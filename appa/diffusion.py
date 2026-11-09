r"""Diffusion helpers."""

import math
import torch
import torch.nn as nn

from einops import rearrange
from functools import partial
from omegaconf import DictConfig, open_dict
from torch import Tensor
from torch import distributed as dist
from typing import Callable, List, Union

from appa.grid import create_icosphere, icosphere_nhops_edges
from appa.math import (
    gmres,
    merge_gradients_by_channel_priority,
    merge_gradients_by_channel_priority_accumulate,
    merge_gradients_by_channel_priority_top1,
    merge_gradients_by_group_priority,
    merge_gradients_by_priority,
    project_conflicting_gradients,
)
from appa.nn.sptgraph import SpatioTemporalGraphDiT
from appa.nn.triggers import disable_checkpointing, disable_xfa
from appa.nn.vit import CondViDiT, ViDiT


class Denoiser(nn.Module):
    r"""Denoiser model with EDM-style preconditioning.

    .. math:: d(x_t) \approx E[x | x_t]

    References:
        | Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
        | https://arxiv.org/abs/2206.00364

    Arguments:
        backbone: A noise conditional network.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()

        self.backbone = backbone

    def forward(self, xt: Tensor, sigma_t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            xt: The noisy tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*, 1)`.
            kwargs: Keyword arguments passed to the backbone.

        Returns:
            The denoised tensor :math:`d(x_t)`, with shape :math:`(*, D)`.
        """

        c_skip = 1 / (sigma_t**2 + 1)
        c_out = sigma_t / torch.sqrt(sigma_t**2 + 1)
        c_in = 1 / torch.sqrt(sigma_t**2 + 1)
        c_noise = 1e1 * torch.log(sigma_t)
        c_noise = c_noise.squeeze(dim=-1)

        x_out = self.backbone(c_in * xt, c_noise, **kwargs)

        return c_skip * xt + c_out * x_out


class MMPSDenoiser(nn.Module):
    r"""
    Creates an MMPS denoiser module.

    References:
        | Learning Diffusion Priors from Observations by Expectation Maximization (Rozet et al., 2024)
        | https://arxiv.org/abs/2405.13712

    Arguments:
        denoiser: A Gaussian denoiser.
        A: The forward operator :math:`x \mapsto Ax`. It should take in a vector :math:`x` of shape :math:`(B, D)` and a "blanket_id" integer, and return a vector of shape :math:`(M, B)`.
        y: An observation :math:`y \sim \mathcal{N}(Ax, \diag(\sigma_y ^ 2))` of shape :math:`(M)`, or a list thereof. If a list, corresponds to the observation for each different blanket.
        var_y: The observation variance matrix :math:`\diag(\sigma_y ^ 2)`, or a list thereof. The variance is assumed diagonal with shape :math:`()`, :math:`(D)` or :math:`(D, D)`.
        tweedie_covariance: Whether to use the Tweedie covariance formula or not.
            If :py:`False`, use :math:`\Sigma_t` instead.
        iterations: The number of solver iterations.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        A: Callable[[Tensor, int], Tensor],
        y: Union[Tensor, List[Tensor]],
        var_y: Union[Tensor, List[Tensor]],
        tweedie_covariance: bool = True,
        iterations: int = 1,
    ):
        super().__init__()

        self.denoiser = denoiser

        self.A = A

        if isinstance(y, list):
            self.register_buffer("y", torch.nested.as_nested_tensor(y))
            self.register_buffer("var_y", torch.nested.as_nested_tensor(var_y))
        else:
            self.register_buffer("y", torch.as_tensor(y))
            self.register_buffer("var_y", torch.as_tensor(var_y))

        self.tweedie_covariance = tweedie_covariance

        self.solve = partial(gmres, iterations=iterations)

    def forward(self, x_t: Tensor, sigma_t: Tensor, blanket_id: int = 0, **kwargs):
        var_t = sigma_t**2 / (1 + sigma_t**2)

        selfA = partial(self.A, blanket_id=blanket_id)
        selfy = self.y[blanket_id] if self.y.is_nested else self.y
        selfvar_y = self.var_y[blanket_id] if self.var_y.is_nested else self.var_y

        if selfy is None or len(selfy) == 0:
            return self.denoiser(x_t, sigma_t, **kwargs)  # Unconditioned blanket.

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            y_hat = selfA(x_hat)

        def A(v):
            with (
                disable_xfa(),
                disable_checkpointing(),
            ):  # jvp incompatible with xformers and checkpointing
                return torch.func.jvp(selfA, (x_hat.detach(),), (v,))[1]

        def At(v):
            return torch.autograd.grad(y_hat, x_hat, v, retain_graph=True)[0]

        # fmt: off
        if self.tweedie_covariance:
            def cov_x(v):
                return sigma_t**2 * torch.autograd.grad(x_hat, x_t, v, retain_graph=True)[0]
        else:
            def cov_x(v):
                return var_t * v
        # fmt: on

        def cov_y(v):
            return selfvar_y * v + A(cov_x(At(v)))

        grad = selfy - y_hat
        grad = self.solve(A=cov_y, b=grad)
        score = torch.autograd.grad(y_hat, x_t, grad)[0]

        return x_hat + sigma_t**2 * score


class MultiInstrumentMMPSDenoiser(MMPSDenoiser):
    """Merge independently solved per-instrument MMPS guidance."""

    def __init__(
        self,
        denoiser,
        A,
        y,
        var_y,
        instrument_indices,
        instrument_masks=None,
        instrument_operators=None,
        merge="sum",
        pcgrad_group_size=None,
        pcgrad_overlap_only=False,
        pcgrad_shuffle=False,
        ppgf_overlap_only=False,
        ppgf_aligned_clip=0.0,
        ppgf_order_group_count=None,
        ppgf_bypass_instrument_id=None,
        ppgf_bypass_mask=None,
        ppgf_bypass_merge_mode="retained_ppgf_plus_bypass",
        tweedie_covariance=True,
        iterations=1,
        instrument_gradient_masks=None,
    ):
        super().__init__(
            denoiser=denoiser,
            A=A,
            y=y,
            var_y=var_y,
            tweedie_covariance=tweedie_covariance,
            iterations=iterations,
        )
        if merge not in {
            "sum",
            "pcgrad",
            "ppgf",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
            "spatial_vector_ppgf",
        }:
            raise ValueError(f"Unknown multi-instrument guidance merge: {merge}")
        self.instrument_indices = instrument_indices
        self.instrument_masks = instrument_masks
        self.instrument_gradient_masks = instrument_gradient_masks
        self.instrument_operators = instrument_operators
        self.merge = merge
        self.pcgrad_group_size = pcgrad_group_size
        self.pcgrad_overlap_only = pcgrad_overlap_only and merge == "pcgrad"
        self.pcgrad_shuffle = pcgrad_shuffle and merge == "pcgrad"
        self.channel_ppgf_accumulate = merge == "channel_ppgf_accumulate"
        self.channel_ppgf_top1 = merge == "channel_ppgf_top1"
        self.channel_ppgf = merge in {
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
        }
        self.spatial_vector_ppgf = merge == "spatial_vector_ppgf"
        self.ppgf_order_group_count = (
            None if ppgf_order_group_count is None else int(ppgf_order_group_count)
        )
        self.ppgf_overlap_only = ppgf_overlap_only and merge in {
            "ppgf",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
            "spatial_vector_ppgf",
        }
        if (
            self.channel_ppgf_accumulate or self.channel_ppgf_top1
        ) and self.ppgf_overlap_only:
            raise ValueError(
                f"{merge} requires ppgf_overlap_only=false"
            )
        if self.channel_ppgf and self.pcgrad_group_size is None:
            raise ValueError(
                f"{merge} requires a latent channel group size"
            )
        if self.spatial_vector_ppgf:
            if self.pcgrad_group_size is None:
                raise ValueError(
                    "spatial_vector_ppgf requires a latent vector group size"
                )
            if self.ppgf_order_group_count is None or self.ppgf_order_group_count < 1:
                raise ValueError(
                    "spatial_vector_ppgf requires a positive order-group count"
                )
        ppgf_aligned_clip = float(ppgf_aligned_clip)
        if not -1.0 <= ppgf_aligned_clip <= 1.0:
            raise ValueError("ppgf_aligned_clip must be in [-1, 1]")
        self.ppgf_aligned_clip = (
            ppgf_aligned_clip
            if merge in {"ppgf", "spatial_vector_ppgf"}
            else 0.0
        )
        self.ppgf_ordered = merge in {
            "ppgf",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
            "spatial_vector_ppgf",
        }
        if (self.pcgrad_overlap_only or self.ppgf_overlap_only) and instrument_masks is None:
            raise ValueError("instrument_masks are required for overlap-only merging")
        num_instruments = max(
            (len(indices) for indices in instrument_indices), default=0
        )
        if instrument_gradient_masks is not None:
            if len(instrument_gradient_masks) != len(instrument_indices):
                raise ValueError(
                    "instrument_gradient_masks must contain one entry per blanket"
                )
            normalized_gradient_masks = []
            for blanket_id, (indices, masks) in enumerate(
                zip(
                    instrument_indices,
                    instrument_gradient_masks,
                    strict=True,
                )
            ):
                if len(masks) != len(indices):
                    raise ValueError(
                        "instrument_gradient_masks blanket "
                        f"{blanket_id} must contain one entry per instrument"
                    )
                normalized_masks = []
                for instrument_id, mask in enumerate(masks):
                    if mask is None:
                        normalized_masks.append(None)
                        continue
                    mask = torch.as_tensor(mask, dtype=torch.bool).detach()
                    if mask.numel() == 0:
                        raise ValueError(
                            "instrument gradient mask cannot be empty: "
                            f"blanket={blanket_id}, instrument={instrument_id}"
                        )
                    normalized_masks.append(mask)
                normalized_gradient_masks.append(normalized_masks)
            self.instrument_gradient_masks = normalized_gradient_masks
        bypass_is_configured = ppgf_bypass_instrument_id is not None
        if (
            self.channel_ppgf_accumulate or self.channel_ppgf_top1
        ) and bypass_is_configured:
            raise ValueError(f"{merge} does not support PPGF bypass")
        if bypass_is_configured != (ppgf_bypass_mask is not None):
            raise ValueError(
                "ppgf_bypass_instrument_id and ppgf_bypass_mask must be set together"
            )
        ppgf_bypass_merge_mode = str(ppgf_bypass_merge_mode).lower()
        valid_bypass_merge_modes = {
            "retained_ppgf_plus_bypass",
            "sum_all",
        }
        if ppgf_bypass_merge_mode not in valid_bypass_merge_modes:
            raise ValueError(
                "ppgf_bypass_merge_mode must be one of "
                f"{sorted(valid_bypass_merge_modes)}"
            )
        if not bypass_is_configured and ppgf_bypass_merge_mode != (
            "retained_ppgf_plus_bypass"
        ):
            raise ValueError(
                "ppgf_bypass_merge_mode requires a configured PPGF bypass"
            )
        self.ppgf_bypass_merge_mode = ppgf_bypass_merge_mode
        self.ppgf_bypass_instrument_id = None
        if not bypass_is_configured:
            self.register_buffer("ppgf_bypass_mask", None)
        else:
            if not self.ppgf_ordered:
                raise ValueError("PPGF bypass requires a PPGF guidance merge")
            bypass_instrument_id = int(ppgf_bypass_instrument_id)
            if not 0 <= bypass_instrument_id < num_instruments:
                raise ValueError(
                    f"PPGF bypass instrument ID is out of range: {bypass_instrument_id}"
                )
            bypass_mask = torch.as_tensor(
                ppgf_bypass_mask,
                dtype=torch.bool,
                device=self.var_y.device,
            )
            if bypass_mask.ndim != 1 or bypass_mask.numel() == 0:
                raise ValueError(
                    "ppgf_bypass_mask must be a non-empty one-dimensional mask"
                )
            self.ppgf_bypass_instrument_id = bypass_instrument_id
            self.register_buffer(
                "ppgf_bypass_mask",
                bypass_mask,
            )
        if instrument_operators is not None:
            if len(instrument_operators) != num_instruments:
                raise ValueError(
                    "instrument_operators must contain one operator per instrument"
                )
            if not all(callable(operator) for operator in instrument_operators):
                raise TypeError("each instrument operator must be callable")

        self.num_instruments = num_instruments
        diagnostic_options = {"device": self.var_y.device, "dtype": torch.float32}
        self.register_buffer(
            "cosine_sum",
            torch.zeros(num_instruments, num_instruments, **diagnostic_options),
        )
        self.register_buffer(
            "cosine_count",
            torch.zeros(num_instruments, num_instruments, **diagnostic_options),
        )
        self.register_buffer(
            "conflict_count",
            torch.zeros(num_instruments, num_instruments, **diagnostic_options),
        )
        self.register_buffer(
            "guidance_norm_sum", torch.zeros(num_instruments, **diagnostic_options)
        )
        self.register_buffer(
            "guidance_norm_count", torch.zeros(num_instruments, **diagnostic_options)
        )

    def _group_masks(self, blanket_id, instrument_ids, batch_size, device):
        if not (self.pcgrad_overlap_only or self.ppgf_overlap_only):
            return None
        return [
            self.instrument_masks[blanket_id][instrument_id]
            .to(device=device)
            .reshape(1, -1)
            .expand(batch_size, -1)
            .reshape(-1)
            for instrument_id in instrument_ids
        ]

    def _apply_instrument_gradient_mask(
        self,
        score,
        blanket_id,
        instrument_id,
    ):
        """Hard-mask one instrument score before diagnostics and merging."""
        if self.instrument_gradient_masks is None:
            return score
        mask = self.instrument_gradient_masks[blanket_id][instrument_id]
        if mask is None:
            return score
        if score.ndim < 2:
            raise ValueError(
                "instrument gradient masking requires a leading batch dimension"
            )
        elements_per_sample = score[0].numel()
        if mask.numel() != elements_per_sample:
            raise ValueError(
                "instrument gradient mask width does not match one denoiser "
                f"sample: {mask.numel()} != {elements_per_sample} "
                f"(blanket={blanket_id}, instrument={instrument_id})"
            )
        expanded = mask.to(device=score.device).reshape(
            1,
            *score.shape[1:],
        ).expand_as(score)
        return score.masked_fill(~expanded, 0)

    def _local_projection_orders(self, projection_orders, instrument_ids):
        if projection_orders is None:
            return None
        projection_orders = torch.as_tensor(
            projection_orders,
            dtype=torch.long,
            device=self.var_y.device,
        )
        global_shape = (self.num_instruments, self.num_instruments)
        if projection_orders.ndim == 2:
            if tuple(projection_orders.shape) != global_shape:
                raise ValueError(
                    "global PCGrad orders must have shape " f"{global_shape}"
                )

            global_to_local = {
                instrument_id: local_id
                for local_id, instrument_id in enumerate(instrument_ids)
            }
            local_orders = []
            expected = list(range(self.num_instruments))
            for instrument_id in instrument_ids:
                global_order = [
                    int(index)
                    for index in projection_orders[instrument_id].tolist()
                ]
                if sorted(global_order) != expected:
                    raise ValueError(
                        "each PCGrad order must permute all instrument IDs"
                    )
                local_orders.append([
                    global_to_local[index]
                    for index in global_order
                    if index in global_to_local
                ])
            return local_orders

        if projection_orders.ndim != 3 or tuple(projection_orders.shape[1:]) != (
            self.num_instruments,
            self.num_instruments,
        ):
            raise ValueError(
                "spatial PCGrad orders must have shape "
                f"(num_order_groups, {self.num_instruments}, "
                f"{self.num_instruments})"
            )
        num_order_groups = int(projection_orders.shape[0])
        expected = torch.arange(
            self.num_instruments,
            dtype=torch.long,
            device=projection_orders.device,
        ).expand(num_order_groups, self.num_instruments, -1)
        if not torch.equal(
            torch.sort(projection_orders, dim=-1).values,
            expected,
        ):
            raise ValueError(
                "each spatial PCGrad order must permute all instrument IDs"
            )

        mapping = torch.full(
            (self.num_instruments,),
            -1,
            dtype=torch.long,
            device=projection_orders.device,
        )
        instrument_ids = torch.as_tensor(
            instrument_ids,
            dtype=torch.long,
            device=projection_orders.device,
        )
        mapping[instrument_ids] = torch.arange(
            len(instrument_ids),
            device=projection_orders.device,
        )
        target_orders = projection_orders.index_select(1, instrument_ids)
        local_orders = mapping[target_orders]
        return local_orders[local_orders >= 0].reshape(
            num_order_groups,
            len(instrument_ids),
            len(instrument_ids),
        )

    def _local_priority_order(self, ppgf_order, instrument_ids):
        if tuple(ppgf_order.shape) != (self.num_instruments,):
            raise ValueError(
                "PPGF order must have shape " f"({self.num_instruments},)"
            )

        global_order = [int(index) for index in ppgf_order.tolist()]
        expected = list(range(self.num_instruments))
        if sorted(global_order) != expected:
            raise ValueError("PPG:F order must permute all instrument IDs")
        global_to_local = {
            instrument_id: local_id
            for local_id, instrument_id in enumerate(instrument_ids)
        }
        return [
            global_to_local[index]
            for index in global_order
            if index in global_to_local
        ]

    def _local_channel_priority_orders(self, ppgf_order, instrument_ids):
        expected_shape = (
            self.pcgrad_group_size,
            self.num_instruments,
        )
        if tuple(ppgf_order.shape) != expected_shape:
            raise ValueError(
                "Channel PPGF orders must have shape " f"{expected_shape}"
            )
        expected = torch.arange(
            self.num_instruments,
            dtype=torch.long,
            device=ppgf_order.device,
        ).expand(self.pcgrad_group_size, -1)
        if not torch.equal(torch.sort(ppgf_order, dim=1).values, expected):
            raise ValueError(
                "each Channel PPGF order must permute all instrument IDs"
            )

        mapping = torch.full(
            (self.num_instruments,),
            -1,
            dtype=torch.long,
            device=ppgf_order.device,
        )
        instrument_ids = torch.as_tensor(
            instrument_ids,
            dtype=torch.long,
            device=ppgf_order.device,
        )
        mapping[instrument_ids] = torch.arange(
            len(instrument_ids), device=ppgf_order.device
        )
        local_orders = mapping[ppgf_order]
        return local_orders[local_orders >= 0].reshape(
            self.pcgrad_group_size, len(instrument_ids)
        )

    def _local_group_priority_orders(self, ppgf_order, instrument_ids):
        expected_shape = (
            self.ppgf_order_group_count,
            self.num_instruments,
        )
        if tuple(ppgf_order.shape) != expected_shape:
            raise ValueError(
                "Spatial Vector PPGF orders must have shape " f"{expected_shape}"
            )
        expected = torch.arange(
            self.num_instruments,
            dtype=torch.long,
            device=ppgf_order.device,
        ).expand(self.ppgf_order_group_count, -1)
        if not torch.equal(torch.sort(ppgf_order, dim=1).values, expected):
            raise ValueError(
                "each Spatial Vector PPGF order must permute all instrument IDs"
            )

        mapping = torch.full(
            (self.num_instruments,),
            -1,
            dtype=torch.long,
            device=ppgf_order.device,
        )
        instrument_ids = torch.as_tensor(
            instrument_ids,
            dtype=torch.long,
            device=ppgf_order.device,
        )
        mapping[instrument_ids] = torch.arange(
            len(instrument_ids),
            device=ppgf_order.device,
        )
        local_orders = mapping[ppgf_order]
        return local_orders[local_orders >= 0].reshape(
            self.ppgf_order_group_count,
            len(instrument_ids),
        )

    def _merge_ppgf_subset(
        self,
        scores,
        instrument_ids,
        group_masks,
        ppgf_order,
        conflict_weight,
    ):
        """Merge a selected set of instruments with the configured PPGF rule."""
        if not scores:
            return None
        if len(scores) == 1 or conflict_weight == 0.0:
            return torch.stack(scores).sum(0)
        if ppgf_order is None:
            raise ValueError("PPGF requires a timestep-level priority order")
        if self.channel_ppgf_top1:
            orders = self._local_channel_priority_orders(
                ppgf_order,
                instrument_ids,
            )
            return merge_gradients_by_channel_priority_top1(
                scores,
                orders=orders,
                group_size=self.pcgrad_group_size,
                conflict_weight=conflict_weight,
            )
        if self.channel_ppgf_accumulate:
            orders = self._local_channel_priority_orders(
                ppgf_order,
                instrument_ids,
            )
            return merge_gradients_by_channel_priority_accumulate(
                scores,
                orders=orders,
                group_size=self.pcgrad_group_size,
                conflict_weight=conflict_weight,
            )
        if self.channel_ppgf:
            orders = self._local_channel_priority_orders(
                ppgf_order,
                instrument_ids,
            )
            return merge_gradients_by_channel_priority(
                scores,
                orders=orders,
                group_size=self.pcgrad_group_size,
                group_masks=group_masks,
                conflict_weight=conflict_weight,
            )
        if self.spatial_vector_ppgf:
            orders = self._local_group_priority_orders(
                ppgf_order,
                instrument_ids,
            )
            return merge_gradients_by_group_priority(
                scores,
                orders=orders,
                group_size=self.pcgrad_group_size,
                group_masks=group_masks,
                aligned_clip=self.ppgf_aligned_clip,
                conflict_weight=conflict_weight,
            )
        order = self._local_priority_order(ppgf_order, instrument_ids)
        return merge_gradients_by_priority(
            scores,
            order=order,
            group_size=self.pcgrad_group_size,
            group_masks=group_masks,
            aligned_clip=self.ppgf_aligned_clip,
            conflict_weight=conflict_weight,
        )

    def _expanded_ppgf_bypass_mask(self, score):
        if score.ndim < 2:
            raise ValueError("PPGF bypass expects a leading batch dimension")
        elements_per_sample = score[0].numel()
        if self.ppgf_bypass_mask.numel() != elements_per_sample:
            raise ValueError(
                "PPGF bypass mask width does not match one denoiser sample: "
                f"{self.ppgf_bypass_mask.numel()} != {elements_per_sample}"
            )
        shape = (1, *score.shape[1:])
        return self.ppgf_bypass_mask.reshape(shape).expand_as(score)

    def _record_diagnostics(self, scores, instrument_ids, group_masks=None):
        with torch.no_grad():
            for local_i, instrument_i in enumerate(instrument_ids):
                norm_i = torch.linalg.vector_norm(scores[local_i].float())
                self.guidance_norm_sum[instrument_i] += norm_i
                self.guidance_norm_count[instrument_i] += 1
                for local_j in range(local_i + 1, len(scores)):
                    instrument_j = instrument_ids[local_j]
                    score_i = scores[local_i].float()
                    score_j = scores[local_j].float()
                    if self.pcgrad_group_size is None:
                        score_i = score_i.reshape(1, -1)
                        score_j = score_j.reshape(1, -1)
                    else:
                        score_i = score_i.reshape(-1, self.pcgrad_group_size)
                        score_j = score_j.reshape(-1, self.pcgrad_group_size)
                    dot = torch.sum(score_i * score_j, dim=-1)
                    denominator = (
                        torch.linalg.vector_norm(score_i, dim=-1)
                        * torch.linalg.vector_norm(score_j, dim=-1)
                    )
                    valid = denominator > 1e-12
                    if group_masks is not None:
                        valid = valid & group_masks[local_i] & group_masks[local_j]
                    cosine = dot[valid] / denominator[valid]
                    cosine_sum = cosine.sum()
                    pair_count = valid.sum().to(self.cosine_count.dtype)
                    conflict_count = (cosine < 0).sum().to(self.conflict_count.dtype)
                    self.cosine_sum[instrument_i, instrument_j] += cosine_sum
                    self.cosine_sum[instrument_j, instrument_i] += cosine_sum
                    self.cosine_count[instrument_i, instrument_j] += pair_count
                    self.cosine_count[instrument_j, instrument_i] += pair_count
                    self.conflict_count[instrument_i, instrument_j] += conflict_count
                    self.conflict_count[instrument_j, instrument_i] += conflict_count

    def diagnostics(self):
        cosine_mean = self.cosine_sum / self.cosine_count.clamp_min(1)
        conflict_fraction = self.conflict_count / self.cosine_count.clamp_min(1)
        norm_mean = self.guidance_norm_sum / self.guidance_norm_count.clamp_min(1)
        return {
            "cosine_mean": cosine_mean.detach().cpu(),
            "conflict_fraction": conflict_fraction.detach().cpu(),
            "pair_count": self.cosine_count.detach().cpu(),
            "guidance_norm_mean": norm_mean.detach().cpu(),
        }

    def forward(
        self,
        x_t,
        sigma_t,
        blanket_id=0,
        pcgrad_orders=None,
        ppgf_order=None,
        ppgf_merge_weight=1.0,
        **kwargs,
    ):
        var_t = sigma_t**2 / (1 + sigma_t**2)
        selfA = partial(self.A, blanket_id=blanket_id)
        selfy = self.y[blanket_id] if self.y.is_nested else self.y
        selfvar_y = self.var_y[blanket_id] if self.var_y.is_nested else self.var_y
        if selfy is None or len(selfy) == 0:
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            return x_hat

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            y_hat = (
                selfA(x_hat) if self.instrument_operators is None else None
            )

        def cov_x(vector):
            if self.tweedie_covariance:
                return sigma_t**2 * torch.autograd.grad(
                    x_hat, x_t, vector, retain_graph=True
                )[0]
            return var_t * vector

        scores = []
        instrument_ids = []
        for instrument_id, indices in enumerate(self.instrument_indices[blanket_id]):
            if indices.numel() == 0:
                continue
            indices = indices.to(x_hat.device)
            with torch.enable_grad():
                if self.instrument_operators is None:
                    # Sampling runs under torch.no_grad(). Re-enable grad for the
                    # slice so At_i can still form a VJP.
                    y_hat_i = y_hat[indices]
                    operator_i = None
                else:
                    operator_i = partial(
                        self.instrument_operators[instrument_id],
                        blanket_id=blanket_id,
                    )
                    y_hat_i = operator_i(x_hat)
                    if y_hat_i.numel() != indices.numel():
                        raise ValueError(
                            f"instrument {instrument_id} operator returned "
                            f"{y_hat_i.numel()} values for {indices.numel()} observations"
                        )
            residual_i = selfy[indices] - y_hat_i
            var_y_i = selfvar_y[indices]

            def A_i(vector, operator=operator_i, selected=indices):
                with disable_xfa(), disable_checkpointing():
                    tangent = torch.func.jvp(
                        operator if operator is not None else selfA,
                        (x_hat.detach(),),
                        (vector,),
                    )[1]
                    return tangent if operator is not None else tangent[selected]

            def At_i(vector, output=y_hat_i):
                return torch.autograd.grad(
                    output, x_hat, vector, retain_graph=True
                )[0]

            def cov_y_i(vector, variance=var_y_i, forward=A_i, adjoint=At_i):
                return variance * vector + forward(cov_x(adjoint(vector)))

            solved = self.solve(A=cov_y_i, b=residual_i)
            score_i = torch.autograd.grad(
                y_hat_i, x_t, solved, retain_graph=True
            )[0]
            score_i = self._apply_instrument_gradient_mask(
                score_i,
                blanket_id,
                instrument_id,
            )
            scores.append(score_i)
            instrument_ids.append(instrument_id)
            ### vis score_i norm
            score_i_norm = torch.linalg.norm(score_i,dim=-1)
            # print(f'score{score_i.shape},norm{score_i_norm.shape},x_hat {x_hat.shape}')

            # print(f"Instrumen_id {instrument_id} score_i mean{score_i_norm.mean()} max{score_i_norm.max()} min{score_i_norm.min()}")


        if not scores:
            return x_hat

        group_masks = self._group_masks(
            blanket_id, instrument_ids, x_t.shape[0], x_t.device
        )
        self._record_diagnostics(scores, instrument_ids, group_masks)
        merged_score = None
        if self.merge == "pcgrad" and len(scores) > 1:
            if self.pcgrad_shuffle and pcgrad_orders is None:
                raise ValueError("Shuffled PCGrad requires timestep-level projection orders")
            projection_orders = (
                self._local_projection_orders(pcgrad_orders, instrument_ids)
                if self.pcgrad_shuffle
                else None
            )
            scores = project_conflicting_gradients(
                scores,
                group_size=self.pcgrad_group_size,
                group_masks=group_masks,
                projection_orders=projection_orders,
            )
        elif self.merge in {
            "ppgf",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
            "spatial_vector_ppgf",
        } and len(scores) > 1:
            ppgf_merge_weight = float(ppgf_merge_weight)
            if not 0.0 <= ppgf_merge_weight <= 1.0:
                raise ValueError("ppgf_merge_weight must be in [0, 1]")
            if ppgf_merge_weight == 0.0:
                merged_score = torch.stack(scores).sum(0)
            elif ppgf_order is None:
                raise ValueError("PPGF requires a timestep-level priority order")
            elif self.channel_ppgf_top1:
                channel_orders = self._local_channel_priority_orders(
                    ppgf_order,
                    instrument_ids,
                )
                merged_score = merge_gradients_by_channel_priority_top1(
                    scores,
                    orders=channel_orders,
                    group_size=self.pcgrad_group_size,
                    conflict_weight=ppgf_merge_weight,
                )
            elif self.channel_ppgf_accumulate:
                channel_orders = self._local_channel_priority_orders(
                    ppgf_order,
                    instrument_ids,
                )
                merged_score = merge_gradients_by_channel_priority_accumulate(
                    scores,
                    orders=channel_orders,
                    group_size=self.pcgrad_group_size,
                    conflict_weight=ppgf_merge_weight,
                )
            elif self.channel_ppgf:
                channel_orders = self._local_channel_priority_orders(
                    ppgf_order,
                    instrument_ids,
                )
                merged_score = merge_gradients_by_channel_priority(
                    scores,
                    orders=channel_orders,
                    group_size=self.pcgrad_group_size,
                    group_masks=group_masks,
                    conflict_weight=ppgf_merge_weight,
                )
            elif self.spatial_vector_ppgf:
                group_orders = self._local_group_priority_orders(
                    ppgf_order,
                    instrument_ids,
                )
                merged_score = merge_gradients_by_group_priority(
                    scores,
                    orders=group_orders,
                    group_size=self.pcgrad_group_size,
                    group_masks=group_masks,
                    aligned_clip=self.ppgf_aligned_clip,
                    conflict_weight=ppgf_merge_weight,
                )
            else:
                priority_order = self._local_priority_order(
                    ppgf_order,
                    instrument_ids,
                )
                merged_score = merge_gradients_by_priority(
                    scores,
                    order=priority_order,
                    group_size=self.pcgrad_group_size,
                    group_masks=group_masks,
                    aligned_clip=self.ppgf_aligned_clip,
                    conflict_weight=ppgf_merge_weight,
                )
            if (
                self.ppgf_bypass_instrument_id is not None
                and self.ppgf_bypass_instrument_id in instrument_ids
            ):
                if self.ppgf_bypass_merge_mode == "sum_all":
                    bypass_score = torch.stack(scores).sum(0)
                else:
                    bypass_local_id = instrument_ids.index(
                        self.ppgf_bypass_instrument_id
                    )
                    retained_local_ids = [
                        local_id
                        for local_id in range(len(scores))
                        if local_id != bypass_local_id
                    ]
                    retained_scores = [
                        scores[local_id] for local_id in retained_local_ids
                    ]
                    retained_instrument_ids = [
                        instrument_ids[local_id] for local_id in retained_local_ids
                    ]
                    retained_masks = (
                        None
                        if group_masks is None
                        else [group_masks[local_id] for local_id in retained_local_ids]
                    )
                    retained_merged = self._merge_ppgf_subset(
                        retained_scores,
                        retained_instrument_ids,
                        retained_masks,
                        ppgf_order,
                        ppgf_merge_weight,
                    )
                    bypass_score = scores[bypass_local_id]
                    if retained_merged is not None:
                        bypass_score = retained_merged + bypass_score
                bypass_mask = self._expanded_ppgf_bypass_mask(merged_score)
                merged_score = torch.where(
                    bypass_mask,
                    bypass_score,
                    merged_score,
                )

        if merged_score is None:
            merged_score = torch.stack(scores).sum(0)
        return x_hat + sigma_t**2 * merged_score


class NoisedObsDenoiser(nn.Module):
    r"""Noised-observation (DDRM / RePaint replacement-style) guidance for o2e.

    Alternative to MMPSDenoiser. Instead of the DPS/Tweedie route (linearize the
    operator at x̂ and solve a coupled system with the denoiser Jacobian Jᵀ), this
    noises the observation along the SAME VE forward diffusion as x and injects it
    at the x_t (input) level. It only applies when the operator is the IDENTITY
    restricted to a latent-pixel mask (o2e): y lives in the SAME latent space as x,
    so y_t and x_t[m] are the same variable at the same noise level and can be
    replaced pixel-for-pixel (no SVD needed — S's SVD is trivially 0/1).

    y0 is the observation (E[x0 | obs][m] = y0). Each ``forward`` (called by the
    UNMODIFIED PC sampler) runs the denoiser once, then FUSES the observation into the
    denoiser's x̂ at the observed pixels — a precision-weighted (Kalman-gain) update,
    the VE translation of FPS-SMC's ``p_mean_variance`` fusion (Dou & Song 2023):

        x̂ = denoiser(x_t)
        x̂[m] ← (1 - w)·x̂[m] + w·y0 ,   w = σ_t² / (σ_t² + var_y)

    Derivation of w (why σ_t², not var_t). At the x_t noise level the state's own
    (prior/proposal) variance is σ_t² — the VE forward is ``x_t = x0 + σ_t·ε`` with NO
    signal attenuation, so unlike FPS's VP form ``ᾱ_t·σ²`` the observation variance is
    NOT rescaled by any ᾱ; it stays ``var_y`` in the same units as x. FPS builds
    ``coef = posterior_var / (ᾱ_t·σ²)`` and fuses with ``coef/(coef+1)``; the VE analog
    is ``coef = σ_t²/var_y`` ⇒ ``w = coef/(coef+1) = σ_t²/(σ_t²+var_y)`` — the standard
    Kalman gain ``Σ_prior/(Σ_prior+Σ_obs)``. (An earlier version used
    ``var_t=σ_t²/(1+σ_t²)`` here; that is the ΠGDM Tweedie-covariance for the x0/Jᵀ
    route, wrong for an x_t-level fusion because it saturates to 1 at high σ_t and so
    under-trusts the obs early in sampling. σ_t² is the correct prior variance.)

    Behaviour across the schedule: high σ_t (early) → w→1, trust the obs (anchor the
    coarse structure); low σ_t (late) → w→0, trust the denoiser (it refines detail).
    Unobserved pixels keep x̂ untouched — the obs reaches them only ACROSS steps via
    the denoiser prior, NOT within-step (there is no Jᵀ back-projection here, unlike
    MMPS; z/wind may benefit less than directly-observed temp/humidity).

    Cost: one denoiser call + one masked lerp, no GMRES / jvp / vjp / autograd.

    Arguments:
        denoiser: A Gaussian denoiser (same object MMPS wraps).
        obs_by_blanket: list indexed by blanket_id; each entry is a list of tuples
            ``(t_local, m_lat, y_sel, var_y_col)`` for that blanket's observed
            timesteps, where
              t_local   : int, frame index within the blanket [0, blanket_size)
              m_lat     : (N,) bool mask of observed latent pixels
              y_sel     : (M*C,) obs values at masked pixels, pixel-major/channel-minor
              var_y_col : (C,) per-channel observation variance σ_y² = var_y.
        num_latent_channels: C, latent channels per pixel.
        blanket_size: K, frames per blanket.
        use_var_y: If False (default), var_y=0 in the weight ⇒ w=1 ⇒ x̂[m] is fully
            OVERWRITTEN by y0 (treat the obs as clean/exact — hard anchor at observed
            pixels, at every noise level). If True, use the calibrated per-channel
            var_y in the Kalman weight ``w = σ_t²/(σ_t²+var_y)`` so the fusion soft-
            blends: it trusts the obs at high σ_t and hands off to the denoiser as
            σ_t → σ_min (where w → σ_min²/(σ_min²+var_y) ≈ 0).

    REPLACEABILITY BOUNDARY — when can you guide x_t with a noised obs y_t directly
    (this replacement), and when must you fall back to gradient guidance (MMPS)? The
    deciding factor is the observation operator A in ``y = A(x) + noise``:

      * A = selection matrix S (o2e, A(z)=z[:,m,:]): y_t and x_t[m] are the SAME
        variable at the SAME noise level — one-to-one, same space. Direct
        replacement is EXACT. This class.
      * A = a general LINEAR operator (non-selection: a linear conv/mapper, blur,
        downsample): y is a linear MIX of x's coords, so NO per-coordinate y↔x
        correspondence → cannot replace pixel-wise. DDRM's fix is to replace in A's
        SVD domain (rotate x by Vᵀ so A becomes diagonal, replace per spectral
        component). Correct but needs A's SVD (for conv: FFT-diagonalization +
        boundary care) — not implemented here.
      * A NONLINEAR (e2s's 5×5+SiLU mapper, obsop's decode+MLP): no SVD / cannot be
        diagonalized → replacement is impossible → MUST use gradient guidance
        (DPS/MMPS: H=∂A/∂x̂ linearization + Jᵀ propagation).

    Corollary for the three lines: o2e → this replacement; e2s (nonlinear mapper) &
    obsop → MMPSDenoiser. Even a purely-linear e2s mapper (the 1×1/3×3/5×5
    no-activation ladder) is DDRM-SVD-able in theory but the conv SVD is fiddly, so
    gradient guidance is the practical choice there too.

    WHY NOISED y_t IS BOUND TO THIS (not to MMPS): MMPS compares the innovation at
    the CLEAN x0 level (via x̂=E[x0|x_t]), so it should use the CLEANEST y — injecting
    extra noise there only degrades it. Noised y_t makes sense ONLY at the x_t level
    (same noise σ_t, no x̂ bridge), which is exactly this replacement. So "noised y_t"
    and "goes through x̂/Jᵀ" are mutually exclusive: pick x_t-level replacement (y_t,
    no Jᵀ, propagation via the net) OR MMPS (clean y, x0-level, linear Jᵀ).

    Ref: DDRM (Kawar et al. 2022, arXiv:2201.11793), RePaint (Lugmayr et al. 2022,
    arXiv:2201.09865). MMPS is the DPS-family gradient route (Rozet et al. 2405.13712).
    """

    def __init__(
        self,
        denoiser: Denoiser,
        obs_by_blanket: List,
        num_latent_channels: int,
        blanket_size: int,
        use_var_y: bool = False,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.obs_by_blanket = obs_by_blanket
        self.num_latent_channels = num_latent_channels
        self.blanket_size = blanket_size
        self.use_var_y = use_var_y

    def forward(self, x_t: Tensor, sigma_t: Tensor, blanket_id: int = 0, **kwargs):
        x_hat = self.denoiser(x_t, sigma_t, **kwargs)

        obs = self.obs_by_blanket[blanket_id] if blanket_id < len(self.obs_by_blanket) else None
        if not obs:
            return x_hat  # Unconditioned blanket.

        C = self.num_latent_channels
        K = self.blanket_size
        # (B, K*N*C) -> (B, K, N, C). CLONE so the in-place fusion never mutates the
        # denoiser's output tensor (rearrange returns a view; mutating it could leak
        # across steps if the denoiser caches/reuses buffers).
        z = rearrange(x_hat, "B (K N C) -> B K N C", K=K, C=C).clone()

        # Kalman-gain prior variance at this level: σ_t² (VE forward, no ᾱ rescale).
        sig = sigma_t.reshape(()) if torch.is_tensor(sigma_t) else torch.as_tensor(sigma_t)
        sig2 = (sig.to(z.dtype)) ** 2

        for t_local, m_lat, y_sel, var_y_col in obs:
            m = m_lat.to(z.device)
            M = int(m.sum())
            if M == 0:
                continue
            y0 = y_sel.to(z.device, z.dtype).view(M, C)             # (M, C) obs
            frame = z[:, t_local]                                   # (B, N, C) view of z

            if not self.use_var_y:
                # var_y -> 0  =>  w = σ_t²/(σ_t²+0) = 1  =>  hard overwrite by y0
                # (treat the obs as clean/exact at observed pixels, every level).
                frame[:, m, :] = y0.unsqueeze(0)
            else:
                # Kalman gain w = σ_t² / (σ_t² + var_y), per channel (var_y is (C,)).
                vy = var_y_col.to(z.device, z.dtype).view(1, C)     # (1, C) σ_y²
                w = (sig2 / (sig2 + vy)).to(z.dtype)                # (1, C)
                frame[:, m, :] = (1 - w) * frame[:, m, :] + w * y0.unsqueeze(0)

        return rearrange(z, "B K N C -> B (K N C)")


def noised_obs_step(
    y_next: Tensor,
    y0: Tensor,
    sigma_t: Union[float, Tensor],
    sigma_next: Union[float, Tensor],
    c_rate: float = 1.0,
    noise: Tensor = None,
) -> Tensor:
    r"""One VE reverse step of the noised-observation sequence (FPS-style recurrence).

    Builds the noised observation :math:`y_t` at noise level :math:`\sigma_t` from the
    clean observation :math:`y_0` and the previous (higher-noise) sample
    :math:`y_{t+1}` at level :math:`\sigma_{t+1}`, following the SAME VE forward
    diffusion as the state (:math:`x_t = x_0 + \sigma_t\,\varepsilon`, no signal decay):

    .. math::
        y_t = y_0 + \sqrt{1-c}\,\frac{\sigma_t}{\sigma_{t+1}}\,(y_{t+1} - y_0)
                  + \sqrt{c}\,\sigma_t\,\varepsilon ,\quad \varepsilon\sim\mathcal N(0,I)

    This is the VE translation of FPS-SMC's VP recurrence
    ``new_y = c2·y0 + c1·(y_{t+1} − c3·y0) + c4·noise`` (Dou & Song, NeurIPS 2023).
    Under VE the signal is not attenuated, so the two FPS signal coefficients collapse
    to 1 (``c2 = c3 = 1``) and only the noise coefficients remain:

    ==========================================  =====================================
    FPS (VP, :math:`\bar\alpha` schedule)       Appa VE (:math:`\sigma` schedule)
    ==========================================  =====================================
    ``c2 = √ᾱ_t``           (× y0, signal)       ``1``
    ``c3 = √ᾱ_{t+1}``       (× y0, subtract)     ``1``
    ``c1 = √((1-c)(1-ᾱ_t)/(1-ᾱ_{t+1}))``         ``√(1-c)·σ_t/σ_{t+1}``  (reuse old noise)
    ``c4 = √c·√(1-ᾱ_t)``                          ``√c·σ_t``              (inject new noise)
    ==========================================  =====================================

    ``c_rate`` (= FPS's ``c``) trades off reusing :math:`y_{t+1}`'s noise vs injecting
    fresh noise: ``c=0`` → deterministic (VE-DDIM, :math:`y_t,y_{t+1}` fully correlated);
    ``c=1`` → independent draws each step. Any ``c`` gives the exact marginal
    :math:`y_t\sim\mathcal N(y_0,\sigma_t^2)` (same distribution as :math:`x_t` at
    observed pixels — the "same diffusion coefficient" requirement), because
    ``reuse²·σ_{t+1}² + inject² = (1-c)σ_t² + c·σ_t² = σ_t²`` provided ``y_{t+1}`` is
    itself :math:`\mathcal N(y_0,\sigma_{t+1}^2)` (preserved by induction from the
    :func:`build_noised_obs_sequence` initialisation).

    ``y0`` is treated as the CLEAN observation here (its calibrated encoder noise
    :math:`\sigma_y` is folded into the fusion step downstream, not into this
    sequence). If instead you want :math:`y_t` to match :math:`x_t` accounting for
    :math:`\sigma_y`, replace ``σ_t`` with ``√(σ_t²−σ_y²)`` (valid only for
    :math:`\sigma_t\ge\sigma_y`).

    Arguments:
        y_next: :math:`y_{t+1}`, the previous higher-noise observation, any shape.
        y0: The clean observation :math:`y_0`, same shape as ``y_next``.
        sigma_t: Target noise level :math:`\sigma_t` (scalar or broadcastable).
        sigma_next: Previous noise level :math:`\sigma_{t+1}` (> 0).
        c_rate: Noise-refresh rate :math:`c \in [0, 1]`.
        noise: Optional pre-drawn :math:`\varepsilon` (same shape as ``y0``); a fresh
            ``randn_like(y0)`` is used when ``None``.

    Returns:
        :math:`y_t`, same shape as ``y0``.
    """
    if noise is None:
        noise = torch.randn_like(y0)
    reuse = math.sqrt(1.0 - c_rate) * (sigma_t / sigma_next)  # c1: carry y_{t+1}'s noise
    inject = math.sqrt(c_rate) * sigma_t                      # c4: fresh noise
    return y0 + reuse * (y_next - y0) + inject * noise


def build_noised_obs_sequence(
    y0: Tensor,
    sigmas,
    c_rate: float = 1.0,
    noise: List = None,
) -> List:
    r"""Build the full noised-observation sequence :math:`y_{t}` over a VE schedule.

    Runs :func:`noised_obs_step` from the noisiest level down, mirroring FPS's
    ``p_sample_loop`` which constructs the whole ``ys`` list up front. The result is
    aligned one-to-one with the sampler's own :math:`\sigma_t` points so that at
    reverse step ``i`` the guidance can fuse the state with ``ys[i]``.

    Arguments:
        y0: Clean observation :math:`y_0`, any shape.
        sigmas: Sequence of noise levels in DESCENDING order (``sigmas[0]`` = highest,
            e.g. ``σ_max``; ``sigmas[-1]`` = lowest). Must match the state sampler's
            per-step ``σ_t`` values.
        c_rate: Noise-refresh rate passed to :func:`noised_obs_step`.
        noise: Optional list of pre-drawn noise tensors (``len(sigmas)`` entries, each
            ``randn_like(y0)``) for reproducibility; fresh noise is drawn when ``None``.

    Returns:
        List ``ys`` of length ``len(sigmas)`` with ``ys[i] ~ N(y0, sigmas[i]²)``.
        ``ys[0]`` is initialised as ``y0 + sigmas[0]·ε`` (≈ pure noise when
        ``σ_max`` dominates, matching FPS's ``y_last = A(randn)``).
    """
    n = len(sigmas)
    ys = [None] * n
    eps0 = noise[0] if noise is not None else torch.randn_like(y0)
    ys[0] = y0 + sigmas[0] * eps0
    for i in range(1, n):
        eps = noise[i] if noise is not None else None
        ys[i] = noised_obs_step(ys[i - 1], y0, sigmas[i], sigmas[i - 1], c_rate, eps)
    return ys


# TODO: Support for blanket_id.
class SDADenoiser(nn.Module):
    r"""Creates an sda-like guided denoiser module.

    References:
        | Score-based Data Assimiliation (Rozet et al., 2023)
        | https://arxiv.org/abs/2306.10574

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        gamma: A coefficient :math:`\gamma \approx \diag(A A^\top)`.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        gamma: Tensor = 1.0,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    def forward(self, x_t: Tensor, sigma_t: Tensor, **kwargs):
        var_t = sigma_t**2 / (1 + sigma_t**2)

        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_()
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)

            y_hat = self.A(x_hat)

            log_p = (self.y - y_hat) ** 2 / (self.var_y + self.gamma * var_t)

            log_p = -1 / 2 * log_p.sum()

        score = torch.autograd.grad(log_p, x_t)[0]

        return x_hat + sigma_t**2 * score


# TODO: Support for blanket_id.
class W2CDenoiser(nn.Module):
    r"""Creates an simplified sda-like guided denoiser module.

    References:
        | A Generative Framework for Probabilistic, Spatiotemporally Coherent Downscaling
        | of Climate Simulation (Schmidt et al., 2024)
        | https://arxiv.org/abs/2412.15361

    Arguments:
        denoiser: A Gaussian denoiser.
        y: An observation :math:`y \sim \mathcal{N}(A(x), \Sigma_y)`, with shape :math:`(*, D)`.
        A: The forward operator :math:`x \mapsto A(x)`.
        var_y: The noise variance :math:`\Sigma_y`.
        gamma: A coefficient :math:`\gamma \approx \diag(A A^\top)`.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        var_y: Tensor,
        gamma: Tensor = 1.0,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.A = A

        self.register_buffer("y", torch.as_tensor(y))
        self.register_buffer("var_y", torch.as_tensor(var_y))
        self.register_buffer("gamma", torch.as_tensor(gamma))

    def forward(self, x_t: Tensor, sigma_t: Tensor, **kwargs):
        var_t = sigma_t**2 / (1 + sigma_t**2)

        x_hat = self.denoiser(x_t, sigma_t, **kwargs)

        with torch.enable_grad():
            x_hat = x_hat.detach().requires_grad_()
            y_hat = self.A(x_hat)

            log_p = (self.y - y_hat) ** 2 / (self.var_y + self.gamma * var_t)
            log_p = -1 / 2 * log_p.sum()

        grad = torch.autograd.grad(log_p, x_hat)[0]

        return x_hat + var_t * grad


class TrajectoryDenoiser(nn.Module):
    r"""Wraps a blanket denoising model into a full trajectory model.

    Arguments:
        denoiser: A denoiser backbone that operates on blanket.
        blanket_size: The size of the backbone blanket.
        blanket_stride: The stride between two consecutive blankets in a trajectory.
        state_size: The size of an individual state.
        distributed: Whether to use distributed processing.
        pass_blanket_ids: Whether to pass the blanket id to the backbone denoiser (e.g., conditioning).
        mode: Mode of trajectory reconstruction from individual windows.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        blanket_size: int,
        blanket_stride: int,
        state_size: int,
        distributed: bool = False,
        pass_blanket_ids: bool = False,
        mode: str = "symmetrical",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.blanket_size = blanket_size
        self.blanket_stride = blanket_stride
        self.state_size = state_size
        self.distributed = distributed
        self.pass_blanket_ids = pass_blanket_ids
        self.mode = mode

        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(
        self,
        x_t,
        t,
        date,
        pcgrad_orders=None,
        ppgf_order=None,
        ppgf_merge_weight=None,
    ):
        r"""Processes a trajectory blanket by blanket.

        Arguments:
            x_t: The input flattened trajectory with shape :math:`(B, T * D)`, with D the state size.
            t: The diffusion time tensor with shape :math:`()`.
            date: The date tensor of the trajectory used as a timestamp context, with shape :math:`(B, T, 4)`.
            pcgrad_orders: Optional global instrument-order matrix generated once per diffusion
                timestep and shared by every rank and blanket.
            ppgf_order: Optional global PPGF priority vector, or channel-by-source
                priority matrix for channel PPGF, generated once per diffusion timestep
                and shared by every rank and blanket.
            ppgf_merge_weight: Optional timestep-level interpolation weight between
                naive sum (zero) and full PPGF (one).

        Returns:
            The output denoised flat trajectory with shape :math:`(B, T * D)`.
        """

        B = x_t.shape[0]

        x_t, date = self.unfold(x_t, date)

        if self.distributed:
            x_t = x_t.tensor_split(self.world_size, dim=0)
            date = date.tensor_split(self.world_size, dim=0)
            sizes = [x.shape[0] for x in x_t]
            max_size = max(sizes)
            x_t, date = x_t[self.rank], date[self.rank]

        x_t_list = []
        for blanket_id, (x, d) in enumerate(
            zip(x_t.tensor_split(x_t.shape[0]), date.tensor_split(x_t.shape[0]))
        ):
            if self.pass_blanket_ids:
                kw = {}
                if pcgrad_orders is not None:
                    kw["pcgrad_orders"] = pcgrad_orders
                if ppgf_order is not None:
                    kw["ppgf_order"] = ppgf_order
                if ppgf_merge_weight is not None:
                    kw["ppgf_merge_weight"] = ppgf_merge_weight
                x_t_list.append(
                    self.denoiser(x, t, date=d, blanket_id=blanket_id, **kw)
                )
            else:
                x_t_list.append(self.denoiser(x, t, date=d))
        x_t = torch.cat(x_t_list)

        if self.distributed:
            x_t = torch.cat([
                x_t,
                x_t.new_full((max_size - x_t.shape[0], *x_t.shape[1:]), 0),
            ]).contiguous()
            x_ts = [torch.zeros_like(x_t) for _ in range(self.world_size)]
            dist.all_gather(x_ts, x_t)
            x_ts = [x[: sizes[i]] for i, x in enumerate(x_ts)]
            x_t = torch.cat(x_ts, dim=0)
        x_t = self.fold(x_t, batch_size=B)
        return x_t

    def unfold(self, x, d):
        x = rearrange(x, "B (T D) -> B T D", D=self.state_size)
        T = x.shape[1]

        assert (
            (T - self.blanket_size) % self.blanket_stride == 0
        ), f"K={self.blanket_size} and S={self.blanket_stride} do not fit in {T}."
        num_segments = (T - self.blanket_size) // self.blanket_stride + 1

        x = x.unfold(1, self.blanket_size, self.blanket_stride)
        x = rearrange(x, "B N D K -> (B N) (K D)", N=num_segments)

        d = d.unfold(1, self.blanket_size, self.blanket_stride)
        d = rearrange(d, "B N F K -> (B N) K F")

        return x, d

    def fold(self, x, batch_size):
        x = rearrange(x, "(B N) (K D) -> B N K D", B=batch_size, D=self.state_size)

        segments = []
        if x.shape[1] > 1:
            if self.mode == "symmetrical":
                i = (self.blanket_size - self.blanket_stride) // 2
                j = i + self.blanket_stride
                segments.append(x[:, 0, :j])
                segments.extend(x[:, 1:-1, i:j].unbind(1))
                segments.append(x[:, -1, i:])

                x = torch.cat(segments, dim=1)
            elif self.mode == "causal":
                i = self.blanket_size - self.blanket_stride
                segments.append(x[:, 0, :])
                segments.extend(x[:, 1:, i:].unbind(1))

                x = torch.cat(segments, dim=1)
            elif self.mode == "average":
                length = (x.shape[1] - 1) * self.blanket_stride + self.blanket_size
                accum_x = torch.zeros(x.shape[0], length, x.shape[-1])
                accum_count = torch.zeros(x.shape[0], length, 1)
                for i in range(x.shape[1]):
                    accum_x[
                        :, i * self.blanket_stride : i * self.blanket_stride + self.blanket_size
                    ] += x[:, i]
                    accum_count[
                        :, i * self.blanket_stride : i * self.blanket_stride + self.blanket_size
                    ] += 1

                x = accum_x / accum_count
            else:
                raise NotImplementedError(
                    f"Trajectory reconstruction mode {self.mode} not implemented."
                )
        else:
            x = x[:, 0]

        return rearrange(x, "B T D -> B (T D)")


class CondTrajectoryDenoiser(nn.Module):
    """TrajectoryDenoiser that also handles condition unfolding."""

    def __init__(
        self,
        denoiser: nn.Module,
        blanket_size: int,
        blanket_stride: int,
        state_size: int,
        cond_token_dim: int,
        distributed: bool = False,
        mode: str = "symmetrical",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.blanket_size = blanket_size
        self.blanket_stride = blanket_stride
        self.state_size = state_size
        self.cond_token_dim = cond_token_dim
        self.mode = mode
        self.distributed = distributed

        if distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, x_t, t, date, cond):
        """
        Args:
            x_t: (B, T * D)
            t: () or (B,)
            date: (B, T, 4)
            cond: (B, T, N, C)
        """
        B = x_t.shape[0]
        T = date.shape[1]
        num_segments = (T - self.blanket_size) // self.blanket_stride + 1

        # Unfold x_t and date
        x_t = rearrange(x_t, "B (T D) -> B T D", D=self.state_size)
        x_t = x_t.unfold(1, self.blanket_size, self.blanket_stride)
        x_t = rearrange(x_t, "B N D K -> (B N) (K D)", N=num_segments)

        date = date.unfold(1, self.blanket_size, self.blanket_stride)
        date = rearrange(date, "B N F K -> (B N) K F")

        # Unfold cond: (B, T, N, C) -> windows of blanket_size along T
        cond = cond.unfold(1, self.blanket_size, self.blanket_stride)
        # (B, num_segments, N, C, K) -> (B*N_seg, K, N, C)
        cond = rearrange(cond, "B N S C K -> (B N) K S C", N=num_segments)

        use_distributed = self.distributed and num_segments > self.world_size

        if use_distributed:
            x_t = x_t.tensor_split(self.world_size, dim=0)
            date = date.tensor_split(self.world_size, dim=0)
            cond = cond.tensor_split(self.world_size, dim=0)
            sizes = [x.shape[0] for x in x_t]
            max_size = max(sizes)
            x_t, date, cond = x_t[self.rank], date[self.rank], cond[self.rank]

        x_t_list = []
        for x, d, c in zip(
            x_t.tensor_split(x_t.shape[0]),
            date.tensor_split(date.shape[0]),
            cond.tensor_split(cond.shape[0]),
        ):
            x_t_list.append(self.denoiser(x, t, date=d, cond=c))
        x_t = torch.cat(x_t_list)

        if use_distributed:
            x_t = torch.cat([
                x_t,
                x_t.new_full((max_size - x_t.shape[0], *x_t.shape[1:]), 0),
            ]).contiguous()
            x_ts = [torch.zeros_like(x_t) for _ in range(self.world_size)]
            dist.all_gather(x_ts, x_t)
            x_ts = [x[: sizes[i]] for i, x in enumerate(x_ts)]
            x_t = torch.cat(x_ts, dim=0)

        # Fold back
        x_t = rearrange(x_t, "(B N) (K D) -> B N K D", B=B, D=self.state_size)
        segments = []
        if x_t.shape[1] > 1:
            if self.mode == "symmetrical":
                i = (self.blanket_size - self.blanket_stride) // 2
                j = i + self.blanket_stride
                segments.append(x_t[:, 0, :j])
                segments.extend(x_t[:, 1:-1, i:j].unbind(1))
                segments.append(x_t[:, -1, i:])
                x_t = torch.cat(segments, dim=1)
            elif self.mode == "causal":
                i = self.blanket_size - self.blanket_stride
                segments.append(x_t[:, 0, :])
                segments.extend(x_t[:, 1:, i:].unbind(1))
                x_t = torch.cat(segments, dim=1)
            else:
                x_t = x_t[:, 0]
        else:
            x_t = x_t[:, 0]

        return rearrange(x_t, "B T D -> B (T D)")


class DenoiserLoss(nn.Module):
    r"""Loss for a denoiser model.

    .. math:: \lambda_t || d(x_t) - x ||^2

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
    """

    def __init__(self, denoiser: Denoiser, std_min_penalty: float = 0):
        super().__init__()

        self.denoiser = denoiser
        self.std_min_penalty = std_min_penalty

    def forward(self, x: Tensor, sigma_t: Tensor, **kwargs) -> Tensor:
        r"""
        Arguments:
            x: The clean tensor, with shape :math:`(*, D)`.
            sigma_t: The noise std, with shape :math:`(*, 1)`.
            kwargs: Keyword arguments passed to the denoiser.

        Returns:
            The reduced denoising loss, with shape :math:`()`.
        """

        lmb_t = (1 / (sigma_t + self.std_min_penalty) ** 2 + 1).squeeze(-1)

        z = torch.randn_like(x)
        # sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # print(f'x shape {x.shape} sigma_t {sigma_t.shape} z {z.shape}')
        xt = x + sigma_t * z

        dxt = self.denoiser(xt, sigma_t, **kwargs)

        error = dxt - x

        return torch.mean(lmb_t * torch.mean(error**2, dim=-1))


class Schedule(nn.Module):
    r"""Base class for noise schedules.

    Arguments:
        sigma_min: The minimum noise std :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The maximum noise std :math:`\sigma_\max \in \mathbb{R}_+`.
        bfloat16: Whether to use bfloat16 precision or not (float32).
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e3, bfloat16: bool = False):
        super().__init__()

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

        self.t_max = 1 - 1 / 256 if bfloat16 else 1.0

        if bfloat16:
            print(
                "Warning: Using bfloat16 precision, "
                "please make sure to use schedule.sigma_tmax (t_max ≠ 1.0!)."
            )

    def sigma_tmax(self) -> Tensor:
        r"""Returns the maximum noise std :math:`\sigma_{t_{max}}`.

        Returns:
            The noise std :math:`\sigma_{t_{max}}`.
        """

        return self.forward(torch.tensor([self.t_max]))


class LogLinearSchedule(Schedule):
    r"""Log-linear noise schedule.

    .. math:: \sigma_t = \exp(\log(a) (1 - t) + \log(b) t)

    Arguments:
        a: The noise lower bound.
        b: The noise upper bound.
    """

    def forward(self, t: Tensor) -> Tensor:
        r"""
        Arguments:
            t: The schedule time, with shape :math:`(*)`.

        Returns:
            The noise std :math:`\sigma_t`, with shape :math:`(*, 1)`.
        """

        return torch.exp(
            self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * t
        ).unsqueeze(-1)


class LogLogitSchedule(Schedule):
    r"""Creates a log-logit noise schedule.

    .. math::
        \sigma_t & = \sqrt{\sigma_\min \sigma_\max} \exp(\rho \logit t)

    See also:
        :func:`torch.logit`

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
        spread: The spread factor :math:`\rho \in \mathbb{R}_+`.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        bfloat16: bool = False,
        spread: float = 2.0,
    ):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, bfloat16=bfloat16)

        self.eps = math.sqrt(sigma_min / sigma_max) ** (1 / spread)
        self.log_sigma_med = math.log(sigma_min * sigma_max) / 2
        self.spread = spread

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.spread * torch.logit(t * (1 - 2 * self.eps) + self.eps) + self.log_sigma_med
        ).unsqueeze(-1)


class RectifiedSchedule(Schedule):
    r"""Creates a rectified noise schedule.

    .. math::
        \sigma_t & = \frac{t + (1 - t) \sigma_\min}{t (\sigma_\max^{-1} - 1) + 1}

    Arguments:
        sigma_min: The initial noise scale :math:`\sigma_\min \in \mathbb{R}_+`.
        sigma_max: The final noise scale :math:`\sigma_\max \in \mathbb{R}_+`.
    """

    def forward(self, t: Tensor) -> Tensor:
        return ((t + (1 - t) * self.sigma_min) / (t * (1 / self.sigma_max - 1) + 1)).unsqueeze(-1)


def create_schedule(
    train_cfg: DictConfig,
    bfloat16: bool = False,
    device: str = "cuda",
) -> Schedule:
    r"""Create a noise schedule."""

    schedules = {
        "rectified": RectifiedSchedule,
        "loglogit": LogLogitSchedule,
        "loglinear": LogLinearSchedule,
    }

    noise_schedule = train_cfg.noise_schedule
    sigma_min = train_cfg.sigma_min
    sigma_max = train_cfg.sigma_max

    if noise_schedule not in schedules:
        raise ValueError(f"Invalid noise schedule: {noise_schedule}")

    return schedules[noise_schedule](
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        bfloat16=bfloat16,
    ).to(device)


def create_denoiser(
    diffusion_cfg: DictConfig,
    ae_cfg: DictConfig,
    distributed: bool = False,
    device: str = "cuda",
    overrides: dict = None,
):
    r"""Create a denoiser model with an SPTGraphDiT backbone.

    Arguments:
        diffusion_cfg: Diffusion configuration.
        ae_cfg: Autoencoder configuration.
        distributed: Whether to use distributed data parallel.
        device: Device to use.
        overrides: Optional overrides for the score network configuration.

    Returns:
        A denoiser model.
    """

    denoiser_cfg = diffusion_cfg.backbone

    if overrides:
        with open_dict(denoiser_cfg):
            for k in overrides.keys():
                denoiser_cfg[k] = overrides[k]

    if "graphdit" in denoiser_cfg.name:
        grid, _ = create_icosphere(ae_cfg.ae.ico_divisions[-1])

        if denoiser_cfg.self_attention_hops is None or denoiser_cfg.cross_attention_hops is None:
            self_edges = None
            cross_edges = None
        else:
            self_edges = icosphere_nhops_edges(grid, grid, denoiser_cfg.self_attention_hops)
            cross_edges = icosphere_nhops_edges(grid, grid, denoiser_cfg.cross_attention_hops)

        dit = SpatioTemporalGraphDiT(
            input_grid=grid,
            input_channels=diffusion_cfg.channels,
            self_edges=self_edges,
            cross_edges=cross_edges,
            **denoiser_cfg,
        ).to(device)
    elif "cond_vit" in denoiser_cfg.name:
        H, W, C = ae_cfg.ae.latent_shape
        dit = CondViDiT(
            in_channels=C,
            shape=(H, W),
            **denoiser_cfg,
        ).to(device)
    elif "vit" in denoiser_cfg.name:
        # assert "cae" in ae_cfg.ae.name, f"ViDiT not compatible with {ae_cfg.ae.name}"

        H, W, C = ae_cfg.ae.latent_shape
        dit = ViDiT(
            in_channels=C,
            shape=(H, W),
            **denoiser_cfg,
        ).to(device)
    else:
        raise NotImplementedError(f"Unknown denoiser type with name {denoiser_cfg.name}")

    if distributed:
        dit = torch.nn.parallel.DistributedDataParallel(module=dit, device_ids=[device])

    return Denoiser(dit).to(device)
