r"""Diffusion helpers."""

import math
import torch
import torch.nn as nn

from abc import abstractmethod
from torch import Tensor
from torch import distributed as dist
from typing import Callable

from appa.diffusion import Schedule
from appa.math import gauss_legendre


class Sampler(nn.Module):
    r"""Base class for samplers, implementing the init API but leaving the forward to be implemetned by subclasses.

    .. math:: x_s = x_t - \tau (x_t - d(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau` is determined by the noise schedule.

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: Schedule,
        steps: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.schedule = schedule
        self.steps = steps

    @abstractmethod
    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from :math:`p(x_1)`, with shape :math:`(*, D)`.

        Returns:
            A data tensor from :math:`p(x_0 | x_1)`, with shape :math:`(*, D)`.
        """

        pass


class DDPMSampler(Sampler):
    r"""DDPM sampler for the reverse SDE.

    .. math:: x_s = x_t - \tau (x_t - d(x_t)) + \sigma_s \sqrt{\tau} \epsilon

    where :math:`\tau = 1 - \frac{\sigma_s^2}{\sigma_t^2}`.

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        silent: bool = True,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps)

        self.silent = silent

    def forward(self, x1: Tensor) -> Tensor:
        r"""
        Arguments:
            x1: A noise tensor from :math:`p(x_1)`, with shape :math:`(*, D)`.

        Returns:
            A data tensor from :math:`p(x_0 | x_1)`, with shape :math:`(*, D)`.
        """

        dt = torch.as_tensor(1 / self.steps, device=x1.device)
        time = torch.linspace(self.schedule.t_max, dt, self.steps, device=x1.device)

        xt = x1
        for i, t in enumerate(time):
            xt = self.step(xt, t, t - dt)

            if not self.silent:
                print(f"Diffusion step {i + 1}/{self.steps} done.", flush=True)

        return xt

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)
        tau = 1 - (sigma_s / sigma_t) ** 2
        eps = torch.randn_like(xt)

        return xt - tau * (xt - self.denoiser(xt, sigma_t)) + sigma_s * torch.sqrt(tau) * eps


class DDIMSampler(DDPMSampler):
    r"""DDIM sampler for the reverse SDE.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - d(x_t))

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.

    """

    def step(self, xt: Tensor, t: Tensor, s: Tensor) -> Tensor:
        sigma_s, sigma_t = self.schedule(s), self.schedule(t)

        return xt - (1 - sigma_s / sigma_t) * (xt - self.denoiser(xt, sigma_t))


class RewindDDIMSampler(DDIMSampler):
    r"""DDIM sampler for the reverse SDE with forward jumps.

    .. math:: x_s = x_t - (1 - \frac{\sigma_s}{\sigma_t}) (x_t - d(x_t))

    Arguments:
        rewind_steps: Number of inner denoising steps before forward diffusion rewind to the next outer step.
        jump: Whether to perform large jumps backward (true) or small steps.

    """

    def __init__(self, rewind_after: int = 16, jump: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.rewind_after = rewind_after
        self.jump = jump

    @torch.no_grad()
    def forward(self, x1: Tensor, **kwargs) -> Tensor:
        x_t = x1

        dt = torch.as_tensor(1 / self.steps, device=x1.device)

        for i, t in enumerate(torch.linspace(1, dt, self.steps)):
            kmax = min(self.rewind_after, int(t / dt))
            s = t - kmax * dt

            if self.jump:
                x_s = self.step(x_t, t, s, **kwargs)
            else:
                x_s = x_t
                for ti in torch.linspace(t, s + dt, kmax, device=x1.device):
                    x_s = self.step(x_s, ti, ti - dt, **kwargs)

            sigma_s = self.schedule(s)
            sigma_t = self.schedule(t - dt)

            B = torch.sqrt(sigma_t**2 - sigma_s**2)
            x_t = x_s + B * torch.randn_like(x_s)

            if not self.silent:
                print(f"Diffusion step {i + 1}/{self.steps} done.", flush=True)

        x = x_t

        return x


class LMSSampler(Sampler):
    r"""Creates a linear multi-step (LMS) sampler.

    References:
        | k-diffusion (Katherine Crowson)
        | https://github.com/crowsonkb/k-diffusion
        | azula library (François Rozet)
        | https://github.com/francois-rozet/azula

    Arguments:
        denoiser: A denoiser model :math:`d(x_t) \approx E[x | x_t]`.
        schedule: The noise schedule.
        steps: The number of sampling steps.
        order: The order of the multi-step method.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        order: int = 3,
        silent: bool = True,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps)

        self.order = order
        self.silent = silent

    @staticmethod
    def adams_bashforth(t: Tensor, i: int, order: int = 3) -> Tensor:
        r"""Returns the coefficients of the :math:`N`-th order Adams-Bashforth method.

        Wikipedia:
            https://wikipedia.org/wiki/Linear_multistep_method

        Arguments:
            t: The integration variable, with shape :math:`(T)`.
            i: The integration step.
            order: The method order :math:`N`.

        Returns:
            The Adams-Bashforth coefficients, with shape :math:`(N)`.
        """

        ti = t[i]
        tj = t[i - order : i]
        tk = torch.cat((tj, tj)).unfold(0, order, 1)[:order, 1:]
        tj_tk = tj[..., None] - tk

        # Lagrange basis
        def lj(t):
            return torch.prod((t[..., None, None] - tk) / tj_tk, dim=-1)

        # Adams-Bashforth coefficients
        cj = gauss_legendre(lj, tj[-1], ti, n=order // 2 + 1)

        return cj

    @torch.no_grad()
    def forward(self, x1: Tensor, **kwargs) -> Tensor:
        # Ok to go to 0 because enumerate(time[:-1])
        time = torch.linspace(self.schedule.t_max, 0, self.steps + 1, device=x1.device)

        sigmas = self.schedule(time).squeeze()
        ratio = sigmas.double()

        xt = x1

        derivatives = []

        for i, sigma_t in enumerate(sigmas[:-1]):
            q_t = self.denoiser(xt, sigma_t, **kwargs)
            z_t = (xt - q_t) / sigma_t

            derivatives.append(z_t)

            if len(derivatives) > self.order:
                derivatives.pop(0)

            coefficients = self.adams_bashforth(ratio, i + 1, order=len(derivatives))
            coefficients = coefficients.to(xt)

            delta = sum(c * d for c, d in zip(coefficients, derivatives, strict=True))

            xt = xt + delta

            if not self.silent:
                print(f"Diffusion step {i + 1}/{self.steps} done.", flush=True)

        x0 = xt

        return x0


class PCSampler(DDPMSampler):
    r"""Creates a predictor-corrector (PC) sampler.

    References:
        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        denoiser: A Gaussian denoiser.
        corrections: The number of Langevin corrections per step.
        delta: The amplitude of Langevin corrections.
        kwargs: Keyword arguments passed to :class:`Sampler`.
    """

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        corrections: int = 1,
        delta: float = 0.01,
        pcgrad_shuffle: bool = False,
        pcgrad_num_tasks: int = 0,
        pcgrad_seed: int = None,
        pcgrad_order_mode: str = "random",
        pcgrad_uncertainty: Tensor = None,
        pcgrad_uncertainty_strength: float = 1.0,
        order_log: bool = True,
        **kwargs,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps, **kwargs)

        self.denoiser = denoiser
        self.corrections = corrections
        self.pcgrad_shuffle = pcgrad_shuffle
        self.pcgrad_num_tasks = pcgrad_num_tasks
        self.pcgrad_order_mode = pcgrad_order_mode
        self.pcgrad_uncertainty_strength = float(pcgrad_uncertainty_strength)
        self.order_log = bool(order_log)
        self._pcgrad_order_step = 0
        if self.pcgrad_shuffle and self.pcgrad_num_tasks < 2:
            raise ValueError("Shuffled PCGrad requires at least two tasks")
        if self.pcgrad_order_mode not in {"random", "uncertainty", "fixed"}:
            raise ValueError(f"Unknown PCGrad order mode: {self.pcgrad_order_mode!r}")
        if self.pcgrad_uncertainty_strength < 0:
            raise ValueError("PCGrad uncertainty strength must be non-negative")

        self._pcgrad_fixed_order = None
        self._pcgrad_fixed_order_on_device = None
        if self.pcgrad_order_mode in {"uncertainty", "fixed"}:
            if pcgrad_uncertainty is None:
                mode_name = self.pcgrad_order_mode.capitalize()
                raise ValueError(
                    f"{mode_name} PCGrad ordering requires instrument uncertainties"
                )
            pcgrad_uncertainty = torch.as_tensor(
                pcgrad_uncertainty, dtype=torch.float32, device="cpu"
            )
            if self.pcgrad_order_mode == "uncertainty" and (
                pcgrad_uncertainty.shape != (self.pcgrad_num_tasks,)
            ):
                raise ValueError(
                    "Instrument uncertainty must have shape " f"({self.pcgrad_num_tasks},)"
                )
            if self.pcgrad_order_mode == "fixed" and not (
                pcgrad_uncertainty.shape == (self.pcgrad_num_tasks,)
                or (
                    pcgrad_uncertainty.ndim == 2
                    and pcgrad_uncertainty.shape[0] > 0
                    and pcgrad_uncertainty.shape[1] == self.pcgrad_num_tasks
                )
            ):
                raise ValueError(
                    "Fixed PCGrad uncertainty must have shape "
                    f"({self.pcgrad_num_tasks},) or "
                    f"(num_order_groups, {self.pcgrad_num_tasks})"
                )
            if not torch.isfinite(pcgrad_uncertainty).all() or not torch.all(
                pcgrad_uncertainty > 0
            ):
                raise ValueError("Instrument uncertainties must be finite and positive")
            self.pcgrad_uncertainty = pcgrad_uncertainty
            if self.pcgrad_order_mode == "fixed":
                # High-uncertainty sources are projected first, so the most
                # reliable sources enter last and have the final influence.
                fixed_order = torch.argsort(
                    pcgrad_uncertainty,
                    dim=-1,
                    descending=True,
                    stable=True,
                )
                if fixed_order.ndim == 1:
                    self._pcgrad_fixed_order = fixed_order.expand(
                        self.pcgrad_num_tasks, -1
                    ).clone()
                else:
                    self._pcgrad_fixed_order = fixed_order.unsqueeze(1).expand(
                        -1,
                        self.pcgrad_num_tasks,
                        -1,
                    ).clone()
        else:
            self.pcgrad_uncertainty = None

        self._pcgrad_generator = None
        if self.pcgrad_order_mode != "fixed":
            self._pcgrad_generator = torch.Generator(device="cpu")
            if pcgrad_seed is None:
                self._pcgrad_generator.seed()
            else:
                self._pcgrad_generator.manual_seed(int(pcgrad_seed))

        self.register_buffer("delta", torch.as_tensor(delta))

    def _sample_pcgrad_orders(self, device, timestep=None):
        if not self.pcgrad_shuffle:
            return None

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        shape = (self.pcgrad_num_tasks, self.pcgrad_num_tasks)
        if self.pcgrad_order_mode == "fixed":
            cached_order = self._pcgrad_fixed_order_on_device
            if cached_order is None or cached_order.device != torch.device(device):
                cached_order = self._pcgrad_fixed_order.to(device=device)
                self._pcgrad_fixed_order_on_device = cached_order
            orders = cached_order
        elif rank == 0:
            if self.pcgrad_order_mode == "random":
                orders = torch.stack([
                    torch.randperm(
                        self.pcgrad_num_tasks,
                        generator=self._pcgrad_generator,
                    )
                    for _ in range(self.pcgrad_num_tasks)
                ])
            else:
                log_uncertainty = torch.log(self.pcgrad_uncertainty)
                log_uncertainty = log_uncertainty - log_uncertainty.mean()
                orders = []
                for _ in range(self.pcgrad_num_tasks):
                    uniform = torch.rand(
                        self.pcgrad_num_tasks,
                        generator=self._pcgrad_generator,
                    ).clamp_(1e-7, 1 - 1e-7)
                    gumbel = -torch.log(-torch.log(uniform))
                    priority = self.pcgrad_uncertainty_strength * log_uncertainty + gumbel
                    orders.append(torch.argsort(priority, descending=True))
                orders = torch.stack(orders)
            orders = orders.to(device=device)
        else:
            orders = torch.empty(shape, dtype=torch.long, device=device)

        if distributed and self.pcgrad_order_mode != "fixed":
            dist.broadcast(orders, src=0)
        if timestep is not None:
            if rank == 0 and self.order_log:
                timestep_text = f", t={float(timestep.detach().cpu()):.6g}"
                if orders.ndim == 3:
                    sample = orders[: min(4, orders.shape[0]), 0]
                    order_text = (
                        f", spatial_order_groups={orders.shape[0]}, "
                        f"spatial_order_sample={sample.detach().cpu().tolist()}"
                    )
                else:
                    order_text = f", orders={orders.detach().cpu().tolist()}"
                print(
                    f"[PCGrad order] diffusion_step={self._pcgrad_order_step}"
                    f"{timestep_text}, mode={self.pcgrad_order_mode}"
                    f"{order_text}",
                    flush=True,
                )
            self._pcgrad_order_step += 1
        return orders

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        s: Tensor,
        generator: torch.Generator = None,
        **kwargs,
    ) -> Tensor:
        sigma_s = self.schedule(s)
        sigma_t = self.schedule(t)
        if self.pcgrad_shuffle:
            kwargs = {
                **kwargs,
                "pcgrad_orders": self._sample_pcgrad_orders(x_t.device, timestep=t),
            }

        # Corrector
        for _ in range(self.corrections):
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            noise = torch.randn(
                x_t.shape,
                dtype=x_t.dtype,
                device=x_t.device,
                generator=generator,
            )
            x_t = x_t + self.delta * (x_hat - x_t) + torch.sqrt(2 * self.delta) * sigma_t * noise

        # Predictor
        x_hat = self.denoiser(x_t, sigma_t, **kwargs)
        x_s = x_hat + sigma_s / sigma_t * (x_t - x_hat)

        return x_s


class PPGFSampler(DDPMSampler):
    """Predictor-corrector sampler with an independent PPGF priority process."""

    def __init__(
        self,
        denoiser: Callable[[Tensor, Tensor], Tensor],
        schedule: nn.Module = None,
        steps: int = 256,
        corrections: int = 1,
        delta: float = 0.01,
        ppgf_num_tasks: int = 0,
        ppgf_num_channels: int = 0,
        ppgf_channelwise: bool = False,
        ppgf_seed: int = None,
        ppgf_order_mode: str = "gumbel",
        ppgf_uncertainty: Tensor = None,
        ppgf_uncertainty_strength: float = 1.0,
        ppgf_uncertainty_strength_mode: str = "constant",
        ppgf_uncertainty_strength_start: float = 0.0,
        ppgf_uncertainty_strength_end: float = 10.0,
        ppgf_uncertainty_strength_ramp_steps: int = 16,
        ppgf_anneal_mode: str = "none",
        ppgf_anneal_sigma_high: float = 3.0,
        ppgf_anneal_sigma_low: float = 1.0,
        ppgf_anneal_full_steps: int = 16,
        ppgf_anneal_blend_steps: int = 32,
        ppgf_anneal_sum_steps: int = 16,
        order_log: bool = True,
        **kwargs,
    ):
        super().__init__(denoiser=denoiser, schedule=schedule, steps=steps, **kwargs)

        self.corrections = corrections
        self.ppgf_num_tasks = int(ppgf_num_tasks)
        self.ppgf_num_channels = int(ppgf_num_channels)
        self.ppgf_channelwise = bool(ppgf_channelwise)
        self.ppgf_order_mode = str(ppgf_order_mode)
        self.ppgf_uncertainty_strength = float(ppgf_uncertainty_strength)
        self.ppgf_uncertainty_strength_mode = str(ppgf_uncertainty_strength_mode).lower()
        self.ppgf_uncertainty_strength_start = float(ppgf_uncertainty_strength_start)
        self.ppgf_uncertainty_strength_end = float(ppgf_uncertainty_strength_end)
        self.ppgf_uncertainty_strength_ramp_steps = int(ppgf_uncertainty_strength_ramp_steps)
        self.ppgf_anneal_mode = str(ppgf_anneal_mode).lower()
        self.ppgf_anneal_sigma_high = float(ppgf_anneal_sigma_high)
        self.ppgf_anneal_sigma_low = float(ppgf_anneal_sigma_low)
        self.ppgf_anneal_full_steps = int(ppgf_anneal_full_steps)
        self.ppgf_anneal_blend_steps = int(ppgf_anneal_blend_steps)
        self.ppgf_anneal_sum_steps = int(ppgf_anneal_sum_steps)
        self.order_log = bool(order_log)
        if self.ppgf_channelwise and self.ppgf_num_channels < 1:
            raise ValueError("Channel PPGF requires a positive channel count")
        self._ppgf_order_shape = (
            (self.ppgf_num_channels, self.ppgf_num_tasks)
            if self.ppgf_channelwise
            else (self.ppgf_num_tasks,)
        )
        self._ppgf_order_step = 0
        if self.ppgf_num_tasks < 2:
            raise ValueError("PPGF requires at least two tasks")
        if self.ppgf_order_mode not in {"random", "gumbel", "fixed"}:
            raise ValueError(f"Unknown PPGF order mode: {self.ppgf_order_mode!r}")
        if not math.isfinite(self.ppgf_uncertainty_strength) or self.ppgf_uncertainty_strength < 0:
            raise ValueError("PPGF uncertainty strength must be finite and non-negative")
        if self.ppgf_uncertainty_strength_mode not in {"constant", "linear_steps"}:
            raise ValueError(
                "Unknown PPGF uncertainty strength mode: "
                f"{self.ppgf_uncertainty_strength_mode!r}"
            )
        if self.ppgf_uncertainty_strength_mode == "linear_steps":
            if self.ppgf_order_mode != "gumbel":
                raise ValueError("Linear PPGF uncertainty strength requires gumbel ordering")
            if not (
                math.isfinite(self.ppgf_uncertainty_strength_start)
                and self.ppgf_uncertainty_strength_start >= 0.0
                and math.isfinite(self.ppgf_uncertainty_strength_end)
                and self.ppgf_uncertainty_strength_end >= 0.0
            ):
                raise ValueError(
                    "PPGF uncertainty strength endpoints must be finite and non-negative"
                )
            if not 1 <= self.ppgf_uncertainty_strength_ramp_steps <= self.steps:
                raise ValueError(
                    "PPGF uncertainty strength ramp steps must be in " f"[1, {self.steps}]"
                )
        if self.ppgf_anneal_mode not in {"none", "cosine", "cosine_steps"}:
            raise ValueError(f"Unknown PPGF anneal mode: {self.ppgf_anneal_mode!r}")
        if not (
            math.isfinite(self.ppgf_anneal_sigma_low)
            and math.isfinite(self.ppgf_anneal_sigma_high)
            and 0.0 < self.ppgf_anneal_sigma_low < self.ppgf_anneal_sigma_high
        ):
            raise ValueError(
                "PPGF anneal sigma thresholds must satisfy "
                "0 < sigma_low < sigma_high and both must be finite"
            )
        if self.ppgf_anneal_mode == "cosine_steps":
            step_counts = (
                self.ppgf_anneal_full_steps,
                self.ppgf_anneal_blend_steps,
                self.ppgf_anneal_sum_steps,
            )
            if any(value < 0 for value in step_counts):
                raise ValueError("PPGF anneal step counts must be non-negative")
            if sum(step_counts) != self.steps:
                raise ValueError(
                    "PPGF anneal step counts must sum to sampler steps: "
                    f"{step_counts} != {self.steps}"
                )

        self._ppgf_fixed_order = None
        self._ppgf_fixed_order_on_device = None
        self._ppgf_anneal_step = 0
        if self.ppgf_order_mode in {"gumbel", "fixed"}:
            if ppgf_uncertainty is None:
                mode_name = self.ppgf_order_mode.capitalize()
                raise ValueError(f"{mode_name} PPGF requires instrument uncertainties")
            uncertainty = torch.as_tensor(
                ppgf_uncertainty,
                dtype=torch.float32,
                device="cpu",
            )
            if tuple(uncertainty.shape) != self._ppgf_order_shape:
                raise ValueError("PPGF uncertainty must have shape " f"{self._ppgf_order_shape}")
            if not torch.isfinite(uncertainty).all() or not torch.all(uncertainty > 0):
                raise ValueError("PPGF uncertainties must be finite and positive")
            self.ppgf_uncertainty = uncertainty
            if self.ppgf_order_mode == "fixed":
                self._ppgf_fixed_order = torch.argsort(
                    uncertainty,
                    dim=-1,
                    descending=True,
                    stable=True,
                )
        else:
            self.ppgf_uncertainty = None

        self._ppgf_generator = None
        if self.ppgf_order_mode != "fixed":
            self._ppgf_generator = torch.Generator(device="cpu")
            if ppgf_seed is None:
                self._ppgf_generator.seed()
            else:
                self._ppgf_generator.manual_seed(int(ppgf_seed))

        self.register_buffer("delta", torch.as_tensor(delta))

    def _ppgf_uncertainty_strength_at_step(self, diffusion_step=None):
        """Return the Gumbel uncertainty strength for one diffusion timestep."""
        if self.ppgf_uncertainty_strength_mode == "constant":
            return self.ppgf_uncertainty_strength

        step = self._ppgf_anneal_step if diffusion_step is None else int(diffusion_step)
        if not 0 <= step < self.steps:
            raise ValueError(
                f"PPGF uncertainty strength diffusion step must be in [0, {self.steps}); "
                f"got {step}"
            )
        if self.ppgf_uncertainty_strength_ramp_steps == 1:
            return self.ppgf_uncertainty_strength_end
        if step == 0:
            return self.ppgf_uncertainty_strength_start
        if step >= self.ppgf_uncertainty_strength_ramp_steps - 1:
            return self.ppgf_uncertainty_strength_end

        fraction = step / (self.ppgf_uncertainty_strength_ramp_steps - 1)
        return (
            self.ppgf_uncertainty_strength_start
            + (self.ppgf_uncertainty_strength_end - self.ppgf_uncertainty_strength_start)
            * fraction
        )

    def _ppgf_merge_weight(self, sigma_t, diffusion_step=None):
        """Interpolate from full PPGF at high sigma to exact sum at low sigma."""
        if self.ppgf_anneal_mode == "none":
            return 1.0
        if self.ppgf_anneal_mode == "cosine_steps":
            step = self._ppgf_anneal_step if diffusion_step is None else int(diffusion_step)
            if not 0 <= step < self.steps:
                raise ValueError(
                    f"PPGF anneal diffusion step must be in [0, {self.steps}); got {step}"
                )
            if step < self.ppgf_anneal_full_steps:
                return 1.0
            blend_stop = self.ppgf_anneal_full_steps + self.ppgf_anneal_blend_steps
            if step >= blend_stop:
                return 0.0
            blend_position = step - self.ppgf_anneal_full_steps
            blend_fraction = (blend_position + 1) / (self.ppgf_anneal_blend_steps + 1)
            return 0.5 + 0.5 * math.cos(math.pi * blend_fraction)

        sigma = float(torch.as_tensor(sigma_t).detach().float().reshape(-1)[0].cpu())
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("PPGF annealing requires a finite non-negative sigma")
        if sigma >= self.ppgf_anneal_sigma_high:
            return 1.0
        if sigma <= self.ppgf_anneal_sigma_low:
            return 0.0

        log_fraction = (math.log(sigma) - math.log(self.ppgf_anneal_sigma_low)) / (
            math.log(self.ppgf_anneal_sigma_high) - math.log(self.ppgf_anneal_sigma_low)
        )
        return 0.5 - 0.5 * math.cos(math.pi * log_fraction)

    def forward(self, x1: Tensor) -> Tensor:
        self._ppgf_order_step = 0
        self._ppgf_anneal_step = 0
        return super().forward(x1)

    def _sample_ppgf_order(
        self,
        device,
        timestep=None,
        merge_weight=None,
        uncertainty_strength=None,
    ):
        if uncertainty_strength is None:
            uncertainty_strength = self._ppgf_uncertainty_strength_at_step()
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if self.ppgf_order_mode == "fixed":
            cached_order = self._ppgf_fixed_order_on_device
            if cached_order is None or cached_order.device != torch.device(device):
                cached_order = self._ppgf_fixed_order.to(device=device)
                self._ppgf_fixed_order_on_device = cached_order
            order = cached_order
        elif rank == 0:
            if self.ppgf_order_mode == "random":
                if self.ppgf_channelwise:
                    order = torch.stack([
                        torch.randperm(
                            self.ppgf_num_tasks,
                            generator=self._ppgf_generator,
                        )
                        for _ in range(self.ppgf_num_channels)
                    ])
                else:
                    order = torch.randperm(
                        self.ppgf_num_tasks,
                        generator=self._ppgf_generator,
                    )
            else:
                log_uncertainty = torch.log(self.ppgf_uncertainty)
                log_uncertainty = log_uncertainty - log_uncertainty.mean(dim=-1, keepdim=True)
                uniform = torch.rand(
                    self._ppgf_order_shape,
                    generator=self._ppgf_generator,
                ).clamp_(1e-7, 1 - 1e-7)
                gumbel = -torch.log(-torch.log(uniform))
                priority = uncertainty_strength * log_uncertainty + gumbel
                order = torch.argsort(priority, dim=-1, descending=True)
            order = order.to(device=device)
        else:
            order = torch.empty(
                self._ppgf_order_shape,
                dtype=torch.long,
                device=device,
            )

        if distributed and self.ppgf_order_mode != "fixed":
            dist.broadcast(order, src=0)
        if timestep is not None:
            if rank == 0 and self.order_log:
                timestep_text = f", t={float(timestep.detach().cpu()):.6g}"
                merge_text = (
                    "" if merge_weight is None else f", merge_weight={float(merge_weight):.6g}"
                )
                strength_text = (
                    f", uncertainty_strength={float(uncertainty_strength):.6g}"
                    if self.ppgf_order_mode == "gumbel"
                    else ""
                )
                if self.ppgf_channelwise:
                    sample = order[: min(4, self.ppgf_num_channels)]
                    order_text = f"channel_order_sample={sample.detach().cpu().tolist()}"
                else:
                    order_text = f"order={order.detach().cpu().tolist()}"
                print(
                    f"[PPGF order] diffusion_step={self._ppgf_order_step}"
                    f"{timestep_text}, mode={self.ppgf_order_mode}"
                    f"{strength_text}{merge_text}, "
                    f"{order_text}",
                    flush=True,
                )
            self._ppgf_order_step += 1
        return order

    def step(
        self,
        x_t: Tensor,
        t: Tensor,
        s: Tensor,
        generator: torch.Generator = None,
        ppgf_order: Tensor = None,
        **kwargs,
    ) -> Tensor:
        sigma_s = self.schedule(s)
        sigma_t = self.schedule(t)
        # Sample once at timestep scope; all correctors and the predictor reuse it.
        ppgf_merge_weight = self._ppgf_merge_weight(
            sigma_t,
            diffusion_step=self._ppgf_anneal_step,
        )
        ppgf_uncertainty_strength = self._ppgf_uncertainty_strength_at_step(
            diffusion_step=self._ppgf_anneal_step,
        )
        # TrajectoryDenoiser forwards this same vector to every local blanket.
        if ppgf_order is None:
            ppgf_order = self._sample_ppgf_order(
                x_t.device,
                timestep=t,
                merge_weight=(ppgf_merge_weight if self.ppgf_anneal_mode != "none" else None),
                uncertainty_strength=ppgf_uncertainty_strength,
            )
        kwargs = {
            **kwargs,
            "ppgf_order": ppgf_order,
        }
        if self.ppgf_anneal_mode != "none":
            kwargs["ppgf_merge_weight"] = ppgf_merge_weight

        for _ in range(self.corrections):
            x_hat = self.denoiser(x_t, sigma_t, **kwargs)
            if generator is None:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.randn(
                    x_t.shape,
                    dtype=x_t.dtype,
                    device=x_t.device,
                    generator=generator,
                )
            x_t = x_t + self.delta * (x_hat - x_t) + torch.sqrt(2 * self.delta) * sigma_t * noise

        x_hat = self.denoiser(x_t, sigma_t, **kwargs)
        x_s = x_hat + sigma_s / sigma_t * (x_t - x_hat)
        self._ppgf_anneal_step += 1
        return x_s


def select_sampler(name: str):
    r"""Return a sampler class based on a short name.

    Arguments:
        name: The name of the sampler, one of "pc", "ddpm", "ddim", "rewind", or "lms".

    Returns:
        A sampler class corresponding to the given name.
    """

    if name == "pc":
        return PCSampler
    elif name == "ddpm":
        return DDPMSampler
    elif name == "ddim":
        return DDIMSampler
    elif name == "rewind":
        return RewindDDIMSampler
    elif name == "lms":
        return LMSSampler
    else:
        raise ValueError(
            f"Unknown sampler type: {name}. Options are 'pc', 'ddpm', 'ddim', 'rewind', and 'lms'."
        )
