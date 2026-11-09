r"""ERA5-latent -> Satellite-latent mapper.

Mirrors the structure of `appa/nn/cae.py:ConvDecoder`:
  - 1x1 conv to project ERA5 latent channels to a base hidden width
  - per-stage [ResBlocks ... + Conv + Unpatchify(2x)] using PixelShuffle for
    learnable spatial upsampling
  - one final F.interpolate at the end of the trunk to land on the exact
    satellite latent shape (since the era<->sate ratio is not a power of 2)
  - 1x1 conv to project to the satellite latent channel count

For ERA5 latent (~23x47) -> ATMS latent (~180x362) the upsampling is ~7.83x in
H, which is between 2^2=4x and 2^3=8x. We use 3 PixelShuffle stages (8x total),
landing on (184, 376), then F.interpolate down to (180, 362). The bilinear step
is small (~0.96x) and only "fine-aligns" — virtually all of the spatial growth
is done by learnable PixelShuffle.
"""

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cae import ResBlock
from .layers import ConvNd, Unpatchify


class EraToSateNet(nn.Module):
    r"""Maps ERA5 latent (H_e, W_e, C_e) -> ATMS latent (H_s, W_s, C_s).

    Arguments:
        in_shape:  (H_e, W_e, C_e)  -- source ERA5 latent shape.
        out_shape: (H_s, W_s, C_s)  -- target satellite latent shape.
        hid_channels: list of hidden widths per stage. Each stage performs one
            PixelShuffle 2x upsampling (so total stride = 2 ** len(hid_channels))
            and reduces channels from hid_channels[i] to hid_channels[i+1] via
            the up-conv.
        hid_blocks:  number of ResBlocks per stage (matches len(hid_channels)).
        kernel_size: kernel size of all convs.
        ffn_factor:  FFN expansion factor inside each ResBlock.
        dropout:     dropout in ResBlock FFN.
        checkpointing: gradient-checkpoint each ResBlock.
        identity_init: init up-conv as identity-like (matches ConvDecoder).
    """

    def __init__(
        self,
        in_shape: Tuple[int, int, int],
        out_shape: Tuple[int, int, int],
        hid_channels: Sequence[int] = (256, 128, 64),
        hid_blocks: Sequence[int] = (2, 2, 2),
        kernel_size: int = 3,
        ffn_factor: int = 1,
        dropout: float = 0.0,
        checkpointing: bool = False,
        identity_init: bool = True,
        same_grid: bool = False,
        simple_act: bool = False,
        activation: str = "silu",
        use_norm: bool = True,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)
        self.same_grid = bool(same_grid)
        self.simple_act = bool(simple_act)
        self.use_norm = bool(use_norm)

        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        H_e, W_e, C_e = self.in_shape
        H_s, W_s, C_s = self.out_shape

        n_stages = len(hid_channels)
        self.n_stages = n_stages
        self.spatial = 2
        stride = (2, 2)
        self.stride = stride

        # Same-size mode: hid_channels=() → no PixelShuffle stages, a single
        # conv C_e -> C_s at the shared grid. Used when the ERA and sate latents
        # share a grid (e.g. the ERA-aligned sate AE, both 23x47x128), so the
        # mapper is a LINEAR per-pixel (kernel_size=1) or local-neighborhood
        # (kernel_size>1) op — its Jacobian is a CONSTANT matrix (no activation),
        # so MMPS stays an EXACT Kalman gain, matching o2e. kernel_size>1 lets a
        # sate-latent pixel read an ERA neighborhood (capacity-ladder step ②)
        # while keeping exact-KF; same-padding preserves the grid so out_proj is
        # a no-op and the final F.interpolate is skipped.
        if n_stages == 0:
            if (H_e, W_e) != (H_s, W_s):
                raise ValueError(
                    f"hid_channels=() (0-stage mode) requires equal ERA and sate "
                    f"latent grids, got in ({H_e},{W_e}) != out ({H_s},{W_s}). "
                    f"Use a non-empty hid_channels for upsampling."
                )
            self.pre_final_hw = (H_e, W_e)
            # Single conv does the whole C_e -> C_s map; out_proj is a no-op.
            # padding = k//2 (same-size); zeros mode matches the PixelShuffle
            # stages below. identity_init puts the +1 at the kernel CENTER so a
            # k>1 kernel still inits ≈ per-pixel identity + small neighborhood.
            self.in_proj = ConvNd(
                C_e, C_s, spatial=self.spatial, kernel_size=kernel_size,
                padding=kernel_size // 2, padding_mode="zeros",
                identity_init=bool(identity_init),
            )
            self.stages = nn.ModuleList()
            self.out_proj = nn.Identity()
            return

        # Same-grid NONLINEAR mode (capacity-ladder ⑤/⑥): equal ERA and sate
        # grids, a trunk of ResBlocks at the FIXED resolution (no PixelShuffle
        # upsampling). ResBlock carries LayerNorm+SiLU → the map is NONLINEAR →
        # its Jacobian is state-dependent → MMPS becomes an EKF (NOT the exact
        # KF the linear 0-stage mode gives — see [[e2s-mapper-capacity-vs-kf-alignment]]).
        # kernel_size selects the flavor: k=1 → per-pixel channel MLP (⑤,
        # channel-nonlinear only); k>1 → + spatial mixing (⑥).
        if self.same_grid:
            if (H_e, W_e) != (H_s, W_s):
                raise ValueError(
                    f"same_grid=True requires equal ERA and sate latent grids, "
                    f"got in ({H_e},{W_e}) != out ({H_s},{W_s})."
                )
            self.pre_final_hw = (H_e, W_e)
            kw = dict(
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode="zeros",
            )

            # Simplest-nonlinear mode (capacity-ladder "near-linear"): a SINGLE
            # activation between two same-size convs — conv(k) → act → conv(k).
            # Unlike the ResBlock trunk below (LayerNorm+SiLU+FFN+residual per
            # block = strong nonlinearity), there is exactly ONE nonlinearity, so
            # the map stays close to a linear op while its Jacobian is still
            # state-dependent → MMPS is a (mild) EKF, not the exact KF of the
            # 0-stage linear mode. identity_init makes both convs center-identity
            # kernels so the map starts at act(z) ≈ z (SiLU/GELU ≈ identity near
            # 0), i.e. it departs from linear only as training bends it.
            if self.simple_act:
                act_cls = {
                    "silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU,
                    "tanh": nn.Tanh,
                }[str(activation).lower()]
                hid = hid_channels[0] if len(hid_channels) else C_e
                self.in_proj = ConvNd(
                    C_e, hid, spatial=self.spatial,
                    identity_init=bool(identity_init), **kw,
                )
                # The activation is stored as a stage "block" so the generic
                # forward (in_proj → stages → out_proj) runs it unchanged.
                self.stages = nn.ModuleList([nn.ModuleList([act_cls()])])
                self.out_proj = ConvNd(
                    hid, C_s, spatial=self.spatial,
                    identity_init=bool(identity_init), **kw,
                )
                return

            # 1x1 lift C_e -> first hidden width.
            self.in_proj = ConvNd(
                C_e, hid_channels[0], spatial=self.spatial, kernel_size=1, padding=0
            )
            self.stages = nn.ModuleList()
            for i, (n_blocks, ch) in enumerate(zip(hid_blocks, hid_channels)):
                blocks = nn.ModuleList()
                for _ in range(n_blocks):
                    blocks.append(
                        ResBlock(
                            channels=ch,
                            ffn_factor=ffn_factor,
                            spatial=self.spatial,
                            dropout=dropout,
                            checkpointing=checkpointing,
                            use_norm=self.use_norm,
                            **kw,
                        )
                    )
                # Width change BETWEEN stages only (1x1, preserves grid); the
                # final stage's width is reduced to C_s by out_proj.
                if i + 1 < n_stages:
                    blocks.append(
                        ConvNd(
                            ch, hid_channels[i + 1], spatial=self.spatial,
                            kernel_size=1, padding=0,
                        )
                    )
                self.stages.append(blocks)
            self.out_proj = ConvNd(
                hid_channels[-1], C_s, spatial=self.spatial, kernel_size=1, padding=0
            )
            return

        # Total upsampling factor from PixelShuffle alone.
        total_up = 2 ** n_stages
        # Spatial size right before the final F.interpolate.
        self.pre_final_hw = (H_e * total_up, W_e * total_up)
        if (
            self.pre_final_hw[0] < H_s or self.pre_final_hw[1] < W_s
        ):
            raise ValueError(
                f"len(hid_channels)={n_stages} gives only {total_up}x upsampling, "
                f"target needs at least max({H_s/H_e:.2f}x, {W_s/W_e:.2f}x). "
                f"Pick hid_channels with a longer list."

            )

        kw = dict(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="zeros",
        )

        # Project input ERA5 channels to first hidden width.
        self.in_proj = ConvNd(
            C_e, hid_channels[0], spatial=self.spatial, kernel_size=1, padding=0
        )

        # Per-stage trunk: ResBlocks + (Conv + Unpatchify) PixelShuffle 2x.
        self.stages = nn.ModuleList()
        for i, (n_blocks, ch) in enumerate(zip(hid_blocks, hid_channels)):
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(
                    ResBlock(
                        channels=ch,
                        ffn_factor=ffn_factor,
                        spatial=self.spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kw,
                    )
                )

            # PixelShuffle 2x: produces channels for the next stage (or stays at
            # ch on the last stage so the final 1x1 only needs to reduce ch -> C_s).
            next_ch = hid_channels[i + 1] if i + 1 < n_stages else hid_channels[-1]
            blocks.append(
                nn.Sequential(
                    ConvNd(
                        ch,
                        next_ch * math.prod(stride),
                        spatial=self.spatial,
                        identity_init=identity_init,
                        **kw,
                    ),
                    Unpatchify(patch_size=stride),
                )
            )
            self.stages.append(blocks)

        # Final 1x1 to satellite-latent channel count.
        self.out_proj = ConvNd(
            hid_channels[-1], C_s, spatial=self.spatial, kernel_size=1, padding=0
        )

    def forward(self, z_era: torch.Tensor) -> torch.Tensor:
        r"""
        Arguments:
            z_era: ERA5 latent, shape (B, H_e * W_e, C_e).
        Returns:
            z_sate_pred: predicted ATMS latent, shape (B, H_s * W_s, C_s).
        """
        B = z_era.shape[0]
        H_e, W_e, C_e = self.in_shape
        H_s, W_s, C_s = self.out_shape

        x = z_era.reshape(B, H_e, W_e, C_e).permute(0, 3, 1, 2).contiguous()
        x = self.in_proj(x)

        for blocks in self.stages:
            for block in blocks:
                x = block(x)

        # x is now at (H_e * 2^n, W_e * 2^n) which is >= (H_s, W_s).
        # Final fine-align to the exact satellite latent shape.
        if (x.shape[-2], x.shape[-1]) != (H_s, W_s):
            x = F.interpolate(
                x, size=(H_s, W_s), mode="bilinear", align_corners=False
            )

        x = self.out_proj(x)  # (B, C_s, H_s, W_s)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H_s * W_s, C_s)
        return x
