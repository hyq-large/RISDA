r"""Conv Auto-Encoder (CAE) building blocks."""

__all__ = [
    "ConvEncoder",
    "ConvDecoder",
    "ConvAE",
    "conv_ae",
]

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Sequence, Tuple, Union

from appa.nn import triggers
from appa.nn.swin import SwinBlock, to_2tuple

from .layers import (
    ConvNd,
    LayerNorm,
    Patchify,
    Unpatchify,
)


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Creates a residual block module.

    Arguments:
        channels: The number of channels :math:`C`.
        ffn_factor: The channel factor in the FFN.
        spatial: The number of spatial dimensions :math:`N`.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        channels: int,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        use_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.checkpointing = checkpointing

        # Norm (use_norm=False replaces it with identity — for ablating LayerNorm
        # in the e2s mapper without touching any CAE encoder/decoder that relies
        # on it; default True keeps every existing caller byte-identical).
        self.norm = LayerNorm(dim=-spatial - 1) if use_norm else nn.Identity()

        # FFN
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        self.ffn[-1].weight.data.mul_(1e-2)

    def _forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C, L_1, ..., L_N)`.
        """

        y = self.norm(x)
        y = self.ffn(y)

        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SwinResBlock(nn.Module):
    r"""Adapter wrapping SwinBlock to have the same [B, C, H, W] interface as ResBlock."""

    def __init__(
        self,
        channels: int,
        input_size: Tuple[int, int],
        num_heads: int = 4,
        window_size: Union[int, Sequence[int]] = (5, 8),
        shift_size: Union[int, Sequence[int]] = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.checkpointing = checkpointing
        window_size = to_2tuple(window_size)
        shift_size = to_2tuple(shift_size)

        self.block = SwinBlock(
            dim=channels,
            input_size=input_size,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )

    def _forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


def _make_blocks(
    num_blocks: int,
    channels: int,
    block_type: str = "resblock",
    input_size: Optional[Tuple[int, int]] = None,
    num_heads: int = 4,
    window_size: Union[int, Sequence[int]] = (5, 8),
    mlp_ratio: float = 4.0,
    drop_path: float = 0.0,
    ffn_factor: int = 1,
    spatial: int = 2,
    dropout: Optional[float] = None,
    checkpointing: bool = False,
    **conv_kwargs,
) -> nn.ModuleList:
    r"""Factory to create a list of blocks (ResBlock or SwinResBlock)."""
    blocks = nn.ModuleList()
    window_size_tuple = to_2tuple(window_size)

    for i in range(num_blocks):
        if block_type == "swin":
            shift = (0, 0) if i % 2 == 0 else (window_size_tuple[0] // 2, window_size_tuple[1] // 2)
            blocks.append(
                SwinResBlock(
                    channels=channels,
                    input_size=input_size,
                    num_heads=num_heads,
                    window_size=window_size_tuple,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    checkpointing=checkpointing,
                )
            )
        else:
            blocks.append(
                ResBlock(
                    channels,
                    ffn_factor=ffn_factor,
                    spatial=spatial,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **conv_kwargs,
                )
            )
    return blocks


class MultiBranchBlock(nn.Module):
    r"""Multi-branch residual block with independent parameters per branch.

    Arguments:
        channels: The working channel width for each branch.
        num_branches: Number of parallel branches (excluding global branch).
        num_blocks: Number of ResBlocks per branch.
        ffn_factor: Channel expansion factor in ResBlock FFN.
        spatial: Number of spatial dimensions.
        input_channels: Per-branch input channel list (for level 0 variable-group split).
            If None and is_first_level=True, uses independent projection.
            If None and is_first_level=False, chunks the input into branches directly.
        global_branch: Whether to add an extra branch that sees the full input.
        global_input_channels: Input channels for the global branch at level 0.
            If None, uses the same as other branches (channels).
        is_first_level: Whether this is the first level (needs input projections).
        dropout: Dropout rate.
        checkpointing: Whether to use gradient checkpointing.
        identity_init: Whether to use identity-like initialization for projections.
        kwargs: Conv keyword arguments (kernel_size, padding, etc.).
    """

    def __init__(
        self,
        channels: int,
        num_branches: int,
        num_blocks: int,
        ffn_factor: int = 1,
        spatial: int = 2,
        input_channels: Optional[Sequence[int]] = None,
        global_branch: bool = False,
        global_input_channels: Optional[int] = None,
        is_first_level: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
        block_type: str = "resblock",
        input_size: Optional[Tuple[int, int]] = None,
        num_heads: int = 4,
        window_size: Union[int, Sequence[int]] = (5, 8),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.num_branches = num_branches
        self.input_channels = input_channels
        self.global_branch = global_branch
        self.is_first_level = is_first_level
        self.total_branches = num_branches + (1 if global_branch else 0)

        if input_channels is not None:
            self.input_projs = nn.ModuleList([
                ConvNd(in_ch, channels, spatial=spatial, **kwargs)
                for in_ch in input_channels
            ])
        elif is_first_level:
            self.input_projs = nn.ModuleList([
                ConvNd(channels, channels, spatial=spatial, identity_init=identity_init, **kwargs)
                for _ in range(num_branches)
            ])
        else:
            self.input_projs = None

        block_kwargs = dict(
            block_type=block_type, input_size=input_size, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, drop_path=drop_path,
            ffn_factor=ffn_factor, spatial=spatial, dropout=dropout,
            checkpointing=checkpointing, **kwargs,
        )

        if global_branch:
            if global_input_channels is not None:
                self.global_proj = ConvNd(
                    global_input_channels, channels, spatial=spatial, **kwargs
                )
            elif is_first_level:
                self.global_proj = ConvNd(
                    channels, channels, spatial=spatial, identity_init=identity_init, **kwargs
                )
            else:
                self.global_proj = None
            self.global_blocks = _make_blocks(num_blocks, channels, **block_kwargs)

        self.branches = nn.ModuleList([
            _make_blocks(num_blocks, channels, **block_kwargs)
            for _ in range(num_branches)
        ])

        self.merge = ConvNd(
            self.total_branches * channels, self.total_branches * channels,
            spatial=spatial, identity_init=identity_init, **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.input_channels is not None:
            chunks = x.split(self.input_channels, dim=1)
        elif self.input_projs is None:
            chunks = x.chunk(self.total_branches, dim=1)
        else:
            chunks = [x] * self.num_branches

        outs = []
        if self.input_projs is not None:
            for proj, blocks, chunk in zip(self.input_projs, self.branches, chunks):
                h = proj(chunk)
                for block in blocks:
                    h = block(h)
                outs.append(h)
        else:
            for i, blocks in enumerate(self.branches):
                h = chunks[i]
                for block in blocks:
                    h = block(h)
                outs.append(h)

        if self.global_branch:
            if self.global_proj is not None:
                h = self.global_proj(x)
            else:
                h = chunks[-1]
            for block in self.global_blocks:
                h = block(h)
            outs.append(h)

        return self.merge(torch.cat(outs, dim=1))


class TruncateChannels(nn.Module):
    r"""Truncates the input tensor to keep only the first `channels` channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :self.channels]


class PadChannels(nn.Module):
    r"""Zero-pads the input tensor to reach `target_channels` channels."""

    def __init__(self, target_channels: int):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, x: Tensor) -> Tensor:
        pad_size = self.target_channels - x.shape[1]
        if pad_size > 0:
            padding = x.new_zeros(x.shape[0], pad_size, *x.shape[2:])
            return torch.cat([x, padding], dim=1)
        return x


class MultiBranchOutput(nn.Module):
    r"""Per-branch output projection for decoder final level.

    Takes the full total_branches * channels input, chunks into branches,
    and applies per-branch Conv to produce variable-group-sized outputs.
    The global branch chunk is discarded (it has no corresponding output channels).
    """

    def __init__(
        self,
        channels: int,
        output_channels: Sequence[int],
        num_heads: int = 6,
        spatial: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_output_branches = len(output_channels)
        self.output_projs = nn.ModuleList([
            ConvNd(channels, out_ch, spatial=spatial, **kwargs)
            for out_ch in output_channels
        ])

    def forward(self, x: Tensor) -> Tensor:
        chunks = x.chunk(self.num_heads, dim=1)
        outs = []
        for i, proj in enumerate(self.output_projs):
            outs.append(proj(chunks[i]))
        return torch.cat(outs, dim=1)


class ConvEncoder(nn.Module):
    r"""Creates an conv encoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
        channel_groups: Optional list of per-variable-group channel counts (e.g. [13,13,13,13,13,4]).
            Enables multi-branch architecture when provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
        channel_groups: Optional[Sequence[int]] = None,
        global_branch: bool = False,
        block_type: str = "resblock",
        padded_size: Optional[Tuple[int, int]] = None,
        num_heads: int = 4,
        window_size: Union[int, Sequence[int]] = (5, 8),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.patch = Patchify(patch_size=patch_size)
        self.channel_groups = channel_groups
        self.descent = nn.ModuleList()

        if channel_groups is not None:
            num_branches = len(channel_groups)
            total_branches = num_branches + (1 if global_branch else 0)
            input_channels_after_patch = [ch * math.prod(patch_size) for ch in channel_groups]
            global_in_ch = sum(input_channels_after_patch) if global_branch else None

            for i, num_blocks in enumerate(hid_blocks):
                blocks = nn.ModuleList()
                level_size = (padded_size[0] // (stride[0] ** i),
                              padded_size[1] // (stride[1] ** i)) if padded_size else None

                if i > 0:
                    prev_total_ch = total_branches * hid_channels[i - 1]
                    if pixel_shuffle:
                        blocks.append(
                            nn.Sequential(
                                Patchify(patch_size=stride),
                                ConvNd(
                                    prev_total_ch * math.prod(stride),
                                    total_branches * hid_channels[i],
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                            )
                        )
                    else:
                        blocks.append(
                            ConvNd(
                                prev_total_ch,
                                total_branches * hid_channels[i],
                                spatial=spatial,
                                stride=stride,
                                identity_init=identity_init,
                                **kwargs,
                            )
                        )

                blocks.append(
                    MultiBranchBlock(
                        channels=hid_channels[i],
                        num_branches=num_branches,
                        num_blocks=num_blocks,
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        input_channels=input_channels_after_patch if i == 0 else None,
                        global_branch=global_branch,
                        global_input_channels=global_in_ch if i == 0 else None,
                        is_first_level=(i == 0),
                        dropout=dropout,
                        checkpointing=checkpointing,
                        identity_init=identity_init,
                        block_type=block_type,
                        input_size=level_size,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                        **kwargs,
                    )
                )

                if i + 1 == len(hid_blocks):
                    blocks.append(
                        ConvNd(
                            total_branches * hid_channels[i],
                            out_channels,
                            spatial=spatial,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

                self.descent.append(blocks)
        else:
            for i, num_blocks in enumerate(hid_blocks):
                blocks = nn.ModuleList()
                level_size = (padded_size[0] // (stride[0] ** i),
                              padded_size[1] // (stride[1] ** i)) if padded_size else None

                if i > 0:
                    if pixel_shuffle:
                        blocks.append(
                            nn.Sequential(
                                Patchify(patch_size=stride),
                                ConvNd(
                                    hid_channels[i - 1] * math.prod(stride),
                                    hid_channels[i],
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                            )
                        )
                    else:
                        blocks.append(
                            ConvNd(
                                hid_channels[i - 1],
                                hid_channels[i],
                                spatial=spatial,
                                stride=stride,
                                identity_init=identity_init,
                                **kwargs,
                            )
                        )
                else:
                    blocks.append(
                        ConvNd(
                            math.prod(patch_size) * in_channels,
                            hid_channels[i],
                            spatial=spatial,
                            **kwargs,
                        )
                    )

                for blk in _make_blocks(
                    num_blocks, hid_channels[i],
                    block_type=block_type, input_size=level_size,
                    num_heads=num_heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, drop_path=drop_path,
                    ffn_factor=ffn_factor, spatial=spatial,
                    dropout=dropout, checkpointing=checkpointing, **kwargs,
                ):
                    blocks.append(blk)

                if i + 1 == len(hid_blocks):
                    blocks.append(
                        ConvNd(
                            hid_channels[i],
                            out_channels,
                            spatial=spatial,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

                self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 / 2^D, ..., L_N  / 2^D)`.
        """

        x = self.patch(x)

        for blocks in self.descent:
            for block in blocks:
                x = block(x)

        return x


class ConvDecoder(nn.Module):
    r"""Creates a conv decoder module.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        hid_channels: The numbers of channels at each depth.
        hid_blocks: The numbers of hidden blocks at each depth.
        kernel_size: The kernel size of all convolutions.
        stride: The stride of the downsampling convolutions.
        pixel_shuffle: Whether to use pixel shuffling or not.
        spatial: The number of spatial dimensions.
        periodic: Whether the spatial dimensions are periodic or not.
        dropout: The dropout rate in :math:`[0, 1]`.
        checkpointing: Whether to use gradient checkpointing or not.
        identity_init: Initialize down/upsampling convolutions as identity.
        channel_groups: Optional list of per-variable-group channel counts.
            Enables multi-branch architecture when provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
        channel_groups: Optional[Sequence[int]] = None,
        global_branch: bool = False,
        block_type: str = "resblock",
        padded_size: Optional[Tuple[int, int]] = None,
        num_heads: int = 4,
        window_size: Union[int, Sequence[int]] = (5, 8),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial

        if isinstance(stride, int):
            stride = [stride] * spatial

        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.unpatch = Unpatchify(patch_size=patch_size)
        self.channel_groups = channel_groups
        self.ascent = nn.ModuleList()

        if channel_groups is not None:
            num_branches = len(channel_groups)
            total_branches = num_branches + (1 if global_branch else 0)
            output_channels_before_unpatch = [ch * math.prod(patch_size) for ch in channel_groups]

            for i, num_blocks in reversed(list(enumerate(hid_blocks))):
                blocks = nn.ModuleList()
                cur_total_ch = total_branches * hid_channels[i]
                level_size = (padded_size[0] // (stride[0] ** i),
                              padded_size[1] // (stride[1] ** i)) if padded_size else None

                if i + 1 == len(hid_blocks):
                    blocks.append(
                        ConvNd(
                            in_channels,
                            cur_total_ch,
                            spatial=spatial,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

                blocks.append(
                    MultiBranchBlock(
                        channels=hid_channels[i],
                        num_branches=num_branches,
                        num_blocks=num_blocks,
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        input_channels=None,
                        global_branch=global_branch,
                        global_input_channels=None,
                        is_first_level=False,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        identity_init=identity_init,
                        block_type=block_type,
                        input_size=level_size,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path,
                        **kwargs,
                    )
                )

                if i > 0:
                    prev_total_ch = total_branches * hid_channels[i - 1]
                    if pixel_shuffle:
                        blocks.append(
                            nn.Sequential(
                                ConvNd(
                                    cur_total_ch,
                                    prev_total_ch * math.prod(stride),
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                                Unpatchify(patch_size=stride),
                            )
                        )
                    else:
                        blocks.append(
                            nn.Sequential(
                                nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                                ConvNd(
                                    cur_total_ch,
                                    prev_total_ch,
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                            )
                        )
                else:
                    blocks.append(
                        MultiBranchOutput(
                            channels=hid_channels[i],
                            output_channels=output_channels_before_unpatch,
                            num_heads=total_branches,
                            spatial=spatial,
                            **kwargs,
                        )
                    )

                self.ascent.append(blocks)
        else:
            for i, num_blocks in reversed(list(enumerate(hid_blocks))):
                blocks = nn.ModuleList()
                level_size = (padded_size[0] // (stride[0] ** i),
                              padded_size[1] // (stride[1] ** i)) if padded_size else None

                if i + 1 == len(hid_blocks):
                    blocks.append(
                        ConvNd(
                            in_channels,
                            hid_channels[i],
                            spatial=spatial,
                            identity_init=identity_init,
                            **kwargs,
                        )
                    )

                for blk in _make_blocks(
                    num_blocks, hid_channels[i],
                    block_type=block_type, input_size=level_size,
                    num_heads=num_heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, drop_path=drop_path,
                    ffn_factor=ffn_factor, spatial=spatial,
                    dropout=dropout, checkpointing=checkpointing, **kwargs,
                ):
                    blocks.append(blk)

                if i > 0:
                    if pixel_shuffle:
                        blocks.append(
                            nn.Sequential(
                                ConvNd(
                                    hid_channels[i],
                                    hid_channels[i - 1] * math.prod(stride),
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                                Unpatchify(patch_size=stride),
                            )
                        )
                    else:
                        blocks.append(
                            nn.Sequential(
                                nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                                ConvNd(
                                    hid_channels[i],
                                    hid_channels[i - 1],
                                    spatial=spatial,
                                    identity_init=identity_init,
                                    **kwargs,
                                ),
                            )
                        )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i],
                            math.prod(patch_size) * out_channels,
                            spatial=spatial,
                            **kwargs,
                        )
                    )

                self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor, with shape :math:`(B, C_i, L_1, ..., L_N)`.

        Returns:
            The output tensor, with shape :math:`(B, C_o, L_1 \times 2^D, ..., L_N  \times 2^D)`.
        """

        for blocks in self.ascent:
            for block in blocks:
                x = block(x)

        x = self.unpatch(x)

        return x


class ConvAE(nn.Module):
    r"""Creates a conv auto-encoder module.

    Arguments:
        encoder: An encoder module.
        decoder: A decoder module.
        shape: The input shape.
        resize: The resize factor.
        saturation: The type of latent saturation.
        noise: The latent noise's standard deviation.
        vae: Whether to make the encoder variational. When enabled, an extra
            log-variance head is attached on top of the encoder output (which
            becomes the posterior mean :math:`\mu`). In training mode, ``encode``
            returns a reparameterized sample :math:`\mu + \sigma \odot \epsilon`
            and the per-element KL to :math:`\mathcal{N}(0, 1)` is stashed on
            ``self.last_kl``; in eval mode it returns :math:`\mu` (deterministic),
            so downstream dumping/diffusion is unaffected.
        latent_channels: Number of latent channels (only needed to build the
            log-variance head when ``vae`` is enabled).
        logvar_min: Lower clamp for the predicted log-variance (stability).
        logvar_max: Upper clamp for the predicted log-variance (stability).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        shape: Sequence[int] = (721, 1440),
        resize: int = 32,
        saturation: str = "softclip2",
        saturation_bound: float = 5.0,
        noise: float = 0.0,
        vae: bool = False,
        latent_channels: Optional[int] = None,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        spatial: int = 2,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.shape = shape
        self.resize = resize

        self.saturation = saturation
        self.saturation_bound = saturation_bound
        self.noise = noise

        # Variational head. Kept entirely separate from the deterministic path
        # so that vae=False is byte-identical to the non-variational AE.
        self.vae = vae
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.last_kl = None
        if vae:
            assert latent_channels is not None, "latent_channels required when vae=True"
            # 1x1 conv reading the same final feature map as mu, predicting logvar.
            # Zero-init weight + zero bias => logvar starts at 0 (sigma=1), i.e.
            # the sampler begins as mu + N(0,1) noise and KL starts near mu^2/2.
            self.logvar_head = ConvNd(
                latent_channels, latent_channels, spatial=spatial, kernel_size=1
            )
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)

    def description(self):
        # TO IMPROVE
        return "Conv autoencoder"

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        elif self.saturation == "softclip":
            return x / (1 + abs(x) / self.saturation_bound)
        elif self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / self.saturation_bound))
        elif self.saturation == "tanh":
            return torch.tanh(x / self.saturation_bound) * self.saturation_bound
        elif self.saturation == "asinh":
            return torch.arcsinh(x)
        elif self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        else:
            raise ValueError(f"unknown saturation '{self.saturation}'")

    def pad(self, x: Tensor) -> Tensor:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f

        x = torch.nn.functional.pad(x, pad=(w // 2, w - w // 2, 0, 0), mode="circular")
        x = torch.nn.functional.pad(x, pad=(0, 0, h // 2, h - h // 2), mode="constant")

        return x

    def unpad(self, x: Tensor) -> Tensor:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f

        if h > 0:
            x = x[..., h // 2 : h // 2 - h, :]

        if w > 0:
            x = x[..., :, w // 2 : w // 2 - w]

        return x

    @property
    def latent_shape(self) -> int:
        H, W = self.shape
        f = self.resize
        h, w = -H % f, -W % f + 2 * f
        h, w = (H + h) // f, (W + w) // f

        latent_channels = self.encoder.descent[-1][-1].out_channels

        return h, w, latent_channels

    def encode(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        H, W = self.shape
        x = rearrange(x, "... (H W) C -> ... C H W", H=H, W=W)
        # if not (H % self.resize == 0 and W % self.resize == 0):
        x = self.pad(x)
        h = self.encoder(x)

        if not self.vae:
            z = self.saturate(h)
            z = rearrange(z, "... C H W -> ... (H W) C")
            return z

        # Variational path: mu is the (saturated) encoder output, logvar comes
        # from a parallel head reading the same pre-saturation feature map.
        mu = self.saturate(h)
        logvar = self.logvar_head(h).clamp(self.logvar_min, self.logvar_max)

        # Per-element KL[N(mu, sigma^2) || N(0, 1)] = 0.5*(mu^2 + sigma^2 - logvar - 1).
        # Stash the mean so train_v1.py can read it without changing the
        # encode/forward return signature (keeps every other caller untouched).
        self.last_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean()

        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu  # deterministic at eval/dump time

        z = rearrange(z, "... C H W -> ... (H W) C")
        return z

    def decode(self, z: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tensor:
        H, W, _ = self.latent_shape
        if self.noise > 0:
            z = z + self.noise * torch.randn_like(z)
        z = rearrange(z, "... (H W) C -> ... C H W", H=H, W=W)
        x = self.decoder(z)
        # if not (H % self.resize == 0 and W % self.resize == 0):
        x = self.unpad(x)
        x = rearrange(x, "... C H W -> ... (H W) C")
        return x

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        z = self.encode(x, t, c)
        y = self.decode(z, t, c)
        return z, y


def conv_ae(
    in_channels: int,
    lat_channels: int,
    spatial: int = 2,
    # AE
    shape: Sequence[int] = (721, 1440),
    resize: int = 32,
    saturation: str = "softclip2",
    saturation_bound: float = 5.0,
    noise_level: float = 0.0,
    # Asymmetric reconstruction: decoder output channel count. Defaults to
    # in_channels (symmetric, the historical behaviour -> every existing caller
    # is unchanged). Set < in_channels to feed the encoder auxiliary input
    # channels (e.g. cos-zenith / time-offset) that should NOT be reconstructed:
    # the decoder then only rebuilds the first `out_channels` physical channels.
    out_channels: Optional[int] = None,
    # Variational
    vae: bool = False,
    logvar_min: float = -10.0,
    logvar_max: float = 10.0,
    # Multi-branch
    channel_groups: Optional[Sequence[int]] = None,
    # Ignore
    latent_shape: Optional[Tuple] = None,
    name: Optional[str] = None,
    context_channels: Optional[int] = None,
    # Passthrough
    **kwargs,
) -> ConvAE:
    r"""Instantiates a conv auto-encoder."""

    if out_channels is None:
        out_channels = in_channels

    H, W = shape
    f = resize
    h_pad, w_pad = -H % f, -W % f + 2 * f
    padded_size = (H + h_pad, W + w_pad)

    encoder = ConvEncoder(
        in_channels=in_channels,
        out_channels=lat_channels,
        spatial=spatial,
        channel_groups=channel_groups,
        padded_size=padded_size,
        **kwargs,
    )

    decoder = ConvDecoder(
        in_channels=lat_channels,
        out_channels=out_channels,
        spatial=spatial,
        channel_groups=channel_groups,
        padded_size=padded_size,
        **kwargs,
    )

    autoencoder = ConvAE(
        encoder,
        decoder,
        shape=shape,
        resize=resize,
        saturation=saturation,
        saturation_bound=saturation_bound,
        noise=noise_level,
        vae=vae,
        latent_channels=lat_channels,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
        spatial=spatial,
    )

    return autoencoder


if __name__ == "__main__":
    ae = conv_ae(
        in_channels=69,
        lat_channels=320,
        shape=[120, 240],
        resize=8,
        saturation="softclip2",
        noise_level=0.0,
        hid_channels=[32, 64, 128,256],
        hid_blocks=[3, 3, 3, 3],
        pixel_shuffle=True,
        ffn_factor=1,
        patch_size=[1, 1],
        identity_init=True,
        channel_groups=[13,13,13,13,13,4],
        global_branch=True,
    )
    print(ae)
    print(f"\nTotal params: {sum(p.numel() for p in ae.parameters()):,}")

    swin_ae = conv_ae(
        in_channels=69,
        lat_channels=320,
        shape=[120, 240],
        resize=8,
        saturation="softclip2",
        noise_level=0.0,
        hid_channels=[32,64,128,256],
        hid_blocks=[6, 6, 6, 6],
        num_heads=8,
        pixel_shuffle=True,
        window_size=[5,8],
        patch_size=[1, 1],
        identity_init=True,
        channel_groups=[13,13,13,13,13,4],
        global_branch=True,
        block_type='swin'
    )
    print(f"\nTotal params: {sum(p.numel() for p in swin_ae.parameters()):,}")
