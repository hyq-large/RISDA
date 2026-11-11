import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from appa.nn import triggers
from appa.nn.swin import SwinBlock, to_2tuple

from .layers import ConvNd, LayerNorm, Patchify, Unpatchify


class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
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
        self.norm = LayerNorm(dim=-spatial - 1) if use_norm else nn.Identity()
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )
        self.ffn[-1].weight.data.mul_(0.01)

    def _forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.ffn(y)
        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SwinResBlock(nn.Module):
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
        (B, C, H, W) = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.block(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
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
            self.input_projs = nn.ModuleList(
                [ConvNd(in_ch, channels, spatial=spatial, **kwargs) for in_ch in input_channels]
            )
        elif is_first_level:
            self.input_projs = nn.ModuleList(
                [
                    ConvNd(
                        channels, channels, spatial=spatial, identity_init=identity_init, **kwargs
                    )
                    for _ in range(num_branches)
                ]
            )
        else:
            self.input_projs = None
        block_kwargs = dict(
            block_type=block_type,
            input_size=input_size,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            ffn_factor=ffn_factor,
            spatial=spatial,
            dropout=dropout,
            checkpointing=checkpointing,
            **kwargs,
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
        self.branches = nn.ModuleList(
            [_make_blocks(num_blocks, channels, **block_kwargs) for _ in range(num_branches)]
        )
        self.merge = ConvNd(
            self.total_branches * channels,
            self.total_branches * channels,
            spatial=spatial,
            identity_init=identity_init,
            **kwargs,
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
            for proj, blocks, chunk in zip(self.input_projs, self.branches, chunks, strict=False):
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
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return x[:, : self.channels]


class PadChannels(nn.Module):
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
        self.output_projs = nn.ModuleList(
            [ConvNd(channels, out_ch, spatial=spatial, **kwargs) for out_ch in output_channels]
        )

    def forward(self, x: Tensor) -> Tensor:
        chunks = x.chunk(self.num_heads, dim=1)
        outs = []
        for i, proj in enumerate(self.output_projs):
            outs.append(proj(chunks[i]))
        return torch.cat(outs, dim=1)


class ConvEncoder(nn.Module):
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
            padding=tuple((k // 2 for k in kernel_size)),
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
                level_size = (
                    (padded_size[0] // stride[0] ** i, padded_size[1] // stride[1] ** i)
                    if padded_size
                    else None
                )
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
                        is_first_level=i == 0,
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
                level_size = (
                    (padded_size[0] // stride[0] ** i, padded_size[1] // stride[1] ** i)
                    if padded_size
                    else None
                )
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
                    num_blocks,
                    hid_channels[i],
                    block_type=block_type,
                    input_size=level_size,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    ffn_factor=ffn_factor,
                    spatial=spatial,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **kwargs,
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
        x = self.patch(x)
        for blocks in self.descent:
            for block in blocks:
                x = block(x)
        return x


class ConvDecoder(nn.Module):
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
            padding=tuple((k // 2 for k in kernel_size)),
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
                level_size = (
                    (padded_size[0] // stride[0] ** i, padded_size[1] // stride[1] ** i)
                    if padded_size
                    else None
                )
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
                level_size = (
                    (padded_size[0] // stride[0] ** i, padded_size[1] // stride[1] ** i)
                    if padded_size
                    else None
                )
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
                    num_blocks,
                    hid_channels[i],
                    block_type=block_type,
                    input_size=level_size,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    ffn_factor=ffn_factor,
                    spatial=spatial,
                    dropout=dropout,
                    checkpointing=checkpointing,
                    **kwargs,
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
        for blocks in self.ascent:
            for block in blocks:
                x = block(x)
        x = self.unpatch(x)
        return x
