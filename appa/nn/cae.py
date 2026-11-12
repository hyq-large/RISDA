import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from appa.nn import triggers

from .layers import ConvNd, LayerNorm, Patchify, Unpatchify


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


def _make_blocks(
    num_blocks, channels, ffn_factor=1, spatial=2, dropout=None, checkpointing=False, **conv_kwargs
):
    return nn.ModuleList(
        [
            ResBlock(
                channels,
                ffn_factor=ffn_factor,
                spatial=spatial,
                dropout=dropout,
                checkpointing=checkpointing,
                **conv_kwargs,
            )
            for _ in range(num_blocks)
        ]
    )


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
    ):
        super().__init__()
        if channel_groups is not None or global_branch or block_type != "resblock":
            raise ValueError(
                "Only the single-branch residual convolutional architecture is supported"
            )
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
        self.descent = nn.ModuleList()
        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()
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
    ):
        super().__init__()
        if channel_groups is not None or global_branch or block_type != "resblock":
            raise ValueError(
                "Only the single-branch residual convolutional architecture is supported"
            )
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
        self.ascent = nn.ModuleList()
        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()
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
