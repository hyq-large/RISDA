import math
from typing import Hashable, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from appa.date import get_local_time_encoding, get_year_progress_encoding

from .attention import MultiheadSelfAttention
from .layers import Patchify, SineEncoding, Unpatchify


class ViTBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mod_features: int = 0,
        ffn_factor: int = 4,
        spatial: int = 2,
        rope: bool = True,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        if mod_features > 0:
            self.ada_zero = nn.Sequential(
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, 4 * channels),
                Rearrange("... (n C) -> n ... 1 C", n=4),
            )
            self.ada_zero[-2].weight.data.mul_(0.01)
        else:
            self.ada_zero = nn.Parameter(torch.randn(4, channels))
            self.ada_zero.data.mul_(0.01)
        self.msa = MultiheadSelfAttention(channels, **kwargs)
        if rope:
            amplitude = 100.0 ** (-torch.rand(channels // 2))
            direction = torch.nn.functional.normalize(torch.randn(spatial, channels // 2), dim=0)
            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

    def _forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
    ) -> Tensor:
        if self.theta is None:
            theta = None
        else:
            theta = torch.einsum("...ij,jk", coo, self.theta)
        if torch.is_tensor(self.ada_zero):
            (a, b, c, d) = self.ada_zero
        else:
            (a, b, c, d) = self.ada_zero(mod)
        y = (a + 1) * self.norm(x) + b
        y = y + self.msa(y, theta, mask)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)
        if skip is not None:
            y = (y + d * skip) * torch.rsqrt(1 + d * d)
        return y

    def forward(
        self,
        x: Tensor,
        mod: Optional[Tensor] = None,
        coo: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        skip: Optional[Tensor] = None,
    ) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, mod, coo, mask, skip, use_reentrant=False)
        else:
            return self._forward(x, mod, coo, mask, skip)


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        unpatch_size: Union[int, Sequence[int], None] = None,
        window_size: Union[int, Sequence[int], None] = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial
        if unpatch_size is None:
            unpatch_size = patch_size
        elif isinstance(unpatch_size, int):
            unpatch_size = [unpatch_size] * spatial
        self.patch = Patchify(patch_size, channel_last=True)
        self.unpatch = Unpatchify(unpatch_size, channel_last=True)
        self.in_proj = nn.Linear(
            math.prod(patch_size) * (in_channels + cond_channels), hid_channels
        )
        self.out_proj = nn.Linear(hid_channels, math.prod(patch_size) * out_channels)
        self.positional_embedding = nn.Sequential(
            SineEncoding(hid_channels),
            Rearrange("... N C -> ... (N C)"),
            nn.Linear(spatial * hid_channels, hid_channels),
        )
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    channels=hid_channels, mod_features=mod_features, spatial=spatial, **kwargs
                )
                for _ in range(hid_blocks)
            ]
        )
        self.spatial = spatial
        if window_size is None:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = (window_size,) * spatial
        else:
            self.window_size = tuple(window_size)

    @staticmethod
    def coo_and_mask(
        shape: Sequence[int],
        spatial: int,
        window_size: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        assert isinstance(shape, Hashable)
        assert isinstance(window_size, Hashable)
        coo = (torch.arange(size, device=device) for size in shape)
        coo = torch.cartesian_prod(*coo)
        coo = torch.reshape(coo, shape=(-1, spatial))
        if window_size is None:
            mask = None
        else:
            delta = torch.abs(coo[:, None] - coo[None, :])
            delta = torch.minimum(delta, delta.new_tensor(shape) - delta)
            mask = torch.all(delta <= coo.new_tensor(window_size) // 2, dim=-1)
        return (coo.to(dtype=dtype), mask)

    def forward(
        self, x: Tensor, mod: Optional[Tensor] = None, cond: Optional[Tensor] = None
    ) -> Tensor:
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.patch(x)
        x = self.in_proj(x)
        shape = x.shape[-self.spatial - 1 : -1]
        (coo, mask) = self.coo_and_mask(
            shape,
            spatial=self.spatial,
            window_size=self.window_size,
            dtype=x.dtype,
            device=x.device,
        )
        x = skip = torch.flatten(x, -self.spatial - 1, -2)
        x = x + self.positional_embedding(coo)
        for block in self.blocks:
            x = block(x, mod, coo=coo, mask=mask, skip=skip)
        x = torch.unflatten(x, sizes=shape, dim=-2)
        x = self.out_proj(x)
        x = self.unpatch(x)
        return x


class ViDiT(ViT):
    def __init__(
        self,
        in_channels: int,
        blanket_size: int = 4,
        shape: Sequence[int] = (15, 32),
        cond_channels: int = 0,
        mod_features: int = 0,
        hid_channels: int = 1024,
        hid_blocks: int = 3,
        spatial: int = 3,
        patch_size: Union[int, Sequence[int]] = 1,
        window_size: Union[int, Sequence[int], None] = None,
        name: str = "",
        arch: str = "",
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            cond_channels=4 * blanket_size + cond_channels,
            mod_features=mod_features,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            spatial=spatial,
            patch_size=patch_size,
            unpatch_size=patch_size,
            window_size=window_size,
            **kwargs,
        )
        self.blanket_size = blanket_size
        self.shape = shape
        self.diffusion_time_encoder = SineEncoding(features=mod_features)

    def description(self):
        return "Patched DiT"

    def encode_date(self, date: Tensor):
        year_progress = get_year_progress_encoding(date).flatten(-2)
        local_hour = get_local_time_encoding(date, torch.zeros(1).to(date)).squeeze(-2).flatten(-2)
        date = torch.cat([year_progress, local_hour], dim=-1)
        for _ in range(len(self.shape) + 1):
            date = date.unsqueeze(-1)
        date = date.repeat(1, 1, self.blanket_size, *self.shape)
        return date

    def forward(self, x: Tensor, t: Tensor, date: Tensor):
        (H, W) = self.shape
        x = rearrange(x, "B (T H W C) -> B C T H W", T=self.blanket_size, H=H, W=W)
        cond = self.encode_date(date)
        mod = self.diffusion_time_encoder(t)
        x = super().forward(x, mod, cond)
        return rearrange(x, "B C T H W -> B (T H W C)")
