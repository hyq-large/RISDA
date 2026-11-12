from typing import Sequence, Union

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn


def ConvNd(
    in_channels: int, out_channels: int, spatial: int = 2, identity_init: bool = False, **kwargs
) -> nn.Module:
    CONVS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if spatial in CONVS:
        Conv = CONVS[spatial]
    else:
        raise NotImplementedError()
    conv = Conv(in_channels, out_channels, **kwargs)
    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]
        eye = torch.zeros_like(conv.weight.data)
        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1
        conv.weight.data.mul_(0.01)
        conv.weight.data.add_(eye)
    return conv


class LayerNorm(nn.Module):
    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-05):
        super().__init__()
        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype in (torch.float32, torch.float64):
            x32 = x
        else:
            x32 = x.to(dtype=torch.float32)
        (variance, mean) = torch.var_mean(x32, dim=self.dim, keepdim=True)
        y32 = (x32 - mean) * torch.rsqrt(variance + self.eps)
        return y32.to(dtype=x.dtype)


class SineEncoding(nn.Module):
    def __init__(self, features: int, omega: float = 100.0):
        super().__init__()
        assert features % 2 == 0
        freqs = torch.linspace(0, 1, features // 2, dtype=torch.float64)
        freqs = omega ** (-freqs)
        self.register_buffer("freqs", freqs.to(torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x = x[..., None]
        return torch.cat((torch.sin(self.freqs * x), torch.cos(self.freqs * x)), dim=-1)


def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (axis_length,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=axis_length)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=axis_length)
    elif len(patch_size) == 2:
        (h, w) = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        (axis_length, h, w) = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=axis_length, h=h, w=w
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=axis_length, h=h, w=w
            )
    elif len(patch_size) == 4:
        (axis_length, h, w, z) = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... L H W Z (C l h w z)",
                l=axis_length,
                h=h,
                w=w,
                z=z,
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... (C l h w z) L H W Z",
                l=axis_length,
                h=h,
                w=w,
                z=z,
            )
    else:
        raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (axis_length,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=axis_length)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=axis_length)
    elif len(patch_size) == 2:
        (h, w) = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        (axis_length, h, w) = patch_size
        if channel_last:
            return Rearrange(
                "... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=axis_length, h=h, w=w
            )
        else:
            return Rearrange(
                "... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=axis_length, h=h, w=w
            )
    elif len(patch_size) == 4:
        (axis_length, h, w, z) = patch_size
        if channel_last:
            return Rearrange(
                "... L H W Z (C l h w z) -> ... C (L l) (H h) (W w) (Z z)",
                l=axis_length,
                h=h,
                w=w,
                z=z,
            )
        else:
            return Rearrange(
                "... (C l h w z) L H W Z -> ... C (L l) (H h) (W w) (Z z)",
                l=axis_length,
                h=h,
                w=w,
                z=z,
            )
    else:
        raise NotImplementedError()
