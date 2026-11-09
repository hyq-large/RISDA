r"""Shared layers and modules."""

__all__ = [
    "ConvNd",
    "LayerNorm",
    "SineEncoding",
    "SineLayer",
    "SirenEmbedding",
    "Patchify",
    "Unpatchify",
]

import math
import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Sequence, Union


def Linear(in_features: int, out_features: int, identity_init: bool = False, **kwargs):
    r"""Returns a linear layer with identity init option.

    Arguments:
        in_features: The number of input features :math:`F_i`.
        out_features: The number of output features :math:`F_o`.
        identity_init: Initialize the layer as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.
    """

    linear = nn.Linear(in_features, out_features, **kwargs)

    if identity_init:
        id_offset = torch.cat(
            [torch.eye(in_features)] * (out_features // in_features)
            + [torch.eye(out_features % in_features, in_features)],
            dim=0,
        )
        linear.weight.data.mul_(1e-2)
        linear.weight.data.add_(id_offset)

    return linear


def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    r"""Returns an N-dimensional convolutional layer.

    Arguments:
        in_channels: The number of input channels :math:`C_i`.
        out_channels: The number of output channels :math:`C_o`.
        spatial: The number of spatial dimensions :math:`N`.
        identity_init: Initialize the convolution as a (pseudo-)identity.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    CONVS = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d,
    }

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

        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)

    return conv


class LayerNorm(nn.Module):
    r"""Creates a layer that standardizes features along a dimension.

    .. math:: y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}}

    References:
       | Layer Normalization (Lei Ba et al., 2016)
       | https://arxiv.org/abs/1607.06450

    Arguments:
        dim: The dimension(s) to standardize.
        eps: A numerical stability term.
    """

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()

        self.dim = dim if isinstance(dim, int) else tuple(dim)

        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input tensor :math:`x`, with shape :math:(*).

        Returns:
            The standardized tensor :math:`y`, with shape :math:`(*)`.
        """

        if x.dtype in (torch.float32, torch.float64):
            x32 = x
        else:
            x32 = x.to(dtype=torch.float32)

        variance, mean = torch.var_mean(x32, dim=self.dim, keepdim=True)

        y32 = (x32 - mean) * torch.rsqrt(variance + self.eps)

        return y32.to(dtype=x.dtype)


class SineEncoding(nn.Module):
    r"""Creates a sinusoidal positional encoding.

    .. math::
        e_{2i} & = \sin \left( x \times \omega^\frac{-2i}{D} \right) \\
        e_{2i+1} & = \cos \left( x \times \omega^\frac{-2i}{D} \right)

    References:
        | Attention Is All You Need (Vaswani et al., 2017)
        | https://arxiv.org/abs/1706.03762

    Arguments:
        features: The number of embedding features :math:`F`. Must be even.
        omega: The maximum frequency :math:`\omega`.
    """

    def __init__(self, features: int, omega: float = 1e2):
        super().__init__()

        assert features % 2 == 0

        freqs = torch.linspace(0, 1, features // 2, dtype=torch.float64)
        freqs = omega ** (-freqs)

        self.register_buffer("freqs", freqs.to(torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The position :math:`x`, with shape :math:`(*)`.

        Returns:
            The embedding vector :math:`e`, with shape :math:`(*, F)`.
        """

        x = x[..., None]

        return torch.cat(
            (
                torch.sin(self.freqs * x),
                torch.cos(self.freqs * x),
            ),
            dim=-1,
        )


class SineLayer(nn.Module):
    r"""Adapted implementation of SineLayer for SirenNet.

    Reference:
        | https://github.com/vsitzmann/siren

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        is_first: Whether the layer is the first of :class:`SirenNet` or not.
        omega_0: The boosting factor of the layer (described in supplement Sec. 1.5 or original paper).
    """

    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights(is_first)

    @torch.no_grad()
    def init_weights(self, is_first):
        if is_first:
            weight_bound = 1 / self.in_features
        else:
            weight_bound = math.sqrt(6 / self.in_features) / self.omega_0

        self.linear.weight.uniform_(-weight_bound, weight_bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenEmbedding(nn.Module):
    r"""Creates a positional embedding module based on SirenNet.

    Reference:
        | Implicit Neural Representations with Periodic Activation Functions
        | https://arxiv.org/abs/2006.09661

        | Geographic Location Encoding with Spherical Harmonics and Sinusoidal Representation Networks
        | https://arxiv.org/abs/2310.06743

    Arguments:
        n_pos_features: Number of positional features.
        channels: The number of output channels :math:`C`.
        n_layers: The number of hidden layers.
    """

    def __init__(
        self, n_pos_features: int, channels: int, n_layers: int, checkpointing: bool = False
    ):
        super().__init__()

        layers = []
        layers.append(SineLayer(n_pos_features, channels, is_first=True))
        layers.extend([SineLayer(channels, channels, is_first=False) for _ in range(n_layers)])
        layers.append(nn.Linear(channels, channels))

        self.siren = nn.Sequential(*layers)

        self.checkpointing = checkpointing

    def _forward(self, coords: Tensor) -> Tensor:
        return self.siren(coords)

    def forward(self, coords: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, coords, use_reentrant=False)
        else:
            return self._forward(coords)


def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        else:
            return Rearrange("... C (L l) -> ... (C l) L", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        else:
            return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
        else:
            return Rearrange("... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... L H W Z (C l h w z)", l=l, h=h, w=w, z=z
            )
        else:
            return Rearrange(
                "... C (L l) (H h) (W w) (Z z) -> ... (C l h w z) L H W Z", l=l, h=h, w=w, z=z
            )
    else:
        raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        else:
            return Rearrange("... (C l) L -> ... C (L l)", l=l)
    elif len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        else:
            return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    elif len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        else:
            return Rearrange("... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
    elif len(patch_size) == 4:
        l, h, w, z = patch_size
        if channel_last:
            return Rearrange(
                "... L H W Z (C l h w z) -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z
            )
        else:
            return Rearrange(
                "... (C l h w z) L H W Z -> ... C (L l) (H h) (W w) (Z z)", l=l, h=h, w=w, z=z
            )
    else:
        raise NotImplementedError()
