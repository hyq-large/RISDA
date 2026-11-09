import numpy as np
import pandas as pd
import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)

__all__ = ['time_encoding', "ResBlock", "SwinBlock"]


def exists(val):
    return val is not None

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def init_swin(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def deep_norm_fill(m, base):
    init_scale = np.power(base, 0.25)
    for name, p in m.named_parameters():
        if (
            "fc1" in name
            or "fc2" in name
            or "out_proj" in name
        ):
            p.data.div_(init_scale)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        temb_dim=0,
        dropout=0,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()

        self.scale_shift = scale_shift

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_dim, eps=1e-6, affine=True),
            nn.SiLU(),
            conv_class(in_dim, out_dim, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, 2 * out_dim if scale_shift else out_dim)
        ) if temb_dim > 0 else nn.Identity()


        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=out_dim, eps=1e-6, affine=True),
            nn.SiLU() if scale_shift else nn.Identity(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_class(out_dim, out_dim, 3, padding=1)
            ),
        )

        if in_dim == out_dim:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_dim, out_dim, 1)


    def forward(self, x, emb=None):
        h = self.in_layers(x)
        if emb is None:
            h = self.out_layers(h)
        else:
            emb = self.emb_layers(emb)
            emb = rearrange(emb, 'n c -> n c 1 1')
            if self.scale_shift:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = emb.chunk(2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb
                h = self.out_layers(h)
        return self.skip_connection(x) + h

class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        temb_dim: int = 0,
        num_layers: int = 1,
        up=True,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels + skip_channels if i == 0 else out_channels
            resnets.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.up = up
        if up:
            self.upsample = Upsample(out_channels, out_channels, use_deconv=True)


    def forward(self, x, temb=None, output_size=None):
        for blk in self.resnets:
            x = blk(x, temb)

        if self.up:
            x = self.upsample(x, output_size)
        return x

class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int = 0,
        num_layers: int = 1,
        down=True,
        scale_shift=False,
        conv_class=nn.Conv2d,
    ):
        super().__init__()
        self.down = down

        if down:
            self.downsample = Downsample(in_channels, out_channels, use_conv=True)

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                ResBlock(
                    out_channels if down else in_channels,
                    out_channels,
                    temb_dim=temb_dim,
                    scale_shift=scale_shift,
                    conv_class=conv_class,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, temb=None):

        if self.down:
            x = self.downsample(x)

        for blk in self.resnets:
            x = blk(x, temb)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, use_conv=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2

        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels,  out_channels=None, use_conv=False, use_deconv=False, conv_class=nn.Conv2d):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_deconv = use_deconv

        if use_deconv:
            self.op = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.op = conv_class(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x, output_size=None):
        if self.use_deconv:
            return self.op(x)

        if output_size is None:
            y = F.interpolate(x.float(), scale_factor=2.0, mode="nearest").to(x)
        else:
            y = F.interpolate(x.float(), size=tuple(output_size), mode="nearest").to(x)

        if self.use_conv:
            y = self.op(y)

        return y

class PeriodicConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert max(self.padding) > 0

    def forward(self, x):
        x = F.pad(x, (self.padding[1], self.padding[1], 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.padding[0], self.padding[0]), mode="constant", value=0)
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups
        )
        return x

conv_class = PeriodicConv2d
norm_layer = partial(nn.LayerNorm, eps=1e-6)
linear_class = nn.Linear

conv_class = PeriodicConv2d
norm_layer = partial(nn.LayerNorm, eps=1e-6)
linear_class = nn.Linear

class CubeEmbed(nn.Module):
    def __init__(self,
            in_chans=3,
            in_frames=1,
            patch_size=4,
            embed_dim=96,
            norm_layer=None,
            flatten=False,
            proj_type='conv2d',
        ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans * in_frames, embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=0
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if x.ndim == 5:
            x = rearrange(x, 'n t c h w -> n (t c) h w')

        x = self.proj(x)

        if self.flatten:
            x = rearrange(x, 'n c h w -> n (h w) c')
            x = self.norm(x)

        return x



class CubeEmbed(nn.Module):
    def __init__(self,
            in_chans=3,
            in_frames=1,
            patch_size=4,
            embed_dim=96,
            norm_layer=None,
            flatten=False,
            proj_type='conv2d',
        ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans * in_frames, embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=0
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if x.ndim == 5:
            x = rearrange(x, 'n t c h w -> n (t c) h w')

        x = self.proj(x)

        if self.flatten:
            x = rearrange(x, 'n c h w -> n (h w) c')
            x = self.norm(x)

        return x

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class MCDropout(nn.Module):
    def __init__(self, p=0):
        super().__init__()
        self.p = p
        self.mask = None

    def init_mask(self, x):
        self.mask = torch.empty_like(x).bernoulli_(1 - self.p)

    def forward(self, x, step=0):
        if self.p == 0:
            return x

        if step == 0 or self.mask is None:
            self.init_mask(x)

        x = x.div(1 - self.p) * self.mask
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

class GeGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        inner_dim = int(hidden_features * (2 / 3))
        self.fc1 = nn.Linear(in_features, inner_dim * 2, bias=False)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: (win_h, win_w)

    Returns:
        windows: (num_windows*B, win_h, win_w, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: (win_h, win_w)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        # self.softmax = nn.Softmax(dim=-1)
        # self.gating = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # gating = self.gating(x).sigmoid()

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.int().view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttentionV2(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_weight = nn.Parameter(torch.randn(dim*3, dim)*0.02)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv_weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table.to(x)).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.int().view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = mask.to(x)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_size (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, input_size,
                 window_size=7, shift_size=0, attn_type='v1', mask_type='hw',
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, init_values=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        self.window_size = list(to_2tuple(window_size))
        self.shift_size = list(to_2tuple(shift_size))
        self.mlp_ratio = mlp_ratio

        if self.input_size[0] <= self.window_size[0]:
            self.shift_size[0] = 0
            self.window_size[0] = self.input_size[0]

        if self.input_size[1] <= self.window_size[1]:
            self.shift_size[1] = 0
            self.window_size[1] = self.input_size[1]

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        if attn_type == 'v1':
            self.attn = WindowAttention(
                dim, window_size=self.window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        elif attn_type == 'v2':
            self.attn = WindowAttentionV2(
                dim, window_size=self.window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()


        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GeGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # self.mlp = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_hidden_dim)


        if max(self.shift_size) > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            w_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    if mask_type == 'h':
                        img_mask[:, h, :, :] = cnt
                    elif mask_type == 'w':
                        img_mask[:, :, w, :] = cnt
                    else:
                        img_mask[:, h, w, :] = cnt

                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)


    def _init_norm(self):
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm1.weight, 0)
        nn.init.constant_(self.norm2.bias, 0)
        nn.init.constant_(self.norm2.weight, 0)


    def forward(self, x):
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        # cyclic shift
        if max(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)


        x = shortcut + self.drop_path(self.ls1(x))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_size={self.input_size}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_size
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_size (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_size={self.input_size}, dim={self.dim}"

    def flops(self):
        H, W = self.input_size
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_size (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_size, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(depth):
            drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path

            block = SwinBlock(
                dim=dim, input_size=input_size,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )

            self.blocks.append(block)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_size, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_down = x
        if self.downsample is not None:
            x_down = self.downsample(x)
        return x, x_down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_size={self.input_size}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        input_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, input_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        input_size = [input_size[0] // patch_size[0], input_size[1] // patch_size[1]]

        self.input_size = input_size
        self.patch_size = patch_size
        self.input_size = input_size

        self.num_patch = input_size[0] * input_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.input_size[0] and W == self.input_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patch_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        input_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
            self,
            in_chans=3,
            input_size=128,
            patch_size=2,
            num_classes=0,
            window_size=8,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, **kwargs
        ):
        super().__init__()
        self.input_size = input_size

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbed(
            input_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        num_patch = self.patch_embed.num_patch
        input_size = self.patch_embed.input_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patch, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(dim=dim,
                               input_size=(input_size[0] // (2 ** i_layer),
                                           input_size[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        return x


    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patch_size[0] * self.patch_size[1] // (2 ** self.num_layers)
        return flops

class SwinLayer(nn.Module):

    def __init__(
            self,
            in_chans,
            embed_dim,
            input_size,
            window_size,
            depth=4,
            num_heads=8,
            mlp_ratio=4.,
            drop_path=0,
            attn_type="v1",
            mask_type="hw",
            deep_norm=True,
            scale_shift=False,
        ):

        super().__init__()


        self.depth = depth
        self.input_size = input_size

        self.blocks = nn.ModuleList()

        for i in range(depth):
            blk = SwinBlock(
                dim=embed_dim,
                input_size=input_size,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attn_type=attn_type,
                mask_type=mask_type,
                norm_layer=norm_layer,
            )
            self.blocks.append(blk)

        # self.in_proj = nn.Linear(in_chans, embed_dim, bias=False)
        # self.out_proj = conv_class(embed_dim, embed_dim, kernel_size=3, padding=1)

        self.in_chans = in_chans
        self.embed_dim = embed_dim



    def forward(self, h):
        # state = rearrange(state, 'n c h w -> n (h w) c')
        # state = self.in_proj(state)

        for i, blk in enumerate(self.blocks):
            h = blk(h)

        # state = rearrange(state,  'n (h w) c -> n c h w', h=self.image_size[0])
        # state = self.out_proj(state)
        return h

class USwinDecoder(nn.Module):
    def __init__(
            self,
            in_chans,
            out_chans,
            in_frames,
            embed_dim,
            image_size,
            window_size=8,
            patch_size=4,
            down_times=0,
            depths=[6, 6, 6, 6],
            num_heads=8,
            drop_path=0,
            mlp_ratio=4,
            attn_type="v1",
            mask_type='hw',
            ape=False,
            deep_norm=True,
            latent_dim=1024,
        ):
        super().__init__()
        self.ape = ape
        self.deep_norm = deep_norm
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times


        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=True,
        )
        if self.ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(self.feat_size), embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        layers = []
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        input_size = [sz // (patch_size*(2 ** down_times)) for sz in image_size]

        for i in range(self.num_layers):
            dpr_i =  dpr[sum(depths[:i]) : sum(depths[:i+1])]
            layer = SwinLayer(
                embed_dim,
                embed_dim,
                input_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr_i,
                attn_type=attn_type,
                mask_type=mask_type,
                deep_norm=deep_norm,
            )
            layers.append(layer)
            self.add_module(f"norm{i}", norm_layer(embed_dim))

        self.layers = nn.ModuleList(layers)

        self.fpn = nn.Sequential(
            nn.Linear(embed_dim * self.num_layers, latent_dim),
            nn.GELU(),
        )

        # self.apply(init_swin)

        if self.deep_norm:
            deep_norm_fill(self, base=2*sum(depths))

        self.head = zero_module(nn.Linear(embed_dim, out_chans*patch_size*patch_size))


        # self.mc_dropout = MCDropout(p=0.3)


    def forward(self, input,emb=None, const=None, noise=None):
        x = input

        x = rearrange(x, 'n t c h w -> n (t c) h w')
        h = self.patch_embed(x)

        # h = self.mc_dropout(h, step)

        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)

        # pos_embed
        if self.ape:
            h = h + self.pos_embed

        # attention (swin or vit)


        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        # import pdb;pdb.set_trace()
        # To. latent Space

        h = self.fpn(torch.cat(outs, dim=-1))


        # unet up



        out = self.head(h)
        out = rearrange(out,
            'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
            h=self.feat_size[0],
            p1=self.patch_size,
            p2=self.patch_size,
        )

        if out.shape[-2:] != input.shape[-2:]:
            out = F.interpolate(
                out.float(),
                size=input.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(input)


        return out




class USwinDecoder_E(nn.Module):
    def __init__(
            self,
            in_chans,
            out_chans,
            in_frames,
            embed_dim,
            image_size,
            window_size=8,
            patch_size=4,
            down_times=0,
            depths=[6, 6, 6, 6],
            num_heads=8,
            drop_path=0,
            mlp_ratio=4,
            attn_type="v1",
            mask_type='hw',
            ape=False,
            deep_norm=True,
            latent_dim=1024,
        ):
        super().__init__()
        self.ape = ape
        self.deep_norm = deep_norm
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times


        self.patch_embed = CubeEmbed(
            in_chans=in_chans,
            in_frames=in_frames,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            flatten=True,
        )
        if self.ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(self.feat_size), embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        layers = []
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        input_size = [sz // (patch_size*(2 ** down_times)) for sz in image_size]

        for i in range(self.num_layers):
            dpr_i =  dpr[sum(depths[:i]) : sum(depths[:i+1])]
            layer = SwinLayer(
                embed_dim,
                embed_dim,
                input_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr_i,
                attn_type=attn_type,
                mask_type=mask_type,
                deep_norm=deep_norm,
            )
            layers.append(layer)
            self.add_module(f"norm{i}", norm_layer(embed_dim))

        self.layers = nn.ModuleList(layers)
        self.to_latent = TokenFCN(embed_dim,2*latent_dim, hidden_dim=embed_dim, drop=drop_path)
        # self.fpn = nn.Sequential(
        #     nn.Linear(embed_dim * self.num_layers, latent_dim),
        #     nn.GELU(),
        # )

        # self.apply(init_swin)

        if self.deep_norm:
            deep_norm_fill(self, base=2*sum(depths))

        self.head = zero_module(nn.Linear(embed_dim, out_chans*patch_size*patch_size))


        # self.mc_dropout = MCDropout(p=0.3)


    def forward(self, input,emb=None, const=None, noise=None):
        x = input
        x = rearrange(x, 'n t c h w -> n (t c) h w')
        h = self.patch_embed(x)



        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)

        # pos_embed
        if self.ape:
            h = h + self.pos_embed

        # attention (swin or vit)

        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        h = torch.cat(outs, dim=-1)
        latent = self.to_latent(h)
        # return latent
        return DiagonalGaussianDistribution(latent)






class USwinDecoder_D(nn.Module):
    def __init__(
            self,
            in_chans,
            out_chans,
            in_frames,
            embed_dim,
            image_size,
            window_size=8,
            patch_size=4,
            down_times=0,
            depths=[6, 6, 6, 6],
            num_heads=8,
            drop_path=0,
            mlp_ratio=4,
            attn_type="v1",
            mask_type='hw',
            ape=False,
            deep_norm=True,
            latent_dim=1024,
        ):
        super().__init__()
        self.ape = ape
        self.deep_norm = deep_norm
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.feat_size = [sz // patch_size for sz in image_size]

        self.num_layers = len(depths)
        self.down_times = down_times

        if self.ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(self.feat_size), embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        layers = []
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        input_size = [sz // (patch_size*(2 ** down_times)) for sz in image_size]

        for i in range(self.num_layers):
            dpr_i =  dpr[sum(depths[:i]) : sum(depths[:i+1])]
            layer = SwinLayer(
                embed_dim,
                embed_dim,
                input_size,
                window_size,
                depth=depths[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr_i,
                attn_type=attn_type,
                mask_type=mask_type,
                deep_norm=deep_norm,
            )
            layers.append(layer)
            self.add_module(f"norm{i}", norm_layer(embed_dim))

        self.layers = nn.ModuleList(layers)


        # self.apply(init_swin)

        if self.deep_norm:
            deep_norm_fill(self, base=2*sum(depths))

        self.head = zero_module(nn.Linear(embed_dim, out_chans*patch_size*patch_size))

        self.from_latent = TokenFCN(latent_dim, embed_dim, hidden_dim=embed_dim, drop=drop_path)
        # self.mc_dropout = MCDropout(p=0.3)


    def forward(self, input,emb=None, const=None, noise=None):
        h = self.from_latent(input)

        # h = self.mc_dropout(h, step)
        if exists(noise):
            noise = F.interpolate(
                noise,
                size=self.feat_size,
                mode="bilinear",
                align_corners=False,
            )
            noise = rearrange(noise, 'n c h w -> n (h w) c')
            h = h + noise.to(h)
        # pos_embed
        if self.ape:
            h = h + self.pos_embed
        # attention (swin or vit)
        outs = []
        for i, blk in enumerate(self.layers):
            h = blk(h)
            out = getattr(self, f"norm{i}")(h)
            outs.append(out)
        h = torch.cat(outs, dim=-1)
        # import pdb;pdb.set_trace()
        # To. latent Space
        # h = self.fpn(torch.cat(outs, dim=-1))
        # unet up
        out = self.head(h)
        out = rearrange(out,
            'n (h w) (p1 p2 c) -> n c (h p1) (w p2)',
            h=self.feat_size[0],
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return out

        # if out.shape[-2:] != input.shape[-2:]:
        #     out = F.interpolate(
        #         out.float(),
        #         size=input.shape[-2:],
        #         mode="bilinear",
        #         align_corners=False,
        #     ).to(input)









def time_encoding(init_time, num_steps, freq=6):
    '''
    This function generates time encodings for the given initial time,
    number of steps, and frequency of time intervals. The approach is
    based on the description provided in the paper
    "GraphCast: Learning skillful medium-range global weather forecasting"
    by DeepMind (https://arxiv.org/abs/2212.12794).
    '''

    init_time = np.array([init_time])
    tembs = []
    for i in range(num_steps):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'h') for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = torch.from_numpy(np.array(times, dtype=np.float32))
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return torch.stack(tembs)




class TokenFCN(nn.Module):
    """
    Shared FCN applied to each token (last dim).
    """
    def __init__(self, in_dim, out_dim, hidden_dim=None, drop=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (..., in_dim) -> (..., out_dim)
        return self.net(x)



class FuXiAE(nn.Module):
    def __init__(self,
        chans=70,
        in_frames=1,
        embed_dim=1536,
        image_size=(720, 1440),
        window_size=8,
        patch_size=4,
        down_times=0,
        latent_dim=1024,
        num_heads=24,
        depths=[8],
        attn_type='v2',
        mask_type='h',
        drop_path=0.2,):
        super().__init__()
        in_chans = chans
        out_chans = chans
        self.encoder = USwinDecoder_E(in_chans=in_chans,out_chans=out_chans,in_frames=in_frames,embed_dim=embed_dim,image_size=image_size,window_size=window_size,latent_dim=latent_dim,
            patch_size=patch_size,num_heads=num_heads,depths=depths,attn_type=attn_type,mask_type=mask_type,drop_path =drop_path,)
        self.decoder = USwinDecoder_D(in_chans=in_chans,out_chans=out_chans,in_frames=in_frames,embed_dim=embed_dim,image_size=image_size,window_size=window_size,latent_dim=latent_dim,
            patch_size=patch_size,num_heads=num_heads,depths=depths,attn_type=attn_type,mask_type=mask_type,drop_path =drop_path,)
    def forward(self,x,return_latent=False,sample_posterior=False):
        posterior = self.encoder(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        rec = self.decoder(z)
        if return_latent:
            return (posterior,rec)
        return rec





class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        # sample = randn_tensor(
        #     self.mean.shape,
        #     generator=generator,
        #     device=self.parameters.device,
        #     dtype=self.parameters.dtype,
        # )
        sample = torch.randn_like(self.mean)
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return (0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[2],
                )).mean()
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[2],
                )

    def nll(self, sample: torch.Tensor, dims) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean






if __name__ == '__main__':
    bg = torch.randn(1,1,13,120,240).cuda().half()

    model = FuXiAE(chans=13,depths=[6],window_size=10,patch_size=4,image_size=(120,240),latent_dim=128).cuda().half()
    for name, m in model.named_children():
        n = sum(p.numel() for p in m.parameters())
        print(f"{n:,}\t{name}\t({m.__class__.__name__})")
    z,output = model(bg,return_latent=True)
    import pdb;pdb.set_trace()
    kl_loss = z.kl()
    print(kl_loss)

    print(output.shape)
