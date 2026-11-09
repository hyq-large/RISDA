r"""Spatio-temporal graph network building blocks."""

import functools
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import LongTensor, Tensor
from torch.utils.checkpoint import checkpoint

from appa.date import get_local_time_encoding, get_year_progress_encoding
from appa.grid import latlon_to_xyz
from appa.nn.graph import GraphAttention
from appa.nn.layers import SineEncoding, SirenEmbedding


@functools.cache
def create_spatio_temporal_edges(
    num_nodes: int, self_edges: LongTensor, cross_edges: LongTensor, blanket_size: int
) -> LongTensor:
    """Creates edges for a spatio-temporal graph.

    Arguments:
        num_nodes : Number of spatial nodes.
        self_edges: Tensor of shape :math:`(Es, 2)`, edges between nodes at the same timestep.
        cross_edges: Tensor of shape :math:`(Ec, 2)`, edges between nodes at different timesteps.
        blanket_size: The size of the blanket :math:`K`.

    Returns:
        edge_index: Tensor of shape :math:`(E_total, 2)`, edges in the spatio-temporal graph.
    """

    spatial = (
        torch.arange(blanket_size, device=self_edges.device)[..., None, None] * num_nodes
        + self_edges
    )
    spatial = rearrange(spatial, "K E ... -> (K E) ...")

    E = cross_edges.shape[0]
    temporal = (
        torch.arange(blanket_size, device=self_edges.device)[..., None, None] * num_nodes
        + cross_edges
    )
    temporal = rearrange(temporal, "K E ... -> (K E) ...")
    temporal_q, temporal_k = temporal.unbind(-1)

    temporal = [
        torch.stack([temporal_q, temporal_k.roll(-(k + 1) * E, dims=-1)], dim=-1)
        for k in range(blanket_size - 1)
    ]

    return torch.cat([spatial, *temporal])


class GraphDiTBlock(nn.Module):
    r"""A graph DiT block that applies graph attention in the spatial or temporal dimensions
    and a feed-forward network.

    Arguments:
        channels: Number of channels in the input features.
        heads: Number of attention heads.
        num_nodes : Number of spatial nodes.
        self_edges: Tensor of shape :math:`(Es, 2)`, edges between nodes at the same timestep.
        cross_edges: Tensor of shape :math:`(Ec, 2)`, edges between nodes at different timesteps.
        mod_features: Number of modulation features for diffusion time.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        num_nodes: int,
        self_edges: LongTensor,
        cross_edges: LongTensor,
        mod_features: int,
        context_features: int = 3
        + 4,  # 3 for positions and 4 for sine/cosine of year progress and day progress
        use_xfa: bool = False,  # XFA does not work for the DiT, so we default to False.
        checkpointing: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.heads = heads

        self.num_nodes = num_nodes
        self.register_buffer("self_edges", self_edges)
        self.register_buffer("cross_edges", cross_edges)

        self.modulation = nn.Sequential(
            nn.Linear(mod_features, mod_features),
            nn.SiLU(),
            nn.Linear(mod_features, 3 * channels),
            Rearrange("... (n C) -> n ... 1 C", n=3),
        )
        self.modulation[-2].weight.data.mul_(1e-2)

        self.context_embedding = SirenEmbedding(
            context_features, channels=channels, n_layers=1, checkpointing=checkpointing
        )

        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.attn = GraphAttention(
            input_channels=channels,
            hidden_channels=channels,
            output_linear=True,
            heads=heads,
            qk_norm=True,
            use_xfa=use_xfa,
        )

        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels), nn.SiLU(), nn.Linear(4 * channels, channels)
        )

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor, time_encoding: Tensor, context_encoding: Tensor) -> Tensor:
        r"""Applies the graph DiT block.

        Arguments:
            x: The input features with shape `(B, T * N, C)`.
            time_encoding: The diffusion encoded time tensor with shape `(F)` or `(B, F)`.
            context_encoding: The encoded context, used for additive positional embedding, with shape :math:`(B, T, N, C)`.

        Returns:
            Tensor: The output features with shape `(B, T * N, C)`.
        """

        B, TN, C = x.shape
        assert TN % self.num_nodes == 0

        if self.self_edges is None or self.cross_edges is None:
            edges = None
        else:
            edges = create_spatio_temporal_edges(
                self.num_nodes, self.self_edges, self.cross_edges, TN // self.num_nodes
            )

        context = rearrange(self.context_embedding(context_encoding), "B T N C -> B (T N) C")

        a, b, c = self.modulation(time_encoding)
        y = (a + 1) * self.norm(x + context) + b
        y = y + self.attn(y, y, edges)
        y = self.ffn(y)
        y = (x + c * y) * torch.rsqrt(1 + c * c)

        return y

    def forward(self, x: Tensor, time_encoding: Tensor, context_encoding: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(
                self._forward, x, time_encoding, context_encoding, use_reentrant=False
            )
        else:
            return self._forward(x, time_encoding, context_encoding)


class SpatioTemporalGraphDiT(nn.Module):
    r"""Creates a GraphDiT made of successive spatio-temporal blocks :class:`GraphDiTBlock`.

    Arguments:
        input_grid: Input grid latitude/longitude pairs used as positional context, with shape :math:`(N, 2)`.
        input_channels: Number of input channels per node.
        hidden_channels: Number of hidden channels per node.
        hidden_blocks: Number of hidden spatio-temporal attention blocks.
        self_edges: Edges that connect spatially nodes of a graph , with shape :math:`(Es, 2)`.
        cross_edges: Edges that connect spatially nodes of a graph to nodes of graphs at every other time indices, with shape :math:`(Et, 2)`.
        mod_features: Number of modulation features for diffusion time.
        heads: The number of attention heads in the spatio-temporal attention.
        use_xfa: Whether to use xformers attention or not.
        checkpointing: Whether to checkpoint dit blocks ot not.
    """

    def __init__(
        self,
        input_grid: Tensor,
        input_channels: int,
        hidden_channels: int,
        hidden_blocks: int,
        self_edges: Tensor,
        cross_edges: Tensor,
        mod_features: int = 256,
        heads: int = 1,
        use_xfa: bool = False,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer("grid_coord", latlon_to_xyz(input_grid))
        self.register_buffer("in_longitudes", input_grid[:, -1])
        self.input_channels = input_channels
        self.num_spatial_nodes = input_grid.shape[0]

        self.blocks = nn.ModuleList([
            GraphDiTBlock(
                channels=hidden_channels,
                heads=heads,
                num_nodes=self.num_spatial_nodes,
                self_edges=self_edges,
                cross_edges=cross_edges,
                mod_features=mod_features,
                context_features=3 + 4,
                use_xfa=use_xfa,
                checkpointing=checkpointing,
            )
            for _ in range(hidden_blocks)
        ])

        # sine encoding for diffusion time
        self.diffusion_time_encoder = SineEncoding(features=mod_features)

        self.in_proj = nn.Linear(input_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, input_channels)

    def description(self):
        # TO IMPROVE
        return "Graph DiT"

    def encode_date(self, date: Tensor):
        year_progress = get_year_progress_encoding(date).unsqueeze(-2)
        local_hour = get_local_time_encoding(date, self.in_longitudes)
        year_progress = year_progress.repeat(1, 1, local_hour.shape[-2], 1)
        date = torch.cat([year_progress, local_hour], dim=-1)

        return date

    def forward(self, x: Tensor, t: Tensor, date: Tensor):
        r"""
        Arguments:
            x: The input flattened graph sequence with shape :math:`(B, D)` where :math:`D = T * N * C`.
            t: The diffusion time tensor with shape :math:`()` or :math:`(B)`
            date: The date tensor of the graph sequence used as a timestamp context, with shape :math:`(B, T, 4)`.

        Returns:
            The output denoised flat graph sequence with shape :math:`(B, T * N * C)`.
        """

        B, D = x.shape
        T = D // (self.num_spatial_nodes * self.input_channels)
        assert D == T * self.num_spatial_nodes * self.input_channels

        x = rearrange(
            x,
            "B (T N C) -> B (T N) C",
            N=self.num_spatial_nodes,
            C=self.input_channels,
        )

        date = self.encode_date(date)
        grid = self.grid_coord[None, None, ...].repeat(B, T, 1, 1)
        context = torch.cat([date, grid], dim=-1)

        t = self.diffusion_time_encoder(t)

        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x, t, context)

        x = self.out_proj(x)
        x = rearrange(
            x,
            "B (T N) C -> B (T N C)",
            N=self.num_spatial_nodes,
            C=self.input_channels,
        )

        return x
