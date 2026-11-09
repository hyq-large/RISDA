r"""Graph autoencoder implementation."""

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional, Sequence, Tuple

from appa.date import get_local_time_encoding, get_year_progress_encoding
from appa.grid import (
    create_edges,
    create_icosphere,
    create_N320,
    icosphere_nhops_edges,
    latlon_to_xyz,
)
from appa.nn.graph import GraphPoolAttention, GraphSelfAttention
from appa.nn.layers import Linear, SirenEmbedding


class GraphAE(nn.Module):
    r"""Creates a graph auto-encoder inspired from Graphcast's architecture.

    Illustration:
        The graph auto-encoder is decomposed in modules, each working on a specific query grid.
        An encoder module is structured as

            KEY_GRID => PROJ_KEY --> QUERY_GRID <-S-> QUERY_GRID

        with '=>' denoting a channel projection, '-->' an attention pooling block queried by query grid features
        and <-S-> are `S` self-attention blocks on the query grid.
        The auto-encoder is structured as

        N_320 => N_320                                                -- N_320 => N_320
                    |                                                 |
                    -- ICO-x                                      -- ICO-x
                            |                                     |
                            -- ...                              ...
                                    ICO-l => ICO_latent => ICO-l

        where intermediate grids are explicited as ICO-?.
        The latent icosphere is projected onto desired channels after self-attention as well as output grid which
        is projected back to input grid channels. In the ascending part, upper grids acts as keys and lower ones as queries.
        For the decoder, the symmetric applies from ICO_latent to the original grid.

    Arguments:
        in_channels: Number of input state channels :math:`C_i` corresponding to physical variables and their levels.
        latent_channels: Number of latent channels.
        context_channels: Number of context channels :math:`C_c`.
        hid_channels: Number of channels at each intermediate level.
        heads: Number of attention heads at each intermediate level.
        ico_divisions: Number of icosahedron subdivisions for each intermediate resolution.
        self_attention_blocks: Number of self-attention blocks at each intermediate resolution.
        pooling_area: Number of surface per point used for pooling at each intermediate resolution.
        use_hop_pooling: Whether to use hop for pooling area.
        self_attention_hops: Number of hops for self-attention at each intermediate level. If not specified, use multi-mesh attention.
        saturation: Saturation function used for latent regularization. Either softclip, softclip2, tanh or asinh.
        saturation_bound: Maximal absolute value of the saturation output. Does not apply for asinh.
        dropout: Dropout rate.
        identity_init: Whether to use identity like initialisation for pooling and projection blocks.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        context_channels: int,
        hid_channels: Sequence[int],
        heads: Sequence[int],
        ico_divisions: Sequence[int],
        self_attention_blocks: Sequence[int],
        pooling_area: Sequence[float],
        use_hop_pooling: bool = False,
        self_attention_hops: Optional[Sequence[int]] = None,
        saturation: str = "softclip2",
        saturation_bound: float = 5.0,
        noise_level: float = 0.0,
        dropout: Optional[float] = None,
        identity_init: bool = False,
        checkpointing: bool = False,
        use_xfa: bool = True,
        **kwargs,
    ):
        super().__init__()

        # CREATE PROBLEM-RELATED FEATURES
        self.icosphere_graphs = [create_icosphere(N) for N in ico_divisions]
        enc_grids = [create_N320(), *[vertices for vertices, edges in self.icosphere_graphs]]

        self.register_buffer("in_longitudes", enc_grids[0][:, -1])

        # Create pixel to ico static edges for context
        static_pool_indices = []
        for grid in enc_grids[1:]:
            static_indices = create_edges(grid, enc_grids[0], neighbors=1)[:, -1]
            static_pool_indices.append(static_indices)
        self.static_pool_indices_cat = nn.Buffer(torch.cat(static_pool_indices))
        self.static_pool_sizes = list(map(len, static_pool_indices))
        if not use_hop_pooling:
            # The whole earth surface should be covered
            min_area = 2 / math.sqrt(3)
            for area in pooling_area:
                assert area > min_area, f"pooling area must be at least {min_area}, found {area}."

            # As the distance is almost constant between points on the icosphere,
            # pooling is made using multiple of arc corresponding to area per point of the query grid.
            pool_args = [
                {"max_arc": n_area * 2 * math.asin(math.sqrt(1 / grid.shape[0]))}
                for n_area, grid in zip(pooling_area, enc_grids[1:])
            ]
        else:
            for n_hops in pooling_area:
                assert (
                    n_hops % 1 == 0 and n_hops >= 1
                ), f"must use integer area (>= 1) when using hops pooling, found {area}."
            pool_args = pooling_area

        pool_edges = []
        pool_q_features = []
        pool_k_features = []
        encoder_q_idx = []
        decoder_q_idx = []

        for key, query, pool_arg in zip(enc_grids[:-1], enc_grids[1:], pool_args):
            if use_hop_pooling:
                ed = icosphere_nhops_edges(query, key, n_hops=pool_arg)
            else:
                ed = create_edges(query, key, **pool_arg)
            pool_edges.append(ed)

            # Source indices for query creation
            encoder_q_idx.append(create_edges(query, key, neighbors=1)[:, -1])
            decoder_q_idx.append(create_edges(key, query, neighbors=1)[:, -1])

            feat_q = latlon_to_xyz(query)
            feat_k = latlon_to_xyz(key)

            pool_q_features.append(feat_q)
            pool_k_features.append(feat_k)

        if self_attention_hops is None:
            sa_edges = [edges for vertices, edges in self.icosphere_graphs]
        else:
            sa_edges = [
                icosphere_nhops_edges(vertices, vertices, n_hops=n_hops)
                for (vertices, _), n_hops in zip(self.icosphere_graphs, self_attention_hops)
            ]

        # CREATE AE ARCHITECTURE

        self.encoder = []
        self.decoder = []
        self.encoder_context_embedding = []
        self.decoder_context_embedding = []

        for i in range(len(hid_channels)):
            if i == 0:
                in_dim = in_channels
                out_dim = in_channels
                self.encoder_context_embedding.append(
                    SirenEmbedding(
                        4 + context_channels,
                        channels=in_dim,
                        n_layers=1,
                        checkpointing=checkpointing,
                    )
                )
            else:
                in_dim = out_dim = hid_channels[i - 1]

            self.encoder.append(
                GraphPoolAttention(
                    q_coords=pool_q_features[i],
                    k_coords=pool_k_features[i],
                    edges=pool_edges[i],
                    in_channels=in_dim,
                    hid_channels=hid_channels[i],
                    out_channels=hid_channels[i],
                    heads=heads[i],
                    k_to_q_idx=encoder_q_idx[i],
                    dropout=dropout,
                    identity_init=identity_init,
                    checkpointing=checkpointing,
                    use_xfa=use_xfa,
                )
            )

            self.decoder.insert(
                0,
                GraphPoolAttention(
                    q_coords=pool_k_features[i],
                    k_coords=pool_q_features[i],
                    edges=pool_edges[i].flip(-1),
                    in_channels=hid_channels[i],
                    hid_channels=hid_channels[i],
                    out_channels=out_dim,
                    heads=heads[i],
                    k_to_q_idx=decoder_q_idx[i],
                    dropout=dropout,
                    identity_init=identity_init,
                    checkpointing=checkpointing,
                    use_xfa=use_xfa,
                ),
            )

            self.encoder_context_embedding.append(
                SirenEmbedding(
                    4 + context_channels,
                    channels=hid_channels[i],
                    n_layers=1,
                    checkpointing=checkpointing,
                )
            )

            self.decoder_context_embedding.insert(
                0,
                SirenEmbedding(
                    4 + context_channels,
                    channels=hid_channels[i],
                    n_layers=1,
                    checkpointing=checkpointing,
                ),
            )

            for _ in range(self_attention_blocks[i]):
                self.encoder.append(
                    GraphSelfAttention(
                        channels=hid_channels[i],
                        edges=sa_edges[i],
                        heads=heads[i],
                        dropout=dropout,
                        checkpointing=checkpointing,
                        use_xfa=use_xfa,
                    )
                )
                self.decoder.insert(
                    0,
                    GraphSelfAttention(
                        channels=hid_channels[i],
                        edges=sa_edges[i],
                        heads=heads[i],
                        dropout=dropout,
                        checkpointing=checkpointing,
                        use_xfa=use_xfa,
                    ),
                )

            if i + 1 == len(hid_channels):
                self.encoder.append(
                    Linear(
                        hid_channels[i],
                        latent_channels,
                        identity_init=identity_init,
                    )
                )
                self.decoder.insert(
                    0,
                    Linear(
                        latent_channels,
                        hid_channels[i],
                        identity_init=identity_init,
                    ),
                )

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.encoder_context_embedding = nn.ModuleList(self.encoder_context_embedding)
        self.decoder_context_embedding = nn.ModuleList(self.decoder_context_embedding)

        self.saturation_bound = saturation_bound
        self.saturation = saturation
        self.noise_level = noise_level

    @property
    def latent_shape(self) -> Tuple[int, int]:
        num_nodes = self.icosphere_graphs[-1][0].shape[0]
        latent_channels = self.encoder[-1].out_channels

        return num_nodes, latent_channels

    def description(self):
        # TODO: Improve description
        return "Graph autoencoder"

    def static_pool(self, context: Tensor) -> Sequence[Tensor]:
        r"""Pools the context to each ICO level.

        Arguments:
            context: The auto-encoding context, with shape :math:`(*, N, K)`.

        Returns:
            A list of pooled context, each with shape :math:`(*, N_i, K)`.
        """

        static_pool_indices = torch.split(self.static_pool_indices_cat, self.static_pool_sizes)

        return [context[..., i, :] for i in static_pool_indices]

    def saturate(self, z: Tensor) -> Tensor:
        if self.saturation is None:
            return z
        elif self.saturation == "softclip":
            return z / (1 + torch.abs(z) / self.saturation_bound)
        elif self.saturation == "softclip2":
            return z * torch.rsqrt(1 + torch.square(z / self.saturation_bound))
        elif self.saturation == "tanh":
            return self.saturation_bound * torch.tanh(z / self.saturation_bound)
        elif self.saturation == "asinh":
            return torch.asinh(z)
        else:
            raise ValueError(f"Unknown saturation function: {self.saturation}")

    def encode_date(self, date: Tensor):
        year_progress = get_year_progress_encoding(date).unsqueeze(-2)
        local_hour = get_local_time_encoding(date, self.in_longitudes)
        year_progress = year_progress.repeat(1, local_hour.shape[1], 1)
        date = torch.cat([year_progress, local_hour], dim=-1)

        return date

    def forward(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None):
        r"""
        Arguments:
            x: State tensor containing variables of interest, with shape :math:`(B, N, Z)`.
            t: Timestamp tensor, with shape :math:`(B, 4)`.
            c: Context tensor, with shape :math:`(B, N, K)`.
        """
        z = self.encode(x, t, c)
        x = self.decode(z, t, c)
        return z, x

    def encode(self, x: Tensor, t: Tensor, c: Optional[Tensor] = None):
        # Compose context as timestamp features and given physical context
        t = self.encode_date(t)
        if c is not None:
            hr_context = torch.cat([t, c], dim=-1)
        else:
            hr_context = t
        pooled_context = self.static_pool(hr_context)

        context_idx = 0
        x = x + self.encoder_context_embedding[context_idx](hr_context)

        for e_block in self.encoder:
            x = e_block(x)
            if isinstance(e_block, GraphPoolAttention):
                context_idx += 1
                x = x + self.encoder_context_embedding[context_idx](
                    pooled_context[context_idx - 1]
                )

        x = self.saturate(x)

        return x

    def decode(self, z: Tensor, t: Tensor, c: Optional[Tensor] = None):
        # Compose context as timestamp features and given physical context
        t = self.encode_date(t)
        if c is not None:
            hr_context = torch.cat([t, c], dim=-1)
        else:
            hr_context = t
        pooled_context = self.static_pool(hr_context)
        pooled_context.reverse()

        z = z + torch.randn_like(z) * self.noise_level

        context_idx = 0
        for d_block in self.decoder:
            if isinstance(d_block, GraphPoolAttention):
                z = z + self.decoder_context_embedding[context_idx](pooled_context[context_idx])
                context_idx += 1
            z = d_block(z)

        return z
