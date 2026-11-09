r"""Graph network building blocks."""

import functools
import math
import numpy as np
import torch
import torch.nn as nn
import warnings

with warnings.catch_warnings(action="ignore"):
    import xformers.components.attention.core as xfa
    import xformers.sparse as xfs

from einops import rearrange
from torch import LongTensor, Tensor
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.checkpoint import checkpoint
from typing import Optional

import appa.nn.triggers as triggers

from .layers import Linear, SirenEmbedding


class GraphAttention(nn.Module):
    r"""Creates a multi-head graph attention module.

    Arguments:
        input_channels: Number of input channels :math:`C`.
        hidden_channels: Number of hidden channels used for attention.
        heads: Number of attention heads.
        output_linear: Whether the output of multi-head attention is recombined by a linear layer or not.
        chunks: Number of chunks used to split the attention computation.
        qk_norm: Whether to use Query-Key RMS-normalization or not.
        checkpointing: Whether to use gradient checkpointing or not.
        use_xfa: Whether to use xformers sparse attention or not.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        heads: int = 1,
        output_linear: bool = False,
        chunks: int = 4,
        qk_norm: bool = True,
        checkpointing: bool = False,
        use_xfa: bool = True,
    ):
        super().__init__()

        assert hidden_channels % heads == 0, "channels must be a multiple of heads."

        self.q_proj = nn.Linear(input_channels, hidden_channels)
        self.kv_proj = nn.Linear(input_channels, 2 * hidden_channels)

        if output_linear:
            self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        else:
            self.out_proj = nn.Identity()

        if qk_norm:
            self.qk_norm = nn.RMSNorm(hidden_channels // heads, elementwise_affine=False)
        else:
            self.qk_norm = nn.Identity()

        self.heads = heads
        self.chunks = chunks
        self.checkpointing = checkpointing
        self.use_xfa = use_xfa and xfa._has_cpp_library
        self.xfa_warning = False  # Flag for warning about XFA issues

    @staticmethod
    @functools.cache
    def as_sparse_mask(
        edges: LongTensor,
        M: int,
        N: int,
        divisible_by: int = 4,
    ) -> xfs.SparseCSRTensor:
        nonzeros = edges[:, 0] * N + edges[:, 1]
        nonzeros = set(nonzeros.tolist())
        zeros = []

        assert any(
            i % 4 == 0 for i in range(len(nonzeros), M * N)
        ), f"Could not make edges count divisible by {divisible_by}."

        for i in range(M * N):
            if (len(nonzeros) + len(zeros)) % divisible_by == 0:
                break
            elif i not in nonzeros:
                zeros.append(i)

        values = torch.cat((
            torch.zeros(len(zeros), dtype=bool, device=edges.device),
            torch.ones(len(nonzeros), dtype=bool, device=edges.device),
        ))

        indices = torch.cat((
            torch.as_tensor(sorted(zeros), dtype=torch.int64, device=edges.device),
            torch.as_tensor(sorted(nonzeros), dtype=torch.int64, device=edges.device),
        ))

        row_indices, col_indices = indices // N, indices % N
        row_offsets, col_indices = xfs.utils._coo_to_csr(M, N, row_indices, col_indices)
        row_offsets, col_indices = row_offsets.to(torch.int32), col_indices.to(torch.int32)

        return xfs.SparseCSRTensor(row_offsets, col_indices, values[None], (1, M, N))

    @staticmethod
    def fallback_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        edges: LongTensor,
        chunks: int = 1,
    ) -> Tensor:
        r"""Computes a scaled dot-product attention over edges.

        Arguments:
            q: The query tensor :math:`q`, with shape :math:`(*, M, C)`.
            k: The key tensor :math:`k, with shape :math:`(*, N, C)`.
            v: The value tensor :math:`v`, with shape :math:`(*, N, D)`.
            edges: The attention edges, with shape :math:`(E, 2)`.
            chunks: The number of chunks used to split the attention computation.

        Returns:
            The output vector :math:`y`, with shape :math:`(*, M, D)`.
        """

        assert q.ndim == k.ndim == v.ndim

        M, C = q.shape[-2:]
        D = v.shape[-1]

        batch = torch.broadcast_shapes(
            q.shape[:-2],
            k.shape[:-2],
        )

        i, j = torch.unbind(edges, dim=-1)

        # Weights
        weights = []

        for ii, jj in zip(
            torch.chunk(i, chunks, dim=-1),
            torch.chunk(j, chunks, dim=-1),
        ):
            qi = q[..., ii, :]
            kj = k[..., jj, :]

            weights.append(torch.linalg.vecdot(qi, kj))

        weights = torch.cat(weights, dim=-1)
        weights = weights / math.sqrt(C)
        weights = torch.exp(weights)

        total = torch.zeros((*batch, M), dtype=q.dtype, device=q.device)
        total = total.index_add(index=i, source=weights, dim=-1)
        total = total[..., i]

        weights = weights / total

        # Values
        y = torch.zeros((*batch, M, D), dtype=q.dtype, device=q.device)

        for ii, jj, w in zip(
            torch.chunk(i, chunks, dim=-1),
            torch.chunk(j, chunks, dim=-1),
            torch.chunk(weights, chunks, dim=-1),
        ):
            vj = v[..., jj, :]
            vi = w[..., None] * vj

            y = y.index_add(index=ii, source=vi, dim=-2)

        return y

    @staticmethod
    def xfa_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        edges: LongTensor,
    ) -> Tensor:
        r"""Computes a scaled dot-product attention over edges using XFA.

        Arguments:
            q: The query tensor :math:`q`, with shape :math:`(*, M, C)`.
            k: The key tensor :math:`k, with shape :math:`(*, N, C)`.
            v: The value tensor :math:`v`, with shape :math:`(*, N, D)`.
            edges: The attention edges, with shape :math:`(E, 2)`.

        Returns:
            The output vector :math:`y`, with shape :math:`(*, M, D)`.
        """

        M, N = q.shape[-2], k.shape[-2]

        shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
        q = q.expand(*shape, -1, -1)
        k = k.expand(*shape, -1, -1)
        v = v.expand(*shape, -1, -1)

        y = xfa.scaled_dot_product_attention(
            q=rearrange(q, "... M C -> (...) M C"),
            k=rearrange(k, "... N C -> (...) N C"),
            v=rearrange(v, "... N C -> (...) N C"),
            att_mask=xfa.SparseCS._wrap(GraphAttention.as_sparse_mask(edges, M, N)),
        )

        if shape:
            return y.unflatten(0, shape)
        else:
            return y

    def _forward(
        self,
        q: Tensor,
        k: Tensor,
        edges: LongTensor,
    ) -> Tensor:
        q = self.q_proj(q)
        kv = self.kv_proj(k)

        q = rearrange(q, "... M (H C) -> ... H M C", H=self.heads)
        k, v = rearrange(kv, "... N (n H C) -> n ... H N C", H=self.heads, n=2)

        q, k = self.qk_norm(q), self.qk_norm(k)

        batch_size = max((np.prod(t.shape[:-2]) for t in (q, k)))

        if triggers.ENABLE_XFA and self.use_xfa and batch_size > 65536 and not self.xfa_warning:
            self.xfa_warning = True
            warnings.warn(
                f"Batch size of {batch_size} (> 65536) used with XFA, falling back to default attention.",
                UserWarning,
                stacklevel=1,
            )

        if edges is None:  # full attention
            y = scaled_dot_product_attention(q, k, v)
        else:
            if triggers.ENABLE_XFA and self.use_xfa and batch_size < 65536:
                y = self.xfa_attention(q, k, v, edges)
            else:
                y = self.fallback_attention(q, k, v, edges, chunks=self.chunks)

        y = rearrange(y, "... H M C -> ... M (H C)")

        return self.out_proj(y)

    def forward(self, query: Tensor, key: Tensor, edges: LongTensor) -> Tensor:
        r"""
        Arguments:
            query: The query graph, with shape :math:`(*, M, C)`.
            key: The key graph, with shape :math:`(*, N, C)`.
            edges: The attention edges, with shape :math:`(E, 2)`.

        Returns:
            The output graph, with shape :math:`(*, M, C)`.
        """

        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, query, key, edges, use_reentrant=False)
        else:
            return self._forward(query, key, edges)


class GraphPoolAttention(nn.Module):
    r"""Creates a graph attention pooling module.

    Arguments:
        q_coords: The positional features of the query nodes, with shape :math:`(M, F)`.
        k_coords: The positional features of the key nodes, with shape :math:`(N, F)`.
        edges: The query-key edge indices, with shape :math:`(E, 2)`.
        in_channels: The number of input channels.
        hid_channels: The number of channels for the attention.
        out_channels: The number of output channels.
        heads: Number of attention heads.
        identity_init: Whether to use identity-like weight initialization.
    """

    def __init__(
        self,
        q_coords: Tensor,
        k_coords: Tensor,
        edges: Tensor,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        heads: int = 1,
        k_to_q_idx: Optional[Tensor] = None,
        identity_init: bool = False,
        dropout: float = None,
        checkpointing: bool = False,
        use_xfa: bool = True,
    ):
        super().__init__()

        assert (
            q_coords.shape[-1] == k_coords.shape[-1]
        ), "Keys and queries must have the same coordinates dimension."

        _, F = q_coords.shape

        self.features_proj = SirenEmbedding(F, in_channels, n_layers=1)
        self.q_bias = SirenEmbedding(F, hid_channels, n_layers=1)
        self.q_bias.siren[-1].weight.data.mul_(1e-2)

        self.attn = GraphAttention(
            input_channels=in_channels,
            hidden_channels=hid_channels,
            heads=heads,
            output_linear=False,
            use_xfa=use_xfa,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hid_channels, hid_channels * 4),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(hid_channels * 4, hid_channels),
        )
        self.ffn[-1].weight.data.mul_(1e-2)

        self.norm_ffn = nn.LayerNorm(hid_channels, elementwise_affine=False)

        self.out_proj = Linear(hid_channels, out_channels, identity_init=identity_init)

        if identity_init:
            id_offset = torch.cat(
                [torch.eye(in_channels)] * (hid_channels // in_channels)
                + [torch.eye(hid_channels % in_channels, in_channels)],
                dim=0,
            )
            self.attn.kv_proj.weight.data[hid_channels:].mul_(1e-2)
            self.attn.kv_proj.weight.data[hid_channels:].add_(id_offset)

            id_offset = torch.cat(
                [torch.eye(hid_channels)] * (out_channels // hid_channels)
                + [torch.eye(out_channels % hid_channels, hid_channels)],
                dim=0,
            )
            self.out_proj.weight.data.mul_(1e-2)
            self.out_proj.weight.data.add_(id_offset)

        self.register_buffer("q_coords", q_coords)
        self.register_buffer("k_coords", k_coords)
        self.register_buffer("edges", edges)
        if k_to_q_idx is not None:
            self.register_buffer("k_to_q_idx", k_to_q_idx)
        else:
            self.k_to_q_idx = None

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        q = self.features_proj(self.q_coords)
        k = x + self.features_proj(self.k_coords)

        while q.ndim < k.ndim:
            q = q[None]

        if self.k_to_q_idx is not None:
            q = q + x[:, self.k_to_q_idx]

        y = self.attn(q, k, self.edges) + self.q_bias(self.q_coords)
        y = y + self.ffn(self.norm_ffn(y))

        return self.out_proj(y)

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class GraphSelfAttention(nn.Module):
    r"""Creates a graph self-attention module.

    Arguments:
        channels: The number of input channels :math:`C`.
        edges: The query-key edge indices, with shape :math:`(E, 2)`.
        heads: Number of attention heads.
        dropout: The dropout rate.
    """

    def __init__(
        self,
        channels: int,
        edges: Tensor,
        heads: int = 1,
        dropout: float = None,
        checkpointing: bool = False,
        use_xfa: bool = True,
    ):
        super().__init__()

        self.attn = GraphAttention(
            input_channels=channels,
            hidden_channels=channels,
            output_linear=True,
            heads=heads,
            use_xfa=use_xfa,
        )
        self.attn.kv_proj.weight.data[channels:].mul_(1e-2)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
        )
        self.ffn[-1].weight.data.mul_(1e-2)

        self.norm_att = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(channels, elementwise_affine=False)

        self.register_buffer("edges", edges)

        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        r"""
        Arguments:
            x: The input graph, with shape :math:`(B, N, C)`.

        Returns:
            The output graph, with shape :math:`(B, N, C)`.
        """

        x_res = self.norm_att(x)

        y = x + self.attn(x_res, x_res, self.edges)
        y = x + self.ffn(self.norm_ffn(y))

        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing and triggers.ENABLE_CHECKPOINTING:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


# fmt: off
def monkey_coo_to_csr(m, n, row_indices, column_indices):
    row_offsets = row_indices.bincount(minlength=m).cumsum(0, dtype=row_indices.dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))
    return row_offsets, column_indices

def monkey_round_nnz(mask, divisible_by=4):
    nnz = torch.count_nonzero(mask)
    cunz = torch.cumsum(~mask.flatten(), dim=0)
    flip = cunz <= (-nnz) % divisible_by

    return torch.logical_or(mask, flip.reshape_as(mask))

def monkey_masked_matmul(cls, a, b, mask):
    assert mask.shape[1] == a.shape[1]
    assert mask.shape[2] == b.shape[2]

    values = mask._SparseCSRTensor__values
    row_indices = mask._SparseCSRTensor__row_indices
    row_offsets = mask._SparseCSRTensor__row_offsets
    column_indices = mask._SparseCSRTensor__column_indices
    tansp_info = mask._SparseCSRTensor__transp_info

    out = xfs._csr_ops._sddmm.apply(
        a.contiguous(),
        b.transpose(-2, -1).contiguous(),
        row_indices,
        row_offsets,
        column_indices,
        tansp_info,
    )
    out = torch.where(values, out, float("-inf"))

    return cls._wrap(
        mask.shape,
        out,
        row_indices,
        row_offsets,
        column_indices,
        tansp_info,
    )

xfs.utils._coo_to_csr = monkey_coo_to_csr
xfs.utils._round_nnz = monkey_round_nnz
xfs.SparseCSRTensor._masked_matmul = classmethod(monkey_masked_matmul)
# fmt: on
