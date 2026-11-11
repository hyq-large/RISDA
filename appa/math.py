from typing import Callable, Optional

import torch
from torch import Tensor


def gmres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if dtype is None:
        dtype = torch.float64
    epsilon = torch.finfo(dtype).smallest_normal
    if x0 is None:
        r = b
    else:
        r = b - A(x0)
    r = r.to(dtype)

    def normalize(x):
        norm = torch.linalg.vector_norm(x, dim=-1)
        x = x / torch.clip(norm[..., None], min=epsilon)
        return (x, norm)

    def rotation(a, b):
        c = torch.clip(torch.sqrt(a * a + b * b), min=epsilon)
        return (a / c, -b / c)

    V = [None for _ in range(iterations + 1)]
    B = [None for _ in range(iterations + 1)]
    H = [[None for _ in range(iterations)] for _ in range(iterations + 1)]
    cs = [None for _ in range(iterations)]
    ss = [None for _ in range(iterations)]
    (V[0], B[0]) = normalize(r)
    for j in range(iterations):
        v = V[j].to(b)
        w = A(v).to(dtype)
        for i in range(j + 1):
            H[i][j] = torch.einsum("...i,...i", w, V[i])
            w = w - H[i][j][..., None] * V[i]
        (w, w_norm) = normalize(w)
        H[j + 1][j] = w_norm
        V[j + 1] = w
        for i in range(j):
            tmp = cs[i] * H[i][j] - ss[i] * H[i + 1][j]
            H[i + 1][j] = cs[i] * H[i + 1][j] + ss[i] * H[i][j]
            H[i][j] = tmp
        (cs[j], ss[j]) = rotation(H[j][j], H[j + 1][j])
        H[j][j] = cs[j] * H[j][j] - ss[j] * H[j + 1][j]
        B[j + 1] = ss[j] * B[j]
        B[j] = cs[j] * B[j]
        for i in range(j + 1, iterations + 1):
            H[i][j] = torch.zeros_like(H[j][j])
    (V, B, H) = (V[:-1], B[:-1], H[:-1])
    V = torch.stack(V, dim=-2)
    B = torch.stack(B, dim=-1)
    H = torch.stack([torch.stack(Hi, dim=-1) for Hi in H], dim=-2)
    y = torch.linalg.solve_triangular(
        H + epsilon * torch.eye(iterations, dtype=dtype, device=H.device),
        B.unsqueeze(dim=-1),
        upper=True,
    ).squeeze(dim=-1)
    if x0 is None:
        x = torch.einsum("...ij,...i", V, y)
    else:
        x = x0 + torch.einsum("...ij,...i", V, y)
    return x.to(b)
