import torch

from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from torch import Tensor
from typing import Callable, Iterable, Optional, Sequence, Tuple


def project_conflicting_gradients(
    gradients,
    group_size=None,
    group_masks=None,
    eps=1e-12,
    projection_orders=None,
):
    """Apply PCGrad globally or with one projection order per vector group.

    A two-dimensional ``projection_orders`` tensor has shape ``(S, S)`` and
    supplies one global permutation for each of the ``S`` target gradients. A
    three-dimensional tensor has shape ``(G, S, S)`` and supplies those target
    permutations independently for ``G`` spatial groups. Spatial orders repeat
    over leading batch/time copies of the same ``G`` groups.
    """
    if not gradients:
        return []
    original_shapes = [gradient.shape for gradient in gradients]
    if group_size is None:
        grouped = [gradient.reshape(1, -1) for gradient in gradients]
    else:
        grouped = [gradient.reshape(-1, group_size) for gradient in gradients]

    num_gradients = len(grouped)
    num_groups = grouped[0].shape[0]
    spatial_orders = None
    if projection_orders is None:
        projection_orders = [range(num_gradients) for _ in range(num_gradients)]
    else:
        order_tensor = torch.as_tensor(
            projection_orders,
            dtype=torch.long,
            device=grouped[0].device,
        )
        if order_tensor.ndim == 3:
            if tuple(order_tensor.shape[1:]) != (
                num_gradients,
                num_gradients,
            ):
                raise ValueError(
                    "spatial PCGrad projection orders must have shape "
                    f"(num_order_groups, {num_gradients}, {num_gradients})"
                )
            num_order_groups = int(order_tensor.shape[0])
            if num_order_groups < 1 or num_groups % num_order_groups != 0:
                raise ValueError(
                    "gradient vector-group count must be a positive multiple "
                    "of the spatial PCGrad order-group count"
                )
            expected = torch.arange(
                num_gradients,
                dtype=torch.long,
                device=order_tensor.device,
            ).expand(num_order_groups, num_gradients, -1)
            if not torch.equal(
                torch.sort(order_tensor, dim=-1).values,
                expected,
            ):
                raise ValueError(
                    "each spatial PCGrad projection order must be a full permutation"
                )
            spatial_orders = order_tensor.repeat(
                num_groups // num_order_groups,
                1,
                1,
            )
        elif order_tensor.ndim == 2:
            projection_orders = order_tensor.tolist()
        else:
            raise ValueError(
                "PCGrad projection orders must have shape (S,S) or (G,S,S)"
            )

    normalized_orders = None
    if spatial_orders is None:
        if len(projection_orders) != num_gradients:
            raise ValueError(
                "projection_orders must contain one permutation per gradient"
            )
        normalized_orders = []
        for order in projection_orders:
            order = [int(index) for index in order]
            if sorted(order) != list(range(num_gradients)):
                raise ValueError(
                    "each PCGrad projection order must be a full permutation"
                )
            normalized_orders.append(order)

    projected = [gradient.clone() for gradient in grouped]
    if spatial_orders is None:
        for i, gradient in enumerate(projected):
            for j in normalized_orders[i]:
                if i == j:
                    continue
                other = grouped[j]
                dot = torch.sum(gradient * other, dim=-1, keepdim=True)
                conflict = dot < 0
                if group_masks is not None:
                    overlap = group_masks[i].reshape(-1, 1) & group_masks[j].reshape(
                        -1, 1
                    )
                    conflict = conflict & overlap
                coefficient = torch.where(
                    conflict,
                    dot
                    / torch.sum(other.square(), dim=-1, keepdim=True).clamp_min(eps),
                    torch.zeros_like(dot),
                )
                gradient = gradient - coefficient * other
            projected[i] = gradient
    else:
        source_gradients = torch.stack(grouped, dim=1)
        group_indices = torch.arange(num_groups, device=source_gradients.device)
        source_masks = None
        if group_masks is not None:
            if len(group_masks) != num_gradients:
                raise ValueError("group_masks must contain one mask per gradient")
            masks = [mask.reshape(-1).bool() for mask in group_masks]
            if any(mask.numel() != num_groups for mask in masks):
                raise ValueError(
                    "each group mask must contain one value per vector group"
                )
            source_masks = torch.stack(masks, dim=1)

        for i, gradient in enumerate(projected):
            for position in range(num_gradients):
                other_ids = spatial_orders[:, i, position]
                other = source_gradients[group_indices, other_ids]
                active = (other_ids != i).unsqueeze(-1)
                dot = torch.sum(gradient * other, dim=-1, keepdim=True)
                conflict = (dot < 0) & active
                if source_masks is not None:
                    overlap = source_masks[:, i] & source_masks[
                        group_indices,
                        other_ids,
                    ]
                    conflict = conflict & overlap.unsqueeze(-1)
                coefficient = torch.where(
                    conflict,
                    dot
                    / torch.sum(other.square(), dim=-1, keepdim=True).clamp_min(eps),
                    torch.zeros_like(dot),
                )
                gradient = gradient - coefficient * other
            projected[i] = gradient
    return [
        gradient.reshape(shape)
        for gradient, shape in zip(projected, original_shapes, strict=True)
    ]


def merge_gradients_by_priority(
    gradients,
    order,
    group_size=None,
    group_masks=None,
    eps=1e-12,
    aligned_clip=0.0,
    conflict_weight=1.0,
):
    """Sequentially merge gradients while protecting higher-priority directions.

    ``aligned_clip`` controls the positive parallel component. Zero is additive
    fusion, positive values interpolate toward max fusion, and negative values
    amplify the next (higher-priority) gradient's aligned direction. Orthogonal
    components remain unchanged.
    """
    if not gradients:
        raise ValueError("gradients must contain at least one tensor")
    aligned_clip = float(aligned_clip)
    if not -1.0 <= aligned_clip <= 1.0:
        raise ValueError("aligned_clip must be in [-1, 1]")
    conflict_weight = float(conflict_weight)
    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be in [0, 1]")

    original_shape = gradients[0].shape
    if any(gradient.shape != original_shape for gradient in gradients[1:]):
        raise ValueError("all gradients must have the same shape")

    if group_size is None:
        grouped = [gradient.reshape(1, -1) for gradient in gradients]
    else:
        grouped = [gradient.reshape(-1, group_size) for gradient in gradients]

    num_gradients = len(grouped)
    order = [int(index) for index in order]
    if sorted(order) != list(range(num_gradients)):
        raise ValueError("priority order must be a full permutation")

    masks = None
    if group_masks is not None:
        if len(group_masks) != num_gradients:
            raise ValueError("group_masks must contain one mask per gradient")
        num_groups = grouped[0].shape[0]
        masks = [mask.reshape(-1, 1).bool() for mask in group_masks]
        if any(mask.shape[0] != num_groups for mask in masks):
            raise ValueError("each group mask must contain one value per vector group")

    accumulator = grouped[order[0]].clone()
    accumulated_mask = masks[order[0]].clone() if masks is not None else None
    for index in order[1:]:
        next_gradient = grouped[index]
        dot = torch.sum(accumulator * next_gradient, dim=-1, keepdim=True)
        next_norm_square = torch.sum(
            next_gradient.square(), dim=-1, keepdim=True
        )
        coefficient = dot / next_norm_square.clamp_min(eps)
        if masks is not None:
            overlap = accumulated_mask & masks[index]
        else:
            overlap = torch.ones_like(dot, dtype=torch.bool)

        conflict_coefficient = torch.where(
            (dot < 0) & overlap, coefficient, torch.zeros_like(coefficient)
        )
        aligned_coefficient = torch.where(
            (dot > 0) & overlap,
            coefficient.clamp(min=0.0, max=1.0),
            torch.zeros_like(coefficient),
        )
        accumulator = accumulator - conflict_coefficient * next_gradient
        accumulator = (
            accumulator
            + next_gradient
            - aligned_clip * aligned_coefficient * next_gradient
        )
        if masks is not None:
            accumulated_mask = accumulated_mask | masks[index]

    if conflict_weight < 1.0:
        summed = torch.stack(grouped).sum(0)
        accumulator = summed + conflict_weight * (accumulator - summed)

    return accumulator.reshape(original_shape)


def merge_gradients_by_group_priority(
    gradients,
    orders,
    group_size,
    group_masks=None,
    eps=1e-12,
    aligned_clip=0.0,
    conflict_weight=1.0,
):
    """Apply vector PPGF with one source order per spatial vector group.

    The order matrix has one row per spatial group and is reused for every
    leading batch/time copy. Each group contains group_size scalar channels and
    follows the same vector projection as merge_gradients_by_priority.
    """
    if not gradients:
        raise ValueError("gradients must contain at least one tensor")
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    aligned_clip = float(aligned_clip)
    if not -1.0 <= aligned_clip <= 1.0:
        raise ValueError("aligned_clip must be in [-1, 1]")
    conflict_weight = float(conflict_weight)
    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be in [0, 1]")

    original_shape = gradients[0].shape
    if any(gradient.shape != original_shape for gradient in gradients[1:]):
        raise ValueError("all gradients must have the same shape")
    if gradients[0].numel() % group_size != 0:
        raise ValueError("gradient size must be divisible by group_size")

    grouped = [gradient.reshape(-1, group_size) for gradient in gradients]
    num_gradients = len(grouped)
    num_groups = grouped[0].shape[0]
    orders = torch.as_tensor(
        orders,
        dtype=torch.long,
        device=grouped[0].device,
    )
    if orders.ndim != 2 or orders.shape[1] != num_gradients:
        raise ValueError(
            "group priority orders must have shape "
            f"(num_order_groups, {num_gradients})"
        )
    num_order_groups = int(orders.shape[0])
    if num_order_groups < 1 or num_groups % num_order_groups != 0:
        raise ValueError(
            "gradient vector-group count must be a positive multiple of "
            "the priority-order group count"
        )
    expected = torch.arange(
        num_gradients,
        dtype=torch.long,
        device=orders.device,
    ).expand(num_order_groups, -1)
    if not torch.equal(torch.sort(orders, dim=1).values, expected):
        raise ValueError("each group priority order must be a full permutation")
    orders = orders.repeat(num_groups // num_order_groups, 1)

    stacked = torch.stack(grouped, dim=0)
    gather_index = orders.t().unsqueeze(-1).expand(
        -1,
        -1,
        group_size,
    )
    ordered = torch.gather(stacked, 0, gather_index)

    ordered_masks = None
    if group_masks is not None:
        if len(group_masks) != num_gradients:
            raise ValueError("group_masks must contain one mask per gradient")
        masks = [mask.reshape(-1).bool() for mask in group_masks]
        if any(mask.numel() != num_groups for mask in masks):
            raise ValueError("each group mask must contain one value per vector group")
        ordered_masks = torch.gather(
            torch.stack(masks, dim=0),
            0,
            orders.t(),
        )

    accumulator = ordered[0].clone()
    accumulated_mask = (
        ordered_masks[0].unsqueeze(-1) if ordered_masks is not None else None
    )
    for position in range(1, num_gradients):
        next_gradient = ordered[position]
        dot = torch.sum(accumulator * next_gradient, dim=-1, keepdim=True)
        next_norm_square = torch.sum(next_gradient.square(), dim=-1, keepdim=True)
        coefficient = dot / next_norm_square.clamp_min(eps)
        if ordered_masks is not None:
            next_mask = ordered_masks[position].unsqueeze(-1)
            overlap = accumulated_mask & next_mask
            accumulated_mask = accumulated_mask | next_mask
        else:
            overlap = torch.ones_like(dot, dtype=torch.bool)

        conflict_coefficient = torch.where(
            (dot < 0) & overlap, coefficient, torch.zeros_like(coefficient)
        )
        aligned_coefficient = torch.where(
            (dot > 0) & overlap,
            coefficient.clamp(min=0.0, max=1.0),
            torch.zeros_like(coefficient),
        )
        accumulator = accumulator - conflict_coefficient * next_gradient
        accumulator = (
            accumulator
            + next_gradient
            - aligned_clip * aligned_coefficient * next_gradient
        )

    if conflict_weight < 1.0:
        summed = stacked.sum(0)
        accumulator = summed + conflict_weight * (accumulator - summed)

    return accumulator.reshape(original_shape)


def merge_gradients_by_channel_priority(
    gradients,
    orders,
    group_size,
    group_masks=None,
    conflict_weight=1.0,
):
    """Merge scalar gradient channels using one priority order per channel.

    Sources enter from lower to higher priority. At each latent scalar, an
    opposite-sign higher-priority value replaces the accumulator; otherwise it
    is added. With masks, replacement is restricted to accumulated observation
    overlap, while non-overlapping values remain additive.
    """
    group_size = int(group_size)
    conflict_weight = float(conflict_weight)
    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be in [0, 1]")

    original_shape = gradients[0].shape
    if any(gradient.shape != original_shape for gradient in gradients[1:]):
        raise ValueError("all gradients must have the same shape")
    if gradients[0].numel() % group_size != 0:
        raise ValueError("gradient size must be divisible by group_size")

    grouped = [gradient.reshape(-1, group_size) for gradient in gradients]
    num_gradients = len(grouped)
    num_groups = grouped[0].shape[0]
    orders = torch.as_tensor(
        orders,
        dtype=torch.long,
        device=grouped[0].device,
    )
    expected_shape = (group_size, num_gradients)
    if tuple(orders.shape) != expected_shape:
        raise ValueError(
            "channel priority orders must have shape " f"{expected_shape}"
        )
    expected = torch.arange(
        num_gradients,
        dtype=torch.long,
        device=orders.device,
    ).expand(group_size, -1)
    if not torch.equal(torch.sort(orders, dim=1).values, expected):
        raise ValueError("each channel priority order must be a full permutation")

    stacked = torch.stack(grouped, dim=0)
    gather_index = orders.t().reshape(
        num_gradients, 1, group_size
    ).expand(-1, num_groups, -1)
    ordered = torch.gather(stacked, 0, gather_index)

    ordered_masks = None
    if group_masks is not None:
        if len(group_masks) != num_gradients:
            raise ValueError("group_masks must contain one mask per gradient")
        masks = [mask.reshape(-1).bool() for mask in group_masks]
        if any(mask.numel() != num_groups for mask in masks):
            raise ValueError("each group mask must contain one value per vector group")
        stacked_masks = torch.stack(masks, dim=0).unsqueeze(-1).expand(
            -1, -1, group_size
        )
        ordered_masks = torch.gather(stacked_masks, 0, gather_index)

    accumulator = ordered[0].clone()
    accumulated_mask = ordered_masks[0].clone() if ordered_masks is not None else None
    for position in range(1, num_gradients):
        next_gradient = ordered[position]
        conflict = (
            (accumulator != 0)
            & (next_gradient != 0)
            & (torch.signbit(accumulator) != torch.signbit(next_gradient))
        )
        if ordered_masks is not None:
            next_mask = ordered_masks[position]
            conflict = conflict & accumulated_mask & next_mask
            accumulated_mask = accumulated_mask | next_mask
        accumulator = torch.where(
            conflict,
            next_gradient,
            accumulator + next_gradient,
        )

    if conflict_weight < 1.0:
        summed = stacked.sum(0)
        accumulator = summed + conflict_weight * (accumulator - summed)

    return accumulator.reshape(original_shape)


def merge_gradients_by_channel_priority_accumulate(
    gradients,
    orders,
    group_size,
    conflict_weight=1.0,
):
    """Accumulate gradients aligned with each channel's most reliable source.

    ``orders`` lists sources from lower to higher priority for every scalar
    channel. The final source is the reference: a nonzero reference keeps and
    sums every source value with the same sign, while a zero reference produces
    zero. ``conflict_weight`` interpolates this result with the exact source
    sum and is kept for the common PPGF sampler interface.
    """
    group_size = int(group_size)
    conflict_weight = float(conflict_weight)
    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be in [0, 1]")
    if not gradients:
        raise ValueError("gradients must contain at least one tensor")

    original_shape = gradients[0].shape
    if any(gradient.shape != original_shape for gradient in gradients[1:]):
        raise ValueError("all gradients must have the same shape")
    if gradients[0].numel() % group_size != 0:
        raise ValueError("gradient size must be divisible by group_size")

    grouped = [gradient.reshape(-1, group_size) for gradient in gradients]
    num_gradients = len(grouped)
    num_groups = grouped[0].shape[0]
    orders = torch.as_tensor(
        orders,
        dtype=torch.long,
        device=grouped[0].device,
    )
    expected_shape = (group_size, num_gradients)
    if tuple(orders.shape) != expected_shape:
        raise ValueError(
            "channel priority orders must have shape " f"{expected_shape}"
        )
    expected = torch.arange(
        num_gradients,
        dtype=torch.long,
        device=orders.device,
    ).expand(group_size, -1)
    if not torch.equal(torch.sort(orders, dim=1).values, expected):
        raise ValueError("each channel priority order must be a full permutation")

    stacked = torch.stack(grouped, dim=0)
    gather_index = orders.t().reshape(
        num_gradients, 1, group_size
    ).expand(-1, num_groups, -1)
    ordered = torch.gather(stacked, 0, gather_index)
    reference = ordered[-1]
    same_sign = (
        reference.unsqueeze(0).ne(0)
        & ordered.ne(0)
        & (torch.signbit(ordered) == torch.signbit(reference).unsqueeze(0))
    )
    accumulated = torch.where(same_sign, ordered, 0).sum(dim=0)

    if conflict_weight < 1.0:
        summed = stacked.sum(0)
        accumulated = summed + conflict_weight * (accumulated - summed)

    return accumulated.reshape(original_shape)


def merge_gradients_by_channel_priority_top1(
    gradients,
    orders,
    group_size,
    conflict_weight=1.0,
):
    """Keep only the highest-priority available source at each scalar channel.

    ``orders`` lists sources from lower to higher priority. The final source is
    selected independently for every spatial latent channel. When a window is
    missing a source, the caller supplies the corresponding reduced local
    ordering, so the final entry automatically falls back to the most reliable
    source that is actually available.
    """
    group_size = int(group_size)
    conflict_weight = float(conflict_weight)
    if not 0.0 <= conflict_weight <= 1.0:
        raise ValueError("conflict_weight must be in [0, 1]")
    if not gradients:
        raise ValueError("gradients must contain at least one tensor")

    original_shape = gradients[0].shape
    if any(gradient.shape != original_shape for gradient in gradients[1:]):
        raise ValueError("all gradients must have the same shape")
    if gradients[0].numel() % group_size != 0:
        raise ValueError("gradient size must be divisible by group_size")

    grouped = [gradient.reshape(-1, group_size) for gradient in gradients]
    num_gradients = len(grouped)
    num_groups = grouped[0].shape[0]
    orders = torch.as_tensor(
        orders,
        dtype=torch.long,
        device=grouped[0].device,
    )
    expected_shape = (group_size, num_gradients)
    if tuple(orders.shape) != expected_shape:
        raise ValueError(
            "channel priority orders must have shape " f"{expected_shape}"
        )
    expected = torch.arange(
        num_gradients,
        dtype=torch.long,
        device=orders.device,
    ).expand(group_size, -1)
    if not torch.equal(torch.sort(orders, dim=1).values, expected):
        raise ValueError("each channel priority order must be a full permutation")

    stacked = torch.stack(grouped, dim=0)
    gather_index = orders.t().reshape(
        num_gradients, 1, group_size
    ).expand(-1, num_groups, -1)
    selected = torch.gather(stacked, 0, gather_index)[-1]
    if conflict_weight < 1.0:
        summed = stacked.sum(0)
        selected = summed + conflict_weight * (selected - summed)
    return selected.reshape(original_shape)


def gmres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system :math:`Ax = b` with generalized minimal residual (GMRES) iterations.

    The matrix :math:`A \in \mathbb{R}^{D \times D}` can be non-symmetric non-definite.

    Wikipedia:
        https://wikipedia.org/wiki/Generalized_minimal_residual_method

    Warning:
        This function is optimized for GPU execution. To avoid CPU-GPU communication,
        all iterations are performed regardless of convergence.

    Arguments:
        A: The linear operator :math:`x \mapsto Ax`.
        b: The right-hand side vector :math:`b`, with shape :math:`(*, D)`.
        x0: An initial guess :math:`x_0`, with shape :math:`(*, D)`. If :py:`None`, use
            :math:`x_0 = 0` instead.
        iterations: The number of GMRES iterations :math:`n`.
        dtype: The data type used for intermediate computations. If :py:`None`, use
            :class:`torch.float64` instead.

    Returns:
        The :math:`n`-th iteration :math:`x_n`, with shape :math:`(*, D)`.
    """

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

        return x, norm

    def rotation(a, b):
        c = torch.clip(torch.sqrt(a * a + b * b), min=epsilon)
        return a / c, -b / c

    V = [None for _ in range(iterations + 1)]
    B = [None for _ in range(iterations + 1)]
    H = [[None for _ in range(iterations)] for _ in range(iterations + 1)]
    cs = [None for _ in range(iterations)]
    ss = [None for _ in range(iterations)]

    V[0], B[0] = normalize(r)

    for j in range(iterations):
        v = V[j].to(b)
        w = A(v).to(dtype)

        # Apply Arnoldi iteration to get the j+1-th basis
        for i in range(j + 1):
            H[i][j] = torch.einsum("...i,...i", w, V[i])
            w = w - H[i][j][..., None] * V[i]
        w, w_norm = normalize(w)
        H[j + 1][j] = w_norm
        V[j + 1] = w

        # Apply Givens rotation
        for i in range(j):
            tmp = cs[i] * H[i][j] - ss[i] * H[i + 1][j]
            H[i + 1][j] = cs[i] * H[i + 1][j] + ss[i] * H[i][j]
            H[i][j] = tmp

        cs[j], ss[j] = rotation(H[j][j], H[j + 1][j])
        H[j][j] = cs[j] * H[j][j] - ss[j] * H[j + 1][j]

        # Update residual vector
        B[j + 1] = ss[j] * B[j]
        B[j] = cs[j] * B[j]

        # Fill with zeros
        for i in range(j + 1, iterations + 1):
            H[i][j] = torch.zeros_like(H[j][j])

    V, B, H = V[:-1], B[:-1], H[:-1]

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


class GaussLegendre(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
        *phi: Tensor,
    ) -> Tensor:
        ctx.f, ctx.n = f, n
        ctx.save_for_backward(a, b, *phi)

        return GaussLegendre.quadrature(f, a, b, n)

    @staticmethod
    def backward(ctx, grad_area: Tensor) -> Tuple[Tensor, ...]:
        f, n = ctx.f, ctx.n
        a, b, *phi = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        if phi:
            with torch.enable_grad():
                area = GaussLegendre.quadrature(f, a, b, n)

            grad_phi = torch.autograd.grad(
                area, phi, grad_area, create_graph=True, allow_unused=True
            )
        else:
            grad_phi = ()

        return (None, grad_a, grad_b, None, *grad_phi)

    @staticmethod
    @lru_cache(maxsize=None)
    def nodes(n: int, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Returns the nodes and weights for a :math:`n`-point Gauss-Legendre
        quadrature over the interval :math:`[0, 1]`.

        See :func:`numpy.polynomial.legendre.leggauss`.
        """

        nodes, weights = leggauss(n)

        nodes = (nodes + 1) / 2
        weights = weights / 2

        kwargs.setdefault("dtype", torch.get_default_dtype())

        return (
            torch.as_tensor(nodes, **kwargs),
            torch.as_tensor(weights, **kwargs),
        )

    @staticmethod
    def quadrature(
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        n: int,
    ) -> Tensor:
        nodes, weights = GaussLegendre.nodes(n, dtype=a.dtype, device=a.device)
        nodes = torch.lerp(
            a[..., None],
            b[..., None],
            nodes,
        ).movedim(-1, 0)

        return (b - a) * torch.tensordot(weights, f(nodes), dims=1)


def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
    phi: Iterable[Tensor] = (),
) -> Tensor:
    r"""Estimates the definite integral of a function :math:`f_\phi(x)` from :math:`a`
    to :math:`b` using a :math:`n`-point Gauss-Legendre quadrature.

    .. math:: \int_a^b f_\phi(x) ~ dx \approx (b - a) \sum_{i = 1}^n w_i f_\phi(x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Gauss-Legendre_quadrature

    References:
        | zuko library (François Rozet)
        | https://github.com/probabilists/zuko/

    Arguments:
        f: A univariate function :math:`f_\phi`.
        a: The lower limit :math:`a`.
        b: The upper limit :math:`b`.
        n: The number of points :math:`n` at which the function is evaluated.
        phi: The parameters :math:`\phi` of :math:`f_\phi`.

    Returns:
        The definite integral estimation.

    Example:
        >>> f = lambda x: torch.exp(-x**2)
        >>> a, b = torch.tensor([-0.69, 4.2])
        >>> gauss_legendre(f, a, b, n=16)
        tensor(1.4807)
    """

    return GaussLegendre.apply(f, a, b, n, *phi)


def str_to_ids(ids_str: str) -> Sequence[int]:
    """
    Convert a string of IDs to a list of integers.

    e.g.:
        '1,2,3'    -> [1, 2, 3]
        '1-3'      -> [1, 2, 3]
        '1,2,8-10' -> [1, 2, 8, 9, 10]

    Args:
        id_list_str: Formatted string of IDs.

    Returns:
        The list of IDs. (list of int)

    """
    id_list_str = ids_str.split(",")
    id_list = []

    for id_ in id_list_str:
        if "-" in id_:
            start, end = id_.split("-")
            id_list += list(range(int(start), int(end) + 1))
        else:
            id_list.append(int(id_))

    return id_list
