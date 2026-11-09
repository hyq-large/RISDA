import pytest
import torch
import torch.nn as nn

from appa.diffusion import MultiInstrumentMMPSDenoiser, TrajectoryDenoiser
from appa.math import (
    merge_gradients_by_channel_priority,
    merge_gradients_by_channel_priority_accumulate,
    merge_gradients_by_channel_priority_top1,
    merge_gradients_by_group_priority,
    merge_gradients_by_priority,
    project_conflicting_gradients,
)
from appa.sampling import PCSampler, PPGFSampler


def test_pcgrad_leaves_non_conflicting_gradients_unchanged():
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])]
    projected = project_conflicting_gradients(gradients)
    assert torch.equal(projected[0], gradients[0])
    assert torch.equal(projected[1], gradients[1])


def test_pcgrad_removes_pairwise_negative_components():
    gradients = [torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.0])]
    projected = project_conflicting_gradients(gradients)
    assert torch.dot(projected[0], gradients[1]) >= -1e-6
    assert torch.dot(projected[1], gradients[0]) >= -1e-6
    assert torch.allclose(projected[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(projected[1], torch.tensor([-0.5, 0.5]))


def test_pcgrad_projects_each_pixel_group_independently():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 2.0, 0.0]]),
    ]
    projected = project_conflicting_gradients(gradients, group_size=2)
    assert torch.allclose(projected[0], torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(projected[1], torch.tensor([[0.0, 0.0, 2.0, 0.0]]))


def test_pcgrad_only_projects_conflicts_in_observation_overlap():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, -1.0, 0.0]]),
    ]
    masks = [torch.tensor([True, True]), torch.tensor([True, False])]
    projected = project_conflicting_gradients(gradients, group_size=2, group_masks=masks)
    assert torch.allclose(projected[0], torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    assert torch.allclose(projected[1], torch.tensor([[0.0, 0.0, -1.0, 0.0]]))


def test_pcgrad_keeps_overlapping_non_conflicting_groups():
    gradients = [torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]])]
    masks = [torch.tensor([True]), torch.tensor([True])]
    projected = project_conflicting_gradients(gradients, group_size=2, group_masks=masks)
    assert torch.equal(projected[0], gradients[0])
    assert torch.equal(projected[1], gradients[1])


class _IdentityDenoiser(nn.Module):
    def forward(self, x_t, sigma_t, **kwargs):
        return x_t


class _AllToAllDenoiser(nn.Module):
    def forward(self, x_t, sigma_t, **kwargs):
        return x_t.sum(dim=-1, keepdim=True).repeat(1, x_t.shape[-1])


def test_multi_instrument_hard_masks_score_before_diagnostics_and_sum():
    common = {
        "denoiser": _AllToAllDenoiser(),
        "A": lambda x, blanket_id=0: x[:, 0].reshape(-1),
        "y": [torch.tensor([3.0])],
        "var_y": [torch.ones(1)],
        "instrument_indices": [[torch.tensor([0])]],
        "merge": "sum",
        "tweedie_covariance": False,
        "iterations": 1,
    }
    unmasked = MultiInstrumentMMPSDenoiser(**common)
    masked = MultiInstrumentMMPSDenoiser(
        **common,
        instrument_gradient_masks=[[
            torch.tensor([True, False]),
        ]],
    )

    with torch.no_grad():
        unmasked_output = unmasked(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
        )
        masked_output = masked(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
        )

    assert torch.allclose(unmasked_output, torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(masked_output, torch.tensor([[2.0, 0.0]]))
    assert torch.allclose(masked.guidance_norm_sum, torch.tensor([2.0]))
    assert torch.equal(masked.guidance_norm_count, torch.ones(1))


def test_multi_instrument_rejects_gradient_mask_with_wrong_score_width():
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_AllToAllDenoiser(),
        A=lambda x, blanket_id=0: x[:, 0].reshape(-1),
        y=[torch.tensor([3.0])],
        var_y=[torch.ones(1)],
        instrument_indices=[[torch.tensor([0])]],
        instrument_gradient_masks=[[
            torch.tensor([True, False, True]),
        ]],
        merge="sum",
        tweedie_covariance=False,
        iterations=1,
    )

    with pytest.raises(ValueError, match="mask width"):
        denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
        )


def test_multi_instrument_masks_only_selected_score_before_ppgf(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["gradients"] = [
            gradient.detach().clone() for gradient in gradients
        ]
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr(
        "appa.diffusion.merge_gradients_by_channel_priority",
        merge,
    )
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_AllToAllDenoiser(),
        A=lambda x, blanket_id=0: torch.stack(
            (x[:, 0], x[:, 0]),
            dim=-1,
        ).reshape(-1),
        y=[torch.tensor([3.0, 6.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[torch.tensor([0]), torch.tensor([1])]],
        instrument_gradient_masks=[[
            None,
            torch.tensor([True, False]),
        ]],
        merge="channel_ppgf",
        pcgrad_group_size=2,
        tweedie_covariance=False,
        iterations=1,
    )

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=torch.tensor([
                [0, 1],
                [0, 1],
            ]),
        )

    assert torch.allclose(captured["gradients"][0], torch.tensor([[2.0, 2.0]]))
    assert torch.allclose(captured["gradients"][1], torch.tensor([[4.0, 0.0]]))
    assert torch.allclose(output, torch.tensor([[6.0, 2.0]]))
    assert torch.allclose(
        denoiser.guidance_norm_sum,
        torch.tensor([8.0**0.5, 4.0]),
    )


def test_multi_instrument_mmps_keeps_slice_grad_under_no_grad():
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[torch.tensor([0]), torch.tensor([1])]],
        merge="pcgrad",
        pcgrad_group_size=1,
        tweedie_covariance=False,
        iterations=1,
    )
    with torch.no_grad():
        output = denoiser(torch.zeros(1, 2), torch.tensor(1.0), blanket_id=0)
    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()
    assert denoiser.guidance_norm_sum.device == denoiser.var_y.device
    assert denoiser.cosine_sum.device == denoiser.var_y.device


def test_multi_instrument_mmps_supports_independent_operators():
    def unused_joint_operator(x, blanket_id=0):
        raise AssertionError("joint operator must not run for independent tasks")

    operators = [
        lambda x, blanket_id=0: x[:, 0].reshape(-1),
        lambda x, blanket_id=0: (2.0 * x[:, 1]).reshape(-1),
    ]
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=unused_joint_operator,
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[torch.tensor([0]), torch.tensor([1])]],
        instrument_operators=operators,
        merge="sum",
        tweedie_covariance=False,
        iterations=1,
    )

    with torch.no_grad():
        output = denoiser(torch.zeros(1, 2), torch.tensor(1.0), blanket_id=0)

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_pcgrad_respects_explicit_projection_order():
    gradients = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([-1.0, 2.0]),
    ]
    first_1_then_2 = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    first_2_then_1 = [[0, 2, 1], [0, 1, 2], [0, 1, 2]]

    projected_a = project_conflicting_gradients(gradients, projection_orders=first_1_then_2)
    projected_b = project_conflicting_gradients(gradients, projection_orders=first_2_then_1)

    assert torch.allclose(projected_a[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(projected_b[0], torch.tensor([0.6, 0.6]))


def test_pcgrad_uses_a_different_fixed_order_for_each_spatial_vector():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
        torch.tensor([[-1.0, 2.0, -1.0, 2.0]]),
    ]
    spatial_orders = torch.tensor([
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 2, 1], [0, 1, 2], [0, 1, 2]],
    ])

    projected = project_conflicting_gradients(
        gradients,
        group_size=2,
        projection_orders=spatial_orders,
    )

    assert torch.allclose(
        projected[0],
        torch.tensor([[0.5, 0.5, 0.6, 0.6]]),
    )


def test_spatial_pcgrad_orders_repeat_over_leading_frame_groups():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0] * 2]),
        torch.tensor([[-1.0, 1.0, -1.0, 1.0] * 2]),
        torch.tensor([[-1.0, 2.0, -1.0, 2.0] * 2]),
    ]
    spatial_orders = torch.tensor([
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        [[0, 2, 1], [0, 1, 2], [0, 1, 2]],
    ])

    projected = project_conflicting_gradients(
        gradients,
        group_size=2,
        projection_orders=spatial_orders,
    )

    assert torch.allclose(
        projected[0],
        torch.tensor([[0.5, 0.5, 0.6, 0.6] * 2]),
    )


def test_spatial_pcgrad_rejects_non_permutation_orders():
    gradients = [torch.ones(1, 4), -torch.ones(1, 4)]
    invalid_orders = torch.tensor([
        [[0, 0], [0, 1]],
        [[0, 1], [0, 1]],
    ])

    with pytest.raises(ValueError, match="full permutation"):
        project_conflicting_gradients(
            gradients,
            group_size=2,
            projection_orders=invalid_orders,
        )


class _IdentitySchedule(nn.Module):
    def forward(self, time):
        return time


def test_pc_sampler_reuses_orders_within_timestep_and_resamples_next_step():
    recorded_orders = []

    def denoiser(x_t, sigma_t, pcgrad_orders=None):
        recorded_orders.append(pcgrad_orders.clone())
        return x_t

    sampler = PCSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        pcgrad_shuffle=True,
        pcgrad_num_tasks=5,
        pcgrad_seed=0,
    )
    x_t = torch.zeros(1, 2)
    sampler.step(x_t, torch.tensor(1.0), torch.tensor(0.5))
    sampler.step(x_t, torch.tensor(0.5), torch.tensor(0.25))

    assert len(recorded_orders) == 6
    assert all(torch.equal(recorded_orders[0], order) for order in recorded_orders[:3])
    assert all(torch.equal(recorded_orders[3], order) for order in recorded_orders[3:])
    assert not torch.equal(recorded_orders[0], recorded_orders[3])
    expected_ids = list(range(5))
    for order_matrix in (recorded_orders[0], recorded_orders[3]):
        assert all(sorted(row.tolist()) == expected_ids for row in order_matrix)


def test_pc_sampler_prints_orders_once_per_timestep(capsys):
    sampler = PCSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        corrections=2,
        pcgrad_shuffle=True,
        pcgrad_num_tasks=3,
        pcgrad_seed=0,
    )

    sampler.step(torch.zeros(1, 2), torch.tensor(1.0), torch.tensor(0.5))

    output = capsys.readouterr().out
    assert output.count("[PCGrad order]") == 1
    assert "diffusion_step=0" in output
    assert "t=1" in output
    assert "mode=random" in output
    assert "orders=[[" in output


def test_pc_sampler_uses_broadcast_orders_on_nonzero_rank(monkeypatch):
    expected = torch.tensor([
        [2, 0, 1],
        [0, 2, 1],
        [1, 0, 2],
    ])

    monkeypatch.setattr("appa.sampling.dist.is_available", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.is_initialized", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.get_rank", lambda: 1)

    def broadcast(orders, src):
        assert src == 0
        orders.copy_(expected)

    monkeypatch.setattr("appa.sampling.dist.broadcast", broadcast)
    sampler = PCSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        pcgrad_shuffle=True,
        pcgrad_num_tasks=3,
        pcgrad_seed=0,
    )

    orders = sampler._sample_pcgrad_orders(torch.device("cpu"))
    assert torch.equal(orders, expected)


def test_multi_instrument_pcgrad_filters_global_orders_for_missing_instrument():
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[
            [
                torch.tensor([0]),
                torch.empty(0, dtype=torch.long),
                torch.tensor([1]),
            ]
        ],
        merge="pcgrad",
        pcgrad_group_size=1,
        pcgrad_shuffle=True,
        tweedie_covariance=False,
        iterations=1,
    )
    orders = torch.tensor([
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
    ])

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            pcgrad_orders=orders,
        )

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_multi_instrument_localizes_spatial_orders_when_a_source_is_missing():
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[
            torch.tensor([0]),
            torch.empty(0, dtype=torch.long),
            torch.tensor([1]),
        ]],
        merge="pcgrad",
        pcgrad_group_size=1,
        pcgrad_shuffle=True,
        tweedie_covariance=False,
        iterations=1,
    )
    spatial_orders = torch.tensor([
        [
            [1, 2, 0],
            [0, 1, 2],
            [0, 1, 2],
        ],
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 1, 0],
        ],
    ])

    localized = denoiser._local_projection_orders(
        spatial_orders,
        instrument_ids=[0, 2],
    )

    expected = torch.tensor([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ])
    assert torch.equal(localized, expected)


def test_uncertainty_weighted_orders_put_reliable_instruments_later():
    sampler = PCSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        pcgrad_shuffle=True,
        pcgrad_num_tasks=3,
        pcgrad_seed=0,
        pcgrad_order_mode="uncertainty",
        pcgrad_uncertainty=torch.tensor([4.0, 1.0, 0.25]),
        pcgrad_uncertainty_strength=2.0,
    )

    position_sum = torch.zeros(3)
    num_rows = 0
    for _ in range(1000):
        orders = sampler._sample_pcgrad_orders(torch.device("cpu"))
        position_sum += torch.argsort(orders, dim=1).float().sum(dim=0)
        num_rows += len(orders)

    mean_position = position_sum / num_rows
    assert mean_position[0] < mean_position[1] < mean_position[2]


def test_fixed_pcgrad_order_puts_reliable_instruments_last_and_reuses_order():
    recorded_orders = []

    def denoiser(x_t, sigma_t, pcgrad_orders=None):
        recorded_orders.append(pcgrad_orders.clone())
        return x_t

    sampler = PCSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        pcgrad_shuffle=True,
        pcgrad_num_tasks=3,
        pcgrad_seed=0,
        pcgrad_order_mode="fixed",
        pcgrad_uncertainty=torch.tensor([1.0, 0.25, 4.0]),
        pcgrad_uncertainty_strength=1.0,
    )
    x_t = torch.zeros(1, 2)
    sampler.step(x_t, torch.tensor(1.0), torch.tensor(0.5))
    sampler.step(x_t, torch.tensor(0.5), torch.tensor(0.25))

    expected = torch.tensor([[2, 0, 1]]).expand(3, -1)
    assert len(recorded_orders) == 6
    assert all(torch.equal(order, expected) for order in recorded_orders)
    assert sampler._pcgrad_generator is None


def test_fixed_pcgrad_order_does_not_depend_on_uncertainty_strength():
    common = {
        "denoiser": lambda x_t, sigma_t, **kwargs: x_t,
        "schedule": _IdentitySchedule(),
        "pcgrad_shuffle": True,
        "pcgrad_num_tasks": 3,
        "pcgrad_order_mode": "fixed",
        "pcgrad_uncertainty": torch.tensor([1.0, 0.25, 4.0]),
    }
    weak = PCSampler(**common, pcgrad_uncertainty_strength=0.01)
    strong = PCSampler(**common, pcgrad_uncertainty_strength=100.0)

    weak_order = weak._sample_pcgrad_orders(torch.device("cpu"))
    strong_order = strong._sample_pcgrad_orders(torch.device("cpu"))

    assert torch.equal(weak_order, strong_order)


def test_fixed_pcgrad_builds_and_reuses_one_order_per_spatial_vector():
    recorded_orders = []

    def denoiser(x_t, sigma_t, pcgrad_orders=None):
        recorded_orders.append(pcgrad_orders.clone())
        return x_t

    sampler = PCSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        pcgrad_shuffle=True,
        pcgrad_num_tasks=3,
        pcgrad_seed=0,
        pcgrad_order_mode="fixed",
        pcgrad_uncertainty=torch.tensor([
            [1.0, 0.25, 4.0],
            [0.1, 5.0, 1.0],
        ]),
        pcgrad_uncertainty_strength=100.0,
    )
    x_t = torch.zeros(1, 2)
    sampler.step(x_t, torch.tensor(1.0), torch.tensor(0.5))
    sampler.step(x_t, torch.tensor(0.5), torch.tensor(0.25))

    expected = torch.tensor([
        [[2, 0, 1], [2, 0, 1], [2, 0, 1]],
        [[1, 2, 0], [1, 2, 0], [1, 2, 0]],
    ])
    assert len(recorded_orders) == 6
    assert all(torch.equal(order, expected) for order in recorded_orders)
    assert sampler._pcgrad_generator is None


def test_ppgf_projects_accumulator_before_adding_next_gradient():
    gradients = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 1.0]),
    ]

    merged = merge_gradients_by_priority(gradients, order=[0, 1])

    assert torch.allclose(merged, torch.tensor([-0.5, 1.5]))
    assert torch.allclose(
        torch.dot(merged, gradients[1]),
        gradients[1].square().sum(),
    )


@pytest.mark.parametrize(
    ("accumulator", "expected"),
    [
        ([2.0, 3.0], [2.0, 3.0]),
        ([0.5, 3.0], [1.0, 3.0]),
        ([0.0, 3.0], [1.0, 3.0]),
    ],
)
def test_ppgf_aligned_clip_keeps_larger_parallel_component(accumulator, expected):
    merged = merge_gradients_by_priority(
        [torch.tensor(accumulator), torch.tensor([1.0, 0.0])],
        order=[0, 1],
        aligned_clip=1.0,
    )

    assert torch.allclose(merged, torch.tensor(expected))


def test_ppgf_aligned_clip_interpolates_from_sum_to_max():
    gradients = [torch.tensor([2.0, 3.0]), torch.tensor([1.0, 0.0])]

    merged = merge_gradients_by_priority(gradients, order=[0, 1], aligned_clip=0.5)

    assert torch.allclose(merged, torch.tensor([2.5, 3.0]))


def test_ppgf_negative_aligned_clip_amplifies_higher_priority_direction():
    gradients = [torch.tensor([2.0, 3.0]), torch.tensor([1.0, 0.0])]

    merged = merge_gradients_by_priority(gradients, order=[0, 1], aligned_clip=-0.4)

    assert torch.allclose(merged, torch.tensor([3.4, 3.0]))


def test_ppgf_aligned_clip_preserves_conflict_projection():
    gradients = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 1.0]),
    ]

    merged = merge_gradients_by_priority(gradients, order=[0, 1], aligned_clip=1.0)

    assert torch.allclose(merged, torch.tensor([-0.5, 1.5]))


def test_ppgf_aligned_clip_only_applies_in_accumulated_overlap():
    gradients = [
        torch.tensor([[2.0, 0.0, 2.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    ]
    masks = [torch.tensor([True, True]), torch.tensor([True, False])]

    merged = merge_gradients_by_priority(
        gradients,
        order=[0, 1],
        group_size=2,
        group_masks=masks,
        aligned_clip=1.0,
    )

    assert torch.allclose(merged, torch.tensor([[2.0, 0.0, 3.0, 0.0]]))


@pytest.mark.parametrize("aligned_clip", [-1.1, 1.1])
def test_ppgf_rejects_invalid_aligned_clip(aligned_clip):
    with pytest.raises(ValueError, match="aligned_clip must be in"):
        merge_gradients_by_priority(
            [torch.ones(2), torch.ones(2)],
            order=[0, 1],
            aligned_clip=aligned_clip,
        )


def test_ppgf_is_priority_order_sensitive_and_preserves_last_gradient():
    gradients = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([1.0, -2.0]),
    ]

    merged_012 = merge_gradients_by_priority(gradients, order=[0, 1, 2])
    merged_021 = merge_gradients_by_priority(gradients, order=[0, 2, 1])

    assert not torch.allclose(merged_012, merged_021)
    assert torch.dot(merged_012, gradients[2]) >= gradients[2].square().sum() - 1e-6
    assert torch.dot(merged_021, gradients[1]) >= gradients[1].square().sum() - 1e-6


def test_ppgf_projects_grouped_conflicts_only_in_accumulated_overlap():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[-2.0, 0.0, -2.0, 0.0]]),
    ]
    masks = [torch.tensor([True, True]), torch.tensor([True, False])]

    merged = merge_gradients_by_priority(
        gradients,
        order=[0, 1],
        group_size=2,
        group_masks=masks,
    )

    assert torch.allclose(merged, torch.tensor([[-2.0, 0.0, -1.0, 0.0]]))


@pytest.mark.parametrize("order_mode", ["random", "gumbel"])
def test_ppgf_sampler_reuses_one_order_within_timestep_and_resamples_next_step(order_mode):
    recorded_orders = []

    def denoiser(x_t, sigma_t, ppgf_order=None):
        recorded_orders.append(ppgf_order.clone())
        return x_t

    sampler = PPGFSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        ppgf_num_tasks=5,
        ppgf_seed=0,
        ppgf_order_mode=order_mode,
        ppgf_uncertainty=torch.ones(5) if order_mode == "gumbel" else None,
    )
    x_t = torch.zeros(1, 2)
    sampler.step(x_t, torch.tensor(1.0), torch.tensor(0.5))
    sampler.step(x_t, torch.tensor(0.5), torch.tensor(0.25))

    assert len(recorded_orders) == 6
    assert all(torch.equal(recorded_orders[0], order) for order in recorded_orders[:3])
    assert all(torch.equal(recorded_orders[3], order) for order in recorded_orders[3:])
    assert not torch.equal(recorded_orders[0], recorded_orders[3])
    assert recorded_orders[0].shape == (5,)
    assert sorted(recorded_orders[0].tolist()) == list(range(5))


def test_ppgf_sampler_accepts_one_explicit_order_for_multiple_particles(monkeypatch):
    recorded_orders = []

    def denoiser(x_t, sigma_t, ppgf_order=None):
        recorded_orders.append(ppgf_order.clone())
        return x_t

    sampler = PPGFSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=1,
        ppgf_num_tasks=3,
        ppgf_seed=0,
        ppgf_order_mode="random",
    )

    def unexpected_sample(*args, **kwargs):
        raise AssertionError("an explicit timestep order must not be resampled")

    monkeypatch.setattr(sampler, "_sample_ppgf_order", unexpected_sample)
    order = torch.tensor([2, 0, 1])
    for particle_seed in (11, 29):
        sampler.step(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            torch.tensor(0.5),
            generator=torch.Generator().manual_seed(particle_seed),
            ppgf_order=order,
        )

    assert len(recorded_orders) == 4
    assert all(torch.equal(received, order) for received in recorded_orders)


def test_ppgf_sampler_explicit_generator_preserves_particle_rng_stream():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: torch.zeros_like(x_t),
        schedule=_IdentitySchedule(),
        corrections=2,
        ppgf_num_tasks=3,
        ppgf_seed=0,
        ppgf_order_mode="random",
    )
    state = torch.ones(1, 6)
    order = torch.tensor([0, 1, 2])
    t = torch.tensor(0.8)
    s = torch.tensor(0.4)

    torch.manual_seed(1234)
    baseline = sampler.step(state.clone(), t, s, ppgf_order=order)
    particle = sampler.step(
        state.clone(),
        t,
        s,
        generator=torch.Generator().manual_seed(1234),
        ppgf_order=order,
    )

    assert torch.equal(particle, baseline)


def test_ppgf_gumbel_sampler_requires_instrument_uncertainties():
    with pytest.raises(ValueError, match="Gumbel PPGF requires"):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=_IdentitySchedule(),
            ppgf_num_tasks=3,
            ppgf_order_mode="gumbel",
        )


def test_ppgf_fixed_sampler_requires_instrument_uncertainties():
    with pytest.raises(ValueError, match="Fixed PPGF requires"):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=_IdentitySchedule(),
            ppgf_num_tasks=3,
            ppgf_order_mode="fixed",
        )


def test_ppgf_fixed_order_is_stable_reused_and_does_not_consume_rng():
    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=5,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.tensor([1.0, 4.0, 4.0, 0.5, 2.0]),
        ppgf_uncertainty_strength=0.0,
    )

    first = sampler._sample_ppgf_order(torch.device("cpu"))
    second = sampler._sample_ppgf_order(torch.device("cpu"))

    assert torch.equal(first, torch.tensor([1, 2, 4, 0, 3]))
    assert torch.equal(second, first)
    assert sampler._ppgf_generator is None
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_ppgf_linear_steps_strength_is_exactly_0_to_10_over_16_steps():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        steps=64,
        ppgf_num_tasks=2,
        ppgf_order_mode="gumbel",
        ppgf_uncertainty=torch.ones(2),
        ppgf_uncertainty_strength=10.0,
        ppgf_uncertainty_strength_mode="linear_steps",
        ppgf_uncertainty_strength_start=0.0,
        ppgf_uncertainty_strength_end=10.0,
        ppgf_uncertainty_strength_ramp_steps=16,
    )

    strengths = [
        sampler._ppgf_uncertainty_strength_at_step(diffusion_step=step) for step in range(64)
    ]

    assert strengths[0] == 0.0
    assert strengths[1] == pytest.approx(10.0 / 15.0)
    assert strengths[15] == 10.0
    assert strengths[16:] == [10.0] * 48
    assert strengths[:16] == sorted(strengths[:16])


def test_ppgf_linear_steps_strength_advances_once_per_sampler_timestep(monkeypatch):
    sampled_strengths = []

    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        steps=3,
        corrections=2,
        ppgf_num_tasks=2,
        ppgf_order_mode="gumbel",
        ppgf_uncertainty=torch.ones(2),
        ppgf_uncertainty_strength_mode="linear_steps",
        ppgf_uncertainty_strength_start=0.0,
        ppgf_uncertainty_strength_end=10.0,
        ppgf_uncertainty_strength_ramp_steps=3,
        order_log=False,
    )

    def sample_order(device, timestep=None, merge_weight=None, uncertainty_strength=None):
        sampled_strengths.append(uncertainty_strength)
        return torch.tensor([0, 1], device=device)

    monkeypatch.setattr(sampler, "_sample_ppgf_order", sample_order)
    state = torch.zeros(1, 2)
    state = sampler.step(state, torch.tensor(1.0), torch.tensor(0.9))
    state = sampler.step(state, torch.tensor(0.9), torch.tensor(0.8))
    sampler.step(state, torch.tensor(0.8), torch.tensor(0.7))

    assert sampled_strengths == pytest.approx([0.0, 5.0, 10.0])
    assert sampler._ppgf_anneal_step == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ppgf_uncertainty_strength_mode": "quadratic"}, "Unknown PPGF"),
        ({"ppgf_uncertainty_strength_start": -1.0}, "endpoints"),
        ({"ppgf_uncertainty_strength_end": float("inf")}, "endpoints"),
        ({"ppgf_uncertainty_strength_ramp_steps": 0}, "ramp steps"),
        ({"ppgf_uncertainty_strength_ramp_steps": 65}, "ramp steps"),
    ],
)
def test_ppgf_linear_steps_strength_rejects_invalid_configuration(overrides, message):
    kwargs = {
        "ppgf_uncertainty_strength_mode": "linear_steps",
        "ppgf_uncertainty_strength_start": 0.0,
        "ppgf_uncertainty_strength_end": 10.0,
        "ppgf_uncertainty_strength_ramp_steps": 16,
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **denoiser_kwargs: x_t,
            schedule=_IdentitySchedule(),
            steps=64,
            ppgf_num_tasks=2,
            ppgf_order_mode="gumbel",
            ppgf_uncertainty=torch.ones(2),
            **kwargs,
        )


def test_ppgf_linear_steps_strength_requires_gumbel_ordering():
    with pytest.raises(ValueError, match="requires gumbel ordering"):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=_IdentitySchedule(),
            steps=64,
            ppgf_num_tasks=2,
            ppgf_order_mode="random",
            ppgf_uncertainty_strength_mode="linear_steps",
        )


def test_ppgf_cosine_anneal_uses_log_sigma_and_exact_endpoints():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=2,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.ones(2),
        ppgf_anneal_mode="cosine",
        ppgf_anneal_sigma_high=3.0,
        ppgf_anneal_sigma_low=1.0,
    )

    assert sampler._ppgf_merge_weight(torch.tensor(3.0)) == 1.0
    assert sampler._ppgf_merge_weight(torch.tensor(1.0)) == 0.0
    assert sampler._ppgf_merge_weight(torch.tensor(0.0)) == 0.0
    assert sampler._ppgf_merge_weight(torch.tensor(3.0**0.5)) == pytest.approx(0.5)


def test_ppgf_cosine_steps_anneal_is_exactly_16_32_16():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        steps=64,
        ppgf_num_tasks=2,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.ones(2),
        ppgf_anneal_mode="cosine_steps",
        ppgf_anneal_full_steps=16,
        ppgf_anneal_blend_steps=32,
        ppgf_anneal_sum_steps=16,
    )

    weights = [
        sampler._ppgf_merge_weight(torch.tensor(float("nan")), diffusion_step=step)
        for step in range(64)
    ]

    assert sum(weight == 1.0 for weight in weights) == 16
    assert sum(0.0 < weight < 1.0 for weight in weights) == 32
    assert sum(weight == 0.0 for weight in weights) == 16
    assert weights[15] == 1.0
    assert 0.0 < weights[16] < 1.0
    assert 0.0 < weights[47] < 1.0
    assert weights[48] == 0.0
    assert weights[16:48] == sorted(weights[16:48], reverse=True)


def test_ppgf_cosine_steps_advances_once_per_sampler_timestep():
    recorded_weights = []

    def denoiser(x_t, sigma_t, ppgf_order=None, ppgf_merge_weight=None):
        recorded_weights.append(ppgf_merge_weight)
        return x_t

    sampler = PPGFSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        steps=3,
        corrections=2,
        ppgf_num_tasks=2,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.ones(2),
        ppgf_anneal_mode="cosine_steps",
        ppgf_anneal_full_steps=1,
        ppgf_anneal_blend_steps=1,
        ppgf_anneal_sum_steps=1,
        order_log=False,
    )

    state = torch.zeros(1, 2)
    state = sampler.step(state, torch.tensor(1.0), torch.tensor(0.9))
    state = sampler.step(state, torch.tensor(0.9), torch.tensor(0.8))
    sampler.step(state, torch.tensor(0.8), torch.tensor(0.7))

    assert recorded_weights == pytest.approx([1.0] * 3 + [0.5] * 3 + [0.0] * 3)
    assert sampler._ppgf_anneal_step == 3


@pytest.mark.parametrize(
    ("full_steps", "blend_steps", "sum_steps", "message"),
    [
        (16, 32, 15, "sum to sampler steps"),
        (16, -1, 49, "non-negative"),
    ],
)
def test_ppgf_cosine_steps_rejects_invalid_counts(
    full_steps,
    blend_steps,
    sum_steps,
    message,
):
    with pytest.raises(ValueError, match=message):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=_IdentitySchedule(),
            steps=64,
            ppgf_num_tasks=2,
            ppgf_order_mode="fixed",
            ppgf_uncertainty=torch.ones(2),
            ppgf_anneal_mode="cosine_steps",
            ppgf_anneal_full_steps=full_steps,
            ppgf_anneal_blend_steps=blend_steps,
            ppgf_anneal_sum_steps=sum_steps,
        )


@pytest.mark.parametrize(
    ("sigma_low", "sigma_high"),
    [(0.0, 3.0), (1.0, 1.0), (3.0, 1.0), (1.0, float("inf"))],
)
def test_ppgf_cosine_anneal_rejects_invalid_sigma_thresholds(sigma_low, sigma_high):
    with pytest.raises(ValueError, match="sigma thresholds"):
        PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=_IdentitySchedule(),
            ppgf_num_tasks=2,
            ppgf_order_mode="fixed",
            ppgf_uncertainty=torch.ones(2),
            ppgf_anneal_mode="cosine",
            ppgf_anneal_sigma_high=sigma_high,
            ppgf_anneal_sigma_low=sigma_low,
        )


def test_ppgf_sampler_forwards_annealed_weight_to_all_denoiser_calls():
    recorded_weights = []

    def denoiser(x_t, sigma_t, ppgf_order=None, ppgf_merge_weight=None):
        recorded_weights.append(ppgf_merge_weight)
        return x_t

    sampler = PPGFSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        ppgf_num_tasks=2,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.ones(2),
        ppgf_anneal_mode="cosine",
        ppgf_anneal_sigma_high=3.0,
        ppgf_anneal_sigma_low=1.0,
    )

    sampler.step(
        torch.zeros(1, 2),
        torch.tensor(3.0**0.5),
        torch.tensor(1.0),
    )

    assert recorded_weights == pytest.approx([0.5, 0.5, 0.5])


def test_channel_ppgf_fixed_order_is_independent_per_channel():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=3,
        ppgf_num_channels=2,
        ppgf_channelwise=True,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.tensor([
            [3.0, 3.0, 1.0],
            [0.5, 4.0, 2.0],
        ]),
    )

    order = sampler._sample_ppgf_order(torch.device("cpu"))

    assert torch.equal(order, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_ppgf_random_order_uses_uniform_randperm_without_uncertainties():
    seed = 17
    expected_generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = torch.randperm(5, generator=expected_generator)
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=5,
        ppgf_seed=seed,
        ppgf_order_mode="random",
    )

    order = sampler._sample_ppgf_order(torch.device("cpu"))

    assert torch.equal(order, expected)
    assert sampler.ppgf_uncertainty is None


def test_ppgf_sampler_broadcasts_one_vector_to_nonzero_rank(monkeypatch):
    expected = torch.tensor([2, 0, 1])
    monkeypatch.setattr("appa.sampling.dist.is_available", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.is_initialized", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.get_rank", lambda: 1)

    def broadcast(order, src):
        assert src == 0
        order.copy_(expected)

    monkeypatch.setattr("appa.sampling.dist.broadcast", broadcast)
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=3,
        ppgf_seed=0,
        ppgf_uncertainty=torch.ones(3),
    )

    order = sampler._sample_ppgf_order(torch.device("cpu"))
    assert torch.equal(order, expected)


def test_ppgf_gumbel_order_puts_reliable_instruments_later():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=3,
        ppgf_seed=0,
        ppgf_order_mode="gumbel",
        ppgf_uncertainty=torch.tensor([4.0, 1.0, 0.25]),
        ppgf_uncertainty_strength=2.0,
    )

    position_sum = torch.zeros(3)
    for _ in range(1000):
        order = sampler._sample_ppgf_order(torch.device("cpu"))
        position_sum += torch.argsort(order).float()

    mean_position = position_sum / 1000
    assert mean_position[0] < mean_position[1] < mean_position[2]


def test_multi_instrument_ppgf_filters_order_for_missing_instrument():
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[
            [
                torch.tensor([0]),
                torch.empty(0, dtype=torch.long),
                torch.tensor([1]),
            ]
        ],
        merge="ppgf",
        pcgrad_group_size=1,
        tweedie_covariance=False,
        iterations=1,
    )

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=torch.tensor([1, 2, 0]),
        )

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_spatial_vector_ppgf_uses_one_vector_order_per_spatial_group():
    gradients = [
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[-1.0, 1.0, -1.0, 1.0]]),
    ]
    orders = torch.tensor([
        [0, 1],
        [1, 0],
    ])

    merged = merge_gradients_by_group_priority(
        gradients,
        orders=orders,
        group_size=2,
    )

    torch.testing.assert_close(
        merged,
        torch.tensor([[-0.5, 1.5, 1.0, 1.0]]),
    )


def test_multi_instrument_spatial_vector_ppgf_filters_missing_sources(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["orders"] = kwargs["orders"].clone()
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr("appa.diffusion.merge_gradients_by_group_priority", merge)
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[
            torch.tensor([0]),
            torch.empty(0, dtype=torch.long),
            torch.tensor([1]),
        ]],
        merge="spatial_vector_ppgf",
        pcgrad_group_size=1,
        ppgf_order_group_count=2,
        tweedie_covariance=False,
        iterations=1,
    )
    global_orders = torch.tensor([[1, 2, 0], [0, 1, 2]])

    with torch.no_grad():
        denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=global_orders,
        )

    assert torch.equal(captured["orders"], torch.tensor([[1, 0], [0, 1]]))


def test_multi_instrument_ppgf_forwards_aligned_clip(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["aligned_clip"] = kwargs["aligned_clip"]
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr("appa.diffusion.merge_gradients_by_priority", merge)
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[torch.tensor([0]), torch.tensor([1])]],
        merge="ppgf",
        pcgrad_group_size=1,
        ppgf_aligned_clip=0.75,
        tweedie_covariance=False,
        iterations=1,
    )

    with torch.no_grad():
        denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=torch.tensor([0, 1]),
        )

    assert captured["aligned_clip"] == 0.75


def test_trajectory_denoiser_forwards_ppgf_order_to_every_blanket():
    class RecordingDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x_t, sigma_t, blanket_id, ppgf_order=None, **kwargs):
            self.calls.append((blanket_id, ppgf_order))
            return x_t

    recorder = RecordingDenoiser()
    denoiser = TrajectoryDenoiser(
        recorder,
        blanket_size=2,
        blanket_stride=1,
        state_size=1,
        pass_blanket_ids=True,
    )
    order = torch.tensor([2, 0, 1])

    output = denoiser(
        torch.zeros(1, 3),
        torch.tensor(1.0),
        date=torch.zeros(1, 3, 4),
        ppgf_order=order,
    )

    assert output.shape == (1, 3)
    assert [blanket_id for blanket_id, _ in recorder.calls] == [0, 1]
    assert all(received is order for _, received in recorder.calls)


def test_trajectory_denoiser_forwards_ppgf_merge_weight_to_every_blanket():
    class RecordingDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            x_t,
            sigma_t,
            blanket_id,
            ppgf_merge_weight=None,
            **kwargs,
        ):
            self.calls.append((blanket_id, ppgf_merge_weight))
            return x_t

    recorder = RecordingDenoiser()
    denoiser = TrajectoryDenoiser(
        recorder,
        blanket_size=2,
        blanket_stride=1,
        state_size=1,
        pass_blanket_ids=True,
    )

    output = denoiser(
        torch.zeros(1, 3),
        torch.tensor(1.0),
        date=torch.zeros(1, 3, 4),
        ppgf_merge_weight=0.25,
    )

    assert output.shape == (1, 3)
    assert recorder.calls == [(0, 0.25), (1, 0.25)]


def test_channel_ppgf_uses_a_different_source_order_per_channel():
    gradients = [
        torch.tensor([[1.0, -1.0]]),
        torch.tensor([[-2.0, 3.0]]),
    ]
    orders = torch.tensor([
        [0, 1],
        [1, 0],
    ])

    merged = merge_gradients_by_channel_priority(
        gradients,
        orders=orders,
        group_size=2,
    )

    assert torch.equal(merged, torch.tensor([[-2.0, -1.0]]))


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (2.0, -3.0, -3.0),
        (-2.0, 3.0, 3.0),
    ],
)
def test_channel_ppgf_replaces_opposite_signs_symmetrically(first, second, expected):
    merged = merge_gradients_by_channel_priority(
        [torch.tensor([first]), torch.tensor([second])],
        orders=torch.tensor([[0, 1]]),
        group_size=1,
    )

    assert torch.equal(merged, torch.tensor([expected]))


def test_channel_ppgf_conflict_weight_interpolates_to_exact_sum():
    gradients = [
        torch.tensor([2.0]),
        torch.tensor([-3.0]),
    ]
    orders = torch.tensor([[0, 1]])
    summed = merge_gradients_by_channel_priority(
        gradients, orders=orders, group_size=1, conflict_weight=0.0
    )
    halfway = merge_gradients_by_channel_priority(
        gradients, orders=orders, group_size=1, conflict_weight=0.5
    )

    assert torch.equal(summed, torch.tensor([-1.0]))
    assert torch.equal(halfway, torch.tensor([-2.0]))


def test_vector_ppgf_conflict_weight_interpolates_to_exact_sum():
    gradients = [
        torch.tensor([2.0]),
        torch.tensor([-3.0]),
    ]
    summed = merge_gradients_by_priority(
        gradients, order=[0, 1], group_size=1, conflict_weight=0.0
    )
    halfway = merge_gradients_by_priority(
        gradients, order=[0, 1], group_size=1, conflict_weight=0.5
    )

    assert torch.equal(summed, torch.tensor([-1.0]))
    assert torch.equal(halfway, torch.tensor([-2.0]))


def test_channel_ppgf_adds_same_sign_and_zero_values():
    gradients = [
        torch.tensor([[2.0, -2.0, 0.0]]),
        torch.tensor([[3.0, -4.0, 5.0]]),
    ]

    merged = merge_gradients_by_channel_priority(
        gradients,
        orders=torch.tensor([[0, 1], [0, 1], [0, 1]]),
        group_size=3,
    )

    assert torch.equal(merged, torch.tensor([[5.0, -6.0, 5.0]]))


def test_channel_ppgf_accumulate_uses_most_reliable_source_sign():
    gradients = [
        torch.tensor([1.0]),
        torch.tensor([-2.0]),
        torch.tensor([3.0]),
    ]
    orders = torch.tensor([[0, 1, 2]])

    accumulated = merge_gradients_by_channel_priority_accumulate(
        gradients,
        orders=orders,
        group_size=1,
    )
    original = merge_gradients_by_channel_priority(
        gradients,
        orders=orders,
        group_size=1,
    )

    assert torch.equal(accumulated, torch.tensor([4.0]))
    assert torch.equal(original, torch.tensor([3.0]))


def test_channel_ppgf_accumulate_returns_zero_for_zero_reference():
    merged = merge_gradients_by_channel_priority_accumulate(
        [torch.tensor([2.0]), torch.tensor([-3.0]), torch.tensor([0.0])],
        orders=torch.tensor([[0, 1, 2]]),
        group_size=1,
    )

    assert torch.equal(merged, torch.tensor([0.0]))


def test_channel_ppgf_accumulate_sums_negative_reference_sign():
    merged = merge_gradients_by_channel_priority_accumulate(
        [torch.tensor([-1.0]), torch.tensor([2.0]), torch.tensor([-3.0])],
        orders=torch.tensor([[0, 1, 2]]),
        group_size=1,
    )

    assert torch.equal(merged, torch.tensor([-4.0]))


def test_channel_ppgf_accumulate_uses_independent_spatial_channel_references():
    merged = merge_gradients_by_channel_priority_accumulate(
        [
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[-2.0, -2.0]]),
            torch.tensor([[3.0, 3.0]]),
        ],
        orders=torch.tensor([
            [0, 1, 2],
            [2, 0, 1],
        ]),
        group_size=2,
    )

    assert torch.equal(merged, torch.tensor([[4.0, -2.0]]))


def test_channel_ppgf_top1_uses_independent_spatial_channel_sources():
    merged = merge_gradients_by_channel_priority_top1(
        [
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[10.0, 20.0, 30.0]]),
            torch.tensor([[100.0, 200.0, 300.0]]),
        ],
        orders=torch.tensor([
            [0, 1, 2],
            [2, 0, 1],
            [1, 2, 0],
        ]),
        group_size=3,
    )

    assert torch.equal(merged, torch.tensor([[100.0, 20.0, 3.0]]))


def test_channel_ppgf_top1_conflict_weight_interpolates_to_exact_sum():
    gradients = [torch.tensor([1.0]), torch.tensor([3.0])]
    orders = torch.tensor([[0, 1]])

    summed = merge_gradients_by_channel_priority_top1(
        gradients,
        orders=orders,
        group_size=1,
        conflict_weight=0.0,
    )
    halfway = merge_gradients_by_channel_priority_top1(
        gradients,
        orders=orders,
        group_size=1,
        conflict_weight=0.5,
    )

    assert torch.equal(summed, torch.tensor([4.0]))
    assert torch.equal(halfway, torch.tensor([3.5]))


def test_channel_ppgf_replaces_only_in_observation_overlap():
    gradients = [
        torch.tensor([[1.0, -1.0, 1.0, -1.0]]),
        torch.tensor([[-2.0, 2.0, -2.0, 2.0]]),
    ]
    masks = [torch.tensor([True, True]), torch.tensor([True, False])]

    merged = merge_gradients_by_channel_priority(
        gradients,
        orders=torch.tensor([[0, 1], [0, 1]]),
        group_size=2,
        group_masks=masks,
    )

    assert torch.equal(merged, torch.tensor([[-2.0, 2.0, -1.0, 1.0]]))


def test_multi_instrument_channel_ppgf_filters_missing_sources(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["orders"] = kwargs["orders"].clone()
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr(
        "appa.diffusion.merge_gradients_by_channel_priority",
        merge,
    )
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[
            [
                torch.tensor([0]),
                torch.empty(0, dtype=torch.long),
                torch.tensor([1]),
            ]
        ],
        merge="channel_ppgf",
        pcgrad_group_size=2,
        tweedie_covariance=False,
        iterations=1,
    )
    global_orders = torch.tensor([
        [1, 2, 0],
        [0, 1, 2],
    ])

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=global_orders,
        )

    assert torch.isfinite(output).all()
    assert torch.equal(captured["orders"], torch.tensor([[1, 0], [0, 1]]))


def test_multi_instrument_dispatches_channel_ppgf_accumulate(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["orders"] = kwargs["orders"].clone()
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr(
        "appa.diffusion.merge_gradients_by_channel_priority_accumulate",
        merge,
    )
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[[torch.tensor([0]), torch.tensor([1])]],
        merge="channel_ppgf_accumulate",
        pcgrad_group_size=2,
        ppgf_overlap_only=False,
        tweedie_covariance=False,
        iterations=1,
    )
    global_orders = torch.tensor([
        [0, 1],
        [1, 0],
    ])

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=global_orders,
        )

    assert torch.isfinite(output).all()
    assert torch.equal(captured["orders"], global_orders)


def test_multi_instrument_top1_filters_missing_sources(monkeypatch):
    captured = {}

    def merge(gradients, **kwargs):
        captured["orders"] = kwargs["orders"].clone()
        return torch.stack(gradients).sum(0)

    monkeypatch.setattr(
        "appa.diffusion.merge_gradients_by_channel_priority_top1",
        merge,
    )
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([1.0, -1.0])],
        var_y=[torch.ones(2)],
        instrument_indices=[
            [
                torch.tensor([0]),
                torch.empty(0, dtype=torch.long),
                torch.tensor([1]),
            ]
        ],
        merge="channel_ppgf_top1",
        pcgrad_group_size=2,
        ppgf_overlap_only=False,
        tweedie_covariance=False,
        iterations=1,
    )
    global_orders = torch.tensor([
        [1, 2, 0],
        [0, 1, 2],
    ])

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=global_orders,
        )

    assert torch.isfinite(output).all()
    assert torch.equal(captured["orders"], torch.tensor([[1, 0], [0, 1]]))


def test_channel_ppgf_gumbel_orders_are_channelwise_and_uncertainty_weighted():
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=_IdentitySchedule(),
        ppgf_num_tasks=3,
        ppgf_num_channels=2,
        ppgf_channelwise=True,
        ppgf_seed=0,
        ppgf_order_mode="gumbel",
        ppgf_uncertainty=torch.tensor([
            [9.0, 1.0, 0.1],
            [0.1, 1.0, 9.0],
        ]),
        ppgf_uncertainty_strength=2.0,
    )
    position_sum = torch.zeros(2, 3)

    for _ in range(1000):
        orders = sampler._sample_ppgf_order(torch.device("cpu"))
        assert orders.shape == (2, 3)
        assert torch.equal(
            torch.sort(orders, dim=1).values,
            torch.arange(3).expand(2, -1),
        )
        position_sum += torch.argsort(orders, dim=1).float()

    mean_position = position_sum / 1000
    assert mean_position[0, 0] < mean_position[0, 1] < mean_position[0, 2]
    assert mean_position[1, 2] < mean_position[1, 1] < mean_position[1, 0]


def test_channel_ppgf_sampler_reuses_matrix_within_timestep():
    recorded_orders = []

    def denoiser(x_t, sigma_t, ppgf_order=None):
        recorded_orders.append(ppgf_order.clone())
        return x_t

    sampler = PPGFSampler(
        denoiser=denoiser,
        schedule=_IdentitySchedule(),
        corrections=2,
        ppgf_num_tasks=3,
        ppgf_num_channels=4,
        ppgf_channelwise=True,
        ppgf_seed=0,
        ppgf_order_mode="random",
    )

    sampler.step(
        torch.zeros(1, 4),
        torch.tensor(1.0),
        torch.tensor(0.5),
    )

    assert len(recorded_orders) == 3
    assert recorded_orders[0].shape == (4, 3)
    assert all(torch.equal(recorded_orders[0], order) for order in recorded_orders[1:])


def test_channel_ppgf_can_bypass_one_instrument_on_selected_scalars():
    operators = [
        lambda x, blanket_id=0: x.flatten(),
        lambda x, blanket_id=0: x.flatten(),
        lambda x, blanket_id=0: x.flatten(),
    ]
    denoiser = MultiInstrumentMMPSDenoiser(
        denoiser=_IdentityDenoiser(),
        A=lambda x, blanket_id=0: x.flatten(),
        y=[torch.tensor([3.0, 3.0, 1.5, 1.5, -4.5, -4.5])],
        var_y=[torch.ones(6)],
        instrument_indices=[[
            torch.tensor([0, 1]),
            torch.tensor([2, 3]),
            torch.tensor([4, 5]),
        ]],
        instrument_operators=operators,
        merge="channel_ppgf",
        pcgrad_group_size=2,
        ppgf_bypass_instrument_id=2,
        ppgf_bypass_mask=torch.tensor([False, True]),
        tweedie_covariance=False,
        iterations=1,
    )

    with torch.no_grad():
        output = denoiser(
            torch.zeros(1, 2),
            torch.tensor(1.0),
            blanket_id=0,
            ppgf_order=torch.tensor([
                [0, 1, 2],
                [0, 1, 2],
            ]),
        )

    assert torch.allclose(output, torch.tensor([[-3.0, 0.0]]), atol=1e-6)
