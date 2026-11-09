import pytest
import torch

from omegaconf import OmegaConf

from appa.math import merge_gradients_by_channel_priority
from appa.sampling import PPGFSampler
from experiments.observations.build_spatial_ordering_cache import (
    RAW_SPATIAL_KIND,
    SPATIAL_PPGF_MATRIX_KIND,
    build_spatial_ordering_payload,
)
from experiments.observations.build_random_spatial_ppgf_ordering_cache import (
    RANDOM_ORDERING_METRIC,
    build_random_spatial_ordering_payload,
)
from experiments.observations.evaluate_guided_six_frame import (
    configure_pcgrad_ordering_cache,
    configure_source_uncertainties,
    load_channel_ppgf_ordering_cache,
)


def _spatial_payload():
    variance = torch.arange(1, 25, dtype=torch.float32).reshape(2, 2, 2, 3) / 10
    counts = torch.tensor([
        [[2, 3], [4, 5]],
        [[0, 1], [2, 3]],
    ])
    variance[1, 0, 0] = torch.nan
    variance[1, 0, 1] = torch.nan
    return {
        "kind": RAW_SPATIAL_KIND,
        "period": "2025-01-01T00--2025-03-31T23",
        "time_scope": "hourly",
        "expected_hours": 2160,
        "mask_frac": 0.0,
        "sources": ("atms", "convi"),
        "latent_shape": (2, 2, 3),
        "variance": variance,
        "counts": counts,
    }


def test_spatial_ordering_fills_zero_and_single_count_positions(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)

    payload = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
        minimum_count=2,
        fill_value=1.0,
    )

    assert payload["kind"] == SPATIAL_PPGF_MATRIX_KIND
    assert torch.equal(payload["variance"][1, 0, 0], torch.ones(3))
    assert torch.equal(payload["variance"][1, 0, 1], torch.ones(3))
    torch.testing.assert_close(
        payload["variance"][0],
        raw_payload["variance"][0],
    )
    torch.testing.assert_close(
        payload["variance"][1, 1],
        raw_payload["variance"][1, 1],
    )
    assert payload["filled_position_count"] == 2
    assert payload["filled_scalar_count"] == 6
    assert payload["filled_position_count_by_source"] == {
        "atms": 0,
        "convi": 2,
    }
    assert payload["zero_count_position_count"] == 1
    assert payload["single_count_position_count"] == 1
    assert payload["zero_count_position_count_by_source"]["convi"] == 1
    assert payload["single_count_position_count_by_source"]["convi"] == 1
    assert payload["fill_value"] == 1.0


def test_spatial_loader_preserves_hwc_flatten_order_and_checks_shape(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)
    payload = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
    )
    cache = tmp_path / "ordering.pth"
    torch.save(payload, cache)

    channel, metadata = load_channel_ppgf_ordering_cache(
        cache,
        expected_sources=("atms", "convi"),
        expected_channels=3,
        expected_spatial_shape=(2, 2, 3),
    )

    expected = payload["variance"].reshape(2, -1).transpose(0, 1)
    assert torch.equal(channel, expected)
    assert metadata["ordering_granularity"] == "spatial_channel"
    assert metadata["spatial_shape"] == (2, 2, 3)
    assert metadata["order_size"] == 12
    assert metadata["fill_value"] == 1.0
    expected_spatial_vector = (
        payload["variance"]
        .clamp_min(1e-12)
        .log()
        .mean(dim=-1)
        .exp()
        .reshape(2, -1)
        .transpose(0, 1)
    )
    torch.testing.assert_close(
        metadata["spatial_vector"],
        expected_spatial_vector,
    )
    assert metadata["spatial_vector"].shape == (4, 2)
    assert metadata["spatial_vector_order_size"] == 4

    payload["variance"] = payload["variance"][..., :2]
    torch.save(payload, cache)
    with pytest.raises(RuntimeError, match="ordering shape mismatch"):
        load_channel_ppgf_ordering_cache(
            cache,
            expected_sources=("atms", "convi"),
            expected_channels=3,
            expected_spatial_shape=(2, 2, 3),
        )


def test_spatial_vector_cache_configures_fixed_pcgrad(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)
    payload = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
    )
    cache = tmp_path / "ordering.pth"
    torch.save(payload, cache)
    config = OmegaConf.create({
        "guidance_merge": "pcgrad",
        "pcgrad_order_mode": "fixed",
        "pcgrad_shuffle": True,
        "pcgrad_overlap_only": False,
        "pcgrad_ordering_cache": str(cache),
    })
    sources = [{"name": "atms"}, {"name": "convi"}]

    metadata = configure_pcgrad_ordering_cache(
        config,
        sources,
        latent_channels=3,
        spatial_shape=(2, 2, 3),
    )

    assert metadata["ordering_granularity"] == "spatial_channel"
    assert metadata["spatial_vector"].shape == (4, 2)
    expected = (
        payload["variance"]
        .clamp_min(1e-12)
        .log()
        .mean(dim=-1)
        .exp()
        .reshape(2, -1)
        .transpose(0, 1)
    )
    torch.testing.assert_close(metadata["spatial_vector"], expected)

    config.pcgrad_overlap_only = True
    with pytest.raises(ValueError, match="pcgrad_overlap_only=false"):
        configure_pcgrad_ordering_cache(
            config,
            sources,
            latent_channels=3,
            spatial_shape=(2, 2, 3),
        )


def test_spatial_ordering_supports_fixed_or_gumbel_full_latent_channel_ppgf(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)
    payload = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
    )
    cache = tmp_path / "ordering.pth"
    torch.save(payload, cache)
    config = OmegaConf.create({
        "variance_floor": 1e-8,
        "var_y_mode": "channelwise",
        "var_y_constant": 0.1,
        "ppgf_ordering_source": "matrix_cache",
        "ppgf_ordering_cache": str(cache),
        "guidance_merge": "channel_ppgf",
        "ppgf_order_mode": "gumbel",
        "ppgf_overlap_only": False,
    })
    sources = [
        {"name": "atms", "var_y": torch.ones(3)},
        {"name": "convi", "var_y": torch.ones(3)},
    ]

    metadata = configure_source_uncertainties(
        config,
        sources,
        route="encoder",
        latent_channels=3,
        effective_std=torch.ones(3),
        spatial_shape=(2, 2, 3),
    )

    assert metadata["ordering_granularity"] == "spatial_channel"
    assert metadata["channel"].shape == (12, 2)

    config.guidance_merge = "spatial_vector_ppgf"
    config.ppgf_order_mode = "fixed"
    metadata = configure_source_uncertainties(
        config,
        sources,
        route="encoder",
        latent_channels=3,
        effective_std=torch.ones(3),
        spatial_shape=(2, 2, 3),
    )
    expected_spatial_vector = (
        payload["variance"]
        .clamp_min(1e-12)
        .log()
        .mean(dim=-1)
        .exp()
        .reshape(2, -1)
        .transpose(0, 1)
    )
    torch.testing.assert_close(
        metadata["spatial_vector"],
        expected_spatial_vector,
    )

    config.guidance_merge = "channel_ppgf"
    config.ppgf_order_mode = "random"
    with pytest.raises(ValueError, match="ppgf_order_mode=fixed or gumbel"):
        configure_source_uncertainties(
            config,
            sources,
            route="encoder",
            latent_channels=3,
            effective_std=torch.ones(3),
            spatial_shape=(2, 2, 3),
        )

    config.ppgf_order_mode = "fixed"
    config.ppgf_overlap_only = True
    with pytest.raises(ValueError, match="requires ppgf_overlap_only=false"):
        configure_source_uncertainties(
            config,
            sources,
            route="encoder",
            latent_channels=3,
            effective_std=torch.ones(3),
            spatial_shape=(2, 2, 3),
        )


def test_spatial_gumbel_orders_are_per_latent_resampled_and_seed_reproducible():
    uncertainty = torch.tensor([[1.0, 2.0, 4.0]]).expand(12, -1).clone()

    def make_sampler():
        return PPGFSampler(
            denoiser=lambda x_t, sigma_t, **kwargs: x_t,
            schedule=lambda value: value,
            ppgf_num_tasks=3,
            ppgf_num_channels=12,
            ppgf_channelwise=True,
            ppgf_seed=17,
            ppgf_order_mode="gumbel",
            ppgf_uncertainty=uncertainty,
            ppgf_uncertainty_strength=0.5,
        )

    first_sampler = make_sampler()
    second_sampler = make_sampler()
    first_step = first_sampler._sample_ppgf_order(torch.device("cpu"))
    second_step = first_sampler._sample_ppgf_order(torch.device("cpu"))

    assert first_step.shape == (12, 3)
    assert torch.equal(
        torch.sort(first_step, dim=1).values,
        torch.arange(3).expand(12, -1),
    )
    assert torch.unique(first_step, dim=0).shape[0] > 1
    assert not torch.equal(first_step, second_step)
    assert torch.equal(
        first_step,
        second_sampler._sample_ppgf_order(torch.device("cpu")),
    )
    assert torch.equal(
        second_step,
        second_sampler._sample_ppgf_order(torch.device("cpu")),
    )


def test_spatial_latent_orders_are_reused_across_six_frames():
    latent_size = 4
    orders = torch.tensor([
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ])
    low_priority = torch.ones(6, latent_size)
    high_priority = -2.0 * torch.ones(6, latent_size)

    merged = merge_gradients_by_channel_priority(
        [low_priority, high_priority],
        orders=orders,
        group_size=latent_size,
    )

    expected_frame = torch.tensor([-2.0, 1.0, -2.0, 1.0])
    assert torch.equal(merged, expected_frame.expand(6, -1))


def test_fixed_ppgf_order_is_local_and_never_broadcast(monkeypatch):
    monkeypatch.setattr("appa.sampling.dist.is_available", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.is_initialized", lambda: True)
    monkeypatch.setattr("appa.sampling.dist.get_rank", lambda: 1)

    def unexpected_broadcast(*args, **kwargs):
        raise AssertionError("deterministic fixed order must not be broadcast")

    monkeypatch.setattr("appa.sampling.dist.broadcast", unexpected_broadcast)
    sampler = PPGFSampler(
        denoiser=lambda x_t, sigma_t, **kwargs: x_t,
        schedule=lambda value: value,
        ppgf_num_tasks=3,
        ppgf_order_mode="fixed",
        ppgf_uncertainty=torch.tensor([1.0, 3.0, 2.0]),
    )

    first = sampler._sample_ppgf_order(torch.device("cpu"))
    second = sampler._sample_ppgf_order(torch.device("cpu"))

    assert torch.equal(first, torch.tensor([1, 2, 0]))
    assert second.data_ptr() == first.data_ptr()


def test_random_spatial_ordering_is_fixed_per_seed_and_uniform_per_scalar(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)
    reference = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
    )
    reference_cache = tmp_path / "reference.pth"
    torch.save(reference, reference_cache)

    first = build_random_spatial_ordering_payload(
        reference,
        reference_cache_path=reference_cache,
        seed=17,
    )
    repeated = build_random_spatial_ordering_payload(
        reference,
        reference_cache_path=reference_cache,
        seed=17,
    )
    different = build_random_spatial_ordering_payload(
        reference,
        reference_cache_path=reference_cache,
        seed=18,
    )

    assert first["ordering_metric"] == RANDOM_ORDERING_METRIC
    assert first["variance"].shape == (2, 2, 2, 3)
    expected_ranks = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    assert torch.equal(
        torch.sort(first["variance"], dim=0).values,
        expected_ranks.expand_as(first["variance"]),
    )
    assert torch.equal(first["variance"], repeated["variance"])
    assert not torch.equal(first["variance"], different["variance"])


def test_random_spatial_vector_ordering_is_shared_across_channels(tmp_path):
    raw_payload = _spatial_payload()
    raw_cache = tmp_path / "raw.pth"
    torch.save(raw_payload, raw_cache)
    reference = build_spatial_ordering_payload(
        raw_payload,
        raw_cache_path=raw_cache,
    )
    reference_cache = tmp_path / "reference.pth"
    torch.save(reference, reference_cache)

    payload = build_random_spatial_ordering_payload(
        reference,
        reference_cache_path=reference_cache,
        seed=17,
        granularity="spatial_vector",
    )

    values = payload["variance"]
    orders = torch.argsort(values, dim=0, descending=True, stable=True)
    assert values.shape == (2, 2, 2, 3)
    assert payload["randomization_granularity"] == (
        "independent_per_H_lat_W_lat_shared_C_lat"
    )
    assert torch.equal(values[..., :1].expand_as(values), values)
    assert torch.equal(orders[..., :1].expand_as(orders), orders)

    cache = tmp_path / "spatial_vector_random.pth"
    torch.save(payload, cache)
    config = OmegaConf.create({
        "guidance_merge": "pcgrad",
        "pcgrad_order_mode": "fixed",
        "pcgrad_shuffle": True,
        "pcgrad_overlap_only": False,
        "pcgrad_ordering_cache": str(cache),
    })
    metadata = configure_pcgrad_ordering_cache(
        config,
        [{"name": "atms"}, {"name": "convi"}],
        latent_channels=3,
        spatial_shape=(2, 2, 3),
    )
    assert metadata["spatial_vector"].shape == (4, 2)
    assert metadata["random_seed"] == 17
