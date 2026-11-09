import numpy as np
import pytest
import torch

from datetime import datetime
from omegaconf import OmegaConf
from pathlib import Path
from types import SimpleNamespace

from experiments.observations import evaluate_guided_six_frame as evaluator
from experiments.observations.evaluate_guided_six_frame import (
    ANCHOR_FRAME_INDEX,
    CHANNEL_PPGF_MATRIX_KIND,
    ENSEMBLE_MEMBER_SEED_STRIDE,
    ENSEMBLE_REDUCTION,
    EXPECTED_SPLIT_ANCHORS,
    RESIDUAL_VAR_Y_KIND,
    SPATIAL_VAR_Y_KIND,
    _ensemble_anchor_payload,
    _source_contract,
    apply_var_y_mode,
    ensemble_sample_seed,
    enumerate_split_anchors,
    extract_anchor_frame,
    file_identity,
    finalize_weighted_rmse,
    load_channel_ppgf_ordering_cache,
    load_calibrated_var_y_cache,
    load_spatial_var_y_cache,
    partition_anchor_indices,
    ppgf_ordering_uncertainty,
    select_encoder_observation_variance,
    select_source_names,
    select_split_anchor_range,
    timestamp_key,
    window_times,
)






def test_ensemble_seed_schedule_keeps_member_zero_aligned():
    assert ensemble_sample_seed(0, 17, 0) == 17
    assert ensemble_sample_seed(0, 17, 1) == 17 + ENSEMBLE_MEMBER_SEED_STRIDE
    with pytest.raises(ValueError, match="non-negative"):
        ensemble_sample_seed(0, 17, -1)


def test_ensemble_anchor_payload_is_float32_member_mean():
    members = [
        {
            "latent": torch.tensor([[1.0, 3.0]], dtype=torch.float16),
            "sample_index": 0,
            "sample_seed": 7,
            "seed_anchor_index": 7,
        },
        {
            "latent": torch.tensor([[3.0, 7.0]], dtype=torch.float32),
            "sample_index": 1,
            "sample_seed": 1_000_010,
            "seed_anchor_index": 7,
        },
    ]

    payload = _ensemble_anchor_payload(members)

    assert payload["latent"].dtype == torch.float32
    assert torch.equal(payload["latent"], torch.tensor([[2.0, 5.0]]))
    assert payload["ensemble"] == {
        "num_members": 2,
        "reduction": ENSEMBLE_REDUCTION,
        "member_seeds": [7, 1_000_010],
    }
    assert "sample_index" not in payload
    assert "sample_seed" not in payload
    assert "seed_anchor_index" not in payload


def test_multi_sample_generation_saves_members_mean_and_resumes(tmp_path, monkeypatch):
    anchor = datetime(2025, 4, 1, 6)
    calls = []

    def fake_sample(
        config,
        runtime,
        selected_anchor,
        anchor_index,
        contract_hash,
        rank,
        *,
        sample_index,
        seed_anchor_index,
    ):
        calls.append((anchor_index, sample_index, seed_anchor_index))
        return {
            "kind": evaluator.ANCHOR_KIND,
            "contract_hash": contract_hash,
            "anchor_index": anchor_index,
            "anchor": evaluator.timestamp_tensor(selected_anchor),
            "window_timestamps": torch.stack([
                evaluator.timestamp_tensor(value)
                for value in evaluator.window_times(selected_anchor)
            ]),
            "frame_number": evaluator.ANCHOR_FRAME_NUMBER,
            "latent_coordinate": "diffusion_standardized",
            "latent": torch.full((2, 3), float(sample_index)),
            "sample_index": sample_index,
            "sample_seed": evaluator.ensemble_sample_seed(
                int(config.seed),
                seed_anchor_index,
                sample_index,
            ),
            "seed_anchor_index": seed_anchor_index,
            "source_value_counts": {},
            "guidance_diagnostics": None,
        }

    monkeypatch.setattr(evaluator, "sample_anchor", fake_sample)
    config = OmegaConf.create({
        "seed": 11,
        "num_samples_per_anchor": 3,
        "log_every": 1,
        "cuda_cleanup_interval": 100,
    })
    runtime = {"latent_shape": (2, 3)}

    for _ in range(2):
        evaluator.generate_local_anchors(
            config,
            runtime,
            [anchor],
            [7],
            [0],
            tmp_path,
            "contract",
            rank=0,
        )

    assert calls == [(0, 0, 7), (0, 1, 7), (0, 2, 7)]
    for sample_index in range(3):
        assert (
            tmp_path
            / "anchors_members"
            / "0000"
            / f"{sample_index:03d}.pth"
        ).is_file()
    ensemble = evaluator.safe_load(tmp_path / "anchors" / "0000.pth", map_location="cpu")
    assert torch.equal(ensemble["latent"], torch.ones(2, 3))
    assert ensemble["ensemble"]["num_members"] == 3


def test_window_is_hourly_and_frame_four_is_anchor():
    anchor = datetime(2025, 4, 1, 6)
    times = window_times(anchor)

    assert times == [
        datetime(2025, 4, 1, 3),
        datetime(2025, 4, 1, 4),
        datetime(2025, 4, 1, 5),
        anchor,
        datetime(2025, 4, 1, 7),
        datetime(2025, 4, 1, 8),
    ]
    assert times[ANCHOR_FRAME_INDEX] == anchor


def test_rank_partition_is_balanced_disjoint_and_complete():
    partitions = [partition_anchor_indices(731, rank, 8) for rank in range(8)]
    flattened = [index for partition in partitions for index in partition]

    assert sorted(flattened) == list(range(731))
    assert len(flattened) == len(set(flattened))
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 1


def test_extract_anchor_frame_returns_only_fourth_frame():
    window = torch.arange(6 * 2).reshape(6, 2)
    assert torch.equal(extract_anchor_frame(window), window[3])
    with pytest.raises(ValueError, match="6 frames"):
        extract_anchor_frame(window[:5])


def test_source_selection_supports_single_multi_and_no_obs():
    catalog = {
        "atms": {"encoder": {}, "decoder": {}},
        "convi": {"encoder": {}, "decoder": {}},
    }
    assert select_source_names(["ATMS"], "encoder", catalog) == ["atms"]
    assert select_source_names(["atms", "convi"], "decoder", catalog) == [
        "atms",
        "convi",
    ]
    assert select_source_names([], "no_obs", catalog) == []
    with pytest.raises(ValueError, match="duplicates"):
        select_source_names(["atms", "atms"], "encoder", catalog)


def test_full_observation_convi_source_contract_is_audited(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = model_dir / "config.yaml"
    checkpoint = model_dir / "model_best.pth"
    var_y_cache = model_dir / "var_y.pth"
    config.write_text("model: test\n")
    checkpoint.write_bytes(b"checkpoint")
    var_y_cache.write_bytes(b"variance")
    source = {
        "name": "convi",
        "kind": "convi_obs_encoder",
        "model_dir": model_dir,
        "checkpoint_path": checkpoint,
        "var_y_cache": var_y_cache,
        "var_y": torch.tensor([0.5, 0.7]),
        "var_y_mode": "channelwise",
        "var_y_constant": None,
        "dataset": SimpleNamespace(
            observation_data={
                "variant": "qc",
                "surf_dir": "/observations/surf_qc",
                "icoa_dir": "/observations/icoads_qc",
            },
        ),
    }

    contract = _source_contract(source)

    assert contract["convi_observation_data"] == {
        "variant": "qc",
        "surf_dir": "/observations/surf_qc",
        "icoa_dir": "/observations/icoads_qc",
    }
    assert contract["convi_observation_scope"] == "all_valid_qc_observations"


def test_full_observation_convi_dataset_does_not_require_manifest(monkeypatch):
    class FakeDataset:
        def __len__(self):
            return 1

    def fake_conviobs(**kwargs):
        assert str(kwargs["surf_dir"]).endswith("/surf_qc")
        assert str(kwargs["icoa_dir"]).endswith("/icoads_qc")
        return FakeDataset()

    monkeypatch.setattr(evaluator, "CONVIOBS", fake_conviobs)
    dataset = evaluator._build_convi_dataset(
        SimpleNamespace(
            surf_dir="training-surf",
            icoa_dir="training-icoa",
            mean_std_dir="training-mean-std",
        ),
        OmegaConf.create({
            "observation_variant": "qc",
            "surf_dir": "/observations/surf_qc",
            "icoa_dir": "/observations/icoads_qc",
            "mean_std_dir": "/observations/mean_std",
        }),
    )

    assert len(dataset) == 1
    assert dataset.observation_variant == "qc"


def test_surface_only_convi_dataset_never_opens_icoads(monkeypatch):
    class FakeDataset:
        def __len__(self):
            return 1

    def fake_conviobs(**kwargs):
        assert str(kwargs["surf_dir"]).endswith("/surf_qc")
        assert kwargs["icoa_dir"] is None
        return FakeDataset()

    monkeypatch.setattr(evaluator, "CONVIOBS", fake_conviobs)
    dataset = evaluator._build_convi_dataset(
        SimpleNamespace(kind="convi", icoa_dir=None),
        OmegaConf.create({
            "observation_variant": "qc",
            "surface_only": True,
            "surf_dir": "/observations/surf_qc",
            "icoa_dir": None,
            "mean_std_dir": "/observations/mean_std",
        }),
    )

    assert dataset.observation_data["scope"] == "surf_only"
    assert dataset.observation_data["icoa_dir"] is None


def test_dense_era_surface_encoder_uses_runtime_convi_qc_dataset(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "dense_encoder"
    latent_dir = tmp_path / "latents"
    model_dir.mkdir()
    latent_dir.mkdir()
    OmegaConf.save(
        OmegaConf.create({
            "era_latent_dir": str(latent_dir),
            "data": {"kind": "era_surface_dense"},
            "model": {},
        }),
        model_dir / "config.yaml",
    )
    (model_dir / "model_best.pth").touch()
    var_y_cache = model_dir / "convi_qc_var_y.pth"
    var_y_cache.touch()
    fake_dataset = SimpleNamespace()
    load_call = {}

    monkeypatch.setattr(
        evaluator,
        "_build_convi_dataset",
        lambda _data_config, _convi_config: fake_dataset,
    )
    monkeypatch.setattr(
        evaluator,
        "_configure_convi_era5_truth_values",
        lambda _dataset, _convi_config, _coordinates: None,
    )

    def fake_load_encoder(model_path, _ae, in_channels, checkpoint, _device):
        load_call.update({
            "model_dir": Path(model_path),
            "in_channels": in_channels,
            "checkpoint": checkpoint,
        })
        return object()

    monkeypatch.setattr(evaluator, "load_obs2era_encoder", fake_load_encoder)
    monkeypatch.setattr(
        evaluator,
        "load_calibrated_var_y_cache",
        lambda *_args, **_kwargs: (torch.ones(2), {}),
    )
    source = evaluator._load_encoder_source(
        "convi",
        OmegaConf.create({
            "model_dir": str(model_dir),
            "checkpoint": "best",
            "var_y_cache": str(var_y_cache),
            "reliability_cache": None,
            "calibration_period": "2025-01-01T00--2025-03-31T23",
        }),
        latent_dir,
        (4, 2),
        {
            "ae": SimpleNamespace(),
            "raw_std": torch.ones(2),
            "effective_std": torch.ones(2),
        },
        torch.device("cpu"),
        1e-8,
        convi_config=OmegaConf.create({}),
    )

    assert source["kind"] == "convi_obs_encoder"
    assert source["dataset"] is fake_dataset
    assert source["encoder_model_data_kind"] == "era_surface_dense"
    assert load_call == {
        "model_dir": model_dir,
        "in_channels": 5,
        "checkpoint": "best",
    }


def test_dense_era_truth_surface_input_sets_every_mask_value_to_one():
    from appa.data.datasets import CONVIOBS

    dataset = CONVIOBS.__new__(CONVIOBS)
    dataset.grid_shape = (2, 3)
    dataset.mean = np.zeros(
        (1, CONVIOBS.NUM_ERA_CHANNELS, 1, 1), dtype=np.float32
    )
    dataset.std = np.ones(
        (1, CONVIOBS.NUM_ERA_CHANNELS, 1, 1), dtype=np.float32
    )
    def fail_if_convi_is_read(_date):
        raise AssertionError("dense ERA full-mask mode must not read CONVI values")

    dataset._prepare_surface = fail_if_convi_is_read
    replacement = np.arange(24, dtype=np.float32).reshape(4, 2, 3)

    encoder_input, union_mask = dataset.prepare_surface_encoder_input(
        np.datetime64("2025-01-10T00"),
        normalized_surface_values=replacement,
        full_surface_mask=True,
    )

    assert torch.equal(encoder_input[:4], torch.from_numpy(replacement))
    assert torch.equal(encoder_input[4:], torch.ones((1, 2, 3)))
    assert torch.equal(union_mask, torch.ones((1, 2, 3)))


def test_calibration_cache_loader_validates_identity_and_converts_encoder_coordinate(tmp_path):
    checkpoint = tmp_path / "model_best.pth"
    torch.save({"weight": torch.tensor(1)}, checkpoint)
    cache = tmp_path / "obs2era_var_y.pth"
    payload = {
        "kind": RESIDUAL_VAR_Y_KIND,
        "representation": "encoder",
        "source": "atms",
        "window_start": "2025-01-01T00",
        "window_end": "2025-03-31T23",
        "window_end_inclusive": True,
        **evaluator.calibration_period_checks("2025-01-01T00--2025-03-31T23"),
        "num_requested_times": 2160,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "variance": torch.tensor([4.0, 9.0]),
    }
    torch.save(payload, cache)

    variance, loaded = load_calibrated_var_y_cache(
        cache,
        expected_representation="encoder",
        expected_source="atms",
        expected_channels=2,
        expected_checkpoint=checkpoint,
        coordinate_scale=torch.tensor([0.5, 2.0]),
    )

    assert loaded["kind"] == RESIDUAL_VAR_Y_KIND
    assert torch.equal(variance, torch.tensor([1.0, 36.0]))

    payload["representation"] = "decoder"
    torch.save(payload, cache)
    with pytest.raises(RuntimeError, match="representation mismatch"):
        load_calibrated_var_y_cache(
            cache,
            expected_representation="encoder",
            expected_source="atms",
            expected_channels=2,
        )


def test_calibration_cache_loader_accepts_only_matching_model_directory_relocation(tmp_path):
    model_dir = tmp_path / "new_root" / "obs2era_atms_1h"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "model_best.pth"
    torch.save({"weight": torch.tensor(1)}, checkpoint)
    cache = model_dir / "obs2era_var_y.pth"
    old_checkpoint = tmp_path / "old_root" / "obs2era_atms_1h" / "model_best.pth"
    payload = {
        "kind": RESIDUAL_VAR_Y_KIND,
        "representation": "encoder",
        "source": "atms",
        "window_start": "2025-01-01T00",
        "window_end": "2025-03-31T23",
        "window_end_inclusive": True,
        **evaluator.calibration_period_checks("2025-01-01T00--2025-03-31T23"),
        "num_requested_times": 2160,
        "checkpoint_path": str(old_checkpoint),
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "variance": torch.tensor([4.0, 9.0]),
    }
    torch.save(payload, cache)

    variance, _ = load_calibrated_var_y_cache(
        cache,
        expected_representation="encoder",
        expected_source="atms",
        expected_channels=2,
        expected_checkpoint=checkpoint,
    )
    assert torch.equal(variance, payload["variance"])

    payload["checkpoint_path"] = str(tmp_path / "old_root" / "obs2era_other_1h" / "model_best.pth")
    torch.save(payload, cache)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        load_calibrated_var_y_cache(
            cache,
            expected_representation="encoder",
            expected_source="atms",
            expected_channels=2,
            expected_checkpoint=checkpoint,
        )


def test_spatial_var_y_loader_selects_source_rows_and_converts_coordinates(
    tmp_path,
):
    sources = ("atms", "amsua", "mhs", "mwts", "mwhs", "convi")
    latent_shape = (2, 3, 4)
    checkpoint_paths = {}
    source_metadata = {}
    for source in sources:
        model_dir = tmp_path / f"obs2era_{source}_1h"
        model_dir.mkdir()
        checkpoint = model_dir / "model_best.pth"
        checkpoint.write_bytes(source.encode("utf-8"))
        checkpoint_paths[source] = checkpoint
        stat = checkpoint.stat()
        source_metadata[source] = {
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        }

    variance = (
        torch.arange(1, 7, dtype=torch.float32)
        .reshape(6, 1, 1, 1)
        .expand(6, *latent_shape)
        .clone()
    )
    payload = {
        "kind": SPATIAL_VAR_Y_KIND,
        "period": "2025-01-01T00--2025-03-31T23",
        "time_scope": "hourly",
        **evaluator.calibration_period_checks("2025-01-01T00--2025-03-31T23"),
        "num_requested_times": 2160,
        "mask_frac": 0.0,
        "ordering_ready": False,
        "sources": sources,
        "latent_shape": latent_shape,
        "layout": "(source, H_lat, W_lat, C_lat)",
        "variance": variance,
        "counts": torch.full((6, 2, 3), 2, dtype=torch.int64),
        "source_metadata": source_metadata,
    }
    cache = tmp_path / "spatial_raw.pth"
    torch.save(payload, cache)

    loaded, metadata = load_spatial_var_y_cache(
        cache,
        expected_sources=("mwhs", "atms"),
        expected_spatial_shape=latent_shape,
        expected_channels=4,
        expected_checkpoints={
            "mwhs": checkpoint_paths["mwhs"],
            "atms": checkpoint_paths["atms"],
        },
        coordinate_scale=torch.tensor([0.5, 1.0, 2.0, 3.0]),
    )

    assert metadata["kind"] == SPATIAL_VAR_Y_KIND
    assert loaded.shape == (2, 6, 4)
    expected_scale = torch.tensor([0.25, 1.0, 4.0, 9.0])
    torch.testing.assert_close(loaded[0], 5.0 * expected_scale.expand(6, 4))
    torch.testing.assert_close(loaded[1], expected_scale.expand(6, 4))

    payload["counts"][4, 0, 0] = 1
    torch.save(payload, cache)
    with pytest.raises(RuntimeError, match="at least two calibration samples"):
        load_spatial_var_y_cache(
            cache,
            expected_sources=("mwhs",),
            expected_spatial_shape=latent_shape,
            expected_channels=4,
            expected_checkpoints={"mwhs": checkpoint_paths["mwhs"]},
            coordinate_scale=torch.ones(4),
        )

    payload["counts"][4, 0, 1] = 0
    payload["variance"][4, 0, 1] = float("nan")
    torch.save(payload, cache)
    fallback = torch.tensor([[0.25, 0.5, 0.75, 1.0]])
    loaded, _ = load_spatial_var_y_cache(
        cache,
        expected_sources=("mwhs",),
        expected_spatial_shape=latent_shape,
        expected_channels=4,
        expected_checkpoints={"mwhs": checkpoint_paths["mwhs"]},
        coordinate_scale=torch.tensor([0.5, 1.0, 2.0, 3.0]),
        channelwise_fallback=fallback,
    )

    assert torch.equal(loaded[0, 0], fallback[0])
    assert torch.equal(loaded[0, 1], fallback[0])
    expected_supported = 5.0 * torch.tensor([0.25, 1.0, 4.0, 9.0])
    torch.testing.assert_close(loaded[0, 2], expected_supported)


def test_encoder_observation_variance_selects_matching_spatial_tokens():
    spatial = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    mask = torch.tensor([True, False, True, False])
    selected = select_encoder_observation_variance(
        {"var_y": spatial, "var_y_mode": "spatial"},
        mask,
        latent_pixels=4,
        latent_channels=3,
    )
    assert torch.equal(selected, torch.cat((spatial[0], spatial[2])))

    channelwise = select_encoder_observation_variance(
        {"var_y": torch.tensor([0.1, 0.2, 0.3]), "var_y_mode": "channelwise"},
        mask,
        latent_pixels=4,
        latent_channels=3,
    )
    assert torch.equal(
        channelwise,
        torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]),
    )


def test_convi_loss_spatial_top1_uses_fixed_stable_final_source(
    tmp_path, monkeypatch
):
    cache = tmp_path / "spatial_ordering.pth"
    cache.touch()
    uncertainty = torch.tensor([
        [2.0, 3.0, 4.0, 5.0, 6.0, 1.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 3.0, 4.0, 5.0, 6.0, 1.0],
        [2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
    ])
    metadata = {
        "path": cache,
        "kind": "spatial-ordering",
        "ordering_metric": "mse",
        "source_order": ["atms", "amsua", "mhs", "mwts", "mwhs", "convi"],
        "ordering_granularity": "spatial_channel",
        "order_size": 4,
        "channel": uncertainty,
    }

    def fake_load(path, **kwargs):
        assert Path(path) == cache
        assert kwargs["expected_channels"] == 4
        assert kwargs["expected_spatial_shape"] == (1, 1, 4)
        return uncertainty, metadata

    monkeypatch.setattr(
        evaluator,
        "load_channel_ppgf_ordering_cache",
        fake_load,
    )
    sources = [
        {
            "name": name,
            "kind": "convi_obs_encoder" if name == "convi" else "obs2era",
        }
        for name in ("atms", "amsua", "mhs", "mwts", "mwhs", "convi")
    ]
    config = OmegaConf.create({
        "convi_loss_top1_ordering_cache": str(cache),
        "guidance_merge": "sum",
        "var_y_mode": "spatial",
        "mask_frac": 0.0,
    })

    result = evaluator.configure_convi_loss_spatial_top1(
        config,
        sources,
        route="encoder",
        latent_channels=4,
        spatial_shape=(1, 1, 4),
    )

    assert torch.equal(
        sources[-1]["convi_loss_scalar_mask"],
        torch.tensor([[True, False, True, False]]),
    )
    assert result["selected_scalar_count"] == 2
    assert result["selected_fraction"] == 0.5


def test_convi_loss_scalar_filter_keeps_values_variance_and_operator_aligned():
    source = {
        "convi_loss_scalar_mask": torch.tensor([
            [True, False],
            [True, True],
            [False, True],
        ]),
        "var_y": torch.tensor([
            [10.0, 11.0],
            [12.0, 13.0],
            [14.0, 15.0],
        ]),
        "var_y_mode": "spatial",
    }
    observation = {
        "m_lat": torch.tensor([True, False, True]),
        "y_sel": torch.tensor([0.0, 1.0, 4.0, 5.0]),
    }

    scalar_filter = evaluator.apply_encoder_observation_scalar_filter(
        source,
        observation,
        latent_pixels=3,
        latent_channels=2,
    )
    variance = select_encoder_observation_variance(
        source,
        observation["m_lat"],
        latent_pixels=3,
        latent_channels=2,
    )[scalar_filter]
    prediction = evaluator.select_encoder_state_values(
        torch.arange(6, dtype=torch.float32).reshape(1, 3, 2),
        observation,
    )

    assert torch.equal(scalar_filter, torch.tensor([True, False, False, True]))
    assert torch.equal(observation["y_sel"], torch.tensor([0.0, 5.0]))
    assert torch.equal(variance, torch.tensor([10.0, 15.0]))
    assert torch.equal(prediction, torch.tensor([0.0, 5.0]))


def test_convi_gradient_mask_expands_only_convi_m_lat_over_channels():
    config = OmegaConf.create({
        "convi_gradient_latent_mask": True,
        "guidance_merge": "sum",
    })
    sources = [
        {"name": "atms", "kind": "obs2era"},
        {"name": "convi", "kind": "convi_obs_encoder"},
    ]
    source_masks = torch.tensor([
        [[True, True, False], [False, True, True]],
        [[True, False, False], [False, True, False]],
    ])

    masks = evaluator.build_convi_gradient_latent_masks(
        config,
        sources,
        source_masks,
        latent_channels=2,
    )

    assert masks[0] is None
    assert torch.equal(
        masks[1],
        torch.tensor([
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
        ]),
    )


def test_convi_gradient_mask_is_opt_in_and_rejects_joint_merge():
    sources = [
        {"name": "atms", "kind": "obs2era"},
        {"name": "convi", "kind": "convi_obs_encoder"},
    ]
    source_masks = torch.ones(2, 6, 3, dtype=torch.bool)
    disabled = OmegaConf.create({
        "convi_gradient_latent_mask": False,
        "guidance_merge": "joint",
    })
    assert (
        evaluator.build_convi_gradient_latent_masks(
            disabled,
            sources,
            source_masks,
            latent_channels=2,
        )
        is None
    )

    enabled = OmegaConf.create({
        "convi_gradient_latent_mask": True,
        "guidance_merge": "joint",
    })
    with pytest.raises(ValueError, match="independent-source"):
        evaluator.build_convi_gradient_latent_masks(
            enabled,
            sources,
            source_masks,
            latent_channels=2,
        )


def test_configure_spatial_var_y_supports_sum_and_spatial_channel_ppgf(
    tmp_path, monkeypatch
):
    cache = tmp_path / "spatial_raw.pth"
    cache.touch()
    checkpoints = {}
    channel_caches = {}
    sources = []
    for source_name in ("atms", "convi"):
        checkpoint = tmp_path / f"{source_name}_best.pth"
        checkpoint.touch()
        channel_cache = tmp_path / f"{source_name}_channel.pth"
        channel_cache.touch()
        checkpoints[source_name] = checkpoint
        channel_caches[source_name] = channel_cache
        sources.append({
            "name": source_name,
            "checkpoint_path": checkpoint,
            "var_y_cache": channel_cache,
            "var_y": torch.ones(3),
        })

    requested = {}

    def fake_load(path, **kwargs):
        requested["path"] = Path(path)
        requested.update(kwargs)
        return torch.stack((torch.ones(4, 3), torch.full((4, 3), 2.0))), {
            "kind": SPATIAL_VAR_Y_KIND,
            "sources": ("atms", "amsua", "mhs", "mwts", "mwhs", "convi"),
            "layout": "(source,H_lat,W_lat,C_lat)",
            "counts": torch.full((6, 2, 2), 9),
        }

    monkeypatch.setattr(evaluator, "load_spatial_var_y_cache", fake_load)
    config = OmegaConf.create({
        "variance_floor": 1e-8,
        "var_y_mode": "spatial",
        "var_y_constant": 0.1,
        "var_y_scale": 1.0,
        "spatial_var_y_cache": str(cache),
        "calibration_period": "2025-01-01T00--2025-03-31T23",
        "mask_frac": 0.0,
        "guidance_merge": "sum",
        "ppgf_ordering_source": "var_y",
        "ppgf_reliability_metric": "native_rmse",
    })

    result = evaluator.configure_source_uncertainties(
        config,
        sources,
        route="encoder",
        latent_channels=3,
        effective_std=torch.ones(3),
        spatial_shape=(2, 2, 3),
        coordinate_scale=torch.ones(3),
    )

    assert result is None
    assert requested["expected_sources"] == ["atms", "convi"]
    assert requested["expected_checkpoints"] == checkpoints
    assert torch.equal(requested["channelwise_fallback"], torch.ones(2, 3))
    assert torch.equal(sources[0]["var_y"], torch.ones(4, 3))
    assert torch.equal(sources[1]["var_y"], torch.full((4, 3), 2.0))
    assert sources[0]["channelwise_var_y_cache"] == channel_caches["atms"]
    assert sources[1]["var_y_cache"] == cache.resolve()
    assert sources[1]["spatial_var_y"]["raw_source_index"] == 5
    assert (
        sources[1]["spatial_var_y"]["channelwise_fallback_position_count"]
        == 0
    )

    ordering_cache = tmp_path / "spatial_ordering.pth"
    ordering_cache.touch()
    ordering_metadata = {
        "ordering_granularity": "spatial_channel",
        "spatial_shape": (2, 2, 3),
        "order_size": 12,
    }

    def fake_load_ordering(path, **kwargs):
        assert Path(path) == ordering_cache
        assert kwargs["expected_sources"] == ["atms", "convi"]
        assert kwargs["expected_spatial_shape"] == (2, 2, 3)
        return torch.ones(12, 2), ordering_metadata

    monkeypatch.setattr(
        evaluator,
        "load_channel_ppgf_ordering_cache",
        fake_load_ordering,
    )
    config.ppgf_ordering_source = "matrix_cache"
    config.ppgf_ordering_cache = str(ordering_cache)
    config.ppgf_order_mode = "fixed"
    config.ppgf_overlap_only = False
    for merge in (
        "channel_ppgf",
        "channel_ppgf_accumulate",
        "channel_ppgf_top1",
    ):
        config.guidance_merge = merge
        result = evaluator.configure_source_uncertainties(
            config,
            sources,
            route="encoder",
            latent_channels=3,
            effective_std=torch.ones(3),
            spatial_shape=(2, 2, 3),
            coordinate_scale=torch.ones(3),
        )
        assert result is ordering_metadata

    config.guidance_merge = "pcgrad"
    with pytest.raises(ValueError, match="requires guidance_merge=sum"):
        evaluator.configure_source_uncertainties(
            config,
            sources,
            route="encoder",
            latent_channels=3,
            effective_std=torch.ones(3),
            spatial_shape=(2, 2, 3),
            coordinate_scale=torch.ones(3),
        )


def test_var_y_mode_supports_channelwise_and_one_global_constant():
    calibrated = torch.tensor([0.2, 0.7])

    channelwise = apply_var_y_mode(
        calibrated,
        mode="channelwise",
        constant=0.1,
        variance_floor=1e-8,
    )
    constant = apply_var_y_mode(
        calibrated,
        mode="constant",
        constant=0.1,
        variance_floor=1e-8,
        scale=0.25,
    )

    assert torch.equal(channelwise, calibrated)
    assert torch.equal(constant, torch.tensor([0.025, 0.025]))
    assert torch.equal(
        apply_var_y_mode(
            calibrated,
            mode="channelwise",
            constant=None,
            variance_floor=1e-8,
            scale=0.25,
        ),
        torch.tensor([0.05, 0.175]),
    )
    with pytest.raises(ValueError, match="var_y_scale"):
        apply_var_y_mode(
            calibrated,
            mode="channelwise",
            constant=None,
            variance_floor=1e-8,
            scale=0.0,
        )
    with pytest.raises(ValueError, match="below variance_floor"):
        apply_var_y_mode(
            calibrated,
            mode="constant",
            constant=1e-9,
            variance_floor=1e-8,
        )




def test_channel_ppgf_matrix_cache_loader_validates_and_transposes(tmp_path):
    cache = tmp_path / "ordering.pth"
    payload = {
        "kind": CHANNEL_PPGF_MATRIX_KIND,
        "sources": ("atms", "mhs"),
        "mask_frac": 0.2,
        "variance": torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]),
    }
    torch.save(payload, cache)

    channel, metadata = load_channel_ppgf_ordering_cache(
        cache,
        expected_sources=("atms", "mhs"),
        expected_channels=3,
    )

    assert torch.equal(
        channel,
        torch.tensor([
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]),
    )
    assert metadata["cache_mask_frac"] == 0.2
    assert metadata["source_order"] == ["atms", "mhs"]
    assert metadata["vector_reduction"] == "geometric_mean_over_latent_channels"
    expected_vector = torch.tensor([
        (1.0 * 2.0 * 3.0) ** (1.0 / 3.0),
        (4.0 * 5.0 * 6.0) ** (1.0 / 3.0),
    ])
    assert torch.allclose(metadata["vector"], expected_vector)

    payload["variance"][0, 0] = 0.0
    torch.save(payload, cache)
    with pytest.raises(RuntimeError, match="invalid values"):
        load_channel_ppgf_ordering_cache(
            cache,
            expected_sources=("atms", "mhs"),
            expected_channels=3,
        )


def test_matrix_cache_ordering_supports_channel_and_vector_ppgf():
    channel = torch.tensor([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ])
    vector = torch.tensor([
        (1.0 * 2.0 * 3.0) ** (1.0 / 3.0),
        (4.0 * 5.0 * 6.0) ** (1.0 / 3.0),
    ])
    config = {"ppgf_ordering_source": "matrix_cache"}
    runtime = {
        "sources": [{"name": "atms"}, {"name": "mhs"}],
        "ppgf_ordering_cache": {"channel": channel, "vector": vector},
    }

    assert torch.equal(
        ppgf_ordering_uncertainty(config, runtime, channelwise=True),
        channel,
    )
    assert torch.equal(
        ppgf_ordering_uncertainty(config, runtime, channelwise=False),
        vector,
    )


def test_ppgf_reliability_ordering_shapes_match_vector_and_channel_modes():
    config = {"ppgf_ordering_source": "guidance_reliability"}
    runtime = {
        "latent_shape": (3, 2),
        "sources": [
            {
                "name": "atms",
                "ppgf_reliability": {
                    "channel": torch.tensor([1.0, 2.0]),
                    "vector": torch.tensor(2.0**0.5),
                },
            },
            {
                "name": "mhs",
                "ppgf_reliability": {
                    "channel": torch.tensor([3.0, 4.0]),
                    "vector": torch.tensor(12.0**0.5),
                },
            },
        ],
    }

    vector = ppgf_ordering_uncertainty(config, runtime, channelwise=False)
    channel = ppgf_ordering_uncertainty(config, runtime, channelwise=True)

    torch.testing.assert_close(
        vector,
        torch.tensor([2.0**0.5, 12.0**0.5]),
    )
    assert torch.equal(channel, torch.tensor([[1.0, 3.0], [2.0, 4.0]]))


def test_finalize_rmse_takes_square_root_after_global_sums():
    normalized, physical = finalize_weighted_rmse(
        squared_error_sum=torch.tensor([4.0, 18.0]),
        weight_sum=torch.tensor([1.0, 2.0]),
        physical_std=torch.tensor([10.0, 0.5]),
    )

    assert torch.allclose(normalized, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(physical, torch.tensor([20.0, 1.5], dtype=torch.float64))


def test_timestamp_key_requires_exact_hour_fields():
    assert timestamp_key(datetime(2025, 9, 30, 18)) == (2025, 9, 30, 18)
    with pytest.raises(ValueError, match="four fields"):
        timestamp_key([2025, 9, 30])


def test_convi_union_pixel_counts_match_periodic_latent_blocks():
    from experiments.observations.convi_obs_encoder import (
        _pixel_mask_to_latent_counts,
    )

    pixel_mask = torch.tensor([[
        [True, True, False, False],
        [True, True, False, False],
    ]])
    counts = _pixel_mask_to_latent_counts(pixel_mask, resize=2)

    assert torch.equal(counts, torch.tensor([0, 4, 0, 4]))


def test_dense_convi_bypass_config_requires_spatial_fixed_encoder_ppgf():
    config = OmegaConf.create({
        "ppgf_convi_bypass_threshold": 50,
        "guidance_merge": "channel_ppgf",
        "ppgf_order_mode": "fixed",
    })
    sources = [
        {"name": "atms", "kind": "obs2era"},
        {"name": "convi", "kind": "convi_obs_encoder"},
    ]

    threshold = evaluator.validate_ppgf_convi_bypass_config(
        config,
        sources,
        {"ordering_granularity": "spatial_channel"},
    )

    assert threshold == 50


def test_ppgf_accumulate_config_accepts_global_fixed_channel_cache():
    config = OmegaConf.create({
        "guidance_merge": "channel_ppgf_accumulate",
        "ppgf_order_mode": "fixed",
        "ppgf_ordering_source": "matrix_cache",
        "ppgf_overlap_only": False,
        "ppgf_anneal_mode": "none",
        "ppgf_uncertainty_strength_mode": "constant",
        "ppgf_aligned_clip": 0.0,
        "ppgf_convi_bypass_threshold": None,
    })

    evaluator.validate_ppgf_accumulate_config(
        config,
        {"ordering_granularity": "channel", "order_size": 128},
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"ppgf_order_mode": "gumbel"}, "ppgf_order_mode=fixed"),
        ({"ppgf_overlap_only": True}, "ppgf_overlap_only=false"),
        ({"ppgf_anneal_mode": "cosine_steps"}, "ppgf_anneal_mode=none"),
    ],
)
def test_ppgf_accumulate_config_rejects_unsupported_modes(override, message):
    values = {
        "guidance_merge": "channel_ppgf_accumulate",
        "ppgf_order_mode": "fixed",
        "ppgf_ordering_source": "matrix_cache",
        "ppgf_overlap_only": False,
        "ppgf_anneal_mode": "none",
        "ppgf_uncertainty_strength_mode": "constant",
        "ppgf_aligned_clip": 0.0,
        "ppgf_convi_bypass_threshold": None,
        **override,
    }

    with pytest.raises(ValueError, match=message):
        evaluator.validate_ppgf_accumulate_config(
            OmegaConf.create(values),
            {"ordering_granularity": "channel", "order_size": 128},
        )


def test_ppgf_accumulate_config_accepts_spatial_cache():
    config = OmegaConf.create({
        "guidance_merge": "channel_ppgf_accumulate",
        "ppgf_order_mode": "fixed",
        "ppgf_ordering_source": "matrix_cache",
        "ppgf_overlap_only": False,
        "ppgf_anneal_mode": "none",
        "ppgf_uncertainty_strength_mode": "constant",
        "ppgf_aligned_clip": 0.0,
        "ppgf_convi_bypass_threshold": None,
    })

    evaluator.validate_ppgf_accumulate_config(
        config,
        {
            "ordering_granularity": "spatial_channel",
            "spatial_shape": (23, 47, 128),
            "order_size": 23 * 47 * 128,
        },
    )


def test_ppgf_top1_config_accepts_spatial_cache():
    config = OmegaConf.create({
        "guidance_merge": "channel_ppgf_top1",
        "ppgf_order_mode": "fixed",
        "ppgf_ordering_source": "matrix_cache",
        "ppgf_overlap_only": False,
        "ppgf_anneal_mode": "none",
        "ppgf_uncertainty_strength_mode": "constant",
        "ppgf_aligned_clip": 0.0,
        "ppgf_convi_bypass_threshold": None,
    })

    evaluator.validate_ppgf_top1_config(
        config,
        {
            "ordering_granularity": "spatial_channel",
            "spatial_shape": (23, 47, 128),
            "order_size": 23 * 47 * 128,
        },
    )
