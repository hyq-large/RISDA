import pytest
import torch
from omegaconf import OmegaConf
from experiments.observations.evaluate_guided_six_frame import CHANNEL_NAMES, _metric_targets
from experiments.observations.evaluate_guided_six_frame_metrics import (
    _accumulate_sparse_physical, _latitude_weights, finalize_sparse_rmse,
    _resolve_metrics_output_dir,
)

def test_sparse_physical_metric_filters_channels_and_uses_both_normalizations():
    prediction = torch.zeros(69, 1, 2)
    prediction[0, 0, 0] = 1.0
    observation = {
        "pixel_index": torch.tensor([0, 1]),
        "channel_index": torch.tensor([0, 65]),
        "values": torch.tensor([1.0, 99.0]),
    }
    prediction_mean = torch.zeros(69)
    prediction_std = torch.ones(69)
    prediction_mean[0] = 10.0
    prediction_std[0] = 2.0
    observation_mean = torch.zeros(69)
    observation_std = torch.ones(69)
    observation_mean[0] = 5.0
    observation_std[0] = 3.0
    squared_error_sum = torch.zeros(69, dtype=torch.float64)
    counts = torch.zeros(69, dtype=torch.int64)

    selected = _accumulate_sparse_physical(
        prediction,
        observation,
        channel_min=0,
        channel_max=65,
        prediction_mean=prediction_mean,
        prediction_std=prediction_std,
        observation_mean=observation_mean,
        observation_std=observation_std,
        squared_error_sum=squared_error_sum,
        counts=counts,
    )

    assert selected == 1
    assert squared_error_sum[0].item() == 16.0
    assert counts[0].item() == 1
    assert counts[65].item() == 0

def test_latitude_weight_modes_are_explicit():
    assert torch.equal(_latitude_weights(3, "none", "cpu"), torch.ones(3))
    cosine = _latitude_weights(3, "cosine", "cpu")
    assert cosine[0].item() == pytest.approx(0.0, abs=1e-6)
    assert cosine[1].item() == pytest.approx(1.0)
    assert cosine[2].item() == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="latitude weighting"):
        _latitude_weights(3, "invalid", "cpu")

def test_sparse_rmse_preserves_empty_channels_as_nan():
    result = finalize_sparse_rmse(torch.tensor([9.0, 0.0]), torch.tensor([1, 0]))

    assert result[0].item() == 3.0
    assert torch.isnan(result[1])

def test_qc_metric_output_subdir_stays_inside_run_directory(tmp_path):
    run_dir = tmp_path / "validation"
    run_dir.mkdir()

    assert _resolve_metrics_output_dir(run_dir, None) == run_dir.resolve()
    assert _resolve_metrics_output_dir(run_dir, "metrics_qc") == (run_dir / "metrics_qc").resolve()
    for invalid in ("", ".", "../escape", str(tmp_path / "absolute")):
        with pytest.raises(ValueError, match="metrics-output-subdir"):
            _resolve_metrics_output_dir(run_dir, invalid)

def test_all_metric_target_selects_every_era_channel():
    config = OmegaConf.create({"metrics": {"target_names": "all"}})

    names, indices = _metric_targets(config)

    assert names == list(CHANNEL_NAMES)
    assert indices == list(range(69))
