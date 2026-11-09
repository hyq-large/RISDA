import numpy as np
import pandas as pd
import pytest
import torch

from experiments.observations import train_obs2era as training


def _paired_inputs(tmp_path, monkeypatch, hours):
    times = [pd.Timestamp(f"2024-01-01T{hour:02d}") for hour in hours]
    month = tmp_path / "2024" / "01"
    month.mkdir(parents=True)
    for time in times:
        np.save(month / f"{time:%Y%m%d_%H}.npy", np.ones((69, 2, 3), dtype=np.float32))

    class FakeSatellite:
        obs_channel = 1

        def __init__(self, **kwargs):
            self.times = times

        def __getitem__(self, index):
            return torch.ones(1, 2, 2, 3), None

    monkeypatch.setattr(training, "SatelliteAEDataset", FakeSatellite)
    return dict(
        era_data_path=tmp_path,
        sate_data_path=tmp_path,
        sate_name="atms",
        sate_standard_dir=tmp_path,
    )


def test_hourly_pairing_keeps_non_synoptic_hours_and_converts_targets(tmp_path, monkeypatch):
    kwargs = _paired_inputs(tmp_path, monkeypatch, [1, 2])
    stats = {}
    for name, value in (
        ("source_mean", 10), ("source_std", 2),
        ("target_mean", 4), ("target_std", 4),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, np.full(69, value, dtype=np.float32))
        stats[f"{name}_path"] = str(path)
    dataset = training.Atms2EraPairDataset(
        **kwargs, era_cadence_hours=1, era_normalization=stats,
    )
    assert len(dataset) == 2
    assert dataset.dates[:, 3].tolist() == [1, 2]
    for index, hour in enumerate([1, 2]):
        _, target, _, date = dataset[index]
        # (1 * 2 + 10 - 4) / 4 = 2 in the ERA decoder's target basis.
        torch.testing.assert_close(target, torch.full((69, 2, 3), 2.0))
        assert date[-1].item() == hour


def test_default_pairing_preserves_six_hourly_data(tmp_path, monkeypatch):
    kwargs = _paired_inputs(tmp_path, monkeypatch, [0, 1, 6])
    dataset = training.Atms2EraPairDataset(**kwargs)
    assert dataset.dates[:, 3].tolist() == [0, 6]
    torch.testing.assert_close(dataset[0][1], torch.ones(69, 2, 3))


def test_six_hourly_pairing_rejects_unused_normalization(tmp_path, monkeypatch):
    kwargs = _paired_inputs(tmp_path, monkeypatch, [0])
    with pytest.raises(ValueError, match="requires era_cadence_hours=1"):
        training.Atms2EraPairDataset(**kwargs, era_normalization={"source_mean_path": "unused"})
