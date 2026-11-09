import numpy as np
import pytest
import torch

from appa.data.datasets import ERA5DataNpy_2016_2025_1h


def _write_hour(tmp_path, values):
    data_dir = tmp_path / "era5"
    month_dir = data_dir / "2017" / "01"
    month_dir.mkdir(parents=True)
    np.save(month_dir / "20170101_00.npy", values)
    return data_dir


def _write_stat(tmp_path, name, values):
    path = tmp_path / f"{name}.npy"
    np.save(path, values)
    return str(path)


def test_hourly_dataset_converts_normalization_basis(tmp_path):
    normalized = np.linspace(-2.0, 2.0, 69 * 2 * 3, dtype=np.float32).reshape(69, 2, 3)
    data_dir = _write_hour(tmp_path, normalized)

    source_mean = np.linspace(100.0, 168.0, 70, dtype=np.float32)
    source_std = np.linspace(1.0, 7.9, 70, dtype=np.float32)
    target_mean = np.linspace(50.0, 84.5, 70, dtype=np.float32)
    target_std = np.linspace(2.0, 8.9, 70, dtype=np.float32)
    paths = {
        "source_mean_path": _write_stat(tmp_path, "source_mean", source_mean),
        "source_std_path": _write_stat(tmp_path, "source_std", source_std),
        "target_mean_path": _write_stat(tmp_path, "target_mean", target_mean),
        "target_std_path": _write_stat(tmp_path, "target_std", target_std),
    }

    dataset = ERA5DataNpy_2016_2025_1h(
        data_path=data_dir,
        time_range=["2017-01-01T00", "2017-01-01T00"],
        **paths,
    )
    actual, date = dataset[0]
    expected = (
        normalized * source_std[:69, None, None]
        + source_mean[:69, None, None]
        - target_mean[:69, None, None]
    ) / target_std[:69, None, None]

    torch.testing.assert_close(actual[0], torch.from_numpy(expected))
    torch.testing.assert_close(date, torch.tensor([[2017, 1, 1, 0]]))


def test_hourly_dataset_identity_basis_is_unchanged(tmp_path):
    normalized = np.linspace(-1.0, 1.0, 69 * 2 * 3, dtype=np.float32).reshape(69, 2, 3)
    data_dir = _write_hour(tmp_path, normalized)
    mean = _write_stat(tmp_path, "mean", np.arange(70, dtype=np.float32))
    std = _write_stat(tmp_path, "std", np.arange(70, dtype=np.float32) + 1.0)

    dataset = ERA5DataNpy_2016_2025_1h(
        data_path=data_dir,
        time_range=["2017-01-01T00", "2017-01-01T00"],
        source_mean_path=mean,
        source_std_path=std,
        target_mean_path=mean,
        target_std_path=std,
    )
    actual, _ = dataset[0]

    torch.testing.assert_close(actual[0], torch.from_numpy(normalized), rtol=0, atol=0)


def test_hourly_dataset_requires_complete_statistics(tmp_path):
    with pytest.raises(ValueError, match="all four statistics paths"):
        ERA5DataNpy_2016_2025_1h(
            data_path=tmp_path,
            source_mean_path="mean.npy",
        )
