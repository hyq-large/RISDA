import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import ConventionalObservations, SatelliteObservations
from ..reproduction import EraFields, hourly_times


def interval_times(paths, interval):
    hours = hourly_times(*interval)
    selected = sorted(time for time in paths if hours[0] <= time <= hours[-1])
    if not selected:
        raise ValueError(f"No samples in {list(interval)}")
    return selected


class EraDataset(Dataset):
    def __init__(self, data, interval, require_all=False):
        self.data_config = data
        self.fields = EraFields(data.era5)
        self.mean = self.fields.stat(data.target_mean)
        self.std = self.fields.stat(data.target_std)
        if (self.std <= 0).any():
            raise ValueError("Target ERA5 standard deviation must be positive")
        self.times = interval_times(self.fields.paths, interval)
        if require_all:
            self.fields.require(hourly_times(*interval))
        self.input_channels = 69

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        field = (self.fields.read(self.times[index]) - self.mean) / self.std
        return torch.from_numpy(field)


class ObservationDataset(EraDataset):
    def __init__(self, data, interval, stage):
        super().__init__(data, interval)
        self.stage = stage
        if stage == "convi":
            self.observations = ConventionalObservations(
                data.surf_dir, data.icoa_dir, data.mean_std_dir,
                data.era5.grid_shape, channel_masks=True,
            )
            available = set(self.observations.surf) | set(self.observations.icoa)
        else:
            self.observations = SatelliteObservations(
                data.sate_data_path, data.sate_standard_dir, data.sate_name, data.use_satzen
            )
            available = set(self.observations.paths)
        self.times = [time for time in self.times if time in available]
        if not self.times:
            raise ValueError("No exact-hour ERA5/observation pairs in the selected interval")
        self.input_channels = self.observations.input_channels

    def __getitem__(self, index):
        observations, mask = self.observations.read(self.times[index])
        if tuple(observations.shape[-2:]) != tuple(self.fields.config.grid_shape):
            raise ValueError(f"Observation grid differs from ERA5 at {self.times[index]}")
        if self.stage == "convi":
            return observations, observations[:4].clone(), observations[4:8].clone()
        return observations, super().__getitem__(index), mask


class LatentWindows(Dataset):
    def __init__(self, path, interval, expected_shape):
        self.path = str(path)
        self.file = None
        with h5py.File(self.path, "r") as file:
            dates = np.asarray(file["dates"])
            shape = tuple(file["latents"].shape[1:])
            if shape != (expected_shape[0] * expected_shape[1], expected_shape[2]):
                raise ValueError("Latent HDF5 shape does not match the autoencoder")
            if len(dates) != len(file["latents"]):
                raise ValueError("Latent dates and values have different lengths")
            self.complete = np.asarray(file["complete"], bool) if "complete" in file else np.ones(len(dates), bool)
        self.dates = dates.astype(np.int64)
        times = [datetime(*map(int, row)) for row in dates]
        if any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Latent dates must be unique and increasing")
        bounds = hourly_times(*interval)
        self.indices = [
            i for i in range(len(times) - 5)
            if bounds[0] <= times[i] and times[i + 5] <= bounds[-1]
            and times[i + 3].hour % 6 == 0
            and times[i + 5] - times[i] == timedelta(hours=5)
            and self.complete[i:i + 6].all()
        ]
        self.times = [times[i + 3] for i in self.indices]
        if not self.indices:
            raise ValueError("No complete six-hour latent windows in the selected interval")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        start = self.indices[index]
        latent = np.asarray(self.file["latents"][start:start + 6], dtype=np.float32)
        if not np.isfinite(latent).all():
            raise ValueError(f"Nonfinite latent at {self.times[index]}")
        return torch.from_numpy(latent), torch.from_numpy(self.dates[start:start + 6].copy())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state


def dataset_identity(dataset):
    digest = hashlib.sha256()
    for time in dataset.times:
        digest.update(time.isoformat().encode())
    paths = []
    if isinstance(dataset, EraDataset):
        paths.extend(dataset.fields.paths[t] for t in dataset.times)
        paths.extend(Path(dataset.data_config[k]) for k in ("target_mean", "target_std"))
        if dataset.fields.config.coordinate == "normalized":
            paths.extend(Path(dataset.fields.config[k]) for k in ("mean", "std"))
        if isinstance(dataset, ObservationDataset):
            data = dataset.data_config
            norm = Path(data.mean_std_dir if dataset.stage == "convi" else data.sate_standard_dir)
            name = "era5" if dataset.stage == "convi" else data.sate_name
            paths.extend(norm / f"{kind}_{name}.npy" for kind in ("mean", "std"))
        for attr in ("paths", "surf", "icoa"):
            indexed = getattr(getattr(dataset, "observations", None), attr, {})
            paths.extend(indexed[t] for t in dataset.times if t in indexed)
    else:
        paths.append(Path(dataset.path))
    for path in paths:
        stat = path.stat()
        digest.update(json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns]).encode())
    return {"samples": len(dataset), "first": dataset.times[0].isoformat(),
            "last": dataset.times[-1].isoformat(), "index_sha256": digest.hexdigest()}
