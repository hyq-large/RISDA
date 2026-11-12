from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.nn import functional as F

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
CHANNELS = tuple(f"{v}{p}" for v in ("z", "t", "u", "v", "r") for p in LEVELS) + (
    "t2m",
    "u10",
    "v10",
    "msl",
)

SATELLITE_CHANNELS = {"atms": 22, "amsua": 15, "mhs": 5, "mwts": 17, "mwhs": 15}


def index_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = {}
    for path in sorted(directory.rglob("*.nc")):
        try:
            time = datetime.strptime(path.stem.split("_")[-1], "%Y%m%d%H")
        except ValueError:
            continue
        if time in paths:
            raise ValueError(f"Duplicate observation timestamp: {paths[time]}, {path}")
        paths[time] = path
    return paths


def load_normalization(directory, name, channels):
    mean = np.load(Path(directory) / f"mean_{name}.npy").astype(np.float32)
    std = np.load(Path(directory) / f"std_{name}.npy").astype(np.float32)
    if mean.size != channels or std.size != channels:
        raise ValueError(f"Expected {channels} normalization channels for {name}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"Invalid normalization for {name}")
    return mean.reshape(1, channels, 1, 1), std.reshape(1, channels, 1, 1)


def latent_mask(pixel_mask, resize):
    mask = pixel_mask.float()[None]
    height, width = mask.shape[-2:]
    pad_h, pad_w = (-height) % resize, (-width) % resize + 2 * resize
    mask = F.pad(mask, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode="circular")
    mask = F.pad(mask, (0, 0, pad_h // 2, pad_h - pad_h // 2))
    return F.avg_pool2d(mask, resize).reshape(-1) > 0


class SatelliteObservations:
    def __init__(self, directory, normalization, name, use_satzen=False):
        self.name = name
        self.channels = SATELLITE_CHANNELS[name]
        self.use_satzen = bool(use_satzen)
        self.paths = index_files(Path(directory) / name)
        self.mean, self.std = load_normalization(normalization, name, self.channels)

    @property
    def input_channels(self):
        return self.channels + 1 + int(self.use_satzen)

    def read(self, time):
        if time not in self.paths:
            return None
        with xr.open_dataset(self.paths[time]) as dataset:
            values = dataset.x.values.astype(np.float32)
        if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] < self.channels:
            raise ValueError(f"Invalid observation shape in {self.paths[time]}")
        values_normalized = (values[:, : self.channels] - self.mean) / self.std
        finite = np.isfinite(values_normalized)
        mask = finite.any(axis=1, keepdims=True).astype(np.float32)
        channels = [np.where(finite, values_normalized, 0), mask]
        if self.use_satzen:
            index = self.channels + (3 if self.name in {"mwts", "mwhs"} else 0)
            if values.shape[1] <= index:
                raise ValueError(f"Missing zenith angle in {self.paths[time]}")
            angle = np.cos(np.deg2rad(values[:, index : index + 1]))
            channels.append(np.nan_to_num(angle, nan=0.0))
        return torch.from_numpy(np.concatenate(channels, axis=1)[0]), torch.from_numpy(mask[0])


class ConventionalObservations:
    def __init__(self, surf_dir, icoa_dir, normalization, grid_shape, channel_masks=False):
        self.surf = index_files(surf_dir)
        self.icoa = {} if icoa_dir is None else index_files(icoa_dir)
        mean, std = load_normalization(normalization, "era5", 69)
        self.mean, self.std = mean[:, 65:], std[:, 65:]
        self.grid_shape = tuple(grid_shape)
        self.channel_masks = bool(channel_masks)

    @property
    def input_channels(self):
        return 8 if self.channel_masks else 5

    def _read(self, paths, time):
        if time not in paths:
            return np.full((1, 4, *self.grid_shape), np.nan, dtype=np.float32)
        with xr.open_dataset(paths[time]) as dataset:
            values = dataset.sel(channel=["t2m", "u10", "v10", "slp"]).x.values.astype(np.float32)
        if values.shape != (1, 4, *self.grid_shape):
            raise ValueError(f"Invalid surface grid shape in {paths[time]}")
        values[:, -1] *= 100.0
        return values

    def read(self, time):
        if time not in self.surf and time not in self.icoa:
            return None
        surf, icoa = self._read(self.surf, time), self._read(self.icoa, time)
        values = np.where(np.isfinite(surf), surf, icoa)
        finite = np.isfinite(values)
        values = np.where(finite, (values - self.mean) / self.std, 0.0)
        union = finite.any(axis=1, keepdims=True).astype(np.float32)
        masks = finite.astype(np.float32) if self.channel_masks else union
        return torch.from_numpy(np.concatenate([values, masks], axis=1)[0]), torch.from_numpy(
            union[0]
        )


class EraFields:
    def __init__(self, config):
        self.config = config
        self.paths = {}
        for path in sorted(Path(config.directory).rglob("*.npy")):
            for pattern in ("%Y%m%d_%H", "%Y%m%dT%H", "%Y%m%d%H"):
                try:
                    time = datetime.strptime(path.stem, pattern)
                    break
                except ValueError:
                    time = None
            if time is None:
                continue
            if time in self.paths:
                raise ValueError(f"Duplicate ERA5 timestamp: {time}")
            self.paths[time] = path
        if not self.paths:
            raise ValueError("No hourly ERA5 .npy files found")
        if config.coordinate not in ("physical", "normalized"):
            raise ValueError("ERA5 coordinate must be physical or normalized")
        self.mean = self.std = None
        if config.coordinate == "normalized":
            self.mean = self.stat(config.mean)
            self.std = self.stat(config.std)
            if (self.std <= 0).any():
                raise ValueError("ERA5 standard deviation must be positive")

    @staticmethod
    def stat(path):
        value = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        if value.size != 69 or not np.isfinite(value).all():
            raise ValueError(f"Expected 69 finite normalization values: {path}")
        return value.reshape(69, 1, 1)

    def require(self, times):
        missing = [t.isoformat() for t in times if t not in self.paths]
        if missing:
            raise ValueError(f"Missing {len(missing)} ERA5 times; first: {missing[0]}")

    def read(self, time):
        if time not in self.paths:
            raise FileNotFoundError(f"Missing exact-hour ERA5 field: {time}")
        value = np.asarray(np.load(self.paths[time]), dtype=np.float32)
        if value.shape != (69, *self.config.grid_shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid ERA5 field: {self.paths[time]}")
        if self.mean is not None:
            value = value * self.std + self.mean
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite physical ERA5 field: {time}")
        return value
