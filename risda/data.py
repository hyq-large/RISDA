from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.nn import functional as F

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
