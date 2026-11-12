import json
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .config import anchor_times
from .environment import ROOT, setup

LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
CHANNELS = tuple(f"{v}{p}" for v in ("z", "t", "u", "v", "r") for p in LEVELS) + (
    "t2m",
    "u10",
    "v10",
    "msl",
)


def read_config(path, overrides=()):
    setup()
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def default_config(name):
    import sys

    local = ROOT / "configs" / name
    return local if local.is_file() else Path(sys.prefix) / "share/risda" / name


def json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def hourly_times(start, end):
    return anchor_times(OmegaConf.create({"start": start, "end": end, "step_hours": 1}))


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
