"""Timestamp-aligned ERA-latent and satellite-TB data for direct E2O."""

import h5py
import numpy as np
import pandas as pd
import torch
import xarray as xr

from pathlib import Path
from torch.utils.data import Dataset

from appa.data.datasets import SatelliteAEDataset


DIRECT_E2O_DATA_CONTRACT = "exact_hour_cos_satzen_v2"


def validate_direct_e2o_data_contract(data_config):
    """Reject checkpoints/configs that do not carry the current input contract."""
    contract = str(data_config.get("contract", ""))
    if contract != DIRECT_E2O_DATA_CONTRACT:
        raise RuntimeError(
            "Direct E2O data contract mismatch: expected "
            f"{DIRECT_E2O_DATA_CONTRACT!r}, got {contract!r}. "
            "The cos(satellite-zenith) decoder must be trained from scratch."
        )
    if not bool(data_config.get("use_satzen", False)):
        raise RuntimeError(
            "Direct E2O requires data.use_satzen=true under contract "
            f"{DIRECT_E2O_DATA_CONTRACT!r}"
        )
    try:
        cadence_hours = int(data_config.get("cadence_hours"))
    except (TypeError, ValueError):
        raise RuntimeError(
            "Direct E2O requires a positive integer data.cadence_hours"
        ) from None
    if cadence_hours <= 0:
        raise RuntimeError(
            "Direct E2O requires a positive integer data.cadence_hours"
        )
    return cadence_hours


def timestamp_key(value):
    """Convert a timestamp-like value to a stable ``(Y, M, D, H)`` key."""
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 4:
        return tuple(int(part) for part in value)
    timestamp = pd.Timestamp(value)
    return timestamp.year, timestamp.month, timestamp.day, timestamp.hour


def strict_time_intersection(reference_dates, latent_dates):
    """Map each reference date to the exact matching latent row.

    Reference order is retained. Duplicate dates are rejected because they
    would make the observation-to-latent pairing ambiguous.
    """
    latent_index = {}
    for index, date in enumerate(latent_dates):
        key = timestamp_key(date)
        if key in latent_index:
            raise ValueError(f"duplicate ERA latent timestamp: {key}")
        latent_index[key] = index

    matched_reference = []
    matched_latent = []
    seen_reference = set()
    for index, date in enumerate(reference_dates):
        key = timestamp_key(date)
        if key in seen_reference:
            raise ValueError(f"duplicate O2E reference timestamp: {key}")
        seen_reference.add(key)
        if key in latent_index:
            matched_reference.append(index)
            matched_latent.append(latent_index[key])
    return (
        np.asarray(matched_reference, dtype=np.int64),
        np.asarray(matched_latent, dtype=np.int64),
    )


def validate_latent_cadence(latent_dates, expected_hours):
    """Require a complete, ordered latent date axis at the configured cadence."""
    dates = np.asarray(latent_dates, dtype=np.int64)
    if dates.ndim != 2 or dates.shape[1] != 4:
        raise ValueError(
            f"ERA latent dates must have shape (N,4); got {dates.shape}"
        )
    if len(dates) < 2:
        raise ValueError("ERA latent dates need at least two timestamps")
    timestamps = pd.to_datetime(
        {
            "year": dates[:, 0],
            "month": dates[:, 1],
            "day": dates[:, 2],
            "hour": dates[:, 3],
        },
        errors="raise",
    )
    hour_values = np.asarray(
        timestamps, dtype="datetime64[h]"
    ).astype(np.int64)
    deltas = np.diff(hour_values)
    bad = np.flatnonzero(deltas != int(expected_hours))
    if len(bad):
        index = int(bad[0])
        raise ValueError(
            "ERA latent cadence mismatch: expected consecutive "
            f"{int(expected_hours)}h steps, but {tuple(dates[index])} -> "
            f"{tuple(dates[index + 1])} is {int(deltas[index])}h"
        )


def load_tb_target_at_time(satellite, timestamp):
    """Load exact-hour standardized TB, validity, and cos(zenith).

    The target retains a validity mask for every TB channel. No neighbouring
    hour is read or used to fill missing pixels. ATMS/AMSUA/MHS use their raw
    satellite-zenith channel; MWTS/MWHS expose Earth incidence through the
    same satzen_channel attribute.
    """
    channel_count = int(satellite.obs_channel)
    mean = np.asarray(satellite.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(satellite.std, dtype=np.float32).reshape(-1)
    if mean.size != channel_count or std.size != channel_count:
        raise ValueError(
            f"expected {channel_count} TB statistics, got mean={mean.size}, std={std.size}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("TB mean/std must be finite and every std must be positive")

    path = satellite._all_paths.get(pd.Timestamp(timestamp))
    if path is None:
        return (
            np.zeros((channel_count, 720, 1440), dtype=np.float32),
            np.zeros((channel_count, 720, 1440), dtype=bool),
            np.zeros((1, 720, 1440), dtype=np.float32),
        )

    with xr.open_dataset(path) as dataset:
        frame = np.asarray(dataset.x.values[0], dtype=np.float32)
    raw = frame[:channel_count]
    if raw.ndim != 3 or raw.shape[0] != channel_count:
        raise ValueError(
            "satellite frame must have shape "
            f"({channel_count},H,W); got {raw.shape}"
        )
    satzen_channel = int(satellite.satzen_channel)
    if not 0 <= satzen_channel < frame.shape[0]:
        raise ValueError(
            f"satellite zenith channel {satzen_channel} is outside raw frame "
            f"with {frame.shape[0]} channels"
        )
    zenith_degrees = frame[satzen_channel]
    if zenith_degrees.shape != raw.shape[1:]:
        raise ValueError(
            "satellite zenith field must share the TB spatial shape; "
            f"got {zenith_degrees.shape} and {raw.shape[1:]}"
        )
    valid = np.isfinite(raw)
    standardized = (raw - mean[:, None, None]) / std[:, None, None]
    standardized = np.where(valid, standardized, 0.0).astype(np.float32)
    zenith_valid = np.isfinite(zenith_degrees) & valid.any(axis=0)
    cos_satzen = np.zeros(zenith_degrees.shape, dtype=np.float32)
    cos_satzen[zenith_valid] = np.cos(
        np.deg2rad(zenith_degrees[zenith_valid])
    ).astype(np.float32)
    return standardized, valid, cos_satzen[None]


class EraLatentSatelliteTbDataset(Dataset):
    """Pair native ERA latents with real gridded satellite TB.

    The latent date axis is authoritative and must match the configured regular
    cadence. Samples are the exact timestamp intersection with satellite files;
    no ERA physical array or separate ERA file index is needed.
    """

    def __init__(
        self,
        era_latent_path,
        cadence_hours,
        satellite_data_path,
        satellite_name,
        satellite_standard_dir,
    ):
        self.era_latent_path = str(Path(era_latent_path))
        self.satellite = SatelliteAEDataset(
            data_path=satellite_data_path,
            sate_name=satellite_name,
            standard_dir=satellite_standard_dir,
            blanket_size=1,
            time_range=None,
            use_satzen=True,
        )
        self.satellite_name = str(satellite_name)
        self.n_obs = int(self.satellite.obs_channel)
        if not self.satellite.times:
            raise RuntimeError(f"no {self.satellite_name} timestamps were found")

        latent_path = Path(self.era_latent_path)
        if not latent_path.is_file():
            raise FileNotFoundError(f"ERA latent dump not found: {latent_path}")
        with h5py.File(latent_path, "r") as handle:
            if "latents" not in handle or "dates" not in handle:
                raise KeyError(f"{latent_path} must contain 'latents' and 'dates'")
            latent_dates = np.asarray(handle["dates"][:], dtype=np.int64)
            self.latent_sample_shape = tuple(handle["latents"].shape[1:])
        self.cadence_hours = int(cadence_hours)
        validate_latent_cadence(latent_dates, self.cadence_hours)

        satellite_dates = np.asarray(
            [timestamp_key(value) for value in self.satellite.times],
            dtype=np.int64,
        )
        satellite_rows, latent_rows = strict_time_intersection(
            satellite_dates, latent_dates
        )
        if not len(satellite_rows):
            raise RuntimeError(
                f"no exact timestamp overlap between {self.satellite_name} "
                f"and {latent_path}"
            )
        self.satellite_indices = satellite_rows
        self.latent_indices = latent_rows
        self.dates = satellite_dates[satellite_rows]
        self.alignment = {
            "satellite_samples": int(len(satellite_dates)),
            "latent_samples": int(len(latent_dates)),
            "aligned_samples": int(len(self.dates)),
            "missing_from_latent_dump": int(
                len(satellite_dates) - len(self.dates)
            ),
            "missing_from_satellite": int(
                len(latent_dates) - len(self.dates)
            ),
            "latent_cadence_hours": self.cadence_hours,
        }
        self._latent_handle = None

    def __len__(self):
        return len(self.dates)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_latent_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_latent_handle", None)
        if handle is not None:
            handle.close()

    def _get_latent_handle(self):
        if self._latent_handle is None:
            self._latent_handle = h5py.File(self.era_latent_path, "r")
        return self._latent_handle

    def _load_tb_target(self, center_time):
        return load_tb_target_at_time(self.satellite, center_time)

    def __getitem__(self, index):
        latent = np.asarray(
            self._get_latent_handle()["latents"][self.latent_indices[index]],
            dtype=np.float32,
        )
        satellite_index = int(self.satellite_indices[index])
        center_time = self.satellite.times[satellite_index]
        target, valid, cos_satzen = self._load_tb_target(center_time)
        target = np.moveaxis(target, 0, -1).reshape(-1, self.n_obs)
        valid = np.moveaxis(valid, 0, -1).reshape(-1, self.n_obs)
        cos_satzen = np.moveaxis(cos_satzen, 0, -1).reshape(-1, 1)
        return (
            torch.from_numpy(latent),
            torch.from_numpy(target),
            torch.from_numpy(valid),
            torch.from_numpy(cos_satzen),
            torch.from_numpy(self.dates[index].copy()),
        )


def split_train_valid_indices(dates):
    """Train before 2025 and validate on January-March 2025."""
    dates = np.asarray(dates)
    if dates.ndim != 2 or dates.shape[1] != 4:
        raise ValueError(f"dates must have shape (N,4); got {dates.shape}")
    if len(dates) < 2:
        raise ValueError("at least two aligned samples are required")
    valid_mask = (
        (dates[:, 0] == 2025)
        & (dates[:, 1] >= 1)
        & (dates[:, 1] <= 3)
    )
    valid = np.flatnonzero(valid_mask)
    if not len(valid):
        raise ValueError("Q1 2025 contains no validation samples")
    train = np.flatnonzero(dates[:, 0] < 2025)
    if not len(train):
        raise ValueError("Q1 2025 validation leaves no training samples")
    return train, valid, "2025-01-01..2025-03-31"
