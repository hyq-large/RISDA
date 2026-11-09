"""Guided sampling on a configurable UTC anchor axis.

Each independent window uses exact-hour observations at [T-3, T-2, T-1, T,
T+1, T+2]. Only frame four at T is retained. Ranks own independent anchors;
there is no overlap stitching or distributed collective during sampling.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import numpy as np
import os
import random
import sys
import time
import torch
import torch.distributed as dist

from datetime import datetime, timedelta
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Mapping, Sequence

from appa.config.hydra import compose
from appa.data.datasets import CONVIOBS, ERA5DataNpy_2016_2025_1h, SatelliteAEDataset
from appa.diffusion import (
    Denoiser,
    MMPSDenoiser,
    MultiInstrumentMMPSDenoiser,
    TrajectoryDenoiser,
    create_schedule,
)
from appa.sampling import PPGFSampler, select_sampler
from appa.save import load_auto_encoder, load_denoiser, safe_load
from experiments.observations.build_spatial_ordering_cache import (
    SPATIAL_PPGF_MATRIX_KIND,
)
from experiments.observations.convi_obs_encoder import (
    CONVI_OBS_ENCODER_CHANNELS,
    load_convi_obs_encoder_latent_at_time,
)
from experiments.observations.convi_physical import physical_normalization_contract
from experiments.observations.direct_e2o_assimilation import (
    load_direct_e2o_at_time,
    load_direct_e2o_source,
    select_direct_e2o_prediction,
)
from experiments.observations.reanalysis_obs2era_dual import (
    channel_ppgf_uncertainty,
    filter_physical_observation_by_block_mask,
    load_convi_at_time,
    load_obs2era_latent_at_time,
    pcgrad_order_uncertainty,
)
from risda.models.encoders import load_obs2era_encoder

PROTOCOL_KIND = "guided_independent_six_frame_v2"
ANCHOR_KIND = "guided_independent_six_frame_anchor_v2"
COLLECTION_KIND = "guided_independent_frame4_collection_v2"
METRICS_KIND = "guided_independent_frame4_rmse_v2"
RESIDUAL_VAR_Y_KIND = "hourly_encoder_centered_residual_var_y_v1"
SPATIAL_VAR_Y_KIND = "mask_frac_obs_encoder_spatial_uncertainty_v1"
SPATIAL_VAR_Y_MINIMUM_COUNT = 2
PPGF_RELIABILITY_METRICS = ("native_std", "native_rmse")
CHANNEL_PPGF_MATRIX_KIND = "mask_frac_obs_encoder_channel_uncertainty"
PPGF_CONVI_BYPASS_DEFAULT_MERGE_MODE = "retained_ppgf_plus_bypass"
PPGF_CONVI_BYPASS_MERGE_MODES = (
    PPGF_CONVI_BYPASS_DEFAULT_MERGE_MODE,
    "sum_all",
)

WINDOW_OFFSETS = (-3, -2, -1, 0, 1, 2)
WINDOW_SIZE = 6
ANCHOR_FRAME_NUMBER = 4
ANCHOR_FRAME_INDEX = ANCHOR_FRAME_NUMBER - 1
ANCHOR_INTERVAL_HOURS = 6
ENSEMBLE_MEMBER_SEED_STRIDE = 1_000_003
ENSEMBLE_REDUCTION = "arithmetic_mean_in_diffusion_standardized_latent_space"

# Default convenience axis. Actual runs always use config.timeline.
SPLIT_ANCHOR_RANGES = {"evaluation": (("2025-04-01T00", "2025-06-30T18"),)}
SPLIT_BOUNDS = {key: value[0] for key, value in SPLIT_ANCHOR_RANGES.items()}
EXPECTED_SPLIT_ANCHORS = {"evaluation": 364}
from risda.timeline import calibration_checks as calibration_period_checks, select_anchors


def _check_calibration_period(payload: Mapping, period: str) -> None:
    for field, expected in calibration_period_checks(period).items():
        if payload.get(field) != expected:
            raise RuntimeError(f"calibration {field} mismatch: {payload.get(field)!r} != {expected!r}")

CONVI_OBSERVATION_VARIANT = "qc"
CONVI_GUIDANCE_VALUE_SOURCES = (
    "observation",
    "era5_truth",
    "era5_truth_full_mask",
)
CONVI_QC_DIRECTORY_NAMES = {
    "surf_dir": "surf_qc",
    "icoa_dir": "icoads_qc",
    "temp_dir": "temp_time_qc",
}

PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ATMOSPHERIC_VARIABLES = ("z", "t", "u", "v", "r")
SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")
CHANNEL_NAMES = tuple(
    [f"{variable}{level}" for variable in ATMOSPHERIC_VARIABLES for level in PRESSURE_LEVELS]
    + list(SURFACE_VARIABLES)
)


def _parse_hour(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H")


def timestamp_key(value) -> tuple[int, int, int, int]:
    """Return a strict ``(year, month, day, hour)`` key."""
    if isinstance(value, datetime):
        return value.year, value.month, value.day, value.hour
    if isinstance(value, np.datetime64):
        value = np.datetime_as_string(value.astype("datetime64[h]"), unit="h")
        return timestamp_key(_parse_hour(value))
    values = torch.as_tensor(value).reshape(-1).tolist()
    if len(values) != 4:
        raise ValueError(f"timestamp must have four fields; got {values}")
    return tuple(int(item) for item in values)


def timestamp_tensor(value: datetime) -> torch.Tensor:
    return torch.tensor(timestamp_key(value), dtype=torch.long)


def format_hour(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H")


def serialized_split_anchor_ranges(split: str) -> list[list[str]]:
    """Return the exact inclusive anchor ranges recorded in run contracts."""
    split = str(split).lower()
    if split not in SPLIT_ANCHOR_RANGES:
        raise ValueError(f"split must be one of {tuple(SPLIT_ANCHOR_RANGES)}; got {split!r}")
    return [list(values) for values in SPLIT_ANCHOR_RANGES[split]]


def enumerate_split_anchors(split: str) -> list[datetime]:
    """Enumerate every retained 6-hour anchor in a protocol split."""
    split = str(split).lower()
    if split not in SPLIT_ANCHOR_RANGES:
        raise ValueError(f"split must be one of {tuple(SPLIT_ANCHOR_RANGES)}; got {split!r}")
    anchors = []
    for start_text, end_text in SPLIT_ANCHOR_RANGES[split]:
        start, end = (_parse_hour(value) for value in (start_text, end_text))
        current = start
        while current <= end:
            anchors.append(current)
            current += timedelta(hours=ANCHOR_INTERVAL_HOURS)
    expected = EXPECTED_SPLIT_ANCHORS[split]
    if len(anchors) != expected:
        raise RuntimeError(f"{split} produced {len(anchors)} anchors, expected {expected}")
    return anchors


def select_split_anchor_range(
    split: str,
    start_time: str | datetime | None = None,
    end_time: str | datetime | None = None,
) -> tuple[list[datetime], list[int]]:
    """Select an inclusive contiguous range from one canonical protocol split.

    The returned indices always refer to the complete canonical split. They
    keep member-zero seeds aligned between subset and full-split runs.
    """
    complete = enumerate_split_anchors(split)

    def parse_boundary(value, name: str, default: datetime) -> datetime:
        if value is None:
            return default
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = _parse_hour(str(value))
            except ValueError as error:
                raise ValueError(
                    f"{name} must use YYYY-MM-DDTHH; got {value!r}"
                ) from error
        if parsed not in complete:
            raise ValueError(
                f"{name}={format_hour(parsed)} is not a 6-hour anchor in the {split} split"
            )
        return parsed

    start = parse_boundary(start_time, "anchor_start_time", complete[0])
    end = parse_boundary(end_time, "anchor_end_time", complete[-1])
    if start > end:
        raise ValueError(
            f"anchor_start_time={format_hour(start)} is after "
            f"anchor_end_time={format_hour(end)}"
        )
    indices = [index for index, anchor in enumerate(complete) if start <= anchor <= end]
    if not indices:
        raise RuntimeError("selected anchor range is empty")
    return [complete[index] for index in indices], indices


def serialized_anchor_ranges(anchors: Sequence[datetime]) -> list[list[str]]:
    """Serialize one ordered anchor axis while preserving temporal gaps."""
    anchors = list(anchors)
    if not anchors:
        raise ValueError("anchor sequence cannot be empty")
    ranges = []
    start = previous = anchors[0]
    for anchor in anchors[1:]:
        if anchor - previous != timedelta(hours=ANCHOR_INTERVAL_HOURS):
            ranges.append([format_hour(start), format_hour(previous)])
            start = anchor
        previous = anchor
    ranges.append([format_hour(start), format_hour(previous)])
    return ranges


def ensemble_sample_seed(
    base_seed: int,
    source_anchor_index: int,
    sample_index: int,
) -> int:
    """Return the deterministic seed for one anchor/member pair."""
    source_anchor_index = int(source_anchor_index)
    sample_index = int(sample_index)
    if source_anchor_index < 0 or sample_index < 0:
        raise ValueError("anchor and sample indices must be non-negative")
    return (
        int(base_seed)
        + source_anchor_index
        + sample_index * ENSEMBLE_MEMBER_SEED_STRIDE
    )


def _num_samples_per_anchor(config: Mapping) -> int:
    raw = config.get("num_samples_per_anchor", 1)
    if isinstance(raw, bool):
        raise ValueError("num_samples_per_anchor must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("num_samples_per_anchor must be a positive integer") from error
    if value <= 0 or value != raw:
        raise ValueError("num_samples_per_anchor must be a positive integer")
    return value


def window_times(anchor: datetime) -> list[datetime]:
    """Return the six exact observation/generation hours for one anchor."""
    return [anchor + timedelta(hours=offset) for offset in WINDOW_OFFSETS]


def select_config_anchors(config, *, canonical=False):
    """Use the shared timeline; retain canonical indices for subset seeds."""
    return select_anchors(config, canonical=canonical)


def partition_anchor_indices(num_anchors: int, rank: int, world_size: int) -> list[int]:
    """Assign complete windows to ranks without any cross-rank window stitching."""
    if num_anchors < 0:
        raise ValueError("num_anchors must be non-negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size: {rank}/{world_size}")
    return list(range(rank, num_anchors, world_size))


def extract_anchor_frame(window: torch.Tensor) -> torch.Tensor:
    """Extract protocol frame four from a tensor whose first dimension is time."""
    if window.shape[0] != WINDOW_SIZE:
        raise ValueError(f"window must contain {WINDOW_SIZE} frames; got {window.shape[0]}")
    return window[ANCHOR_FRAME_INDEX]


def select_source_names(
    enabled_sources: Sequence[str], route: str, source_catalog: Mapping
) -> list[str]:
    """Validate a stable, duplicate-free source order for one route."""
    route = str(route).lower()
    if route == "no_obs":
        if enabled_sources:
            raise ValueError("route=no_obs requires enabled_sources=[]")
        return []
    if route not in {"encoder", "decoder"}:
        raise ValueError(f"route must be encoder, decoder, or no_obs; got {route!r}")
    names = [str(name).lower() for name in enabled_sources]
    if not names:
        raise ValueError(f"route={route} requires at least one enabled source")
    if len(set(names)) != len(names):
        raise ValueError(f"enabled_sources contains duplicates: {names}")
    missing = [name for name in names if name not in source_catalog]
    if missing:
        raise ValueError(f"unknown observation sources: {missing}")
    unavailable = [name for name in names if route not in source_catalog[name]]
    if unavailable:
        raise ValueError(f"sources do not support route={route}: {unavailable}")
    return names


def _same_path(left, right) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _checkpoint_path(model_dir: Path, checkpoint: str) -> Path:
    if str(checkpoint) not in {"best", "last"}:
        raise ValueError("evaluation source checkpoint must be best or last")
    return model_dir / f"model_{checkpoint}.pth"


def _optional_resolved_path(config, field: str) -> Path | None:
    value = config.get(field, None)
    return None if value is None else Path(value).expanduser().resolve()


def file_identity(path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _matches_model_directory_relocation(
    cache_path: Path,
    recorded_checkpoint: Path,
    expected_checkpoint: Path,
) -> bool:
    """Recognize a calibration cache moved together with its unchanged model directory."""
    return (
        cache_path.parent == expected_checkpoint.parent
        and recorded_checkpoint.parent.name == expected_checkpoint.parent.name
        and recorded_checkpoint.name == expected_checkpoint.name
    )


def load_calibrated_var_y_cache(
    cache_path,
    expected_representation: str,
    expected_source: str,
    expected_channels: int,
    expected_checkpoint=None,
    coordinate_scale: torch.Tensor | None = None,
    variance_floor: float = 1e-8,
    expected_period: str = "2025-01-01T00--2025-03-31T23",
) -> tuple[torch.Tensor, dict]:
    """Load centered residual variance for an explicitly selected period.

    ``coordinate_scale`` is used only to convert Encoder variance from the raw
    latent-statistics coordinate used during calibration into the effective
    diffusion coordinate ``sqrt(std**2 + ae_noise**2)``.  It is a deterministic
    coordinate conversion, not an observation-trust scale.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Calibration var_y cache is missing: {cache_path}")
    payload = safe_load(cache_path, map_location="cpu")
    _check_calibration_period(payload, expected_period)
    checks = {
        "kind": f"hourly_{expected_representation}_centered_residual_var_y_v1",
        "representation": str(expected_representation).lower(),
    }
    for field, expected in checks.items():
        actual = payload.get(field)
        if str(actual).lower() != str(expected).lower():
            raise RuntimeError(
                f"Calibration var_y {field} mismatch in {cache_path}: {actual!r} != {expected!r}"
            )
    if str(payload.get("source", "")).lower() != str(expected_source).lower():
        raise RuntimeError(
            f"Calibration var_y source mismatch in {cache_path}: "
            f"{payload.get('source')!r} != {expected_source!r}"
        )
    if payload.get("window_end_inclusive") is not True:
        raise RuntimeError(f"Calibration var_y cache is not end-inclusive: {cache_path}")
    if payload.get("num_requested_times") != calibration_period_checks(expected_period)["expected_hours"]:
        raise RuntimeError(f"calibration cache did not process every candidate hour: {cache_path}")

    if expected_checkpoint is not None:
        checkpoint = Path(expected_checkpoint).expanduser().resolve()
        identity = payload.get("metadata", {}).get("identities", {}).get("checkpoint", {})
        recorded = Path(payload.get("checkpoint_path", identity.get("path", ""))).expanduser().resolve()
        if recorded != checkpoint and not _matches_model_directory_relocation(
            cache_path,
            recorded,
            checkpoint,
        ):
            raise RuntimeError(
                f"Calibration var_y checkpoint mismatch in {cache_path}: {recorded} != {checkpoint}"
            )
        recorded_mtime = int(payload.get("checkpoint_mtime_ns", identity.get("mtime_ns", -1)))
        if recorded_mtime != checkpoint.stat().st_mtime_ns:
            raise RuntimeError(
                f"Calibration var_y checkpoint changed after calibration: {checkpoint}; "
                "regenerate the cache"
            )


    variance = torch.as_tensor(payload.get("variance"), dtype=torch.float32).reshape(-1)
    if variance.shape != (int(expected_channels),):
        raise RuntimeError(
            f"Calibration var_y shape mismatch in {cache_path}: "
            f"{tuple(variance.shape)} != {(int(expected_channels),)}"
        )
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise RuntimeError(f"Calibration var_y contains invalid values: {cache_path}")
    if coordinate_scale is not None:
        coordinate_scale = torch.as_tensor(coordinate_scale, dtype=torch.float32).reshape(-1)
        if coordinate_scale.shape != variance.shape:
            raise ValueError("coordinate_scale must have one value per var_y channel")
        variance = variance * coordinate_scale.square()
    variance_floor = float(variance_floor)
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    return variance.clamp_min(variance_floor), payload


def _validate_spatial_var_y_checkpoint(
    cache_path: Path,
    source: str,
    metadata: Mapping,
    expected_checkpoint: Path,
) -> None:
    identity = metadata.get("checkpoint")
    if not isinstance(identity, Mapping):
        raise RuntimeError(
            f"spatial var_y cache has no checkpoint identity for {source}: "
            f"{cache_path}"
        )
    expected_checkpoint = expected_checkpoint.expanduser().resolve()
    recorded_checkpoint = Path(identity.get("path", "")).expanduser().resolve()
    same_path = recorded_checkpoint == expected_checkpoint
    relocated_with_model = (
        recorded_checkpoint.name == expected_checkpoint.name
        and recorded_checkpoint.parent.name == expected_checkpoint.parent.name
    )
    if not same_path and not relocated_with_model:
        raise RuntimeError(
            f"spatial var_y checkpoint mismatch for {source}: "
            f"{recorded_checkpoint} != {expected_checkpoint}"
        )
    stat = expected_checkpoint.stat()
    if int(identity.get("size", -1)) != int(stat.st_size):
        raise RuntimeError(
            f"spatial var_y checkpoint size changed for {source}: "
            f"{expected_checkpoint}"
        )
    if int(identity.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
        raise RuntimeError(
            f"spatial var_y checkpoint mtime changed for {source}: "
            f"{expected_checkpoint}"
        )


def load_spatial_var_y_cache(
    cache_path,
    *,
    expected_sources: Sequence[str],
    expected_spatial_shape: Sequence[int],
    expected_channels: int,
    expected_checkpoints: Mapping[str, Path],
    coordinate_scale: torch.Tensor,
    channelwise_fallback: torch.Tensor | None = None,
    variance_floor: float = 1e-8,
    expected_period: str = "2025-01-01T00--2025-03-31T23",
) -> tuple[torch.Tensor, dict]:
    """Load centered spatial Encoder variance for selected source rows.

    The returned tensor follows the requested source order and has shape
    ``(sources, H_lat*W_lat, C_lat)`` in the effective diffusion coordinate.
    Raw MSE/ordering caches are deliberately rejected. A fixed spatial
    position needs at least two covered calibration samples for a centered variance;
    positions below that threshold use the supplied calibrated channelwise variance
    in the effective diffusion coordinate.
    """
    cache_path = Path(cache_path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"spatial var_y cache is missing: {cache_path}")
    payload = safe_load(cache_path, map_location="cpu")
    checks = {
        "kind": SPATIAL_VAR_Y_KIND,
        "period": expected_period,
        "time_scope": "hourly",
        "mask_frac": 0.0,
        "ordering_ready": False,
    }
    _check_calibration_period(payload, expected_period)
    for field, expected in checks.items():
        actual = payload.get(field)
        if actual != expected:
            raise RuntimeError(
                f"spatial var_y {field} mismatch in {cache_path}: "
                f"{actual!r} != {expected!r}"
            )

    cache_sources = tuple(
        str(value).lower() for value in payload.get("sources", ())
    )
    if not cache_sources or len(set(cache_sources)) != len(cache_sources):
        raise RuntimeError("spatial variance source order must be nonempty and unique")
    requested_sources = tuple(str(value).lower() for value in expected_sources)
    if not requested_sources or len(set(requested_sources)) != len(requested_sources):
        raise ValueError("expected_sources must be non-empty and duplicate-free")
    missing = [source for source in requested_sources if source not in cache_sources]
    if missing:
        raise RuntimeError(f"spatial var_y cache is missing sources: {missing}")

    latent_shape = tuple(int(value) for value in payload.get("latent_shape", ()))
    spatial_shape = tuple(int(value) for value in expected_spatial_shape)
    expected_channels = int(expected_channels)
    if len(spatial_shape) == 3:
        if spatial_shape[-1] != expected_channels:
            raise ValueError("expected_spatial_shape channel count is inconsistent")
        spatial_shape = spatial_shape[:2]
    if len(spatial_shape) != 2:
        raise ValueError("expected_spatial_shape must be (H,W) or (H,W,C)")
    if latent_shape != (*spatial_shape, expected_channels):
        raise RuntimeError(
            f"spatial var_y latent shape {latent_shape} != "
            f"{(*spatial_shape, expected_channels)}"
        )

    variance = torch.as_tensor(payload.get("variance"), dtype=torch.float32)
    counts = torch.as_tensor(payload.get("counts"), dtype=torch.int64)
    expected_variance_shape = (len(cache_sources), *latent_shape)
    expected_count_shape = (len(cache_sources), *spatial_shape)
    if tuple(variance.shape) != expected_variance_shape:
        raise RuntimeError(
            f"spatial variance shape {tuple(variance.shape)} != "
            f"{expected_variance_shape}"
        )
    if tuple(counts.shape) != expected_count_shape:
        raise RuntimeError(
            f"spatial count shape {tuple(counts.shape)} != {expected_count_shape}"
        )

    source_indices = torch.tensor(
        [cache_sources.index(source) for source in requested_sources],
        dtype=torch.long,
    )
    variance = variance.index_select(0, source_indices)
    counts = counts.index_select(0, source_indices)
    if bool((counts < 0).any()):
        raise RuntimeError("spatial var_y counts contain negative values")
    supported_positions = counts >= SPATIAL_VAR_Y_MINIMUM_COUNT
    supported_scalars = supported_positions.unsqueeze(-1).expand_as(variance)
    supported_variance = variance[supported_scalars]
    if not bool(torch.isfinite(supported_variance).all()) or bool(
        (supported_variance < 0).any()
    ):
        raise RuntimeError(
            "supported spatial var_y centered variance contains non-finite "
            "or negative values"
        )
    if bool((~supported_positions).any()) and channelwise_fallback is None:
        affected = {
            source: int(
                (counts[index] < SPATIAL_VAR_Y_MINIMUM_COUNT).sum()
            )
            for index, source in enumerate(requested_sources)
            if bool(
                (counts[index] < SPATIAL_VAR_Y_MINIMUM_COUNT).any()
            )
        }
        raise RuntimeError(
            "spatial var_y requires at least two calibration samples at every selected "
            "token unless calibrated channelwise fallback is supplied; "
            f"unsupported positions={affected}"
        )

    fallback = None
    if channelwise_fallback is not None:
        fallback = torch.as_tensor(
            channelwise_fallback, dtype=torch.float32
        )
        expected_fallback_shape = (len(requested_sources), expected_channels)
        if tuple(fallback.shape) != expected_fallback_shape:
            raise ValueError(
                "channelwise spatial-var_y fallback shape "
                f"{tuple(fallback.shape)} != {expected_fallback_shape}"
            )
        if not bool(torch.isfinite(fallback).all()) or bool(
            (fallback < 0).any()
        ):
            raise ValueError(
                "channelwise spatial-var_y fallback must be finite and "
                "non-negative"
            )

    source_metadata = payload.get("source_metadata")
    if not isinstance(source_metadata, Mapping):
        raise RuntimeError("spatial var_y cache has no source_metadata mapping")
    for source in requested_sources:
        metadata = source_metadata.get(source)
        if not isinstance(metadata, Mapping):
            raise RuntimeError(
                f"spatial var_y cache has no metadata for source {source}"
            )
        checkpoint = expected_checkpoints.get(source)
        if checkpoint is None:
            raise ValueError(f"expected checkpoint is missing for source {source}")
        _validate_spatial_var_y_checkpoint(
            cache_path,
            source,
            metadata,
            Path(checkpoint),
        )

    coordinate_scale = torch.as_tensor(
        coordinate_scale, dtype=torch.float32
    ).reshape(-1)
    if coordinate_scale.shape != (expected_channels,):
        raise ValueError(
            "coordinate_scale must have one value per ERA latent channel"
        )
    if not bool(torch.isfinite(coordinate_scale).all()) or bool(
        (coordinate_scale <= 0).any()
    ):
        raise ValueError("coordinate_scale must be finite and positive")
    variance = variance * coordinate_scale.square().view(1, 1, 1, -1)
    if bool((~supported_positions).any()):
        fallback_grid = fallback.view(
            len(requested_sources), 1, 1, expected_channels
        )
        variance = torch.where(
            supported_positions.unsqueeze(-1), variance, fallback_grid
        )
    if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise RuntimeError(
            "effective spatial var_y contains non-finite or negative values"
        )

    variance_floor = float(variance_floor)
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    return (
        variance.clamp_min(variance_floor).reshape(
            len(requested_sources), math.prod(spatial_shape), expected_channels
        ),
        payload,
    )


def apply_var_y_mode(
    var_y: torch.Tensor,
    mode: str,
    constant: float | None,
    variance_floor: float,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return the effective MMPS covariance without changing its channel shape."""
    values = torch.as_tensor(var_y)
    if values.numel() == 0 or not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("loaded var_y must be non-empty, finite, and positive")
    variance_floor = float(variance_floor)
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("var_y_scale must be finite and positive")
    mode = str(mode).lower()
    if mode in {"channelwise", "spatial"}:
        effective = values
    elif mode == "source_mean":
        effective = torch.full_like(values, values.mean())
    elif mode != "constant":
        raise ValueError("var_y_mode must be channelwise, spatial, source_mean, or constant")
    else:
        if constant is None:
            raise ValueError("var_y_constant is required when var_y_mode=constant")
        constant = float(constant)
        if not math.isfinite(constant) or constant <= 0:
            raise ValueError("var_y_constant must be finite and positive")
        if constant < variance_floor:
            raise ValueError(
                f"var_y_constant={constant} is below variance_floor={variance_floor}"
            )
        effective = torch.full_like(values, constant)
    return (effective * scale).clamp_min(variance_floor)


def _validate_recorded_file(identity: Mapping, label: str) -> Path:
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"guidance reliability is missing {label} identity")
    path = Path(identity.get("path", "")).expanduser().resolve()
    actual = file_identity(path)
    expected = {
        "path": str(path),
        "size": int(identity.get("size", -1)),
        "mtime_ns": int(identity.get("mtime_ns", -1)),
    }
    if actual != expected:
        raise RuntimeError(
            f"guidance reliability {label} changed after calibration: " f"{actual} != {expected}"
        )
    return path


def load_channel_ppgf_ordering_cache(
    cache_path,
    expected_sources: Sequence[str],
    expected_channels: int,
    expected_spatial_shape: Sequence[int] | None = None,
    expected_period: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Load channel-only or spatial-channel ordering values for PPGF."""
    cache_path = Path(cache_path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"Channel PPGF ordering cache is missing: {cache_path}")
    payload = safe_load(cache_path, map_location="cpu")
    if expected_period is not None:
        _check_calibration_period(payload, expected_period)
    if expected_period is not None:
        if payload.get("source_statistic") != "mse" or payload.get("includes_bias") is not True:
            raise RuntimeError("Reliability must use residual MSE including bias")
        if not torch.equal(torch.as_tensor(payload["variance"]), torch.as_tensor(payload["mse"])):
            raise RuntimeError("Ordering transport field must equal MSE")
    kind = payload.get("kind")
    if kind not in {CHANNEL_PPGF_MATRIX_KIND, SPATIAL_PPGF_MATRIX_KIND}:
        raise RuntimeError(
            f"Channel PPGF ordering cache kind mismatch in {cache_path}: "
            f"{kind!r} is not one of "
            f"{(CHANNEL_PPGF_MATRIX_KIND, SPATIAL_PPGF_MATRIX_KIND)!r}"
        )

    expected_sources = tuple(str(source).lower() for source in expected_sources)
    cache_sources = tuple(str(source).lower() for source in payload.get("sources", ()))
    if cache_sources != expected_sources:
        raise RuntimeError(
            f"Channel PPGF ordering source mismatch in {cache_path}: "
            f"{cache_sources} != {expected_sources}"
        )
    source_major = torch.as_tensor(payload.get("variance"), dtype=torch.float32)
    spatial_shape = None
    if kind == CHANNEL_PPGF_MATRIX_KIND:
        expected_shape = (len(expected_sources), int(expected_channels))
        ordering_granularity = "channel"
        vector_reduction = "geometric_mean_over_latent_channels"
    else:
        if expected_spatial_shape is None:
            raise RuntimeError(
                "spatial Channel PPGF cache requires the expected latent spatial shape"
            )
        spatial_shape = tuple(int(value) for value in expected_spatial_shape)
        if (
            len(spatial_shape) != 3
            or spatial_shape[-1] != int(expected_channels)
            or any(value <= 0 for value in spatial_shape)
        ):
            raise RuntimeError(f"invalid expected spatial latent shape: {spatial_shape}")
        cache_spatial_shape = tuple(int(value) for value in payload.get("latent_shape", ()))
        if cache_spatial_shape != spatial_shape:
            raise RuntimeError(
                f"spatial Channel PPGF latent shape mismatch in {cache_path}: "
                f"{cache_spatial_shape} != {spatial_shape}"
            )
        expected_shape = (len(expected_sources), *spatial_shape)
        ordering_granularity = "spatial_channel"
        vector_reduction = "geometric_mean_over_spatial_positions_and_latent_channels"
        if payload.get("ordering_ready") is not True:
            raise RuntimeError(f"spatial Channel PPGF cache is not ordering-ready: {cache_path}")
    if source_major.shape != expected_shape:
        raise RuntimeError(
            f"Channel PPGF ordering shape mismatch in {cache_path}: "
            f"{tuple(source_major.shape)} != {expected_shape}"
        )
    flattened_source_major = source_major.reshape(len(expected_sources), -1)
    if not torch.isfinite(source_major).all() or (source_major <= 0).any():
        raise RuntimeError(f"Channel PPGF ordering cache contains invalid values: {cache_path}")
    if "ordering_only" in payload and payload["ordering_only"] is not True:
        raise RuntimeError(f"Channel PPGF cache is not marked ordering-only: {cache_path}")

    spatial_vector_values = None
    spatial_vector_reduction = None
    if ordering_granularity == "spatial_channel":
        spatial_vector_values = (
            source_major.clamp_min(1e-12)
            .log()
            .mean(dim=-1)
            .exp()
            .reshape(len(expected_sources), -1)
            .transpose(0, 1)
            .contiguous()
        )
        spatial_vector_reduction = "geometric_mean_over_latent_channels_per_pixel"

    selected_times = payload.get("selected_times")
    vector_values = flattened_source_major.clamp_min(1e-12).log().mean(dim=1).exp()
    metadata = {
        "path": cache_path,
        "kind": payload["kind"],
        "source_order": list(cache_sources),
        "cache_mask_frac": (
            None if payload.get("mask_frac") is None else float(payload["mask_frac"])
        ),
        "ordering_metric": str(payload.get("ordering_metric", "centered_residual_variance")),
        "calibration_kind": payload.get("calibration_kind"),
        "period": payload.get("period"),
        "time_scope": payload.get("time_scope"),
        "expected_hours": payload.get("expected_hours"),
        "num_selected_times": (
            len(selected_times) if isinstance(selected_times, (list, tuple)) else None
        ),
        "ordering_granularity": ordering_granularity,
        "spatial_shape": spatial_shape,
        "order_size": int(flattened_source_major.shape[1]),
        "minimum_count": payload.get("minimum_count"),
        "fill_value": payload.get("fill_value"),
        "missing_value_policy": payload.get("missing_value_policy"),
        "filled_position_count": payload.get("filled_position_count"),
        "filled_scalar_count": payload.get("filled_scalar_count"),
        "filled_position_count_by_source": payload.get("filled_position_count_by_source"),
        "zero_count_position_count": payload.get("zero_count_position_count"),
        "single_count_position_count": payload.get("single_count_position_count"),
        "zero_count_position_count_by_source": payload.get("zero_count_position_count_by_source"),
        "single_count_position_count_by_source": payload.get(
            "single_count_position_count_by_source"
        ),
        "raw_cache": payload.get("raw_cache"),
        "random_seed": payload.get("random_seed"),
        "randomization_granularity": payload.get("randomization_granularity"),
        "priority_semantics": payload.get("priority_semantics"),
        "reference_cache": payload.get("reference_cache"),
        "analysis_calibration": payload.get("analysis_calibration"),
        "encoder_ordering_provenance": payload.get("encoder_ordering_provenance"),
        "channel": flattened_source_major.transpose(0, 1).contiguous(),
        "vector": vector_values.contiguous(),
        "vector_reduction": vector_reduction,
        "spatial_vector": spatial_vector_values,
        "spatial_vector_reduction": spatial_vector_reduction,
        "spatial_vector_order_size": (
            None
            if spatial_vector_values is None
            else int(spatial_vector_values.shape[0])
        ),
    }
    return metadata["channel"], metadata


def finalize_weighted_rmse(
    squared_error_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    physical_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish split-wide RMSE only after all frames/ranks have been summed."""
    squared_error_sum = torch.as_tensor(squared_error_sum, dtype=torch.float64)
    weight_sum = torch.as_tensor(weight_sum, dtype=torch.float64)
    physical_std = torch.as_tensor(physical_std, dtype=torch.float64)
    if not (squared_error_sum.shape == weight_sum.shape == physical_std.shape):
        raise ValueError("RMSE accumulator tensors must share a shape")
    if (weight_sum <= 0).any():
        raise RuntimeError("RMSE accumulator contains an empty channel")
    normalized = torch.sqrt(squared_error_sum / weight_sum)
    return normalized, normalized * physical_std


def _atomic_torch_save(payload, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(text: str, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contract_hash(payload: Mapping) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_latent_coordinates(latent_dir: Path, denoiser_config: DictConfig, device):
    autoencoder_config = OmegaConf.load(latent_dir / "ae" / "config.yaml")
    ae = autoencoder_config.ae
    stats = safe_load(latent_dir / "stats.pth", map_location="cpu")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(1, 1, -1)
    raw_std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(1, 1, -1)
    if not torch.isfinite(mean).all():
        raise ValueError("latent mean contains non-finite values")
    if not torch.isfinite(raw_std).all() or (raw_std <= 0).any():
        raise ValueError("latent std must be finite and positive")
    inject_noise = bool(denoiser_config.train.get("inject_ae_noise", False))
    noise_level = float(ae.get("noise_level", 0.0)) if inject_noise else 0.0
    effective_std = torch.sqrt(raw_std.square() + noise_level**2)
    return {
        "ae_file_config": autoencoder_config,
        "ae": ae,
        "mean": mean.to(device),
        "raw_std": raw_std.to(device),
        "effective_std": effective_std.to(device),
        "noise_level": noise_level,
    }


def _validate_convi_qc_config(convi_config) -> None:
    variant = str(convi_config.get("observation_variant", "")).lower()
    if variant != CONVI_OBSERVATION_VARIANT:
        raise RuntimeError(
            "CONVI guidance/evaluation requires convi.observation_variant=qc; " f"got {variant!r}"
        )


def _resolve_convi_qc_directory(value, field: str) -> Path:
    path = Path(value).expanduser().resolve()
    expected_name = CONVI_QC_DIRECTORY_NAMES[field]
    if path.name != expected_name:
        raise RuntimeError(
            f"CONVI {field} must select the QC directory {expected_name!r}; got {path}"
        )
    return path


def _build_convi_dataset(data_config, convi_config) -> CONVIOBS:
    if convi_config is None:
        raise RuntimeError("CONVI guidance requires a local CONVI configuration")
    _validate_convi_qc_config(convi_config)
    surface_only = bool(convi_config.get("surface_only", False))
    if str(getattr(data_config, "kind", "")).lower() == "convi":
        model_surface_only = getattr(data_config, "icoa_dir", None) is None
        if surface_only is not model_surface_only:
            raise RuntimeError(
                "CONVI guidance scope must match the trained Encoder: "
                f"surface_only={surface_only}, model_surface_only="
                f"{model_surface_only}"
            )
    surf_dir = _resolve_convi_qc_directory(convi_config.surf_dir, "surf_dir")
    icoa_dir = (
        None
        if surface_only
        else _resolve_convi_qc_directory(convi_config.icoa_dir, "icoa_dir")
    )
    mean_std_dir = Path(convi_config.mean_std_dir).expanduser().resolve()
    dataset = CONVIOBS(
        surf_dir=surf_dir,
        icoa_dir=icoa_dir,
        mean_std_dir=mean_std_dir,
        skip_temp=True,
        skip_humidity=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("CONVI dataset contains no observation timestamps")
    dataset.observation_variant = CONVI_OBSERVATION_VARIANT
    dataset.observation_data = {
        "variant": CONVI_OBSERVATION_VARIANT,
        "scope": "surf_only" if surface_only else "merged_surf_icoads",
        "surf_dir": str(surf_dir),
        "icoa_dir": str(icoa_dir) if icoa_dir is not None else None,
        "mean_std_dir": str(mean_std_dir),
    }
    return dataset


def _time_index(times) -> dict[tuple[int, int, int, int], int]:
    return {timestamp_key(value): index for index, value in enumerate(times)}


def _load_era_channel_stat(path, label: str) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if values.ndim != 1 or values.size < len(CHANNEL_NAMES):
        raise ValueError(
            f"{label} must contain at least {len(CHANNEL_NAMES)} channels; "
            f"got shape {values.shape}"
        )
    values = values[: len(CHANNEL_NAMES)]
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values: {path}")
    return values


def _configure_convi_era5_truth_values(
    dataset: CONVIOBS,
    convi_config,
    latent_coordinates,
) -> dict | None:
    value_source = str(
        convi_config.get("guidance_value_source", "observation")
    ).lower()
    if value_source not in CONVI_GUIDANCE_VALUE_SOURCES:
        raise ValueError(
            "convi.guidance_value_source must be observation, era5_truth, or "
            "era5_truth_full_mask; "
            f"got {value_source!r}"
        )
    if value_source == "observation":
        return None
    full_surface_mask = value_source == "era5_truth_full_mask"

    ae_config = latent_coordinates["ae_file_config"]
    normalization = ae_config.normalization
    data_path = Path(ae_config.data_path).expanduser().resolve()
    source_mean_path = Path(normalization.source_mean_path).expanduser().resolve()
    source_std_path = Path(normalization.source_std_path).expanduser().resolve()
    target_mean_path = Path(normalization.target_mean_path).expanduser().resolve()
    target_std_path = Path(normalization.target_std_path).expanduser().resolve()
    for label, path in (
        ("ERA5 data root", data_path),
        ("ERA5 source mean", source_mean_path),
        ("ERA5 source std", source_std_path),
        ("ERA5 target mean", target_mean_path),
        ("ERA5 target std", target_std_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    source_mean = _load_era_channel_stat(source_mean_path, "ERA5 source mean")
    source_std = _load_era_channel_stat(source_std_path, "ERA5 source std")
    target_mean = _load_era_channel_stat(target_mean_path, "ERA5 target mean")
    target_std = _load_era_channel_stat(target_std_path, "ERA5 target std")
    if (source_std <= 0).any() or (target_std <= 0).any():
        raise ValueError("ERA5 source/target standard deviations must be positive")

    convi_mean = np.asarray(dataset.mean, dtype=np.float32).reshape(-1)
    convi_std = np.asarray(dataset.std, dtype=np.float32).reshape(-1)
    if not np.array_equal(target_mean, convi_mean) or not np.array_equal(
        target_std, convi_std
    ):
        raise RuntimeError(
            "ERA5 target mean/std do not exactly match the CONVI encoder "
            "mean_era5.npy/std_era5.npy basis"
        )

    surface = slice(CONVIOBS.NUM_TEMP_CHANNELS, CONVIOBS.NUM_ERA_CHANNELS)
    return {
        "data_path": data_path,
        "source_mean_path": source_mean_path,
        "source_std_path": source_std_path,
        "target_mean_path": target_mean_path,
        "target_std_path": target_std_path,
        "scale": (source_std[surface] / target_std[surface]).reshape(4, 1, 1),
        "offset": (
            (source_mean[surface] - target_mean[surface]) / target_std[surface]
        ).reshape(4, 1, 1),
        "full_surface_mask": full_surface_mask,
    }


def _load_convi_era5_truth_surface(source: Mapping, timestamp: datetime):
    truth = source.get("convi_era5_truth_values")
    if truth is None:
        return None
    path = (
        truth["data_path"]
        / f"{timestamp.year:04d}"
        / f"{timestamp.month:02d}"
        / f"{timestamp:%Y%m%d_%H}.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(f"exact-hour ERA5 guidance frame is missing: {path}")
    stored = np.load(path, mmap_mode="r")
    expected_shape = (len(CHANNEL_NAMES), 720, 1440)
    if stored.shape != expected_shape:
        raise RuntimeError(
            f"ERA5 guidance frame has shape {stored.shape}, expected {expected_shape}: {path}"
        )
    surface = np.asarray(
        stored[CONVIOBS.NUM_TEMP_CHANNELS : CONVIOBS.NUM_ERA_CHANNELS],
        dtype=np.float32,
    )
    normalized = surface * truth["scale"] + truth["offset"]
    if not np.isfinite(normalized).all():
        raise RuntimeError(f"ERA5 guidance frame produced non-finite values: {path}")
    return normalized


def _load_encoder_source(
    name: str,
    source_config,
    latent_dir: Path,
    latent_shape: tuple[int, int],
    latent_coordinates,
    device,
    variance_floor: float,
    convi_config=None,
) -> dict:
    model_dir = Path(source_config.model_dir).expanduser().resolve()
    model_config = OmegaConf.load(model_dir / "config.yaml")
    if not _same_path(model_config.era_latent_dir, latent_dir):
        raise RuntimeError(
            f"{name} Encoder was trained for {model_config.era_latent_dir}, not {latent_dir}"
        )
    data_config = model_config.data
    data_kind = str(data_config.get("kind", "sate")).lower()
    convi_era5_truth_values = None
    if data_kind == "sate":
        dataset = SatelliteAEDataset(
            data_path=data_config.sate_data_path,
            sate_name=str(data_config.sate_name),
            standard_dir=data_config.sate_standard_dir,
            blanket_size=1,
            time_range=None,
            use_satzen=bool(data_config.get("use_satzen", False)),
        )
        actual_name = str(data_config.sate_name).lower()
        in_channels = int(dataset.get_dim())
        kind = "obs2era"
        time_to_idx = _time_index(dataset.times)
    elif data_kind in {"convi", "era_surface_dense"}:
        if name != "convi":
            raise ValueError(
                f"data.kind={data_kind!r} can only serve the CONVI source"
            )
        dataset = _build_convi_dataset(data_config, convi_config)
        actual_name = "convi"
        in_channels = CONVI_OBS_ENCODER_CHANNELS
        kind = "convi_obs_encoder"
        time_to_idx = None
        convi_era5_truth_values = _configure_convi_era5_truth_values(
            dataset,
            convi_config,
            latent_coordinates,
        )
    else:
        raise ValueError(f"unsupported Encoder data.kind for {name}: {data_kind!r}")
    if actual_name != name:
        raise RuntimeError(
            f"source catalog name {name!r} disagrees with model source {actual_name!r}"
        )

    checkpoint = str(source_config.get("checkpoint", "best"))
    checkpoint_path = _checkpoint_path(model_dir, checkpoint)
    encoder = load_obs2era_encoder(
        model_dir,
        latent_coordinates["ae"],
        in_channels,
        checkpoint,
        device,
    )
    channels = latent_shape[-1]
    coordinate_scale = (
        latent_coordinates["raw_std"].cpu() / latent_coordinates["effective_std"].cpu()
    ).reshape(-1)
    var_y, _ = load_calibrated_var_y_cache(
        source_config.var_y_cache,
        expected_representation="encoder",
        expected_source=name,
        expected_channels=channels,
        expected_checkpoint=checkpoint_path,
        coordinate_scale=coordinate_scale,
        expected_period=str(source_config.calibration_period),
        variance_floor=variance_floor,
    )
    source = {
        "name": name,
        "kind": kind,
        "model_dir": model_dir,
        "checkpoint_path": checkpoint_path,
        "var_y_cache": Path(source_config.var_y_cache).expanduser().resolve(),
        "reliability_cache": _optional_resolved_path(
            source_config,
            "reliability_cache",
        ),
        "var_y": var_y.to(device),
        "encoder": encoder,
        "dataset": dataset,
        "time_to_idx": time_to_idx,
        "encoder_model_data_kind": data_kind,
    }
    if kind == "convi_obs_encoder" and convi_era5_truth_values is not None:
        source["convi_era5_truth_values"] = convi_era5_truth_values
    return source


def _load_decoder_source(
    name: str,
    source_config,
    latent_dir: Path,
    latent_shape: tuple[int, int],
    device,
    variance_floor: float,
) -> dict:
    model_dir = Path(source_config.model_dir).expanduser().resolve()
    checkpoint = str(source_config.get("checkpoint", "best"))
    source = load_direct_e2o_source(
        model_dir,
        checkpoint,
        latent_dir,
        latent_shape,
        device,
    )
    actual_name = str(source["sate_dataset"].sate_name).lower()
    if actual_name != name:
        raise RuntimeError(f"source catalog name {name!r} disagrees with decoder {actual_name!r}")
    var_y, _ = load_calibrated_var_y_cache(
        source_config.var_y_cache,
        expected_representation="decoder",
        expected_source=name,
        expected_channels=source["sate_dataset"].obs_channel,
        expected_checkpoint=source["checkpoint_path"],
        variance_floor=variance_floor,
        expected_period=str(source_config.calibration_period),
    )
    source.update({
        "name": name,
        "kind": "direct_e2o",
        "var_y_cache": Path(source_config.var_y_cache).expanduser().resolve(),
        "reliability_cache": _optional_resolved_path(
            source_config,
            "reliability_cache",
        ),
        "var_y": var_y.to(device),
    })
    return source


def resolve_source_route(config: DictConfig, name: str) -> str:
    """Keep the satellite route while optionally selecting physical CONVI."""
    route = str(config.route).lower()
    space = config.get("convi", {}).get("guidance_space")
    if space is not None and str(space).lower() not in {"encoder", "physical"}:
        raise ValueError("convi.guidance_space must be null, encoder, or physical")
    if name != "convi" or space is None:
        return route
    if route == "no_obs":
        raise ValueError("CONVI guidance cannot be selected with route=no_obs")
    selected = "decoder" if str(space).lower() == "physical" else "encoder"
    merge = str(config.guidance_merge).lower()
    calibrated_ppgf = (
        merge == "channel_ppgf"
        and str(config.get("ppgf_ordering_source", "var_y")) == "matrix_cache"
    )
    if selected != route and merge != "sum" and not calibrated_ppgf:
        raise ValueError(
            "Mixed CONVI guidance requires guidance_merge=sum or "
            "channel_ppgf with an analysis-calibrated matrix_cache"
        )
    return selected


def source_variance_options(config: DictConfig, kind: str) -> tuple[str, float | None]:
    mode = str(config.get("var_y_mode", "channelwise")).lower()
    constant = config.get("var_y_constant")
    if kind == "convi_physical":
        convi = config.get("convi", {})
        mode = str(convi.get("var_y_mode") or mode).lower()
        if convi.get("var_y_constant") is not None:
            constant = convi.var_y_constant
        if mode not in {"constant", "channelwise"}:
            raise ValueError("Physical CONVI var_y_mode must be constant or channelwise")
    return mode, constant


def load_convi_physical_var_y(cache_path, latent_dir, dataset, normalization, *,
                            variance_floor=1e-8, strict=False,
                            expected_period="2025-01-01T00--2025-03-31T23"):
    """Load four physical variances, with the same QC normalization as guidance."""
    values, payload = load_calibrated_var_y_cache(
        cache_path,
        expected_representation="decoder",
        expected_source="convi",
        expected_channels=4,
        expected_checkpoint=Path(latent_dir) / "ae" / "model.pth",
        variance_floor=variance_floor,
        expected_period=expected_period,
    )
    for field, expected in {
        "space": "normalized_era_surface",
        "decoder_kind": "frozen_era_autoencoder_physical",
        "era_channel_indices": [65, 66, 67, 68],
        "surface_channel_names": list(SURFACE_VARIABLES),
    }.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"Physical CONVI cache {field} mismatch")
    if strict:
        if payload.get("convi_observation_data") != dataset.observation_data:
            raise RuntimeError("Physical CONVI cache uses different QC observation data")
        if payload.get("era_normalization") != normalization:
            raise RuntimeError("Physical CONVI cache uses different ERA normalization")

    return values, payload


def _load_convi_physical_source(
    name: str,
    source_config,
    latent_dir: Path,
    era_autoencoder,
    device,
    variance_floor: float,
    convi_config,
    var_y_mode="channelwise",
    latent_coordinates=None,
) -> dict:
    value_source = str(
        convi_config.get("guidance_value_source", "observation")
    ).lower()
    if value_source not in {"observation", "era5_truth"}:
        raise ValueError(
            "physical CONVI supports observation or era5_truth at QC masks"
        )
    model_dir = Path(source_config.data_model_dir).expanduser().resolve()
    model_config = OmegaConf.load(model_dir / "config.yaml")
    if not _same_path(model_config.era_latent_dir, latent_dir):
        raise RuntimeError("CONVI data model and diffusion use different ERA latent spaces")
    dataset = _build_convi_dataset(model_config.data, convi_config)
    checkpoint_path = latent_dir / "ae" / "model.pth"
    normalization = physical_normalization_contract(
        latent_dir / "ae" / "config.yaml", convi_config.mean_std_dir
    )
    # Constant physical variance needs neither a CONVI Encoder nor a channelwise cache.
    cache_path = None
    if var_y_mode == "constant":
        var_y = torch.ones(4)
    else:
        cache_path = Path(source_config.var_y_cache).expanduser().resolve()
        var_y, _ = load_convi_physical_var_y(
            cache_path, latent_dir, dataset, normalization,
            variance_floor=variance_floor,
            strict=False,
            expected_period=str(source_config.calibration_period),
        )
    source = {
        "name": name,
        "kind": "convi_physical",
        "model_dir": model_dir,
        "checkpoint_path": checkpoint_path,
        "var_y_cache": cache_path,
        "reliability_cache": _optional_resolved_path(
            source_config,
            "reliability_cache",
        ),
        "var_y": var_y.to(device),
        "dataset": dataset,
        "era_autoencoder": era_autoencoder,
        "era_normalization": normalization,
    }
    if value_source == "era5_truth":
        if latent_coordinates is None:
            raise ValueError("physical ERA5 truth requires latent normalization coordinates")
        source["convi_era5_truth_values"] = _configure_convi_era5_truth_values(
            dataset, convi_config, latent_coordinates,
        )
    return source


def _validate_denoiser_protocol(denoiser_config: DictConfig) -> None:
    expected = {
        "blanket_size": WINDOW_SIZE,
        "blanket_dt": 1,
        "anchor_frame": ANCHOR_FRAME_NUMBER,
        "anchor_hour_interval": ANCHOR_INTERVAL_HOURS,
    }
    for field, expected_value in expected.items():
        actual = denoiser_config.train.get(field, None)
        if actual is None or int(actual) != expected_value:
            raise RuntimeError(
                f"diffusion train.{field} must be {expected_value} for this protocol; got {actual}"
            )


def configure_source_uncertainties(
    config: DictConfig,
    sources: Sequence[dict],
    route: str,
    latent_channels: int,
    effective_std: torch.Tensor,
    spatial_shape: Sequence[int] | None = None,
    coordinate_scale: torch.Tensor | None = None,
) -> dict | None:
    """Apply the MMPS covariance mode and attach optional PPGF reliability."""
    var_y_mode = str(config.get("var_y_mode", "channelwise")).lower()
    var_y_scale = float(config.get("var_y_scale", 1.0))
    if not math.isfinite(var_y_scale) or var_y_scale <= 0:
        raise ValueError("var_y_scale must be finite and positive")
    variance_floor = float(config.variance_floor)
    merge = str(config.guidance_merge).lower()
    if var_y_mode == "spatial":
        if str(route).lower() != "encoder":
            raise ValueError("var_y_mode=spatial supports only the Encoder route")
        if merge not in {
            "sum",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
        }:
            raise ValueError(
                "var_y_mode=spatial requires guidance_merge=sum, "
                "channel_ppgf, channel_ppgf_accumulate, or channel_ppgf_top1"
            )
        if float(config.mask_frac) != 0.0:
            raise ValueError("var_y_mode=spatial requires mask_frac=0")
        if spatial_shape is None or coordinate_scale is None:
            raise ValueError(
                "var_y_mode=spatial requires latent spatial shape and "
                "coordinate scale"
            )
        cache_value = config.get("spatial_var_y_cache", None)
        if cache_value is None:
            raise ValueError(
                "spatial_var_y_cache is required when var_y_mode=spatial"
            )
        # Physical CONVI retains its four-channel physical covariance. Only
        # Encoder sources use the ERA-latent spatial covariance cache.
        spatial_sources = [s for s in sources if s.get("kind") != "convi_physical"]
        if not spatial_sources:
            raise ValueError("spatial var_y requires at least one Encoder source")
        channelwise_fallback_rows = []
        for source in spatial_sources:
            channelwise = source.get("channelwise_var_y")
            if channelwise is None:
                channelwise = torch.as_tensor(source["var_y"]).detach().clone()
                if tuple(channelwise.shape) != (int(latent_channels),):
                    raise RuntimeError(
                        f"{source['name']} channelwise var_y shape "
                        f"{tuple(channelwise.shape)} != "
                        f"{(int(latent_channels),)}"
                    )
                source["channelwise_var_y"] = channelwise
                source["channelwise_var_y_cache"] = source["var_y_cache"]
            channelwise_fallback_rows.append(
                torch.as_tensor(channelwise).detach().cpu().float()
            )
        channelwise_fallback = torch.stack(channelwise_fallback_rows)
        cache_path = Path(cache_value).expanduser().resolve()
        spatial_values, spatial_payload = load_spatial_var_y_cache(
            cache_path,
            expected_sources=[source["name"] for source in spatial_sources],
            expected_spatial_shape=spatial_shape,
            expected_channels=latent_channels,
            expected_checkpoints={
                source["name"]: source["checkpoint_path"] for source in spatial_sources
            },
            coordinate_scale=coordinate_scale,
            channelwise_fallback=channelwise_fallback,
            variance_floor=variance_floor,
            expected_period=str(config.calibration_period),
        )
        cache_sources = tuple(
            str(value).lower()
            for value in spatial_payload.get("sources", ())
        )
        cache_counts = torch.as_tensor(
            spatial_payload["counts"], dtype=torch.int64
        )
        for index, source in enumerate(spatial_sources):
            source_name = source["name"]
            cache_source_index = cache_sources.index(source_name)
            source["var_y_cache"] = cache_path
            source["var_y"] = spatial_values[index].to(
                device=source["var_y"].device
            )
            source["spatial_var_y"] = {
                "kind": spatial_payload["kind"],
                "statistic": "centered_population_variance",
                "layout": "(H_lat*W_lat, C_lat)",
                "raw_layout": spatial_payload.get("layout"),
                "raw_source_index": cache_source_index,
                "count_min": int(cache_counts[cache_source_index].min()),
                "count_max": int(cache_counts[cache_source_index].max()),
                "minimum_spatial_count": SPATIAL_VAR_Y_MINIMUM_COUNT,
                "spatial_position_count": int(
                    (
                        cache_counts[cache_source_index]
                        >= SPATIAL_VAR_Y_MINIMUM_COUNT
                    ).sum()
                ),
                "channelwise_fallback_position_count": int(
                    (
                        cache_counts[cache_source_index]
                        < SPATIAL_VAR_Y_MINIMUM_COUNT
                    ).sum()
                ),
                "missing_value_policy": (
                    "count >= 2 uses fixed-position Calibrated centered variance; "
                    "count < 2 uses the same source's covered-sample "
                    "channelwise centered variance"
                ),
            }
            if config.get("calibration_period") is not None:
                source["spatial_var_y"]["missing_value_policy"] = (
                    "count >= 2 uses calibrated spatial centered variance; "
                    "count < 2 uses same-source calibrated channelwise centered variance"
                )
    for source in sources:
        source_mode, source_constant = source_variance_options(config, source.get("kind", ""))
        source["var_y"] = apply_var_y_mode(
            source["var_y"],
            source_mode,
            source_constant,
            variance_floor,
            scale=var_y_scale,
        )
        source["var_y_mode"] = source_mode
        source["var_y_constant"] = float(source_constant) if source_mode == "constant" else None
        source["var_y_scale"] = var_y_scale

    ordering_source = str(config.get("ppgf_ordering_source", "var_y")).lower()
    if ordering_source not in {"var_y", "matrix_cache"}:
        raise ValueError(
            "ppgf_ordering_source must be var_y or matrix_cache"
        )
    if (
        var_y_mode == "spatial"
        and merge in {
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
        }
        and ordering_source != "matrix_cache"
    ):
        raise ValueError(
            "spatial var_y with Channel PPGF requires "
            "ppgf_ordering_source=matrix_cache"
        )
    reliability_metric = str(config.get("ppgf_reliability_metric", "native_rmse")).lower()
    if reliability_metric not in PPGF_RELIABILITY_METRICS:
        raise ValueError(f"ppgf_reliability_metric must be one of {PPGF_RELIABILITY_METRICS}")
    if ordering_source == "matrix_cache":
        if merge not in {
            "ppgf",
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
            "spatial_vector_ppgf",
        }:
            raise ValueError(
                "matrix_cache ordering requires guidance_merge=ppgf, "
                "channel_ppgf, channel_ppgf_accumulate, channel_ppgf_top1, "
                "or spatial_vector_ppgf"
            )
        cache_path = config.get("ppgf_ordering_cache", None)
        if cache_path is None:
            raise ValueError("ppgf_ordering_cache is required for matrix_cache ordering")
        _, cache = load_channel_ppgf_ordering_cache(
            cache_path,
            expected_sources=[source["name"] for source in sources],
            expected_channels=latent_channels,
            expected_spatial_shape=spatial_shape,
            expected_period=config.get("calibration_period"),
        )
        if cache["ordering_granularity"] == "spatial_channel":
            if merge not in {
                "channel_ppgf",
                "channel_ppgf_accumulate",
                "channel_ppgf_top1",
                "spatial_vector_ppgf",
            }:
                raise ValueError(
                    "spatial ordering cache requires guidance_merge=channel_ppgf, "
                    "channel_ppgf_accumulate, channel_ppgf_top1, or "
                    "spatial_vector_ppgf"
                )
            order_mode = str(config.ppgf_order_mode).lower()
            if order_mode not in {"fixed", "gumbel"}:
                raise ValueError(
                    "spatial ordering cache requires " "ppgf_order_mode=fixed or gumbel"
                )
            if bool(config.ppgf_overlap_only):
                raise ValueError("spatial ordering cache requires ppgf_overlap_only=false")
        elif merge == "spatial_vector_ppgf":
            raise ValueError(
                "spatial_vector_ppgf requires a spatial-channel ordering cache"
            )
        return cache
    if merge == "spatial_vector_ppgf":
        raise ValueError("spatial_vector_ppgf requires ppgf_ordering_source=matrix_cache")
    return None


def configure_pcgrad_ordering_cache(
    config: DictConfig,
    sources: Sequence[dict],
    latent_channels: int,
    spatial_shape: Sequence[int],
) -> dict | None:
    """Load per-spatial-vector ordering values used by fixed PCGrad."""
    cache_value = config.get("pcgrad_ordering_cache", None)
    if cache_value is None:
        return None
    if str(config.guidance_merge).lower() != "pcgrad":
        raise ValueError("pcgrad_ordering_cache requires guidance_merge=pcgrad")
    if str(config.pcgrad_order_mode).lower() != "fixed":
        raise ValueError("pcgrad_ordering_cache requires pcgrad_order_mode=fixed")
    if not bool(config.pcgrad_shuffle):
        raise ValueError("pcgrad_ordering_cache requires pcgrad_shuffle=true")
    if bool(config.pcgrad_overlap_only):
        raise ValueError("Spatial Fixed PCGrad requires pcgrad_overlap_only=false")

    _, cache = load_channel_ppgf_ordering_cache(
        cache_value,
        expected_sources=[source["name"] for source in sources],
        expected_channels=latent_channels,
        expected_spatial_shape=spatial_shape,
    )
    if cache["ordering_granularity"] != "spatial_channel":
        raise ValueError(
            "Spatial Fixed PCGrad requires a spatial-channel ordering cache"
        )
    if (
        cache["random_seed"] is not None
        and cache["randomization_granularity"]
        != "independent_per_H_lat_W_lat_shared_C_lat"
    ):
        raise ValueError(
            "random Spatial Fixed PCGrad requires one random order per "
            "(H_lat,W_lat), shared by all latent channels"
        )
    spatial_values = cache.get("spatial_vector")
    expected_shape = (
        math.prod(tuple(int(value) for value in spatial_shape[:-1])),
        len(sources),
    )
    if spatial_values is None or tuple(spatial_values.shape) != expected_shape:
        actual_shape = None if spatial_values is None else tuple(spatial_values.shape)
        raise RuntimeError(
            "Spatial Fixed PCGrad ordering shape mismatch: "
            f"{actual_shape} != {expected_shape}"
        )
    return cache


def configure_convi_loss_spatial_top1(
    config: DictConfig,
    sources: Sequence[dict],
    route: str,
    latent_channels: int,
    spatial_shape: Sequence[int],
) -> dict | None:
    """Keep CONVI loss scalars where fixed spatial ordering ranks it first."""
    cache_value = config.get("convi_loss_top1_ordering_cache", None)
    if cache_value is None:
        return None
    if str(route).lower() != "encoder":
        raise ValueError("CONVI top1 loss filtering supports only the Encoder route")
    if str(config.guidance_merge).lower() != "sum":
        raise ValueError("CONVI top1 loss filtering requires guidance_merge=sum")
    if str(config.get("var_y_mode", "channelwise")).lower() != "spatial":
        raise ValueError("CONVI top1 loss filtering requires var_y_mode=spatial")
    if float(config.mask_frac) != 0.0:
        raise ValueError("CONVI top1 loss filtering requires mask_frac=0")

    source_names = [str(source["name"]).lower() for source in sources]
    convi_ids = [index for index, name in enumerate(source_names) if name == "convi"]
    if len(convi_ids) != 1 or len(sources) < 2:
        raise ValueError(
            "CONVI top1 loss filtering requires one CONVI source plus other sources"
        )
    convi_id = convi_ids[0]
    if sources[convi_id]["kind"] != "convi_obs_encoder":
        raise ValueError("CONVI top1 loss filtering requires CONVI Encoder guidance")

    _, ordering_cache = load_channel_ppgf_ordering_cache(
        cache_value,
        expected_sources=source_names,
        expected_channels=latent_channels,
        expected_spatial_shape=spatial_shape,
    )
    if ordering_cache["ordering_granularity"] != "spatial_channel":
        raise ValueError(
            "CONVI top1 loss filtering requires a spatial-channel ordering cache"
        )
    spatial_shape = tuple(int(value) for value in spatial_shape)
    expected_order_size = math.prod(spatial_shape)
    if int(ordering_cache["order_size"]) != expected_order_size:
        raise RuntimeError(
            "CONVI top1 ordering width does not match the latent state: "
            f"{ordering_cache['order_size']} != {expected_order_size}"
        )

    uncertainty = ordering_cache["channel"]
    fixed_order = torch.argsort(
        uncertainty,
        dim=1,
        descending=True,
        stable=True,
    )
    top1_source = fixed_order[:, -1]
    scalar_mask = top1_source.eq(convi_id).reshape(-1, latent_channels)
    selected_count = int(scalar_mask.sum())
    if selected_count == 0:
        raise RuntimeError("CONVI is never top1 in the spatial ordering cache")
    sources[convi_id]["convi_loss_scalar_mask"] = scalar_mask

    channels_per_token = scalar_mask.sum(dim=1)
    metadata = {
        "cache": file_identity(ordering_cache["path"]),
        "kind": ordering_cache["kind"],
        "ordering_metric": ordering_cache["ordering_metric"],
        "source_order": ordering_cache["source_order"],
        "selection_rule": (
            "torch.argsort(descending=True, stable=True) final source is CONVI"
        ),
        "layout": "(H_lat,W_lat,C_lat)",
        "shape": list(spatial_shape),
        "selected_scalar_count": selected_count,
        "total_scalar_count": int(scalar_mask.numel()),
        "selected_fraction": selected_count / scalar_mask.numel(),
        "selected_channels_per_token_min": int(channels_per_token.min()),
        "selected_channels_per_token_mean": float(
            channels_per_token.float().mean()
        ),
        "selected_channels_per_token_max": int(channels_per_token.max()),
    }
    sources[convi_id]["convi_loss_spatial_top1"] = metadata
    return metadata


def _validate_fixed_channel_ppgf_reference_config(
    config: DictConfig,
    ordering_cache: Mapping | None,
    merge: str,
) -> None:
    """Validate a fixed-order merge that uses the most reliable source."""
    if str(config.guidance_merge).lower() != merge:
        return
    if str(config.ppgf_order_mode).lower() != "fixed":
        raise ValueError(f"{merge} requires ppgf_order_mode=fixed")
    if str(config.get("ppgf_ordering_source", "var_y")).lower() != "matrix_cache":
        raise ValueError(
            f"{merge} requires ppgf_ordering_source=matrix_cache"
        )
    if bool(config.ppgf_overlap_only):
        raise ValueError(f"{merge} requires ppgf_overlap_only=false")
    if str(config.get("ppgf_anneal_mode", "none")).lower() != "none":
        raise ValueError(f"{merge} requires ppgf_anneal_mode=none")
    if str(
        config.get("ppgf_uncertainty_strength_mode", "constant")
    ).lower() != "constant":
        raise ValueError(
            f"{merge} requires ppgf_uncertainty_strength_mode=constant"
        )
    if float(config.ppgf_aligned_clip) != 0.0:
        raise ValueError(f"{merge} requires ppgf_aligned_clip=0")
    if config.get("ppgf_convi_bypass_threshold", None) is not None:
        raise ValueError(f"{merge} does not support CONVI bypass")
    if ordering_cache is None:
        raise ValueError(f"{merge} requires a loaded ordering cache")
    granularity = ordering_cache.get("ordering_granularity")
    order_size = int(ordering_cache.get("order_size", -1))
    if granularity == "channel" and order_size == 128:
        return
    if (
        granularity == "spatial_channel"
        and tuple(ordering_cache.get("spatial_shape") or ()) == (23, 47, 128)
        and order_size == 23 * 47 * 128
    ):
        return
    raise ValueError(
        f"{merge} requires a global 6x128 or spatial 6x23x47x128 ordering cache"
    )


def validate_ppgf_accumulate_config(
    config: DictConfig,
    ordering_cache: Mapping | None,
) -> None:
    """Keep PPGF-Accumulate on its intentionally narrow fixed-order contract."""
    _validate_fixed_channel_ppgf_reference_config(
        config,
        ordering_cache,
        "channel_ppgf_accumulate",
    )


def validate_ppgf_top1_config(
    config: DictConfig,
    ordering_cache: Mapping | None,
) -> None:
    """Keep Top-1-only guidance on the same audited fixed-order contract."""
    _validate_fixed_channel_ppgf_reference_config(
        config,
        ordering_cache,
        "channel_ppgf_top1",
    )


def validate_ppgf_convi_bypass_config(
    config: DictConfig,
    sources: Sequence[dict],
    ordering_cache: Mapping | None,
) -> int | None:
    """Validate and normalize the optional dense-CONVI PPGF bypass threshold."""
    raw_threshold = config.get("ppgf_convi_bypass_threshold", None)
    if raw_threshold is None:
        return None
    if isinstance(raw_threshold, bool):
        raise ValueError("ppgf_convi_bypass_threshold must be a positive integer")
    try:
        numeric_threshold = float(raw_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ppgf_convi_bypass_threshold must be a positive integer"
        ) from error
    threshold = int(numeric_threshold)
    if (
        not math.isfinite(numeric_threshold)
        or numeric_threshold != threshold
        or threshold < 1
    ):
        raise ValueError("ppgf_convi_bypass_threshold must be a positive integer")

    if str(config.guidance_merge).lower() != "channel_ppgf":
        raise ValueError("CONVI PPGF bypass requires guidance_merge=channel_ppgf")
    if str(config.ppgf_order_mode).lower() != "fixed":
        raise ValueError("CONVI PPGF bypass currently requires ppgf_order_mode=fixed")
    if (
        ordering_cache is None
        or ordering_cache.get("ordering_granularity") != "spatial_channel"
    ):
        raise ValueError("CONVI PPGF bypass requires a spatial-channel ordering cache")

    convi_ids = [
        source_id
        for source_id, source in enumerate(sources)
        if source["name"] == "convi"
    ]
    if len(convi_ids) != 1 or len(sources) < 2:
        raise ValueError("CONVI PPGF bypass requires CONVI plus at least one other source")
    if sources[convi_ids[0]]["kind"] != "convi_obs_encoder":
        raise ValueError("CONVI PPGF bypass currently requires the encoder route")
    return threshold


def validate_ppgf_convi_bypass_merge_mode(
    config: DictConfig,
    threshold: int | None,
) -> str:
    """Validate the dense-region merge used by the optional CONVI bypass."""
    mode = str(
        config.get(
            "ppgf_convi_bypass_merge_mode",
            PPGF_CONVI_BYPASS_DEFAULT_MERGE_MODE,
        )
    ).lower()
    if mode not in PPGF_CONVI_BYPASS_MERGE_MODES:
        raise ValueError(
            "ppgf_convi_bypass_merge_mode must be one of "
            f"{list(PPGF_CONVI_BYPASS_MERGE_MODES)}"
        )
    if threshold is None and mode != PPGF_CONVI_BYPASS_DEFAULT_MERGE_MODE:
        raise ValueError(
            "ppgf_convi_bypass_merge_mode requires "
            "ppgf_convi_bypass_threshold"
        )
    return mode


def describe_ppgf_convi_bypass_rule(mode: str) -> str:
    """Describe the sparse/dense merge split recorded in run manifests."""
    if mode == "sum_all":
        dense_rule = "coverage>=threshold: six-source raw MMPS sum"
    else:
        dense_rule = (
            "coverage>=threshold: satellite-only PPGF + raw CONVI MMPS"
        )
    return "coverage<threshold: six-source PPGF; " + dense_rule



def load_generation_runtime(config: DictConfig, names: Sequence[str], device) -> dict:
    """Load the diffusion once per rank and every selected observation source once."""
    model_path = Path(config.model_path).expanduser().resolve()
    latent_dir = model_path.parents[1]
    denoiser_config = compose(model_path / "config.yaml")
    _validate_denoiser_protocol(denoiser_config)
    coordinates = _load_latent_coordinates(latent_dir, denoiser_config, device)
    ae_shape = tuple(int(value) for value in coordinates["ae"].latent_shape)
    latent_shape = (ae_shape[0] * ae_shape[1], ae_shape[2])
    state_size = math.prod(latent_shape)

    route = str(config.route).lower()
    convi_config = config.get("convi", None)
    physical_autoencoder = None
    source_routes = {name: resolve_source_route(config, name) for name in names}
    if "convi" in names and source_routes["convi"] == "decoder":
        physical_autoencoder = load_auto_encoder(
            latent_dir / "ae", "model", device="cpu", eval_mode=True
        )
        physical_autoencoder.eval().requires_grad_(False)
        physical_autoencoder.decoder.to(device)

    variance_floor = float(config.variance_floor)
    sources = []
    for name in names:
        source_config = config.source_catalog[name][source_routes[name]]
        kind = str(source_config.kind).lower()
        if kind in {"satellite_encoder", "convi_encoder"}:
            source = _load_encoder_source(
                name,
                source_config,
                latent_dir,
                latent_shape,
                coordinates,
                device,
                variance_floor,
                convi_config=convi_config,
            )
        elif kind == "direct_e2o":
            source = _load_decoder_source(
                name,
                source_config,
                latent_dir,
                latent_shape,
                device,
                variance_floor,
            )
        elif kind == "convi_physical":
            if physical_autoencoder is None:
                raise RuntimeError("CONVI physical guidance requires the frozen ERA decoder")
            source = _load_convi_physical_source(
                name,
                source_config,
                latent_dir,
                physical_autoencoder,
                device,
                variance_floor,
                convi_config,
                var_y_mode=source_variance_options(config, kind)[0],
                latent_coordinates=coordinates,
            )
        else:
            raise ValueError(f"unsupported source kind for {name}: {kind!r}")
        sources.append(source)

    ppgf_ordering_cache = configure_source_uncertainties(
        config,
        sources,
        route,
        latent_shape[-1],
        coordinates["effective_std"].cpu(),
        spatial_shape=ae_shape,
        coordinate_scale=(
            coordinates["raw_std"].cpu()
            / coordinates["effective_std"].cpu()
        ),
    )
    pcgrad_ordering_cache = configure_pcgrad_ordering_cache(
        config,
        sources,
        latent_shape[-1],
        ae_shape,
    )
    convi_loss_spatial_top1 = configure_convi_loss_spatial_top1(
        config,
        sources,
        route,
        latent_shape[-1],
        ae_shape,
    )
    validate_ppgf_accumulate_config(config, ppgf_ordering_cache)
    validate_ppgf_top1_config(config, ppgf_ordering_cache)
    ppgf_convi_bypass_threshold = validate_ppgf_convi_bypass_config(
        config,
        sources,
        ppgf_ordering_cache,
    )

    ppgf_convi_bypass_merge_mode = validate_ppgf_convi_bypass_merge_mode(
        config,
        ppgf_convi_bypass_threshold,
    )

    model_target = str(config.model_target).lower()
    if model_target not in {"best", "last"}:
        raise ValueError("model_target must be best or last")
    backbone = load_denoiser(
        model_path,
        best=model_target == "best",
        overrides={"checkpointing": True},
    ).backbone.to(device)
    backbone.eval().requires_grad_(False)
    base_denoiser = Denoiser(backbone).to(device)

    return {
        "model_path": model_path,
        "latent_dir": latent_dir,
        "denoiser_config": denoiser_config,
        "coordinates": coordinates,
        "latent_shape": latent_shape,
        "state_size": state_size,
        "device": device,
        "sources": sources,
        "ppgf_ordering_cache": ppgf_ordering_cache,
        "pcgrad_ordering_cache": pcgrad_ordering_cache,
        "convi_loss_spatial_top1": convi_loss_spatial_top1,
        "ppgf_convi_bypass_threshold": ppgf_convi_bypass_threshold,
        "ppgf_convi_bypass_merge_mode": ppgf_convi_bypass_merge_mode,
        "physical_autoencoder": physical_autoencoder,
        "backbone": backbone,
        "base_denoiser": base_denoiser,
        "noise_schedule": create_schedule(denoiser_config.train).to(device),
    }


def release_generation_runtime(runtime: dict) -> None:
    """Release generation models before loading the split-wide metric decoder."""
    runtime["sources"].clear()
    for key in (
        "physical_autoencoder",
        "base_denoiser",
        "backbone",
        "noise_schedule",
        "coordinates",
    ):
        runtime[key] = None
    gc.collect()
    torch.cuda.empty_cache()


def _load_source_observation(source, timestamp: datetime, runtime, mask_frac: float):
    time_string = format_hour(timestamp)
    device = runtime["device"]
    coordinates = runtime["coordinates"]
    resize = int(coordinates["ae"].resize)
    if source["kind"] == "obs2era":
        return load_obs2era_latent_at_time(
            time_string,
            source["encoder"],
            coordinates["mean"],
            coordinates["effective_std"],
            source["dataset"],
            source["time_to_idx"],
            resize,
            device,
            mask_frac,
        )
    if source["kind"] == "convi_obs_encoder":
        normalized_surface_values = _load_convi_era5_truth_surface(
            source,
            timestamp,
        )
        truth = source.get("convi_era5_truth_values")
        return load_convi_obs_encoder_latent_at_time(
            time_string,
            source["dataset"],
            source["encoder"],
            coordinates["mean"],
            coordinates["effective_std"],
            resize,
            device,
            mask_frac,
            normalized_surface_values=normalized_surface_values,
            full_surface_mask=bool(
                truth is not None and truth.get("full_surface_mask", False)
            ),
        )
    if source["kind"] == "direct_e2o":
        observation = load_direct_e2o_at_time(
            time_string,
            source["sate_dataset"],
            source["time_to_idx"],
            device,
        )
        if observation is None:
            return None
        pixel_mask = observation.pop("pixel_mask")
        return filter_physical_observation_by_block_mask(
            observation,
            pixel_mask,
            resize,
            mask_frac=mask_frac,
        )
    if source["kind"] == "convi_physical":
        observation = load_convi_at_time(
            time_string,
            source["dataset"],
            resize,
            device,
            mask_frac=mask_frac,
        )
        if observation is not None and source.get("convi_era5_truth_values") is not None:
            surface = _load_convi_era5_truth_surface(source, timestamp)
            replace_convi_physical_values(observation, surface)
        return observation
    raise RuntimeError(f"unknown loaded source kind: {source['kind']!r}")


def replace_convi_physical_values(observation, normalized_surface):
    """Replace values only, preserving each QC variable/pixel pair and QC support."""
    values = torch.as_tensor(normalized_surface, dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("ERA5 surface must have shape (4,H,W)")
    channels = observation["channel_local"].detach().cpu().long()
    pixels = observation["pixel_index"].detach().cpu().long()
    selected = values.reshape(4, -1)[channels, pixels]
    if selected.shape != observation["values"].shape or not torch.isfinite(selected).all():
        raise RuntimeError("ERA5 truth selection does not match CONVI QC observations")
    observation["values"] = selected.to(observation["values"].device)


def select_encoder_observation_variance(
    source: Mapping,
    latent_mask: torch.Tensor,
    latent_pixels: int,
    latent_channels: int,
) -> torch.Tensor:
    """Select MMPS variance in exactly the same token/channel order as y_sel."""
    latent_mask = torch.as_tensor(latent_mask, dtype=torch.bool).reshape(-1)
    if latent_mask.shape != (int(latent_pixels),):
        raise RuntimeError(
            f"Encoder latent mask shape {tuple(latent_mask.shape)} != "
            f"{(int(latent_pixels),)}"
        )
    values = torch.as_tensor(source["var_y"])
    mode = str(source.get("var_y_mode", "channelwise")).lower()
    if mode == "spatial":
        expected_shape = (int(latent_pixels), int(latent_channels))
        if tuple(values.shape) != expected_shape:
            raise RuntimeError(
                f"spatial var_y shape {tuple(values.shape)} != {expected_shape}"
            )
        return values[latent_mask.to(device=values.device)].reshape(-1)

    values = values.reshape(-1)
    if values.shape != (int(latent_channels),):
        raise RuntimeError(
            f"channel var_y shape {tuple(values.shape)} != "
            f"{(int(latent_channels),)}"
        )
    pixels = int(latent_mask.sum())
    return (
        values.view(1, latent_channels)
        .expand(pixels, latent_channels)
        .reshape(-1)
    )


def apply_encoder_observation_scalar_filter(
    source: Mapping,
    observation: dict,
    latent_pixels: int,
    latent_channels: int,
) -> torch.Tensor | None:
    """Filter one Encoder observation with an optional static scalar mask."""
    full_mask = source.get("convi_loss_scalar_mask")
    if full_mask is None:
        return None
    values = observation["y_sel"]
    latent_mask = observation["m_lat"].to(device=values.device, dtype=torch.bool)
    full_mask = torch.as_tensor(
        full_mask,
        dtype=torch.bool,
        device=values.device,
    )
    expected_shape = (int(latent_pixels), int(latent_channels))
    if tuple(full_mask.shape) != expected_shape:
        raise RuntimeError(
            f"CONVI loss scalar mask shape {tuple(full_mask.shape)} != "
            f"{expected_shape}"
        )
    if tuple(latent_mask.shape) != (int(latent_pixels),):
        raise RuntimeError(
            f"Encoder latent mask shape {tuple(latent_mask.shape)} != "
            f"{(int(latent_pixels),)}"
        )
    selected_mask = full_mask[latent_mask].reshape(-1)
    if values.numel() != selected_mask.numel():
        raise RuntimeError(
            "Encoder observation width does not match the selected scalar mask: "
            f"{values.numel()} != {selected_mask.numel()}"
        )
    observation["latent_scalar_filter"] = selected_mask
    observation["y_sel"] = values[selected_mask]
    return selected_mask


def select_encoder_state_values(
    state: torch.Tensor,
    observation: Mapping,
) -> torch.Tensor:
    """Apply the observation token mask and optional scalar loss filter."""
    selected = state[:, observation["m_lat"], :].reshape(state.shape[0], -1)
    scalar_filter = observation.get("latent_scalar_filter")
    if scalar_filter is not None:
        scalar_filter = torch.as_tensor(
            scalar_filter,
            dtype=torch.bool,
            device=selected.device,
        ).reshape(-1)
        if scalar_filter.numel() != selected.shape[1]:
            raise RuntimeError(
                "Encoder state width does not match the scalar loss filter: "
                f"{selected.shape[1]} != {scalar_filter.numel()}"
            )
        selected = selected[:, scalar_filter]
    return selected.reshape(-1)


def build_convi_gradient_latent_masks(
    config: DictConfig,
    sources: Sequence[Mapping],
    source_masks: torch.Tensor,
    latent_channels: int,
    device=None,
) -> list[torch.Tensor | None] | None:
    """Expand the dynamic CONVI m_lat mask over all latent channels."""
    if not bool(config.get("convi_gradient_latent_mask", False)):
        return None
    if str(config.guidance_merge).lower() == "joint":
        raise ValueError(
            "convi_gradient_latent_mask requires an independent-source "
            "guidance merge"
        )
    if len(sources) < 2:
        raise ValueError(
            "convi_gradient_latent_mask requires CONVI plus at least one "
            "other guidance source"
        )
    convi_ids = [
        source_id
        for source_id, source in enumerate(sources)
        if str(source["name"]).lower() == "convi"
    ]
    if len(convi_ids) != 1:
        raise ValueError(
            "convi_gradient_latent_mask requires exactly one CONVI source"
        )
    convi_id = convi_ids[0]
    if str(sources[convi_id]["kind"]).lower() != "convi_obs_encoder":
        raise ValueError(
            "convi_gradient_latent_mask currently requires the Encoder route"
        )

    masks = torch.as_tensor(source_masks, dtype=torch.bool)
    if masks.ndim != 3 or masks.shape[0] != len(sources):
        raise ValueError(
            "source_masks must have shape (sources, frames, latent_pixels); "
            f"got {tuple(masks.shape)}"
        )
    channels = int(latent_channels)
    if channels < 1:
        raise ValueError("latent_channels must be positive")
    convi_mask = (
        masks[convi_id]
        .unsqueeze(-1)
        .expand(*masks[convi_id].shape, channels)
        .reshape(-1)
        .contiguous()
    )
    if device is not None:
        convi_mask = convi_mask.to(device=device, non_blocking=True)
    result: list[torch.Tensor | None] = [None] * len(sources)
    result[convi_id] = convi_mask
    return result


def _prepare_window_guidance(config: DictConfig, runtime: dict, anchor: datetime):
    """Build one six-hour observation vector and its exact matching operators."""
    sources = runtime["sources"]
    device = runtime["device"]
    latent_pixels, latent_channels = runtime["latent_shape"]
    convi_bypass_threshold = runtime.get("ppgf_convi_bypass_threshold")
    convi_bypass_merge_mode = runtime["ppgf_convi_bypass_merge_mode"]
    convi_bypass_source_id = None
    convi_physical_pixel_counts = None
    if convi_bypass_threshold is not None:
        convi_bypass_source_id = next(
            source_id
            for source_id, source in enumerate(sources)
            if source["name"] == "convi"
        )
        convi_physical_pixel_counts = torch.zeros(
            WINDOW_SIZE,
            latent_pixels,
            dtype=torch.int64,
        )
    times = window_times(anchor)
    dates = torch.stack([timestamp_tensor(value) for value in times])[None].to(device)
    per_frame = []
    value_counts = {source["name"]: [] for source in sources}
    mask_frac = float(config.mask_frac)
    for timestamp in times:
        observations = []
        for source in sources:
            observation = _load_source_observation(source, timestamp, runtime, mask_frac)
            if (
                observation is not None
                and source["kind"] in {"obs2era", "convi_obs_encoder"}
            ):
                apply_encoder_observation_scalar_filter(
                    source,
                    observation,
                    latent_pixels,
                    latent_channels,
                )
            observations.append(observation)
            values = (
                None
                if observation is None
                else observation.get("y_sel", observation.get("values"))
            )
            value_counts[source["name"]].append(0 if values is None else int(values.numel()))
        per_frame.append(observations)

    y_parts = []
    variance_parts = []
    source_index_parts = [[] for _ in sources]
    source_masks = torch.zeros(len(sources), WINDOW_SIZE, latent_pixels, dtype=torch.bool)
    offset = 0
    for time_index, observations in enumerate(per_frame):
        for source_id, (source, observation) in enumerate(zip(sources, observations, strict=True)):
            if observation is None:
                continue
            source_masks[source_id, time_index] = observation["m_lat"].detach().cpu()
            if source_id == convi_bypass_source_id:
                raw_counts = observation.get("physical_pixel_counts")
                if raw_counts is None:
                    raise RuntimeError(
                        "CONVI PPGF bypass requires physical pixel counts"
                    )
                pixel_counts = raw_counts.detach().to(
                    device="cpu",
                    dtype=torch.int64,
                ).reshape(-1)
                if pixel_counts.numel() != latent_pixels:
                    raise RuntimeError(
                        "CONVI physical pixel count width does not match the latent grid: "
                        f"{pixel_counts.numel()} != {latent_pixels}"
                    )
                block_capacity = int(runtime["coordinates"]["ae"].resize) ** 2
                if int(pixel_counts.min()) < 0 or int(pixel_counts.max()) > block_capacity:
                    raise RuntimeError(
                        "CONVI physical pixel counts are outside the latent-block capacity"
                    )
                convi_physical_pixel_counts[time_index] = pixel_counts
            if source["kind"] in {"obs2era", "convi_obs_encoder"}:
                observed_values = observation["y_sel"]
                observation_variance = select_encoder_observation_variance(
                    source,
                    observation["m_lat"],
                    latent_pixels,
                    latent_channels,
                )
                scalar_filter = observation.get("latent_scalar_filter")
                if scalar_filter is not None:
                    observation_variance = observation_variance[
                        scalar_filter.to(device=observation_variance.device)
                    ]
            else:
                observed_values = observation["values"]
                observation_variance = source["var_y"][observation["channel_local"]]
            if observation_variance.numel() != observed_values.numel():
                raise RuntimeError(
                    f"{source['name']} observation/var_y length mismatch: "
                    f"{observed_values.numel()} != {observation_variance.numel()}"
                )
            length = int(observed_values.numel())
            y_parts.append(observed_values)
            variance_parts.append(observation_variance)
            source_index_parts[source_id].append(torch.arange(offset, offset + length))
            offset += length

    if bool(config.get("save_source_latent_masks", False)):
        runtime["source_anchor_latent_masks"] = source_masks[:, ANCHOR_FRAME_INDEX].clone()
    if bool(config.get("save_window_latents", False)):
        runtime["source_window_latent_masks"] = source_masks.clone()
    y_obs = torch.cat(y_parts) if y_parts else torch.empty(0, device=device)
    var_y = torch.cat(variance_parts) if variance_parts else torch.empty(0, device=device)
    source_indices = [
        torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        for parts in source_index_parts
    ]
    coordinates = runtime["coordinates"]

    def make_operator(source_ids):
        source_ids = tuple(source_ids)

        def operator(blanket, blanket_id=0):
            del blanket_id
            states = blanket.reshape(blanket.shape[0], WINDOW_SIZE, latent_pixels, latent_channels)
            values = []
            for time_index in range(WINDOW_SIZE):
                for source_id in source_ids:
                    source = sources[source_id]
                    observation = per_frame[time_index][source_id]
                    if observation is None:
                        continue
                    state = states[:, time_index]
                    if source["kind"] in {"obs2era", "convi_obs_encoder"}:
                        values.append(
                            select_encoder_state_values(state, observation)
                        )
                        continue
                    native = state * coordinates["effective_std"] + coordinates["mean"]
                    if source["kind"] == "direct_e2o":
                        prediction = source["decoder"].forward_at_pixels(
                            native,
                            observation["pixel_index"],
                            observation["cos_satzen"],
                        )
                        values.append(
                            select_direct_e2o_prediction(prediction, observation["valid_mask"])
                        )
                        continue
                    date = dates[:, time_index]
                    if date.shape[0] != state.shape[0]:
                        date = date.expand(state.shape[0], -1)
                    decoded = source["era_autoencoder"].decode(native, date)
                    values.append(
                        decoded[
                            :,
                            observation["pixel_index"],
                            observation["channel_index"],
                        ].reshape(-1)
                    )
            return torch.cat(values) if values else blanket.new_empty(0)

        return operator

    if not sources:
        return runtime["base_denoiser"], None, dates, value_counts

    all_operator = make_operator(range(len(sources)))
    iterations = int(config.diffusion.mmps_iters)
    tweedie = bool(config.diffusion.get("tweedie_covariance", True))
    merge = str(config.guidance_merge).lower()
    valid_merges = {
        "joint",
        "sum",
        "pcgrad",
        "ppgf",
        "channel_ppgf",
        "channel_ppgf_accumulate",
        "channel_ppgf_top1",
        "spatial_vector_ppgf",
    }
    if merge not in valid_merges:
        raise ValueError(f"guidance_merge must be one of {sorted(valid_merges)}")
    instrument_gradient_masks = build_convi_gradient_latent_masks(
        config,
        sources,
        source_masks,
        latent_channels,
        device=device,
    )
    if len(sources) == 1 or merge == "joint":
        wrapped = MMPSDenoiser(
            runtime["base_denoiser"],
            all_operator,
            y_obs,
            var_y,
            iterations=iterations,
            tweedie_covariance=tweedie,
        )
        return wrapped, None, dates, value_counts

    merge_group_size = latent_channels
    ppgf_order_group_count = None
    ordering_cache = runtime.get("ppgf_ordering_cache")
    if (
        merge in {
            "channel_ppgf",
            "channel_ppgf_accumulate",
            "channel_ppgf_top1",
        }
        and ordering_cache is not None
        and ordering_cache["ordering_granularity"] == "spatial_channel"
    ):
        expected_order_size = latent_pixels * latent_channels
        if int(ordering_cache["order_size"]) != expected_order_size:
            raise RuntimeError(
                "spatial ordering width does not match the runtime latent state: "
                f"{ordering_cache['order_size']} != {expected_order_size}"
            )
        merge_group_size = expected_order_size
    elif merge == "spatial_vector_ppgf":
        if (
            ordering_cache is None
            or ordering_cache["ordering_granularity"] != "spatial_channel"
        ):
            raise RuntimeError(
                "spatial_vector_ppgf requires a loaded spatial-channel ordering cache"
            )
        expected_order_groups = latent_pixels
        if (
            int(ordering_cache["spatial_vector_order_size"])
            != expected_order_groups
        ):
            raise RuntimeError(
                "spatial vector ordering height does not match latent pixels: "
                f"{ordering_cache['spatial_vector_order_size']} "
                f"!= {expected_order_groups}"
            )
        ppgf_order_group_count = expected_order_groups

    ppgf_bypass_mask = None
    ppgf_convi_bypass_metadata = None
    if convi_physical_pixel_counts is not None:
        dense_mask = convi_physical_pixel_counts >= convi_bypass_threshold
        ppgf_bypass_mask = (
            dense_mask.unsqueeze(-1)
            .expand(WINDOW_SIZE, latent_pixels, latent_channels)
            .reshape(-1)
            .to(device=device)
        )
        ppgf_convi_bypass_metadata = {
            "threshold": int(convi_bypass_threshold),
            "merge_mode": convi_bypass_merge_mode,
            "comparison": "physical_pixel_count_greater_than_or_equal",
            "count_semantics": (
                "unique QC SURF/ICOADS union pixels per frame "
                "and periodically padded latent block"
            ),
            "physical_pixel_counts": convi_physical_pixel_counts.to(torch.int16),
            "dense_latent_blocks_per_frame": dense_mask.sum(dim=1).tolist(),
        }

    source_operators = [make_operator([source_id]) for source_id in range(len(sources))]
    wrapped = MultiInstrumentMMPSDenoiser(
        runtime["base_denoiser"],
        all_operator,
        y_obs,
        var_y,
        instrument_indices=[source_indices],
        instrument_masks=[list(source_masks.unbind(0))],
        instrument_gradient_masks=(
            None
            if instrument_gradient_masks is None
            else [instrument_gradient_masks]
        ),
        instrument_operators=source_operators,
        merge=merge,
        pcgrad_group_size=merge_group_size,
        pcgrad_overlap_only=bool(config.pcgrad_overlap_only),
        pcgrad_shuffle=bool(config.pcgrad_shuffle),
        ppgf_overlap_only=bool(config.ppgf_overlap_only),
        ppgf_aligned_clip=float(config.ppgf_aligned_clip),
        ppgf_order_group_count=ppgf_order_group_count,
        ppgf_bypass_instrument_id=convi_bypass_source_id,
        ppgf_bypass_mask=ppgf_bypass_mask,
        ppgf_bypass_merge_mode=convi_bypass_merge_mode,
        iterations=iterations,
        tweedie_covariance=tweedie,
    )
    if ppgf_convi_bypass_metadata is not None:
        wrapped.ppgf_convi_bypass_metadata = ppgf_convi_bypass_metadata
    return wrapped, wrapped, dates, value_counts


def ppgf_ordering_uncertainty(
    config: DictConfig,
    runtime: dict,
    channelwise: bool,
    spatial_vector: bool = False,
) -> torch.Tensor:
    """Return global, scalar-channel, or spatial-vector ordering uncertainty."""
    if channelwise and spatial_vector:
        raise ValueError("PPGF ordering cannot be channelwise and spatial-vector")
    sources = runtime["sources"]
    ordering_source = str(config.get("ppgf_ordering_source", "var_y")).lower()
    if ordering_source == "matrix_cache":
        cache = runtime.get("ppgf_ordering_cache")
        if cache is None:
            raise RuntimeError("PPGF ordering matrix cache was not loaded")
        if spatial_vector:
            values = cache.get("spatial_vector")
            if values is None:
                raise RuntimeError(
                    "spatial-vector PPGF ordering was not loaded from the cache"
                )
            return values
        if channelwise:
            return cache["channel"]
        return cache["vector"]
    if spatial_vector:
        raise ValueError("spatial-vector PPGF ordering requires matrix_cache")
    if ordering_source == "guidance_reliability":
        missing = [source["name"] for source in sources if "ppgf_reliability" not in source]
        if missing:
            raise RuntimeError(f"PPGF guidance reliability was not loaded for sources: {missing}")
        if channelwise:
            return torch.stack(
                [source["ppgf_reliability"]["channel"] for source in sources],
                dim=1,
            )
        return torch.stack([source["ppgf_reliability"]["vector"] for source in sources])
    if ordering_source != "var_y":
        raise ValueError(
            "ppgf_ordering_source must be var_y or matrix_cache"
        )
    if not channelwise:
        return torch.stack([pcgrad_order_uncertainty(source["var_y"]) for source in sources])
    return torch.stack(
        [
            channel_ppgf_uncertainty(
                source["var_y"],
                (
                    "obs2era"
                    if source["kind"] in {"obs2era", "convi_obs_encoder"}
                    else "direct_e2o"
                ),
                runtime["latent_shape"][-1],
            )
            for source in sources
        ],
        dim=1,
    )


def _build_sampler(
    config: DictConfig,
    runtime: dict,
    denoiser,
    multi_guidance,
    dates: torch.Tensor,
    anchor_seed: int,
    rank: int,
    blanket_stride: int = WINDOW_SIZE,
    reconstruction_mode: str = "symmetrical",
):
    trajectory_denoiser = TrajectoryDenoiser(
        denoiser,
        blanket_size=WINDOW_SIZE,
        blanket_stride=int(blanket_stride),
        state_size=runtime["state_size"],
        distributed=False,
        pass_blanket_ids=bool(runtime["sources"]),
        mode=str(reconstruction_mode),
    )
    trajectory_denoiser = partial(trajectory_denoiser, date=dates)
    sampler_type = str(config.diffusion.sampler.type)
    sampler_class = select_sampler(sampler_type)
    sampler_kwargs = dict(OmegaConf.to_container(config.diffusion.sampler.config, resolve=True))

    if multi_guidance is not None:
        var_y_uncertainties = torch.stack([
            pcgrad_order_uncertainty(source["var_y"]) for source in runtime["sources"]
        ])
        if multi_guidance.ppgf_ordered:
            if sampler_type != "pc":
                raise ValueError("PPGF requires the PC sampler")
            sampler_class = PPGFSampler
            groupwise_ordering = (
                multi_guidance.channel_ppgf or multi_guidance.spatial_vector_ppgf
            )
            ordering_uncertainty = ppgf_ordering_uncertainty(
                config,
                runtime,
                channelwise=multi_guidance.channel_ppgf,
                spatial_vector=multi_guidance.spatial_vector_ppgf,
            )
            sampler_kwargs.update(
                ppgf_num_tasks=multi_guidance.num_instruments,
                ppgf_num_channels=(
                    int(ordering_uncertainty.shape[0]) if groupwise_ordering else 0
                ),
                ppgf_channelwise=groupwise_ordering,
                ppgf_seed=anchor_seed,
                ppgf_order_mode=str(config.ppgf_order_mode),
                ppgf_uncertainty=ordering_uncertainty,
                ppgf_uncertainty_strength=float(config.ppgf_uncertainty_strength),
                ppgf_uncertainty_strength_mode=str(
                    config.get("ppgf_uncertainty_strength_mode", "constant")
                ),
                ppgf_uncertainty_strength_start=float(
                    config.get("ppgf_uncertainty_strength_start", 0.0)
                ),
                ppgf_uncertainty_strength_end=float(
                    config.get("ppgf_uncertainty_strength_end", 10.0)
                ),
                ppgf_uncertainty_strength_ramp_steps=int(
                    config.get("ppgf_uncertainty_strength_ramp_steps", 16)
                ),
                ppgf_anneal_mode=str(config.get("ppgf_anneal_mode", "none")),
                ppgf_anneal_sigma_high=float(config.get("ppgf_anneal_sigma_high", 3.0)),
                ppgf_anneal_sigma_low=float(config.get("ppgf_anneal_sigma_low", 1.0)),
                ppgf_anneal_full_steps=int(config.get("ppgf_anneal_full_steps", 16)),
                ppgf_anneal_blend_steps=int(config.get("ppgf_anneal_blend_steps", 32)),
                ppgf_anneal_sum_steps=int(config.get("ppgf_anneal_sum_steps", 16)),
                order_log=rank == 0,
            )
        elif multi_guidance.pcgrad_shuffle:
            if sampler_type != "pc":
                raise ValueError("shuffled PCGrad requires the PC sampler")
            pcgrad_uncertainties = var_y_uncertainties
            pcgrad_ordering_cache = runtime.get("pcgrad_ordering_cache")
            if pcgrad_ordering_cache is not None:
                pcgrad_uncertainties = pcgrad_ordering_cache["spatial_vector"]
            sampler_kwargs.update(
                pcgrad_shuffle=True,
                pcgrad_num_tasks=multi_guidance.num_instruments,
                pcgrad_seed=anchor_seed,
                pcgrad_order_mode=str(config.pcgrad_order_mode),
                pcgrad_uncertainty=pcgrad_uncertainties,
                pcgrad_uncertainty_strength=float(config.pcgrad_uncertainty_strength),
                order_log=rank == 0,
            )

    steps = (
        int(runtime["denoiser_config"].valid.denoising_steps)
        if config.diffusion.num_steps is None
        else int(config.diffusion.num_steps)
    )
    return sampler_class(
        denoiser=trajectory_denoiser,
        steps=steps,
        schedule=runtime["noise_schedule"],
        silent=True,
        **sampler_kwargs,
    )


def _precision_context(precision: str):
    if precision == "float32":
        return contextlib.nullcontext()
    if precision not in {"float16", "bfloat16"}:
        raise ValueError("precision must be float16, bfloat16, or float32")
    return torch.autocast(device_type="cuda", dtype=getattr(torch, precision))


def sample_anchor(
    config: DictConfig,
    runtime: dict,
    anchor: datetime,
    anchor_index: int,
    contract_hash: str,
    rank: int,
    *,
    sample_index: int | None = None,
    seed_anchor_index: int | None = None,
) -> dict:
    """Generate one independent window and retain only its fourth frame."""
    effective_sample_index = 0 if sample_index is None else int(sample_index)
    effective_seed_anchor_index = (
        int(anchor_index) if seed_anchor_index is None else int(seed_anchor_index)
    )
    anchor_seed = ensemble_sample_seed(
        int(config.seed),
        effective_seed_anchor_index,
        effective_sample_index,
    )
    random.seed(anchor_seed)
    np.random.seed(anchor_seed % (2**32))
    torch.manual_seed(anchor_seed)
    torch.cuda.manual_seed_all(anchor_seed)

    denoiser, multi_guidance, dates, value_counts = _prepare_window_guidance(
        config, runtime, anchor
    )
    sampler = _build_sampler(
        config,
        runtime,
        denoiser,
        multi_guidance,
        dates,
        anchor_seed,
        rank,
    )
    generator = torch.Generator(device=runtime["device"]).manual_seed(anchor_seed)
    noise = torch.randn(
        1,
        WINDOW_SIZE * runtime["state_size"],
        generator=generator,
        device=runtime["device"],
    )
    start = noise * runtime["noise_schedule"].sigma_tmax().to(runtime["device"])
    with torch.no_grad(), _precision_context(str(config.precision)):
        sampled = sampler(start)
    window = sampled.reshape(1, WINDOW_SIZE, *runtime["latent_shape"])[0]
    frame_four = extract_anchor_frame(window).detach().to(dtype=torch.float32, device="cpu")

    diagnostics = None
    if multi_guidance is not None:
        diagnostics = multi_guidance.diagnostics()
    payload = {
        "kind": ANCHOR_KIND,
        "contract_hash": contract_hash,
        "anchor_index": int(anchor_index),
        "anchor": timestamp_tensor(anchor),
        "window_timestamps": torch.stack([
            timestamp_tensor(value) for value in window_times(anchor)
        ]),
        "frame_number": ANCHOR_FRAME_NUMBER,
        "latent_coordinate": "diffusion_standardized",
        "latent": frame_four,
        "source_value_counts": value_counts,
        "guidance_diagnostics": diagnostics,
    }
    if bool(config.get("save_source_latent_masks", False)):
        payload["source_anchor_latent_masks"] = runtime["source_anchor_latent_masks"].clone()
    if bool(config.get("save_window_latents", False)):
        payload["window_latents"] = window.detach().to(dtype=torch.float32, device="cpu")
        payload["source_window_latent_masks"] = runtime["source_window_latent_masks"].clone()
    if (sample_index is not None or bool(config.get("save_window_latents", False))):
        payload.update({
            "sample_index": effective_sample_index,
            "sample_seed": anchor_seed,
            "seed_anchor_index": effective_seed_anchor_index,
        })
    bypass_metadata = getattr(
        multi_guidance,
        "ppgf_convi_bypass_metadata",
        None,
    )
    if bypass_metadata is not None:
        payload["ppgf_convi_bypass"] = bypass_metadata
    return payload


def _source_contract(source: Mapping) -> dict:
    var_y = source["var_y"].detach().cpu().float()
    var_y_mode = str(source["var_y_mode"])
    contract = {
        "name": source["name"],
        "kind": source["kind"],
        "model_config": file_identity(Path(source["model_dir"]) / "config.yaml"),
        "checkpoint": file_identity(source["checkpoint_path"]),
        "calibration_var_y": (
            file_identity(source["var_y_cache"])
            if source.get("var_y_cache") is not None else None
        ),
        "var_y_mode": var_y_mode,
        "var_y_constant": source["var_y_constant"],
        "var_y_channels": (
            int(var_y.shape[-1])
            if var_y_mode == "spatial"
            else int(var_y.numel())
        ),
        "var_y_min": float(var_y.min()),
        "var_y_mean": float(var_y.mean()),
        "var_y_max": float(var_y.max()),
    }
    if var_y_mode == "spatial":
        contract["var_y_shape"] = list(var_y.shape)
        contract["var_y_spatial_tokens"] = int(var_y.shape[0])
        contract["var_y_values"] = int(var_y.numel())
        contract["channelwise_calibration_var_y"] = file_identity(
            source["channelwise_var_y_cache"]
        )
        contract["spatial_var_y"] = dict(source["spatial_var_y"])
    var_y_scale = float(source.get("var_y_scale", 1.0))
    if var_y_scale != 1.0:
        contract["var_y_scale"] = var_y_scale
    reliability = source.get("ppgf_reliability")
    if reliability is not None:
        channel_values = reliability["channel"].float()
        contract["ppgf_guidance_reliability"] = {
            "cache": file_identity(reliability["path"]),
            "kind": reliability["kind"],
            "contract_hash": reliability["contract_hash"],
            "metric": reliability["metric"],
            "num_used_anchors": reliability["num_used_anchors"],
            "num_missing_anchors": reliability["num_missing_anchors"],
            "vector_geometric_mean": float(reliability["vector"]),
            "channel_min": float(channel_values.min()),
            "channel_mean": float(channel_values.mean()),
            "channel_max": float(channel_values.max()),
        }
    if source["kind"] == "convi_physical":
        contract["era_normalization"] = source["era_normalization"]
        contract["decoder_input_space"] = "native_era_latent"
    if source["kind"] in {"convi_obs_encoder", "convi_physical"}:
        dataset = source["dataset"]
        observation_data = getattr(dataset, "observation_data", None)
        if not isinstance(observation_data, Mapping) or observation_data.get("variant") != "qc":
            raise RuntimeError(
                "CONVI source did not retain an auditable QC observation-data contract"
            )
        contract["convi_observation_data"] = dict(observation_data)
        contract["convi_observation_scope"] = "all_valid_qc_observations"
        truth = source.get("convi_era5_truth_values")
        if truth is not None:
            full_surface_mask = bool(truth.get("full_surface_mask", False))
            contract["convi_guidance_values"] = {
                "source": (
                    "exact_hour_era5_truth_full_grid"
                    if full_surface_mask
                    else "exact_hour_era5_truth_at_convi_variable_masks"
                ),
                "era5_data_path": str(truth["data_path"]),
                "source_mean": file_identity(truth["source_mean_path"]),
                "source_std": file_identity(truth["source_std_path"]),
                "target_mean": file_identity(truth["target_mean_path"]),
                "target_std": file_identity(truth["target_std_path"]),
                "normalization": (
                    "stored_hourly_era5 -> physical -> CONVI encoder ERA basis"
                ),
                "mask": (
                    "all four surface variables and union mask valid globally"
                    if full_surface_mask
                    else "unchanged QC CONVI per-variable availability"
                ),
                "var_y": (
                    "Calibrated centered spatial variance from Dense-ERA full-mask "
                    "cache"
                    if var_y_mode == "spatial"
                    else "unchanged real-CONVI-calibrated cache"
                ),
            }
    return contract


def build_generation_manifest(
    config: DictConfig,
    runtime: dict,
    names: Sequence[str],
    split: str,
    *,
    anchors: Sequence[datetime] | None = None,
    source_anchor_indices: Sequence[int] | None = None,
) -> dict:
    canonical_anchors, _ = select_config_anchors(config, canonical=True)
    anchors = canonical_anchors if anchors is None else list(anchors)
    if not anchors:
        raise ValueError("generation requires at least one anchor")
    canonical_index = {
        timestamp_key(anchor): index for index, anchor in enumerate(canonical_anchors)
    }
    try:
        expected_source_indices = [
            canonical_index[timestamp_key(anchor)] for anchor in anchors
        ]
    except KeyError as error:
        raise ValueError("generation anchors must belong to the canonical split") from error
    if source_anchor_indices is None:
        source_anchor_indices = expected_source_indices
    else:
        source_anchor_indices = [int(index) for index in source_anchor_indices]
        if source_anchor_indices != expected_source_indices:
            raise ValueError("source anchor indices do not match the selected timestamps")
    if source_anchor_indices != sorted(set(source_anchor_indices)):
        raise ValueError("generation anchors must be strictly increasing and unique")
    num_samples_per_anchor = _num_samples_per_anchor(config)
    is_anchor_subset = anchors != canonical_anchors

    model_checkpoint = runtime["model_path"] / f"model_{str(config.model_target).lower()}.pth"
    bypass_merge_mode = runtime["ppgf_convi_bypass_merge_mode"]
    ordering_cache = runtime.get("ppgf_ordering_cache")
    ordering_cache_contract = None
    if ordering_cache is not None:
        channel_values = ordering_cache["channel"].float()
        vector_values = ordering_cache["vector"].float()
        ordering_cache_contract = {
            "cache": file_identity(ordering_cache["path"]),
            "kind": ordering_cache["kind"],
            "source_order": ordering_cache["source_order"],
            "cache_mask_frac": ordering_cache["cache_mask_frac"],
            "ordering_metric": ordering_cache["ordering_metric"],
            "calibration_kind": ordering_cache["calibration_kind"],
            "period": ordering_cache["period"],
            "time_scope": ordering_cache["time_scope"],
            "expected_hours": ordering_cache["expected_hours"],
            "num_selected_times": ordering_cache["num_selected_times"],
            "shape": list(channel_values.shape),
            "vector_reduction": ordering_cache["vector_reduction"],
            "vector_shape": list(vector_values.shape),
            "value_min": float(channel_values.min()),
            "value_mean": float(channel_values.mean()),
            "value_max": float(channel_values.max()),
            "vector_value_min": float(vector_values.min()),
            "vector_value_mean": float(vector_values.mean()),
            "vector_value_max": float(vector_values.max()),
        }
        spatial_vector_values = ordering_cache.get("spatial_vector")
        if spatial_vector_values is not None:
            spatial_vector_values = spatial_vector_values.float()
            ordering_cache_contract.update({
                "spatial_vector_reduction": ordering_cache[
                    "spatial_vector_reduction"
                ],
                "spatial_vector_shape": list(spatial_vector_values.shape),
                "spatial_vector_value_min": float(spatial_vector_values.min()),
                "spatial_vector_value_mean": float(spatial_vector_values.mean()),
                "spatial_vector_value_max": float(spatial_vector_values.max()),
            })
        for key in (
            "ordering_granularity",
            "spatial_shape",
            "order_size",
            "minimum_count",
            "fill_value",
            "missing_value_policy",
            "filled_position_count",
            "filled_scalar_count",
            "filled_position_count_by_source",
            "zero_count_position_count",
            "single_count_position_count",
            "zero_count_position_count_by_source",
            "single_count_position_count_by_source",
            "raw_cache",
        ):
            ordering_cache_contract[key] = ordering_cache[key]
    pcgrad_ordering_cache = runtime.get("pcgrad_ordering_cache")
    pcgrad_ordering_cache_contract = None
    if pcgrad_ordering_cache is not None:
        spatial_values = pcgrad_ordering_cache["spatial_vector"].float()
        pcgrad_ordering_cache_contract = {
            "cache": file_identity(pcgrad_ordering_cache["path"]),
            "kind": pcgrad_ordering_cache["kind"],
            "source_order": pcgrad_ordering_cache["source_order"],
            "ordering_metric": pcgrad_ordering_cache["ordering_metric"],
            "ordering_granularity": pcgrad_ordering_cache[
                "ordering_granularity"
            ],
            "spatial_shape": pcgrad_ordering_cache["spatial_shape"],
            "spatial_vector_reduction": pcgrad_ordering_cache[
                "spatial_vector_reduction"
            ],
            "spatial_vector_shape": list(spatial_values.shape),
            "spatial_vector_value_min": float(spatial_values.min()),
            "spatial_vector_value_mean": float(spatial_values.mean()),
            "spatial_vector_value_max": float(spatial_values.max()),
            "random_seed": pcgrad_ordering_cache["random_seed"],
            "randomization_granularity": pcgrad_ordering_cache[
                "randomization_granularity"
            ],
            "priority_semantics": pcgrad_ordering_cache["priority_semantics"],
            "reference_cache": pcgrad_ordering_cache["reference_cache"],
            "fixed_sort": (
                "descending fixed random rank; first source is projected first "
                "and final source has the highest priority"
                if pcgrad_ordering_cache["random_seed"] is not None
                else (
                    "descending spatial-vector MSE; high-uncertainty source "
                    "first, lowest-MSE source last"
                )
            ),
            "reuse": (
                "one order per spatial vector, shared by all 128 channels and "
                "fixed across six frames, diffusion steps, and anchors"
            ),
        }
    generation_contract = {
        "protocol_kind": PROTOCOL_KIND,
        "split": split,
        "split_bounds": [format_hour(anchors[0]), format_hour(anchors[-1])],
        "anchor_ranges": serialized_anchor_ranges(anchors),
        "num_anchors": len(anchors),
        "window_offsets_hours": list(WINDOW_OFFSETS),
        "anchor_frame_number": ANCHOR_FRAME_NUMBER,
        "anchor_interval_hours": int(config.timeline.evaluation.step_hours),
        "timeline": OmegaConf.to_container(config.timeline, resolve=True),
        "method": str(config.id),
        "exact_hour_observations": True,
        "independent_windows": True,
        "route": str(config.route).lower(),
        "source_order": list(names),
        "diffusion_config": file_identity(runtime["model_path"] / "config.yaml"),
        "diffusion_checkpoint": file_identity(model_checkpoint),
        "diffusion": OmegaConf.to_container(config.diffusion, resolve=True),
        "precision": str(config.precision),
        "seed": int(config.seed),
        "mask_frac": float(config.mask_frac),
        "variance_floor": float(config.variance_floor),
        "observation_covariance": {
            "var_y_mode": str(config.get("var_y_mode", "channelwise")),
            "var_y_constant": (
                float(config.var_y_constant)
                if str(config.get("var_y_mode", "channelwise")).lower() == "constant"
                else None
            ),
            **(
                {"var_y_scale": float(config.get("var_y_scale", 1.0))}
                if float(config.get("var_y_scale", 1.0)) != 1.0
                else {}
            ),
            **(
                {
                    "spatial_var_y_cache": file_identity(
                        config.spatial_var_y_cache
                    ),
                    "spatial_var_y_statistic": "centered_population_variance",
                    "spatial_var_y_layout": "(source,H_lat,W_lat,C_lat)",
                    "spatial_var_y_minimum_count": (
                        SPATIAL_VAR_Y_MINIMUM_COUNT
                    ),
                    "spatial_var_y_missing_value_policy": (
                        "count >= 2 uses fixed-position Calibrated centered variance; "
                        "count < 2 uses covered-sample calibrated channelwise "
                        "centered variance"
                    ),
                }
                if str(config.get("var_y_mode", "channelwise")).lower()
                == "spatial"
                else {}
            ),
        },
        "guidance": {
            "merge": str(config.guidance_merge),
            **(
                {
                    "convi_gradient_latent_mask": True,
                    "convi_gradient_mask_contract": {
                        "source": "convi",
                        "mask": "per-frame dynamic observation m_lat",
                        "layout_before_channel_expansion": (
                            "(6, H_lat*W_lat)"
                        ),
                        "channel_broadcast": "all latent channels",
                        "application_stage": (
                            "after per-source MMPS autograd score and before "
                            "diagnostics or gradient merge"
                        ),
                        "outside_mask": "exact_zero",
                    }
                }
                if bool(config.get("convi_gradient_latent_mask", False))
                else {}
            ),
            "pcgrad_overlap_only": bool(config.pcgrad_overlap_only),
            "pcgrad_shuffle": bool(config.pcgrad_shuffle),
            "pcgrad_order_mode": str(config.pcgrad_order_mode),
            "pcgrad_uncertainty_strength": float(config.pcgrad_uncertainty_strength),
            "pcgrad_ordering_cache": pcgrad_ordering_cache_contract,
            "ppgf_overlap_only": bool(config.ppgf_overlap_only),
            "ppgf_aligned_clip": float(config.ppgf_aligned_clip),
            "ppgf_order_mode": str(config.ppgf_order_mode),
            "ppgf_ordering_source": str(config.get("ppgf_ordering_source", "var_y")),
            "ppgf_ordering_cache": ordering_cache_contract,
            "ppgf_reliability_metric": str(config.get("ppgf_reliability_metric", "native_rmse")),
            "ppgf_uncertainty_strength": float(config.ppgf_uncertainty_strength),
            **(
                {
                    "convi_loss_spatial_top1": runtime[
                        "convi_loss_spatial_top1"
                    ]
                }
                if runtime.get("convi_loss_spatial_top1") is not None
                else {}
            ),
            **(
                {
                    "ppgf_convi_bypass_threshold": int(
                        config.ppgf_convi_bypass_threshold
                    ),
                    "ppgf_convi_bypass_merge_mode": bypass_merge_mode,
                    "ppgf_convi_bypass_rule": describe_ppgf_convi_bypass_rule(
                        bypass_merge_mode
                    ),
                }
                if config.get("ppgf_convi_bypass_threshold", None) is not None
                else {}
            ),
            **(
                {
                    "ppgf_uncertainty_strength_mode": str(
                        config.get("ppgf_uncertainty_strength_mode", "constant")
                    ),
                    "ppgf_uncertainty_strength_start": float(
                        config.get("ppgf_uncertainty_strength_start", 0.0)
                    ),
                    "ppgf_uncertainty_strength_end": float(
                        config.get("ppgf_uncertainty_strength_end", 10.0)
                    ),
                    "ppgf_uncertainty_strength_ramp_steps": int(
                        config.get("ppgf_uncertainty_strength_ramp_steps", 16)
                    ),
                }
                if str(config.get("ppgf_uncertainty_strength_mode", "constant")).lower()
                != "constant"
                else {}
            ),
            "ppgf_anneal_mode": str(config.get("ppgf_anneal_mode", "none")),
            "ppgf_anneal_sigma_high": float(config.get("ppgf_anneal_sigma_high", 3.0)),
            "ppgf_anneal_sigma_low": float(config.get("ppgf_anneal_sigma_low", 1.0)),
            "ppgf_anneal_full_steps": int(config.get("ppgf_anneal_full_steps", 16)),
            "ppgf_anneal_blend_steps": int(config.get("ppgf_anneal_blend_steps", 32)),
            "ppgf_anneal_sum_steps": int(config.get("ppgf_anneal_sum_steps", 16)),
        },
        "sources": [_source_contract(source) for source in runtime["sources"]],
        "latent_coordinate": {
            "stats": file_identity(runtime["latent_dir"] / "stats.pth"),
            "ae": file_identity(runtime["latent_dir"] / "ae" / "config.yaml"),
            "effective_std": "sqrt(stats.std^2 + ae.noise_level^2)",
            "ae_noise_level": float(runtime["coordinates"]["noise_level"]),
        },
    }
    if bool(config.get("save_source_latent_masks", False)):
        generation_contract["source_latent_mask_storage"] = {
            "field": "source_anchor_latent_masks",
            "layout": "(source, latent_pixel)",
            "frame_number": ANCHOR_FRAME_NUMBER,
            "scope": "exact-hour direct observation support, after observation filtering",
        }
    if bool(config.get("save_window_latents", False)):
        if num_samples_per_anchor != 1:
            raise ValueError("full-window calibration requires exactly one member")
        generation_contract["window_storage"] = {
            "latent_field": "window_latents", "mask_field": "source_window_latent_masks",
            "latent_layout": "(6,latent_pixel,channel)",
            "mask_layout": "(source,6,latent_pixel)",
        }
    if config.get("calibration_period") is not None:
        generation_contract["calibration_period"] = str(config.calibration_period)
        if config.get("rebuttal_calibration") is not None:
            generation_contract["rebuttal_calibration"] = OmegaConf.to_container(
                config.rebuttal_calibration, resolve=True,
            )
        if ordering_cache_contract is not None:
            ordering_cache_contract["expected_hours"] = ordering_cache.get("expected_hours")
        covariance = generation_contract["observation_covariance"]
        if "spatial_var_y_missing_value_policy" in covariance:
            covariance["spatial_var_y_missing_value_policy"] = (
                "count >= 2 uses calibrated spatial centered variance; "
                "count < 2 uses same-period same-source channelwise centered variance"
            )
    if is_anchor_subset:
        generation_contract["anchor_subset"] = {
            "inclusive": True,
            "start_time": format_hour(anchors[0]),
            "end_time": format_hour(anchors[-1]),
            "source_split_anchor_index_range": [
                source_anchor_indices[0],
                source_anchor_indices[-1],
            ],
            "parent_split_bounds": [format_hour(canonical_anchors[0]), format_hour(canonical_anchors[-1])],
            "parent_anchor_ranges": serialized_anchor_ranges(canonical_anchors),
            "parent_num_anchors": len(canonical_anchors),
        }
    if num_samples_per_anchor != 1:
        generation_contract["sampling_ensemble"] = {
            "num_samples_per_anchor": num_samples_per_anchor,
            "reduction": ENSEMBLE_REDUCTION,
            "member_seed_schedule": (
                "seed + canonical_split_anchor_index + "
                f"sample_index * {ENSEMBLE_MEMBER_SEED_STRIDE}"
            ),
            "member_zero_matches_single_sample_seed": True,
            "member_storage": "anchors_members/{anchor_index:04d}/{sample_index:03d}.pth",
            "retained_storage": "anchors/{anchor_index:04d}.pth",
        }
    return {
        "kind": PROTOCOL_KIND,
        "contract_hash": _contract_hash(generation_contract),
        "generation_contract": generation_contract,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "diffusion_training_validation_caveat": (
            "The current diffusion checkpoint historically used full-2025 validation. "
            "This evaluator uses the explicitly recorded calibration and evaluation periods, "
            "but the checkpoint-selection caveat must be disclosed with results."
        ),
    }


def _ensure_manifest(target_dir: Path, manifest: Mapping) -> None:
    """Create or verify the immutable generation contract for safe resume."""
    path = target_dir / "manifest.json"
    if path.exists():
        current = json.loads(path.read_text())
        if current.get("contract_hash") != manifest["contract_hash"]:
            raise RuntimeError(
                f"output directory has a different generation contract: {target_dir}"
            )
        return
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(text, path)
    current = json.loads(path.read_text())
    if current.get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError(f"manifest race produced a mismatched contract in {target_dir}")


def _anchor_path(anchor_dir: Path, anchor_index: int) -> Path:
    return anchor_dir / f"{anchor_index:04d}.pth"


def _anchor_member_path(
    members_dir: Path,
    anchor_index: int,
    sample_index: int,
) -> Path:
    return members_dir / f"{anchor_index:04d}" / f"{sample_index:03d}.pth"


def _validate_anchor_payload(
    payload: Mapping,
    anchor: datetime,
    anchor_index: int,
    contract_hash: str,
    latent_shape: tuple[int, int] | None = None,
) -> None:
    checks = {
        "kind": ANCHOR_KIND,
        "contract_hash": contract_hash,
        "anchor_index": int(anchor_index),
        "frame_number": ANCHOR_FRAME_NUMBER,
        "latent_coordinate": "diffusion_standardized",
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"anchor {anchor_index} {field} mismatch: "
                f"{payload.get(field)!r} != {expected!r}"
            )
    if timestamp_key(payload["anchor"]) != timestamp_key(anchor):
        raise RuntimeError(f"anchor {anchor_index} timestamp mismatch")
    expected_window = [timestamp_key(value) for value in window_times(anchor)]
    actual_window = [timestamp_key(value) for value in payload["window_timestamps"]]
    if actual_window != expected_window:
        raise RuntimeError(f"anchor {anchor_index} does not contain exact-hour window times")
    if latent_shape is not None and tuple(payload["latent"].shape) != tuple(latent_shape):
        raise RuntimeError(
            f"anchor {anchor_index} latent shape mismatch: "
            f"{tuple(payload['latent'].shape)} != {tuple(latent_shape)}"
        )


def _validate_anchor_member_payload(
    payload: Mapping,
    anchor: datetime,
    anchor_index: int,
    sample_index: int,
    source_anchor_index: int,
    base_seed: int,
    contract_hash: str,
    latent_shape: tuple[int, int],
) -> None:
    _validate_anchor_payload(
        payload,
        anchor,
        anchor_index,
        contract_hash,
        latent_shape,
    )
    checks = {
        "sample_index": int(sample_index),
        "seed_anchor_index": int(source_anchor_index),
        "sample_seed": ensemble_sample_seed(
            base_seed,
            source_anchor_index,
            sample_index,
        ),
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"anchor {anchor_index} member {sample_index} {field} mismatch: "
                f"{payload.get(field)!r} != {expected!r}"
            )


def _ensemble_anchor_payload(member_payloads: Sequence[Mapping]) -> dict:
    if not member_payloads:
        raise ValueError("cannot average an empty member sequence")
    payload = dict(member_payloads[0])
    member_seeds = [int(member["sample_seed"]) for member in member_payloads]
    payload["latent"] = torch.stack(
        [torch.as_tensor(member["latent"], dtype=torch.float32) for member in member_payloads],
        dim=0,
    ).mean(dim=0)
    for field in ("sample_index", "sample_seed", "seed_anchor_index"):
        payload.pop(field, None)
    payload["ensemble"] = {
        "num_members": len(member_payloads),
        "reduction": ENSEMBLE_REDUCTION,
        "member_seeds": member_seeds,
    }
    payload["guidance_diagnostics_representative_member"] = 0
    return payload


def _validate_ensemble_anchor_payload(
    payload: Mapping,
    anchor: datetime,
    anchor_index: int,
    source_anchor_index: int,
    base_seed: int,
    num_samples_per_anchor: int,
    contract_hash: str,
    latent_shape: tuple[int, int],
) -> None:
    _validate_anchor_payload(
        payload,
        anchor,
        anchor_index,
        contract_hash,
        latent_shape,
    )
    ensemble = payload.get("ensemble")
    if not isinstance(ensemble, Mapping):
        raise RuntimeError(f"anchor {anchor_index} has no ensemble metadata")
    expected_seeds = [
        ensemble_sample_seed(base_seed, source_anchor_index, sample_index)
        for sample_index in range(num_samples_per_anchor)
    ]
    checks = {
        "num_members": int(num_samples_per_anchor),
        "reduction": ENSEMBLE_REDUCTION,
        "member_seeds": expected_seeds,
    }
    for field, expected in checks.items():
        if ensemble.get(field) != expected:
            raise RuntimeError(
                f"anchor {anchor_index} ensemble {field} mismatch: "
                f"{ensemble.get(field)!r} != {expected!r}"
            )


def generate_local_anchors(
    config: DictConfig,
    runtime: dict,
    anchors: Sequence[datetime],
    source_anchor_indices: Sequence[int],
    local_indices: Sequence[int],
    target_dir: Path,
    contract_hash: str,
    rank: int,
) -> None:
    anchor_dir = target_dir / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    members_dir = target_dir / "anchors_members"
    num_samples_per_anchor = _num_samples_per_anchor(config)
    if num_samples_per_anchor > 1:
        members_dir.mkdir(parents=True, exist_ok=True)
    if len(source_anchor_indices) != len(anchors):
        raise ValueError("source anchor index count differs from selected anchors")
    log_every = max(1, int(config.log_every))
    for local_position, anchor_index in enumerate(local_indices):
        anchor = anchors[anchor_index]
        source_anchor_index = int(source_anchor_indices[anchor_index])
        path = _anchor_path(anchor_dir, anchor_index)
        if num_samples_per_anchor == 1:
            reuse = False
            if path.is_file():
                try:
                    payload = safe_load(path, map_location="cpu")
                    _validate_anchor_payload(
                        payload,
                        anchor,
                        anchor_index,
                        contract_hash,
                        runtime["latent_shape"],
                    )
                    if (bool(config.get("save_window_latents", False))):
                        _validate_anchor_member_payload(
                            payload, anchor, anchor_index, 0, source_anchor_index,
                            int(config.seed), contract_hash, runtime["latent_shape"],
                        )
                    reuse = True
                except Exception as error:
                    print(
                        f"[rank {rank}] recomputing invalid anchor file {path}: {error}",
                        flush=True,
                    )
            if not reuse:
                started = time.monotonic()
                payload = sample_anchor(
                    config,
                    runtime,
                    anchor,
                    anchor_index,
                    contract_hash,
                    rank,
                    seed_anchor_index=source_anchor_index,
                )
                _atomic_torch_save(payload, path)
                elapsed = time.monotonic() - started
                status = f"generated in {elapsed:.1f}s"
            else:
                status = "reused"
        else:
            member_payloads = []
            generated_members = 0
            for sample_index in range(num_samples_per_anchor):
                member_path = _anchor_member_path(
                    members_dir,
                    anchor_index,
                    sample_index,
                )
                member_path.parent.mkdir(parents=True, exist_ok=True)
                member_payload = None
                if member_path.is_file():
                    try:
                        member_payload = safe_load(member_path, map_location="cpu")
                        _validate_anchor_member_payload(
                            member_payload,
                            anchor,
                            anchor_index,
                            sample_index,
                            source_anchor_index,
                            int(config.seed),
                            contract_hash,
                            runtime["latent_shape"],
                        )
                    except Exception as error:
                        print(
                            f"[rank {rank}] recomputing invalid member "
                            f"{member_path}: {error}",
                            flush=True,
                        )
                        member_payload = None
                if member_payload is None:
                    started = time.monotonic()
                    member_payload = sample_anchor(
                        config,
                        runtime,
                        anchor,
                        anchor_index,
                        contract_hash,
                        rank,
                        sample_index=sample_index,
                        seed_anchor_index=source_anchor_index,
                    )
                    _atomic_torch_save(member_payload, member_path)
                    generated_members += 1
                    print(
                        f"[rank {rank}] anchor {anchor_index + 1}/{len(anchors)} "
                        f"member {sample_index + 1}/{num_samples_per_anchor} "
                        f"{format_hour(anchor)} generated in "
                        f"{time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                member_payloads.append(member_payload)

            ensemble_payload = _ensemble_anchor_payload(member_payloads)
            reuse = False
            if path.is_file():
                try:
                    saved_payload = safe_load(path, map_location="cpu")
                    _validate_ensemble_anchor_payload(
                        saved_payload,
                        anchor,
                        anchor_index,
                        source_anchor_index,
                        int(config.seed),
                        num_samples_per_anchor,
                        contract_hash,
                        runtime["latent_shape"],
                    )
                    if not torch.equal(
                        torch.as_tensor(saved_payload["latent"]),
                        ensemble_payload["latent"],
                    ):
                        raise RuntimeError("saved latent is not the member arithmetic mean")
                    reuse = True
                except Exception as error:
                    print(
                        f"[rank {rank}] recomputing invalid ensemble anchor "
                        f"{path}: {error}",
                        flush=True,
                    )
            if not reuse:
                _atomic_torch_save(ensemble_payload, path)
            status = (
                f"members generated={generated_members} "
                f"reused={num_samples_per_anchor - generated_members}; "
                f"ensemble {'reused' if reuse else 'written'}"
            )
        if local_position % log_every == 0 or local_position + 1 == len(local_indices):
            print(
                f"[rank {rank}] anchor {anchor_index + 1}/{len(anchors)} "
                f"{format_hour(anchor)} {status}",
                flush=True,
            )
        if (local_position + 1) % int(config.cuda_cleanup_interval) == 0:
            gc.collect()
            torch.cuda.empty_cache()


def merge_anchor_outputs(
    anchors: Sequence[datetime],
    target_dir: Path,
    contract_hash: str,
    latent_shape: tuple[int, int],
    manifest: Mapping,
) -> Path:
    """Merge sorted frame-four latents only; six-frame trajectories are never gathered."""
    anchor_dir = target_dir / "anchors"
    latents = []
    source_value_counts = []
    for anchor_index, anchor in enumerate(anchors):
        path = _anchor_path(anchor_dir, anchor_index)
        if not path.is_file():
            raise RuntimeError(f"missing anchor output after distributed run: {path}")
        payload = safe_load(path, map_location="cpu")
        _validate_anchor_payload(
            payload,
            anchor,
            anchor_index,
            contract_hash,
            latent_shape,
        )
        latents.append(payload["latent"])
        source_value_counts.append(payload.get("source_value_counts", {}))
    output = {
        "kind": COLLECTION_KIND,
        "contract_hash": contract_hash,
        "protocol_kind": PROTOCOL_KIND,
        "split": manifest["generation_contract"]["split"],
        "route": manifest["generation_contract"]["route"],
        "source_order": manifest["generation_contract"]["source_order"],
        "frame_number": ANCHOR_FRAME_NUMBER,
        "window_offsets_hours": torch.tensor(WINDOW_OFFSETS),
        "timestamps": torch.stack([timestamp_tensor(anchor) for anchor in anchors]),
        "frame4_latents": torch.stack(latents),
        "latent_coordinate": "diffusion_standardized",
        "source_value_counts": source_value_counts,
        "num_anchors": len(anchors),
    }
    sampling_ensemble = manifest["generation_contract"].get("sampling_ensemble")
    if sampling_ensemble is not None:
        output["sampling_ensemble"] = dict(sampling_ensemble)
    destination = target_dir / "frame4_latents.pth"
    _atomic_torch_save(output, destination)
    return destination


def _metric_targets(config: DictConfig) -> tuple[list[str], list[int]]:
    configured = config.metrics.target_names
    names = (
        list(CHANNEL_NAMES)
        if isinstance(configured, str) and configured.lower() == "all"
        else [str(value) for value in configured]
    )
    if not names:
        raise ValueError("metrics.target_names cannot be empty")
    if len(set(names)) != len(names):
        raise ValueError("metrics.target_names contains duplicates")
    unknown = [name for name in names if name not in CHANNEL_NAMES]
    if unknown:
        raise ValueError(f"unknown metric channels: {unknown}")
    return names, [CHANNEL_NAMES.index(name) for name in names]


def _truth_dataset(
    ae_file_config: DictConfig,
    anchors: Sequence[datetime],
    data_path=None,
) -> tuple[ERA5DataNpy_2016_2025_1h, dict]:
    normalization = ae_file_config.normalization
    dataset = ERA5DataNpy_2016_2025_1h(
        data_path=data_path or ae_file_config.data_path,
        blanket_size=1,
        time_range=(format_hour(anchors[0]), format_hour(anchors[-1])),
        source_mean_path=normalization.source_mean_path,
        source_std_path=normalization.source_std_path,
        target_mean_path=normalization.target_mean_path,
        target_std_path=normalization.target_std_path,
    )
    index = {
        timestamp_key(date): dataset_index for dataset_index, date in enumerate(dataset.dates)
    }
    missing = [format_hour(anchor) for anchor in anchors if timestamp_key(anchor) not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"ERA5 truth is missing {len(missing)} split anchors; first missing: {preview}"
        )
    return dataset, index


def compute_local_metrics(
    config: DictConfig,
    latent_dir: Path,
    anchors: Sequence[datetime],
    local_indices: Sequence[int],
    target_dir: Path,
    contract_hash: str,
    latent_shape: tuple[int, int],
    device,
    rank: int,
) -> dict:
    """Decode each rank's frame-four subset and accumulate RMSE sufficient statistics."""
    denoiser_config = compose(Path(config.model_path) / "config.yaml")
    coordinates = _load_latent_coordinates(latent_dir, denoiser_config, device)
    ae_file_config = coordinates["ae_file_config"]
    autoencoder = load_auto_encoder(latent_dir / "ae", "model", device="cpu", eval_mode=True)
    autoencoder.eval().requires_grad_(False).to(device)
    target_names, target_channels = _metric_targets(config)
    target_channel_tensor = torch.tensor(target_channels, dtype=torch.long, device=device)
    truth, truth_index = _truth_dataset(
        ae_file_config,
        anchors,
        data_path=config.metrics.get("data_path", None),
    )

    height, width = (int(value) for value in coordinates["ae"].shape)
    latitude_weight = torch.cos(
        torch.linspace(math.pi / 2, -math.pi / 2, height, device=device)
    ).clamp_min(0)
    latitude_weight = latitude_weight.view(1, height, 1)
    squared_error_sum = torch.zeros(len(target_names), dtype=torch.float64, device=device)
    weight_sum = torch.zeros(len(target_names), dtype=torch.float64, device=device)
    anchor_dir = target_dir / "anchors"
    log_every = max(1, int(config.metrics.log_every))

    for local_position, anchor_index in enumerate(local_indices):
        anchor = anchors[anchor_index]
        payload = safe_load(_anchor_path(anchor_dir, anchor_index), map_location="cpu")
        _validate_anchor_payload(
            payload,
            anchor,
            anchor_index,
            contract_hash,
            latent_shape,
        )
        latent = payload["latent"].to(device=device, dtype=torch.float32)[None]
        native = latent * coordinates["effective_std"] + coordinates["mean"]
        date = timestamp_tensor(anchor)[None].to(device)
        with torch.no_grad(), _precision_context(str(config.metrics.precision)):
            decoded = autoencoder.decode(native, date)
        prediction = (
            decoded.reshape(1, height, width, len(CHANNEL_NAMES))
            .permute(0, 3, 1, 2)[0]
            .index_select(0, target_channel_tensor)
            .float()
        )
        truth_state, _ = truth[truth_index[timestamp_key(anchor)]]
        target = (
            truth_state[0]
            .index_select(0, target_channel_tensor.cpu())
            .to(device=device, dtype=torch.float32)
        )
        difference = prediction - target
        valid = torch.isfinite(difference)
        weighted_square = torch.where(
            valid,
            difference.square() * latitude_weight,
            torch.zeros((), dtype=difference.dtype, device=device),
        )
        squared_error_sum += weighted_square.sum(dim=(-2, -1), dtype=torch.float64)
        valid_weight = torch.where(
            valid,
            latitude_weight.expand_as(difference),
            torch.zeros((), dtype=difference.dtype, device=device),
        )
        weight_sum += valid_weight.sum(dim=(-2, -1), dtype=torch.float64)
        if local_position % log_every == 0 or local_position + 1 == len(local_indices):
            print(
                f"[rank {rank}] RMSE frame {anchor_index + 1}/{len(anchors)} "
                f"{format_hour(anchor)}",
                flush=True,
            )
        del latent, native, decoded, prediction, truth_state, target, difference

    target_std = np.asarray(
        np.load(ae_file_config.normalization.target_std_path),
        dtype=np.float64,
    ).squeeze()
    if target_std.ndim != 1 or target_std.size < len(CHANNEL_NAMES):
        raise RuntimeError("target normalization std does not contain 69 channels")
    physical_std = torch.as_tensor(target_std[target_channels], dtype=torch.float64)
    del autoencoder
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "target_names": target_names,
        "squared_error_sum": squared_error_sum.cpu(),
        "weight_sum": weight_sum.cpu(),
        "physical_std": physical_std,
        "num_frames": torch.tensor(len(local_indices), dtype=torch.int64),
    }


def write_metrics(
    target_dir: Path,
    split: str,
    route: str,
    source_order: Sequence[str],
    metric_totals: Mapping,
    expected_frames: int,
    make_plot: bool,
) -> Path:
    num_frames = int(metric_totals["num_frames"])
    if num_frames != expected_frames:
        raise RuntimeError(f"metric frame count is {num_frames}, expected {expected_frames}")
    normalized, physical = finalize_weighted_rmse(
        metric_totals["squared_error_sum"],
        metric_totals["weight_sum"],
        metric_totals["physical_std"],
    )
    target_names = list(metric_totals["target_names"])
    payload = {
        "kind": METRICS_KIND,
        "split": split,
        "route": route,
        "source_order": list(source_order),
        "num_frames": num_frames,
        "aggregation": (
            "latitude-weighted squared errors summed over every frame-4 anchor "
            "in the complete split, then one square root"
        ),
        "target_names": target_names,
        "squared_error_sum": metric_totals["squared_error_sum"],
        "weight_sum": metric_totals["weight_sum"],
        "rmse_normalized": normalized,
        "rmse_physical": physical,
    }
    destination = target_dir / "rmse.pth"
    _atomic_torch_save(payload, destination)

    rows = ["channel,rmse_normalized,rmse_physical"]
    lines = [
        f"split={split} route={route} sources={','.join(source_order) or 'none'}",
        f"frames={num_frames} aggregation=split-wide-frame4",
        "channel          RMSE_normalized        RMSE_physical",
    ]
    for name, normalized_value, physical_value in zip(
        target_names, normalized.tolist(), physical.tolist(), strict=True
    ):
        rows.append(f"{name},{normalized_value:.10g},{physical_value:.10g}")
        lines.append(f"{name:<10} {normalized_value:>18.8g} {physical_value:>20.8g}")
    rows.append(f"mean,{normalized.mean().item():.10g},{physical.mean().item():.10g}")
    lines.append(
        f"{'mean':<10} {normalized.mean().item():>18.8g} " f"{physical.mean().item():>20.8g}"
    )
    _atomic_write_text("\n".join(rows) + "\n", target_dir / "rmse.csv")
    _atomic_write_text("\n".join(lines) + "\n", target_dir / "rmse.log")

    summary = {
        "kind": METRICS_KIND,
        "split": split,
        "route": route,
        "source_order": list(source_order),
        "num_frames": num_frames,
        "target_names": target_names,
        "rmse_normalized": normalized.tolist(),
        "rmse_physical": physical.tolist(),
        "mean_rmse_normalized": normalized.mean().item(),
        "mean_rmse_physical": physical.mean().item(),
        "diffusion_training_validation_caveat": (
            "The current diffusion checkpoint historically used full-2025 validation."
        ),
    }
    _atomic_write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        target_dir / "rmse.json",
    )

    if make_plot:
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(figsize=(16, 5))
        positions = np.arange(len(target_names))
        axis.bar(positions, physical.numpy())
        axis.set_xticks(positions)
        axis.set_xticklabels(target_names, rotation=45, ha="right")
        axis.set_ylabel("RMSE (physical units)")
        axis.set_title(f"{split}: split-wide frame-4 RMSE")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()
        figure.savefig(target_dir / "rmse.png", dpi=150)
        plt.close(figure)
    return destination


def _initialize_cpu_coordination(world_size: int, timeout_seconds: float) -> None:
    """Initialize Gloo only after diffusion sampling has completely finished."""
    if world_size <= 1:
        return
    if dist.is_initialized():
        raise RuntimeError("distributed process group was initialized during sampling")
    dist.init_process_group(
        backend="gloo",
        timeout=timedelta(seconds=float(timeout_seconds)),
    )


def _all_reduce_metric_totals(metric_totals: dict, world_size: int) -> None:
    if world_size <= 1:
        return
    for field in ("squared_error_sum", "weight_sum", "num_frames"):
        dist.all_reduce(metric_totals[field], op=dist.ReduceOp.SUM)


def _run_id(config: DictConfig, names: Sequence[str]) -> str:
    if not config.get("id"):
        raise ValueError("Select a paper method with python -m risda run METHOD")
    return str(config.id)


def main(config_path: str | None = None) -> None:
    overrides = sys.argv[1:]
    if overrides and overrides[0] == "--config":
        if len(overrides) < 2:
            raise ValueError("--config requires a YAML file")
        config_path, overrides = overrides[1], overrides[2:]
    if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage: torchrun --standalone --nproc_per_node=N "
            "evaluate_guided_six_frame.py --config <resolved-run-config.yaml>. "
            "Use python -m risda run METHOD for the public interface."
        )
        return
    if config_path is None:
        raise ValueError("Use python -m risda run METHOD, or supply --config with a resolved run config")
    if dist.is_initialized():
        raise RuntimeError(
            "do not initialize torch.distributed before this evaluator; "
            "ranks sample independent anchors"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("guided diffusion evaluation requires CUDA")

    config = compose(config_path, overrides=overrides)
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)
    split = str(config.split).lower()
    anchors, source_anchor_indices = select_config_anchors(config)
    num_samples_per_anchor = _num_samples_per_anchor(config)
    route = str(config.route).lower()
    names = select_source_names(
        list(config.enabled_sources),
        route,
        config.source_catalog,
    )
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid torchrun rank/world size: {rank}/{world_size}")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if bool(config.deterministic):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    target_dir = Path(config.output_root).expanduser().resolve() / _run_id(config, names) / split
    target_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_generation_runtime(config, names, device)
    latent_dir = runtime["latent_dir"]
    latent_shape = runtime["latent_shape"]
    manifest = build_generation_manifest(
        config,
        runtime,
        names,
        split,
        anchors=anchors,
        source_anchor_indices=source_anchor_indices,
    )
    _ensure_manifest(target_dir, manifest)
    if rank == 0:
        OmegaConf.save(config, target_dir / "config.yaml")
        print(
            f"[six-frame] split={split} anchors={len(anchors)} route={route} "
            f"sources={names or ['none']} ranks={world_size}",
            flush=True,
        )
        print(
            f"[six-frame] anchor_range={format_hour(anchors[0])}.."
            f"{format_hour(anchors[-1])} samples_per_anchor={num_samples_per_anchor} "
            f"total_diffusion_samples={len(anchors) * num_samples_per_anchor}",
            flush=True,
        )
        print(
            "[six-frame] independent windows; exact-hour observations on all six "
            f"frames; retaining {'all six frames and masks' if config.get('save_window_latents', False) else 'frame 4 only'}; multi-member runs retain the latent "
            "ensemble mean; no NCCL collectives during sampling",
            flush=True,
        )

    local_indices = partition_anchor_indices(len(anchors), rank, world_size)
    generate_local_anchors(
        config,
        runtime,
        anchors,
        source_anchor_indices,
        local_indices,
        target_dir,
        manifest["contract_hash"],
        rank,
    )
    release_generation_runtime(runtime)
    del runtime

    _initialize_cpu_coordination(
        world_size,
        float(config.coordination_timeout_seconds),
    )
    metric_totals = None
    if bool(config.metrics.enabled):
        metric_totals = compute_local_metrics(
            config,
            latent_dir,
            anchors,
            local_indices,
            target_dir,
            manifest["contract_hash"],
            latent_shape,
            device,
            rank,
        )
        _all_reduce_metric_totals(metric_totals, world_size)
    if world_size > 1:
        dist.destroy_process_group()

    if rank != 0:
        return
    collection_path = merge_anchor_outputs(
        anchors,
        target_dir,
        manifest["contract_hash"],
        latent_shape,
        manifest,
    )
    print(f"[six-frame] wrote {collection_path}", flush=True)
    if metric_totals is not None:
        metrics_path = write_metrics(
            target_dir,
            split,
            route,
            names,
            metric_totals,
            len(anchors),
            bool(config.metrics.make_plot),
        )
        print(f"[six-frame] wrote {metrics_path}", flush=True)
    print(
        "[six-frame] disclosure: the current diffusion checkpoint historically "
        "used full-2025 validation; see the recorded evaluation and calibration periods.",
        flush=True,
    )


if __name__ == "__main__":
    main()
