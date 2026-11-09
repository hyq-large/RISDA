"""Encoder residuals, observation support and float64 population moments."""
from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from appa.data.datasets import ERA5DataNpy_2016_2025_1h, SatelliteAEDataset
from experiments.observations.convi_obs2era_data import (
    CONVI_QC_TO_ERA5_QC_MASK, ERA5_QC_MASK_TO_ERA5_QC_MASK,
    SURFACE_CHANNEL_INDICES, Convi2EraPairDataset,
)
from risda.models.encoders import load_obs2era_encoder, resolve_surface_preprocessing
_CONVI_CALIBRATION_ALIGNMENT_MODE = CONVI_QC_TO_ERA5_QC_MASK

def timestamp_key(value) -> tuple[int, int, int, int]:
    """Return a strict (year, month, day, hour) key."""
    if isinstance(value, datetime):
        return value.year, value.month, value.day, value.hour
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value.astype("datetime64[h]"), unit="h")
        parsed = datetime.strptime(text, "%Y-%m-%dT%H")
        return timestamp_key(parsed)
    values = torch.as_tensor(value).reshape(-1).tolist()
    if len(values) != 4:
        raise ValueError(f"timestamp must have four fields; got {values}")
    return tuple(int(item) for item in values)

def format_hour(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H")

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def file_identity(path: Path, *, include_sha256: bool = False) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    identity = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = _sha256(path)
    return identity

def centered_standardized_residual(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    latent_mean: torch.Tensor,
    raw_std: torch.Tensor,
) -> torch.Tensor:
    denominator = raw_std.clamp_min(1e-8)
    return (
        (prediction - latent_mean) / denominator
        - (truth - latent_mean) / denominator
    )

def _time_index(times) -> dict[tuple[int, int, int, int], int]:
    result = {}
    for index, value in enumerate(times):
        key = timestamp_key(value)
        if key in result:
            raise RuntimeError(f"observation dataset has duplicate timestamp {key}")
        result[key] = index
    return result

def _build_dataset(
    source: str,
    data_config,
    *,
    convi_surface_only: bool = False,
    convi_surf_dir: Path | None = None,
    convi_icoa_dir: Path | None = None,
    convi_alignment_mode: str = _CONVI_CALIBRATION_ALIGNMENT_MODE,
):
    data_kind = str(data_config.get("kind", "sate")).lower()
    if data_kind == "convi":
        calibration_alignment_mode = str(convi_alignment_mode).lower()
        if calibration_alignment_mode != _CONVI_CALIBRATION_ALIGNMENT_MODE:
            raise ValueError(
                "CONVI calibration input must use real CONVI QC values; "
                f"got alignment_mode={calibration_alignment_mode!r}"
            )
        actual_source = str(data_config.get("name", "convi")).lower()
        surf_dir = Path(
            convi_surf_dir
            if convi_surf_dir is not None
            else data_config.surf_dir
        ).expanduser().resolve()
        configured_icoa_dir = (
            convi_icoa_dir
            if convi_icoa_dir is not None
            else data_config.get("icoa_dir", None)
        )
        effective_surface_only = (
            bool(convi_surface_only) or configured_icoa_dir is None
        )
        icoa_dir = (
            None
            if effective_surface_only
            else Path(configured_icoa_dir).expanduser().resolve()
        )
        for label, path in (("SURF", surf_dir), ("ICOADS", icoa_dir)):
            if path is not None and not path.is_dir():
                raise FileNotFoundError(
                    f"CONVI {label} directory is missing: {path}"
                )
        normalization = OmegaConf.to_container(
            data_config.era_normalization,
            resolve=True,
            enum_to_str=True,
        )
        surface_preprocessing = OmegaConf.to_container(
            data_config.get("surface_preprocessing", None),
            resolve=True,
            enum_to_str=True,
        )
        dataset = Convi2EraPairDataset(
            era_data_path=data_config.era_data_path,
            surf_dir=surf_dir,
            icoa_dir=icoa_dir,
            mean_std_dir=data_config.mean_std_dir,
            era_dataset_cls=ERA5DataNpy_2016_2025_1h,
            era_dataset_kwargs=normalization,
            surface_preprocessing=surface_preprocessing,
            alignment_mode=calibration_alignment_mode,
        )
        data_kind = "convi_aligned"
        in_channels = int(dataset.input_channels)
    elif data_kind == "sate":
        actual_source = str(data_config.sate_name).lower()
        dataset = SatelliteAEDataset(
            data_path=data_config.sate_data_path,
            sate_name=actual_source,
            standard_dir=data_config.sate_standard_dir,
            blanket_size=1,
            time_range=None,
            use_satzen=bool(data_config.get("use_satzen", False)),
        )
        in_channels = int(dataset.get_dim())
    else:
        raise ValueError(f"unsupported Encoder data.kind {data_kind!r}")
    if actual_source != source:
        raise RuntimeError(
            f"requested source {source!r} does not match model source "
            f"{actual_source!r}"
        )
    return data_kind, dataset, in_channels

def _load_observation(data_kind: str, dataset, index: int):
    if data_kind == "convi_aligned":
        observation, _physical_target, physical_mask, _date = dataset[index]
        physical_mask = torch.as_tensor(physical_mask, dtype=torch.float32)
        if physical_mask.ndim != 3 or physical_mask.shape[0] <= max(
            SURFACE_CHANNEL_INDICES
        ):
            raise RuntimeError(
                "aligned CONVI physical mask must retain the 69-channel "
                f"surface positions; got {tuple(physical_mask.shape)}"
            )
        pixel_mask = physical_mask[list(SURFACE_CHANNEL_INDICES)].amax(
            dim=0,
            keepdim=True,
        )
    elif data_kind == "sate":
        merged, _ = dataset[index]
        observation = merged[0]
        pixel_mask = observation[
            dataset.obs_channel : dataset.obs_channel + 1
        ]
    else:
        raise ValueError(f"unsupported observation data kind: {data_kind!r}")
    observation = torch.as_tensor(observation, dtype=torch.float32)
    pixel_mask = torch.as_tensor(pixel_mask, dtype=torch.float32)
    if observation.ndim != 3:
        raise RuntimeError(
            f"Encoder input must have shape (C,H,W); got "
            f"{tuple(observation.shape)}"
        )
    if tuple(pixel_mask.shape) != (1, *observation.shape[-2:]):
        raise RuntimeError(
            f"pixel mask shape {tuple(pixel_mask.shape)} does not match "
            f"observation shape {tuple(observation.shape)}"
        )
    return observation, pixel_mask

def _obs_mask_to_frac(mask_pix: torch.Tensor, resize: int) -> torch.Tensor:
    height, width = mask_pix.shape[-2:]
    factor = int(resize)
    pad_h = (-height) % factor
    pad_w = (-width) % factor + 2 * factor
    if pad_w > 0:
        mask_pix = F.pad(
            mask_pix,
            (pad_w // 2, pad_w - pad_w // 2, 0, 0),
            mode="circular",
        )
    if pad_h > 0:
        mask_pix = F.pad(
            mask_pix,
            (0, 0, pad_h // 2, pad_h - pad_h // 2),
            mode="constant",
            value=0,
        )
    return F.avg_pool2d(mask_pix, kernel_size=(factor, factor))[0, 0].reshape(-1)

def finalize_spatial_moments(
    sum_error: torch.Tensor,
    sum_square_error: torch.Tensor,
    spatial_counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Finalize raw population moments independently at every latent position."""
    sum_error = torch.as_tensor(sum_error, dtype=torch.float64)
    sum_square_error = torch.as_tensor(sum_square_error, dtype=torch.float64)
    spatial_counts = torch.as_tensor(spatial_counts, dtype=torch.int64)
    if sum_error.shape != sum_square_error.shape:
        raise ValueError("spatial residual moment tensors must share a shape")
    if sum_error.ndim != 2:
        raise ValueError("spatial residual moments must have shape (pixels, channels)")
    if spatial_counts.shape != sum_error.shape[:1]:
        raise ValueError("spatial counts must have shape (pixels,)")
    if bool((spatial_counts < 0).any()):
        raise ValueError("spatial counts cannot be negative")

    bias = torch.full_like(sum_error, float("nan"))
    mse = torch.full_like(sum_error, float("nan"))
    valid = spatial_counts > 0
    if bool(valid.any()):
        denominator = spatial_counts[valid].double().unsqueeze(1)
        bias[valid] = sum_error[valid] / denominator
        mse[valid] = sum_square_error[valid] / denominator

    variance = torch.full_like(sum_error, float("nan"))
    variance[valid] = (mse[valid] - bias[valid].square()).clamp_min(0)
    return {
        "bias": bias.float(),
        "mse": mse.float(),
        "rmse": mse.sqrt().float(),
        "variance": variance.float(),
        "std": variance.sqrt().float(),
        "counts": spatial_counts,
    }

def finalize_channelwise_moments_from_spatial_sums(
    sum_error: torch.Tensor,
    sum_square_error: torch.Tensor,
    spatial_counts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pool the exact spatial float64 sums into one statistic per channel.

    This is deliberately finalized from the same accumulators as the spatial
    cache. It therefore has exactly the same model inputs, selected latent
    blocks, inference precision, and ERA5 latent truth as the spatial
    statistics, without a second Encoder pass.
    """
    sum_error = torch.as_tensor(sum_error, dtype=torch.float64)
    sum_square_error = torch.as_tensor(sum_square_error, dtype=torch.float64)
    spatial_counts = torch.as_tensor(spatial_counts, dtype=torch.int64)
    if sum_error.shape != sum_square_error.shape or sum_error.ndim != 2:
        raise ValueError(
            "spatial residual sums must share shape (pixels, channels)"
        )
    if spatial_counts.shape != sum_error.shape[:1]:
        raise ValueError("spatial counts must have shape (pixels,)")
    if bool((spatial_counts < 0).any()):
        raise ValueError("spatial counts cannot be negative")

    total_count = int(spatial_counts.sum())
    if total_count <= 0:
        raise RuntimeError("no supported latent blocks for channelwise moments")
    denominator = float(total_count)
    bias = sum_error.sum(dim=0) / denominator
    mse = sum_square_error.sum(dim=0) / denominator
    variance = (mse - bias.square()).clamp_min(0)
    counts = torch.full(
        (sum_error.shape[1],),
        total_count,
        dtype=torch.int64,
    )
    return {
        "var_y": variance.float(),
        "bias": bias.float(),
        "mse": mse.float(),
        "rmse": mse.sqrt().float(),
        "variance": variance.float(),
        "std": variance.sqrt().float(),
        "counts": counts,
    }
