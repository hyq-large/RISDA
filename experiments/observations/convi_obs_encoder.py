"""Dedicated mask-aware CONVIOBS-to-ERA latent observation path."""

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from appa.save import safe_load
from risda.models.encoders import (
    CRESSMAN_SURFACE_RAW_CHANNELS,
    obs2era_encoder_uses_cressman,
    resolve_surface_preprocessing,
)


CONVI_OBS_ENCODER_CHANNELS = 5
CONVI_OBS_ENCODER_KIND = "convi_obs2era"


def _pixel_mask_to_latent_fraction(pixel_mask, resize):
    """Pool the union pixel mask onto the periodically padded ERA latent grid."""
    mask = pixel_mask.float()[None]
    height, width = mask.shape[-2:]
    factor = int(resize)
    pad_h = (-height) % factor
    pad_w = (-width) % factor + 2 * factor
    if pad_w > 0:
        mask = F.pad(
            mask,
            (pad_w // 2, pad_w - pad_w // 2, 0, 0),
            mode="circular",
        )
    if pad_h > 0:
        mask = F.pad(
            mask,
            (0, 0, pad_h // 2, pad_h - pad_h // 2),
            mode="constant",
            value=0,
        )
    fraction = F.avg_pool2d(mask, kernel_size=(factor, factor))
    return rearrange(fraction, "n c h w -> (n h w c)")


def _pixel_mask_to_latent_mask(pixel_mask, resize, mask_frac=0.0):
    """Pool the union pixel mask onto the ERA latent grid."""
    fraction = _pixel_mask_to_latent_fraction(pixel_mask, resize)
    if mask_frac <= 0.0:
        return fraction > 0.0
    return fraction >= (float(mask_frac) - 1e-6)


def _pixel_mask_to_latent_counts(pixel_mask, resize):
    """Count unique union-mask physical pixels in each ERA latent block."""
    factor = int(resize)
    fraction = _pixel_mask_to_latent_fraction(pixel_mask, factor)
    return (fraction * (factor * factor)).round().to(torch.int64)


def prepare_convi_obs_encoder_input(
    dataset,
    time,
    encoder,
    normalized_surface_values=None,
    full_surface_mask=False,
):
    """Build the raw input layout required by a saved CONVI encoder."""
    return dataset.prepare_surface_encoder_input(
        time,
        normalized_surface_values=normalized_surface_values,
        full_surface_mask=full_surface_mask,
        include_channel_masks=obs2era_encoder_uses_cressman(encoder),
    )



def encode_convi_obs_encoder_latent(
    dataset,
    encoder,
    era_mean,
    era_std,
    era_resize,
    time,
    device,
    mask_frac=0.0,
    return_pixel_counts=False,
    normalized_surface_values=None,
    full_surface_mask=False,
):
    """Encode one CONVI frame into standardized ERA latent."""
    encoder_input, pixel_mask = prepare_convi_obs_encoder_input(
        dataset,
        time,
        encoder,
        normalized_surface_values=normalized_surface_values,
        full_surface_mask=full_surface_mask,
    )
    encoder_input = encoder_input.to(device, non_blocking=True)[None]
    expected_channels = (
        CRESSMAN_SURFACE_RAW_CHANNELS
        if obs2era_encoder_uses_cressman(encoder)
        else CONVI_OBS_ENCODER_CHANNELS
    )
    if encoder_input.shape[1] != expected_channels:
        raise RuntimeError(
            f"CONVI obs encoder input must have {expected_channels} channels; "
            f"got {encoder_input.shape[1]}"
        )
    with torch.no_grad():
        latent_native = encoder(encoder_input)
    latent_std = (latent_native - era_mean) / era_std.clamp_min(1e-8)
    pixel_mask = pixel_mask.to(device)
    latent_mask = _pixel_mask_to_latent_mask(pixel_mask, era_resize, mask_frac)
    if latent_std.shape[1] != latent_mask.numel():
        raise RuntimeError(
            "CONVI obs encoder mask shape does not match encoder output: "
            f"{latent_mask.numel()} != {latent_std.shape[1]}"
        )
    if return_pixel_counts:
        pixel_counts = _pixel_mask_to_latent_counts(pixel_mask, era_resize)
        return latent_std, latent_mask, pixel_counts
    return latent_std, latent_mask


def calibrate_var_y_convi_obs_encoder(
    dataset,
    encoder,
    era_h5_path,
    era_resize,
    era_mean,
    era_std,
    device,
    max_frames=64,
    valid_frac=0.05,
):
    """Estimate centered residual moments for the dedicated CONVI encoder."""
    with h5py.File(era_h5_path, "r") as era_h5:
        era_dates = np.asarray(era_h5["dates"])
        era_index = {
            tuple(int(value) for value in date): index
            for index, date in enumerate(era_dates)
        }
        times = [
            time
            for time in sorted(dataset.times)
            if (
                int(time.year),
                int(time.month),
                int(time.day),
                int(time.hour),
            )
            in era_index
        ]
        if not times:
            raise RuntimeError("CONVIOBS has no timestamps aligned with era5.h5")

        n_valid = max(1, int(round(len(times) * float(valid_frac))))
        selected_times = times[-n_valid:][: int(max_frames)]
        channels = int(era_mean.numel())
        residual_sum = torch.zeros(channels, dtype=torch.float64, device=device)
        squared_error = torch.zeros(channels, dtype=torch.float64, device=device)
        counts = torch.zeros(channels, dtype=torch.float64, device=device)

        for time in selected_times:
            latent_obs, latent_mask = encode_convi_obs_encoder_latent(
                dataset,
                encoder,
                era_mean,
                era_std,
                era_resize,
                time,
                device,
                mask_frac=0.0,
            )
            if not bool(latent_mask.any()):
                continue
            key = (int(time.year), int(time.month), int(time.day), int(time.hour))
            latent_true = torch.as_tensor(
                era_h5["latents"][era_index[key]], device=device
            )[None]
            latent_true = (latent_true - era_mean) / era_std.clamp_min(1e-8)
            residual = (latent_obs - latent_true).double()
            support = latent_mask.view(1, -1, 1)
            residual_sum += (residual * support).sum(dim=(0, 1))
            squared_error += (residual.square() * support).sum(dim=(0, 1))
            counts += support.sum().expand(channels)

    if (counts == 0).any():
        raise RuntimeError("CONVI obs encoder calibration found no supported blocks")
    bias = residual_sum / counts
    mse = squared_error / counts
    variance = (mse - bias.square()).clamp_min(0)
    return {
        "bias": bias.float().cpu(),
        "mse": mse.float().cpu(),
        "rmse": mse.sqrt().float().cpu(),
        "variance": variance.float().cpu(),
        "std": variance.sqrt().float().cpu(),
        "counts": counts.long().cpu(),
        "selected_times": [str(time) for time in selected_times],
        "num_aligned_frames": len(times),
        "num_tail_frames": n_valid,
    }


def convi_obs_encoder_residual_metadata(
    config, dataset, era_dir, encoder_model_dir, checkpoint
):
    """Identity metadata for a dedicated CONVI obs2era variance cache."""
    era_dir = Path(era_dir)
    model_dir = Path(encoder_model_dir).resolve()
    checkpoint_name = (
        "model_best.pth" if str(checkpoint) == "best" else "model_last.pth"
    )
    checkpoint_path = model_dir / checkpoint_name
    config_path = model_dir / "config.yaml"
    preprocessing = resolve_surface_preprocessing(OmegaConf.load(config_path))
    input_layout = (
        "4 normalized values,4 per-channel masks -> cressman -> "
        "4 interpolated values,4 dilated masks,confidence"
        if preprocessing is not None
        else "t2m,u10,v10,msl,union_mask"
    )

    latent_stats = era_dir / "stats.pth"
    return {
        "version": 1,
        "input_layout": input_layout,
        "surface_preprocessing": preprocessing,
        "fill": "normalized_zero",
        "mask": "any_surface_channel_then_any_pixel_in_latent_block",
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_checkpoint_mtime_ns": (
            checkpoint_path.stat().st_mtime_ns
            if checkpoint_path.exists()
            else None
        ),
        "encoder_config": str(config_path),
        "encoder_config_mtime_ns": (
            config_path.stat().st_mtime_ns if config_path.exists() else None
        ),
        "latent_stats": str(latent_stats),
        "latent_stats_mtime_ns": (
            latent_stats.stat().st_mtime_ns if latent_stats.exists() else None
        ),
        "surf_dir": str(config.surf_dir),
        "icoa_dir": str(config.icoa_dir),
        "mean_std_dir": str(config.mean_std_dir),
        "valid_frac": float(config.get("var_y_valid_frac", 0.05)),
        "max_frames": int(config.get("var_y_max_batches", 64)),
    }


def get_var_y_convi_obs_encoder(
    config,
    dataset,
    era_dir,
    era_mean,
    encoder_model_dir,
    checkpoint,
    device,
):
    """Load the dedicated encoder's centered 128-channel residual variance."""
    channels = int(era_mean.numel())
    mode = str(config.get("var_y_mode", "rmse")).lower()
    floor = float(config.get("obs_covariance", 1e-4))
    if mode == "scalar":
        return torch.full((channels,), floor, dtype=torch.float32, device=device)
    if mode != "rmse":
        raise ValueError(f"Unknown var_y_mode: {mode!r}")
    if not bool(config.get("use_residual_var_y", False)):
        raise ValueError(
            "latent_obs_encoder with var_y_mode=rmse requires "
            "use_residual_var_y=true"
        )

    configured_cache = config.get("obs_encoder_residual_var_y_cache", None)
    cache_path = (
        Path(configured_cache)
        if configured_cache
        else Path(encoder_model_dir) / "convi_online_residual_var_y.pth"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"CONVI obs encoder residual variance cache is missing: {cache_path}"
        )
    cached = safe_load(cache_path, map_location="cpu")
    if cached.get("kind") != CONVI_OBS_ENCODER_KIND:
        raise RuntimeError(
            f"CONVI obs encoder variance kind mismatch in {cache_path}"
        )
    expected_source = str(config.get("name", "convi"))
    if str(cached.get("source")) != expected_source:
        raise RuntimeError(
            f"CONVI obs encoder variance source mismatch in {cache_path}"
        )

    variance = torch.as_tensor(cached.get("variance"), dtype=torch.float32)
    bias = torch.as_tensor(cached.get("bias"), dtype=torch.float32)
    if variance.shape != (channels,) or bias.shape != (channels,):
        raise RuntimeError(
            f"CONVI obs encoder residual cache must contain {channels} channels"
        )
    if (
        not torch.isfinite(variance).all()
        or not torch.isfinite(bias).all()
        or (variance < 0).any()
    ):
        raise RuntimeError(f"invalid CONVI obs encoder residual cache: {cache_path}")

    metadata = convi_obs_encoder_residual_metadata(
        config, dataset, era_dir, encoder_model_dir, checkpoint
    )
    if cached.get("metadata") != metadata:
        raise RuntimeError(
            "CONVI obs encoder residual variance metadata mismatch in "
            f"{cache_path}; regenerate the cache"
        )
    scale = float(config.get("var_y_scale", 1.0))
    return (variance.to(device) * scale).clamp_min(floor)


def load_convi_obs_encoder_latent_at_time(
    time_str,
    dataset,
    encoder,
    era_mean,
    era_std,
    era_resize,
    device,
    mask_frac=0.0,
    normalized_surface_values=None,
    full_surface_mask=False,
):
    """Load one five-channel CONVI frame and return selected ERA latent."""
    latent_std, latent_mask, pixel_counts = encode_convi_obs_encoder_latent(
        dataset,
        encoder,
        era_mean,
        era_std,
        era_resize,
        np.datetime64(time_str, "h"),
        device,
        mask_frac=mask_frac,
        return_pixel_counts=True,
        normalized_surface_values=normalized_surface_values,
        full_surface_mask=full_surface_mask,
    )
    if not bool(latent_mask.any()):
        return None
    return {
        "m_lat": latent_mask,
        "y_sel": latent_std[0, latent_mask, :].reshape(-1),
        "physical_pixel_counts": pixel_counts,
    }
