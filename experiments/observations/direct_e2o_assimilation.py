"""Direct E2O decoder adapters for physical-TB MMPS assimilation."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from omegaconf import OmegaConf

from appa.data.datasets import SatelliteAEDataset
from appa.save import safe_load, safe_save
from experiments.observations.direct_e2o_data import (
    DIRECT_E2O_DATA_CONTRACT,
    EraLatentSatelliteTbDataset,
    load_tb_target_at_time,
    validate_direct_e2o_data_contract,
)
from risda.models.decoders import EraLatentToTbDecoder


DIRECT_E2O_RESIDUAL_CACHE_KIND = "direct_e2o_residual_cos_satzen_v2"


def _checkpoint_path(model_dir, checkpoint):
    checkpoint = str(checkpoint)
    if checkpoint in {"best", "last"}:
        return Path(model_dir) / f"model_{checkpoint}.pth"
    path = Path(checkpoint)
    return path if path.is_absolute() else Path(model_dir) / path


def _same_path(left, right):
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def build_direct_e2o_decoder(model_config, out_channels, device):
    """Reconstruct a trained Direct E2O decoder from its saved config."""
    ae_path = Path(model_config.era_latent_dir) / "ae" / "config.yaml"
    if not ae_path.is_file():
        raise FileNotFoundError(f"ERA autoencoder config not found: {ae_path}")
    autoencoder = OmegaConf.load(ae_path).get("ae")
    if autoencoder is None:
        raise KeyError(f"{ae_path} does not contain an 'ae' config")
    latent_shape = tuple(int(value) for value in autoencoder.latent_shape)
    model = EraLatentToTbDecoder(
        out_channels=int(out_channels),
        lat_channels=latent_shape[2],
        shape=tuple(int(value) for value in autoencoder.shape),
        resize=int(autoencoder.resize),
        zenith_hidden_channels=int(
            model_config.model.zenith_hidden_channels
        ),
        hid_channels=tuple(model_config.model.hid_channels),
        hid_blocks=tuple(model_config.model.hid_blocks),
        kernel_size=int(model_config.model.kernel_size),
        ffn_factor=int(model_config.model.get("ffn_factor", 1)),
        dropout=float(model_config.model.get("dropout", 0.0)),
        checkpointing=bool(model_config.model.get("checkpointing", False)),
        pixel_shuffle=bool(model_config.model.get("pixel_shuffle", True)),
        identity_init=bool(model_config.model.get("identity_init", True)),
    )
    return model.to(device)


def load_direct_e2o_source(
    model_dir, checkpoint, expected_latent_dir, expected_latent_shape, device
):
    """Load one decoder and the real-TB dataset used by its training run."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Direct E2O config not found: {config_path}")
    model_config = OmegaConf.load(config_path)
    validate_direct_e2o_data_contract(model_config.data)
    if not _same_path(model_config.era_latent_dir, expected_latent_dir):
        raise RuntimeError(
            "Direct E2O decoder was trained for a different ERA latent: "
            f"{model_config.era_latent_dir} != {expected_latent_dir}"
        )

    data_config = model_config.data
    satellite = SatelliteAEDataset(
        data_path=data_config.sate_data_path,
        sate_name=str(data_config.sate_name),
        standard_dir=data_config.sate_standard_dir,
        blanket_size=1,
        time_range=None,
        use_satzen=True,
    )
    decoder = build_direct_e2o_decoder(
        model_config, satellite.obs_channel, device
    )
    decoder_shape = (
        decoder.latent_height * decoder.latent_width,
        decoder.lat_channels,
    )
    if tuple(expected_latent_shape) != decoder_shape:
        raise RuntimeError(
            "Direct E2O decoder latent shape mismatch: "
            f"{decoder_shape} != {tuple(expected_latent_shape)}"
        )

    checkpoint_path = _checkpoint_path(model_dir, checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Direct E2O checkpoint not found: {checkpoint_path}"
        )
    decoder.load_state_dict(
        safe_load(checkpoint_path, map_location=device), strict=True
    )
    decoder.eval().requires_grad_(False)
    time_to_index = {
        (int(t.year), int(t.month), int(t.day), int(t.hour)): index
        for index, t in enumerate(satellite.times)
    }
    return {
        "model_dir": model_dir,
        "model_config": model_config,
        "checkpoint_path": checkpoint_path,
        "decoder": decoder,
        "sate_dataset": satellite,
        "time_to_idx": time_to_index,
    }


def prepare_direct_e2o_observation(target, valid, cos_satzen, device):
    """Compact dense TB and its pixel-aligned cosine zenith condition."""
    target = torch.as_tensor(target, dtype=torch.float32)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    cos_satzen = torch.as_tensor(cos_satzen, dtype=torch.float32)
    if target.shape != valid.shape or target.ndim != 3:
        raise ValueError(
            "Direct E2O target and validity must share shape (C,H,W); "
            f"got {tuple(target.shape)} and {tuple(valid.shape)}"
        )
    expected_zenith_shape = (1, *target.shape[1:])
    if tuple(cos_satzen.shape) != expected_zenith_shape:
        raise ValueError(
            "Direct E2O cos_satzen must have shape (1,H,W); "
            f"expected {expected_zenith_shape}, got {tuple(cos_satzen.shape)}"
        )
    if not torch.isfinite(cos_satzen).all():
        raise ValueError("Direct E2O cos_satzen must be finite")

    target = target.movedim(0, -1)
    valid = valid.movedim(0, -1)
    cos_satzen = cos_satzen.movedim(0, -1)
    pixel_mask = valid.any(dim=-1)
    pixel_index = torch.nonzero(
        pixel_mask.reshape(-1), as_tuple=False
    ).flatten()
    if pixel_index.numel() == 0:
        return None

    channels = target.shape[-1]
    target_points = target.reshape(-1, channels).index_select(0, pixel_index)
    valid_points = valid.reshape(-1, channels).index_select(0, pixel_index)
    cos_satzen_points = cos_satzen.reshape(-1, 1).index_select(
        0, pixel_index
    )
    channel_local = (
        torch.arange(channels)
        .view(1, channels)
        .expand(pixel_index.numel(), channels)[valid_points]
    )
    values = target_points[valid_points]
    return {
        "pixel_mask": pixel_mask,
        "pixel_index": pixel_index.to(device, non_blocking=True),
        "valid_mask": valid_points.to(device, non_blocking=True),
        "cos_satzen": cos_satzen_points.to(device, non_blocking=True),
        "channel_local": channel_local.to(device, non_blocking=True),
        "values": values.to(device, non_blocking=True),
    }


def load_direct_e2o_at_time(
    time_str, sate_dataset, time_to_idx, device
):
    """Load compact real standardized TB values for one assimilation time."""
    timestamp = pd.Timestamp(np.datetime64(time_str, "h"))
    key = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
    )
    index = time_to_idx.get(key)
    if index is None:
        return None
    center_time = sate_dataset.times[index]
    target, valid, cos_satzen = load_tb_target_at_time(
        sate_dataset, center_time
    )
    return prepare_direct_e2o_observation(
        target, valid, cos_satzen, device
    )


def select_direct_e2o_prediction(prediction, valid_mask):
    """Flatten compact decoder output in the target's channel-valid order."""
    if prediction.ndim != 3:
        raise ValueError("prediction must have shape (batch,pixels,channels)")
    if tuple(prediction.shape[1:]) != tuple(valid_mask.shape):
        raise ValueError(
            "Direct E2O prediction selection mismatch: "
            f"{tuple(prediction.shape[1:])} != {tuple(valid_mask.shape)}"
        )
    expanded_valid = valid_mask.unsqueeze(0).expand(
        prediction.shape[0], -1, -1
    )
    return torch.masked_select(prediction, expanded_valid)


def finalize_residual_moments(sum_error, sum_square_error, counts):
    """Return per-channel residual MSE and centered population variance."""
    sum_error = torch.as_tensor(sum_error, dtype=torch.float64)
    sum_square_error = torch.as_tensor(sum_square_error, dtype=torch.float64)
    counts = torch.as_tensor(counts, dtype=torch.int64)
    if not (sum_error.shape == sum_square_error.shape == counts.shape):
        raise ValueError("residual moment tensors must share a shape")
    if (counts <= 0).any():
        missing = torch.nonzero(counts <= 0).flatten().tolist()
        raise RuntimeError(f"no Direct E2O residuals for channels {missing}")
    bias = sum_error / counts
    mse = sum_square_error / counts
    return {
        "bias": bias,
        "mse": mse,
        "variance": (mse - bias.square()).clamp_min(0),
        "counts": counts,
    }


def _precision_context(device, precision):
    precision = str(precision).lower()
    if device.type != "cuda" or precision in {"float32", "fp32"}:
        return nullcontext()
    if precision in {"float16", "fp16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    if precision in {"bfloat16", "bf16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported Direct E2O precision: {precision!r}")


def calibrate_direct_e2o_residuals(
    decoder, model_config, checkpoint_path, device, max_frames=64,
    valid_frac=0.05,
):
    """Calibrate physical-TB residual moments on the standard tail slice."""
    dataset = EraLatentSatelliteTbDataset(
        era_latent_path=(
            Path(model_config.era_latent_dir)
            / str(model_config.data.get("latent_file", "era5.h5"))
        ),
        cadence_hours=int(model_config.data.cadence_hours),
        satellite_data_path=model_config.data.sate_data_path,
        satellite_name=model_config.data.sate_name,
        satellite_standard_dir=model_config.data.sate_standard_dir,
    )
    num_tail = max(1, int(round(len(dataset) * float(valid_frac))))
    selected_indices = list(
        range(len(dataset) - num_tail, len(dataset))
    )[: int(max_frames)]
    channels = int(dataset.n_obs)
    sum_error = torch.zeros(channels, dtype=torch.float64)
    sum_square_error = torch.zeros(channels, dtype=torch.float64)
    counts = torch.zeros(channels, dtype=torch.int64)
    selected_times = []
    precision = str(model_config.train.get("precision", "float16"))

    with torch.no_grad():
        for frame_number, index in enumerate(selected_indices, start=1):
            latent, target, valid, cos_satzen, date = dataset[index]
            latent = latent.unsqueeze(0).to(device, non_blocking=True)
            target = target.unsqueeze(0).to(device, non_blocking=True)
            valid = valid.unsqueeze(0).to(device, non_blocking=True)
            cos_satzen = cos_satzen.unsqueeze(0).to(
                device, non_blocking=True
            )
            with _precision_context(device, precision):
                prediction = decoder(latent, cos_satzen)
            residual = prediction.float() - target.float()
            masked = torch.where(valid, residual, 0.0)
            sum_error += masked.sum(dim=(0, 1)).double().cpu()
            sum_square_error += masked.square().sum(dim=(0, 1)).double().cpu()
            counts += valid.sum(dim=(0, 1)).cpu()
            year, month, day, hour = date.tolist()
            selected_times.append(
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}"
            )
            if (
                frame_number == 1
                or frame_number % 8 == 0
                or frame_number == len(selected_indices)
            ):
                print(
                    "[direct-e2o] residual calibration "
                    f"{model_config.data.sate_name}: "
                    f"frame {frame_number}/{len(selected_indices)}",
                    flush=True,
                )

    moments = finalize_residual_moments(
        sum_error, sum_square_error, counts
    )
    return {
        **moments,
        "kind": DIRECT_E2O_RESIDUAL_CACHE_KIND,
        "data_contract": DIRECT_E2O_DATA_CONTRACT,
        "source": str(model_config.data.sate_name),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_mtime_ns": Path(checkpoint_path).stat().st_mtime_ns,
        "selected_times": selected_times,
        "num_aligned_frames": len(dataset),
        "num_tail_frames": num_tail,
        "valid_frac": float(valid_frac),
        "max_frames": int(max_frames),
    }


def _cache_matches(
    cached, source, checkpoint_path, expected_channels, max_frames, valid_frac
):
    return (
        cached.get("kind") == DIRECT_E2O_RESIDUAL_CACHE_KIND
        and cached.get("data_contract") == DIRECT_E2O_DATA_CONTRACT
        and str(cached.get("source")) == str(source)
        and str(cached.get("checkpoint_path"))
        == str(Path(checkpoint_path).resolve())
        and int(cached.get("checkpoint_mtime_ns", -1))
        == Path(checkpoint_path).stat().st_mtime_ns
        and int(cached.get("max_frames", -1)) == int(max_frames)
        and float(cached.get("valid_frac", -1.0)) == float(valid_frac)
        and torch.as_tensor(cached.get("variance", [])).shape
        == (int(expected_channels),)
    )


def get_direct_e2o_var_y(
    source, var_y_mode, obs_covariance, var_y_scale,
    use_residual_var_y, max_frames, valid_frac, device,
):
    """Load or calibrate the Direct E2O TB-space observation variance."""
    channels = int(source["sate_dataset"].obs_channel)
    floor = float(obs_covariance)
    mode = str(var_y_mode).lower()
    if mode == "scalar":
        return torch.full((channels,), floor, device=device)
    if mode != "rmse":
        raise ValueError(f"Unknown var_y_mode: {mode!r}")

    cache_path = Path(
        source.get("var_y_cache")
        or source["model_dir"] / "direct_e2o_residual_var_y.pth"
    )
    checkpoint_path = source["checkpoint_path"]
    source_name = source["sate_dataset"].sate_name
    rank = dist.get_rank() if dist.is_initialized() else 0
    cached = None
    if rank == 0 and cache_path.is_file():
        candidate = safe_load(cache_path, map_location="cpu")
        if _cache_matches(
            candidate, source_name, checkpoint_path, channels,
            max_frames, valid_frac,
        ):
            cached = candidate
    if rank == 0 and cached is None:
        cached = calibrate_direct_e2o_residuals(
            source["decoder"],
            source["model_config"],
            checkpoint_path,
            device,
            max_frames=max_frames,
            valid_frac=valid_frac,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        safe_save(cached, cache_path)
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        cached = safe_load(cache_path, map_location="cpu")
    if not _cache_matches(
        cached, source_name, checkpoint_path, channels,
        max_frames, valid_frac,
    ):
        raise RuntimeError(f"invalid Direct E2O variance cache: {cache_path}")

    statistic = "variance" if use_residual_var_y else "mse"
    raw = torch.as_tensor(cached[statistic], dtype=torch.float32, device=device)
    result = (raw * float(var_y_scale)).clamp_min(floor)
    if rank == 0:
        print(
            f"[direct-e2o] {source_name} var_y={statistic}; "
            f"frames={len(cached['selected_times'])}; cache={cache_path}; "
            f"mean={result.mean().item():.6g}; "
            f"range=[{result.min().item():.6g}, {result.max().item():.6g}]",
            flush=True,
        )
    return result
