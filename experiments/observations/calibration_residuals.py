"""Normalized observation residuals and the frozen ERA5 physical decoder."""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from appa.save import load_auto_encoder
from experiments.observations.convi_physical import physical_normalization_contract
from experiments.observations.direct_e2o_assimilation import (
    build_direct_e2o_decoder, finalize_residual_moments,
)
from experiments.observations.direct_e2o_data import (
    EraLatentSatelliteTbDataset, validate_direct_e2o_data_contract,
)

def timestamp_key(value):
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 4:
        return tuple(int(part) for part in value)
    timestamp = pd.Timestamp(value)
    return timestamp.year, timestamp.month, timestamp.day, timestamp.hour

def precision_context(device, precision):
    precision = str(precision).lower()
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    if precision == "float16":
        return torch.autocast("cuda", dtype=torch.float16)
    if precision == "bfloat16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported precision: {precision!r}")

def decode_era_latent(
    decoder,
    latent,
    *,
    latent_hw,
    output_hw,
    resize,
):
    """Decode native ERA latents and remove the AE's spatial padding."""
    if latent.ndim != 3:
        raise ValueError(
            f"ERA latent must have shape (B,spatial,channels); got {latent.shape}"
        )
    latent_h, latent_w = (int(value) for value in latent_hw)
    output_h, output_w = (int(value) for value in output_hw)
    if latent.shape[1] != latent_h * latent_w:
        raise ValueError(
            f"ERA latent spatial size {latent.shape[1]} does not match "
            f"latent_hw={latent_hw}"
        )

    latent_grid = latent.transpose(1, 2).reshape(
        latent.shape[0], latent.shape[2], latent_h, latent_w
    )
    decoded = decoder(latent_grid)
    pad_h = (-output_h) % int(resize)
    pad_w = (-output_w) % int(resize) + 2 * int(resize)
    start_h = pad_h // 2
    start_w = pad_w // 2
    decoded = decoded[
        ...,
        start_h : start_h + output_h,
        start_w : start_w + output_w,
    ]
    if tuple(decoded.shape[-2:]) != (output_h, output_w):
        raise RuntimeError(
            "ERA decoder output cannot be cropped to the configured grid: "
            f"got {tuple(decoded.shape[-2:])}, expected {(output_h, output_w)}"
        )
    return decoded.flatten(2).transpose(1, 2)

def load_era_physical_decoder(era_latent_dir, device):
    """Load only the frozen ERA AE decoder needed by physical CONVI."""
    ae_dir = Path(era_latent_dir) / "ae"
    config_path = ae_dir / "config.yaml"
    checkpoint_file = ae_dir / "model.pth"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)

    autoencoder = load_auto_encoder(
        ae_dir,
        "model",
        device="cpu",
        eval_mode=True,
        eval_noise_level=0.0,
    )
    latent_shape = tuple(int(value) for value in autoencoder.latent_shape)
    if len(latent_shape) != 3:
        raise ValueError(
            "physical CONVI decoding requires a 3-D ERA AE latent shape; "
            f"got {latent_shape}"
        )
    latent_h, latent_w, latent_channels = latent_shape
    output_hw = tuple(int(value) for value in autoencoder.shape)
    if len(output_hw) != 2:
        raise ValueError(f"ERA AE output shape must be two-dimensional: {output_hw}")
    resize = int(autoencoder.resize)
    decoder = autoencoder.decoder.to(device)
    decoder.eval().requires_grad_(False)
    if hasattr(autoencoder, "encoder"):
        autoencoder.encoder = None
    del autoencoder
    return (
        decoder,
        (latent_h, latent_w),
        latent_channels,
        output_hw,
        resize,
        ae_dir,
        config_path,
        checkpoint_file,
    )
