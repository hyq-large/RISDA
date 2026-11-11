import inspect
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

from appa.diffusion import Denoiser, LogLinearSchedule, LogLogitSchedule, RectifiedSchedule
from appa.nn.cae import ConvDecoder
from appa.nn.triggers import skip_init
from appa.nn.vit import ViDiT
from appa.save import safe_load


class EraDecoder(nn.Module):
    def __init__(self, ae_config, checkpoint, device):
        super().__init__()
        options = OmegaConf.to_container(ae_config, resolve=True)
        self.shape = tuple(options["shape"])
        self.latent_shape = tuple(options["latent_shape"])
        height, width = self.shape
        resize = int(options["resize"])
        self.pad_h = (-height) % resize
        self.pad_w = (-width) % resize + 2 * resize
        allowed = set(inspect.signature(ConvDecoder).parameters)
        kwargs = {key: value for key, value in options.items() if key in allowed}
        kwargs.update(
            in_channels=self.latent_shape[-1],
            out_channels=int(options.get("out_channels") or options["in_channels"]),
            dropout=0.0,
            checkpointing=False,
            padded_size=(height + self.pad_h, width + self.pad_w),
        )
        with skip_init():
            self.decoder = ConvDecoder(**kwargs)
        weights = safe_load(checkpoint)
        decoder_weights = {
            key.removeprefix("decoder."): value
            for key, value in weights.items()
            if key.startswith("decoder.")
        }
        self.decoder.load_state_dict(decoder_weights, strict=True)
        self.to(device).eval().requires_grad_(False)

    def forward(self, native_latent):
        height, width, channels = self.latent_shape
        values = native_latent.reshape(-1, height, width, channels).permute(0, 3, 1, 2)
        values = self.decoder(values)
        top, left = self.pad_h // 2, self.pad_w // 2
        return values[..., top : top + self.shape[0], left : left + self.shape[1]]


def load_prior(directory, checkpoint, latent_dir, device):
    directory, latent_dir = Path(directory), Path(latent_dir)
    config = OmegaConf.load(directory / "config.yaml")
    ae = OmegaConf.load(latent_dir / "ae/config.yaml").ae
    if "cae" not in str(ae.name) or bool(ae.get("vae", False)):
        raise ValueError("This runtime requires the deterministic convolutional ERA5 autoencoder")
    backbone = OmegaConf.to_container(config.backbone, resolve=True)
    if backbone.get("arch", "vit") != "vit" or "cond" in str(backbone.get("name", "")):
        raise ValueError("This runtime requires the unconditional ViDiT prior")
    if int(backbone.get("blanket_size", 0)) != 6:
        raise ValueError("The prior must use six-frame windows")
    if int(config.train.get("blanket_dt", 1)) != 1 or not config.train.get("standardize", True):
        raise ValueError("The prior must use hourly standardized latent states")
    backbone.update(checkpointing=True, dropout=0.0)
    with skip_init():
        network = ViDiT(
            in_channels=int(ae.latent_shape[-1]), shape=tuple(ae.latent_shape[:2]), **backbone
        )
    network.load_state_dict(safe_load(directory / f"model_{checkpoint}.pth"), strict=True)
    network.to(device).eval().requires_grad_(False)
    stats = safe_load(latent_dir / "stats.pth")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device).reshape(1, 1, -1)
    raw_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, 1, -1)
    if mean.numel() != int(ae.latent_shape[-1]) or raw_std.shape != mean.shape:
        raise ValueError("Latent statistics do not match the prior channels")
    if not torch.isfinite(mean).all() or not torch.isfinite(raw_std).all() or (raw_std <= 0).any():
        raise ValueError("Invalid latent normalization")
    noise = float(ae.get("noise_level", 0)) if config.train.get("inject_ae_noise", False) else 0.0
    std = torch.sqrt(raw_std.square() + noise**2)
    schedules = {
        "rectified": RectifiedSchedule,
        "loglinear": LogLinearSchedule,
        "loglogit": LogLogitSchedule,
    }
    schedule = schedules[str(config.train.noise_schedule)](
        sigma_min=float(config.train.sigma_min),
        sigma_max=float(config.train.sigma_max),
    ).to(device)
    return Denoiser(network), schedule, ae, mean, std
