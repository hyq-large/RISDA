import inspect
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from appa.diffusion import Denoiser, LogLinearSchedule, LogLogitSchedule, RectifiedSchedule
from appa.nn.cae import ConvDecoder, ConvEncoder
from appa.nn.vit import ViDiT
from appa.save import safe_load

from ..encoders import ObsToEraEncoder, resolve_surface_preprocessing
from ..models import EraDecoder
from ..reproduction import LEVELS


def prepare_ae(config):
    ae = config.ae
    if ae.get("vae", False) or ae.saturation != "softclip2":
        raise ValueError("Training requires a deterministic softclip2 autoencoder")
    if int(ae.resize) != 2 ** (len(ae.hid_channels) - 1) or list(ae.patch_size) != [1, 1]:
        raise ValueError("Resize must match the convolutional stages with patch_size=[1,1]")
    height, width = map(int, ae.shape)
    factor = int(ae.resize)
    ae.in_channels = 69
    ae.latent_shape = [
        (height + (-height) % factor) // factor,
        (width + (-width) % factor) // factor + 2,
        int(ae.lat_channels),
    ]
    return ae


class Autoencoder(nn.Module):
    def __init__(self, ae):
        super().__init__()
        options = OmegaConf.to_container(ae, resolve=True)
        self.shape = tuple(ae.shape)
        self.latent_shape = tuple(ae.latent_shape)
        self.noise = float(ae.noise_level)
        factor = int(ae.resize)
        self.pad_h = (-self.shape[0]) % factor
        self.pad_w = (-self.shape[1]) % factor + 2 * factor
        allowed = set(inspect.signature(ConvEncoder).parameters)
        kwargs = {k: v for k, v in options.items() if k in allowed}
        kwargs.update(padded_size=(self.shape[0] + self.pad_h, self.shape[1] + self.pad_w))
        kwargs.update(in_channels=69, out_channels=int(ae.lat_channels))
        self.encoder = ConvEncoder(**kwargs)
        kwargs.update(in_channels=int(ae.lat_channels), out_channels=69)
        self.decoder = ConvDecoder(**kwargs)

    def encode(self, field):
        field = F.pad(
            field, (self.pad_w // 2, self.pad_w - self.pad_w // 2, 0, 0), mode="circular"
        )
        field = F.pad(field, (0, 0, self.pad_h // 2, self.pad_h - self.pad_h // 2))
        latent = self.encoder(field)
        latent = latent * torch.rsqrt(1 + (latent / 5).square())
        return latent.permute(0, 2, 3, 1).flatten(1, 2)

    def decode(self, latent, noise=False):
        if noise and self.noise:
            latent = latent + self.noise * torch.randn_like(latent)
        field = self.decoder(latent.reshape(-1, *self.latent_shape).permute(0, 3, 1, 2))
        top, left = self.pad_h // 2, self.pad_w // 2
        return field[..., top : top + self.shape[0], left : left + self.shape[1]]

    def forward(self, field):
        latent = self.encode(field)
        return self.decode(latent, noise=True), latent


def reconstruction_weights(shape, latitude=True, levels="v1"):
    height, width = shape
    lat = torch.cos(torch.linspace(math.pi / 2, -math.pi / 2, height)) + 1 / width
    if not latitude:
        lat = torch.ones(height)
    if levels not in ("v1", None, False):
        raise ValueError("Only the ERA69 pressure-level layout is supported")
    pressure = torch.tensor(LEVELS, dtype=torch.float32)
    channels = torch.cat([(pressure / pressure.sum()).repeat(5), torch.tensor([1, .1, .1, .1])])
    if not levels:
        channels = torch.ones(69)
    return channels[None, :, None, None] * lat[None, None, :, None]


def masked_reconstruction(prediction, target, mask):
    error = (prediction.float() - target.float()).square()
    weights = mask.float().expand_as(error)
    return (error * weights).sum() / weights.sum().clamp_min(1)


class AEProblem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Autoencoder(config.ae)
        self.register_buffer("weights", reconstruction_weights(
            config.ae.shape, config.loss.latitude_weighting, config.loss.level_weighting
        ))
        self.l2 = float(config.loss.l2_weight)
        if config.loss.error != "mse":
            raise ValueError("The reconstruction objective requires MSE")

    def forward(self, field):
        reconstructed, latent = self.model(field)
        return ((reconstructed.float() - field.float()).square() * self.weights).mean() + (
            self.l2 * latent.float().square().mean()
        )


class EncoderProblem(nn.Module):
    def __init__(self, config, stage, channels, device):
        super().__init__()
        latent = Path(config.era_latent_dir)
        ae = OmegaConf.load(latent / "ae/config.yaml").ae
        if float(config.loss.latent_weight) != 0:
            raise ValueError("This implementation uses the physical reconstruction objective")
        self.model = ObsToEraEncoder(
            in_channels=channels, lat_channels=int(ae.latent_shape[-1]),
            shape=tuple(ae.shape), resize=int(ae.resize),
            surface_preprocessing=resolve_surface_preprocessing(config) if stage == "convi" else None,
            **OmegaConf.to_container(config.model, resolve=True),
        )
        self.decoder = EraDecoder(ae, latent / "ae/model.pth", device)
        self.convi = stage == "convi"
        self.weight = float(config.loss.physical_weight)

    def train(self, mode=True):
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(self, observations, target, mask):
        prediction = self.decoder(self.model(observations))
        if self.convi:
            prediction = prediction[:, -4:]
        return self.weight * masked_reconstruction(prediction, target, mask)


class PriorProblem(nn.Module):
    def __init__(self, config):
        super().__init__()
        ae = OmegaConf.load(Path(config.latent_dir) / "ae/config.yaml").ae
        self.model = ViDiT(
            in_channels=int(ae.latent_shape[-1]), shape=tuple(ae.latent_shape[:2]),
            **OmegaConf.to_container(config.backbone, resolve=True),
        )
        stats = safe_load(Path(config.latent_dir) / "stats.pth")
        mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(1, 1, 1, -1)
        std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape_as(mean)
        if mean.numel() != int(ae.latent_shape[-1]) or not torch.isfinite(mean).all():
            raise ValueError("Invalid latent mean")
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid latent standard deviation")
        self.noise = float(ae.noise_level) if config.train.inject_ae_noise else 0
        self.register_buffer("mean", mean)
        self.register_buffer("std", torch.sqrt(std.square() + self.noise**2))
        schedules = {"rectified": RectifiedSchedule, "loglinear": LogLinearSchedule,
                     "loglogit": LogLogitSchedule}
        self.schedule = schedules[config.train.noise_schedule](
            sigma_min=float(config.train.sigma_min), sigma_max=float(config.train.sigma_max)
        )
        self.penalty = float(config.loss.sigma_min_penalty)

    def forward(self, native, dates):
        if self.noise:
            native = native + self.noise * torch.randn_like(native)
        clean = ((native - self.mean) / self.std).flatten(1)
        sigma = self.schedule(torch.rand(clean.shape[0], device=clean.device))
        noisy = clean + sigma * torch.randn_like(clean)
        estimate = Denoiser(self.model)(noisy, sigma, date=dates)
        error = (estimate.float() - clean).square().mean(-1)
        return (error * (1 + (sigma.squeeze(-1) + self.penalty).pow(-2))).mean()
