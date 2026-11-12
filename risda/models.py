import inspect
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from appa.diffusion import Denoiser, LogLinearSchedule, LogLogitSchedule, RectifiedSchedule
from appa.nn.cae import ConvDecoder, ConvEncoder
from appa.nn.triggers import skip_init
from appa.nn.vit import ViDiT
from appa.save import safe_load
from appa.transforms import cressman_interp_multi_channel


CRESSMAN_SURFACE_RAW_CHANNELS = 8


CRESSMAN_SURFACE_ENCODER_CHANNELS = 9


def resolve_surface_preprocessing(config):
    if config is None:
        return None
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if isinstance(config, str):
        raw = {"name": config}
    elif isinstance(config, dict):
        if "data" in config:
            config = config["data"]
        raw = config.get("surface_preprocessing", config if "name" in config else None)
    else:
        raise TypeError(
            f"surface preprocessing must be a mapping, string, or None; got {type(config).__name__}"
        )
    if raw is None:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True, enum_to_str=True)
    if isinstance(raw, str):
        raw = {"name": raw}
    raw = dict(raw)
    name = str(raw.get("name", "none")).lower()
    if name in {"", "none", "disabled"}:
        return None
    if name != "cressman":
        raise ValueError(f"unsupported surface preprocessing: {name!r}")
    result = {
        "name": name,
        "radius": int(raw.get("radius", 10)),
        "eps": float(raw.get("eps", 0.0001)),
        "confidence_ignore": int(raw.get("confidence_ignore", 0)),
    }
    if result["radius"] < 1:
        raise ValueError("Cressman radius must be positive")
    if result["eps"] <= 0.0:
        raise ValueError("Cressman eps must be positive")
    if not 0 <= result["confidence_ignore"] < 4:
        raise ValueError("Cressman confidence_ignore must be in [0, 4)")
    return result


def obs2era_encoder_raw_input_channels(config, fallback):
    preprocessing = resolve_surface_preprocessing(config)
    if preprocessing is not None:
        return CRESSMAN_SURFACE_RAW_CHANNELS
    return int(fallback)


class CressmanSurfacePreprocessor(nn.Module):
    def __init__(self, radius=10, eps=0.0001, confidence_ignore=0):
        super().__init__()
        self.radius = int(radius)
        self.eps = float(eps)
        self.confidence_ignore = int(confidence_ignore)

    def forward(self, observations):
        if observations.ndim != 4:
            raise ValueError(
                f"Cressman surface input must have shape (B,8,H,W); got {tuple(observations.shape)}"
            )
        if observations.shape[1] != CRESSMAN_SURFACE_RAW_CHANNELS:
            raise ValueError(
                f"Cressman surface input must contain [4 values | 4 masks]; got {observations.shape[1]} channels"
            )
        original_dtype = observations.dtype
        with torch.autocast(device_type=observations.device.type, enabled=False):
            values = observations[:, :4].float().unsqueeze(1)
            masks = observations[:, 4:8].float().unsqueeze(1)
            (values_with_confidence, dilated_masks) = cressman_interp_multi_channel(
                values,
                masks,
                confidence_ignore=self.confidence_ignore,
                R=self.radius,
                eps=self.eps,
            )
            prepared = torch.cat(
                (
                    values_with_confidence[:, 0, :4],
                    dilated_masks[:, 0],
                    values_with_confidence[:, 0, 4:5],
                ),
                dim=1,
            )
        return prepared.to(dtype=original_dtype)


class ObsToEraEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        lat_channels,
        shape,
        resize,
        saturation_bound=5.0,
        surface_preprocessing=None,
        **encoder_kwargs,
    ):
        super().__init__()
        preprocessing = resolve_surface_preprocessing(surface_preprocessing)
        self.raw_in_channels = int(in_channels)
        self.surface_preprocessing = preprocessing
        self.surface_preprocessing_name = (
            preprocessing["name"] if preprocessing is not None else None
        )
        if preprocessing is None:
            self.surface_preprocessor = None
            learned_in_channels = self.raw_in_channels
        else:
            if self.raw_in_channels != CRESSMAN_SURFACE_RAW_CHANNELS:
                raise ValueError(
                    f"Cressman Encoder expects eight raw channels [4 values | 4 masks]; got {self.raw_in_channels}"
                )
            self.surface_preprocessor = CressmanSurfacePreprocessor(
                radius=preprocessing["radius"],
                eps=preprocessing["eps"],
                confidence_ignore=preprocessing["confidence_ignore"],
            )
            learned_in_channels = CRESSMAN_SURFACE_ENCODER_CHANNELS
        self.learned_in_channels = learned_in_channels
        (height, width) = shape
        factor = int(resize)
        padded_size = (height + -height % factor, width + -width % factor + 2 * factor)
        self.encoder = ConvEncoder(
            in_channels=learned_in_channels,
            out_channels=lat_channels,
            channel_groups=None,
            padded_size=padded_size,
            **encoder_kwargs,
        )
        self.shape = tuple(shape)
        self.resize = int(resize)
        self.saturation_bound = float(saturation_bound)

    def _pad(self, values):
        (height, width) = self.shape
        factor = self.resize
        pad_h = -height % factor
        pad_w = -width % factor + 2 * factor
        values = F.pad(values, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode="circular")
        return F.pad(values, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode="constant")

    def forward(self, observations):
        if self.surface_preprocessor is not None:
            observations = self.surface_preprocessor(observations)
        hidden = self.encoder(self._pad(observations))
        latent = hidden * torch.rsqrt(1 + torch.square(hidden / self.saturation_bound))
        return rearrange(latent, "b c h w -> b (h w) c")


def load_obs2era_encoder(model_dir, era_autoencoder_config, in_channels, checkpoint, device):
    model_dir = Path(model_dir)
    config = OmegaConf.load(model_dir / "config.yaml")
    preprocessing = resolve_surface_preprocessing(config)
    raw_in_channels = obs2era_encoder_raw_input_channels(config, in_channels)
    model = ObsToEraEncoder(
        in_channels=raw_in_channels,
        lat_channels=int(era_autoencoder_config.latent_shape[2]),
        shape=tuple(era_autoencoder_config.shape),
        resize=int(era_autoencoder_config.resize),
        hid_channels=tuple(config.model.hid_channels),
        hid_blocks=tuple(config.model.hid_blocks),
        kernel_size=int(config.model.kernel_size),
        ffn_factor=int(config.model.get("ffn_factor", 1)),
        dropout=0.0,
        surface_preprocessing=preprocessing,
        checkpointing=False,
        pixel_shuffle=bool(config.model.get("pixel_shuffle", True)),
        identity_init=bool(config.model.get("identity_init", True)),
    )
    checkpoint_name = "model_best.pth" if str(checkpoint) == "best" else "model_last.pth"
    state = safe_load(model_dir / checkpoint_name, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


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
