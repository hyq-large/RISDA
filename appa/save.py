r"""Save and load tools for autoencoder models."""

import shutil
import torch

from omegaconf import OmegaConf, open_dict
from pathlib import Path
from typing import Union

from appa.config.hydra import compose
from appa.diffusion import Denoiser, create_denoiser
from appa.nn.cae import ConvAE, conv_ae
from appa.nn.gae import GraphAE
from appa.nn.triggers import skip_init


def safe_save(obj: object, path: Union[str, Path]) -> None:
    r"""Safely save an object to a file using torch.save

    Arguments:
        obj: Object to save.
        path: Path to save the object.
    """

    path = Path(path)

    if path.exists():
        path_prev = path.with_suffix(".prev.pth")
        try:
            shutil.copy2(path, path_prev)
        except (FileNotFoundError, OSError):
            pass

    torch.save(obj, path)


def safe_load(path: Union[str, Path], map_location: str = "cpu") -> object:
    r"""Safely load an object from a file using torch.load

    Arguments:
        path: Path to file to save
        cpu: Whether to load the model on CPU.
    """

    path = Path(path)

    try:
        return torch.load(path, weights_only=False, map_location=map_location)
    except Exception:
        # If loading fails, try to load the previous file
        path_prev = path.with_suffix(".prev.pth")
        return torch.load(path_prev, weights_only=False, map_location=map_location)


def select_ae_architecture(name_ae: str):
    if "ico" in name_ae:
        return GraphAE
    elif "cae" in name_ae:
        return conv_ae
    else:
        raise NotImplementedError(f"Unknown ae type with name {name_ae}")


def load_auto_encoder(
    path: Path,
    model_name: str = "model_best",
    device: str = "cuda",
    eval_mode: bool = False,
    eval_noise_level: float = 0.0,
) -> Union[GraphAE, ConvAE]:
    r"""Load and return a trained autoencoder model (eval state).

    Arguments:
        path: Directory where autoencoder model is stored.
        model_name: Model file name (e.g., "model_best", "model_last", "model").
        device: Device to load the model on.
        eval: Whether to set the model in eval mode.
        eval_noise_level: Noise level to use if eval is True (default is 0.0). If None, uses the noise level from training.
    """

    with open(path / "config.yaml", "r") as file:
        config_ae = OmegaConf.load(file).get("ae")
        name_ae = config_ae.pop("name", None)

        config_ae.pop("latent_shape", None)

        if eval_mode:
            with open_dict(config_ae):
                config_ae.checkpointing = True
                if eval_noise_level is not None:
                    config_ae.noise_level = eval_noise_level

        # Backward compatible with run cg33oqw8
        config_ae.setdefault("context_channels", 1)

    # Model weights checkpoint
    ckpt_path = path / f"{model_name}.pth"
    checkpoint = safe_load(ckpt_path, map_location=device)

    ae_arch = select_ae_architecture(name_ae)
    if eval_mode:
        with skip_init():
            ae = ae_arch(**config_ae)
    else:
        ae = ae_arch(**config_ae)
    ae.load_state_dict(checkpoint)

    if eval_mode:
        ae.eval()

    return ae


def load_denoiser(
    path: Path, best: bool = True, device: str = "cuda", overrides: dict = None,name=None,
    ckpt_path=None,
) -> Denoiser:
    r"""Load and return a trained denoiser model (eval state).

    Arguments:
        path: Directory where denoiser model is stored.
        best: Whether to load the best model or the last one saved.
        device: Device to load the model on.
        name: If given, load model_step_{name*2000}.pth.
        ckpt_path: If given (absolute, or relative to `path`), load THIS checkpoint
            file directly, overriding best/name. Lets you point at an arbitrary
            weight file, e.g. a backed-up old best (model_best_121repro.pth).
    """

    with open(path.parents[3] / "config.yaml", "r") as file:
        ae_cfg = OmegaConf.load(file)
        ae_cfg.pop("name", None)

        # Backward compatible with run cg33oqw8
        ae_cfg.setdefault("ae.context_channels", 1)

    diffusion_cfg = compose(path / "config.yaml")

    denoiser = create_denoiser(
        diffusion_cfg,
        ae_cfg,
        device=device,
        overrides=overrides,
    )

    # Model weights checkpoint. Priority: explicit ckpt_path > step name > best/last.
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = path / ckpt_path
    else:
        model_name = "model_best.pth" if best else "model_last.pth"
        if name is not None:
            model_name = f"model_step_{int(name)*2000}.pth"
        ckpt_path = path / model_name
    print(f'Loading ckpt from {ckpt_path}')
    checkpoint = safe_load(ckpt_path, map_location=device)
    denoiser.backbone.load_state_dict(checkpoint)
    denoiser.backbone.eval()

    return denoiser
