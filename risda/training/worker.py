import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from ..reproduction import hourly_times, read_config
from .data import EraDataset, LatentWindows, ObservationDataset
from .engine import seed_all, train_loop
from .models import AEProblem, EncoderProblem, PriorProblem, prepare_ae


def prepare_config(config, stage):
    OmegaConf.set_struct(config, False)
    if int(config.seed) != 0:
        raise ValueError("The reproduction configuration uses base seed 0")
    train_times = hourly_times(*config.dates.training)
    valid_times = hourly_times(*config.dates.validation)
    if max(train_times[0], valid_times[0]) <= min(train_times[-1], valid_times[-1]):
        raise ValueError("Training and validation intervals must not overlap")
    if stage == "ae":
        prepare_ae(config)
        if list(config.data.era5.grid_shape) != list(config.ae.shape):
            raise ValueError("ERA5 data and autoencoder grid shapes must match")
    elif stage == "prior":
        if not config.train.standardize or config.train.blanket_dt != 1:
            raise ValueError("The prior requires standardized hourly latents")
        if config.backbone.blanket_size != 6 or config.train.anchor_frame != 4 or config.train.anchor_hour_interval != 6:
            raise ValueError("The prior uses six frames anchored at frame four every six hours")
    return config


def build_problem(config, stage, device):
    if stage == "prior":
        ae = OmegaConf.load(Path(config.latent_dir) / "ae/config.yaml").ae
        path = Path(config.latent_dir) / config.latent_data
        datasets = [LatentWindows(path, config.dates[key], ae.latent_shape)
                    for key in ("training", "validation")]
        problem = PriorProblem(config)
    elif stage == "ae":
        datasets = [EraDataset(config.data, config.dates[key]) for key in ("training", "validation")]
        problem = AEProblem(config)
    else:
        ae = OmegaConf.load(Path(config.era_latent_dir) / "ae/config.yaml").ae
        if tuple(ae.shape) != tuple(config.data.era5.grid_shape):
            raise ValueError("ERA5 data and frozen decoder grid shapes must match")
        datasets = [ObservationDataset(config.data, config.dates[key], stage)
                    for key in ("training", "validation")]
        problem = EncoderProblem(config, stage, datasets[0].input_channels, device)
    return problem, *datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("ae", "prior", "satellite", "convi"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args, overrides = parser.parse_known_args()
    config = prepare_config(read_config(args.config, overrides), args.stage)
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        seed_all(int(config.seed))
        problem, training, validation = build_problem(config, args.stage, device)
        train_loop(config, problem, training, validation, device, args.resume, args.stop_after)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
