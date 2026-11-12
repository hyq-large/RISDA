import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from appa.save import safe_load

from ..config import hourly_times, json_write, read_config
from .data import EraDataset, dataset_identity
from .engine import atomic_save, autocast, seed_all
from .models import Autoencoder


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_latents(config, device, resume=False):
    seed_all(int(config.seed))
    source, output = Path(config.ae_dir), Path(config.latent_dir)
    ae_config = OmegaConf.load(source / "config.yaml")
    checkpoint = source / f"model_{config.checkpoint}.pth"
    dataset = EraDataset(config.data, config.dates.export, require_all=True)
    if tuple(dataset.fields.config.grid_shape) != tuple(ae_config.ae.shape):
        raise ValueError("Export grid differs from the trained autoencoder")
    for key in ("target_mean", "target_std"):
        if sha256(config.data[key]) != sha256(ae_config.data[key]):
            raise ValueError("Export normalization must match AE training")
    identity = {
        "config": OmegaConf.to_container(config, resolve=True),
        "ae_config_sha256": sha256(source / "config.yaml"),
        "ae_checkpoint_sha256": sha256(checkpoint),
        "dataset": dataset_identity(dataset),
    }
    encoded = json.dumps(identity, sort_keys=True)
    output.mkdir(parents=True, exist_ok=True)
    final = output / "era5.h5"
    partial = output / "era5.partial.h5"
    if final.exists():
        with h5py.File(final, "r") as file:
            if resume and file.attrs.get("inputs") == encoded:
                print(f"Latent export already complete: {final}")
                return
        raise FileExistsError("Latent export exists; use a new latent_dir for different inputs")
    if partial.exists() and not resume:
        raise FileExistsError("Partial export exists; use --resume")
    model = Autoencoder(ae_config.ae).to(device).eval()
    model.load_state_dict(safe_load(checkpoint), strict=True)
    shape = tuple(ae_config.ae.latent_shape)
    native_shape = (shape[0] * shape[1], shape[2])
    dates = np.asarray([[t.year, t.month, t.day, t.hour] for t in dataset.times], dtype=np.int32)
    mode = "r+" if partial.exists() else "w"
    with h5py.File(partial, mode) as file:
        if mode == "w":
            file.attrs["inputs"] = encoded
            file.create_dataset("latents", (len(dataset), *native_shape), dtype="f4", chunks=(1, *native_shape))
            file.create_dataset("dates", data=dates)
            file.create_dataset("complete", data=np.zeros(len(dataset), dtype=bool))
        elif file.attrs.get("inputs") != encoded or not np.array_equal(file["dates"][:], dates):
            raise ValueError("Cannot resume export with different inputs or dates")
        missing = np.flatnonzero(~np.asarray(file["complete"], bool)).tolist()
        loader = DataLoader(
            torch.utils.data.Subset(dataset, missing), batch_size=int(config.batch_size),
            num_workers=int(config.num_workers), generator=torch.Generator().manual_seed(int(config.seed)),
        )
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                with autocast(device, config.precision):
                    native = model.encode(batch.to(device)).float().cpu().numpy()
                if not np.isfinite(native).all():
                    raise FloatingPointError("Nonfinite exported latent")
                indices = missing[offset:offset + len(native)]
                file["latents"][indices] = native
                file.flush()
                file["complete"][indices] = True
                file.flush()
                offset += len(native)
                if offset % 100 == 0 or offset == len(missing):
                    print(f"Exported {offset}/{len(missing)} remaining hours", flush=True)
    (output / "ae").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(ae_config, output / "ae/config.yaml")
    atomic_save(model.cpu().state_dict(), output / "ae/model.pth")
    partial.replace(final)
    json_write(identity, output / "export.json")


def latent_statistics(config):
    output = Path(config.latent_dir)
    ae_config = OmegaConf.load(output / "ae/config.yaml")
    bounds = hourly_times(*config.dates.statistics)
    train_bounds = hourly_times(*ae_config.dates.training)
    if bounds[0] < train_bounds[0] or bounds[-1] > train_bounds[-1]:
        raise ValueError("Latent statistics must stay within the AE training interval")
    with h5py.File(output / "era5.h5", "r") as file:
        dates = [datetime(*map(int, row)) for row in file["dates"][:]]
        index = {time: i for i, time in enumerate(dates)}
        if len(index) != len(dates) or any(t not in index for t in bounds):
            raise ValueError("Statistics require every training hour with unique dates")
        indices = sorted(index[t] for t in bounds)
        if "complete" in file and not np.asarray(file["complete"])[indices].all():
            raise ValueError("Statistics cannot use incomplete latent rows")
        count = 0
        mean = np.zeros(file["latents"].shape[-1], dtype=np.float64)
        m2 = np.zeros_like(mean)
        for start in range(0, len(indices), int(config.stats_batch_size)):
            batch = np.asarray(file["latents"][indices[start:start + int(config.stats_batch_size)]], dtype=np.float64)
            batch = batch.reshape(-1, len(mean))
            if not np.isfinite(batch).all():
                raise ValueError("Nonfinite native latent in statistics interval")
            n = len(batch)
            local_mean = batch.mean(0)
            delta = local_mean - mean
            m2 += ((batch - local_mean)**2).sum(0) + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        std = np.sqrt(m2 / count)
        if (std <= 0).any() or not np.isfinite(std).all():
            raise ValueError("Latent standard deviations must be finite and positive")
    atomic_save({
        "mean": torch.tensor(mean, dtype=torch.float32),
        "std": torch.tensor(std, dtype=torch.float32),
        "count": count, "hours": len(indices), "start": str(config.dates.statistics[0]),
        "end": str(config.dates.statistics[1]),
    }, output / "stats.pth")
    json_write({"count": count, "hours": len(indices), "start": bounds[0].isoformat(),
                "end": bounds[-1].isoformat(), "coordinate": "native", "ddof": 0},
               output / "stats.json")
    print(f"Saved training-only latent statistics from {len(indices)} hours: {output / 'stats.pth'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dump", "stats"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args, overrides = parser.parse_known_args()
    config = read_config(args.config, overrides)
    if args.stage == "dump":
        dump_latents(config, torch.device(args.device), args.resume)
    else:
        latent_statistics(config)


if __name__ == "__main__":
    main()
