import argparse
import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from appa.save import safe_load

from .config import ROOT, anchor_times, file_identity, freeze, load_config, setup
from .data import ConventionalObservations, SatelliteObservations, latent_mask, load_normalization
from .guidance import Uncertainty, VCLG, sample
from .models import EraDecoder, load_obs2era_encoder, load_prior, resolve_surface_preprocessing


WINDOW_OFFSETS = (-3, -2, -1, 0, 1, 2)


def timestamp(time):
    return torch.tensor([time.year, time.month, time.day, time.hour])


def precision_context(precision, device):
    if precision == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, precision))


def load_sources(config, ae, device):
    sources = []
    for name in config.sources:
        spec = config.encoders[name]
        model_config = OmegaConf.load(Path(spec.model_dir) / "config.yaml")
        if name == "convi":
            trained_surface_only = model_config.data.get("icoa_dir") is None
            if trained_surface_only != (config.data.icoa_dir is None):
                raise ValueError("CONVI ICOADS selection must match the encoder configuration")
            data = ConventionalObservations(
                config.data.surf_dir,
                config.data.icoa_dir,
                config.data.normalization,
                ae.shape,
                channel_masks=resolve_surface_preprocessing(model_config) is not None,
            )
        else:
            if model_config.data.get("sate_name") != name:
                raise ValueError(f"Encoder source mismatch: {name}")
            data = SatelliteObservations(
                config.data.observations,
                config.data.normalization,
                name,
                model_config.data.get("use_satzen", False),
            )
        encoder = load_obs2era_encoder(
            spec.model_dir, ae, data.input_channels, spec.checkpoint, device
        )
        sources.append({"name": name, "data": data, "encoder": encoder})
    return sources


def window_observations(sources, anchor, ae, mean, std, uncertainty, device):
    pixels = int(ae.latent_shape[0]) * int(ae.latent_shape[1])
    channels = int(ae.latent_shape[-1])
    grid = torch.arange(pixels * channels, device=device).reshape(pixels, channels)
    result = []
    for source_id, source in enumerate(sources):
        indices, values, variances, counts = [], [], [], []
        for frame, offset in enumerate(WINDOW_OFFSETS):
            raw = source["data"].read(anchor + timedelta(hours=offset))
            if raw is None:
                counts.append(0)
                continue
            inputs, mask = raw
            if tuple(inputs.shape[-2:]) != tuple(ae.shape):
                raise ValueError(f"Observation grid differs from the encoder: {source['name']}")
            support = latent_mask(mask.to(device), int(ae.resize))
            if support.numel() != pixels:
                raise ValueError("Observation mask differs from the latent grid")
            if not support.any():
                counts.append(0)
                continue
            with torch.no_grad():
                latent = source["encoder"](inputs.to(device)[None])
            if tuple(latent.shape) != (1, pixels, channels):
                raise ValueError("Encoder output differs from the prior latent grid")
            normalized = (latent - mean) / std
            selected = normalized[0, support].reshape(-1)
            if not torch.isfinite(selected).all():
                raise ValueError(f"Nonfinite encoded observations: {source['name']}")
            indices.append(grid[support].reshape(-1) + frame * pixels * channels)
            values.append(selected)
            variances.append(uncertainty.variance[source_id, support].reshape(-1))
            counts.append(selected.numel())
        result.append(
            {
                "indices": torch.cat(indices)
                if indices
                else torch.empty(0, dtype=torch.long, device=device),
                "values": torch.cat(values) if values else torch.empty(0, device=device),
                "variance": torch.cat(variances) if variances else torch.empty(0, device=device),
                "counts": counts,
            }
        )
    return result


def observation_signature(sources, anchor):
    records = []
    for source in sources:
        data = source["data"]
        groups = (
            [data.surf, data.icoa] if isinstance(data, ConventionalObservations) else [data.paths]
        )
        for offset in WINDOW_OFFSETS:
            time = anchor + timedelta(hours=offset)
            for paths in groups:
                path = paths.get(time)
                if path is not None:
                    stat = path.stat()
                    records.append([str(path), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_anchor(payload, index, anchor, shape, contract, signature):
    if (
        payload.get("contract") != contract
        or payload.get("observation_signature") != signature
        or payload.get("index") != index
        or not torch.equal(payload["date"], timestamp(anchor))
    ):
        raise ValueError(f"Saved anchor inputs differ at {anchor}")
    if tuple(payload["latent"].shape) != shape or not torch.isfinite(payload["latent"]).all():
        raise ValueError(f"Invalid saved latent at {anchor}")


def run(config, run_root, contract, device, rank=0, world_size=1):
    prior, schedule, ae, mean, std = load_prior(
        config.prior.model_dir,
        config.prior.checkpoint,
        config.prior.latent_dir,
        device,
    )
    sources = load_sources(config, ae, device)
    uncertainty = Uncertainty.load(
        config.uncertainty.variance,
        config.uncertainty.reliability,
        config.sources,
        tuple(ae.latent_shape),
        scale=config.uncertainty.var_y_scale,
        floor=config.uncertainty.variance_floor,
        device=device,
    )
    order = uncertainty.priority_order()
    shape = (int(ae.latent_shape[0]) * int(ae.latent_shape[1]), int(ae.latent_shape[2]))
    decoder = None
    if config.save_decoded:
        decoder = EraDecoder(ae, Path(config.prior.latent_dir) / "ae/model.pth", device)
        physical_mean, physical_std = load_normalization(config.data.normalization, "era5", 69)
        physical_mean = torch.from_numpy(physical_mean).to(device)
        physical_std = torch.from_numpy(physical_std).to(device)
    anchors = anchor_times(config.timeline)
    for index in range(rank, len(anchors), world_size):
        anchor = anchors[index]
        path = run_root / "anchors" / f"{index:06d}.pth"
        signature = observation_signature(sources, anchor)
        if path.exists():
            payload = safe_load(path)
            validate_anchor(payload, index, anchor, shape, contract, signature)
        else:
            seed = int(config.seed) + index
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            observations = window_observations(sources, anchor, ae, mean, std, uncertainty, device)
            guidance = VCLG(prior, observations, order, int(config.sampling.solver_iterations))
            dates = torch.stack([timestamp(anchor + timedelta(hours=h)) for h in WINDOW_OFFSETS])[
                None
            ].to(device)
            generator = torch.Generator(device=device).manual_seed(seed)
            initial = torch.randn(1, 6 * shape[0] * shape[1], generator=generator, device=device)
            initial *= schedule.sigma_tmax().to(device)
            with precision_context(config.precision, device):
                sampled = sample(
                    guidance,
                    schedule,
                    initial,
                    dates,
                    steps=int(config.sampling.steps),
                    corrections=int(config.sampling.corrections),
                    delta=float(config.sampling.delta),
                )
            latent = sampled.reshape(6, *shape)[3].float().cpu()
            payload = {
                "date": timestamp(anchor),
                "index": index,
                "seed": seed,
                "latent": latent,
                "contract": contract,
                "observation_signature": signature,
                "source_value_counts": {
                    s["name"]: o["counts"] for s, o in zip(sources, observations, strict=True)
                },
            }
            atomic_save(payload, path)
            del guidance, observations, sampled
        if decoder is not None:
            output = run_root / "era5" / f"{anchor:%Y%m%dT%H}.npy"
            if not output.exists():
                with torch.no_grad(), precision_context(config.precision, device):
                    native = payload["latent"].to(device)[None] * std + mean
                    field = decoder(native).float() * physical_std + physical_mean
                if not torch.isfinite(field).all():
                    raise RuntimeError(f"Nonfinite decoded ERA5 field at {anchor}")
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, field[0].cpu().numpy())
                temporary.replace(output)
        print(f"[rank {rank}] {index + 1}/{len(anchors)} {anchor:%Y-%m-%dT%H}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        payloads = [safe_load(run_root / "anchors" / f"{i:06d}.pth") for i in range(len(anchors))]
        for i, (anchor, payload) in enumerate(zip(anchors, payloads, strict=True)):
            validate_anchor(
                payload, i, anchor, shape, contract, observation_signature(sources, anchor)
            )
        atomic_save(
            {
                "dates": torch.stack([p["date"] for p in payloads]),
                "latents": torch.stack([p["latent"] for p in payloads]),
                "coordinate": "diffusion_standardized",
                "contract": contract,
            },
            run_root / "trajectory.pth",
        )
        print(f"Completed {len(anchors)} anchors: {run_root}", flush=True)


def worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    run_root = args.config.parent.parent
    contract = json.loads((run_root / "launch/inputs.json").read_text())["contract"]
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=6))
    try:
        run(config, run_root, contract, device, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def inputs(config):
    paths = [
        Path(config.uncertainty.variance),
        Path(config.uncertainty.reliability),
        Path(config.prior.model_dir) / "config.yaml",
        Path(config.prior.model_dir) / f"model_{config.prior.checkpoint}.pth",
        Path(config.prior.latent_dir) / "stats.pth",
        Path(config.prior.latent_dir) / "ae/config.yaml",
    ]
    if config.save_decoded:
        paths.append(Path(config.prior.latent_dir) / "ae/model.pth")
    names = set(config.sources) - {"convi"}
    if "convi" in config.sources or config.save_decoded:
        names.add("era5")
    for name in names:
        paths.extend(
            Path(config.data.normalization) / f"{prefix}_{name}.npy" for prefix in ("mean", "std")
        )
    for name in config.sources:
        spec = config.encoders[name]
        paths.extend(
            (
                Path(spec.model_dir) / "config.yaml",
                Path(spec.model_dir) / f"model_{spec.checkpoint}.pth",
            )
        )
    record = {
        "config": OmegaConf.to_container(config, resolve=True),
        "files": [file_identity(path) for path in sorted(set(paths))],
        "code": {
            str(path.relative_to(ROOT)): file_identity(path)["sha256"]
            for package in ("risda", "appa")
            for path in sorted((ROOT / package).rglob("*.py"))
        },
    }
    contract = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return {"contract": contract, **record}


def logged(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as handle:
        handle.write("\n$ " + shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def main():
    setup()
    parser = argparse.ArgumentParser(description="RISDA inference with PPGF+VCLG")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/assimilation.yaml"
            if (ROOT / "configs/assimilation.yaml").is_file()
            else Path(sys.prefix) / "share/risda/assimilation.yaml"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if any("=" not in item for item in overrides):
        parser.error("Overrides must use key=value")
    config = load_config(args.config, overrides)
    run_root = Path(config.output_root).expanduser().resolve() / "PPGF+VCLG"
    frozen = run_root / "launch/config.yaml"
    command = [
        sys.executable,
        "-B",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        "--module",
        "risda.inference",
        "--config",
        str(frozen),
    ]
    print(OmegaConf.to_yaml(config, resolve=True))
    print(f"Anchors: {len(anchor_times(config.timeline))}")
    print(shlex.join(command))
    if args.dry_run:
        return
    provenance = inputs(config)
    freeze(frozen, OmegaConf.to_yaml(config, resolve=True))
    freeze(run_root / "launch/inputs.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    logged(command, run_root / "logs/inference.log")


if __name__ == "__main__":
    worker_main()
