import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from appa.save import safe_load

from .config import anchor_times, load_config
from .data import latent_mask
from .inference import atomic_save, load_sources, precision_context
from .reproduction import default_config, json_write, read_config
from .run import file_identity, freeze


class ResidualMoments:
    def __init__(self, pixels, channels):
        self.sum = torch.zeros(pixels, channels, dtype=torch.float64)
        self.square = torch.zeros_like(self.sum)
        self.count = torch.zeros(pixels, dtype=torch.int64)
        self.channel_sum = torch.zeros(channels, dtype=torch.float64)
        self.channel_square = torch.zeros_like(self.channel_sum)

    def update(self, residual, channel_residual, mask):
        mask = mask.cpu().bool()
        residual, channel_residual = residual.cpu().double(), channel_residual.cpu().double()
        if residual.shape != self.sum.shape or channel_residual.shape != self.sum.shape:
            raise ValueError("Residual shape differs from the latent grid")
        if (
            not torch.isfinite(residual[mask]).all()
            or not torch.isfinite(channel_residual[mask]).all()
        ):
            raise ValueError("Nonfinite residual at an observed latent position")
        self.sum[mask] += residual[mask]
        self.square[mask] += residual[mask].square()
        self.count += mask.long()
        self.channel_sum += channel_residual[mask].sum(0)
        self.channel_square += channel_residual[mask].square().sum(0)

    def finish(self, minimum_count=2):
        if self.count.sum() < 2:
            raise ValueError(
                "Calibration requires at least two observed latent residuals per source"
            )
        n = self.count.sum()
        channel_bias = self.channel_sum / n
        channel_mse = self.channel_square / n
        channel_variance = (channel_mse - channel_bias.square()).clamp_min(0)
        denominator = self.count.clamp_min(1)[:, None]
        bias = self.sum / denominator
        mse = self.square / denominator
        variance = (mse - bias.square()).clamp_min(0)
        fallback = self.count < minimum_count
        variance[fallback] = channel_variance
        mse[fallback] = channel_mse
        return {
            "variance_raw": variance.float(),
            "mse_raw": mse.float(),
            "bias_raw": bias.float(),
            "counts": self.count,
            "fallback": fallback,
            "channel_variance_raw": channel_variance.float(),
            "channel_mse_raw": channel_mse.float(),
        }


def file_stat(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def provenance(config, assimilation, times):
    latent = Path(assimilation.prior.latent_dir)
    paths = [
        latent / "stats.pth",
        latent / "ae/config.yaml",
        Path(assimilation.prior.model_dir) / "config.yaml",
    ]
    for name in assimilation.sources:
        spec = assimilation.encoders[name]
        paths.extend(
            [
                Path(spec.model_dir) / "config.yaml",
                Path(spec.model_dir) / f"model_{spec.checkpoint}.pth",
            ]
        )
        tag = "era5" if name == "convi" else name
        paths.extend(
            Path(assimilation.data.normalization) / f"{prefix}_{tag}.npy"
            for prefix in ("mean", "std")
        )
    return {
        "config": OmegaConf.to_container(config, resolve=True),
        "assimilation": OmegaConf.to_container(assimilation, resolve=True),
        "files": [file_identity(path) for path in sorted(set(paths))],
        "latent_file": file_stat(config.latent_path),
        "times": [t.isoformat() for t in times],
        "code": file_identity(__file__),
    }


@torch.inference_mode()
def calibrate_source(source, times, truth, truth_index, ae, raw_std, config, device):
    shape = tuple(int(v) for v in ae.latent_shape)
    moments = ResidualMoments(shape[0] * shape[1], shape[2])
    channel_precision = (
        config.convi_channel_precision
        if source["name"] == "convi"
        else config.satellite_channel_precision
    )
    observed, missing, empty = [], [], []
    for i, time in enumerate(times):
        raw = source["data"].read(time)
        if raw is None:
            missing.append(time.isoformat())
            continue
        inputs, mask = raw
        if tuple(inputs.shape[-2:]) != tuple(ae.shape):
            raise ValueError("Observation and ERA5 grids differ")
        support = latent_mask(mask.to(device), int(ae.resize))
        if not support.any():
            empty.append(time.isoformat())
            continue
        target = torch.as_tensor(truth[truth_index[time]], device=device, dtype=torch.float32)
        if tuple(target.shape) != (shape[0] * shape[1], shape[2]):
            raise ValueError("Expected native ERA latents with shape (H_lat*W_lat,C_lat)")
        inputs = inputs.to(device)[None]
        with precision_context(config.spatial_precision, device):
            predicted = source["encoder"](inputs)[0].float()
        residual = (predicted - target) / raw_std
        if channel_precision == config.spatial_precision:
            channel_residual = residual
        else:
            with precision_context(channel_precision, device):
                channel_prediction = source["encoder"](inputs)[0].float()
            channel_residual = (channel_prediction - target) / raw_std
        moments.update(residual, channel_residual, support)
        observed.append(time.isoformat())
        if i % 100 == 0:
            print(f"{source['name']}: {i + 1}/{len(times)} hours", flush=True)
    return {
        **moments.finish(int(config.minimum_count)),
        "observed_times": observed,
        "missing_times": missing,
        "empty_times": empty,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate PPGF+VCLG from masked Encoder residuals"
    )
    parser.add_argument("--config", type=Path, default=default_config("calibration.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    config = read_config(args.config, overrides)
    reference = Path(config.assimilation_config)
    if not reference.is_absolute():
        reference = args.config.resolve().parent / reference
    assimilation = load_config(reference)
    times = anchor_times(config.timeline)
    if int(config.timeline.step_hours) != 1 or config.minimum_count < 2:
        raise ValueError("Use hourly calibration and minimum_count >= 2")
    if times[-1] >= anchor_times(assimilation.timeline)[0]:
        raise ValueError("Calibration must end before the first evaluation anchor")
    for precision in (
        config.spatial_precision,
        config.satellite_channel_precision,
        config.convi_channel_precision,
    ):
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported calibration precision")
    print(f"Calibration: {len(times)} hourly candidates, sources={list(assimilation.sources)}")
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small verification")
    torch.set_num_threads(1)
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    record = provenance(config, assimilation, times)
    output = Path(config.output)
    freeze(output / "inputs.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    ae = OmegaConf.load(Path(assimilation.prior.latent_dir) / "ae/config.yaml").ae
    prior = OmegaConf.load(Path(assimilation.prior.model_dir) / "config.yaml")
    stats = safe_load(Path(assimilation.prior.latent_dir) / "stats.pth")
    raw_std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device).reshape(1, -1)
    if not torch.isfinite(raw_std).all() or (raw_std <= 0).any():
        raise ValueError("Invalid native latent standard deviation")
    noise = float(ae.get("noise_level", 0)) if prior.train.get("inject_ae_noise", False) else 0.0
    coordinate_scale = (raw_std.square() / (raw_std.square() + noise**2)).cpu()
    results = []
    with h5py.File(config.latent_path, "r") as handle:
        dates = [datetime(*(int(v) for v in row)) for row in handle["dates"][:]]
        if len(dates) != len(set(dates)) or len(dates) != len(handle["latents"]):
            raise ValueError("Latent timestamps must be unique and match the latent count")
        truth_index = {date: i for i, date in enumerate(dates)}
        missing = [time for time in times if time not in truth_index]
        if missing:
            raise ValueError(f"ERA latent calibration coverage is incomplete: {missing[0]}")
        for name in assimilation.sources:
            one_source = OmegaConf.create(OmegaConf.to_container(assimilation, resolve=True))
            one_source.sources = [name]
            source = load_sources(one_source, ae, device)[0]
            groups = (
                [source["data"].surf, source["data"].icoa]
                if name == "convi"
                else [source["data"].paths]
            )
            observations = [file_stat(paths[t]) for paths in groups for t in times if t in paths]
            freeze(output / f"{name}_observations.json", json.dumps(observations, indent=2) + "\n")
            result = calibrate_source(
                source, times, handle["latents"], truth_index, ae, raw_std, config, device
            )
            current_observations = [
                file_stat(paths[t]) for paths in groups for t in times if t in paths
            ]
            if current_observations != observations:
                raise RuntimeError("Observation files changed during calibration")
            result["source"] = name
            atomic_save(result, output / f"{name}_moments.pth")
            results.append(result)
            del source
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if provenance(config, assimilation, times) != record:
        raise RuntimeError("Calibration model or latent inputs changed during computation")
    shape = tuple(int(v) for v in ae.latent_shape)
    shared = {
        "sources": list(assimilation.sources),
        "latent_shape": list(shape),
        "calibration_start": str(config.timeline.start),
        "calibration_end": str(config.timeline.end),
    }
    variance = torch.stack([r["variance_raw"] * coordinate_scale for r in results])
    reliability = torch.stack([r["mse_raw"] for r in results])
    atomic_save(
        {
            **shared,
            "variance": variance.reshape(len(results), *shape),
            "coordinate": "diffusion_standardized",
        },
        config.variance_output,
    )
    atomic_save(
        {
            **shared,
            "reliability": reliability.reshape(len(results), *shape),
            "coordinate": "raw_std_standardized",
            "lower_is_more_reliable": True,
        },
        config.reliability_output,
    )
    json_write(
        {
            **shared,
            "candidate_hours": len(times),
            "sources_used": {
                r["source"]: {
                    "observed_hours": len(r["observed_times"]),
                    "missing_hours": len(r["missing_times"]),
                    "empty_hours": len(r["empty_times"]),
                    "fallback_positions": int(r["fallback"].sum()),
                }
                for r in results
            },
            "variance": file_identity(config.variance_output),
            "reliability": file_identity(config.reliability_output),
        },
        output / "summary.json",
    )
    print(f"Variance: {config.variance_output}\nReliability: {config.reliability_output}")


if __name__ == "__main__":
    main()
