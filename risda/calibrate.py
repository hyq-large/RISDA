#!/usr/bin/env python
"""Paper Encoder residual calibration, with explicit dates and resumable shards.

The spatial calibration kernel supplies the unmodified data, mask, Encoder,
standardization and moment routines. No diffusion sampling or training occurs.
Satellite channelwise moments use a separate fp16 forward, as in the actual
paper caches; spatial moments and all CONVI moments use fp32.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/assimilation.yaml"
OUTPUT = ROOT / "outputs/calibration/2025-01-01T00--2025-03-31T23"
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from appa.config.hydra import compose
from experiments.observations import (
    calibration_spatial as paper,
)

SOURCES = ("atms", "amsua", "mhs", "mwts", "mwhs", "convi")
from risda.run import PRESETS
EXPERIMENTS = tuple(PRESETS)
START, END = "2025-01-01T00", "2025-03-31T23"
PERIOD = f"{START}--{END}"
FIELDS = ("bias", "mse", "variance", "std", "rmse")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def hourly_axis(start=None, end=None):
    start, end = start or START, end or END
    first = datetime.strptime(start, "%Y-%m-%dT%H")
    last = datetime.strptime(end, "%Y-%m-%dT%H")
    if last < first:
        raise ValueError("Calibration must have start <= end")
    return [first + timedelta(hours=i) for i in range(int((last-first).total_seconds()/3600)+1)]


def requested_times(smoke=False):
    times = hourly_axis()
    if smoke:
        times = [times[i] for i in np.linspace(0, len(times)-1, min(16, len(times)), dtype=int)]
    return times


def stage_dir(smoke=False):
    return OUTPUT / 'smoke' if smoke else OUTPUT

def prepare():
    """Freeze the actual paper assets before starting any residual inference."""
    reference = REFERENCE
    cfg = compose(reference)
    OmegaConf.resolve(cfg)
    if tuple(cfg.enabled_sources) != SOURCES:
        raise ValueError('Calibration source order must match assimilation_config.enabled_sources')
    if cfg.calibration_period != PERIOD or Path(cfg.calibration_root).resolve() != OUTPUT:
        raise ValueError('Calibration output/period must match assimilation configuration')
    model_specs = {}
    for source in SOURCES:
        entry = cfg.source_catalog[source].encoder
        model_dir = Path(entry.model_dir).resolve()
        model_cfg = OmegaConf.load(model_dir / "config.yaml")
        latent_dir = Path(model_cfg.era_latent_dir).resolve()
        if latent_dir != Path(cfg.model_path).resolve().parents[1]:
            raise ValueError(f'{source} uses a different latent directory from the prior')
        if source == "convi":
            for key in ("surf_dir", "icoa_dir", "mean_std_dir"):
                model_cfg.data[key] = cfg.convi[key]
            assert model_cfg.data.alignment_mode == "convi_qc_to_convi_qc"
            assert model_cfg.data.surface_preprocessing.name == "cressman"
            assert int(model_cfg.data.surface_preprocessing.radius) == 10
        identities = {
            "checkpoint": paper.file_identity(model_dir / f"model_{entry.checkpoint}.pth", include_sha256=True),
            "model_config": paper.file_identity(model_dir / "config.yaml", include_sha256=True),
            "era_stats": paper.file_identity(latent_dir / "stats.pth", include_sha256=True),
            "era_latents": paper.file_identity(latent_dir / "era5.h5"),
            "ae_config": paper.file_identity(latent_dir / "ae/config.yaml", include_sha256=True),
        }
        # Retain normalization-file identities in addition to their paths.
        normalization = {}
        for key in ("mean_std_dir", "sate_standard_dir"):
            if model_cfg.data.get(key):
                path = Path(model_cfg.data[key])
                for p in sorted(path.glob("*.npy")):
                    normalization[str(p)] = paper.file_identity(p, include_sha256=True)
        model_specs[source] = {
            "model_dir": str(model_dir), "checkpoint": str(entry.checkpoint),
            "identities": identities, "normalization_files": normalization,
            "data": OmegaConf.to_container(model_cfg.data, resolve=True),
            "spatial_precision": "float32",
            "channelwise_precision": "float32" if source == "convi" else "float16",
        }
    assert len({v["identities"]["era_stats"]["sha256"] for v in model_specs.values()}) == 1
    protocol = {
        "version": 1, "period": PERIOD, "window_start": START, "window_end": END,
        "time_scope": "hourly", "expected_hours": len(hourly_axis()),
        "sources": list(SOURCES), "experiments": list(EXPERIMENTS),
        "mask_frac": 0.0, "minimum_spatial_count": 2, "var_y_scale": 1,
        "likelihood_statistic": "centered_population_variance",
        "ordering_statistic": "standardized_encoder_residual_mse_including_bias",
        "ppgf_order_mode": "fixed", "retrain": False,
        "models": model_specs,
        "reference": paper.file_identity(reference, include_sha256=True),
        "implementation": {
            "runner": paper.file_identity(Path(__file__), include_sha256=True),
            "paper_calibrator": paper.file_identity(Path(paper.__file__), include_sha256=True),
        },
    }
    path = OUTPUT / "protocol/calibration_config.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError(f"Frozen protocol differs from current files: {path}")
    save_json(protocol, path)
    return protocol


def check_identities(spec):
    for identity in (*spec["identities"].values(), *spec["normalization_files"].values()):
        now = paper.file_identity(Path(identity["path"]), include_sha256="sha256" in identity)
        if now != identity:
            raise RuntimeError(f"Calibration input changed: {identity['path']}")


def shard_path(directory, source, rank, world_size):
    return directory / "work" / source / f"rank_{rank:02d}_of_{world_size:02d}.pth"


def empty_totals(pixels, channels):
    return {
        "sum": torch.zeros(pixels, channels, dtype=torch.float64),
        "square": torch.zeros(pixels, channels, dtype=torch.float64),
        "counts": torch.zeros(pixels, dtype=torch.int64),
        "channel_sum": torch.zeros(channels, dtype=torch.float64),
        "channel_square": torch.zeros(channels, dtype=torch.float64),
    }


def accumulate(totals, residual32, channel_residual, mask):
    if not torch.isfinite(residual32[mask]).all() or not torch.isfinite(channel_residual[mask]).all():
        raise RuntimeError("Non-finite residual at an observed latent position")
    totals["sum"][mask] += residual32[mask]
    totals["square"][mask] += residual32[mask].square()
    totals["counts"] += mask.long()
    totals["channel_sum"] += channel_residual[mask].sum(0)
    totals["channel_square"] += channel_residual[mask].square().sum(0)


def finalize(totals, shape):
    h, w, c = shape
    spatial = paper.finalize_spatial_moments(totals["sum"], totals["square"], totals["counts"])
    spatial = {k: v.reshape(h, w) if k == "counts" else v.reshape(h, w, c) for k, v in spatial.items()}
    channel = paper.finalize_channelwise_moments_from_spatial_sums(
        totals["channel_sum"].unsqueeze(0), totals["channel_square"].unsqueeze(0),
        totals["counts"].sum().reshape(1),
    )
    pooled_fp32 = paper.finalize_channelwise_moments_from_spatial_sums(
        totals["sum"], totals["square"], totals["counts"],
    )
    for moments in (spatial, channel, pooled_fp32):
        valid = moments["counts"] > 0
        assert torch.allclose(moments["mse"][valid], (moments["variance"] + moments["bias"].square())[valid], rtol=3e-5, atol=3e-7)
    return spatial, channel, pooled_fp32


def worker(args):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    protocol = json.loads((OUTPUT / "protocol/calibration_config.json").read_text())
    spec = protocol["models"][args.source]
    check_identities(spec)
    directory = stage_dir(args.smoke)
    times = requested_times(args.smoke)
    indices = list(range(args.rank, len(times), args.world_size))
    model_dir = Path(spec["model_dir"])
    model_cfg = OmegaConf.load(model_dir / "config.yaml")
    ae_cfg = OmegaConf.load(spec["identities"]["ae_config"]["path"]).ae
    shape = tuple(int(v) for v in ae_cfg.latent_shape)
    h, w, channels = shape
    contract = {"protocol": protocol, "source": args.source, "rank": args.rank,
                "world_size": args.world_size, "smoke": args.smoke,
                "requested_times": [paper.format_hour(t) for t in times]}
    path = shard_path(directory, args.source, args.rank, args.world_size)
    state = {"contract": contract, "next_offset": 0, "complete": False,
             "totals": empty_totals(h*w, channels), "frames": []}
    if path.exists():
        state = load(path)
        if state["contract"] != contract:
            raise RuntimeError(f"Shard contract changed: {path}")
        if state["complete"]:
            print(f"reuse complete {path}", flush=True)
            return
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    kind, dataset, in_channels = paper._build_dataset(args.source, OmegaConf.create(spec["data"]))
    observation_times = dataset.dates if kind == "convi_aligned" else dataset.times
    obs_index = paper._time_index(observation_times)
    encoder = paper.load_obs2era_encoder(model_dir, ae_cfg, in_channels, spec["checkpoint"], device)
    encoder.eval()
    stats = load(spec["identities"]["era_stats"]["path"])
    mean = stats["mean"].float().to(device).reshape(1, 1, -1)
    std = stats["std"].float().to(device).reshape(1, 1, -1)
    assert torch.isfinite(std).all() and (std > 0).all()
    totals = {k: v.to(device) for k, v in state["totals"].items()}
    started = time.monotonic()

    def checkpoint(complete=False):
        state["totals"] = {k: v.cpu() for k, v in totals.items()}
        state["complete"] = complete
        state["updated_utc"] = utc_now()
        save_torch(state, path)
        print(f"{args.source} rank {args.rank}: {state['next_offset']}/{len(indices)} "
              f"hours, {time.monotonic()-started:.1f}s, complete={complete}", flush=True)

    with h5py.File(spec["identities"]["era_latents"]["path"], "r") as truth_file, torch.inference_mode():
        era_index = paper._time_index(np.asarray(truth_file["dates"]))
        for offset in range(state["next_offset"], len(indices)):
            timestamp = times[indices[offset]]
            key = paper.timestamp_key(timestamp)
            if key not in era_index:
                raise RuntimeError(f"Missing ERA latent truth: {timestamp}")
            frame = {"timestamp": paper.format_hour(timestamp), "observation_present": key in obs_index,
                     "supported_blocks": 0}
            if key in obs_index:
                observation, pixel_mask = paper._load_observation(kind, dataset, obs_index[key])
                fraction = paper._obs_mask_to_frac(pixel_mask.unsqueeze(0).to(device), int(ae_cfg.resize))
                mask = fraction > 0
                frame["supported_blocks"] = int(mask.sum())
                if frame["supported_blocks"]:
                    observation = observation.unsqueeze(0).to(device)
                    truth = torch.as_tensor(truth_file["latents"][era_index[key]], dtype=torch.float32, device=device).unsqueeze(0)
                    prediction32 = encoder(observation).float()
                    residual32 = paper.centered_standardized_residual(prediction32, truth, mean, std)[0].double()
                    if spec["channelwise_precision"] == "float16":
                        with torch.autocast("cuda", dtype=torch.float16):
                            prediction16 = encoder(observation)
                        channel_residual = paper.centered_standardized_residual(prediction16.float(), truth, mean, std)[0].double()
                    else:
                        channel_residual = residual32
                    accumulate(totals, residual32, channel_residual, mask)
            state["frames"].append(frame)
            state["next_offset"] = offset + 1
            if state["next_offset"] % 24 == 0:
                checkpoint()
    check_identities(spec)
    checkpoint(complete=True)


def write_csv(rows, path):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fill_spatial(raw, counts, channel):
    return torch.where((counts >= 2).unsqueeze(-1), raw, channel.reshape(1, 1, -1))


def merge_source(source, world_size, smoke, protocol):
    directory = stage_dir(smoke)
    spec = protocol["models"][source]
    check_identities(spec)
    totals = None
    frames = []
    for rank in range(world_size):
        state = load(shard_path(directory, source, rank, world_size))
        assert state["complete"] and state["contract"]["protocol"] == protocol
        assert state["contract"]["source"] == source and state["contract"]["rank"] == rank
        expected_times = [paper.format_hour(t) for t in requested_times(smoke)[rank::world_size]]
        assert [f["timestamp"] for f in state["frames"]] == expected_times
        if totals is None:
            totals = {k: v.clone() for k, v in state["totals"].items()}
        else:
            for k in totals:
                totals[k] += state["totals"][k]
        frames.extend(state["frames"])
    frames.sort(key=lambda f: f["timestamp"])
    assert [f["timestamp"] for f in frames] == [paper.format_hour(t) for t in requested_times(smoke)]
    assert int(totals["counts"].sum()) == sum(f["supported_blocks"] for f in frames)
    shape = tuple(OmegaConf.load(spec["identities"]["ae_config"]["path"]).ae.latent_shape)
    spatial, channel, pooled32 = finalize(totals, shape)
    data = spec["data"]
    metadata = {
        **spec["identities"], "model_dir": spec["model_dir"], "model_data": data,
        "precision": "float32; no autocast", "channelwise_precision": spec["channelwise_precision"],
        "period": PERIOD, "first_time": START, "last_time": END,
        "standardization": "((prediction-mean)/raw_std)-((truth-mean)/raw_std)",
        "calibration_latent_truth": "timestamp-aligned ERA5 native latent",
        "selection_mask": "original QC union pooled to latent; mask_frac > 0",
        "surface_preprocessing": data.get("surface_preprocessing"),
        "training_alignment_mode": data.get("alignment_mode"),
        "calibration_alignment_mode": "convi_qc_to_era5_qc_mask" if source == "convi" else None,
    }
    common = {
        "source": source, "period": PERIOD, "window_start": START, "window_end": END,
        "window_end_inclusive": True, "time_scope": "hourly", "expected_hours": len(hourly_axis()),
        "smoke": smoke, "num_requested_times": len(frames), "mask_frac": 0.0,
        "latent_shape": shape, "checkpoint": spec["checkpoint"], "metadata": metadata,
        "selected_times": [f["timestamp"] for f in frames if f["observation_present"]],
        "used_timestamps": [f["timestamp"] for f in frames if f["supported_blocks"]],
        "missing_observation_timestamps": [f["timestamp"] for f in frames if not f["observation_present"]],
        "zero_support_timestamps": [f["timestamp"] for f in frames if f["observation_present"] and not f["supported_blocks"]],
        "num_used_times": sum(f["supported_blocks"] > 0 for f in frames),
    }
    raw = {**common, "kind": "hourly_encoder_residual_spatial_v1", "layout": "(H_lat,W_lat,C_lat)",
           **spatial, "channelwise": channel, "channelwise_from_spatial_fp32": pooled32}
    channel_payload = {
        **common, "kind": "hourly_encoder_centered_residual_var_y_v1",
        "representation": "encoder", "space": "standardized_era_latent", **channel,
        "model_dir": spec["model_dir"], "checkpoint_path": spec["identities"]["checkpoint"]["path"],
        "checkpoint_mtime_ns": spec["identities"]["checkpoint"]["mtime_ns"],
        "precision": spec["channelwise_precision"],
        "variance_definition": "centered population variance: E[e^2]-E[e]^2; denominator N",
    }
    save_torch(raw, directory / "raw" / f"{source}_spatial.pth")
    save_torch(channel_payload, directory / "channelwise" / f"{source}_var_y.pth")
    write_csv(frames, directory / "raw" / f"{source}_hourly_counts.csv")
    return raw, channel_payload


def build_bundle(world_size, smoke, protocol):
    directory = stage_dir(smoke)
    rows, channel_rows, raws, channels = [], [], [], []
    for source in SOURCES:
        raw, channel = merge_source(source, world_size, smoke, protocol)
        raws.append(raw)
        channels.append(channel)
        counts = raw["counts"]
        rows.append({"source": source, "requested_hours": raw["num_requested_times"],
                     "used_hours": raw["num_used_times"], "missing_hours": len(raw["missing_observation_timestamps"]),
                     "zero_support_hours": len(raw["zero_support_timestamps"]),
                     "supported_block_times": int(counts.sum()), "spatial_positions_ge2": int((counts >= 2).sum()),
                     "fallback_positions": int((counts < 2).sum()),
                     "channel_variance_mean": float(channel["variance"].mean()),
                     "channel_mse_mean": float(channel["mse"].mean()),
                     "channel_bias_squared_mean": float(channel["bias"].square().mean())})
        for c in range(len(channel["variance"])):
            channel_rows.append({"source": source, "channel": c, "count": int(channel["counts"][c]),
                                 **{k: float(channel[k][c]) for k in FIELDS}})
    common = {"sources": list(SOURCES), "period": PERIOD, "window_start": START, "window_end": END,
              "time_scope": "hourly", "expected_hours": len(hourly_axis()), "smoke": smoke,
              "mask_frac": 0.0, "latent_shape": raws[0]["latent_shape"], "source_axis": 0}
    channel_bundle = {**common, "kind": "encoder_channelwise_calibration_bundle_v1",
                      **{k: torch.stack([ch[k] for ch in channels]) for k in (*FIELDS, "var_y", "counts")}}
    spatial_bundle = {**common, "kind": "mask_frac_obs_encoder_spatial_uncertainty_v1",
                      **{k: torch.stack([r[k] for r in raws]) for k in (*FIELDS, "counts")},
                      "layout": "(source,H_lat,W_lat,C_lat)", "ordering_ready": False,
                      "era_stats": protocol["models"][SOURCES[0]]["identities"]["era_stats"],
                      "source_metadata": {s:r["metadata"] for s,r in zip(SOURCES,raws)},
                      "missing_value_policy": "count<2 uses same-source channelwise centered variance"}
    mse = torch.stack([fill_spatial(r["mse"], r["counts"], ch["mse"]) for r,ch in zip(raws,channels)])
    assert torch.isfinite(mse).all() and (mse > 0).all()
    ordering = {**common, "kind": "mask_frac_obs_encoder_spatial_channel_ordering_v1",
                "ordering_ready": True, "ordering_only": True, "ordering_metric": "standardized_residual_mse_including_bias",
                "source_statistic": "mse", "includes_bias": True, "smaller_is_more_reliable": True,
                "transport_field": "variance", "transport_field_semantics": "legacy schema alias containing MSE",
                "minimum_count": 2, "fallback_policy": "same_source_channelwise_residual_mse",
                "variance": mse, "mse": mse.clone(), "counts": spatial_bundle["counts"],
                "filled_position_count_by_source": {row["source"]:row["fallback_positions"] for row in rows}}
    artifacts = {"encoder_channelwise.pth": channel_bundle,
                 "encoder_spatial_var_y.pth": spatial_bundle,
                 "encoder_spatial_mse_ordering.pth": ordering}
    channel_mse = channel_bundle['mse'].float()
    artifacts['encoder_channel_mse_ordering.pth'] = {
        **common, 'kind': 'mask_frac_obs_encoder_channel_uncertainty',
        'ordering_ready': True, 'ordering_only': True,
        'source_statistic': 'mse', 'includes_bias': True,
        'ordering_metric': 'standardized_residual_mse_including_bias',
        'smaller_is_more_reliable': True, 'variance': channel_mse, 'mse': channel_mse.clone(),
    }
    for name, value in artifacts.items():
        save_torch(value, directory / name)
    write_csv(rows, directory / "summary.csv")
    write_csv(channel_rows, directory / "channelwise_errors.csv")
    manifest = {"created_utc": utc_now(), "status": "complete", "protocol": protocol, "smoke": smoke,
                "summary": rows, "outputs": {name:paper.file_identity(directory/name,include_sha256=True) for name in artifacts}}
    for source in SOURCES:
        for name in (f"channelwise/{source}_var_y.pth", f"raw/{source}_spatial.pth"):
            manifest["outputs"][name] = paper.file_identity(directory/name, include_sha256=True)
    save_json(manifest, directory / "manifest.json")
    print(json.dumps(rows, indent=2), flush=True)


def run(args):
    torch.set_num_threads(1)
    directory = stage_dir(args.smoke)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = prepare()
    gpus = args.gpus.split(",")
    if not gpus or len(gpus) != len(set(gpus)) or not all(s.isdigit() for s in gpus):
        raise ValueError("--gpus must be a comma-separated list of distinct device indices")
    status = {"started_utc": utc_now(), "pid": os.getpid(), "status": "running", "gpus": gpus,
              "smoke": args.smoke, "completed_sources": []}
    save_json(status, directory / "status.json")
    (directory / "logs").mkdir(exist_ok=True)
    processes = []
    try:
        for source in SOURCES:
            status.update(source=source, updated_utc=utc_now())
            save_json(status, directory / "status.json")
            processes, handles = [], []
            for rank, gpu in enumerate(gpus):
                command = [sys.executable, "-m", "risda.calibrate", "worker", "--source", source,
                           "--rank", str(rank), "--world-size", str(len(gpus))]
                if args.smoke:
                    command.append("--smoke")
                log = (directory / "logs" / f"{source}_rank{rank}.log").open("a")
                handles.append(log)
                log.write(f"\n{utc_now()} {command!r}\n")
                log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
                processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                failures = [p.returncode for p in processes if p.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f"{source} worker failed with exit codes {failures}; see {directory/'logs'}")
                time.sleep(2)
            if any(p.returncode != 0 for p in processes):
                raise RuntimeError(f"{source} failed; see {directory/'logs'}")
            for handle in handles:
                handle.close()
            merge_source(source, len(gpus), args.smoke, protocol)
            status["completed_sources"].append(source)
            save_json(status, directory / "status.json")
        build_bundle(len(gpus), args.smoke, protocol)
        status.update(status="complete", finished_utc=utc_now())
    except BaseException as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        status.update(status="failed", error=str(exc), finished_utc=utc_now())
        raise
    finally:
        save_json(status, directory / "status.json")


def main():
    from risda.calibration_config import configure
    remaining = configure(sys.modules[__name__])
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for mode in ("run", "launch"):
        p = subs.add_parser(mode)
        p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
        p.add_argument("--smoke", action="store_true")
    subs.add_parser("prepare")
    p = subs.add_parser("worker")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args(remaining)
    if args.command == "worker":
        worker(args)
    elif args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args)
    else:
        directory = stage_dir(args.smoke)
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "risda.calibrate", "run", "--gpus", args.gpus]
        if args.smoke:
            command.append("--smoke")
        with (directory / "run.log").open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"))
        (directory / "run.pid").write_text(str(process.pid) + "\n")
        print(f"Started calibration PID {process.pid}; logs: {directory/'run.log'}")


if __name__ == "__main__":
    main()
