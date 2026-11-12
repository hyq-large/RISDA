import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from appa.save import safe_load

from .config import anchor_times
from .inference import timestamp
from .reproduction import CHANNELS, EraFields, default_config, json_write, read_config
from .run import file_identity


def squared_totals(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and ERA5 reference must share shape (C,H,W)")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction or ERA5 reference")
    weights = np.maximum(np.cos(np.linspace(np.pi / 2, -np.pi / 2, target.shape[-2])), 0)
    error = prediction.astype(np.float64) - target.astype(np.float64)
    total = (error**2 * weights[None, :, None]).sum(axis=(-2, -1))
    weight = np.full(target.shape[0], weights.sum() * target.shape[-1], dtype=np.float64)
    return total, weight


def evaluate(run_dir, fields):
    config = OmegaConf.load(run_dir / "launch/config.yaml")
    inputs = json.loads((run_dir / "launch/inputs.json").read_text())
    trajectory = safe_load(run_dir / "trajectory.pth")
    times = anchor_times(config.timeline)
    expected_dates = torch.stack([timestamp(time) for time in times])
    if trajectory.get("contract") != inputs["contract"] or not torch.equal(
        trajectory["dates"], expected_dates
    ):
        raise ValueError("Completed trajectory does not match the configured evaluation times")
    if len(trajectory["latents"]) != len(times):
        raise ValueError("Incomplete trajectory")
    fields.require(times)
    squared, weight = np.zeros(69, dtype=np.float64), np.zeros(69, dtype=np.float64)
    records = []
    for time in times:
        path = run_dir / "era5" / f"{time:%Y%m%dT%H}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing decoded analysis; run with save_decoded=true: {path}")
        prediction = np.load(path)
        total, count = squared_totals(prediction, fields.read(time))
        squared += total
        weight += count
        records.append(
            {
                "time": time.isoformat(),
                "prediction": file_identity(path),
                "reference": file_identity(fields.paths[time]),
            }
        )
    if (weight <= 0).any():
        raise ValueError("No valid evaluation weights")
    return np.sqrt(squared / weight), {
        "generation_contract": inputs["contract"],
        "num_analyses": len(times),
        "squared_error_sum": squared.tolist(),
        "weight_sum": weight.tolist(),
        "inputs": records,
        "latitude_weighting": "cosine",
        "coordinate": "physical",
        "channels": list(CHANNELS),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate every retained analysis against exact-hour ERA5"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, default=default_config("data.yaml"))
    args = parser.parse_args()
    config = read_config(args.data_config)
    result, report = evaluate(args.run_dir, EraFields(config.era5))
    output = args.run_dir / "metrics"
    output.mkdir(exist_ok=True)
    with (output / "era69_rmse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "rmse"])
        writer.writerows(zip(CHANNELS, result, strict=True))
    lines = [f"ERA5 RMSE; physical units; cosine latitude; analyses={report['num_analyses']}"]
    lines.extend(f"{name:<10} {value:.9g}" for name, value in zip(CHANNELS, result, strict=True))
    (output / "era69_rmse.log").write_text("\n".join(lines) + "\n")
    json_write(
        {
            **report,
            "rmse": result.tolist(),
            "data_config": OmegaConf.to_container(config, resolve=True),
        },
        output / "era69_rmse.json",
    )
    print(output / "era69_rmse.log")


if __name__ == "__main__":
    main()
