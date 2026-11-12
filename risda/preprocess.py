import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .reproduction import EraFields, default_config, hourly_times, json_write, read_config
from .run import file_identity, freeze


def normalize(field, mean, std):
    if not np.isfinite(field).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Nonfinite ERA5 field or invalid target standard deviation")
    result = (field - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite standardized ERA5 field")
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Convert gridded ERA5 arrays to the frozen ERA normalization"
    )
    parser.add_argument("--config", type=Path, default=default_config("data.yaml"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    fields = EraFields(config.era5)
    times = hourly_times(args.start, args.end)
    fields.require(times)
    if args.output.resolve() == Path(config.era5.directory).resolve():
        raise ValueError("Use a separate output directory for standardized ERA5 arrays")
    mean, std = EraFields.stat(config.target_mean), EraFields.stat(config.target_std)
    if (std <= 0).any():
        raise ValueError("Target standard deviation must be positive")
    freeze(args.output / "preprocessing.yaml", OmegaConf.to_yaml(config, resolve=True))
    normalization_dir = args.output / "mean_std"
    normalization_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("era5_1h_mean.npy", mean), ("era5_1h_std.npy", std)):
        path = normalization_dir / name
        if path.exists() and not np.array_equal(np.load(path), values.reshape(-1)):
            raise ValueError("Existing output normalization differs")
        np.save(path, values.reshape(-1))
    records = []
    for time in times:
        target = normalize(fields.read(time), mean, std)
        path = args.output / f"{time:%Y/%m/%Y%m%d_%H}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not np.array_equal(np.load(path), target):
                raise ValueError(f"Existing output differs: {path}")
        else:
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, target)
            temporary.replace(path)
        records.append(
            {
                "time": time.isoformat(),
                "source": file_identity(fields.paths[time]),
                "output": file_identity(path),
            }
        )
    json_write(
        {
            "times": records,
            "target_mean": file_identity(config.target_mean),
            "target_std": file_identity(config.target_std),
            "coordinate": "normalized",
            "statistics_fitted": False,
        },
        args.output / f"manifest_{times[0]:%Y%m%dT%H}_{times[-1]:%Y%m%dT%H}.json",
    )
    print(f"Prepared {len(times)} hourly ERA5 fields in {args.output}")


if __name__ == "__main__":
    main()
