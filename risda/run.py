import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from .config import anchor_times, load_config
from .environment import ROOT, setup


def file_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


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


def freeze(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Run inputs changed; choose a new output_root: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


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
