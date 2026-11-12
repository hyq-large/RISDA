import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .environment import setup
from .reproduction import default_config, read_config

CONFIGS = {
    "ae": "era5_ae.yaml", "prior": "latent_prior.yaml",
    "satellite": "satellite_encoder.yaml", "convi": "convi_encoder.yaml",
    "dump": "latents.yaml", "stats": "latents.yaml",
}


def main():
    setup()
    parser = argparse.ArgumentParser(description="Train the models used by PPGF+VCLG")
    parser.add_argument("stage", choices=tuple(CONFIGS))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.gpus < 1 or any("=" not in item or item.startswith("--") for item in overrides):
        parser.error("Use a positive GPU count and configuration overrides as key=value")
    exporting = args.stage in ("dump", "stats")
    if (exporting or args.device == "cpu") and args.gpus != 1:
        parser.error("Export/statistics and the CPU launcher require --gpus 1")
    if exporting and args.stop_after is not None:
        parser.error("--stop-after applies to training stages")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be a positive optimizer-step count")
    config_path = (args.config or default_config("training/" + CONFIGS[args.stage])).resolve()
    config = read_config(config_path, overrides)
    if not exporting:
        from .training.worker import prepare_config
        config = prepare_config(config, args.stage)
        batch = int(config.train.batch_size_per_step)
        if batch % (args.gpus * int(config.train.microbatch)):
            parser.error("Global batch must be divisible by GPU count times microbatch")
    module = "risda.training.export" if exporting else "risda.training.worker"
    command = [sys.executable]
    if args.gpus > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus)]
    command += ["-m", module, args.stage, "--config", str(config_path), "--device", args.device]
    if args.resume:
        command += ["--resume"]
    if args.stop_after is not None:
        command += ["--stop-after", str(args.stop_after)]
    command += overrides
    print(shlex.join(command), flush=True)
    if args.dry_run:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(config, resolve=True), flush=True)
        return
    subprocess.run(command, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
