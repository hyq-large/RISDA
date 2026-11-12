import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("atms", "amsua", "mhs", "mwts", "mwhs", "convi")


def setup():
    defaults = {
        "RISDA_DATA_ROOT": ROOT / "data",
        "RISDA_MODEL_ROOT": ROOT / "models",
        "RISDA_OUTPUT_ROOT": ROOT / "outputs",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    data = Path(os.environ["RISDA_DATA_ROOT"])
    models = Path(os.environ["RISDA_MODEL_ROOT"])
    for key, value in {
        "RISDA_OBS_ROOT": data / "observations",
        "RISDA_MEAN_STD": data / "normalization",
        "RISDA_LATENT_DIR": models / "era_latents",
        "RISDA_DENOISER_DIR": models / "era_latents/denoisers/prior",
        "RISDA_VARIANCE": models / "variance.pth",
        "RISDA_RELIABILITY": models / "reliability.pth",
    }.items():
        os.environ.setdefault(key, str(value))
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def default_config(name):
    local = ROOT / "configs" / name
    return local if local.is_file() else Path(sys.prefix) / "share/risda" / name


def read_config(path, overrides=()):
    setup()
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def anchor_times(timeline):
    start = datetime.strptime(str(timeline.start), "%Y-%m-%dT%H")
    end = datetime.strptime(str(timeline.end), "%Y-%m-%dT%H")
    step = timeline.step_hours
    if isinstance(step, bool) or int(step) != step or step < 1:
        raise ValueError("timeline.step_hours must be a positive integer")
    seconds = (end - start).total_seconds()
    if seconds < 0 or seconds % (3600 * step):
        raise ValueError("Timeline endpoints must follow the configured cadence")
    return [start + timedelta(hours=i * step) for i in range(int(seconds // (3600 * step)) + 1)]


def hourly_times(start, end):
    return anchor_times(OmegaConf.create({"start": start, "end": end, "step_hours": 1}))


def load_config(path, overrides=()):
    config = read_config(path, overrides)
    sources = list(config.sources)
    if not sources or len(sources) != len(set(sources)) or set(sources) - set(SOURCES):
        raise ValueError("Select unique sources from atms, amsua, mhs, mwts, mwhs, convi")
    if config.precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported inference precision")
    for name in sources:
        if config.encoders[name].checkpoint not in ("best", "last"):
            raise ValueError(f"Invalid checkpoint selection for {name}")
    if config.prior.checkpoint not in ("best", "last"):
        raise ValueError("Invalid prior checkpoint selection")
    anchor_times(config.timeline)
    for key in ("variance", "reliability"):
        if not config.uncertainty.get(key):
            raise ValueError(f"uncertainty.{key} is required")
    if (
        Path(config.uncertainty.variance).resolve()
        == Path(config.uncertainty.reliability).resolve()
    ):
        raise ValueError("Variance and reliability require separate input files")
    return config


def file_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


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


def json_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
