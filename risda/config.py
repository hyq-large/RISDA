from datetime import datetime, timedelta
from pathlib import Path

from omegaconf import OmegaConf

SOURCES = ("atms", "amsua", "mhs", "mwts", "mwhs", "convi")


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


def load_config(path, overrides=()):
    config = OmegaConf.load(path)
    OmegaConf.set_struct(config, True)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
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
