"""Resolve the same timeline, sources and assets used by assimilation."""
import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from appa.config.hydra import compose
from .environment import ROOT, setup
from .timeline import period_id, validate_timeline


def configure(module):
    setup()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', default=os.environ.get(
        'RISDA_CALIBRATION_CONFIG', str(ROOT / 'configs/calibration.yaml')))
    args, remaining = parser.parse_known_args()
    path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(path)
    evaluation = Path(cfg.assimilation_config)
    if not evaluation.is_absolute():
        evaluation = path.parent / evaluation
    shared = compose(evaluation)
    validate_timeline(shared)
    sources = tuple(shared.enabled_sources)
    allowed = {'atms', 'amsua', 'mhs', 'mwts', 'mwhs', 'convi'}
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= allowed:
        raise ValueError('enabled_sources must contain distinct supported sources')
    module.CALIBRATION_CONFIG = cfg
    module.CONFIG_PATH = path
    module.REFERENCE = evaluation.resolve()
    module.OUTPUT = Path(shared.calibration_root).expanduser().resolve()
    module.SOURCES = sources
    module.START = str(shared.timeline.calibration.start)
    module.END = str(shared.timeline.calibration.end)
    module.PERIOD = period_id(module.START, module.END)
    os.environ['RISDA_CALIBRATION_CONFIG'] = str(path)
    return remaining
