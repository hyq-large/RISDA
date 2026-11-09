"""Data-free integrity, import and configuration checks for a source checkout."""
import ast
import importlib
import json
from pathlib import Path

from .environment import ROOT, setup


def main():
    setup()
    for path in ROOT.rglob('*.py'):
        if any(p.startswith('.') for p in path.relative_to(ROOT).parts):
            continue
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    modules = [
        'appa.nn.cae', 'appa.diffusion', 'appa.sampling', 'risda.models.encoders',
        'risda.models.decoders', 'experiments.autoencoder.train_1h',
        'experiments.autoencoder.dump_1h', 'experiments.diffusion.train_diff_1h',
        'experiments.observations.train_obs2era',
        'experiments.observations.train_convi_cressman_obs2era',
        'experiments.observations.train_era5_latent_to_tb',
        'experiments.observations.evaluate_guided_six_frame',
        'experiments.observations.evaluate_guided_six_frame_metrics',
        'scripts.data.latent_stats_v1', 'risda.calibrate', 'risda.calibrate_decoder',
    ]
    for name in modules:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Import came from outside this checkout: {name}')
    from .run import PRESETS, make_config
    from experiments.observations.evaluate_guided_six_frame import select_config_anchors
    for example, count in [('assimilation.yaml', 364)]:
        for preset in PRESETS:
            cfg = make_config(ROOT/'configs'/example, preset, allow_missing_constant=True)
            anchors, _ = select_config_anchors(cfg)
            if len(anchors) != count or cfg.ppgf_order_mode != 'fixed':
                raise RuntimeError(f'Invalid preset {preset}: {example}')
    print(json.dumps({'imports': len(modules), 'presets': len(PRESETS),
                      'external_data_read': False, 'sampling': False, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()

