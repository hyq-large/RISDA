"""Run an explicitly configured RISDA comparison, with frozen config and logs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

from .environment import ROOT, setup

# Public labels follow Table 1. PPGF_SD shares priorities across locations;
# both PPGF variants use spatially resolved likelihood variances.
PRESETS = {
    'Appa': ('decoder', 'sum', 'constant', None),
    'AGF_m': ('encoder', 'sum', 'source_mean', None),
    'AGF+VCLG': ('encoder', 'sum', 'channelwise', None),
    'PCGrad+VCLG': ('encoder', 'pcgrad', 'channelwise', None),
    'PPGF_SD+VCLG': ('encoder', 'channel_ppgf', 'spatial', 'channel'),
    'PPGF+VCLG': ('encoder', 'channel_ppgf', 'spatial', 'spatial'),
}


def make_config(path, experiment, overrides=(), allow_missing_constant=False):
    from appa.config.hydra import compose
    from .timeline import validate_timeline, calibration_checks
    cfg = compose(path, overrides)
    validate_timeline(cfg)
    route, merge, variance, ordering = PRESETS[experiment]
    if experiment == 'AGF+VCLG':
        variance = str(cfg.vclg_variance)
        if variance not in {'channelwise', 'spatial'}:
            raise ValueError('vclg_variance must be channelwise or spatial')
    cfg.id = experiment
    cfg.route, cfg.guidance_merge, cfg.var_y_mode = route, merge, variance
    cfg.var_y_constant = None
    cfg.ppgf_order_mode = 'fixed'
    cfg.pcgrad_order_mode = 'fixed'
    cfg.pcgrad_shuffle = False
    cfg.ppgf_overlap_only = False
    cfg.ppgf_ordering_source = 'matrix_cache' if ordering else 'var_y'
    cfg.spatial_var_y_cache = None
    cfg.ppgf_ordering_cache = None
    cfg.convi.guidance_space = None
    cfg.convi.var_y_mode = None
    cfg.convi.var_y_constant = None
    cfg.metrics.enabled = False
    if variance == 'spatial':
        cfg.spatial_var_y_cache = str(Path(cfg.calibration_root) / 'encoder_spatial_var_y.pth')
    if ordering:
        cfg.ppgf_ordering_cache = str(Path(cfg.calibration_root) / f'encoder_{ordering}_mse_ordering.pth')
    if experiment == 'Appa':
        constant_path = Path(cfg.calibration_root) / 'decoder/decoder_constant.json'
        if constant_path.is_file():
            value = json.loads(constant_path.read_text())
            checks = calibration_checks(cfg.calibration_period)
            # The scalar file carries dates/counts, with hourly scope implicit
            # in the decoder calibration manifest checked by preflight.
            checks.pop('time_scope')
            if any(value.get(k) != v for k, v in checks.items()) or value.get('sources') != list(cfg.enabled_sources):
                raise ValueError('Decoder constant protocol does not match this run')
            cfg.var_y_constant = float(value['var_y'])
            if not math.isfinite(cfg.var_y_constant) or cfg.var_y_constant <= 0:
                raise ValueError('Calibrated constant must be finite and positive')
        elif not allow_missing_constant:
            raise FileNotFoundError(f'Calibrate the Decoder first: {constant_path}')
    return cfg


def identity(path):
    path = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def preflight(cfg):
    """Require completed calibration and unchanged model/statistic inputs."""
    routes = ['decoder'] if cfg.route == 'decoder' else ['encoder']
    if cfg.convi.guidance_space == 'physical' and 'decoder' not in routes:
        routes.append('decoder')
    records = {}
    for route in routes:
        directory = Path(cfg.calibration_root) / ('decoder' if route == 'decoder' else '')
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_text())
        protocol = manifest['protocol']
        if manifest['status'] != 'complete' or manifest.get('smoke'):
            raise ValueError(f'Calibration is incomplete or is a smoke run: {path}')
        if protocol['period'] != cfg.calibration_period or protocol['sources'] != list(cfg.enabled_sources):
            raise ValueError(f'Calibration source order/period differs: {path}')
        for name, spec in protocol['models'].items():
            entry = cfg.source_catalog[name][route]
            actual_dir = (Path(cfg.model_path).parents[1] / 'ae'
                          if route == 'decoder' and name == 'convi' else Path(entry.model_dir))
            if actual_dir.resolve() != Path(spec['model_dir']).resolve():
                raise ValueError(f'{name} {route} model differs from calibration')
            if Path(spec['identities']['era_latents']['path']).parent.resolve() != Path(cfg.model_path).resolve().parents[1]:
                raise ValueError(f'{name} latent coordinates differ from the prior')
            for old in (*spec['identities'].values(), *spec['normalization_files'].values()):
                if 'sha256' in old:
                    now = identity(old['path'])
                    if now['sha256'] != old['sha256']:
                        raise ValueError(f'Calibration input changed: {old["path"]}')
                    records[now['path']] = now
            cache = Path(entry.var_y_cache)
            records[str(cache)] = identity(cache)
        for old in manifest['outputs'].values():
            now = identity(old['path'])
            if now['sha256'] != old['sha256']:
                raise ValueError(f'Calibration output changed: {old["path"]}')
            records[now['path']] = now
        records[str(path)] = identity(path)
    latent = Path(cfg.model_path).parents[1]
    for path in (Path(cfg.model_path)/'config.yaml',
                 Path(cfg.model_path)/f'model_{cfg.model_target}.pth',
                 latent/'stats.pth', latent/'ae/config.yaml', latent/'ae/model.pth'):
        records[str(path)] = identity(path)
    return records


def freeze(path, text):
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f'Run inputs changed; use a new output_root: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def logged(command, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as handle:
        handle.write('\n$ '+shlex.join(command)+'\n')
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=PRESETS)
    parser.add_argument('gpus', nargs='?', type=int, default=1)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/assimilation.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Print resolved config/commands without accessing model or data files')
    parser.add_argument('--check-runtime', action='store_true', help='With --dry-run: also load the real runtime on CPU, without sampling')
    parser.add_argument('--metrics-only', action='store_true')
    parser.add_argument('--skip-metrics', action='store_true')
    args, overrides = parser.parse_known_args()
    if args.gpus < 1:
        parser.error('GPU count must be positive')
    if args.check_runtime and not args.dry_run:
        parser.error('--check-runtime requires --dry-run')
    cfg = make_config(args.config.resolve(), args.experiment, overrides,
                      allow_missing_constant=args.dry_run and not args.check_runtime)
    run_root = Path(cfg.output_root).expanduser().resolve() / cfg.id
    run_dir = run_root / cfg.split
    config_path = run_root / 'launch/config.yaml'
    launch = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
              f'--nproc_per_node={args.gpus}']
    commands = [launch+[str(ROOT/'experiments/observations/evaluate_guided_six_frame.py'), '--config', str(config_path)],
                launch+[str(ROOT/'experiments/observations/evaluate_guided_six_frame_metrics.py'), '--run-dir', str(run_dir)]]
    print(OmegaConf.to_yaml(cfg, resolve=True))
    for command in commands:
        print(shlex.join(command))
    if args.dry_run and not args.check_runtime:
        if cfg.var_y_mode == 'constant' and cfg.var_y_constant is None or cfg.convi.var_y_mode == 'constant' and cfg.convi.var_y_constant is None:
            print('Calibrated constant is pending; execution requires decoder/decoder_constant.json.')
        return
    provenance = preflight(cfg)
    if args.check_runtime:
        import torch
        from experiments.observations import evaluate_guided_six_frame as ev
        runtime = ev.load_generation_runtime(cfg, list(cfg.enabled_sources), torch.device('cpu'))
        manifest = ev.build_generation_manifest(cfg, runtime, list(cfg.enabled_sources), str(cfg.split))
        print(f'CPU runtime validated; no sampling. Contract: {manifest["contract_hash"]}')
        return
    freeze(config_path, OmegaConf.to_yaml(cfg, resolve=True))
    freeze(run_root/'launch/inputs.json', json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    if not args.metrics_only:
        logged(commands[0], run_root/'logs/generation.log')
    if not args.skip_metrics:
        logged(commands[1], run_root/'logs/metrics.log')


if __name__ == '__main__':
    main()

