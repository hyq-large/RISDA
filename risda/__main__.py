"""RISDA command line front door; training kernels retain their original names."""
import argparse
import os
import runpy
import subprocess
import sys

from .environment import ROOT, setup


def main():
    setup()
    parser = argparse.ArgumentParser(description='RISDA training, calibration and assimilation')
    parser.add_argument('stage', choices=(
        'train', 'train-decoder', 'calibrate', 'calibrate-decoder',
        'run', 'metrics', 'check',
    ))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.stage == 'train':
        raise SystemExit(subprocess.call(['bash', str(ROOT / 'run_train_latent_channels.sh'), *rest], cwd=ROOT))
    if args.stage == 'train-decoder':
        from .train_decoder import main as run
        sys.argv = ['risda train-decoder', *rest]
        return run()
    modules = {
        'calibrate': 'risda.calibrate',
        'calibrate-decoder': 'risda.calibrate_decoder',
        'run': 'risda.run',
        'check': 'risda.check',
        'metrics': 'experiments.observations.evaluate_guided_six_frame_metrics',
    }
    os.chdir(ROOT)
    sys.argv = [f'risda {args.stage}', *rest]
    runpy.run_module(modules[args.stage], run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()

