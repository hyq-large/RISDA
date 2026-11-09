"""Train one exact-hour satellite forward decoder for physical-space controls."""
import argparse
import os
import shlex
import subprocess
import sys

from .environment import ROOT, setup


def main():
    setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('atms','amsua','mhs','mwts','mwhs'))
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--latent-dir', default=os.environ['RISDA_LATENT_DIR'])
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={args.gpus}', str(ROOT/'experiments/observations/train_era5_latent_to_tb.py'),
               f'era_latent_dir={args.latent_dir}', f'data.sate_name={args.source}',
               'data.cadence_hours=1', f'data.era_data_path={os.environ["RISDA_ERA5_1H"]}',
               f'+id=direct_e2o_{args.source}', *overrides]
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT/'experiments/observations', check=True)

