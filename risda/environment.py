"""Portable defaults shared by the public command line entrypoints."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup():
    defaults = {
        'RISDA_DATA_ROOT': str(ROOT / 'data'),
        'RISDA_OUTPUT_ROOT': str(ROOT / 'outputs'),
        'RISDA_MODEL_ROOT': str(ROOT / 'models'),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    data = Path(os.environ['RISDA_DATA_ROOT']).expanduser().resolve()
    output = Path(os.environ['RISDA_OUTPUT_ROOT']).expanduser().resolve()
    models = Path(os.environ['RISDA_MODEL_ROOT']).expanduser().resolve()
    paths = {
        'RISDA_ROOT': ROOT,
        'RISDA_ERA5_1H': data / 'era5_hourly',
        'RISDA_ERA5_6H': data / 'era5_6hourly',
        'RISDA_OBS_ROOT': data / 'observations',
        'RISDA_MEAN_STD': data / 'normalization',
        'RISDA_LATENT_DIR': models / 'era_latents',
        'RISDA_DENOISER_DIR': models / 'era_latents/denoisers/prior',
    }
    for key, value in paths.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    # Prevent an inherited research checkout from satisfying missing release imports.
    os.environ['PYTHONPATH'] = str(ROOT)

