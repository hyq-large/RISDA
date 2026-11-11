import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
