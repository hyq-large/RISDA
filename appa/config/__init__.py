r"""Global paths and configuration helpers.

Runtime paths are environment-driven so the repository never encodes a
particular cluster layout. Command-line Hydra overrides remain the preferred
way to select experiment-specific data and model directories.
"""

__all__ = [
    "compose",
]

import os

from pathlib import Path

from .hydra import compose

PROJECT = Path(
    os.environ.get("RISDA_ROOT", Path(__file__).resolve().parents[2])
).expanduser()
DATA_ROOT = Path(
    os.environ.get("RISDA_DATA_ROOT", PROJECT / "data")
).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("RISDA_OUTPUT_ROOT", PROJECT / "outputs")
).expanduser()

PATH_AE = OUTPUT_ROOT / "ae"

PATH_ERA5 = Path(
    os.environ.get(
        "RISDA_ERA5_ZARR",
        DATA_ROOT / "era5_1993-2021-1h-1440x721.zarr",
    )
).expanduser()
PATH_STAT = Path(
    os.environ.get(
        "RISDA_ERA5_STATS",
        DATA_ROOT / "stats_era5_1993-2021-1h-1440x721.zarr",
    )
).expanduser()
PATH_MASK = Path(
    os.environ.get(
        "RISDA_ERA5_MASKS",
        DATA_ROOT / "masks_era5_1993-2021-1h-1440x721.zarr",
    )
).expanduser()
