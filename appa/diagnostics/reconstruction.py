r"""Tools to visualize reconstruction capabilities of an autoencoder."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pathlib import Path
from torch import Tensor

from ..config import PATH_MASK
from ..data.const import ERA5_PRESSURE_LEVELS
from ..diagnostics.const import CMAPS_SURF, UNITS


def plot_reconstruction(
    path: Path,
    dataset_target: xr.Dataset,
    dataset_pred: xr.Dataset,
    time: Tensor,
    rescaled_units: bool = False,
) -> None:
    r"""Compare target and predicted reconstructions.

    Information:
        Quantiles: 2% and 98% are used to set the colorbar limits.

    Arguments:
        path: Directory to save the output figures.
        dataset_target: Dataset containing target values.
        dataset_pred: Dataset containing predicted values.
        time: Tensor representing the time sequence used for folder naming.
        rescaled_units: If True, physical units are shown in the plot title.
    """
    assert all(len(ds.dims) == 3 for ds in [dataset_target, dataset_pred]), (
        "ERROR (plot_reconstruction): Dataset must be 3-dimensional (latitude, longitude, levels) "
        "for reconstruction plots generation."
    )

    path_folder = path / "_".join(map(str, time.tolist())) / "reconstructions"
    path_folder.mkdir(parents=True, exist_ok=True)
    date_str = "{}-{:02d}-{:02d} {:02d}:00".format(*[t.item() for t in time])

    # Europe centered mask
    mask = np.roll(xr.open_zarr(PATH_MASK)["sea_surface_temperature_mask"].values, 720, axis=1)

    for (var_name, x), (_, x_pred) in zip(
        dataset_target.data_vars.items(), dataset_pred.data_vars.items()
    ):
        path_var = path_folder / var_name
        path_var.mkdir(parents=True, exist_ok=True)
        units = ("[" + UNITS.get(var_name, "UNKNOWN") + "]") if rescaled_units else ""

        # Adding level dimension
        x = x.values[np.newaxis, :, :] if x.ndim == 2 else x
        x_pred = x_pred.values[np.newaxis, :, :] if x_pred.ndim == 2 else x_pred

        # Masking land areas
        x = np.where(mask, x, np.nan) if var_name == "sea_surface_temperature" else x
        x_pred = (
            np.where(mask, x_pred, np.nan) if var_name == "sea_surface_temperature" else x_pred
        )

        levels = x.shape[0]
        for l in range(levels):
            # Saturating colorbar for 4% of total values
            x_min, x_max = np.nanquantile(x[l], [0.02, 0.98])
            plt_properties = {
                "cmap": CMAPS_SURF.get(var_name, "viridis"),
                "vmin": x_min,
                "vmax": x_max,
            }
            title = f"{var_name} {units} | {date_str} | "
            title += f"Level {ERA5_PRESSURE_LEVELS[l]}" if levels > 1 else "Surface"
            filename = f"{ERA5_PRESSURE_LEVELS[l]}.png" if x.shape[0] > 1 else "surface.png"

            fig, axs = plt.subplots(2, 1, figsize=(12, 12))
            axs[0].imshow(x[l], **plt_properties)
            axs[1].imshow(x_pred[l], **plt_properties)
            axs[0].set_title(title, pad=15)
            axs[0].set_xticks([])
            fig.subplots_adjust(hspace=0.1)
            fig.colorbar(
                axs[0].images[0],
                ax=axs,
                orientation="vertical",
                fraction=0.0492,
                pad=0.03,
                use_gridspec=True,
            )

            for ax, label in zip(axs, ["Target", "Prediction"]):
                ax.text(
                    0.98,
                    0.97,
                    label,
                    color="white",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    bbox=dict(facecolor="black", edgecolor="black", boxstyle="round,pad=0.3"),
                )

            fig.savefig(path_var / filename, dpi=300)
            plt.close(fig)
