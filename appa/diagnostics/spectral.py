r"""Tools to visualize spectral reconstruction capabilities of an autoencoder."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from torch_harmonics import RealSHT

from ..data.const import ERA5_PRESSURE_LEVELS
from ..diagnostics.const import CMAPS_LINE, EARTH_RADIUS


def _to_tensor(dataset: xr.Dataset) -> Tensor:
    """Helper function to convert xarray Dataset to PyTorch Tensor."""
    array = dataset.to_stacked_array(
        new_dim="z_total", sample_dims=("longitude", "latitude")
    ).transpose("z_total", ...)
    return torch.as_tensor(array.load().data)


def plot_spectral_reconstruction(
    path: Path,
    dataset_target: xr.Dataset,
    dataset_pred: xr.Dataset,
    time: Tensor,
) -> None:
    r"""Compares physical state spectra between prediction and target.

    Figures:
        Top: Relative Absolute Error between prediction and target spectra.
        Main: Comparison of prediction (colored) and target (black) spectra.

    Arguments:
        path: Directory to save the output figures.
        dataset_target: Dataset containing target values.
        dataset_pred: Dataset containing predicted values.
        time: Tensor representing the time sequence used for folder naming.
    """
    assert all(len(ds.dims) == 3 for ds in [dataset_target, dataset_pred]), (
        "ERROR (plot_spectral_reconstruction): Dataset must be 3-dimensional (latitude, longitude, levels) "
        "for spectral plots generation."
    )

    path_folder = path / "_".join(map(str, time.tolist())) / "spectral" / "reconstructions"
    path_folder.mkdir(parents=True, exist_ok=True)
    date_str = "{}-{:02d}-{:02d} {:02d}:00".format(*[t.item() for t in time])

    for (var_name, x), (_, x_pred) in zip(
        dataset_target.data_vars.items(), dataset_pred.data_vars.items()
    ):
        path_var = path_folder / var_name
        path_var.mkdir(parents=True, exist_ok=True)

        # Adding level dimension
        x = x.values[np.newaxis, :, :] if x.ndim == 2 else x.values
        x_pred = x_pred.values[np.newaxis, :, :] if x_pred.ndim == 2 else x_pred.values

        x = torch.from_numpy(x)
        x_pred = torch.from_numpy(x_pred)

        # Spherical Harmonics Transform.
        sht = RealSHT(*x_pred.shape[1:], grid="equiangular")

        # Remove frequency 0 corresponding to infinite wavelength
        coeff_pred, coeff_target = sht(x_pred).abs().sum(-2)[:, 1:], sht(x).abs().sum(-2)[:, 1:]
        rae = (coeff_target - coeff_pred).abs() / coeff_target.abs()
        k_values = 2 * np.pi * EARTH_RADIUS / np.arange(1, rae.shape[1] + 1)

        levels = coeff_pred.shape[0]
        for l in range(levels):
            title = f"{var_name} | {date_str} | "
            title += f"Level {ERA5_PRESSURE_LEVELS[l]}" if levels > 1 else "Surface"
            filename = f"{ERA5_PRESSURE_LEVELS[l]}.png" if x.shape[0] > 1 else "surface.png"

            fig, axs = plt.subplots(2, 1, figsize=(12, 8))
            axs[0].plot(k_values, rae[l] * 100, color="k")
            axs[0].set_title(title, pad=15)
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
            axs[0].set_ylabel("Relative Absolute Error [%]", fontsize=12)
            axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
            axs[0].invert_xaxis()

            axs[1].plot(
                k_values, coeff_pred[l], color=CMAPS_LINE.get(var_name, "blue"), label="Prediction"
            )
            axs[1].plot(k_values, coeff_target[l], color="k", label="Target")
            axs[1].set_xscale("log")
            axs[1].set_yscale("log")
            axs[1].set_xlabel(r"Wavelength $\lambda$ [km]", fontsize=12)
            axs[1].set_ylabel(r"$E(\lambda)$", fontsize=12)
            axs[1].grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)
            axs[1].invert_xaxis()
            axs[1].legend(loc="upper right")

            fig.savefig(path_var / filename, dpi=300)
            plt.close(fig)


def plot_spectral_cae(
    path: Path,
    dataset_target: xr.Dataset,
    dataset_pred: xr.Dataset,
    time: Tensor,
    normalize: bool = True,
) -> None:
    r"""Plot spectral cumulative absolute error

    Information
        The cumulative absolute error is computed by summing the absolute
        differences between the spectral coefficients of the target and
        predicted datasets, calculated at each level and across all variables.

    Arguments:
        path: Directory to save the output figures.
        dataset_target: Dataset containing target values.
        dataset_pred: Dataset containing predicted values.
        time: Tensor representing the time sequence used for folder naming.
        normalize: Whether to normalize the cumulative error (max to 1) or not.
    """
    assert all(len(ds.dims) == 3 for ds in [dataset_target, dataset_pred]), (
        "ERROR (plot_spectral_cae): Dataset must be 3-dimensional (latitude, longitude, levels) "
        "for spectral plots generation."
    )

    path_folder = path / "_".join(map(str, time.tolist())) / "spectral" / "cumulative"
    path_folder.mkdir(parents=True, exist_ok=True)
    filename = "cumulative_absolute_error.png"

    x = _to_tensor(dataset_target)
    x_pred = _to_tensor(dataset_pred)

    # Spherical Harmonics Transform.
    sht = RealSHT(*x_pred.shape[1:], grid="equiangular")

    # Remove frequency 0 corresponding to infinite wavelength
    coeff_pred, coeff_target = sht(x_pred).abs().sum(-2)[:, 1:], sht(x).abs().sum(-2)[:, 1:]
    ae = ((coeff_target - coeff_pred).abs()).sum(0)

    cae = ae.cumsum(0)
    if normalize:
        cae = cae / cae[-1]

    label = "Normalized Cumulative Absolute Error" if normalize else "Cumulative Absolute Error"
    k_values = 2 * np.pi * EARTH_RADIUS / np.arange(1, len(cae) + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, cae)
    plt.xscale("log")
    ax = plt.gca()
    ax.invert_xaxis()
    plt.grid()
    plt.xlabel(r"Wavelength $\lambda$ [km]", fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.savefig(path_folder / filename, dpi=300)
    plt.close()
