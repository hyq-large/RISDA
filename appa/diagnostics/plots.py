r"""Visualization helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from matplotlib.figure import Figure
from torch import Tensor
from typing import Iterable, List, Optional, Sequence, Union

from appa.date import format_blanket_date


def plot_spheres_grid(
    blankets: Tensor,
    date: Tensor,
    coordinates: Sequence[Tensor],
    cmap: str = "jet",
    figsize: int = 3,
    vmin: Optional[Union[List[float], float]] = None,
    vmax: Optional[Union[List[float], float]] = None,
    row_titles: Optional[List[str]] = None,
):
    r"""Plot sampled blankets on a grid.

    Arguments:
        blankets: The blanket tensor, with shape :math:`(batch, blanket_size, N)`.
        date: The date tensor, with shape :math:`(batch, blanket_size, 4)`. (year, month, day, hour)
        coordinates: A tuple of coordinate tensors, with shape :math:`(3, N)`. (x, y, z)
        cmap: The colormap shared for each plot. Defaults to 'jet'.
        figsize: The size of a figure cell. Defaults to 3.
        vmin: The minimum value of the colormap. If a list, it should have the same length as the number of images.
        vmax: The maximum value of the colormap. If a list, it should have the same length as the number of images.
        row_titles: List of titles written on the left of each row. Defaults to None.

    Returns:
        The figure containing the differents blankets in the specified grid.
    """

    B, blanket_size = blankets.shape[:2]
    x, y, z = coordinates

    fig = plt.figure(figsize=(figsize * blanket_size, figsize * B))
    fig.suptitle(format_blanket_date(date))
    plt.tight_layout()
    subplot_idx = 1
    for j in range(B):
        for k in range(blanket_size):
            ax = fig.add_subplot(
                B,
                blanket_size,
                subplot_idx,
                projection="3d",
                elev=53,
                azim=9,
            )
            ax.scatter(x, y, z, c=blankets[j, k], cmap=cmap, vmin=vmin, vmax=vmax, s=0.5)
            ax.set_axis_off()

            if k == 0:
                ax.text2D(
                    0,
                    0.475,
                    row_titles[j],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=15,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            subplot_idx += 1

    return fig


def plot_image_grid(
    data: Tensor,
    shape: Sequence[int],
    mask: Tensor = None,
    cmap: str = "jet",
    figsize: int = 8,
    axis: bool = False,
    vmin: Optional[Union[List[float], float]] = None,
    vmax: Optional[Union[List[float], float]] = None,
    x_titles: Optional[List[str]] = None,
    y_titles: Optional[List[str]] = None,
    fontsize: int = 32,
    border=0.1,
    masked_values_color: str = "white",
    tex_font: bool = True,
) -> Figure:
    """A function to display data images on a grid.

    Arguments:
        data: Tensor containing the sequence of images, with shape :math:`(N, H, W)`.
        For Earth data, H and W are the  and latitudinallongitudinal dimensions.
        shape: The shape of the figure, as a tuple (row, columns) such that N = row * columns.
        mask: The transparency mask (e.g. to highlight land surfaces). Defaults to None.
        cmap: The colormap shared for each plot. Defaults to 'jet'.
        figsize: The size of a figure cell.
        axis: Whether to display axis or not. Defaults to False.
        vmin: The minimum value of the colormap. If a list, it should have the same length as the number of images.
        vmax: The maximum value of the colormap. If a list, it should have the same length as the number of images.
        x_titles: List of titles written below each column. Defaults to None.
        y_titles: List of titles written on the left of each row. Defaults to None.
        fontsize: The fontsize of the x/y titles. Defaults to 32.

    Returns:
        The figure containing the differents data images in the specified grid.
    """

    H, W = data[0].shape

    figsize = (figsize * shape[1], figsize * shape[0] * H / W)
    f, a = plt.subplots(shape[0], shape[1], figsize=figsize)

    if tex_font:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

    if mask is not None:
        mask = mask.roll(W // 2, dims=-1)

    data = data.roll(W // 2, dims=-1)

    cmap = mpl.colormaps.get_cmap(cmap)
    cmap.set_bad(color=masked_values_color)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if shape[0] == 1 and shape[1] == 1:
                ax = a
            elif shape[0] == 1:
                ax = a[j]
            elif shape[1] == 1:
                ax = a[i]
            else:
                ax = a[i, j]

            if not axis:
                ax.axis("off")

            if mask is not None:
                ax.imshow(mask, cmap="grey")
            if vmin is not None and vmax is not None:
                vmin_val = vmin[i * shape[1] + j] if isinstance(vmin, Iterable) else vmin
                vmax_val = vmax[i * shape[1] + j] if isinstance(vmax, Iterable) else vmax
                ax.imshow(data[i * shape[1] + j], cmap=cmap, vmin=vmin_val, vmax=vmax_val)
            else:
                ax.imshow(data[i * shape[1] + j], cmap=cmap, alpha=0.9)

    plt.tight_layout()
    if isinstance(border, float):
        plt.subplots_adjust(left=border, right=1 - border, top=1 - border, bottom=border)
    else:
        plt.subplots_adjust(*border)

    if x_titles:
        for j, title in enumerate(x_titles):
            if len(a.shape) == 1:
                col_pos = a[j].get_position().bounds
                top_row_pos = a[j].get_position().bounds
            else:
                col_pos = a[0, j].get_position().bounds
                top_row_pos = a[0, j].get_position().bounds
            f.text(
                col_pos[0] + col_pos[2] / 2,
                top_row_pos[1] + top_row_pos[3] + 0.005,
                r"\textbf{" + title + r"}" if tex_font else title,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    if y_titles:
        for i, title in enumerate(y_titles):
            if len(a.shape) == 1:
                row_pos = a[i].get_position().bounds
            else:
                row_pos = a[i, 0].get_position().bounds
            f.text(
                row_pos[0] - 0.0095,
                row_pos[1] + row_pos[3] / 2,
                r"\textbf{" + title + r"}" if tex_font else title,
                ha="center",
                va="center",
                rotation=90,
                fontsize=fontsize,
            )

    return f


def vorticity(u: Tensor, v: Tensor) -> Tensor:
    """A function that computes the vorticity of given velocity fields.

    Arguments:
        u: The horizontal velocity tensor, with shape :math:`(La, Lo)`.
        v: The vertical velocity tensor, with shape :math:`(La, Lo)`.

    Returns:
        The vorticity tensor, with shape :math:`(La, Lo)`.
    """

    uv = torch.stack((u, v))
    uv = torch.nn.functional.pad(uv, pad=(1, 1), mode="circular")

    (du_dy,) = torch.gradient(uv[0], dim=-2)
    (dv_dx,) = torch.gradient(uv[1], dim=-1)

    y = dv_dx - du_dy
    y = y[..., :, 1:-1]

    return y
