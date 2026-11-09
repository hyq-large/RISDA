r"""Losses used for training auto encoder"""

import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from typing import Optional

from appa.data.const import (
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
)


def graphcast_level_weights(levels: list, reduction: str = "sum") -> Tensor:
    r"""Returns the level weights of each variable following Graphcast's implementation.

    References:
        | GraphCast: Learning skillful medium-range global weather forecasting
        | https://arxiv.org/abs/2212.12794

    """
    weights = torch.zeros(
        len(ERA5_SURFACE_VARIABLES) + len(levels) * len(ERA5_ATMOSPHERIC_VARIABLES)
    )

    p_levels = torch.as_tensor(levels, dtype=torch.float32)
    if reduction == "sum":
        lin_atm_weights = p_levels / p_levels.sum()
    elif reduction == "mean":
        lin_atm_weights = p_levels / p_levels.mean()

    for i, var in enumerate(ERA5_SURFACE_VARIABLES):
        if "2m_temperature" in var:
            weights[i] = 1.0
        else:
            weights[i] = 0.1

    weights[len(ERA5_SURFACE_VARIABLES) :] = lin_atm_weights.repeat(
        len(ERA5_ATMOSPHERIC_VARIABLES)
    )

    return weights


def graphcast_level_weights_v1(
    levels: list,
    num_atmo_vars: int = 5,
    num_surface_vars: int = 4,
    reduction: str = "sum",
) -> Tensor:
    r"""Level weights for the v1 channel layout: [atmo(num_atmo_vars * levels)] + [surface].

    Channel order: z@levels, t@levels, u@levels, v@levels, q@levels, t2m, u10, v10, msl.
    """
    num_levels = len(levels)
    total = num_atmo_vars * num_levels + num_surface_vars
    weights = torch.zeros(total)

    p_levels = torch.as_tensor(levels, dtype=torch.float32)
    if reduction == "sum":
        lin_atm_weights = p_levels / p_levels.sum()
    elif reduction == "mean":
        lin_atm_weights = p_levels / p_levels.mean()

    weights[:num_atmo_vars * num_levels] = lin_atm_weights.repeat(num_atmo_vars)

    # Surface: t2m gets weight 1.0, others 0.1
    surface_start = num_atmo_vars * num_levels
    weights[surface_start] = 1.0      # t2m
    weights[surface_start + 1] = 0.1  # u10
    weights[surface_start + 2] = 0.1  # v10
    weights[surface_start + 3] = 0.1  # msl

    return weights


class AELoss(nn.Module):
    r"""Auto-encoder loss.

    Arguments:
        criterion: Criterion used for the loss.
        N_lat: Number of latitude points.
        N_lon: Number of longitude points.
        latitude_weighting: Whether to weight the loss across latitudes or not.
        level_weighting: Whether to weight the loss across pressure levels or not.
        levels: List of pressure levels.
        channel_weights: Optional per-channel weights [C] to balance gradient contribution.
            Computed as mean(MSE) / MSE_c from a prior evaluation run.
        balance_channels: Whether to dynamically balance per-channel loss via EMA.
        balance_momentum: EMA momentum for dynamic channel balancing.
    """

    def __init__(
        self,
        criterion: str,
        N_lat: int = 120,
        N_lon: int = 240,
        latitude_weighting: bool = True,
        level_weighting: bool = False,
        levels: Optional[list] = ERA5_PRESSURE_LEVELS,
        channel_weights: Optional[Tensor] = None,
        balance_channels: bool = False,
        balance_momentum: float = 0.99,
    ):
        super().__init__()
        if criterion == "mse":
            self.loss = nn.MSELoss(reduction="none")
        elif criterion == "mae":
            self.loss = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        self.N_lat = N_lat
        self.N_lon = N_lon
        self.balance_channels = balance_channels
        self.balance_momentum = balance_momentum

        if latitude_weighting:
            lat_weights = (
                torch.linspace(torch.pi / 2, -torch.pi / 2, self.N_lat).cos() + 1 / self.N_lon
            )
        else:
            lat_weights = None
        self.register_buffer("latitude_weights", lat_weights)

        if level_weighting:
            if level_weighting == "v1":
                level_weights = graphcast_level_weights_v1(levels)
            else:
                level_weights = graphcast_level_weights(levels)
        else:
            level_weights = None
        self.register_buffer("level_weights", level_weights)

        self.register_buffer("channel_weights", channel_weights)
        self.register_buffer("channel_ema", None)

    def forward(self, input: Tensor, target: Tensor):
        loss = self.loss(input, target)

        if self.latitude_weights is not None:
            loss = rearrange(loss, "B (Lat Lon) Z -> B Lon Z Lat", Lat=self.N_lat)
            loss = loss * self.latitude_weights.to(loss)
            loss = rearrange(loss, "B Lon Z Lat -> B (Lat Lon) Z")

        if self.level_weights is not None:
            loss = loss * self.level_weights.to(loss)

        if self.channel_weights is not None:
            loss = loss * self.channel_weights.to(loss)

        if self.balance_channels:
            mse_per_channel = loss.detach().mean(dim=(0, 1))  # [C]
            if self.channel_ema is None:
                self.channel_ema = mse_per_channel
            else:
                self.channel_ema = self.balance_momentum * self.channel_ema + (1 - self.balance_momentum) * mse_per_channel
            weights = 1 / (self.channel_ema + 1e-8)
            loss = loss * weights

        return loss.mean()
