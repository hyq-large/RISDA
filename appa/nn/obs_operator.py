r"""Neural observation operator: maps ERA5 atmospheric state + satellite
geometry/identity + lat/lon to ATMS brightness temperatures.

Per-pixel (point-to-point) MLP. Inputs are flattened over the spatial dims
to one big sample stream; the operator is applied independently to each
pixel.
"""

from typing import Sequence

import torch
import torch.nn as nn


class ObsOperatorMLP(nn.Module):
    r"""Point-wise MLP observation operator.

    Inputs (concatenated along the feature dim):
        - era5:        (..., n_era5)         normalized
        - cos_satzen:  (..., 1)              in [-1, 1]; cos(deg2rad(satzen))
        - satname_id:  (..., 1)  long        integer in [0, num_satnames)
        - pos_enc:     (..., 4)              [sin(lat), cos(lat), sin(lon), cos(lon)]

    Output:
        - tb_norm:     (..., n_tb)           ATMS TB in normalized space
                                              (caller can de-normalize with std/mean)

    Args:
        n_era5:           number of ERA5 input channels (default 69)
        n_tb:             number of TB output channels (default 22 for ATMS)
        num_satnames:     number of distinct satellite identifiers
        satname_embed_dim: embedding width for satname (default 8)
        hidden:           list of hidden widths
        dropout:          dropout rate inside MLP (default 0.0)
    """

    def __init__(
        self,
        n_era5: int = 69,
        n_tb: int = 22,
        num_satnames: int = 2,
        satname_embed_dim: int = 8,
        hidden: Sequence[int] = (256, 256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_era5 = int(n_era5)
        self.n_tb = int(n_tb)
        self.num_satnames = int(num_satnames)
        self.satname_embed_dim = int(satname_embed_dim)

        self.satname_embed = nn.Embedding(self.num_satnames, self.satname_embed_dim)

        # 1 (cos satzen) + embed_dim (satname) + 4 (lat/lon sincos)
        in_dim = self.n_era5 + 1 + self.satname_embed_dim + 4

        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, self.n_tb))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def lat_lon_pos_enc(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
        r"""Returns [sin(lat), cos(lat), sin(lon), cos(lon)] in last dim.

        Args:
            lat: (...,) float, in degrees, range [-90, 90].
            lon: (...,) float, in degrees, range [0, 360) or [-180, 180).
        """
        lat_rad = lat * torch.pi / 180.0
        lon_rad = lon * torch.pi / 180.0
        return torch.stack(
            [lat_rad.sin(), lat_rad.cos(), lon_rad.sin(), lon_rad.cos()],
            dim=-1,
        )

    def forward(
        self,
        era5: torch.Tensor,
        cos_satzen: torch.Tensor,
        satname_id: torch.Tensor,
        pos_enc: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Args:
            era5:       (..., n_era5)
            cos_satzen: (..., 1)
            satname_id: (..., 1) long  -- int satellite id
            pos_enc:    (..., 4)
        Returns:
            tb_norm:    (..., n_tb)
        """
        sat_emb = self.satname_embed(satname_id.squeeze(-1))   # (..., embed_dim)
        feat = torch.cat([era5, cos_satzen, sat_emb, pos_enc], dim=-1)
        return self.mlp(feat)
