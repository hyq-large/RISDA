"""Dense ERA surface fields paired with the full ERA state for O2E training."""

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from appa.data.datasets import ERA5DataNpy_2016_2025_1h


class DenseEraSurface2EraDataset(Dataset):
    """Use dense ERA ``t2m/u10/v10/msl`` as a five-channel O2E input.

    The hourly ERA archive is converted by ``ERA5DataNpy_2016_2025_1h`` from
    its native 1h-statistics basis into the target basis used by the frozen ERA
    autoencoder. The four surface channels and the 69-channel target therefore
    share exactly the decoder's normalization. The fifth input channel and the
    physical-loss mask are all ones because the ERA input is globally dense.
    """

    input_channels = 5
    surface_channel_indices = (65, 66, 67, 68)
    surface_channel_names = ("t2m", "u10", "v10", "msl")

    def __init__(
        self,
        era_data_path,
        era_normalization,
        time_range=None,
        era_dataset_cls=ERA5DataNpy_2016_2025_1h,
    ):
        normalization = dict(era_normalization)
        self.era = era_dataset_cls(
            data_path=era_data_path,
            blanket_size=1,
            time_range=time_range,
            **normalization,
        )
        if len(self.era) == 0:
            raise RuntimeError("Hourly ERA dataset contains no frames")

        self.dates = np.asarray(
            [
                [timestamp.year, timestamp.month, timestamp.day, timestamp.hour]
                for value in self.era.times
                for timestamp in [pd.Timestamp(value)]
            ],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.era)

    def __getitem__(self, index):
        era_frame, _ = self.era[index]
        era_gt = era_frame[0].float()
        if era_gt.ndim != 3 or era_gt.shape[0] != 69:
            raise ValueError(
                "Dense ERA O2E expects one 69-channel frame; "
                f"got shape {tuple(era_gt.shape)}"
            )

        surface = era_gt[list(self.surface_channel_indices)]
        full_mask = torch.ones_like(surface[:1])
        observation = torch.cat((surface, full_mask), dim=0)
        date = torch.from_numpy(self.dates[index].copy())
        return observation, era_gt, full_mask, date
