"""Dense ERA surface training inputs masked by dynamic CONVI latent blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from appa.data.datasets import CONVIOBS, ERA5DataNpy_2016_2025_1h


def expand_union_mask_to_latent_blocks(pixel_mask, resize):
    """Expand every occupied ERA latent token to its physical-pixel block."""
    mask = torch.as_tensor(pixel_mask, dtype=torch.float32)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "union pixel mask must have shape (1,H,W); got "
            f"{tuple(mask.shape)}"
        )
    factor = int(resize)
    if factor <= 0:
        raise ValueError(f"resize must be positive; got {factor}")

    height, width = mask.shape[-2:]
    pad_h = (-height) % factor
    pad_w = (-width) % factor + 2 * factor
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = mask[None]
    if pad_w:
        padded = F.pad(
            padded,
            (pad_left, pad_right, 0, 0),
            mode="circular",
        )
    if pad_h:
        padded = F.pad(
            padded,
            (0, 0, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

    occupied = F.max_pool2d(
        padded,
        kernel_size=factor,
        stride=factor,
    ) > 0
    expanded = F.interpolate(
        occupied.float(),
        scale_factor=factor,
        mode="nearest",
    )
    expanded = expanded[
        ...,
        pad_top : pad_top + height,
        pad_left : pad_left + width,
    ]
    expected_shape = (1, 1, height, width)
    if tuple(expanded.shape) != expected_shape:
        raise RuntimeError(
            "expanded latent-block mask shape mismatch: "
            f"{tuple(expanded.shape)} != {expected_shape}"
        )
    return expanded[0]


def _timestamp_key(value):
    timestamp = pd.Timestamp(value)
    return (
        int(timestamp.year),
        int(timestamp.month),
        int(timestamp.day),
        int(timestamp.hour),
    )


class DenseEraLatentBlockMask2EraDataset(Dataset):
    """Pair dense ERA truth with exact-hour dynamic CONVI block masks.

    The model input is ``[t2m,u10,v10,msl,block_mask]``. All four ERA5 surface
    values are retained inside an occupied latent-aligned block and zeroed
    elsewhere. The same block mask is returned as the physical-loss mask.
    """

    input_channels = 5
    surface_channel_indices = (65, 66, 67, 68)
    surface_channel_names = ("t2m", "u10", "v10", "msl")

    def __init__(
        self,
        era_data_path,
        era_normalization,
        convi_surf_dir,
        convi_icoa_dir,
        convi_mean_std_dir,
        block_resize,
        time_range=None,
        era_dataset_cls=ERA5DataNpy_2016_2025_1h,
        convi_dataset_cls=CONVIOBS,
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
        self.convi = convi_dataset_cls(
            surf_dir=convi_surf_dir,
            icoa_dir=convi_icoa_dir,
            mean_std_dir=convi_mean_std_dir,
            skip_temp=True,
            skip_humidity=True,
        )
        if len(self.convi.times) == 0:
            raise RuntimeError("CONVI QC dataset contains no timestamps")
        self.block_resize = int(block_resize)
        if self.block_resize <= 0:
            raise ValueError("block_resize must be positive")

        convi_by_key = {
            _timestamp_key(timestamp): timestamp
            for timestamp in self.convi.times
        }
        era_indices = []
        convi_times = []
        dates = []
        for era_index, timestamp in enumerate(self.era.times):
            key = _timestamp_key(timestamp)
            convi_time = convi_by_key.get(key)
            if convi_time is None:
                continue
            era_indices.append(era_index)
            convi_times.append(convi_time)
            dates.append(key)
        if not era_indices:
            raise RuntimeError(
                "ERA5 and CONVI QC datasets have no exact-hour overlap"
            )
        self.era_indices = np.asarray(era_indices, dtype=np.int64)
        self.convi_times = tuple(convi_times)
        self.dates = np.asarray(dates, dtype=np.int64)
        # Calibration code indexes this dataset rather than the unfiltered ERA
        # archive, so expose the exact aligned axis explicitly.
        self.times = self.dates
        self.mask_mode = "convi_latent_block"

    def __len__(self):
        return len(self.era_indices)

    def __getitem__(self, index):
        era_frame, _ = self.era[int(self.era_indices[index])]
        era_gt = era_frame[0].float()
        if era_gt.ndim != 3 or era_gt.shape[0] != 69:
            raise ValueError(
                "Dense ERA O2E expects one 69-channel frame; "
                f"got shape {tuple(era_gt.shape)}"
            )

        _sparse_values, sparse_union_mask = (
            self.convi.prepare_surface_encoder_input(self.convi_times[index])
        )
        block_mask = expand_union_mask_to_latent_blocks(
            sparse_union_mask,
            self.block_resize,
        )
        surface = era_gt[list(self.surface_channel_indices)]
        observation = torch.cat(
            (surface * block_mask, block_mask),
            dim=0,
        )
        date = torch.from_numpy(self.dates[index].copy())
        return observation, era_gt, block_mask, date
