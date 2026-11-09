"""Paired CONVIOBS and ERA5 data for the dedicated obs2era encoder."""

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from appa.data.datasets import CONVIOBS, ERA5DataNpy_2017_2025
from appa.date import get_date_features

CONVI_QC_TO_ERA5 = "convi_qc_to_era5"
CONVI_QC_TO_ERA5_QC_MASK = "convi_qc_to_era5_qc_mask"
CONVI_QC_TO_QC = "convi_qc_to_convi_qc"
ERA5_QC_MASK_TO_ERA5_QC_MASK = "era5_qc_mask_to_era5_qc_mask"
CONVI_ALIGNMENT_MODES = {
    CONVI_QC_TO_ERA5,
    CONVI_QC_TO_ERA5_QC_MASK,
    CONVI_QC_TO_QC,
    ERA5_QC_MASK_TO_ERA5_QC_MASK,
}
SURFACE_CHANNEL_INDICES = (65, 66, 67, 68)


def build_convi_training_sample(
    qc_values_and_masks,
    union_mask,
    era_gt,
    *,
    alignment_mode,
    use_cressman,
):
    """Build one auditable Encoder input, physical target, and loss mask.

    The three aligned modes supervise only the four surface variables and retain
    their original, per-variable QC masks. In particular, pixels introduced by
    Cressman dilation never become physical-loss targets.
    """
    alignment_mode = str(alignment_mode).lower()
    if alignment_mode not in CONVI_ALIGNMENT_MODES:
        raise ValueError(
            "unsupported CONVI training alignment_mode: "
            f"{alignment_mode!r}; expected one of {sorted(CONVI_ALIGNMENT_MODES)}"
        )
    if qc_values_and_masks.ndim != 3 or qc_values_and_masks.shape[0] != 8:
        raise ValueError(
            "QC surface input must contain [4 values | 4 masks]; got "
            f"shape {tuple(qc_values_and_masks.shape)}"
        )
    if union_mask.ndim != 3 or union_mask.shape[0] != 1:
        raise ValueError(
            "QC union mask must have shape (1,H,W); got "
            f"{tuple(union_mask.shape)}"
        )
    if era_gt.ndim != 3 or era_gt.shape[0] < max(SURFACE_CHANNEL_INDICES) + 1:
        raise ValueError(
            "ERA target must have at least 69 channels; got "
            f"shape {tuple(era_gt.shape)}"
        )

    qc_values = qc_values_and_masks[:4]
    channel_masks = qc_values_and_masks[4:8]
    era5_surface_at_qc = (
        era_gt[list(SURFACE_CHANNEL_INDICES)] * channel_masks
    )
    if alignment_mode == ERA5_QC_MASK_TO_ERA5_QC_MASK:
        input_values = era5_surface_at_qc
    else:
        input_values = qc_values

    encoder_mask = channel_masks if use_cressman else union_mask
    encoder_input = torch.cat((input_values, encoder_mask), dim=0)

    if alignment_mode == CONVI_QC_TO_ERA5:
        return encoder_input.float(), era_gt.float(), union_mask.float()

    # Aligned supervision is surface-only. Keeping all unobserved and
    # non-surface target entries at zero makes accidental mask leakage visible
    # while the 69-channel shape remains compatible with the frozen decoder.
    physical_target = torch.zeros_like(era_gt)
    physical_mask = torch.zeros_like(era_gt)
    if alignment_mode == CONVI_QC_TO_QC:
        target_values = qc_values
    else:
        target_values = era5_surface_at_qc
    physical_target[list(SURFACE_CHANNEL_INDICES)] = target_values
    physical_mask[list(SURFACE_CHANNEL_INDICES)] = channel_masks
    return encoder_input.float(), physical_target.float(), physical_mask.float()


class Convi2EraPairDataset(Dataset):
    """Pair CONVI/ERA surface inputs with an explicitly aligned target."""

    input_channels = 5

    def __init__(
        self,
        era_data_path,
        surf_dir,
        icoa_dir,
        mean_std_dir,
        era_dataset_cls=ERA5DataNpy_2017_2025,
        era_dataset_kwargs=None,
        surface_preprocessing=None,
        alignment_mode=CONVI_QC_TO_ERA5_QC_MASK,
    ):
        self.surface_preprocessing = dict(surface_preprocessing or {})
        preprocessing_name = str(
            self.surface_preprocessing.get("name", "none")
        ).lower()
        if preprocessing_name not in {"none", "cressman"}:
            raise ValueError(
                "unsupported CONVI surface preprocessing: "
                f"{preprocessing_name!r}"
            )
        self.use_cressman = preprocessing_name == "cressman"
        self.input_channels = 8 if self.use_cressman else 5
        self.alignment_mode = str(alignment_mode).lower()
        if self.alignment_mode not in CONVI_ALIGNMENT_MODES:
            raise ValueError(
                "unsupported CONVI training alignment_mode: "
                f"{self.alignment_mode!r}; expected one of "
                f"{sorted(CONVI_ALIGNMENT_MODES)}"
            )

        self.convi = CONVIOBS(
            surf_dir=surf_dir,
            icoa_dir=icoa_dir,
            mean_std_dir=mean_std_dir,
            skip_temp=True,
            skip_humidity=True,
        )
        if len(self.convi) == 0:
            raise RuntimeError("CONVIOBS found no surface-observation timestamps")

        convi_times = [pd.Timestamp(time) for time in self.convi.times]
        first, last = min(convi_times), max(convi_times)
        self.era = era_dataset_cls(
            data_path=era_data_path,
            blanket_size=1,
            time_range=[
                first.strftime("%Y%m%d"),
                last.strftime("%Y%m%dT%H"),
            ],
            **(era_dataset_kwargs or {}),
        )

        era_index = {
            (time.year, time.month, time.day, time.hour): index
            for index, value in enumerate(self.era.times)
            for time in [pd.Timestamp(value)]
        }
        pairs = []
        dates = []
        for convi_index, time in enumerate(convi_times):
            key = (time.year, time.month, time.day, time.hour)
            era_idx = era_index.get(key)
            if era_idx is not None:
                pairs.append((era_idx, convi_index))
                dates.append(key)
        if not pairs:
            raise RuntimeError(
                "No overlapping timestamps between ERA5 and CONVIOBS"
            )

        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.dates = np.asarray(dates, dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        era_index, convi_index = self.pairs[index]
        time = self.convi.times[int(convi_index)]
        qc_values_and_masks, union_mask = self.convi.prepare_surface_encoder_input(
            time,
            include_channel_masks=True,
        )

        era_frame, _ = self.era[int(era_index)]
        era_gt = era_frame[0].float()
        obs, physical_target, physical_mask = build_convi_training_sample(
            qc_values_and_masks.float(),
            union_mask.float(),
            era_gt,
            alignment_mode=self.alignment_mode,
            use_cressman=self.use_cressman,
        )

        year, month, day, hour = self.dates[index]
        date = get_date_features(
            np.datetime64(
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}"
            )
        )
        return obs, physical_target, physical_mask, date
