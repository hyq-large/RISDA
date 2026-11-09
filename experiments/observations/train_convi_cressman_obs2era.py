#!/usr/bin/env python
"""Train the 1h nine-channel Cressman CONVI-to-ERA encoder."""

import sys

from functools import partial
from pathlib import Path

from omegaconf import OmegaConf

from appa.config.hydra import compose
from appa.data.datasets import ERA5DataNpy_2016_2025_1h
from experiments.observations.convi_obs2era_data import Convi2EraPairDataset
from experiments.observations.train_obs2era import main as train

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "train_convi_cressman_obs2era.yaml"
)


if __name__ == "__main__":
    config = compose(CONFIG_PATH, overrides=sys.argv[1:])
    normalization = OmegaConf.to_container(
        config.data.era_normalization,
        resolve=True,
        enum_to_str=True,
    )
    pair_dataset = partial(
        Convi2EraPairDataset,
        era_dataset_cls=ERA5DataNpy_2016_2025_1h,
        era_dataset_kwargs=normalization,
    )
    train(config, convi_pair_dataset_factory=pair_dataset)

