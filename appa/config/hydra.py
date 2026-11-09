r"""Hydra helpers."""

__all__ = [
    "compose",
]

import hydra
import os
import warnings

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Sequence, Union


def compose(
    config_file: Union[str, Path],
    overrides: Sequence[str] = (),
) -> DictConfig:
    r"""Composes an Hydra configuration.

    Arguments:
        config_file: A configuration file path.
        overrides: The overridden settings, as a sequence of strings :py:`"key=value"`.

    Returns:
        A configuration.
    """

    assert os.path.isfile(config_file), f"{config_file} does not exists."

    config_file = os.path.abspath(config_file)
    config_dir, config_name = os.path.split(config_file)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*missing `_self_`.*")

        with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)

    OmegaConf.resolve(cfg)

    return cfg
