"""Normalization contract for CONVI guidance through the frozen ERA decoder."""

import hashlib
import numpy as np

from omegaconf import OmegaConf
from pathlib import Path


def physical_normalization_contract(ae_config_path, mean_std_dir):
    """Require identical ERA target/observation statistics and record their bytes.

    Decoder predictions and observations are compared in normalized ERA space.
    Only latent coordinates are unstandardized before calling the decoder.
    """
    config = OmegaConf.load(ae_config_path)
    normalization = config.get("normalization", {})
    contract = {"space": "normalized_era_surface", "era_channel_indices": [65, 66, 67, 68]}
    for statistic in ("mean", "std"):
        target = normalization.get(f"target_{statistic}_path")
        if target is None:
            raise RuntimeError(f"ERA AE config lacks target {statistic} normalization")
        target = Path(target).expanduser().resolve()
        observation = (Path(mean_std_dir) / f"{statistic}_era5.npy").expanduser().resolve()
        arrays = []
        for path in (target, observation):
            array = np.asarray(np.load(path), dtype=np.float32).squeeze()
            if array.ndim != 1 or array.size < 69 or not np.isfinite(array[:69]).all():
                raise ValueError(f"Invalid ERA {statistic} statistics: {path}")
            arrays.append(array[:69])
        if statistic == "std" and (arrays[0] <= 0).any():
            raise ValueError("ERA standard deviations must be positive")
        if not np.array_equal(*arrays):
            raise RuntimeError(f"CONVI and ERA Decoder {statistic} normalization differ")
        contract[statistic] = {
            "decoder_target_path": str(target),
            "observation_path": str(observation),
            "float32_sha256": hashlib.sha256(arrays[0].tobytes()).hexdigest(),
            "surface_values": arrays[0][65:69].tolist(),
        }
    return contract
