from dataclasses import dataclass
from pathlib import Path

import torch


def load_matrix(path, key, sources, latent_shape):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("sources") != list(sources):
        raise ValueError(f"{path}: sources must match the configured source order")
    if key not in payload:
        raise ValueError(f"{path}: missing {key}")
    matrix = torch.as_tensor(payload[key], dtype=torch.float32)
    height, width, channels = latent_shape
    expected = (len(sources), height, width, channels)
    if tuple(matrix.shape) not in (expected, (len(sources), height * width, channels)):
        raise ValueError(f"{path}: expected {expected} or its flattened spatial layout")
    if not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(f"{path}: {key} must be finite and nonnegative")
    return matrix.reshape(len(sources), height * width, channels)


@dataclass
class Uncertainty:
    variance: torch.Tensor
    reliability: torch.Tensor

    @classmethod
    def load(
        cls,
        variance_path,
        reliability_path,
        sources,
        latent_shape,
        scale=1.0,
        floor=1e-8,
        device="cpu",
    ):
        if not 0 < float(scale) < float("inf") or not 0 < float(floor) < float("inf"):
            raise ValueError("Variance scale and floor must be finite and positive")
        variance = load_matrix(variance_path, "variance", sources, latent_shape)
        reliability = load_matrix(reliability_path, "reliability", sources, latent_shape)
        variance = (variance * float(scale)).clamp_min(float(floor))
        if not torch.isfinite(variance).all():
            raise ValueError("Scaled variance must be finite")
        return cls(variance.to(device), reliability.to(device))

    def priority_order(self):
        return torch.argsort(self.reliability.flatten(1).T, dim=-1, descending=True, stable=True)
