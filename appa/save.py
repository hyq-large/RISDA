from pathlib import Path

import torch


def safe_load(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location, weights_only=False)
