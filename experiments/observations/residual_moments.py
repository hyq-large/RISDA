"""Centered residual moments for diagnostics."""
import torch

def finalize_moments(sum_values, sum_squares, counts):
    """Return population moments, with variance centered by the residual bias."""
    sum_values = torch.as_tensor(sum_values, dtype=torch.float64)
    sum_squares = torch.as_tensor(sum_squares, dtype=torch.float64)
    counts = torch.as_tensor(counts, dtype=torch.int64)
    if not (
        sum_values.shape == sum_squares.shape == counts.shape
    ):
        raise ValueError("moment tensors must share a shape")
    if (counts <= 0).any():
        missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"no residual samples for channels {missing}")
    bias = sum_values / counts
    mse = sum_squares / counts
    variance = (mse - bias.square()).clamp_min(0)
    return {
        "bias": bias,
        "mse": mse,
        "rmse": mse.sqrt(),
        "variance": variance,
        "std": variance.sqrt(),
        "counts": counts,
    }
