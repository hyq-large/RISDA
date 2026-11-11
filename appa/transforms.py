import torch
from torch.nn import functional as F


def cressman_interp_multi_channel(obs_value, obs_mask, confidence_ignore=0, R=10, eps=0.0001):
    if obs_value.ndim != 5 or obs_mask.ndim != 5:
        raise ValueError(
            f"obs_value and obs_mask must both have shape (B,T,C,H,W); got {tuple(obs_value.shape)} and {tuple(obs_mask.shape)}"
        )
    if obs_value.shape != obs_mask.shape:
        raise ValueError(
            f"obs_value and obs_mask shapes must match; got {tuple(obs_value.shape)} and {tuple(obs_mask.shape)}"
        )
    if not obs_value.is_floating_point():
        raise TypeError("obs_value must be floating point")
    radius = int(R)
    if radius < 1:
        raise ValueError(f"R must be a positive integer; got {R!r}")
    epsilon = float(eps)
    if not epsilon > 0.0:
        raise ValueError(f"eps must be positive; got {eps!r}")
    (batch, time, channels, height, width) = obs_value.shape
    ignored = int(confidence_ignore)
    confidence_channels = channels - ignored
    if ignored < 0 or confidence_channels < 1:
        raise ValueError(
            f"confidence_ignore must leave at least one channel; got C={channels}, confidence_ignore={confidence_ignore!r}"
        )
    mask = (obs_mask > 0).to(dtype=obs_value.dtype)
    values = torch.where(
        (mask > 0) & torch.isfinite(obs_value),
        obs_value,
        torch.zeros((), dtype=obs_value.dtype, device=obs_value.device),
    )
    flat_values = values.reshape(batch * time * channels, 1, height, width)
    flat_mask = mask.reshape(batch * time * channels, 1, height, width)
    coordinates = torch.arange(-radius, radius + 1, dtype=obs_value.dtype, device=obs_value.device)
    (yy, xx) = torch.meshgrid(coordinates, coordinates, indexing="ij")
    distance_squared = xx.square() + yy.square()
    radius_squared = float(radius * radius)
    kernel = (radius_squared - distance_squared) / (radius_squared + distance_squared + epsilon)
    kernel = torch.where(
        distance_squared < radius_squared, kernel, torch.zeros_like(kernel)
    ).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)

    def periodic_longitude_convolution(tensor):
        tensor = F.pad(tensor, (radius, radius, 0, 0), mode="circular")
        tensor = F.pad(tensor, (0, 0, radius, radius), mode="constant", value=0.0)
        return F.conv2d(tensor, kernel)

    weighted_value = periodic_longitude_convolution(flat_values)
    support_weight = periodic_longitude_convolution(flat_mask)
    has_support = support_weight > 0
    interpolated = torch.where(
        has_support,
        weighted_value / support_weight.clamp_min(epsilon),
        torch.zeros_like(weighted_value),
    )
    interpolated = torch.where(flat_mask > 0, flat_values, interpolated)
    density = support_weight / kernel.sum().clamp_min(epsilon)
    density = density.clamp_(0.0, 1.0)
    density = torch.where(flat_mask > 0, torch.ones_like(density), density)
    interpolated = interpolated.reshape(batch, time, channels, height, width)
    dilated_mask = has_support.to(obs_value.dtype).reshape(batch, time, channels, height, width)
    density = density.reshape(batch, time, channels, height, width)
    confidence = density[:, :, :confidence_channels].mean(dim=2, keepdim=True)
    return (torch.cat((interpolated, confidence), dim=2), dilated_mask)
