import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from appa.math import gmres


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


def merge_ppgf(gradients, order):
    if not gradients:
        raise ValueError("At least one source gradient is required")
    shape = gradients[0].shape
    if any(gradient.shape != shape for gradient in gradients):
        raise ValueError("Source gradients must share the same shape")
    if order.ndim != 2 or order.shape[1] != len(gradients):
        raise ValueError("Priority order must have shape (latent scalars, sources)")
    width = order.shape[0]
    if width < 1 or gradients[0].numel() % width:
        raise ValueError("Priority order does not match the latent grid")
    stacked = torch.stack([gradient.reshape(-1, width) for gradient in gradients])
    index = order.T[:, None, :].expand(-1, stacked.shape[1], -1)
    ordered = torch.gather(stacked, 0, index)
    accumulator = ordered[0].clone()
    for gradient in ordered[1:]:
        conflict = (
            (accumulator != 0)
            & (gradient != 0)
            & (torch.signbit(accumulator) != torch.signbit(gradient))
        )
        accumulator = torch.where(conflict, gradient, accumulator + gradient)
    return accumulator.reshape(shape)


class VCLG(nn.Module):
    def __init__(self, denoiser, observations, priority_order, iterations=2):
        super().__init__()
        if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 1:
            raise ValueError("Solver iterations must be a positive integer")
        self.denoiser = denoiser
        self.observations = observations
        self.iterations = int(iterations)
        self.register_buffer("priority_order", priority_order)

    def forward(self, noisy, sigma, **kwargs):
        with torch.enable_grad():
            noisy = noisy.detach().requires_grad_()
            predicted = self.denoiser(noisy, sigma, **kwargs)
        gradients = []
        for observation in self.observations:
            indices = observation["indices"]
            if indices.numel() == 0:
                gradients.append(torch.zeros_like(noisy))
                continue
            with torch.enable_grad():
                selected = predicted.reshape(-1)[indices]
            residual = observation["values"] - selected
            variance = observation["variance"]

            def covariance(vector, indices=indices, variance=variance):
                adjoint = torch.zeros_like(predicted).reshape(-1).scatter(0, indices, vector)
                propagated = torch.autograd.grad(
                    predicted,
                    noisy,
                    adjoint.reshape_as(predicted),
                    retain_graph=True,
                )[0]
                return variance * vector + (sigma.square() * propagated).reshape(-1)[indices]

            solved = gmres(covariance, residual, iterations=self.iterations)
            gradient = torch.autograd.grad(selected, noisy, solved, retain_graph=True)[0]
            gradients.append(gradient)
        return predicted + sigma.square() * merge_ppgf(gradients, self.priority_order)


@torch.no_grad()
def sample(denoiser, schedule, initial, dates, steps=64, corrections=2, delta=0.1):
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError("Sampling steps must be a positive integer")
    if isinstance(corrections, bool) or int(corrections) != corrections or corrections < 0:
        raise ValueError("Corrections must be a nonnegative integer")
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("Correction amplitude must be finite and nonnegative")
    dt = torch.as_tensor(1 / steps, device=initial.device)
    times = torch.linspace(schedule.t_max, dt, steps, device=initial.device)
    state = initial
    for time in times:
        sigma = schedule(time)
        for _ in range(corrections):
            predicted = denoiser(state, sigma, date=dates)
            state = (
                state
                + delta * (predicted - state)
                + math.sqrt(2 * delta) * sigma * torch.randn_like(state)
            )
        predicted = denoiser(state, sigma, date=dates)
        state = predicted + schedule(time - dt) / sigma * (state - predicted)
        if not torch.isfinite(state).all():
            raise RuntimeError(f"Nonfinite sample at diffusion time {float(time)}")
    return state
