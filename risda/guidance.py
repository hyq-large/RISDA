import torch
from torch import nn

from appa.math import gmres


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
