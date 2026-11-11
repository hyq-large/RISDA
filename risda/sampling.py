import math

import torch


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
