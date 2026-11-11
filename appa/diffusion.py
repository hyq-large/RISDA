import math

import torch
from torch import Tensor, nn


class Denoiser(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, xt: Tensor, sigma_t: Tensor, **kwargs) -> Tensor:
        c_skip = 1 / (sigma_t**2 + 1)
        c_out = sigma_t / torch.sqrt(sigma_t**2 + 1)
        c_in = 1 / torch.sqrt(sigma_t**2 + 1)
        c_noise = 10.0 * torch.log(sigma_t)
        c_noise = c_noise.squeeze(dim=-1)
        x_out = self.backbone(c_in * xt, c_noise, **kwargs)
        return c_skip * xt + c_out * x_out


class Schedule(nn.Module):
    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 1000.0, bfloat16: bool = False):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
        self.t_max = 1 - 1 / 256 if bfloat16 else 1.0
        if bfloat16:
            print(
                "Warning: Using bfloat16 precision, please make sure to use schedule.sigma_tmax (t_max ≠ 1.0!)."
            )

    def sigma_tmax(self) -> Tensor:
        return self.forward(torch.tensor([self.t_max]))


class LogLinearSchedule(Schedule):
    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * t
        ).unsqueeze(-1)


class LogLogitSchedule(Schedule):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 1000.0,
        bfloat16: bool = False,
        spread: float = 2.0,
    ):
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, bfloat16=bfloat16)
        self.eps = math.sqrt(sigma_min / sigma_max) ** (1 / spread)
        self.log_sigma_med = math.log(sigma_min * sigma_max) / 2
        self.spread = spread

    def forward(self, t: Tensor) -> Tensor:
        return torch.exp(
            self.spread * torch.logit(t * (1 - 2 * self.eps) + self.eps) + self.log_sigma_med
        ).unsqueeze(-1)


class RectifiedSchedule(Schedule):
    def forward(self, t: Tensor) -> Tensor:
        return ((t + (1 - t) * self.sigma_min) / (t * (1 / self.sigma_max - 1) + 1)).unsqueeze(-1)
