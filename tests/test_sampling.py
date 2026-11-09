r"""Tests for the appa.diffusion module."""

import pytest
import torch
import torch.nn as nn

from functools import partial
from torch import Tensor
from typing import Any

from appa.diffusion import Denoiser, RectifiedSchedule
from appa.sampling import DDIMSampler, DDPMSampler, LMSSampler


class DummyBackbone(nn.Module):
    def __init__(self, in_sz: int):
        super().__init__()

        self.l1 = nn.Linear(in_sz, in_sz)
        self.l2 = nn.Linear(1, in_sz)

    def forward(self, x: Tensor, t: Tensor, context: Any = None):
        t = t.unsqueeze(-1)

        return self.l1(x) + self.l2(t)


@pytest.mark.parametrize("sampler", [DDIMSampler, DDPMSampler, LMSSampler])
@pytest.mark.parametrize("features", [5])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("context", [True, False])
def test_samplers(sampler: nn.Module, features: int, batch_size: int, context: bool):
    net = DummyBackbone(features)

    if context:
        d = partial(Denoiser(net), context=torch.rand(1))
    else:
        d = Denoiser(net)

    s = sampler(d, schedule=RectifiedSchedule())

    x1 = torch.randn(batch_size, features)
    x0 = s(x1)

    assert x0.shape == x1.shape
