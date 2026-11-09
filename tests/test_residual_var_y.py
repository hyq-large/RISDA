import torch

from experiments.observations.residual_moments import finalize_moments


def test_finalize_moments_separates_bias_from_residual_variance():
    residuals = torch.tensor(
        [
            [1.0, -2.0],
            [3.0, -2.0],
            [5.0, -2.0],
        ],
        dtype=torch.float64,
    )
    moments = finalize_moments(
        residuals.sum(0),
        residuals.square().sum(0),
        torch.tensor([3, 3]),
    )

    assert torch.allclose(
        moments["bias"], torch.tensor([3.0, -2.0], dtype=torch.float64)
    )
    assert torch.allclose(
        moments["mse"],
        torch.tensor([35.0 / 3.0, 4.0], dtype=torch.float64),
    )
    assert torch.allclose(
        moments["variance"],
        torch.tensor([8.0 / 3.0, 0.0], dtype=torch.float64),
    )


def test_finalize_moments_rejects_missing_channels():
    try:
        finalize_moments(
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([1, 0]),
        )
    except RuntimeError as error:
        assert "channels [1]" in str(error)
    else:
        raise AssertionError("missing channels should be rejected")
