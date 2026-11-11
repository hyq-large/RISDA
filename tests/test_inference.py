from datetime import datetime

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf

from appa.nn.cae import ConvDecoder
from appa.nn.vit import ViDiT
from risda.config import anchor_times
from risda.data import ConventionalObservations, SatelliteObservations, latent_mask
from risda.encoders import ObsToEraEncoder
from risda.guidance import VCLG, merge_ppgf
from risda.inference import run
from risda.uncertainty import Uncertainty


def test_ppgf_matches_priority_stop_rule():
    generator = torch.Generator().manual_seed(17)
    gradients = torch.randint(-3, 4, (6, 2, 12), generator=generator).float()
    costs = torch.rand(12, 6, generator=generator)
    order = costs.argsort(dim=-1, descending=True, stable=True)
    expected = torch.zeros(2, 12)
    for frame in range(2):
        for channel in range(12):
            total = 0.0
            for source in reversed(order[channel].tolist()):
                value = float(gradients[source, frame, channel])
                if total * value < 0:
                    break
                total += value
            expected[frame, channel] = total
    torch.testing.assert_close(merge_ppgf(list(gradients), order), expected)


def test_vclg_matches_dense_covariance():
    class LinearPrior(torch.nn.Module):
        def forward(self, state, sigma):
            return state @ matrix

    matrix = torch.tensor([[0.6, 0.1, 0.0], [0.1, 0.5, 0.15], [0.0, 0.15, 0.7]])
    noisy = torch.tensor([[0.5, -0.4, 0.2]])
    sigma = torch.tensor([0.8])
    observations = [
        {
            "indices": torch.tensor([0, 2]),
            "values": torch.tensor([1.2, -0.3]),
            "variance": torch.tensor([0.2, 0.4]),
        },
        {
            "indices": torch.tensor([1, 2]),
            "values": torch.tensor([0.8, 0.6]),
            "variance": torch.tensor([0.3, 0.1]),
        },
    ]
    order = torch.tensor([[0, 1], [1, 0], [0, 1]])
    gradients = []
    predicted = noisy @ matrix
    for observation in observations:
        index = observation["indices"]
        covariance = torch.diag(observation["variance"]) + sigma.square() * matrix[index][:, index]
        solved = torch.linalg.solve(covariance, observation["values"] - predicted[0, index])
        gradients.append((matrix[:, index] @ solved)[None])
    expected = predicted + sigma.square() * merge_ppgf(gradients, order)
    with torch.no_grad():
        actual = VCLG(LinearPrior(), observations, order, iterations=2)(noisy, sigma)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_uncertainty_inputs_and_scaling(tmp_path):
    sources = ["atms", "convi"]
    variance = torch.arange(8).reshape(2, 1, 2, 2).float()
    costs = torch.tensor([[[[2.0, 0.0], [3.0, 1.0]]], [[[1.0, 0.0], [4.0, 0.5]]]])
    var_path, rel_path = tmp_path / "variance.pth", tmp_path / "reliability.pth"
    torch.save({"sources": sources, "variance": variance}, var_path)
    torch.save({"sources": sources, "reliability": costs}, rel_path)
    result = Uncertainty.load(var_path, rel_path, sources, (1, 2, 2), scale=0.1)
    torch.testing.assert_close(result.variance, (variance.reshape(2, 2, 2) * 0.1).clamp_min(1e-8))
    torch.testing.assert_close(result.reliability, costs.reshape(2, 2, 2))
    assert result.priority_order().tolist() == [[0, 1], [0, 1], [1, 0], [0, 1]]
    with pytest.raises(ValueError, match="source order"):
        Uncertainty.load(var_path, rel_path, sources[::-1], (1, 2, 2))
    torch.save({"sources": sources, "variance": variance * float("nan")}, var_path)
    with pytest.raises(ValueError, match="finite"):
        Uncertainty.load(var_path, rel_path, sources, (1, 2, 2))


def write_nc(path, values, channels):
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"x": (("time", "channel", "lat", "lon"), values)}, coords={"channel": channels}
    ).to_netcdf(path, engine="h5netcdf")


@pytest.mark.parametrize(
    "name,channels", [("atms", 22), ("amsua", 15), ("mhs", 5), ("mwts", 17), ("mwhs", 15)]
)
def test_satellite_exact_hour_and_angle(tmp_path, name, channels):
    obs, norm = tmp_path / "observations", tmp_path / "norm"
    norm.mkdir()
    np.save(norm / f"mean_{name}.npy", np.ones(channels) * 100)
    np.save(norm / f"std_{name}.npy", np.ones(channels) * 10)
    values = np.full((1, channels + 7, 2, 2), np.nan, dtype=np.float32)
    values[0, :channels, 0, 0] = 120
    angle = channels + (3 if name in {"mwts", "mwhs"} else 0)
    values[0, angle, 0, 0] = 60
    write_nc(obs / name / f"{name}_2025040100.nc", values, np.arange(channels + 7))
    data = SatelliteObservations(obs, norm, name, use_satzen=True)
    result, mask = data.read(datetime(2025, 4, 1))
    torch.testing.assert_close(result[:channels, 0, 0], torch.full((channels,), 2.0))
    assert result[-1, 0, 0].item() == pytest.approx(0.5, abs=1e-6)
    assert mask.sum() == 1
    assert data.read(datetime(2025, 4, 1, 1)) is None


def test_surface_precedence_units_and_original_mask(tmp_path):
    norm = tmp_path / "norm"
    norm.mkdir()
    np.save(norm / "mean_era5.npy", np.zeros(69))
    np.save(norm / "std_era5.npy", np.ones(69))
    surf = np.full((1, 4, 4, 4), np.nan, dtype=np.float32)
    icoa = surf.copy()
    surf[0, 0, 1, 1] = 280
    surf[0, 3, 1, 1] = 1000
    icoa[0, 0, 1, 1] = 290
    icoa[0, 1, 1, 1] = 3
    channels = ["t2m", "u10", "v10", "slp"]
    write_nc(tmp_path / "surf/s_2025040100.nc", surf, channels)
    write_nc(tmp_path / "icoa/i_2025040100.nc", icoa, channels)
    data = ConventionalObservations(tmp_path / "surf", tmp_path / "icoa", norm, (4, 4), True)
    values, mask = data.read(datetime(2025, 4, 1))
    assert values[0, 1, 1] == 280
    assert values[1, 1, 1] == 3
    assert values[3, 1, 1] == 100000
    assert values[6, 1, 1] == 0
    assert mask.sum() == 1
    assert latent_mask(mask, 2).reshape(2, 4).tolist() == [[False, True, False, True], [False] * 4]


def test_timeline_context_does_not_change_anchor_count():
    dates = anchor_times(
        OmegaConf.create({"start": "2025-04-01T00", "end": "2025-06-30T18", "step_hours": 6})
    )
    assert len(dates) == 364
    assert dates[0] == datetime(2025, 4, 1)
    with pytest.raises(ValueError):
        anchor_times(
            OmegaConf.create({"start": "2025-04-01T00", "end": "2025-04-01T05", "step_hours": 6})
        )


def test_complete_inference_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    model = tmp_path / "encoder"
    latent = tmp_path / "latent"
    prior_dir = latent / "denoisers/prior"
    normalization = tmp_path / "normalization"
    for path in (model, latent / "ae", prior_dir, normalization):
        path.mkdir(parents=True)
    ae = {
        "name": "cae",
        "shape": [4, 4],
        "resize": 2,
        "latent_shape": [2, 4, 2],
        "lat_channels": 2,
        "in_channels": 69,
        "hid_channels": [4, 8],
        "hid_blocks": [1, 1],
        "noise_level": 0.0,
        "pixel_shuffle": True,
        "ffn_factor": 1,
        "kernel_size": 3,
    }
    OmegaConf.save(OmegaConf.create({"ae": ae}), latent / "ae/config.yaml")
    decoder = ConvDecoder(
        in_channels=2, out_channels=69, hid_channels=[4, 8], hid_blocks=[1, 1], padded_size=(4, 8)
    )
    torch.save(
        {f"decoder.{key}": value for key, value in decoder.state_dict().items()},
        latent / "ae/model.pth",
    )
    torch.save({"mean": torch.zeros(2), "std": torch.ones(2)}, latent / "stats.pth")
    encoder = ObsToEraEncoder(
        5, 2, (4, 4), 2, hid_channels=[4, 8], hid_blocks=[1, 1], kernel_size=3
    )
    torch.save(encoder.state_dict(), model / "model_best.pth")
    OmegaConf.save(
        OmegaConf.create(
            {
                "data": {"icoa_dir": None},
                "model": {"hid_channels": [4, 8], "hid_blocks": [1, 1], "kernel_size": 3},
            }
        ),
        model / "config.yaml",
    )
    backbone = {
        "name": "vit",
        "arch": "vit",
        "mod_features": 8,
        "hid_channels": 8,
        "hid_blocks": 1,
        "attention_heads": 2,
        "ffn_factor": 2,
        "rope": False,
        "blanket_size": 6,
        "patch_size": [1, 1, 1],
    }
    network = ViDiT(in_channels=2, shape=(2, 4), **backbone)
    torch.save(network.state_dict(), prior_dir / "model_best.pth")
    OmegaConf.save(
        OmegaConf.create(
            {
                "backbone": backbone,
                "train": {"noise_schedule": "rectified", "sigma_min": 0.01, "sigma_max": 1.0},
            }
        ),
        prior_dir / "config.yaml",
    )
    np.save(normalization / "mean_era5.npy", np.zeros(69))
    np.save(normalization / "std_era5.npy", np.ones(69))
    values = np.full((1, 4, 4, 4), np.nan, dtype=np.float32)
    values[0, :, 1, 1] = [280, 2, 1, 1000]
    write_nc(tmp_path / "surf/s_2025040100.nc", values, ["t2m", "u10", "v10", "slp"])
    variance_path, reliability_path = tmp_path / "variance.pth", tmp_path / "reliability.pth"
    torch.save({"sources": ["convi"], "variance": torch.ones(1, 2, 4, 2)}, variance_path)
    torch.save({"sources": ["convi"], "reliability": torch.ones(1, 2, 4, 2)}, reliability_path)
    config = OmegaConf.create(
        {
            "prior": {"model_dir": str(prior_dir), "latent_dir": str(latent), "checkpoint": "best"},
            "sources": ["convi"],
            "encoders": {"convi": {"model_dir": str(model), "checkpoint": "best"}},
            "data": {
                "surf_dir": str(tmp_path / "surf"),
                "icoa_dir": None,
                "normalization": str(normalization),
            },
            "uncertainty": {
                "variance": str(variance_path),
                "reliability": str(reliability_path),
                "var_y_scale": 1.0,
                "variance_floor": 1e-8,
            },
            "sampling": {"steps": 2, "corrections": 0, "delta": 0.1, "solver_iterations": 2},
            "timeline": {"start": "2025-04-01T00", "end": "2025-04-01T00", "step_hours": 6},
            "precision": "float32",
            "save_decoded": True,
            "seed": 0,
        }
    )
    output = tmp_path / "output"
    run(config, output, "test-contract", torch.device("cpu"))
    payload = torch.load(output / "trajectory.pth", weights_only=True)
    assert payload["latents"].shape == (1, 8, 2)
    assert np.load(output / "era5/20250401T00.npy").shape == (69, 4, 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed anchors must not be sampled again")

    monkeypatch.setattr("risda.inference.sample", forbidden)
    run(config, output, "test-contract", torch.device("cpu"))
    with pytest.raises(ValueError, match="inputs differ"):
        run(config, output, "different-contract", torch.device("cpu"))
