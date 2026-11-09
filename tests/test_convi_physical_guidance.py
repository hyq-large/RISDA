import numpy as np
import pytest
import torch
from datetime import datetime
from omegaconf import OmegaConf
from types import SimpleNamespace
from experiments.observations import evaluate_guided_six_frame as evaluator
from experiments.observations.convi_physical import physical_normalization_contract

def _config():
    return OmegaConf.create({
        'guidance_merge':'sum', 'mask_frac':0.0, 'variance_floor':1e-8,
        'diffusion':{'mmps_iters':2, 'tweedie_covariance':True},
    })

def _normalization(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    for kind, values in (("mean", np.arange(69)), ("std", np.arange(69) + 1)):
        np.save(stats / f"{kind}_era5.npy", np.asarray(values, dtype=np.float32))
    ae = tmp_path / "ae.yaml"
    OmegaConf.save(
        {
            "normalization": {
                "target_mean_path": str(stats / "mean_era5.npy"),
                "target_std_path": str(stats / "std_era5.npy"),
            }
        },
        ae,
    )
    return ae, stats

def test_physical_normalization_rejects_equal_shape_wrong_basis(tmp_path):
    ae, stats = _normalization(tmp_path)
    contract = physical_normalization_contract(ae, stats)
    assert contract["mean"]["surface_values"] == [65, 66, 67, 68]
    other = tmp_path / "other"
    other.mkdir()
    np.save(other / "mean_era5.npy", np.arange(69, dtype=np.float32) + 1)
    np.save(other / "std_era5.npy", np.arange(69, dtype=np.float32) + 1)
    with pytest.raises(RuntimeError, match="mean normalization differ"):
        physical_normalization_contract(ae, other)

def test_constant_physical_source_does_not_read_a_variance_cache(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    OmegaConf.save({"era_latent_dir": str(tmp_path), "data": {}}, model / "config.yaml")
    dataset = SimpleNamespace(num_observation_channels=4)
    monkeypatch.setattr(evaluator, "_build_convi_dataset", lambda *_: dataset)
    monkeypatch.setattr(
        evaluator,
        "physical_normalization_contract",
        lambda *_: {"space": "normalized_era_surface"},
    )

    def no_cache(*args, **kwargs):
        pytest.fail("Constant physical guidance must not open a calibration cache")

    monkeypatch.setattr(evaluator, "load_convi_physical_var_y", no_cache)
    source = evaluator._load_convi_physical_source(
        "convi",
        OmegaConf.create({"data_model_dir": str(model)}),
        tmp_path,
        object(),
        "cpu",
        1e-8,
        OmegaConf.create({"mean_std_dir": str(tmp_path)}),
        var_y_mode="constant",
    )
    assert source["var_y_cache"] is None
    assert source["var_y"].shape == (4,)
    assert "encoder" not in source

def test_mixed_operator_uses_native_latent_and_sparse_channel_variance(monkeypatch):
    config = _config()
    config.diffusion.mmps_iters = 2
    config.diffusion.tweedie_covariance = True
    for key, value in {
        "pcgrad_overlap_only": True,
        "pcgrad_shuffle": True,
        "ppgf_overlap_only": True,
        "ppgf_aligned_clip": 0.0,
    }.items():
        config[key] = value

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

        def decode(self, native, date):
            # Four surface channels after the 65 pressure-level variables.
            zero = native.new_zeros(*native.shape[:-1], 65)
            surface = torch.stack(
                (
                    native[..., 0] ** 2,
                    native[..., 1],
                    native[..., 0],
                    self.weight * native[..., 1],
                ),
                dim=-1,
            )
            return torch.cat((zero, surface), dim=-1)

    decoder = Decoder()
    sources = [
        {
            "name": "atms",
            "kind": "obs2era",
            "var_y": torch.tensor([0.3, 0.4]),
            "var_y_mode": "channelwise",
        },
        {
            "name": "convi",
            "kind": "convi_physical",
            "era_autoencoder": decoder,
            "var_y": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "var_y_mode": "channelwise",
        },
    ]
    satellite = {"y_sel": torch.arange(4.0), "m_lat": torch.tensor([True, False, True])}
    physical = {
        "values": torch.tensor([10.0, 20.0]),
        "m_lat": torch.tensor([False, True, True]),
        "pixel_index": torch.tensor([1, 2]),
        "channel_index": torch.tensor([65, 68]),
        "channel_local": torch.tensor([0, 3]),
    }
    monkeypatch.setattr(
        evaluator,
        "_load_source_observation",
        lambda source, *_: satellite if source["name"] == "atms" else physical,
    )
    captured = {}

    def capture(base, op, y, var_y, **kwargs):
        captured.update(op=op, y=y, var_y=var_y, **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(evaluator, "MultiInstrumentMMPSDenoiser", capture)
    runtime = {
        "sources": sources,
        "device": "cpu",
        "latent_shape": (3, 2),
        "base_denoiser": None,
        "ppgf_convi_bypass_merge_mode": "retained_ppgf_plus_bypass",
        "coordinates": {
            "effective_std": torch.tensor([2.0, 3.0]),
            "mean": torch.tensor([5.0, 7.0]),
        },
    }
    evaluator._prepare_window_guidance(config, runtime, datetime(2025, 1, 10))
    state = torch.ones(1, 6 * 3 * 2, requires_grad=True)
    ops = captured["instrument_operators"]
    assert torch.equal(ops[0](state), torch.ones(24))
    assert torch.equal(ops[1](state), torch.tensor([49.0, 20.0] * 6))
    grad = torch.autograd.grad(ops[1](state).sum(), state)[0].reshape(6, 3, 2)
    assert torch.equal(grad[:, 1, 0], torch.full((6,), 28.0))
    assert torch.equal(grad[:, 2, 1], torch.full((6,), 6.0))
    assert torch.equal(grad[:, 0], torch.zeros(6, 2))
    assert decoder.weight.grad is None
    assert torch.equal(captured["var_y"], torch.tensor([0.3, 0.4, 0.3, 0.4, 0.1, 0.4] * 6))
    assert captured["merge"] == "sum"
