import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace


from experiments.observations.direct_e2o_assimilation import (
    finalize_residual_moments,
    prepare_direct_e2o_observation,
    select_direct_e2o_prediction,
)
from experiments.observations.direct_e2o_data import (
    DIRECT_E2O_DATA_CONTRACT,
    load_tb_target_at_time,
    split_train_valid_indices,
    strict_time_intersection,
    validate_direct_e2o_data_contract,
    validate_latent_cadence,
)
from experiments.observations.reanalysis_obs2era_dual import (
    channel_ppgf_uncertainty,
    filter_physical_observation_by_block_mask,
    load_ppgf_ordering_cache,
    select_source_configs,
)
from experiments.observations.train_era5_latent_to_tb import summarize_metrics
from risda.models.decoders import EraLatentToTbDecoder, masked_channel_sums


def test_strict_time_intersection_preserves_o2e_order():
    reference = [
        (2024, 1, 1, 0),
        (2024, 1, 1, 6),
        (2024, 1, 1, 12),
    ]
    latent = [
        (2024, 1, 1, 12),
        (2024, 1, 1, 0),
    ]

    reference_rows, latent_rows = strict_time_intersection(reference, latent)

    np.testing.assert_array_equal(reference_rows, [0, 2])
    np.testing.assert_array_equal(latent_rows, [1, 0])


def test_strict_time_intersection_rejects_duplicate_latents():
    date = (2024, 1, 1, 0)
    with pytest.raises(ValueError, match="duplicate ERA latent"):
        strict_time_intersection([date], [date, date])


def test_direct_e2o_contract_rejects_zenith_free_configs():
    with pytest.raises(RuntimeError, match="contract mismatch"):
        validate_direct_e2o_data_contract({})
    with pytest.raises(RuntimeError, match="use_satzen=true"):
        validate_direct_e2o_data_contract(
            {
                "contract": DIRECT_E2O_DATA_CONTRACT,
                "use_satzen": False,
            }
        )
    with pytest.raises(RuntimeError, match="cadence_hours"):
        validate_direct_e2o_data_contract(
            {
                "contract": DIRECT_E2O_DATA_CONTRACT,
                "use_satzen": True,
            }
        )
    assert validate_direct_e2o_data_contract(
        {
            "contract": DIRECT_E2O_DATA_CONTRACT,
            "use_satzen": True,
            "cadence_hours": 1,
        }
    ) == 1


def test_latent_cadence_distinguishes_hourly_and_six_hourly_axes():
    hourly = np.array(
        [
            [2024, 1, 1, 22],
            [2024, 1, 1, 23],
            [2024, 1, 2, 0],
            [2024, 1, 2, 1],
        ]
    )
    validate_latent_cadence(hourly, expected_hours=1)
    with pytest.raises(ValueError, match="expected consecutive 6h"):
        validate_latent_cadence(hourly, expected_hours=6)

    six_hourly = np.array(
        [
            [2024, 1, 1, 0],
            [2024, 1, 1, 6],
            [2024, 1, 1, 12],
        ]
    )
    validate_latent_cadence(six_hourly, expected_hours=6)


def test_load_tb_target_reads_only_exact_hour(monkeypatch):
    center = pd.Timestamp("2024-01-01T06:00:00")
    exact_tb = np.array(
        [
            [[10.0, np.nan], [np.nan, np.nan]],
            [[np.nan, 30.0], [np.nan, np.nan]],
        ],
        dtype=np.float32,
    )
    zenith = np.array([[60.0, 0.0], [30.0, np.nan]], dtype=np.float32)
    exact = np.concatenate((exact_tb, zenith[None]), axis=0)
    neighbour = np.full((3, 2, 2), 999.0, dtype=np.float32)
    frames = {"exact": exact, "neighbour": neighbour}
    opened = []

    class FakeDataset:
        def __init__(self, frame):
            self.x = SimpleNamespace(values=frame[None])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def open_dataset(path):
        opened.append(path)
        return FakeDataset(frames[path])

    monkeypatch.setattr(
        "experiments.observations.direct_e2o_data.xr.open_dataset",
        open_dataset,
    )
    satellite = SimpleNamespace(
        obs_channel=2,
        mean=np.array([0.0, 10.0], dtype=np.float32),
        std=np.array([10.0, 10.0], dtype=np.float32),
        satzen_channel=2,
        _all_paths={
            center: "exact",
            center - pd.Timedelta(hours=1): "neighbour",
        },
    )

    target, valid, cos_satzen = load_tb_target_at_time(
        satellite, center
    )

    assert opened == ["exact"]
    assert target[0, 0, 0] == pytest.approx(1.0)
    assert not valid[1, 0, 0]
    assert target[1, 0, 0] == pytest.approx(0.0)
    assert not valid[:, 1, 0].any()
    np.testing.assert_allclose(
        cos_satzen,
        [[[0.5, 1.0], [0.0, 0.0]]],
        atol=1e-6,
    )


def test_load_tb_target_does_not_fall_back_to_neighbour(monkeypatch):
    center = pd.Timestamp("2024-01-01T06:00:00")
    monkeypatch.setattr(
        "experiments.observations.direct_e2o_data.xr.open_dataset",
        lambda _path: pytest.fail("a neighbouring file must not be opened"),
    )
    satellite = SimpleNamespace(
        obs_channel=2,
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        satzen_channel=2,
        _all_paths={center - pd.Timedelta(hours=1): "neighbour"},
    )

    target, valid, cos_satzen = load_tb_target_at_time(
        satellite, center
    )

    assert target.shape == (2, 720, 1440)
    assert not target.any()
    assert not valid.any()
    assert cos_satzen.shape == (1, 720, 1440)
    assert not cos_satzen.any()



def test_masked_channel_sums_ignores_missing_targets():
    prediction = torch.tensor([[[1.0, 100.0], [3.0, 4.0]]])
    target = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    valid = torch.tensor([[[True, False], [True, True]]])

    squared_error, count = masked_channel_sums(prediction, target, valid)

    torch.testing.assert_close(squared_error, torch.tensor([5.0, 9.0]))
    torch.testing.assert_close(count, torch.tensor([2.0, 1.0]))


def test_metrics_and_calendar_split():
    dates = np.array([
        [2023, 1, 1, 0],
        [2024, 1, 1, 0],
        [2025, 1, 1, 0],
        [2025, 3, 31, 23],
        [2025, 4, 1, 0],
    ])
    train, valid, split = split_train_valid_indices(dates)
    np.testing.assert_array_equal(train, [0, 1])
    np.testing.assert_array_equal(valid, [2, 3])
    assert 4 not in train
    assert 4 not in valid
    assert split == "2025-01-01..2025-03-31"

    metrics = summarize_metrics(
        squared_error=[4.0, 9.0],
        count=[1.0, 1.0],
        channel_std=[2.0, 10.0],
    )
    assert metrics["normalized_rmse"] == pytest.approx(np.sqrt(6.5))
    assert metrics["per_channel_normalized_rmse"] == pytest.approx([2.0, 3.0])
    assert metrics["per_channel_kelvin_rmse"] == pytest.approx([4.0, 30.0])


def test_decoder_returns_dense_tb_grid():
    decoder = EraLatentToTbDecoder(
        out_channels=2,
        lat_channels=3,
        shape=(8, 16),
        resize=4,
        zenith_hidden_channels=4,
        hid_channels=(4, 8, 16),
        hid_blocks=(1, 1, 1),
        kernel_size=3,
        ffn_factor=1,
        dropout=0.0,
        checkpointing=False,
        pixel_shuffle=True,
        identity_init=True,
    )
    latent_height, latent_width, latent_channels = decoder.latent_shape
    latent = torch.randn(1, latent_height * latent_width, latent_channels)
    cos_satzen = torch.zeros(1, 8 * 16, 1)

    output = decoder(latent, cos_satzen)

    assert output.shape == (1, 8 * 16, 2)


def test_decoder_zenith_head_and_sparse_pixels_are_consistent():
    decoder = EraLatentToTbDecoder(
        out_channels=2,
        lat_channels=3,
        shape=(8, 16),
        resize=4,
        zenith_hidden_channels=4,
        hid_channels=(4, 8, 16),
        hid_blocks=(1, 1, 1),
        kernel_size=3,
        ffn_factor=1,
        dropout=0.0,
        checkpointing=False,
        pixel_shuffle=True,
        identity_init=True,
    )
    latent_height, latent_width, latent_channels = decoder.latent_shape
    latent = torch.randn(2, latent_height * latent_width, latent_channels)
    zero_zenith = torch.zeros(2, 8 * 16, 1)
    one_zenith = torch.ones(2, 8 * 16, 1)

    with torch.no_grad():
        decoder.zenith_head[0].weight.zero_()
        decoder.zenith_head[0].bias.zero_()
        decoder.zenith_head[0].weight[:, -1].fill_(1.0)
        decoder.zenith_head[-1].weight.fill_(0.25)
        decoder.zenith_head[-1].bias.zero_()

    zero_output = decoder(latent, zero_zenith)
    one_output = decoder(latent, one_zenith)
    assert not torch.allclose(zero_output, one_output)

    pixel_index = torch.tensor([0, 7, 63, 127])
    sparse_output = decoder.forward_at_pixels(
        latent,
        pixel_index,
        one_zenith[0].index_select(0, pixel_index),
    )
    torch.testing.assert_close(
        sparse_output,
        one_output.index_select(1, pixel_index),
    )



def test_direct_e2o_observation_and_prediction_share_flattening_order():
    target = np.array(
        [
            [[10.0, 11.0, 12.0]],
            [[20.0, 21.0, 22.0]],
        ],
        dtype=np.float32,
    )
    valid = np.array(
        [
            [[True, False, True]],
            [[False, True, True]],
        ]
    )
    cos_satzen = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    observation = prepare_direct_e2o_observation(
        target, valid, cos_satzen, torch.device("cpu")
    )

    torch.testing.assert_close(
        observation["values"], torch.tensor([10.0, 21.0, 12.0, 22.0])
    )
    torch.testing.assert_close(
        observation["channel_local"], torch.tensor([0, 1, 0, 1])
    )
    torch.testing.assert_close(
        observation["cos_satzen"],
        torch.tensor([[0.1], [0.2], [0.3]]),
    )

    prediction = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    selected = select_direct_e2o_prediction(
        prediction,
        observation["valid_mask"],
    )
    torch.testing.assert_close(selected, torch.tensor([1.0, 4.0, 5.0, 6.0]))

    selected.sum().backward()
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]),
    )


def test_direct_e2o_centered_residual_variance():
    moments = finalize_residual_moments(
        sum_error=torch.tensor([4.0, 4.0]),
        sum_square_error=torch.tensor([10.0, 8.0]),
        counts=torch.tensor([2, 2]),
    )

    torch.testing.assert_close(moments["bias"], torch.tensor([2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(
        moments["mse"], torch.tensor([5.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        moments["variance"], torch.tensor([1.0, 0.0], dtype=torch.float64)
    )


def test_channel_ppgf_broadcasts_direct_e2o_physical_uncertainty():
    uncertainty = channel_ppgf_uncertainty(
        torch.tensor([1.0, 4.0]),
        source_kind="direct_e2o",
        latent_channels=3,
    )

    torch.testing.assert_close(uncertainty, torch.full((3,), 2.0))


def _save_guided_latent_rmse_matrix(path):
    rmse = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    torch.save(
        {
            "kind": "guided_latent_native_rmse_matrix_6x128_v1",
            "source_order": ["atms", "convi"],
            "polar_guidance": "decoder",
            "convi_guidance": "encoder",
            "mask_frac": 0.01,
            "rmse_native": rmse,
        },
        path,
    )
    return rmse


def test_ppgf_ordering_cache_loads_exact_route_and_matrix(tmp_path):
    cache_path = tmp_path / "ordering.pth"
    expected = _save_guided_latent_rmse_matrix(cache_path)

    loaded = load_ppgf_ordering_cache(
        cache_path,
        expected_sources=["atms", "convi"],
        latent_channels=4,
        expected_mask_frac=0.01,
        expected_polar_guidance="decoder",
        expected_convi_guidance="encoder",
    )

    torch.testing.assert_close(loaded, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_mask_frac", 0.0, "mask_frac mismatch"),
        ("expected_polar_guidance", "encoder", "polar_guidance mismatch"),
        ("expected_convi_guidance", "physical", "convi_guidance mismatch"),
    ],
)
def test_ppgf_ordering_cache_rejects_route_mismatch(
    tmp_path, field, value, message
):
    cache_path = tmp_path / "ordering.pth"
    _save_guided_latent_rmse_matrix(cache_path)
    arguments = {
        "expected_sources": ["atms", "convi"],
        "latent_channels": 4,
        "expected_mask_frac": 0.01,
        "expected_polar_guidance": "decoder",
        "expected_convi_guidance": "encoder",
    }
    arguments[field] = value

    with pytest.raises(RuntimeError, match=message):
        load_ppgf_ordering_cache(cache_path, **arguments)


def test_ppgf_ordering_cache_rejects_source_order(tmp_path):
    cache_path = tmp_path / "ordering.pth"
    _save_guided_latent_rmse_matrix(cache_path)

    with pytest.raises(RuntimeError, match="source order mismatch"):
        load_ppgf_ordering_cache(
            cache_path,
            expected_sources=["convi", "atms"],
            latent_channels=4,
            expected_mask_frac=0.01,
            expected_polar_guidance="decoder",
            expected_convi_guidance="encoder",
        )







def test_source_filter_selects_satellite_or_physical_convi():
    instruments = [{"name": "atms"}, {"name": "mhs"}]

    selected, convi_enabled = select_source_configs(
        instruments, True, "mhs"
    )
    assert selected == [{"name": "mhs"}]
    assert not convi_enabled

    selected, convi_enabled = select_source_configs(
        instruments, True, "convi"
    )
    assert selected == []
    assert convi_enabled

    with pytest.raises(ValueError, match="matched 0"):
        select_source_configs(instruments, True, "missing")


def test_true_block_filter_aligns_direct_e2o_and_convi_values():
    height = width = 64
    channels = 2
    target = np.zeros((channels, height, width), dtype=np.float32)
    valid = np.zeros_like(target, dtype=bool)
    cos_satzen = np.arange(
        height * width, dtype=np.float32
    ).reshape(1, height, width)

    low_density_pixels = [(0, longitude) for longitude in range(10)]
    retained_pixels = [(32, 32 + offset) for offset in range(11)]
    for offset, (latitude, longitude) in enumerate(low_density_pixels):
        target[0, latitude, longitude] = 10.0 + offset
        valid[0, latitude, longitude] = True
    for offset, (latitude, longitude) in enumerate(retained_pixels):
        target[0, latitude, longitude] = 100.0 + offset
        valid[0, latitude, longitude] = True
        if offset % 2 == 0:
            target[1, latitude, longitude] = 200.0 + offset
            valid[1, latitude, longitude] = True

    direct = prepare_direct_e2o_observation(
        target, valid, cos_satzen, torch.device("cpu")
    )
    pixel_mask = direct.pop("pixel_mask")
    scalar_pixel_index = direct["pixel_index"][:, None].expand_as(
        direct["valid_mask"]
    )[direct["valid_mask"]]
    convi = {
        "pixel_index": scalar_pixel_index,
        "channel_index": direct["channel_local"] + 65,
        "channel_local": direct["channel_local"],
        "values": direct["values"],
    }

    unchanged = filter_physical_observation_by_block_mask(
        direct, pixel_mask, resize=32, mask_frac=0.0
    )
    torch.testing.assert_close(unchanged["values"], direct["values"])
    torch.testing.assert_close(
        unchanged["pixel_index"], direct["pixel_index"]
    )
    torch.testing.assert_close(
        unchanged["cos_satzen"], direct["cos_satzen"]
    )

    filtered_direct = filter_physical_observation_by_block_mask(
        direct, pixel_mask, resize=32, mask_frac=0.01
    )
    filtered_convi = filter_physical_observation_by_block_mask(
        convi, pixel_mask, resize=32, mask_frac=0.01
    )

    expected_pixels = torch.tensor(
        [latitude * width + longitude for latitude, longitude in retained_pixels]
    )
    torch.testing.assert_close(
        filtered_direct["pixel_index"], expected_pixels
    )
    torch.testing.assert_close(
        filtered_direct["cos_satzen"],
        torch.from_numpy(cos_satzen.reshape(-1, 1)).index_select(
            0, expected_pixels
        ),
    )
    assert filtered_direct["values"].numel() == 17
    assert filtered_direct["block_filter_values_before"] == 27
    assert filtered_direct["block_filter_values_after"] == 17
    assert filtered_direct["block_filter_pixels_before"] == 21
    assert filtered_direct["block_filter_pixels_after"] == 11
    # The retained longitude block also has one circular duplicate.
    assert int(filtered_direct["m_lat"].sum()) == 2

    prediction = torch.from_numpy(target).movedim(0, -1).reshape(
        1, height * width, channels
    )
    compact_prediction = prediction.index_select(
        1, filtered_direct["pixel_index"]
    )
    selected = select_direct_e2o_prediction(
        compact_prediction, filtered_direct["valid_mask"]
    )
    torch.testing.assert_close(selected, filtered_direct["values"])
    torch.testing.assert_close(
        filtered_convi["values"], filtered_direct["values"]
    )
    torch.testing.assert_close(
        filtered_convi["channel_local"],
        filtered_direct["channel_local"],
    )
    assert torch.unique(filtered_convi["pixel_index"]).numel() == 11
