import torch

from experiments.observations.convi_obs2era_data import (
    CONVI_QC_TO_ERA5,
    CONVI_QC_TO_ERA5_QC_MASK,
    CONVI_QC_TO_QC,
    ERA5_QC_MASK_TO_ERA5_QC_MASK,
    SURFACE_CHANNEL_INDICES,
    build_convi_training_sample,
)
from experiments.observations.train_obs2era import _masked_physical_mse


def _sample_fields():
    qc = torch.zeros(8, 2, 3)
    for channel, (row, column), value in zip(
        range(4),
        ((0, 0), (0, 1), (1, 0), (1, 2)),
        (1.0, 2.0, 3.0, 4.0),
        strict=True,
    ):
        qc[channel, row, column] = value
        qc[channel + 4, row, column] = 1.0
    union = qc[4:8].any(dim=0, keepdim=True).float()
    era = torch.arange(69 * 2 * 3, dtype=torch.float32).reshape(69, 2, 3)
    return qc, union, era


def test_qc_to_qc_uses_raw_qc_values_and_per_variable_loss_masks():
    qc, union, era = _sample_fields()

    encoder_input, target, mask = build_convi_training_sample(
        qc,
        union,
        era,
        alignment_mode=CONVI_QC_TO_QC,
        use_cressman=True,
    )

    assert torch.equal(encoder_input, qc)
    assert target.shape == mask.shape == era.shape
    assert torch.equal(target[list(SURFACE_CHANNEL_INDICES)], qc[:4])
    assert torch.equal(mask[list(SURFACE_CHANNEL_INDICES)], qc[4:8])
    assert target[:65].count_nonzero().item() == 0
    assert mask[:65].count_nonzero().item() == 0
    assert mask.sum().item() == 4


def test_era5_qc_mask_alignment_uses_identical_input_and_target_values():
    qc, union, era = _sample_fields()
    expected_surface = era[list(SURFACE_CHANNEL_INDICES)] * qc[4:8]

    encoder_input, target, mask = build_convi_training_sample(
        qc,
        union,
        era,
        alignment_mode=ERA5_QC_MASK_TO_ERA5_QC_MASK,
        use_cressman=True,
    )

    assert torch.equal(encoder_input[:4], expected_surface)
    assert torch.equal(encoder_input[4:8], qc[4:8])
    assert torch.equal(target[list(SURFACE_CHANNEL_INDICES)], expected_surface)
    assert torch.equal(mask[list(SURFACE_CHANNEL_INDICES)], qc[4:8])
    assert target[:65].count_nonzero().item() == 0
    assert mask.sum().item() == 4


def test_qc_to_era5_qc_mask_uses_raw_convi_input_and_surface_era5_target():
    qc, union, era = _sample_fields()
    expected_surface_target = (
        era[list(SURFACE_CHANNEL_INDICES)] * qc[4:8]
    )

    encoder_input, target, mask = build_convi_training_sample(
        qc,
        union,
        era,
        alignment_mode=CONVI_QC_TO_ERA5_QC_MASK,
        use_cressman=False,
    )

    assert encoder_input.shape == (5, 2, 3)
    assert torch.equal(encoder_input[:4], qc[:4])
    assert torch.equal(encoder_input[4:], union)
    assert target.shape == mask.shape == era.shape
    assert torch.equal(
        target[list(SURFACE_CHANNEL_INDICES)],
        expected_surface_target,
    )
    assert torch.equal(mask[list(SURFACE_CHANNEL_INDICES)], qc[4:8])
    assert target[:65].count_nonzero().item() == 0
    assert mask[:65].count_nonzero().item() == 0
    assert mask.sum().item() == 4


def test_historical_qc_to_era5_mode_is_unchanged():
    qc, union, era = _sample_fields()

    encoder_input, target, mask = build_convi_training_sample(
        qc,
        union,
        era,
        alignment_mode=CONVI_QC_TO_ERA5,
        use_cressman=True,
    )

    assert torch.equal(encoder_input, qc)
    assert torch.equal(target, era)
    assert torch.equal(mask, union)


def test_channelwise_physical_mask_averages_only_selected_variable_points():
    prediction = torch.zeros(1, 2, 3)
    target = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])
    channel_mask = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
    shared_mask = torch.tensor([[[1.0], [0.0]]])

    assert torch.equal(
        _masked_physical_mse(prediction, target, channel_mask),
        torch.tensor(5.0),
    )
    assert torch.allclose(
        _masked_physical_mse(prediction, target, shared_mask),
        torch.tensor(14.0 / 3.0),
    )
