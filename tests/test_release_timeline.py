"""End-to-end contracts for configurable dates and the paper comparison methods."""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from appa.math import merge_gradients_by_channel_priority
from risda.environment import ROOT, setup
from risda.run import PRESETS, make_config
from risda.timeline import calibration_checks, select_anchors, time_axis, validate_timeline
from experiments.observations import evaluate_guided_six_frame as ev
from experiments.observations import evaluate_guided_six_frame_metrics as metrics


def test_default_and_configured_timeline_preserve_hourly_calibration_and_subset_seeds(monkeypatch, tmp_path):
    monkeypatch.setenv('RISDA_OUTPUT_ROOT', str(tmp_path))
    setup()
    cfg = make_config(ROOT/'configs/assimilation.yaml', 'PPGF+VCLG')
    calibration, evaluation = validate_timeline(cfg)
    assert len(calibration) == 2160 and len(evaluation) == 364
    assert evaluation[0] == datetime(2025, 4, 1)
    assert evaluation[-1] == datetime(2025, 6, 30, 18)
    assert list(cfg.enabled_sources) == ['atms', 'amsua', 'mhs', 'mwts', 'mwhs', 'convi']
    cfg.anchor_start_time = '2025-04-02T00'
    cfg.anchor_end_time = '2025-04-02T12'
    selected, indices = select_anchors(cfg)
    assert indices == [4, 5, 6] and selected == evaluation[4:7]
    assert select_anchors(cfg, canonical=True)[0] == evaluation
    custom = make_config(ROOT/'configs/assimilation.yaml', 'PPGF+VCLG', [
        'timeline.calibration.start=2023-11-02T03', 'timeline.calibration.end=2023-11-03T19',
        'timeline.evaluation.start=2024-02-29T00', 'timeline.evaluation.end=2024-03-01T00',
        'timeline.evaluation.step_hours=3'])
    assert len(validate_timeline(custom)[0]) == 41
    assert len(ev.select_config_anchors(custom)[0]) == 9
    assert custom.calibration_period == '2023-11-02T03--2023-11-03T19'
    assert custom.calibration_period in custom.calibration_root


@pytest.mark.parametrize('start,end,step', [
    ('2025-01-02T00', '2025-01-01T00', 1),
    ('2025-01-01T00', '2025-01-01T07', 6),
    ('2025-01-01T00', '2025-01-01T06', 0),
    ('2025-01-01T00', '2025-01-01T06', True),
])
def test_invalid_time_axes_are_rejected(start, end, step):
    with pytest.raises(ValueError):
        time_axis(start, end, step)


def test_overlapping_calibration_and_test_anchors_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('RISDA_OUTPUT_ROOT', str(tmp_path))
    setup()
    with pytest.raises(ValueError, match='before'):
        make_config(ROOT/'configs/assimilation.yaml', 'AGF+VCLG', [
            'timeline.evaluation.start=2025-03-31T18'])


def test_paper_methods_and_source_mean_covariance(monkeypatch, tmp_path):
    monkeypatch.setenv('RISDA_OUTPUT_ROOT', str(tmp_path))
    setup()
    assert set(PRESETS) == {'Appa', 'AGF_m', 'AGF+VCLG', 'PCGrad+VCLG', 'PPGF_SD+VCLG', 'PPGF+VCLG'}
    for method in ('PPGF_SD+VCLG', 'PPGF+VCLG'):
        cfg = make_config(ROOT/'configs/assimilation.yaml', method)
        assert cfg.var_y_mode == 'spatial' and cfg.ppgf_order_mode == 'fixed'
        assert cfg.ppgf_ordering_cache.endswith(
            'encoder_channel_mse_ordering.pth' if method.startswith('PPGF_SD') else 'encoder_spatial_mse_ordering.pth')
    values = ev.apply_var_y_mode(torch.tensor([1., 5.]), 'source_mean', None, 1e-8, scale=2)
    torch.testing.assert_close(values, torch.tensor([6., 6.]))


def test_fixed_fusion_matches_paper_high_priority_stop_rule():
    generator = torch.Generator().manual_seed(73)
    gradients = torch.randint(-3, 4, (6, 7, 13), generator=generator).float()
    mse = torch.rand(13, 6, generator=generator)
    low_to_high = mse.argsort(dim=-1, descending=True, stable=True)
    actual = merge_gradients_by_channel_priority(list(gradients), low_to_high, 13)
    expected = torch.zeros(7, 13)
    for pixel in range(7):
        for channel in range(13):
            for source in reversed(low_to_high[channel].tolist()):
                value = gradients[source, pixel, channel]
                if expected[pixel, channel] * value < 0:
                    break
                expected[pixel, channel] += value
    torch.testing.assert_close(actual, expected)


def test_cache_rejects_changed_dates_or_incomplete_hour_count(tmp_path):
    period = '2024-12-31T22--2025-01-01T02'
    payload = {**calibration_checks(period), 'kind': ev.RESIDUAL_VAR_Y_KIND,
               'representation': 'encoder', 'source': 'mwhs',
               'window_end_inclusive': True, 'num_requested_times': 5,
               'variance': torch.tensor([1., 2.])}
    path = tmp_path/'cache.pth'
    torch.save(payload, path)
    ev.load_calibrated_var_y_cache(path, 'encoder', 'mwhs', 2, expected_period=period)
    with pytest.raises(RuntimeError, match='mismatch'):
        ev.load_calibrated_var_y_cache(path, 'encoder', 'mwhs', 2,
                                      expected_period='2024-12-31T22--2025-01-01T03')
    payload['num_requested_times'] = 4
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='every candidate hour'):
        ev.load_calibrated_var_y_cache(path, 'encoder', 'mwhs', 2, expected_period=period)


def _saved_run(tmp_path):
    latent_dir = tmp_path/'latents'
    (latent_dir/'ae').mkdir(parents=True)
    cfg = OmegaConf.create({'id': 'PPGF+VCLG', 'split': 'evaluation', 'route': 'encoder',
        'enabled_sources': ['mwts', 'mwhs', 'convi'], 'num_samples_per_anchor': 1,
        'timeline': {'calibration': {'start':'2024-12-31T00', 'end':'2024-12-31T23'},
                     'evaluation': {'start':'2025-01-01T00', 'end':'2025-01-01T06', 'step_hours':6}}})
    ae = {'ae': {'latent_shape':[1, 2, 3], 'shape':[2, 2]}, 'data_path': 'unused'}
    OmegaConf.save(ae, latent_dir/'ae/config.yaml')
    torch.save({'mean':torch.zeros(3), 'std':torch.ones(3)}, latent_dir/'stats.pth')
    anchors, _ = ev.select_config_anchors(cfg)
    contract = {'protocol_kind':ev.PROTOCOL_KIND, 'split':cfg.split, 'route':cfg.route,
        'source_order':list(cfg.enabled_sources), 'num_anchors':len(anchors),
        'split_bounds':[ev.format_hour(t) for t in (anchors[0],anchors[-1])],
        'anchor_ranges':ev.serialized_anchor_ranges(anchors), 'anchor_interval_hours':6,
        'timeline':OmegaConf.to_container(cfg.timeline), 'anchor_frame_number':4,
        'window_offsets_hours':list(ev.WINDOW_OFFSETS), 'independent_windows':True,
        'latent_coordinate': {'stats':ev.file_identity(latent_dir/'stats.pth'),
                              'ae':ev.file_identity(latent_dir/'ae/config.yaml'), 'ae_noise_level':0.0}}
    contract_hash = ev._contract_hash(contract)
    run_dir = tmp_path/'run'
    run_dir.mkdir()
    OmegaConf.save(cfg, run_dir/'config.yaml')
    (run_dir/'manifest.json').write_text(json.dumps({'kind':ev.PROTOCOL_KIND,
        'generation_contract':contract, 'contract_hash':contract_hash}))
    for index, anchor in enumerate(anchors):
        ev._atomic_torch_save({'kind':ev.ANCHOR_KIND, 'contract_hash':contract_hash,
            'anchor_index':index, 'anchor':ev.timestamp_tensor(anchor),
            'window_timestamps':torch.stack([ev.timestamp_tensor(t) for t in ev.window_times(anchor)]),
            'frame_number':4, 'latent_coordinate':'diffusion_standardized',
            'latent':torch.zeros(2,3)}, ev._anchor_path(run_dir/'anchors',index))
    ev.merge_anchor_outputs(anchors, run_dir, contract_hash, (2,3),
        {'generation_contract':contract})
    return run_dir


def test_saved_timeline_and_completion_are_checked(tmp_path):
    directory = _saved_run(tmp_path)
    run = metrics._load_run(directory)
    assert len(run['anchors']) == 2
    cfg = OmegaConf.load(directory/'config.yaml')
    cfg.timeline.evaluation.end = '2025-01-01T12'
    OmegaConf.save(cfg, directory/'config.yaml')
    with pytest.raises(RuntimeError, match='differ'):
        metrics._load_run(directory)


def test_metrics_decode_once_per_anchor_and_write_only_era_and_temp(tmp_path, monkeypatch):
    run = metrics._load_run(_saved_run(tmp_path))
    run['metrics_output_dir'] = run['run_dir']
    calls = []
    class Decoder(torch.nn.Module):
        def decode(self, native, date):
            calls.append(date.tolist())
            return torch.ones(1,4,69)
    monkeypatch.setattr(metrics,'load_auto_encoder', lambda *a,**kw:Decoder())
    monkeypatch.setattr(metrics,'_load_coordinates', lambda *a: {
        'mean':torch.zeros(1,1,3), 'effective_std':torch.ones(1,1,3),
        'target_mean':torch.zeros(69), 'target_std':torch.full((69,),2.)})
    truth = [(torch.zeros(1,69,2,2),None)]*2
    monkeypatch.setattr(metrics,'_truth_dataset', lambda *a,**kw: (
        truth, {ev.timestamp_key(t):i for i,t in enumerate(run['anchors'])}))
    temp = SimpleNamespace(grid_shape=(2,2), prepare_sparse=lambda t: {
        'pixel_index':torch.tensor([0]), 'channel_index':torch.tensor([13]),
        'values':torch.tensor([3.])})
    monkeypatch.setattr(metrics,'_convi_normalization',lambda *a:(torch.zeros(69),torch.full((69,),4.)))
    totals = metrics.compute_local_totals(run,[0,1],temp,precision='float32',device='cpu',era_latitude_weighting='none')
    result = metrics.write_metrics(run,totals,{'enabled':True},'none')
    assert len(calls) == 2
    assert result['era69']['rmse_physical'] == [2.]*69
    assert result['temp']['rmse_physical'][13] == 10.
    assert result['temp']['counts'][13] == 2
    assert result['temp']['rmse_physical'][0] is None
    assert {p.name for p in run['run_dir'].glob('*rmse.csv')} == {'era69_rmse.csv','convi_temp_all_rmse.csv'}
    totals['num_frames'] -= 1
    with pytest.raises(RuntimeError,match='incomplete'):
        metrics.write_metrics(run,totals,{'enabled':True})
