"""Boundary checks: public calibration caches must work in the actual evaluator."""
import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from appa.nn.cae import conv_ae
from experiments.observations import evaluate_guided_six_frame as ev
from risda import calibrate as cal
from risda import calibrate_decoder as dec
from risda.environment import ROOT, setup
from risda.run import PRESETS, make_config


@pytest.mark.parametrize('period,sources,start,end', [
    ('2024-11-17T03--2024-11-19T19', ('mwts','mwhs','convi'), '2024-11-17T03','2024-11-19T19'),
    ('2025-01-01T00--2025-03-31T23', ('atms','amsua','mhs','mwts','mwhs','convi'), '2025-01-01T00','2025-03-31T23'),
])
def test_calibration_bundle_roundtrip(tmp_path, monkeypatch, period, sources, start, end):
    for name, value in {'OUTPUT': tmp_path/'calibration', 'PERIOD':period,
                        'START':start,'END':end,'SOURCES':sources}.items():
        monkeypatch.setattr(cal,name,value)
    shape = (2,3,4)
    pixels, channels = 6,4
    ae = tmp_path/'ae.yaml'
    OmegaConf.save({'ae': {'latent_shape':list(shape)}},ae)
    stats = tmp_path/'stats.pth'
    torch.save({'mean':torch.zeros(channels),'std':torch.ones(channels)},stats)
    ident = lambda p: cal.paper.file_identity(p,include_sha256=True)
    protocol = {'models':{}}
    times = cal.hourly_axis()
    checkpoints = {}
    for source in sources:
        checkpoint = tmp_path/f'{source}.pth'
        checkpoint.write_bytes(b'synthetic checkpoint identity')
        checkpoints[source] = checkpoint
        spec = {'model_dir':str(tmp_path/source),'checkpoint':'best','data':{},
                'channelwise_precision':'float32','normalization_files':{},
                'identities':{'checkpoint':ident(checkpoint), 'ae_config':ident(ae),
                              'era_stats':ident(stats)}}
        protocol['models'][source] = spec
        # Two positions have no samples, one has just one: exercise spatial fallback.
        counts = torch.tensor([len(times),len(times),len(times),1,0,0])
        bias = torch.arange(1,channels+1,dtype=torch.float64).expand(pixels,-1)/10
        variance = torch.full_like(bias,0.2)
        total = cal.empty_totals(pixels,channels)
        total.update(sum=bias*counts[:,None],square=(variance+bias.square())*counts[:,None],counts=counts)
        total['channel_sum'] = total['sum'].sum(0)
        total['channel_square'] = total['square'].sum(0)
        frames = [{'timestamp':cal.paper.format_hour(t), 'observation_present':True,
                   'supported_blocks':4 if i == 0 else 3} for i,t in enumerate(times)]
        state = {'complete':True,'contract':{'protocol':protocol,'source':source,'rank':0},
                 'totals':total,'frames':frames}
        # Write after all source identities exist, because contracts contain the whole protocol.
        spec['_state'] = state
    states = {source:protocol['models'][source].pop('_state') for source in sources}
    for source,state in states.items():
        cal.save_torch(state,cal.shard_path(cal.stage_dir(),source,0,1))
    cal.build_bundle(1,False,protocol)
    rows = []
    scale = torch.full((channels,),0.5)
    for source in sources:
        variance, payload = ev.load_calibrated_var_y_cache(
            cal.stage_dir()/'channelwise'/f'{source}_var_y.pth',
            'encoder',source,channels,checkpoints[source],coordinate_scale=scale,
            expected_period=period)
        torch.testing.assert_close(variance,torch.full((channels,),0.05))
        torch.testing.assert_close(payload['mse'],payload['variance']+payload['bias'].square())
        rows.append(variance)
    spatial,_ = ev.load_spatial_var_y_cache(
        cal.stage_dir()/'encoder_spatial_var_y.pth',expected_sources=sources,
        expected_spatial_shape=shape,expected_channels=channels,
        expected_checkpoints=checkpoints,coordinate_scale=scale,
        channelwise_fallback=torch.stack(rows),expected_period=period)
    torch.testing.assert_close(spatial,torch.full((len(sources),pixels,channels),0.05))
    cfg = OmegaConf.create({
        'var_y_mode':'spatial', 'var_y_constant':None, 'var_y_scale':1.0,
        'variance_floor':1e-8, 'guidance_merge':'channel_ppgf', 'route':'encoder',
        'mask_frac':0.0, 'ppgf_order_mode':'fixed', 'ppgf_overlap_only':False,
        'ppgf_ordering_source':'matrix_cache', 'calibration_period':period,
        'spatial_var_y_cache':str(cal.stage_dir()/'encoder_spatial_var_y.pth'),
        'ppgf_ordering_cache':str(cal.stage_dir()/'encoder_channel_mse_ordering.pth')})
    source_runtime = [{'name':name, 'var_y':row, 'checkpoint_path':checkpoints[name],
                       'var_y_cache':cal.stage_dir()/'channelwise'/f'{name}_var_y.pth'}
                      for name,row in zip(sources,rows)]
    result = ev.configure_source_uncertainties(cfg, source_runtime, route='encoder',
        latent_channels=channels, effective_std=torch.ones(channels), spatial_shape=shape,
        coordinate_scale=scale)
    assert result['ordering_granularity'] == 'channel'
    assert all(tuple(source['var_y'].shape) == (pixels, channels) for source in source_runtime)
    for level in ('channel','spatial'):
        order,_ = ev.load_channel_ppgf_ordering_cache(
            cal.stage_dir()/f'encoder_{level}_mse_ordering.pth',sources,channels,shape,
            expected_period=period)
        assert torch.isfinite(order).all() and (order > 0).all()


def test_decoder_constant_weights_sources_not_channels(monkeypatch):
    monkeypatch.setattr(dec,'SOURCES',('small','large'))
    result = dec.mean_constants({
        'small':{'variance':torch.tensor([1.]),'mse':torch.tensor([2.]),'counts':torch.tensor([3])},
        'large':{'variance':torch.tensor([3.,3.,3.]),'mse':torch.tensor([4.,4.,4.]),'counts':torch.tensor([7,7,7])},
    })
    assert result['var_y'] == 2.0
    assert result['source_weights'] == {'small':0.5,'large':0.5}
    assert result['variance']['equal_channel'] == 2.5


def test_public_presets_keep_common_inputs_and_fixed_order(tmp_path,monkeypatch):
    monkeypatch.setenv('RISDA_OUTPUT_ROOT',str(tmp_path))
    setup()
    configs = [make_config(ROOT/'configs/assimilation.yaml',p,allow_missing_constant=True) for p in PRESETS]
    for field in ('model_path','enabled_sources','seed','split','timeline','diffusion','var_y_scale'):
        values = [OmegaConf.to_container(c[field],resolve=True) if OmegaConf.is_config(c[field]) else c[field] for c in configs]
        assert all(v == values[0] for v in values)
    assert all(c.ppgf_order_mode == 'fixed' for c in configs)
    assert all(c.convi.guidance_value_source == 'observation' for c in configs)


def test_smoke_constant_cannot_enter_public_run(tmp_path,monkeypatch):
    monkeypatch.setenv('RISDA_OUTPUT_ROOT',str(tmp_path))
    setup()
    path = tmp_path/'calibration/2025-01-01T00--2025-03-31T23/decoder/decoder_constant.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'smoke':True}))
    with pytest.raises(ValueError,match='protocol'):
        make_config(ROOT/'configs/assimilation.yaml','Appa')


def test_small_era69_ae_train_save_reload(tmp_path):
    kwargs = dict(in_channels=69,lat_channels=4,shape=(8,16),resize=2,
                  hid_channels=[8,16],hid_blocks=[1,1],pixel_shuffle=True,
                  ffn_factor=1,dropout=0.0,noise_level=0.0)
    model = conv_ae(**kwargs)
    optim = torch.optim.AdamW(model.parameters(),lr=1e-3)
    x = torch.randn(1,8*16,69)
    date = torch.tensor([[2024,10,1,0]])
    latent,reconstruction = model(x,date)
    loss = (reconstruction-x).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
    optim.step()
    path = tmp_path/'ae.pth'
    torch.save(model.state_dict(),path)
    restored = conv_ae(**kwargs)
    restored.load_state_dict(torch.load(path,weights_only=True))
    model.eval()
    restored.eval()
    with torch.no_grad():
        z,y = model(x,date)
        z2,y2 = restored(x,date)
    assert y.shape == x.shape and z.shape[-1] == 4
    torch.testing.assert_close(z,z2)
    torch.testing.assert_close(y,y2)

