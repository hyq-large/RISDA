"""Evaluate saved analyses against ERA5 (69 channels) and optional TEMP.

Each retained frame is decoded once. ERA5 uses cosine-latitude-weighted RMSE;
TEMP uses physical RMSE at all valid upper-air observation records. Surface
observations are assimilation inputs, not a separate observation metric.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from appa.data.datasets import CONVIOBS
from appa.save import load_auto_encoder, safe_load
from experiments.observations.evaluate_guided_six_frame import (
    ANCHOR_FRAME_NUMBER, CHANNEL_NAMES, COLLECTION_KIND, ENSEMBLE_REDUCTION,
    PROTOCOL_KIND, WINDOW_OFFSETS, _anchor_path, _atomic_torch_save,
    _atomic_write_text, _contract_hash, _initialize_cpu_coordination,
    _num_samples_per_anchor, _precision_context, _resolve_convi_qc_directory,
    _truth_dataset, _validate_anchor_payload, file_identity, finalize_weighted_rmse,
    format_hour, partition_anchor_indices, select_config_anchors,
    serialized_anchor_ranges, timestamp_key, timestamp_tensor,
)

ERA_METRICS_KIND = "risda_era69_rmse_v1"
ERA_UNWEIGHTED_METRICS_KIND = "risda_era69_unweighted_rmse_v1"
TEMP_METRICS_KIND = "risda_temp_rmse_v1"
EVALUATION_KIND = "risda_evaluation_v1"
TEMP_CHANNEL_STOP = CONVIOBS.NUM_TEMP_CHANNELS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--metrics-output-subdir')
    parser.add_argument('--data-path', type=Path, help='ERA5 truth directory override')
    parser.add_argument('--precision', choices=('float32', 'float16', 'bfloat16'))
    parser.add_argument('--era-latitude-weighting', choices=('cosine', 'none'), default='cosine')
    parser.add_argument('--log-every', type=int, default=8)
    parser.add_argument('--coordination-timeout-seconds', type=int, default=21600)
    return parser.parse_args()


def _load_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    paths = {"config": run_dir/'config.yaml', "manifest": run_dir/'manifest.json',
             "trajectory": run_dir/'frame4_latents.pth'}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config = OmegaConf.load(paths['config'])
    manifest = json.loads(paths['manifest'].read_text())
    contract = manifest['generation_contract']
    if manifest.get('kind') != PROTOCOL_KIND or contract.get('protocol_kind') != PROTOCOL_KIND:
        raise RuntimeError('Unsupported generation protocol')
    if _contract_hash(contract) != manifest['contract_hash']:
        raise RuntimeError('Generation contract hash mismatch')
    anchors, source_indices = select_config_anchors(config)
    canonical, _ = select_config_anchors(config, canonical=True)
    checks = {
        'split': str(config.split), 'route': str(config.route),
        'source_order': list(config.enabled_sources), 'num_anchors': len(anchors),
        'split_bounds': [format_hour(anchors[0]), format_hour(anchors[-1])],
        'anchor_ranges': serialized_anchor_ranges(anchors),
        'anchor_interval_hours': int(config.timeline.evaluation.step_hours),
        'timeline': OmegaConf.to_container(config.timeline, resolve=True),
        'anchor_frame_number': ANCHOR_FRAME_NUMBER,
        'window_offsets_hours': list(WINDOW_OFFSETS), 'independent_windows': True,
    }
    for key, expected in checks.items():
        if contract.get(key) != expected:
            raise RuntimeError(f'Saved config and generation {key} differ')
    subset = contract.get('anchor_subset')
    if anchors != canonical:
        expected = {'inclusive': True, 'start_time': format_hour(anchors[0]),
                    'end_time': format_hour(anchors[-1]),
                    'source_split_anchor_index_range': [source_indices[0], source_indices[-1]],
                    'parent_split_bounds': [format_hour(canonical[0]), format_hour(canonical[-1])],
                    'parent_anchor_ranges': serialized_anchor_ranges(canonical),
                    'parent_num_anchors': len(canonical)}
        if subset != expected:
            raise RuntimeError('Generation anchor subset differs from configured timeline')
    elif subset is not None:
        raise RuntimeError('Full timeline unexpectedly has an anchor subset')
    num_samples = _num_samples_per_anchor(config)
    ensemble = contract.get('sampling_ensemble')
    if num_samples == 1 and ensemble is not None:
        raise RuntimeError('Single-sample run has ensemble metadata')
    if num_samples > 1 and (not isinstance(ensemble, Mapping)
                           or ensemble.get('num_samples_per_anchor') != num_samples
                           or ensemble.get('reduction') != ENSEMBLE_REDUCTION):
        raise RuntimeError('Sampling ensemble metadata mismatch')
    coordinate = contract['latent_coordinate']
    latent_dir = Path(coordinate['stats']['path']).resolve().parent
    ae_config_path = latent_dir/'ae/config.yaml'
    if Path(coordinate['ae']['path']).resolve() != ae_config_path:
        raise RuntimeError('Saved AE config is outside the latent directory')
    for item in (coordinate['stats'], coordinate['ae']):
        if file_identity(item['path']) != item:
            raise RuntimeError(f'Latent coordinate file changed: {item["path"]}')
    ae_config = OmegaConf.load(ae_config_path)
    h, w, channels = map(int, ae_config.ae.latent_shape)
    shape = (h*w, channels)
    # mmap keeps this completion check from duplicating the full trajectory in RAM.
    trajectory = torch.load(paths['trajectory'], map_location='cpu', weights_only=False, mmap=True)
    for key, expected in {'kind': COLLECTION_KIND, 'contract_hash': manifest['contract_hash'],
                          'num_anchors': len(anchors), 'frame_number': ANCHOR_FRAME_NUMBER,
                          'latent_coordinate': 'diffusion_standardized'}.items():
        if trajectory.get(key) != expected:
            raise RuntimeError(f'Completed trajectory {key} mismatch')
    if tuple(trajectory['frame4_latents'].shape) != (len(anchors), *shape):
        raise RuntimeError('Completed trajectory shape mismatch')
    if [timestamp_key(t) for t in trajectory['timestamps']] != [timestamp_key(t) for t in anchors]:
        raise RuntimeError('Completed trajectory timestamps mismatch')
    return {'run_dir': run_dir, 'paths': paths, 'config': config, 'manifest': manifest,
            'contract': contract, 'contract_hash': manifest['contract_hash'],
            'protocol_kind': PROTOCOL_KIND, 'anchor_frame_number': ANCHOR_FRAME_NUMBER,
            'split': str(config.split), 'anchors': anchors, 'route': str(config.route),
            'source_order': list(config.enabled_sources), 'num_samples_per_anchor': num_samples,
            'latent_dir': latent_dir, 'latent_shape': shape, 'ae_file_config': ae_config,
            'ae_noise_level': float(coordinate['ae_noise_level'])}


def _validate_generation_anchor_payload(run, payload, anchor, anchor_index):
    _validate_anchor_payload(payload, anchor, anchor_index, run['contract_hash'], run['latent_shape'])
    if not torch.isfinite(payload['latent']).all():
        raise RuntimeError(f'Nonfinite generated latent at anchor {anchor_index}')
    if run['num_samples_per_anchor'] > 1:
        ensemble = payload.get('ensemble') or {}
        if (ensemble.get('num_members') != run['num_samples_per_anchor']
                or ensemble.get('reduction') != ENSEMBLE_REDUCTION):
            raise RuntimeError('Anchor ensemble metadata mismatch')


def _build_temp_dataset(convi):
    if not bool(convi.get('metrics_include_temp', True)):
        return None, {'enabled': False}
    paths = {'temp_dir': _resolve_convi_qc_directory(convi.temp_dir, 'temp_dir'),
             'temp_bias_dir': Path(convi.temp_bias_dir).expanduser().resolve(),
             'mean_std_dir': Path(convi.mean_std_dir).expanduser().resolve()}
    dataset = CONVIOBS(surf_dir=None, icoa_dir=None, **paths, skip_temp=False, skip_humidity=False)
    return dataset, {'enabled': True, **{k: str(v) for k, v in paths.items()}}


def _resolve_metrics_output_dir(run_dir: Path, subdir: str | None) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if subdir is None:
        return run_dir
    value = str(subdir).strip()
    if not value:
        raise ValueError("--metrics-output-subdir cannot be empty")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError("--metrics-output-subdir must be a non-empty child of the run directory")
    output_dir = (run_dir / relative).resolve()
    if not output_dir.is_relative_to(run_dir):
        raise ValueError("--metrics-output-subdir escapes the run directory")
    return output_dir

def _normalization_vector(path, label: str) -> torch.Tensor:
    value = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if value.ndim != 1 or value.size < len(CHANNEL_NAMES):
        raise RuntimeError(f"{label} must contain at least 69 channels")
    value = value[: len(CHANNEL_NAMES)]
    if not np.isfinite(value).all():
        raise RuntimeError(f"{label} contains non-finite values")
    return torch.from_numpy(value.copy())

def _load_coordinates(run: Mapping, device) -> dict:
    stats = safe_load(run["latent_dir"] / "stats.pth", map_location="cpu")
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32).reshape(1, 1, -1)
    raw_std = torch.as_tensor(stats["std"], dtype=torch.float32).reshape(1, 1, -1)
    if (
        mean.shape != raw_std.shape
        or mean.shape[-1] != run["latent_shape"][-1]
        or not torch.isfinite(mean).all()
        or not torch.isfinite(raw_std).all()
        or (raw_std <= 0).any()
    ):
        raise RuntimeError("invalid latent coordinate statistics")
    effective_std = torch.sqrt(raw_std.square() + float(run["ae_noise_level"]) ** 2)
    normalization = run["ae_file_config"].normalization
    target_mean = _normalization_vector(normalization.target_mean_path, "ERA target mean")
    target_std = _normalization_vector(normalization.target_std_path, "ERA target std")
    if (target_std <= 0).any():
        raise RuntimeError("ERA target standard deviations must be positive")
    return {
        "mean": mean.to(device),
        "effective_std": effective_std.to(device),
        "target_mean": target_mean.to(device),
        "target_std": target_std.to(device),
    }

def _convi_normalization(dataset: CONVIOBS, device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.from_numpy(dataset.mean.reshape(-1).copy()).to(device=device, dtype=torch.float32)
    std = torch.from_numpy(dataset.std.reshape(-1).copy()).to(device=device, dtype=torch.float32)
    if (
        mean.numel() != len(CHANNEL_NAMES)
        or std.numel() != len(CHANNEL_NAMES)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise RuntimeError("invalid CONVI physical normalization")
    return mean, std

def _latitude_weights(height: int, mode: str, device) -> torch.Tensor:
    mode = str(mode).lower()
    if mode == "cosine":
        return torch.cos(
            torch.linspace(math.pi / 2, -math.pi / 2, int(height), device=device)
        ).clamp_min(0)
    if mode == "none":
        return torch.ones(int(height), dtype=torch.float32, device=device)
    raise ValueError(f"unsupported latitude weighting: {mode}")

def _accumulate_sparse_physical(
    prediction: torch.Tensor,
    observation,
    *,
    channel_min: int,
    channel_max: int,
    prediction_mean: torch.Tensor,
    prediction_std: torch.Tensor,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    squared_error_sum: torch.Tensor,
    counts: torch.Tensor,
    latitude_weights: torch.Tensor | None = None,
    weight_sum: torch.Tensor | None = None,
) -> int:
    if observation is None:
        return 0
    pixel = observation["pixel_index"].to(device=prediction.device, dtype=torch.long)
    channel = observation["channel_index"].to(device=prediction.device, dtype=torch.long)
    values = observation["values"].to(device=prediction.device, dtype=torch.float32)
    selected = (channel >= int(channel_min)) & (channel < int(channel_max))
    if not bool(selected.any()):
        return 0
    pixel = pixel[selected]
    channel = channel[selected]
    values = values[selected]
    width = int(prediction.shape[-1])
    row = torch.div(pixel, width, rounding_mode="floor")
    column = pixel.remainder(width)
    predicted_normalized = prediction[
        channel,
        row,
        column,
    ]
    predicted_physical = predicted_normalized * prediction_std[channel] + prediction_mean[channel]
    observed_physical = values * observation_std[channel] + observation_mean[channel]
    residual = (predicted_physical - observed_physical).to(torch.float64)
    if not torch.isfinite(residual).all():
        raise RuntimeError("Nonfinite residual at a valid TEMP record")
    if (latitude_weights is None) != (weight_sum is None):
        raise ValueError("latitude_weights and weight_sum must be supplied together")
    if latitude_weights is None:
        squared_error_sum.scatter_add_(0, channel, residual.square())
    else:
        latitude_weights = torch.as_tensor(
            latitude_weights, dtype=torch.float64, device=prediction.device
        ).reshape(-1)
        if latitude_weights.numel() != prediction.shape[-2]:
            raise ValueError("latitude weight count differs from prediction height")
        record_weights = latitude_weights[row]
        squared_error_sum.scatter_add_(0, channel, residual.square() * record_weights)
        weight_sum.scatter_add_(0, channel, record_weights)
    counts.scatter_add_(
        0,
        channel,
        torch.ones_like(channel, dtype=torch.int64),
    )
    return int(channel.numel())

def finalize_sparse_rmse(
    squared_error_sum: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    squared_error_sum = torch.as_tensor(squared_error_sum, dtype=torch.float64)
    counts = torch.as_tensor(counts, dtype=torch.int64)
    if squared_error_sum.shape != counts.shape:
        raise ValueError("sparse RMSE sums and counts must share a shape")
    if (squared_error_sum < 0).any() or (counts < 0).any():
        raise ValueError("sparse RMSE totals cannot be negative")
    result = torch.full_like(squared_error_sum, float("nan"))
    valid = counts > 0
    result[valid] = torch.sqrt(squared_error_sum[valid] / counts[valid].double())
    return result

def _json_values(values: torch.Tensor) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in torch.as_tensor(values).tolist()
    ]

def _attach_evaluation_provenance(payload: dict, run: Mapping) -> None:
    provenance = run.get("evaluation_provenance")
    if provenance is not None:
        payload["evaluation_provenance"] = dict(provenance)

def _write_era_metrics(
    run: Mapping,
    totals: Mapping,
    latitude_weighting: str,
) -> dict:
    latitude_weighting = str(latitude_weighting).lower()
    if latitude_weighting not in {"cosine", "none"}:
        raise ValueError(f"unsupported ERA latitude weighting: {latitude_weighting}")
    latitude_weighted = latitude_weighting == "cosine"
    scope = (
        "era69_full_map_cosine_latitude_weighted"
        if latitude_weighted
        else "era69_full_map_unweighted"
    )
    normalized, physical = finalize_weighted_rmse(
        totals["era_squared_error_sum"],
        totals["era_weight_sum"],
        totals["physical_std"],
    )
    payload = {
        "kind": ERA_METRICS_KIND if latitude_weighted else ERA_UNWEIGHTED_METRICS_KIND,
        "generation_contract_hash": run["contract_hash"],
        "split": run["split"],
        "route": run["route"],
        "source_order": run["source_order"],
        "num_frames": int(totals["num_frames"]),
        "frame_number": int(run.get("anchor_frame_number", ANCHOR_FRAME_NUMBER)),
        "scope": scope,
        "target_names": list(CHANNEL_NAMES),
        "aggregation": (
            f"{('cosine-latitude-weighted' if latitude_weighted else 'unweighted')} "
            "squared errors summed over every retained offset-zero anchor in the complete "
            "split, followed by one square root"
        ),
        "latitude_weighted": latitude_weighted,
        "latitude_weighting": latitude_weighting,
        "squared_error_sum": totals["era_squared_error_sum"],
        "weight_sum": totals["era_weight_sum"],
        "rmse_normalized": normalized,
        "rmse_physical": physical,
    }
    _attach_evaluation_provenance(payload, run)
    destination = run["metrics_output_dir"] / "era69_rmse.pth"
    _atomic_torch_save(payload, destination)

    rows = ["channel,rmse_normalized,rmse_physical"]
    lines = [
        f"ERA69 split={run['split']} route={run['route']} "
        f"sources={','.join(run['source_order']) or 'none'}",
        f"frames={int(totals['num_frames'])} frame={run.get('anchor_frame_number', ANCHOR_FRAME_NUMBER)} "
        f"latitude_weighted={str(latitude_weighted).lower()}",
        "channel,rmse_normalized,rmse_physical",
    ]
    for name, normalized_value, physical_value in zip(
        CHANNEL_NAMES, normalized.tolist(), physical.tolist(), strict=True
    ):
        row = f"{name},{normalized_value:.10g},{physical_value:.10g}"
        rows.append(row)
        lines.append(row)
    _atomic_write_text("\n".join(rows) + "\n", destination.with_suffix(".csv"))
    _atomic_write_text("\n".join(lines) + "\n", destination.with_suffix(".log"))
    return {
        "path": str(destination),
        "target_names": list(CHANNEL_NAMES),
        "rmse_normalized": normalized.tolist(),
        "rmse_physical": physical.tolist(),
        "scope": scope,
        "latitude_weighted": latitude_weighted,
        "latitude_weighting": latitude_weighting,
    }

def compute_local_totals(run, local_indices, temp_all, *, data_path=None,
                         era_latitude_weighting='cosine', precision='float16',
                         log_every=8, device='cpu', rank=0):
    coordinates = _load_coordinates(run, device)
    autoencoder = load_auto_encoder(run['latent_dir']/'ae', 'model', device='cpu', eval_mode=True)
    autoencoder.eval().requires_grad_(False).to(device)
    truth, truth_index = _truth_dataset(run['ae_file_config'], run['anchors'], data_path=data_path)
    height, width = map(int, run['ae_file_config'].ae.shape)
    if temp_all is not None and tuple(temp_all.grid_shape) != (height, width):
        raise RuntimeError('TEMP and ERA grids differ')
    weights = _latitude_weights(height, era_latitude_weighting, device).view(1, height, 1)
    totals = {key: torch.zeros(69, dtype=torch.float64, device=device) for key in
              ('era_squared_error_sum', 'era_weight_sum', 'temp_squared_error_sum')}
    totals['temp_counts'] = torch.zeros(69, dtype=torch.int64, device=device)
    temp_mean = temp_std = None
    if temp_all is not None:
        temp_mean, temp_std = _convi_normalization(temp_all, device)
    temp_frames = 0
    for position, index in enumerate(local_indices):
        anchor = run['anchors'][index]
        payload = safe_load(_anchor_path(run['run_dir']/'anchors', index), map_location='cpu')
        _validate_generation_anchor_payload(run, payload, anchor, index)
        latent = payload['latent'].to(device=device, dtype=torch.float32)[None]
        native = latent * coordinates['effective_std'] + coordinates['mean']
        date = timestamp_tensor(anchor)[None].to(device)
        with torch.no_grad(), _precision_context(precision):
            decoded = autoencoder.decode(native, date)
        prediction = decoded.reshape(1, height, width, 69).permute(0, 3, 1, 2)[0].float()
        truth_state, _ = truth[truth_index[timestamp_key(anchor)]]
        target = truth_state[0].to(device=device, dtype=torch.float32)
        difference = prediction - target
        if not torch.isfinite(difference).all():
            raise RuntimeError(f'Nonfinite prediction or ERA truth at {anchor}')
        totals['era_squared_error_sum'] += (difference.double().square()*weights).sum((-2,-1))
        totals['era_weight_sum'] += weights.expand_as(difference).double().sum((-2,-1))
        count = 0
        if temp_all is not None:
            count = _accumulate_sparse_physical(
                prediction, temp_all.prepare_sparse(np.datetime64(anchor, 'h')),
                channel_min=0, channel_max=TEMP_CHANNEL_STOP,
                prediction_mean=coordinates['target_mean'], prediction_std=coordinates['target_std'],
                observation_mean=temp_mean, observation_std=temp_std,
                squared_error_sum=totals['temp_squared_error_sum'], counts=totals['temp_counts'])
        temp_frames += int(count > 0)
        if position % max(1, int(log_every)) == 0 or position+1 == len(local_indices):
            print(f'[rank {rank}] metrics {index+1}/{len(run["anchors"])} TEMP records={count}', flush=True)
    del autoencoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**{k: v.cpu() for k, v in totals.items()},
            'num_frames': torch.tensor(len(local_indices)),
            'temp_matched_frames': torch.tensor(temp_frames),
            'physical_std': coordinates['target_std'].double().cpu()}


def _reduce_totals(totals, world_size):
    if world_size > 1:
        for key in totals:
            if key != 'physical_std':
                dist.all_reduce(totals[key], op=dist.ReduceOp.SUM)


def write_metrics(run, totals, observation_data, latitude_weighting='cosine'):
    if int(totals['num_frames']) != len(run['anchors']):
        raise RuntimeError('Refusing to report metrics for an incomplete evaluation')
    destination = run['metrics_output_dir']
    destination.mkdir(parents=True, exist_ok=True)
    era = _write_era_metrics(run, totals, latitude_weighting)
    temp = None
    if observation_data['enabled']:
        errors = totals['temp_squared_error_sum'][:TEMP_CHANNEL_STOP]
        counts = totals['temp_counts'][:TEMP_CHANNEL_STOP]
        rmse = finalize_sparse_rmse(errors, counts)
        payload = {'kind': TEMP_METRICS_KIND, 'generation_contract_hash': run['contract_hash'],
                   'target_names': list(CHANNEL_NAMES[:TEMP_CHANNEL_STOP]),
                   'scope': 'all_valid_qc_temp_records', 'observation_data': observation_data,
                   'num_frames': int(totals['num_frames']),
                   'matched_frames': int(totals['temp_matched_frames']),
                   'squared_error_sum': errors, 'counts': counts, 'rmse_physical': rmse}
        _atomic_torch_save(payload, destination/'convi_temp_all_rmse.pth')
        rows = ['channel,rmse_physical,count']
        rows += [f'{name},{value:.10g},{int(count)}' for name, value, count in
                 zip(CHANNEL_NAMES[:TEMP_CHANNEL_STOP], rmse, counts, strict=True)]
        _atomic_write_text('\n'.join(rows)+'\n', destination/'convi_temp_all_rmse.csv')
        _atomic_write_text('\n'.join(rows)+'\n', destination/'convi_temp_all_rmse.log')
        temp = {'rmse_physical': _json_values(rmse), 'counts': counts.tolist(),
                'matched_frames': int(totals['temp_matched_frames'])}
    result = {'kind': EVALUATION_KIND, 'method': str(run['config'].id),
              'generation_contract_hash': run['contract_hash'],
              'timeline': OmegaConf.to_container(run['config'].timeline, resolve=True),
              'num_frames': int(totals['num_frames']), 'era69': era, 'temp': temp,
              'observation_data': observation_data,
              'evaluation_provenance': run.get('evaluation_provenance')}
    _atomic_write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', destination/'evaluation_metrics.json')
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Trajectory metric decoding requires CUDA')
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.cuda.set_device(device)
    run = _load_run(args.run_dir)
    run['metrics_output_dir'] = _resolve_metrics_output_dir(run['run_dir'], args.metrics_output_subdir)
    temp, observation_data = _build_temp_dataset(run['config'].convi)
    precision = args.precision or str(run['config'].metrics.precision)
    run['evaluation_provenance'] = {
        'precision': precision, 'era_latitude_weighting': args.era_latitude_weighting,
        'data_path': str(args.data_path or run['ae_file_config'].data_path),
        'trajectory': file_identity(run['paths']['trajectory']),
        'ae_checkpoint': file_identity(run['latent_dir']/'ae/model.pth')}
    indices = partition_anchor_indices(len(run['anchors']), rank, world_size)
    totals = compute_local_totals(run, indices, temp, data_path=args.data_path,
                                 era_latitude_weighting=args.era_latitude_weighting,
                                 precision=precision, log_every=args.log_every, device=device, rank=rank)
    _initialize_cpu_coordination(world_size, args.coordination_timeout_seconds)
    _reduce_totals(totals, world_size)
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        write_metrics(run, totals, observation_data, args.era_latitude_weighting)


if __name__ == '__main__':
    main()
