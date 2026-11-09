r"""Multi-source ERA5-latent reanalysis with real observations.

Satellite instruments can either be encoded online into ERA latent space,
where their operators are masked identities:

    y_i = standardize(obs2era_encode_i(real_obs_i))
    A_i(z) = z[:, latent_mask_i, :]

or assimilated as physical standardized TB through a trained Direct E2O
decoder:

    y_i = standardized_real_TB_i
    A_i(z) = direct_e2o_i(destandardize_latent(z))[valid_swath_values]

An optional CONVIOBS task merges TEMP, SURF, and ICOADS (SURF wins surface
collisions). TEMP humidity can be excluded. In the default ``physical`` mode,
every finite observation is compared directly with the normalized ERA field
decoded from the clean latent:

    y_c = normalize_era(conventional_obs)
    A_c(z) = era_autoencoder.decode(destandardize_latent(z))[pixel, channel]

The fixed observation layout is [timestep][source][values]. ``joint`` stacks all
sources into one coupled MMPS solve. ``sum``, ``pcgrad``, ``ppgf``, and
``channel_ppgf`` solve each source independently. PPGF projects accumulated
lower-priority guidance before adding the next source. Channel PPGF instead samples
one source order per latent channel and performs scalar sign fusion: opposite signs
replace with the later value, while equal signs add. Source-specific operators keep
SATE JVPs out of the ERA decoder for satellite-only tasks. CONVIOBS is one additional
guidance task, and its direct latent support is
the set of 32x32 AE blocks containing at least one physical observation.

Physical CONVIOBS var_y is either a fixed scalar shared with the satellite sources
or per-channel population Var(obs_norm). The latent observation encoder uses
centered per-channel residual variance against aligned ERA5 truth latents. Both
calibrations use the first configured frames in the final 5% of timestamps aligned
with era5.h5.

Usage: torchrun --nproc-per-node <NUM_GPUS> reanalysis_obs2era_dual.py model_path=...
"""

import copy
import gc
import h5py
import math
import numpy as np
import os
import random
import shutil
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
import warnings

from datetime import timedelta
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pathlib import Path

from appa.config.hydra import compose
from appa.data.datasets import CONVIOBS, SatelliteAEDataset
from appa.date import create_trajectory_timestamps, get_date_features
from appa.diffusion import (
    Denoiser,
    MMPSDenoiser,
    MultiInstrumentMMPSDenoiser,
    TrajectoryDenoiser,
    create_schedule,
)
from appa.math import str_to_ids
from appa.sampling import PPGFSampler, select_sampler
from appa.save import load_auto_encoder, load_denoiser, safe_load, safe_save
from experiments.observations.convi_obs_encoder import (
    CONVI_OBS_ENCODER_CHANNELS,
    get_var_y_convi_obs_encoder,
    load_convi_obs_encoder_latent_at_time,
)
from experiments.observations.direct_e2o_assimilation import (
    get_direct_e2o_var_y,
    load_direct_e2o_at_time,
    load_direct_e2o_source,
    select_direct_e2o_prediction,
)
from risda.models.encoders import load_obs2era_encoder

TRAJECTORY_DT_HOURS = 6
OBS_DT_HOURS = 6

GRID_HW = (720, 1440)


def pcgrad_order_uncertainty(var_y, scale=1.0):
    """Return a source's scalar Gumbel-order uncertainty without changing var_y."""
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("pcgrad_uncertainty_scale must be finite and positive")
    return (
        var_y.detach()
        .float()
        .clamp_min(1e-12)
        .log()
        .mean()
        .exp()
        .mul(scale)
        .cpu()
    )


def channel_ppgf_uncertainty(
    var_y, source_kind, latent_channels, scale=1.0
):
    """Return one ordering uncertainty for each latent channel."""
    values = var_y.detach().float().reshape(-1)
    if values.numel() == latent_channels:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "pcgrad_uncertainty_scale must be finite and positive"
            )
        return values.clamp_min(1e-12).mul(scale).cpu()

    if source_kind in {"convi", "direct_e2o"} and values.numel() > 0:
        geometric_mean = pcgrad_order_uncertainty(values, scale)
        return geometric_mean.expand(latent_channels).clone()

    raise ValueError(
        "channel_ppgf requires one uncertainty per latent "
        f"channel ({latent_channels}); source kind {source_kind!r} "
        f"has {values.numel()}"
    )


GUIDED_LATENT_RMSE_MATRIX_KIND = (
    "guided_latent_native_rmse_matrix_6x128_v1"
)


def load_ppgf_ordering_cache(
    cache_path,
    expected_sources,
    latent_channels,
    expected_mask_frac,
    expected_polar_guidance,
    expected_convi_guidance,
):
    """Load and audit a route-matched source-by-channel RMSE matrix.

    The matrix is used only for PPGF ordering. MMPS covariance continues to
    come from each source's residual ``var_y`` cache.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"PPGF ordering cache is missing: {cache_path}"
        )
    cached = safe_load(cache_path, map_location="cpu")
    actual_kind = cached.get("kind")
    if actual_kind != GUIDED_LATENT_RMSE_MATRIX_KIND:
        raise RuntimeError(
            f"PPGF ordering cache kind mismatch in {cache_path}: "
            f"{actual_kind!r} != {GUIDED_LATENT_RMSE_MATRIX_KIND!r}"
        )

    sources = [
        str(value).lower() for value in cached.get("source_order", ())
    ]
    expected = [str(value).lower() for value in expected_sources]
    if sources != expected:
        raise RuntimeError(
            f"PPGF ordering source order mismatch in {cache_path}: "
            f"{sources} != {expected}"
        )

    cached_mask_frac = float(cached.get("mask_frac", float("nan")))
    if not math.isclose(
        cached_mask_frac,
        float(expected_mask_frac),
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            f"PPGF ordering mask_frac mismatch in {cache_path}: "
            f"{cached_mask_frac} != {float(expected_mask_frac)}"
        )

    route_fields = {
        "polar_guidance": str(expected_polar_guidance).lower(),
        "convi_guidance": str(expected_convi_guidance).lower(),
    }
    for field, expected_value in route_fields.items():
        actual_value = str(cached.get(field, "")).lower()
        if actual_value != expected_value:
            raise RuntimeError(
                f"PPGF ordering {field} mismatch in {cache_path}: "
                f"{actual_value!r} != {expected_value!r}"
            )

    rmse = torch.as_tensor(cached.get("rmse_native"), dtype=torch.float32)
    expected_shape = (len(expected), int(latent_channels))
    if rmse.shape != expected_shape:
        raise RuntimeError(
            f"PPGF ordering RMSE shape mismatch in {cache_path}: "
            f"{tuple(rmse.shape)} != {expected_shape}"
        )
    if not torch.isfinite(rmse).all() or not torch.all(rmse > 0):
        raise RuntimeError(
            f"invalid PPGF ordering RMSE values in {cache_path}"
        )
    return rmse


def select_source_configs(instruments, convi_enabled, source_filter=None):
    """Select one physical guidance source for latent calibration runs."""
    selected = list(instruments or [])
    if source_filter is None:
        return selected, bool(convi_enabled)

    source_filter = str(source_filter).lower()
    if source_filter == "convi":
        if not convi_enabled:
            raise ValueError("source_filter=convi requires convi.enabled=true")
        return [], True

    selected = [
        instrument
        for instrument in selected
        if str(instrument.get("name", "")).lower() == source_filter
    ]
    if len(selected) != 1:
        raise ValueError(
            f"source_filter={source_filter!r} matched {len(selected)} "
            "instrument entries; each entry needs a unique name"
        )
    return selected, False


def load_residual_var_y_cache(
    cache_path, expected_channels, expected_kind, expected_source=None
):
    """Load a centered residual-variance cache with strict identity checks."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"residual variance cache is missing: {cache_path}"
        )
    cached = safe_load(cache_path, map_location="cpu")
    if cached.get('kind') != expected_kind:
        raise RuntimeError(
            f"residual variance kind mismatch in {cache_path}: "
            f"{cached.get('kind')} != {expected_kind}"
        )
    if (
        expected_source is not None
        and str(cached.get('source')) != str(expected_source)
    ):
        raise RuntimeError(
            f"residual variance source mismatch in {cache_path}: "
            f"{cached.get('source')} != {expected_source}"
        )
    variance = torch.as_tensor(cached["variance"], dtype=torch.float32)
    if variance.shape != (int(expected_channels),):
        raise RuntimeError(
            f"residual variance shape mismatch in {cache_path}: "
            f"{tuple(variance.shape)} != {(int(expected_channels),)}"
        )
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise RuntimeError(f"invalid residual variance values in {cache_path}")
    return variance, cached


def _abs_hour_to_time_str(start_date, start_hour, abs_hour):
    """(start_date 'YYYY-MM-DD', start_hour int, abs_hour int) -> 'YYYY-MM-DDTHH'."""
    base = np.datetime64(f"{start_date}T{int(start_hour):02d}", "h")
    t = base + np.timedelta64(int(abs_hour), "h")
    return str(t)


# =============================================
def _obs_mask_to_frac(mask_pix, resize):
    """Observed fraction within each f x f block, as a (lat_H*lat_W,) float
    tensor (block-mean of the 0/1 pixel mask). Mirrors reanalysis_e2s.py:
    circular-pad longitude, zero-pad latitude, then avg_pool.

    mask_pix: (1, 1, H, W) float {0,1} obs pixel mask.
    """
    H, W = mask_pix.shape[-2:]
    f = int(resize)
    pad_h = (-H) % f
    pad_w = (-W) % f + 2 * f
    if pad_w > 0:
        mask_pix = F.pad(mask_pix, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode="circular")
    if pad_h > 0:
        mask_pix = F.pad(mask_pix, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode="constant", value=0)
    frac_lat = F.avg_pool2d(mask_pix, kernel_size=(f, f))      # (1, 1, lat_H, lat_W)
    return rearrange(frac_lat, "n c h w -> n (h w) c")[0, :, 0]  # (S,)


# =============================================
# Latent-grid mask thresholding (shared recipe with reanalysis_e2s.py)
# =============================================
def _frac_to_mask(frac_lat, mask_frac=0.0):
    """Threshold an observed-fraction tensor into a bool latent-grid mask.

        mask_frac <= 0   -> ANY observed pixel makes it valid (loosest)
        mask_frac == 0.5 -> a majority of the block must be observed
        mask_frac == 1.0 -> the WHOLE block must be observed (logical AND)
    """
    if mask_frac <= 0.0:
        return frac_lat > 0.0
    # tiny epsilon so mask_frac=1.0 still admits fully-observed blocks despite
    # avg_pool float rounding.
    return frac_lat >= (float(mask_frac) - 1e-6)


def _flat_pixel_to_latent_block(pixel_index, grid_shape, resize):
    """Map original-grid flat pixels to the matching central latent blocks.

    The mapping mirrors _obs_mask_to_frac: latitude is zero padded and
    longitude receives one circular block on each side. Original pixels map
    through the central (non-duplicated) longitude span.
    """
    pixel_index = torch.as_tensor(pixel_index, dtype=torch.long)
    height, width = (int(value) for value in grid_shape)
    resize = int(resize)
    if resize <= 0:
        raise ValueError("resize must be positive")
    if pixel_index.ndim != 1:
        raise ValueError("pixel_index must be one-dimensional")
    if pixel_index.numel() and (
        int(pixel_index.min()) < 0
        or int(pixel_index.max()) >= height * width
    ):
        raise ValueError("pixel_index is outside the observation grid")

    pad_h = (-height) % resize
    pad_w = (-width) % resize + 2 * resize
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    latent_width = (width + pad_w) // resize
    latitude = torch.div(pixel_index, width, rounding_mode="floor")
    longitude = pixel_index % width
    return (
        torch.div(latitude + pad_top, resize, rounding_mode="floor")
        * latent_width
        + torch.div(longitude + pad_left, resize, rounding_mode="floor")
    )


def filter_physical_observation_by_block_mask(
    observation, pixel_mask, resize, mask_frac=0.0,
):
    """Apply the latent block threshold to compact physical observations.

    pixel_mask is the channel-union spatial mask used to decide whether a
    block is dense enough. CONVI stores one pixel_index per scalar value,
    while Direct E2O stores one row per pixel plus a two-dimensional
    valid_mask and one cos_satzen condition per row. Both layouts are filtered
    without changing their original value/channel ordering, so y_obs, cov_y,
    zenith, and the forward operator remain element-for-element aligned.
    """
    observation = dict(observation)
    pixel_mask = torch.as_tensor(pixel_mask, dtype=torch.bool).cpu()
    if pixel_mask.ndim != 2:
        raise ValueError("pixel_mask must have shape (height, width)")
    if "pixel_index" not in observation or "values" not in observation:
        raise ValueError("physical observation needs pixel_index and values")

    fraction = _obs_mask_to_frac(
        pixel_mask.float()[None, None], int(resize)
    ).cpu()
    latent_mask = _frac_to_mask(fraction, float(mask_frac)).cpu()
    values_before = int(observation["values"].numel())
    pixels_before = int(pixel_mask.sum())

    if float(mask_frac) > 0.0:
        pixel_index = observation["pixel_index"]
        latent_index = _flat_pixel_to_latent_block(
            pixel_index, pixel_mask.shape, resize
        )
        keep_by_pixel = latent_mask.to(pixel_index.device)[latent_index]

        if "valid_mask" in observation:
            valid_mask = observation["valid_mask"]
            if (
                valid_mask.ndim != 2
                or valid_mask.shape[0] != pixel_index.numel()
            ):
                raise ValueError(
                    "Direct E2O valid_mask must have one row per pixel_index"
                )
            cos_satzen = observation.get("cos_satzen")
            if (
                cos_satzen is None
                or cos_satzen.ndim != 2
                or cos_satzen.shape
                != (pixel_index.numel(), 1)
            ):
                raise ValueError(
                    "Direct E2O cos_satzen must have shape "
                    "(len(pixel_index),1)"
                )
            scalar_keep = keep_by_pixel[:, None].expand_as(valid_mask)[
                valid_mask
            ]
            if scalar_keep.numel() != observation["values"].numel():
                raise RuntimeError(
                    "Direct E2O value order disagrees with valid_mask"
                )
            observation["pixel_index"] = pixel_index[keep_by_pixel]
            observation["valid_mask"] = valid_mask[keep_by_pixel]
            observation["cos_satzen"] = cos_satzen[keep_by_pixel]
            for key in ("channel_local", "values"):
                observation[key] = observation[key][scalar_keep]
        else:
            for key in (
                "pixel_index",
                "channel_index",
                "channel_local",
                "values",
            ):
                if key in observation:
                    if observation[key].shape[0] != keep_by_pixel.numel():
                        raise ValueError(
                            f"{key} must align with physical scalar observations"
                        )
                    observation[key] = observation[key][keep_by_pixel]

    values_after = int(observation["values"].numel())
    if values_after == 0:
        return None
    retained_pixels = torch.unique(
        observation["pixel_index"].detach().cpu()
    ).numel()
    observation["m_lat"] = latent_mask.to(
        observation["values"].device, non_blocking=True
    )
    observation["block_filter_mask_frac"] = float(mask_frac)
    observation["block_filter_values_before"] = values_before
    observation["block_filter_values_after"] = values_after
    observation["block_filter_pixels_before"] = pixels_before
    observation["block_filter_pixels_after"] = int(retained_pixels)
    return observation


# =============================================
# var_y calibration (per ERA5-latent-channel error variance)
# =============================================
def calibrate_var_y_obs2era(
    encoder, sate_dataset, era_h5_path, era_resize, era_mean, era_std, device,
    max_frames=64, valid_frac=0.05,
):
    """Per ERA5-latent-channel observation variance in STANDARDIZED latent space,
    computed ONLINE (no pre-dump).

    var_y[c] = E[(encode_obs2era(obs) - z_era_truth)^2] over the tail valid_frac
    of the ATMS frames that timestamp-align with era5.h5, masked to valid latent
    pixels, in standardized latent units. This is the obs2era encoder's own
    modelling error per ERA5-latent channel — the analogue of e2s's
    per-ATMS-latent-channel var_y.

    encoder    : the loaded ObsToEraEncoder (online).
    sate_dataset: SatelliteAEDataset yielding real ATMS frames.
    era_h5_path: era5.h5 (ground-truth ERA5 latents/dates).
    """
    era_h5 = h5py.File(era_h5_path, "r")
    era_dates = np.asarray(era_h5["dates"])     # (N_e, 4)
    era_key = {tuple(int(x) for x in d): i for i, d in enumerate(era_dates)}

    # Timestamp-align ATMS frames with era5.h5; tail valid_frac is validation.
    pairs = []                                  # (era_idx, sate_idx)
    for j, t in enumerate(sate_dataset.times):
        k = (int(t.year), int(t.month), int(t.day), int(t.hour))
        i = era_key.get(k)
        if i is not None:
            pairs.append((i, j))
    pairs = np.asarray(sorted(pairs), dtype=np.int64)
    n_valid = max(1, int(round(len(pairs) * valid_frac)))
    pairs = pairs[-n_valid:]

    n_obs = sate_dataset.obs_channel
    C = int(era_mean.numel())
    local_se = torch.zeros(C, device=device)
    local_n = torch.zeros(C, device=device)

    with torch.no_grad():
        for b in range(min(len(pairs), max_frames)):
            i_era, j_sate = int(pairs[b, 0]), int(pairs[b, 1])
            merged, _ = sate_dataset[j_sate]                   # (1, 23/24, H, W)
            obs = merged[0].to(device)                         # (23/24, H, W)
            z_o = encoder(obs.unsqueeze(0))                    # (1, S, C) native
            z_e = torch.as_tensor(era_h5["latents"][i_era]).to(device).unsqueeze(0)  # (1, S, C)

            mask_pix = obs[n_obs:n_obs + 1].unsqueeze(0)       # (1, 1, H, W)
            frac = _obs_mask_to_frac(mask_pix, era_resize)     # (S,)
            m = (frac > 0.0).float().view(1, -1, 1)            # any-observed for calibration

            z_e_std = (z_e - era_mean) / era_std.clamp_min(1e-8)
            z_o_std = (z_o - era_mean) / era_std.clamp_min(1e-8)
            se = (z_o_std - z_e_std).square()                  # (1, S, C)
            local_se += (se * m).sum(dim=(0, 1))
            local_n += m.sum(dim=(0, 1)).clamp_min(0)

    era_h5.close()
    return local_se / local_n.clamp_min(1.0)                   # (C,)


def get_var_y_per_channel_obs2era(
    config, encoder, sate_dataset, era_dir, era_resize, era_mean, era_std, device,
):
    """Return a (C,) tensor of per-channel var_y in standardized ERA5 latent space.

    var_y_mode:
        'rmse'   -> online-calibrated per-channel obs2era residual variance
                    (recommended). Cached at obs2era_model_dir/obs2era_online_var_y.pth,
                    keyed on the obs2era checkpoint mtime.
        'scalar' -> obs_covariance broadcast to all channels.
    A floor of obs_covariance is applied in 'rmse' mode to avoid zero variance.

    var_y_scale (rmse mode only): a single global coefficient c applied to the
    calibrated VARIANCE, var_y = c * var_y_rmse, BEFORE the obs_covariance floor.
    Keeps the per-channel relative shape from the calibration while turning one knob
    for overall observation trust: c<1 trusts obs more (stronger guidance, toward the
    "accurate observation" limit), c>1 trusts obs less. c=1.0 is unchanged. Scaling
    before the floor keeps obs_covariance as a hard lower bound so small c can't
    collapse a channel's variance to ~0 (which would make the GMRES solve of
    (Sigma_y + H Sigma H^T) ill-conditioned). In the dual/N-instrument path each
    instrument passes its OWN var_y_scale (from its `instruments[]` entry), so obs
    trust is tunable PER INSTRUMENT. Ignored in scalar mode.
    """
    C = int(era_mean.numel())
    mode = str(config.get("var_y_mode", "rmse")).lower()
    floor = float(config.get("obs_covariance", 1e-4))
    scale = float(config.get("var_y_scale", 1.0))

    if mode == "scalar":
        return torch.full((C,), floor, device=device)

    model_dir = Path(config.obs2era_model_dir)
    ckpt_name = "model_best.pth" if config.get("obs2era_checkpoint", "best") == "best" else "model_last.pth"
    ckpt_mtime = int(os.path.getmtime(model_dir / ckpt_name))
    if bool(config.get("use_residual_var_y", False)):
        cache_path = model_dir / "obs2era_online_residual_var_y.pth"
        var_y, _ = load_residual_var_y_cache(
            cache_path,
            expected_channels=C,
            expected_kind="sate",
            expected_source=sate_dataset.sate_name,
        )
        return (var_y.to(device) * scale).clamp_min(floor)

    cache_path = model_dir / "obs2era_online_var_y.pth"

    var_y = None
    if cache_path.exists():
        cached = safe_load(cache_path, map_location="cpu")
        if int(cached.get("ckpt_mtime", -1)) == ckpt_mtime:
            var_y = cached["var_y"].to(device)

    if var_y is None:
        var_y = calibrate_var_y_obs2era(
            encoder, sate_dataset, Path(era_dir) / "era5.h5", era_resize,
            era_mean, era_std, device,
            max_frames=int(config.get("var_y_max_batches", 64)),
        )
        if dist.get_rank() == 0:
            safe_save({"var_y": var_y.cpu(), "ckpt_mtime": ckpt_mtime}, cache_path)

    # Cache stores the RAW calibrated var_y (scale-independent), so var_y_scale
    # can be swept without re-calibrating. Apply the global coefficient here, then
    # floor. c=1.0 -> identical to the pre-existing behaviour.
    return (var_y * scale).clamp_min(floor)


# =============================================
# =============================================
# Conventional-observation variance calibration
# =============================================
def calibrate_var_y_convi(
    dataset, era_h5_path, max_frames=64, valid_frac=0.05
):
    """Estimate enabled CONVIOBS channel variances on the SATE tail split."""
    with h5py.File(era_h5_path, "r") as era_h5:
        era_keys = {
            tuple(int(value) for value in date)
            for date in np.asarray(era_h5["dates"])
        }

    times = [
        time
        for time in sorted(dataset.times)
        if (int(time.year), int(time.month), int(time.day), int(time.hour))
        in era_keys
    ]
    if not times:
        raise RuntimeError("CONVIOBS has no timestamps aligned with era5.h5")

    n_valid = max(1, int(round(len(times) * float(valid_frac))))
    times = times[-n_valid:][: int(max_frames)]
    num_channels = dataset.num_observation_channels
    sums = np.zeros(num_channels, dtype=np.float64)
    square_sums = np.zeros(num_channels, dtype=np.float64)
    counts = np.zeros(num_channels, dtype=np.int64)

    for time in times:
        observation = dataset.prepare_sparse(time)
        if observation is None:
            continue
        channels = observation["channel_local"].numpy()
        values = observation["values"].numpy().astype(np.float64, copy=False)
        counts += np.bincount(channels, minlength=num_channels)
        sums += np.bincount(channels, weights=values, minlength=num_channels)
        square_sums += np.bincount(
            channels, weights=values * values, minlength=num_channels
        )

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise RuntimeError(
            f"CONVIOBS calibration has no valid values for channels {missing}"
        )

    means = sums / counts
    variance = np.maximum(square_sums / counts - means * means, 0.0)
    return torch.from_numpy(variance.astype(np.float32)), counts, times


def get_var_y_convi(config, dataset, era_dir, model_path, device):
    """Load, estimate, or fix enabled ERA-normalized CONVIOBS variances."""
    valid_frac = float(config.get("var_y_valid_frac", 0.05))
    max_frames = int(config.get("var_y_max_batches", 64))
    mode = str(config.get("var_y_mode", "rmse")).lower()
    floor = float(config.get("obs_covariance", 1e-4))
    if mode == "scalar":
        var_y = torch.full(
            (dataset.num_observation_channels,),
            floor,
            dtype=torch.float32,
            device=device,
        )
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                "[obs2era-dual] CONVIOBS fixed scalar "
                f"var_y={floor:g} for {dataset.num_observation_channels} channels",
                flush=True,
            )
        return var_y
    if mode != "rmse":
        raise ValueError(f"Unknown var_y_mode: {mode!r}")

    scale = float(config.get("var_y_scale", 1.0))
    use_residual_var_y = bool(config.get("use_residual_var_y", False))
    cache_value = config.get("var_y_cache", None)
    cache_path = (
        Path(cache_value)
        if cache_value
        else Path(model_path) / "convi_obs_norm_var_y.pth"
    )
    if use_residual_var_y:
        residual_cache_value = config.get("residual_var_y_cache", None)
        cache_path = (
            Path(residual_cache_value)
            if residual_cache_value
            else Path(model_path) / "convi_obs_norm_residual_var_y.pth"
        )
    metadata = {
        "surf_dir": str(config.surf_dir),
        "icoa_dir": str(config.icoa_dir),
        "mean_std_dir": str(config.mean_std_dir),
        "temp_dir": (
            str(config.get("temp_dir")) if config.get("temp_dir", None) else None
        ),
        "temp_bias_dir": (
            str(config.get("temp_bias_dir"))
            if config.get("temp_bias_dir", None)
            else None
        ),
        "skip_temp": bool(config.get("skip_temp", True)),
        "skip_humidity": bool(config.get("skip_humidity", True)),
        "observation_channel_indices": dataset.observation_channel_indices.tolist(),
        "valid_frac": valid_frac,
        "max_frames": max_frames,
    }

    rank = dist.get_rank() if dist.is_initialized() else 0
    raw_var = None
    counts = None
    selected_times = None
    if rank == 0 and cache_path.exists():
        if use_residual_var_y:
            raw_var, cached = load_residual_var_y_cache(
                cache_path,
                expected_channels=dataset.num_observation_channels,
                expected_kind="convi",
                expected_source=str(config.get("name", "convi")),
            )
            expected_indices = torch.as_tensor(
                dataset.observation_channel_indices
            ).cpu()
            cached_indices = torch.as_tensor(
                cached.get("global_channel_indices", [])
            ).cpu()
            if not torch.equal(cached_indices, expected_indices):
                raise RuntimeError(
                    "CONVI residual variance channels do not match enabled "
                    f"channels in {cache_path}"
                )
            counts = cached.get("counts")
            selected_times = cached.get("selected_times")
        else:
            cached = safe_load(cache_path, map_location="cpu")
            if cached.get("metadata") == metadata:
                raw_var = cached["var_y"].float()
                counts = cached.get("counts")
                selected_times = cached.get("selected_times")

    if rank == 0 and raw_var is None:
        if use_residual_var_y:
            raise FileNotFoundError(
                f"CONVI residual variance cache is missing: {cache_path}"
            )
        raw_var, counts, selected = calibrate_var_y_convi(
            dataset,
            Path(era_dir) / "era5.h5",
            max_frames=max_frames,
            valid_frac=valid_frac,
        )
        selected_times = [str(time) for time in selected]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        safe_save(
            {
                "var_y": raw_var,
                "counts": torch.as_tensor(counts),
                "selected_times": selected_times,
                "metadata": metadata,
            },
            cache_path,
        )

    var_y = torch.empty(
        dataset.num_observation_channels, dtype=torch.float32, device=device
    )
    if rank == 0:
        var_y.copy_(raw_var.to(device))
    if dist.is_initialized():
        dist.broadcast(var_y, src=0)

    if rank == 0:
        print(
            "[obs2era-dual] CONVIOBS calibration: "
            f"tail={valid_frac:.3f}, frames={len(selected_times)}, "
            f"counts={torch.as_tensor(counts).tolist()}, raw_var_y={raw_var.tolist()}",
            flush=True,
        )
    return (var_y * scale).clamp_min(floor)



def load_convi_at_time(
    time_str, dataset, era_resize, device, mask_frac=0.0,
):
    """Load enabled conventional observations and their direct latent support."""
    observation = dataset.prepare_sparse(np.datetime64(time_str, "h"))
    if observation is None:
        return None

    pixel_mask = observation.pop("pixel_mask")
    observation = filter_physical_observation_by_block_mask(
        observation,
        pixel_mask,
        era_resize,
        mask_frac=mask_frac,
    )
    if observation is None:
        return None
    for key in ("pixel_index", "channel_index", "channel_local", "values"):
        observation[key] = observation[key].to(device, non_blocking=True)
    observation["m_lat"] = observation["m_lat"].to(
        device, non_blocking=True
    )
    return observation


# =============================================
# Online: real ATMS frame -> standardized ERA5 latent observation
# =============================================
def load_obs2era_latent_at_time(
    time_str, encoder, era_mean, era_std, sate_dataset, time_to_idx,
    era_resize, device, mask_frac=0.0,
):
    """Encode the real ATMS frame at `time_str` ONLINE into the ERA5 latent,
    standardize it, and select the valid latent pixels (observed pixel fraction
    thresholded by mask_frac at run time).

    Returns a dict on `device`:
        m_lat: (S,)              bool   valid ERA5-latent pixels
        y_sel: (num_valid * C,)  float  standardized obs2era latent at valid
                                        pixels, row-major (pixel-major,
                                        channel-minor), the SAME order A(z) yields.
    or None if the timestamp has no obs frame / no valid pixels.
    """
    base = np.datetime64(time_str, "h")
    key = (int(str(base)[:4]), int(str(base)[5:7]), int(str(base)[8:10]), int(str(base)[11:13]))
    idx = time_to_idx.get(key)
    if idx is None:
        return None

    merged, _ = sate_dataset[idx]                 # (1, C, H, W) torch float32, C=23 or 24
    obs = merged[0].to(device)                    # (C, H, W) = [TB, mask (, cos_satzen)]
    # The mask always sits at obs_channel; optional satzen follows it.
    n_obs = sate_dataset.obs_channel

    # ObsToEraEncoder.forward wants (B, C, H, W) and pads internally.
    with torch.no_grad():
        z = encoder(obs.unsqueeze(0))                              # (1, S, C) native ERA latent
    z_std = (z - era_mean) / era_std.clamp_min(1e-8)              # standardized

    # Pixel mask -> latent-grid observed fraction, then threshold by mask_frac.
    mask_pix = obs[n_obs:n_obs + 1].unsqueeze(0)                 # (1, 1, H, W)
    frac = _obs_mask_to_frac(mask_pix, era_resize)               # (S,) float

    # Diagnostic: valid latent pixels per mask_frac threshold for THIS frame.
    S = int(frac.numel())
    counts = {r: int(_frac_to_mask(frac, r).sum()) for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)}
    # print(
    #     f"  [mask_frac@{time_str}] S={S} | "
    #     + " ".join(f"{r:.1f}->{n} ({100.0*n/S:.1f}%)" for r, n in counts.items()),
    #     flush=True,
    # )

    m_lat = _frac_to_mask(frac, mask_frac)                       # (S,) bool
    if int(m_lat.sum()) == 0:
        return None

    y_sel = z_std[0, m_lat, :].reshape(-1)
    return {"m_lat": m_lat, "y_sel": y_sel}


# =============================================
# Forward operator A(z) = z, selected to valid latent pixels (IDENTITY)
# =============================================
def observator_obs2era(
    blanket, blanket_id, blanket_size, blanket_stride,
    num_latent_channels, obs_fn, slice_fn=None,
):
    """Observation function for ERA5-latent-space assimilation with obs2era.

    The obs2era observation already lives in the ERA5 latent space, so there is
    NO mapper / decode: A(z) just selects the valid latent pixels of the
    trajectory directly (identity operator restricted to the swath).

    blanket: (B, K * N_era * C_era). Returns concatenated observed values.
    """
    blanket = rearrange(
        blanket, "B (K N C) -> B K N C", K=blanket_size, C=num_latent_channels,
    )
    if slice_fn is None:
        time_indices = slice(0, blanket_size)
    else:
        time_indices = slice_fn(blanket_id)

    y = []
    for i, t in enumerate(range(blanket_size)[time_indices]):
        z_era_t = blanket[:, t]                                  # (B, N_era, C_era) standardized
        observed_values, _ = obs_fn(z_era_t, blanket_id * blanket_stride + t)
        y.append(observed_values)
    return torch.cat(y)


def reanalysis(
    model_path,
    model_target,
    diffusion,
    assimilation_length,
    padded_trajectory_size,
    obs_covariance,
    blanket_overlap,
    start_date,
    start_hour,
    precision,
    target_dir,
    instruments,
    convi=None,
    var_y_mode="rmse",
    var_y_max_batches=64,
    use_residual_var_y=False,
    no_obs=False,
    seed=None,
    deterministic=False,
    mask_frac=0.0,
    guidance_merge="joint",
    pcgrad_overlap_only=False,
    pcgrad_shuffle=False,
    pcgrad_order_mode="random",
    pcgrad_uncertainty_strength=1.0,
    ppgf_overlap_only=False,
    ppgf_aligned_clip=0.0,
    ppgf_order_mode="gumbel",
    ppgf_uncertainty_strength=1.0,
    ppgf_ordering_cache=None,
    source_filter=None,
    ckpt_index=None,
    **_unused,
):
    rank = dist.get_rank()
    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    model_path = Path(model_path)
    latent_dir = model_path.parents[1]               # ERA5 latent dir (era5.h5 + stats.pth)
    denoiser_cfg = compose(model_path / "config.yaml")
    precision = getattr(torch, precision)
    use_bfloat16 = precision == torch.bfloat16
    convi_enabled = bool(
        not no_obs and convi is not None and convi.get("enabled", False)
    )
    instruments, convi_enabled = select_source_configs(
        instruments, convi_enabled, source_filter
    )
    convi_guidance_space = (
        str(convi.get("guidance_space", "physical")).lower()
        if convi_enabled
        else "physical"
    )
    if convi_guidance_space not in {"physical", "latent_obs_encoder"}:
        raise ValueError(
            "convi.guidance_space must be physical or latent_obs_encoder; "
            f"got {convi_guidance_space!r}"
        )

    trajectory_dt = TRAJECTORY_DT_HOURS
    blanket_size = denoiser_cfg.train.blanket_size
    blanket_stride = blanket_size - blanket_overlap
    diffusion_steps = (
        denoiser_cfg.valid.denoising_steps
        if diffusion.num_steps is None
        else diffusion.num_steps
    )

    # ERA5 latent stats (the space the guided trajectory + the observation live in).
    era_stats_d = safe_load(latent_dir / "stats.pth")
    era_mean = era_stats_d["mean"].to(device)
    era_std = era_stats_d["std"].to(device)

    # CONVIOBS either compares against the decoded ERA field or uses the frozen
    # ERA encoder to make a zero-filled latent observation.
    ae = load_auto_encoder(latent_dir / "ae", "model", device="cpu", eval_mode=True)
    latent_shape = ae.latent_shape
    if len(latent_shape) == 3:
        latent_shape = latent_shape[0] * latent_shape[1], latent_shape[2]
    state_size = math.prod(latent_shape)
    latent_channels = latent_shape[-1]
    if convi_enabled:
        ae.requires_grad_(False)
        if convi_guidance_space == "physical":
            ae.decoder.to(device)
        else:
            ae.encoder.to(device)
        era_autoencoder = ae
    else:
        era_autoencoder = None
        del ae

    # ERA5 latent resize factor (pixel -> latent downsample) for the mask pooling.
    era_ae_cfg = OmegaConf.load(latent_dir / "ae" / "config.yaml").get("ae")
    era_resize = int(era_ae_cfg.resize)

    # ---- obs2era observation sources (ONLINE encode, no pre-dump) ----
    # One source per instrument (e.g. ATMS + AMSUA); each has its own encoder,
    # its own SatelliteAEDataset, and its own per-ERA5-channel var_y, but all
    # encode into the SAME ERA5 latent space. They are STACKED into one joint
    # observation for MMPS (see module docstring). A fixed instrument ORDER is
    # kept for the whole run so the forward operator, y_obs_list and cov_y_list
    # concatenate identically.
    #
    # no_obs: leave y_obs empty everywhere -> MMPSDenoiser degrades to the plain
    #         denoiser, an unconditional sample. A fixed seed makes the guided and
    #         no-obs runs share the SAME starting noise.
    sources = []  # Fixed task order: SATE instruments, then optional CONVIOBS.

    if guidance_merge not in {
        "joint", "sum", "pcgrad", "ppgf", "channel_ppgf"
    }:
        raise ValueError(
            "guidance_merge must be one of joint, sum, pcgrad, ppgf, "
            f"channel_ppgf; got {guidance_merge!r}"
        )
    if pcgrad_order_mode not in {"random", "uncertainty"}:
        raise ValueError(
            "pcgrad_order_mode must be random or uncertainty; "
            f"got {pcgrad_order_mode!r}"
        )
    if pcgrad_uncertainty_strength < 0:
        raise ValueError("pcgrad_uncertainty_strength must be non-negative")
    if ppgf_order_mode not in {"random", "gumbel", "fixed"}:
        raise ValueError(
            "ppgf_order_mode must be random, gumbel, or fixed; "
            f"got {ppgf_order_mode!r}"
        )
    if ppgf_uncertainty_strength < 0:
        raise ValueError("ppgf_uncertainty_strength must be non-negative")
    ppgf_aligned_clip = float(ppgf_aligned_clip)
    if not -1.0 <= ppgf_aligned_clip <= 1.0:
        raise ValueError("ppgf_aligned_clip must be in [-1, 1]")

    if no_obs:
        if rank == 0:
            print("[obs2era-dual] no_obs=True — unconditional baseline (no guidance).", flush=True)
    else:
        if len(instruments) == 0 and not convi_enabled:
            raise ValueError(
                "at least one satellite instrument or CONVI source is required"
            )
        if rank == 0 and source_filter is not None:
            print(
                f"[obs2era-dual] source_filter={source_filter}; "
                "running one independent guidance source",
                flush=True,
            )

        for inst in instruments:
            model_dir = Path(inst["model_dir"])
            checkpoint = inst.get("checkpoint", "best")
            inst_var_y_scale = float(inst.get("var_y_scale", 1.0))  # per-instrument trust knob
            representation = str(
                inst.get("representation", "obs2era")
            ).lower()
            if representation not in {"obs2era", "direct_e2o"}:
                raise ValueError(
                    "instrument representation must be obs2era or direct_e2o; "
                    f"got {representation!r}"
                )

            if representation == "direct_e2o":
                source = load_direct_e2o_source(
                    model_dir,
                    checkpoint,
                    latent_dir,
                    latent_shape,
                    device,
                )
                source.update(
                    {
                        "name": source["sate_dataset"].sate_name,
                        "kind": "direct_e2o",
                        "pcgrad_uncertainty_scale": inst.get(
                            "pcgrad_uncertainty_scale", 1.0
                        ),
                        "var_y_cache": inst.get("var_y_cache", None),
                    }
                )
                source["var_y"] = get_direct_e2o_var_y(
                    source,
                    var_y_mode=var_y_mode,
                    obs_covariance=obs_covariance,
                    var_y_scale=inst_var_y_scale,
                    use_residual_var_y=use_residual_var_y,
                    max_frames=int(
                        inst.get("var_y_max_batches", var_y_max_batches)
                    ),
                    valid_frac=float(inst.get("var_y_valid_frac", 0.05)),
                    device=device,
                )
                sources.append(source)
                if rank == 0:
                    print(
                        "[obs2era-dual] Direct E2O instrument "
                        f"'{source['name']}' ({model_dir.name}); "
                        f"TB channels={source['var_y'].numel()}; "
                        f"checkpoint={source['checkpoint_path'].name}",
                        flush=True,
                    )
                continue

            # Guardrail: each obs2era must have been trained on THIS denoiser's ERA5 latent.
            o2e_cfg = OmegaConf.load(model_dir / "config.yaml")
            if str(o2e_cfg.get("era_latent_dir", "")).rstrip("/") != str(latent_dir).rstrip("/"):
                warnings.warn(
                    f"obs2era era_latent_dir ({o2e_cfg.get('era_latent_dir')}) != denoiser "
                    f"latent_dir ({latent_dir}); standardization spaces may be inconsistent.",
                    stacklevel=2,
                )

            # Real obs dataset (paths + sate_name from THIS model's training config).
            # use_satzen is per-instrument: read from EACH model's saved config so
            # get_dim() matches that encoder's input conv (ATMS/AMSUA may differ).
            o2e_data = o2e_cfg.get("data")
            sate_name = str(o2e_data.sate_name)
            o2e_use_satzen = bool(o2e_data.get("use_satzen", False))
            sate_dataset = SatelliteAEDataset(
                data_path=o2e_data.sate_data_path,
                sate_name=sate_name,
                standard_dir=o2e_data.sate_standard_dir,
                blanket_size=1,
                time_range=None,
                use_satzen=o2e_use_satzen,
            )
            if rank == 0:
                print(
                    f"[obs2era-dual] {sate_name}: use_satzen={o2e_use_satzen}  "
                    f"encoder in_channels={sate_dataset.get_dim()} "
                    f"(mask@{sate_dataset.obs_channel} kept)",
                    flush=True,
                )
            time_to_idx = {
                (int(t.year), int(t.month), int(t.day), int(t.hour)): i
                for i, t in enumerate(sate_dataset.times)
            }

            # Online obs2era encoder (in_channels from this instrument's get_dim()).
            encoder = load_obs2era_encoder(
                model_dir, era_ae_cfg, sate_dataset.get_dim(), checkpoint, device,
            )

            var_y_per_channel = get_var_y_per_channel_obs2era(
                OmegaConf.create({
                    "var_y_mode": var_y_mode,
                    "obs_covariance": obs_covariance,
                    "var_y_max_batches": var_y_max_batches,
                    "obs2era_model_dir": str(model_dir),
                    "obs2era_checkpoint": checkpoint,
                    "var_y_scale": inst_var_y_scale,   # per-instrument obs-trust knob
                    "use_residual_var_y": use_residual_var_y,
                }),
                encoder, sate_dataset, latent_dir, era_resize, era_mean, era_std, device,
            )  # (C,)

            sources.append({
                "name": sate_name,
                "kind": "obs2era",
                "model_dir": model_dir,
                "encoder": encoder,
                "sate_dataset": sate_dataset,
                "time_to_idx": time_to_idx,
                "var_y": var_y_per_channel,
                "pcgrad_uncertainty_scale": inst.get(
                    "pcgrad_uncertainty_scale", 1.0
                ),
            })

            if rank == 0:
                rmse_lat = torch.sqrt(var_y_per_channel) * era_std
                print(
                    f"[obs2era-dual] instrument '{sate_name}' ({model_dir.name}) "
                    f"var_y_mode={var_y_mode}  residual={use_residual_var_y}  "
                    f"var_y_scale={inst_var_y_scale}  "
                    f"per-channel rmse (native latent units) "
                    f"mean={rmse_lat.mean().item():.4f}, range "
                    f"[{rmse_lat.min().item():.4f}, {rmse_lat.max().item():.4f}]",
                    flush=True,
                )

        if convi_enabled:
            include_temp = bool(convi.get("include_temp", False))
            if include_temp and convi_guidance_space == "latent_obs_encoder":
                raise ValueError(
                    "latent_obs_encoder is fixed to t2m/u10/v10/msl and "
                    "requires convi.include_temp=false"
                )
            required = ["surf_dir", "icoa_dir", "mean_std_dir"]
            if include_temp:
                required += ["temp_dir", "temp_bias_dir"]
            missing = [key for key in required if not convi.get(key)]
            if missing:
                raise ValueError(
                    f"CONVIOBS enabled but configuration is missing: {missing}"
                )
            convi_dataset = CONVIOBS(
                surf_dir=convi.surf_dir,
                icoa_dir=convi.icoa_dir,
                mean_std_dir=convi.mean_std_dir,
                temp_dir=convi.get("temp_dir", None),
                temp_bias_dir=convi.get("temp_bias_dir", None),
                skip_temp=not include_temp,
                skip_humidity=bool(convi.get("skip_humidity", True)),
            )
            if len(convi_dataset) == 0:
                raise RuntimeError("CONVIOBS found no enabled observation timestamps")
            convi_variance_config = OmegaConf.create(
                {
                    "surf_dir": str(convi.surf_dir),
                    "icoa_dir": str(convi.icoa_dir),
                    "mean_std_dir": str(convi.mean_std_dir),
                    "temp_dir": convi.get("temp_dir", None),
                    "temp_bias_dir": convi.get("temp_bias_dir", None),
                    "skip_temp": not include_temp,
                    "skip_humidity": bool(convi.get("skip_humidity", True)),
                    "var_y_cache": convi.get("var_y_cache", None),
                    "latent_residual_var_y_cache": convi.get(
                        "latent_residual_var_y_cache", None
                    ),
                    "obs_encoder_residual_var_y_cache": convi.get(
                        "obs_encoder_residual_var_y_cache", None
                    ),
                    "var_y_valid_frac": convi.get("var_y_valid_frac", 0.05),
                    "var_y_max_batches": convi.get(
                        "var_y_max_batches", var_y_max_batches
                    ),
                    "var_y_scale": convi.get("var_y_scale", 1.0),
                    "var_y_mode": var_y_mode,
                    "obs_covariance": obs_covariance,
                    "use_residual_var_y": use_residual_var_y,
                    "residual_var_y_cache": convi.get(
                        "residual_var_y_cache", None
                    ),
                    "name": str(convi.get("name", "convi")),
                }
            )
            convi_encoder = None
            if convi_guidance_space == "latent_obs_encoder":
                model_value = convi.get("obs_encoder_model_dir", None)
                if not model_value:
                    raise ValueError(
                        "latent_obs_encoder requires convi.obs_encoder_model_dir"
                    )
                encoder_model_dir = Path(model_value).resolve()
                encoder_checkpoint = str(
                    convi.get("obs_encoder_checkpoint", "best")
                )
                encoder_config = OmegaConf.load(
                    encoder_model_dir / "config.yaml"
                )
                encoder_kind = str(encoder_config.data.get("kind", "sate"))
                if encoder_kind != "convi":
                    raise RuntimeError(
                        "CONVI obs encoder config has wrong data.kind: "
                        f"{encoder_kind!r}"
                    )
                configured_latent_dir = str(encoder_config.era_latent_dir)
                if configured_latent_dir.rstrip("/") != str(latent_dir).rstrip("/"):
                    raise RuntimeError(
                        "CONVI obs encoder was trained for a different ERA latent: "
                        f"{configured_latent_dir} != {latent_dir}"
                    )
                convi_encoder = load_obs2era_encoder(
                    encoder_model_dir,
                    era_ae_cfg,
                    CONVI_OBS_ENCODER_CHANNELS,
                    encoder_checkpoint,
                    device,
                )
                convi_var_y = get_var_y_convi_obs_encoder(
                    convi_variance_config,
                    convi_dataset,
                    latent_dir,
                    era_mean,
                    encoder_model_dir,
                    encoder_checkpoint,
                    device,
                )
                convi_kind = "convi_obs_encoder"
            else:
                convi_var_y = get_var_y_convi(
                    convi_variance_config,
                    convi_dataset,
                    latent_dir,
                    model_path,
                    device,
                )
                convi_kind = "convi"
            sources.append(
                {
                    "name": str(convi.get("name", "convi")),
                    "kind": convi_kind,
                    "dataset": convi_dataset,
                    "encoder": convi_encoder,
                    "var_y": convi_var_y,
                    "pcgrad_uncertainty_scale": convi.get(
                        "pcgrad_uncertainty_scale", 1.0
                    ),
                }
            )
            if rank == 0:
                print(
                    "[obs2era-dual] CONVIOBS added as one task: "
                    f"guidance_space={convi_guidance_space}, "
                    f"observation_channels={convi_dataset.num_observation_channels}, "
                    f"guidance_channels={convi_var_y.numel()}, "
                    f"var_y mean={convi_var_y.mean().item():.4g}, range "
                    f"[{convi_var_y.min().item():.4g}, "
                    f"{convi_var_y.max().item():.4g}]",
                    flush=True,
                )

        if rank == 0:
            print(
                f"[obs2era-dual] {len(sources)} instrument(s) stacked, fixed order: "
                + ", ".join(s["name"] for s in sources),
                flush=True,
            )

    channel_uncertainty = None
    ordering_rmse = None
    ordering_cache_path = (
        Path(ppgf_ordering_cache).resolve()
        if ppgf_ordering_cache is not None
        else None
    )
    if ordering_cache_path is not None:
        if guidance_merge not in {"ppgf", "channel_ppgf"}:
            raise ValueError(
                "ppgf_ordering_cache requires guidance_merge=ppgf or "
                "guidance_merge=channel_ppgf"
            )
        polar_kinds = {
            source["kind"]
            for source in sources
            if str(source["name"]).lower() != "convi"
        }
        polar_route_by_kind = {
            "obs2era": "encoder",
            "direct_e2o": "decoder",
        }
        if (
            len(polar_kinds) != 1
            or next(iter(polar_kinds)) not in polar_route_by_kind
        ):
            raise RuntimeError(
                "PPGF ordering cache requires one uniform polar guidance "
                f"route; source kinds are {sorted(polar_kinds)}"
            )
        polar_guidance = polar_route_by_kind[next(iter(polar_kinds))]
        convi_sources = [
            source
            for source in sources
            if str(source["name"]).lower() == "convi"
        ]
        if len(convi_sources) != 1:
            raise RuntimeError(
                "PPGF ordering cache requires exactly one CONVI source"
            )
        convi_route_by_kind = {
            "convi_obs_encoder": "encoder",
            "convi": "physical",
        }
        convi_kind = convi_sources[0]["kind"]
        if convi_kind not in convi_route_by_kind:
            raise RuntimeError(
                "unsupported CONVI guidance kind for PPGF ordering cache: "
                f"{convi_kind!r}"
            )
        convi_guidance = convi_route_by_kind[convi_kind]
        ordering_rmse = load_ppgf_ordering_cache(
            ordering_cache_path,
            [source["name"] for source in sources],
            latent_channels,
            mask_frac,
            polar_guidance,
            convi_guidance,
        )

    instrument_uncertainty = None
    if sources:
        if ordering_rmse is not None:
            scaled_ordering_rmse = torch.stack([
                row.mul(float(source["pcgrad_uncertainty_scale"]))
                for row, source in zip(ordering_rmse, sources)
            ])
            instrument_uncertainty = (
                scaled_ordering_rmse.clamp_min(1e-12).log().mean(dim=1).exp()
            )
            if rank == 0:
                uncertainty_text = ", ".join(
                    f"{source['name']}={uncertainty.item():.6f}"
                    for source, uncertainty in zip(
                        sources, instrument_uncertainty
                    )
                )
                print(
                    "[obs2era-dual] route-matched PPGF ordering cache: "
                    f"{ordering_cache_path}; vector reduction="
                    "geometric_mean(native_RMSE); MMPS cov_y remains "
                    "residual var_y; order_uncertainty=["
                    f"{uncertainty_text}]",
                    flush=True,
                )
        else:
            instrument_uncertainty = torch.stack([
                pcgrad_order_uncertainty(
                    source["var_y"], source["pcgrad_uncertainty_scale"]
                )
                for source in sources
            ])
        if guidance_merge == "channel_ppgf":
            if ordering_rmse is not None:
                channel_uncertainty = scaled_ordering_rmse.T.contiguous().cpu()
                if rank == 0:
                    print(
                        "[obs2era-dual] Channel PPGF uses the full "
                        f"{tuple(ordering_rmse.shape)} native-RMSE matrix; "
                        "MMPS cov_y is unchanged",
                        flush=True,
                    )
            else:
                channel_values = []
                for source in sources:
                    channel_value = channel_ppgf_uncertainty(
                        source["var_y"],
                        source["kind"],
                        latent_channels,
                        source["pcgrad_uncertainty_scale"],
                    )
                    channel_values.append(channel_value)
                    if (
                        rank == 0
                        and source["kind"] in {"convi", "direct_e2o"}
                        and source["var_y"].numel() != latent_channels
                    ):
                        print(
                            "[obs2era-dual] channel_ppgf physical-source "
                            f"uncertainty for {source['name']}: geometric mean="
                            f"{channel_value[0].item():.6f}, broadcast to "
                            f"{latent_channels} latent channels",
                            flush=True,
                        )
                channel_uncertainty = torch.stack(
                    channel_values, dim=1
                ).cpu()
        if rank == 0 and guidance_merge == "pcgrad" and pcgrad_shuffle:
            uncertainty_text = ", ".join(
                f"{source['name']}={uncertainty.item():.6f}"
                for source, uncertainty in zip(sources, instrument_uncertainty)
            )
            print(
                f"[obs2era-dual] PCGrad order_mode={pcgrad_order_mode}; "
                f"uncertainty_strength={pcgrad_uncertainty_strength}; "
                f"order_uncertainty=[{uncertainty_text}]",
                flush=True,
            )
        if rank == 0 and guidance_merge == "ppgf":
            mode_detail = ""
            if ppgf_order_mode in {"gumbel", "fixed"}:
                uncertainty_text = ", ".join(
                    f"{source['name']}={uncertainty.item():.6f}"
                    for source, uncertainty in zip(sources, instrument_uncertainty)
                )
                mode_detail = f"; order_uncertainty=[{uncertainty_text}]"
            print(
                f"[obs2era-dual] PPGF order_mode={ppgf_order_mode}; "
                f"uncertainty_strength={ppgf_uncertainty_strength}; "
                f"aligned_clip={ppgf_aligned_clip}"
                f"{mode_detail}",
                flush=True,
            )
        if rank == 0 and guidance_merge == "channel_ppgf":
            mode_detail = ""
            if ppgf_order_mode in {"gumbel", "fixed"}:
                mode_detail = (
                    f"; uncertainty_shape={tuple(channel_uncertainty.shape)}"
                )
            print(
                f"[obs2era-dual] Channel PPGF order_mode={ppgf_order_mode}; "
                f"uncertainty_strength={ppgf_uncertainty_strength}; "
                f"fusion=opposite_sign_replace,same_sign_add{mode_detail}",
                flush=True,
            )

    if use_bfloat16:
        torch.set_default_dtype(torch.bfloat16)

    noise_schedule = create_schedule(denoiser_cfg.train).to(device)
    backbone = load_denoiser(
        model_path,
        best=model_target == "best",
        overrides={"checkpointing": True},
        name=ckpt_index,
    ).backbone.to(device)
    backbone.requires_grad_(False)

    if rank == 0:
        timestamps = create_trajectory_timestamps(
            start_date, start_hour, assimilation_length, trajectory_dt
        )
        safe_save(timestamps[None], target_dir / "timestamps.pt")
        del timestamps

    timestamps = create_trajectory_timestamps(
        start_date, start_hour, padded_trajectory_size, trajectory_dt
    )[None]

    num_blankets = (padded_trajectory_size - blanket_size) // blanket_stride + 1
    num_gpus = dist.get_world_size()
    absolute_blanket_ids = torch.arange(num_blankets).tensor_split(num_gpus)[rank]
    rank_start_idx = absolute_blanket_ids[0] * blanket_stride
    rank_end_idx = absolute_blanket_ids[-1] * blanket_stride + blanket_size

    denoiser = Denoiser(backbone).cuda()
    if use_bfloat16:
        denoiser = denoiser.to(torch.bfloat16)

    if (not no_obs) and absolute_blanket_ids[0] * blanket_stride < assimilation_length:
        # ---- Pre-load obs (standardized ERA5 latent + mask) per step ----
        # local_obs[absolute_step] is a LIST over instruments (fixed `sources`
        # order); each entry is {m_lat, y_sel} or None if that instrument has no
        # frame at this step. A step is stored if ANY instrument observed it.
        local_obs = {}
        local_assim_end = min(rank_end_idx, assimilation_length)
        for absolute_step in range(rank_start_idx, local_assim_end):
            abs_hour = absolute_step * trajectory_dt
            if abs_hour % OBS_DT_HOURS != 0:
                continue
            time_str = _abs_hour_to_time_str(start_date, start_hour, abs_hour)
            per_inst = []
            for source in sources:
                if source["kind"] == "obs2era":
                    observation = load_obs2era_latent_at_time(
                        time_str,
                        source["encoder"],
                        era_mean,
                        era_std,
                        source["sate_dataset"],
                        source["time_to_idx"],
                        era_resize,
                        device,
                        mask_frac,
                    )
                elif source["kind"] == "direct_e2o":
                    observation = load_direct_e2o_at_time(
                        time_str,
                        source["sate_dataset"],
                        source["time_to_idx"],
                        device,
                    )
                    if observation is not None:
                        pixel_mask = observation.pop("pixel_mask")
                        observation = filter_physical_observation_by_block_mask(
                            observation,
                            pixel_mask,
                            era_resize,
                            mask_frac=mask_frac,
                        )
                elif source["kind"] == "convi_obs_encoder":
                    observation = load_convi_obs_encoder_latent_at_time(
                        time_str,
                        source["dataset"],
                        source["encoder"],
                        era_mean,
                        era_std,
                        era_resize,
                        device,
                        mask_frac,
                    )
                else:
                    observation = load_convi_at_time(
                        time_str,
                        source["dataset"],
                        era_resize,
                        device,
                        mask_frac=mask_frac,
                    )
                per_inst.append(observation)
            if all(o is None for o in per_inst):
                if rank == 0:
                    print(f"  [warn] no obs frame for {time_str} (any instrument)", flush=True)
                continue
            local_obs[absolute_step] = per_inst

        if rank == 0:
            total_obs = sum(
                int((o.get("y_sel", o.get("values"))).numel())
                for per_inst in local_obs.values()
                for o in per_inst
                if o is not None
            )
            print(
                f"[obs2era-dual] rank0 local obs: {len(local_obs)} steps, "
                f"{total_obs:.3e} observed values total (all instruments)",
                flush=True,
            )
            filtered_observations = [
                observation
                for per_inst in local_obs.values()
                for observation in per_inst
                if observation is not None
                and "block_filter_values_before" in observation
            ]
            if filtered_observations:
                values_before = sum(
                    observation["block_filter_values_before"]
                    for observation in filtered_observations
                )
                values_after = sum(
                    observation["block_filter_values_after"]
                    for observation in filtered_observations
                )
                pixels_before = sum(
                    observation["block_filter_pixels_before"]
                    for observation in filtered_observations
                )
                pixels_after = sum(
                    observation["block_filter_pixels_after"]
                    for observation in filtered_observations
                )
                print(
                    "[obs2era-dual] true block filter: "
                    f"mask_frac={float(mask_frac):g}; values="
                    f"{values_after}/{values_before}; spatial_pixels="
                    f"{pixels_after}/{pixels_before}",
                    flush=True,
                )

        # One heterogeneous operator serves joint MMPS; source-specific partials
        # avoid running the ERA decoder while solving a SATE-only task.
        def source_obs_fn(z_era, t_in_rank, source_ids):
            absolute_step = rank_start_idx + t_in_rank
            absolute_step = (
                absolute_step.item()
                if torch.is_tensor(absolute_step)
                else absolute_step
            )
            empty = torch.empty(0, device=z_era.device, dtype=z_era.dtype)
            if absolute_step >= assimilation_length:
                return empty, empty
            per_inst = local_obs.get(int(absolute_step), None)
            if per_inst is None:
                return empty, empty

            batch_size = z_era.shape[0]
            values = []
            for source_id in source_ids:
                source = sources[source_id]
                observation = per_inst[source_id]
                if observation is None:
                    continue
                if source["kind"] in {"obs2era", "convi_obs_encoder"}:
                    mask = observation["m_lat"]
                    values.append(z_era[:, mask, :].reshape(-1))
                    continue
                if source["kind"] == "direct_e2o":
                    z_native = z_era * era_std + era_mean
                    prediction = source["decoder"].forward_at_pixels(
                        z_native,
                        observation["pixel_index"],
                        observation["cos_satzen"],
                    )
                    values.append(
                        select_direct_e2o_prediction(
                            prediction,
                            observation["valid_mask"],
                        )
                    )
                    continue

                z_native = z_era * era_std + era_mean
                date = timestamps[:, int(absolute_step)].to(z_era.device)
                if batch_size != date.shape[0]:
                    date = date.expand(batch_size, -1)
                decoded = era_autoencoder.decode(z_native, date)
                pixel = observation["pixel_index"]
                channel = observation["channel_index"]
                values.append(decoded[:, pixel, channel].reshape(-1))

            if not values:
                return empty, empty
            return torch.cat(values), empty

        # ---- Pre-compute y_obs and cov_y per blanket ----
        y_obs_list = []
        cov_y_list = []
        timestamp_obs_list = []
        instrument_indices_list = []
        instrument_masks_list = []

        for blanket_id in range(len(absolute_blanket_ids)):
            absolute_start_idx = rank_start_idx + blanket_id * blanket_stride

            y_obs = []
            cov_y = []
            t_obs = []
            instrument_index_parts = [[] for _ in sources]
            instrument_masks = torch.zeros(
                len(sources), blanket_size, latent_shape[0], dtype=torch.bool
            )
            observation_offset = 0
            for t_local in range(blanket_size):
                absolute_step = absolute_start_idx + t_local
                if absolute_step >= assimilation_length:
                    break

                abs_hour = absolute_step * trajectory_dt
                time_str = _abs_hour_to_time_str(start_date, start_hour, abs_hour)
                t_obs.append(get_date_features(np.datetime64(time_str)))

                if abs_hour % OBS_DT_HOURS != 0:
                    continue

                per_inst = local_obs.get(int(absolute_step), None)
                if per_inst is None:
                    continue
                # Stack instruments in the SAME fixed order A(x) produces, so the
                # joint y / cov_y line up element-for-element with the operator.
                for instrument_id, (src, obs) in enumerate(zip(sources, per_inst)):
                    if obs is None:
                        continue
                    instrument_masks[instrument_id, t_local] = obs["m_lat"].cpu()
                    if src["kind"] in {"obs2era", "convi_obs_encoder"}:
                        observed_values = obs["y_sel"]
                        channels = int(src["var_y"].numel())
                        pixels = int(obs["m_lat"].sum())
                        observation_variance = (
                            src["var_y"]
                            .view(1, channels)
                            .expand(pixels, channels)
                            .reshape(-1)
                        )
                    else:
                        observed_values = obs["values"]
                        observation_variance = src["var_y"][obs["channel_local"]]
                    segment_length = observed_values.numel()
                    y_obs.append(observed_values)
                    cov_y.append(observation_variance)
                    instrument_index_parts[instrument_id].append(
                        torch.arange(
                            observation_offset,
                            observation_offset + segment_length,
                            dtype=torch.long,
                        )
                    )
                    observation_offset += segment_length

            if y_obs:
                y_obs_list.append(torch.cat(y_obs, dim=-1))
                cov_y_list.append(torch.cat(cov_y, dim=-1))
            else:
                y_obs_list.append(torch.empty(0, device=device))
                cov_y_list.append(torch.empty(0, device=device))
            instrument_indices_list.append([
                torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
                for parts in instrument_index_parts
            ])
            instrument_masks_list.append(list(instrument_masks.unbind(0)))
            timestamp_obs_list.append(torch.stack(t_obs).cuda())

        slice_fn = lambda blanket_id: slice(
            0, max(0, assimilation_length - rank_start_idx - blanket_id * blanket_stride)
        )

        def make_source_operator(source_ids):
            return partial(
                observator_obs2era,
                blanket_size=blanket_size,
                blanket_stride=blanket_stride,
                num_latent_channels=latent_channels,
                obs_fn=partial(source_obs_fn, source_ids=tuple(source_ids)),
                slice_fn=slice_fn,
            )

        A = make_source_operator(range(len(sources)))
        instrument_operators = [
            make_source_operator([source_id])
            for source_id in range(len(sources))
        ]

        if guidance_merge == "joint":
            denoiser = MMPSDenoiser(
                denoiser, A, y_obs_list, cov_y_list,
                iterations=diffusion.mmps_iters,
                tweedie_covariance=diffusion.get("tweedie_covariance", True),
            )
            guidance_denoiser = None
        else:
            denoiser = MultiInstrumentMMPSDenoiser(
                denoiser, A, y_obs_list, cov_y_list,
                instrument_indices=instrument_indices_list,
                instrument_masks=instrument_masks_list,
                instrument_operators=(
                    instrument_operators
                    if any(
                        source["kind"] in {"convi", "direct_e2o"}
                        for source in sources
                    )
                    else None
                ),
                merge=guidance_merge,
                pcgrad_group_size=latent_channels,
                pcgrad_overlap_only=pcgrad_overlap_only,
                pcgrad_shuffle=pcgrad_shuffle,
                ppgf_overlap_only=ppgf_overlap_only,
                ppgf_aligned_clip=ppgf_aligned_clip,
                iterations=diffusion.mmps_iters,
                tweedie_covariance=diffusion.get("tweedie_covariance", True),
            )
            guidance_denoiser = denoiser
            if rank == 0:
                print(
                    f"[obs2era-dual] guidance_merge={guidance_merge}; "
                    f"overlap_only={pcgrad_overlap_only}; "
                    f"shuffle={pcgrad_shuffle}; "
                    f"order_mode={pcgrad_order_mode}; "
                    f"ppgf_overlap_only={ppgf_overlap_only}; "
                    f"ppgf_aligned_clip={ppgf_aligned_clip}; "
                    f"ppgf_strength={ppgf_uncertainty_strength}; "
                    "solving each instrument independently.",
                    flush=True,
                )
    else:
        guidance_denoiser = None

    trajectory_denoiser = TrajectoryDenoiser(
        denoiser,
        blanket_size=blanket_size,
        blanket_stride=blanket_stride,
        state_size=state_size,
        distributed=True,
        pass_blanket_ids=not no_obs,
        mode="symmetrical",
    )
    denoiser = partial(trajectory_denoiser, date=timestamps.cuda())

    @torch.no_grad()
    def sample():
        sampler_type = diffusion.sampler.type
        sampler_cls = select_sampler(sampler_type)
        sampler_kwargs = dict(diffusion.sampler.config)
        if guidance_denoiser is not None and guidance_denoiser.ppgf_ordered:
            if sampler_type != "pc":
                raise ValueError("PPGF currently requires the PC sampler dynamics")
            sampler_cls = PPGFSampler
            sampler_kwargs.update(
                ppgf_num_tasks=guidance_denoiser.num_instruments,
                ppgf_num_channels=(
                    latent_channels if guidance_denoiser.channel_ppgf else 0
                ),
                ppgf_channelwise=guidance_denoiser.channel_ppgf,
                ppgf_seed=seed,
                ppgf_order_mode=ppgf_order_mode,
                ppgf_uncertainty=(
                    (
                        channel_uncertainty
                        if guidance_denoiser.channel_ppgf
                        else instrument_uncertainty
                    )
                    if ppgf_order_mode in {"gumbel", "fixed"}
                    else None
                ),
                ppgf_uncertainty_strength=ppgf_uncertainty_strength,
            )
        elif guidance_denoiser is not None and guidance_denoiser.pcgrad_shuffle:
            if sampler_type != "pc":
                raise ValueError("Shuffled PCGrad currently requires the PC sampler")
            sampler_kwargs.update(
                pcgrad_shuffle=True,
                pcgrad_num_tasks=guidance_denoiser.num_instruments,
                pcgrad_seed=seed,
                pcgrad_order_mode=pcgrad_order_mode,
                pcgrad_uncertainty=instrument_uncertainty,
                pcgrad_uncertainty_strength=pcgrad_uncertainty_strength,
            )
        sampler = sampler_cls(
            denoiser=denoiser,
            steps=diffusion_steps,
            schedule=noise_schedule,
            silent=rank > 0,
            **sampler_kwargs,
        )
        # Fixed seed -> identical starting noise across guided and no-obs runs.
        # Seed the GLOBAL RNGs too: the PC sampler injects Langevin noise via
        # torch.randn_like, which draws from the global (not the local `gen`)
        # generator. Same per-rank offset keeps ranks independent yet
        # run-to-run reproducible.
        if seed is None:
            x1 = torch.randn(len(timestamps), padded_trajectory_size * state_size, device=device)
        else:
            rank_seed = int(seed) + 10_000 * rank
            random.seed(rank_seed)
            np.random.seed(rank_seed % (2**32))
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed_all(rank_seed)
            gen = torch.Generator(device=device).manual_seed(rank_seed)
            x1 = torch.randn(
                len(timestamps), padded_trajectory_size * state_size,
                generator=gen, device=device,
            )
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        samp_start = (x1 * noise_schedule.sigma_tmax().cuda()).flatten(1).cuda()
        return sampler(samp_start).reshape((-1, padded_trajectory_size, *latent_shape)).cpu()

    if precision != torch.float16:
        sampled_traj = sample()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            sampled_traj = sample()
    if guidance_denoiser is not None:
        for statistic in (
            guidance_denoiser.cosine_sum,
            guidance_denoiser.cosine_count,
            guidance_denoiser.conflict_count,
            guidance_denoiser.guidance_norm_sum,
            guidance_denoiser.guidance_norm_count,
        ):
            dist.all_reduce(statistic, op=dist.ReduceOp.SUM)

    if rank == 0:
        sampled_traj = sampled_traj[:, :assimilation_length]
        safe_save(sampled_traj, target_dir / "trajectory.pt")
        if guidance_denoiser is not None:
            safe_save(
                {
                    "guidance_merge": guidance_merge,
                    "pcgrad_group_size": guidance_denoiser.pcgrad_group_size,
                    "pcgrad_overlap_only": guidance_denoiser.pcgrad_overlap_only,
                    "pcgrad_shuffle": guidance_denoiser.pcgrad_shuffle,
                    "pcgrad_order_mode": pcgrad_order_mode,
                    "pcgrad_uncertainty_strength": pcgrad_uncertainty_strength,
                    "ppgf_overlap_only": guidance_denoiser.ppgf_overlap_only,
                    "ppgf_aligned_clip": guidance_denoiser.ppgf_aligned_clip,
                    "ppgf_order_mode": ppgf_order_mode,
                    "ppgf_uncertainty_strength": ppgf_uncertainty_strength,
                    "instrument_uncertainty": instrument_uncertainty,
                    "channel_uncertainty": channel_uncertainty,
                    "ppgf_ordering_cache": (
                        str(ordering_cache_path)
                        if ordering_cache_path is not None
                        else None
                    ),
                    "ppgf_ordering_rmse": ordering_rmse,
                    "instrument_order": [source["name"] for source in sources],
                    **guidance_denoiser.diagnostics(),
                },
                target_dir / "guidance_diagnostics.pth",
            )

    return sampled_traj


def main(config_path="configs/reanalysis_obs2era_dual.yaml"):
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(
            f"Usage: torchrun --nproc-per-node <NUM_GPUS> {Path(sys.argv[0]).name} "
            "model_path=... [+id=<id>] [param=value ...] "
            f"(defaults from {config_path})"
        )
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist_timeout_seconds = float(os.environ.get("APPA_DIST_TIMEOUT_SECONDS", 3600))
    if not math.isfinite(dist_timeout_seconds) or dist_timeout_seconds <= 0:
        raise ValueError("APPA_DIST_TIMEOUT_SECONDS must be finite and positive")
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
        timeout=timedelta(seconds=dist_timeout_seconds),
    )
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"[distributed] collective timeout={dist_timeout_seconds:g}s",
            flush=True,
        )

    config = compose(config_path, overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    unique_id = config.pop("id") if "id" in config else "default_obs2era"
    ckpt_index = config.pop("ckpt_index", None)
    if ckpt_index is not None:
        unique_id = f"{unique_id}_step{int(ckpt_index) * 2000}"

    configured_output_root = config.get("output_root", None)
    output_root = (
        Path(configured_output_root)
        if configured_output_root is not None
        else Path(config.model_path) / "reanalysis"
    )
    target_dir = output_root / unique_id
    if rank == 0:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_config = copy.deepcopy(config)
        saved_config.unique_id = unique_id
        OmegaConf.save(saved_config, target_dir / "config.yaml")

    assimilation_lengths = str(config.pop("assimilation_lengths"))
    blanket_size = compose(f"{config.model_path}/config.yaml").train.blanket_size
    overlap = config.blanket_overlap
    blanket_stride = blanket_size - overlap

    num_samples_per_date = config.pop("num_samples_per_date")
    start_dates = config.pop("start_dates")
    start_dates = [start_date for start_date in start_dates for _ in range(num_samples_per_date)]
    start_dates = [
        (day, int(hour.split("h")[0])) for date in start_dates for day, hour in [date.split(" ")]
    ]
    start_dates, start_hours = zip(*start_dates)
    num_samples = len(start_dates)

    for assimilation_length in str_to_ids(ids_str=assimilation_lengths):
        trajectory_size = max(blanket_size, assimilation_length)
        while (trajectory_size - blanket_size) % blanket_stride != 0:
            trajectory_size += 1

        num_blankets = (trajectory_size - blanket_size) // blanket_stride + 1
        num_gpus = dist.get_world_size()
        if num_gpus > num_blankets:
            warnings.warn(
                f"Number of GPUs ({num_gpus}) exceeds blankets ({num_blankets}). "
                f"Some GPUs will be idle.",
                stacklevel=2,
            )

        window_target_dir = target_dir / f"{assimilation_length}h"

        for i in range(num_samples):
            target_dir_traj = window_target_dir / f"tmp_{i}"
            if rank == 0:
                target_dir_traj.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            if rank == 0:
                print(f"Reanalysis {assimilation_length}h — sample {i + 1}/{num_samples}")

            reanalysis(
                assimilation_length=assimilation_length,
                padded_trajectory_size=trajectory_size,
                start_date=start_dates[i],
                start_hour=start_hours[i],
                target_dir=target_dir_traj,
                ckpt_index=ckpt_index,
                **config,
            )

            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

        if rank == 0:
            trajectories = []
            timestamps = []
            for i in range(num_samples):
                target_dir_traj = window_target_dir / f"tmp_{i}"
                trajectory = safe_load(target_dir_traj / "trajectory.pt")
                timestamp = safe_load(target_dir_traj / "timestamps.pt")
                trajectories.append(trajectory)
                timestamps.append(timestamp)

                diagnostics_path = target_dir_traj / "guidance_diagnostics.pth"
                if diagnostics_path.exists():
                    shutil.move(
                        diagnostics_path,
                        window_target_dir / f"guidance_diagnostics_{i}.pth",
                    )

                shutil.rmtree(target_dir_traj)
                del trajectory, timestamp
                gc.collect()

            trajectories = torch.cat(trajectories)
            timestamps = torch.cat(timestamps)
            trajectories = rearrange(trajectories, "(d s) ... -> d s ...", s=num_samples_per_date)
            timestamps = rearrange(timestamps, "(d s) ... -> d s ...", s=num_samples_per_date)

            safe_save(trajectories, window_target_dir / "trajectories.pt")
            safe_save(timestamps, window_target_dir / "timestamps.pt")

            print("Saved to", window_target_dir)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
