r"""Train ATMS-obs -> ERA5-latent inversion encoder (DDP, multi-GPU).

The REVERSE direction of train_e2s.py. Instead of pushing the ERA5 state toward
the satellite observation space, we learn a single encoder that maps the REAL
ATMS brightness-temperature observation frame DIRECTLY to the ERA5 autoencoder's
latent space, decode that latent through the FROZEN ERA5 decoder to a physical
ERA5 field, and supervise it against the ERA5 ground truth — but ONLY on the
pixels where an ATMS observation actually exists.

    obs (23/24ch pixel) --[new ConvEncoder, trained]--> z_era_hat (native latent)
        --[frozen ERA5 decoder]-->  x_hat (69ch normalized field)
        ----- masked MSE vs ERA5 truth, mask = obs swath -----> loss

This is an observation->state direct regressor: no diffusion, no MMPS. It answers
"how well can a network reconstruct the ERA5 state from sparse ATMS obs alone",
as a baseline for the latent diffusion assimilation.

Design notes
------------
* The new encoder reuses `appa.nn.cae.conv_ae`'s ConvEncoder (single-branch,
  channel_groups=None), input channels = 23 = obs(22) | mask(1), or 24 with
  cos_satzen appended when data.use_satzen=true — the exact frame
  `SatelliteAEDataset` yields. Unlike the sate AE line, o2e KEEPS the mask channel
  in the encoder input (it decodes the full ERA field, so mask is not a leak);
  satzen is a pure extra geometry input, nothing is dropped.
  resize=32 lands 720x1440 -> 23x47, latent channels 128 — the SAME latent shape
  the frozen ERA5 decoder expects.
* The encoder output is the NATIVE (softclip2-saturated) latent, the same space
  the ERA5 decoder was trained on, so it is decoded directly with no
  de-standardization (stats.pth is not needed on the forward path).
* ERA5 ground truth comes from ERA5DataNpy_2017_2025 in NORMALIZED space, the
  same space the decoder outputs, so the two are directly comparable.

Launch:
    cd experiments/observations
    torchrun --nproc_per_node=8 train_obs2era.py \
        era_latent_dir=/.../ae/<era_runid>/latents/<dump_id> \
        +id=<run_name>
"""

import itertools
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb

from datetime import timedelta
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset

from appa.config.hydra import compose
from appa.data.datasets import (
    ERA5DataNpy_2016_2025_1h,
    ERA5DataNpy_2017_2025,
    SatelliteAEDataset,
)
from appa.date import get_date_features
from appa.optim import get_optimizer, safe_gd_step
from appa.save import load_auto_encoder, safe_load, safe_save
from experiments.observations.convi_obs2era_data import (
    CONVI_QC_TO_ERA5,
    CONVI_QC_TO_ERA5_QC_MASK,
    Convi2EraPairDataset,
)
from experiments.observations.dense_era_surface_obs2era_data import (
    DenseEraSurface2EraDataset,
)
from risda.models.encoders import ObsToEraEncoder, resolve_surface_preprocessing


# =====================================================================
# Dataset: pair real ATMS obs frames with ERA5 ground-truth fields
# =====================================================================
class Atms2EraPairDataset(Dataset):
    """Pairs an ATMS observation frame (SatelliteAEDataset, 23/24ch) with the
    ERA5 ground-truth field (ERA5DataNpy_2017_2025, 69ch normalized) at the
    SAME timestamp.

    Returns per-sample:
        obs:    (C, H, W)    [obs(22) | mask(1)] normalized obs space,
                             (+ cos_satzen(1) appended => 24ch when use_satzen).
        era_gt: (69, H, W)   ERA5 truth, normalized (== ERA decoder output space)
        mask:   (1,  H, W)   0/1, 1 where any ATMS obs exists (== obs[n_obs])
        date:   (4,)         date features (year, month, day, hour)

    The ATMS .nc reads happen on the fly inside SatelliteAEDataset; ERA frames
    are mmap'd npy loads. Both datasets are constructed once; pairing is by exact
    (year, month, day, hour) timestamp match.
    """

    def __init__(
        self,
        era_data_path,
        sate_data_path,
        sate_name,
        sate_standard_dir,
        use_satzen=False,
        era_cadence_hours=6,
        era_normalization=None,
    ):
        if era_cadence_hours not in (1, 6):
            raise ValueError("era_cadence_hours must be 1 or 6")
        if era_cadence_hours == 6 and era_normalization:
            raise ValueError("era_normalization requires era_cadence_hours=1")
        self.sate = SatelliteAEDataset(
            data_path=sate_data_path,
            sate_name=sate_name,
            standard_dir=sate_standard_dir,
            blanket_size=1,
            time_range=None,
            use_satzen=bool(use_satzen),
        )
        self.n_obs = self.sate.obs_channel  # 22 for ATMS (mask sits at n_obs)

        # Narrow the ERA npy walk to the ATMS time span (ERA5DataNpy_2017_2025
        # globs the disk on construction, so bounding it is much faster).
        sate_times = list(self.sate.times)
        if len(sate_times) == 0:
            raise RuntimeError("SatelliteAEDataset found no timestamps.")
        t0, t1 = min(sate_times), max(sate_times)
        start = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
        if era_cadence_hours == 1:
            start += f"T{t0.hour:02d}"
        end = f"{t1.year:04d}{t1.month:02d}{t1.day:02d}T{t1.hour:02d}"
        era_dataset_cls = (
            ERA5DataNpy_2016_2025_1h
            if era_cadence_hours == 1
            else ERA5DataNpy_2017_2025
        )
        self.era = era_dataset_cls(
            data_path=era_data_path, blanket_size=1, time_range=[start, end],
            **(dict(era_normalization or {}) if era_cadence_hours == 1 else {}),
        )

        # Build (y, m, d, h) -> index maps and intersect.
        def key(t):
            return (int(t.year), int(t.month), int(t.day), int(t.hour))

        era_key_to_idx = {}
        for i, t in enumerate(self.era.times):
            # self.era.times entries are np.datetime64 / pd.Timestamp-like.
            import pandas as pd
            ts = pd.Timestamp(t)
            era_key_to_idx[(ts.year, ts.month, ts.day, ts.hour)] = i

        pairs = []
        dates = []
        for j, t in enumerate(self.sate.times):
            k = key(t)
            i = era_key_to_idx.get(k)
            if i is not None:
                pairs.append((i, j))
                dates.append(k)
        if not pairs:
            raise RuntimeError("No overlapping timestamps between ERA5 npy and ATMS .nc.")

        self.pairs = np.asarray(pairs, dtype=np.int64)  # (P, 2): (era_idx, sate_idx)
        self.dates = np.asarray(dates, dtype=np.int64)  # (P, 4)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i_era, j_sate = self.pairs[idx]

        merged, _ = self.sate[int(j_sate)]      # (1, 23/24, H, W)
        obs = merged[0].float()                 # (23/24, H, W)
        mask = obs[self.n_obs : self.n_obs + 1] # (1, H, W) — 0/1 obs presence

        era_frame, _ = self.era[int(i_era)]     # (1, 69, H, W) normalized
        era_gt = era_frame[0].float()           # (69, H, W)

        r = self.dates[idx]
        date = get_date_features(
            np.datetime64(f"{r[0]:04d}-{r[1]:02d}-{r[2]:02d}T{r[3]:02d}")
        )
        return obs, era_gt, mask, date


# =====================================================================
# Model: ATMS obs (23/24ch pixel) -> ERA native latent
# =====================================================================

def _pixel_mask_to_latent_frac(mask_pix, resize):
    """Downsample a 0/1 pixel obs mask onto the ERA latent grid, returning the
    observed FRACTION of each resize x resize block.

    mask_pix: (B, 1, H, W) float {0,1}.  Returns (B, HW_lat, 1) float in [0,1].

    Uses the SAME pad recipe as ObsToEraEncoder._pad / reanalysis_obs2era's
    _obs_mask_to_frac (circular-pad longitude, zero-pad latitude, then avg_pool),
    so the resulting latent mask is aligned pixel-for-pixel with the latents the
    encoder actually produces and with the mask used at assimilation time.
    """
    H, W = 720, 1440
    f = int(resize)
    pad_h = (-H) % f
    pad_w = (-W) % f + 2 * f
    if pad_w > 0:
        mask_pix = F.pad(mask_pix, (pad_w // 2, pad_w - pad_w // 2, 0, 0), mode="circular")
    if pad_h > 0:
        mask_pix = F.pad(mask_pix, (0, 0, pad_h // 2, pad_h - pad_h // 2), mode="constant", value=0)
    frac = F.avg_pool2d(mask_pix, kernel_size=(f, f))     # (B, 1, H_lat, W_lat)
    return rearrange(frac, "b c h w -> b (h w) c")         # (B, HW_lat, 1)


def _frac_to_latent_mask(frac, latent_mask_frac=0.0):
    """Threshold an observed-fraction tensor into a 0/1 latent mask, matching
    reanalysis_obs2era._frac_to_mask:
        <= 0  -> any observed pixel in the block (loosest)
        == 1  -> the whole block observed (strictest)
    """
    if latent_mask_frac <= 0.0:
        return (frac > 0.0).float()
    return (frac >= (latent_mask_frac - 1e-6)).float()


def _masked_physical_mse(prediction, target, mask):
    """Masked MSE for either a shared spatial mask or a channelwise mask."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must share shape (B,HW,C); got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if mask.ndim != 3 or mask.shape[:2] != prediction.shape[:2]:
        raise ValueError(
            "physical mask must have shape (B,HW,1|C); got "
            f"{tuple(mask.shape)} for prediction {tuple(prediction.shape)}"
        )
    squared_error = (prediction - target).square()
    if mask.shape[-1] == 1:
        # Preserve the historical loss scale: average all output channels at
        # each selected pixel, then average selected pixels.
        squared_error = squared_error.mean(dim=-1, keepdim=True)
    elif mask.shape[-1] != prediction.shape[-1]:
        raise ValueError(
            "physical mask channels must be 1 or match the target; got "
            f"{mask.shape[-1]} and {prediction.shape[-1]}"
        )
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def main(
    cfg,
    convi_pair_dataset_factory=Convi2EraPairDataset,
):
    # --- DDP init ---
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    era_dir = Path(cfg.era_latent_dir)

    # --- Dataset ---
    if is_main:
        print("[obs2era] dist ready; building paired dataset...", flush=True)
    data_kind = str(cfg.data.get("kind", "sate")).lower()
    if data_kind == "era_surface_dense":
        full = DenseEraSurface2EraDataset(
            era_data_path=cfg.data.era_data_path,
            era_normalization=OmegaConf.to_container(
                cfg.data.era_normalization, resolve=True, enum_to_str=True
            ),
            time_range=OmegaConf.to_container(
                cfg.data.get("time_range"), resolve=True, enum_to_str=True
            ),
        )
    elif data_kind == "convi":
        full = convi_pair_dataset_factory(
            era_data_path=cfg.data.era_data_path,
            surf_dir=cfg.data.surf_dir,
            icoa_dir=cfg.data.icoa_dir,
            mean_std_dir=cfg.data.mean_std_dir,
            surface_preprocessing=resolve_surface_preprocessing(cfg),
            alignment_mode=cfg.data.get(
                "alignment_mode",
                CONVI_QC_TO_ERA5_QC_MASK,
            ),
        )
    elif data_kind == "sate":
        full = Atms2EraPairDataset(
            era_data_path=cfg.data.era_data_path,
            sate_data_path=cfg.data.sate_data_path,
            sate_name=cfg.data.sate_name,
            sate_standard_dir=cfg.data.sate_standard_dir,
            use_satzen=bool(cfg.data.get("use_satzen", False)),
            era_cadence_hours=int(cfg.data.get("era_cadence_hours", 6)),
            era_normalization=OmegaConf.to_container(
                cfg.data.era_normalization, resolve=True
            ) if cfg.data.get("era_normalization") else None,
        )
    else:
        raise ValueError(f"unsupported obs2era data.kind: {data_kind!r}")
    n = len(full)
    if is_main:
        print(f"[obs2era] world_size={world_size}", flush=True)
        print(f"[obs2era] era npy:   {cfg.data.era_data_path}", flush=True)
        if data_kind == "era_surface_dense":
            print(
                "[obs2era] input=dense ERA [t2m,u10,v10,msl | all-one mask] "
                f"in decoder normalization, samples={n}",
                flush=True,
            )
        elif data_kind == "convi":
            print(f"[obs2era] CONVI SURF: {cfg.data.surf_dir}", flush=True)
            print(
                f"[obs2era] alignment_mode={full.alignment_mode}, "
                f"paired samples={n}",
                flush=True,
            )
        else:
            print(f"[obs2era] sate .nc:  {cfg.data.sate_data_path}", flush=True)
            print(f"[obs2era] n_obs={full.n_obs}, paired samples={n}", flush=True)

    # --- Train before 2025; validate only January-March 2025. ---
    valid_year = 2025
    years = full.dates[:, 0].astype(np.int64)
    months = full.dates[:, 1].astype(np.int64)
    vmask = (
        (years == valid_year)
        & (months >= 1)
        & (months <= 3)
    )
    if not vmask.any():
        raise RuntimeError(f"Q1 {valid_year} contains no validation samples")
    train_indices = np.where(years < valid_year)[0]
    valid_indices = np.where(vmask)[0]
    if not len(train_indices):
        raise RuntimeError(f"valid_year={valid_year} leaves no training samples")

    train_set = Subset(full, train_indices.tolist())
    valid_set = Subset(full, valid_indices.tolist())
    if is_main:
        kind = (
            f"{valid_year}-01-01..{valid_year}-03-31"
        )
        print(
            f"[obs2era] train={len(train_indices)}, valid={len(valid_indices)} (split: {kind})",
            flush=True,
        )

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    valid_sampler = DistributedSampler(
        valid_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
        sampler=train_sampler, drop_last=True, persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
        sampler=valid_sampler, drop_last=False, persistent_workers=False,
    )

    # --- ERA latent shape (from the AE config in the dump dir) ---
    with open(era_dir / "ae" / "config.yaml") as f:
        era_ae_cfg = OmegaConf.load(f).get("ae")
    era_lat = tuple(era_ae_cfg.latent_shape)   # (H_e, W_e, C_e)
    lat_channels = int(era_lat[2])
    if is_main:
        print(f"[obs2era] era latent shape = {era_lat} (lat_channels={lat_channels})", flush=True)

    # --- New observation encoder -> ERA native latent ---
    surface_preprocessing = (
        resolve_surface_preprocessing(cfg) if data_kind == "convi" else None
    )

    if data_kind in {"convi", "era_surface_dense"}:
        in_channels = full.input_channels
        if is_main:
            if surface_preprocessing is not None:
                description = (
                    "raw=8 [4 values | 4 masks] -> Cressman -> learned=9 "
                    "[4 interpolated values | 4 dilated masks | confidence]"
                )
            else:
                mask_kind = "all-one" if data_kind == "era_surface_dense" else "union"
                description = f"in_channels=5 (4 values | {mask_kind} mask)"
            print(f"[obs2era] encoder input: {description}", flush=True)
    else:
        in_channels = full.sate.get_dim()
        if is_main:
            print(
                f"[obs2era] use_satzen={full.sate.use_satzen}  encoder in_channels="
                f"{in_channels} (obs {full.n_obs} | mask"
                + (" | cos_satzen" if full.sate.use_satzen else "")
                + ")",
                flush=True,
            )
    model = ObsToEraEncoder(
        in_channels=in_channels,
        lat_channels=lat_channels,
        shape=(720, 1440),
        resize=int(era_ae_cfg.resize),
        hid_channels=tuple(cfg.model.hid_channels),
        hid_blocks=tuple(cfg.model.hid_blocks),
        kernel_size=cfg.model.kernel_size,
        ffn_factor=cfg.model.ffn_factor,
        dropout=cfg.model.dropout,
        surface_preprocessing=surface_preprocessing,
        checkpointing=bool(cfg.model.get("checkpointing", False)),
        pixel_shuffle=bool(cfg.model.get("pixel_shuffle", True)),
        identity_init=bool(cfg.model.get("identity_init", True)),
    ).to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[obs2era] encoder params = {n_params/1e6:.2f} M", flush=True)
    model = DistributedDataParallel(model, device_ids=[device])

    # --- Frozen ERA5 decoder ---
    if is_main:
        print("[obs2era] loading frozen ERA5 decoder (341M ckpt)...", flush=True)
    era_ae = load_auto_encoder(
        era_dir / "ae", model_name="model", device=str(device), eval_mode=True,
    )
    era_ae.to(device).eval()
    for p in era_ae.parameters():
        p.requires_grad_(False)

    # --- Loss config (physical field loss + optional latent-space masked L2) ---
    loss_cfg = cfg.get("loss") or {}
    physical_weight = float(loss_cfg.get("physical_weight", 1.0))
    latent_weight = float(loss_cfg.get("latent_weight", 0.0))
    latent_mask_frac = float(loss_cfg.get("latent_mask_frac", 0.0))
    latent_standardize = bool(loss_cfg.get("latent_standardize", False))
    era_resize = int(era_ae_cfg.resize)
    if (
        data_kind == "convi"
        and full.alignment_mode != CONVI_QC_TO_ERA5
        and latent_weight != 0.0
    ):
        raise ValueError(
            "aligned CONVI QC/ERA-at-QC training is surface-only and requires "
            "loss.latent_weight=0"
        )

    # Latent standardization stats (only needed when the latent term is on AND
    # standardization is requested). Native latent = softclip2-saturated space
    # shared by ObsToEraEncoder and era_ae.encode; stats.pth lives in the dump dir.
    lat_mean = lat_std = None
    if latent_weight > 0.0 and latent_standardize:
        stats_path = era_dir / "stats.pth"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"loss.latent_standardize=true but {stats_path} is missing; "
                "set latent_standardize=false or point era_latent_dir at a dump with stats.pth."
            )
        stats_d = safe_load(stats_path, map_location="cpu")
        lat_mean = stats_d["mean"].to(device)
        lat_std = stats_d["std"].to(device)
    if is_main:
        print(
            f"[obs2era] loss: physical_weight={physical_weight} latent_weight={latent_weight}"
            + (f" (latent_mask_frac={latent_mask_frac}, standardize={latent_standardize})"
               if latent_weight > 0.0 else " (latent term OFF)"),
            flush=True,
        )

    optimizer, scheduler = get_optimizer(
        params=model.parameters(),
        update_steps=cfg.train.update_steps,
        **cfg.optim,
    )

    # --- Output dir + resume ---
    output_group = str(cfg.get("output_group", "obs2era"))
    runpath = Path(cfg.output_dir) / output_group / cfg.id
    if is_main:
        runpath.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    start_step = 0
    best_val_loss = float("inf")
    resume_meta = runpath / "metadata.yaml"
    resume_model = runpath / "model_last.pth"
    if resume_meta.exists() and resume_model.exists():
        if is_main:
            print(f"[resume] loading from {runpath}", flush=True)
        for obj, name in zip(
            (model.module, optimizer, scheduler),
            ("model", "optimizer", "scheduler"),
            strict=True,
        ):
            obj.load_state_dict(safe_load(runpath / f"{name}_last.pth", map_location=device))
        meta = OmegaConf.load(resume_meta)
        start_step = int(meta["last_step_done"]) + 1
        best_val_loss = float(meta["best_val_loss"])
        if is_main:
            print(f"[resume] step={start_step}, best_val={best_val_loss:.6g}", flush=True)

    if is_main:
        OmegaConf.save(cfg, runpath / "config.yaml")

    # --- Wandb (rank 0) ---
    run = None
    if is_main and (cfg.wandb_entity is not None or os.environ.get("WANDB_API_KEY")):
        run = wandb.init(
            project="appa_obs2era", name=str(cfg.id), id=str(cfg.id),
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True), resume="allow",
        )

    # --- Training setup ---
    scaler = torch.amp.grad_scaler.GradScaler("cuda")
    grad_acc = max(1, cfg.train.batch_size_per_step // (cfg.train.batch_size * world_size))
    dummy_date = torch.zeros(cfg.train.batch_size, 4, device=device, dtype=torch.long)

    def cycle(loader, sampler):
        for epoch in itertools.count():
            sampler.set_epoch(epoch)
            for b in loader:
                yield b

    train_iter = cycle(train_loader, train_sampler)
    if is_main:
        print("[obs2era] entering training loop...", flush=True)

    def combined_loss(obs, physical_target, mask):
        """Encode input and apply the configured physical/latent supervision.

        Encode obs -> latent once, then:
          * PHYSICAL term: decode through the frozen ERA AE, then use either a
            shared (B,1,H,W) mask or a per-channel (B,69,H,W) mask.
          * LATENT term (only if latent_weight>0): masked L2 between the predicted
            latent and the ground-truth ERA latent (era_ae.encode(gt), frozen), on
            the obs mask downsampled to the latent grid.

        Returns (total, phys_detached, lat_detached) so components can be logged.
        """
        z_hat = model(obs)                                # (B, HW_lat, C_lat) native
        bs = z_hat.shape[0]
        date = dummy_date[:bs]

        B, C, H, W = physical_target.shape
        gt = physical_target.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # --- physical-space masked field loss ---
        x_hat = era_ae.decode(z_hat, date, None)          # (B, HW, 69) normalized
        m = mask.permute(0, 2, 3, 1).reshape(B, H * W, mask.shape[1])
        phys = _masked_physical_mse(x_hat, gt, m)

        total = physical_weight * phys
        lat = torch.zeros((), device=z_hat.device)

        # --- latent-space masked L2 (optional) ---
        if latent_weight > 0.0:
            # Ground-truth latent: encode the ERA5 GT field through the SAME frozen
            # ERA AE. Non-VAE + eval => deterministic mu, matching the dumped latents
            # and the space z_hat lives in. No grad flows into the target.
            with torch.no_grad():
                z_true = era_ae.encode(gt, date, None)    # (B, HW_lat, C_lat) native
            if latent_standardize:
                z_hat_c = (z_hat - lat_mean) / lat_std.clamp_min(1e-8)
                z_true_c = (z_true - lat_mean) / lat_std.clamp_min(1e-8)
            else:
                z_hat_c, z_true_c = z_hat, z_true
            lat_err = (z_hat_c - z_true_c.detach()).square()       # (B, HW_lat, C_lat)
            lat_per_pix = lat_err.mean(dim=-1, keepdim=True)       # (B, HW_lat, 1)
            latent_pixel_mask = (
                mask
                if mask.shape[1] == 1
                else mask.any(dim=1, keepdim=True).to(mask.dtype)
            )
            m_lat = _frac_to_latent_mask(
                _pixel_mask_to_latent_frac(latent_pixel_mask, era_resize),
                latent_mask_frac,
            )                                                      # (B, HW_lat, 1)
            lat = (lat_per_pix * m_lat).sum() / m_lat.sum().clamp_min(1.0)
            total = total + latent_weight * lat

        return total, phys.detach(), lat.detach()

    def run_batch(batch):
        obs, physical_target, mask, _date = batch
        obs = obs.to(device, non_blocking=True)
        physical_target = physical_target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        return combined_loss(obs, physical_target, mask)

    # --- Training loop ---
    log_t = time.time()
    train_losses = []
    train_phys = []
    train_lat = []
    for step in range(start_step, cfg.train.update_steps):
        model.train()
        for acc in range(grad_acc):
            batch = next(train_iter)
            sync_now = (acc + 1 == grad_acc)
            sync_ctx = model.no_sync() if not sync_now else _nullctx()
            with sync_ctx:
                loss, phys, lat = run_batch(batch)
                scaler.scale(loss / grad_acc).backward()
        safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip, scaler=scaler)
        scheduler.step()
        train_losses.append(loss.detach())
        train_phys.append(phys)
        train_lat.append(lat)

        if (step + 1) % cfg.train.log_interval == 0:
            stats_t = torch.stack([
                torch.stack(train_losses).mean(),
                torch.stack(train_phys).mean(),
                torch.stack(train_lat).mean(),
            ])
            dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
            stats_t = stats_t / world_size
            if is_main:
                dt = time.time() - log_t
                mean_loss, mean_phys, mean_lat = (x.item() for x in stats_t)
                print(
                    f"step {step+1}/{cfg.train.update_steps}  loss={mean_loss:.4f}  "
                    f"(phys={mean_phys:.4f} lat={mean_lat:.4f})  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"{dt/cfg.train.log_interval:.2f}s/step",
                    flush=True,
                )
                if run is not None:
                    run.log({
                        "train/loss": mean_loss,
                        "train/loss_physical": mean_phys,
                        "train/loss_latent": mean_lat,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/step_time": dt / cfg.train.log_interval,
                    }, step=step + 1)
                log_t = time.time()
            train_losses = []
            train_phys = []
            train_lat = []

        if (step + 1) % cfg.train.valid_interval == 0:
            model.eval()
            # [sum_total, sum_phys, sum_lat, count]
            local = torch.zeros(4, device=device)
            with torch.no_grad():
                for batch in valid_loader:
                    v_total, v_phys, v_lat = run_batch(batch)
                    local[0] += v_total.float()
                    local[1] += v_phys.float()
                    local[2] += v_lat.float()
                    local[3] += 1.0
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            n = local[3].clamp_min(1.0)
            v_mean, v_phys, v_lat = (local[0] / n).item(), (local[1] / n).item(), (local[2] / n).item()
            if is_main:
                print(
                    f"  [valid] step {step+1}  loss={v_mean:.4f} "
                    f"(phys={v_phys:.4f} lat={v_lat:.4f})",
                    flush=True,
                )
                if run is not None:
                    run.log({
                        "valid/loss": v_mean,
                        "valid/loss_physical": v_phys,
                        "valid/loss_latent": v_lat,
                    }, step=step + 1)
                if v_mean < best_val_loss:
                    best_val_loss = v_mean
                    safe_save(model.module.state_dict(), runpath / "model_best.pth")

        if (step + 1) % cfg.train.save_interval == 0 and is_main:
            for obj, name in zip(
                (model.module, optimizer, scheduler),
                ("model", "optimizer", "scheduler"),
                strict=True,
            ):
                safe_save(obj.state_dict(), runpath / f"{name}_last.pth")
            with open(runpath / "metadata.yaml", "w") as f:
                OmegaConf.save(
                    {"last_step_done": step, "best_val_loss": best_val_loss, "runid": str(cfg.id)},
                    f,
                )

    if is_main and run is not None:
        run.finish()
    if is_main:
        print("[obs2era] done.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    cfg = compose("configs/train_obs2era.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)
    if "id" not in cfg:
        cfg.id = wandb.util.generate_id()
    main(cfg)
