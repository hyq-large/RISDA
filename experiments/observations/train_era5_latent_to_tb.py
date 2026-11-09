r"""Train one instrument-specific direct E2O decoder.

The model maps a real native ERA latent directly to a dense, standardized
brightness-temperature field. Only valid channel values on the real satellite
swath contribute to the loss:

    era5.h5 latent -> dense base TB --+
                                      +-> 1x1 head -> standardized TB
                  per-pixel cos(satzen)
                                             | real per-channel swath mask
                                             v
                                      masked pixel-channel MSE

Training timestamps first reproduce the corresponding O2E encoder's exact
6-hour ERA/satellite intersection, then take the strict intersection with
``era5.h5``. No ERA physical field is loaded during training.
"""

import itertools
import numpy as np
import os
import sys
import time
import torch
import torch.distributed as dist
import wandb

from contextlib import nullcontext
from datetime import timedelta
from omegaconf import OmegaConf
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from appa.config.hydra import compose
from appa.optim import get_optimizer, safe_gd_step
from appa.save import safe_load, safe_save
from experiments.observations.direct_e2o_data import (
    DIRECT_E2O_DATA_CONTRACT,
    EraLatentSatelliteTbDataset,
    split_train_valid_indices,
    validate_direct_e2o_data_contract,
)
from risda.models.decoders import EraLatentToTbDecoder, masked_channel_sums


def precision_context(precision):
    precision = str(precision).lower()
    if precision in {"float32", "fp32"}:
        return nullcontext()
    if precision in {"float16", "fp16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision in {"bfloat16", "bf16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported precision: {precision!r}")


def build_dataset(cfg):
    cadence_hours = validate_direct_e2o_data_contract(cfg.data)
    return EraLatentSatelliteTbDataset(
        era_latent_path=Path(cfg.era_latent_dir) / str(cfg.data.latent_file),
        cadence_hours=cadence_hours,
        satellite_data_path=cfg.data.sate_data_path,
        satellite_name=cfg.data.sate_name,
        satellite_standard_dir=cfg.data.sate_standard_dir,
    )


def load_era_autoencoder_config(era_latent_dir):
    path = Path(era_latent_dir) / "ae" / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"ERA autoencoder config not found: {path}")
    config = OmegaConf.load(path).get("ae")
    if config is None:
        raise KeyError(f"{path} does not contain an 'ae' config")
    return config


def build_decoder(cfg, dataset, device):
    autoencoder = load_era_autoencoder_config(cfg.era_latent_dir)
    shape = tuple(int(value) for value in autoencoder.shape)
    latent_shape = tuple(int(value) for value in autoencoder.latent_shape)
    model = EraLatentToTbDecoder(
        out_channels=dataset.n_obs,
        lat_channels=latent_shape[2],
        shape=shape,
        resize=int(autoencoder.resize),
        zenith_hidden_channels=int(cfg.model.zenith_hidden_channels),
        hid_channels=tuple(cfg.model.hid_channels),
        hid_blocks=tuple(cfg.model.hid_blocks),
        kernel_size=int(cfg.model.kernel_size),
        ffn_factor=int(cfg.model.get("ffn_factor", 1)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        checkpointing=bool(cfg.model.get("checkpointing", False)),
        pixel_shuffle=bool(cfg.model.get("pixel_shuffle", True)),
        identity_init=bool(cfg.model.get("identity_init", True)),
    )
    expected = (
        model.latent_height * model.latent_width,
        model.lat_channels,
    )
    if tuple(dataset.latent_sample_shape) != expected:
        raise ValueError(
            f"era5.h5 latent shape {dataset.latent_sample_shape} does not match "
            f"the ERA AE/decoder shape {expected}"
        )
    return model.to(device)


def summarize_metrics(squared_error, count, channel_std):
    squared_error = np.asarray(squared_error, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    channel_std = np.asarray(channel_std, dtype=np.float64).reshape(-1)
    if squared_error.shape != count.shape or count.shape != channel_std.shape:
        raise ValueError("metric vectors must have the same channel shape")
    if count.sum() <= 0:
        raise RuntimeError("validation contains no finite satellite TB values")

    normalized_rmse = np.full_like(squared_error, np.nan)
    observed = count > 0
    normalized_rmse[observed] = np.sqrt(squared_error[observed] / count[observed])
    kelvin_rmse = normalized_rmse * channel_std
    return {
        "normalized_rmse": float(np.sqrt(squared_error.sum() / count.sum())),
        "mean_channel_normalized_rmse": float(np.nanmean(normalized_rmse)),
        "mean_channel_kelvin_rmse": float(np.nanmean(kelvin_rmse)),
        "per_channel_normalized_rmse": normalized_rmse.tolist(),
        "per_channel_kelvin_rmse": kelvin_rmse.tolist(),
        "per_channel_valid_count": count.astype(np.int64).tolist(),
    }


def evaluate(model, loader, device, precision, channel_std):
    model.eval()
    channel_count = int(np.asarray(channel_std).size)
    totals = torch.zeros((2, channel_count), device=device, dtype=torch.float64)
    with torch.no_grad():
        for latent, target, valid, cos_satzen, _date in loader:
            latent = latent.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            cos_satzen = cos_satzen.to(device, non_blocking=True)
            with precision_context(precision):
                prediction = model(latent, cos_satzen)
            squared_error, count = masked_channel_sums(prediction, target, valid)
            totals[0] += squared_error.to(dtype=torch.float64)
            totals[1] += count.to(dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return summarize_metrics(
        totals[0].cpu().numpy(),
        totals[1].cpu().numpy(),
        channel_std,
    )


def save_training_state(model, optimizer, scheduler, runpath, metadata):
    safe_save(model.state_dict(), runpath / "model_last.pth")
    safe_save(optimizer.state_dict(), runpath / "optimizer_last.pth")
    safe_save(scheduler.state_dict(), runpath / "scheduler_last.pth")
    OmegaConf.save(metadata, runpath / "metadata.yaml")


class _NullSync:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def main(cfg):
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    if is_main:
        print("[direct-e2o] building exact latent/OBS-aligned dataset...", flush=True)
    dataset = build_dataset(cfg)
    train_indices, valid_indices, split_name = split_train_valid_indices(
        dataset.dates,
    )
    train_set = Subset(dataset, train_indices.tolist())
    valid_set = Subset(dataset, valid_indices.tolist())
    rank_valid_set = Subset(valid_set, list(range(rank, len(valid_set), world_size)))
    if is_main:
        alignment = dataset.alignment
        print(
            f"[direct-e2o] instrument={dataset.satellite_name} "
            f"channels={dataset.n_obs} latent_shape={dataset.latent_sample_shape} "
            f"cadence={dataset.cadence_hours}h "
            f"contract={DIRECT_E2O_DATA_CONTRACT}",
            flush=True,
        )
        print(
            "[direct-e2o] timestamps: "
            f"latent={alignment['latent_samples']} "
            f"satellite={alignment['satellite_samples']} "
            f"aligned={alignment['aligned_samples']} "
            f"missing_latent={alignment['missing_from_latent_dump']} "
            f"missing_satellite={alignment['missing_from_satellite']}",
            flush=True,
        )
        print(
            f"[direct-e2o] train={len(train_set)} valid={len(valid_set)} " f"split={split_name}",
            flush=True,
        )

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader_kwargs = {
        "batch_size": int(cfg.train.batch_size),
        "num_workers": int(cfg.train.num_workers),
        "pin_memory": bool(cfg.train.get("pin_memory", True)),
        "persistent_workers": bool(
            int(cfg.train.num_workers) > 0 and cfg.train.get("persistent_workers", True)
        ),
    }
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        rank_valid_set,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = build_decoder(cfg, dataset, device)
    if is_main:
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"[direct-e2o] decoder parameters={parameter_count / 1e6:.2f}M",
            flush=True,
        )
    model = DistributedDataParallel(model, device_ids=[device])

    optimizer, scheduler = get_optimizer(
        params=model.parameters(),
        update_steps=int(cfg.train.update_steps),
        **cfg.optim,
    )
    effective_unit = int(cfg.train.batch_size) * world_size
    effective_batch = int(cfg.train.batch_size_per_step)
    if effective_batch % effective_unit:
        raise ValueError(
            "train.batch_size_per_step must be divisible by " "train.batch_size * world_size"
        )
    accumulation_steps = effective_batch // effective_unit
    precision = str(cfg.train.get("precision", "float16"))
    scaler = torch.amp.GradScaler("cuda", enabled=precision.lower() in {"float16", "fp16"})

    runpath = Path(cfg.output_dir) / "direct_e2o" / str(cfg.id)
    if is_main:
        runpath.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    start_step = 0
    best_rmse = float("inf")
    last_metrics = None
    metadata_path = runpath / "metadata.yaml"
    if metadata_path.exists() and (runpath / "model_last.pth").exists():
        model.module.load_state_dict(safe_load(runpath / "model_last.pth", map_location=device))
        optimizer.load_state_dict(safe_load(runpath / "optimizer_last.pth", map_location=device))
        scheduler.load_state_dict(safe_load(runpath / "scheduler_last.pth", map_location=device))
        metadata = OmegaConf.load(metadata_path)
        start_step = int(metadata.last_step_done) + 1
        best_rmse = float(metadata.get("best_valid_normalized_rmse", float("inf")))
        if is_main:
            print(
                f"[direct-e2o] resume step={start_step} best_rmse={best_rmse:.6g}",
                flush=True,
            )
    if is_main:
        OmegaConf.save(cfg, runpath / "config.yaml")

    run = None
    if is_main and (cfg.wandb_entity is not None or os.environ.get("WANDB_API_KEY")):
        run = wandb.init(
            project="risda_direct_e2o",
            name=str(cfg.id),
            id=str(cfg.id),
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )

    def cycling_batches():
        for epoch in itertools.count():
            train_sampler.set_epoch(epoch)
            yield from train_loader

    train_iterator = iter(cycling_batches())
    train_losses = []
    log_start = time.time()
    last_step_done = start_step - 1
    last_valid_step = -1
    channel_std = np.asarray(dataset.satellite.std, dtype=np.float64).reshape(-1)

    for step in range(start_step, int(cfg.train.update_steps)):
        model.train()
        for accumulation_index in range(accumulation_steps):
            latent, target, valid, cos_satzen, _date = next(train_iterator)
            latent = latent.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            cos_satzen = cos_satzen.to(device, non_blocking=True)
            synchronize = accumulation_index + 1 == accumulation_steps
            sync_context = _NullSync() if synchronize else model.no_sync()
            with sync_context:
                with precision_context(precision):
                    prediction = model(latent, cos_satzen)
                    squared_error, count = masked_channel_sums(prediction, target, valid)
                    loss = squared_error.sum() / count.sum().clamp_min(1.0)
                scaler.scale(loss / accumulation_steps).backward()
        safe_gd_step(
            optimizer,
            grad_clip=cfg.optim.grad_clip,
            scaler=scaler,
        )
        scheduler.step()
        train_losses.append(loss.detach())
        last_step_done = step

        if (step + 1) % int(cfg.train.log_interval) == 0:
            mean_loss = torch.stack(train_losses).mean()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size
            if is_main:
                elapsed = time.time() - log_start
                print(
                    f"step {step + 1}/{cfg.train.update_steps} "
                    f"loss={mean_loss.item():.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"{elapsed / len(train_losses):.2f}s/step",
                    flush=True,
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss": mean_loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=step + 1,
                    )
            train_losses = []
            log_start = time.time()

        if (step + 1) % int(cfg.train.valid_interval) == 0:
            last_metrics = evaluate(
                model.module,
                valid_loader,
                device,
                precision,
                channel_std,
            )
            last_valid_step = step
            if is_main:
                print(
                    f"  [valid] step={step + 1} "
                    f"normalized_rmse={last_metrics['normalized_rmse']:.5f} "
                    f"mean_channel_K_rmse={last_metrics['mean_channel_kelvin_rmse']:.3f}",
                    flush=True,
                )
                if run is not None:
                    log_values = {
                        "valid/normalized_rmse": last_metrics["normalized_rmse"],
                        "valid/mean_channel_normalized_rmse": last_metrics[
                            "mean_channel_normalized_rmse"
                        ],
                        "valid/mean_channel_kelvin_rmse": last_metrics["mean_channel_kelvin_rmse"],
                    }
                    for index, value in enumerate(
                        last_metrics["per_channel_normalized_rmse"], start=1
                    ):
                        log_values[f"valid/channel_{index:02d}_normalized_rmse"] = value
                    run.log(log_values, step=step + 1)
                if last_metrics["normalized_rmse"] < best_rmse:
                    best_rmse = last_metrics["normalized_rmse"]
                    safe_save(
                        model.module.state_dict(),
                        runpath / "model_best.pth",
                    )

        if (step + 1) % int(cfg.train.save_interval) == 0 and is_main:
            save_training_state(
                model.module,
                optimizer,
                scheduler,
                runpath,
                {
                    "runid": str(cfg.id),
                    "instrument": dataset.satellite_name,
                    "data_contract": DIRECT_E2O_DATA_CONTRACT,
                    "uses_cos_satzen": True,
                    "cadence_hours": dataset.cadence_hours,
                    "last_step_done": last_step_done,
                    "best_valid_normalized_rmse": best_rmse,
                    "last_valid_metrics": last_metrics,
                    "alignment": dataset.alignment,
                    "split": split_name,
                    "train_samples": len(train_set),
                    "valid_samples": len(valid_set),
                },
            )

    if last_valid_step != last_step_done:
        last_metrics = evaluate(
            model.module,
            valid_loader,
            device,
            precision,
            channel_std,
        )
        if is_main and last_metrics["normalized_rmse"] < best_rmse:
            best_rmse = last_metrics["normalized_rmse"]
            safe_save(model.module.state_dict(), runpath / "model_best.pth")

    if is_main:
        save_training_state(
            model.module,
            optimizer,
            scheduler,
            runpath,
            {
                "runid": str(cfg.id),
                "instrument": dataset.satellite_name,
                "data_contract": DIRECT_E2O_DATA_CONTRACT,
                "uses_cos_satzen": True,
                "cadence_hours": dataset.cadence_hours,
                "last_step_done": last_step_done,
                "best_valid_normalized_rmse": best_rmse,
                "last_valid_metrics": last_metrics,
                "alignment": dataset.alignment,
                "split": split_name,
                "train_samples": len(train_set),
                "valid_samples": len(valid_set),
            },
        )
        if run is not None:
            run.finish()
        print(
            f"[direct-e2o] done: {runpath} best_rmse={best_rmse:.6g}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    config = compose("configs/train_era5_latent_to_tb.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)
    if "id" not in config:
        config.id = wandb.util.generate_id()
    main(config)
