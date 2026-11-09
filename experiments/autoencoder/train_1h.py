r"""Autoencoder training script."""

import argparse
import cloudpickle
import dask
import dawgz
import itertools
import os
import re
import wandb

from functools import partial
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path

import appa

from appa.config import PATH_AE, PATH_ERA5, PATH_STAT
from appa.config import OUTPUT_ROOT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    ERA5_ATMOSPHERIC_VARIABLES,
    ERA5_PRESSURE_LEVELS,
    ERA5_SURFACE_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset, ERADataNpy, ERA5DataNpy_2016_2025_1h
from appa.data.transforms import StandardizeTransform
from appa.diagnostics.plots import plot_image_grid
from appa.losses import AELoss
from appa.optim import get_optimizer, safe_gd_step
from appa.save import safe_load, safe_save, select_ae_architecture

cloudpickle.register_pickle_by_value(appa)


def train(
    runid: str,
    cfg: DictConfig,
    fork_lap: int = 0,
    lap: int = 0,
    output_dir: str = None,
):
    r"""Train for one lap an autoencoder.

    Args:
        runid: The run id.
        cfg: The training configuration as a DictConfig.
        fork_lap: If starting from a checkpoint, the lap to fork from.
        lap: The current lap number.
    """
    import matplotlib.pyplot as plt
    import os
    import time
    import torch
    import torch.distributed as dist

    from datetime import timedelta
    from einops import rearrange
    from math import ceil
    from torch.amp.grad_scaler import GradScaler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import trange

    # DDP
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    rank = dist.get_rank()
    print(f'RANK {rank}')
    world_size = dist.get_world_size()
    num_nodes = ceil(world_size / torch.cuda.device_count())
    num_cpu = len(os.sched_getaffinity(0)) * num_nodes
    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    dask.config.set(scheduler="synchronous")

    torch.set_float32_matmul_precision("high")

    assert (
        cfg.train.batch_size_per_step % (cfg.train.batch_element_per_gpu * world_size) == 0
    ), "Batch size must be divisible by number of GPU * samples per GPU"

    # Data
    n_valid_samples = cfg.valid.n_valid_samples
    n_snapshots = cfg.valid.n_valid_snapshots
    lnum_samples = [
        None,
        # Pad the number of samples to be divisible by the world size
        n_valid_samples + (-n_valid_samples) % world_size,
        n_snapshots + (-n_snapshots) % world_size,
    ]

    splits = [
        DATASET_DATES_TRAINING,
        DATASET_DATES_VALIDATION,
        DATASET_DATES_VALIDATION,
    ]

    pressure_levels = (
        SUB_PRESSURE_LEVELS if cfg.train.sub_pressure_levels else ERA5_PRESSURE_LEVELS
    )

    # For v1 datasets (ERADataNpy with 13 levels), always use SUB_PRESSURE_LEVELS for loss weighting
    if cfg.loss.level_weighting == "v1":
        pressure_levels = SUB_PRESSURE_LEVELS

    datasets = [
        ERA5DataNpy_2016_2025_1h(
            data_path=cfg.data_path,
            blanket_size=cfg.train.trajectory_size,
            time_range=time_range,
            **{
                name: cfg.normalization[name]
                for name in (
                    "source_mean_path",
                    "source_std_path",
                    "target_mean_path",
                    "target_std_path",
                )
            },
        )
        for time_range in (
            ["2017-01-01T00", "2024-12-31T23"],
            ["2025-01-01T00", "2025-03-31T23"],
            ["2025-01-01T00", "2025-03-31T23"],
        )
    ]

    # Validation is always non-shuffled
    shuffles = [cfg.train.shuffle_training, False, False]

    train_loader, valid_quant_loader, valid_quali_loader = (
        get_dataloader(
            dataset=dataset,
            batch_size=cfg.train.batch_element_per_gpu,
            num_workers=4,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
            persistent_workers=True,
            drop_last=True
        )
        for dataset, shuffle in zip(datasets, shuffles)
    )

    # Model
    with open_dict(cfg):
        num_surface_variables = len(ERA5_SURFACE_VARIABLES) if ERA5_SURFACE_VARIABLES else 0
        num_atmospheric_variables = (
            (len(ERA5_ATMOSPHERIC_VARIABLES) * len(pressure_levels))
            if ERA5_ATMOSPHERIC_VARIABLES and pressure_levels
            else 0
        )
        num_context_variables = len(CONTEXT_VARIABLES) if CONTEXT_VARIABLES else 0
        num_context_variables = 0
        cfg.ae["in_channels"] = len(datasets[0].target_chn)
        cfg.ae["context_channels"] = num_context_variables

    ae_arch = select_ae_architecture(cfg.ae.name)
    ae = ae_arch(**cfg.ae).to(device)

    latent_shape = ae.latent_shape

    ae = DistributedDataParallel(
        ae,
        device_ids=[device],
    )

    optimizer, scheduler = get_optimizer(
        params=ae.parameters(),
        update_steps=cfg.train.update_steps,
        **cfg.optim,
    )
    PATH = Path(output_dir or OUTPUT_ROOT) / "ae"
    runpath = PATH / f"{runid}"
    if rank == 0:
        runpath.mkdir(parents=True, exist_ok=True)

    # In-place resume: if runpath already has a model_last.pth + metadata.yaml,
    # load them and continue from last_step_done + 1. Triggered by reusing the
    # same runid (e.g. via run_pipeline_v4*.sh --ae-id <runid>).
    resume_meta = runpath / "metadata.yaml"
    resume_model = runpath / "model_last.pth"
    if resume_meta.exists() and resume_model.exists():
        if rank == 0:
            print(f"[resume] found checkpoint in {runpath}, loading...", flush=True)
        for obj, name in zip(
            (ae.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
        ):
            ckpt_path = runpath / f"{name}_last.pth"
            ckpt = safe_load(ckpt_path, map_location=device)
            if name == "model":
                obj.load_state_dict(ckpt, strict=False)
            else:
                obj.load_state_dict(ckpt)

        with open(resume_meta, "r") as f:
            metadata = OmegaConf.load(f)
            start_step = int(metadata["last_step_done"]) + 1
            best_val_loss = float(metadata["best_val_loss"])
        if rank == 0:
            print(f"[resume] continuing from step {start_step}, best_val_loss={best_val_loss:.6g}", flush=True)
    elif lap > 0:
        prev_runpath = PATH / f"{lap - 1}"

        if not prev_runpath.exists() or not (prev_runpath / "metadata.yaml").exists():
            print("Previous lap does not exist or ran for too short. Exiting.")

            dist.destroy_process_group()
            return

        for obj, name in zip(
            (ae.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
        ):
            ckpt_path = prev_runpath / f"{name}_last.pth"
            ckpt = safe_load(ckpt_path, map_location=device)

            if name == "model":
                obj.load_state_dict(ckpt, strict=False)
            else:
                obj.load_state_dict(ckpt)

        # Metadata json
        with open(prev_runpath / "metadata.yaml", "r") as f:
            metadata = OmegaConf.load(f)
            start_step = metadata["last_step_done"] + 1
            best_val_loss = float(metadata["best_val_loss"])
    else:
        start_step = 0
        best_val_loss = float("inf")

    if start_step >= cfg.train.update_steps:
        dist.destroy_process_group()
        return

    # Logging
    if rank == 0:
        wandb_config = OmegaConf.to_container(cfg).copy()

        # Additional config fields
        wandb_config["num_model_params"] = sum(p.numel() for p in ae.parameters())
        wandb_config["path"] = runpath
        wandb_config["hardware"] = {
            "num_nodes": num_nodes,
            "num_gpus": world_size,
            "num_cpus": num_cpu,
        }
        wandb_config["ae"]["latent_shape"] = latent_shape

        # Run name format
        run_name = f"{runid} {cfg.ae.name}"

        description = f"AE {cfg.ae.name} " + ae.module.description()

        if "forked_from" in cfg:
            description = f"Forked from {cfg.forked_from}.\n" + description

        wandb_args = dict(
            name=run_name,
            project="appa_ae",
            entity=cfg.wandb_entity,
            config=wandb_config,
            group=f"train_{runid}",
            resume="allow",
            notes=description,
        )

        # WAITING FOR DEPLOYEMENT OF FORK_FROM IN WANDB
        # if cfg.fork_from and lap == fork_lap:
        #     run = wandb.init(fork_from=f"{runid}?step={start_step}", **wandb_args)
        #     runid = run.id

        run = wandb.init(id=runid, **wandb_args)

        OmegaConf.save(wandb_config, runpath / "config.yaml")

        # Kill switch
        open(runpath / ".running", "a").close()

    scaler = GradScaler("cuda")
    precision = getattr(torch, cfg.train.precision)

    # Load channel weights if provided (from prior reconstruction evaluation)
    channel_weights = None
    if hasattr(cfg.loss, "channel_weights_path") and cfg.loss.channel_weights_path:
        import h5py
        with h5py.File(cfg.loss.channel_weights_path, "r") as f:
            mse_norm = torch.from_numpy(f["mse_normalized"][:]).mean(dim=0)  # [T, C]
            mse_per_channel = mse_norm[0]  # [C], take first timestep
        channel_weights = mse_per_channel.mean() / mse_per_channel  # mean(MSE) / MSE_c
        channel_weights = channel_weights / channel_weights.mean()  # normalize so mean weight = 1
        if rank == 0:
            print(f"Channel weights loaded from {cfg.loss.channel_weights_path}")
            print(f"  min={channel_weights.min().item():.3f}, max={channel_weights.max().item():.3f}")

    model_loss = AELoss(
        criterion=cfg.loss.error,
        latitude_weighting=cfg.loss.latitude_weighting,
        level_weighting=cfg.loss.level_weighting,
        levels=pressure_levels,
        N_lat=cfg.loss.N_lat,
        N_lon=cfg.loss.N_lon,
        channel_weights=channel_weights,
        balance_channels=getattr(cfg.loss, "balance_channels", False),
        balance_momentum=getattr(cfg.loss, "balance_momentum", 0.99),
    )

    if rank == 0:
        update_steps = trange(start_step, cfg.train.update_steps, ncols=88, ascii=True)
    else:
        update_steps = range(start_step, cfg.train.update_steps)

    grad_acc_steps = cfg.train.batch_size_per_step // (
        cfg.train.batch_element_per_gpu * world_size
    )

    count_log_train = 0
    count_log_val = 0
    count_log_save = 0
    losses = []
    grads = []

    # Train
    ae.train()

    def distributed_mean_std(values):
        values = values.detach().to(dtype=torch.float64)
        moments = torch.stack(
            (
                values.sum(),
                values.square().sum(),
                values.new_tensor(values.numel()),
            )
        )
        dist.all_reduce(moments, op=dist.ReduceOp.SUM)
        mean = moments[0] / moments[2]
        variance = (moments[1] / moments[2] - mean.square()).clamp_min(0)
        return mean.item(), variance.sqrt().item()

    def cycle(loader):
        for epoch in itertools.count():
            if loader.sampler is not None:
                loader.sampler.set_epoch(epoch)
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    dist.barrier()

    for step in update_steps:
        start = time.time()

        if rank == 0 and not os.path.isfile(runpath / ".running"):
            print("Run aborted by removing the .running file. Stopping...")
            wandb.finish()
            job_id = os.environ["SLURM_ARRAY_JOB_ID"]
            os.system(f"scancel {job_id}")
            dist.destroy_process_group()
            return

        for acc_step in range(grad_acc_steps):
            state, timestamp = next(train_iter)

            state = rearrange(
                state.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            )
            # context = rearrange(
            #     context.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
            # )
            timestamp = rearrange(timestamp.to(device, non_blocking=True), "B T D -> (B T) D")
            context = None
            # Forward
            with torch.autocast(device_type="cuda", dtype=precision):
                if acc_step + 1 == grad_acc_steps:
                    # Only synchronize the last step
                    z, x = ae(state, timestamp, context)
                else:
                    with ae.no_sync():
                        z, x = ae(state, timestamp, context)

                recon_loss = model_loss(x, state)
                if getattr(ae.module, "vae", False):
                    # Variational latent regularization: KL[q(z|x) || N(0,1)],
                    # stashed on the model during encode(). Linear warmup of beta
                    # lets the AE learn reconstruction first (anti-collapse).
                    kl = ae.module.last_kl
                    warmup = getattr(cfg.loss, "kl_warmup_steps", 0)
                    beta = cfg.loss.kl_weight * (
                        min(1.0, (step + 1) / warmup) if warmup and warmup > 0 else 1.0
                    )
                    loss = recon_loss + beta * kl
                    last_kl_val = kl.detach()
                    last_beta_val = beta
                else:
                    loss = recon_loss + cfg.loss.l2_weight * (z**2).mean()

            # Backward
            if acc_step + 1 == grad_acc_steps:
                # Only synchronize the last step
                scaler.scale(loss / grad_acc_steps).backward()
                grad = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip, scaler=scaler)
                grads.append(grad)
            else:
                with ae.no_sync():
                    scaler.scale(loss / grad_acc_steps).backward()

            losses.append(loss.detach())

        # LR scheduler
        scheduler.step()

        count_log_train += cfg.train.batch_size_per_step
        count_log_val += cfg.train.batch_size_per_step
        count_log_save += cfg.train.batch_size_per_step

        if rank == 0:
            logs = {}
            logs["train/update_step_time"] = time.time() - start
            logs["train/samples_seen"] = cfg.train.batch_size_per_step * (step + 1)
            logs["train/update_steps_done"] = step + 1

        # Logging
        if cfg.train.log_interval <= count_log_train:
            losses = torch.stack(losses)
            grads = torch.stack(grads)

            loss_mean, loss_std = distributed_mean_std(losses)
            grad_mean, grad_std = distributed_mean_std(grads)

            if rank == 0:
                logs["train/losses/mean"] = loss_mean
                logs["train/losses/std"] = loss_std
                logs["train/grad_norm/mean"] = grad_mean
                logs["train/grad_norm/std"] = grad_std
                logs["train/lr"] = optimizer.param_groups[0]["lr"]
                if getattr(ae.module, "vae", False):
                    logs["train/kl"] = last_kl_val.item()
                    logs["train/kl_beta"] = last_beta_val
                    logs["train/recon_loss"] = recon_loss.detach().item()

        # Validation
        if cfg.valid.log_interval <= count_log_val:
            with torch.no_grad():
                ae.eval()

                # Set all model's gradients to None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # Quantitative validation.
                # Cap the number of validation batches per rank so the full
                # validation window (hourly frames) is NOT scanned every time:
                # honour the intended cfg.valid.n_valid_samples budget and keep
                # runtime bounded. All ranks run the same count so validation
                # collectives stay balanced.
                n_valid_batches = max(
                    1,
                    -(-n_valid_samples // (world_size * cfg.train.batch_element_per_gpu)),
                )
                valid_losses = []
                atm_errors = []
                for vbatch, (state, timestamp) in enumerate(valid_quant_loader):
                    state = rearrange(state, "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(
                        device, non_blocking=True
                    )
                    # context = rearrange(context, "B T Z Lat Lon -> (B T) (Lat Lon) Z").to(
                    #     device, non_blocking=True
                    # )
                    context = None
                    timestamp = rearrange(
                        timestamp.to(device, non_blocking=True), "B T D -> (B T) D"
                    )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=precision,
                    ):
                        _, x = ae(state, timestamp, context)
                        loss = model_loss(state, x)
                        rmse = torch.sqrt((x - state).square().mean(dim=1)).mean(dim=0)
                        # surf_error = rmse[: len(ERA5_SURFACE_VARIABLES)]
                        atm_error = rmse[:]
                    valid_losses.append(loss)
                    atm_errors.append(rmse)

                    del x, state, context

                    if vbatch + 1 >= n_valid_batches:
                        break

                valid_losses = torch.stack(valid_losses)
                atm_error = torch.stack(atm_errors).mean(dim=0)
                torch.cuda.empty_cache()

                # Qualitative validation.
                # This is what OOM'd: snapshots were accumulated on-GPU over the
                # ENTIRE validation window. Cap to the intended n_valid_snapshots
                # budget with the same number of batches on every rank.
                n_snap_batches = max(
                    1,
                    -(-n_snapshots // (world_size * cfg.train.batch_element_per_gpu)),
                )
                snapshots = []
                dates = []
                for sbatch, (data, timestamp) in enumerate(valid_quali_loader):
                    state = rearrange(
                        data.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                    ).nan_to_num()
                    # context = rearrange(
                    #     context.to(device, non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                    # ).nan_to_num()
                    context = None
                    timestamp = rearrange(
                        timestamp.to(device, non_blocking=True), "B T D -> (B T) D"
                    )

                    with torch.autocast(device_type="cuda", dtype=precision):

                        _, x = ae(state, timestamp, context)
                    del context
                    # print(f'Model take in state shape {state.shape}, output x shape {x.shape}')
                    x = rearrange(
                        x,
                        "(B T) (Lat Lon) Z -> B T Z Lat Lon",
                        Lon=data.shape[-1],
                        B=data.shape[0],
                    ).cpu()
                    state = rearrange(
                        state,
                        "(B T) (Lat Lon) Z -> B T Z Lat Lon",
                        Lon=data.shape[-1],
                        B=data.shape[0],
                    ).cpu()
                    snapshot = torch.stack(
                        [state[:, 0, 0, :, :], x[:, 0, 0, :, :]], dim=1
                    ).to(device)
                    snapshots.append(snapshot)
                    dates.append(timestamp.to(device))
                    del x, data, state

                    if sbatch + 1 >= n_snap_batches:
                        break

                snapshots = torch.cat(snapshots, dim=0)
                dates = torch.cat(dates, dim=0)

                valid_loss_mean, valid_loss_std = distributed_mean_std(valid_losses)
                dist.all_reduce(atm_error, op=dist.ReduceOp.SUM)
                atm_error /= world_size

                snapshot_slots = snapshots.new_zeros(
                    (n_snapshots, *snapshots.shape[1:])
                )
                date_slots = dates.new_zeros((n_snapshots, *dates.shape[1:]))
                for local_idx in range(snapshots.shape[0]):
                    global_idx = rank * snapshots.shape[0] + local_idx
                    if global_idx < n_snapshots:
                        snapshot_slots[global_idx].copy_(snapshots[local_idx])
                        date_slots[global_idx].copy_(dates[local_idx])

                dist.all_reduce(snapshot_slots, op=dist.ReduceOp.SUM)
                dist.all_reduce(date_slots, op=dist.ReduceOp.SUM)

                if rank == 0:
                    logs["valid/losses/mean"] = valid_loss_mean
                    logs["valid/losses/std"] = valid_loss_std

                    atm_error = atm_error.cpu()
                    # Surface
                    # for channel_id, var_surf in enumerate(ERA5_SURFACE_VARIABLES):
                        # logs[f"valid/surface_rmse/{var_surf}"] = surf_error[channel_id].item()
                    for channel_id, var_name in enumerate(datasets[0].target_chn):
                        logs[f"valid/rmse/{var_name}"] = atm_error[channel_id].item()

                    # Order snapshot GT/reconstruction pairs by date.
                    snapshots = snapshot_slots.cpu()
                    dates = date_slots.cpu().reshape(-1, 4)
                    dates = [d.tolist() for d in dates]
                    dates = [
                        f"{date[0]:04d}/{date[1]:02d}/{date[2]:02d} {date[3]:02d}h"
                        for date in dates
                    ]
                    dates, snapshots = zip(*sorted(zip(dates, snapshots)))
                    dates = dates[:n_snapshots]
                    snapshots = torch.stack(snapshots)
                    snapshots = snapshots[:n_snapshots]
                    # Set the same vmap for all snapshots
                    vmin = snapshots[:, 0].min().item()
                    vmax = snapshots[:, 0].max().item()
                    snapshots = rearrange(snapshots, "B T ... -> (T B) ...")

                    fig = plot_image_grid(
                        snapshots,
                        shape=(2, n_snapshots),
                        vmin=vmin,
                        vmax=vmax,
                        x_titles=dates,
                        fontsize=24,
                        tex_font=False,
                    )
                    logs["valid/snapshots"] = wandb.Image(fig)
                    plt.close()

                    if logs["valid/losses/mean"] < best_val_loss:
                        best_val_loss = logs["valid/losses/mean"]
                        safe_save(
                            ae.module.state_dict(),
                            runpath / "model_best.pth",
                        )

                del snapshots, snapshot_slots, date_slots, valid_losses, atm_errors
                torch.cuda.empty_cache()

                # Reset to train after validation
                ae.train()

                count_log_val = 0

        if rank == 0 and count_log_save >= cfg.train.save_interval:
            for obj, name in zip(
                (ae.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
            ):
                safe_save(
                    obj.state_dict(),
                    runpath / f"{name}_last.pth",
                )

            with open(runpath / "metadata.yaml", "w") as f:
                OmegaConf.save(
                    {
                        "last_step_done": step,
                        "best_val_loss": best_val_loss,
                        "lap": lap,
                        "runid": runid,
                    },
                    f,
                )

            count_log_save = 0

        if cfg.valid.log_interval <= count_log_val or cfg.train.log_interval <= count_log_train:
            if cfg.train.log_interval <= count_log_train:
                count_log_train = 0
                losses = []
                grads = []
                if rank == 0:
                    update_steps.set_postfix(loss=logs["train/losses/mean"])
            if rank == 0:
                run.log(logs, step=step)

    if rank == 0:
        run.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str, help="Hydra overrides.")
    parser.add_argument("--cpus-per-gpu", type=int, default=8, help="Number of CPU cores per GPU.")
    parser.add_argument("--gpus", type=int, default=4, help="Total number of GPUs.")
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Enforces a number of nodes. Useful to avoid needing full nodes.",
    )
    parser.add_argument("--ram", type=str, default="60GB", help="Amount of RAM per GPU.")
    parser.add_argument("--time", type=str, default="2-00:00:00", help="Time limit.")
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="Slurm partition.",
    )
    parser.add_argument("--lap-start", type=int, default=0, help="Lap number to start from.")
    parser.add_argument("--laps", type=int, default=1, help="Maximum number of laps to perform.")
    parser.add_argument(
        "--fork", type=str, default=None, help="Restart the a given runid/lap with a new id."
    )
    parser.add_argument(
        "--continue",
        dest="continue_",
        type=str,
        default=None,
        help="Restart the a given runid/lap with the same id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (defaults to RISDA_OUTPUT_ROOT).",
    )
    parser.add_argument(
        "--runid",
        type=str,
        default=None,
        help="Reuse an existing runid. If <output-dir>/ae/<runid>/model_last.pth exists, "
             "training resumes from last_step_done + 1.",
    )

    args = parser.parse_args()

    lap_start = args.lap_start
    laps = args.laps

    assert not (args.continue_ and args.fork), "Cannot continue and fork at the same time."

    if args.continue_:
        runid, lap_start = args.continue_.split("/")
        lap_start = int(lap_start) + 1
        config_path = PATH_AE / args.continue_ / "config.yaml"
    elif args.fork:
        lap_start = int(args.fork.split("/")[1]) + 1
        config_path = PATH_AE / args.fork / "config.yaml"

        runid = wandb.util.generate_id()
        forked_lap_path = PATH_AE / args.fork
        new_lap_path = PATH_AE / runid / f"{lap_start - 1}"

        # Create symlink to the run we fork from
        os.makedirs(PATH_AE / runid, exist_ok=True)
        new_lap_path.symlink_to(forked_lap_path)

        args.overrides = [f"++forked_from={args.fork}"] + args.overrides
    else:
        config_path = "./configs/train_1h.yaml"
        runid = wandb.util.generate_id()

    cfg = compose(
        config_file=config_path,
        overrides=args.overrides,
    )

    if args.nodes is not None:
        assert args.gpus % args.nodes == 0, "Number of GPUs must be divisible by number of nodes."

        num_nodes = args.nodes
        num_gpus = args.gpus // num_nodes
    else:
        gpu_per_nodes = 8 if args.partition == "ia" else 4  # gpu
        num_nodes = args.gpus // gpu_per_nodes if args.gpus >= gpu_per_nodes else 1
        num_gpus = min(args.gpus, gpu_per_nodes)  # Local number of GPUs.

    match = re.match(r"(\d+)([A-Za-z]+)", args.ram)
    ram_amount, unit = int(match.group(1)) * num_gpus, match.group(2)
    ram = f"{ram_amount}{unit}"

    if num_nodes > 1:
        interpreter = f"torchrun --nnodes {num_nodes} --nproc-per-node {num_gpus} --rdzv_backend=c10d --rdzv_endpoint=$SLURMD_NODENAME:12345 --rdzv_id=$SLURM_JOB_ID"
    else:
        interpreter = f"torchrun --nnodes 1 --nproc-per-node {num_gpus} --standalone"



    if args.runid is not None:
        runid = args.runid
    else:
        runid = wandb.util.generate_id()
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"PIPELINE_AE_RUNID={runid}")
    train(runid=runid, cfg=cfg, fork_lap=0, lap=0, output_dir=args.output_dir)

    # dawgz.schedule(
    #     dawgz.job(
    #         f=partial(train, runid, cfg, lap_start),
    #         name=f"appa ae {runid}",
    #         nodes=num_nodes,
    #         cpus=args.cpus_per_gpu * num_gpus,
    #         gpus=num_gpus,
    #         ram=ram,
    #         time=args.time,
    #         partition=args.partition,
    #         account=cfg.slurm_account,
    #         array=range(lap_start, lap_start + args.laps),
    #         array_throttle=1,
    #     ),
    #     name=f"appa ae {runid}",
    #     backend="slurm",
    #     interpreter=interpreter,
    #     env=[
    #         "export OMP_NUM_THREADS=" + f"{args.cpus_per_gpu}",
    #         "export WANDB_SILENT=true",
    #         "export XDG_CACHE_HOME=$HOME/.cache",
    #         "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
    #     ],
    # )
