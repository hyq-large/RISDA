r"""Latent diffusion training script."""

import argparse
import cloudpickle
import dawgz
import h5py
import os
import re
import wandb

from datetime import datetime
from einops import rearrange
from functools import partial
from omegaconf import DictConfig, open_dict
from pathlib import Path

import appa

from appa.config import OUTPUT_ROOT
from appa.config.hydra import compose
from appa.data.const import DATASET_DATES_TRAINING, DATASET_DATES_VALIDATION
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import (
    AnchoredLatentBlanketDataset,
    ERAData_v1,
)
from appa.diagnostics.plots import plot_image_grid, plot_spheres_grid
from appa.diffusion import DenoiserLoss, create_denoiser, create_schedule
from appa.grid import create_icosphere, latlon_to_xyz
from appa.optim import get_optimizer, safe_gd_step
from appa.sampling import DDIMSampler
from appa.save import safe_load, safe_save

cloudpickle.register_pickle_by_value(appa)


def train(
    runid: str,
    cfg: DictConfig,
    fork_lap: int = 0,
    lap: int = 0,
    output_dir: str = None,
):
    r"""Train for one lap a latent diffusion model.

    Args:
        runid: The run id.
        cfg: The training configuration as a DictConfig.
        fork_lap: If starting from a checkpoint, the lap to fork from.
        lap: The current lap number.
    """
    import itertools
    import matplotlib.pyplot as plt
    import os
    import time
    import torch
    import torch.distributed as dist

    from datetime import timedelta
    from math import ceil
    from omegaconf import OmegaConf
    from torch.amp.grad_scaler import GradScaler
    from tqdm import trange

    # Parallelism
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    print(f"RANK {rank}")
    world_size = dist.get_world_size()
    num_nodes = ceil(world_size / torch.cuda.device_count())
    num_cpu = len(os.sched_getaffinity(0)) * num_nodes
    device_id = os.environ.get("LOCAL_RANK", rank)
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    torch.set_float32_matmul_precision("high")

    assert (
        cfg.train.batch_size_per_step % (cfg.train.batch_element_per_gpu * world_size) == 0
    ), "Batch size must be divisible by number of GPU * samples per GPU"
    temporal_patch = (
        cfg.backbone.patch_size
        if isinstance(cfg.backbone.patch_size, int)
        else cfg.backbone.patch_size[0]
    )
    if cfg.train.blanket_size % temporal_patch != 0:
        raise ValueError(
            f"blanket_size={cfg.train.blanket_size} must be divisible by the "
            f"temporal patch size {temporal_patch}."
        )

    # assert cfg.ae_run is not None, "Autoencoder run id and lap must be specified."
    # assert cfg.latent_dump is not None, "Latent dump id must be specified."

    PATH_AE = Path(output_dir or OUTPUT_ROOT) / "ae"

    ae_cfg = compose(
        config_file=PATH_AE / cfg.ae_run / "config.yaml",
    )

    latent_folder = PATH_AE / cfg.ae_run / "latents" / cfg.latent_dump
    latent_data = latent_folder / cfg.latent_data
    assert (
        latent_data.exists()
    ), "Latent data must exist. Run the script scripts/run_latents.py to generate it."

    requested_splits = (
        ["2017-01-01", "2024-12-31", 0, 23],
        ["2025-01-01", "2025-03-31", 0, 23],
    )
    with h5py.File(latent_data, "r") as latent_file:
        first_date = latent_file["dates"][0]
        last_date = latent_file["dates"][-1]
    available_start = datetime(*(int(value) for value in first_date))
    available_end = datetime(*(int(value) for value in last_date))

    dataset_splits = []
    for start_date, end_date, start_hour, end_hour in requested_splits:
        requested_start = datetime.fromisoformat(start_date).replace(hour=start_hour)
        requested_end = datetime.fromisoformat(end_date).replace(hour=end_hour)
        actual_start = max(requested_start, available_start)
        actual_end = min(requested_end, available_end)
        if actual_start > actual_end:
            raise ValueError(
                f"Requested latent range {requested_start}..{requested_end} does "
                f"not overlap available range {available_start}..{available_end}."
            )
        if rank == 0 and (actual_start != requested_start or actual_end != requested_end):
            print(
                "Clipping latent date range "
                f"{requested_start}..{requested_end} to "
                f"{actual_start}..{actual_end}.",
                flush=True,
            )
        dataset_splits.append(
            [
                actual_start.strftime("%Y-%m-%d"),
                actual_end.strftime("%Y-%m-%d"),
                actual_start.hour,
                actual_end.hour,
            ]
        )

    if "inject_ae_noise" in cfg.train and cfg.train.inject_ae_noise:
        noise_level = ae_cfg.ae.noise_level
    else:
        noise_level = 0.0

    train_dataset, valid_dataset = (
        AnchoredLatentBlanketDataset(
            latent_data,
            start_date,
            end_date,
            cfg.train.blanket_size,
            start_hour=start_hour,
            end_hour=end_hour,
            standardize=cfg.train.standardize,
            stride=cfg.train.blanket_dt,
            noise_level=noise_level,
            anchor_position=cfg.train.anchor_frame - 1,
            anchor_hour_interval=cfg.train.anchor_hour_interval,
        )
        for start_date, end_date, start_hour, end_hour in dataset_splits
    )

    dataloader_params = dict(
        batch_size=cfg.train.batch_element_per_gpu,
        num_workers=num_cpu // world_size,
        rank=rank,
        world_size=world_size,
    )

    train_loader = get_dataloader(
        dataset=train_dataset,
        shuffle=cfg.train.shuffle_training,
        persistent_workers=True,
        **dataloader_params,
    )

    # Pad number of validation samples to be divisible by world_size
    n_valid_dates = cfg.valid.n_valid_dates
    n_valid_dates_pad = n_valid_dates + (-n_valid_dates) % world_size
    valid_dates_indices = torch.linspace(0, len(valid_dataset) - 1, n_valid_dates_pad).int()

    # TODO: Batch?
    n_quant_valid_samples = cfg.valid.n_valid_samples
    n_quant_valid_samples_pad = n_quant_valid_samples + (-n_quant_valid_samples) % world_size
    valid_quant_indices = torch.linspace(
        0, len(valid_dataset) - 1, n_quant_valid_samples_pad
    ).int()

    if "graph" in cfg.backbone.name:
        pass
        # grid, _ = create_icosphere(ae_cfg.ae.ico_divisions[-1])
        # x, y, z = latlon_to_xyz(grid).unbind(-1)
    else:
        grid = None
        x, y, z = None, None, None

    schedule = create_schedule(cfg.train, device)

    with open_dict(cfg):
        cfg.backbone["blanket_size"] = cfg.train.blanket_size
    denoise = create_denoiser(
        diffusion_cfg=cfg,
        ae_cfg=ae_cfg,
        distributed=True,
        device=device,
    )

    model = denoise.backbone

    # Optional finetune init: load weights from an existing denoiser checkpoint
    init_from = cfg.get("init_from", None)
    if init_from is not None:
        init_which = cfg.get("init_which", "last")
        ckpt_path = Path(init_from) / f"model_{init_which}.pth"
        state = safe_load(ckpt_path, map_location="cpu")
        # model is wrapped in DDP — load into .module
        model.module.load_state_dict(state)
        if rank == 0:
            print(f"[finetune] loaded weights from {ckpt_path}", flush=True)

    model_loss = DenoiserLoss(denoise, std_min_penalty=cfg.loss.sigma_min_penalty).to(device)

    optimizer, scheduler = get_optimizer(
        params=model.parameters(),
        update_steps=cfg.train.update_steps,
        **cfg.optim,
    )


    run_path = latent_folder / "denoisers" / f"{runid}"
    lap_path = run_path
    if rank == 0:
        lap_path.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")

    if start_step >= cfg.train.update_steps:
        dist.destroy_process_group()
        return

    # Logging
    if rank == 0:
        wandb_config = OmegaConf.to_container(cfg).copy()

        # Additional config fields
        wandb_config["num_model_params"] = sum(p.numel() for p in model.parameters())
        wandb_config["path"] = run_path
        wandb_config["ae_path"] = PATH_AE / cfg.ae_run
        wandb_config["latent_dump_path"] = latent_folder
        wandb_config["hardware"] = {
            "num_nodes": num_nodes,
            "num_gpus": world_size,
            "num_cpus": num_cpu,
        }

        run_name = f"K{cfg.train.blanket_size}-{cfg.backbone.name}-{runid}"

        description = f"Denoiser {cfg.backbone.name} " + model.module.description()

        if "forked_from" in cfg:
            description = f"Forked from {cfg.forked_from}.\n" + description

        wandb_args = dict(
            project="appa_denoiser",
            name=run_name,
            entity=cfg.wandb_entity,
            config=wandb_config,
            group=f"train_{runid}",
            resume="allow",
            notes=description,
        )

        run = wandb.init(id=runid, **wandb_args)

        OmegaConf.save(cfg, lap_path / "config.yaml")

        # Kill switch
        open(lap_path / ".running", "a").close()

    scaler = GradScaler("cuda")
    precision = getattr(torch, cfg.train.precision)

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
    model.train()

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

        if rank == 0 and not os.path.isfile(lap_path / ".running"):
            print("Run aborted by removing the .running file. Stopping...")
            wandb.finish()
            job_id = os.environ["SLURM_ARRAY_JOB_ID"]
            os.system(f"scancel {job_id}")
            dist.destroy_process_group()
            return

        for acc_step in range(grad_acc_steps):
            data, context = next(train_iter)
            data = data.flatten(1).to(device)
            # data = data.to(device)
            context = context.to(device)
            sigma_t = schedule(torch.rand(data.shape[0]).to(data))

            # Forward
            with torch.autocast(device_type="cuda", dtype=precision):
                if acc_step + 1 == grad_acc_steps:
                    # Only synchronize the last step
                    loss = model_loss(data, sigma_t, date=context)
                else:
                    with model.no_sync():
                        loss = model_loss(data, sigma_t, date=context)

            # Backward
            if acc_step + 1 == grad_acc_steps:
                # Only synchronize the last step
                scaler.scale(loss / grad_acc_steps).backward()
                grad = safe_gd_step(optimizer, grad_clip=cfg.optim.grad_clip, scaler=scaler)
                grads.append(grad)
            else:
                with model.no_sync():
                    scaler.scale(loss / grad_acc_steps).backward()

            losses.append(loss.detach())

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

        # Validation
        if cfg.valid.log_interval <= count_log_val:
            with torch.no_grad():
                model.eval()

                # Set all model's gradients to None
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

                # Qualitative validation
                #   For each date, generate n_valid_samples_per_date samples
                #   conditioned on that time of the year and plot them
                #   alongside ground truth at the same date.
                snapshots = []
                dates = []
                samples_per_gpu = n_valid_dates_pad // world_size
                start_idx = rank * samples_per_gpu
                for idx in valid_dates_indices[start_idx : start_idx + samples_per_gpu]:
                    gt, context = valid_dataset[idx]
                    gt, context = gt[None], context[None]
                    context = context.cuda()

                    conditioned_denoiser = partial(denoise, date=context)
                    sampler = DDIMSampler(
                        denoiser=conditioned_denoiser,
                        steps=cfg.valid.denoising_steps,
                        schedule=schedule,
                    )

                    generated_samples = [gt.cuda()]

                    for _ in range(cfg.valid.n_valid_samples_per_date):
                        sampled_state = sampler(
                            sampler.schedule(torch.ones(1).cuda())
                            * torch.randn(
                                1, cfg.train.blanket_size * torch.tensor(gt.shape[2:]).prod()
                            ).cuda()
                        ).reshape((-1, cfg.train.blanket_size, *gt.shape[2:]))

                        generated_samples.append(sampled_state)

                    generated_samples = torch.cat(generated_samples)

                    snapshots.append(generated_samples)
                    dates.append(context)

                snapshots = torch.stack(snapshots)
                dates = torch.cat(dates)
                snapshot_slots = snapshots.new_zeros(
                    (n_valid_dates, *snapshots.shape[1:])
                )
                date_slots = dates.new_zeros((n_valid_dates, *dates.shape[1:]))
                for local_idx, global_idx in enumerate(
                    range(start_idx, start_idx + samples_per_gpu)
                ):
                    if global_idx < n_valid_dates:
                        snapshot_slots[global_idx].copy_(snapshots[local_idx])
                        date_slots[global_idx].copy_(dates[local_idx])

                dist.all_reduce(snapshot_slots, op=dist.ReduceOp.SUM)
                dist.all_reduce(date_slots, op=dist.ReduceOp.SUM)

                if rank == 0:
                    num_channels = cfg.valid.n_valid_channels
                    channels = torch.linspace(0, train_dataset.get_dim() - 1, num_channels).int()

                    snapshots_to_log = snapshot_slots.cpu()
                    dates_to_log = date_slots.cpu()
                    row_titles = ["GT"] + ["Sample"] * cfg.valid.n_valid_samples_per_date

                    for i, (valid_blankets, date) in enumerate(
                        zip(snapshots_to_log, dates_to_log)
                    ):
                        valid_blankets = valid_blankets.cpu()  #  [B,blanket_size,C,H,W]
                        # valid_blankets = rearrange(valid_blankets,"N B C H W -> N B H W C")
                        date = date.cpu()

                        for channel_id in channels:
                            valid_blankets_c = valid_blankets[..., channel_id]
                            vmin = valid_blankets_c[0].min().item()
                            vmax = valid_blankets_c[0].max().item()
                            if "graph" in cfg.backbone.name:
                                fig = plot_spheres_grid(
                                    blankets=valid_blankets_c,
                                    date=date,
                                    coordinates=(x, y, z),
                                    vmin=vmin,
                                    vmax=vmax,
                                    row_titles=row_titles,
                                )
                            else:
                                H, W = model.module.shape
                                B, K = valid_blankets_c.shape[:2]
                                valid_blankets_c = rearrange(
                                    valid_blankets_c, "B K (H W) -> (B K) H W", H=H, W=W
                                )
                                fig = plot_image_grid(
                                    valid_blankets_c,
                                    shape=(B, K),
                                    vmin=vmin,
                                    vmax=vmax,  # ADD x_titles ?
                                    fontsize=24,
                                    tex_font=False,
                                )

                            logs[f"valid/snapshot_{i}/channel_{channel_id}"] = wandb.Image(fig)
                            plt.close()

                del snapshots, dates, snapshot_slots, date_slots, generated_samples
                torch.cuda.empty_cache()

                # Quantitative validation
                valid_losses = []
                samples_per_gpu = n_quant_valid_samples_pad // world_size
                for idx in valid_quant_indices[
                    samples_per_gpu * rank : samples_per_gpu * (rank + 1)
                ]:
                    data, context = valid_dataset[idx]
                    data = data[None]
                    context = context[None]

                    data = data.flatten(1).to(device)
                    context = context.to(device)
                    sigma_t = schedule(torch.rand(data.shape[0]).to(data))
                    with torch.autocast(device_type="cuda", dtype=precision):
                        loss = model_loss(data, sigma_t, date=context)
                    valid_losses.append(loss)

                    del context, data, sigma_t

                valid_losses = torch.stack(valid_losses)
                valid_loss_mean, valid_loss_std = distributed_mean_std(valid_losses)

                if rank == 0:
                    logs["valid/losses/mean"] = valid_loss_mean
                    logs["valid/losses/std"] = valid_loss_std

                    if logs["valid/losses/mean"] < best_val_loss:
                        best_val_loss = logs["valid/losses/mean"]
                        safe_save(
                            model.module.state_dict(),
                            lap_path / "model_best.pth",
                        )

                del valid_losses
                torch.cuda.empty_cache()

            # Reset to train after validation
            model.train()

            count_log_val = 0

        if rank == 0 and count_log_save >= cfg.train.save_interval:
            for obj, name in zip(
                (model.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
            ):
                safe_save(
                    obj.state_dict(),
                    lap_path / f"{name}_last.pth",
                )

            with open(lap_path / "metadata.yaml", "w") as f:
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

        if rank == 0 and step%2000 == 0:
            for obj, name in zip(
                (model.module, optimizer, scheduler), ("model", "optimizer", "scheduler")
            ):
                safe_save(
                    obj.state_dict(),
                    lap_path / f"{name}_step_{step}.pth",
                )

        if cfg.valid.log_interval <= count_log_val or cfg.train.log_interval <= count_log_train:
            if cfg.train.log_interval <= count_log_train:
                count_log_train = 0
                losses = []
                grads = []
            if rank == 0:
                update_steps.set_postfix(loss=logs["train/losses/mean"])
                run.log(logs, step=step)

    if rank == 0:
        run.finish()

    # DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", type=str, help="Hydra overrides.")
    parser.add_argument("--cpus-per-gpu", type=int, default=8, help="Number of CPU cores per GPU.")
    parser.add_argument("--gpus", type=int, default=8, help="Total number of GPUs.")
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
    parser.add_argument("--laps", type=int, default=2, help="Maximum number of laps to perform.")
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
        "--output_dir",
        type=str,
        default=None,
        help="Output root directory (defaults to RISDA_OUTPUT_ROOT).",
    )


    # Use parse_known_args so positional Hydra overrides ("foo.bar=...", "++baz=...")
    # appearing *after* an --option (e.g. --output_dir=...) are not rejected.
    # parse_args() with nargs="*" positional only accepts positionals BEFORE the
    # first --option; anything after lands in unknowns.  We merge unknowns into overrides.
    args, _unknown = parser.parse_known_args()
    args.overrides = list(args.overrides) + list(_unknown)

    lap_start = args.lap_start
    laps = args.laps

    assert not (args.continue_ and args.fork), "Cannot continue and fork at the same time."

    if args.continue_:
        runid, lap_start = args.continue_.split("/")[-2:]
        lap_start = int(lap_start) + 1
        config_path = f"{args.continue_}/config.yaml"
    elif args.fork:
        lap_start = int(args.fork.split("/")[-1]) + 1
        config_path = f"{args.fork}/config.yaml"

        runid = wandb.util.generate_id()
        forked_lap_path = Path(args.fork)
        new_lap_path = forked_lap_path.parent.parent / runid / f"{lap_start - 1}"

        # Create symlink to the run we fork from
        os.makedirs(new_lap_path.parent, exist_ok=True)
        new_lap_path.symlink_to(forked_lap_path)

        args.overrides = [f"++forked_from={args.fork}"] + args.overrides
    else:
        config_path = "./configs/train_diff_1h.yaml"
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




    runid = wandb.util.generate_id()
    print(f"PIPELINE_DENOISER_RUNID={runid}")
    train(runid=runid, cfg=cfg, fork_lap=0, lap=0,output_dir=args.output_dir)
    # dawgz.schedule(
    #     dawgz.job(
    #         f=partial(train, runid, cfg, lap_start),
    #         name=f"appa dit {runid}",
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
    #     name=f"appa dit {runid}",
    #     backend="slurm",
    #     interpreter=interpreter,
    #     env=[
    #         "export OMP_NUM_THREADS=" + f"{args.cpus_per_gpu}",
    #         "export WANDB_SILENT=true",
    #         "export XDG_CACHE_HOME=$HOME/.cache",
    #         "export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor",
    #     ],
    # )
