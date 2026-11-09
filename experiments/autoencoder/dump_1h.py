r"""Encode ERA5 into latent space with an AE (no SLURM)."""

import dask
import gc
import h5py
import numpy as np
import shutil
import sys
import time
import torch

from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path

from appa.config import PATH_ERA5, PATH_STAT
from appa.config.hydra import compose
from appa.data.const import (
    CONTEXT_VARIABLES,
    ERA5_VARIABLES,
    SUB_PRESSURE_LEVELS,
)
from appa.data.dataloaders import get_dataloader
from appa.data.datasets import ERA5Dataset, ERADataNpy, ERA5DataNpy_2016_2025_1h
from appa.data.transforms import StandardizeTransform
from appa.date import split_interval
from appa.save import load_auto_encoder


def dump_to_latent(config):
    num_chunks = config.num_chunks
    start_date = config.start_date
    end_date = config.end_date
    start_hour = config.start_hour
    end_hour = config.end_hour
    data_path = config.data_path
    model_path = Path(config.model_path)
    use_best = config.checkpoint == "best"
    print(f'use_best {use_best}')

    save_every = config.save_every
    log_every = config.log_every

    output_path = model_path / "latents" / config.id
    (output_path / "tmp").mkdir(parents=True, exist_ok=True)

    # Copy AE
    (output_path / "ae").mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path / "config.yaml", output_path / "ae" / "config.yaml")
    shutil.copy2(
        model_path / ("model_best.pth" if use_best else "model_last.pth"),
        output_path / "ae" / "model.pth",
    )

    ae_config = OmegaConf.load(model_path / "config.yaml")
    normalization_kwargs = {}
    if "normalization" in ae_config:
        normalization_kwargs = {
            name: ae_config.normalization[name]
            for name in (
                "source_mean_path", "source_std_path",
                "target_mean_path", "target_std_path",
            )
        }

    time_intervals = split_interval(
        num_chunks, start_date, end_date, start_hour, end_hour
    )

    # --- Load AE once ---
    dask.config.set(scheduler="synchronous")

    autoencoder = load_auto_encoder(output_path / "ae", model_name="model")
    autoencoder.cuda()
    autoencoder.eval()

    latent_state_shape = autoencoder.latent_shape
    if len(latent_state_shape) == 3:
        latent_state_shape = (
            latent_state_shape[0] * latent_state_shape[1],
            latent_state_shape[2],
        )

    # st = StandardizeTransform(
    #     path=PATH_STAT,
    #     state_variables=ERA5_VARIABLES,
    #     context_variables=CONTEXT_VARIABLES,
    #     levels=SUB_PRESSURE_LEVELS,
    # )

    # --- Dump chunks sequentially ---
    for rank in range(num_chunks):
        print(f"\n=== Chunk {rank+1}/{num_chunks} ===", flush=True)

        time_interval = time_intervals[rank]
        chunk_start, chunk_end, chunk_start_hour, chunk_end_hour = time_interval


        dataset= ERA5DataNpy_2016_2025_1h(
            data_path=data_path,
            blanket_size=1,
            time_range=[
                f"{chunk_start}T{chunk_start_hour:02d}",
                f"{chunk_end}T{chunk_end_hour:02d}",
            ],
            **normalization_kwargs,
        )
        print(config.batch_size)
        dataloader = get_dataloader(
            dataset,
            batch_size=config.batch_size,
            num_workers=8,
            prefetch_factor=1,
            shuffle=False,
        )

        num_samples = len(dataset)
        len_dl = len(dataloader)
        print(f"Chunk samples: {num_samples}, batches: {len_dl}", flush=True)

        dump_tmp_file = h5py.File(output_path / f"tmp/{rank}.h5", "w")
        dump_tmp_file.create_dataset("latents", (num_samples, *latent_state_shape), "float32")
        dump_tmp_file.create_dataset("dates", (num_samples, 4), "int32")

        dumped_states = []
        timestamps = []
        curr_idx = 0
        start_time = time.time()

        def save_progress():
            nonlocal curr_idx, dumped_states, timestamps, start_time

            if not dumped_states:
                return

            dumped_states = np.concatenate(dumped_states, axis=0)
            timestamps = np.concatenate(timestamps, axis=0)
            dump_tmp_file["latents"][curr_idx : curr_idx + len(dumped_states)] = dumped_states
            dump_tmp_file["dates"][curr_idx : curr_idx + len(dumped_states)] = timestamps
            curr_idx += len(dumped_states)

            dumped_states = []
            timestamps = []
            gc.collect()

        with torch.no_grad():
            for idx, (state, date) in enumerate(dataloader):
                state = rearrange(
                    state.to("cuda", non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                )
                # context = rearrange(
                #     context.to("cuda", non_blocking=True), "B T Z Lat Lon -> (B T) (Lat Lon) Z"
                # )

                context = None

                date = rearrange(date.to("cuda", non_blocking=True), "B T D -> (B T) D")

                z = autoencoder.encode(state, date, context)

                dumped_states.append(z.cpu().numpy())
                timestamps.append(date.cpu().numpy())

                if (idx + 1) % log_every == 0:
                    elapsed_time = time.time() - start_time
                    print(
                        f"{idx + 1}/{len_dl} - Avg. time per sample: {(elapsed_time / log_every):.2f}s",
                        flush=True,
                    )
                    start_time = time.time()

                if len(dumped_states) >= save_every:
                    save_progress()

        save_progress()
        dump_tmp_file.close()

    # --- Merge all chunks ---
    print("\n=== Merging chunks ===", flush=True)

    total_dataset = ERA5DataNpy_2016_2025_1h(
            data_path=data_path,
            blanket_size=1,
            time_range=[
                f"{start_date}T{start_hour:02d}",
                f"{end_date}T{end_hour:02d}",
            ],
            **normalization_kwargs,
        )
    total_num_samples = len(total_dataset)
    del total_dataset

    dump_file = h5py.File(output_path / config.dump_name, "w")
    dump_file.create_dataset("latents", (total_num_samples, *latent_state_shape), "float32")
    dump_file.create_dataset("dates", (total_num_samples, 4), "int32")

    curr_idx = 0
    for rank in range(num_chunks):
        chunk_file = h5py.File(output_path / f"tmp/{rank}.h5", "r")

        latents = chunk_file["latents"][:]
        dates = chunk_file["dates"][:]

        print(f"Chunk {rank} shape: {latents.shape}, {dates.shape}", flush=True)

        num_samples_chunk = min(latents.shape[0], total_num_samples - curr_idx)
        latents = latents[:num_samples_chunk]
        dates = dates[:num_samples_chunk]

        dump_file["latents"][curr_idx : curr_idx + num_samples_chunk] = latents
        dump_file["dates"][curr_idx : curr_idx + num_samples_chunk] = dates
        curr_idx += num_samples_chunk

        chunk_file.close()

    print(dump_file["latents"].shape, dump_file["dates"].shape, flush=True)
    dump_file.close()

    shutil.rmtree(output_path / "tmp")
    print("Done!", flush=True)


if __name__ == "__main__":
    config = compose("configs/dump_1h.yaml", overrides=sys.argv[1:])
    OmegaConf.set_readonly(config, False)
    OmegaConf.set_struct(config, False)

    if "id" not in config:
        import wandb
        config["id"] = wandb.util.generate_id()

    if config.data_path == "era5":
        config.data_path = PATH_ERA5

    dump_to_latent(config)
