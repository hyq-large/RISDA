import contextlib
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Subset

from ..config import json_write
from .data import dataset_identity
from .soap import SOAP


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def optimizer_for(model, config):
    kwargs = dict(
        lr=float(config.optim.learning_rate), betas=tuple(config.optim.betas),
        eps=float(config.optim.get("eps", 1e-8)), weight_decay=float(config.optim.weight_decay),
    )
    if config.optim.optimizer == "soap":
        optimizer = SOAP(
            model.parameters(), **kwargs,
            precondition_frequency=int(config.optim.precondition_frequency),
            max_precond_size=int(config.optim.precondition_size),
            merge_dims=bool(config.optim.merge_dims),
        )
    elif config.optim.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError("Supported optimizers: soap, adamw")
    schedule = config.optim.scheduler
    if schedule not in (None, "linear"):
        raise ValueError("Supported learning-rate schedules: null, linear")
    def factor(step):
        decay = max(0.0, 1 - step / config.train.update_steps) if schedule == "linear" else 1.0
        return decay * min(1.0, (step + 1) / (int(config.optim.warmup) + 1))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class RemainingBatches:
    def __init__(self, sampler, offset):
        self.sampler, self.offset = sampler, offset

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return max(0, len(self.sampler) - self.offset)


class BatchStream:
    def __init__(self, dataset, microbatch, rank, world, workers, seed, cursor=(0, 0)):
        self.dataset, self.microbatch = dataset, microbatch
        self.rank, self.world, self.workers, self.seed = rank, world, workers, seed
        self.epoch, self.offset = cursor
        self.iterator = None

    def next(self):
        while True:
            if self.iterator is None:
                sampler = DistributedSampler(
                    self.dataset, self.world, self.rank, shuffle=True, seed=self.seed, drop_last=True
                )
                sampler.set_epoch(self.epoch)
                batches = BatchSampler(sampler, self.microbatch, drop_last=True)
                if len(batches) == 0:
                    raise ValueError("Training split is smaller than one distributed microbatch")
                generator = torch.Generator().manual_seed(self.seed + self.epoch)
                self.iterator = iter(DataLoader(
                    self.dataset, batch_sampler=RemainingBatches(batches, self.offset),
                    num_workers=self.workers, pin_memory=torch.cuda.is_available(),
                    generator=generator,
                ))
            try:
                batch = next(self.iterator)
                self.offset += 1
                return batch
            except StopIteration:
                self.epoch += 1
                self.offset = 0
                self.iterator = None


def move_batch(batch, device):
    if torch.is_tensor(batch):
        return (batch.to(device, non_blocking=True),)
    return tuple(value.to(device, non_blocking=True) for value in batch)


def autocast(device, precision):
    if precision == "float32":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=getattr(torch, precision))


def validate(problem, dataset, config, device, rank, world):
    saved_rng = rng_state(device)
    was_training = problem.training
    try:
        seed_all(int(config.seed) + rank)
        problem.eval()
        count = min(len(dataset), int(config.train.validation_samples or len(dataset)))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(config.seed)))[:count]
        subset = Subset(dataset, indices.tolist()[rank::world])
        loader = DataLoader(
            subset, batch_size=int(config.train.microbatch), num_workers=int(config.train.num_workers),
            generator=torch.Generator().manual_seed(int(config.seed)),
        )
        total = torch.zeros(2, dtype=torch.float64, device=device)
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                with autocast(device, config.train.precision):
                    loss = problem(*batch)
                n = batch[0].shape[0]
                total[0] += loss.double() * n
                total[1] += n
        if world > 1:
            dist.all_reduce(total)
        loss = (total[0] / total[1]).item()
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        return loss
    finally:
        problem.train(was_training)
        restore_rng(saved_rng, device)


def train_loop(config, problem, training, validation, device, resume=False, stop_after=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    micro = int(config.train.microbatch)
    batch = int(config.train.batch_size_per_step)
    if micro < 1 or batch < micro * world or batch % (micro * world):
        raise ValueError("Global batch must be divisible by microbatch times world size")
    if device.type == "cpu" and config.train.precision == "float16":
        raise ValueError("CPU training requires train.precision=float32 or bfloat16")
    accum = batch // (micro * world)
    output = Path(config.output_dir)
    resolved = OmegaConf.to_container(config, resolve=True)
    identities = {"training": dataset_identity(training), "validation": dataset_identity(validation)}
    reference = config.get("era_latent_dir", config.get("latent_dir"))
    if reference is not None:
        names = ["ae/config.yaml", "ae/model.pth"]
        if "latent_dir" in config:
            names.append("stats.pth")
        identities["artifacts"] = {}
        for name in names:
            path = Path(reference) / name
            stat = path.stat()
            identities["artifacts"][name] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    conflict = [None]
    if rank == 0 and output.exists() and any(output.iterdir()) and not resume:
        conflict[0] = f"Output exists: {output}. Use --resume or a different output_dir"
    if world > 1:
        dist.broadcast_object_list(conflict, src=0)
    if conflict[0]:
        raise FileExistsError(conflict[0])
    output.mkdir(parents=True, exist_ok=True)
    state = None
    if resume:
        state = torch.load(output / "training_state.pth", map_location="cpu", weights_only=False)
        if state["config"] != resolved or state["inputs"] != identities or state["world_size"] != world:
            raise ValueError("Resume requires the same configuration, input files and world size")
    if rank == 0:
        OmegaConf.save(config, output / "config.yaml")
        json_write(identities, output / "inputs.json")
    problem.to(device)
    optimizer, scheduler = optimizer_for(problem.model, config)
    scaler = torch.amp.GradScaler(device.type, enabled=config.train.precision == "float16")
    step, best, cursor = 0, float("inf"), (0, 0)
    seed_all(int(config.seed) + rank)
    if state is not None:
        problem.model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best, cursor = state["step"], state["best_loss"], state["cursor"]
        restore_rng(state["rng"][rank], device)
        del state
    model = DistributedDataParallel(
        problem, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False
    ) if world > 1 else problem
    stream = BatchStream(
        training, micro, rank, world, int(config.train.num_workers), int(config.seed), cursor
    )
    end = int(config.train.update_steps)
    if stop_after is not None:
        end = min(end, int(stop_after))
    if end <= step:
        if rank == 0:
            print(f"Already at step {step}; requested stop is {end}", flush=True)
        return
    def save_state():
        local_rng = rng_state(device)
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, local_rng)
        else:
            states[0] = local_rng
        if rank == 0:
            weights = problem.model.state_dict()
            atomic_save(weights, output / "model_last.pth")
            atomic_save({
                "model": weights, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best_loss": best, "cursor": (stream.epoch, stream.offset),
                "rng": states, "world_size": world, "config": resolved, "inputs": identities,
            }, output / "training_state.pth")
        if world > 1:
            dist.barrier()
    log_interval = int(config.train.log_interval)
    valid_interval = int(config.train.valid_interval)
    save_interval = int(config.train.save_interval)
    if min(log_interval, valid_interval, save_interval) <= 0:
        raise ValueError("Training intervals must be positive optimizer-step counts")
    problem.train()
    while step < end:
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for index in range(accum):
            batch_tensors = move_batch(stream.next(), device)
            sync = model.no_sync() if world > 1 and index < accum - 1 else contextlib.nullcontext()
            with sync:
                with autocast(device, config.train.precision):
                    loss = model(*batch_tensors)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
                scaler.scale(loss / accum).backward()
            total_loss += loss.detach() / accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(problem.model.parameters(), float(config.optim.grad_clip))
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
        step += 1
        record = {"step": step, "samples": step * batch}
        if step % log_interval == 0 or step == end:
            if world > 1:
                dist.all_reduce(total_loss)
            record.update(train_loss=total_loss.item() / world, learning_rate=scheduler.get_last_lr()[0])
        if step % valid_interval == 0 or step == end:
            score = validate(problem, validation, config, device, rank, world)
            record["validation_loss"] = score
            if score < best:
                best = score
                if rank == 0:
                    atomic_save(problem.model.state_dict(), output / "model_best.pth")
        if rank == 0 and len(record) > 2:
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            with (output / "train.log").open("a") as file:
                file.write(line + "\n")
        if step % save_interval == 0 or step == end:
            save_state()
