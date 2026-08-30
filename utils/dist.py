"""Minimal DDP plumbing for the three training scripts (branch OBQA-qwen).

Launched under `torchrun --nproc_per_node=N`, every rank holds a full model
replica and sees 1/N of each batch (DistributedSampler); gradients are averaged
by DDP, so `per-rank batch x N x grad_accum` is the effective batch. Without
torchrun (WORLD_SIZE unset) everything degrades to the single-process behaviour
that produced all Granite artefacts: plain DataLoader(shuffle=True), no
wrapping, rank 0 == the only process.
"""
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler


class DistInfo:
    def __init__(self, rank, world, local_rank):
        self.rank, self.world, self.local_rank = rank, world, local_rank

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.world > 1

    @property
    def device(self):
        return f"cuda:{self.local_rank}"


def init_distributed():
    """-> DistInfo. Initialises NCCL when launched by torchrun (WORLD_SIZE > 1)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"[DDP] rank {rank}/{world} local_rank {local_rank} device cuda:{local_rank}")
    return DistInfo(rank, world, local_rank)


def make_loader(dataset, batch_size, collate_fn, shuffle, seed, info: DistInfo):
    """DataLoader that is the legacy loader in single-process mode and a
    DistributedSampler-backed one under DDP. Call `set_epoch(loader, epoch)` at
    every epoch start so the per-epoch shuffle differs across epochs."""
    if not info.distributed:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    sampler = DistributedSampler(dataset, num_replicas=info.world, rank=info.rank, shuffle=shuffle,
                                 seed=seed, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler)


def set_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def wrap_ddp(model, info: DistInfo):
    if not info.distributed:
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    # All trainable params (LoRA A/B, or the router heads) take part in every
    # forward, so no unused-parameter search is needed.
    return DDP(model, device_ids=[info.local_rank], output_device=info.local_rank, find_unused_parameters=False)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: float, info: DistInfo) -> float:
    """Mean of a python float across ranks (identical early-stop decisions)."""
    if not info.distributed:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / info.world)


def barrier(info: DistInfo):
    if info.distributed:
        dist.barrier()


def cleanup(info: DistInfo):
    if info.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
