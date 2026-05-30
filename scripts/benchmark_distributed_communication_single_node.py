#!/usr/bin/env python3
"""Benchmark single-node distributed all-reduce communication.

Example GPU run:
    uv run python scripts/benchmark_distributed_communication_single_node.py 2

Local CPU smoke test:
    uv run python scripts/benchmark_distributed_communication_single_node.py 2 --backend gloo --device cpu --sizes-mb 1 --steps 2
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BYTES_PER_FLOAT32 = 4
DEFAULT_SIZES_MB = (1, 10, 100, 1024)


@dataclass(frozen=True)
class SizeResult:
    size_mb: int
    numel: int
    rank_times_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.rank_times_ms)

    @property
    def std_ms(self) -> float:
        if len(self.rank_times_ms) == 1:
            return 0.0
        return statistics.stdev(self.rank_times_ms)

    @property
    def min_ms(self) -> float:
        return min(self.rank_times_ms)

    @property
    def max_ms(self) -> float:
        return max(self.rank_times_ms)

    @property
    def effective_gib_per_s(self) -> float:
        seconds = self.max_ms / 1000
        if seconds == 0:
            return float("inf")
        return (self.size_mb / 1024) / seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark torch.distributed all-reduce on one node.")
    parser.add_argument("num_gpus", type=int, help="Number of GPU worker processes to launch. With --device cpu, this is the number of CPU processes.")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=list(DEFAULT_SIZES_MB), help="Float32 tensor sizes in MiB to benchmark.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup all-reduce iterations before timing.")
    parser.add_argument("--steps", type=int, default=20, help="Measured all-reduce iterations.")
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto", help="Distributed backend.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Tensor device to benchmark.")
    parser.add_argument("--master-addr", default="127.0.0.1", help="MASTER_ADDR for the local process group.")
    parser.add_argument("--master-port", default="29500", help="MASTER_PORT for the local process group.")
    return parser.parse_args()


def resolve_backend_and_device(requested_backend: str, requested_device: str, world_size: int) -> tuple[str, str]:
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if requested_device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = requested_device

    if device == "cuda" and cuda_count < world_size:
        raise RuntimeError(f"Requested {world_size} CUDA workers, but only {cuda_count} CUDA devices are available.")

    if requested_backend == "auto":
        backend = "nccl" if device == "cuda" else "gloo"
    else:
        backend = requested_backend

    if backend == "nccl" and device != "cuda":
        raise RuntimeError("NCCL requires --device cuda.")

    return backend, device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_numel(size_mb: int) -> int:
    return size_mb * 1024 * 1024 // BYTES_PER_FLOAT32


def benchmark_size(size_mb: int, warmup_steps: int, steps: int, device: torch.device) -> SizeResult:
    tensor = torch.ones(tensor_numel(size_mb), dtype=torch.float32, device=device)

    dist.barrier()
    for _ in range(warmup_steps):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    synchronize(device)
    dist.barrier()

    start = time.perf_counter()
    for _ in range(steps):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / steps

    rank_times: list[float | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rank_times, elapsed_ms)
    return SizeResult(size_mb=size_mb, numel=tensor.numel(), rank_times_ms=[float(t) for t in rank_times])


def worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    backend, device_type = resolve_backend_and_device(args.backend, args.device, world_size)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        if rank == 0:
            print(f"backend={backend} device={device_type} world_size={world_size} warmup_steps={args.warmup_steps} steps={args.steps}")
            print("size_mib,numel,mean_ms,std_ms,min_ms,max_ms,effective_gib_per_s")

        for size_mb in args.sizes_mb:
            result = benchmark_size(size_mb=size_mb, warmup_steps=args.warmup_steps, steps=args.steps, device=device)
            if rank == 0:
                print(f"{result.size_mb},{result.numel},{result.mean_ms:.3f},{result.std_ms:.3f},{result.min_ms:.3f},{result.max_ms:.3f},{result.effective_gib_per_s:.3f}")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.num_gpus <= 0:
        raise ValueError("num_gpus must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")

    mp.spawn(worker, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)


if __name__ == "__main__":
    main()
