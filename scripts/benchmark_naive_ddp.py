#!/usr/bin/env python3
"""Benchmark naive DDP training.

Assignment-style run:
    uv run python scripts/benchmark_naive_ddp.py --model-size xl --world-size 2

CPU smoke test:
    uv run python scripts/benchmark_naive_ddp.py --device cpu --backend gloo --model-size tiny --world-size 2 --steps 2 --warmup-steps 1
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from tests.adapters import ddp_on_after_backward, get_ddp


MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training on one node.")
    parser.add_argument("--world-size", type=int, default=2, help="Number of worker processes / GPUs.")
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="xl")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_backend_and_device(args: argparse.Namespace) -> tuple[str, str]:
    if args.device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = args.device

    if device_type == "cuda" and torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"Requested {args.world_size} CUDA workers, but only {torch.cuda.device_count()} CUDA devices are available.")

    if args.backend == "auto":
        backend = "nccl" if device_type == "cuda" else "gloo"
    else:
        backend = args.backend

    if backend == "nccl" and device_type != "cuda":
        raise RuntimeError("NCCL requires CUDA tensors.")

    return backend, device_type


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batch(args: argparse.Namespace, local_batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    return inputs, targets


def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    preset = MODEL_PRESETS[args.model_size]
    return BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=preset["d_model"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        d_ff=preset["d_ff"],
        rope_theta=10_000.0,
    ).to(device)


def train_step(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    local_batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    inputs, targets = make_batch(args, local_batch_size, device)

    step_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()

    synchronize(device)
    comm_start = time.perf_counter()
    ddp_on_after_backward(model, optimizer)
    synchronize(device)
    comm_s = time.perf_counter() - comm_start

    optimizer.step()
    synchronize(device)
    step_s = time.perf_counter() - step_start
    return step_s, comm_s


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
        min(values),
        max(values),
    )


def worker(rank: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    backend, device_type = resolve_backend_and_device(args)

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=args.world_size)
    try:
        if args.global_batch_size % args.world_size != 0:
            raise ValueError("--global-batch-size must be divisible by --world-size.")
        local_batch_size = args.global_batch_size // args.world_size

        torch.manual_seed(args.seed + rank)
        model = get_ddp(make_model(args, device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for _ in range(args.warmup_steps):
            train_step(args, model, optimizer, local_batch_size, device)
        dist.barrier()

        step_times: list[float] = []
        comm_times: list[float] = []
        for _ in range(args.steps):
            step_s, comm_s = train_step(args, model, optimizer, local_batch_size, device)
            step_times.append(step_s)
            comm_times.append(comm_s)
        dist.barrier()

        rank_result = {
            "step_ms": [t * 1000 for t in step_times],
            "comm_ms": [t * 1000 for t in comm_times],
        }
        gathered: list[dict[str, list[float]] | None] = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathered, rank_result)

        if rank == 0:
            all_step = [t for result in gathered if result is not None for t in result["step_ms"]]
            all_comm = [t for result in gathered if result is not None for t in result["comm_ms"]]
            step_mean, step_std, step_min, step_max = summarize(all_step)
            comm_mean, comm_std, comm_min, comm_max = summarize(all_comm)
            print(f"backend={backend} device={device_type} world_size={args.world_size} model_size={args.model_size}")
            print("metric,mean_ms,std_ms,min_ms,max_ms")
            print(f"train_step,{step_mean:.3f},{step_std:.3f},{step_min:.3f},{step_max:.3f}")
            print(f"ddp_communication,{comm_mean:.3f},{comm_std:.3f},{comm_min:.3f},{comm_max:.3f}")
            print(f"communication_fraction,{comm_mean / step_mean:.4f},,,")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")

    mp.spawn(worker, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
