#!/usr/bin/env python3
"""Profile optimizer state sharding memory and speed for the xl DDP setup.

Assignment-style run on a 2-GPU node:
    uv run python scripts/benchmark_optimizer_state_sharding.py

Tiny CPU smoke test:
    uv run python scripts/benchmark_optimizer_state_sharding.py --device cpu --backend gloo --model-size tiny --context-length 8 --vocab-size 100 --global-batch-size 2 --steps 1 --warmup-steps 0
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import ddp_on_after_backward, get_ddp
from cs336_systems.sharded_optimizer import get_sharded_optimizer


WORLD_SIZE = 2
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark optimizer state sharding for 1 node x 2 GPUs.")
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
    device_type = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_type == "auto":
        device_type = "cpu"

    if device_type == "cuda" and torch.cuda.device_count() < WORLD_SIZE:
        raise RuntimeError(f"Requested {WORLD_SIZE} CUDA workers, but only {torch.cuda.device_count()} CUDA devices are available.")

    backend = "nccl" if args.backend == "auto" and device_type == "cuda" else args.backend
    if backend == "auto":
        backend = "gloo"
    if backend == "nccl" and device_type != "cuda":
        raise RuntimeError("NCCL requires CUDA tensors.")
    return backend, device_type


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_gib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    return allocated, peak


def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    cfg = MODEL_PRESETS[args.model_size]
    return BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10_000.0,
    ).to(device)


def make_batch(args: argparse.Namespace, local_batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (local_batch_size, args.context_length), device=device)
    return input_ids, targets


def make_optimizer(mode: str, model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if mode == "regular":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if mode == "sharded":
        return get_sharded_optimizer(model.parameters(), torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"unknown optimizer mode: {mode}")


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    local_batch_size: int,
    device: torch.device,
) -> float:
    input_ids, targets = make_batch(args, local_batch_size, device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()
    ddp_on_after_backward(model, optimizer)
    optimizer.step()
    synchronize(device)
    return time.perf_counter() - start


def collect_rank_dict(rank_result: dict[str, Any]) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, rank_result)
    return [result for result in gathered if result is not None]


def run_mode(mode: str, rank: int, args: argparse.Namespace, device: torch.device) -> None:
    local_batch_size = args.global_batch_size // WORLD_SIZE
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed + rank)
    model = get_ddp(make_model(args, device))
    optimizer = make_optimizer(mode, model, args)
    after_init_alloc, after_init_peak = memory_gib(device)

    optimizer.zero_grad(set_to_none=True)
    input_ids, targets = make_batch(args, local_batch_size, device)
    logits = model(input_ids)
    loss = cross_entropy(logits, targets)
    loss.backward()
    ddp_on_after_backward(model, optimizer)
    before_step_alloc, before_step_peak = memory_gib(device)

    optimizer.step()
    after_step_alloc, after_step_peak = memory_gib(device)

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, args, local_batch_size, device)
    dist.barrier()

    step_times_ms = [1000 * train_step(model, optimizer, args, local_batch_size, device) for _ in range(args.steps)]
    dist.barrier()

    rank_result = {
        "mode": mode,
        "rank": rank,
        "after_init_alloc_gib": after_init_alloc,
        "after_init_peak_gib": after_init_peak,
        "before_step_alloc_gib": before_step_alloc,
        "before_step_peak_gib": before_step_peak,
        "after_step_alloc_gib": after_step_alloc,
        "after_step_peak_gib": after_step_peak,
        "step_times_ms": step_times_ms,
    }
    results = collect_rank_dict(rank_result)
    if rank == 0:
        print_mode_summary(mode, results)

    del optimizer, model, input_ids, targets, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dist.barrier()


def max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def fmt_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def print_mode_summary(mode: str, results: list[dict[str, Any]]) -> None:
    print(f"\noptimizer_mode={mode}")
    print("memory_point,max_allocated_gib,max_peak_gib")
    for point in ("after_init", "before_step", "after_step"):
        max_alloc = max_optional([result[f"{point}_alloc_gib"] for result in results])
        max_peak = max_optional([result[f"{point}_peak_gib"] for result in results])
        print(f"{point},{fmt_optional(max_alloc)},{fmt_optional(max_peak)}")

    all_times = [time_ms for result in results for time_ms in result["step_times_ms"]]
    mean = statistics.fmean(all_times)
    std = statistics.stdev(all_times) if len(all_times) > 1 else 0.0
    print("timing_metric,mean_ms,std_ms,min_ms,max_ms")
    print(f"train_step,{mean:.3f},{std:.3f},{min(all_times):.3f},{max(all_times):.3f}")


def worker(rank: int, args: argparse.Namespace) -> None:
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    backend, device_type = resolve_backend_and_device(args)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=WORLD_SIZE)
    try:
        if args.global_batch_size % WORLD_SIZE != 0:
            raise ValueError("--global-batch-size must be divisible by 2.")
        if rank == 0:
            print(f"backend={backend} device={device_type} world_size=2 model_size={args.model_size}")
        for mode in ("regular", "sharded"):
            run_mode(mode, rank, args, device)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    mp.spawn(worker, args=(args,), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
