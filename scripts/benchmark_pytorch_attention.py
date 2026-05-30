from __future__ import annotations

import argparse
import itertools
import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmarking import resolve_device, resolve_dtype, synchronize

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True)
class AttentionBenchmarkResult:
    variant: str
    d_model: int
    sequence_length: int
    forward_mean_ms: float | None
    forward_std_ms: float | None
    pre_backward_memory_gib: float | None
    backward_mean_ms: float | None
    backward_std_ms: float | None
    oom: bool
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch scaled dot-product attention over assignment 2 sizes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action="store_true", help="Use a causal attention mask.")
    parser.add_argument("--variants", choices=("eager", "compiled"), nargs="+", default=["eager"])
    return parser.parse_args()


def make_inputs(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, d_model)
    Q = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    K = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    V = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return Q, K, V


def make_mask(sequence_length: int, device: torch.device, causal: bool) -> torch.Tensor | None:
    if not causal:
        return None
    positions = torch.arange(sequence_length, device=device)
    return positions[:, None] >= positions[None, :]


def clear_cuda_after_oom(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def time_forward(
    attention_fn: AttentionFn,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None,
    warmup_steps: int,
    steps: int,
) -> tuple[float, float]:
    Q, K, V = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=False)

    with torch.no_grad():
        for _ in range(warmup_steps):
            attention_fn(Q, K, V, mask)
            synchronize(device)

        timings_s: list[float] = []
        for _ in range(steps):
            start = timeit.default_timer()
            attention_fn(Q, K, V, mask)
            synchronize(device)
            timings_s.append(timeit.default_timer() - start)

    return statistics.fmean(timings_s) * 1_000, statistics.stdev(timings_s) * 1_000 if len(timings_s) > 1 else 0.0


def measure_pre_backward_memory(
    attention_fn: AttentionFn,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None,
) -> float:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    Q, K, V = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=True)
    output = attention_fn(Q, K, V, mask)
    loss = output.sum()
    synchronize(device)

    if device.type == "cuda":
        memory_gib = torch.cuda.memory_allocated(device) / 1024**3
    else:
        memory_gib = float("nan")

    del loss, output, Q, K, V
    clear_cuda_after_oom(device)
    return memory_gib


def time_backward(
    attention_fn: AttentionFn,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None,
    warmup_steps: int,
    steps: int,
) -> tuple[float, float]:
    timings_s: list[float] = []
    for _ in range(warmup_steps):
        Q, K, V = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=True)
        output = attention_fn(Q, K, V, mask)
        output.sum().backward()
        synchronize(device)

    for _ in range(steps):
        Q, K, V = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=True)
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        synchronize(device)
        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        timings_s.append(timeit.default_timer() - start)

    return statistics.fmean(timings_s) * 1_000, statistics.stdev(timings_s) * 1_000 if len(timings_s) > 1 else 0.0


def get_attention_fn(variant: str) -> AttentionFn:
    if variant == "eager":
        return scaled_dot_product_attention
    if variant == "compiled":
        torch._dynamo.reset()
        return torch.compile(scaled_dot_product_attention)
    raise ValueError(f"unknown attention variant: {variant}")


def benchmark_one(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    variant: str,
    d_model: int,
    sequence_length: int,
) -> AttentionBenchmarkResult:
    mask = make_mask(sequence_length, device, args.causal)
    attention_fn = get_attention_fn(variant)
    try:
        forward_mean_ms, forward_std_ms = time_forward(
            attention_fn,
            args.batch_size,
            sequence_length,
            d_model,
            device,
            dtype,
            mask,
            args.warmup_steps,
            args.steps,
        )
        pre_backward_memory_gib = measure_pre_backward_memory(attention_fn, args.batch_size, sequence_length, d_model, device, dtype, mask)
        backward_mean_ms, backward_std_ms = time_backward(
            attention_fn,
            args.batch_size,
            sequence_length,
            d_model,
            device,
            dtype,
            mask,
            args.warmup_steps,
            args.steps,
        )
    except torch.cuda.OutOfMemoryError as exc:
        clear_cuda_after_oom(device)
        return AttentionBenchmarkResult(
            variant=variant,
            d_model=d_model,
            sequence_length=sequence_length,
            forward_mean_ms=None,
            forward_std_ms=None,
            pre_backward_memory_gib=None,
            backward_mean_ms=None,
            backward_std_ms=None,
            oom=True,
            error=str(exc).splitlines()[0],
        )

    return AttentionBenchmarkResult(
        variant=variant,
        d_model=d_model,
        sequence_length=sequence_length,
        forward_mean_ms=forward_mean_ms,
        forward_std_ms=forward_std_ms,
        pre_backward_memory_gib=pre_backward_memory_gib,
        backward_mean_ms=backward_mean_ms,
        backward_std_ms=backward_std_ms,
        oom=False,
    )


def format_optional(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "OOM"
    return f"{value:.{precision}f}"


def print_csv(results: list[AttentionBenchmarkResult]) -> None:
    print("variant,d_model,sequence_length,forward_mean_ms,forward_std_ms,pre_backward_memory_gib,backward_mean_ms,backward_std_ms,status")
    for result in results:
        status = "OOM" if result.oom else "ok"
        print(
            f"{result.variant},"
            f"{result.d_model},"
            f"{result.sequence_length},"
            f"{format_optional(result.forward_mean_ms)},"
            f"{format_optional(result.forward_std_ms)},"
            f"{format_optional(result.pre_backward_memory_gib)},"
            f"{format_optional(result.backward_mean_ms)},"
            f"{format_optional(result.backward_std_ms)},"
            f"{status}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if dtype is None:
        raise RuntimeError("dtype must resolve to a torch dtype")
    if device.type != "cuda":
        raise RuntimeError("The assignment memory benchmark is intended to run on CUDA.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    results: list[AttentionBenchmarkResult] = []
    for variant in args.variants:
        results.extend(
            benchmark_one(args, device, dtype, variant, d_model, sequence_length)
            for d_model, sequence_length in itertools.product(args.d_models, args.sequence_lengths)
        )
    print_csv(results)


if __name__ == "__main__":
    main()
