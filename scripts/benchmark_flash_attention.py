from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import triton.testing

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmarking import resolve_device, resolve_dtype, synchronize
from cs336_systems.flash_attention import FlashAttentionTritonFunction


AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FlashBenchmarkResult:
    implementation: str
    dtype: str
    sequence_length: int
    d_model: int
    batch_size: int
    q_tile_size: int
    k_tile_size: int
    forward_ms: float | None
    backward_ms: float | None
    forward_backward_ms: float | None
    status: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark causal batch-size-1 PyTorch attention against the assignment's Triton "
            "FlashAttention implementation using triton.testing.do_bench."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Assignment default is 1.")
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        help="Sequence lengths to sweep.",
    )
    parser.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128], help="Embedding dimensions to sweep.")
    parser.add_argument("--dtypes", choices=("bfloat16", "float32"), nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--implementations", choices=("pytorch", "triton"), nargs="+", default=["pytorch", "triton"])
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=25, help="Warmup duration passed to triton.testing.do_bench.")
    parser.add_argument("--rep-ms", type=int, default=100, help="Measurement duration passed to triton.testing.do_bench.")
    parser.add_argument("--q-tile-size", type=int, default=16, help="Tile size used by FlashAttentionTritonFunction.")
    parser.add_argument("--k-tile-size", type=int, default=16, help="Tile size used by FlashAttentionTritonFunction.")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path. Results are always printed to stdout.")
    return parser.parse_args()


def make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, sequence_length, d_model)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device)
    return positions[:, None] >= positions[None, :]


def make_attention_fn(implementation: str, causal_mask: torch.Tensor) -> AttentionFn:
    if implementation == "pytorch":

        def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return scaled_dot_product_attention(q, k, v, causal_mask)

        return pytorch_attention

    if implementation == "triton":

        def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return FlashAttentionTritonFunction.apply(q, k, v, True)

        return triton_attention

    raise ValueError(f"unknown implementation: {implementation}")


def clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def is_oom_error(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def bench_ms(fn: Callable[[], object], warmup_ms: int, rep_ms: int) -> float:
    result = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean")
    return float(result)


def benchmark_forward(
    attention_fn: AttentionFn,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q, k, v = make_inputs(
        batch_size=batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )

    def run_forward() -> torch.Tensor:
        with torch.no_grad():
            return attention_fn(q, k, v)

    return bench_ms(run_forward, warmup_ms, rep_ms)


def benchmark_backward(
    attention_fn: AttentionFn,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q, k, v = make_inputs(
        batch_size=batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_output = torch.randn_like(q)
    output = attention_fn(q, k, v)
    synchronize(device)

    def run_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.autograd.grad(output, (q, k, v), grad_output, retain_graph=True)

    return bench_ms(run_backward, warmup_ms, rep_ms)


def benchmark_forward_backward(
    attention_fn: AttentionFn,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q, k, v = make_inputs(
        batch_size=batch_size,
        sequence_length=sequence_length,
        d_model=d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_output = torch.randn_like(q)

    def run_forward_backward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = attention_fn(q, k, v)
        return torch.autograd.grad(output, (q, k, v), grad_output)

    return bench_ms(run_forward_backward, warmup_ms, rep_ms)


def benchmark_one(
    args: argparse.Namespace,
    *,
    implementation: str,
    dtype_name: str,
    dtype: torch.dtype,
    sequence_length: int,
    d_model: int,
    device: torch.device,
) -> FlashBenchmarkResult:
    try:
        causal_mask = make_causal_mask(sequence_length, device)
        attention_fn = make_attention_fn(implementation, causal_mask)

        forward_ms = benchmark_forward(
            attention_fn,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            d_model=d_model,
            device=device,
            dtype=dtype,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        clear_cuda(device)

        backward_ms = benchmark_backward(
            attention_fn,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            d_model=d_model,
            device=device,
            dtype=dtype,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        clear_cuda(device)

        forward_backward_ms = benchmark_forward_backward(
            attention_fn,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            d_model=d_model,
            device=device,
            dtype=dtype,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
        clear_cuda(device)
    except Exception as error:
        clear_cuda(device)
        status = "OOM" if is_oom_error(error) else "error"
        return FlashBenchmarkResult(
            implementation=implementation,
            dtype=dtype_name,
            sequence_length=sequence_length,
            d_model=d_model,
            batch_size=args.batch_size,
            q_tile_size=args.q_tile_size,
            k_tile_size=args.k_tile_size,
            forward_ms=None,
            backward_ms=None,
            forward_backward_ms=None,
            status=status,
            error=str(error).splitlines()[0],
        )

    return FlashBenchmarkResult(
        implementation=implementation,
        dtype=dtype_name,
        sequence_length=sequence_length,
        d_model=d_model,
        batch_size=args.batch_size,
        q_tile_size=args.q_tile_size,
        k_tile_size=args.k_tile_size,
        forward_ms=forward_ms,
        backward_ms=backward_ms,
        forward_backward_ms=forward_backward_ms,
        status="ok",
    )


def format_latency(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def result_to_row(result: FlashBenchmarkResult) -> dict[str, str | int]:
    return {
        "implementation": result.implementation,
        "dtype": result.dtype,
        "sequence_length": result.sequence_length,
        "d_model": result.d_model,
        "batch_size": result.batch_size,
        "q_tile_size": result.q_tile_size,
        "k_tile_size": result.k_tile_size,
        "forward_ms": format_latency(result.forward_ms),
        "backward_ms": format_latency(result.backward_ms),
        "forward_backward_ms": format_latency(result.forward_backward_ms),
        "status": result.status,
        "error": result.error or "",
    }


def write_csv(results: Iterable[FlashBenchmarkResult], output: Path | None) -> None:
    fieldnames = [
        "implementation",
        "dtype",
        "sequence_length",
        "d_model",
        "batch_size",
        "q_tile_size",
        "k_tile_size",
        "forward_ms",
        "backward_ms",
        "forward_backward_ms",
        "status",
        "error",
    ]
    rows = [result_to_row(result) for result in results]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            file_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            file_writer.writeheader()
            file_writer.writerows(rows)


def make_csv_writer(handle: object) -> csv.DictWriter:
    fieldnames = [
        "implementation",
        "dtype",
        "sequence_length",
        "d_model",
        "batch_size",
        "q_tile_size",
        "k_tile_size",
        "forward_ms",
        "backward_ms",
        "forward_backward_ms",
        "status",
        "error",
    ]
    return csv.DictWriter(handle, fieldnames=fieldnames)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark uses Triton do_bench and FlashAttention kernels, so it requires CUDA.")

    FlashAttentionTritonFunction.Q_TILE_SIZE = args.q_tile_size
    FlashAttentionTritonFunction.K_TILE_SIZE = args.k_tile_size
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    stdout_writer = make_csv_writer(sys.stdout)
    stdout_writer.writeheader()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    output_handle = args.output.open("w", newline="") if args.output is not None else None
    file_writer = make_csv_writer(output_handle) if output_handle is not None else None

    try:
        if file_writer is not None:
            file_writer.writeheader()

        for implementation, dtype_name, sequence_length, d_model in itertools.product(args.implementations, args.dtypes, args.sequence_lengths, args.d_models):
            dtype = resolve_dtype(dtype_name)
            if dtype is None:
                raise ValueError(f"unknown dtype: {dtype_name}")

            print(
                f"benchmarking implementation={implementation} dtype={dtype_name} sequence_length={sequence_length} d_model={d_model}",
                file=sys.stderr,
                flush=True,
            )
            result = benchmark_one(
                args,
                implementation=implementation,
                dtype_name=dtype_name,
                dtype=dtype,
                sequence_length=sequence_length,
                d_model=d_model,
                device=device,
            )
            row = result_to_row(result)
            stdout_writer.writerow(row)
            sys.stdout.flush()
            if file_writer is not None and output_handle is not None:
                file_writer.writerow(row)
                output_handle.flush()
    finally:
        if output_handle is not None:
            output_handle.close()


if __name__ == "__main__":
    main()
