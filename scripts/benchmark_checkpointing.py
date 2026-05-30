from __future__ import annotations

import argparse
import json
import statistics
import timeit
from dataclasses import asdict, dataclass

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmarking import MODEL_PRESETS, resolve_device, resolve_dtype, synchronize


@dataclass(frozen=True)
class CheckpointingResult:
    chunk_size: int
    warmup_steps: int
    steps: int
    mean_s: float
    std_s: float
    peak_memory_gib: float
    last_loss: float


class OneLevelCheckpointedTransformerLM(torch.nn.Module):
    """BasicsTransformerLM wrapper that checkpoints contiguous chunks of Transformer blocks."""

    def __init__(self, model: BasicsTransformerLM, chunk_size: int):
        super().__init__()
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.model = model
        self.chunk_size = chunk_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.token_embeddings(input_ids)

        for start in range(0, len(self.model.layers), self.chunk_size):
            layers = self.model.layers[start : start + self.chunk_size]

            def run_chunk(hidden: torch.Tensor, layers: torch.nn.ModuleList = layers) -> torch.Tensor:
                for layer in layers:
                    hidden = layer(hidden)
                return hidden

            x = checkpoint(run_chunk, x, use_reentrant=False)

        x = self.model.ln_final(x)
        return self.model.lm_head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep one-level activation checkpointing chunk sizes.")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="xl")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def make_model(args: argparse.Namespace, device: torch.device, chunk_size: int) -> OneLevelCheckpointedTransformerLM:
    preset = MODEL_PRESETS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=preset["d_model"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        d_ff=preset["d_ff"],
        rope_theta=10_000.0,
    )
    model = model.to(device=device, dtype=resolve_dtype(args.dtype))
    return OneLevelCheckpointedTransformerLM(model, chunk_size=chunk_size)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    return input_ids, targets


def benchmark_chunk_size(args: argparse.Namespace, device: torch.device, chunk_size: int) -> CheckpointingResult:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()

    model = make_model(args, device, chunk_size)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    input_ids, targets = make_batch(args, device)

    last_loss = float("nan")

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    model.train()
    for _ in range(args.warmup_steps):
        last_loss = float(step().cpu())
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_s: list[float] = []
    for _ in range(args.steps):
        start = timeit.default_timer()
        last_loss = float(step().cpu())
        synchronize(device)
        timings_s.append(timeit.default_timer() - start)

    peak_memory_gib = float("nan")
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    return CheckpointingResult(
        chunk_size=chunk_size,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        mean_s=statistics.fmean(timings_s),
        std_s=statistics.stdev(timings_s) if len(timings_s) > 1 else 0.0,
        peak_memory_gib=peak_memory_gib,
        last_loss=last_loss,
    )


def print_text(results: list[CheckpointingResult]) -> None:
    fastest = min(result.mean_s for result in results)
    print("chunk_size,peak_memory_gib,mean_s,std_s,relative_step_time,last_loss")
    for result in results:
        print(
            f"{result.chunk_size},"
            f"{result.peak_memory_gib:.3f},"
            f"{result.mean_s:.3f},"
            f"{result.std_s:.3f},"
            f"{result.mean_s / fastest:.3f},"
            f"{result.last_loss:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for CUDA memory profiling.")

    results = [benchmark_chunk_size(args, device, chunk_size) for chunk_size in args.chunk_sizes]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_text(results)


if __name__ == "__main__":
    main()
