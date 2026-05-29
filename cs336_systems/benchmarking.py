from __future__ import annotations

import argparse
import json
import math
import statistics
import timeit
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import torch

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW as BasicsAdamW

if TYPE_CHECKING:
    from collections.abc import Iterator


MODEL_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 128, "d_ff": 512, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    model_size: str
    vocab_size: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    rope_theta: float | None
    warmup_steps: int
    steps: int
    device: str
    dtype: str
    autocast_dtype: str | None
    optimizer: str
    lr: float
    weight_decay: float
    seed: int
    compile: bool
    forward_requires_grad: bool
    regenerate_batch: bool
    nvtx: bool
    nvtx_attention: bool
    memory_snapshot: str | None
    memory_history_max_entries: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the CS336 basics Transformer forward/backward/training step.")

    parser.add_argument("--mode", choices=("forward", "backward", "train"), default="train", help="Work to time per step.")
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="small", help="Named model preset. Individual dimension flags override it.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE by passing rope_theta=None.")

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"), help="Parameter dtype for the model.")
    parser.add_argument("--autocast-dtype", default=None, choices=("float16", "bfloat16"), help="Use torch.autocast for forward/loss computation.")
    parser.add_argument("--optimizer", default="basics-adamw", choices=("basics-adamw", "torch-adamw"))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="Wrap the model with torch.compile before benchmarking.")
    parser.add_argument("--forward-requires-grad", action="store_true", help="Keep autograd enabled in forward-only mode.")
    parser.add_argument("--regenerate-batch", action="store_true", help="Generate a fresh random batch before each timed or warmup step.")
    parser.add_argument("--nvtx", action="store_true", help="Annotate benchmark phases with NVTX ranges for Nsight Systems.")
    parser.add_argument("--nvtx-attention", action="store_true", help="Patch basics attention with NVTX subranges for attention scores, softmax, and value matmul.")
    parser.add_argument("--memory-snapshot", default=None, help="Write a PyTorch CUDA memory snapshot pickle for the measured benchmark steps.")
    parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000, help="Maximum memory history entries to keep when --memory-snapshot is enabled.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text summary.")

    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return device


def resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_config(args: argparse.Namespace, device: torch.device) -> BenchmarkConfig:
    preset = MODEL_PRESETS[args.model_size]
    return BenchmarkConfig(
        mode=args.mode,
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        d_model=args.d_model if args.d_model is not None else preset["d_model"],
        d_ff=args.d_ff if args.d_ff is not None else preset["d_ff"],
        num_layers=args.num_layers if args.num_layers is not None else preset["num_layers"],
        num_heads=args.num_heads if args.num_heads is not None else preset["num_heads"],
        rope_theta=None if args.no_rope else args.rope_theta,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        device=device.type,
        dtype=args.dtype,
        autocast_dtype=args.autocast_dtype,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        compile=args.compile,
        forward_requires_grad=args.forward_requires_grad,
        regenerate_batch=args.regenerate_batch,
        nvtx=args.nvtx,
        nvtx_attention=args.nvtx_attention,
        memory_snapshot=args.memory_snapshot,
        memory_history_max_entries=args.memory_history_max_entries,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def nvtx_range(config: BenchmarkConfig, device: torch.device, name: str) -> Iterator[None]:
    if (config.nvtx or config.nvtx_attention) and device.type == "cuda":
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def install_annotated_attention(config: BenchmarkConfig, device: torch.device) -> None:
    if not config.nvtx_attention:
        return

    def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        d_k = K.shape[-1]

        with nvtx_range(config, device, "scaled_dot_product_attention"):
            with nvtx_range(config, device, "attention_scores"):
                attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

            if mask is not None:
                with nvtx_range(config, device, "attention_mask"):
                    attention_scores = torch.where(mask, attention_scores, float("-inf"))

            with nvtx_range(config, device, "attention_softmax"):
                attention_weights = softmax(attention_scores, dim=-1)

            with nvtx_range(config, device, "attention_value_matmul"):
                return torch.matmul(attention_weights, V)

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )
    return input_ids, targets


def make_model(config: BenchmarkConfig, device: torch.device) -> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
    model = model.to(device=device, dtype=resolve_dtype(config.dtype))
    if config.compile:
        model = torch.compile(model)
    return model


def make_optimizer(config: BenchmarkConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "basics-adamw":
        return BasicsAdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == "torch-adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def make_step_fn(
    config: BenchmarkConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Callable[[], torch.Tensor | None]:
    autocast_dtype = resolve_dtype(config.autocast_dtype)

    def autocast_context():
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    if config.mode == "forward":
        model.eval()

        def forward_step() -> None:
            with nvtx_range(config, device, "get_batch"):
                input_ids, _ = batch_fn()
            grad_context = nullcontext() if config.forward_requires_grad else torch.no_grad()
            with grad_context, autocast_context(), nvtx_range(config, device, "forward"):
                model(input_ids)

        return forward_step

    model.train()

    def backward_or_train_step() -> torch.Tensor:
        if optimizer is None:
            raise RuntimeError("optimizer is required for backward/train modes")

        with nvtx_range(config, device, "get_batch"):
            input_ids, targets = batch_fn()
        with nvtx_range(config, device, "zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            with nvtx_range(config, device, "forward"):
                logits = model(input_ids)
            with nvtx_range(config, device, "loss"):
                loss = cross_entropy(logits, targets)
        with nvtx_range(config, device, "backward"):
            loss.backward()
        if config.mode == "train":
            with nvtx_range(config, device, "optimizer_step"):
                optimizer.step()
        return loss.detach()

    return backward_or_train_step


def run_benchmark(config: BenchmarkConfig, device: torch.device) -> dict[str, Any]:
    if config.memory_snapshot is not None and device.type != "cuda":
        raise RuntimeError("--memory-snapshot requires a CUDA device.")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    install_annotated_attention(config, device)
    model = make_model(config, device)
    optimizer = make_optimizer(config, model) if config.mode in {"backward", "train"} else None
    static_batch = make_batch(config, device)

    def batch_fn() -> tuple[torch.Tensor, torch.Tensor]:
        return make_batch(config, device) if config.regenerate_batch else static_batch

    step = make_step_fn(config, model, optimizer, batch_fn, device)

    last_loss: float | None = None
    with nvtx_range(config, device, "warmup"):
        for step_idx in range(config.warmup_steps):
            with nvtx_range(config, device, f"warmup_step_{step_idx}"):
                result = step()
                synchronize(device)
                if result is not None:
                    last_loss = float(result.cpu())

    synchronize(device)
    timings_s: list[float] = []
    if config.memory_snapshot is not None:
        snapshot_path = Path(config.memory_snapshot)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=config.memory_history_max_entries, clear_history=True)
    try:
        with nvtx_range(config, device, "benchmark"):
            for step_idx in range(config.steps):
                with nvtx_range(config, device, f"step_{step_idx}"):
                    start = timeit.default_timer()
                    result = step()
                    synchronize(device)
                    elapsed = timeit.default_timer() - start
                    if result is not None:
                        last_loss = float(result.cpu())
                    timings_s.append(elapsed)
    finally:
        if config.memory_snapshot is not None:
            torch.cuda.memory._dump_snapshot(config.memory_snapshot)
            torch.cuda.memory._record_memory_history(enabled=None)

    peak_memory_bytes: int | None = None
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)

    mean_s = statistics.fmean(timings_s) if timings_s else float("nan")
    std_s = statistics.stdev(timings_s) if len(timings_s) > 1 else 0.0

    return {
        "config": asdict(config),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "mean_s": mean_s,
        "std_s": std_s,
        "min_s": min(timings_s) if timings_s else float("nan"),
        "max_s": max(timings_s) if timings_s else float("nan"),
        "timings_s": timings_s,
        "last_loss": last_loss,
        "peak_memory_bytes": peak_memory_bytes,
    }


def print_text_summary(results: dict[str, Any]) -> None:
    config = results["config"]
    print(f"mode: {config['mode']}")
    print(
        "model: "
        f"{config['model_size']} "
        f"(layers={config['num_layers']}, d_model={config['d_model']}, d_ff={config['d_ff']}, heads={config['num_heads']})"
    )
    print(f"batch/context/vocab: {config['batch_size']}/{config['context_length']}/{config['vocab_size']}")
    print(f"device/dtype/autocast: {config['device']}/{config['dtype']}/{config['autocast_dtype']}")
    print(f"parameters: {results['num_parameters']:,}")
    print(f"warmup/steps: {config['warmup_steps']}/{config['steps']}")
    print(f"mean: {results['mean_s'] * 1_000:.3f} ms")
    print(f"std: {results['std_s'] * 1_000:.3f} ms")
    print(f"min/max: {results['min_s'] * 1_000:.3f} / {results['max_s'] * 1_000:.3f} ms")
    if results["last_loss"] is not None:
        print(f"last_loss: {results['last_loss']:.6f}")
    if results["peak_memory_bytes"] is not None:
        print(f"peak_cuda_memory: {results['peak_memory_bytes'] / 1024**3:.3f} GiB")
    if config["memory_snapshot"] is not None:
        print(f"memory_snapshot: {config['memory_snapshot']}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = make_config(args, device)
    results = run_benchmark(config, device)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_text_summary(results)


if __name__ == "__main__":
    main()
