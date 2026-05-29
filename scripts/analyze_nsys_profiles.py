#!/usr/bin/env python3
"""Summarize Nsight Systems SQLite exports for Problem 1.2.

The profiling commands export both `.nsys-rep` and `.sqlite` files. This script
reads the SQLite export and ignores warmup by restricting kernel summaries to
the measured `step_0` NVTX range.
"""

from __future__ import annotations

import argparse
import glob
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


NS_IN_MS = 1_000_000
ATTENTION_RANGES = ("attention_scores", "attention_softmax", "attention_value_matmul")
MATMUL_PATTERNS = (
    "gemm",
    "sgemm",
    "xmma",
    "cublas",
    "cutlass",
    "matmul",
    "bmm",
    "mm_kernel",
)


@dataclass(frozen=True)
class ProfileName:
    model: str
    context: int
    mode: str


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    @property
    def ms(self) -> float:
        return (self.end - self.start) / NS_IN_MS


@dataclass(frozen=True)
class KernelSummary:
    name: str
    count: int
    ms: float


def profile_name(path: Path) -> ProfileName:
    match = re.match(r"(?P<model>.+)_ctx(?P<context>\d+)_(?P<mode>forward|backward|train)\.sqlite$", path.name)
    if not match:
        raise ValueError(f"Could not parse profile filename: {path}")
    return ProfileName(
        model=match.group("model"),
        context=int(match.group("context")),
        mode=match.group("mode"),
    )


def is_matmul_kernel(name: str) -> bool:
    lower = name.lower()
    return any(pattern in lower for pattern in MATMUL_PATTERNS)


def measured_step(conn: sqlite3.Connection) -> Range:
    row = conn.execute("select start, end from NVTX_EVENTS where text = 'step_0' and end is not null").fetchone()
    if row is None:
        raise ValueError("Profile does not contain a completed NVTX range named step_0")
    return Range(start=row[0], end=row[1])


def named_range_ms(conn: sqlite3.Connection, name: str, containing: Range) -> float | None:
    rows = conn.execute(
        """
        select start, end
        from NVTX_EVENTS
        where text = ?
          and end is not null
          and start >= ?
          and end <= ?
        """,
        (name, containing.start, containing.end),
    ).fetchall()
    if not rows:
        return None
    return sum(end - start for start, end in rows) / NS_IN_MS


def kernel_summaries(conn: sqlite3.Connection, containing: Range) -> list[KernelSummary]:
    rows = conn.execute(
        """
        select strings.value, count(*) as launches, sum(k.end - k.start) / 1000000.0 as ms
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds strings on strings.id = k.demangledName
        where k.start >= ?
          and k.end <= ?
        group by strings.value
        order by ms desc
        """,
        (containing.start, containing.end),
    ).fetchall()
    return [KernelSummary(name=name, count=count, ms=ms) for name, count, ms in rows]


def ranges_for_text(conn: sqlite3.Connection, text: str, containing: Range) -> list[Range]:
    rows = conn.execute(
        """
        select start, end
        from NVTX_EVENTS
        where text = ?
          and end is not null
          and start >= ?
          and end <= ?
        order by start
        """,
        (text, containing.start, containing.end),
    ).fetchall()
    return [Range(start=start, end=end) for start, end in rows]


def launched_kernel_time_in_ranges(conn: sqlite3.Connection, ranges: list[Range]) -> float:
    """Return CUDA kernel time for kernels launched by CUDA APIs in ranges.

    Nested NVTX ranges usually do not synchronize before they close. Joining
    kernels through CUDA runtime correlation IDs tracks the kernels launched
    while the CPU was inside the range, even if the GPU executes them after the
    NVTX range end timestamp.
    """
    total_ns = 0
    for nvtx_range in ranges:
        row = conn.execute(
            """
            select coalesce(sum(k.end - k.start), 0)
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId = r.correlationId
            where r.start >= ?
              and r.end <= ?
            """,
            (nvtx_range.start, nvtx_range.end),
        ).fetchone()
        total_ns += row[0] or 0
    return total_ns / NS_IN_MS


def shorten(name: str, limit: int) -> str:
    name = " ".join(name.split())
    if len(name) <= limit:
        return name
    return name[: limit - 3] + "..."


def summarize_profile(path: Path, top: int, name_width: int) -> None:
    parsed = profile_name(path)
    conn = sqlite3.connect(path)
    step = measured_step(conn)
    kernels = kernel_summaries(conn, step)

    total_kernel_ms = sum(kernel.ms for kernel in kernels)
    matmul_ms = sum(kernel.ms for kernel in kernels if is_matmul_kernel(kernel.name))
    matmul_pct = 100 * matmul_ms / total_kernel_ms if total_kernel_ms else 0.0
    top_kernel = kernels[0] if kernels else KernelSummary(name="--", count=0, ms=0.0)
    non_matmul = [kernel for kernel in kernels if not is_matmul_kernel(kernel.name)]

    print(f"## {parsed.model} ctx={parsed.context} mode={parsed.mode}")
    print(f"- measured step wall time: {step.ms:.3f} ms")
    forward_ms = named_range_ms(conn, "forward", step)
    if forward_ms is not None:
        print(f"- forward NVTX wall time inside step: {forward_ms:.3f} ms")
    print(f"- cumulative CUDA kernel time inside step: {total_kernel_ms:.3f} ms")
    print(f"- matmul-like CUDA kernel time: {matmul_ms:.3f} ms ({matmul_pct:.1f}% of cumulative kernel time)")
    print(f"- top CUDA kernel: {top_kernel.ms:.3f} ms across {top_kernel.count} launches")
    print(f"  `{shorten(top_kernel.name, name_width)}`")

    if parsed.mode == "forward":
        print("- top non-matmul CUDA kernels:")
        for kernel in non_matmul[:top]:
            print(f"  - {kernel.ms:.3f} ms across {kernel.count} launches: `{shorten(kernel.name, name_width)}`")

    attention_rows = []
    for range_name in ATTENTION_RANGES:
        ranges = ranges_for_text(conn, range_name, step)
        if ranges:
            attention_rows.append(
                (
                    range_name,
                    len(ranges),
                    sum(nvtx_range.ms for nvtx_range in ranges),
                    launched_kernel_time_in_ranges(conn, ranges),
                )
            )
    if attention_rows:
        print("- attention NVTX subranges:")
        for range_name, count, wall_ms, kernel_ms in attention_rows:
            print(f"  - {range_name}: {wall_ms:.3f} ms wall / {kernel_ms:.3f} ms CUDA kernels across {count} ranges")
    print()


def comparison_table(paths: list[Path]) -> None:
    rows = []
    top_by_profile: dict[tuple[str, int, str], KernelSummary] = {}
    for path in paths:
        parsed = profile_name(path)
        conn = sqlite3.connect(path)
        step = measured_step(conn)
        kernels = kernel_summaries(conn, step)
        total_kernel_ms = sum(kernel.ms for kernel in kernels)
        matmul_ms = sum(kernel.ms for kernel in kernels if is_matmul_kernel(kernel.name))
        top_by_profile[(parsed.model, parsed.context, parsed.mode)] = kernels[0]
        rows.append((parsed.model, parsed.context, parsed.mode, step.ms, total_kernel_ms, matmul_ms, 100 * matmul_ms / total_kernel_ms if total_kernel_ms else 0.0))

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    print("# Compact Summary")
    print("| Model | Ctx | Mode | Step wall ms | CUDA kernel ms | Matmul-like ms | Matmul-like % |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for model, context, mode, step_ms, total_kernel_ms, matmul_ms, matmul_pct in rows:
        print(f"| {model} | {context} | {mode} | {step_ms:.3f} | {total_kernel_ms:.3f} | {matmul_ms:.3f} | {matmul_pct:.1f}% |")
    print()

    grouped: dict[tuple[str, int], dict[str, KernelSummary]] = defaultdict(dict)
    for key, kernel in top_by_profile.items():
        model, context, mode = key
        grouped[(model, context)][mode] = kernel

    print("# Forward vs Backward Dominant Kernel Check")
    print("| Model | Ctx | Forward top launches | Forward top ms | Same top in backward? | Backward top launches | Backward top ms |")
    print("|---|---:|---:|---:|---|---:|---:|")
    for (model, context), by_mode in sorted(grouped.items()):
        if "forward" not in by_mode or "backward" not in by_mode:
            continue
        forward = by_mode["forward"]
        backward = by_mode["backward"]
        same = "yes" if forward.name == backward.name else "no"
        print(f"| {model} | {context} | {forward.count} | {forward.ms:.3f} | {same} | {backward.count} | {backward.ms:.3f} |")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles",
        nargs="*",
        default=["profiles/nsys/*_ctx*_*.sqlite"],
        help="SQLite profile files or glob patterns to analyze.",
    )
    parser.add_argument("--top", type=int, default=8, help="Number of non-matmul kernels to show for forward profiles.")
    parser.add_argument("--name-width", type=int, default=150, help="Maximum printed CUDA kernel name width.")
    parser.add_argument("--details", action="store_true", help="Print per-profile details after compact tables.")
    return parser.parse_args()


def expand_profiles(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths), key=lambda path: (profile_name(path).model, profile_name(path).context, profile_name(path).mode))


def main() -> None:
    args = parse_args()
    paths = expand_profiles(args.profiles)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing profile(s): {', '.join(str(path) for path in missing)}")

    comparison_table(paths)
    if args.details:
        print("# Per-Profile Details")
        for path in paths:
            summarize_profile(path, top=args.top, name_width=args.name_width)


if __name__ == "__main__":
    main()
