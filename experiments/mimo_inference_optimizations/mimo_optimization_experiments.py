#!/usr/bin/env python3
"""Small, runnable experiments for MiMo-V2.5 inference optimizations.

This file is intentionally not a MiMo implementation. MiMo-V2.5 is a huge
MoE/multimodal model. Instead, each experiment builds a small PyTorch model or
simulator that isolates one production idea from Xiaomi's inference blog:

1. Hybrid SWA KV/decode: full KV for global layers, window KV for SWA layers.
2. Length bucketing: reduce padding/straggler work during prefill batches.
3. Cache-affinity scheduling: route requests by prefix-hit score and load.
4. Encoder batching: batch image preprocessing + encoder forward calls.
5. MTP-style speculative decode: verify several draft tokens per target call.

Public implementation notes used while writing this:
- vLLM's MiMo-V2 MTP layer combines token embeddings with previous hidden
  states and uses the SWA attention config.
- vLLM's MiMo-V2 omni vision encoder has full-attention blocks and window
  attention blocks, with data-parallel encoder deployment in the recipe.
- SGLang's MiMo-V2.5 recipe exposes SWA and EAGLE/MTP serving flags such as
  --swa-full-tokens-ratio and --enable-multi-layer-eagle.

Run:
    python mimo_optimization_experiments.py --quick --device auto
    python mimo_optimization_experiments.py --experiments hybrid_swa,mtp

Colab:
    Upload this file and run the companion notebook, or run this script directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for the model experiments.\n"
            "Install it with one of:\n"
            "  pip install torch\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu121 torch\n"
            "Colab normally already includes torch; choose Runtime > Change runtime type > GPU."
        ) from exc
    return torch, nn, F


@dataclass
class BenchRow:
    experiment: str
    case: str
    metric: str
    baseline: float
    optimized: float
    speedup: float
    unit: str
    note: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)


def pick_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def median_time_ms(
    fn: Callable[[], Any],
    *,
    torch,
    device,
    repeats: int,
    warmup: int,
) -> float:
    for _ in range(warmup):
        fn()
    sync(torch, device)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(torch, device)
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def bytes_to_mib(num_bytes: float) -> float:
    return num_bytes / (1024.0 * 1024.0)


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    out = ["  ".join(col.ljust(widths[col]) for col in columns)]
    out.append("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        out.append("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    return "\n".join(out)


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def parse_count(value: str) -> int:
    """Parse counts like 4096, 4k, 128k, or 1m using binary units."""
    text = value.strip().lower().replace("_", "")
    if not text:
        raise ValueError("empty count")
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1024
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1024 * 1024
        text = text[:-1]
    return int(float(text) * multiplier)


def parse_count_list(raw: str) -> list[int]:
    values = [parse_count(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("count list cannot be empty")
    return values


# ---------------------------------------------------------------------------
# Experiment 1: Hybrid SWA KV/decode
# ---------------------------------------------------------------------------


def decode_attention(torch, F, q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def run_hybrid_swa_experiment(args) -> list[BenchRow]:
    torch, _, F = require_torch()
    device = pick_device(torch, args.device)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    layers = args.hybrid_layers or (12 if args.quick else 24)
    heads = 4
    head_dim = 32
    window = args.hybrid_window
    full_every = 6  # 5 SWA layers + 1 full layer, like the blog's 5:1 ratio.
    full_layers = [i for i in range(layers) if (i + 1) % full_every == 0]
    swa_layers = [i for i in range(layers) if i not in full_layers]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    itemsize = torch.tensor([], dtype=dtype).element_size()
    lengths = (
        parse_count_list(args.hybrid_lengths)
        if args.hybrid_lengths
        else ([256, 512, 1024] if args.quick else [512, 1024, 2048, 4096])
    )

    rows: list[BenchRow] = []
    printed: list[dict[str, str]] = []

    for seq_len in lengths:
        q_full = torch.randn(layers, 1, heads, 1, head_dim, device=device, dtype=dtype)
        k_full = torch.randn(layers, 1, heads, seq_len, head_dim, device=device, dtype=dtype)
        v_full = torch.randn(layers, 1, heads, seq_len, head_dim, device=device, dtype=dtype)
        swa_len = min(window, seq_len)
        k_swa = torch.randn(len(swa_layers), 1, heads, swa_len, head_dim, device=device, dtype=dtype)
        v_swa = torch.randn(len(swa_layers), 1, heads, swa_len, head_dim, device=device, dtype=dtype)

        def full_decode():
            acc = None
            for layer in range(layers):
                out = decode_attention(torch, F, q_full[layer], k_full[layer], v_full[layer])
                acc = out if acc is None else acc + out
            return acc

        def hybrid_decode():
            acc = None
            swa_idx = 0
            for layer in range(layers):
                if layer in full_layers:
                    out = decode_attention(torch, F, q_full[layer], k_full[layer], v_full[layer])
                else:
                    out = decode_attention(torch, F, q_full[layer], k_swa[swa_idx], v_swa[swa_idx])
                    swa_idx += 1
                acc = out if acc is None else acc + out
            return acc

        full_ms = median_time_ms(
            full_decode,
            torch=torch,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        hybrid_ms = median_time_ms(
            hybrid_decode,
            torch=torch,
            device=device,
            repeats=args.repeats,
            warmup=args.warmup,
        )

        full_kv_bytes = layers * 2 * heads * seq_len * head_dim * itemsize
        hybrid_kv_bytes = (
            len(full_layers) * seq_len + len(swa_layers) * swa_len
        ) * 2 * heads * head_dim * itemsize
        rows.append(
            BenchRow(
                "hybrid_swa",
                f"seq={seq_len}",
                "decode_attention_time",
                full_ms,
                hybrid_ms,
                full_ms / hybrid_ms if hybrid_ms else float("inf"),
                "ms",
                f"{len(full_layers)} full + {len(swa_layers)} SWA, W={window}",
            )
        )
        rows.append(
            BenchRow(
                "hybrid_swa",
                f"seq={seq_len}",
                "kv_cache_memory",
                bytes_to_mib(full_kv_bytes),
                bytes_to_mib(hybrid_kv_bytes),
                full_kv_bytes / hybrid_kv_bytes if hybrid_kv_bytes else float("inf"),
                "MiB",
                "equal KV heads in this toy model",
            )
        )
        printed.append(
            {
                "seq": str(seq_len),
                "full_ms": fmt(full_ms),
                "hybrid_ms": fmt(hybrid_ms),
                "time_speedup": fmt(full_ms / hybrid_ms if hybrid_ms else float("inf")),
                "full_kv_mib": fmt(bytes_to_mib(full_kv_bytes)),
                "hybrid_kv_mib": fmt(bytes_to_mib(hybrid_kv_bytes)),
                "kv_reduction": fmt(full_kv_bytes / hybrid_kv_bytes if hybrid_kv_bytes else float("inf")),
            }
        )

    print("\n[hybrid_swa] full attention vs hybrid SWA decode/KV")
    print(table(printed, ["seq", "full_ms", "hybrid_ms", "time_speedup", "full_kv_mib", "hybrid_kv_mib", "kv_reduction"]))
    return rows


# ---------------------------------------------------------------------------
# Shared tiny transformer components for prefill and encoder experiments
# ---------------------------------------------------------------------------


def build_tiny_block_class(torch, nn, F):
    class TinyBlock(nn.Module):
        def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
            super().__init__()
            self.dim = dim
            self.heads = heads
            self.head_dim = dim // heads
            self.ln1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Linear(dim * mlp_ratio, dim),
            )

        def forward(self, x, causal: bool = False):
            bsz, seq_len, dim = x.shape
            qkv = self.qkv(self.ln1(x))
            qkv = qkv.view(bsz, seq_len, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
            x = x + self.proj(y)
            x = x + self.mlp(self.ln2(x))
            return x

    return TinyBlock


# ---------------------------------------------------------------------------
# Experiment 2: Length bucketing
# ---------------------------------------------------------------------------


def make_batches(lengths: list[int], batch_size: int) -> list[list[int]]:
    return [lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)]


def bucket_lengths(lengths: list[int], batch_size: int, bucket_edges: list[int]) -> list[list[int]]:
    buckets: dict[int, list[int]] = {edge: [] for edge in bucket_edges}
    for length in lengths:
        for edge in bucket_edges:
            if length <= edge:
                buckets[edge].append(length)
                break
        else:
            buckets[bucket_edges[-1]].append(length)
    grouped: list[list[int]] = []
    for edge in bucket_edges:
        grouped.extend(make_batches(sorted(buckets[edge]), batch_size))
    return grouped


def padded_tokens(groups: list[list[int]]) -> int:
    return sum(max(group) * len(group) for group in groups if group)


def run_length_bucketing_experiment(args) -> list[BenchRow]:
    torch, nn, F = require_torch()
    device = pick_device(torch, args.device)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    TinyBlock = build_tiny_block_class(torch, nn, F)

    class TinyPrefillModel(nn.Module):
        def __init__(self, dim: int, heads: int, layers: int):
            super().__init__()
            self.blocks = nn.ModuleList([TinyBlock(dim, heads) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            for block in self.blocks:
                x = block(x, causal=True)
            return self.norm(x)

    dim = 96 if args.quick else 160
    layers = 2 if args.quick else 4
    heads = 4
    batch_size = 8
    n_requests = 48 if args.quick else 128
    candidate_lengths = [64, 96, 128, 192, 256, 384, 512] if args.quick else [
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
    ]
    rng = random.Random(args.seed)
    lengths = [rng.choice(candidate_lengths) for _ in range(n_requests)]
    shuffled = lengths[:]
    rng.shuffle(shuffled)

    random_groups = make_batches(shuffled, batch_size)
    # Tiny analog of Xiaomi's 0-64K / 64K-256K / 256K-1M production buckets.
    bucket_edges = [128, 512, 2048]
    bucketed_groups = bucket_lengths(shuffled, batch_size, bucket_edges)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = TinyPrefillModel(dim, heads, layers).to(device=device, dtype=dtype).eval()

    def prepare(groups: list[list[int]]):
        return [
            torch.randn(len(group), max(group), dim, device=device, dtype=dtype)
            for group in groups
            if group
        ]

    random_batches = prepare(random_groups)
    bucketed_batches = prepare(bucketed_groups)

    @torch.no_grad()
    def run_batches(batches):
        acc = None
        for x in batches:
            y = model(x)
            acc = y[:, -1].mean() if acc is None else acc + y[:, -1].mean()
        return acc

    random_ms = median_time_ms(
        lambda: run_batches(random_batches),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    bucket_ms = median_time_ms(
        lambda: run_batches(bucketed_batches),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    real = sum(lengths)
    random_pad = padded_tokens(random_groups)
    bucket_pad = padded_tokens(bucketed_groups)

    print("\n[length_bucketing] random batches vs length buckets")
    print(
        table(
            [
                {
                    "policy": "random",
                    "ms": fmt(random_ms),
                    "padded_tokens": str(random_pad),
                    "pad_waste": pct((random_pad - real) / random_pad),
                },
                {
                    "policy": "bucketed",
                    "ms": fmt(bucket_ms),
                    "padded_tokens": str(bucket_pad),
                    "pad_waste": pct((bucket_pad - real) / bucket_pad),
                },
            ],
            ["policy", "ms", "padded_tokens", "pad_waste"],
        )
    )

    return [
        BenchRow(
            "length_bucketing",
            f"{n_requests}_requests",
            "prefill_time",
            random_ms,
            bucket_ms,
            random_ms / bucket_ms if bucket_ms else float("inf"),
            "ms",
            "same tiny transformer, different batching policy",
        ),
        BenchRow(
            "length_bucketing",
            f"{n_requests}_requests",
            "padded_token_work",
            float(random_pad),
            float(bucket_pad),
            random_pad / bucket_pad if bucket_pad else float("inf"),
            "tokens",
            "proxy for attention/MoE work imbalance",
        ),
    ]


# ---------------------------------------------------------------------------
# Experiment 3: Cache-affinity scheduler simulation
# ---------------------------------------------------------------------------


@dataclass
class SimRequest:
    request_id: int
    prefix_id: int
    prefix_tokens: int
    new_tokens: int
    output_tokens: int
    arrival_s: float


@dataclass
class SimWorker:
    worker_id: int
    available_s: float = 0.0
    total_busy_s: float = 0.0
    cache: dict[int, int] | None = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = {}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((p / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def generate_requests(seed: int, count: int) -> list[SimRequest]:
    rng = random.Random(seed)
    prefix_lengths = {
        i: rng.choice([512, 1024, 2048, 4096, 8192])
        for i in range(24)
    }
    # Hot prefixes mimic shared system prompts / repeated agent contexts.
    hot_prefixes = [0, 1, 2, 3, 4, 5]
    requests = []
    t = 0.0
    for i in range(count):
        t += rng.expovariate(8.0)
        if rng.random() < 0.72:
            prefix_id = rng.choice(hot_prefixes)
        else:
            prefix_id = rng.randrange(24)
        requests.append(
            SimRequest(
                request_id=i,
                prefix_id=prefix_id,
                prefix_tokens=prefix_lengths[prefix_id],
                new_tokens=rng.choice([128, 256, 512, 768]),
                output_tokens=rng.choice([64, 128, 256]),
                arrival_s=t,
            )
        )
    return requests


def cached_tokens(worker: SimWorker, req: SimRequest) -> int:
    assert worker.cache is not None
    return min(worker.cache.get(req.prefix_id, 0), req.prefix_tokens)


def service_time(req: SimRequest, cache_hit_tokens: int) -> tuple[float, int]:
    uncached = max(req.prefix_tokens - cache_hit_tokens, 0) + req.new_tokens
    # Synthetic but intentionally skewed toward prefill, where prefix hits matter.
    duration_s = 0.020 + uncached * 0.000030 + req.output_tokens * 0.000012
    return duration_s, uncached


def simulate_scheduler(
    requests: list[SimRequest],
    *,
    workers_count: int,
    optimized: bool,
    match_weight: float = 1.0,
    aging_tokens_per_s: float = 240.0,
) -> dict[str, float]:
    workers = [SimWorker(i) for i in range(workers_count)]
    waiting: list[SimRequest] = []
    pending = list(requests)
    now = 0.0
    ttfts: list[float] = []
    uncached_tokens_total = 0

    while pending or waiting:
        next_arrival = pending[0].arrival_s if pending else float("inf")
        next_worker = min(w.available_s for w in workers)
        now = min(next_arrival, next_worker if waiting else next_arrival)

        while pending and pending[0].arrival_s <= now:
            waiting.append(pending.pop(0))

        available = [w for w in workers if w.available_s <= now]
        while waiting and available:
            if not optimized:
                req = waiting.pop(0)
                worker = min(available, key=lambda w: (w.available_s, w.total_busy_s, w.worker_id))
            else:
                def req_priority(req_: SimRequest) -> float:
                    best_hit = max(cached_tokens(w, req_) for w in available)
                    uncached = req_.prefix_tokens - best_hit + req_.new_tokens
                    waited = max(now - req_.arrival_s, 0.0)
                    return uncached - waited * aging_tokens_per_s

                req = min(waiting, key=req_priority)
                waiting.remove(req)

                avg_busy = sum(w.total_busy_s for w in workers) / len(workers)
                max_busy = max(max(w.total_busy_s, avg_busy) for w in workers) + 1e-6

                def worker_score(worker_: SimWorker) -> float:
                    hit_pct = cached_tokens(worker_, req) / max(req.prefix_tokens, 1)
                    load = worker_.total_busy_s / max_busy
                    return match_weight * hit_pct - load

                worker = max(available, key=lambda w: (worker_score(w), -w.worker_id))

            hit = cached_tokens(worker, req)
            dur, uncached = service_time(req, hit)
            start = max(now, req.arrival_s, worker.available_s)
            # TTFT proxy: wait + prefill portion. Decode is not included.
            prefill_s = 0.020 + uncached * 0.000030
            ttfts.append(start - req.arrival_s + prefill_s)
            worker.available_s = start + dur
            worker.total_busy_s += dur
            assert worker.cache is not None
            worker.cache[req.prefix_id] = req.prefix_tokens
            uncached_tokens_total += uncached
            available.remove(worker)

        if not pending and waiting and not available:
            now = min(w.available_s for w in workers)

    makespan = max(w.available_s for w in workers)
    return {
        "mean_ttft_ms": statistics.mean(ttfts) * 1000.0,
        "p90_ttft_ms": percentile(ttfts, 90) * 1000.0,
        "p99_ttft_ms": percentile(ttfts, 99) * 1000.0,
        "makespan_s": makespan,
        "uncached_tokens": float(uncached_tokens_total),
    }


def run_scheduler_experiment(args) -> list[BenchRow]:
    count = 300 if args.quick else 1200
    workers = 4 if args.quick else 8
    requests = generate_requests(args.seed, count)
    baseline = simulate_scheduler(requests, workers_count=workers, optimized=False)
    optimized = simulate_scheduler(requests, workers_count=workers, optimized=True)

    print("\n[scheduler] load-only FIFO vs cache-affinity + uncached-token priority")
    print(
        table(
            [
                {
                    "policy": "load_fifo",
                    "p90_ttft_ms": fmt(baseline["p90_ttft_ms"]),
                    "p99_ttft_ms": fmt(baseline["p99_ttft_ms"]),
                    "uncached_tokens": str(int(baseline["uncached_tokens"])),
                    "makespan_s": fmt(baseline["makespan_s"]),
                },
                {
                    "policy": "cache_affinity",
                    "p90_ttft_ms": fmt(optimized["p90_ttft_ms"]),
                    "p99_ttft_ms": fmt(optimized["p99_ttft_ms"]),
                    "uncached_tokens": str(int(optimized["uncached_tokens"])),
                    "makespan_s": fmt(optimized["makespan_s"]),
                },
            ],
            ["policy", "p90_ttft_ms", "p99_ttft_ms", "uncached_tokens", "makespan_s"],
        )
    )

    return [
        BenchRow(
            "scheduler",
            f"{count}_requests",
            "p90_ttft",
            baseline["p90_ttft_ms"],
            optimized["p90_ttft_ms"],
            baseline["p90_ttft_ms"] / optimized["p90_ttft_ms"],
            "ms",
            "synthetic request stream",
        ),
        BenchRow(
            "scheduler",
            f"{count}_requests",
            "uncached_prefill_tokens",
            baseline["uncached_tokens"],
            optimized["uncached_tokens"],
            baseline["uncached_tokens"] / optimized["uncached_tokens"],
            "tokens",
            "lower is better; speedup is avoided work ratio",
        ),
    ]


# ---------------------------------------------------------------------------
# Experiment 4: Encoder batching
# ---------------------------------------------------------------------------


def run_encoder_batching_experiment(args) -> list[BenchRow]:
    torch, nn, F = require_torch()
    device = pick_device(torch, args.device)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    TinyBlock = build_tiny_block_class(torch, nn, F)

    class TinyImageEncoder(nn.Module):
        def __init__(self, dim: int, heads: int, layers: int):
            super().__init__()
            self.patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
            self.blocks = nn.ModuleList([TinyBlock(dim, heads) for _ in range(layers)])
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.patch(x)
            x = x.flatten(2).transpose(1, 2)
            for block in self.blocks:
                x = block(x, causal=False)
            return self.head(self.norm(x).mean(dim=1))

    image_count = 24 if args.quick else 96
    dim = 96 if args.quick else 160
    layers = 2 if args.quick else 4
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    target_hw = 128 if args.quick else 192
    raw_hw = 160 if args.quick else 256

    encoder = TinyImageEncoder(dim, 4, layers).to(device=device, dtype=dtype).eval()
    raw_cpu = torch.rand(image_count, 3, raw_hw, raw_hw, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    @torch.no_grad()
    def serial_per_request():
        outs = []
        for img in raw_cpu:
            x = F.interpolate(
                img.unsqueeze(0),
                size=(target_hw, target_hw),
                mode="bilinear",
                align_corners=False,
            )
            x = ((x - mean) / std).to(device=device, dtype=dtype)
            outs.append(encoder(x))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def batched_encoder():
        x = F.interpolate(
            raw_cpu,
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
        )
        x = ((x - mean) / std).to(device=device, dtype=dtype)
        return encoder(x)

    @torch.no_grad()
    def batched_device_preprocess():
        x = raw_cpu.to(device=device, dtype=dtype)
        mean_d = mean.to(device=device, dtype=dtype)
        std_d = std.to(device=device, dtype=dtype)
        x = F.interpolate(
            x,
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - mean_d) / std_d
        return encoder(x)

    serial_ms = median_time_ms(
        serial_per_request,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    batched_ms = median_time_ms(
        batched_encoder,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    device_ms = median_time_ms(
        batched_device_preprocess,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print("\n[encoder_batching] per-request encoder vs cross-request batch")
    print(
        table(
            [
                {"policy": "serial", "ms": fmt(serial_ms), "images": str(image_count)},
                {"policy": "batched_cpu_pre", "ms": fmt(batched_ms), "images": str(image_count)},
                {"policy": "batched_device_pre", "ms": fmt(device_ms), "images": str(image_count)},
            ],
            ["policy", "ms", "images"],
        )
    )

    return [
        BenchRow(
            "encoder_batching",
            f"{image_count}_images",
            "encoder_time",
            serial_ms,
            batched_ms,
            serial_ms / batched_ms if batched_ms else float("inf"),
            "ms",
            "serial request loop vs one batched forward",
        ),
        BenchRow(
            "encoder_batching",
            f"{image_count}_images",
            "device_preprocess_time",
            batched_ms,
            device_ms,
            batched_ms / device_ms if device_ms else float("inf"),
            "ms",
            "CPU tensor preprocessing vs device preprocessing",
        ),
    ]


# ---------------------------------------------------------------------------
# Experiment 5: MTP-style speculative decode
# ---------------------------------------------------------------------------


def run_mtp_experiment(args) -> list[BenchRow]:
    torch, nn, F = require_torch()
    device = pick_device(torch, args.device)
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    class TinyPatternTarget(nn.Module):
        """Toy autoregressive target model.

        The learned-looking heavy layers create target-model cost. The logits
        intentionally follow a simple pattern so a cheap draft can have high or
        low acceptance, making speculative decode behavior easy to see.
        """

        def __init__(self, vocab: int, dim: int, depth: int):
            super().__init__()
            self.vocab = vocab
            self.embed = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
            self.to_logits = nn.Linear(dim, vocab, bias=False)

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = x + F.gelu(layer(x))
            logits = 0.0001 * self.to_logits(x)
            correct = (input_ids + 1) % self.vocab
            logits = logits.scatter_add(
                -1,
                correct.unsqueeze(-1),
                torch.full((*correct.shape, 1), 10.0, device=input_ids.device),
            )
            return logits

    vocab = 512
    dim = 192 if args.quick else 384
    depth = 4 if args.quick else 8
    gen_len = 160 if args.quick else 512
    draft_k = 4
    target = TinyPatternTarget(vocab, dim, depth).to(device).eval()
    rng = random.Random(args.seed)

    def draft_tokens(start_token: int, k: int, error_rate: float) -> list[int]:
        toks = []
        cur = start_token
        for _ in range(k):
            nxt = (cur + 1) % vocab
            if rng.random() < error_rate:
                nxt = (nxt + rng.randrange(2, 31)) % vocab
            toks.append(nxt)
            cur = nxt
        return toks

    @torch.no_grad()
    def vanilla_decode():
        cur = 7
        out = []
        calls = 0
        for _ in range(gen_len):
            ids = torch.tensor([[cur]], device=device, dtype=torch.long)
            logits = target(ids)
            cur = int(torch.argmax(logits[0, -1]).item())
            out.append(cur)
            calls += 1
        return out, calls

    @torch.no_grad()
    def speculative_decode(error_rate: float):
        cur = 7
        out = []
        calls = 0
        accepted = 0
        while len(out) < gen_len:
            remaining = gen_len - len(out)
            props = draft_tokens(cur, min(draft_k, remaining), error_rate)
            verify_input = [cur] + props[:-1]
            ids = torch.tensor([verify_input], device=device, dtype=torch.long)
            logits = target(ids)
            target_next = torch.argmax(logits[0], dim=-1).tolist()
            calls += 1
            rejected = False
            for prop, truth in zip(props, target_next):
                if prop == int(truth):
                    out.append(prop)
                    cur = prop
                    accepted += 1
                    if len(out) >= gen_len:
                        break
                else:
                    out.append(int(truth))
                    cur = int(truth)
                    rejected = True
                    break
            if not rejected and len(out) < gen_len:
                # All drafted tokens accepted. Continue from the last accepted token.
                pass
        return out, calls, accepted

    vanilla_ms = median_time_ms(
        vanilla_decode,
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    high_accept_error = 0.03
    low_accept_error = 0.35
    spec_high_ms = median_time_ms(
        lambda: speculative_decode(high_accept_error),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    spec_low_ms = median_time_ms(
        lambda: speculative_decode(low_accept_error),
        torch=torch,
        device=device,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    _, vanilla_calls = vanilla_decode()
    _, high_calls, high_acc = speculative_decode(high_accept_error)
    _, low_calls, low_acc = speculative_decode(low_accept_error)

    print("\n[mtp] vanilla greedy decode vs MTP-style speculative verification")
    print(
        table(
            [
                {
                    "policy": "vanilla",
                    "ms": fmt(vanilla_ms),
                    "target_calls": str(vanilla_calls),
                    "accept_rate": "-",
                },
                {
                    "policy": "spec_high_accept",
                    "ms": fmt(spec_high_ms),
                    "target_calls": str(high_calls),
                    "accept_rate": pct(high_acc / gen_len),
                },
                {
                    "policy": "spec_low_accept",
                    "ms": fmt(spec_low_ms),
                    "target_calls": str(low_calls),
                    "accept_rate": pct(low_acc / gen_len),
                },
            ],
            ["policy", "ms", "target_calls", "accept_rate"],
        )
    )

    return [
        BenchRow(
            "mtp",
            "high_accept",
            "decode_time",
            vanilla_ms,
            spec_high_ms,
            vanilla_ms / spec_high_ms if spec_high_ms else float("inf"),
            "ms",
            f"draft_k={draft_k}, error_rate={high_accept_error}",
        ),
        BenchRow(
            "mtp",
            "low_accept",
            "decode_time",
            vanilla_ms,
            spec_low_ms,
            vanilla_ms / spec_low_ms if spec_low_ms else float("inf"),
            "ms",
            f"draft_k={draft_k}, error_rate={low_accept_error}",
        ),
    ]


EXPERIMENTS: dict[str, Callable[[Any], list[BenchRow]]] = {
    "hybrid_swa": run_hybrid_swa_experiment,
    "length_bucketing": run_length_bucketing_experiment,
    "scheduler": run_scheduler_experiment,
    "encoder_batching": run_encoder_batching_experiment,
    "mtp": run_mtp_experiment,
}


def parse_experiments(raw: str) -> list[str]:
    if raw == "all":
        return list(EXPERIMENTS)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {', '.join(unknown)}")
    return names


def print_summary(rows: list[BenchRow]) -> None:
    print("\n[summary] baseline vs optimized")
    printable = [
        {
            "experiment": row.experiment,
            "case": row.case,
            "metric": row.metric,
            "baseline": fmt(row.baseline),
            "optimized": fmt(row.optimized),
            "speedup": fmt(row.speedup),
            "unit": row.unit,
        }
        for row in rows
    ]
    print(table(printable, ["experiment", "case", "metric", "baseline", "optimized", "speedup", "unit"]))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        default="all",
        help=f"Comma-separated names or 'all'. Choices: {', '.join(EXPERIMENTS)}",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--quick", action="store_true", help="Use smaller settings for fast smoke runs.")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats per case.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per case.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, help="Optional path to write raw result rows as JSON.")
    parser.add_argument(
        "--hybrid-lengths",
        help="Hybrid SWA sequence lengths, e.g. 4k,8k,16k,32k,64k,128k.",
    )
    parser.add_argument(
        "--hybrid-window",
        type=int,
        default=128,
        help="Sliding-window size for Hybrid SWA layers.",
    )
    parser.add_argument(
        "--hybrid-layers",
        type=int,
        help="Override Hybrid SWA layer count. Defaults to 12 with --quick, otherwise 24.",
    )
    parser.add_argument("--list", action="store_true", help="List experiments and exit.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        for name in EXPERIMENTS:
            print(name)
        return 0

    all_rows: list[BenchRow] = []
    for name in parse_experiments(args.experiments):
        all_rows.extend(EXPERIMENTS[name](args))

    print_summary(all_rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(row) for row in all_rows], indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
