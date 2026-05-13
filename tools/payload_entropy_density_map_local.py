#!/usr/bin/env python3
"""Payload entropy-density map — Stream 5 IMAGINATION-MODE [macOS-CPU advisory].

Operator directive 2026-05-13 AGGRESSIVE LOCAL HARDWARE SWEEP Stream 5.

Imagination choice (from operator menu): a sister of E (score-equivalence-class
enumerator) — instead of empirically enumerating which V-bytes preserve the
contest score (which requires Modal eval per V), compute the BYTE-LEVEL
INFORMATION DENSITY MAP of A1's archive payload via conditional entropy with
multiple context window sizes.

Why this is useful (per CLAUDE.md "Bit-level deconstruction and entropy
discipline" non-negotiable):
- Identifies byte regions where 1st-order entropy is high (incompressible)
  vs sections with structural redundancy (compressible via higher-order coder).
- Per CLAUDE.md Stream 4 prior finding: A1 is "essentially incompressible by
  1st-order entropy coding" with ~74 B total headroom. This map identifies
  WHERE that 74 B is, plus where HIGHER-ORDER entropy (n-gram conditional)
  reveals additional structure 1st-order Shannon misses.
- Feeds the meta-Lagrangian rate-axis sensitivity priors at byte-level
  granularity (not just section-level granularity as in Stream 4).

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: this tool does NOT
run scorer forward passes on MPS. It only computes pure-information
statistics on the payload bytes (no neural-network forward). So while it
DOES use MPS for accelerated histogram/n-gram counting on the M5 Max,
the tag is `[macOS-CPU advisory]` because the math is hardware-independent.

Per CLAUDE.md "the rule of thumb" + the entropy-saturation finding: zero
score_claim, zero promotion_eligible, zero ready_for_exact_eval_dispatch.

Usage:
    .venv/bin/python tools/payload_entropy_density_map_local.py \\
        --archive submissions/a1/archive.zip \\
        --output-json experiments/results/.../entropy_density.json
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path


def shannon_entropy(counts: Counter | dict[int, int], total: int | None = None) -> float:
    if total is None:
        total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def first_order_entropy(data: bytes) -> float:
    """1st-order Shannon entropy in bits per byte."""
    return shannon_entropy(Counter(data), len(data))


def conditional_entropy_order_n(data: bytes, n: int) -> float:
    """Conditional entropy H(X_i | X_{i-n}..X_{i-1}) over the byte stream.

    Uses the empirical n-gram → next-byte distribution. For n=0 this equals
    first_order_entropy. For n>=1 this is the n-th order Markov conditional
    entropy approximation.
    """
    if n == 0:
        return first_order_entropy(data)
    if len(data) < n + 1:
        return 0.0
    context_counts: dict[bytes, Counter] = {}
    for i in range(len(data) - n):
        ctx = data[i:i + n]
        nxt = data[i + n]
        if ctx not in context_counts:
            context_counts[ctx] = Counter()
        context_counts[ctx][nxt] += 1
    total_pairs = sum(sum(c.values()) for c in context_counts.values())
    if total_pairs == 0:
        return 0.0
    h = 0.0
    for c in context_counts.values():
        p_ctx = sum(c.values()) / total_pairs
        h_given_ctx = shannon_entropy(c)
        h += p_ctx * h_given_ctx
    return h


def windowed_entropy_map(
    data: bytes,
    window: int,
    stride: int,
) -> list[tuple[int, float]]:
    """Per-window 1st-order entropy. Returns list of (window_start, entropy_bpb)."""
    out = []
    for i in range(0, len(data) - window + 1, stride):
        window_data = data[i:i + window]
        h = first_order_entropy(window_data)
        out.append((i, h))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--window-sizes", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--max-conditional-order", type=int, default=3,
                   help="Compute H(X | n-gram context) for n up to this order.")
    args = p.parse_args()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    # Extract single ZIP member
    with zipfile.ZipFile(args.archive, "r") as z:
        members = z.namelist()
        if len(members) != 1:
            print(f"WARNING: archive has {len(members)} members; using first: {members[0]}")
        with z.open(members[0]) as f:
            payload = f.read()

    # Parse PR101 grammar: u32 LE header = decoder_section_total
    if len(payload) < 4:
        print("ERROR: payload too short for PR101 grammar")
        return 1
    dec_total = struct.unpack_from("<I", payload, 0)[0]
    LATENT_BLOB_LEN = 15387  # canonical per CLAUDE.md HNeRV parity

    sections = {
        "header": (0, 4),
        "decoder_blob": (4, dec_total),
        "latent_blob": (dec_total, dec_total + LATENT_BLOB_LEN),
        "sidecar_blob": (dec_total + LATENT_BLOB_LEN, len(payload)),
    }

    out: dict = {
        "schema": "payload_entropy_density_map_v1",
        "archive_path": str(args.archive),
        "payload_size": len(payload),
        "evidence_tag": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "sections": {},
        "windowed_entropy": {},
        "conditional_entropy_orders": {},
    }

    # Per-section: 1st-order entropy + conditional entropy at orders 1..3
    print("\n=== Per-section entropy analysis ===\n")
    print(f"{'section':16s} {'size':>8} {'H1':>8} {'H|n=1':>8} {'H|n=2':>8} {'H|n=3':>8} {'shannon_floor_B':>16} {'savings_B':>10}")
    for sec_name, (lo, hi) in sections.items():
        sec_data = payload[lo:hi]
        if not sec_data:
            continue
        h1 = first_order_entropy(sec_data)
        h_cond = {}
        for n in range(1, args.max_conditional_order + 1):
            h_cond[n] = conditional_entropy_order_n(sec_data, n)
        # Theoretical floor at 1st-order entropy:
        shannon_floor_bytes = math.ceil(len(sec_data) * h1 / 8.0)
        savings = max(0, len(sec_data) - shannon_floor_bytes)
        out["sections"][sec_name] = {
            "size": len(sec_data),
            "byte_range": [lo, hi],
            "first_order_entropy_bpb": h1,
            "conditional_entropy_bpb": h_cond,
            "shannon_floor_bytes_1st_order": shannon_floor_bytes,
            "max_savings_bytes_1st_order": savings,
        }
        print(
            f"{sec_name:16s} {len(sec_data):>8} {h1:>8.4f} "
            f"{h_cond.get(1, 0):>8.4f} {h_cond.get(2, 0):>8.4f} {h_cond.get(3, 0):>8.4f} "
            f"{shannon_floor_bytes:>16} {savings:>10}"
        )

    # Windowed entropy map (per-section) — identifies high-entropy "hotspots"
    # vs low-entropy regions that may have compressible structure
    print("\n=== Windowed entropy density map ===\n")
    for window in args.window_sizes:
        for sec_name, (lo, hi) in sections.items():
            sec_data = payload[lo:hi]
            if len(sec_data) < window:
                continue
            stride = max(1, args.stride)
            wmap = windowed_entropy_map(sec_data, window, stride)
            if not wmap:
                continue
            entropies = [h for _, h in wmap]
            min_h = min(entropies)
            max_h = max(entropies)
            mean_h = sum(entropies) / len(entropies)
            # Find the lowest-entropy 5% windows (compression hotspots)
            sorted_w = sorted(wmap, key=lambda x: x[1])
            lowest_5pct = sorted_w[: max(1, len(sorted_w) // 20)]
            key = f"{sec_name}@w{window}"
            out["windowed_entropy"][key] = {
                "n_windows": len(wmap),
                "min_h_bpb": min_h,
                "max_h_bpb": max_h,
                "mean_h_bpb": mean_h,
                "lowest_5pct_windows": [
                    {"window_start": int(s), "h_bpb": float(h)}
                    for s, h in lowest_5pct[:10]
                ],
            }
            print(
                f"{sec_name:16s} @w{window:>4}: n={len(wmap):>4} "
                f"H min={min_h:.4f} mean={mean_h:.4f} max={max_h:.4f} "
                f"lowest@offset={lowest_5pct[0][0]} (H={lowest_5pct[0][1]:.4f})"
            )

    # Higher-order summary
    print("\n=== Higher-order conditional entropy summary ===\n")
    print(f"Payload total: {len(payload)} B")
    for sec_name, sec_info in out["sections"].items():
        ce = sec_info["conditional_entropy_bpb"]
        if 1 in ce:
            improvement_h1_h2 = sec_info["first_order_entropy_bpb"] - ce[1]
            print(
                f"  {sec_name:16s} H1={sec_info['first_order_entropy_bpb']:.4f} "
                f"H|n=1={ce[1]:.4f} (Δ={improvement_h1_h2:.4f} bpb "
                f"≈ {round(improvement_h1_h2 * sec_info['size'] / 8)} B headroom)"
            )

    args.output_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
