#!/usr/bin/env python3
"""Compute per-tensor empirical entropy from PR101's quantized weights.

Uses ``tac.pr101_split_brotli_codec._quantize_tensor`` to apply the same
INT8 symmetric quantization PR101 uses, then computes per-element
Shannon entropy for each of the 28 tensors.

The output feeds into ``tac.score_geometry_shannon_floor.compute_shannon_floor()``
as the ``per_tensor_empirical_bits`` argument — converting the loose
uniform upper bound into a tight empirical lower bound on archive bytes.

Bonus: also reports the per-tensor histogram and the entropy-vs-uniform
ratio per tensor, which is operationally useful — tensors with the
highest entropy gap are the most-skewed and benefit most from
arithmetic coding (vs brotli's static Huffman).

CLAUDE.md compliance: pure CPU + numpy + math; no scorer load; no
contest score claims (only entropy bounds).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)
from tac.score_geometry_shannon_floor import compute_shannon_floor  # noqa: E402

TOOL_NAME = "tools/pr101_per_tensor_empirical_entropy.py"
SCHEMA_VERSION = "pr101_per_tensor_empirical_entropy.v1"


def _shannon_entropy_bits(symbols: np.ndarray) -> tuple[float, dict[int, int]]:
    """Compute Shannon entropy of a 1D symbol array. Returns (bits, counts)."""
    if symbols.size == 0:
        return 0.0, {}
    flat = symbols.flatten().tolist()
    counts = Counter(int(s) for s in flat)
    total = sum(counts.values())
    h = 0.0
    for n in counts.values():
        if n == 0:
            continue
        p = n / total
        h -= p * math.log2(p)
    return h, dict(counts)


def compute_per_tensor_entropy(
    state_dict_path: Path,
    *,
    n_quant: int = N_QUANT,
) -> dict[str, Any]:
    """Quantize each tensor in the saved state_dict, compute entropy.

    Returns a manifest with per-tensor entropy + the empirical Shannon
    floor when those entropies are plugged into compute_shannon_floor().
    """
    import torch

    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[dict[str, Any]] = []
    per_tensor_bits: dict[str, float] = {}
    uniform_bits = math.log2(n_quant) if n_quant > 1 else 0.0

    for name, shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        tensor = state_dict[name]
        qt = _quantize_tensor(name, tensor, n_quant=n_quant)
        h_bits, counts = _shannon_entropy_bits(qt.q_i8)
        ratio = h_bits / uniform_bits if uniform_bits > 0 else 1.0
        rows.append({
            "name": name,
            "shape": list(shape),
            "n_elements": int(qt.q_i8.size),
            "scale": qt.scale,
            "n_unique_symbols": len(counts),
            "n_quant_max": n_quant,
            "uniform_bits_per_element": uniform_bits,
            "empirical_bits_per_element": h_bits,
            "entropy_ratio_vs_uniform": ratio,
            "min_symbol": int(qt.q_i8.min()),
            "max_symbol": int(qt.q_i8.max()),
            "mode_symbol": max(counts.items(), key=lambda kv: kv[1])[0] if counts else 0,
            "mode_count": max(counts.values()) if counts else 0,
        })
        per_tensor_bits[name] = h_bits

    # Compute empirical Shannon floor (ARCHIVE-LEVEL bound)
    archive_overhead = 15_387 + 607 + 100
    floor_report = compute_shannon_floor(
        schema=FIXED_STATE_SCHEMA,
        n_quant=n_quant,
        schema_label="PR101_FIXED_STATE_SCHEMA",
        per_tensor_empirical_bits=per_tensor_bits,
        archive_overhead_bytes=archive_overhead,
    )

    # Aggregate stats
    total_elements = sum(r["n_elements"] for r in rows)
    weighted_avg_entropy = (
        sum(r["empirical_bits_per_element"] * r["n_elements"] for r in rows)
        / total_elements
    ) if total_elements > 0 else 0.0
    most_skewed = sorted(rows, key=lambda r: r["entropy_ratio_vs_uniform"])[:5]
    least_skewed = sorted(
        rows, key=lambda r: -r["entropy_ratio_vs_uniform"]
    )[:5]

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "n_quant": n_quant,
        "uniform_bits_per_element": uniform_bits,
        "weighted_avg_empirical_bits_per_element": weighted_avg_entropy,
        "weighted_avg_entropy_ratio_vs_uniform": (
            weighted_avg_entropy / uniform_bits if uniform_bits > 0 else 1.0
        ),
        "total_elements": total_elements,
        "uniform_floor_bytes": floor_report.total_bytes_uniform_floor,
        "empirical_floor_bytes": floor_report.total_bytes_empirical_floor,
        "uniform_to_empirical_savings_bytes": (
            floor_report.total_bytes_uniform_floor
            - (floor_report.total_bytes_empirical_floor or 0)
        ),
        "score_at_empirical_floor_zero_distortion":
            floor_report.score_at_empirical_floor_zero_distortion,
        "per_tensor": rows,
        "top_5_most_skewed_tensors": [
            {"name": r["name"], "entropy_ratio": r["entropy_ratio_vs_uniform"]}
            for r in most_skewed
        ],
        "top_5_least_skewed_tensors": [
            {"name": r["name"], "entropy_ratio": r["entropy_ratio_vs_uniform"]}
            for r in least_skewed
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True,
                        help="Saved torch state_dict (e.g. extracted from PR101 archive)")
    parser.add_argument("--n-quant", type=int, default=N_QUANT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = compute_per_tensor_entropy(args.state_dict_path, n_quant=args.n_quant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote per-tensor entropy manifest to {args.output}")
    print(
        f"Total: {manifest['total_elements']:,} elements; "
        f"weighted-avg empirical = {manifest['weighted_avg_empirical_bits_per_element']:.4f} bits/elem "
        f"(vs uniform {manifest['uniform_bits_per_element']:.4f}); "
        f"ratio {manifest['weighted_avg_entropy_ratio_vs_uniform']:.4f}"
    )
    print(
        f"Uniform floor: {manifest['uniform_floor_bytes']:,} bytes; "
        f"Empirical floor: {manifest['empirical_floor_bytes']:,} bytes; "
        f"Savings: {manifest['uniform_to_empirical_savings_bytes']:,} bytes; "
        f"Empirical-floor score: {manifest['score_at_empirical_floor_zero_distortion']:.5f}"
    )
    print("Most-skewed tensors (lowest entropy ratio):")
    for row in manifest["top_5_most_skewed_tensors"]:
        print(f"  {row['name']}: ratio={row['entropy_ratio']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
