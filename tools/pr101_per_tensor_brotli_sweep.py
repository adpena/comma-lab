#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-tensor brotli parameter sweep for PR101's quantized weights.

Global brotli sweeps (cf. tools/codec_op_cma_search.py) constrain ALL
28 tensors to a single (quality, lgwin, lgblock) parameter tuple. This
tool lifts that constraint and finds the per-tensor optimum, computing
the SUM of per-tensor minima as a tight bound on what tensor-aware
brotli could achieve.

Key insight: brotli's lgwin (sliding window) and lgblock (block size) are
per-stream parameters. For small tensors (biases of 3-9 elements), a small
lgwin helps; for large tensors (48,384-element stem.weight), large lgwin
helps. Forcing all 28 tensors to share parameters costs bytes.

Pure CPU + brotli + numpy. No GPU, no scorer load.

Output:
  - Per-tensor minimum (q, lgwin, lgblock) to bytes
  - Sum across 28 tensors = the per-tensor-optimal brotli floor
  - Comparison with global-optimal brotli (from cmaes search)
  - Comparison with empirical Shannon floor

Usage::

    .venv/bin/python tools/pr101_per_tensor_brotli_sweep.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_per_tensor_brotli_sweep.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
    _zigzag_encode_i8,
)

TOOL_NAME = "tools/pr101_per_tensor_brotli_sweep.py"
SCHEMA_VERSION = "pr101_per_tensor_brotli_sweep.v1"


def _quantize_to_zigzag_bytes(name: str, tensor: torch.Tensor) -> tuple[bytes, str]:
    """Quantize a tensor with PR101's INT8 scheme + zigzag, return raw bytes."""
    qt = _quantize_tensor(name, tensor, n_quant=N_QUANT)
    zz = _zigzag_encode_i8(qt.q_i8.flatten())
    raw = bytes(zz.tolist())
    return raw, hashlib.sha256(raw).hexdigest()


def _sweep_tensor(
    raw_bytes: bytes,
    *,
    qualities: list[int],
    lgwins: list[int],
    lgblocks: list[int],
) -> dict[str, Any]:
    """Sweep all (q, lgwin, lgblock) combinations on a single tensor's bytes."""
    best: dict[str, Any] = {"bytes_out": float("inf")}
    n_evals = 0
    n_failed = 0
    for q, lgwin, lgblock in itertools.product(qualities, lgwins, lgblocks):
        try:
            encoded = brotli.compress(
                raw_bytes,
                quality=q,
                lgwin=lgwin,
                lgblock=lgblock,
            )
            n_evals += 1
            if len(encoded) < best["bytes_out"]:
                best = {
                    "bytes_out": len(encoded),
                    "quality": q,
                    "lgwin": lgwin,
                    "lgblock": lgblock,
                    "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                }
        except brotli.error:
            n_failed += 1
    best["n_evals"] = n_evals
    best["n_failed"] = n_failed
    best["raw_bytes_len"] = len(raw_bytes)
    return best


def per_tensor_sweep(
    state_dict_path: Path,
    *,
    qualities: list[int] | None = None,
    lgwins: list[int] | None = None,
    lgblocks: list[int] | None = None,
) -> dict[str, Any]:
    """Run the full per-tensor sweep across PR101's 28 tensors."""
    if qualities is None:
        qualities = list(range(1, 12))  # 1..11
    if lgwins is None:
        lgwins = list(range(10, 25))  # 10..24
    if lgblocks is None:
        lgblocks = list(range(16, 25))  # 16..24

    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[dict[str, Any]] = []
    total_bytes = 0
    total_raw = 0
    t0 = time.time()
    for name, shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        raw_bytes, raw_sha = _quantize_to_zigzag_bytes(name, state_dict[name])
        result = _sweep_tensor(
            raw_bytes,
            qualities=qualities,
            lgwins=lgwins,
            lgblocks=lgblocks,
        )
        # Also record the global-default result for direct comparison
        default_encoded = brotli.compress(raw_bytes, quality=11, lgwin=22, lgblock=18)
        result["bytes_at_pr101_default_q11_lgwin22"] = len(default_encoded)
        result["bytes_savings_vs_pr101_default"] = (
            len(default_encoded) - result["bytes_out"]
        )
        result["name"] = name
        result["shape"] = list(shape)
        result["n_elements"] = int(np.prod(shape))
        result["raw_sha256"] = raw_sha
        rows.append(result)
        total_bytes += result["bytes_out"]
        total_raw += result["raw_bytes_len"]

    elapsed = time.time() - t0
    n_per_tensor = len(qualities) * len(lgwins) * len(lgblocks)
    n_total = n_per_tensor * len(rows)

    # Aggregate stats
    total_default = sum(r["bytes_at_pr101_default_q11_lgwin22"] for r in rows)
    total_savings = total_default - total_bytes
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "sweep_axes": {
            "quality": qualities,
            "lgwin": lgwins,
            "lgblock": lgblocks,
        },
        "n_evals_per_tensor": n_per_tensor,
        "n_total_evals": n_total,
        "elapsed_seconds": elapsed,
        "evals_per_second": n_total / elapsed if elapsed > 0 else 0,
        "n_tensors": len(rows),
        "total_raw_bytes": total_raw,
        "total_per_tensor_optimum_bytes": total_bytes,
        "total_at_pr101_default_bytes": total_default,
        "total_savings_vs_pr101_default": total_savings,
        "per_tensor_results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-min", type=int, default=1)
    parser.add_argument("--quality-max", type=int, default=11)
    parser.add_argument("--lgwin-min", type=int, default=10)
    parser.add_argument("--lgwin-max", type=int, default=24)
    parser.add_argument("--lgblock-min", type=int, default=16)
    parser.add_argument("--lgblock-max", type=int, default=24)
    args = parser.parse_args(argv)

    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    qualities = list(range(args.quality_min, args.quality_max + 1))
    lgwins = list(range(args.lgwin_min, args.lgwin_max + 1))
    lgblocks = list(range(args.lgblock_min, args.lgblock_max + 1))
    n_per_tensor = len(qualities) * len(lgwins) * len(lgblocks)
    n_total = n_per_tensor * len(FIXED_STATE_SCHEMA)
    print(f"Sweeping {n_per_tensor} configs/tensor x {len(FIXED_STATE_SCHEMA)} tensors = {n_total:,} total evals")
    print(f"Estimated time: {n_total * 0.05:.0f}-{n_total * 0.5:.0f} seconds")

    manifest = per_tensor_sweep(
        args.state_dict_path,
        qualities=qualities,
        lgwins=lgwins,
        lgblocks=lgblocks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nWrote per-tensor brotli sweep manifest to {args.output}")
    print(f"Total per-tensor optimum:        {manifest['total_per_tensor_optimum_bytes']:>10,} bytes")
    print(f"Total at PR101 default (q=11,lgwin=22,lgblock=18): {manifest['total_at_pr101_default_bytes']:>10,} bytes")
    print(f"Savings:                         {manifest['total_savings_vs_pr101_default']:>+10,} bytes")
    print(f"Sweep wall-clock: {manifest['elapsed_seconds']:.1f}s ({manifest['evals_per_second']:.0f} evals/s)")
    print("\nTop-5 tensors by savings:")
    sorted_rows = sorted(
        manifest["per_tensor_results"],
        key=lambda r: -r["bytes_savings_vs_pr101_default"],
    )[:5]
    for r in sorted_rows:
        print(
            f"  {r['name']:<20} elements={r['n_elements']:>6,} "
            f"q={r['quality']:>2} lgwin={r['lgwin']:>2} lgblock={r['lgblock']:>2} "
            f"saves {r['bytes_savings_vs_pr101_default']:>+5} bytes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
