#!/usr/bin/env python3
"""Adaptive arithmetic coding planning bound for PR101.

The deepest answer to "compress the PMF overhead": don't transmit a PMF
at all. Adaptive arithmetic coding (AAC) updates the PMF symbol-by-symbol
from the already-encoded prefix. Encoder and decoder both run the same
update rule; no PMF is transmitted.

For each symbol s_n at position n:
  - Encode at PMF_n derived from {s_1, ..., s_{n-1}} (Laplace-smoothed)
  - Update counts: count[s_n] += 1
  - Decoder mirrors exactly: encoded bit count is the only thing transmitted

This tool computes the THEORETICAL adaptive-AC bits without actually running
constriction (constriction's Categorical is static; full adaptive AC needs
a custom Model). The theoretical sum-of-conditional-entropies is the
provable lower bound that an arithmetic coder approaches within ~1 byte
of overhead.

Mathematical statement:
  Adaptive AC bits = -sum_n log2(
      (count_{s_n}(n-1) + alpha) / (n - 1 + K * alpha)
  )
where alpha = Laplace smoothing parameter, K = alphabet size.

For IID-within-tensor symbols, this converges to N * H_empirical as N grows.
For mixed non-IID data, this simple running-count baseline is not a full
joint-entropy model. It answers the narrower reproducible question: can a
zero-PMF adaptive marginal coder beat the current Brotli substrate?

Pure CPU + numpy. No GPU, no scorer load, no network.

Usage::

    .venv/bin/python tools/pr101_adaptive_arithmetic_coding.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_adaptive_ac.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_adaptive_arithmetic_coding.py"
SCHEMA_VERSION = "pr101_adaptive_arithmetic_coding.v1"
N_CATEGORIES = 255
EVIDENCE_GRADE = "derivation"
EVIDENCE_SEMANTICS = "cpu_theoretical_adaptive_ac_bound"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144


def _adaptive_ac_bits(
    symbols: np.ndarray,
    *,
    alpha: float = 1.0,
    initial_counts: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute adaptive arithmetic coding bits using running counts.

    Returns (total_bits, final_counts) so callers can chain across tensors.
    """
    counts = (
        np.zeros(N_CATEGORIES, dtype=np.float64)
        if initial_counts is None
        else initial_counts.copy()
    )
    total_seen = float(counts.sum())
    K_alpha = N_CATEGORIES * alpha
    bits = 0.0
    for s in symbols:
        p = (counts[s] + alpha) / (total_seen + K_alpha)
        bits += -math.log2(p)
        counts[s] += 1.0
        total_seen += 1.0
    return bits, counts


def _separate_per_tensor_aac(
    state_dict: dict, *, alpha: float = 1.0
) -> tuple[float, list[dict[str, Any]]]:
    """Per-tensor adaptive AC: each tensor starts from uniform.
    Equivalent to per-tensor empirical floor in the limit but without
    transmitting any PMF."""
    rows: list[dict[str, Any]] = []
    total_bits = 0.0
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        bits, _ = _adaptive_ac_bits(symbols, alpha=alpha)
        rows.append({
            "name": name,
            "n_elements": int(symbols.size),
            "adaptive_ac_bits": bits,
            "adaptive_ac_bytes": math.ceil(bits / 8),
            "amortized_bits_per_element": bits / symbols.size,
        })
        total_bits += bits
    return total_bits, rows


def _joint_aac_across_tensors(
    state_dict: dict, *, alpha: float = 1.0
) -> tuple[float, list[dict[str, Any]]]:
    """Running-count AC: all 28 tensors encoded as one schema-order stream.

    The PMF accumulates across all symbols. This reuses cross-tensor marginal
    statistics, but it is not a full joint context model.
    """
    counts = np.zeros(N_CATEGORIES, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    total_bits = 0.0
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        bits, counts = _adaptive_ac_bits(
            symbols, alpha=alpha, initial_counts=counts
        )
        rows.append({
            "name": name,
            "n_elements": int(symbols.size),
            "running_pmf_total_seen_after": int(counts.sum()),
            "joint_aac_bits_for_this_tensor": bits,
            "amortized_bits_per_element": bits / symbols.size,
        })
        total_bits += bits
    return total_bits, rows


def _interleaved_joint_aac(
    state_dict: dict, *, alpha: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    """Sequential joint adaptive AC where tensors are concatenated in
    schema-order, but with a small overhead to mark per-tensor boundaries
    (so decoder knows where each tensor ends).

    Boundary overhead: 28 x (4 bytes uint32 length prefix) = 112 bytes.
    """
    all_symbols: list[int] = []
    boundaries: list[int] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        all_symbols.extend(int(s) for s in symbols)
        boundaries.append(symbols.size)
    bits, _ = _adaptive_ac_bits(np.array(all_symbols, dtype=np.int32), alpha=alpha)
    boundary_overhead = 4 * len(boundaries)
    return bits, {
        "boundary_overhead_bytes": boundary_overhead,
        "n_total_symbols": len(all_symbols),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Laplace smoothing (default 1.0)")
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    input_bytes = args.state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit("loaded file is not a dict")

    t0 = time.time()
    per_tensor_bits, per_tensor_rows = _separate_per_tensor_aac(state_dict, alpha=args.alpha)
    joint_bits, joint_rows = _joint_aac_across_tensors(state_dict, alpha=args.alpha)
    interleaved_bits, interleaved_meta = _interleaved_joint_aac(state_dict, alpha=args.alpha)
    elapsed = time.time() - t0

    archive_overhead = 16_094
    boundary_overhead = interleaved_meta["boundary_overhead_bytes"]

    per_tensor_bytes = math.ceil(per_tensor_bits / 8)
    joint_bytes = math.ceil(joint_bits / 8)
    interleaved_bytes = math.ceil(interleaved_bits / 8)

    per_tensor_total = per_tensor_bytes + boundary_overhead + archive_overhead
    joint_total = joint_bytes + boundary_overhead + archive_overhead
    interleaved_total = interleaved_bytes + boundary_overhead + archive_overhead

    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(args.state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "theoretical_bits_only",
            "no_actual_adaptive_coder_bitstream",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_categories": N_CATEGORIES,
        "alpha": args.alpha,
        "elapsed_seconds": elapsed,
        "per_tensor_aac_bits": per_tensor_bits,
        "per_tensor_aac_bytes": per_tensor_bytes,
        "per_tensor_aac_total_archive": per_tensor_total,
        "joint_aac_bits": joint_bits,
        "joint_aac_bytes": joint_bytes,
        "joint_aac_total_archive": joint_total,
        "interleaved_aac_bits": interleaved_bits,
        "interleaved_aac_bytes": interleaved_bytes,
        "interleaved_aac_total_archive": interleaved_total,
        "boundary_overhead_bytes": boundary_overhead,
        "archive_overhead_bytes": archive_overhead,
        "comparison_brotli_optuna_optimum_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "savings_per_tensor_aac_vs_brotli": (
            REFERENCE_BROTLI_OPTUNA_BYTES - per_tensor_total
        ),
        "savings_joint_aac_vs_brotli": (
            REFERENCE_BROTLI_OPTUNA_BYTES - joint_total
        ),
        "savings_interleaved_aac_vs_brotli": (
            REFERENCE_BROTLI_OPTUNA_BYTES - interleaved_total
        ),
        "per_tensor_results": per_tensor_rows,
        "joint_results": joint_rows,
        "interleaved_meta": interleaved_meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Elapsed: {elapsed:.1f}s on {sum(r['n_elements'] for r in per_tensor_rows):,} symbols")
    print()
    print("Per-tensor AAC (each tensor starts from uniform):")
    print(f"  payload bytes:                  {per_tensor_bytes:>10,}")
    print(f"  + boundary overhead (28 x 4):   {boundary_overhead:>10}")
    print(f"  + archive overhead:             {archive_overhead:>10,}")
    print(f"  = total:                        {per_tensor_total:>10,} bytes")
    print(
        f"  vs brotli+Optuna {REFERENCE_BROTLI_OPTUNA_BYTES:,}:       "
        f"{REFERENCE_BROTLI_OPTUNA_BYTES - per_tensor_total:>+10,}"
    )
    print()
    print("Running-count AAC (one schema-order stream):")
    print(f"  payload bytes:                  {joint_bytes:>10,}")
    print(f"  + boundary overhead:            {boundary_overhead:>10}")
    print(f"  + archive overhead:             {archive_overhead:>10,}")
    print(f"  = total:                        {joint_total:>10,} bytes")
    print(
        f"  vs brotli+Optuna {REFERENCE_BROTLI_OPTUNA_BYTES:,}:       "
        f"{REFERENCE_BROTLI_OPTUNA_BYTES - joint_total:>+10,}"
    )
    print()
    print("Interleaved AAC (one stream, schema order):")
    print(f"  payload bytes:                  {interleaved_bytes:>10,}")
    print(f"  + boundary overhead:            {boundary_overhead:>10}")
    print(f"  + archive overhead:             {archive_overhead:>10,}")
    print(f"  = total:                        {interleaved_total:>10,} bytes")
    print(
        f"  vs brotli+Optuna {REFERENCE_BROTLI_OPTUNA_BYTES:,}:       "
        f"{REFERENCE_BROTLI_OPTUNA_BYTES - interleaved_total:>+10,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
