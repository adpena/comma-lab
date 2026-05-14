#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Constriction with a SHARED PMF across all 28 tensors.

The per-tensor PMF approach (cf.
``tools/pr101_constriction_marginal_floor.py``) costs 14,280 bytes of
PMF overhead. This variant encodes all 28 tensors against a SINGLE
shared empirical PMF computed from all 228,958 symbols pooled.

Pros:
  - PMF overhead drops from 14,280 to 510 bytes (single PMF, not 28)
  - Pooled PMF is "near-canonical" since neural-network weight
    distributions are highly self-similar across layers (mostly
    Laplacian-like)

Cons:
  - Slight ratio loss because per-tensor distributions DO differ
    (biases are sparser; rgb tensors have different scales)

Operational comparison vs brotli:
  - brotli static-Huffman uses ~14 bytes of header per stream (one
    stream); about equivalent overhead to one PMF here
  - brotli adapts within its sliding window; this shared-PMF approach
    does NOT adapt at all: pure marginal coding

This tool's output is the "naive joint-floor approximation", not as
good as a learned hyperprior, but a useful upper bound on the true
joint floor.

CLAUDE.md compliance: pure CPU + numpy + constriction. No scorer load,
no GPU, no contest score claims.

Usage::

    .venv/bin/python tools/pr101_constriction_shared_pmf_floor.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_constriction_shared_pmf.json
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

TOOL_NAME = "tools/pr101_constriction_shared_pmf_floor.py"
SCHEMA_VERSION = "pr101_constriction_shared_pmf_floor.v1"
N_CATEGORIES = 255
EVIDENCE_GRADE = "empirical"
EVIDENCE_SEMANTICS = "cpu_shared_pmf_range_coder_packet_estimate"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES = 175_916
REFERENCE_PER_TENSOR_CONSTRICTION_BYTES = 190_718


def _int8_to_index(arr_i8: np.ndarray) -> np.ndarray:
    return (arr_i8.astype(np.int32) + 127).astype(np.int32)


def encode_shared_pmf(state_dict_path: Path) -> dict[str, Any]:
    import constriction
    cmodel = constriction.stream.model
    RangeEncoder = constriction.stream.queue.RangeEncoder

    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    # Pool all symbols across all tensors first
    pooled: list[np.ndarray] = []
    per_tensor_meta: list[dict[str, Any]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        idx = _int8_to_index(qt.q_i8.flatten())
        pooled.append(idx)
        per_tensor_meta.append({"name": name, "n_elements": int(idx.size)})
    all_symbols = np.concatenate(pooled)

    # Empirical PMF over all symbols pooled, with Laplace smoothing
    counts = np.bincount(all_symbols, minlength=N_CATEGORIES).astype(np.float64)
    counts += 1.0  # Laplace smoothing
    shared_pmf = counts / counts.sum()

    # Theoretical bits (no smoothing for the bound; use raw fractions
    # but skip zero-count bins for log)
    raw_counts = np.bincount(all_symbols, minlength=N_CATEGORIES).astype(np.float64)
    raw_pmf = raw_counts / raw_counts.sum()
    bits = 0.0
    for c, p in zip(raw_counts, raw_pmf, strict=False):
        if c > 0 and p > 0:
            bits += c * (-math.log2(p))

    # Encode each tensor against the SHARED PMF
    t0 = time.time()
    m = cmodel.Categorical(shared_pmf, perfect=False)
    enc = RangeEncoder()
    enc.encode(all_symbols.astype(np.int32), m)
    encoded_words = enc.get_compressed()
    encoded_bytes = encoded_words.tobytes()
    elapsed = time.time() - t0

    pmf_overhead = N_CATEGORIES * 2  # 510 bytes (single shared PMF as float16)
    archive_overhead = 15_387 + 607 + 100  # 16,094 bytes

    total_with_pmf = len(encoded_bytes) + pmf_overhead
    total_with_archive = total_with_pmf + archive_overhead

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "packet_estimate_only",
            "shared_pmf_not_wired_into_decoder",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_tensors": len(per_tensor_meta),
        "total_symbols": int(all_symbols.size),
        "n_categories": N_CATEGORIES,
        "smoothing": 1.0,
        "elapsed_seconds": elapsed,
        "shared_pmf_n_unique_symbols": int(np.count_nonzero(np.bincount(all_symbols, minlength=N_CATEGORIES))),
        "encoded_payload_bytes": len(encoded_bytes),
        "pmf_overhead_bytes": pmf_overhead,
        "total_with_pmf_overhead": total_with_pmf,
        "archive_overhead_bytes": archive_overhead,
        "total_archive_with_overhead": total_with_archive,
        "theoretical_bits_at_pooled_pmf": bits,
        "theoretical_bytes_at_pooled_pmf": math.ceil(bits / 8.0),
        "comparison_brotli_optuna_optimum_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "savings_vs_brotli_optuna": REFERENCE_BROTLI_OPTUNA_BYTES - total_with_archive,
        "comparison_per_tensor_marginal_floor_bytes": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES,
        "savings_vs_per_tensor_marginal_floor": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES - total_with_archive,
        "comparison_per_tensor_constriction_marginal_bytes": REFERENCE_PER_TENSOR_CONSTRICTION_BYTES,
        "savings_vs_per_tensor_constriction": REFERENCE_PER_TENSOR_CONSTRICTION_BYTES - total_with_archive,
        "per_tensor": per_tensor_meta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = encode_shared_pmf(args.state_dict_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Total symbols pooled:                  {manifest['total_symbols']:>10,}")
    print(f"Encoded payload (shared-PMF range):    {manifest['encoded_payload_bytes']:>10,} bytes")
    print(f"Theoretical at pooled-PMF:             {manifest['theoretical_bytes_at_pooled_pmf']:>10,} bytes")
    print(f"PMF overhead (one shared PMF, fp16):   {manifest['pmf_overhead_bytes']:>10,} bytes")
    print(f"Archive overhead:                      {manifest['archive_overhead_bytes']:>10,} bytes")
    print(f"Total archive with overhead:           {manifest['total_archive_with_overhead']:>10,} bytes")
    print()
    print(f"vs brotli+Optuna  (178,144 B):         {manifest['savings_vs_brotli_optuna']:>+10,} bytes")
    print(f"vs per-tensor marginal floor (175,916): {manifest['savings_vs_per_tensor_marginal_floor']:>+10,} bytes")
    print(f"vs per-tensor constriction (190,718):  {manifest['savings_vs_per_tensor_constriction']:>+10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
