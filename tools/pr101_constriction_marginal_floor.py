#!/usr/bin/env python3
"""Empirical marginal-floor encoder using constriction range coder.

This is the operational counterpart to ``compute_shannon_floor``.
The marginal-floor calculator predicts ceil(N * H) bytes; this tool
ACTUALLY encodes PR101's quantized weights with that distribution and
measures the real bytes-out, including all overhead (per-tensor PMF
header + range-coder finalization).

The constriction range coder is provably within ~1 byte of the
arithmetic-coding-theoretic minimum given the supplied probability
distributions. So this empirically lower-bounds what ANY entropy coder
operating on the marginal PMFs can achieve.

Compared to brotli (which uses static Huffman with adaptive context),
the marginal range coder:
  + matches the per-tensor empirical PMF exactly (no static-Huffman
    waste)
  - misses cross-tensor context (each tensor coded independently)
  - pays per-tensor PMF-storage overhead (~256 * 4 bytes per tensor;
    can be reduced with shared PMF or implicit predictor)

The cross-context version is in
``tools/pr101_constriction_hyperprior_floor.py``.

CLAUDE.md compliance: pure CPU + numpy + constriction + tac.pr101
quantizer. No scorer load, no GPU, no contest score claims.

Usage::

    .venv/bin/python tools/pr101_constriction_marginal_floor.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_constriction_marginal.json
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)
from tac.repo_io import sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/pr101_constriction_marginal_floor.py"
SCHEMA_VERSION = "pr101_constriction_marginal_floor.v1"
EVIDENCE_GRADE = "empirical"
EVIDENCE_SEMANTICS = "cpu_range_coder_marginal_pmf_packet_estimate"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES = 175_916

# int8 symbol space [-127, 127] maps to 255 categories.
N_CATEGORIES = 255


def _int8_to_index(arr_i8: np.ndarray) -> np.ndarray:
    """Map int8 in [-127, 127] to uint8 index in [0, 254]."""
    shifted = arr_i8.astype(np.int32) + 127
    if (shifted < 0).any() or (shifted > 254).any():
        raise ValueError(f"int8 symbols outside [-127, 127] range: "
                         f"min={shifted.min() - 127}, max={shifted.max() - 127}")
    return shifted.astype(np.int32)


def _empirical_pmf_with_smoothing(
    symbols: np.ndarray, n_cats: int, smoothing: float = 1.0
) -> np.ndarray:
    """Empirical PMF with Laplace smoothing.

    Smoothing prevents zero probabilities (which would be invalid for
    arithmetic coding). The default smoothing of 1 (add-one Laplace)
    adds <1 byte of overhead per tensor on PR101 since N >> 255.
    """
    counts = np.zeros(n_cats, dtype=np.float64)
    flat = symbols.flatten()
    for i in range(flat.size):
        counts[int(flat[i])] += 1.0
    counts += smoothing
    return counts / counts.sum()


def _shannon_bits(symbols: np.ndarray, pmf: np.ndarray) -> float:
    """Compute Shannon bits used to encode `symbols` against `pmf`.
    This is the theoretical minimum; arithmetic coder approaches this."""
    flat = symbols.flatten()
    counts = np.zeros(pmf.size, dtype=np.int64)
    for i in range(flat.size):
        counts[int(flat[i])] += 1
    bits = 0.0
    for c, p in zip(counts, pmf, strict=False):
        if c > 0 and p > 0:
            bits += c * (-math.log2(p))
    return bits


def encode_marginal_floor(
    state_dict_path: Path,
) -> dict[str, Any]:
    """Encode PR101 weights using constriction with per-tensor empirical PMFs."""
    import constriction
    cmodel = constriction.stream.model
    RangeEncoder = constriction.stream.queue.RangeEncoder

    input_sha256 = sha256_file(state_dict_path)
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[dict[str, Any]] = []
    total_encoded_bytes = 0
    total_pmf_overhead_bytes = 0
    total_theoretical_bits = 0.0
    t0 = time.time()

    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = _int8_to_index(qt.q_i8.flatten())
        pmf = _empirical_pmf_with_smoothing(symbols, N_CATEGORIES, smoothing=1.0)
        # Encode with range coder using per-tensor categorical model
        m = cmodel.Categorical(pmf, perfect=False)
        enc = RangeEncoder()
        enc.encode(symbols.astype(np.int32), m)
        encoded = enc.get_compressed()
        # `encoded` is a numpy uint32 array; convert to bytes
        encoded_bytes = encoded.tobytes()
        # PMF overhead: store the PMF as float16 or quantized int16. Use
        # float16 = 2 bytes per category x 255 = 510 bytes/tensor as a
        # plausible deployable overhead. (Could tune lower with quantized
        # PMF or implicit predictor.)
        pmf_overhead = N_CATEGORIES * 2  # 510 bytes
        # Theoretical bits at empirical PMF
        theoretical_bits = _shannon_bits(symbols, pmf)
        rows.append({
            "name": name,
            "n_elements": int(symbols.size),
            "encoded_payload_bytes": len(encoded_bytes),
            "pmf_overhead_bytes": pmf_overhead,
            "theoretical_bits_at_empirical_pmf": theoretical_bits,
            "theoretical_bytes_at_empirical_pmf": math.ceil(theoretical_bits / 8.0),
            "n_unique_symbols": int(np.count_nonzero(np.bincount(symbols, minlength=N_CATEGORIES))),
        })
        total_encoded_bytes += len(encoded_bytes)
        total_pmf_overhead_bytes += pmf_overhead
        total_theoretical_bits += theoretical_bits

    elapsed = time.time() - t0

    archive_overhead = 15_387 + 607 + 100  # latent_blob + sidecar + zip header
    total_with_pmf_overhead = total_encoded_bytes + total_pmf_overhead_bytes
    total_with_archive_overhead = total_with_pmf_overhead + archive_overhead

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
            "pmf_storage_not_wired_into_decoder",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_tensors": len(rows),
        "n_categories": N_CATEGORIES,
        "smoothing": 1.0,
        "elapsed_seconds": elapsed,
        "total_encoded_payload_bytes": total_encoded_bytes,
        "total_pmf_overhead_bytes": total_pmf_overhead_bytes,
        "total_with_pmf_overhead": total_with_pmf_overhead,
        "total_archive_overhead_bytes": archive_overhead,
        "total_archive_with_overhead": total_with_archive_overhead,
        "total_theoretical_bits_at_empirical_pmf": total_theoretical_bits,
        "total_theoretical_bytes_at_empirical_pmf": math.ceil(total_theoretical_bits / 8.0),
        "comparison_brotli_optuna_optimum_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "savings_vs_brotli_optuna": REFERENCE_BROTLI_OPTUNA_BYTES - total_with_archive_overhead,
        "comparison_per_tensor_marginal_floor_bytes": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES,
        "savings_vs_marginal_floor": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES - total_with_archive_overhead,
        "per_tensor": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = encode_marginal_floor(args.state_dict_path)
    write_json(args.output, manifest)

    print(f"Wrote {args.output}")
    print(f"Total encoded payload:           {manifest['total_encoded_payload_bytes']:>10,} bytes")
    print(f"PMF overhead (255 cats x 2 B x 28 tensors): {manifest['total_pmf_overhead_bytes']:>10,} bytes")
    print(f"Total with PMF overhead:         {manifest['total_with_pmf_overhead']:>10,} bytes")
    print(f"Archive overhead (latent+sidecar+zip):       {manifest['total_archive_overhead_bytes']:>10,} bytes")
    print(f"Total archive with overhead:     {manifest['total_archive_with_overhead']:>10,} bytes")
    print()
    print(f"vs brotli+Optuna (178,144 B):    {manifest['savings_vs_brotli_optuna']:>+10,} bytes")
    print(f"vs marginal floor (175,916 B):   {manifest['savings_vs_marginal_floor']:>+10,} bytes")
    print()
    print(f"Theoretical bits at empirical PMF: {manifest['total_theoretical_bits_at_empirical_pmf']:,.1f}")
    print(f"  -> bytes:                        {manifest['total_theoretical_bytes_at_empirical_pmf']:,}")
    print(f"  + archive overhead = floor:      {manifest['total_theoretical_bytes_at_empirical_pmf'] + manifest['total_archive_overhead_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
