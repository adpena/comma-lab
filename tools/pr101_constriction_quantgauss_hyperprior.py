#!/usr/bin/env python3
"""Balle-style QuantizedGaussian hyperprior on PR101 weights.

Per the joint-entropy subagent verdict (memo
``feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md``),
the deployable joint-floor approach is "constriction + learned
2-tensor-context hyperprior". This tool implements the simplest version:

For each tensor:
  1. Compute (mean, std) of its int8-quantized symbols (zigzag-decoded)
  2. Encode (mean, std) as fp16: 4 bytes overhead per tensor
  3. Use constriction's QuantizedGaussian(mean, std) as the coding model
  4. Encode the symbols against the per-tensor Gaussian

Total overhead: 28 x 4 = 112 bytes (vs 14,280 for empirical PMF).

If neural-network INT8 weight distributions ARE roughly Gaussian (after
zero-meaning), the Gaussian payload bytes will be close to the empirical
floor without paying the per-tensor PMF transmission cost.

If they're NOT Gaussian (e.g., heavy-tailed Laplacian), the
QuantizedLaplace fallback should do better. We compute both and pick
the per-tensor minimum.

CLAUDE.md compliance: pure CPU + numpy + constriction. No scorer load,
no GPU, no contest score claims. The output is the deployable joint-
floor estimate (predicted 155-165 KB; subagent estimate 155 KB).

Usage::

    .venv/bin/python tools/pr101_constriction_quantgauss_hyperprior.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_constriction_quantgauss.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

TOOL_NAME = "tools/pr101_constriction_quantgauss_hyperprior.py"
SCHEMA_VERSION = "pr101_constriction_quantgauss_hyperprior.v1"
EVIDENCE_GRADE = "empirical"
EVIDENCE_SEMANTICS = "cpu_parametric_hyperprior_packet_estimate"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES = 175_916
REFERENCE_PER_TENSOR_CONSTRICTION_BYTES = 190_718
REFERENCE_SHARED_PMF_CONSTRICTION_BYTES = 203_196
REFERENCE_JOINT_FLOOR_PREDICTION_BYTES = 155_000


def _try_encode_quantgauss(
    symbols_centered: np.ndarray,
    mean: float,
    std: float,
    *,
    use_laplace: bool = False,
) -> tuple[int, float, str]:
    """Encode `symbols_centered` (int range, zero-mean) with
    QuantizedGaussian or QuantizedLaplace. Returns (bytes, theoretical_bits, model_name)."""
    import constriction
    cmodel = constriction.stream.model
    RangeEncoder = constriction.stream.queue.RangeEncoder
    # constriction's QuantizedGaussian / QuantizedLaplace parameterize on
    # the continuous (mean, std) of an underlying Gaussian/Laplace; the
    # model quantizes to integers in the supplied range.
    n_cats = 255
    lo, hi = 0, n_cats - 1  # integer support
    if use_laplace:
        m = cmodel.QuantizedLaplace(min_symbol_inclusive=lo, max_symbol_inclusive=hi)
        # Laplace API: encode method needs (symbols, mean_array, scale_array)
        # where each symbol gets its own (mean, scale). We pass scalars
        # broadcast to all symbols.
        means = np.full(symbols_centered.size, mean, dtype=np.float64)
        scales = np.full(symbols_centered.size, std, dtype=np.float64)
        enc = RangeEncoder()
        enc.encode(symbols_centered.astype(np.int32), m, means, scales)
    else:
        m = cmodel.QuantizedGaussian(min_symbol_inclusive=lo, max_symbol_inclusive=hi)
        means = np.full(symbols_centered.size, mean, dtype=np.float64)
        stds = np.full(symbols_centered.size, std, dtype=np.float64)
        enc = RangeEncoder()
        enc.encode(symbols_centered.astype(np.int32), m, means, stds)
    encoded = enc.get_compressed()
    encoded_bytes = encoded.tobytes()
    return len(encoded_bytes), 0.0, ("laplace" if use_laplace else "gaussian")


def encode_quantgauss_hyperprior(state_dict_path: Path) -> dict[str, Any]:
    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")

    rows: list[dict[str, Any]] = []
    total_payload_bytes = 0
    total_hyperprior_overhead = 0
    t0 = time.time()

    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        # Map int8 [-127, 127] to index [0, 254]
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        # Compute mean + std on the indexed symbols
        mean = float(np.mean(symbols))
        std = float(np.std(symbols))
        if std < 1e-3:
            std = 1e-3  # avoid degenerate Gaussian
        # Try Gaussian
        try:
            g_bytes, _, _ = _try_encode_quantgauss(symbols, mean, std, use_laplace=False)
        except Exception:
            g_bytes = -1
        # Try Laplace
        try:
            l_bytes, _, _ = _try_encode_quantgauss(symbols, mean, std, use_laplace=True)
        except Exception:
            l_bytes = -1
        # Pick best (Gaussian or Laplace; need 1 bit to flag which model in archive)
        if g_bytes > 0 and l_bytes > 0:
            best_bytes = min(g_bytes, l_bytes)
            best_model = "gaussian" if g_bytes <= l_bytes else "laplace"
        elif g_bytes > 0:
            best_bytes = g_bytes
            best_model = "gaussian"
        elif l_bytes > 0:
            best_bytes = l_bytes
            best_model = "laplace"
        else:
            raise RuntimeError(f"both Gaussian and Laplace encoders failed for {name}")

        # Hyperprior overhead per tensor: 2 fp16 (mean, std) + 1 bit for
        # model selection = 4.13 bytes; round up to 5 bytes for header.
        hyperprior_overhead = 5

        rows.append({
            "name": name,
            "n_elements": int(symbols.size),
            "mean": mean,
            "std": std,
            "gaussian_bytes": g_bytes,
            "laplace_bytes": l_bytes,
            "best_bytes": best_bytes,
            "best_model": best_model,
            "hyperprior_overhead_bytes": hyperprior_overhead,
        })
        total_payload_bytes += best_bytes
        total_hyperprior_overhead += hyperprior_overhead

    elapsed = time.time() - t0

    archive_overhead = 15_387 + 607 + 100
    total_with_overhead = total_payload_bytes + total_hyperprior_overhead + archive_overhead

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
            "parametric_model_not_wired_into_decoder",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_tensors": len(rows),
        "elapsed_seconds": elapsed,
        "total_payload_bytes": total_payload_bytes,
        "total_hyperprior_overhead_bytes": total_hyperprior_overhead,
        "total_archive_overhead_bytes": archive_overhead,
        "total_archive_with_overhead": total_with_overhead,
        "comparison_brotli_optuna_optimum_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "savings_vs_brotli_optuna": REFERENCE_BROTLI_OPTUNA_BYTES - total_with_overhead,
        "comparison_per_tensor_marginal_floor_bytes": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES,
        "savings_vs_marginal_floor": REFERENCE_PER_TENSOR_MARGINAL_FLOOR_BYTES - total_with_overhead,
        "comparison_per_tensor_constriction": REFERENCE_PER_TENSOR_CONSTRICTION_BYTES,
        "savings_vs_per_tensor_constriction": REFERENCE_PER_TENSOR_CONSTRICTION_BYTES - total_with_overhead,
        "comparison_shared_pmf_constriction": REFERENCE_SHARED_PMF_CONSTRICTION_BYTES,
        "savings_vs_shared_pmf_constriction": REFERENCE_SHARED_PMF_CONSTRICTION_BYTES - total_with_overhead,
        "subagent_predicted_deployable_joint_floor": REFERENCE_JOINT_FLOOR_PREDICTION_BYTES,
        "gap_to_subagent_prediction": total_with_overhead - REFERENCE_JOINT_FLOOR_PREDICTION_BYTES,
        "per_tensor": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    manifest = encode_quantgauss_hyperprior(args.state_dict_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Total payload (Gaussian/Laplace per-tensor): {manifest['total_payload_bytes']:>10,} bytes")
    print(f"Hyperprior overhead (28 x 5 B):              {manifest['total_hyperprior_overhead_bytes']:>10,} bytes")
    print(f"Archive overhead:                            {manifest['total_archive_overhead_bytes']:>10,} bytes")
    print(f"Total archive:                               {manifest['total_archive_with_overhead']:>10,} bytes")
    print()
    print(f"vs brotli+Optuna  (178,144):                 {manifest['savings_vs_brotli_optuna']:>+10,} bytes")
    print(f"vs marginal floor (175,916):                 {manifest['savings_vs_marginal_floor']:>+10,} bytes")
    print(f"vs per-tensor constriction (190,718):        {manifest['savings_vs_per_tensor_constriction']:>+10,} bytes")
    print(f"vs shared-PMF constriction (203,196):        {manifest['savings_vs_shared_pmf_constriction']:>+10,} bytes")
    print()
    print(f"vs subagent prediction (155,000 deployable joint floor): "
          f"{manifest['gap_to_subagent_prediction']:>+,} bytes above")
    # Model preference distribution
    n_gauss = sum(1 for r in manifest["per_tensor"] if r["best_model"] == "gaussian")
    n_laplace = manifest["n_tensors"] - n_gauss
    print(f"\nModel preference: {n_gauss} Gaussian + {n_laplace} Laplace tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
