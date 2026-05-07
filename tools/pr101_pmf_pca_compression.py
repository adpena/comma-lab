#!/usr/bin/env python3
"""PCA-based PMF compression falsification probe for PR101.

The 28 per-tensor PMFs form a 28x255 matrix M where each row is a
fp16-quantized probability vector. A low-rank PMF hypothesis would make
the PMF side information compressible as a basis plus per-tensor
coefficients. This tool measures that hypothesis rather than assuming it.

This tool applies truncated PCA at rank K and measures:
  - Reconstruction error (max KL-divergence per tensor)
  - Total transmitted bytes:
      basis K x 255 fp16  +  per-tensor coefficients (28 x K fp16)
  - Comparison with brotli-concat-PMFs (3,513 bytes empirical)

The output is planning evidence only. It does not emit a decoder bitstream
or a score-affecting archive. Pure CPU + numpy. No GPU, no scorer load.

Usage::

    .venv/bin/python tools/pr101_pmf_pca_compression.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_pmf_pca.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
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
)

TOOL_NAME = "tools/pr101_pmf_pca_compression.py"
SCHEMA_VERSION = "pr101_pmf_pca_compression.v1"
N_CATEGORIES = 255
EVIDENCE_GRADE = "empirical"
EVIDENCE_SEMANTICS = "cpu_pmf_side_information_planning_probe"
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
REFERENCE_PER_TENSOR_RAW_PMF_BYTES = 190_718
REFERENCE_PER_TENSOR_BROTLI_PMF_BYTES = 179_951
REFERENCE_JOINT_FLOOR_PREDICTION_BYTES = 155_000


def _build_pmf_matrix(state_dict: dict) -> np.ndarray:
    """Build a 28 x N_CATEGORIES matrix of per-tensor empirical PMFs."""
    pmfs: list[np.ndarray] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        counts = np.bincount(symbols, minlength=N_CATEGORIES).astype(np.float64)
        pmf = (counts + 1.0) / (counts.sum() + N_CATEGORIES)  # Laplace smooth
        pmfs.append(pmf)
    return np.array(pmfs)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q). Direction matters; use this to bound coding loss."""
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log2(p / q)))


def _theoretical_bits_at_pmf(symbols: np.ndarray, pmf: np.ndarray) -> float:
    """Shannon bits to encode `symbols` against `pmf`."""
    counts = np.bincount(symbols, minlength=N_CATEGORIES).astype(np.float64)
    bits = 0.0
    for c, p in zip(counts, pmf, strict=False):
        if c > 0 and p > 0:
            bits += c * (-math.log2(p))
    return bits


def pca_compress(state_dict_path: Path, k_values: list[int]) -> dict[str, Any]:
    """Sweep PCA rank K, measuring overhead bytes + payload bytes."""
    input_bytes = state_dict_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise SystemExit(f"loaded {state_dict_path} is not a dict")
    M = _build_pmf_matrix(state_dict)  # (28, 255)
    max_rank = min(M.shape)
    if not k_values:
        raise SystemExit("at least one K value is required")
    bad_k = [k for k in k_values if k <= 0 or k > max_rank]
    if bad_k:
        raise SystemExit(f"K values outside valid range [1, {max_rank}]: {bad_k}")

    # Center the PMFs around their mean (so PCA captures variation, not the
    # mean shape)
    mean_pmf = M.mean(axis=0)  # (255,)
    M_centered = M - mean_pmf  # (28, 255)

    # SVD: M_centered = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    # U: (28, 28), S: (28,), Vt: (28, 255)

    # Pre-compute per-tensor symbols for payload calc
    per_tensor_symbols: list[np.ndarray] = []
    for name, _ in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, state_dict[name], n_quant=N_QUANT)
        symbols = (qt.q_i8.astype(np.int32) + 127).flatten()
        per_tensor_symbols.append(symbols)

    # Brotli baselines
    raw_pmfs_bytes = M.astype(np.float16).tobytes()
    brotli_concat_bytes = len(brotli.compress(raw_pmfs_bytes, quality=11))

    sweep_rows: list[dict[str, Any]] = []
    for k in k_values:
        # Truncated PCA: keep first K basis vectors and coefficients
        U_k = U[:, :k]  # (28, k)
        S_k = S[:k]     # (k,)
        Vt_k = Vt[:k]   # (k, 255)
        # Reconstruct centered PMFs
        M_recon = U_k @ np.diag(S_k) @ Vt_k + mean_pmf  # (28, 255)
        # Renormalize each row + clip negatives
        M_recon = np.clip(M_recon, 1e-8, None)
        M_recon = M_recon / M_recon.sum(axis=1, keepdims=True)

        # Overhead: basis (k x 255 fp16) + mean (255 fp16)
        # + per-tensor coefficients (28 x k fp16).
        basis_bytes = k * N_CATEGORIES * 2  # k x 255 x 2
        mean_bytes = N_CATEGORIES * 2       # 255 x 2 = 510
        coef_bytes = 28 * k * 2             # 28 x k x 2
        raw_overhead_bytes = basis_bytes + mean_bytes + coef_bytes
        # Also try brotli-compressing the overhead blob
        overhead_blob = (
            (Vt_k.astype(np.float16).tobytes())
            + (mean_pmf.astype(np.float16).tobytes())
            + ((U_k * S_k).astype(np.float16).tobytes())
        )
        brotli_overhead_bytes = len(brotli.compress(overhead_blob, quality=11))

        # Per-tensor payload at the reconstructed PMF
        total_theoretical_bits = 0.0
        kl_divergences = []
        for i, symbols in enumerate(per_tensor_symbols):
            recon_pmf = M_recon[i]
            true_pmf = M[i]
            kl = _kl_divergence(true_pmf, recon_pmf)
            kl_divergences.append(kl)
            bits = _theoretical_bits_at_pmf(symbols, recon_pmf)
            total_theoretical_bits += bits

        total_payload_bytes = math.ceil(total_theoretical_bits / 8.0)
        archive_overhead = 16_094

        # Two final budgets: raw-overhead and brotli-compressed-overhead
        total_with_raw = total_payload_bytes + raw_overhead_bytes + archive_overhead
        total_with_brotli = total_payload_bytes + brotli_overhead_bytes + archive_overhead

        sweep_rows.append({
            "k": k,
            "basis_bytes_raw_fp16": basis_bytes,
            "mean_bytes_raw_fp16": mean_bytes,
            "coef_bytes_raw_fp16": coef_bytes,
            "raw_overhead_bytes": raw_overhead_bytes,
            "brotli_compressed_overhead_bytes": brotli_overhead_bytes,
            "max_kl_divergence_per_tensor": max(kl_divergences),
            "mean_kl_divergence_per_tensor": float(np.mean(kl_divergences)),
            "theoretical_payload_bits": total_theoretical_bits,
            "theoretical_payload_bytes": total_payload_bytes,
            "total_archive_with_raw_overhead": total_with_raw,
            "total_archive_with_brotli_overhead": total_with_brotli,
        })

    best = min(sweep_rows, key=lambda r: r["total_archive_with_brotli_overhead"])
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
            "planning_probe_only",
            "no_actual_pca_pmf_decoder_bitstream",
            "no_archive_substitution_performed",
            "missing_exact_cuda_auth_eval",
        ],
        "n_tensors": 28,
        "n_categories": N_CATEGORIES,
        "raw_pmf_bytes_fp16": len(raw_pmfs_bytes),
        "brotli_concat_pmf_bytes": brotli_concat_bytes,
        "best_k_by_total_archive_with_brotli_overhead": best,
        "singular_values_spectrum": [float(s) for s in S],
        "spectrum_decay_pct": [
            float(S[i] / S[0]) for i in range(min(len(S), 10))
        ],
        "comparison_brotli_optuna_optimum_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "comparison_per_tensor_constriction_with_raw_pmfs_bytes": REFERENCE_PER_TENSOR_RAW_PMF_BYTES,
        "comparison_per_tensor_constriction_with_brotli_pmfs_bytes": REFERENCE_PER_TENSOR_BROTLI_PMF_BYTES,
        "subagent_predicted_deployable_joint_floor_bytes": REFERENCE_JOINT_FLOOR_PREDICTION_BYTES,
        "k_sweep": sweep_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-values", type=str, default="1,2,3,4,5,6,8,10,15,20",
                        help="Comma-separated K values to sweep")
    args = parser.parse_args(argv)
    if not args.state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict_path}")

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    manifest = pca_compress(args.state_dict_path, k_values=k_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Spectrum decay (first 10):     {[f'{x:.3f}' for x in manifest['spectrum_decay_pct']]}")
    print(f"Brotli concat PMF baseline:    {manifest['brotli_concat_pmf_bytes']:,} bytes")
    print()
    print(
        f"{'K':>3} {'raw_ovh':>9} {'br_ovh':>8} {'payload':>10} "
        f"{'total_brotli':>13} {'vs_brotli178144':>18} {'max_KL':>9}"
    )
    for r in manifest["k_sweep"]:
        delta = r["total_archive_with_brotli_overhead"] - REFERENCE_BROTLI_OPTUNA_BYTES
        print(
            f"{r['k']:>3} {r['raw_overhead_bytes']:>9,} "
            f"{r['brotli_compressed_overhead_bytes']:>8,} "
            f"{r['theoretical_payload_bytes']:>10,} "
            f"{r['total_archive_with_brotli_overhead']:>13,} "
            f"{delta:>+18,} {r['max_kl_divergence_per_tensor']:>9.4f}"
        )
    print()
    best = min(manifest["k_sweep"], key=lambda r: r["total_archive_with_brotli_overhead"])
    print(
        f"Best K={best['k']}: total {best['total_archive_with_brotli_overhead']:,} bytes "
        f"(vs brotli {REFERENCE_BROTLI_OPTUNA_BYTES:,}: "
        f"{best['total_archive_with_brotli_overhead'] - REFERENCE_BROTLI_OPTUNA_BYTES:+,})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
