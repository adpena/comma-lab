#!/usr/bin/env python3
"""PCA-based PMF compression for PR101.

The 28 per-tensor PMFs form a 28×255 matrix M where each row is a
fp16-quantized probability vector. Empirically these PMFs are highly
correlated (all near-Laplacian peaks at zero) — the SVD rank is bounded
by 28 but the effective rank is far smaller (~3-5).

This tool applies truncated PCA at rank K and measures:
  - Reconstruction error (max KL-divergence per tensor)
  - Total transmitted bytes:
      basis K × 255 fp16  +  per-tensor coefficients (28 × K fp16)
  - Comparison with brotli-concat-PMFs (3,513 bytes empirical)

Predicted at K=3: ~1,700 bytes total PMF overhead → 1.8 KB savings vs
brotli concat. Combined with per-tensor empirical range coder: total
archive = 160,344 + 1,700 + 16,094 = 178,138 B ≈ tied with brotli.

K=5 sweep: trades 1 KB more overhead for likely <0.1% payload increase.
Net: should beat brotli by 0.5-1.5 KB.

Pure CPU + numpy. No GPU, no scorer load.

Usage::

    .venv/bin/python tools/pr101_pmf_pca_compression.py \\
        --state-dict-path experiments/.../pr101_decoder_state_dict.pt \\
        --output reports/pr101_pmf_pca.json
"""
from __future__ import annotations

import argparse
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


def _build_pmf_matrix(state_dict: dict) -> np.ndarray:
    """Build a 28 × N_CATEGORIES matrix of per-tensor empirical PMFs."""
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
    """KL(p || q) — direction matters; use this to bound coding loss."""
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
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    M = _build_pmf_matrix(state_dict)  # (28, 255)

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

        # Overhead: basis (k × 255 fp16) + mean (255 fp16) + per-tensor coefs (28 × k fp16)
        basis_bytes = k * N_CATEGORIES * 2  # k × 255 × 2
        mean_bytes = N_CATEGORIES * 2       # 255 × 2 = 510
        coef_bytes = 28 * k * 2             # 28 × k × 2
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

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "input_state_dict": str(state_dict_path),
        "n_tensors": 28,
        "n_categories": N_CATEGORIES,
        "raw_pmf_bytes_fp16": len(raw_pmfs_bytes),
        "brotli_concat_pmf_bytes": brotli_concat_bytes,
        "singular_values_spectrum": [float(s) for s in S],
        "spectrum_decay_pct": [
            float(S[i] / S[0]) for i in range(min(len(S), 10))
        ],
        "comparison_brotli_optuna_optimum_bytes": 178144,
        "comparison_per_tensor_constriction_with_raw_pmfs_bytes": 190718,
        "comparison_per_tensor_constriction_with_brotli_pmfs_bytes": 179951,
        "subagent_predicted_deployable_joint_floor_bytes": 155000,
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
    print(f"\n{'K':>3} {'raw_ovh':>9} {'br_ovh':>8} {'payload':>10} {'total_brotli':>13} {'vs_brotli178144':>18} {'max_KL':>9}")
    for r in manifest["k_sweep"]:
        delta = r["total_archive_with_brotli_overhead"] - 178144
        print(f"{r['k']:>3} {r['raw_overhead_bytes']:>9,} {r['brotli_compressed_overhead_bytes']:>8,} "
              f"{r['theoretical_payload_bytes']:>10,} {r['total_archive_with_brotli_overhead']:>13,} "
              f"{delta:>+18,} {r['max_kl_divergence_per_tensor']:>9.4f}")
    print()
    best = min(manifest["k_sweep"], key=lambda r: r["total_archive_with_brotli_overhead"])
    print(f"Best K={best['k']}: total {best['total_archive_with_brotli_overhead']:,} bytes "
          f"(vs brotli 178,144: {best['total_archive_with_brotli_overhead'] - 178144:+,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
