#!/usr/bin/env python3
"""PR101 sensitivity-aware per-tensor quantization with Mallat wavelet
importance proxy (Phase A3-alt, council reactivation criterion #2 from
``feedback_pr101_sensitivity_aware_xavier_l2_proxy_DEFERRED_pending_real_hessian_20260508.md``).

The Xavier-L2 proxy was empirically falsified on PR101 substrate: at
``avg_budget=0.05/eta=1.0`` it landed at 159,979 B vs the uniform analytical
baseline at 156,344 B, a +3,635 B regression. The structural reason recorded in
the falsification memo is that magnitude-RMS is *not* compression-hardness on a
near-iid substrate — high-RMS tensors are not always the ones that compress
poorly, so an inverse-magnitude budget tilt has nothing to extract.

This sister tool replaces the importance proxy with **Mallat wavelet-coefficient
energy**: each tensor is reshaped to a 2D matrix and decomposed with a 2-level
Daubechies-4 wavelet decomposition. The importance score is the per-element sum
of squared *detail* coefficient magnitudes across all 6 detail subbands (LH/HL/HH
at level 1 and level 2). The intuition (Mallat 1989; Mallat & Zhong 1992) is that
wavelet detail energy captures local frequency content — tensors with high-
frequency content compress poorly relative to tensors with mostly low-frequency
(smooth) content. This is a closer analogue of "compression-hardness" than
magnitude-RMS without requiring scorer load.

Reuses the budget allocator and per-tensor K encoder from the Xavier-L2 tool;
ONLY the importance step is replaced. CPU-only, MPS-research-signal grade.

Reactivation: if Mallat ALSO regresses vs uniform, the next reactivation
criterion is real Hessian-trace importance (one CUDA forward+backward through
SegNet/PoseNet) or compression-hardness empirical proxy (per-tensor K=2 vs K=8
brotli-ratio comparison).

CLAUDE.md compliance:
- ``score_claim=False``, ``promotion_eligible=False``, ``byte_proxy_only=True``
- ``cuda_eval_worth_testing=False`` — proxy-only research signal, NOT a dispatch
  candidate (sister to the Xavier-L2 tool which set the same flags)
- ``dispatch_blockers=["awaiting_mallat_vs_xavier_vs_uniform_comparison",
  "missing_exact_cuda_auth_eval", "no_runtime_dequantize_path_built",
  "byte_closed_lossy_coarsening_runtime_packet_missing",
  "requires_exact_cuda_auth_eval_before_any_score_use",
  "proxy_rel_err_not_score_evidence"]``
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pywt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr101_lossy_coarsening_analytical import (  # noqa: E402
    PR101_BROTLI_BASELINE_BYTES,
    TensorBlob,
    collect_tensors,
    encode_with_per_tensor_K,
    find_best_K_for_tensor,
)
from pr101_sensitivity_aware_quantization import (  # noqa: E402
    TensorImportance,
    allocate_per_tensor_budgets,
)

TOOL_NAME = "tools/pr101_sensitivity_aware_mallat_wavelet.py"
SCHEMA_VERSION = "pr101_sensitivity_aware_mallat_wavelet.v1"
SENSITIVITY_PROXY = "mallat_wavelet"
WAVELET_NAME = "db4"
WAVELET_LEVEL = 2
DISPATCH_BLOCKERS = [
    "awaiting_mallat_vs_xavier_vs_uniform_comparison",
    "missing_exact_cuda_auth_eval",
    "no_runtime_dequantize_path_built",
    "byte_closed_lossy_coarsening_runtime_packet_missing",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "proxy_rel_err_not_score_evidence",
]


def _reshape_to_2d(arr: np.ndarray) -> np.ndarray:
    """Reshape a 0D/1D/3D/4D numeric array to a 2D matrix.

    Strategy:
      * 0D scalar     → (1, 1) padded.
      * 1D arrays     → near-square reshape (rows = ceil(sqrt(n)), pad with zeros).
      * 2D arrays     → unchanged.
      * 3D/4D arrays  → flatten leading dims into rows, keep last dim as cols.

    Returns a contiguous float64 array.

    .. note::
       For 1D arrays we *also* expose a 1D-wavelet path in
       :func:`_mallat_detail_energy_1d`; this 2D reshape is only used for
       genuine multi-dimensional tensors. Reshaping a 1D ramp to a 2D matrix
       destroys the original adjacency, so 1D inputs should be handled with
       1D wavedec for an interpretable detail-energy measurement.
    """
    if arr.ndim == 0:
        return np.asarray([[float(arr.item())]], dtype=np.float64)
    if arr.ndim == 1:
        n = arr.size
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        pad = rows * cols - n
        if pad > 0:
            padded = np.concatenate([arr.astype(np.float64), np.zeros(pad, dtype=np.float64)])
        else:
            padded = arr.astype(np.float64)
        return padded.reshape(rows, cols)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    # 3D / 4D / higher: collapse all but the last dimension into rows
    last = arr.shape[-1]
    rows = int(np.prod(arr.shape[:-1]))
    return arr.astype(np.float64).reshape(rows, last)


def _mallat_detail_energy_1d(signal: np.ndarray) -> float:
    """1D Mallat detail energy: sum of squared detail-coefficient magnitudes."""
    if signal.size == 0:
        return 0.0
    wavelet = pywt.Wavelet(WAVELET_NAME)
    filter_len = wavelet.dec_len
    if signal.size < filter_len:
        return 0.0
    max_level = pywt.dwt_max_level(signal.size, filter_len)
    level = min(WAVELET_LEVEL, max_level)
    if level < 1:
        return 0.0
    coeffs = pywt.wavedec(
        signal.astype(np.float64), wavelet=WAVELET_NAME, level=level, mode="periodization",
    )
    # coeffs = [cA_n, cD_n, ..., cD_1]; detail subbands are coeffs[1:].
    energy = 0.0
    for cD in coeffs[1:]:
        energy += float(np.sum(cD * cD))
    return energy


def _mallat_detail_energy(matrix: np.ndarray) -> float:
    """Sum of squared detail-coefficient magnitudes from a 2D db4 decomp.

    For matrices smaller than the wavelet filter length on the smaller axis,
    falls back to a 1D wavedec along the longer axis. If both axes are smaller
    than the filter length, returns 0.0 (no resolvable high-frequency content
    at this scale).
    """
    if matrix.size == 0:
        return 0.0
    if matrix.ndim != 2:
        raise ValueError(f"_mallat_detail_energy expects 2D, got shape {matrix.shape}")

    wavelet = pywt.Wavelet(WAVELET_NAME)
    filter_len = wavelet.dec_len  # db4 → 8
    min_dim = min(matrix.shape)
    max_dim = max(matrix.shape)

    if max_dim < filter_len:
        # Both axes too small for any meaningful decomposition at this filter.
        return 0.0

    if min_dim < filter_len:
        # Fall back to 1D wavedec along the long axis (preserves adjacency
        # structure when the natural shape is rectangular and short on one side,
        # e.g. (1728, 28) or (3, 18×3×3) after reshape).
        long_axis = int(np.argmax(matrix.shape))
        if long_axis == 1:
            rows = matrix
        else:
            rows = matrix.T
        # Sum 1D detail energies across all rows.
        energy = 0.0
        for r in rows:
            energy += _mallat_detail_energy_1d(np.ascontiguousarray(r))
        return energy

    max_level = pywt.dwt_max_level(min_dim, filter_len)
    level = min(WAVELET_LEVEL, max_level)
    if level < 1:
        return 0.0

    coeffs = pywt.wavedec2(matrix, wavelet=WAVELET_NAME, level=level, mode="periodization")
    # coeffs = [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    energy = 0.0
    for entry in coeffs[1:]:
        cH, cV, cD = entry
        energy += float(np.sum(cH * cH))
        energy += float(np.sum(cV * cV))
        energy += float(np.sum(cD * cD))
    return energy


def compute_mallat_wavelet_importance(
    tensors: list[TensorBlob],
) -> list[TensorImportance]:
    """Compute per-tensor importance via Mallat wavelet detail energy.

    For each tensor blob (np.int32 symbols in [-127, 127] from the PR101 quantizer):

    1. Reshape to a 2D matrix via :func:`_reshape_to_2d`.
    2. Apply :func:`_mallat_detail_energy` (2-level db4 decomposition).
    3. Importance := ``detail_energy / max(numel, 1)`` (per-element normalization
       so importance is comparable across tensors of different sizes).
    4. ``rms`` (legacy schema field) is reported as the same magnitude-RMS the
       Xavier-L2 proxy used, so downstream analysis can compare both proxies on
       the same TensorImportance schema.

    Raises:
        ValueError: if a tensor blob is empty (0 elements).
    """
    out: list[TensorImportance] = []
    for tb in tensors:
        symbols = tb.raw
        numel = int(symbols.size)
        if numel == 0:
            raise ValueError(f"empty tensor blob: {tb.name}")
        # legacy magnitude-RMS for cross-proxy comparison
        sym_f64 = symbols.astype(np.float64)
        rms = float(np.sqrt(np.mean(sym_f64 * sym_f64)))

        # Use native 1D wavedec for 1D tensors (preserves natural adjacency);
        # fall through to 2D reshape + wavedec2 for 2D+ tensors. For 1D arrays
        # smaller than the filter length (e.g. PR101 rgb_*.bias with 3 elems)
        # _mallat_detail_energy_1d returns 0.0 (well-defined fallback).
        if symbols.ndim == 1:
            energy = _mallat_detail_energy_1d(symbols)
        else:
            matrix = _reshape_to_2d(symbols)
            energy = _mallat_detail_energy(matrix)
        importance = energy / max(numel, 1)

        out.append(
            TensorImportance(
                name=tb.name,
                importance=importance,
                numel=numel,
                rms=rms,
            )
        )
    return out


def encode_sensitivity_weighted_mallat(
    tensors: list[TensorBlob],
    *,
    average_budget: float,
    eta: float = 1.0,
    brotli_quality: int = 11,
) -> dict:
    """Encode PR101 tensors with Mallat-wavelet-weighted per-tensor budgets."""
    importances = compute_mallat_wavelet_importance(tensors)
    budgets = allocate_per_tensor_budgets(
        importances, average_budget=average_budget, eta=eta,
    )
    Ks: list[int] = []
    achieved: list[float] = []
    for tb, budget in zip(tensors, budgets, strict=True):
        K, rel_err = find_best_K_for_tensor(tb.raw, budget)
        Ks.append(K)
        achieved.append(rel_err)
    enc = encode_with_per_tensor_K(tensors, Ks, brotli_quality)
    enc["sensitivity_proxy"] = SENSITIVITY_PROXY
    enc["wavelet"] = WAVELET_NAME
    enc["wavelet_level"] = WAVELET_LEVEL
    enc["eta"] = eta
    enc["average_budget"] = average_budget
    enc["per_tensor_importance"] = [
        {"name": ti.name, "importance": ti.importance, "numel": ti.numel, "rms": ti.rms}
        for ti in importances
    ]
    enc["per_tensor_budget"] = budgets
    enc["per_tensor_achieved_rel_err"] = achieved
    return enc


# Xavier-L2 falsification anchors (from the DEFERRED memo). Used to compute
# delta_vs_xavier_l2 columns when the same (avg_budget, eta) cell is swept.
XAVIER_L2_ANCHOR_BYTES: dict[tuple[float, float], int] = {
    (0.020, 0.0): 176_990,
    (0.020, 0.5): 177_432,
    (0.020, 1.0): 176_631,
    (0.050, 0.0): 156_344,
    (0.050, 0.5): 157_871,
    (0.050, 1.0): 159_979,
}
# The eta=0 (uniform) result is identical for both proxies (the allocator
# emits average_budget for every tensor regardless of importance), but we keep
# the row populated so the cross-reference table renders cleanly.
UNIFORM_ANCHOR_BYTES: dict[float, int] = {
    0.020: 176_990,
    0.050: 156_344,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument(
        "--average-budgets",
        type=str,
        default="0.020,0.050",
        help="Comma-separated AVERAGE rel_err budgets (matches Xavier-L2 sweep grid).",
    )
    p.add_argument(
        "--etas",
        type=str,
        default="0.0,0.5,1.0",
        help="Comma-separated tilt strengths (eta=0 == uniform baseline).",
    )
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    timestamp = _dt.datetime.now(tz=_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / f"experiments/results/pr101_sensitivity_aware_mallat_wavelet_{timestamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tensors = collect_tensors(args.state_dict)
    avg_budgets = [float(b) for b in args.average_budgets.split(",")]
    etas = [float(e) for e in args.etas.split(",")]

    print(f"[mallat-wavelet] state_dict: {args.state_dict}")
    print(f"[mallat-wavelet] {len(tensors)} tensors, "
          f"total {sum(t.raw.size for t in tensors):,} symbols")
    print(f"[mallat-wavelet] PR101 brotli baseline: "
          f"{PR101_BROTLI_BASELINE_BYTES:,} B")
    print(f"[mallat-wavelet] sensitivity proxy: {SENSITIVITY_PROXY} "
          f"({WAVELET_NAME}, level={WAVELET_LEVEL})")
    print()

    rows: list[dict] = []
    for ab in avg_budgets:
        for eta in etas:
            enc = encode_sensitivity_weighted_mallat(
                tensors,
                average_budget=ab,
                eta=eta,
                brotli_quality=args.brotli_quality,
            )
            uniform_b = UNIFORM_ANCHOR_BYTES.get(ab)
            xavier_b = XAVIER_L2_ANCHOR_BYTES.get((ab, eta))
            row = {
                "tool": TOOL_NAME,
                "schema_version": SCHEMA_VERSION,
                "average_budget": ab,
                "eta": eta,
                "archive_bytes": enc["archive_bytes"],
                "rel_err": enc["rel_err"],
                "delta_vs_pr101_baseline": enc["archive_bytes"]
                - PR101_BROTLI_BASELINE_BYTES,
                "delta_vs_uniform": (
                    enc["archive_bytes"] - uniform_b if uniform_b is not None else None
                ),
                "delta_vs_xavier_l2": (
                    enc["archive_bytes"] - xavier_b if xavier_b is not None else None
                ),
                "evidence_grade": "[byte-anchor; sensitivity_proxy=mallat_wavelet]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "byte_proxy_only": True,
                "cuda_eval_worth_testing": False,
                "sensitivity_proxy": SENSITIVITY_PROXY,
                "wavelet": WAVELET_NAME,
                "wavelet_level": WAVELET_LEVEL,
                "council_finding_reference": (
                    "Round 2 finding 10 reactivation criterion #2 "
                    "(Mallat wavelet-coefficient importance, Phase A3-alt)"
                ),
            }
            rows.append(row)
            delta_uni = row["delta_vs_uniform"]
            delta_xav = row["delta_vs_xavier_l2"]
            uni_str = f"{delta_uni:+,}" if delta_uni is not None else "n/a"
            xav_str = f"{delta_xav:+,}" if delta_xav is not None else "n/a"
            print(
                f"  avg_budget={ab:.3f} eta={eta:.1f}: "
                f"archive={enc['archive_bytes']:>7,} B "
                f"(Δ_baseline {row['delta_vs_pr101_baseline']:+,} B, "
                f"Δ_uniform {uni_str} B, "
                f"Δ_xavier_l2 {xav_str} B) "
                f"rel_err={enc['rel_err']:.4f}"
            )

    # Per-budget cross-proxy summary
    summary: list[dict] = []
    for ab in avg_budgets:
        ab_rows = [r for r in rows if r["average_budget"] == ab]
        baseline = next((r for r in ab_rows if r["eta"] == 0.0), None)
        best = min(ab_rows, key=lambda r: r["archive_bytes"])
        if baseline is None:
            continue
        savings_vs_uniform = baseline["archive_bytes"] - best["archive_bytes"]
        xavier_at_best = XAVIER_L2_ANCHOR_BYTES.get((ab, best["eta"]))
        summary.append(
            {
                "average_budget": ab,
                "uniform_baseline_archive_bytes": baseline["archive_bytes"],
                "best_mallat_archive_bytes": best["archive_bytes"],
                "best_eta": best["eta"],
                "savings_vs_uniform": savings_vs_uniform,
                "xavier_l2_at_best_eta": xavier_at_best,
                "delta_vs_xavier_l2_at_best_eta": (
                    best["archive_bytes"] - xavier_at_best
                    if xavier_at_best is not None else None
                ),
                "passes_council_threshold": savings_vs_uniform >= 3000,
            }
        )

    if rows:
        any_row_beats_uniform = any(
            r["delta_vs_uniform"] is not None and r["delta_vs_uniform"] < 0
            for r in rows
        )
        any_row_beats_xavier = any(
            r["delta_vs_xavier_l2"] is not None and r["delta_vs_xavier_l2"] < 0
            for r in rows
        )
    else:
        any_row_beats_uniform = False
        any_row_beats_xavier = False

    if any(s["passes_council_threshold"] for s in summary):
        verdict = "empirical_anchor"
    elif any_row_beats_uniform:
        verdict = "incremental_improvement_insufficient"
    elif any_row_beats_xavier:
        verdict = "incremental_improvement_insufficient"
    else:
        verdict = "DEFERRED_pending_real_hessian"

    manifest = {
        "tool": TOOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "state_dict_path": str(args.state_dict.resolve()),
        "sensitivity_proxy": SENSITIVITY_PROXY,
        "wavelet": WAVELET_NAME,
        "wavelet_level": WAVELET_LEVEL,
        "rows": rows,
        "summary": summary,
        "xavier_l2_reference_anchors": [
            {"average_budget": k[0], "eta": k[1], "archive_bytes": v}
            for k, v in sorted(XAVIER_L2_ANCHOR_BYTES.items())
        ],
        "council_finding_reference": (
            "Round 2 finding 10 reactivation criterion #2 "
            "(Mallat wavelet-coefficient importance, Phase A3-alt)"
        ),
        "evidence_grade": "[byte-anchor; sensitivity_proxy=mallat_wavelet]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "verdict": verdict,
    }
    manifest_path = args.output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")

    print()
    print("Cross-proxy summary (savings vs uniform, vs Xavier-L2):")
    for s in summary:
        verdict_str = "PASS" if s["passes_council_threshold"] else "FAIL"
        delta_xav = s["delta_vs_xavier_l2_at_best_eta"]
        delta_xav_str = f"{delta_xav:+,}" if delta_xav is not None else "n/a"
        print(
            f"  avg_budget={s['average_budget']:.3f}: "
            f"savings_vs_uniform={s['savings_vs_uniform']:+,} B  "
            f"Δ_xavier_l2_at_best_eta={delta_xav_str} B  "
            f"council_threshold={verdict_str}"
        )
    print()
    print(f"verdict: {verdict}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
