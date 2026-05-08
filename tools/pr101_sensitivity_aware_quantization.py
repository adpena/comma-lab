#!/usr/bin/env python3
"""PR101 sensitivity-aware per-tensor quantization (Phase A2).

Council mandate (commit 231abcee, `.omx/research/grand_council_extreme_rigor_track_1_20260508.md`):
Decision 3 — sensitivity-aware quantization is UNANIMOUS / HIGHEST EV-PER-$
($1 CPU work, predicted -8-15 KB savings vs uniform-budget allocation at fixed
average rel_err on PR101 substrate).

The existing analytical tool (`tools/pr101_lossy_coarsening_analytical.py`)
allocates a UNIFORM rel_err budget across all 28 PR101 tensors. This tool
applies UNIWARD-style discipline at the weight level: tensors with higher
importance to SegNet/PoseNet output get a TIGHTER per-tensor rel_err budget;
tensors with lower importance get a LOOSER budget. The total byte count drops
because the lower-importance tensors compress more aggressively, while the
high-importance tensors maintain fidelity where it matters.

CPU-only Fisher-diagonal proxy
==============================

Computing the true Hessian-trace per tensor requires forward+backward through
SegNet/PoseNet on actual frames — heavy CUDA work. As a CPU-feasible proxy,
this tool uses the **Xavier-aware L2 norm**:

    importance(W) = ||W||_2 / sqrt(numel(W))

This is the per-element RMS amplitude of the tensor. It approximates the
diagonal Fisher information per parameter (under the assumption that gradient
magnitudes scale with weight magnitudes, which holds for ReLU-class
activations). Tensors with high RMS amplitude are quantization-sensitive;
tensors near zero are quantization-insensitive.

This proxy is INDEPENDENT of any score-gradient computation. It is the
council's "$1 CPU" version of Decision 3. A future refinement (Phase A3-alt
Mallat wavelet importance, council finding) can replace this with
wavelet-coefficient importance from a single CPU forward pass.

Wire format
-----------
Same as the analytical tool: 28 × 1 byte per-tensor K + brotli payload +
canonical PR101 archive overhead.

CLAUDE.md compliance: pure-CPU analytical, evidence tagged
``[byte-anchor; sensitivity_proxy=xavier_l2]``. No score claim, no promotion
eligibility, no scorer load. The savings are a byte-side proxy that must be
runtime-validated (decode + exact CUDA auth eval) before any score impact.

Falsification: Per council memo, A2 PASSES if sensitivity-weighted produces
≥3 KB additional savings vs uniform-budget at the same average rel_err on
PR101 substrate. A2 FAILS if savings <1 KB (would suggest the L2-norm proxy
doesn't track real importance — re-tag measured-config-retired, NOT KILL).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import brotli  # noqa: F401  imported for parity with the analytical tool
import numpy as np
import torch  # noqa: F401  imported for parity with the analytical tool

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pr101_lossy_coarsening_analytical import (  # noqa: E402
    ARCHIVE_OVERHEAD_BYTES,
    PR101_BROTLI_BASELINE_BYTES,
    TensorBlob,
    collect_tensors,
    encode_with_per_tensor_K,
    find_best_K_for_tensor,
)
from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA  # noqa: E402

TOOL_NAME = "tools/pr101_sensitivity_aware_quantization.py"
SCHEMA_VERSION = "pr101_sensitivity_aware_quantization.v1"
SENSITIVITY_PROXY = "xavier_l2"


@dataclass(frozen=True)
class TensorImportance:
    """Per-tensor importance score under the Xavier-aware L2 proxy."""

    name: str
    importance: float
    numel: int
    rms: float


def compute_xavier_l2_importance(tensors: list[TensorBlob]) -> list[TensorImportance]:
    """Compute per-tensor importance via Xavier-aware L2 norm.

    importance(W) = sqrt( mean(W^2) ) = ||W||_2 / sqrt(numel(W))

    Returns one TensorImportance per input tensor in the same order as the
    PR101 FIXED_STATE_SCHEMA.
    """
    out: list[TensorImportance] = []
    for tb in tensors:
        symbols = tb.raw.astype(np.float64)
        numel = int(symbols.size)
        if numel == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(symbols * symbols)))
        out.append(
            TensorImportance(
                name=tb.name,
                importance=rms,
                numel=numel,
                rms=rms,
            )
        )
    return out


def allocate_per_tensor_budgets(
    importances: list[TensorImportance],
    *,
    average_budget: float,
    eta: float = 1.0,
    floor: float = 0.001,
    cap: float = 0.20,
) -> list[float]:
    """Allocate per-tensor rel_err budgets weighted inversely by importance.

    The total budget mass is conserved: sum(budgets * numel) ≈
    average_budget * sum(numel). High-importance tensors receive tighter
    budgets (smaller rel_err allowed); low-importance receive looser.

    Args:
        importances: per-tensor importance scores.
        average_budget: target average rel_err across the substrate.
        eta: tilt strength. eta=0 → uniform allocation (== analytical
            baseline). eta=1 → fully inverse-proportional. Higher eta
            concentrates budget more on low-importance tensors.
        floor: minimum per-tensor budget (avoid zero-budget for any tensor).
        cap: maximum per-tensor budget (avoid catastrophic distortion on
            any single tensor).

    Returns:
        List of per-tensor rel_err budgets in the same order as input.
    """
    if average_budget < 0.0:
        raise ValueError("average_budget must be non-negative")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    if floor < 0.0 or cap <= floor:
        raise ValueError("require 0 <= floor < cap")

    n = len(importances)
    if n == 0:
        return []

    total_numel = sum(ti.numel for ti in importances)
    if total_numel == 0:
        return [average_budget] * n

    imps = np.array([ti.importance for ti in importances], dtype=np.float64)
    numels = np.array([ti.numel for ti in importances], dtype=np.float64)

    # Inverse weights: w_i = 1 / (importance_i + epsilon)^eta
    eps = 1e-9 + float(imps[imps > 0].min()) * 0.01 if (imps > 0).any() else 1e-9
    inv_weights = 1.0 / np.power(imps + eps, eta)

    # Normalize so the parameter-mass-weighted average equals average_budget
    raw_budgets = inv_weights * average_budget
    weighted_avg = float(np.sum(raw_budgets * numels) / total_numel)
    if weighted_avg <= 0.0:
        return [average_budget] * n
    scale = average_budget / weighted_avg
    budgets = raw_budgets * scale

    # Clamp
    budgets = np.clip(budgets, floor, cap)

    # Renormalize after clamp (best-effort; clamp may drift mean by a bit)
    weighted_avg_post = float(np.sum(budgets * numels) / total_numel)
    if weighted_avg_post > 0.0 and abs(weighted_avg_post - average_budget) > 1e-6:
        # one more rescale pass; only the unclamped budgets respond
        unclamped_mask = (budgets > floor + 1e-9) & (budgets < cap - 1e-9)
        if unclamped_mask.any():
            adjustment = (
                (average_budget - float(np.sum(budgets[~unclamped_mask] * numels[~unclamped_mask]) / total_numel))
                / float(np.sum(budgets[unclamped_mask] * numels[unclamped_mask]) / total_numel)
            )
            if adjustment > 0:
                budgets[unclamped_mask] *= adjustment
                budgets = np.clip(budgets, floor, cap)

    return budgets.tolist()


def encode_sensitivity_weighted(
    tensors: list[TensorBlob],
    *,
    average_budget: float,
    eta: float = 1.0,
    brotli_quality: int = 11,
) -> dict:
    """Encode PR101 tensors with sensitivity-weighted per-tensor budgets.

    Returns a dict containing the encoder result, the per-tensor importance
    scores, the allocated budgets, and the achieved per-tensor rel_errs.
    """
    importances = compute_xavier_l2_importance(tensors)
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
    enc["eta"] = eta
    enc["average_budget"] = average_budget
    enc["per_tensor_importance"] = [
        {"name": ti.name, "importance": ti.importance, "numel": ti.numel}
        for ti in importances
    ]
    enc["per_tensor_budget"] = budgets
    enc["per_tensor_achieved_rel_err"] = achieved
    return enc


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
        default="0.02,0.03,0.05",
        help="Comma-separated AVERAGE rel_err budgets across the substrate.",
    )
    p.add_argument(
        "--etas",
        type=str,
        default="0.0,0.5,1.0,2.0",
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
            / f"experiments/results/pr101_sensitivity_aware_quant_{timestamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tensors = collect_tensors(args.state_dict)
    avg_budgets = [float(b) for b in args.average_budgets.split(",")]
    etas = [float(e) for e in args.etas.split(",")]

    rows: list[dict] = []
    for ab in avg_budgets:
        for eta in etas:
            enc = encode_sensitivity_weighted(
                tensors,
                average_budget=ab,
                eta=eta,
                brotli_quality=args.brotli_quality,
            )
            row = {
                "tool": TOOL_NAME,
                "schema_version": SCHEMA_VERSION,
                "average_budget": ab,
                "eta": eta,
                "archive_bytes": enc["archive_bytes"],
                "rel_err": enc["rel_err"],
                "delta_vs_pr101_baseline": enc["archive_bytes"]
                - PR101_BROTLI_BASELINE_BYTES,
                "evidence_grade": "[byte-anchor; sensitivity_proxy=xavier_l2]",
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "byte_proxy_only": True,
                "sensitivity_proxy": SENSITIVITY_PROXY,
                "council_finding_reference": "Round 2 finding 10",
            }
            rows.append(row)
            print(
                f"  avg_budget={ab:.3f} eta={eta:.1f}: "
                f"archive={enc['archive_bytes']:>7,} B "
                f"(Δ {row['delta_vs_pr101_baseline']:+,} B) "
                f"rel_err={enc['rel_err']:.4f}"
            )

    # Find the best (lowest archive_bytes) across the sweep at each
    # average_budget; compare to the eta=0 uniform baseline at the same budget.
    summary: list[dict] = []
    for ab in avg_budgets:
        ab_rows = [r for r in rows if r["average_budget"] == ab]
        baseline = next((r for r in ab_rows if r["eta"] == 0.0), None)
        best = min(ab_rows, key=lambda r: r["archive_bytes"])
        if baseline is None:
            continue
        savings_vs_uniform = baseline["archive_bytes"] - best["archive_bytes"]
        summary.append(
            {
                "average_budget": ab,
                "uniform_baseline_archive_bytes": baseline["archive_bytes"],
                "best_sensitivity_weighted_archive_bytes": best["archive_bytes"],
                "best_eta": best["eta"],
                "savings_vs_uniform": savings_vs_uniform,
                "passes_council_threshold": savings_vs_uniform >= 3000,
            }
        )

    manifest = {
        "tool": TOOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "state_dict_path": str(args.state_dict.resolve()),
        "rows": rows,
        "summary": summary,
        "council_finding_reference": "Round 2 finding 10",
        "evidence_grade": "[byte-anchor; sensitivity_proxy=xavier_l2]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "dispatch_blockers": [
            "missing_exact_cuda_auth_eval",
            "no_runtime_dequantize_path_built",
            "byte_closed_lossy_coarsening_runtime_packet_missing",
            "requires_exact_cuda_auth_eval_before_any_score_use",
            "proxy_rel_err_not_score_evidence",
        ],
    }
    manifest_path = args.output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}")

    if summary:
        print("\nCouncil-threshold check (savings vs uniform >= 3 KB):")
        for s in summary:
            verdict = "PASS" if s["passes_council_threshold"] else "FAIL"
            print(
                f"  avg_budget={s['average_budget']:.3f}: "
                f"savings={s['savings_vs_uniform']:+,} B  [{verdict}]"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
