#!/usr/bin/env python3
"""PR101 Joint-ADMM × continuous lossy_coarsening — Path B step 6 empirical anchor.

The high-EV combination of two empirical winners:
- Path B step 5: Joint-ADMM Lagrangian-allocation MECHANISM (allocates
  per-tensor distortion non-uniformly via λ bisection)
- Subagent D: continuous lossy_coarsening (per-tensor K-step rounding;
  achieves 156,344 B at 3.86% rel_err on PR101)

This tool composes them: use the Lagrangian allocation MECHANISM over the
LOSSY_COARSENING continuous K basis instead of the discrete sparsity grid.

## Hypothesis

Subagent D's tool runs per-tensor K search with a UNIFORM per-tensor
rel_err budget (each tensor finds best K subject to budget). The Lagrangian
formulation should let cheap-K tensors take MORE rel_err and expensive-K
tensors take LESS, within the same global RMS budget.

## Method

1. For each tensor and each K ∈ {1, 2, ..., 64}: compute (achieved per-tensor
   rel_err, contribution-to-joint-brotli proxy)
2. For Lagrangian λ: each tensor picks K minimizing
   (per_tensor_byte_proxy(K) + λ * rel_err(K)^2)
3. Encode joint payload with selected per-tensor Ks via subagent D's
   `encode_with_per_tensor_K`. Measure ACTUAL brotli bytes.
4. Bisect λ to satisfy global RMS rel_err target.
5. Compare to subagent D's UNIFORM-budget baseline (each tensor finds best K
   subject to budget = global RMS target).

## Falsification scope

`falsification_scope="lagrangian_allocation_x_lossy_coarsening_continuous_K_only"`.
The full primal-dual ADMM with consensus constraints remains UNTESTED.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

# Reuse subagent D's machinery
from pr101_lossy_coarsening_analytical import (  # noqa: E402
    TensorBlob,
    find_best_K_for_tensor,
    encode_with_per_tensor_K,
    ARCHIVE_OVERHEAD_BYTES,
)

TOOL_NAME = "tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_admm_x_lossy_coarsening_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Path-B-step-6 ADMM × continuous lossy_coarsening test]"

K_RANGE = list(range(1, 65))


def precompute_per_tensor_K_curves(tensors: list[TensorBlob]) -> list[list[dict]]:
    """For each tensor, compute (K, achieved_rel_err, per_tensor_byte_proxy)
    for K ∈ K_RANGE. byte_proxy = brotli(rounded_int8) of THIS tensor alone
    (an over-estimate of joint contribution but monotone in K)."""
    curves: list[list[dict]] = []
    for tb in tensors:
        rows = []
        abs_sum = float(np.abs(tb.raw).astype(np.float64).sum()) + 1e-12
        for K in K_RANGE:
            rounded = np.round(tb.raw / K) * K
            err = float(np.abs(rounded - tb.raw).astype(np.float64).sum())
            re = err / abs_sum
            rounded_i8 = rounded.clip(-127, 127).astype(np.int8)
            byte_proxy = len(brotli.compress(rounded_i8.tobytes(), quality=11, lgwin=16, lgblock=19))
            rows.append({"K": K, "rel_err": re, "byte_proxy": byte_proxy})
        curves.append(rows)
    return curves


def lagrangian_select_Ks(curves: list[list[dict]], lam: float) -> tuple[list[int], list[float]]:
    """For Lagrangian λ, each tensor picks K minimizing
    (byte_proxy + λ * rel_err^2)."""
    Ks: list[int] = []
    rel_errs: list[float] = []
    for tensor_curve in curves:
        cost = [r["byte_proxy"] + lam * r["rel_err"] ** 2 for r in tensor_curve]
        idx = int(np.argmin(cost))
        Ks.append(tensor_curve[idx]["K"])
        rel_errs.append(tensor_curve[idx]["rel_err"])
    return Ks, rel_errs


def bisect_admm_for_global_rms(
    tensors: list[TensorBlob],
    curves: list[list[dict]],
    rms_target: float,
) -> dict:
    """Bisect λ to satisfy global RMS rel_err target (joint-encoded)."""
    lo, hi = 0.0, 1e15
    last_admm = None
    for _ in range(80):
        mid = (lo + hi) / 2 if hi < 1e15 else lo * 10 + 1
        Ks, rel_errs = lagrangian_select_Ks(curves, mid)
        result = encode_with_per_tensor_K(tensors, Ks)
        rms = result["rel_err"]  # joint achieved RMS
        last_admm = {"lambda": mid, **result, "rms_rel_err": rms}
        if rms <= rms_target:
            hi = mid
        else:
            lo = mid
        if hi == lo or abs(hi - lo) < 1e-12:
            break
    return last_admm


def greedy_uniform_per_tensor_budget(
    tensors: list[TensorBlob],
    budget: float,
) -> dict:
    """Subagent D's approach: each tensor finds its own best K subject to
    per-tensor rel_err budget = global budget."""
    Ks = []
    for tb in tensors:
        K, _ = find_best_K_for_tensor(tb.raw, budget)
        Ks.append(K)
    return encode_with_per_tensor_K(tensors, Ks)


def run_experiment(state_dict_path: Path, rms_targets: list[float]) -> dict:
    import torch
    # weights_only=True per REVIEW-ENG C4 (2026-05-08): tensor-only state_dict.
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    tensors: list[TensorBlob] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        tensors.append(TensorBlob(name=name, raw=qt.q_i8.astype(np.int32).flatten()))
    n_tensors = len(tensors)

    print(f"  precomputing per-tensor K curves ({n_tensors} × {len(K_RANGE)})...")
    curves = precompute_per_tensor_K_curves(tensors)

    # Baseline: uniform K=1 (lossless)
    baseline = encode_with_per_tensor_K(tensors, [1] * n_tensors)

    comparison: list[dict] = []
    for rms_t in rms_targets:
        # Subagent-D-style greedy: per-tensor budget = global budget
        greedy = greedy_uniform_per_tensor_budget(tensors, rms_t)

        # ADMM: bisect λ for global RMS rel_err
        admm = bisect_admm_for_global_rms(tensors, curves, rms_t)

        savings = greedy["archive_bytes"] - admm["archive_bytes"]
        comparison.append({
            "rms_target": rms_t,
            "greedy_K_per_tensor_budget": {
                "archive_bytes": greedy["archive_bytes"],
                "rel_err": greedy["rel_err"],
                "Ks": greedy["Ks"],
            },
            "admm_K_lagrangian": {
                "archive_bytes": admm["archive_bytes"],
                "rel_err": admm["rel_err"],
                "lambda": admm["lambda"],
                "Ks": admm["Ks"],
            },
            "admm_savings_vs_greedy_bytes": savings,
        })

    best = max(comparison, key=lambda r: r["admm_savings_vs_greedy_bytes"])

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "lagrangian_allocation_x_lossy_coarsening_continuous_K_only",
        "input_state_dict": str(state_dict_path),
        "n_tensors": n_tensors,
        "K_range": [K_RANGE[0], K_RANGE[-1]],
        "rms_targets_swept": rms_targets,
        "baseline_lossless_bytes": baseline["archive_bytes"],
        "comparison_at_rms_targets": comparison,
        "best_admm_savings_bytes": best["admm_savings_vs_greedy_bytes"],
        "best_admm_savings_rms_target": best["rms_target"],
        "headline": (
            f"ADMM × lossy_coarsening continuous-K basis vs subagent-D-style greedy "
            f"per-tensor budget. Best savings: {best['admm_savings_vs_greedy_bytes']:+,} B at "
            f"rms_target={best['rms_target']}"
        ),
        "dispatch_blockers": [
            "byte_rel_err_proxy_only_no_score_test",
            "lagrangian_x_continuous_K_no_iterative_consensus_ADMM",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "iterative_primal_dual_ADMM_with_consensus",
            "score_aware_per_tensor_distortion_weights",
            "ADMM_x_continuous_K_with_CUDA_score_validation",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--rms-targets", type=float, nargs="+",
                   default=[0.01, 0.02, 0.0386, 0.05, 0.10])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    print(f"Path B step 6: ADMM × continuous lossy_coarsening empirical anchor")
    manifest = run_experiment(args.state_dict, args.rms_targets)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(f"  Baseline (uniform K=1, lossless): {manifest['baseline_lossless_bytes']:,} B\n")
    print(f"  {'rms_target':>10s} | {'greedy_b':>9s} {'greedy_re':>10s} | "
          f"{'admm_b':>9s} {'admm_re':>10s} {'λ':>10s} | {'savings':>8s}")
    for r in manifest["comparison_at_rms_targets"]:
        g = r["greedy_K_per_tensor_budget"]
        a = r["admm_K_lagrangian"]
        print(f"  {r['rms_target']:>10.4f} | {g['archive_bytes']:>9,} {g['rel_err']:>10.5f} | "
              f"{a['archive_bytes']:>9,} {a['rel_err']:>10.5f} {a['lambda']:>10.2e} | "
              f"{r['admm_savings_vs_greedy_bytes']:>+8,}")
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        target = next((r for r in manifest["comparison_at_rms_targets"]
                       if abs(r["rms_target"] - 0.0386) < 1e-9),
                      manifest["comparison_at_rms_targets"][0])
        evidence_row = {
            "technique": "joint_admm_x_continuous_lossy_coarsening",
            "empirical_archive_bytes": target["admm_K_lagrangian"]["archive_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "lagrangian_x_continuous_K_byte_anchor_at_rms_target_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "lagrangian_allocation_x_lossy_coarsening_continuous_K_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(rms_target=3.86%; admm_bytes={target['admm_K_lagrangian']['archive_bytes']:,}; "
                f"vs_greedy={target['admm_savings_vs_greedy_bytes']:+,}; "
                f"best_savings={manifest['best_admm_savings_bytes']:+,}@rms={manifest['best_admm_savings_rms_target']})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": ["lagrangian_x_continuous_K_basis"],
            "reactivation_criteria_remaining": manifest["reactivation_criteria_remaining"],
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
