#!/usr/bin/env python3
# ADMM_WAIVED:B4-reviewed historical/planning naming; docstrings or delegated coordinator code clarify whether this is Lagrangian, bridge, or actual iterative ADMM.
"""PR101 Lagrangian per-tensor allocation — Path B step 5 empirical anchor.

NAMING CORRECTION (REVIEW-MATH, 2026-05-08, Dykstra council finding)
--------------------------------------------------------------------
The historical name "Joint-ADMM" is mathematically misleading: this tool
implements **Lagrangian per-tensor allocation** via λ-bisection over
INDEPENDENT per-tensor argmin problems, NOT iterative primal-dual ADMM with
consensus / dual-variable updates. The ``rho_init`` knob, splitting
operators, and primal-dual update rules of full ADMM are absent here.

The full primal-dual ADMM with consensus constraints lives in
``src/tac/joint_admm_coordinator.py`` (Op_GammaJointADMM); this file is a
distinct simpler mechanism. To avoid downstream confusion, the **technique**
field in evidence rows is now ``lagrangian_per_tensor_allocation`` (was
``joint_admm_lagrangian_allocation``); historical rows are NOT rewritten,
they get a ``renamed_to`` forward reference.

The Ω-OPT design predicts -5bp from coordinated cross-component allocation
(0.105 vs HStack 0.110). This tool tests whether per-tensor distortion
allocation via Lagrangian λ beats the uniform-per-tensor-budget greedy
selector tested in Path B step 4.

## Config space enumeration

Technique class: Joint-ADMM cross-component byte/distortion allocation
Full config space: { lagrangian_per_tensor_allocation,
                     primal_dual_ADMM_iterative,
                     dual_decomposition,
                     proximal_gradient }
This tool tests: { lagrangian_per_tensor_allocation } — the simplest
form (no iterative ADMM updates, just λ bisection over per-tensor convex
hulls).
Falsification scope: this tool can constrain the Lagrangian-allocation
form. The full primal-dual ADMM with iterative updates remains UNTESTED.

## Method

Per-tensor cost curve: from Path B step 4's discrete codec measurements
({brotli_only, sparsity@α=0.3/0.5/0.7/0.9}), build the per-tensor convex
hull in (rel_err, bytes) space.

Greedy (Path B step 4 baseline): each tensor gets the same global rel_err
budget; pick most aggressive admissible codec.

ADMM-allocation (this anchor): for a chosen Lagrangian multiplier λ, each
tensor independently solves:
    min_codec  bytes(tensor, codec) + λ * rel_err(tensor, codec)^2
which is per-tensor and trivial. λ is then adjusted by bisection until
the sum-of-squared-rel_err constraint is met.

This isolates the JOINT-ALLOCATION value: ADMM lets cheap-distortion
tensors take MORE distortion and expensive-distortion tensors take LESS,
within the same global budget.

## Expected result

If per-tensor cost curves differ meaningfully, Joint-ADMM beats greedy.
If curves are uniform (PR101's case based on Path B step 4 evidence),
savings are modest. Either way: a clean empirical anchor for the predicted
-5bp Joint-ADMM benefit at the cross-component allocation layer.

## CLAUDE.md compliance

family_falsified=False; falsification_scope set to the specific test;
ready_for_exact_eval_dispatch=False (CPU byte+rel_err proxy, not score).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.cost_curves import (  # noqa: E402
    greedy_uniform_per_tensor_budget_sparsity,
    precompute_per_tensor_sparsity_curves,
)
from tac.optimization.lagrangian_per_tensor_allocation import (  # noqa: E402
    LagrangianPerTensorAllocator,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_omega_opt_joint_admm_allocation_empirical.py"
# v2 schema bump for REVIEW-MATH naming clarification: technique is
# Lagrangian per-tensor allocation, not primal-dual ADMM.
SCHEMA_VERSION = "pr101_omega_opt_lagrangian_per_tensor_allocation_empirical.v2"
SCHEMA_VERSION_HISTORICAL = "pr101_omega_opt_joint_admm_allocation_empirical.v1"
EVIDENCE_GRADE = (
    "[CPU-prep faithful Path-B-step-5 Lagrangian-per-tensor-allocation test "
    "(historical name: Joint-ADMM Lagrangian-allocation)]"
)


# NOTE: encode_brotli_only, encode_sparsity_alpha, the per-tensor sparsity
# curve precompute, the greedy-budget selector, and the Lagrangian allocator
# all delegate to the canonical primitives in tac.codec / tac.optimization.
# This file is a thin orchestrator + CLI for the Path B step 5 anchor.


def measure_curves(
    quantized: list[tuple[str, np.ndarray]], alphas: list[float]
) -> list[list[dict]]:
    """Per-tensor sparsity curves (delegates to tac.codec.cost_curves)."""
    return precompute_per_tensor_sparsity_curves(quantized, alphas)


def greedy_uniform_budget(
    curves: list[list[dict]], budget: float
) -> tuple[int, float]:
    """Greedy uniform-budget selector (delegates to tac.codec.cost_curves)."""
    return greedy_uniform_per_tensor_budget_sparsity(curves, budget)


def lagrangian_allocate(
    curves: list[list[dict]], lam: float
) -> tuple[int, float, list[float]]:
    """Lagrangian per-tensor selection (delegates to tac.optimization)."""
    res = LagrangianPerTensorAllocator().allocate(curves, lam)
    return res.total_bytes, res.rel_err, res.per_tensor_rel_errs


def bisect_lambda_for_rms_target(
    curves: list[list[dict]], rms_target: float
) -> dict:
    """λ-bisection (delegates to tac.optimization).

    Returns the historical key shape used by the manifest writer.
    """
    res = LagrangianPerTensorAllocator().bisect_for_rms_target(
        curves, rms_target, max_iter=60, lam_hi=1e12
    )
    return {
        "lambda": res.lam,
        "total_bytes": res.total_bytes,
        "achieved_rms_rel_err": res.rel_err,
        "per_tensor_rel_errs": res.per_tensor_rel_errs,
        "n_sparsified": sum(1 for e in res.per_tensor_rel_errs if e > 0),
    }


def run_experiment(state_dict_path: Path, alphas: list[float], rms_targets: list[float]) -> dict:
    import torch
    # weights_only=True per REVIEW-ENG C4 (2026-05-08): tensor-only state_dict.
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    quantized: list[tuple[str, np.ndarray]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        quantized.append((name, qt.q_i8.flatten().astype(np.int8)))
    n_tensors = len(quantized)

    print(f"  measuring per-tensor cost curves ({n_tensors} × {len(alphas) + 1} = {n_tensors * (len(alphas) + 1)} encodings)...")
    curves = measure_curves(quantized, alphas)

    baseline_bytes = sum(c[0]["bytes"] for c in curves)  # all-brotli

    comparison: list[dict] = []
    for rms_t in rms_targets:
        # Greedy with uniform budget = rms_t (per-tensor budget = global)
        g_bytes, g_rms = greedy_uniform_budget(curves, rms_t)

        # ADMM-allocation: bisect λ for RMS rel_err ≤ rms_t
        admm = bisect_lambda_for_rms_target(curves, rms_t)

        # ADMM saves vs greedy at the same achieved rel_err
        savings_vs_greedy = g_bytes - admm["total_bytes"]

        comparison.append({
            "rms_target": rms_t,
            "greedy_bytes": g_bytes,
            "greedy_rms": g_rms,
            "admm_lambda": admm["lambda"],
            "admm_bytes": admm["total_bytes"],
            "admm_rms": admm["achieved_rms_rel_err"],
            "admm_n_sparsified": admm["n_sparsified"],
            "admm_savings_vs_greedy_bytes": savings_vs_greedy,
        })

    # The winning case for ADMM
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
        "falsification_scope": "lagrangian_allocation_no_iterative_ADMM_only",
        "input_state_dict": str(state_dict_path),
        "n_tensors": n_tensors,
        "alphas_swept": alphas,
        "rms_targets_swept": rms_targets,
        "baseline_lossless_bytes": baseline_bytes,
        "comparison_at_rms_targets": comparison,
        "best_admm_savings_bytes": best["admm_savings_vs_greedy_bytes"],
        "best_admm_savings_rms_target": best["rms_target"],
        "headline": (
            f"Joint-ADMM Lagrangian-allocation vs greedy uniform-per-tensor budget. "
            f"Best ADMM advantage: {best['admm_savings_vs_greedy_bytes']:+,} B at "
            f"rms_target={best['rms_target']} (greedy={best['greedy_bytes']:,} B, "
            f"admm={best['admm_bytes']:,} B)"
        ),
        "dispatch_blockers": [
            "byte_rel_err_proxy_only_no_score_test",
            "lagrangian_allocation_only_iterative_ADMM_NOT_tested",
            "discrete_codec_basis_only_continuous_lossy_coarsening_NOT_tested_jointly",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "iterative_primal_dual_ADMM_with_consensus_constraints",
            "joint_allocation_with_lossy_coarsening_continuous_basis",
            "ADMM_with_score_aware_per_tensor_distortion_weights",
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
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    p.add_argument("--rms-targets", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    print("Path B step 5: Joint-ADMM Lagrangian-allocation empirical anchor")
    manifest = run_experiment(args.state_dict, args.alphas, args.rms_targets)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_omega_opt_joint_admm_allocation_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(f"  Baseline (all-lossless brotli): {manifest['baseline_lossless_bytes']:,} B\n")
    print(f"  {'rms_target':>10s} | {'greedy_b':>9s} {'greedy_rms':>10s} | "
          f"{'admm_b':>9s} {'admm_rms':>10s} {'λ':>10s} {'sparsified':>4s} | {'savings':>8s}")
    for r in manifest["comparison_at_rms_targets"]:
        print(f"  {r['rms_target']:>10.4f} | {r['greedy_bytes']:>9,} {r['greedy_rms']:>10.5f} | "
              f"{r['admm_bytes']:>9,} {r['admm_rms']:>10.5f} {r['admm_lambda']:>10.2e} {r['admm_n_sparsified']:>4d} | "
              f"{r['admm_savings_vs_greedy_bytes']:>+8,}")
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        target = next((r for r in manifest["comparison_at_rms_targets"] if abs(r["rms_target"] - 0.10) < 1e-9),
                      manifest["comparison_at_rms_targets"][0])
        evidence_row = {
            "technique": "lagrangian_per_tensor_allocation",
            "technique_historical_alias": "joint_admm_lagrangian_allocation",
            "renamed_to": "lagrangian_per_tensor_allocation",
            "renamed_per": "REVIEW-MATH 2026-05-08 Dykstra naming finding",
            "empirical_archive_bytes": target["admm_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "lagrangian_allocation_byte_anchor_at_rms_target_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "lagrangian_allocation_no_iterative_ADMM_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(rms_target=0.10; admm_bytes={target['admm_bytes']:,}; "
                f"vs_greedy={target['admm_savings_vs_greedy_bytes']:+,}; "
                f"best_savings={manifest['best_admm_savings_bytes']:+,}@rms={manifest['best_admm_savings_rms_target']})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": ["lagrangian_allocation"],
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
