#!/usr/bin/env python3
"""PR101 Joint-ADMM cross-component allocation — Path B step 5 empirical anchor.

The Ω-OPT design predicts -5bp from Joint-ADMM coordination across components
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
import struct
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_omega_opt_joint_admm_allocation_empirical.py"
SCHEMA_VERSION = "pr101_omega_opt_joint_admm_allocation_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful Path-B-step-5 Joint-ADMM Lagrangian-allocation test]"


def encode_brotli_only(symbols: np.ndarray) -> tuple[int, float]:
    return len(brotli.compress(symbols.tobytes(), quality=11, lgwin=16, lgblock=19)), 0.0


def encode_sparsity_alpha(symbols: np.ndarray, alpha: float) -> tuple[int, float]:
    n = symbols.size
    if alpha <= 0:
        return encode_brotli_only(symbols)
    n_keep = max(1, int(round((1.0 - alpha) * n)))
    if n_keep >= n:
        return encode_brotli_only(symbols)
    abs_vals = np.abs(symbols.flatten().astype(np.int32))
    top_idx = np.argpartition(abs_vals, n - n_keep)[n - n_keep:]
    top_idx_sorted = np.sort(top_idx)
    nz_values = symbols.flatten()[top_idx_sorted].astype(np.int8)
    recon = np.zeros_like(symbols.flatten(), dtype=np.int8)
    recon[top_idx_sorted] = nz_values
    diff = symbols.flatten().astype(np.float64) - recon.astype(np.float64)
    orig_l2 = float(np.linalg.norm(symbols.flatten().astype(np.float64))) + 1e-12
    rel_err = float(np.linalg.norm(diff)) / orig_l2
    deltas = np.diff(np.concatenate([np.array([0], dtype=np.uint32), top_idx_sorted.astype(np.uint32)])).astype(np.uint32)
    payload = struct.pack("<II", n, nz_values.size) + deltas.tobytes() + nz_values.tobytes()
    return len(brotli.compress(payload, quality=11, lgwin=16, lgblock=19)), rel_err


def measure_curves(quantized: list[tuple[str, np.ndarray]], alphas: list[float]) -> list[list[dict]]:
    """For each tensor, return list of {alpha, bytes, rel_err}."""
    out: list[list[dict]] = []
    for _, syms in quantized:
        rows: list[dict] = []
        b0, e0 = encode_brotli_only(syms)
        rows.append({"alpha": 0.0, "bytes": b0, "rel_err": e0})
        for alpha in alphas:
            b, e = encode_sparsity_alpha(syms, alpha)
            rows.append({"alpha": alpha, "bytes": b, "rel_err": e})
        out.append(rows)
    return out


def greedy_uniform_budget(curves: list[list[dict]], budget: float) -> tuple[int, float]:
    """Each tensor picks the smallest-bytes codec whose rel_err <= budget."""
    total_bytes = 0
    rel_errs: list[float] = []
    for tensor_rows in curves:
        valid = [r for r in tensor_rows if r["rel_err"] <= budget]
        best = min(valid, key=lambda r: r["bytes"])
        total_bytes += best["bytes"]
        rel_errs.append(best["rel_err"])
    rms_rel_err = float(np.sqrt(np.mean([e ** 2 for e in rel_errs])))
    return total_bytes, rms_rel_err


def lagrangian_allocate(curves: list[list[dict]], lam: float) -> tuple[int, float, list[float]]:
    """For Lagrangian λ, each tensor picks codec minimizing (bytes + λ * rel_err^2)."""
    total_bytes = 0
    rel_errs: list[float] = []
    for tensor_rows in curves:
        cost = [r["bytes"] + lam * r["rel_err"] ** 2 for r in tensor_rows]
        idx = int(np.argmin(cost))
        chosen = tensor_rows[idx]
        total_bytes += chosen["bytes"]
        rel_errs.append(chosen["rel_err"])
    rms = float(np.sqrt(np.mean([e ** 2 for e in rel_errs])))
    return total_bytes, rms, rel_errs


def bisect_lambda_for_rms_target(curves: list[list[dict]], rms_target: float) -> dict:
    """Find λ such that the achieved RMS rel_err is ≤ rms_target."""
    # Large λ → all tensors pick lowest rel_err (= brotli_only, rel_err=0)
    # Small λ → all tensors pick highest sparsity (max rel_err)
    lo, hi = 0.0, 1e12
    for _ in range(60):
        mid = (lo + hi) / 2 if hi < 1e12 else lo * 10 + 1
        bytes_total, rms, _ = lagrangian_allocate(curves, mid)
        if rms <= rms_target:
            hi = mid  # achievable; tighten by lowering λ → more rel_err allowed
        else:
            lo = mid  # too much rel_err; raise λ
        if abs(hi - lo) < 1e-9 or hi == lo:
            break
    final_lam = hi
    bytes_total, rms, rel_errs = lagrangian_allocate(curves, final_lam)
    return {
        "lambda": final_lam,
        "total_bytes": bytes_total,
        "achieved_rms_rel_err": rms,
        "per_tensor_rel_errs": rel_errs,
        "n_sparsified": sum(1 for e in rel_errs if e > 0),
    }


def run_experiment(state_dict_path: Path, alphas: list[float], rms_targets: list[float]) -> dict:
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
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

    print(f"Path B step 5: Joint-ADMM Lagrangian-allocation empirical anchor")
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
            "technique": "joint_admm_lagrangian_allocation",
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
