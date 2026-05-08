#!/usr/bin/env python3
"""PR101 Path B step 7 — Fridrich UNIWARD-weighted Lagrangian per-tensor allocation.

Background (CLAUDE.md "Fridrich inverse steganalysis"):
    1. UNIWARD: errors in textured regions are undetectable. Weight loss by
       inverse local variance.
    2. Detector-informed embedding = our TTO approach. Fridrich-approved
       (Yousfi 2022).
    3. Square root law: spread small errors (L∞ penalty), don't concentrate
       large ones.
    4. CNN blind spots: EfficientNet misses DCT statistics, has texture-region
       blind spots.

Path B step 6 (``tools/pr101_omega_opt_admm_x_lossy_coarsening_empirical.py``)
selects per-tensor coarsening step K via a Lagrangian
``cost(t,K) = byte_proxy(t,K) + λ · rel_err(t,K)²``. The cost weighting is
UNIFORM across tensors — every tensor's rel_err is treated identically.

UNIWARD principle inverts this: the *detector's local-variance map* tells us
which embedding regions are blind-spots (high variance = textured = blind).
For score-targeted detector evasion, distortion in HIGH-variance tensors
hurts the detector less than distortion in LOW-variance tensors. We therefore
weight ``rel_err²`` by ``1 / local_variance`` so that the optimizer pushes
*more* error into high-variance (textured) tensors and *less* into
low-variance (smooth) tensors:

    cost_uniward(t, K) = byte_proxy(t, K) + λ · w(t) · rel_err(t, K)²
    w(t) = 1 / (variance_proxy(t) + ε)

Where ``variance_proxy(t)`` is the variance of the int8 weight histogram for
tensor ``t`` — a CPU-cheap analogue of UNIWARD's wavelet-domain residual
variance. ε is a small floor (1e-6) to avoid division blow-up on
near-constant tensors.

This is a CPU byte-proxy anchor only. No CUDA score is implied. Score-aware
weighting (the "true" UNIWARD-Fridrich-Yousfi recipe with detector-in-loop)
remains in the reactivation_criteria_remaining list.

Falsification scope
-------------------
``uniward_weighted_lagrangian_per_tensor_only``: only the variance-weighted
Lagrangian over the lossy_coarsening continuous-K basis is tested. The
combination with iterative primal-dual ADMM consensus, the wavelet-domain
UNIWARD residual proxy, and exact CUDA score validation are NOT tested.
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

# Reuse subagent D's machinery and Path B step 6's helpers
from pr101_lossy_coarsening_analytical import (  # noqa: E402
    TensorBlob,
    encode_with_per_tensor_K,
    find_best_K_for_tensor,
    ARCHIVE_OVERHEAD_BYTES,
)

TOOL_NAME = "tools/pr101_omega_opt_uniward_weighted_allocation.py"
SCHEMA_VERSION = "pr101_omega_opt_uniward_weighted_allocation.v1"
EVIDENCE_GRADE = (
    "[CPU-prep faithful Fridrich-UNIWARD-weighted Path-B-step-7 test]"
)

K_RANGE = list(range(1, 65))
EPS_VARIANCE = 1e-6


# ---------------------------------------------------------------------------
# UNIWARD variance proxy
# ---------------------------------------------------------------------------


def compute_local_variance_proxy(tensors: list[TensorBlob]) -> list[float]:
    """Return per-tensor variance of the int8 symbol distribution.

    This is the CPU analogue of UNIWARD's local-variance residual: a tensor
    whose int8 symbols span a wide dynamic range is "textured" and absorbs
    error well; a tensor whose symbols cluster near zero is "smooth" and
    distortion there is detector-visible.
    """
    variances: list[float] = []
    for tb in tensors:
        symbols = tb.raw.astype(np.float64)
        var = float(np.var(symbols)) if symbols.size > 0 else 0.0
        variances.append(var)
    return variances


def compute_uniward_weights(variances: list[float]) -> list[float]:
    """``w(t) = 1 / (var(t) + ε)`` — inverse-variance per CLAUDE.md UNIWARD rule.

    Higher variance → lower weight → more error budget allocated there.
    Lower variance → higher weight → cost forces less error there.
    """
    return [1.0 / (v + EPS_VARIANCE) for v in variances]


# ---------------------------------------------------------------------------
# Per-tensor K curve precomputation
# ---------------------------------------------------------------------------


def precompute_per_tensor_K_curves(tensors: list[TensorBlob]) -> list[list[dict]]:
    """For each tensor, compute (K, achieved_rel_err, per_tensor_byte_proxy)
    over ``K_RANGE``. ``byte_proxy`` is a per-tensor brotli over the rounded
    int8 — an over-estimate of joint contribution but monotone in K.
    """
    curves: list[list[dict]] = []
    for tb in tensors:
        rows = []
        abs_sum = float(np.abs(tb.raw).astype(np.float64).sum()) + 1e-12
        for K in K_RANGE:
            rounded = np.round(tb.raw / K) * K
            err = float(np.abs(rounded - tb.raw).astype(np.float64).sum())
            re = err / abs_sum
            rounded_i8 = rounded.clip(-127, 127).astype(np.int8)
            byte_proxy = len(
                brotli.compress(rounded_i8.tobytes(), quality=11, lgwin=16, lgblock=19)
            )
            rows.append({"K": K, "rel_err": re, "byte_proxy": byte_proxy})
        curves.append(rows)
    return curves


# ---------------------------------------------------------------------------
# Lagrangian selection (uniform vs UNIWARD-weighted)
# ---------------------------------------------------------------------------


def lagrangian_select_Ks_uniform(
    curves: list[list[dict]], lam: float
) -> tuple[list[int], list[float]]:
    """Path B step 6 baseline: ``cost(K) = byte_proxy + λ · rel_err²``."""
    Ks: list[int] = []
    rel_errs: list[float] = []
    for tensor_curve in curves:
        cost = [r["byte_proxy"] + lam * r["rel_err"] ** 2 for r in tensor_curve]
        idx = int(np.argmin(cost))
        Ks.append(tensor_curve[idx]["K"])
        rel_errs.append(tensor_curve[idx]["rel_err"])
    return Ks, rel_errs


def lagrangian_select_Ks_uniward(
    curves: list[list[dict]], weights: list[float], lam: float
) -> tuple[list[int], list[float]]:
    """UNIWARD: ``cost(K) = byte_proxy + λ · w(t) · rel_err²``.

    ``w(t)`` is the inverse-variance weight; tensors with high variance carry
    a low weight and therefore are encouraged to absorb more rel_err.
    """
    Ks: list[int] = []
    rel_errs: list[float] = []
    for tensor_curve, w in zip(curves, weights, strict=True):
        cost = [r["byte_proxy"] + lam * w * r["rel_err"] ** 2 for r in tensor_curve]
        idx = int(np.argmin(cost))
        Ks.append(tensor_curve[idx]["K"])
        rel_errs.append(tensor_curve[idx]["rel_err"])
    return Ks, rel_errs


# ---------------------------------------------------------------------------
# λ bisection for global RMS rel_err target
# ---------------------------------------------------------------------------


def bisect_for_global_rms_uniward(
    tensors: list[TensorBlob],
    curves: list[list[dict]],
    weights: list[float],
    rms_target: float,
    max_iter: int = 80,
) -> dict:
    """Bisect λ so the joint-encoded RMS rel_err <= ``rms_target`` under
    UNIWARD-weighted Lagrangian selection."""
    lo, hi = 0.0, 1e15
    last_result = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2 if hi < 1e15 else lo * 10 + 1
        Ks, _ = lagrangian_select_Ks_uniward(curves, weights, mid)
        result = encode_with_per_tensor_K(tensors, Ks)
        rms = result["rel_err"]
        last_result = {"lambda": mid, **result, "rms_rel_err": rms}
        if rms <= rms_target:
            hi = mid
        else:
            lo = mid
        if hi == lo or abs(hi - lo) < 1e-12:
            break
    assert last_result is not None
    return last_result


def bisect_for_global_rms_uniform(
    tensors: list[TensorBlob],
    curves: list[list[dict]],
    rms_target: float,
    max_iter: int = 80,
) -> dict:
    """Path B step 6 baseline: bisect λ for ``rel_err²`` Lagrangian."""
    lo, hi = 0.0, 1e15
    last_result = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2 if hi < 1e15 else lo * 10 + 1
        Ks, _ = lagrangian_select_Ks_uniform(curves, mid)
        result = encode_with_per_tensor_K(tensors, Ks)
        rms = result["rel_err"]
        last_result = {"lambda": mid, **result, "rms_rel_err": rms}
        if rms <= rms_target:
            hi = mid
        else:
            lo = mid
        if hi == lo or abs(hi - lo) < 1e-12:
            break
    assert last_result is not None
    return last_result


# ---------------------------------------------------------------------------
# Greedy uniform-budget baseline (Subagent-D-style)
# ---------------------------------------------------------------------------


def greedy_uniform_per_tensor_budget(
    tensors: list[TensorBlob], budget: float
) -> dict:
    """Per-tensor budget allocation — same as Path B step 6 baseline."""
    Ks = []
    for tb in tensors:
        K, _ = find_best_K_for_tensor(tb.raw, budget)
        Ks.append(K)
    return encode_with_per_tensor_K(tensors, Ks)


# ---------------------------------------------------------------------------
# Top-level experiment runner
# ---------------------------------------------------------------------------


def run_experiment(state_dict_path: Path, rms_targets: list[float]) -> dict:
    import torch

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    tensors: list[TensorBlob] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        tensors.append(
            TensorBlob(name=name, raw=qt.q_i8.astype(np.int32).flatten())
        )
    n_tensors = len(tensors)

    print(f"  precomputing per-tensor K curves ({n_tensors} × {len(K_RANGE)})...")
    curves = precompute_per_tensor_K_curves(tensors)

    variances = compute_local_variance_proxy(tensors)
    weights = compute_uniward_weights(variances)
    print(
        f"  variance proxy: min={min(variances):.3e}, max={max(variances):.3e}, "
        f"weight ratio max/min={max(weights)/min(weights):.2e}"
    )

    baseline = encode_with_per_tensor_K(tensors, [1] * n_tensors)

    comparison: list[dict] = []
    for rms_t in rms_targets:
        greedy = greedy_uniform_per_tensor_budget(tensors, rms_t)
        admm_uniform = bisect_for_global_rms_uniform(tensors, curves, rms_t)
        admm_uniward = bisect_for_global_rms_uniward(
            tensors, curves, weights, rms_t
        )

        savings_uniward_vs_greedy = (
            greedy["archive_bytes"] - admm_uniward["archive_bytes"]
        )
        savings_uniward_vs_uniform = (
            admm_uniform["archive_bytes"] - admm_uniward["archive_bytes"]
        )
        comparison.append(
            {
                "rms_target": rms_t,
                "greedy_K_per_tensor_budget": {
                    "archive_bytes": greedy["archive_bytes"],
                    "rel_err": greedy["rel_err"],
                },
                "admm_uniform_lagrangian": {
                    "archive_bytes": admm_uniform["archive_bytes"],
                    "rel_err": admm_uniform["rel_err"],
                    "lambda": admm_uniform["lambda"],
                    "Ks": admm_uniform["Ks"],
                },
                "admm_uniward_weighted_lagrangian": {
                    "archive_bytes": admm_uniward["archive_bytes"],
                    "rel_err": admm_uniward["rel_err"],
                    "lambda": admm_uniward["lambda"],
                    "Ks": admm_uniward["Ks"],
                },
                "uniward_savings_vs_greedy_bytes": savings_uniward_vs_greedy,
                "uniward_savings_vs_uniform_admm_bytes": savings_uniward_vs_uniform,
            }
        )

    best_uniward_vs_uniform = max(
        comparison, key=lambda r: r["uniward_savings_vs_uniform_admm_bytes"]
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cuda_eval_worth_testing": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "family_falsified": False,
        "falsification_scope": "uniward_weighted_lagrangian_per_tensor_only",
        "input_state_dict": str(state_dict_path),
        "n_tensors": n_tensors,
        "K_range": [K_RANGE[0], K_RANGE[-1]],
        "rms_targets_swept": rms_targets,
        "baseline_lossless_bytes": baseline["archive_bytes"],
        "comparison_at_rms_targets": comparison,
        "best_uniward_savings_vs_uniform_bytes": best_uniward_vs_uniform[
            "uniward_savings_vs_uniform_admm_bytes"
        ],
        "best_uniward_savings_rms_target": best_uniward_vs_uniform["rms_target"],
        "headline": (
            "UNIWARD-weighted Lagrangian (1/var rel_err² weights) vs Path B step 6 "
            "uniform Lagrangian. Best UNIWARD savings vs uniform: "
            f"{best_uniward_vs_uniform['uniward_savings_vs_uniform_admm_bytes']:+,} B at "
            f"rms_target={best_uniward_vs_uniform['rms_target']}"
        ),
        "dispatch_blockers": [
            "byte_rel_err_proxy_only_no_score_test",
            "variance_proxy_substitutes_for_uniward_wavelet_residual",
            "no_iterative_primal_dual_ADMM_consensus",
            "missing_exact_cuda_auth_eval",
        ],
        "reactivation_criteria_remaining": [
            "wavelet_domain_uniward_residual_variance_proxy",
            "score_aware_per_tensor_distortion_weights_detector_in_loop",
            "iterative_primal_dual_ADMM_with_consensus",
            "uniward_weighted_lagrangian_with_CUDA_score_validation",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fridrich UNIWARD-weighted Lagrangian per-tensor allocation"
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument(
        "--rms-targets",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.0386, 0.05, 0.10],
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    print(f"Path B step 7: Fridrich UNIWARD-weighted Lagrangian per-tensor allocation")
    manifest = run_experiment(args.state_dict, args.rms_targets)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_uniward_weighted_lagrangian_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(
        f"  Baseline (uniform K=1, lossless): {manifest['baseline_lossless_bytes']:,} B\n"
    )
    print(
        f"  {'rms_target':>10s} | {'greedy_b':>9s} | {'uniform_b':>10s} {'uniform_λ':>10s} | "
        f"{'uniward_b':>10s} {'uniward_λ':>10s} | {'Δ_vs_uniform':>12s}"
    )
    for r in manifest["comparison_at_rms_targets"]:
        g = r["greedy_K_per_tensor_budget"]
        u = r["admm_uniform_lagrangian"]
        w = r["admm_uniward_weighted_lagrangian"]
        print(
            f"  {r['rms_target']:>10.4f} | {g['archive_bytes']:>9,} | "
            f"{u['archive_bytes']:>10,} {u['lambda']:>10.2e} | "
            f"{w['archive_bytes']:>10,} {w['lambda']:>10.2e} | "
            f"{r['uniward_savings_vs_uniform_admm_bytes']:>+12,}"
        )
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        target = next(
            (
                r
                for r in manifest["comparison_at_rms_targets"]
                if abs(r["rms_target"] - 0.0386) < 1e-9
            ),
            manifest["comparison_at_rms_targets"][0],
        )
        evidence_row = {
            "technique": "uniward_weighted_lagrangian_per_tensor",
            "empirical_archive_bytes": target[
                "admm_uniward_weighted_lagrangian"
            ]["archive_bytes"],
            "evidence_grade": EVIDENCE_GRADE,
            "evidence_marker": EVIDENCE_GRADE,
            "evidence_semantics": "uniward_weighted_lagrangian_byte_anchor_no_score",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "proxy_row": True,
            "cuda_eval_worth_testing": False,
            "family_falsified": False,
            "falsification_scope": "uniward_weighted_lagrangian_per_tensor_only",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "dispatch_blockers": manifest["dispatch_blockers"],
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(rms_target=3.86%; "
                f"uniward_bytes={target['admm_uniward_weighted_lagrangian']['archive_bytes']:,}; "
                f"vs_uniform_admm={target['uniward_savings_vs_uniform_admm_bytes']:+,}; "
                f"best_savings={manifest['best_uniward_savings_vs_uniform_bytes']:+,}@"
                f"rms={manifest['best_uniward_savings_rms_target']})"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": [
                "uniward_inverse_variance_lagrangian_weighting"
            ],
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
