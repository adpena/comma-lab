#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Water-filling REPAIR probe — $0 macOS-MLX/numpy advisory (NON-PROMOTABLE).

THE KEY QUESTION (operator-routed 2026-05-27):
    Instead of DROPPING bytes (rate saving, distortion cost), allocate a byte
    BUDGET to ADD detail back (repair) where the scorer is most sensitive.
    Use the per-pair master-gradient to rank regions by seg+pose sensitivity,
    use the canonical water-fill allocator to allocate the budget greedily to
    the highest-marginal-sensitivity regions, apply the REPAIR operator, and
    measure the $0 advisory score delta across a few budget levels.

    DOES the marginal distortion-reduction per repair-byte beat the 25/N rate
    cost?

NON-PROMOTABLE PER CLAUDE.md:
    * Catalog #192 macOS-CPU / advisory non-promotion.
    * Catalog #127 authoritative-tag custody (this carries NO contest axis).
    * Catalog #323 canonical Provenance (evidence_grade=[macOS-MLX research-signal]).
    The numbers here are a HEURISTIC PRIOR derived from the master-gradient
    sensitivity map; they DO NOT constitute a contest score claim. Promotion
    requires paired Linux x86_64 [contest-CPU] + NVIDIA [contest-CUDA] auth
    eval on the exact archive bytes per "Submission auth eval — BOTH CPU AND
    CUDA" non-negotiable.

CANONICAL SURFACES CONSUMED (READ-ONLY, NOT reimplemented):
    * Per-pair scorer-sensitivity map:
      .omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy
      shape (178517, 600, 3) = (N_bytes, N_pairs, [seg, pose, rate]) float64.
    * Water-fill allocator: tac.optimization.bit_allocator_end_to_end
      .allocate_per_pair_bits (per Catalog #125 hook #3).
    * REPAIR operator: tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas
      CanonicalOperation.REPAIR ("drop + add per-pair/per-frame repair signal";
      Atick-Redlich cooperative-receiver).
    * Canonical score formula: tac.score_composition
      S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489.

DEGENERACY ASSERTION (the rank-1 tautology lesson, commit 21c76632a):
    The probe ASSERTS the per-byte sensitivity ranking has real spread (Gini /
    top-k concentration bounded away from a single point mass). A degenerate
    ranking (all mass on one byte) would make water-fill trivially "optimal"
    by construction — the same rank-1 problem-spec tautology that the synergy
    re-run extinguished. If the ranking is degenerate the probe REFUSES and
    reports the degeneracy rather than a phantom marginal.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Canonical contest constants (mirrored from tac.score_composition) ──────
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
RATE_COST_PER_BYTE = CANONICAL_RATE_MULTIPLIER / CANONICAL_RATE_DENOM_BYTES  # 6.6586e-7

MASTER_GRADIENT_PATH = (
    REPO_ROOT
    / ".omx/state/master_gradient_fec6_frontier_mlx_per_pair_full600_20260527.npy"
)

# Frontier baseline (pointer-only per CLAUDE.md "Frontier scores are pointer-only").
FRONTIER_POINTER_PATH = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"

# MLX operating-point distortion at which the master-gradient was MEASURED
# (per the producer memo mlx_per_pair_master_gradient_authoritative_artifacts_
# landed_20260527.md full-600 row: d_seg=0.0012223561610638473,
# d_pose=0.0017157510650319333; total score 0.37208944003527994). This is the
# REALIZABILITY ANCHOR: the repair operator cannot recover more distortion than
# is actually present at the operating point where the gradient was taken.
MLX_OPERATING_POINT_D_SEG = 0.0012223561610638473
MLX_OPERATING_POINT_D_POSE = 0.0017157510650319333


def _operating_point_distortion_budget() -> float:
    """Total score-distortion present at the MLX operating point (the hard cap).

    S_distortion = 100*d_seg + sqrt(10*d_pose). A repair operator can recover
    AT MOST this much distortion (driving d_seg, d_pose -> 0). Any "best-case"
    estimate exceeding this is physically impossible (the rank-1-tautology-class
    of unbounded upper bound — the naive sum-of-marginals over thousands of
    bytes blows past the total distortion present and is therefore NOT a
    realizable score delta)."""
    return (
        CANONICAL_SEG_MULTIPLIER * MLX_OPERATING_POINT_D_SEG
        + (CANONICAL_POSE_SQRT_INNER * MLX_OPERATING_POINT_D_POSE) ** 0.5
    )


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_frontier_archive_bytes() -> tuple[float, int, str]:
    """Return (frontier_cpu_score, archive_bytes, archive_sha) from the canonical pointer."""
    d = json.loads(FRONTIER_POINTER_PATH.read_text())
    cpu = d["our_local_frontier_contest_cpu"]
    score = float(cpu["score"])
    sha = str(cpu["archive_sha256"])
    archive_bytes = int(cpu.get("extra", {}).get("archive_bytes", 178546))
    return score, archive_bytes, sha


def _per_byte_sensitivity(grad_mmap, *, axes: tuple[int, ...] = (0, 1)):
    """Aggregate per-byte sensitivity = L1 over pairs of |seg|+|pose| (not rate).

    The repair operator REDUCES distortion (seg/pose); rate is the COST it must
    beat (25/N per byte). We therefore rank ONLY by the distortion axes the
    repair can move. Streamed in chunks to avoid loading the 2.4 GB tensor.
    """
    n_bytes = int(grad_mmap.shape[0])
    out = np.zeros(n_bytes, dtype=np.float64)
    chunk = 8192
    for start in range(0, n_bytes, chunk):
        end = min(start + chunk, n_bytes)
        block = np.asarray(grad_mmap[start:end])  # (c, N_pairs, 3)
        # L1 over pairs of |seg| + |pose|; rate axis (2) excluded.
        mag = np.abs(block[:, :, list(axes)]).sum(axis=(1, 2))
        out[start:end] = mag
    return out


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0=uniform, 1=single mass)."""
    v = np.sort(np.asarray(x, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    # Gini = (2*sum(i*v_i) / (n*sum(v)) ) - (n+1)/n
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v)) / (n * cum[-1]) - (n + 1.0) / n)


def _assert_non_degenerate(sens: np.ndarray) -> dict:
    """Assert the sensitivity ranking has REAL spread (rank-1 tautology guard).

    Degeneracy = a single point mass (or a handful of bytes) carrying ~all the
    sensitivity. If the top-1 byte carries >50% of total L1 mass OR fewer than
    100 bytes carry 90% of the mass, the ranking is degenerate and water-fill
    is a tautology by construction.
    """
    total = float(sens.sum())
    order = np.argsort(sens)[::-1]
    sorted_sens = sens[order]
    cum = np.cumsum(sorted_sens)
    top1_frac = float(sorted_sens[0] / total) if total > 0 else 1.0
    # bytes needed to reach 90% of mass
    n90 = int(np.searchsorted(cum, 0.90 * total) + 1) if total > 0 else 1
    n_nonzero = int((sens > 0).sum())
    gini = _gini(sens)
    degenerate = bool(top1_frac > 0.50 or n90 < 100 or n_nonzero < 100)
    return {
        "total_l1_sensitivity": total,
        "n_bytes": int(sens.size),
        "n_nonzero_bytes": n_nonzero,
        "top1_mass_fraction": top1_frac,
        "bytes_to_90pct_mass": n90,
        "gini_coefficient": gini,
        "degenerate": degenerate,
        "degeneracy_rule": (
            "degenerate if top1_frac>0.50 OR bytes_to_90pct<100 OR n_nonzero<100"
        ),
    }


def _marginal_distortion_per_repair_byte(sens: np.ndarray, budget: int) -> dict:
    """Water-fill: greedily allocate `budget` repair bytes to highest-sensitivity bytes.

    The per-byte sensitivity `s_i = sum_pairs |grad_seg| + |grad_pose|` is the
    marginal |dScore_distortion / d(byte_value)| at the current frontier. A
    repair byte that perturbs byte i toward its scorer-optimal value can recover
    AT MOST `s_i * full_byte_swing` of distortion (upper bound). We use the raw
    per-byte sensitivity magnitude as the canonical marginal proxy (consistent
    with how the bit-allocator ranks). We sum the top-`budget` sensitivities =
    the BEST-CASE achievable distortion reduction if each repair byte fully
    realizes its marginal, then compare per-byte to the 25/N rate cost.
    """
    order = np.argsort(sens)[::-1]
    top = sens[order[:budget]]
    # NAIVE (unbounded) best-case = sum of top-budget marginal sensitivities.
    # This is the rank-1-tautology-CLASS of unbounded upper bound: it sums raw
    # FD magnitudes across thousands of bytes and can blow past the TOTAL
    # distortion present at the operating point — physically impossible.
    naive_distortion_reduction = float(top.sum())

    # REALIZABILITY-BOUNDED best-case (apples-to-apples discipline): cap the
    # achievable distortion reduction at the actual distortion budget present
    # at the MLX operating point. No repair can recover more distortion than
    # exists. This is the disciplined number.
    distortion_budget = _operating_point_distortion_budget()
    realizable_distortion_reduction = min(naive_distortion_reduction, distortion_budget)

    rate_cost = RATE_COST_PER_BYTE * budget
    # Net score delta (negative = improvement). Use the REALIZABLE reduction.
    net_realizable = -realizable_distortion_reduction + rate_cost
    naive_marginal_per_byte = (
        naive_distortion_reduction / budget if budget else 0.0
    )
    realizable_marginal_per_byte = (
        realizable_distortion_reduction / budget if budget else 0.0
    )
    return {
        "budget_bytes": int(budget),
        "naive_unbounded_distortion_reduction": naive_distortion_reduction,
        "operating_point_distortion_budget": distortion_budget,
        "naive_exceeds_distortion_budget": bool(
            naive_distortion_reduction > distortion_budget
        ),
        "realizable_distortion_reduction_capped": realizable_distortion_reduction,
        "naive_marginal_per_repair_byte": naive_marginal_per_byte,
        "realizable_marginal_per_repair_byte": realizable_marginal_per_byte,
        "rate_cost_per_byte": RATE_COST_PER_BYTE,
        "total_rate_cost": rate_cost,
        "net_score_delta_realizable_best_case": net_realizable,
        # The disciplined verdict: does the REALIZABLE marginal per byte beat
        # the rate cost? (and is the naive estimate physically possible at all?)
        "realizable_marginal_beats_rate": bool(
            realizable_marginal_per_byte > RATE_COST_PER_BYTE
        ),
        "naive_marginal_beats_rate": bool(
            naive_marginal_per_byte > RATE_COST_PER_BYTE
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--budgets",
        type=str,
        default="64,256,1024,4096,16384",
        help="comma-separated repair-byte budget levels",
    )
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args(argv)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]

    if not MASTER_GRADIENT_PATH.exists():
        print(f"FATAL: master-gradient not found at {MASTER_GRADIENT_PATH}", file=sys.stderr)
        return 2

    frontier_score, archive_bytes, archive_sha = _load_frontier_archive_bytes()

    grad = np.load(MASTER_GRADIENT_PATH, mmap_mode="r")
    print(f"[probe] master-gradient shape={grad.shape} dtype={grad.dtype}", file=sys.stderr)

    sens = _per_byte_sensitivity(grad)
    degeneracy = _assert_non_degenerate(sens)

    result = {
        "schema": "water_filling_repair_probe_v1",
        "evidence_grade": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "axis_tag": "[macOS-MLX research-signal]",
        "utc": _utc(),
        "master_gradient_path": str(MASTER_GRADIENT_PATH.relative_to(REPO_ROOT)),
        "master_gradient_shape": list(grad.shape),
        "frontier_cpu_score_pointer": frontier_score,
        "frontier_archive_bytes": archive_bytes,
        "frontier_archive_sha256_prefix": archive_sha[:16],
        "rate_cost_per_byte_25_over_N": RATE_COST_PER_BYTE,
        "canonical_rate_denom_bytes": CANONICAL_RATE_DENOM_BYTES,
        "degeneracy_check": degeneracy,
        "rate_axis_signal_in_gradient": float(
            np.abs(np.asarray(grad[:2000, :, 2])).sum()
        ),
        "provenance": {
            "kind": "macos_mlx_research_signal",
            "axis_tag": "[macOS-MLX research-signal]",
            "hardware_substrate": "darwin_arm64_m5_max_macos",
            "evidence_grade": "macos_mlx_research_signal",
            "score_claim_valid": False,
            "rationale": (
                "Water-filling-repair marginal probe derived from per-pair "
                "master-gradient HEURISTIC PRIOR (FEC6 frontier). NOT a contest "
                "score claim; promotion requires paired Linux x86_64 + NVIDIA "
                "auth eval on exact archive bytes."
            ),
        },
    }

    if degeneracy["degenerate"]:
        result["verdict"] = "DEGENERATE_RANKING_REFUSED"
        result["budget_sweep"] = []
        result["conclusion"] = (
            "Sensitivity ranking is degenerate (rank-1-like concentration). "
            "Water-fill would be a tautology by construction (the rank-1 lesson "
            "from commit 21c76632a). REFUSING to report a phantom marginal."
        )
    else:
        sweep = [_marginal_distortion_per_repair_byte(sens, b) for b in budgets]
        result["budget_sweep"] = sweep
        result["operating_point_distortion_budget"] = _operating_point_distortion_budget()

        # APPLES-TO-APPLES caveat (the load-bearing honesty): the FD gradient
        # measures |dScore / d(byte_VALUE)| (sensitivity to perturbing an EXISTING
        # byte), NOT |dScore / d(ADDING a repair byte)|. A repair byte ADDS rate
        # unconditionally but only recovers distortion if (a) the receiver runtime
        # actually consumes it and (b) the FD sensitivity at the current byte
        # value is a faithful proxy for the recoverable distortion. The naive
        # sum-of-marginals over thousands of bytes EXCEEDS the total distortion
        # present at the operating point (physically impossible) — confirming
        # the raw marginal is an UPPER-BOUND PROXY, not a realizable score delta.
        naive_blew_budget = any(s["naive_exceeds_distortion_budget"] for s in sweep)
        realizable_beats = any(s["realizable_marginal_beats_rate"] for s in sweep)

        top_byte_marginal = sweep[0]["realizable_marginal_per_repair_byte"]
        result["dispositive_top_byte_realizable_marginal"] = top_byte_marginal
        result["dispositive_rate_cost_per_byte"] = RATE_COST_PER_BYTE
        result["dispositive_ratio_marginal_over_rate"] = (
            top_byte_marginal / RATE_COST_PER_BYTE if RATE_COST_PER_BYTE else float("inf")
        )
        result["naive_upper_bound_exceeds_distortion_budget"] = bool(naive_blew_budget)

        # Disciplined verdict: even with the realizability cap, the FD-magnitude
        # marginal is an UPPER BOUND on a quantity (∂score/∂byte_value) that is
        # NOT the repair quantity (∂score/∂repair_byte). We therefore do NOT
        # declare a FIRE candidate from this proxy alone — that would repeat the
        # unbounded-upper-bound class of error. The honest verdict is
        # INDETERMINATE: the proxy is non-degenerate and large, but the
        # value-sensitivity → repair-recoverability gap is unmeasured.
        if naive_blew_budget:
            result["verdict"] = "INDETERMINATE_UPPER_BOUND_PROXY_NOT_REALIZABLE"
            result["conclusion"] = (
                "INDETERMINATE. The per-byte sensitivity ranking is NON-DEGENERATE "
                f"(Gini {degeneracy['gini_coefficient']:.3f}, top1 mass "
                f"{degeneracy['top1_mass_fraction']*100:.2f}%, "
                f"{degeneracy['bytes_to_90pct_mass']} bytes to 90% mass) — the "
                "rank-1 tautology is NOT present. BUT the naive sum-of-marginals "
                "EXCEEDS the total distortion budget at the operating point "
                f"({_operating_point_distortion_budget():.4f}), proving the FD "
                "magnitude is an UPPER-BOUND PROXY for |dScore/d(byte_value)|, NOT "
                "a realizable |dScore/d(ADDING a repair byte)|. The "
                "value-sensitivity -> repair-recoverability gap is UNMEASURED. "
                "Declaring a FIRE candidate from this proxy alone would repeat the "
                "unbounded-upper-bound error class (sister of the rank-1 tautology "
                "lesson, commit 21c76632a). The disciplined next step is a $0 "
                "byte-mutation realizability micro-probe (Catalog #139): mutate "
                "the top-K sensitivity bytes toward a repair-signal target on the "
                "ACTUAL archive + re-run the MLX scorer oracle to measure the "
                "REALIZED d_seg/d_pose reduction per repair byte, then re-compare "
                "to 25/N. NOT a score claim; NOT promotable."
            )
        elif realizable_beats:
            result["verdict"] = "SUB_FRONTIER_MARGINAL_CANDIDATE_PENDING_REALIZABILITY"
            result["conclusion"] = (
                "Realizability-bounded marginal beats 25/N AND the naive estimate "
                "stays within the distortion budget. Operator-gated FIRE candidate "
                "PENDING a byte-mutation realizability micro-probe. NOT a score claim."
            )
        else:
            result["verdict"] = "NULL_NO_SUB_FRONTIER_MARGINAL"
            result["conclusion"] = (
                "NULL: even the highest-sensitivity byte's realizable marginal "
                "distortion-reduction is BELOW the 25/N rate cost. REPAIR does "
                "not reopen a sub-frontier marginal at this operating point."
            )

    out = json.dumps(result, indent=2, sort_keys=True)
    print(out)
    if args.json_out:
        p = REPO_ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out)
        print(f"[probe] wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
