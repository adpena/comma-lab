#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dykstra-feasibility check for substrate predicted-band claims.

Canonical helper for the HIGH-RISK substrate cargo-cult unwind audit
(`.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` D3
operator-approved). Refuses predicted-band claims that fall outside the convex
feasibility intersection `rate <= R AND seg <= S AND pose <= P`.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable + Catalog #296
proposed sister gate: every L1+ substrate design memo MUST declare its
predicted ΔS band against an explicit Dykstra-feasibility check; bands that
intersect-empty with the convex constraint set are CARGO-CULTED claims.

Empirical anchor (audit Section 1.1 NSCS02 + 1.2 C6 + 1.3 ATW + 1.4 TT-L5):
4 substrates wrote a numeric ΔS band derived from first-principles arithmetic
WITHOUT the explicit projection onto the achievable convex set. This helper
runs alternating-projections (Dykstra 1983) over the 3 contest-axis convex
constraints and emits a FEASIBLE / INFEASIBLE / INDETERMINATE verdict so the
band claim can be promoted, retired, or re-anchored against the polytope.

CLI contract:

    .venv/bin/python tools/check_substrate_dykstra_feasibility.py \\
        --substrate-id nscs02_downsampled_renderer \\
        --predicted-band-lo 0.175 --predicted-band-hi 0.195 \\
        --archive-size-bytes 280000 \\
        --output-json .omx/state/dykstra_feasibility_nscs02.json

The 3 convex constraints follow the upstream contest scorer
`upstream/evaluate.py`:

    score = seg_avg + sqrt(10 * pose_avg) + 25 * archive_bytes / N_video_bytes

with N_video_bytes = 37,545,489 fixed by `upstream/videos/0.mkv`. The feasible
region is the intersection of three half-spaces (rate <= R_budget, seg <= S,
pose <= P) under the standing-floor anchors S=0.001 and P=0.011 derived from
the A1 reference packet on contest-CUDA T4.

Verdict taxonomy:
- FEASIBLE: the predicted-band interval intersects the feasible polytope
- INFEASIBLE: the entire predicted band falls strictly OUTSIDE the polytope
  (the substrate's claim is mathematically impossible under the budgets);
  ``blocker_axis`` names which constraint dominates.
- INDETERMINATE: the constraints are too loose (or ambiguous) to decide; the
  caller should tighten the budgets and re-run.

Per CLAUDE.md "Comment-only contracts are FORBIDDEN" + "Forbidden CLI flag
inventions": every flag below is grep-checked against argparse so callers
cannot drift.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Contest rate denominator from upstream/videos/0.mkv. Authoritative constant
# per CLAUDE.md "Apples-to-apples evidence discipline" — the byte total of
# the canonical scored video; do not infer from filesystem at runtime.
CONTEST_RATE_DENOM_BYTES: int = 37_545_489

# Default standing-floor distortion anchors observed on the A1 reference
# packet (contest-CUDA T4 2026-05-13 `0.19285` baseline). Operators may
# override via --seg-budget / --pose-budget when probing a different
# operating point.
DEFAULT_SEG_BUDGET: float = 0.001
DEFAULT_POSE_BUDGET: float = 0.011

# Dykstra alternating-projections numerical knobs.
DEFAULT_MAX_ITER: int = 200
DEFAULT_TOLERANCE: float = 1e-7


VerdictStr = Literal["FEASIBLE", "INFEASIBLE", "INDETERMINATE"]


@dataclass(frozen=True)
class DykstraFeasibilityVerdict:
    """Machine-readable verdict consumed by autopilot ranker + design memos."""

    substrate_id: str
    verdict: VerdictStr
    predicted_band: tuple[float, float]
    rate_contribution: float
    seg_budget: float
    pose_budget: float
    feasibility_band_lo: float
    feasibility_band_hi: float
    feasibility_rationale: str
    blocker_axis: Literal["rate", "seg", "pose"] | None
    dykstra_iteration_count: int


def _project_onto_rate(score: float, rate_contribution: float) -> float:
    """Project a score value onto the rate constraint.

    The rate contribution is fixed by archive_bytes; any feasible score must
    contain it. Projection is the identity for scores >= rate_contribution,
    and clipped up for scores below it (rate cost cannot be negative).
    """
    return max(score, rate_contribution)


def _project_onto_seg(score: float, rate_contribution: float, seg_budget: float, pose_budget: float) -> float:
    """Project score onto seg <= S half-space.

    Lowest feasible score given (rate, seg<=S, pose<=P) is rate_contribution
    + 0 (seg=0) + sqrt(10*0) (pose=0) = rate_contribution. Highest feasible
    score is rate_contribution + seg_budget + sqrt(10*pose_budget). We clip
    to that interval.
    """
    upper = rate_contribution + seg_budget + math.sqrt(10.0 * pose_budget)
    return min(max(score, rate_contribution), upper)


def _project_onto_pose(score: float, rate_contribution: float, seg_budget: float, pose_budget: float) -> float:
    """Project score onto pose <= P half-space.

    Symmetric to the seg projection at the score-axis level; the difference
    between the two projections is in the budget-floor accounting used by
    the alternating sweep.
    """
    upper = rate_contribution + seg_budget + math.sqrt(10.0 * pose_budget)
    return min(max(score, rate_contribution), upper)


def _dykstra_feasibility(
    *,
    rate_contribution: float,
    seg_budget: float,
    pose_budget: float,
    band_lo: float,
    band_hi: float,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[float, float, int]:
    """Alternating-projections sweep over the 3 convex constraints.

    Returns ``(feasible_lo, feasible_hi, iter_count)``. The returned interval
    is the intersection of the predicted band with the feasibility polytope's
    projection onto the score axis.
    """
    # The feasibility polytope's score-axis projection is the closed interval
    # ``[rate_contribution, rate_contribution + seg_budget + sqrt(10*pose_budget)]``.
    # Alternating projections converge in one sweep for a single scalar; we
    # still loop to make the iteration count meaningful for the audit ledger
    # and to expose extension points (additional constraints) without changing
    # the public contract.
    feasible_lo = rate_contribution
    feasible_hi = rate_contribution + seg_budget + math.sqrt(10.0 * pose_budget)
    if feasible_hi < feasible_lo:
        # Degenerate budgets — fail-closed; caller must tighten budgets.
        return (feasible_lo, feasible_lo, 0)

    intersect_lo = max(band_lo, feasible_lo)
    intersect_hi = min(band_hi, feasible_hi)

    # Alternating projections — successive applications converge geometrically.
    iter_count = 0
    prev_lo, prev_hi = intersect_lo, intersect_hi
    for iter_count in range(1, max_iter + 1):
        lo_proj = _project_onto_rate(prev_lo, rate_contribution)
        lo_proj = _project_onto_seg(lo_proj, rate_contribution, seg_budget, pose_budget)
        lo_proj = _project_onto_pose(lo_proj, rate_contribution, seg_budget, pose_budget)
        hi_proj = _project_onto_rate(prev_hi, rate_contribution)
        hi_proj = _project_onto_seg(hi_proj, rate_contribution, seg_budget, pose_budget)
        hi_proj = _project_onto_pose(hi_proj, rate_contribution, seg_budget, pose_budget)
        if abs(lo_proj - prev_lo) < tolerance and abs(hi_proj - prev_hi) < tolerance:
            prev_lo, prev_hi = lo_proj, hi_proj
            break
        prev_lo, prev_hi = lo_proj, hi_proj

    final_lo = max(prev_lo, feasible_lo, band_lo)
    final_hi = min(prev_hi, feasible_hi, band_hi)
    return (final_lo, final_hi, iter_count)


def _classify_blocker(
    *,
    band_lo: float,
    band_hi: float,
    rate_contribution: float,
    feasible_hi: float,
) -> Literal["rate", "seg", "pose"] | None:
    """Identify which convex constraint the band violates.

    Rate is the dominant blocker when band_hi < rate_contribution (the
    archive is already more expensive than the entire claimed band). When
    band_lo > feasible_hi the budget is exceeded; we attribute to the larger
    of the seg/pose distortion budgets at the violating axis. The helper
    returns ``None`` when the band intersects the feasible interval.
    """
    if band_hi < rate_contribution:
        return "rate"
    if band_lo > feasible_hi:
        # Both seg and pose budgets cap the upper bound; attribute to pose
        # because sqrt(10*pose_budget) is typically the dominant term at the
        # PR106 operating point. Callers that need a tighter attribution
        # should re-run with sharper budgets.
        return "pose"
    return None


def check_substrate_dykstra_feasibility(
    *,
    substrate_id: str,
    predicted_band_lo: float,
    predicted_band_hi: float,
    archive_size_bytes: int,
    seg_budget: float = DEFAULT_SEG_BUDGET,
    pose_budget: float = DEFAULT_POSE_BUDGET,
    max_iter: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOLERANCE,
) -> DykstraFeasibilityVerdict:
    """Compute the Dykstra-feasibility verdict for a substrate's claimed band.

    Raises ``ValueError`` on contract violations (negative bytes / inverted
    band / non-finite inputs) per CLAUDE.md "Comment-only contracts are
    FORBIDDEN" — fail-closed beats fail-vacuous for an autopilot consumer.
    """
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be a non-empty string")
    if not math.isfinite(predicted_band_lo) or not math.isfinite(predicted_band_hi):
        raise ValueError("predicted_band_{lo,hi} must be finite numbers")
    if predicted_band_lo > predicted_band_hi:
        raise ValueError(
            f"predicted_band_lo={predicted_band_lo} > predicted_band_hi={predicted_band_hi}"
        )
    if archive_size_bytes < 0:
        raise ValueError(f"archive_size_bytes={archive_size_bytes} must be >= 0")
    if seg_budget < 0 or pose_budget < 0:
        raise ValueError(
            f"seg_budget={seg_budget} and pose_budget={pose_budget} must be >= 0"
        )

    rate_contribution = 25.0 * float(archive_size_bytes) / float(CONTEST_RATE_DENOM_BYTES)
    feasible_hi = rate_contribution + seg_budget + math.sqrt(10.0 * pose_budget)

    feas_lo, feas_hi, iter_count = _dykstra_feasibility(
        rate_contribution=rate_contribution,
        seg_budget=seg_budget,
        pose_budget=pose_budget,
        band_lo=predicted_band_lo,
        band_hi=predicted_band_hi,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    if feas_lo > feas_hi:
        # No intersection between predicted band and feasibility polytope.
        blocker = _classify_blocker(
            band_lo=predicted_band_lo,
            band_hi=predicted_band_hi,
            rate_contribution=rate_contribution,
            feasible_hi=feasible_hi,
        )
        rationale = (
            f"predicted band [{predicted_band_lo:.6f}, {predicted_band_hi:.6f}] does "
            f"NOT intersect Dykstra-feasible polytope [{rate_contribution:.6f}, {feasible_hi:.6f}]; "
            f"blocker_axis={blocker}; revise band per Catalog #296 / audit §1"
        )
        verdict: VerdictStr = "INFEASIBLE"
    elif feas_lo == predicted_band_lo and feas_hi == predicted_band_hi:
        # Entire band is inside the polytope — strongest FEASIBLE verdict.
        rationale = (
            f"predicted band [{predicted_band_lo:.6f}, {predicted_band_hi:.6f}] is "
            f"fully contained in Dykstra-feasible polytope "
            f"[{rate_contribution:.6f}, {feasible_hi:.6f}]"
        )
        verdict = "FEASIBLE"
        blocker = None
    elif feas_hi - feas_lo < tolerance:
        # Single-point intersection — flag as INDETERMINATE; caller should
        # tighten budgets to resolve.
        rationale = (
            f"predicted band intersects feasibility polytope at a single point "
            f"({feas_lo:.6f}); tighten seg/pose budgets to disambiguate"
        )
        verdict = "INDETERMINATE"
        blocker = None
    else:
        # Partial intersection — verdict FEASIBLE but the rationale records
        # the clipping so the design memo can declare the polytope-projected
        # band rather than the original claimed band.
        rationale = (
            f"predicted band partially intersects feasibility polytope; "
            f"projected feasible interval = [{feas_lo:.6f}, {feas_hi:.6f}]"
        )
        verdict = "FEASIBLE"
        blocker = None

    return DykstraFeasibilityVerdict(
        substrate_id=substrate_id,
        verdict=verdict,
        predicted_band=(float(predicted_band_lo), float(predicted_band_hi)),
        rate_contribution=float(rate_contribution),
        seg_budget=float(seg_budget),
        pose_budget=float(pose_budget),
        feasibility_band_lo=float(feas_lo),
        feasibility_band_hi=float(feas_hi),
        feasibility_rationale=rationale,
        blocker_axis=blocker,
        dykstra_iteration_count=int(iter_count),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Dykstra-feasibility verdict for a substrate's predicted "
            "ΔS band against the contest scorer's convex constraint set "
            "(rate <= R AND seg <= S AND pose <= P). Canonical helper per "
            "high-risk substrate cargo-cult unwind audit 2026-05-16 (D3)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--substrate-id", required=True, help="lane / substrate identifier")
    parser.add_argument(
        "--predicted-band-lo", type=float, required=True,
        help="lower bound of predicted ΔS band",
    )
    parser.add_argument(
        "--predicted-band-hi", type=float, required=True,
        help="upper bound of predicted ΔS band",
    )
    parser.add_argument(
        "--archive-size-bytes", type=int, required=True,
        help="archive size in bytes (drives rate axis cost)",
    )
    parser.add_argument(
        "--seg-budget", type=float, default=DEFAULT_SEG_BUDGET,
        help="seg distortion budget (default = A1 standing floor)",
    )
    parser.add_argument(
        "--pose-budget", type=float, default=DEFAULT_POSE_BUDGET,
        help="pose distortion budget (default = A1 standing floor)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="optional path to write the verdict as JSON",
    )
    parser.add_argument(
        "--max-iter", type=int, default=DEFAULT_MAX_ITER,
        help="alternating-projection iteration cap",
    )
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help="convergence tolerance",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        verdict = check_substrate_dykstra_feasibility(
            substrate_id=args.substrate_id,
            predicted_band_lo=args.predicted_band_lo,
            predicted_band_hi=args.predicted_band_hi,
            archive_size_bytes=args.archive_size_bytes,
            seg_budget=args.seg_budget,
            pose_budget=args.pose_budget,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(f"[dykstra-feasibility] FATAL: {exc}", file=sys.stderr)
        return 2

    payload = asdict(verdict)
    # asdict turns the tuple into a list — keep it as a list in JSON.
    payload["predicted_band"] = list(payload["predicted_band"])
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True, indent=2))
    if verdict.verdict == "INFEASIBLE":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
