#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Canvas multi-op composition closed-form prediction sweep (PARADOX-CLOSER Half 2).

DOES multi-op composition beat single-op at the contest-CPU frontier?

This is the $0 closed-form prediction sweep that closes the drop-one-frontier
paradox Half 2. It:

  1. Probes ``.omx/state/master_gradient_anchors.jsonl`` for the per-axis
     (seg/pose/rate) decomposition on the canonical frontier archive (DQS1 sha
     ``7a0da5d0fc327cba`` if present, else fec6 ``6bae0201``).
  2. Populates the 5D canvas via
     ``populate_5d_canvas_from_master_gradient_anchors`` (BUILD-1 entry point).
  3. Runs the 12 operators (4 canonical + 8 extended) through the populated
     canvas to generate candidate multi-op compositions.
  4. Feeds the candidate compositions to the META-LIFT-2 Pareto polytope Dykstra
     solver to compute the PREDICTED optimal multi-op composition's ΔS
     (closed-form Cauchy-Schwarz + Dykstra alternating projections, NOT an
     empirical run).
  5. Compares predicted optimal multi-op ΔS against V14-V2's frontier-crossing
     -7.66e-6 [contest-CPU] (DQS1 baseline 0.1920282830) and emits the 3-outcome
     verdict.

ALL $0 closed-form. NO paid dispatch. macOS-CPU advisory NON-PROMOTABLE per
Catalog #127/#192. Every score-claim row carries score_claim=false +
promotable=false + axis_tag="[predicted]" per Catalog #323/#341.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas import (
    CanonicalOperation,
    ScorerAxis,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators import (
    DropFrameParameters,
    ExtendedOperation,
    MergePairParameters,
    MotionConditionalParameters,
    ReorderPairParameters,
    ReplaceManyParameters,
    ReplaceOneParameters,
    SynthesizeFrameParameters,
    TemporalCoherenceParameters,
    generate_drop_frame_candidates,
    generate_merge_pair_candidates,
    generate_motion_conditional_candidates,
    generate_reorder_pair_candidates,
    generate_replace_many_candidates,
    generate_replace_one_candidates,
    generate_synthesize_frame_candidates,
    generate_temporal_coherence_candidates,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
    populate_5d_canvas_from_master_gradient_anchors,
    populate_per_pair_cells_from_gradient_array,
)
from tac.pareto_polytope_unified_solver.solver import (
    PareDLPProblemSpec,
    solve_pareto_polytope_via_dykstra_projections,
)

# Canonical contest formula constants (Catalog #356 tac.score_composition).
CONTEST_SEG_MULTIPLIER = 100.0
CONTEST_POSE_SQRT_INNER = 10.0
CONTEST_RATE_MULTIPLIER = 25.0
CONTEST_RATE_DENOM_BYTES = 37_545_489

# Frontier reference per task (DQS1 baseline; V14-V2 crossing).
DQS1_BASELINE_CONTEST_CPU = 0.1920282830
V14_V2_FRONTIER_CROSSING_DELTA = -7.66e-6  # V14-V2 [contest-CPU] vs DQS1 baseline

# Frontier archive sha preference order.
DQS1_FRONTIER_SHA = "7a0da5d0fc327cba"
FEC6_FALLBACK_SHA = "6bae0201"


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_frontier_archive_sha(repo_root: Path) -> tuple[str, str]:
    """Return (resolved_full_sha, which) preferring DQS1 then fec6."""
    from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_populator import (
        list_distinct_archives_in_ledger,
    )

    distinct = list_distinct_archives_in_ledger(repo_root=repo_root)
    for sha in distinct:
        if sha.startswith(DQS1_FRONTIER_SHA):
            return sha, "dqs1_frontier"
    for sha in distinct:
        if sha.startswith(FEC6_FALLBACK_SHA):
            return sha, "fec6_fallback"
    raise SystemExit(
        "neither DQS1 frontier nor fec6 fallback archive present in master "
        f"gradient ledger; distinct archives = {distinct}"
    )


def _pose_score_term(d_pose: float) -> float:
    return math.sqrt(CONTEST_POSE_SQRT_INNER * max(d_pose, 0.0))


def _rate_score_term(archive_bytes: float) -> float:
    return CONTEST_RATE_MULTIPLIER * archive_bytes / CONTEST_RATE_DENOM_BYTES


@dataclass(frozen=True)
class OperatorCandidateSummary:
    operation: str
    n_candidates: int
    best_predicted_delta_score: float
    best_predicted_byte_cost: int
    best_axis_decomposition: dict[str, Any] | None


def _run_all_12_operators(canvas, top_n: int = 32) -> list[OperatorCandidateSummary]:
    """Run all 12 operators (4 canonical-start + 8 extended) over the canvas."""
    summaries: list[OperatorCandidateSummary] = []

    # 4 canonical operations: surfaced via the lattice's generate_*_starts which
    # produce ExecutableCandidate rows directly from the operation enum.
    canonical_starts = {
        CanonicalOperation.FULL_DROP: canvas.generate_full_drop_starts,
        CanonicalOperation.REPAIR: canvas.generate_repair_starts,
        CanonicalOperation.MASKED: canvas.generate_masked_starts,
        CanonicalOperation.FEATHERED: canvas.generate_feathered_starts,
    }
    for op, gen in canonical_starts.items():
        try:
            cands = list(gen())
        except Exception as exc:
            summaries.append(
                OperatorCandidateSummary(
                    operation=f"{op.value}::ERROR::{type(exc).__name__}",
                    n_candidates=0,
                    best_predicted_delta_score=0.0,
                    best_predicted_byte_cost=0,
                    best_axis_decomposition=None,
                )
            )
            continue
        summaries.append(_summarize_candidates(op.value, cands))

    # 8 extended operations with canonical default parameters.
    extended = [
        (ExtendedOperation.REPLACE_ONE, generate_replace_one_candidates, ReplaceOneParameters),
        (ExtendedOperation.REPLACE_MANY, generate_replace_many_candidates, ReplaceManyParameters),
        (ExtendedOperation.MERGE_PAIR, generate_merge_pair_candidates, MergePairParameters),
        (ExtendedOperation.REORDER_PAIR, generate_reorder_pair_candidates, ReorderPairParameters),
        (ExtendedOperation.DROP_FRAME, generate_drop_frame_candidates, DropFrameParameters),
        (ExtendedOperation.SYNTHESIZE_FRAME, generate_synthesize_frame_candidates, SynthesizeFrameParameters),
        (ExtendedOperation.MOTION_CONDITIONAL, generate_motion_conditional_candidates, MotionConditionalParameters),
        (ExtendedOperation.TEMPORAL_COHERENCE, generate_temporal_coherence_candidates, TemporalCoherenceParameters),
    ]
    for op, gen, param_cls in extended:
        try:
            params = param_cls()  # canonical defaults
        except TypeError:
            # Some parameter dataclasses require positional args; build minimal.
            params = None
        try:
            cands = [] if params is None else list(gen(canvas, params, top_n=top_n))
        except Exception as exc:
            summaries.append(
                OperatorCandidateSummary(
                    operation=f"{op.value}::ERROR::{type(exc).__name__}::{exc}",
                    n_candidates=0,
                    best_predicted_delta_score=0.0,
                    best_predicted_byte_cost=0,
                    best_axis_decomposition=None,
                )
            )
            continue
        summaries.append(_summarize_candidates(op.value, cands))

    return summaries


def _summarize_candidates(op_value: str, cands: list) -> OperatorCandidateSummary:
    if not cands:
        return OperatorCandidateSummary(
            operation=op_value,
            n_candidates=0,
            best_predicted_delta_score=0.0,
            best_predicted_byte_cost=0,
            best_axis_decomposition=None,
        )
    # Best = most negative predicted_delta_score (largest score reduction).
    best = min(cands, key=lambda c: getattr(c, "predicted_delta_score", 0.0))
    axis_dec = getattr(best, "predicted_axis_decomposition", None)
    axis_payload: dict[str, Any] | None = None
    if axis_dec is not None:
        if hasattr(axis_dec, "as_dict"):
            axis_payload = axis_dec.as_dict()
        elif isinstance(axis_dec, dict):
            axis_payload = axis_dec
    return OperatorCandidateSummary(
        operation=op_value,
        n_candidates=len(cands),
        best_predicted_delta_score=float(getattr(best, "predicted_delta_score", 0.0)),
        best_predicted_byte_cost=int(getattr(best, "predicted_byte_cost", 0) or 0),
        best_axis_decomposition=axis_payload,
    )


def _build_multiop_problem_spec(
    canvas,
    summaries: list[OperatorCandidateSummary],
    frontier_sha: str,
) -> PareDLPProblemSpec:
    """Build a Pareto polytope problem spec treating each productive operator
    as a substrate-axis in the multi-op composition.

    The per-axis gradient L2 norms are derived from the canvas's archive-aggregate
    cells: a productive operator that produces candidates contributes its best
    (|d_seg|, |d_pose|, |rate|) magnitude as the gradient norm on that axis.
    """
    seg = abs(_axis_cell_value(canvas, ScorerAxis.SEGNET_5CLASS))
    pose = abs(_axis_cell_value(canvas, ScorerAxis.POSENET_6D))
    rate = abs(_axis_cell_value(canvas, ScorerAxis.RATE_TERM))

    productive = [s for s in summaries if s.n_candidates > 0 and "ERROR" not in s.operation]
    if not productive:
        # Single-substrate degenerate spec (multi-op produced nothing).
        productive = summaries[:1]

    shas = [f"{frontier_sha[:12]}::{s.operation}" for s in productive]
    # Per-axis gradient L2 norms: weight the canvas-aggregate per-axis magnitude
    # by the operator's best predicted score reduction (more reduction => more
    # leverage on that operator-substrate's axis).
    norms: list[tuple[float, float, float]] = []
    per_axis_caps: list[tuple[float, float, float]] = []
    aggregate_caps: list[float] = []
    for s in productive:
        leverage = max(abs(s.best_predicted_delta_score), 1e-9)
        norms.append((seg * leverage, pose * leverage, rate * leverage))
        # Per-axis byte-budget caps: 10% of the canvas archive byte scale.
        cap = max(abs(s.best_predicted_byte_cost), 1.0)
        per_axis_caps.append((cap, cap, cap))
        aggregate_caps.append(cap * 3.0)

    # Cauchy-Schwarz aggregate upper bound: the canonical bound on total ΔS via
    # the multi-op composition is ||grad||_2 * ||allocation||_2. We set the
    # aggregate upper bound conservatively to the L2 norm of the stacked
    # per-axis gradient norms (the theoretical maximum savings the polytope
    # can deliver).
    grad_l2 = math.sqrt(sum(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 for g in norms))
    cs_bound = grad_l2 if grad_l2 > 0 else 1.0

    return PareDLPProblemSpec(
        substrate_archive_sha256s=tuple(shas),
        per_axis_gradient_l2_norms=tuple(norms),
        per_substrate_per_axis_byte_budget_caps=tuple(per_axis_caps),
        per_substrate_aggregate_byte_budget_caps=tuple(aggregate_caps),
        cauchy_schwarz_aggregate_upper_bound=cs_bound,
        target_aggregate_delta_s=None,
        measurement_axes=tuple("[predicted]" for _ in productive),
    )


def _axis_cell_value(canvas, scorer_axis: ScorerAxis) -> float:
    """Pull the archive-aggregate per-axis component value from the canvas.

    The lattice stores cells in the private sparse store ``_cells`` (no public
    iterator exists; ``query_cell`` requires the full 5-tuple coordinate). We
    iterate the sparse store directly because the multi-op prediction needs the
    per-axis aggregate regardless of receiver/cpu-cuda coordinate.
    """
    store = getattr(canvas, "_cells", {})
    for cell in store.values():
        if getattr(cell, "scorer_axis", None) == scorer_axis:
            return float(getattr(cell, "predicted_delta_score", 0.0))
    return 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-pair-gradient-npy",
        type=str,
        default=None,
        help=(
            "Path to the MLX per-pair master-gradient HEURISTIC-PRIOR artifact "
            "(.npy shape (N_bytes, N_pairs, 3)). When provided, the sweep "
            "re-points the canvas population to the per-pair path "
            "(populate_per_pair_cells_from_gradient_array) which gives the "
            ">=2-distinct-coordinate structure the 12 operators need. "
            "NON-PROMOTABLE macOS-MLX research-signal per Catalog #192/#127/#323. "
            "When omitted, the sweep uses the archive-AGGREGATE path "
            "(populate_5d_canvas_from_master_gradient_anchors) for backward "
            "compatibility with the original PARADOX-CLOSER Half 2 run."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on number of per-pair cells to populate (default: all).",
    )
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root()

    # Step 2: populate canvas. PER-PAIR path (re-pointed source) vs the original
    # archive-AGGREGATE path.
    if args.per_pair_gradient_npy:
        # Re-pointed per-pair HEURISTIC-PRIOR path (the Half-2 resolution unblock).
        manifest = populate_per_pair_cells_from_gradient_array(
            args.per_pair_gradient_npy,
            repo_root=repo_root,
            write_sidecar=False,
            max_pairs=args.max_pairs,
        )
        canvas = manifest.canvas
        frontier_sha = manifest.archive_sha256
        which = "mlx_per_pair_heuristic_prior"
        skip_mode = "per_pair_heuristic_prior_macos_mlx_advisory_nonpromotable"
        cell_count = canvas.cell_count()
    else:
        frontier_sha, which = _find_frontier_archive_sha(repo_root)
        # Advisory anchors skipped by default; if the only anchors for the
        # frontier sha are advisory, opt them in for the closed-form prediction
        # — the output is non-promotable [predicted] by construction.
        try:
            manifest = populate_5d_canvas_from_master_gradient_anchors(
                archive_sha256=frontier_sha,
                repo_root=repo_root,
                write_sidecar=False,
                skip_non_authoritative=True,
            )
            skip_mode = "authoritative_only"
        except Exception:
            manifest = populate_5d_canvas_from_master_gradient_anchors(
                archive_sha256=frontier_sha,
                repo_root=repo_root,
                write_sidecar=False,
                skip_non_authoritative=False,
            )
            skip_mode = "advisory_included_predicted_nonpromotable"

        canvas = manifest.canvas
        cell_count = canvas.cell_count()
        if cell_count == 0:
            # Retry including advisory anchors so the closed-form prediction has cells.
            manifest = populate_5d_canvas_from_master_gradient_anchors(
                archive_sha256=frontier_sha,
                repo_root=repo_root,
                write_sidecar=False,
                skip_non_authoritative=False,
            )
            canvas = manifest.canvas
            cell_count = canvas.cell_count()
            skip_mode = "advisory_included_predicted_nonpromotable"

    # Step 3: run all 12 operators.
    summaries = _run_all_12_operators(canvas, top_n=32)

    # Best single-op predicted ΔS (the single-op baseline for comparison).
    productive = [s for s in summaries if s.n_candidates > 0 and "ERROR" not in s.operation]
    best_single_op = (
        min(productive, key=lambda s: s.best_predicted_delta_score)
        if productive
        else None
    )
    best_single_op_delta = (
        best_single_op.best_predicted_delta_score if best_single_op else 0.0
    )

    # Step 4: build problem spec + run Dykstra solver.
    spec = _build_multiop_problem_spec(canvas, summaries, frontier_sha)
    solution = solve_pareto_polytope_via_dykstra_projections(spec, max_iterations=200, tol=1e-9)

    alloc_obj = solution.allocation  # UnifiedBitBudgetAllocation
    feasible = bool(alloc_obj.feasible)
    feasibility_residual = float(alloc_obj.feasibility_residual)
    aggregate_bytes_allocated = float(alloc_obj.aggregate_total_bytes_allocated)
    # The solver's canonical aggregate_predicted_delta_s is in byte-budget-units
    # (the <grad, allocation> inner product). Convert to contest-CPU score units
    # via the canonical rate denom: the realizable multi-op improvement beyond
    # the single-op base is bounded by the rate-term reduction the optimally
    # allocated bytes deliver. The polytope already enforced Cauchy-Schwarz +
    # per-axis + per-substrate-aggregate feasibility, so this inner product IS
    # the closed-form upper bound on synergy unlock.
    solver_aggregate_delta_s = float(alloc_obj.aggregate_predicted_delta_s)
    # Sign convention: the solver returns a non-negative aggregate (magnitude of
    # achievable score movement). Multi-op SAVINGS reduce score => negative ΔS.
    predicted_multiop_extra_delta = -abs(solver_aggregate_delta_s) / CONTEST_RATE_DENOM_BYTES

    # If the polytope is INFEASIBLE (intersection empty), the multi-op composition
    # cannot realize ANY extra synergy beyond single-op => extra delta is zero.
    if not feasible:
        predicted_multiop_extra_delta = 0.0

    # The predicted optimal multi-op ΔS = best single-op + the polytope-feasible
    # extra (which is bounded by orthogonality of the operator gradients).
    predicted_multiop_delta = best_single_op_delta + predicted_multiop_extra_delta

    # Step 5: 3-outcome verdict vs V14-V2 frontier crossing -7.66e-6.
    epsilon = 5e-7  # operating-point-saturation tolerance band
    if predicted_multiop_delta < V14_V2_FRONTIER_CROSSING_DELTA - epsilon:
        verdict = "MULTIOP_BEATS_V14V2"
        verdict_detail = (
            "predicted multi-op composition ΔS beats V14-V2 frontier crossing; "
            "rank top-3 compositions for FIRE-phase paired CPU+CUDA (operator-gated)"
        )
    elif abs(predicted_multiop_delta - V14_V2_FRONTIER_CROSSING_DELTA) <= epsilon:
        verdict = "OPERATING_POINT_SATURATION_CONFIRMED"
        verdict_detail = (
            "predicted multi-op ΔS ≈ V14-V2; operating-point-saturation CONFIRMED "
            "at this baseline; multi-op needs substrate-class-shift to unlock"
        )
    else:
        verdict = "SINGLE_OP_LOCALLY_OPTIMAL_DROP_MANY_H2_VINDICATED"
        verdict_detail = (
            "predicted multi-op ΔS WORSE than V14-V2; single-op locally optimal at "
            "frontier; DROP-MANY Hypothesis #2 fully vindicated"
        )

    # Rank top-3 operator compositions by best predicted ΔS.
    ranked = sorted(
        productive, key=lambda s: s.best_predicted_delta_score
    )[:3]

    payload: dict[str, Any] = {
        "schema_version": "canvas_multiop_composition_closed_form_prediction_sweep_v1_20260527",
        "frontier_archive_sha256": frontier_sha,
        "frontier_archive_which": which,
        "populator_skip_mode": skip_mode,
        "populator_source_kind": (
            "mlx_per_pair_heuristic_prior"
            if args.per_pair_gradient_npy
            else "master_gradient_anchors_archive_aggregate"
        ),
        "per_pair_gradient_npy": args.per_pair_gradient_npy,
        "per_pair_gradient_npy_sha256": (
            manifest.catalog_323_provenance.get("gradient_npy_sha256")
            if args.per_pair_gradient_npy
            else None
        ),
        "per_pair_pairs_populated": (
            manifest.catalog_323_provenance.get("pairs_populated")
            if args.per_pair_gradient_npy
            else None
        ),
        "canvas_cell_count": cell_count,
        "dqs1_baseline_contest_cpu": DQS1_BASELINE_CONTEST_CPU,
        "v14_v2_frontier_crossing_delta": V14_V2_FRONTIER_CROSSING_DELTA,
        "best_single_op_delta_score": best_single_op_delta,
        "best_single_op_operation": best_single_op.operation if best_single_op else None,
        "predicted_multiop_extra_delta": predicted_multiop_extra_delta,
        "predicted_multiop_delta_score": predicted_multiop_delta,
        "dykstra_converged": bool(solution.converged),
        "dykstra_n_iterations": int(solution.n_iterations_to_convergence),
        "dykstra_feasible": feasible,
        "dykstra_feasibility_residual": feasibility_residual,
        "dykstra_aggregate_bytes_allocated": aggregate_bytes_allocated,
        "dykstra_solver_aggregate_delta_s_raw": solver_aggregate_delta_s,
        "dykstra_axis_tag": solution.axis_tag,
        "dykstra_canonical_equation_id": solution.canonical_equation_id,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "top_3_ranked_compositions": [
            {
                "operation": s.operation,
                "n_candidates": s.n_candidates,
                "best_predicted_delta_score": s.best_predicted_delta_score,
                "best_predicted_byte_cost": s.best_predicted_byte_cost,
                "axis_decomposition": s.best_axis_decomposition,
            }
            for s in ranked
        ],
        "all_operator_summaries": [
            {
                "operation": s.operation,
                "n_candidates": s.n_candidates,
                "best_predicted_delta_score": s.best_predicted_delta_score,
                "best_predicted_byte_cost": s.best_predicted_byte_cost,
            }
            for s in summaries
        ],
        # Catalog #323/#341 canonical Provenance — non-promotable [predicted].
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[predicted]",
        "evidence_grade": "macOS-CPU-advisory-closed-form-prediction-nonpromotable",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
