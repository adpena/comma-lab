# SPDX-License-Identifier: MIT
"""Cathedral consumer for META-LIFT-2 Pareto polytope unified solver.

Per Catalog #335 :class:`tac.cathedral.consumer_contract.CathedralConsumerContract`
+ the 11th standing directive ORDER discipline (ONE canonical Pareto
polytope solver FIRST in :mod:`tac.pareto_polytope_unified_solver`;
per-substrate consumption SECOND through this consumer). Auto-discovered
per Catalog #336/#337 invocation gates by
:func:`tools.cathedral_autopilot_autonomous_loop.discover_and_register_consumers`.

Sister of:
  * :mod:`tac.cathedral_consumers.cross_substrate_master_gradient_analyzer_consumer`
    (META-LIFT-1 — provides ranked-opportunity input to META-LIFT-2)
  * :mod:`tac.cathedral_consumers.master_gradient_aggregate_consumer`
    (Catalog #354 exploit #1 — per-substrate aggregate gradient observability)
  * :mod:`tac.cathedral_consumers.bit_allocator_per_pair_consumer`
    (per-pair bit allocator at a DIFFERENT axis than the unified
    cross-substrate Pareto polytope surface)

Where the sister Catalog #354 exploit consumers operate PER-SUBSTRATE,
and the META-LIFT-1 consumer surfaces ranked opportunities ACROSS
substrates, this consumer is the META-LIFT-2 sister: it loads the
latest canonical Pareto polytope solution from
``.omx/state/pareto_polytope_solutions.jsonl`` (sister of
``.omx/state/cross_substrate_master_gradient_analyses.jsonl`` at the
solver-output sub-surface) and surfaces a per-candidate annotation
listing the candidate's unified bit-budget allocation across all 3 axes.

Per Catalog #341 cathedral consumer routing markers: every return value
carries ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The Pareto polytope allocation is
OBSERVABILITY-ONLY by construction. Promotion of an allocation entry to
a contest dispatch decision REQUIRES paired-CUDA empirical anchor per
CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable.

Hook numbers per Catalog #125 6-hook wire-in:
  * Hook #1 SENSITIVITY_MAP — ACTIVE (per-substrate per-axis allocations
    feed :mod:`tac.sensitivity_map` axis_weights downstream)
  * Hook #2 PARETO_CONSTRAINT — ACTIVE PRIMARY (this consumer IS the
    canonical Pareto polytope consumer per CLAUDE.md "Meta-Lagrangian/
    Pareto solver" non-negotiable; Dim 1 Phase 4 binding implementation)
  * Hook #3 BIT_ALLOCATOR — ACTIVE PRIMARY (per-substrate per-axis
    allocations feed the bit allocator priority cascade per Dim 6
    Step 6.5)
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (this consumer is the
    canonical cathedral entry point for the META-LIFT-2 solver)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (new solution rows
    written via :func:`tac.pareto_polytope_unified_solver.append_solution_locked`
    are read fresh per candidate so the consumer always sees the latest
    posterior; the consumer is STATELESS by design)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (the per-axis allocation IS
    the canonical disambiguator between competing dispatch budget
    routes — a substrate with high seg leverage but low pose leverage
    receives a different allocation than the inverse per CLAUDE.md
    "SegNet vs PoseNet importance — operating-point dependent")

Mission contribution per Catalog #300: ``frontier_breaking_enabler`` —
the canonical Pareto polytope allocation unblocks per-axis dispatch
budget routing (Dim 1 Phase 4 + Dim 6 Step 6.5) without per-substrate
manual tuning. The immediate score-lowering value is via re-routing
dispatch budget from low-leverage substrate-axes to high-leverage ones
according to the canonical Dykstra-projected allocation.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "pareto_polytope_unified_solver_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def _state_dir() -> Path:
    """Locate ``.omx/state/`` for canonical ledger discovery.

    Walks up from this file to find the repo root + ``.omx/state/``.
    Returns a non-existent Path-equivalent (``.omx/state``) if not
    found so the consumer fails GRACEFULLY (returns an empty
    annotation rather than crashing the cathedral autopilot loop).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_solution() -> Mapping[str, Any] | None:
    """Load the most-recent Pareto polytope solution row from the ledger.

    Returns ``None`` if the ledger is missing (graceful degradation per
    Catalog #245 + #248 sister fail-closed disciplines: a missing
    solution does NOT crash the cathedral autopilot loop).
    """
    ledger = _state_dir() / "pareto_polytope_solutions.jsonl"
    if not ledger.exists():
        return None
    raw = ledger.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    # Most-recent row wins (latest written_at_utc).
    latest: Mapping[str, Any] | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Per Catalog #138 sister: lenient-loader fallback for
            # graceful degradation; strict-load is available via
            # tac.pareto_polytope_unified_solver.load_solutions_strict.
            continue
        if not isinstance(row, dict):
            continue
        if latest is None or row.get("written_at_utc", "") > latest.get("written_at_utc", ""):
            latest = row
    return latest


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Triggered when a new master-gradient anchor lands via
    :func:`tac.master_gradient.append_anchor_locked` (which would
    typically be followed by a META-LIFT-1 analyzer re-run via
    ``tools/cross_substrate_master_gradient_cli.py --persist-to-ledger``
    and then a META-LIFT-2 solver re-run via
    ``tools/pareto_polytope_solver_cli.py --persist-to-ledger``).

    This consumer is STATELESS (re-reads the latest Pareto polytope
    solution on every consume call) so the hook is a no-op by design.

    The canonical operator-routable flow is:

    1. Operator runs :mod:`tools.extract_master_gradient` to extract a
       new per-substrate anchor.
    2. ``append_anchor_locked`` writes the row + fires post-anchor
       consumer hooks.
    3. Operator runs :mod:`tools.cross_substrate_master_gradient_cli`
       (with ``--persist-to-ledger``) to rebuild the cross-substrate
       analysis incorporating the new anchor.
    4. Operator runs :mod:`tools.pareto_polytope_solver_cli` (with
       ``--persist-to-ledger``) to rebuild the unified Pareto polytope
       allocation incorporating the new META-LIFT-1 analysis.
    5. Cathedral autopilot ranker invokes this consumer's
       :func:`consume_candidate` per candidate and reads the updated
       solution fresh.

    Per Catalog #327 contest-axis custody: this hook does NOT promote
    diagnostic anchors. The authority filter is applied at META-LIFT-1
    analyzer time via :func:`tac.master_gradient.is_authoritative_axis_anchor`.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Annotates the candidate with its per-axis bit-budget allocation
    (if any) in the latest canonical Pareto polytope solution. The
    annotation is OBSERVABILITY-ONLY per Catalog #341
    (``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[predicted]"``).

    Args:
        candidate: ranker candidate dict; expected to carry
          ``archive_sha256`` (or ``archive_sha`` / ``sha256``) for
          allocation lookup.

    Returns:
        Canonical contribution dict with Pareto polytope allocation
        annotation (Tier A observability-only).
    """
    solution = _load_latest_solution()
    if solution is None:
        return {
            "consumer_name": CONSUMER_NAME,
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "annotation": {
                "pareto_polytope_solution_status": "MISSING_SOLUTION_LEDGER",
                "actionable_hint": (
                    "Run tools/pareto_polytope_solver_cli.py "
                    "--persist-to-ledger to populate the canonical solutions ledger."
                ),
            },
        }

    # Extract candidate sha (defensive — multiple ranker conventions).
    sha = (
        candidate.get("archive_sha256")
        or candidate.get("archive_sha")
        or candidate.get("sha256")
        or ""
    )

    allocation = solution.get("allocation", {})
    substrate_shas = allocation.get("substrate_archive_sha256s", [])
    per_axis = allocation.get("per_substrate_per_axis_allocations", [])
    aggregate = allocation.get("per_substrate_aggregate_allocations", [])
    aggregate_total = float(allocation.get("aggregate_total_bytes_allocated", 0.0))
    aggregate_predicted_delta_s = float(
        allocation.get("aggregate_predicted_delta_s", 0.0)
    )
    feasible = bool(allocation.get("feasible", False))

    # Find this candidate's allocation in the solution.
    candidate_allocation: dict[str, Any] | None = None
    if sha and substrate_shas:
        for idx, substrate_sha in enumerate(substrate_shas):
            if substrate_sha == sha:
                if idx < len(per_axis):
                    axis_alloc = per_axis[idx]
                    candidate_allocation = {
                        "substrate_index": idx,
                        "byte_budget_seg": float(axis_alloc[0]) if len(axis_alloc) > 0 else 0.0,
                        "byte_budget_pose": float(axis_alloc[1]) if len(axis_alloc) > 1 else 0.0,
                        "byte_budget_rate": float(axis_alloc[2]) if len(axis_alloc) > 2 else 0.0,
                        "byte_budget_aggregate": (
                            float(aggregate[idx]) if idx < len(aggregate) else 0.0
                        ),
                    }
                break

    if candidate_allocation:
        pareto_status = "ALLOCATED"
    elif sha and substrate_shas:
        pareto_status = "NOT_IN_SOLUTION"
    else:
        pareto_status = "UNALLOCATED"

    return {
        "consumer_name": CONSUMER_NAME,
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "annotation": {
            "pareto_polytope_solution_id": solution.get("solution_id"),
            "pareto_polytope_solution_status": pareto_status,
            "candidate_archive_sha256": sha,
            "candidate_allocation": candidate_allocation,
            "aggregate_total_bytes_allocated": aggregate_total,
            "aggregate_predicted_delta_s": aggregate_predicted_delta_s,
            "solution_feasible": feasible,
            "n_substrates_in_solution": len(substrate_shas),
            "n_iterations_to_convergence": int(
                solution.get("n_iterations_to_convergence", 0)
            ),
            "converged": bool(solution.get("converged", False)),
            "upstream_meta_lift_1_analysis_id": solution.get(
                "upstream_meta_lift_1_analysis_id"
            ),
            "canonical_equation_id": solution.get("canonical_equation_id"),
            "canonical_equation_status": solution.get("canonical_equation_status"),
            "evidence_grade": solution.get(
                "evidence_grade", "[predicted; pareto-polytope-Dykstra-projections]"
            ),
            "promotion_routing": (
                "OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md "
                "'Apples-to-apples evidence discipline'. Promotion of "
                "allocation entry to contest dispatch decision REQUIRES "
                "paired-CUDA empirical anchor per CLAUDE.md 'Submission "
                "auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT "
                "HARDWARE'."
            ),
        },
    }


__all__ = [
    "CONSUMER_HOOK_NUMBERS",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "consume_candidate",
    "update_from_anchor",
]
