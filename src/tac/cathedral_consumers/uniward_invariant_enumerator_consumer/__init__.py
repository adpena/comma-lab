# SPDX-License-Identifier: MIT
"""Cathedral consumer for META-LIFT-4 UNIWARD canonical-application-surface
invariant enumerator.

Per Catalog #335 :class:`tac.cathedral.consumer_contract.CathedralConsumerContract`
+ the 11th standing directive ORDER discipline (ONE canonical enumerator
FIRST; per-substrate consumption SECOND). Auto-discovered per Catalog #336 /
#337 invocation gates by
:func:`tools.cathedral_autopilot_autonomous_loop.discover_and_register_consumers`.

Sister of:
  * :mod:`tac.cathedral_consumers.cross_substrate_master_gradient_analyzer_consumer`
    (META-LIFT-1 consumer at per-substrate aggregate granularity)
  * :mod:`tac.cathedral_consumers.pareto_polytope_unified_solver_consumer`
    (META-LIFT-2 consumer at Pareto polytope feasibility granularity)
  * :mod:`tac.cathedral_consumers.master_gradient_aggregate_consumer` (Catalog
    #354 exploit #1 — per-substrate aggregate gradient observability)
  * :mod:`tac.cathedral_consumers.cross_substrate_similarity_consumer`
    (SUB_ADDITIVE / SUPER_ADDITIVE / ANTAGONISTIC classification at a DIFFERENT
    axis than the META-LIFT-4 ranked-surface surface)

Where META-LIFT-1 operates per-SUBSTRATE at the aggregate-tensor surface and
META-LIFT-2 operates per-SUBSTRATE-PAIR at the Pareto polytope surface, this
META-LIFT-4 consumer operates per-CANONICAL-APPLICATION-SURFACE at the
UNIWARD invariant surface: it loads the latest canonical enumeration from
``.omx/state/uniward_invariant_enumerations.jsonl`` (sister of
``.omx/state/cross_substrate_master_gradient_analyses.jsonl`` at the
UNIWARD invariant sub-surface) and surfaces a per-candidate annotation
listing the candidate substrate's canonical-application surfaces +
per-axis ranking + UNIWARD applicability verdicts.

Per Catalog #341 cathedral consumer routing markers: every return value
carries ``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
``axis_tag="[predicted]"``. The per-surface UNIWARD invariant is
OBSERVABILITY-ONLY by construction. Promotion of a ranking entry to a
contest score signal REQUIRES paired-CUDA empirical anchor on the specific
surface per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE" non-negotiable.

Hook numbers per Catalog #125 6-hook wire-in:
  * Hook #1 SENSITIVITY_MAP — ACTIVE (per-surface per-axis Taylor
    projections feed :mod:`tac.sensitivity_map` axis_weights downstream)
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-surface UNIWARD applicability +
    ranked Cauchy-Schwarz bound feed the bit allocator priority cascade
    per CLAUDE.md "Bit-level deconstruction and entropy discipline")
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (this consumer is the
    canonical cathedral entry point for the META-LIFT-4 enumeration)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (new enumeration rows
    written via :func:`tac.uniward_invariant_enumerator.append_enumeration_locked`
    are read fresh per candidate so the consumer always sees the latest
    posterior)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (the per-surface UNIWARD-applicable
    verdict IS the canonical disambiguator between Fridrich-natural surfaces
    vs raw-RGB application-domain mismatches; the 5th + 6th-order PARADIGM-
    NULL → 7th-order PARADIGM-VALIDATED transitions empirically validate
    this disambiguator)

Mission contribution per Catalog #300: `frontier_breaking_enabler` —
the canonical UNIWARD invariant enumeration unblocks per-surface
applicability routing across the entire substrate canvas; substrates
with UNIWARD-applicable surfaces become candidates for the canonical
sister B paired-CUDA validation pipeline (per the 7th-order memo
operator-routable next-step).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.cathedral.consumer_contract import HookNumber

CONSUMER_NAME = "uniward_invariant_enumerator_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.BIT_ALLOCATOR,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def _state_dir() -> Path:
    """Locate ``.omx/state/`` for canonical ledger discovery.

    Walks up from this file to find the repo root + ``.omx/state/``. Returns
    a non-existent Path-equivalent (``.omx/state``) if not found so the
    consumer fails GRACEFULLY (returns an empty annotation rather than
    crashing the cathedral autopilot loop).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".omx" / "state"
        if candidate.is_dir():
            return candidate
    return Path(".omx/state")


def _load_latest_enumeration() -> Mapping[str, Any] | None:
    """Load the most-recent UNIWARD invariant enumeration row from the ledger.

    Returns ``None`` if the ledger is missing (graceful degradation per
    Catalog #245 + #248 sister fail-closed disciplines: a missing
    enumeration does NOT crash the cathedral autopilot loop).
    """
    ledger = _state_dir() / "uniward_invariant_enumerations.jsonl"
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
            # Per Catalog #138 sister: lenient-loader fallback for graceful
            # degradation; strict-load is available via
            # tac.uniward_invariant_enumerator.load_enumerations_strict.
            continue
        if not isinstance(row, dict):
            continue
        if latest is None or row.get("written_at_utc", "") > latest.get("written_at_utc", ""):
            latest = row
    return latest


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Triggered when a new master-gradient anchor lands via
    :func:`tac.master_gradient.append_anchor_locked` (the canonical
    upstream signal). This consumer is STATELESS (re-reads the latest
    canonical enumeration on every consume call) so the hook is a no-op
    by design — the enumeration is regenerated on-demand via
    ``tools/uniward_invariant_enumerator_cli.py --persist-to-ledger``.

    The canonical operator-routable flow:

    1. Operator runs :mod:`tools.uniward_invariant_enumerator_cli`
       (with ``--persist-to-ledger``) to populate the canonical
       enumeration ledger.
    2. Cathedral autopilot ranker invokes this consumer's
       :func:`consume_candidate` per candidate and reads the updated
       enumeration fresh.
    3. Downstream consumers (Pareto polytope solver per META-LIFT-2,
       bit-allocator per Dim 6 Step 6.5, sensitivity-map per hook #1)
       inherit the per-surface UNIWARD applicability verdict.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Annotates the candidate with its substrate's canonical-application
    surfaces + per-axis rankings + UNIWARD applicability verdicts. The
    annotation is OBSERVABILITY-ONLY per Catalog #341
    (``predicted_delta_adjustment=0.0`` + ``promotable=False`` +
    ``axis_tag="[predicted]"``).

    Args:
        candidate: ranker candidate dict; expected to carry
          ``substrate_id`` (or ``lane_id`` containing substrate fragment)
          for surface lookup.

    Returns:
        Canonical contribution dict with UNIWARD invariant enumeration
        annotation (Tier A observability-only).
    """
    enumeration = _load_latest_enumeration()
    if enumeration is None:
        return {
            "consumer_name": CONSUMER_NAME,
            "predicted_delta_adjustment": 0.0,
            "promotable": False,
            "axis_tag": "[predicted]",
            "annotation": {
                "uniward_invariant_status": "MISSING_ENUMERATION_LEDGER",
                "actionable_hint": (
                    "Run tools/uniward_invariant_enumerator_cli.py "
                    "--enumerate-all --persist-to-ledger to populate the "
                    "canonical enumeration ledger."
                ),
            },
        }

    # Extract candidate substrate_id (defensive — multiple ranker conventions).
    substrate_id = (
        candidate.get("substrate_id")
        or candidate.get("lane_id", "").split("lane_")[-1]
        or candidate.get("lane_class", "")
        or ""
    )

    surfaces = enumeration.get("surfaces", [])
    verdicts = enumeration.get("verdicts", [])
    rankings_per_axis = enumeration.get("rankings_per_axis", [])

    # Find surfaces belonging to this candidate's substrate (substring match).
    candidate_surfaces: list[dict] = []
    for surface, verdict in zip(surfaces, verdicts):
        s_substrate = surface.get("substrate_id", "")
        s_surface = surface.get("surface_id", "")
        if substrate_id and (
            s_substrate == substrate_id
            or s_substrate in substrate_id
            or substrate_id in s_substrate
            or s_surface in substrate_id
            or substrate_id in s_surface
        ):
            candidate_surfaces.append(
                {
                    "surface_id": s_surface,
                    "surface_kind": surface.get("surface_kind"),
                    "verdict": verdict.get("verdict"),
                    "all_conditions_pass": (
                        verdict.get("condition_1_entropy_coded", False)
                        and verdict.get("condition_2_quantized", False)
                        and verdict.get("condition_3_per_symbol_routable", False)
                        and verdict.get("condition_4_canonical_formula_grounded", False)
                    ),
                    "canonical_reference": verdict.get("canonical_reference_cited"),
                }
            )

    # Find candidate's surface ranks per axis.
    candidate_ranks_per_axis: dict[str, list[dict]] = {"seg": [], "pose": [], "rate": []}
    for ranking in rankings_per_axis:
        axis = ranking.get("axis", "")
        ranked_ids = ranking.get("ranked_surface_ids", [])
        bounds = ranking.get("per_surface_predicted_delta_s_upper_bound", [])
        for i, sid in enumerate(ranked_ids):
            for cand_surf in candidate_surfaces:
                if cand_surf["surface_id"] == sid:
                    candidate_ranks_per_axis[axis].append(
                        {
                            "surface_id": sid,
                            "rank": i + 1,
                            "predicted_delta_s_upper_bound": float(bounds[i]) if i < len(bounds) else 0.0,
                        }
                    )

    if candidate_surfaces:
        uniward_status = "RANKED"
    elif substrate_id:
        uniward_status = "NO_SURFACES_FOUND_FOR_SUBSTRATE"
    else:
        uniward_status = "UNRANKED_NO_SUBSTRATE_ID"

    return {
        "consumer_name": CONSUMER_NAME,
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "axis_tag": "[predicted]",
        "annotation": {
            "uniward_invariant_enumeration_id": enumeration.get("enumeration_id"),
            "uniward_invariant_status": uniward_status,
            "candidate_substrate_id": substrate_id,
            "candidate_surfaces": candidate_surfaces,
            "candidate_ranks_per_axis": candidate_ranks_per_axis,
            "n_applicable_surfaces_total": enumeration.get("n_applicable_surfaces", 0),
            "n_inapplicable_surfaces_total": enumeration.get("n_inapplicable_surfaces", 0),
            "canonical_equation_id": enumeration.get("canonical_equation_id"),
            "canonical_equation_status": enumeration.get("canonical_equation_status"),
            "evidence_grade": enumeration.get(
                "evidence_grade",
                "[predicted; uniward-canonical-application-surface-enumeration]",
            ),
            "promotion_routing": (
                "OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md "
                "'Apples-to-apples evidence discipline'. Promotion of "
                "any UNIWARD applicability ranking to contest score signal "
                "REQUIRES paired-CUDA empirical anchor per CLAUDE.md "
                "'Submission auth eval — BOTH CPU AND CUDA'."
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
