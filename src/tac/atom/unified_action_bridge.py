# SPDX-License-Identifier: MIT
"""Sister-module bridge — ``tac.atom.Atom`` <-> ``tac.unified_action.Action``.

We do NOT modify ``src/tac/unified_action.py`` directly (sister
UNIFIED-ACTION subagent owns it per Catalog #314 absorption-avoidance
discipline + Catalog #230 sister-subagent ownership map). This bridge
provides the canonical helpers callers should use to compose atoms into
the canonical Lagrangian action.

Recommended consumer-side integration:

    from tac.atom import Atom
    from tac.atom.unified_action_bridge import (
        evaluate_action_with_atoms,
        atom_pool_to_meta_lagrangian_ledger,
        atom_pool_to_cathedral_autopilot_candidates,
    )

    # Compose atoms additively into the action's predicted-impact term:
    total_impact_band = evaluate_action_with_atoms(atoms=[a1, a2, a3])

    # Bridge to existing meta_lagrangian_allocator atom shape:
    legacy_atoms = atom_pool_to_meta_lagrangian_ledger([a1, a2, a3])

    # Bridge to cathedral autopilot ranker candidate shape:
    candidates = atom_pool_to_cathedral_autopilot_candidates([a1, a2, a3])

Citations:
  - ``tac.unified_action.Action`` per Catalog #125 hook 2 (Pareto
    constraint) canonical surface.
  - ``tac.meta_lagrangian_allocator.atoms_from_hnerv_decoder_recode_profile``
    canonical legacy atom shape.
  - Operator standing directive 2026-05-18: canonical atom IS the
    META-META-META element; bridges keep downstream consumers uniform.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .atom import Atom
from .types import AtomKind


def evaluate_action_with_atoms(*, atoms: Sequence[Atom]) -> tuple[float, float]:
    """Aggregate a pool of atoms into a (lower, upper) predicted-impact band.

    The aggregation is additive across atoms; this matches Boyd convex-
    optimization §5.1 Lagrangian decomposition where each atom contributes
    one additive term to the meta-Lagrangian. Per CLAUDE.md "Meta-
    Lagrangian/Pareto solver — NON-NEGOTIABLE": *"every stackable or
    substitutive idea should move toward a typed row consumed by the
    planner"* — this helper is the canonical aggregator.

    Per the Dykstra-feasibility discipline (Catalog #296): a naïve additive
    aggregation OVERESTIMATES impact when atoms conflict (sister `conflicts_with_*`
    metadata). Callers that need feasibility-respecting aggregation should
    use ``tac.unified_action.evaluate_with_admm`` for the canonical ADMM
    consensus instead.
    """
    if not atoms:
        return (0.0, 0.0)
    lower = sum(float(a.predicted_impact_delta_s_lower) for a in atoms)
    upper = sum(float(a.predicted_impact_delta_s_upper) for a in atoms)
    return (lower, upper)


def atom_pool_to_meta_lagrangian_ledger(atoms: Sequence[Atom]) -> list[dict[str, Any]]:
    """Convert a Sequence[Atom] into the legacy meta_lagrangian_allocator atom format.

    Calls ``Atom.to_meta_lagrangian_atom()`` on each entry. The result is
    suitable for direct consumption by
    ``tac.meta_lagrangian_allocator.build_atom_ledger`` and sister
    allocator primitives that pre-date the canonical Atom type.
    """
    return [a.to_meta_lagrangian_atom() for a in atoms]


def atom_pool_to_cathedral_autopilot_candidates(
    atoms: Sequence[Atom],
) -> list[dict[str, Any]]:
    """Convert a Sequence[Atom] into the cathedral autopilot ranker candidate shape.

    Mirrors the ``CandidateRow`` schema consumed by
    ``tools/cathedral_autopilot_autonomous_loop.py::rank_candidates``.
    Per Catalog #125 hook 4 (cathedral autopilot dispatch): atoms are the
    canonical input rows; the ranker handles cost-band / venn-deliverability /
    composition-alpha reweighting downstream.

    Documented required wire-in: operator-routable subagent extends
    ``tools/cathedral_autopilot_autonomous_loop.py`` to call this bridge
    on every canonical atom ledger read. Until that lands, callers should
    invoke this helper directly + push results into the ranker's input
    list.
    """
    candidates: list[dict[str, Any]] = []
    for atom in atoms:
        midpoint = (
            atom.predicted_impact_delta_s_lower + atom.predicted_impact_delta_s_upper
        ) / 2.0
        candidate = {
            "candidate_id": atom.atom_id,
            "kind": atom.kind.value,
            "resolution_path": atom.resolution_path.value,
            "predicted_delta": float(midpoint),
            "predicted_delta_lower": float(atom.predicted_impact_delta_s_lower),
            "predicted_delta_upper": float(atom.predicted_impact_delta_s_upper),
            "cost_envelope_usd": float(atom.cost_envelope_usd),
            "evidence_grade": str(atom.provenance.get("evidence_grade", "predicted")),
            "literature_citation": atom.literature_citation,
            "canonical_helper_repo_link": atom.canonical_helper_repo_link,
            # Carry the full atom metadata so downstream consumers (Venn
            # classifier / composition-alpha v2 cascade / Dykstra-feasibility
            # reranker) can inspect kind-specific fields without re-fetching.
            "atom_metadata": dict(atom.metadata),
            "atom_provenance": dict(atom.provenance),
            "wired_hooks": list(atom.wired_hooks),
            "observability_surface": list(atom.observability_surface),
            # Hint to the Venn classifier: kind-specific routing.
            "kind_routing_hint": _kind_routing_hint(atom.kind),
        }
        candidates.append(candidate)
    return candidates


def _kind_routing_hint(kind: AtomKind) -> str:
    """Produce a kind-specific routing hint for the cathedral autopilot ranker.

    The hint is advisory — the autopilot's Venn classifier remains the
    canonical decision surface. Hints help the operator-facing JSON output
    surface kind-specific routing without re-reading the AtomKind enum.
    """
    return {
        AtomKind.ARBITRARY_VALUE: "rank_by_ev_per_dollar",
        AtomKind.META_LAGRANGIAN: "feed_into_meta_lagrangian_allocator",
        AtomKind.CARGO_CULT_ASSUMPTION: "queue_for_unwind_test",
        AtomKind.PREMISE_VERIFICATION: "advisory_only_no_dispatch",
        AtomKind.PROBE_OUTCOME: "consult_for_predecessor_block_per_catalog_313",
        AtomKind.COUNCIL_DELIBERATION: "weight_dispatch_by_council_verdict",
        AtomKind.DISPATCH_CLAIM: "filter_active_claims_no_double_spend",
    }[kind]


__all__ = [
    "atom_pool_to_cathedral_autopilot_candidates",
    "atom_pool_to_meta_lagrangian_ledger",
    "evaluate_action_with_atoms",
]
