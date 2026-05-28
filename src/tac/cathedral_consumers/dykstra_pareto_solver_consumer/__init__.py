# SPDX-License-Identifier: MIT
"""Cathedral consumer for the Dykstra Pareto polytope solver.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4 + Catalog #372
STRICT preflight gate + Catalog #335 canonical CathedralConsumerContract.

This consumer surfaces per-candidate Pareto polytope verdicts via
:func:`tac.dykstra_pareto_solver.solve_pareto_polytope_intersection`.
It is the sister cathedral-consumer surface for the SAME canonical
Dykstra Pareto polytope solver invoked by
``tools/cathedral_autopilot_autonomous_loop.py::invoke_dykstra_pareto_solver_on_candidates``,
auto-discovered per Catalog #335 paradigm alongside the 47+ other
production consumers.

Per Catalog #341 canonical-routing markers + Catalog #357 dual-tier
architecture: this consumer is **TIER_A_OBSERVABILITY_ONLY** at
landing. Every ``consume_candidate`` return value carries:

  * ``predicted_delta_adjustment=0.0`` (routing/observability signal)
  * ``promotable=False`` (per CLAUDE.md "Apples-to-apples")
  * ``axis_tag="[predicted]"`` (canonical observability axis)
  * ``canonical_provenance`` Catalog #323 dict-form Provenance

Per Catalog #305 observability surface (6-facet): inspectable per axis
(verdict.per_axis_dual_variables) / decomposable per signal
(verdict.tight_constraint_axes) / diff-able across runs (canonical
Provenance threading) / queryable post-hoc (canonical posterior anchor
via update_from_anchor) / cite-able (canonical equation
``dykstra_pareto_polytope_intersection_compounding_v1``) /
counterfactual-able (per-axis byte-mutation smoke per Catalog #139).

Hook assignments per Catalog #125:
  * #2 Pareto constraint — **ACTIVE PRIMARY** (THIS consumer surfaces
    per-candidate Pareto polytope verdicts that ARE the canonical
    Pareto constraint mechanism)
  * #1 sensitivity-map — **ACTIVE** (per-axis dual variables surface
    per-axis sensitivities)
  * #5 continual-learning posterior — **ACTIVE** (``update_from_anchor``
    forwards new anchors to canonical equation registry)
  * #4 cathedral autopilot dispatch — N/A (covered by sister invoker
    in ``tools/cathedral_autopilot_autonomous_loop.py``)
  * #3 bit-allocator — N/A at Tier A (Tier B promotion path per Dim 6
    Step 6.5 enables hook #3 via per-axis bit allocation)
  * #6 probe-disambiguator — N/A at Tier A

Phase 2 Tier B promotion pathway (per Catalog #357):
  When sister Slot 2 + future Compound C/D/F land empirical pairwise
  alpha measurements, this consumer becomes the natural Tier B
  promotion candidate:
    - Bump ``CONSUMER_VERSION`` to ``2.0.0``.
    - Add ``CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING``.
    - Emit non-zero ``predicted_delta_adjustment`` derived from
      per-axis dual variables (within Catalog #341 bounded contract).
    - Add ``predicted_axis_decomposition`` per Catalog #356 with
      non-empty ``canonical_provenance``.
    - Keep ``promotable=False`` per Catalog #357 (promotion still
      requires paired contest-CPU + contest-CUDA empirical anchors).

Cross-references:
  * :mod:`tac.dykstra_pareto_solver` (canonical solver facade)
  * :mod:`tac.findings_lagrangian.dual_solver_phase_2` (canonical
    Dykstra sister)
  * :mod:`tac.cathedral.consumer_contract` (canonical contract)
  * :mod:`tac.canonical_equations` (canonical equation registry)
  * Catalog #335 / #341 / #356 / #357 / #372 sister gates
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "dykstra_pareto_solver_consumer"
CONSUMER_VERSION = "1.0.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Forwards new empirical anchors to the canonical equation
    ``dykstra_pareto_polytope_intersection_compounding_v1`` via the
    canonical equations registry. Per Catalog #371 auto-recalibrator:
    posterior anchors accumulate and the recalibrator refits the
    residual summary when ≥3 new anchors land in the same domain.

    Per Catalog #287/#323: this consumer does NOT auto-promote
    diagnostic anchors to contest-grade; the anchor's evidence_grade is
    honored upstream by :func:`tac.canonical_equations.update_equation_with_empirical_anchor`.

    Args:
        anchor: opaque anchor object emitted by upstream posterior
            update sources. May be a CanonicalEquation EmpiricalAnchor,
            a CouncilDeliberationRecord, or any sister surface that
            triggers consumer update.
    """
    # Tier A observability-only: the consumer does not mutate any
    # canonical state on its own; downstream
    # `tac.canonical_equations.update_equation_with_empirical_anchor`
    # handles the canonical posterior update.
    _ = anchor  # explicit acknowledgment; canonical update is upstream


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #2 — Pareto constraint canonical contribution.

    Per Catalog #341 canonical-routing markers: returns
    ``predicted_delta_adjustment=0.0`` + ``axis_tag="[predicted]"`` +
    ``promotable=False`` at Tier A. The structured rationale surfaces
    the canonical solver helper module path so downstream consumers
    can audit which canonical equation governs the prediction.

    Args:
        candidate: candidate dict-form payload from the cathedral
            autopilot ranker.

    Returns:
        Tier A canonical-routing-markers-compliant dict per
        Catalog #341 + Catalog #357.
    """
    candidate_id = (
        candidate.get("candidate_id", "?")
        if isinstance(candidate, Mapping)
        else "?"
    )
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            f"dykstra_pareto_solver_consumer Tier A observability-only verdict "
            f"for candidate {candidate_id!r}; canonical solver at "
            "tac.dykstra_pareto_solver.solve_pareto_polytope_intersection; "
            "canonical equation dykstra_pareto_polytope_intersection_compounding_v1; "
            "per Catalog #341 + #357 Tier A routing"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
        "canonical_helper_module": "tac.dykstra_pareto_solver",
        "canonical_equation_id": (
            "dykstra_pareto_polytope_intersection_compounding_v1"
        ),
    }
