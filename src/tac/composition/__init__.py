"""Composition cell registry: (substrate × primitive × order) tuples.

Per operator directive 2026-05-12 ("stacking and composition on
everything"), this package extends
:mod:`tac.optimization.substrate_composition_matrix` (the pairwise
SUBSTRATE × SUBSTRATE matrix) with a NEW second dimension —
packet-compiler PRIMITIVES — and enumerates the cross-product as typed
:class:`CompositionCell` rows the cathedral autopilot can rank.

Public surface
--------------
- :class:`CompositionCell` — one (substrate × ordered-primitive-pipeline) row.
- :class:`PrimitiveRow` — one packet-compiler primitive inventory row.
- :class:`PrimitiveCategory` — top-level primitive taxonomy.
- :class:`PrimitiveOrderSensitivity` — pipeline ordering rule per category.
- :func:`canonical_primitive_inventory` — 14 packet-compiler primitives.
- :func:`primitive_compatibility` — substrate_class × primitive_category gate.
- :func:`validate_pipeline_ordering` — pipeline ordering / MX validator.
- :func:`enumerate_cells` — the autopilot's substrate-composition input.
- :func:`autopilot_ranking_input` — autopilot CandidateRow-compatible dicts.
- :func:`serialize_enumeration` — JSON-safe enumeration payload.

Score-claim discipline (NON-NEGOTIABLE per CLAUDE.md)
-----------------------------------------------------
Every cell carries ``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False`` until an empirical anchor is
posted via :mod:`tac.continual_learning`. Predicted deltas are derived
from primitive metadata; no number here is an authoritative measurement.

Cross-references
----------------
- :mod:`tac.optimization.substrate_composition_matrix` — pairwise
  SUBSTRATE × SUBSTRATE matrix; this module re-uses its taxonomy.
- :mod:`tac.optimization.autopilot_dispatch_ranking` — consumer.
- :mod:`tac.packet_compiler` — primitive symbol surface.
- :mod:`tac.continual_learning` — posterior anchors (wire-in hook 5).
"""

from __future__ import annotations

from tac.composition.enumerate import (
    ENUMERATION_SCHEMA,
    autopilot_ranking_input,
    enumerate_cells,
    serialize_enumeration,
)
from tac.composition.registry import (
    PLANNING_ONLY,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCHEMA_VERSION,
    SCORE_CLAIM,
    CompositionCell,
    PrimitiveCategory,
    PrimitiveOrderSensitivity,
    PrimitiveRow,
    ScoreAxis,
    SubstrateClass,
    SubstrateRow,
    canonical_primitive_inventory,
    canonical_substrate_inventory,
    primitive_compatibility,
    serialize_primitive_inventory,
    validate_pipeline_ordering,
)

__all__ = [
    # Schema constants.
    "SCHEMA_VERSION",
    "ENUMERATION_SCHEMA",
    "PLANNING_ONLY",
    "SCORE_CLAIM",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    # Types.
    "CompositionCell",
    "PrimitiveRow",
    "PrimitiveCategory",
    "PrimitiveOrderSensitivity",
    # Re-exports from substrate matrix (taxonomy convenience).
    "ScoreAxis",
    "SubstrateClass",
    "SubstrateRow",
    # Inventory + helpers.
    "canonical_primitive_inventory",
    "canonical_substrate_inventory",
    "primitive_compatibility",
    "validate_pipeline_ordering",
    "serialize_primitive_inventory",
    # Enumeration.
    "enumerate_cells",
    "serialize_enumeration",
    "autopilot_ranking_input",
]
