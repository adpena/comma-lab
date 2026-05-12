"""Enumerate (substrate × primitive × order) composition cells.

Per operator directive 2026-05-12 ("stacking and composition on everything"),
:func:`enumerate_cells` produces the autopilot's ranking input as a stream
of :class:`tac.composition.registry.CompositionCell` rows. Each cell
combines:

- ONE substrate from :func:`tac.optimization.substrate_composition_matrix.canonical_substrate_inventory`.
- A pipeline of zero or more PRIMITIVES from :func:`tac.composition.registry.canonical_primitive_inventory`.
- An explicit composition order (matters when the pipeline crosses
  multiple ORDERED_PIPELINE primitives).

The enumeration respects:

1. **Compatibility matrix** (:func:`tac.composition.registry.primitive_compatibility`):
   skip cells where the primitive_category does not apply to the
   substrate_class (e.g., PR101 GOLD does NOT apply to RESIDUAL substrates).
2. **Mutually-exclusive categories**: at most ONE primitive per
   MUTUALLY_EXCLUSIVE category (sign-encoding × 1, schema-elision × 1).
3. **Ordered-pipeline within-category order**: PR101 GOLD trio appears
   strictly in declared order if multiple members are included.
4. **Cross-category order_index monotonicity**: smaller index appears
   first (PR101 storage → sign-encoding → brotli/lzma).

**Score-claim discipline**: every cell carries ``score_claim=False``,
``ready_for_exact_eval_dispatch=False``, ``promotion_eligible=False``.
The autopilot consumes the cells via the existing HALT-and-ASK pattern;
no number here is an empirical measurement.

**Wire-in**: :func:`enumerate_cells` is the autopilot's substrate-
composition-cell source. Register via
:func:`register_with_autopilot_dispatch_ranker` (kept minimal — it
returns the typed input the autopilot ranker accepts; the autopilot's
internal config is NOT mutated here per scope boundary).

Cross-references
----------------
- :mod:`tac.composition.registry` — types + canonical inventory.
- :mod:`tac.optimization.substrate_composition_matrix` — pairwise
  SUBSTRATE × SUBSTRATE matrix (re-used for substrate classification).
- :mod:`tac.optimization.autopilot_dispatch_ranking` — ranker that
  consumes the enumerated cells.
- :mod:`tac.continual_learning` — posterior anchors that update
  per-cell predicted deltas (wire-in hook 5).
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

from tac.composition.registry import (
    SCHEMA_VERSION,
    CompositionCell,
    PrimitiveCategory,
    PrimitiveOrderSensitivity,
    PrimitiveRow,
    SubstrateRow,
    _cell_to_dict,
    canonical_primitive_inventory,
    canonical_substrate_inventory,
    primitive_compatibility,
    validate_pipeline_ordering,
)

ENUMERATION_SCHEMA = "tac_composition_cell_enumeration_v1"


def _cell_id(substrate_id: str, primitive_ids: tuple[str, ...]) -> str:
    """Build a stable, human-readable cell_id.

    Format: ``cell__<substrate_id>__<p1>__<p2>__...__<pN>`` or
    ``cell__<substrate_id>__bare`` if no primitives are applied.
    """
    if not primitive_ids:
        return f"cell__{substrate_id}__bare"
    return "cell__" + substrate_id + "__" + "__".join(primitive_ids)


def _aggregate_predicted_deltas(
    substrate: SubstrateRow,
    primitives: list[PrimitiveRow],
) -> tuple[int, float]:
    """Aggregate predicted bytes_delta and score_delta across primitives.

    The aggregation is INTENTIONALLY conservative (mid-point of each
    primitive's band) and ADDITIVE under the planning-only assumption.
    Per CLAUDE.md "Cross-paradigm composition rules" the truly composite
    delta would require pairwise-alpha correction; that lives in the
    autopilot ranker's :func:`predicted_composite_delta` consumer. THIS
    aggregation feeds the ranker; the ranker applies the alpha correction
    on top.
    """
    bytes_mid = 0
    score_mid = 0.0
    for p in primitives:
        bytes_lo, bytes_hi = p.predicted_bytes_delta_band
        bytes_mid += int(round((bytes_lo + bytes_hi) / 2.0))
        score_lo, score_hi = p.predicted_score_delta_band
        score_mid += (score_lo + score_hi) / 2.0
    # Add substrate's predicted_delta_alone_midpoint() to the score (the
    # cell IS the substrate plus its primitive overlay).
    score_mid += substrate.predicted_delta_alone_midpoint()
    return bytes_mid, score_mid


def _detect_pr101_gold_overlap(
    substrate: SubstrateRow,
    primitive_ids: list[str],
) -> Optional[str]:
    """Detect PR101 GOLD applied to non-HNeRV substrate IDs.

    Per ``feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md``
    lesson 5, PR101 GOLD is HNeRV-family-specific. While the compatibility
    matrix already gates by substrate_class (RENDERER_REPLACEMENT only),
    NOT every RENDERER_REPLACEMENT substrate is HNeRV-derived enough to
    consume PR101's storage-order schema. We surface a soft blocker for
    non-HNeRV renderer-replacements that pull PR101 GOLD primitives so the
    operator sees the assumption explicitly.
    """
    hnerv_family_substrate_ids = {
        # Direct HNeRV-derivatives that PR101 GOLD's schema fits without
        # adaptation. (Empty set for now — Lane 12-v2 NeRV-as-renderer is
        # the canonical HNeRV-substrate target, but the actual HNeRV
        # substrate ships separately and is not in canonical_substrate_inventory yet.)
        # Conservative: require non-empty intersection with the HNeRV-family
        # token set. When the HNeRV substrate(s) land in the inventory,
        # extend this set.
    }
    pr101_gold_present = any(p.startswith("pr101_") for p in primitive_ids)
    if not pr101_gold_present:
        return None
    if substrate.substrate_id in hnerv_family_substrate_ids:
        return None
    # Surface a soft blocker (not a hard skip) so cells still rank for
    # operator review.
    return (
        f"pr101_gold_applied_to_non_hnerv_substrate: substrate_id={substrate.substrate_id} "
        "is RENDERER_REPLACEMENT but not in the canonical HNeRV-family set. "
        "PR101 GOLD schema may need substrate-specific adaptation. Review with operator."
    )


def enumerate_cells(
    substrates: Optional[list[SubstrateRow]] = None,
    primitives: Optional[list[PrimitiveRow]] = None,
    *,
    max_primitives_per_cell: int = 4,
    include_bare_substrate: bool = True,
) -> list[CompositionCell]:
    """Enumerate (substrate × ordered-primitive-pipeline) composition cells.

    Per operator directive 2026-05-12 ("stacking and composition on
    everything"), produces the autopilot ranking input. The enumeration:

    1. Iterates every substrate × primitive-subset combination.
    2. Filters by :func:`primitive_compatibility` (substrate_class ×
       primitive_category gate).
    3. Filters by within-category mutual-exclusion + ordered-pipeline
       constraints via :func:`validate_pipeline_ordering`.
    4. Sorts the pipeline by ``order_index`` to honor cross-category
       monotonicity.
    5. Aggregates per-primitive predicted deltas under the planning-only
       additive assumption.
    6. Emits one :class:`CompositionCell` per valid (substrate, pipeline)
       combination.

    The complexity is O(N_substrates * 2^N_primitives) in the worst case;
    in practice the compatibility-matrix gate and mutually-exclusive
    constraints prune > 99% of the cross-product.

    Parameters
    ----------
    substrates:
        Optional substrate inventory override; defaults to
        :func:`canonical_substrate_inventory`.
    primitives:
        Optional primitive inventory override; defaults to
        :func:`canonical_primitive_inventory`.
    max_primitives_per_cell:
        Cap on the primitive-pipeline length per cell. Defaults to 4 —
        enough to cover PR101 GOLD trio + 1 generic compressor.
    include_bare_substrate:
        If True, emit one cell per substrate with an empty primitive
        pipeline (baseline / no-overlay). Defaults to True.

    Returns
    -------
    list[CompositionCell]
        The enumerated cells, stable-sorted by substrate_id then by
        ``len(primitive_ids), primitive_ids``.

    Notes
    -----
    Per CLAUDE.md "Forbidden score claims": every cell carries
    ``score_claim=False``, ``promotion_eligible=False``,
    ``ready_for_exact_eval_dispatch=False``. The cell's
    ``predicted_score_delta`` is a planning prediction
    ``[predicted; substrate × primitive matrix v1]``.
    """
    if max_primitives_per_cell < 0:
        raise ValueError(
            f"max_primitives_per_cell must be >= 0; got {max_primitives_per_cell}"
        )
    s_inv = substrates if substrates is not None else canonical_substrate_inventory()
    p_inv = primitives if primitives is not None else canonical_primitive_inventory()
    if not s_inv:
        raise ValueError("substrate inventory empty; refusing to enumerate cells")

    primitives_by_id = {p.primitive_id: p for p in p_inv}

    cells: list[CompositionCell] = []

    for substrate in s_inv:
        # Identify the primitives applicable to this substrate's class.
        applicable: list[PrimitiveRow] = [
            p for p in p_inv
            if primitive_compatibility(substrate.substrate_class, p.category)
        ]

        # Optionally emit the bare-substrate baseline (empty pipeline).
        if include_bare_substrate:
            bytes_mid, score_mid = _aggregate_predicted_deltas(substrate, [])
            cells.append(
                CompositionCell(
                    cell_id=_cell_id(substrate.substrate_id, ()),
                    substrate_id=substrate.substrate_id,
                    substrate_class=substrate.substrate_class,
                    primitives=(),
                    composition_order=(),
                    predicted_bytes_delta=bytes_mid,
                    predicted_score_delta=score_mid,
                    compatibility_verdict="compatible_bare_substrate",
                    notes=(
                        f"bare substrate baseline (no primitive overlay); "
                        f"substrate.predicted_delta_alone_midpoint="
                        f"{substrate.predicted_delta_alone_midpoint():.6f}"
                    ),
                )
            )

        if max_primitives_per_cell == 0:
            continue

        # Enumerate primitive subsets up to ``max_primitives_per_cell``.
        # We iterate by combination size (1 .. cap) and filter each.
        cap = min(max_primitives_per_cell, len(applicable))
        for r in range(1, cap + 1):
            for combo in itertools.combinations(applicable, r):
                # Sort by order_index so the cross-category monotonicity
                # constraint is satisfied by construction. Stable sort
                # preserves declared within-category ordering when
                # order_index ties.
                ordered = sorted(combo, key=lambda x: (x.order_index, x.primitive_id))

                # For ORDERED_PIPELINE categories, ensure within-category
                # order matches within_category_order.
                # Group primitives by category; for each ORDERED_PIPELINE
                # category, re-sort its members by within_category_order
                # index. Then re-sort the whole pipeline by order_index
                # to preserve cross-category monotonicity.
                groups: dict[PrimitiveCategory, list[PrimitiveRow]] = {}
                for p in ordered:
                    groups.setdefault(p.category, []).append(p)
                rebuilt: list[PrimitiveRow] = []
                for p in ordered:
                    rebuilt.append(p)  # placeholder; rebuilt below
                # Build a flat reordered list:
                # - for each ORDERED_PIPELINE category, sort members by
                #   their within_category_order.
                # - all other categories keep current order.
                normalized: list[PrimitiveRow] = []
                # Track which primitives we already emitted.
                seen_ids: set[str] = set()
                for p in ordered:
                    if p.primitive_id in seen_ids:
                        continue
                    if p.order_sensitivity == PrimitiveOrderSensitivity.ORDERED_PIPELINE:
                        # Emit all category members in declared order.
                        category_members = groups.get(p.category, [])
                        wco = p.within_category_order
                        ordered_members = sorted(
                            category_members,
                            key=lambda x: wco.index(x.primitive_id) if x.primitive_id in wco else 10**6,
                        )
                        for m in ordered_members:
                            if m.primitive_id not in seen_ids:
                                normalized.append(m)
                                seen_ids.add(m.primitive_id)
                    else:
                        normalized.append(p)
                        seen_ids.add(p.primitive_id)

                # Final pass: stable-sort by order_index across categories.
                normalized.sort(key=lambda x: x.order_index)
                normalized_ids = [p.primitive_id for p in normalized]

                ok, rationale = validate_pipeline_ordering(
                    normalized_ids, primitives_by_id
                )
                if not ok:
                    # The combination violates the ordering / MX rules
                    # (e.g., two sign-encoding strategies). Emit it
                    # with a clear compatibility_verdict so the operator
                    # sees why it was rejected, but mark it un-rankable.
                    cells.append(
                        CompositionCell(
                            cell_id=_cell_id(
                                substrate.substrate_id, tuple(normalized_ids)
                            ),
                            substrate_id=substrate.substrate_id,
                            substrate_class=substrate.substrate_class,
                            primitives=tuple((pid, {}) for pid in normalized_ids),
                            composition_order=tuple(normalized_ids),
                            predicted_bytes_delta=0,
                            predicted_score_delta=0.0,
                            compatibility_verdict="violates_ordering_or_mutually_exclusive",
                            blockers=(
                                f"pipeline_ordering_invalid: {rationale}",
                            ),
                            notes=(
                                f"primitive combination {normalized_ids} "
                                f"violates ordering/MX: {rationale}"
                            ),
                        )
                    )
                    continue

                bytes_mid, score_mid = _aggregate_predicted_deltas(
                    substrate, normalized
                )

                blockers: list[str] = []
                pr101_warn = _detect_pr101_gold_overlap(substrate, normalized_ids)
                if pr101_warn is not None:
                    blockers.append(pr101_warn)

                cells.append(
                    CompositionCell(
                        cell_id=_cell_id(
                            substrate.substrate_id, tuple(normalized_ids)
                        ),
                        substrate_id=substrate.substrate_id,
                        substrate_class=substrate.substrate_class,
                        primitives=tuple((pid, {}) for pid in normalized_ids),
                        composition_order=tuple(normalized_ids),
                        predicted_bytes_delta=bytes_mid,
                        predicted_score_delta=score_mid,
                        compatibility_verdict="compatible",
                        blockers=tuple(blockers),
                        notes=(
                            f"primitive pipeline {normalized_ids} (order: by "
                            f"primitive.order_index, ties broken by primitive_id)"
                        ),
                    )
                )

    # Stable-sort the emitted cells for deterministic output.
    cells.sort(key=lambda c: (c.substrate_id, len(c.primitives), c.primitive_ids()))
    # Drop true duplicates that can arise when ordered_pipeline normalization
    # rebuilds the same pipeline from two different itertools.combinations
    # seed tuples (e.g., a combo {storage_order, byte_maps} and a combo
    # {byte_maps, storage_order} both normalize to (storage_order, byte_maps)).
    seen_cell_ids: set[str] = set()
    deduped: list[CompositionCell] = []
    for cell in cells:
        if cell.cell_id in seen_cell_ids:
            continue
        seen_cell_ids.add(cell.cell_id)
        deduped.append(cell)
    return deduped


def serialize_enumeration(
    cells: list[CompositionCell],
) -> dict[str, Any]:
    """JSON-safe serialization of enumerated cells.

    Per CLAUDE.md "Forbidden /tmp paths" the caller is responsible for
    writing the output to a durable location (``experiments/results/`` /
    ``reports/`` / ``.omx/``); this function only emits the dict.
    """
    return {
        "schema": ENUMERATION_SCHEMA,
        "registry_schema": SCHEMA_VERSION,
        "n_cells": len(cells),
        "n_compatible": sum(1 for c in cells if c.compatibility_verdict == "compatible"),
        "n_bare": sum(
            1 for c in cells if c.compatibility_verdict == "compatible_bare_substrate"
        ),
        "n_violating": sum(
            1 for c in cells
            if c.compatibility_verdict == "violates_ordering_or_mutually_exclusive"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "planning_only_substrate_primitive_matrix_v1",
        "cells": [_cell_to_dict(c) for c in cells],
        "claude_md_compliance_tags": [
            "planning_only_no_score_claim",
            "no_mps_authoritative",
            "no_tmp_paths",
            "substrate_primitive_composition_cell_registry_v1",
        ],
    }


# ── Autopilot wire-in helper ──────────────────────────────────────────────


def autopilot_ranking_input(
    *,
    only_compatible: bool = True,
    only_with_primitives: bool = False,
    substrates: Optional[list[SubstrateRow]] = None,
    primitives: Optional[list[PrimitiveRow]] = None,
    max_primitives_per_cell: int = 4,
) -> list[dict[str, Any]]:
    """Return enumerated cells as autopilot CandidateRow-compatible dicts.

    This is the canonical wire-in surface for the cathedral autopilot's
    substrate-composition source. The autopilot's existing loader at
    :func:`tools.cathedral_autopilot_autonomous_loop.load_candidates_from_substrate_composition_ranking`
    expects a JSON payload with key ``ranked_dispatches`` containing rows
    with ``candidate_id``, ``family``, ``predicted_score_delta``,
    ``expected_information_gain``, ``estimated_dispatch_cost_usd``,
    ``substrate_ids``, ``composition_notes``, ``fits_per_dispatch_cap``,
    ``fits_cumulative_envelope``.

    Per CLAUDE.md "operator_gate_non_negotiable_at_every_dispatch": this
    helper does NOT mutate the autopilot's internal config; it returns
    the typed payload the autopilot loader consumes. The autopilot's
    HALT-and-ASK pattern remains the only path that authorizes a dispatch.

    Parameters
    ----------
    only_compatible:
        Drop cells with ``compatibility_verdict != "compatible"``.
        Defaults True so the autopilot doesn't see ordering-violation
        cells as candidates.
    only_with_primitives:
        Drop bare-substrate cells (empty primitive pipeline). Defaults
        False so the autopilot ranks "substrate-alone" as a baseline
        candidate.
    substrates, primitives, max_primitives_per_cell:
        Forwarded to :func:`enumerate_cells`.
    """
    cells = enumerate_cells(
        substrates=substrates,
        primitives=primitives,
        max_primitives_per_cell=max_primitives_per_cell,
    )
    if only_compatible:
        cells = [
            c for c in cells
            if c.compatibility_verdict
            in ("compatible", "compatible_bare_substrate")
        ]
    if only_with_primitives:
        cells = [c for c in cells if c.primitives]

    out: list[dict[str, Any]] = []
    for c in cells:
        kwargs = c.autopilot_candidate_kwargs()
        # Augment with the substrate_ids field the autopilot loader expects.
        kwargs["substrate_ids"] = [c.substrate_id]
        kwargs["composition_notes"] = kwargs.pop("notes")
        # Default envelope flags True; the autopilot ranker re-evaluates
        # against its own caps. Per CLAUDE.md "operator_gate_non_negotiable":
        # this is a planning artifact, not an authorization.
        kwargs["fits_per_dispatch_cap"] = True
        kwargs["fits_cumulative_envelope"] = True
        kwargs["score_claim"] = False
        kwargs["promotion_eligible"] = False
        kwargs["ready_for_exact_eval_dispatch"] = False
        out.append(kwargs)
    return out


__all__ = [
    "ENUMERATION_SCHEMA",
    "enumerate_cells",
    "serialize_enumeration",
    "autopilot_ranking_input",
]
