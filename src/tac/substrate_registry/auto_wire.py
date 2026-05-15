# SPDX-License-Identifier: MIT
"""Auto-wire query helpers for the 6 Catalog #125 hooks.

Per `.omx/research/substrate_meta_layer_design_20260515.md` § 4.1, the META
layer EXPOSES query helpers and the canonical consumer modules
(``tac.sensitivity_map``, ``tac.cost_band_calibration``, ``tac.continual_learning``,
``tools/cathedral_autopilot_autonomous_loop.py``) READ FROM them. The META
layer never writes to consumer modules — sister-subagent
``WIRE-AND-INTEGRATE-ALL`` owns the consumer-side integration.

This module is a thin read API over ``_REGISTERED_SUBSTRATES``. Each helper
returns either a list of ``SubstrateContract`` instances filtered by the
relevant hook, or a dict mapping ``substrate_id → declared_value`` for hooks
that produce per-substrate scalar metadata.
"""

from __future__ import annotations

from tac.substrate_registry.contract import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
)
from tac.substrate_registry.decorator import get_registered_substrates

__all__ = [
    "query_substrates_by_compliance_token",
    "query_substrates_for_autopilot_ranker",
    "query_substrates_for_bit_allocator_hook",
    "query_substrates_for_continual_learning_anchor_kind",
    "query_substrates_for_pareto_hook",
    "query_substrates_for_probe_disambiguators",
    "query_substrates_for_sensitivity_hook",
]


def _all() -> list[SubstrateContract]:
    """Return every registered SubstrateContract.

    Per adversarial-review finding Q2 (2026-05-15), the registry can be
    mutated out-of-band (e.g., a test that injects a fake object or a
    debugger that overwrites a row). Defensive consumers MUST NOT
    AttributeError on a corrupt row — instead they SKIP it. Operators who
    want corrupt rows surfaced should call
    ``validate_all_registered(prune_corrupt=True)`` from the decorator
    module before invoking any query helper.
    """
    return [
        c
        for c in get_registered_substrates().values()
        if isinstance(c, SubstrateContract)
    ]


def query_substrates_for_sensitivity_hook() -> list[SubstrateContract]:
    """Hook #1 (Catalog #125). Substrates that contribute to sensitivity-map.

    Excludes substrates whose ``hook_sensitivity_contribution`` is
    ``not_applicable_with_rationale`` — those are explicitly opted out.
    """
    return [
        c
        for c in _all()
        if c.hook_sensitivity_contribution != NOT_APPLICABLE_WITH_RATIONALE
    ]


def query_substrates_for_pareto_hook() -> list[SubstrateContract]:
    """Hook #2 (Catalog #125). Substrates that participate in the Pareto front.

    Returns substrates whose ``hook_pareto_constraint`` is non-N/A.
    Consumer ``tac.cost_band_calibration`` reads this list when refreshing the
    Pareto envelope.
    """
    return [
        c
        for c in _all()
        if c.hook_pareto_constraint != NOT_APPLICABLE_WITH_RATIONALE
    ]


def query_substrates_for_bit_allocator_hook() -> list[SubstrateContract]:
    """Hook #3 (Catalog #125). Substrates whose per-tensor importance feeds the
    bit-allocator.

    Returns substrates whose ``hook_bit_allocator_class`` is non-N/A.
    """
    return [
        c
        for c in _all()
        if c.hook_bit_allocator_class != NOT_APPLICABLE_WITH_RATIONALE
    ]


def query_substrates_for_autopilot_ranker() -> dict[str, str]:
    """Hook #4 (Catalog #125). Substrate id → class-shift literature token.

    Used by ``tools/cathedral_autopilot_autonomous_loop.py`` to look up the
    per-substrate class-shift token referenced in
    ``_CLASS_SHIFT_LITERATURE_TOKENS`` (e.g. ``MDL-IBPS``, ``Wyner-Ziv``,
    ``Hafner``, ``DreamerV3``).

    Substrates with ``hook_autopilot_ranker_class_shift_token=None`` are
    omitted (they are within-class baselines).
    """
    return {
        c.id: c.hook_autopilot_ranker_class_shift_token
        for c in _all()
        if c.hook_autopilot_ranker_class_shift_token is not None
    }


def query_substrates_for_continual_learning_anchor_kind() -> dict[str, str]:
    """Hook #5 (Catalog #125). Substrate id → continual-learning anchor kind.

    Used by ``tac.continual_learning.posterior_update_locked`` to know what
    custody kind to expect for empirical anchors from each substrate.

    Substrates with ``not_applicable_with_rationale`` are omitted.
    """
    return {
        c.id: c.hook_continual_learning_anchor_kind
        for c in _all()
        if c.hook_continual_learning_anchor_kind != NOT_APPLICABLE_WITH_RATIONALE
    }


def query_substrates_for_probe_disambiguators() -> dict[str, str]:
    """Hook #6 (Catalog #125). Substrate id → probe-disambiguator path.

    Used by operator workflows to discover which substrates have a registered
    probe, e.g. ``tools/probe_z3_disambiguator.py``. Substrates with
    ``hook_probe_disambiguator=None`` are omitted (rationale must be in
    ``hook_not_applicable_rationale``; that contract is enforced by the
    constructor).
    """
    return {
        c.id: c.hook_probe_disambiguator
        for c in _all()
        if c.hook_probe_disambiguator is not None
    }


def query_substrates_by_compliance_token(token: str) -> list[SubstrateContract]:
    """Return substrates that declare ``token`` in ``catalog_compliance_declarations``.

    E.g. ``query_substrates_by_compliance_token("catalog_205_select_inflate_device_used")``
    returns every substrate that explicitly honors Catalog #205. Used by
    audit tooling to find non-compliant substrates without scanning code.
    """
    return [c for c in _all() if token in c.catalog_compliance_declarations]
