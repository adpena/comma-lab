# SPDX-License-Identifier: MIT
"""Empirical per-X optimal codec planner — emits per-X codec assignment plans.

[verified-against: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md Section 5]
[verified-against: .omx/research/master_gradient_xray_fields_medal_research_wave_20260518.md Section 4.1 sensitivity_mask_aware_quantizr_v1]
[verified-against: Catalog #265 canonical-contract tokens]
[verified-against: Catalog #287 evidence-tag discipline]
[verified-against: Catalog #305 observability surface]
[verified-against: Catalog #323 canonical Provenance contract]

The per-X planner extends `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual`
from per-pair granularity to all X granularities:

    X ∈ {byte, bit, pixel, region, pair, frame, boundary, latent_index, channel, tensor, layer}

Given (X granularity, codec menu, byte budget, sensitivity threshold quantiles)
the planner emits a typed `PerXCodecAssignmentPlan` minimizing predicted contest
score subject to the byte budget.

The CANONICAL FIRST INSTANCE is `plan_per_byte_for_archive_via_sensitivity_quantiles`
which emits the `sensitivity_mask_aware_quantizr_v1` design (top 2% fp16 / next 5% int8
/ next 20% int6 / remaining 73% int4) from the Fields-Medal subagent's memo.

Public API:

- `PerXCodecAssignmentPlan`: typed plan dataclass (frozen; carries Provenance per Catalog #323)
- `PerXAssignmentRow`: single per-X assignment row
- `X_GRANULARITY_VALUES`: canonical X-granularity values
- `CODEC_NAMES`: canonical codec name registry
- `plan_per_byte_for_archive_via_sensitivity_quantiles`: per-byte strategy (THIS RELEASE)
- `plan_per_pair_via_lagrangian_dual`: per-pair strategy (delegates to master_gradient_consumers)
- `codec_bits_per_sample`: codec name → bits-per-sample lookup
"""
from __future__ import annotations

from tac.empirical_per_x_optimal_codec_planner.codec_menu import (
    CODEC_NAMES,
    codec_bits_per_sample,
)
from tac.empirical_per_x_optimal_codec_planner.contract import (
    X_GRANULARITY_VALUES,
    PerXAssignmentRow,
    PerXCodecAssignmentPlan,
    PlannerError,
)
from tac.empirical_per_x_optimal_codec_planner.per_byte_strategy import (
    plan_per_byte_for_archive_via_sensitivity_quantiles,
    plan_per_byte_from_master_gradient,
)


def plan_per_pair_via_lagrangian_dual(*args, **kwargs):
    """Delegate to canonical per-pair Lagrangian dual planner per Section 5.4.

    Wraps `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual`
    output in a `PerXCodecAssignmentPlan` with `x_granularity='pair'`.

    This is a thin convenience adapter; the canonical implementation lives in
    `tac.master_gradient_consumers`. For full Lagrangian dual control, call
    the canonical helper directly.
    """
    from tac.master_gradient_consumers import (
        per_pair_optimal_treatment_plan_via_lagrangian_dual,
    )
    return per_pair_optimal_treatment_plan_via_lagrangian_dual(*args, **kwargs)


__all__ = [
    "CODEC_NAMES",
    "PerXAssignmentRow",
    "PerXCodecAssignmentPlan",
    "PlannerError",
    "X_GRANULARITY_VALUES",
    "codec_bits_per_sample",
    "plan_per_byte_for_archive_via_sensitivity_quantiles",
    "plan_per_byte_from_master_gradient",
    "plan_per_pair_via_lagrangian_dual",
]
