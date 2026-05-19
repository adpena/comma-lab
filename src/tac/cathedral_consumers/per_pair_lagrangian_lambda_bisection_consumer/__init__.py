# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_lagrangian_lambda_bisection`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair Lagrangian λ_R bisection consumer landed 2026-05-19 (commits
418698227 + 27fc83168) per `.omx/research/cable_d_master_gradient_extension_
batch_landed_20260519T055121Z.md`.

The producer fits per-pair λ_R via OLS slope on the per-pair Pareto envelope
(consumer 7). Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable
the per-pair λ_R IS the canonical disambiguator for per-pair rate
allocation; it feeds `tac.optimization.pareto` via per-pair lambda emission.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Per-pair λ_R is DIAGNOSTIC per
CLAUDE.md "Apples-to-apples evidence discipline"; cathedral contribution
is observability-only ([predicted] axis, promotable=False) until paired
empirical anchor lands per Catalog #127.

Hook numbers per Cable D landing memo:
- Hook #1 SENSITIVITY_MAP (per-pair λ_R feeds tac.sensitivity_map.axis_level_reweight)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)

Sister of `per_pair_pareto_envelope_consumer` (producer-chain upstream) +
`per_pair_kkt_residuals_consumer` (producer-chain downstream).
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_lagrangian_lambda_bisection_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-pair λ_R sidecar JSON is canonically persisted via
    `tac.master_gradient_consumers.consumer_output_path` at
    `.omx/state/master_gradient_consumers/`. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns zero-adjustment observability annotation citing per-pair λ_R
    surface. [predicted] axis per CLAUDE.md "Apples-to-apples evidence
    discipline"; promotable=False per Catalog #287/#323.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.per_pair_lagrangian_lambda_bisection "
            "surface available (per-pair λ_R from OLS slope on Pareto envelope; "
            "canonical disambiguator for per-pair rate allocation) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
