# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_volterra_cross_terms`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair Volterra second-order cross-terms consumer landed 2026-05-19
(commits 418698227 + 27fc83168) per `.omx/research/cable_d_master_gradient_
extension_batch_landed_20260519T055121Z.md`.

For each pair-pair (i, j) the producer computes the cosine-style coupling
score over the per-pair distortion gradient profiles. Score in [0, 1];
1 = identical byte-leverage profile (high coupling); 0 = orthogonal
byte-leverage. Volterra cross-terms feed `tac.sensitivity_map.*` for
second-order sensitivity analysis + `tac.optimization.pareto` for
pair-pair-coupled constraint emission.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Hook numbers per Cable D landing memo:
- Hook #1 SENSITIVITY_MAP (second-order Volterra kernel)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
- Hook #2 PARETO_CONSTRAINT (pair-pair coupling feeds tac.optimization.pareto)

Per Cable D's HARD-EARNED audit: Volterra coupling = cosine similarity on
distortion profiles is the canonical second-order kernel form.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_volterra_cross_terms_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.SENSITIVITY_MAP,
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Volterra cross-terms sidecar JSON is canonically persisted via
    `tac.master_gradient_consumers.consumer_output_path` at
    `.omx/state/master_gradient_consumers/`. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution."""
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.per_pair_volterra_cross_terms "
            "surface available (cosine coupling on per-pair distortion gradient "
            "profiles; second-order Volterra kernel; feeds tac.sensitivity_map + "
            "tac.optimization.pareto) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
