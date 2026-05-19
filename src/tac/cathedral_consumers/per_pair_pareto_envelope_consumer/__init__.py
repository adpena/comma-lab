# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_pareto_envelope`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair Pareto envelope consumer landed 2026-05-19 (commits 418698227 +
27fc83168) per `.omx/research/cable_d_master_gradient_extension_batch_landed_
20260519T055121Z.md`.

The producer (`tac.master_gradient_consumers.per_pair_pareto_envelope`)
sweeps bytes ordered by descending rate-gradient magnitude and emits a
per-pair Pareto envelope (cumulative_bytes vs cumulative_distortion). The
envelope feeds `tac.optimization.pareto` via per-pair constraint emission.

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. The producer's output is
DIAGNOSTIC per CLAUDE.md "Apples-to-apples evidence discipline"; the
cathedral contribution is observability-only ([predicted] axis,
promotable=False) until paired empirical anchor lands per Catalog #127.

Hook numbers per Cable D landing memo:
- Hook #2 PARETO_CONSTRAINT (consumed by tac.optimization.pareto)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)

Sister of:
- `atom_consumer` (canonical pattern reference)
- `per_pair_lagrangian_lambda_bisection_consumer` (composable producer chain)
- `per_pair_kkt_residuals_consumer` (consumes lambda from #8 → KKT)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_pareto_envelope_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Reference: per-pair Pareto envelopes are already persisted via the
    canonical `tac.master_gradient_consumers.consumer_output_path` sidecar
    JSON writer at `.omx/state/master_gradient_consumers/`; no additional
    posterior update is required here. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Returns zero-adjustment observability annotation citing the per-pair
    Pareto envelope surface for the candidate's archive. No score
    adjustment — per-pair Pareto envelopes are DIAGNOSTIC [predicted] axis
    per CLAUDE.md "Apples-to-apples evidence discipline".

    Per Catalog #287/#323: contribution is observability-only
    (promotable=False) until a paired empirical anchor demonstrates the
    envelope's predictive power on a contest archive.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.per_pair_pareto_envelope surface "
            "available (per-pair byte-Pareto sweep for tac.optimization.pareto "
            "per-pair constraint emission) [predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
