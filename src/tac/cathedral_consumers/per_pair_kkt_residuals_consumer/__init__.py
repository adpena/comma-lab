# SPDX-License-Identifier: MIT
"""Cathedral consumer for `tac.master_gradient_consumers.per_pair_kkt_residuals`.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for the Cable D
per-pair KKT residuals consumer landed 2026-05-19 (commits 418698227 +
27fc83168) per `.omx/research/cable_d_master_gradient_extension_batch_landed_
20260519T055121Z.md`.

For each pair the producer computes the KKT residual
``||dD/dθ + λ_R · dR/dθ||_2`` where dD = (|g_seg| + |g_pose|) and
dR = |g_rate|. Consumes per-pair λ_R from consumer 8 (or operator-provided).
Per Cable D landing memo §"6-hook wire-in declaration": consumer 12 (KKT
residual) IS the canonical disambiguator for "is the chosen λ_R achieving
per-pair stationarity?" (hook #6 PROBE_DISAMBIGUATOR).

THIS wrapper exposes the canonical CathedralConsumerContract so the
auto-discovery loop ingests the surface. Cathedral contribution is
observability-only ([predicted] axis, promotable=False) per Catalog
#287/#323.

Hook numbers per Cable D landing memo:
- Hook #2 PARETO_CONSTRAINT (consumes consumer 8 λ_R; feeds tac.optimization.pareto)
- Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH (this wrapper)
- Hook #6 PROBE_DISAMBIGUATOR (per-pair stationarity certificate)
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "per_pair_kkt_residuals_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.PARETO_CONSTRAINT,
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.PROBE_DISAMBIGUATOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — continual-learning posterior update.

    Per-pair KKT-residual sidecar JSON is canonically persisted via
    `tac.master_gradient_consumers.consumer_output_path` at
    `.omx/state/master_gradient_consumers/`. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Per-pair KKT residual is the canonical probe-disambiguator for the
    chosen λ_R; HIGH residual = joint codec failing to balance distortion
    vs rate at that pair. Observability-only contribution per Catalog
    #287/#323.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.master_gradient_consumers.per_pair_kkt_residuals surface "
            "available (||dD + λ_R·dR||_2 certificate per pair; canonical "
            "disambiguator for per-pair stationarity; feeds tac.optimization.pareto) "
            "[predicted]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
