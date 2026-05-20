# SPDX-License-Identifier: MIT
# CATHEDRAL_CONSUMER_DEFERRED_OK:axis_tag_is_intentionally_[MPS-PROXY]_per_CLAUDE_md_MPS_auth_eval_is_noise_rule_3_and_forbidden_MPS_derived_strategic_decision_falsification_trap_required_for_downstream_mps_viable_prescreen_consumer_routing_disambiguator_per_slot_II_classification_memo_catalog_341_noncompliance_classification_20260519
"""Cathedral consumer for ``tac.mps_diagnostic`` layerwise MPS drift evidence.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.mps_diagnostic`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.mps_diagnostic`` exposes the canonical layerwise MPS-vs-CPU/CUDA drift
measurement helpers (LayerDriftRecord / measure_layerwise_drift /
identify_drift_cliff_layer / emit_drift_table_markdown). Per CLAUDE.md
"MPS auth eval is NOISE" non-negotiable: MPS results are NEVER score truth.
This consumer is OBSERVABILITY-only — MPS drift evidence is a diagnostic
signal for understanding architecture/runtime mismatch; it does NOT and
cannot adjust a candidate's predicted_score_delta.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "mps_diagnostic_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. MPS drift is diagnostic, never
    posterior-promoted per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. MPS drift is ``[MPS-PROXY]``
    axis per CLAUDE.md "Forbidden MPS-derived strategic decision" + cannot
    be promoted to any contest axis.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.mps_diagnostic canonical layerwise MPS-vs-CPU/CUDA drift "
            "measurement helpers available (LayerDriftRecord / "
            "measure_layerwise_drift / identify_drift_cliff_layer / "
            "emit_drift_table_markdown) — DIAGNOSTIC ONLY [MPS-PROXY]"
        ),
        "axis_tag": "[MPS-PROXY]",
        "promotable": False,
        "confidence": 0.0,
    }
