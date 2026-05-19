# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.mps_gap_experiment`` MPS gap quantification.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.mps_gap_experiment`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.mps_gap_experiment`` exposes canonical MPS gap quantification helpers
(TinyRenderer / build_tiny_renderer / train_on_mps_real_frames /
GapManifest / classify_verdict / compute_gap_components). Per CLAUDE.md
"MPS auth eval is NOISE": this is OBSERVABILITY-only diagnostic; it does
NOT and cannot adjust a candidate's predicted_score_delta. Active sister
subagent ``mps_phase_b_fire_and_harvest_20260519`` owns the per-experiment
verdict harvest path.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "mps_gap_experiment_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — NO-OP. MPS gap experiments are research
    signal only per CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation. MPS gap diagnostic is
    research signal only; cannot be promoted to any contest axis.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.mps_gap_experiment canonical MPS gap quantification helpers "
            "available (TinyRenderer / build_tiny_renderer / "
            "train_on_mps_real_frames / GapManifest / classify_verdict / "
            "compute_gap_components) — DIAGNOSTIC ONLY [MPS-PROXY]"
        ),
        "axis_tag": "[MPS-PROXY]",
        "promotable": False,
        "confidence": 0.0,
    }
