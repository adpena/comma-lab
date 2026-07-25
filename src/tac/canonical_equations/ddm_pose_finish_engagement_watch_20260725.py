# SPDX-License-Identifier: MIT
"""Derived watch window for the DDM pose-finish engagement gate.

This module records the sealed scheduling identity without claiming that a
plateau exists.  The watch becomes eligible only when exact n600 verdicts also
satisfy the trainer's segmentation-plateau predicate.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "ddm_pose_finish_engagement_watch.v1"
LAW_ID = "ddm_pose_finish_engagement_window_20260725"


def derive_pose_finish_engagement_watch(
    *,
    verdict_interval_steps: int,
    ema_span: int,
    hysteresis: int,
    settle_window: int,
    observed_exact_verdicts: int,
    fallback_score_contribution: float,
) -> dict[str, Any]:
    """Return the one-based conditional engagement window.

    A rolling EMA first has enough support at ``ema_span`` verdicts.  Requiring
    ``hysteresis`` confirmations makes the first candidate verdict
    ``ema_span + hysteresis - 1``.  The independent settle window closes one
    verdict later for the sealed 3/3/3 gate:
    ``max(candidate, ema_span + settle_window)``.
    """

    integers = {
        "verdict_interval_steps": verdict_interval_steps,
        "ema_span": ema_span,
        "hysteresis": hysteresis,
        "settle_window": settle_window,
        "observed_exact_verdicts": observed_exact_verdicts,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers.values()):
        raise TypeError("pose-finish watch inputs must be exact integers")
    if verdict_interval_steps <= 0 or min(ema_span, hysteresis, settle_window) <= 0:
        raise ValueError("pose-finish gate lengths must be positive")
    if observed_exact_verdicts < 0:
        raise ValueError("observed_exact_verdicts must be nonnegative")
    fallback = float(fallback_score_contribution)
    if not math.isfinite(fallback) or fallback < 0.0:
        raise ValueError("fallback_score_contribution must be finite and nonnegative")

    candidate_verdict = ema_span + hysteresis - 1
    settled_verdict = max(candidate_verdict, ema_span + settle_window)
    if observed_exact_verdicts < candidate_verdict:
        classification = "PRE_CONDITIONAL_ENGAGEMENT_WINDOW"
    elif observed_exact_verdicts < settled_verdict:
        classification = "CONDITIONAL_CANDIDATE_WINDOW"
    else:
        classification = "CONDITIONAL_SETTLED_WINDOW_OPEN"

    return {
        "schema": SCHEMA,
        "law_id": LAW_ID,
        "epistemic_status": "DERIVED_FROM_SEALED_GATE_CONSTANTS",
        "classification": classification,
        "conditional_on_exact_n600_seg_plateau": True,
        "observed_exact_verdicts": observed_exact_verdicts,
        "candidate_engagement_verdict_index_one_based": candidate_verdict,
        "settled_engagement_verdict_index_one_based": settled_verdict,
        "candidate_engagement_global_step": candidate_verdict * verdict_interval_steps,
        "settled_engagement_global_step": settled_verdict * verdict_interval_steps,
        "sealed_gate": {
            "verdict_interval_steps": verdict_interval_steps,
            "ema_span": ema_span,
            "hysteresis": hysteresis,
            "settle_window": settle_window,
        },
        "fallback": {
            "status": "MEASURED_BANKED_COMPARATOR_NON_PROMOTING",
            "score_contribution": fallback,
            "role": "c1_budget_and_harvest_signal_only",
        },
        "law": (
            "candidate_verdict = ema_span + hysteresis - 1; "
            "settled_verdict = max(candidate_verdict, ema_span + settle_window); "
            "global_step = verdict_index * verdict_interval_steps"
        ),
        "actuation": "NONE",
        "score_claim": False,
    }
