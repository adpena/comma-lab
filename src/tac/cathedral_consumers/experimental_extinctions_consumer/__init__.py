# SPDX-License-Identifier: MIT
"""Cathedral consumer for ``tac.experimental_extinctions`` empirical sweeps.

Per Catalog #335 + tac.cathedral.consumer_contract.CathedralConsumerContract.
Wires the orphan-signal-at-cathedral-autopilot bug class for
``tac.experimental_extinctions`` per wiring + integration audit 2026-05-19
(commit 3821cfb6b).

``tac.experimental_extinctions`` is the canonical helper for empirically-
calibrated sweep extinctions (convergence-aware early-stopping / brotli
quality / codec choice / SegNet boundary sigma / council cadence /
probe-outcome staleness / negation-window FP/FN). These results are
``[empirical]`` axis (per-sweep) but consumer-side they remain non-promotable
without paired-contest-axis verification per CLAUDE.md "Apples-to-apples
evidence discipline" + Catalog #323 canonical Provenance.
"""
from __future__ import annotations

from typing import Any, Mapping

from tac.cathedral.consumer_contract import HookNumber


CONSUMER_NAME = "experimental_extinctions_consumer"
CONSUMER_VERSION = "0.1.0"
CONSUMER_HOOK_NUMBERS = (
    HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    HookNumber.CONTINUAL_LEARNING_POSTERIOR,
)


def update_from_anchor(anchor: Any) -> None:
    """Catalog #125 hook #5 — empirical sweep results emit canonical
    posterior anchors via existing fcntl-locked helpers; no additional
    posterior update required here. NO-OP by design.
    """
    _ = anchor


def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    """Catalog #125 hook #4 — cathedral autopilot ranker contribution.

    Zero-adjustment observability annotation citing canonical empirical-
    sweep surface. Per-candidate sweep verdicts remain non-promotable
    without paired-axis verification.
    """
    _ = candidate
    return {
        "predicted_delta_adjustment": 0.0,
        "rationale": (
            "tac.experimental_extinctions canonical empirical sweep helpers "
            "available (convergence / brotli / codec / sigma / cadence / "
            "probe-decay / negation-window) [empirical-per-sweep, "
            "non-promotable without paired-axis verification]"
        ),
        "axis_tag": "[predicted]",
        "promotable": False,
        "confidence": 0.0,
    }
