# SPDX-License-Identifier: MIT
"""Score-target routing helpers for contest candidate worklists.

These helpers do not make score claims. They only keep operator worklists
focused on candidates whose declared predicted band can plausibly beat the
current score-lowering target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_SCORE_LOWERING_TARGET = 0.19

ScoreTargetStatus = Literal[
    "target_plausible",
    "above_target",
    "unknown_band",
    "invalid_band",
]


@dataclass(frozen=True)
class ScoreTargetDecision:
    """Decision for whether a candidate stays in an active score-lowering queue."""

    active: bool
    status: ScoreTargetStatus
    target_score: float
    predicted_low: float | None
    predicted_high: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "status": self.status,
            "target_score": self.target_score,
            "predicted_low": self.predicted_low,
            "predicted_high": self.predicted_high,
            "reason": self.reason,
        }


def parse_predicted_band(value: Any) -> tuple[float, float] | None:
    """Parse a predicted score band from common manifest/list formats."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        return (min(lo, hi), max(lo, hi))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[0] in "[(" and text[-1:] in "])":
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",")]
        if len(parts) != 2:
            raise ValueError(f"predicted band must contain two values, got {value!r}")
        lo, hi = float(parts[0]), float(parts[1])
        return (min(lo, hi), max(lo, hi))
    raise TypeError(f"unsupported predicted band type: {type(value).__name__}")


def decide_score_target_routing(
    predicted_band: Any,
    *,
    target_score: float = DEFAULT_SCORE_LOWERING_TARGET,
    keep_unknown: bool = True,
) -> ScoreTargetDecision:
    """Return whether a predicted band should stay active for target pursuit.

    A candidate is active when the low end of its predicted band is below the
    target. The high end may still exceed the target; that means the lane is
    risky but plausibly relevant. Bands entirely above target are retained only
    as historical/reference rows by callers that opt into showing them.
    """

    if target_score <= 0.0:
        raise ValueError("target_score must be positive")
    try:
        band = parse_predicted_band(predicted_band)
    except (TypeError, ValueError) as exc:
        return ScoreTargetDecision(
            active=keep_unknown,
            status="invalid_band",
            target_score=float(target_score),
            predicted_low=None,
            predicted_high=None,
            reason=f"invalid predicted band; {'kept' if keep_unknown else 'hidden'}: {exc}",
        )
    if band is None:
        return ScoreTargetDecision(
            active=keep_unknown,
            status="unknown_band",
            target_score=float(target_score),
            predicted_low=None,
            predicted_high=None,
            reason=f"missing predicted band; {'kept' if keep_unknown else 'hidden'}",
        )
    low, high = band
    if low < target_score:
        return ScoreTargetDecision(
            active=True,
            status="target_plausible",
            target_score=float(target_score),
            predicted_low=low,
            predicted_high=high,
            reason=(
                f"predicted low {low:.6f} is below target {target_score:.6f}; "
                "keep for exact-eval routing"
            ),
        )
    return ScoreTargetDecision(
        active=False,
        status="above_target",
        target_score=float(target_score),
        predicted_low=low,
        predicted_high=high,
        reason=(
            f"predicted band [{low:.6f}, {high:.6f}] does not beat target "
            f"{target_score:.6f}; hide from active routing"
        ),
    )
