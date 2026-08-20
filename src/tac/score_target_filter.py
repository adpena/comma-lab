# SPDX-License-Identifier: MIT
"""Score-target routing helpers for contest candidate worklists.

These helpers do not make score claims. They only keep operator worklists
focused on candidates whose declared predicted band can plausibly beat the
current score-lowering target. The target is always loaded from, or verified
against, this checkout's canonical frontier pointer. Numeric target overrides
and arbitrary pointer paths are deliberately not part of the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tac.canonical_frontier_pointer import CANONICAL_FRONTIER_POINTER_PATH
from tac.witness_dsl.dynamic_frontier_target import (
    DynamicFrontierTargetError,
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    verify_dynamic_frontier_target_snapshot,
)

_DYNAMIC_TARGET_REPO_ROOT = Path(__file__).resolve().parents[2]

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
    target_pointer_path: str
    target_pointer_sha256: str
    target_last_refreshed_utc: str
    target_selected_axis: str
    target_selected_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "status": self.status,
            "target_score": self.target_score,
            "predicted_low": self.predicted_low,
            "predicted_high": self.predicted_high,
            "reason": self.reason,
            "target_pointer_path": self.target_pointer_path,
            "target_pointer_sha256": self.target_pointer_sha256,
            "target_last_refreshed_utc": self.target_last_refreshed_utc,
            "target_selected_axis": self.target_selected_axis,
            "target_selected_source": self.target_selected_source,
        }


def _expected_pointer_path() -> str:
    return os.path.abspath(os.fspath(_DYNAMIC_TARGET_REPO_ROOT / CANONICAL_FRONTIER_POINTER_PATH))


def _require_canonical_snapshot(
    snapshot: DynamicFrontierTargetSnapshot,
    *,
    now_utc_iso: str | None,
) -> DynamicFrontierTargetSnapshot:
    if not isinstance(snapshot, DynamicFrontierTargetSnapshot):
        raise TypeError("target_snapshot must be a DynamicFrontierTargetSnapshot")
    if snapshot.pointer_path != _expected_pointer_path():
        raise DynamicFrontierTargetError(
            "score routing refuses a snapshot from a noncanonical pointer path"
        )
    return verify_dynamic_frontier_target_snapshot(snapshot, now_utc_iso=now_utc_iso)


def load_score_target_snapshot(
    *,
    now_utc_iso: str | None = None,
) -> DynamicFrontierTargetSnapshot:
    """Load this checkout's canonical target without accepting a path input."""

    snapshot = load_dynamic_frontier_target(
        repo_root=_DYNAMIC_TARGET_REPO_ROOT,
        now_utc_iso=now_utc_iso,
    )
    return _require_canonical_snapshot(snapshot, now_utc_iso=now_utc_iso)


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
    target_snapshot: DynamicFrontierTargetSnapshot | None = None,
    keep_unknown: bool = True,
    now_utc_iso: str | None = None,
) -> ScoreTargetDecision:
    """Return whether a predicted band should stay active for target pursuit.

    A candidate is active when the low end of its predicted band is below the
    target. The high end may still exceed the target; that means the lane is
    risky but plausibly relevant. Bands entirely above target are retained only
    as historical/reference rows by callers that opt into showing them.
    """

    snapshot = (
        load_score_target_snapshot(now_utc_iso=now_utc_iso)
        if target_snapshot is None
        else _require_canonical_snapshot(target_snapshot, now_utc_iso=now_utc_iso)
    )
    target_score = snapshot.target_score
    custody = {
        "target_pointer_path": snapshot.pointer_path,
        "target_pointer_sha256": snapshot.pointer_sha256,
        "target_last_refreshed_utc": snapshot.last_refreshed_utc,
        "target_selected_axis": snapshot.selected_axis,
        "target_selected_source": snapshot.selected_source,
    }
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
            **custody,
        )
    if band is None:
        return ScoreTargetDecision(
            active=keep_unknown,
            status="unknown_band",
            target_score=float(target_score),
            predicted_low=None,
            predicted_high=None,
            reason=f"missing predicted band; {'kept' if keep_unknown else 'hidden'}",
            **custody,
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
            **custody,
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
        **custody,
    )
