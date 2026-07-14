# SPDX-License-Identifier: MIT
"""Pure resumable controller for one event-scoped stale-costate attempt.

The cached anchor and the later reuse attempt intentionally have different
identity types.  A reuse attempt has a changed current frame and therefore
cannot honestly supply a current exact costate hash.  It instead binds every
anchor-side hash plus the unchanged objective, scorer, and event/control scope.

No scorer, provider, trainer, or checkpoint implementation is imported here.
The caller persists the costate payload at ``AnchorIdentity.payload_path`` and
serializes :class:`ControllerState` beside the normal checkpoint.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Any

SCHEMA_VERSION = "exact_costate_reuse_state_v2"
K_MAX = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TRANSIENT_PREFIXES = (
    "/tmp",
    "/private/tmp",
    "/var/tmp",
    "/private/var/tmp",
    "/var/folders",
    "/private/var/folders",
)


class Phase(StrEnum):
    NEEDS_EXACT_ANCHOR = "needs_exact_anchor"
    REUSE_READY = "reuse_ready"


class StepAction(StrEnum):
    EXACT_ANCHOR = "exact_anchor"
    STALE_REUSE_ATTEMPT = "stale_reuse_attempt"
    FULL_TEACHER_REFRESH = "full_teacher_refresh"


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character sha256")


def _require_durable_path(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must locate a durable payload")
    path = PurePath(value)
    if ".." in path.parts:
        raise ValueError(f"{field_name} must not contain parent traversal")
    normalized = str(path)
    if any(normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in _TRANSIENT_PREFIXES):
        raise ValueError(f"{field_name} must not locate transient storage")


def _verify_payload_bytes(path_text: str, expected_sha256: str) -> None:
    """Fail closed unless the retained anchor payload still has its sealed bytes."""

    _require_durable_path(path_text, "payload_path")
    path = Path(path_text)
    if not path.is_file():
        raise ValueError("anchor payload bytes are unavailable")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("anchor payload bytes are unavailable") from exc
    if digest.hexdigest() != expected_sha256:
        raise ValueError("anchor payload sha256 mismatch")


@dataclass(frozen=True)
class AnchorIdentity:
    """Durable byte identities binding one exact costate anchor."""

    payload_path: str
    payload_sha256: str
    frame_sha256: str
    costate_sha256: str
    objective_sha256: str
    scorer_sha256: str
    control_scope_sha256: str

    def __post_init__(self) -> None:
        _require_durable_path(self.payload_path, "payload_path")
        for name in (
            "payload_sha256",
            "frame_sha256",
            "costate_sha256",
            "objective_sha256",
            "scorer_sha256",
            "control_scope_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        self.verify_payload()

    def verify_payload(self) -> None:
        """Revalidate retained bytes after construction and again before reuse."""

        _verify_payload_bytes(self.payload_path, self.payload_sha256)

    def to_dict(self) -> dict[str, str]:
        return {
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "frame_sha256": self.frame_sha256,
            "costate_sha256": self.costate_sha256,
            "objective_sha256": self.objective_sha256,
            "scorer_sha256": self.scorer_sha256,
            "control_scope_sha256": self.control_scope_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AnchorIdentity:
        if not isinstance(payload, Mapping):
            raise ValueError("anchor identity must be a mapping")
        keys = {
            "payload_path",
            "payload_sha256",
            "frame_sha256",
            "costate_sha256",
            "objective_sha256",
            "scorer_sha256",
            "control_scope_sha256",
        }
        if set(payload) != keys:
            raise ValueError("anchor identity schema mismatch")
        return cls(**{key: payload[key] for key in keys})


@dataclass(frozen=True)
class ReuseAttemptIdentity:
    """Changed-frame identity for a stale reuse attempt.

    There is deliberately no ``current_costate_sha256`` field: computing it
    would defeat the attempted backward skip.  The remaining fields bind the
    attempt to the exact anchor and unchanged control event.
    """

    current_frame_sha256: str
    anchor_payload_sha256: str
    anchor_frame_sha256: str
    anchor_costate_sha256: str
    objective_sha256: str
    scorer_sha256: str
    control_scope_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "current_frame_sha256",
            "anchor_payload_sha256",
            "anchor_frame_sha256",
            "anchor_costate_sha256",
            "objective_sha256",
            "scorer_sha256",
            "control_scope_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def matches_anchor(self, anchor: AnchorIdentity) -> bool:
        return (
            self.anchor_payload_sha256 == anchor.payload_sha256
            and self.anchor_frame_sha256 == anchor.frame_sha256
            and self.anchor_costate_sha256 == anchor.costate_sha256
            and self.objective_sha256 == anchor.objective_sha256
            and self.scorer_sha256 == anchor.scorer_sha256
            and self.control_scope_sha256 == anchor.control_scope_sha256
        )


@dataclass(frozen=True)
class GuardMetrics:
    """Exact full-facet metrics at an anchor or guarded candidate."""

    ce: float
    d_seg: float
    d_pose: float

    def __post_init__(self) -> None:
        values = (self.ce, self.d_seg, self.d_pose)
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError("guard metrics must be finite numbers")
        if any(float(value) < 0.0 for value in values):
            raise ValueError("guard metrics must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {"ce": float(self.ce), "d_seg": float(self.d_seg), "d_pose": float(self.d_pose)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GuardMetrics:
        if not isinstance(payload, Mapping) or set(payload) != {"ce", "d_seg", "d_pose"}:
            raise ValueError("guard metrics schema mismatch")
        return cls(ce=payload["ce"], d_seg=payload["d_seg"], d_pose=payload["d_pose"])


@dataclass(frozen=True)
class ControllerState:
    """Additive JSON-safe state for an event-controlled ``K_max=2`` arm."""

    phase: Phase = Phase.NEEDS_EXACT_ANCHOR
    step_index: int = 0
    rollback_latched: bool = False
    anchor: AnchorIdentity | None = None
    anchor_metrics: GuardMetrics | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if not isinstance(self.phase, Phase):
            raise ValueError("phase must be a Phase")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(self.rollback_latched, bool):
            raise ValueError("rollback_latched must be boolean")
        if (self.anchor is None) != (self.anchor_metrics is None):
            raise ValueError("anchor identity and full-facet baseline metrics must be stored together")
        if self.phase is Phase.REUSE_READY and self.anchor is None:
            raise ValueError("reuse_ready requires an exact anchor and baseline metrics")
        if self.phase is Phase.NEEDS_EXACT_ANCHOR and self.anchor is not None:
            raise ValueError("needs_exact_anchor cannot retain a stale payload")
        if self.rollback_latched and self.phase is not Phase.NEEDS_EXACT_ANCHOR:
            raise ValueError("rollback latch requires needs_exact_anchor phase")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase.value,
            "step_index": self.step_index,
            "rollback_latched": self.rollback_latched,
            "anchor": self.anchor.to_dict() if self.anchor is not None else None,
            "anchor_metrics": self.anchor_metrics.to_dict() if self.anchor_metrics is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControllerState:
        if not isinstance(payload, Mapping):
            raise ValueError("controller state must be a mapping")
        keys = {
            "schema_version",
            "phase",
            "step_index",
            "rollback_latched",
            "anchor",
            "anchor_metrics",
        }
        if set(payload) != keys:
            raise ValueError("controller state schema mismatch")
        try:
            phase = Phase(payload["phase"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown controller phase") from exc
        raw_anchor = payload["anchor"]
        raw_metrics = payload["anchor_metrics"]
        return cls(
            schema_version=payload["schema_version"],
            phase=phase,
            step_index=payload["step_index"],
            rollback_latched=payload["rollback_latched"],
            anchor=None if raw_anchor is None else AnchorIdentity.from_dict(raw_anchor),
            anchor_metrics=None if raw_metrics is None else GuardMetrics.from_dict(raw_metrics),
        )


@dataclass(frozen=True)
class StepDecision:
    action: StepAction
    reason: str
    state: ControllerState


def _latched_refresh(state: ControllerState, reason: str) -> StepDecision:
    failed = ControllerState(
        phase=Phase.NEEDS_EXACT_ANCHOR,
        step_index=state.step_index,
        rollback_latched=True,
    )
    return StepDecision(StepAction.FULL_TEACHER_REFRESH, reason, failed)


def plan_step(
    state: ControllerState,
    attempt: ReuseAttemptIdentity | None = None,
) -> StepDecision:
    """Plan one exact step or one changed-frame reuse within the same event."""

    if state.rollback_latched:
        return StepDecision(StepAction.FULL_TEACHER_REFRESH, "rollback latch is set", state)
    if state.phase is Phase.NEEDS_EXACT_ANCHOR:
        return StepDecision(
            StepAction.EXACT_ANCHOR,
            "event-controlled K_max=2 requires an exact anchor",
            state,
        )
    if attempt is None:
        return _latched_refresh(state, "reuse-attempt identity is missing")
    assert state.anchor is not None  # ControllerState invariant
    try:
        state.anchor.verify_payload()
    except ValueError as exc:
        return _latched_refresh(state, f"anchor payload custody failed: {exc}")
    if not attempt.matches_anchor(state.anchor):
        return _latched_refresh(
            state,
            "anchor payload/frame/costate/objective/scorer/control-scope identity mismatch",
        )
    if attempt.current_frame_sha256 == state.anchor.frame_sha256:
        return _latched_refresh(state, "reuse current frame is bit-identical to the anchor")
    return StepDecision(
        StepAction.STALE_REUSE_ATTEMPT,
        "changed frame is bound to the same event scope; one K_max=2 attempt allowed",
        state,
    )


def record_exact_anchor(
    state: ControllerState,
    identity: AnchorIdentity,
    metrics: GuardMetrics,
) -> ControllerState:
    """Record exact anchor payload custody and its exact full-facet baseline."""

    if state.rollback_latched:
        raise ValueError("rollback latch requires record_full_teacher_refresh")
    if state.phase is not Phase.NEEDS_EXACT_ANCHOR:
        raise ValueError("an exact anchor is only valid in needs_exact_anchor phase")
    identity.verify_payload()
    return ControllerState(
        phase=Phase.REUSE_READY,
        step_index=state.step_index + 1,
        rollback_latched=False,
        anchor=identity,
        anchor_metrics=metrics,
    )


def record_full_teacher_refresh(
    state: ControllerState,
    identity: AnchorIdentity,
    metrics: GuardMetrics,
) -> ControllerState:
    """Discharge rollback/boundary with new payload custody and exact metrics."""

    identity.verify_payload()

    return ControllerState(
        phase=Phase.REUSE_READY,
        step_index=state.step_index + 1,
        rollback_latched=False,
        anchor=identity,
        anchor_metrics=metrics,
    )


def evaluate_reuse_guard(
    state: ControllerState,
    *,
    candidate_metrics: GuardMetrics,
) -> StepDecision:
    """Accept strict CE descent plus nonworsening exact d_seg and d_pose."""

    if (
        state.phase is not Phase.REUSE_READY
        or state.rollback_latched
        or state.anchor is None
        or state.anchor_metrics is None
    ):
        raise ValueError("reuse guard requires an unlatched reuse_ready state")
    baseline = state.anchor_metrics
    accepted = (
        candidate_metrics.ce < baseline.ce
        and candidate_metrics.d_seg <= baseline.d_seg
        and candidate_metrics.d_pose <= baseline.d_pose
    )
    next_state = ControllerState(
        phase=Phase.NEEDS_EXACT_ANCHOR,
        step_index=state.step_index + 1,
        rollback_latched=not accepted,
    )
    return StepDecision(
        StepAction.EXACT_ANCHOR if accepted else StepAction.FULL_TEACHER_REFRESH,
        "full-facet guard accepted; K_max=2 consumed and next step is exact"
        if accepted
        else "full-facet guard rejected; rollback latched",
        next_state,
    )


def force_refresh_boundary(state: ControllerState, boundary: str) -> StepDecision:
    """Invalidate payload custody at every event, stage, or custody boundary."""

    if boundary not in {"event", "stage", "custody_change"}:
        raise ValueError("boundary must be event, stage, or custody_change")
    next_state = ControllerState(
        phase=Phase.NEEDS_EXACT_ANCHOR,
        step_index=state.step_index,
        rollback_latched=False,
    )
    return StepDecision(StepAction.FULL_TEACHER_REFRESH, f"forced {boundary} refresh", next_state)


__all__ = [
    "K_MAX",
    "SCHEMA_VERSION",
    "AnchorIdentity",
    "ControllerState",
    "GuardMetrics",
    "Phase",
    "ReuseAttemptIdentity",
    "StepAction",
    "StepDecision",
    "evaluate_reuse_guard",
    "force_refresh_boundary",
    "plan_step",
    "record_exact_anchor",
    "record_full_teacher_refresh",
]
