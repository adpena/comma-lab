# SPDX-License-Identifier: MIT
"""Typed contract for the canonical task-status append-only ledger."""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "canonical_task_status_v1_20260518"

Status = Literal["pending", "in_progress", "completed", "blocked", "deferred", "cancelled"]
Owner = Literal["claude", "codex", "operator"] | str
TestStatus = Literal["green", "red", "n_a", "pending"]
EventType = Literal["registered", "status_change", "note", "completion", "blocked", "cancelled"]

VALID_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed", "blocked", "deferred", "cancelled"}
)
VALID_TEST_STATUSES: frozenset[str] = frozenset({"green", "red", "n_a", "pending"})
VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {"registered", "status_change", "note", "completion", "blocked", "cancelled"}
)
VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"in_progress", "blocked", "cancelled"}),
    "in_progress": frozenset({"completed", "blocked", "cancelled"}),
    "blocked": frozenset({"in_progress", "deferred", "cancelled"}),
    "deferred": frozenset({"pending", "cancelled"}),
    "completed": frozenset(),
    "cancelled": frozenset(),
}


class CanonicalTaskStatusError(RuntimeError):
    """Base error for canonical task-status failures."""


class CanonicalTaskStatusCorruptError(CanonicalTaskStatusError):
    """Raised when the append-only JSONL ledger cannot be parsed strictly."""


class CanonicalTaskStatusInvalidTransitionError(CanonicalTaskStatusError):
    """Raised when a status transition violates the canonical state machine."""


def _validate_utc_iso(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} is required")
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = _dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != _dt.timedelta(0):
        raise ValueError(f"{field_name} must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class CanonicalTaskStatusRow:
    """One append-only event row in `.omx/state/canonical_task_status.jsonl`."""

    task_id: str
    source_design_memo: str
    title: str
    status: str
    owner: str
    event_type: str
    event_timestamp_utc: str
    event_actor: str
    written_at_utc: str
    written_pid: int
    written_host: str
    schema_version: str = SCHEMA_VERSION
    predicted_cost_usd: float | None = None
    predicted_delta_s_band: tuple[float, float] | None = None
    actual_delta_s: float | None = None
    commit_shas: tuple[str, ...] = field(default_factory=tuple)
    test_status: str = "pending"
    blockers: tuple[str, ...] = field(default_factory=tuple)
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    event_notes: str = ""
    session_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version!r}")
        for field_name in (
            "task_id",
            "source_design_memo",
            "title",
            "owner",
            "event_actor",
            "session_id",
            "written_host",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} is required")
        _validate_utc_iso(self.event_timestamp_utc, "event_timestamp_utc")
        _validate_utc_iso(self.written_at_utc, "written_at_utc")
        if self.started_at_utc is not None:
            _validate_utc_iso(self.started_at_utc, "started_at_utc")
        if self.completed_at_utc is not None:
            _validate_utc_iso(self.completed_at_utc, "completed_at_utc")
        if self.written_pid <= 0:
            raise ValueError("written_pid must be positive")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"invalid event_type: {self.event_type!r}")
        if self.test_status not in VALID_TEST_STATUSES:
            raise ValueError(f"invalid test_status: {self.test_status!r}")
        if self.predicted_cost_usd is not None and float(self.predicted_cost_usd) < 0.0:
            raise ValueError("predicted_cost_usd must be non-negative")
        if self.predicted_delta_s_band is not None:
            lo, hi = self.predicted_delta_s_band
            if float(lo) > float(hi):
                raise ValueError("predicted_delta_s_band lower bound exceeds upper bound")
        if self.actual_delta_s is not None and "[empirical:" not in self.event_notes:
            raise ValueError("actual_delta_s rows must include an [empirical:<path>] event note")
        object.__setattr__(self, "commit_shas", tuple(str(v) for v in self.commit_shas))
        object.__setattr__(self, "blockers", tuple(str(v) for v in self.blockers))
        if self.predicted_delta_s_band is not None:
            lo, hi = self.predicted_delta_s_band
            object.__setattr__(self, "predicted_delta_s_band", (float(lo), float(hi)))

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, Any]) -> CanonicalTaskStatusRow:
        band = obj.get("predicted_delta_s_band")
        parsed_band: tuple[float, float] | None
        if band is None:
            parsed_band = None
        elif isinstance(band, Sequence) and not isinstance(band, str) and len(band) == 2:
            parsed_band = (float(band[0]), float(band[1]))
        else:
            raise ValueError("predicted_delta_s_band must be null or a two-element sequence")
        return cls(
            schema_version=str(obj.get("schema_version", "")),
            task_id=str(obj.get("task_id", "")),
            source_design_memo=str(obj.get("source_design_memo", "")),
            title=str(obj.get("title", "")),
            status=str(obj.get("status", "")),
            owner=str(obj.get("owner", "")),
            predicted_cost_usd=(
                None if obj.get("predicted_cost_usd") is None else float(obj["predicted_cost_usd"])
            ),
            predicted_delta_s_band=parsed_band,
            actual_delta_s=None if obj.get("actual_delta_s") is None else float(obj["actual_delta_s"]),
            commit_shas=tuple(str(v) for v in obj.get("commit_shas", ())),
            test_status=str(obj.get("test_status", "pending")),
            blockers=tuple(str(v) for v in obj.get("blockers", ())),
            started_at_utc=obj.get("started_at_utc"),
            completed_at_utc=obj.get("completed_at_utc"),
            event_type=str(obj.get("event_type", "")),
            event_timestamp_utc=str(obj.get("event_timestamp_utc", "")),
            event_actor=str(obj.get("event_actor", "")),
            event_notes=str(obj.get("event_notes", "")),
            session_id=str(obj.get("session_id", "")),
            written_at_utc=str(obj.get("written_at_utc", "")),
            written_pid=int(obj.get("written_pid", 0)),
            written_host=str(obj.get("written_host", "")),
        )

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "source_design_memo": self.source_design_memo,
            "title": self.title,
            "status": self.status,
            "owner": self.owner,
            "predicted_cost_usd": self.predicted_cost_usd,
            "predicted_delta_s_band": (
                None if self.predicted_delta_s_band is None else list(self.predicted_delta_s_band)
            ),
            "actual_delta_s": self.actual_delta_s,
            "commit_shas": list(self.commit_shas),
            "test_status": self.test_status,
            "blockers": list(self.blockers),
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "event_type": self.event_type,
            "event_timestamp_utc": self.event_timestamp_utc,
            "event_actor": self.event_actor,
            "event_notes": self.event_notes,
            "session_id": self.session_id,
            "written_at_utc": self.written_at_utc,
            "written_pid": self.written_pid,
            "written_host": self.written_host,
        }


def task_id_for_memo_item(source_design_memo: str | Path, item_id: str) -> str:
    """Return the stable canonical task id for a directive item."""

    memo = Path(source_design_memo).name
    return f"{Path(memo).stem}::{item_id}"
