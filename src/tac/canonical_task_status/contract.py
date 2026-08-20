# SPDX-License-Identifier: MIT
"""Typed contract for the canonical task-status append-only ledger."""

from __future__ import annotations

import datetime as _dt
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "canonical_task_status_v1_20260518"

# ---------------------------------------------------------------------------
# ΔS CUSTODY (ddm_op3, 2026-08-03) -- a delta is only meaningful WITH its arguments.
# ---------------------------------------------------------------------------
# MEASURED, 417 ledger rows: 8 carry a typed ``actual_delta_s``, 8 carry a ΔS-shaped
# claim only in free text, 1 overlaps -> 15 distinct rows assert a ΔS, and the existing
# ``[empirical:]`` invariant is able to see exactly ONE of them. The two rows that
# actually misdirected readers this week (a "-0.0983195" and a "-0.0866789 S UNLOCK")
# both carry ``actual_delta_s = None`` and state their number in ``title``.
#
# That is why this is not a one-line addition to the field check: the number moved OUT
# of the field the schema polices and into the prose it does not. Guarding only the
# typed field would have produced a gate that passes both live failures.
#
# Three coordinates are required, because three were separately lost:
#   baseline  -- our own frontier moved SIX times in one day (0.9639878 -> 0.8264972),
#                so a delta with no reference is not stale-ish, it is undefined.
#   term_set  -- a delta over a SUBSET of {seg, pose, rate} is not a ΔS at all. The
#                "-0.0866" above is seg+rate only; its real composed row is +19.22,
#                because the omitted pose term is 234.7x the advertised prize.
#   population -- a prefix of a temporally-correlated list is a scene block, not a
#                sample: n=73 read -0.122 WIN where its own n600 read +0.152 LOSS.

BASELINE_TOKEN = re.compile(r"\[baseline:[^\]=]+=[^\]]+\]")
PARTIAL_TERMS_TOKEN = re.compile(r"\[partial:[^\]]+\]")
POPULATION_TOKEN = re.compile(r"\[n=\d+\]|\[n600\]")

# A ΔS-shaped assertion in free text: a signed 3+-decimal magnitude adjacent to an S
# label. Deliberately narrow -- it matched 8 of 417 rows (1.9%), so it is a claim
# detector, not a number detector.
FREE_TEXT_DELTA_S = re.compile(
    r"[-+−]\s?\d*\.\d{3,}\s*S\b"
    r"|\bd(?:elta)?[_ ]?S\s*[=:]\s*[-+−]?\d*\.\d{3,}"
    r"|ΔS\s*[=:]?\s*[-+−]?\d*\.\d{3,}"
)

# Prose that marks a delta as covering only part of S.
PARTIAL_COMPOSITE_PROSE = re.compile(
    r"seg[_ +]?(?:plus[_ ]?)?rate|seg[-_ ]only|pose[-_ ]only|rate[-_ ]only|seg\+rate|S_add",
    re.IGNORECASE,
)

# An explicit reference COMPARISON in prose ("... vs ref 0.7685479 (-0.0983195) ...").
#
# This pattern exists because of a correction to this module's own first draft. The row
# that misdirected hardest states its delta as a bare parenthesised number with no S
# label at all, so the claim detector above does NOT see it -- and a control that
# "replayed" it with an S appended would have been testing an edited row, not the real
# one. MEASURED: 2 of 417 rows (0.5%), both genuine ΔS claims, so this is specific.
#
# It also sharpens what the rule is actually for. That row DID name its reference. The
# defect was that it named a bare NUMBER -- 0.7685479 -- which no reader can date,
# locate, or re-derive, so nobody could see it had been superseded five times. Naming a
# baseline is necessary; naming a RE-DERIVABLE one is the requirement.
BARE_REFERENCE_PROSE = re.compile(
    r"\bvs\.?\s+(?:ref|reference|baseline|base)\b"
    r"|\bagainst\s+(?:ref|reference|baseline)\b"
    r"|\bbaseline\s*[=:]\s*\d",
    re.IGNORECASE,
)


def delta_s_custody_findings(
    *,
    actual_delta_s: float | None,
    event_notes: str,
    title: str = "",
) -> tuple[str, ...]:
    """Return the ΔS-custody defects of a row. Empty tuple == custodied.

    ONE implementation, TWO policies: the reader WARNS on these findings so the
    append-only ledger stays totally readable (a historical row cannot be rewritten, and
    a reader that raises on history is worse than the defect it reports), while the
    writer REFUSES them so no new uncustodied delta can be minted. That split is the
    same one already used for malformed ``event_notes`` in this module.
    """
    text = f"{title}\n{event_notes}"
    claims_in_free_text = bool(FREE_TEXT_DELTA_S.search(text))
    compares_to_reference = bool(BARE_REFERENCE_PROSE.search(text))
    asserts_delta = actual_delta_s is not None or claims_in_free_text or compares_to_reference
    if not asserts_delta:
        return ()

    findings: list[str] = []
    has_partial_token = bool(PARTIAL_TERMS_TOKEN.search(text))
    reads_partial = bool(PARTIAL_COMPOSITE_PROSE.search(text))

    if not BASELINE_TOKEN.search(text):
        detail = (
            "a bare reference NUMBER is not a baseline -- nobody can date, locate or "
            "re-derive it, which is exactly how a reference that had been superseded "
            "five times kept being divided by"
            if compares_to_reference
            else "our own frontier moved six times in one day"
        )
        findings.append(
            "MISSING_BASELINE: a ΔS asserts a difference, so it is undefined without the "
            "row it was measured against. Add [baseline:<artifact-locator>=<S recomputed "
            f"from components>] -- {detail}"
        )
    if reads_partial and not has_partial_token:
        findings.append(
            "UNDECLARED_PARTIAL_COMPOSITE: this row reads as a delta over a SUBSET of "
            "{seg, pose, rate}, which is not a ΔS. Declare it as [partial:seg+rate] so a "
            "downstream reader cannot drop the qualifier -- one such row advertised "
            "-0.0866 while its composed row measured +19.22"
        )
    if has_partial_token and actual_delta_s is not None:
        findings.append(
            "PARTIAL_IN_TYPED_FIELD: actual_delta_s is the FULL-S delta; a partial "
            "composite must stay in the notes and leave the typed field None"
        )
    if not POPULATION_TOKEN.search(text):
        findings.append(
            "MISSING_POPULATION: state [n600] or [n=<k>] -- a subset of a temporally "
            "correlated video list is a different population, not a smaller sample of "
            "the same one"
        )
    return tuple(findings)

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


def _coerce_event_notes(value: object) -> str:
    """Read-side normalisation of a historical ``event_notes`` value. LOUD, never silent.

    The ledger is APPEND-ONLY (Catalog #110/#113), so a historical row cannot be
    rewritten in place and the reader must stay total. But the previous
    ``str(obj.get("event_notes", ""))`` turned a list into the literal
    ``"['FINAL race verdict...']"`` -- brackets, quotes and all -- and never raised.
    MEASURED 2026-08-01: task 793 (written 2026-07-31T09:20:59Z) carried a mangled
    verdict string through every canonical read since. A crash announces itself; that
    did not. So: join sanely, and WARN so the row gets corrected by append.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        coerced = "; ".join(str(item) for item in value)
    else:
        coerced = str(value)
    warnings.warn(
        f"canonical_task_status: event_notes was {type(value).__name__}, not str -- "
        f"joined for readability; the SOURCE ROW is malformed and must be corrected "
        f"by appending a fixed row (the ledger is append-only)",
        stacklevel=2,
    )
    return coerced


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
        # Free-text fields are ANNOTATED ``str`` but annotations do not enforce.
        # MEASURED 2026-08-01: a list reached event_notes (task 793) and rode the
        # ledger unchallenged, crashing graph-memory recall campaign-wide. Fail CLOSED
        # at construction so no NEW malformed row can ever be written.
        for _name in ("task_id", "title", "status", "owner", "event_notes"):
            _value = getattr(self, _name)
            if not isinstance(_value, str):
                raise TypeError(
                    f"{_name} must be str, got {type(_value).__name__}: {_value!r}"
                )
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
        # ΔS custody: WARN here, REFUSE at the writer. The ledger is append-only, so a
        # reader must stay total over history (see _coerce_event_notes); raising would
        # break campaign-wide recall on 15 legitimately historical rows. The findings are
        # still surfaced so staleness is DETECTABLE rather than discoverable.
        custody = delta_s_custody_findings(
            actual_delta_s=self.actual_delta_s,
            event_notes=self.event_notes,
            title=self.title,
        )
        if custody:
            warnings.warn(
                f"canonical_task_status: task {self.task_id} asserts a ΔS without full "
                f"custody: {'; '.join(custody)}",
                stacklevel=2,
            )
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
            event_notes=_coerce_event_notes(obj.get("event_notes")),
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
