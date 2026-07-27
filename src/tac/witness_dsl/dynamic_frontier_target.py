"""Fail-closed live competitive-target consumption for witness planning.

This module is deliberately read-only.  It turns the canonical frontier
pointer into a source-bound planning snapshot and delegates all score geometry
to :mod:`tac.score_geometry`.  It neither refreshes nor writes the pointer and
does not establish score, evaluation, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from tac import score_geometry
from tac.canonical_frontier_pointer import (
    CANONICAL_FRONTIER_POINTER_PATH,
    POINTER_SCHEMA_VERSION,
    POINTER_STALE_SECONDS,
    CanonicalFrontierPointer,
    effective_frontier_score,
    recompute_effective_frontier,
)

_MAX_POINTER_BYTES: Final = 8 * 1024 * 1024


class DynamicFrontierTargetError(RuntimeError):
    """The canonical competitive target could not be consumed safely."""


@dataclass(frozen=True)
class DynamicFrontierTargetSnapshot:
    """A byte-identity-bound, derived competitive target.

    ``target_score`` is recomputed from custody-bearing constituent rows.  The
    serialized ``effective_frontier`` cache is intentionally not represented
    as an authority field.
    """

    pointer_path: str
    pointer_bytes: int
    pointer_sha256: str
    pointer_device: int
    pointer_inode: int
    pointer_mtime_ns: int
    last_refreshed_utc: str
    source_snapshot_at_utc: str | None
    target_score: float
    selected_axis: str
    selected_source: str
    selected_source_kind: str
    selected_score_precision: str
    selected_custody: str
    selected_evidence_grade: str
    selected_archive_sha256: str | None
    selected_lane_id: str | None
    selected_hardware_substrate: str | None
    selection_rule: str
    research_only: bool = True
    derived_planning_only: bool = True
    score_claim: bool = False
    evaluation_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    pointer_moved: bool = False


@dataclass(frozen=True)
class _StablePointerRead:
    path: Path
    payload: bytes
    sha256: str
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_utc(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise DynamicFrontierTargetError(f"{field} is not a valid UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise DynamicFrontierTargetError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _require_fresh(timestamp: str, *, field: str, now_utc_iso: str) -> None:
    observed = _parse_utc(timestamp, field=field)
    now = _parse_utc(now_utc_iso, field="now_utc_iso")
    age_seconds = (now - observed).total_seconds()
    if age_seconds > POINTER_STALE_SECONDS:
        raise DynamicFrontierTargetError(f"{field} is stale under the canonical 24-hour policy")
    if age_seconds < -300.0:
        raise DynamicFrontierTargetError(f"{field} is implausibly in the future")


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute name without dereferencing any path component."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_named_identity(stable: _StablePointerRead) -> None:
    """Prove the lexical name still denotes the descriptor-verified object."""

    try:
        named = stable.path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DynamicFrontierTargetError("canonical frontier pointer disappeared during read") from exc
    named_identity = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    verified_identity = (
        stable.device,
        stable.inode,
        stable.size,
        stable.mtime_ns,
        stable.ctime_ns,
    )
    if not stat.S_ISREG(named.st_mode) or named_identity != verified_identity:
        raise DynamicFrontierTargetError("canonical frontier pointer path identity changed during read")


def _stable_read(path: Path) -> _StablePointerRead:
    path = _lexical_absolute(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise DynamicFrontierTargetError(f"canonical frontier pointer missing at {path}") from exc
    except OSError as exc:
        raise DynamicFrontierTargetError(
            f"canonical frontier pointer cannot be opened without following links: {exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise DynamicFrontierTargetError("canonical frontier pointer must be a regular file")
        if before.st_size <= 0 or before.st_size > _MAX_POINTER_BYTES:
            raise DynamicFrontierTargetError("canonical frontier pointer byte length is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise DynamicFrontierTargetError("canonical frontier pointer was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise DynamicFrontierTargetError("canonical frontier pointer grew during read")
        after = os.fstat(fd)
    finally:
        os.close(fd)

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        raise DynamicFrontierTargetError("canonical frontier pointer mutated during read")
    payload = b"".join(chunks)
    stable = _StablePointerRead(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
    )
    _require_named_identity(stable)
    return stable


def _pointer_from_bytes(payload: bytes) -> CanonicalFrontierPointer:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicFrontierTargetError(f"canonical frontier pointer is corrupt: {exc}") from exc
    if not isinstance(decoded, Mapping):
        raise DynamicFrontierTargetError("canonical frontier pointer root must be an object")
    try:
        pointer = CanonicalFrontierPointer.from_dict(decoded)
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicFrontierTargetError(f"canonical frontier pointer fields are corrupt: {exc}") from exc
    if pointer.schema_version != POINTER_SCHEMA_VERSION:
        raise DynamicFrontierTargetError(f"unsupported canonical frontier pointer schema {pointer.schema_version!r}")
    return pointer


def _selected_text(row: Mapping[str, object], key: str, *, default: str = "unspecified") -> str:
    value = row.get(key)
    return str(value) if value not in (None, "") else default


def _load_from_path(path: Path, *, now_utc_iso: str) -> DynamicFrontierTargetSnapshot:
    stable = _stable_read(path)
    pointer = _pointer_from_bytes(stable.payload)
    _require_fresh(pointer.last_refreshed_utc, field="last_refreshed_utc", now_utc_iso=now_utc_iso)

    target = effective_frontier_score(pointer)
    selected = recompute_effective_frontier(pointer)
    if target is None or not math.isfinite(target) or target <= 0.0:
        raise DynamicFrontierTargetError("canonical pointer has no finite positive competitive target")
    if not isinstance(selected, Mapping):
        raise DynamicFrontierTargetError("canonical pointer has no custody-bearing selected row")
    try:
        selected_score = float(selected["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicFrontierTargetError("selected constituent row has an invalid score") from exc
    if selected_score != target:
        raise DynamicFrontierTargetError("effective target disagrees with recomputed selected row")

    source = _selected_text(selected, "source")
    source_snapshot_at_utc: str | None = None
    if source == "upstream_official_leaderboard":
        raw_timestamp = selected.get("snapshot_at_utc")
        if not isinstance(raw_timestamp, str) or not raw_timestamp:
            raise DynamicFrontierTargetError("selected official row lacks its source snapshot timestamp")
        _require_fresh(
            raw_timestamp,
            field="upstream_leaderboard_snapshot_at_utc",
            now_utc_iso=now_utc_iso,
        )
        source_snapshot_at_utc = raw_timestamp

    snapshot = DynamicFrontierTargetSnapshot(
        pointer_path=os.fspath(stable.path),
        pointer_bytes=stable.size,
        pointer_sha256=stable.sha256,
        pointer_device=stable.device,
        pointer_inode=stable.inode,
        pointer_mtime_ns=stable.mtime_ns,
        last_refreshed_utc=pointer.last_refreshed_utc,
        source_snapshot_at_utc=source_snapshot_at_utc,
        target_score=target,
        selected_axis=_selected_text(selected, "axis"),
        selected_source=source,
        selected_source_kind=_selected_text(selected, "source_kind"),
        selected_score_precision=_selected_text(selected, "score_precision"),
        selected_custody=_selected_text(selected, "custody"),
        selected_evidence_grade=_selected_text(selected, "evidence_grade"),
        selected_archive_sha256=(
            str(selected["archive_sha256"]) if selected.get("archive_sha256") not in (None, "") else None
        ),
        selected_lane_id=(str(selected["lane_id"]) if selected.get("lane_id") not in (None, "") else None),
        selected_hardware_substrate=(
            str(selected["hardware_substrate"]) if selected.get("hardware_substrate") not in (None, "") else None
        ),
        selection_rule=_selected_text(selected, "selection_rule"),
    )
    # Parsing and constituent recomposition happen after the descriptor read.
    # Recheck the lexical name at the last possible point so an atomic pointer
    # refresh during that work cannot bind old bytes to a new path identity.
    _require_named_identity(stable)
    return snapshot


def load_dynamic_frontier_target(
    *,
    repo_root: Path | str = ".",
    now_utc_iso: str | None = None,
) -> DynamicFrontierTargetSnapshot:
    """Load the canonical competitive target without refresh or mutation."""

    path = Path(repo_root) / CANONICAL_FRONTIER_POINTER_PATH
    return _load_from_path(path, now_utc_iso=now_utc_iso or _utc_now_iso())


def _verify_snapshot(
    snapshot: DynamicFrontierTargetSnapshot,
    *,
    now_utc_iso: str | None,
) -> None:
    if not isinstance(snapshot, DynamicFrontierTargetSnapshot):
        raise TypeError("snapshot must be a DynamicFrontierTargetSnapshot")
    now = now_utc_iso or _utc_now_iso()
    _require_fresh(snapshot.last_refreshed_utc, field="last_refreshed_utc", now_utc_iso=now)
    if snapshot.source_snapshot_at_utc is not None:
        _require_fresh(
            snapshot.source_snapshot_at_utc,
            field="upstream_leaderboard_snapshot_at_utc",
            now_utc_iso=now,
        )
    reopened = _load_from_path(Path(snapshot.pointer_path), now_utc_iso=now)
    if reopened != snapshot:
        raise DynamicFrontierTargetError("canonical frontier pointer identity or derived target changed after snapshot")


def verify_dynamic_frontier_target_snapshot(
    snapshot: DynamicFrontierTargetSnapshot,
    *,
    now_utc_iso: str | None = None,
) -> DynamicFrontierTargetSnapshot:
    """Reopen a snapshot's exact pointer object and return it only if current.

    Non-score consumers use this public guard instead of borrowing a score
    audit merely to obtain pointer freshness and identity verification.
    """

    _verify_snapshot(snapshot, now_utc_iso=now_utc_iso)
    return snapshot


def score_sublevel_against_dynamic_frontier(
    snapshot: DynamicFrontierTargetSnapshot,
    *,
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    reference_bytes: int = score_geometry.CONTEST_REFERENCE_BYTES,
    now_utc_iso: str | None = None,
) -> score_geometry.ScoreSublevelAudit:
    """Delegate a full same-object triple to the canonical sublevel audit."""

    _verify_snapshot(snapshot, now_utc_iso=now_utc_iso)
    return score_geometry.score_sublevel_audit(
        target_score=snapshot.target_score,
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        reference_bytes=reference_bytes,
    )


def score_transition_against_dynamic_frontier(
    snapshot: DynamicFrontierTargetSnapshot,
    *,
    before_d_seg: float,
    before_d_pose: float,
    before_archive_bytes: int,
    after_d_seg: float,
    after_d_pose: float,
    after_archive_bytes: int,
    reference_bytes: int = score_geometry.CONTEST_REFERENCE_BYTES,
    now_utc_iso: str | None = None,
) -> score_geometry.ScoreTransitionAudit:
    """Delegate an exact three-axis finite transition using the live target."""

    _verify_snapshot(snapshot, now_utc_iso=now_utc_iso)
    return score_geometry.score_transition_audit(
        target_score=snapshot.target_score,
        before_d_seg=before_d_seg,
        before_d_pose=before_d_pose,
        before_archive_bytes=before_archive_bytes,
        after_d_seg=after_d_seg,
        after_d_pose=after_d_pose,
        after_archive_bytes=after_archive_bytes,
        reference_bytes=reference_bytes,
    )
