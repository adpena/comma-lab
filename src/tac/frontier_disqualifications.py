"""Append-only eligibility decisions; mirror bytes remain historical provenance.

Identity is (lane, archive), not archive alone: different receivers can share
archive bytes. Missing lane/SHA in legacy consumers is matched conservatively.
The pointer lock serializes journal mutations with pointer publication.
"""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

JOURNAL = Path(".omx/state/frontier_disqualifications.jsonl")
REASON_CLASSES = ("rule118_content_in_code", "decode_budget", "determinism", "custody")


@contextmanager
def eligibility_lock(repo_root: Path | str, *, exclusive: bool = False):
    """Hold packet eligibility through all stages; decisions wait for readers."""
    path = Path(repo_root) / ".omx/state/.frontier_eligibility.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def substantive_rationale(value: object) -> bool:
    """Reject empty, templated, and placeholder explanations (Catalog #287)."""
    return (
        isinstance(value, str)
        and len(value.strip()) >= 12
        and not re.search(r"[<>]|\b(todo|tbd|placeholder|pending|fixme)\b", value, re.I)
        and value.strip().lower() not in {"test rationale", "some rationale", "not applicable"}
    )


def _validate_event(row: dict) -> None:
    if not isinstance(row, dict) or row.get("action") not in {"disqualify", "reinstate"}:
        raise ValueError("invalid disqualification event/action")
    if not isinstance(row.get("lane_id"), str) or not row["lane_id"].strip():
        raise ValueError("disqualification lane_id is required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("archive_sha256", ""))):
        raise ValueError("archive_sha256 must be 64 lowercase hex characters")
    if row.get("reason_class") not in REASON_CLASSES or not substantive_rationale(row.get("rationale")):
        raise ValueError("valid reason class and substantive rationale required")
    for key in ("who", "when", "evidence"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"disqualification {key} is required")
    try:
        timestamp = datetime.fromisoformat(row["when"])
    except ValueError as exc:
        raise ValueError("invalid disqualification timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("disqualification timestamp requires timezone")
    if row["action"] == "reinstate" and (
        row.get("reinstated_at") != row["when"] or row.get("why") != row["rationale"]
    ):
        raise ValueError("reinstatement must carry reinstated_at and why")


def active_disqualifications(repo_root: Path | str) -> list[dict]:
    """Replay every event, failing closed on corruption or invalid transitions."""
    path = Path(repo_root) / JOURNAL
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []  # Legacy repositories have no decisions yet.
    active: dict[tuple[str, str], dict] = {}
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            _validate_event(row)
            key = (row["lane_id"], row["archive_sha256"])
            if row["action"] == "reinstate":
                if key not in active:
                    raise ValueError("cannot reinstate an inactive row")
                del active[key]
            else:
                active[key] = row
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: invalid disqualification journal: {exc}") from exc
    if raw and not raw.endswith("\n"):
        raise ValueError(f"{path}: unterminated journal record; repair custody before append")
    return list(active.values())


def disqualification_for(rows: list[dict], lane_id: str | None, archive_sha256: str | None) -> dict | None:
    """Look up exact identity; legacy missing identity never hides a known veto."""
    return next((row for row in rows
                 if (not lane_id or row["lane_id"] == lane_id)
                 and (not archive_sha256 or row["archive_sha256"] == archive_sha256)), None)


def require_qualified(repo_root: Path | str, lane_id: str | None, archive_sha256: str | None) -> None:
    """Refuse a disqualified row before any packet or pointer publication."""
    row = disqualification_for(active_disqualifications(repo_root), lane_id, archive_sha256)
    if row:
        raise ValueError(f"disqualified {row['lane_id']} {row['archive_sha256']}: "
                         f"{row['reason_class']}: {row['rationale']} (evidence: {row['evidence']})")


def append_disqualification(
    repo_root: Path | str, *, lane_id: str, archive_sha256: str,
    reason_class: str | None = None, evidence: str | None = None,
    rationale: str, who: str | None = None, reinstate: bool = False,
) -> dict:
    """Persist one fsynced decision under lock; never rewrite earlier records."""
    root = Path(repo_root)
    path = root / JOURNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / ".canonical_frontier_pointer.lock"
    with eligibility_lock(root, exclusive=True), lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = active_disqualifications(root)
        prior = disqualification_for(rows, lane_id, archive_sha256)
        if reinstate and (not prior or (prior["lane_id"], prior["archive_sha256"]) != (lane_id, archive_sha256)):
            raise ValueError("cannot reinstate an inactive row")
        evidence = evidence or (prior["evidence"] if reinstate and prior else None)
        evidence_path = root / (evidence or "")
        if not evidence_path.is_file():
            raise ValueError("evidence must name an existing file")
        now = datetime.now(UTC).isoformat()
        row = {
            "action": "reinstate" if reinstate else "disqualify",
            "lane_id": lane_id, "archive_sha256": archive_sha256,
            "reason_class": reason_class or (prior["reason_class"] if reinstate and prior else None),
            "evidence": evidence, "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "rationale": rationale, "who": who or getpass.getuser(), "when": now,
            "reinstated_at": now if reinstate else None, "why": rationale if reinstate else None,
            "score_claim": False,
        }
        _validate_event(row)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return row
