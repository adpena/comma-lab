# SPDX-License-Identifier: MIT
"""Canonical recursive adversarial review ledger.

Mirrors the 4-layer pattern (helper + CLI + STRICT preflight + operator_authorize
wire-in) established by Catalog #245 (Modal call_id ledger) per the design memo
.omx/research/reusable_recursive_adversarial_review_canonical_design_20260517.md.

Persisted append-only at .omx/state/recursive_review_rounds.jsonl under fcntl
LOCK_EX per Catalog #128 / #131 / #245 sister discipline. Each row is a single
RecursiveReviewRound serialized via dataclasses.asdict + json.dumps(sort_keys=True).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "CANONICAL_AXES",
    "CANONICAL_ROTATIONS",
    "RECURSIVE_REVIEW_LEDGER_PATH",
    "VALID_SEVERITIES",
    "VALID_VERDICTS",
    "RecursiveReviewLedgerCorruptError",
    "RecursiveReviewRound",
    "ReviewFinding",
    "append_round_locked",
    "clean_pass_counter_for_bundle",
    "compute_bundle_id",
    "compute_scope_content_sha256",
    "latest_round_by_bundle_id",
    "load_rounds_lenient",
    "load_rounds_strict",
    "query_rounds_by_bundle_id",
    "query_unresolved_critical_findings",
    "update_from_anchor",  # Catalog #265 canonical-contract alias
]

VALID_SEVERITIES: frozenset[str] = frozenset({"CRITICAL", "MEDIUM", "LOW", "CONFIRMS"})
VALID_VERDICTS: frozenset[str] = frozenset(
    {"PROCEED", "PROCEED_WITH_REVISIONS", "DEFER", "KILL_CANDIDATE"}
)
CANONICAL_AXES: frozenset[str] = frozenset(
    {
        "call_sites",
        "phase_interactions",
        "resume_scenarios",
        "edge_cases",
        "default_overrides",
        "comments_vs_code",
        "phase_gate_thresholds",
        "assumption_challenge",
    }
)
CANONICAL_ROTATIONS: frozenset[str] = frozenset(
    {
        "skunkworks_sextet",
        "Z_fresh_eyes",
        "Y_engineering_red",
        "X_theoretical_floor",
        "A_substrate_specialist",
        "B_oss_release",
    }
)

RECURSIVE_REVIEW_LEDGER_PATH = Path(".omx/state/recursive_review_rounds.jsonl")
_LEDGER_LOCK_PATH = Path(".omx/state/.recursive_review.lock")
SEAL_THRESHOLD = 3


class RecursiveReviewLedgerCorruptError(RuntimeError):
    """Raised when load_rounds_strict cannot parse a row [verified-against: Catalog #138]."""


@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str
    axis: str
    severity: str
    member: str
    description: str
    recommended_fix: str

    def __post_init__(self) -> None:
        if self.axis not in CANONICAL_AXES:
            raise ValueError(
                f"axis {self.axis!r} not in canonical 8-axis set per CLAUDE.md "
                f"'Recursive adversarial review protocol' item 8: {sorted(CANONICAL_AXES)}"
            )
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"severity {self.severity!r} not in {sorted(VALID_SEVERITIES)}"
            )
        for field_name in ("finding_id", "member", "description", "recommended_fix"):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"finding field {field_name!r} must be non-empty string")


@dataclass(frozen=True)
class RecursiveReviewRound:
    review_id: str
    bundle_id: str
    scope_paths: tuple[str, ...]
    scope_content_sha256: str
    round_number: int
    council_rotation: str
    council_attendees: tuple[str, ...]
    findings: tuple[ReviewFinding, ...]
    verdict: str
    counter_before: int
    counter_after: int
    reviewed_at_utc: str
    reviewer_agent: str
    related_round_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict {self.verdict!r} not in {sorted(VALID_VERDICTS)}")
        if self.round_number < 1:
            raise ValueError("round_number must be >= 1")
        if self.counter_before < 0 or self.counter_after < 0:
            raise ValueError("counters must be >= 0")
        non_confirms = [f for f in self.findings if f.severity != "CONFIRMS"]
        if non_confirms and self.counter_after != 0:
            raise ValueError(
                "counter_after MUST reset to 0 when round has any non-CONFIRMS finding "
                "[verified-against: CLAUDE.md 'Recursive adversarial review protocol' item 3]"
            )
        if not non_confirms and self.counter_after != self.counter_before + 1:
            raise ValueError("counter_after MUST equal counter_before + 1 for a clean round")


def compute_bundle_id(scope_paths: Sequence[str]) -> str:
    """Stable id from sorted path list."""
    joined = "\n".join(sorted(scope_paths))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def compute_scope_content_sha256(scope_paths: Sequence[str], repo_root: Path | str = ".") -> str:
    """sha256 of concatenated file contents (sorted) at review time."""
    root = Path(repo_root)
    h = hashlib.sha256()
    for rel in sorted(scope_paths):
        p = root / rel
        if not p.exists():
            h.update(b"\x00MISSING\x00")
            h.update(rel.encode())
            continue
        h.update(p.read_bytes())
        h.update(b"\x00SEP\x00")
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_append(target: Path, line: str) -> None:
    """Atomic append via tmp+rename per Catalog #245 sister discipline."""
    _ensure_parent(target)
    existing = target.read_bytes() if target.exists() else b""
    payload = existing + line.encode() + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=str(target.parent), prefix=target.name, suffix=f".tmp.{uuid.uuid4().hex[:12]}", delete=False
    ) as f:
        f.write(payload)
        tmp = Path(f.name)
    os.replace(tmp, target)


def append_round_locked(
    record: RecursiveReviewRound,
    *,
    path: Path | None = None,
    lock_path: Path | None = None,
) -> None:
    """fcntl-locked append per Catalog #128 / #131 / #245."""
    target = path or RECURSIVE_REVIEW_LEDGER_PATH
    lock = lock_path or _LEDGER_LOCK_PATH
    _ensure_parent(lock)
    payload = asdict(record)
    payload["written_at_utc"] = _utc_now()
    payload["written_pid"] = os.getpid()
    payload["written_host"] = socket.gethostname()
    payload["schema_version"] = "recursive_review_round_v1"
    line = json.dumps(payload, sort_keys=True)
    with open(lock, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            expected_counter = clean_pass_counter_for_bundle(
                record.bundle_id,
                path=target,
                scope_content_sha256=record.scope_content_sha256,
            )
            if record.counter_before != expected_counter:
                raise ValueError(
                    "counter_before does not match content-aware ledger state: "
                    f"record={record.counter_before}, expected={expected_counter}. "
                    "If the bundle content changed, the clean-pass counter must reset."
                )
            _atomic_write_append(target, line)
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def load_rounds_lenient(path: Path | None = None) -> list[dict]:
    """Skip malformed rows; return parsed list."""
    target = path or RECURSIVE_REVIEW_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    for line in target.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_rounds_strict(path: Path | None = None) -> list[dict]:
    """Raise + quarantine on parse failure per Catalog #138."""
    target = path or RECURSIVE_REVIEW_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    for idx, line in enumerate(target.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            quarantine = target.with_suffix(f".corrupt.{_utc_now()}")
            target.rename(quarantine)
            raise RecursiveReviewLedgerCorruptError(
                f"recursive_review_rounds.jsonl line {idx + 1}: {exc}; quarantined to {quarantine}"
            ) from exc
        if not isinstance(row, dict):
            quarantine = target.with_suffix(f".corrupt.{_utc_now()}")
            target.rename(quarantine)
            raise RecursiveReviewLedgerCorruptError(
                f"recursive_review_rounds.jsonl line {idx + 1}: non-dict root; quarantined to {quarantine}"
            )
        rows.append(row)
    return rows


def query_rounds_by_bundle_id(
    bundle_id: str, *, path: Path | None = None
) -> list[dict]:
    return [r for r in load_rounds_lenient(path) if r.get("bundle_id") == bundle_id]


def latest_round_by_bundle_id(
    bundle_id: str, *, path: Path | None = None
) -> dict | None:
    rows = query_rounds_by_bundle_id(bundle_id, path=path)
    if not rows:
        return None
    return rows[-1]


def clean_pass_counter_for_bundle(
    bundle_id: str,
    *,
    path: Path | None = None,
    scope_content_sha256: str | None = None,
) -> int:
    latest = latest_round_by_bundle_id(bundle_id, path=path)
    if latest is None:
        return 0
    if (
        scope_content_sha256 is not None
        and latest.get("scope_content_sha256") != scope_content_sha256
    ):
        return 0
    return int(latest.get("counter_after", 0))


def query_unresolved_critical_findings(
    bundle_id: str, *, path: Path | None = None
) -> list[dict]:
    """Returns CRITICAL findings from the latest round; empty if SEALED."""
    latest = latest_round_by_bundle_id(bundle_id, path=path)
    if latest is None:
        return []
    if int(latest.get("counter_after", 0)) >= SEAL_THRESHOLD:
        return []
    return [f for f in latest.get("findings", []) if f.get("severity") == "CRITICAL"]


def update_from_anchor(anchor: dict, *, path: Path | None = None) -> None:
    """Catalog #265 canonical-contract alias.

    Reads a dict-shape anchor (e.g., from continual-learning sister); persists
    as a new round via the canonical helper."""
    findings = tuple(
        ReviewFinding(**f) if not isinstance(f, ReviewFinding) else f
        for f in anchor.get("findings", [])
    )
    record = RecursiveReviewRound(
        review_id=anchor.get("review_id", uuid.uuid4().hex[:12]),
        bundle_id=anchor["bundle_id"],
        scope_paths=tuple(anchor["scope_paths"]),
        scope_content_sha256=anchor["scope_content_sha256"],
        round_number=int(anchor["round_number"]),
        council_rotation=anchor["council_rotation"],
        council_attendees=tuple(anchor["council_attendees"]),
        findings=findings,
        verdict=anchor["verdict"],
        counter_before=int(anchor["counter_before"]),
        counter_after=int(anchor["counter_after"]),
        reviewed_at_utc=anchor.get("reviewed_at_utc", _utc_now()),
        reviewer_agent=anchor.get("reviewer_agent", "anchor-import"),
        related_round_ids=tuple(anchor.get("related_round_ids", ())),
    )
    append_round_locked(record, path=path)
