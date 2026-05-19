# SPDX-License-Identifier: MIT
"""Sister-subagent checkpoint guard — STAGING-surface absorption prevention.

The canonical helper landed by CATALOG-314-PREVENTION-ENHANCEMENT 2026-05-19
per CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
non-negotiable. Sister of Catalog #314 (POST-COMMIT detect) at the PRE-STAGE
surface — together the two close the bare-commit-absorbs-in-flight-files
bug class bidirectionally.

What this guards against
────────────────────────
Bug class anchor (2026-05-19): slot 5 commit ``c8d51ebb5`` absorbed slot 2's
``src/tac/preflight.py`` + ``CLAUDE.md`` edits before slot 2's canonical
serializer call ran. Work landed cleanly under the wrong commit body.
Catalog #157 ``--expected-content-sha256`` caught the secondary effect (file
shas mismatched) but the absorption was downstream of the bare ``git add``
that the ``/commit`` slash command does directly (NOT through
``tools/subagent_commit_serializer.py``).

This helper reads ``.omx/state/subagent_progress.jsonl`` (the canonical
crash-resume checkpoint store, fcntl-locked per Catalog #131), filters to
rows with ``status='in_progress'`` within the configurable lookback window
(default 60 minutes — same window as Catalog #302 + Catalog #314), excludes
the caller's own checkpoint, and refuses if any of the caller's
intended-to-commit files overlap a sister subagent's declared
``files_touched``.

Recommendation taxonomy
───────────────────────
``PROCEED``         — no conflict; caller may proceed with the commit
``ABORT``           — at least one sister subagent has declared the same file
                      with ``status='in_progress'`` AND the file is NOT in
                      ``EXEMPT_FILES``; caller must coordinate via Catalog
                      #230 ownership map before retrying
``WAIT_AND_RETRY``  — at least one sister subagent has declared the same file
                      but its checkpoint is older than half the lookback
                      window (sister may be near completion); caller may
                      retry with exponential backoff per Catalog #131
                      bare-write discipline

Wire-in surfaces
────────────────
1. ``tools/subagent_commit_serializer.py`` — invokes this helper BEFORE
   acquiring the fcntl lock so ABORT/WAIT_AND_RETRY surfaces BEFORE the
   commit machinery runs. Exit codes rc=8/9/10 (renumbered from the brief's
   suggested 5/6/7 to avoid collision with existing serializer rc=5 staged-
   content sha mismatch).
2. ``tools/check_sister_checkpoint_before_git_add.py`` — operator-runnable
   pre-commit hook for the ``/commit`` slash command (which does bare
   ``git add`` + ``git commit`` outside the canonical serializer).
3. ``src/tac/preflight.py`` Catalog #340
   ``check_subagent_commit_serializer_invokes_sister_checkpoint_guard`` —
   META-meta STRICT preflight gate that scans the serializer source for the
   canonical helper invocation BEFORE fcntl lock acquisition.

Paired-env bypass discipline
────────────────────────────
Per Catalog #199 sister rule, intentional override requires BOTH env vars:

    SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE=1
    SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE=<text>

Bare intent without rationale raises ``SystemExit(10)`` in the serializer
wire-in (rc=10 reserved for bare bypass attempt).

Memory: ``feedback_catalog_314_prevention_enhancement_landed_20260519.md``.
Lane: ``lane_catalog_314_prevention_enhancement_20260519``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
from pathlib import Path
from typing import Literal

# ── Constants ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
"""Repo root: src/tac/commit_safety/ lives three levels under repo root."""

CHECKPOINT_JSONL_PATH = REPO_ROOT / ".omx" / "state" / "subagent_progress.jsonl"
"""Canonical crash-resume checkpoint store; managed by
``tools/subagent_checkpoint.py`` per CLAUDE.md "Mandatory crash-resume
protocol" + Catalog #131 + #138."""

DEFAULT_LOOKBACK_MINUTES = 60
"""Default lookback window. Mirrors Catalog #302 + Catalog #314 60-min
absorption window so cross-surface alerts agree on the same time horizon."""

EXEMPT_FILES: frozenset[str] = frozenset({
    # Mirrors Catalog #302 + Catalog #314 _CHECK_*_EXEMPT_FILES (commonly
    # multi-subagent state files; absorption-collision is not a bug-class
    # signature for these because many subagents legitimately append to them
    # concurrently via the canonical fcntl-locked helpers).
    ".omx/state/modal_call_id_ledger.jsonl",
    ".omx/state/active_lane_dispatch_claims.md",
    ".omx/state/commit-serializer.log",
    ".omx/state/catalog-claim.log",
    ".omx/state/subagent_progress.jsonl",
    ".omx/state/lane_registry.json",
    ".omx/state/lane_maturity_audit.log",
    ".omx/state/continual_learning_posterior.jsonl",
    ".omx/state/cost_band_posterior.jsonl",
    ".omx/state/next_catalog_number.txt",
    ".omx/state/probe_outcomes.jsonl",
    ".omx/state/council_deliberation_posterior.jsonl",
    "MEMORY.md",
})

OVERRIDE_ENV_FLAG = "SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE"
OVERRIDE_ENV_RATIONALE = "SUBAGENT_COMMIT_SISTER_CHECKPOINT_OVERRIDE_RATIONALE"
"""Paired-env bypass per Catalog #199. Bare ``OVERRIDE=1`` without
``OVERRIDE_RATIONALE=<text>`` is REFUSED with rc=10 in the serializer."""


Recommendation = Literal["PROCEED", "ABORT", "WAIT_AND_RETRY"]


# ── Typed verdict ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class SisterCheckpointVerdict:
    """Typed verdict returned by ``check_files_against_sister_checkpoints``.

    Attributes
    ----------
    recommendation : Recommendation
        One of ``PROCEED`` / ``ABORT`` / ``WAIT_AND_RETRY``. Callers should
        translate to their exit-code or branching convention.
    conflicts : tuple[tuple[str, tuple[str, ...]], ...]
        For each conflicting sister subagent, a (sister_id, overlapping_files)
        tuple. ``files`` are sorted relative-path strings for stable output.
    diagnostic : str
        Human-readable diagnostic explaining the verdict (printed to stderr
        by the serializer wire-in and the pre-commit hook helper).
    in_flight_subagent_ids : tuple[str, ...]
        IDs of every sister subagent whose checkpoint was considered (may
        include subagents with no file overlap; surfaced for diagnostics).
    checkpoint_path : str
        Resolved path to the canonical checkpoint JSONL store (for
        operator-facing diagnostics).
    """

    recommendation: Recommendation
    conflicts: tuple[tuple[str, tuple[str, ...]], ...]
    diagnostic: str
    in_flight_subagent_ids: tuple[str, ...]
    checkpoint_path: str

    def has_conflict(self) -> bool:
        """True iff at least one sister subagent declared an overlapping file."""
        return bool(self.conflicts)


# ── Internal helpers ─────────────────────────────────────────────────────
def _parse_iso_utc(ts: str | None) -> _dt.datetime | None:
    """Parse an ISO-UTC timestamp; return aware datetime or None on failure."""
    if not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        d = _dt.datetime.fromisoformat(ts)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d
    except (TypeError, ValueError):
        return None


def _normalize_files_touched(raw) -> list[str]:
    """Coerce a checkpoint row's ``files_touched`` field to a flat str list.

    Legacy operator-issued checkpoints sometimes embedded space-separated
    paths inside a single string element; handle that gracefully (mirrors
    Catalog #314 ``_check_314_load_in_flight_subagent_files`` normalization).
    """
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for f in raw:
        if not isinstance(f, str):
            continue
        f = f.strip()
        if not f:
            continue
        # Some legacy rows packed multiple paths into one string.
        if " " in f and not f.startswith("/"):
            out.extend(p.strip() for p in f.split() if p.strip())
        else:
            out.append(f)
    return out


def _load_checkpoints_strict(jsonl_path: Path) -> list[dict]:
    """Read all checkpoint rows; raise on corrupt JSON per Catalog #138.

    Fail-closed loader: any malformed line raises ``CorruptCheckpointError``.
    Missing file returns an empty list (the canonical empty-state behavior;
    matches ``tac.subagent_checkpoint.read_checkpoints``).
    """
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise CorruptCheckpointError(
                        f"{jsonl_path}:{lineno}: malformed JSON: {exc!s}"
                    ) from exc
                if not isinstance(rec, dict):
                    raise CorruptCheckpointError(
                        f"{jsonl_path}:{lineno}: row is not a dict"
                    )
                rows.append(rec)
    except OSError as exc:
        raise CorruptCheckpointError(
            f"{jsonl_path}: OSError reading checkpoint JSONL: {exc!s}"
        ) from exc
    return rows


class CorruptCheckpointError(RuntimeError):
    """Raised by ``_load_checkpoints_strict`` on malformed JSONL.

    Per Catalog #138 ``check_state_writers_strict_load_for_mutating_path``
    fail-closed pattern: corrupt state must surface as a typed exception so
    the caller refuses with rc=4 / rc=8 / etc. rather than silently
    returning an empty list and proceeding.
    """


# ── Public API ───────────────────────────────────────────────────────────
def check_files_against_sister_checkpoints(
    files: list[str],
    *,
    current_subagent_id: str | None = None,
    lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
    checkpoint_path: Path | None = None,
    now_utc: _dt.datetime | None = None,
) -> SisterCheckpointVerdict:
    """Check ``files`` against in-flight sister subagent checkpoints.

    Returns a ``SisterCheckpointVerdict`` whose ``recommendation`` is one of:

    - ``PROCEED`` — no overlap with any sister subagent's declared
      ``files_touched``. Caller may proceed with the commit/staging.
    - ``ABORT`` — at least one sister subagent has declared the same
      non-exempt file with ``status='in_progress'``. Caller MUST coordinate
      via Catalog #230 ownership map (or wait + retry) before staging.
    - ``WAIT_AND_RETRY`` — overlap detected but every overlapping sister's
      checkpoint is older than half the lookback window (the sister may
      be near completion). Caller should retry with exponential backoff;
      if still ABORT after a reasonable delay, escalate to Catalog #230.

    Parameters
    ----------
    files
        Repo-relative path strings the caller intends to stage. Must be a
        list (not a string — coercing a string to ``list('foo.py')``
        would silently produce a list of characters).
    current_subagent_id
        The caller's own ``subagent_id``; excluded from conflict detection
        so the caller does not flag itself. If ``None``, every in-flight
        subagent is considered a potential collision.
    lookback_minutes
        Window for "in-flight" classification. Sister checkpoints written
        more than this many minutes ago are ignored (presumed complete or
        crashed). Default 60 min mirrors Catalog #302 + #314.
    checkpoint_path
        Override the canonical ``.omx/state/subagent_progress.jsonl`` path
        (useful for testing). If ``None``, uses ``CHECKPOINT_JSONL_PATH``.
    now_utc
        Override the current UTC time (useful for testing). If ``None``,
        uses ``datetime.now(timezone.utc)``.

    Raises
    ------
    TypeError
        If ``files`` is not a list of strings.
    CorruptCheckpointError
        If the checkpoint JSONL contains malformed lines. Per Catalog #138
        fail-closed pattern: corrupt state surfaces as an exception so the
        caller refuses rather than silently proceeding.
    """
    if not isinstance(files, list) or not all(isinstance(f, str) for f in files):
        raise TypeError("files must be a list of strings")

    path = Path(checkpoint_path) if checkpoint_path is not None else CHECKPOINT_JSONL_PATH
    now = now_utc if now_utc is not None else _dt.datetime.now(_dt.timezone.utc)
    lookback_seconds = lookback_minutes * 60

    rows = _load_checkpoints_strict(path)

    # Build a "latest checkpoint per subagent" map. A single subagent may
    # write multiple checkpoints (step 1, step 2, ...); only the most-recent
    # row matters for "is this subagent currently in-flight".
    latest_per_subagent: dict[str, dict] = {}
    for row in rows:
        sid = row.get("subagent_id")
        if not isinstance(sid, str) or not sid:
            continue
        latest_per_subagent[sid] = row

    # Filter to in-flight subagents within the lookback window.
    in_flight: list[tuple[str, _dt.datetime, set[str]]] = []
    for sid, row in latest_per_subagent.items():
        if sid == current_subagent_id:
            continue
        status = row.get("status")
        if status != "in_progress":
            continue
        ts = _parse_iso_utc(row.get("written_at_utc"))
        if ts is None:
            continue
        try:
            delta = (now - ts).total_seconds()
        except (TypeError, ValueError):
            continue
        if delta < 0 or delta > lookback_seconds:
            continue
        declared = _normalize_files_touched(row.get("files_touched"))
        # Filter out exempt files — they are commonly multi-subagent state
        # and overlap on them is not a bug-class signature.
        non_exempt = {f for f in declared if f and f not in EXEMPT_FILES}
        if not non_exempt:
            continue
        in_flight.append((sid, ts, non_exempt))

    in_flight_ids = tuple(sorted(sid for sid, _, _ in in_flight))

    # Compute caller's non-exempt file set.
    caller_files = {f for f in files if f and f not in EXEMPT_FILES}
    if not caller_files:
        return SisterCheckpointVerdict(
            recommendation="PROCEED",
            conflicts=(),
            diagnostic=(
                "PROCEED: caller declared no non-exempt files; staging "
                "approved without sister-checkpoint scan."
            ),
            in_flight_subagent_ids=in_flight_ids,
            checkpoint_path=str(path),
        )

    # Detect overlaps per sister subagent.
    conflicts: list[tuple[str, tuple[str, ...], _dt.datetime]] = []
    for sid, ts, sister_files in in_flight:
        overlap = caller_files & sister_files
        if not overlap:
            continue
        conflicts.append((sid, tuple(sorted(overlap)), ts))

    if not conflicts:
        return SisterCheckpointVerdict(
            recommendation="PROCEED",
            conflicts=(),
            diagnostic=(
                f"PROCEED: caller's {len(caller_files)} non-exempt file(s) "
                f"do not overlap any of {len(in_flight)} in-flight sister "
                f"subagent's files_touched within the {lookback_minutes}-"
                f"minute lookback window."
            ),
            in_flight_subagent_ids=in_flight_ids,
            checkpoint_path=str(path),
        )

    # Classify each conflict by age: if EVERY conflict's sister checkpoint
    # is older than half the lookback window, recommend WAIT_AND_RETRY
    # (sister may be near completion). Otherwise recommend ABORT.
    half_window_seconds = lookback_seconds / 2
    every_conflict_older_than_half = all(
        (now - ts).total_seconds() >= half_window_seconds
        for _, _, ts in conflicts
    )
    recommendation: Recommendation = (
        "WAIT_AND_RETRY" if every_conflict_older_than_half else "ABORT"
    )

    # Build human-readable diagnostic. Sort conflicts by sister_id for
    # stable output.
    conflicts_sorted = sorted(conflicts, key=lambda x: x[0])
    conflict_lines = []
    for sid, overlap_files, ts in conflicts_sorted:
        age_min = (now - ts).total_seconds() / 60
        files_str = ", ".join(overlap_files[:5])
        more = "" if len(overlap_files) <= 5 else f" (+{len(overlap_files) - 5} more)"
        conflict_lines.append(
            f"  sister={sid!r} (checkpoint {age_min:.1f} min ago) "
            f"overlaps on: {{ {files_str} }}{more}"
        )

    diagnostic = (
        f"{recommendation}: {len(conflicts)} sister subagent collision(s) "
        f"detected. Per CLAUDE.md \"Bugs must be permanently fixed AND "
        f"self-protected against\" non-negotiable + Catalog #340 STAGING-"
        f"surface PREVENTION (sister of Catalog #314 POST-COMMIT detect).\n"
        + "\n".join(conflict_lines)
        + (
            "\n  Recommendation: coordinate via Catalog #230 ownership map "
            "OR commit via tools/subagent_commit_serializer.py with "
            "--expected-content-sha256 per Catalog #157 (lets fcntl arbitrate)."
            if recommendation == "ABORT"
            else "\n  Recommendation: retry with exponential backoff (sister "
            "may be near completion); escalate to Catalog #230 if still "
            "ABORT after reasonable delay."
        )
    )

    # Strip the third tuple element (timestamp) for the public verdict
    # conflicts field; it's already encoded in the diagnostic string.
    public_conflicts = tuple((sid, files_) for sid, files_, _ in conflicts_sorted)
    return SisterCheckpointVerdict(
        recommendation=recommendation,
        conflicts=public_conflicts,
        diagnostic=diagnostic,
        in_flight_subagent_ids=in_flight_ids,
        checkpoint_path=str(path),
    )


def parse_override_env(env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Parse the paired-env bypass flag + rationale.

    Returns ``(active, rationale)`` where ``active`` is True iff BOTH env
    vars are set with a non-empty rationale (≥4 chars, NOT a placeholder).

    Bare ``OVERRIDE=1`` without rationale → returns ``(False, "")``; the
    serializer wire-in will refuse with rc=10 (bare bypass attempt).
    """
    env = env if env is not None else dict(os.environ)
    flag = env.get(OVERRIDE_ENV_FLAG, "").strip()
    rationale = env.get(OVERRIDE_ENV_RATIONALE, "").strip()
    if flag not in ("1", "true", "True", "yes", "YES"):
        return False, ""
    # Placeholder rationales are rejected so the helper's docstring example
    # cannot self-waive (per Catalog #287 sister discipline).
    placeholders = ("<text>", "<rationale>", "<reason>")
    if not rationale or rationale.lower() in placeholders or len(rationale) < 4:
        return False, ""
    return True, rationale


def bare_override_attempted(env: dict[str, str] | None = None) -> bool:
    """True iff ``OVERRIDE=1`` is set WITHOUT a valid rationale.

    Used by the serializer wire-in to surface rc=10 (paired-env bypass
    discipline violated) rather than silently proceeding past the guard.
    """
    env = env if env is not None else dict(os.environ)
    flag = env.get(OVERRIDE_ENV_FLAG, "").strip()
    if flag not in ("1", "true", "True", "yes", "YES"):
        return False
    active, _ = parse_override_env(env)
    return not active
