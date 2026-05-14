#!/usr/bin/env python3
"""Atomic lane-dispatch claim helper.

Use before paid remote work:
    tools/claim_lane_dispatch.py claim --lane-id public_floor_pvl1 \\
        --platform lightning --instance-job-id exact_eval_... \\
        --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-02T00:20Z \\
        --status eval --notes "T4 promotion"

Recovery note: this helper was lost when subagent worktrees were auto-cleaned
without committing source to git. Only the .pyc cache survived. Rebuilt
2026-05-04 from .pyc string constants + the active_lane_dispatch_claims.md
ledger schema + CLAUDE.md spec.

Behavior contract:
  - Reads .omx/state/active_lane_dispatch_claims.md
  - Refuses to insert a new active (non-terminal) claim for lane_id when an
    active claim already exists within the past --ttl-hours (default 24)
  - Exception: --allow-parallel + --child-of <existing-job-id> + --parallel-reason
    permits a bounded child claim that depends on an active parent
  - Prepends the new row at the top of "## Claims (newest first)" so the
    file remains naturally chronological-newest-first
  - --force bypasses conflict refusal (operator override; logs reason)
  - --dry-run prints the row that would be written without modifying the file
  - Uses fcntl.flock for atomic write; lockfile = claims_path + ".lock"

Exit codes:
  0 - claim recorded (or dry-run succeeded)
  2 - validation error (bad cell content, missing required arg, etc.)
  3 - REFUSING_DISPATCH: conflict detected, no force, no parallel exception
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_ARCHIVE_DIR = Path(".omx/state/dispatch_claims_archive")

# Default for ``--prune --terminal-age-days``: keep terminal rows newer than
# this; archive older terminal rows. 7 days mirrors the T1-E state-hygiene
# spec (2026-05-12 simplification audit).
DEFAULT_PRUNE_TERMINAL_AGE_DAYS = 7

# Header line stamped onto each monthly archive file. Archives are append-only.
ARCHIVE_HEADER_FORMAT = (
    "# Archived dispatch claims — {month_label}\n"
    "\n"
    "Append-only monthly archive of terminal dispatch claim rows pruned from "
    "`.omx/state/active_lane_dispatch_claims.md` by "
    "`tools/claim_lane_dispatch.py prune`. Per CLAUDE.md \"Operator gates "
    "must be wired and used\" + the artifact-kind registry, this file is "
    "HISTORICAL_PROVENANCE — never mutate previously-written rows.\n"
    "\n"
    "## Claims (newest first)\n"
    "\n"
    "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)

HEADER = (
    "# Active lane dispatch claims — cross-agent coordination ledger\n"
    "\n"
    "**MANDATORY for ALL agents (Claude, codex, future):** Before dispatching ANY "
    "training, eval, or remote-GPU job, READ this file FIRST and APPEND a claim row. "
    "Check for conflicts on the same `lane_id` within the past 24h.\n"
    "\n"
    "## Claims (newest first)\n"
    "\n"
    "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)

TERMINAL_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)

_TABLE_HEADER_RE = re.compile(r"^\|\s*timestamp_utc\s*\|", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^\|\s*-+\s*(\|\s*-+\s*)+\|\s*$")
_SHELL_ARGV0_EXPANSION_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/bin/(?:zsh|bash|sh)(?:[./\s]|$)"
)
_SHELL_ARGV0_WAIVER = "SHELL_ARGV0_OK:"
_LANE_ID_RE = re.compile(r"^(?=.*[A-Za-z])[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_RESERVED_LANE_IDS = {
    "0",
    "none",
    "null",
    "undefined",
    "unknown",
    "na",
    "n/a",
    "modal",
    "cuda",
    "cpu",
    "true",
    "false",
}


@dataclass(frozen=True)
class Claim:
    timestamp_utc: str
    agent: str
    lane_id: str
    platform: str
    instance_job_id: str
    predicted_eta_utc: str
    status: str
    notes: str


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0)


def _parse_utc(value: str) -> dt.datetime | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _validate_cell(name: str, value, *, allow_empty: bool = False, allow_space: bool = True) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"VALIDATION_ERROR: {name} must be a string")
    if value != value.strip():
        raise SystemExit(f"VALIDATION_ERROR: {name} must not have leading/trailing whitespace")
    if not allow_empty and not value:
        raise SystemExit(f"VALIDATION_ERROR: {name} must not be empty")
    if "|" in value:
        raise SystemExit(f"VALIDATION_ERROR: {name} must not contain markdown table separators")
    if any(ord(c) < 0x20 and c not in ("\t",) for c in value):
        raise SystemExit(f"VALIDATION_ERROR: {name} must not contain control characters")
    if not allow_space and any(c.isspace() for c in value):
        raise SystemExit(f"VALIDATION_ERROR: {name} must not contain whitespace")
    return value


def _validate_lane_id(value: str) -> str:
    lane_id = _validate_cell("--lane-id", value, allow_space=False)
    if not _lane_id_is_valid(lane_id):
        raise SystemExit(
            "VALIDATION_ERROR: --lane-id must be a real canonical lane id "
            "(letters plus optional digits, dots, underscores, hyphens, or colons); "
            f"got {lane_id!r}"
        )
    return lane_id


def _lane_id_is_valid(value: str) -> bool:
    if not isinstance(value, str):
        return False
    lane_id = value.strip()
    if lane_id != value or not lane_id:
        return False
    lane_id_lower = lane_id.lower()
    return (
        lane_id_lower not in _RESERVED_LANE_IDS
        and _LANE_ID_RE.fullmatch(lane_id) is not None
    )


def _validate_no_shell_argv0_expansion(name: str, value: str) -> None:
    """Reject notes where unescaped ``$0`` was expanded by the caller shell.

    This protects dispatch ledgers from losing cost signal such as ``$0.50``
    turning into ``/bin/zsh.50`` when a user passes double-quoted notes.
    Legitimate shell-path notes can opt in with ``SHELL_ARGV0_OK:<reason>``.
    """

    if _SHELL_ARGV0_WAIVER in value:
        return
    if _SHELL_ARGV0_EXPANSION_RE.search(value):
        raise SystemExit(
            "VALIDATION_ERROR: "
            f"{name} appears to contain shell-expanded $0/argv[0] text "
            "(for example /bin/zsh from an unescaped dollar-cost note). "
            "Pass dollar costs in single quotes or escape '$'. "
            "If this is an intentional shell path, add SHELL_ARGV0_OK:<reason>."
        )


def _validate_claim_inputs(args: argparse.Namespace) -> None:
    _validate_lane_id(args.lane_id)
    _validate_cell("--platform", args.platform, allow_space=False)
    _validate_cell("--instance-job-id", args.instance_job_id, allow_space=False)
    _validate_cell("--agent", args.agent, allow_space=False)
    _validate_cell("--status", args.status, allow_space=False)
    if args.predicted_eta_utc:
        _validate_cell("--predicted-eta-utc", args.predicted_eta_utc, allow_space=False)
    if args.notes is not None:
        notes = _validate_cell("--notes", args.notes, allow_empty=True, allow_space=True)
        _validate_no_shell_argv0_expansion("--notes", notes)
    if args.allow_parallel:
        if not args.child_of:
            raise SystemExit("REFUSING_DISPATCH: --allow-parallel requires --child-of")
        if not args.parallel_reason:
            raise SystemExit("REFUSING_DISPATCH: --allow-parallel requires --parallel-reason")
        _validate_cell("--child-of", args.child_of, allow_space=False)
        parallel_reason = _validate_cell(
            "--parallel-reason", args.parallel_reason, allow_empty=False, allow_space=True
        )
        _validate_no_shell_argv0_expansion("--parallel-reason", parallel_reason)


def _claim_to_row(claim: Claim) -> str:
    cells = (
        claim.timestamp_utc,
        claim.agent,
        claim.lane_id,
        claim.platform,
        claim.instance_job_id,
        claim.predicted_eta_utc,
        claim.status,
        claim.notes,
    )
    return "| " + " | ".join(_escape_cell(c) for c in cells) + " |"


def _parse_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "agent" in line and "lane_id" in line:
            continue
        if _SEPARATOR_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        claims.append(Claim(*cells[:8]))
    return claims


def _is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _latest_claims_by_job(claims: list[Claim]) -> dict[tuple[str, str], Claim]:
    """Return the newest claim row for each ``(lane_id, instance_job_id)``."""

    latest: dict[tuple[str, str], Claim] = {}
    for claim in claims:
        key = (claim.lane_id, claim.instance_job_id)
        prev = latest.get(key)
        if prev is None:
            latest[key] = claim
            continue
        claim_ts = _parse_utc(claim.timestamp_utc)
        prev_ts = _parse_utc(prev.timestamp_utc)
        if prev_ts is None or (claim_ts is not None and claim_ts > prev_ts):
            latest[key] = claim
    return latest


def _claim_age_hours(now_utc: dt.datetime, claim: Claim) -> float | None:
    ts = _parse_utc(claim.timestamp_utc)
    if ts is None:
        return None
    return max((now_utc - ts).total_seconds() / 3600.0, 0.0)


def _claim_is_stale_nonterminal(
    claim: Claim, *, now_utc: dt.datetime, ttl: dt.timedelta
) -> bool:
    if _is_terminal(claim.status):
        return False
    ts = _parse_utc(claim.timestamp_utc)
    return ts is None or now_utc - ts > ttl


def _is_stale_terminal_status(status: str) -> bool:
    return _is_terminal(status) and status.startswith("stale_")


def _summarize_claims(
    claims: list[Claim], *, now_utc: dt.datetime, ttl_hours: float
) -> dict[str, object]:
    """Summarize active and stale nonterminal dispatch claims.

    This is intentionally read-only. It mirrors the conflict logic used by the
    ``claim`` subcommand: a newer terminal row closes an older nonterminal row
    for the same ``(lane_id, instance_job_id)``.
    """

    ttl = dt.timedelta(hours=ttl_hours)
    active: list[dict[str, object]] = []
    stale_nonterminal: list[dict[str, object]] = []
    terminal_latest: list[dict[str, object]] = []
    unparsable_timestamp: list[dict[str, object]] = []
    invalid_lane_id: list[dict[str, object]] = []

    for claim in _latest_claims_by_job(claims).values():
        row = asdict(claim)
        row["terminal"] = _is_terminal(claim.status)
        row["age_hours"] = _claim_age_hours(now_utc, claim)
        ts = _parse_utc(claim.timestamp_utc)
        if ts is None:
            unparsable_timestamp.append(row)
        if not _lane_id_is_valid(claim.lane_id):
            invalid_lane_id.append(row)
        if _is_terminal(claim.status):
            terminal_latest.append(row)
        elif ts is None or now_utc - ts > ttl:
            stale_nonterminal.append(row)
        else:
            active.append(row)

    active.sort(key=lambda row: (str(row["lane_id"]), str(row["instance_job_id"])))
    stale_nonterminal.sort(key=lambda row: (str(row["lane_id"]), str(row["instance_job_id"])))
    terminal_latest.sort(key=lambda row: (str(row["lane_id"]), str(row["instance_job_id"])))
    invalid_lane_id.sort(key=lambda row: (str(row["lane_id"]), str(row["instance_job_id"])))
    return {
        "schema": "pact.dispatch_claim_summary.v1",
        "now_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": ttl_hours,
        "total_latest_jobs": len(active) + len(stale_nonterminal) + len(terminal_latest),
        "active_count": len(active),
        "stale_nonterminal_count": len(stale_nonterminal),
        "terminal_latest_count": len(terminal_latest),
        "unparsable_timestamp_count": len(unparsable_timestamp),
        "invalid_lane_id_count": len(invalid_lane_id),
        "active": active,
        "stale_nonterminal": stale_nonterminal,
        "terminal_latest": terminal_latest,
        "unparsable_timestamp": unparsable_timestamp,
        "invalid_lane_id": invalid_lane_id,
    }


def _load_claims_with_archives(
    claims_path: Path, archive_dir: Path | None
) -> list[Claim]:
    """Load claims from the live ledger plus every monthly archive (if any).

    Used by ``summary --live-only=False`` (the default) to preserve forensic
    visibility into pruned rows.
    """

    text = claims_path.read_text() if claims_path.exists() else HEADER
    claims = list(_parse_claims(text))
    if archive_dir is None or not archive_dir.is_dir():
        return claims
    for archive in sorted(archive_dir.glob("dispatch_claims_*.md")):
        try:
            archive_text = archive.read_text()
        except OSError:
            continue
        claims.extend(_parse_claims(archive_text))
    return claims


def _summary(args: argparse.Namespace) -> int:
    now_utc = _parse_utc(args.now_utc) if args.now_utc else _utc_now()
    if now_utc is None:
        print(f"VALIDATION_ERROR: --now-utc is not ISO-8601: {args.now_utc!r}", file=sys.stderr)
        return 2
    archive_dir = (
        None
        if getattr(args, "live_only", False)
        else args.archive_dir
        if args.archive_dir is not None
        else DEFAULT_ARCHIVE_DIR
        if args.claims_path == DEFAULT_CLAIMS_PATH
        else None
    )
    claims = _load_claims_with_archives(args.claims_path, archive_dir)
    summary = _summarize_claims(
        claims, now_utc=now_utc, ttl_hours=args.ttl_hours
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(
        "CLAIM_SUMMARY "
        f"active={summary['active_count']} "
        f"stale_nonterminal={summary['stale_nonterminal_count']} "
        f"terminal_latest={summary['terminal_latest_count']} "
        f"unparsable_timestamp={summary['unparsable_timestamp_count']} "
        f"invalid_lane_id={summary['invalid_lane_id_count']}"
    )
    for row in summary["active"]:
        print(
            "ACTIVE "
            f"lane_id={row['lane_id']} "
            f"job={row['instance_job_id']} "
            f"platform={row['platform']} "
            f"status={row['status']} "
            f"agent={row['agent']}"
        )
    for row in summary["stale_nonterminal"]:
        print(
            "STALE_NONTERMINAL "
            f"lane_id={row['lane_id']} "
            f"job={row['instance_job_id']} "
            f"platform={row['platform']} "
            f"status={row['status']} "
            f"agent={row['agent']}"
        )
    return 0


@contextlib.contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _claim(args: argparse.Namespace) -> int:
    _validate_claim_inputs(args)

    now_utc = _parse_utc(args.now_utc) if args.now_utc else _utc_now()
    timestamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    new_claim = Claim(
        timestamp_utc=timestamp,
        agent=args.agent,
        lane_id=args.lane_id,
        platform=args.platform,
        instance_job_id=args.instance_job_id,
        predicted_eta_utc=args.predicted_eta_utc or "",
        status=args.status,
        notes=args.notes or "",
    )

    claims_path: Path = args.claims_path
    lock_path = claims_path.with_suffix(claims_path.suffix + ".lock")

    with _file_lock(lock_path):
        text = claims_path.read_text() if claims_path.exists() else HEADER

        existing = _parse_claims(text)

        # conflict detection: a (lane_id, instance_job_id) is active iff its MOST
        # RECENT row is non-terminal. A newer terminal row closes the older
        # nonterminal one (spec from CLAUDE.md + active_lane_dispatch_claims.md).
        #
        # Stale nonterminal rows still represent unresolved paid-work custody.
        # They must be explicitly closed with a terminal stale_* row before any
        # new/dry-run same-lane claim is allowed; ignoring TTL-stale rows was a
        # dispatch-refire footgun.
        ttl = dt.timedelta(hours=args.ttl_hours)
        latest_status_by_job = {
            key: c
            for key, c in _latest_claims_by_job(existing).items()
            if key[0] == args.lane_id
        }

        stale_conflict: list[Claim] = [
            c
            for c in latest_status_by_job.values()
            if _claim_is_stale_nonterminal(c, now_utc=now_utc, ttl=ttl)
        ]
        conflict: list[Claim] = [
            c
            for c in latest_status_by_job.values()
            if not _is_terminal(c.status)
            and not _claim_is_stale_nonterminal(c, now_utc=now_utc, ttl=ttl)
        ]
        closed_instance_job_ids: set[tuple[str, str]] = {
            key for key, c in latest_status_by_job.items() if _is_terminal(c.status)
        }
        if args.child_of and (args.lane_id, args.child_of) in closed_instance_job_ids:
            print(
                f"REFUSING_DISPATCH: --child-of {args.child_of} references a "
                f"terminal claim row for lane_id={args.lane_id}; that dispatch "
                f"already closed (terminal status) so children cannot attach",
                file=sys.stderr,
            )
            return 2
        parent_matches = bool(args.child_of) and any(
            c.instance_job_id == args.child_of for c in conflict
        )

        is_terminal_new = _is_terminal(args.status)
        is_stale_terminal_new = _is_stale_terminal_status(args.status)
        stale_closes_target_job = any(
            c.instance_job_id == args.instance_job_id for c in stale_conflict
        )
        parallel_allowed = args.allow_parallel and parent_matches

        if stale_conflict and not (is_stale_terminal_new and stale_closes_target_job):
            print(
                f"REFUSING_DISPATCH: stale active claim(s) require explicit "
                f"terminal stale_* closure before new/dry-run claims for "
                f"lane_id={args.lane_id}",
                file=sys.stderr,
            )
            for c in stale_conflict:
                age = _claim_age_hours(now_utc, c)
                age_text = "unparsable" if age is None else f"{age:.2f}h"
                print(
                    f"  {c.timestamp_utc} {c.agent} job={c.instance_job_id} "
                    f"status={c.status} age={age_text}",
                    file=sys.stderr,
                )
            print(
                "Close stale work first, e.g. status=stale_superseded_* or "
                "stale_assumed_dead_* with the same --instance-job-id.",
                file=sys.stderr,
            )
            return 3

        if conflict and not is_terminal_new and not args.force and not parallel_allowed:
            print(
                f"REFUSING_DISPATCH: active claim(s) already exist for lane_id={args.lane_id}",
                file=sys.stderr,
            )
            for c in conflict:
                print(
                    f"  {c.timestamp_utc} {c.agent} job={c.instance_job_id} status={c.status}",
                    file=sys.stderr,
                )
            if args.allow_parallel and not parent_matches:
                print(
                    f"REFUSING_DISPATCH: --child-of does not match an active claim for lane_id={args.lane_id}",
                    file=sys.stderr,
                )
            return 3

        new_row = _claim_to_row(new_claim)

        if args.dry_run:
            print(new_row)
            return 0

        # insert new row at top of claims table
        lines = text.splitlines(keepends=True)
        insert_at = None
        for idx, line in enumerate(lines):
            if _SEPARATOR_RE.match(line.rstrip()):
                insert_at = idx + 1
                break
        if insert_at is None:
            text = HEADER + new_row + "\n"
        else:
            lines.insert(insert_at, new_row + "\n")
            text = "".join(lines)

        claims_path.parent.mkdir(parents=True, exist_ok=True)
        with open(claims_path, "w") as f:
            f.write(text)

    parts = [
        f"CLAIM_RECORDED lane_id={args.lane_id}",
        f"platform={args.platform}",
        f"job={args.instance_job_id}",
        f"status={args.status}",
    ]
    if args.child_of:
        parts.append(f"child_of={args.child_of}")
    if args.parallel_reason:
        parts.append(f"parallel_reason={args.parallel_reason}")
    print(" ".join(parts))
    return 0


def _archive_month_key(claim: Claim) -> str | None:
    """Return ``YYYY-MM`` derived from the row timestamp, or None if unparsable."""

    ts = _parse_utc(claim.timestamp_utc)
    if ts is None:
        return None
    return ts.strftime("%Y-%m")


def _archive_path_for_month(archive_dir: Path, month_key: str) -> Path:
    return archive_dir / f"dispatch_claims_{month_key}.md"


def _read_existing_archive(path: Path) -> tuple[list[Claim], list[str]]:
    """Return (parsed claims, raw existing-archive lines) for an archive file.

    If the archive does not exist, returns ([], []) — caller is responsible
    for writing the header.
    """

    if not path.exists():
        return [], []
    text = path.read_text()
    return _parse_claims(text), text.splitlines(keepends=True)


def _plan_prune(
    claims: list[Claim],
    *,
    now_utc: dt.datetime,
    terminal_age_days: float,
    ttl_hours: float,
) -> tuple[list[Claim], list[Claim], list[Claim]]:
    """Partition ``claims`` into (keep, prune, ambiguous).

    Pruning rule:
      - Identify (lane_id, job_id) pairs whose LATEST row is terminal AND
        older than ``terminal_age_days``.
      - All rows belonging to those pairs are eligible to prune.
      - Active or stale-nonterminal pairs (including their entire history)
        are kept on the live ledger — pruning would orphan the active
        custody trail.
      - Pairs whose latest row is terminal but recent (<= terminal_age_days)
        are kept.
      - Unparsable-timestamp rows are surfaced as ambiguous.
    """

    cutoff = now_utc - dt.timedelta(days=terminal_age_days)
    latest = _latest_claims_by_job(claims)
    prunable_keys: set[tuple[str, str]] = set()
    ambiguous_keys: set[tuple[str, str]] = set()
    for key, latest_claim in latest.items():
        ts = _parse_utc(latest_claim.timestamp_utc)
        if ts is None:
            ambiguous_keys.add(key)
            continue
        if _is_terminal(latest_claim.status) and ts < cutoff:
            prunable_keys.add(key)

    keep: list[Claim] = []
    prune: list[Claim] = []
    ambiguous: list[Claim] = []
    for c in claims:
        key = (c.lane_id, c.instance_job_id)
        if key in prunable_keys:
            prune.append(c)
        elif key in ambiguous_keys:
            ambiguous.append(c)
        else:
            keep.append(c)
    return keep, prune, ambiguous


def _build_archive_text_with_appended(
    existing_lines: list[str],
    new_rows: list[Claim],
    *,
    month_key: str,
) -> str:
    """Append ``new_rows`` to an archive file (newest-first within the month).

    Existing rows are PRESERVED unchanged (append-only). New rows are
    deduplicated against existing rows by their full row text and sorted
    newest-first across the merged set.
    """

    existing_claims_parsed: list[Claim] = []
    if existing_lines:
        existing_claims_parsed = _parse_claims("".join(existing_lines))

    # Build deduped set of all rows, keyed by full Claim tuple
    seen: set[tuple[str, ...]] = set()
    merged: list[Claim] = []
    for c in existing_claims_parsed + new_rows:
        key = (
            c.timestamp_utc,
            c.agent,
            c.lane_id,
            c.platform,
            c.instance_job_id,
            c.predicted_eta_utc,
            c.status,
            c.notes,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)

    # Sort newest-first; rows with unparseable timestamps land at the end
    # (after parseable rows, preserving original order among themselves).
    def _sort_key(claim: Claim) -> tuple[int, dt.datetime, str]:
        ts = _parse_utc(claim.timestamp_utc)
        if ts is None:
            return (1, dt.datetime.min.replace(tzinfo=dt.UTC), claim.timestamp_utc)
        return (0, ts, claim.timestamp_utc)

    merged.sort(key=_sort_key, reverse=True)

    header = ARCHIVE_HEADER_FORMAT.format(month_label=month_key)
    rows = "\n".join(_claim_to_row(c) for c in merged) + ("\n" if merged else "")
    return header + rows


def _rebuild_live_ledger(keep: list[Claim], ambiguous: list[Claim]) -> str:
    """Build the new live-ledger text from ``keep`` rows.

    Ambiguous rows (unparseable timestamps) are preserved in the live ledger
    so they continue to receive operator scrutiny.
    """

    seen: set[tuple[str, ...]] = set()
    merged: list[Claim] = []
    for c in keep + ambiguous:
        key = (
            c.timestamp_utc,
            c.agent,
            c.lane_id,
            c.platform,
            c.instance_job_id,
            c.predicted_eta_utc,
            c.status,
            c.notes,
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(c)

    def _sort_key(claim: Claim) -> tuple[int, dt.datetime, str]:
        ts = _parse_utc(claim.timestamp_utc)
        if ts is None:
            return (1, dt.datetime.min.replace(tzinfo=dt.UTC), claim.timestamp_utc)
        return (0, ts, claim.timestamp_utc)

    merged.sort(key=_sort_key, reverse=True)
    rows = "\n".join(_claim_to_row(c) for c in merged) + ("\n" if merged else "")
    return HEADER + rows


def _prune(args: argparse.Namespace) -> int:
    """Archive old terminal rows into monthly files, rewrite the live ledger.

    Atomicity: load + plan + write are all inside one fcntl-locked window
    on ``<claims_path>.lock`` — sister processes (other ``claim`` invocations
    OR another ``prune``) cannot interleave.
    """

    now_utc = _parse_utc(args.now_utc) if args.now_utc else _utc_now()
    if now_utc is None:
        print(
            f"VALIDATION_ERROR: --now-utc is not ISO-8601: {args.now_utc!r}",
            file=sys.stderr,
        )
        return 2
    claims_path: Path = args.claims_path
    archive_dir: Path = args.archive_dir
    lock_path = claims_path.with_suffix(claims_path.suffix + ".lock")
    archive_dir.mkdir(parents=True, exist_ok=True)

    with _file_lock(lock_path):
        text = claims_path.read_text() if claims_path.exists() else HEADER
        claims = _parse_claims(text)
        keep, prune_rows, ambiguous = _plan_prune(
            claims,
            now_utc=now_utc,
            terminal_age_days=args.terminal_age_days,
            ttl_hours=args.ttl_hours,
        )

        # Group prune rows by month for archive routing.
        by_month: dict[str, list[Claim]] = {}
        unparseable_prune: list[Claim] = []
        for c in prune_rows:
            mk = _archive_month_key(c)
            if mk is None:
                unparseable_prune.append(c)
                continue
            by_month.setdefault(mk, []).append(c)

        # Stats for the report.
        summary = {
            "schema": "pact.dispatch_claim_prune.v1",
            "now_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "terminal_age_days": args.terminal_age_days,
            "rows_total_before": len(claims),
            "rows_pruned": len(prune_rows),
            "rows_kept_active_or_recent": len(keep),
            "rows_ambiguous_unparseable_timestamp": len(ambiguous) + len(unparseable_prune),
            "months_archived": sorted(by_month.keys()),
            "archive_files": {},
            "archive_dir": str(archive_dir),
            "claims_path": str(claims_path),
            "dry_run": bool(args.dry_run),
        }

        archive_writes: list[tuple[Path, str]] = []
        for month_key, new_rows in by_month.items():
            archive_path = _archive_path_for_month(archive_dir, month_key)
            _, existing_lines = _read_existing_archive(archive_path)
            archive_text = _build_archive_text_with_appended(
                existing_lines, new_rows, month_key=month_key
            )
            archive_writes.append((archive_path, archive_text))
            summary["archive_files"][month_key] = {
                "path": str(archive_path),
                "rows_appended": len(new_rows),
            }

        live_text = _rebuild_live_ledger(keep, ambiguous + unparseable_prune)
        summary["live_ledger_bytes_after"] = len(live_text.encode("utf-8"))
        summary["live_ledger_bytes_before"] = len(text.encode("utf-8"))

        if args.dry_run:
            if args.format == "json":
                print(json.dumps(summary, indent=2, sort_keys=True))
            else:
                print(
                    "DRY_RUN_PRUNE "
                    f"total_before={summary['rows_total_before']} "
                    f"rows_pruned={summary['rows_pruned']} "
                    f"rows_kept={summary['rows_kept_active_or_recent']} "
                    f"months={summary['months_archived']}"
                )
            return 0

        # Write archives first (append-only is safe if interrupted before live
        # rewrite — the live ledger still has the rows so they aren't lost).
        for archive_path, archive_text in archive_writes:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text(archive_text, encoding="utf-8")

        # Now rewrite the live ledger.
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        claims_path.write_text(live_text, encoding="utf-8")

    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "PRUNE_APPLIED "
            f"total_before={summary['rows_total_before']} "
            f"rows_pruned={summary['rows_pruned']} "
            f"rows_kept={summary['rows_kept_active_or_recent']} "
            f"months={summary['months_archived']} "
            f"live_bytes_before={summary['live_ledger_bytes_before']} "
            f"live_bytes_after={summary['live_ledger_bytes_after']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    claim_p = sub.add_parser("claim", help="Insert a new claim row")
    claim_p.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    claim_p.add_argument("--lane-id", required=True)
    claim_p.add_argument("--platform", required=True)
    claim_p.add_argument("--instance-job-id", required=True)
    claim_p.add_argument("--agent", required=True)
    claim_p.add_argument("--predicted-eta-utc", default="")
    claim_p.add_argument("--status", default="eval")
    claim_p.add_argument("--notes", default="")
    claim_p.add_argument("--ttl-hours", type=float, default=24.0)
    claim_p.add_argument("--now-utc", default="")
    claim_p.add_argument(
        "--allow-parallel",
        action="store_true",
        help="Allow a second active claim in the same lane only when it is a bounded child of an existing active job.",
    )
    claim_p.add_argument(
        "--child-of",
        default="",
        help="Active same-lane instance/job id that this bounded child claim depends on.",
    )
    claim_p.add_argument(
        "--parallel-reason",
        default="",
        help="Short audit reason explaining why this same-lane child can run in parallel.",
    )
    claim_p.add_argument("--force", action="store_true")
    claim_p.add_argument("--dry-run", action="store_true")
    claim_p.set_defaults(func=_claim)
    summary_p = sub.add_parser(
        "summary",
        help="Read-only summary of active and stale nonterminal claim rows",
    )
    summary_p.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    summary_p.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Monthly archive dir (default: .omx/state/dispatch_claims_archive)",
    )
    summary_p.add_argument(
        "--live-only",
        action="store_true",
        help="Scan only the live ledger; ignore monthly archives (default: include archives)",
    )
    summary_p.add_argument("--ttl-hours", type=float, default=24.0)
    summary_p.add_argument("--now-utc", default="")
    summary_p.add_argument("--format", choices=["text", "json"], default="text")
    summary_p.set_defaults(func=_summary)
    prune_p = sub.add_parser(
        "prune",
        help=(
            "Archive terminal rows older than --terminal-age-days into "
            "monthly archive files and rewrite the live ledger"
        ),
    )
    prune_p.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    prune_p.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Monthly archive dir (default: .omx/state/dispatch_claims_archive)",
    )
    prune_p.add_argument(
        "--terminal-age-days",
        type=float,
        default=DEFAULT_PRUNE_TERMINAL_AGE_DAYS,
        help=f"Terminal rows older than this go to archive (default: {DEFAULT_PRUNE_TERMINAL_AGE_DAYS}d)",
    )
    prune_p.add_argument("--ttl-hours", type=float, default=24.0)
    prune_p.add_argument("--now-utc", default="")
    prune_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prune plan without rewriting the ledger or archives",
    )
    prune_p.add_argument("--format", choices=["text", "json"], default="text")
    prune_p.set_defaults(func=_prune)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
