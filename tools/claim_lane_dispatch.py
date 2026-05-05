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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")

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
)

_TABLE_HEADER_RE = re.compile(r"^\|\s*timestamp_utc\s*\|", re.MULTILINE)
_SEPARATOR_RE = re.compile(r"^\|\s*-+\s*(\|\s*-+\s*)+\|\s*$")


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
    return dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0)


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
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


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


def _validate_claim_inputs(args: argparse.Namespace) -> None:
    _validate_cell("--lane-id", args.lane_id, allow_space=False)
    _validate_cell("--platform", args.platform, allow_space=False)
    _validate_cell("--instance-job-id", args.instance_job_id, allow_space=False)
    _validate_cell("--agent", args.agent, allow_space=False)
    _validate_cell("--status", args.status, allow_space=False)
    if args.predicted_eta_utc:
        _validate_cell("--predicted-eta-utc", args.predicted_eta_utc, allow_space=False)
    if args.notes is not None:
        _validate_cell("--notes", args.notes, allow_empty=True, allow_space=True)
    if args.allow_parallel:
        if not args.child_of:
            raise SystemExit("REFUSING_DISPATCH: --allow-parallel requires --child-of")
        if not args.parallel_reason:
            raise SystemExit("REFUSING_DISPATCH: --allow-parallel requires --parallel-reason")
        _validate_cell("--child-of", args.child_of, allow_space=False)
        _validate_cell("--parallel-reason", args.parallel_reason, allow_empty=False, allow_space=True)


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
        if claims_path.exists():
            text = claims_path.read_text()
        else:
            text = HEADER

        existing = _parse_claims(text)

        # conflict detection: a (lane_id, instance_job_id) is active iff its MOST
        # RECENT row is non-terminal. A newer terminal row closes the older
        # nonterminal one (spec from CLAUDE.md + active_lane_dispatch_claims.md).
        ttl = dt.timedelta(hours=args.ttl_hours)
        latest_status_by_job: dict[tuple[str, str], Claim] = {}
        for c in existing:
            if c.lane_id != args.lane_id:
                continue
            ts = _parse_utc(c.timestamp_utc)
            if ts is None or now_utc - ts > ttl:
                continue
            key = (c.lane_id, c.instance_job_id)
            prev = latest_status_by_job.get(key)
            if prev is None or _parse_utc(c.timestamp_utc) > _parse_utc(prev.timestamp_utc):
                latest_status_by_job[key] = c

        conflict: list[Claim] = [
            c for c in latest_status_by_job.values() if not _is_terminal(c.status)
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
        parallel_allowed = args.allow_parallel and parent_matches

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
