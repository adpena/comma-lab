#!/usr/bin/env python3
"""Atomic lane-dispatch claim helper.

Use before paid remote work:

    tools/claim_lane_dispatch.py claim --lane-id public_floor_pvl1 \
        --platform lightning --instance-job-id exact_eval_... \
        --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-02T00:20Z \
        --status eval --notes "T4 promotion"
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
TERMINAL_PREFIXES = (
    "completed_",
    "completed_score=",
    "completed_no_frontier",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
)
HEADER = """# Active lane dispatch claims -- cross-agent coordination ledger

**MANDATORY for ALL agents (Claude, codex, future):** Before dispatching ANY training, eval, or remote-GPU job, READ this file FIRST and APPEND a claim row. Check for conflicts on the same `lane_id` within the past 24h.

## Claims (newest first)

| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |
|---|---|---|---|---|---|---|---|
"""


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
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _parse_utc(value: str) -> dt.datetime | None:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _escape_cell(value: str) -> str:
    return " ".join(value.replace("|", "/").split())


def _validate_cell(name: str, value: str, *, allow_empty: bool = False, allow_space: bool = False) -> None:
    if not isinstance(value, str):
        raise SystemExit(f"{name} must be a string")
    if value != value.strip():
        raise SystemExit(f"{name} must not have leading/trailing whitespace")
    if not value and not allow_empty:
        raise SystemExit(f"{name} must not be empty")
    if "|" in value:
        raise SystemExit(f"{name} must not contain markdown table separators")
    if any(ord(ch) < 32 for ch in value):
        raise SystemExit(f"{name} must not contain control characters")
    if not allow_space and any(ch.isspace() for ch in value):
        raise SystemExit(f"{name} must not contain whitespace")


def _validate_claim_inputs(args: argparse.Namespace) -> None:
    _validate_cell("agent", args.agent)
    _validate_cell("lane_id", args.lane_id)
    _validate_cell("platform", args.platform)
    _validate_cell("instance_job_id", args.instance_job_id)
    _validate_cell("predicted_eta_utc", args.predicted_eta_utc)
    _validate_cell("status", args.status)
    _validate_cell("notes", args.notes, allow_empty=True, allow_space=True)
    if args.child_of is not None:
        _validate_cell("child_of", args.child_of)
    if args.parallel_reason is not None:
        _validate_cell("parallel_reason", args.parallel_reason, allow_space=True)


def _claim_to_row(claim: Claim) -> str:
    cells = [
        claim.timestamp_utc,
        claim.agent,
        claim.lane_id,
        claim.platform,
        claim.instance_job_id,
        claim.predicted_eta_utc,
        claim.status,
        claim.notes,
    ]
    return "| " + " | ".join(_escape_cell(c) for c in cells) + " |\n"


def _parse_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[0] in {"timestamp_utc", "---"}:
            continue
        claims.append(Claim(*cells))
    return claims


def _is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _active_conflicts(
    claims: list[Claim],
    *,
    lane_id: str,
    now: dt.datetime,
    ttl_hours: float,
) -> list[Claim]:
    cutoff = now - dt.timedelta(hours=ttl_hours)
    conflicts: list[Claim] = []
    closed_instance_job_ids: set[str] = set()
    for claim in claims:
        if claim.lane_id != lane_id:
            continue
        if _is_terminal(claim.status):
            closed_instance_job_ids.add(claim.instance_job_id)
            continue
        if claim.instance_job_id in closed_instance_job_ids:
            continue
        ts = _parse_utc(claim.timestamp_utc)
        if ts is None or ts >= cutoff:
            conflicts.append(claim)
    return conflicts


@contextlib.contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.seek(0)
        yield fh
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def claim(args: argparse.Namespace) -> int:
    _validate_claim_inputs(args)
    path = args.claims_path
    now = _parse_utc(args.now_utc) if args.now_utc else _utc_now()
    if now is None:
        raise SystemExit(f"invalid --now-utc: {args.now_utc}")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    new_claim = Claim(
        timestamp_utc=timestamp,
        agent=args.agent,
        lane_id=args.lane_id,
        platform=args.platform,
        instance_job_id=args.instance_job_id,
        predicted_eta_utc=args.predicted_eta_utc,
        status=args.status,
        notes=args.notes,
    )
    with _locked_file(path) as fh:
        text = fh.read()
        if not text.strip():
            text = HEADER
        conflicts = _active_conflicts(
            _parse_claims(text),
            lane_id=args.lane_id,
            now=now,
            ttl_hours=args.ttl_hours,
        )
        parallel_allowed = False
        if conflicts and args.allow_parallel:
            if not args.child_of:
                print(
                    "REFUSING_DISPATCH: --allow-parallel requires --child-of",
                    file=sys.stderr,
                )
                return 2
            if not args.parallel_reason:
                print(
                    "REFUSING_DISPATCH: --allow-parallel requires --parallel-reason",
                    file=sys.stderr,
                )
                return 2
            parent_matches = [
                conflict
                for conflict in conflicts
                if conflict.instance_job_id == args.child_of
            ]
            if not parent_matches:
                print(
                    "REFUSING_DISPATCH: --child-of does not match an active "
                    f"claim for lane_id={args.lane_id}: {args.child_of}",
                    file=sys.stderr,
                )
                return 2
            parallel_allowed = True
        if conflicts and not args.force and not parallel_allowed:
            print(
                f"REFUSING_DISPATCH: active claim(s) already exist for lane_id={args.lane_id}",
                file=sys.stderr,
            )
            for conflict in conflicts:
                print(
                    f"  {conflict.timestamp_utc} {conflict.agent} "
                    f"{conflict.platform} {conflict.instance_job_id} "
                    f"status={conflict.status}",
                    file=sys.stderr,
                )
            return 2
        if parallel_allowed:
            new_claim = Claim(
                timestamp_utc=new_claim.timestamp_utc,
                agent=new_claim.agent,
                lane_id=new_claim.lane_id,
                platform=new_claim.platform,
                instance_job_id=new_claim.instance_job_id,
                predicted_eta_utc=new_claim.predicted_eta_utc,
                status=new_claim.status,
                notes=(
                    f"{new_claim.notes} child_of={args.child_of} "
                    f"parallel_reason={args.parallel_reason}"
                ).strip(),
            )
        if args.dry_run:
            print(_claim_to_row(new_claim).rstrip())
            return 0
        lines = text.splitlines(keepends=True)
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line.startswith("|---|"):
                insert_at = idx + 1
                break
        lines.insert(insert_at, _claim_to_row(new_claim))
        fh.seek(0)
        fh.truncate()
        fh.write("".join(lines))
    print(f"CLAIM_RECORDED lane_id={args.lane_id} platform={args.platform} job={args.instance_job_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    claim_p = sub.add_parser("claim")
    claim_p.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    claim_p.add_argument("--lane-id", required=True)
    claim_p.add_argument("--platform", required=True)
    claim_p.add_argument("--instance-job-id", required=True)
    claim_p.add_argument("--agent", required=True)
    claim_p.add_argument("--predicted-eta-utc", required=True)
    claim_p.add_argument("--status", default="eval")
    claim_p.add_argument("--notes", default="")
    claim_p.add_argument("--ttl-hours", type=float, default=24.0)
    claim_p.add_argument("--now-utc")
    claim_p.add_argument(
        "--allow-parallel",
        action="store_true",
        help=(
            "Allow a second active claim in the same lane only when it is a "
            "bounded child of an existing active job."
        ),
    )
    claim_p.add_argument(
        "--child-of",
        help="Active same-lane instance/job id that this bounded child claim depends on.",
    )
    claim_p.add_argument(
        "--parallel-reason",
        help="Short audit reason explaining why this same-lane child can run in parallel.",
    )
    claim_p.add_argument("--force", action="store_true")
    claim_p.add_argument("--dry-run", action="store_true")
    claim_p.set_defaults(func=claim)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
