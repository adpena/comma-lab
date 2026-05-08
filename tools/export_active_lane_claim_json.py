#!/usr/bin/env python3
"""Export the newest active Level-2 dispatch claim as JSON.

This is a read-only bridge from `.omx/state/active_lane_dispatch_claims.md` to
candidate builders that require structured lane-claim proof. It does not claim,
close, or dispatch any job.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
SCHEMA = "tac_active_lane_claim_json_v1"
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
_SEPARATOR_RE = re.compile(r"^\|\s*-+\s*(\|\s*-+\s*)+\|\s*$")


@dataclass(frozen=True)
class ClaimRow:
    timestamp_utc: str
    agent: str
    lane_id: str
    platform: str
    instance_job_id: str
    predicted_eta_utc: str
    status: str
    notes: str
    raw_row: str = ""


def build_active_lane_claim_json(
    *,
    claims_path: Path,
    lane_id: str,
    instance_job_id: str | None = None,
    now_utc: str | None = None,
    ttl_hours: float = 24.0,
) -> dict[str, Any]:
    """Return structured proof for the newest active matching claim."""

    now = _parse_now(now_utc)
    claims = _parse_claim_rows(claims_path)
    selected, skipped = _select_active_claim(
        claims,
        lane_id=lane_id,
        instance_job_id=instance_job_id,
        now=now,
        ttl_hours=ttl_hours,
    )
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "active": selected is not None,
        "lane_id": lane_id,
        "instance_job_id": instance_job_id or "",
        "claims_path": str(claims_path),
        "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_hours": ttl_hours,
        "claim_count": len(claims),
        "skipped_matching_rows": skipped,
        "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim",
        "claims_file_sha256": _sha256_file(claims_path) if claims_path.exists() else "",
    }
    if selected is None:
        return {
            **base,
            "active": False,
            "claim_status": "",
            "status": "",
            "platform": "",
            "agent": "",
            "timestamp_utc": "",
            "predicted_eta_utc": "",
            "blockers": ["active_lane_claim_not_found"],
        }
    return {
        **base,
        "active": True,
        "lane_id": selected.lane_id,
        "instance_job_id": selected.instance_job_id,
        "claim_status": selected.status,
        "status": selected.status,
        "platform": selected.platform,
        "agent": selected.agent,
        "timestamp_utc": selected.timestamp_utc,
        "predicted_eta_utc": selected.predicted_eta_utc,
        "notes": selected.notes,
        "claim_row_sha256": _sha256_text(selected.raw_row),
        "claim_source": "canonical_claim_file",
        "blockers": [],
    }


def _select_active_claim(
    claims: list[ClaimRow],
    *,
    lane_id: str,
    instance_job_id: str | None,
    now: dt.datetime,
    ttl_hours: float,
) -> tuple[ClaimRow | None, list[dict[str, str]]]:
    closed_jobs: set[str] = set()
    skipped: list[dict[str, str]] = []
    cutoff = now - dt.timedelta(hours=ttl_hours)
    for claim in claims:
        if claim.lane_id != lane_id:
            continue
        if instance_job_id is not None and claim.instance_job_id != instance_job_id:
            continue
        parsed_ts = _parse_utc(claim.timestamp_utc)
        if parsed_ts is None:
            skipped.append(_skip(claim, "invalid_timestamp"))
            continue
        if parsed_ts < cutoff:
            skipped.append(_skip(claim, "outside_ttl"))
            continue
        if claim.instance_job_id in closed_jobs:
            skipped.append(_skip(claim, "closed_by_newer_terminal_row"))
            continue
        if _is_terminal(claim.status):
            closed_jobs.add(claim.instance_job_id)
            skipped.append(_skip(claim, "terminal_status"))
            continue
        return claim, skipped
    return None, skipped


def _parse_claim_rows(claims_path: Path) -> list[ClaimRow]:
    if not claims_path.exists():
        return []
    claims: list[ClaimRow] = []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "agent" in line and "lane_id" in line:
            continue
        if _SEPARATOR_RE.match(line):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        claims.append(ClaimRow(*cells[:8], raw_row=line))
    return claims


def _skip(claim: ClaimRow, reason: str) -> dict[str, str]:
    return {
        "timestamp_utc": claim.timestamp_utc,
        "lane_id": claim.lane_id,
        "instance_job_id": claim.instance_job_id,
        "status": claim.status,
        "reason": reason,
    }


def _parse_now(now_utc: str | None) -> dt.datetime:
    if now_utc:
        parsed = _parse_utc(now_utc)
        if parsed is None:
            raise SystemExit(f"invalid --now-utc: {now_utc}")
        return parsed
    return dt.datetime.now(tz=dt.UTC).replace(microsecond=0)


def _parse_utc(value: str) -> dt.datetime | None:
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _is_terminal(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_PREFIXES)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps_json(payload: dict[str, Any]) -> str:
    """Return stable pretty JSON for file or stdout output."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--instance-job-id")
    parser.add_argument("--now-utc")
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-if-missing", action="store_true")
    args = parser.parse_args(argv)

    payload = build_active_lane_claim_json(
        claims_path=args.claims_path,
        lane_id=args.lane_id,
        instance_job_id=args.instance_job_id,
        now_utc=args.now_utc,
        ttl_hours=args.ttl_hours,
    )
    text = dumps_json(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.fail_if_missing and payload["active"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
