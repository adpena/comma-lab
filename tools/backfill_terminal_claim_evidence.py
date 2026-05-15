#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Backfill no-score cathedral evidence for terminal dispatch-claim rows.

Exact CUDA auth-eval returns must go through ``tools/build_result_review_packet.py``.
This helper covers the other repeated no-signal-loss failure mode: terminal
claim rows such as stale/refused/no-provider-spend attempts that should be
visible to the autopilot evidence ledger but do not have an auth-eval JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.deploy.claims import is_terminal_status

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
DEFAULT_EVIDENCE_PATH = REPO_ROOT / "reports/cathedral_autopilot_evidence.jsonl"
CLAIM_KEYS = (
    "timestamp_utc",
    "agent",
    "lane_id",
    "platform",
    "instance_job_id",
    "predicted_eta_utc",
    "status",
    "notes",
)


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

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.lane_id, self.instance_job_id, self.status)


def parse_claim_rows(path: Path) -> list[ClaimRow]:
    rows: list[ClaimRow] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "timestamp_utc" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < len(CLAIM_KEYS):
            continue
        payload = dict(zip(CLAIM_KEYS, cells[: len(CLAIM_KEYS)], strict=True))
        rows.append(ClaimRow(**payload))
    return rows


def covered_claim_keys(evidence_path: Path) -> set[tuple[str, str, str]]:
    covered: set[tuple[str, str, str]] = set()
    if not evidence_path.is_file():
        return covered
    for line in evidence_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        lane_id = str(payload.get("lane_id") or "")
        job_id = str(payload.get("job_name") or payload.get("instance_job_id") or "")
        status = str(payload.get("dispatch_claim_latest_status") or "")
        if (
            lane_id
            and job_id
            and status
            and payload.get("exact_result_review_packet")
            and payload.get("score_claim") is False
            and payload.get("promotion_eligible") is False
            and payload.get("dispatch_claim_terminal_status_recorded") is True
        ):
            covered.add((lane_id, job_id, status))
        claims = payload.get("covered_terminal_claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            lane_id = str(claim.get("lane_id") or "")
            job_id = str(claim.get("instance_job_id") or "")
            status = str(claim.get("status") or "")
            if lane_id and job_id and status:
                covered.add((lane_id, job_id, status))
    return covered


def terminal_claim_needs_evidence(
    row: ClaimRow,
    *,
    earliest_timestamp_utc: str,
    exact_eval_earliest_timestamp_utc: str,
) -> bool:
    if row.timestamp_utc < earliest_timestamp_utc:
        return False
    if not is_terminal_status(row.status):
        return False
    exact_cuda_status = row.status.startswith("completed_contest_cuda_modal_auth_eval")
    if exact_cuda_status:
        return False
    if row.status.startswith("completed_contest_cpu_modal_auth_eval"):
        return False
    if exact_cuda_status and row.timestamp_utc < exact_eval_earliest_timestamp_utc:
        return False
    return (
        "substrate_" in row.instance_job_id
        or row.lane_id.startswith("lane_substrate")
        or row.lane_id.startswith("lane_pr95_meta_stack")
        or row.lane_id.startswith("lane_time_traveler")
        or row.lane_id.startswith("lane_sabor")
        or row.lane_id.startswith("lane_s2sbs")
        or row.lane_id.startswith("lane_a1_plus")
    )


def dispatch_attempted(row: ClaimRow) -> bool:
    status = row.status.lower()
    notes = row.notes.lower()
    no_provider_markers = (
        "no_cost",
        "no_modal_spend",
        "pre_modal_spawn",
        "pre_provider",
        "refused_dispatch",
        "missing_operator",
    )
    return not any(marker in status or marker in notes for marker in no_provider_markers)


def evidence_row(row: ClaimRow, *, claims_path: Path) -> dict[str, Any]:
    attempted = dispatch_attempted(row)
    return {
        "schema": "cathedral_autopilot_terminal_claim_evidence_v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evidence_grade": "[infrastructure terminal dispatch coverage]",
        "evidence_marker": "[infrastructure terminal dispatch coverage]",
        "evidence_semantics": (
            "terminal_dispatch_claim_no_score_no_signal_loss"
            if attempted
            else "terminal_dispatch_claim_no_provider_spend_no_score"
        ),
        "covered_terminal_claim_count": 1,
        "covered_terminal_claims": [
            {
                "lane_id": row.lane_id,
                "instance_job_id": row.instance_job_id,
                "status": row.status,
            }
        ],
        "dispatch_attempted": attempted,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "proxy_row": True,
        "family_falsified": False,
        "method_family_retired": False,
        "contest_dispatch_verdict": "no_signal_loss_terminal_claim_coverage",
        "measured_config_status": "terminal_dispatch_claim_preserved_no_score",
        "dispatch_blockers": [
            "terminal_claim_has_no_score_authority",
            "exact_result_review_required_before_promotion_or_rank_kill",
        ],
        "reactivation_criteria": [
            "classify any archive/auth-eval artifacts through an exact result-review packet",
            "do not infer model success or failure from terminal infrastructure rows",
        ],
        "source": str(claims_path),
        "claim_timestamp_utc": row.timestamp_utc,
        "claim_agent": row.agent,
        "claim_platform": row.platform,
    }


def missing_rows(
    *,
    claims_path: Path,
    evidence_path: Path,
    earliest_timestamp_utc: str,
    exact_eval_earliest_timestamp_utc: str,
) -> list[ClaimRow]:
    covered = covered_claim_keys(evidence_path)
    rows: list[ClaimRow] = []
    for row in parse_claim_rows(claims_path):
        if not terminal_claim_needs_evidence(
            row,
            earliest_timestamp_utc=earliest_timestamp_utc,
            exact_eval_earliest_timestamp_utc=exact_eval_earliest_timestamp_utc,
        ):
            continue
        if row.key in covered:
            continue
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--evidence-jsonl", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--earliest-timestamp-utc", default="2026-05-13T00:00:00Z")
    parser.add_argument("--exact-eval-earliest-timestamp-utc", default="2026-05-14T06:00:00Z")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no limit")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = missing_rows(
        claims_path=args.claims_path,
        evidence_path=args.evidence_jsonl,
        earliest_timestamp_utc=args.earliest_timestamp_utc,
        exact_eval_earliest_timestamp_utc=args.exact_eval_earliest_timestamp_utc,
    )
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    payloads = [evidence_row(row, claims_path=args.claims_path) for row in rows]
    summary = {
        "missing_terminal_claims": len(rows),
        "dry_run": args.dry_run,
        "evidence_jsonl": str(args.evidence_jsonl),
        "claims_path": str(args.claims_path),
        "rows": [
            {
                "lane_id": row.lane_id,
                "instance_job_id": row.instance_job_id,
                "status": row.status,
            }
            for row in rows
        ],
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    summary["appended"] = len(payloads)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
