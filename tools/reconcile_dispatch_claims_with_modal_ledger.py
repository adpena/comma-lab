#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reconcile active dispatch claims against the Modal call-id ledger.

The claim ledger is intentionally conservative: a job stays active until a
newer terminal claim row closes the same ``(lane_id, instance/job_id)``. Modal
harvesters sometimes append terminal call-id evidence without closing the
original human-readable claim row. This tool makes that mismatch visible and,
with ``--execute``, appends only high-confidence terminal closure rows.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct execution from tools/
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool


REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

import tools.claim_lane_dispatch as claim_dispatch  # noqa: E402


DEFAULT_CLAIMS_PATH = Path(".omx/state/active_lane_dispatch_claims.md")
DEFAULT_MODAL_LEDGER = Path(".omx/state/modal_call_id_ledger.jsonl")
CALL_ID_RE = re.compile(r"\bcall_id=(fc-[A-Z0-9]+)\b")
TERMINAL_LEDGER_STATUSES = frozenset({"harvested", "failed", "pre_spawn_fatal"})


@dataclass(frozen=True)
class ModalEvent:
    row_index: int
    payload: dict[str, Any]

    @property
    def call_id(self) -> str:
        return str(self.payload.get("call_id") or "")

    @property
    def label(self) -> str:
        return str(self.payload.get("label") or "")

    @property
    def lane_id(self) -> str:
        return str(self.payload.get("lane_id") or "")

    @property
    def status(self) -> str:
        return str(self.payload.get("status") or self.payload.get("event_type") or "")

    @property
    def event_type(self) -> str:
        return str(self.payload.get("event_type") or "")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_LEDGER_STATUSES or self.event_type in TERMINAL_LEDGER_STATUSES


@dataclass(frozen=True)
class Reconciliation:
    claim: claim_dispatch.Claim
    match_kind: str
    call_id: str
    terminal_event: ModalEvent | None
    suggested_status: str | None
    confidence: str
    reason: str
    command: list[str] | None

    def to_json(self) -> dict[str, Any]:
        row = {
            "claim": asdict(self.claim),
            "match_kind": self.match_kind,
            "call_id": self.call_id,
            "terminal_event": self.terminal_event.payload if self.terminal_event else None,
            "suggested_status": self.suggested_status,
            "confidence": self.confidence,
            "reason": self.reason,
            "command": self.command,
        }
        return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
    parser.add_argument("--modal-ledger", type=Path, default=DEFAULT_MODAL_LEDGER)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Append high-confidence terminal closure rows via claim_lane_dispatch.py.",
    )
    return parser


def _read_claims(path: Path) -> list[claim_dispatch.Claim]:
    text = path.read_text(encoding="utf-8") if path.exists() else claim_dispatch.HEADER
    return claim_dispatch._parse_claims(text)


def _read_modal_events(path: Path) -> list[ModalEvent]:
    events: list[ModalEvent] = []
    if not path.exists():
        return events
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(ModalEvent(row_index=row_index, payload=payload))
    return events


def _latest_by_call_id(events: list[ModalEvent]) -> dict[str, ModalEvent]:
    latest: dict[str, ModalEvent] = {}
    for event in events:
        if event.call_id:
            latest[event.call_id] = event
    return latest


def _latest_terminal_by_call_id(events: list[ModalEvent]) -> dict[str, ModalEvent]:
    latest: dict[str, ModalEvent] = {}
    for event in events:
        if event.call_id and event.terminal:
            latest[event.call_id] = event
    return latest


def _latest_dispatched_by_lane_label(events: list[ModalEvent]) -> dict[tuple[str, str], ModalEvent]:
    latest: dict[tuple[str, str], ModalEvent] = {}
    for event in events:
        if event.lane_id and event.label:
            latest[(event.lane_id, event.label)] = event
    return latest


def _call_id_from_claim(claim: claim_dispatch.Claim) -> str:
    if claim.instance_job_id.startswith("fc-"):
        return claim.instance_job_id
    match = CALL_ID_RE.search(claim.notes)
    return match.group(1) if match else ""


def _status_from_event(event: ModalEvent) -> str:
    payload = event.payload
    rc = payload.get("rc")
    evidence_grade = str(payload.get("evidence_grade") or "")
    score_axis = str(payload.get("score_axis") or "")
    status = event.status
    if status == "pre_spawn_fatal":
        return f"failed_pre_spawn_fatal_rc_{rc}" if rc is not None else "failed_pre_spawn_fatal"
    if status == "failed" or (isinstance(rc, int) and rc != 0):
        return f"failed_modal_call_recovered_rc_{rc}" if rc is not None else "failed_modal_call_recovered"
    if evidence_grade.startswith("contest-") or score_axis.startswith("contest_"):
        return "completed_modal_auth_eval_recovered"
    return "completed_modal_training_recovered_no_score_claim"


def _note_from_event(event: ModalEvent, match_kind: str) -> str:
    payload = event.payload
    parts = [
        f"Codex reconciliation: terminal Modal evidence matched by {match_kind}",
        f"call_id={event.call_id}",
        f"ledger_row={event.row_index}",
        f"event_status={event.status}",
    ]
    for key in (
        "rc",
        "elapsed_seconds",
        "evidence_grade",
        "score_axis",
        "score_recomputed_from_components",
        "archive_sha256",
        "archive_bytes",
        "harvested_at_utc",
        "failure_class",
    ):
        value = payload.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts)


def _claim_command(
    *,
    repo_root: Path,
    claim: claim_dispatch.Claim,
    status: str,
    notes: str,
) -> list[str]:
    return [
        str(repo_root / ".venv/bin/python"),
        str(repo_root / "tools/claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        claim.lane_id,
        "--platform",
        claim.platform,
        "--instance-job-id",
        claim.instance_job_id,
        "--agent",
        "codex:dispatch_claim_reconciler",
        "--status",
        status,
        "--notes",
        notes,
        "--force",
    ]


def reconcile(
    *,
    repo_root: Path,
    claims_path: Path,
    modal_ledger: Path,
) -> list[Reconciliation]:
    claims = _read_claims(claims_path)
    active_claims = [
        claim
        for claim in claim_dispatch._latest_claims_by_job(claims).values()
        if not claim_dispatch._is_terminal(claim.status) and claim.platform == "modal"
    ]
    events = _read_modal_events(modal_ledger)
    latest_by_call_id = _latest_by_call_id(events)
    terminal_by_call_id = _latest_terminal_by_call_id(events)
    dispatched_by_lane_label = _latest_dispatched_by_lane_label(events)

    rows: list[Reconciliation] = []
    for claim in sorted(active_claims, key=lambda c: (c.lane_id, c.instance_job_id)):
        call_id = _call_id_from_claim(claim)
        match_kind = "claim_call_id"
        if not call_id:
            dispatched = dispatched_by_lane_label.get((claim.lane_id, claim.instance_job_id))
            if dispatched is not None:
                call_id = dispatched.call_id
                match_kind = "lane_label_dispatched_event"

        terminal = terminal_by_call_id.get(call_id) if call_id else None
        if terminal is None and call_id and call_id in latest_by_call_id:
            latest = latest_by_call_id[call_id]
            rows.append(
                Reconciliation(
                    claim=claim,
                    match_kind=match_kind,
                    call_id=call_id,
                    terminal_event=None,
                    suggested_status=None,
                    confidence="active_or_unharvested",
                    reason=f"Modal call_id latest status is nonterminal: {latest.status}",
                    command=None,
                )
            )
            continue
        if terminal is None:
            rows.append(
                Reconciliation(
                    claim=claim,
                    match_kind="no_modal_call_id_match",
                    call_id=call_id,
                    terminal_event=None,
                    suggested_status=None,
                    confidence="manual_review",
                    reason="No call_id was recoverable from the claim row or Modal ledger label.",
                    command=None,
                )
            )
            continue

        status = _status_from_event(terminal)
        notes = _note_from_event(terminal, match_kind)
        rows.append(
            Reconciliation(
                claim=claim,
                match_kind=match_kind,
                call_id=call_id,
                terminal_event=terminal,
                suggested_status=status,
                confidence="high",
                reason="Terminal Modal ledger evidence exists for this exact claim call_id/label.",
                command=_claim_command(
                    repo_root=repo_root,
                    claim=claim,
                    status=status,
                    notes=notes,
                ),
            )
        )
    return rows


def _write_json(path: Path, rows: list[Reconciliation], executed: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pact.dispatch_claim_modal_reconciliation.v1",
        "generated_at_utc": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "high_confidence_terminal_count": sum(1 for row in rows if row.confidence == "high"),
        "manual_review_count": sum(1 for row in rows if row.confidence == "manual_review"),
        "active_or_unharvested_count": sum(1 for row in rows if row.confidence == "active_or_unharvested"),
        "executed": executed,
        "rows": [row.to_json() for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, rows: list[Reconciliation], executed: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    high = [row for row in rows if row.confidence == "high"]
    manual = [row for row in rows if row.confidence == "manual_review"]
    active = [row for row in rows if row.confidence == "active_or_unharvested"]
    lines = [
        "# Dispatch Claim × Modal Ledger Reconciliation",
        "",
        f"- Generated: {datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z')}",
        f"- High-confidence terminal closures: {len(high)}",
        f"- Manual-review rows: {len(manual)}",
        f"- Still active/unharvested rows: {len(active)}",
        f"- Executed closures: {len(executed)}",
        "",
        "## High-confidence terminal closures",
        "",
        "| lane_id | job | matched_call_id | suggested_status | terminal_status | rc | score_axis | score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in high:
        payload = row.terminal_event.payload if row.terminal_event else {}
        score = payload.get("score_recomputed_from_components", payload.get("score", ""))
        lines.append(
            "| "
            + " | ".join(
                [
                    row.claim.lane_id,
                    row.claim.instance_job_id,
                    row.call_id,
                    row.suggested_status or "",
                    row.terminal_event.status if row.terminal_event else "",
                    str(payload.get("rc", "")),
                    str(payload.get("score_axis", "")),
                    str(score),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Manual review / still live", ""])
    lines.append("| lane_id | job | confidence | reason |")
    lines.append("|---|---|---|---|")
    for row in [*manual, *active]:
        lines.append(
            f"| {row.claim.lane_id} | {row.claim.instance_job_id} | {row.confidence} | {row.reason} |"
        )
    if executed:
        lines.extend(["", "## Executed closure rows", ""])
        lines.append("| lane_id | job | status | returncode |")
        lines.append("|---|---|---|---|")
        for item in executed:
            lines.append(
                f"| {item['lane_id']} | {item['instance_job_id']} | {item['status']} | {item['returncode']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute(rows: list[Reconciliation]) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    for row in rows:
        if row.confidence != "high" or row.command is None or row.suggested_status is None:
            continue
        proc = subprocess.run(row.command, text=True, capture_output=True, check=False)
        executed.append(
            {
                "lane_id": row.claim.lane_id,
                "instance_job_id": row.claim.instance_job_id,
                "status": row.suggested_status,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    return executed


def main() -> int:
    args = _parser().parse_args()
    repo_root = args.repo_root.resolve()
    claims_path = (repo_root / args.claims_path).resolve() if not args.claims_path.is_absolute() else args.claims_path
    modal_ledger = (repo_root / args.modal_ledger).resolve() if not args.modal_ledger.is_absolute() else args.modal_ledger
    rows = reconcile(repo_root=repo_root, claims_path=claims_path, modal_ledger=modal_ledger)
    executed = _execute(rows) if args.execute else []
    if args.json_out:
        json_out = (repo_root / args.json_out).resolve() if not args.json_out.is_absolute() else args.json_out
        _write_json(json_out, rows, executed)
    if args.md_out:
        md_out = (repo_root / args.md_out).resolve() if not args.md_out.is_absolute() else args.md_out
        _write_md(md_out, rows, executed)
    high = sum(1 for row in rows if row.confidence == "high")
    manual = sum(1 for row in rows if row.confidence == "manual_review")
    active = sum(1 for row in rows if row.confidence == "active_or_unharvested")
    print(
        "RECONCILE_DISPATCH_CLAIMS "
        f"high_confidence_terminal={high} "
        f"manual_review={manual} "
        f"active_or_unharvested={active} "
        f"executed={len(executed)}"
    )
    for row in rows:
        if row.confidence == "high":
            print(
                "TERMINAL_MATCH "
                f"lane_id={row.claim.lane_id} "
                f"job={row.claim.instance_job_id} "
                f"call_id={row.call_id} "
                f"status={row.suggested_status}"
            )
    return 0 if all(item.get("returncode") == 0 for item in executed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
