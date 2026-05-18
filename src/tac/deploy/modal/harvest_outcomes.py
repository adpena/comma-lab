# SPDX-License-Identifier: MIT
"""Canonical Modal harvest-outcome mirroring into the call-id ledger.

This module is the shared bridge between Modal harvesters and
``tac.deploy.modal.call_id_ledger``. Tool-local harvest code may classify
provider-specific outcomes, but terminal outcomes must be mirrored through this
helper so every ``FunctionCall.from_id(...).get(...)`` consumer closes the same
append-only call-id lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.deploy.modal.call_id_ledger import (
    STATUS_FAILED,
    STATUS_HARVESTED,
    STATUS_STALE,
    TERMINAL_STATUSES,
    latest_status_by_call_id,
    query_by_call_id,
    update_call_id_outcome,
)

__all__ = [
    "append_terminal_call_id_ledger_event",
    "call_ledger_status_from_terminal_harvest",
]


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def call_ledger_status_from_terminal_harvest(
    *,
    harvested: dict[str, Any] | None,
    terminal_claim: dict[str, Any] | None = None,
) -> str | None:
    """Map terminal harvest evidence to a canonical call-id ledger status.

    Returns ``None`` for nonterminal or unclassified rows. Plain provider poll
    timeouts should therefore stay in-flight instead of being terminalized.
    """

    claim_status = str((terminal_claim or {}).get("status") or "")
    harvest_status = str((harvested or {}).get("status") or "")
    crash_kind = str((harvested or {}).get("crash_kind") or "")

    if claim_status == "failed_modal_training_result_cache_expired":
        return STATUS_STALE
    if harvest_status == "expired" or crash_kind == "RESULT_CACHE_EXPIRED":
        return STATUS_STALE
    if claim_status.startswith("failed"):
        return STATUS_FAILED
    if claim_status.startswith("completed"):
        return STATUS_HARVESTED

    rc = None
    if harvested is not None:
        rc = _coerce_int(harvested.get("rc"))
        if rc is None:
            rc = _coerce_int(harvested.get("returncode"))
    if rc == 0:
        return STATUS_HARVESTED
    if rc is not None:
        return STATUS_FAILED
    if harvest_status == "function_timeout" or crash_kind == "FUNCTION_TIMEOUT":
        return STATUS_FAILED
    if harvest_status.startswith("error_"):
        return STATUS_FAILED
    return None


def append_terminal_call_id_ledger_event(
    *,
    repo_root: Path,
    metadata: dict[str, Any],
    harvested: dict[str, Any] | None,
    terminal_claim: dict[str, Any] | None = None,
    agent: str,
) -> dict[str, Any]:
    """Mirror a terminal Modal harvest outcome into the canonical ledger.

    The helper is intentionally idempotent for already-terminal rows that
    already carry all structured signal available from ``harvested``. If a
    previous terminal row was lossy, a new supplement row is appended.
    """

    call_id = str(metadata.get("call_id") or "").strip()
    if not call_id or call_id == "?":
        return {"appended": False, "reason": "metadata_missing_call_id"}

    status = call_ledger_status_from_terminal_harvest(
        harvested=harvested,
        terminal_claim=terminal_claim,
    )
    if status is None:
        return {"appended": False, "reason": "nonterminal_or_unclassified", "call_id": call_id}

    ledger_path = repo_root / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    try:
        latest = latest_status_by_call_id(path=ledger_path).get(call_id)
    except Exception as exc:
        return {
            "appended": False,
            "reason": f"call_id_ledger_status_read_failed:{type(exc).__name__}:{exc}",
            "call_id": call_id,
        }

    rc = None
    elapsed_seconds = None
    score = None
    score_axis = None
    archive_sha256 = None
    archive_bytes = None
    evidence_grade = None
    if isinstance(harvested, dict):
        rc = _coerce_int(harvested.get("rc"))
        if rc is None:
            rc = _coerce_int(harvested.get("returncode"))
        elapsed_seconds = _coerce_float(harvested.get("elapsed_seconds"))
        score = _coerce_float(harvested.get("score"))
        score_axis = harvested.get("score_axis")
        archive_sha256 = harvested.get("archive_sha256")
        archive_bytes = _coerce_int(harvested.get("archive_bytes"))
        evidence_grade = harvested.get("evidence_grade")

    if latest in TERMINAL_STATUSES:
        try:
            lifecycle = query_by_call_id(call_id, path=ledger_path)
        except Exception:
            lifecycle = []
        latest_row = lifecycle[-1] if lifecycle else {}
        missing_structured_signal = any(
            latest_row.get(key) in {None, ""}
            for key, value in (
                ("rc", rc),
                ("elapsed_seconds", elapsed_seconds),
                ("archive_sha256", archive_sha256),
                ("archive_bytes", archive_bytes),
            )
            if value is not None
        )
        if not missing_structured_signal:
            return {
                "appended": False,
                "already_terminal": True,
                "call_id": call_id,
                "latest_status": latest,
            }

    try:
        record = update_call_id_outcome(
            call_id=call_id,
            status=status,
            harvest_result=harvested,
            rc=rc,
            elapsed_seconds=elapsed_seconds,
            score=score,
            score_axis=score_axis if isinstance(score_axis, str) else None,
            archive_sha256=archive_sha256 if isinstance(archive_sha256, str) else None,
            archive_bytes=archive_bytes,
            evidence_grade=evidence_grade if isinstance(evidence_grade, str) else None,
            agent=agent,
            path=ledger_path,
            lock_path=lock_path,
            lane_id=metadata.get("lane_id"),
            label=metadata.get("label"),
            platform=metadata.get("platform", "modal"),
            gpu=metadata.get("gpu"),
            expected_cost_usd=metadata.get("expected_cost_usd"),
            expected_axis=metadata.get("expected_axis"),
            recipe=metadata.get("recipe"),
            dispatched_at_utc=metadata.get("dispatched_at_utc") or metadata.get("dispatched_at"),
            max_seconds=metadata.get("max_seconds"),
            mounted_code_git_head=metadata.get("mounted_code_git_head"),
        )
    except Exception as exc:
        return {
            "appended": False,
            "reason": f"call_id_ledger_append_failed:{type(exc).__name__}:{exc}",
            "call_id": call_id,
            "target_status": status,
        }
    return {
        "appended": True,
        "call_id": call_id,
        "status": status,
        "ledger_path": str(ledger_path),
        "written_at_utc": record.get("written_at_utc"),
    }
