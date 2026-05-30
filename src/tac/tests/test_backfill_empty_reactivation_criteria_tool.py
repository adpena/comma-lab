# SPDX-License-Identifier: MIT
"""Tests for META Finding A canonical 2-landing pattern Landing 2.

Landing 2 = ``tools/backfill_empty_reactivation_criteria_from_next_action.py``
canonical operator-facing tool that scans the probe-outcomes ledger for rows
with EMPTY ``reactivation_criteria`` AND substantive ``next_action`` and
appends NEW ``EVENT_BACKFILL`` rows with auto-derived criteria per the canonical
helper at Landing 1.

Coverage:

- compute_backfill_plan: empty ledger / no candidates / synthetic candidates
- compute_backfill_plan: latest-row-wins (older empty + newer populated → skip)
- compute_backfill_plan: BOTH empty → skipped (HONEST emptiness)
- execute_backfill: APPEND-only invariant (no mutation of original rows)
- execute_backfill: lock semantics (concurrent reads do not crash)
- execute_backfill: integration with canonical ledger module
- CLI: dry-run is default (no mutation)
- CLI: --apply REQUIRES --operator-approved (rc=3)
- CLI: --apply REQUIRES well-formed --operator-approved (rc=2)
- CLI: --apply with valid operator-approved appends rows
- CLI: --json emits machine-readable schema
- Integration: end-to-end synthetic ledger with mixed rows
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.probe_outcomes_ledger import (
    AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL,
    EVENT_BACKFILL,
    VERDICT_DEFER,
    VERDICT_INDEPENDENT,
    VERDICT_PROCEED,
    load_outcomes,
    register_probe_outcome,
)

# Import the backfill tool module by canonical path (since it's a CLI script,
# not in tac.* package). This mirrors the sister-test import pattern used by
# `tools/gc_experiments_results.py` tests.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_PATH = _REPO_ROOT / "tools" / "backfill_empty_reactivation_criteria_from_next_action.py"


def _load_backfill_tool_module():
    """Import the backfill tool module by file path."""
    spec = importlib.util.spec_from_file_location(
        "backfill_empty_reactivation_criteria_from_next_action_test", _TOOL_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BACKFILL_TOOL = _load_backfill_tool_module()


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> tuple[Path, Path]:
    ledger = tmp_path / "probe_outcomes.jsonl"
    lock = tmp_path / "probe_outcomes.jsonl.lock"
    return ledger, lock


# ─────────────────────────────────────────────────────────────────────────
# compute_backfill_plan
# ─────────────────────────────────────────────────────────────────────────


def test_plan_empty_ledger_returns_no_candidates(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 0
    assert plan["ledger_rows_total"] == 0
    assert plan["unique_probe_ids"] == 0
    assert plan["candidates_to_backfill"] == []
    assert plan["skipped_both_empty"] == []
    assert plan["skipped_already_populated"] == []


def test_plan_identifies_empty_criteria_with_substantive_next_action(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # Hand-write a legacy row to simulate pre-Landing-1 EMPTY criteria
    # (the Landing 1 helper would NOT produce this; this simulates the
    # historical landscape).
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_legacy_empty",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "queue paired CUDA + paired CPU anchor",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 1
    assert len(plan["candidates_to_backfill"]) == 1
    cand = plan["candidates_to_backfill"][0]
    assert cand["probe_id"] == "test_legacy_empty"
    assert cand["substrate"] == "test_substrate"
    assert cand["verdict"] == "DEFER"
    assert cand["next_action"] == "queue paired CUDA + paired CPU anchor"
    assert cand["derived_reactivation_criteria"] == [
        "next_action_satisfied: queue paired CUDA + paired CPU anchor"
    ]


def test_plan_skips_both_empty_rows_honest_emptiness(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    # Row with BOTH empty next_action and EMPTY reactivation_criteria.
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_both_empty",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": None,
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 0
    assert "test_both_empty" in plan["skipped_both_empty"]


def test_plan_skips_already_populated_rows(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    register_probe_outcome(
        probe_id="test_already_populated",
        substrate="test_substrate",
        recipe_path=None,
        probe_kind="byte_mutation",
        verdict=VERDICT_DEFER,
        metric_name="test_metric",
        metric_value=0.5,
        next_action="ignored",
        reactivation_criteria=["Operator-supplied criterion"],
        path=ledger,
        lock_path=lock,
    )

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 0
    assert "test_already_populated" in plan["skipped_already_populated"]


def test_plan_latest_row_wins_when_newer_populated(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # First row: empty criteria.
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_latest_wins",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "sister wave lands",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    # Second row (newer): populated criteria via canonical update_probe_outcome
    # simulation; here we just hand-write to control test contents.
    newer_row = dict(legacy_row)
    newer_row["event_type"] = "ratified"
    newer_row["reactivation_criteria"] = ["Operator manually populated"]
    newer_row["written_at_utc"] = "2026-05-15T00:00:00.000000Z"

    ledger.write_text(
        json.dumps(legacy_row, sort_keys=True) + "\n"
        + json.dumps(newer_row, sort_keys=True) + "\n"
    )

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    # Latest row wins → already populated → skip.
    assert plan["total_candidates"] == 0
    assert "test_latest_wins" in plan["skipped_already_populated"]


def test_plan_latest_row_wins_when_newer_still_empty(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_still_empty",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "earlier next_action",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    newer_row = dict(legacy_row)
    newer_row["event_type"] = "ratified"
    newer_row["next_action"] = "later substantive next_action"
    newer_row["written_at_utc"] = "2026-05-15T00:00:00.000000Z"

    ledger.write_text(
        json.dumps(legacy_row, sort_keys=True) + "\n"
        + json.dumps(newer_row, sort_keys=True) + "\n"
    )

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 1
    cand = plan["candidates_to_backfill"][0]
    # Derives from the LATEST row's next_action.
    assert cand["next_action"] == "later substantive next_action"
    assert cand["derived_reactivation_criteria"] == [
        "next_action_satisfied: later substantive next_action"
    ]


# ─────────────────────────────────────────────────────────────────────────
# execute_backfill — APPEND-ONLY invariant
# ─────────────────────────────────────────────────────────────────────────


def test_execute_appends_event_backfill_row(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    # Seed with a legacy empty-criteria row.
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_execute_appends",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "queue paired CUDA + CPU anchor",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 1

    execution = BACKFILL_TOOL.execute_backfill(
        plan,
        operator_approved="testuser:2026-05-30T19:00:00Z",
        ledger_path=ledger,
        lock_path=lock,
    )
    assert execution["appended_count"] == 1
    assert execution["skipped_count"] == 0
    assert execution["appended_probe_ids"] == ["test_execute_appends"]

    rows = load_outcomes(ledger)
    assert len(rows) == 2
    # Original row preserved verbatim (APPEND-ONLY per HISTORICAL_PROVENANCE).
    assert rows[0]["event_type"] == "adjudicated"
    assert rows[0].get("reactivation_criteria") is None
    # New backfill row carries auto-derived criteria + canonical provenance.
    assert rows[1]["event_type"] == EVENT_BACKFILL
    assert rows[1]["probe_id"] == "test_execute_appends"
    assert rows[1]["reactivation_criteria"] == [
        "next_action_satisfied: queue paired CUDA + CPU anchor"
    ]
    assert (
        rows[1]["reactivation_criteria_derivation_provenance"]
        == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL
    )
    assert rows[1]["backfill_operator_approved"] == "testuser:2026-05-30T19:00:00Z"


def test_execute_does_not_mutate_original_rows(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_no_mutation",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "sister wave lands",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    original_serialized = json.dumps(legacy_row, sort_keys=True)
    ledger.write_text(original_serialized + "\n")

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    BACKFILL_TOOL.execute_backfill(
        plan,
        operator_approved="testuser:2026-05-30T19:00:00Z",
        ledger_path=ledger,
        lock_path=lock,
    )

    # Re-read the ledger; the FIRST line must match the original byte-for-byte.
    new_content = ledger.read_text()
    lines = new_content.strip().split("\n")
    assert lines[0] == original_serialized


def test_execute_empty_plan_no_op(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, lock = tmp_ledger
    ledger.write_text("")
    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    execution = BACKFILL_TOOL.execute_backfill(
        plan,
        operator_approved="testuser:2026-05-30T19:00:00Z",
        ledger_path=ledger,
        lock_path=lock,
    )
    assert execution["appended_count"] == 0
    assert "no candidates" in execution["notes"]


def test_execute_skips_concurrently_populated_row(
    tmp_ledger: tuple[Path, Path],
) -> None:
    """If the row becomes populated between plan and execute (e.g. via
    concurrent operator edit), execute_backfill MUST skip it per
    latest-row-wins semantics."""
    ledger, lock = tmp_ledger
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_concurrent_populate",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "queue paired anchor",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 1

    # Simulate concurrent population: append a new ratified row with criteria
    # BEFORE execute_backfill runs.
    concurrent_row = dict(legacy_row)
    concurrent_row["event_type"] = "ratified"
    concurrent_row["reactivation_criteria"] = ["Concurrently populated"]
    concurrent_row["written_at_utc"] = "2026-05-30T18:00:00.000000Z"
    with ledger.open("a") as f:
        f.write(json.dumps(concurrent_row, sort_keys=True) + "\n")

    execution = BACKFILL_TOOL.execute_backfill(
        plan,
        operator_approved="testuser:2026-05-30T19:00:00Z",
        ledger_path=ledger,
        lock_path=lock,
    )
    # Concurrent population observed inside the lock → skip.
    assert execution["appended_count"] == 0
    assert execution["skipped_count"] == 1
    assert execution["skipped_probe_ids"][0]["probe_id"] == "test_concurrent_populate"
    assert (
        execution["skipped_probe_ids"][0]["reason"]
        == "already_populated_concurrent_backfill"
    )


# ─────────────────────────────────────────────────────────────────────────
# CLI — dry-run is default; --apply requires --operator-approved
# ─────────────────────────────────────────────────────────────────────────


def test_cli_dry_run_is_default(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    rc = BACKFILL_TOOL.main(["--ledger-path", str(ledger)])
    assert rc == 0


def test_cli_refuses_apply_without_operator_approved(
    tmp_ledger: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    rc = BACKFILL_TOOL.main(["--ledger-path", str(ledger), "--apply"])
    assert rc == 3
    captured = capsys.readouterr()
    assert "REFUSING_APPLY" in captured.err
    assert "--operator-approved" in captured.err


def test_cli_refuses_apply_with_malformed_operator_approved(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    with pytest.raises(SystemExit) as exc_info:
        BACKFILL_TOOL.main(
            ["--ledger-path", str(ledger), "--apply", "--operator-approved", "noColon"]
        )
    assert "VALIDATION_ERROR" in str(exc_info.value)


def test_cli_refuses_apply_with_empty_handle(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    with pytest.raises(SystemExit) as exc_info:
        BACKFILL_TOOL.main(
            ["--ledger-path", str(ledger), "--apply", "--operator-approved", ":timestamp"]
        )
    assert "handle is empty" in str(exc_info.value)


def test_cli_refuses_apply_with_empty_timestamp(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    with pytest.raises(SystemExit) as exc_info:
        BACKFILL_TOOL.main(
            ["--ledger-path", str(ledger), "--apply", "--operator-approved", "handle:"]
        )
    assert "timestamp is empty" in str(exc_info.value)


def test_cli_apply_with_valid_operator_approved(
    tmp_ledger: tuple[Path, Path],
) -> None:
    ledger, _ = tmp_ledger
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_cli_apply",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "sister wave lands",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")

    rc = BACKFILL_TOOL.main(
        [
            "--ledger-path",
            str(ledger),
            "--apply",
            "--operator-approved",
            "testuser:2026-05-30T19:00:00Z",
        ]
    )
    assert rc == 0

    rows = load_outcomes(ledger)
    assert len(rows) == 2
    assert rows[1]["event_type"] == EVENT_BACKFILL


def test_cli_json_output(
    tmp_ledger: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger, _ = tmp_ledger
    ledger.write_text("")
    rc = BACKFILL_TOOL.main(["--ledger-path", str(ledger), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema"] == (
        "pact.probe_outcomes_reactivation_criteria_backfill_plan.v1"
    )
    assert payload["total_candidates"] == 0


def test_cli_verbose_output_with_candidates(
    tmp_ledger: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger, _ = tmp_ledger
    legacy_row = {
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "test_cli_verbose",
        "substrate": "test_substrate",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "test_metric",
        "metric_value": 0.5,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "queue paired anchor",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    }
    ledger.write_text(json.dumps(legacy_row, sort_keys=True) + "\n")
    rc = BACKFILL_TOOL.main(
        ["--ledger-path", str(ledger), "--dry-run", "--verbose"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "test_cli_verbose" in captured.out
    assert "queue paired anchor" in captured.out


# ─────────────────────────────────────────────────────────────────────────
# Integration — end-to-end synthetic ledger with mixed rows
# ─────────────────────────────────────────────────────────────────────────


def test_end_to_end_mixed_ledger(tmp_ledger: tuple[Path, Path]) -> None:
    """End-to-end: synthetic ledger with empty / populated / both-empty rows.

    After --apply: 1 EVENT_BACKFILL appended; original rows preserved;
    populated and both-empty rows skipped.
    """
    ledger, lock = tmp_ledger

    rows_to_write = []

    # Row 1: empty criteria + substantive next_action (candidate).
    rows_to_write.append({
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "mixed_candidate",
        "substrate": "mixed_substrate_1",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "m",
        "metric_value": 0.0,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "queue paired CUDA + paired CPU anchor",
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    })

    # Row 2: BOTH empty (skipped per HONEST emptiness).
    rows_to_write.append({
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "mixed_both_empty",
        "substrate": "mixed_substrate_2",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "m",
        "metric_value": 0.0,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": None,
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    })

    # Row 3: already populated (skipped).
    rows_to_write.append({
        "schema_version": 1,
        "event_type": "adjudicated",
        "probe_id": "mixed_already_pop",
        "substrate": "mixed_substrate_3",
        "recipe_path": None,
        "probe_kind": "byte_mutation",
        "verdict": "DEFER",
        "metric_name": "m",
        "metric_value": 0.0,
        "threshold": None,
        "threshold_token": None,
        "evidence_path": None,
        "next_action": "sister wave",
        "reactivation_criteria": ["Operator already populated"],
        "blocker_status": "blocking",
        "dispatched_at_utc": None,
        "adjudicated_at_utc": "2026-05-01T00:00:00.000000Z",
        "expires_at_utc": "2026-06-01T00:00:00.000000Z",
        "staleness_window_days": 30,
        "agent": "claude",
        "subagent_id": None,
        "session_id": None,
        "notes": None,
        "written_at_utc": "2026-05-01T00:00:00.000000Z",
        "written_pid": 1,
        "written_host": "test",
    })

    ledger.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows_to_write) + "\n"
    )

    plan = BACKFILL_TOOL.compute_backfill_plan(ledger_path=ledger)
    assert plan["total_candidates"] == 1
    assert plan["candidates_to_backfill"][0]["probe_id"] == "mixed_candidate"
    assert "mixed_both_empty" in plan["skipped_both_empty"]
    assert "mixed_already_pop" in plan["skipped_already_populated"]

    execution = BACKFILL_TOOL.execute_backfill(
        plan,
        operator_approved="testuser:2026-05-30T19:00:00Z",
        ledger_path=ledger,
        lock_path=lock,
    )
    assert execution["appended_count"] == 1

    rows = load_outcomes(ledger)
    assert len(rows) == 4  # 3 original + 1 backfill
    # The backfill row carries auto-derived criteria.
    backfill_rows = [r for r in rows if r.get("event_type") == EVENT_BACKFILL]
    assert len(backfill_rows) == 1
    assert backfill_rows[0]["probe_id"] == "mixed_candidate"
    assert backfill_rows[0]["reactivation_criteria"] == [
        "next_action_satisfied: queue paired CUDA + paired CPU anchor"
    ]
    assert (
        backfill_rows[0]["reactivation_criteria_derivation_provenance"]
        == AUTO_DERIVE_PROVENANCE_FROM_NEXT_ACTION_BACKFILL
    )


def test_cli_subprocess_dry_run_runs_against_live_ledger() -> None:
    """Live-repo regression guard — the CLI dry-run runs cleanly against the
    actual probe-outcomes ledger without crashing. Sister of Catalog #185
    META-meta empirical-state regression guard."""
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOL_PATH),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(_REPO_ROOT / "src"),
        },
    )
    assert result.returncode == 0
    assert "META Finding A canonical 2-landing pattern" in result.stdout
    assert "Total rows in ledger" in result.stdout


def test_event_backfill_is_in_valid_event_types() -> None:
    """The canonical EVENT_BACKFILL constant must be registered in
    VALID_EVENT_TYPES so the ledger's _validate_event_record accepts it."""
    from tac.probe_outcomes_ledger import VALID_EVENT_TYPES

    assert EVENT_BACKFILL in VALID_EVENT_TYPES
