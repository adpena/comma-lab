# SPDX-License-Identifier: MIT
"""Tests for the Modal single-flight refusal + dual-ledger reconciler.

Operator binding 2026-07-15 (memory modal_single_flight_dual_ledger_policy_
20260715): at most ONE live Modal job across ALL lanes unless an explicit
operator override quoted in --notes. This is the ACTUATION half (runtime refusal
in tools/claim_lane_dispatch.py) paired with the WARN-ONLY static gate
check_modal_single_flight_ledger_consistency (src/tac/preflight.py). P0 p0_513.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).resolve().parents[3] / "tools" / "claim_lane_dispatch.py"
spec = importlib.util.spec_from_file_location("claim_lane_dispatch", _HELPER_PATH)
cld = importlib.util.module_from_spec(spec)
sys.modules["claim_lane_dispatch"] = cld
spec.loader.exec_module(cld)


@pytest.fixture
def claims_path(tmp_path):
    return tmp_path / "claims.md"


def _claim(claims_path, lane, job, status="active_dispatched", platform="modal_t4",
           notes="n", *extra):
    return cld.main([
        "claim", "--claims-path", str(claims_path),
        "--lane-id", lane, "--platform", platform, "--instance-job-id", job,
        "--agent", "a", "--status", status, "--notes", notes, *extra,
    ])


# ── helper unit tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("platform,expected", [
    ("modal", True), ("modal_t4", True), ("Modal-A100", True),
    ("modal_cpu", True), ("MODAL", True),
    ("vastai_4090", False), ("lightning", False), ("cpu", False), ("", False),
])
def test_is_modal_platform(platform, expected):
    assert cld._is_modal_platform(platform) is expected


def test_active_modal_across_lanes_excludes_self():
    claims = [
        cld.Claim("2026-07-16T00:00:00Z", "a", "lane_a", "modal_t4", "fc-1", "", "active", ""),
    ]
    # excluding the same (lane, job) yields nothing
    assert cld._active_modal_claims_across_lanes(
        claims, exclude_job=("lane_a", "fc-1")) == []
    # not excluding it yields the row
    assert len(cld._active_modal_claims_across_lanes(
        claims, exclude_job=("other", "x"))) == 1


def test_active_modal_across_lanes_skips_terminal_and_nonmodal():
    claims = [
        cld.Claim("2026-07-16T00:00:00Z", "a", "lane_a", "modal_t4", "fc-1",
                  "", "completed_ok", ""),
        cld.Claim("2026-07-16T00:00:00Z", "a", "lane_b", "vastai", "v-1",
                  "", "active_dispatched", ""),
        cld.Claim("2026-07-16T00:00:00Z", "a", "lane_c", "modal_a100", "fc-2",
                  "", "active_dispatched", ""),
    ]
    out = cld._active_modal_claims_across_lanes(claims, exclude_job=("", ""))
    assert [c.lane_id for c in out] == ["lane_c"]


# ── single-flight refusal ──────────────────────────────────────────────────

def test_first_modal_claim_succeeds(claims_path):
    assert _claim(claims_path, "lane_a", "fc-1") == 0


def test_second_modal_claim_other_lane_refused(claims_path):
    _claim(claims_path, "lane_a", "fc-1")
    rc = _claim(claims_path, "lane_b", "fc-2", platform="modal_a100")
    assert rc == 5


def test_second_modal_claim_override_with_notes_succeeds(claims_path):
    _claim(claims_path, "lane_a", "fc-1")
    rc = _claim(claims_path, "lane_b", "fc-2", "active_dispatched", "modal_a100",
                "operator GO paired AB", "--override")
    assert rc == 0


def test_override_without_notes_refused(claims_path):
    _claim(claims_path, "lane_a", "fc-1")
    with pytest.raises(SystemExit):
        _claim(claims_path, "lane_b", "fc-2", "active_dispatched", "modal_a100",
               "", "--override")


def test_nonmodal_second_claim_not_subject_to_single_flight(claims_path):
    _claim(claims_path, "lane_a", "fc-1")  # modal live
    rc = _claim(claims_path, "lane_v", "v-1", platform="vastai_4090")
    assert rc == 0


def test_terminal_modal_closure_exempt(claims_path):
    _claim(claims_path, "lane_a", "fc-1")  # modal live
    # closing a DIFFERENT modal lane must never be blocked by single-flight
    rc = _claim(claims_path, "lane_b", "fc-9", status="completed_ok",
                platform="modal_a100")
    assert rc == 0


def test_single_flight_clears_after_terminal(claims_path):
    _claim(claims_path, "lane_a", "fc-1")
    # close lane_a with a terminal row for the same job
    assert _claim(claims_path, "lane_a", "fc-1", status="completed_ok") == 0
    # now a new modal lane is allowed
    rc = _claim(claims_path, "lane_b", "fc-2", platform="modal_a100")
    assert rc == 0


def test_dry_run_still_reveals_single_flight_refusal(claims_path):
    _claim(claims_path, "lane_a", "fc-1")
    rc = _claim(claims_path, "lane_b", "fc-2", "active_dispatched", "modal_a100",
                "n", "--dry-run")
    assert rc == 5


# ── reconciler ─────────────────────────────────────────────────────────────

def test_reconcile_ok_when_empty(tmp_path, capsys):
    claims = tmp_path / "c.md"
    ledger = tmp_path / "ledger.jsonl"
    rc = cld.main(["reconcile", "--claims-path", str(claims),
                   "--modal-ledger", str(ledger)])
    assert rc == 0


def test_reconcile_flags_active_claim_no_ledger(tmp_path):
    claims = tmp_path / "c.md"
    ledger = tmp_path / "ledger.jsonl"  # nonexistent -> zero live
    _claim(claims, "lane_a", "fc-1")
    rc = cld.main(["reconcile", "--claims-path", str(claims),
                   "--modal-ledger", str(ledger)])
    assert rc == 6


def test_reconcile_flags_two_live_ledger(tmp_path):
    claims = tmp_path / "c.md"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"call_id": "fc-1", "status": "dispatched",
                    "written_at_utc": "2026-07-16T00:00:00Z", "label": "lane_a"}) + "\n"
        + json.dumps({"call_id": "fc-2", "status": "dispatched",
                      "written_at_utc": "2026-07-16T00:01:00Z", "label": "lane_b"}) + "\n"
    )
    _claim(claims, "lane_a", "fc-1")
    rc = cld.main(["reconcile", "--claims-path", str(claims),
                   "--modal-ledger", str(ledger), "--json"])
    assert rc == 6


def test_reconcile_consistent_single_live_matched(tmp_path):
    claims = tmp_path / "c.md"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"call_id": "fc-1", "status": "dispatched",
                    "written_at_utc": "2026-07-16T00:00:00Z", "label": "lane_a"}) + "\n"
    )
    # active claim carries the call_id in the job field so it matches
    _claim(claims, "lane_a", "fc-1")
    rc = cld.main(["reconcile", "--claims-path", str(claims),
                   "--modal-ledger", str(ledger)])
    assert rc == 0


def test_load_live_modal_ledger_latest_row_wins(tmp_path):
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(
        json.dumps({"call_id": "fc-1", "status": "dispatched",
                    "written_at_utc": "2026-07-16T00:00:00Z"}) + "\n"
        + json.dumps({"call_id": "fc-1", "status": "harvested",
                      "written_at_utc": "2026-07-16T01:00:00Z"}) + "\n"
    )
    # latest row is terminal (harvested) -> zero live
    assert cld._load_live_modal_ledger_calls(ledger) == []


def test_load_live_modal_ledger_missing_file_fail_open(tmp_path):
    assert cld._load_live_modal_ledger_calls(tmp_path / "nope.jsonl") == []
