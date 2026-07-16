"""Tests for check_modal_single_flight_ledger_consistency (#513).

Local-ledger half of the Modal single-flight + dual-ledger policy (2026-07-15):
  (a) <=1 live (non-terminal) call_id in modal_call_id_ledger.jsonl;
  (b) every live call_id has a matching ACTIVE row in active_lane_dispatch_claims.md;
  (c) no live call_id sits non-terminal longer than stale_hours.
FAIL-OPEN, WARN-ONLY. Fixtures are on-disk JSONL + markdown claim tables.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_single_flight_ledger_consistency,
)

_NOW = _dt.datetime(2026, 7, 15, 21, 0, 0, tzinfo=_dt.UTC)


def _iso(dt: _dt.datetime) -> str:
    return dt.isoformat().replace("+00:00", "") + "Z"


def _ledger_row(call_id: str, status: str, *, hours_ago: float = 1.0,
                label: str | None = None, event_type: str | None = None) -> dict:
    ts = _iso(_NOW - _dt.timedelta(hours=hours_ago))
    row = {
        "call_id": call_id,
        "status": status,
        "written_at_utc": ts,
        "dispatched_at_utc": ts,
        "platform": "modal",
    }
    if event_type is not None:
        row["event_type"] = event_type
    if label is not None:
        row["label"] = label
    return row


def _write_ledger(root: Path, rows: list[dict]) -> None:
    p = root / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


_CLAIMS_HEADER = (
    "# Active lane dispatch claims\n\n"
    "| timestamp_utc | agent | lane_id | platform | instance/job_id | eta | status | notes |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _write_claims(root: Path, rows: list[str]) -> None:
    p = root / ".omx" / "state" / "active_lane_dispatch_claims.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_CLAIMS_HEADER + "".join(rows), encoding="utf-8")


def _claim(job_id: str, status: str, *, lane: str = "lane_x", notes: str = "n") -> str:
    return (f"| 2026-07-15T20:00:00Z | claude | {lane} | modal | {job_id} "
            f"|  | {status} | {notes} |\n")


def _run(root: Path, **kw):
    kw.setdefault("now", _NOW)
    return check_modal_single_flight_ledger_consistency(repo_root=root, **kw)


# ---- fail-open --------------------------------------------------------------

def test_missing_ledger_fail_open(tmp_path: Path) -> None:
    assert _run(tmp_path) == []


# ---- (a) single-flight ------------------------------------------------------

def test_single_live_ok(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", label="lbl1")])
    _write_claims(tmp_path, [_claim("lbl1", "active_dispatched")])
    assert _run(tmp_path) == []


def test_two_live_flagged(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [
        _ledger_row("fc-1", "dispatched", label="lbl1"),
        _ledger_row("fc-2", "dispatched", label="lbl2"),
    ])
    _write_claims(tmp_path, [
        _claim("lbl1", "active"), _claim("lbl2", "active"),
    ])
    violations = _run(tmp_path)
    assert any("single-flight max 1" in v for v in violations)


def test_all_terminal_ok(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [
        _ledger_row("fc-1", "harvested"),
        _ledger_row("fc-2", "failed"),
        _ledger_row("fc-3", "pre_spawn_fatal"),
    ])
    _write_claims(tmp_path, [])
    assert _run(tmp_path) == []


def test_latest_row_wins_dispatched_then_harvested(tmp_path: Path) -> None:
    """A call_id dispatched then later harvested is terminal (latest-row-wins)."""
    _write_ledger(tmp_path, [
        _ledger_row("fc-1", "dispatched", hours_ago=5.0),
        _ledger_row("fc-1", "harvested", hours_ago=1.0),
    ])
    _write_claims(tmp_path, [])
    assert _run(tmp_path) == []


def test_max_live_override(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [
        _ledger_row("fc-1", "dispatched", label="lbl1"),
        _ledger_row("fc-2", "dispatched", label="lbl2"),
    ])
    _write_claims(tmp_path, [
        _claim("lbl1", "active"), _claim("lbl2", "active"),
    ])
    # operator override to 2 concurrent -> no single-flight violation
    assert not any("single-flight" in v for v in _run(tmp_path, max_live=2))


# ---- (b) matching active claim ---------------------------------------------

def test_live_without_matching_claim_flagged(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", label="lbl1")])
    _write_claims(tmp_path, [_claim("other-job", "active")])
    violations = _run(tmp_path)
    assert any("NO matching ACTIVE row" in v for v in violations)


def test_live_matched_by_label(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", label="my-label-x")])
    _write_claims(tmp_path, [_claim("my-label-x", "active_timing_smoke_dispatched")])
    assert not any("NO matching ACTIVE" in v for v in _run(tmp_path))


def test_live_matched_by_call_id(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-ABC", "dispatched")])
    _write_claims(tmp_path, [_claim("fc-ABC", "active")])
    assert not any("NO matching ACTIVE" in v for v in _run(tmp_path))


def test_terminal_claim_does_not_match(tmp_path: Path) -> None:
    """A claim whose status is terminal does not satisfy the live-row match."""
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", label="lbl1")])
    _write_claims(tmp_path, [_claim("lbl1", "failed_rc124")])
    assert any("NO matching ACTIVE row" in v for v in _run(tmp_path))


def test_missing_claims_file_skips_check_b(tmp_path: Path) -> None:
    """No claims file -> skip (b), still do (a)+(c); a single fresh live row is OK."""
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", hours_ago=1.0)])
    assert _run(tmp_path) == []


# ---- (c) stale non-terminal -------------------------------------------------

def test_stale_live_flagged(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", hours_ago=48.0, label="lbl1")])
    _write_claims(tmp_path, [_claim("lbl1", "active")])
    violations = _run(tmp_path)
    assert any("non-terminal for" in v for v in violations)


def test_fresh_live_not_stale(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", hours_ago=2.0, label="lbl1")])
    _write_claims(tmp_path, [_claim("lbl1", "active")])
    assert not any("non-terminal for" in v for v in _run(tmp_path))


def test_stale_hours_param(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [_ledger_row("fc-1", "dispatched", hours_ago=2.0, label="lbl1")])
    _write_claims(tmp_path, [_claim("lbl1", "active")])
    assert any("non-terminal for" in v for v in _run(tmp_path, stale_hours=1.0))


# ---- robustness -------------------------------------------------------------

def test_malformed_json_lines_ignored(tmp_path: Path) -> None:
    p = tmp_path / ".omx" / "state" / "modal_call_id_ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json\n" + json.dumps(_ledger_row("fc-1", "harvested")) + "\n",
                 encoding="utf-8")
    _write_claims(tmp_path, [])
    assert _run(tmp_path) == []


def test_event_type_used_when_no_status(tmp_path: Path) -> None:
    row = {"call_id": "fc-1", "event_type": "dispatched",
           "written_at_utc": _iso(_NOW - _dt.timedelta(hours=1)), "label": "lbl1"}
    _write_ledger(tmp_path, [row])
    _write_claims(tmp_path, [_claim("lbl1", "active")])
    # a single fresh live row with a claim is clean
    assert _run(tmp_path) == []


def test_strict_raises(tmp_path: Path) -> None:
    _write_ledger(tmp_path, [
        _ledger_row("fc-1", "dispatched", label="lbl1"),
        _ledger_row("fc-2", "dispatched", label="lbl2"),
    ])
    _write_claims(tmp_path, [_claim("lbl1", "active"), _claim("lbl2", "active")])
    with pytest.raises(PreflightError):
        _run(tmp_path, strict=True)


def test_live_repo_does_not_raise() -> None:
    result = check_modal_single_flight_ledger_consistency(strict=False)
    assert isinstance(result, list)
