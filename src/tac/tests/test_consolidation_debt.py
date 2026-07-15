"""Focused tests for the read-only consolidation-debt cadence monitor."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MONITOR = _load("consolidation_debt", "tools/consolidation_debt.py")
LANDING_GATE = _load("codex_landing_review_gate_for_debt", "tools/codex_landing_review_gate.py")


def _write_ledger(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_landing_count_matches_authoritative_gate_latest_status_fixture(tmp_path):
    ledger = tmp_path / "codex_landing_ledger.jsonl"
    rows = [
        {"label": "reviewed", "stamp": "1", "status": "held_entangled"},
        {"label": "reviewed", "stamp": "1", "status": "reviewed_committed"},
        {"label": "held", "stamp": "2", "status": "held_entangled"},
        {"label": "respawned", "stamp": "3", "status": "respawned"},
        {"label": "unknown", "stamp": "4", "status": "future_state"},
        {"label": "closed", "stamp": "5", "status": "closed"},
    ]
    _write_ledger(ledger, rows)

    latest = LANDING_GATE.latest_dispositions(rows)
    assert MONITOR.TERMINAL_LANDING_STATES == LANDING_GATE.TERMINAL_STATES
    gate_semantics_count = sum(
        row.get("status") not in LANDING_GATE.TERMINAL_STATES
        for row in latest.values()
    )
    assert gate_semantics_count == 2
    assert MONITOR._undispositioned_landings(ledger) == gate_semantics_count


def test_landing_count_ignores_malformed_and_unkeyed_rows(tmp_path):
    ledger = tmp_path / "codex_landing_ledger.jsonl"
    _write_ledger(
        ledger,
        [
            "not-json",
            {"status": "held_entangled"},
            {"label": "missing-stamp", "status": "held_entangled"},
            {"label": "valid", "stamp": "1", "status": "held_entangled"},
        ],
    )
    assert MONITOR._undispositioned_landings(ledger) == 1


def test_missing_landing_ledger_is_zero(tmp_path):
    assert MONITOR._undispositioned_landings(tmp_path / "missing.jsonl") == 0


def test_main_remains_nonblocking_for_consolidate_now(monkeypatch, capsys):
    monkeypatch.setattr(
        MONITOR,
        "compute",
        lambda: {
            "verdict": "CONSOLIDATE-NOW",
            "severity": 2,
            "components": {
                "pile_files": (25, 2),
                "pile_lines": (0, 0),
                "landings": (0, 0),
                "stale_commits": (0, 0),
                "signal_ratio": (0.0, 0),
            },
            "signal_detail": {"memos_24h": 0, "system_intelligence_commits_24h": 0},
        },
    )
    assert MONITOR.main(["--quiet-ok"]) == 0
    assert "CONSOLIDATE-NOW" in capsys.readouterr().out


def test_quiet_ok_suppresses_clean_report(monkeypatch, capsys):
    monkeypatch.setattr(
        MONITOR,
        "compute",
        lambda: {
            "verdict": "OK",
            "severity": 0,
            "components": {},
            "signal_detail": {},
        },
    )
    assert MONITOR.main(["--quiet-ok"]) == 0
    assert capsys.readouterr().out == ""
