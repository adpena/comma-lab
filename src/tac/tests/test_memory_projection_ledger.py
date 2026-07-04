# SPDX-License-Identifier: MIT
"""Tests for the self-calibrating margin ledger (BUILD #294 piece D) in
tools/witness_memory_preflight.py, plus the launcher's calibration-smoke pure helpers (piece B).

Guards: fcntl-locked JSONL append/read; reconcile parses the MEASURED safe_run peak (never
invents); the units-corrected blackbox fallback (tracked ``current_rss_gib`` is KiB/1e6 units —
#205 memory mine §1); calibrated_margin p95 with the labeled 10 GiB fallback under 3 rows; and the
calibration overrun verdict."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import witness_memory_preflight as wmp  # noqa: E402

RAM = 128.0


def _proj():
    return wmp.project_peak_rss_gib(num_pairs=600, verdict_batch=32, total_ram_gib=RAM)


def _mk_launch_sh(tmp_path: Path) -> Path:
    p = tmp_path / "launch.sh"
    p.write_text("python trainer.py \\\n  --num-pairs 600 \\\n  --self-orient \\\n"
                 "  --verdict-batch 32 \\\n")
    return p


# ── append/read + schema ────────────────────────────────────────────────────────────────────────
def test_append_and_read_roundtrip(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    row = wmp.append_ledger_row({"event": "projection", "run_dir": "/x"}, ledger)
    assert "ts" in row
    rows = wmp.read_ledger_rows(ledger)
    assert len(rows) == 1 and rows[0]["run_dir"] == "/x"


def test_read_skips_corrupt_lines(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event": "projection"}\nNOT JSON\n{"event": "reconcile"}\n')
    assert [r["event"] for r in wmp.read_ledger_rows(ledger)] == ["projection", "reconcile"]


def test_record_projection_schema(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    launch = _mk_launch_sh(tmp_path)
    row = wmp.record_projection(tmp_path / "run", launch, _proj(), note="test",
                                ledger_path=ledger)
    for key in ("event", "run_dir", "projected_peak_gib", "config_hash", "ts"):
        assert key in row, key
    assert row["event"] == "projection"
    assert row["projected_peak_gib"] == 67.61
    assert len(row["config_hash"]) == 16


def test_config_hash_stable_and_content_sensitive(tmp_path):
    launch = _mk_launch_sh(tmp_path)
    h1 = wmp.config_hash_from_launch_sh(launch)
    assert h1 == wmp.config_hash_from_launch_sh(launch)
    launch.write_text(launch.read_text().replace("32", "64"))
    assert wmp.config_hash_from_launch_sh(launch) != h1


# ── actual-peak sources (MEASURED only; never invented) ─────────────────────────────────────────
def test_actual_peak_from_run_log_detail_line():
    text = "SAFE_RUN [x] status=ok exit=0 peak_rss=69336MiB elapsed=100.00s limit_rss=90000MiB"
    peak = wmp.actual_peak_from_run_log(text)
    assert peak is not None
    gib, source = peak
    assert abs(gib - 69336 / 1024.0) < 1e-9
    assert source == "run_log_safe_run_peak"


def test_actual_peak_from_run_log_json_row():
    text = 'SAFE_RUN {"label": "x", "status": "ok", "exit": 0, "peak_rss_mib": 61440}'
    peak = wmp.actual_peak_from_run_log(text)
    assert peak is not None and abs(peak[0] - 60.0) < 1e-9


def test_actual_peak_absent_returns_none():
    assert wmp.actual_peak_from_run_log("no telemetry here") is None


def test_blackbox_fallback_units_corrected(tmp_path):
    """Tracked ``current_rss_gib`` is KiB/1e6 UNITS (mine §1) — the reader must convert x0.9537."""
    bb = tmp_path / "memory_blackbox.jsonl"
    run_dir = tmp_path / "levelset_n600_witness_X"
    rows = [
        {"ts": 1, "tracked": [{"label": "levelset_witness_levelset_n600_witness_X",
                               "current_rss_gib": 65.27}]},
        {"ts": 2, "tracked": [{"label": "levelset_witness_levelset_n600_witness_X",
                               "current_rss_gib": 60.0}]},
    ]
    bb.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    peak = wmp.actual_peak_from_blackbox(run_dir, bb)
    assert peak is not None
    gib, source = peak
    assert abs(gib - 65.27 * wmp.TRACKED_RSS_UNIT_TO_GIB) < 1e-9   # 62.25 true GiB
    assert gib < 65.27   # the mislabeled unit over-reads; conversion shrinks it
    assert source == "blackbox_tracked_max_units_corrected"


def test_blackbox_fallback_scans_rotated_archives(tmp_path):
    bb = tmp_path / "memory_blackbox.jsonl"
    bb.write_text(json.dumps({"tracked": [{"label": "levelset_witness_runY",
                                           "current_rss_gib": 50.0}]}) + "\n")
    arch = tmp_path / "archive"
    arch.mkdir()
    (arch / "memory_blackbox_20260703T000000Z.jsonl").write_text(
        json.dumps({"tracked": [{"label": "levelset_witness_runY",
                                 "current_rss_gib": 70.97}]}) + "\n")
    peak = wmp.actual_peak_from_blackbox(tmp_path / "runY", bb)
    assert peak is not None
    assert abs(peak[0] - 70.97 * wmp.TRACKED_RSS_UNIT_TO_GIB) < 1e-9  # archive peak wins


def test_blackbox_fallback_none_when_label_absent(tmp_path):
    bb = tmp_path / "memory_blackbox.jsonl"
    bb.write_text(json.dumps({"tracked": [{"label": "other", "current_rss_gib": 50.0}]}) + "\n")
    assert wmp.actual_peak_from_blackbox(tmp_path / "runZ", bb) is None


# ── reconcile ────────────────────────────────────────────────────────────────────────────────────
def _seed_projection(tmp_path, ledger, run_name="run"):
    run_dir = tmp_path / run_name
    run_dir.mkdir(exist_ok=True)
    launch = _mk_launch_sh(tmp_path)
    wmp.record_projection(run_dir, launch, _proj(), ledger_path=ledger)
    return run_dir


def test_reconcile_requires_projection_row(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(RuntimeError, match="no projection ledger row"):
        wmp.reconcile_run_dir(tmp_path / "nope", ledger_path=ledger,
                              blackbox_path=tmp_path / "absent.jsonl")


def test_reconcile_refuses_to_invent_a_number(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    run_dir = _seed_projection(tmp_path, ledger)
    with pytest.raises(RuntimeError, match="Refusing to invent"):
        wmp.reconcile_run_dir(run_dir, ledger_path=ledger,
                              blackbox_path=tmp_path / "absent.jsonl")


def test_reconcile_from_run_log(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    run_dir = _seed_projection(tmp_path, ledger)
    (run_dir / "run.log").write_text(
        "SAFE_RUN [x] status=ok exit=0 peak_rss=65536MiB elapsed=1s limit_rss=90000MiB")
    row = wmp.reconcile_run_dir(run_dir, ledger_path=ledger,
                                blackbox_path=tmp_path / "absent.jsonl")
    assert row["event"] == "reconcile"
    assert row["actual_peak_gib"] == 64.0
    assert row["residual_gib"] == round(67.61 - 64.0, 3)   # + => over-projection (conservative)
    assert row["in_progress"] is False


def test_reconcile_actual_override_with_source(tmp_path):
    """The #205 backfill path: an externally-MEASURED actual (mine §4) with a mandatory cite."""
    ledger = tmp_path / "ledger.jsonl"
    run_dir = _seed_projection(tmp_path, ledger)
    row = wmp.reconcile_run_dir(
        run_dir, ledger_path=ledger, blackbox_path=tmp_path / "absent.jsonl",
        actual_override=(67.68, "n205_memory_behavior_mine_20260704.md §4"))
    assert row["actual_peak_gib"] == 67.68
    assert row["residual_gib"] == round(67.61 - 67.68, 3) == -0.07
    assert "mine" in row["actual_source"]


def test_reconcile_actual_override_rejects_placeholder_source(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    run_dir = _seed_projection(tmp_path, ledger)
    with pytest.raises(RuntimeError, match="source cite"):
        wmp.reconcile_run_dir(run_dir, ledger_path=ledger,
                              blackbox_path=tmp_path / "absent.jsonl",
                              actual_override=(67.68, "TBD"))


# ── calibrated margin ────────────────────────────────────────────────────────────────────────────
def test_calibrated_margin_fallback_labeled_below_3_rows(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for res in (0.5, -1.0):
        wmp.append_ledger_row({"event": "reconcile", "residual_gib": res}, ledger)
    margin, label = wmp.calibrated_margin(ledger)
    assert margin == wmp.ASSUMED_MARGIN_GIB == 10.0
    assert "assumed_default_insufficient_rows(n=2" in label


def test_calibrated_margin_p95_measured(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    residuals = [0.1, -0.2, 0.5, 1.0, -2.0, 0.3, 0.7, -0.4, 0.9, 1.5,
                 -0.6, 0.2, 0.8, -1.1, 0.4, 0.05, -0.15, 0.25, 0.35, 3.0]
    for r in residuals:
        wmp.append_ledger_row({"event": "reconcile", "residual_gib": r}, ledger)
    margin, label = wmp.calibrated_margin(ledger)
    srt = sorted(abs(r) for r in residuals)
    import math

    assert margin == srt[math.ceil(0.95 * len(srt)) - 1]
    assert label == f"measured_p95_over_{len(residuals)}_reconciled_rows"


def test_calibrated_margin_ignores_projection_rows(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for _ in range(5):
        wmp.append_ledger_row({"event": "projection", "projected_peak_gib": 60.0}, ledger)
    margin, label = wmp.calibrated_margin(ledger)
    assert margin == wmp.ASSUMED_MARGIN_GIB and "n=0" in label


# ── CLI surfaces ────────────────────────────────────────────────────────────────────────────────
def test_cli_reconcile_and_margin(tmp_path, capsys, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(wmp, "LEDGER_PATH", ledger)
    monkeypatch.setattr(wmp, "BLACKBOX_PATH", tmp_path / "absent.jsonl")
    run_dir = _seed_projection(tmp_path, ledger)
    (run_dir / "run.log").write_text("SAFE_RUN [x] status=ok exit=0 peak_rss=65536MiB "
                                     "elapsed=1s limit_rss=90000MiB")
    rc = wmp.main(["--reconcile", str(run_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reconciled" in out and "calibrated margin" in out


def test_cli_reconcile_errors_cleanly_without_projection(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wmp, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(wmp, "BLACKBOX_PATH", tmp_path / "absent.jsonl")
    rc = wmp.main(["--reconcile", str(tmp_path / "nope")])
    assert rc == 5


def test_cli_record_projection_with_backfill_peak(tmp_path, capsys, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(wmp, "LEDGER_PATH", ledger)
    launch = _mk_launch_sh(tmp_path)
    rc = wmp.main(["--launch-sh", str(launch), "--run-dir", str(tmp_path / "run"),
                   "--total-ram-gib", "128", "--record-projection",
                   "--projected-peak-gib", "67.61"])
    assert rc == 0
    rows = wmp.read_ledger_rows(ledger)
    assert rows and rows[0]["projected_peak_gib"] == 67.61


# ── launcher calibration helpers (piece B; pure) ────────────────────────────────────────────────
def test_launcher_parse_safe_run_peak_and_verdict():
    import launch_witness_run as lwr

    assert lwr.parse_safe_run_peak_mib(
        'SAFE_RUN {"label": "c", "status": "ok", "exit": 0, "peak_rss_mib": 20480}') == 20480.0
    assert lwr.parse_safe_run_peak_mib("nothing") is None

    ok, reason = lwr.calibration_verdict(20.0, 21.0, 15.0)   # +5% <= +15%
    assert ok and reason.startswith("OK")
    ok, reason = lwr.calibration_verdict(20.0, 24.0, 15.0)   # +20% > +15%
    assert not ok and "OVERRUN" in reason and "REFUSING" in reason


def test_launcher_calibration_verdict_boundary():
    import launch_witness_run as lwr

    ok, _ = lwr.calibration_verdict(20.0, 23.0, 15.0)  # exactly +15% => OK (<=)
    assert ok
