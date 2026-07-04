# SPDX-License-Identifier: MIT
"""Tests for tools/memory_blackbox.py — the always-on memory black-box recorder + governor daemon.

Guards: sample shape (all trust fields present), fcntl-locked append + 20 MB rotation, the
``--last-crash`` gap/reboot detection (so a future crash yields the trajectory into it), tail, and
the singleton lock.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import memory_blackbox as mbb  # noqa: E402
import system_memory_governor as gov  # noqa: E402


def _snap():
    return gov.SystemMemorySnapshot(
        total_gib=128.0, available_gib=100.0, used_gib=28.0, free_gib=90.0, wired_gib=4.0,
        compressor_gib=1.0, swap_used_gib=0.0, pressure_level=1, load1=3.0, load5=2.0, load15=1.0,
        available_primary_gib=100.0, closure_gib=1.3, closure_ok=True, cross_validated=True,
        discrepancy_gib=0.0, fail_safe=False)


def _job(label="probe", pid=1234):
    return gov.TrackedJob(label=label, pid=pid, pgid=pid, cmd=f"python {label}.py", priority=-60,
                          projected_peak_gib=40.0, current_rss_gib=20.0, paused=False,
                          throttle_eligible=True, own_group_leader=True)


# ── sample shape ────────────────────────────────────────────────────────────────────────────────
def test_sample_once_has_all_trust_and_ceiling_fields(monkeypatch):
    monkeypatch.delenv(gov.SAFETY_FLOOR_ENV, raising=False)  # hermetic vs a leaked floor override
    s = mbb.sample_once(jobs=[_job()], snapshot=_snap())
    for k in ("ts", "ts_iso", "mono", "boottime", "total_gib", "used_gib", "available_gib",
              "pressure", "pressure_level", "adaptive_ceiling_gib", "training_budget_gib",
              "baseline_gib", "system_used_headroom_gib", "tracked_sum_gib", "tracked",
              "closure_ok", "cross_validated", "fail_safe", "load1",
              "safety_floor", "tracked_units"):
        assert k in s, f"missing sample field {k}"
    assert s["tracked_sum_gib"] == 20.0
    assert s["tracked"][0]["label"] == "probe"
    assert s["tracked_units"] == "true_gib"   # GB-F1: tracked rows are TRUE GiB post units-at-read
    # BUILD #298 derived floor: baseline = used(28) - tracked(20) = 8 = measured cp;
    # floor = max(2, 8 + 6.4, 10.24) = 14.4; ceiling = 128 - 14.4 = 113.6; budget = 105.6.
    assert abs(s["baseline_gib"] - 8.0) < 1e-6
    assert abs(s["adaptive_ceiling_gib"] - 113.6) < 1e-6
    assert abs(s["training_budget_gib"] - 105.6) < 1e-6
    # full per-tick decomposition recorded (max observability): which leg won + all leg values
    sf = s["safety_floor"]
    assert sf["winning_leg"] == "measured_cp"
    assert abs(sf["floor_gib"] - 14.4) < 1e-6
    assert abs(sf["measured_cp_rss_gib"] - 8.0) < 1e-6
    assert abs(sf["static_leg_gib"] - 10.24) < 1e-6
    assert abs(sf["cp_headroom_gib"] - 6.4) < 1e-6
    assert sf["smoothed"] is False and abs(sf["applied_floor_gib"] - 14.4) < 1e-6


# ── append + rotation ─────────────────────────────────────────────────────────────────────────
def test_append_and_rotation(tmp_path, monkeypatch):
    live = tmp_path / "memory_blackbox.jsonl"
    lock = tmp_path / ".lock"
    archive = tmp_path / "archive"
    monkeypatch.setattr(mbb, "_BLACKBOX", live)
    monkeypatch.setattr(mbb, "_BLACKBOX_LOCK", lock)
    monkeypatch.setattr(mbb, "_ARCHIVE", archive)
    monkeypatch.setattr(mbb, "ROTATE_BYTES", 300)  # tiny so rotation fires quickly
    for i in range(50):
        mbb.append_sample({"ts": float(i), "x": "y" * 20}, path=live, lock_path=lock)
    # rotation should have moved the over-size live file into the archive at least once
    assert archive.exists()
    rotated = list(archive.glob("memory_blackbox_*.jsonl"))
    assert rotated, "expected at least one rotated archive file"
    assert live.exists()  # the current live file still exists (post-rotation appends)


# ── gap / reboot detection (the crash trajectory) ───────────────────────────────────────────────
def test_find_last_gap_detects_sampler_death():
    samples = [{"ts": t, "boottime": 100} for t in (0, 2, 4, 6)]
    samples += [{"ts": t, "boottime": 100} for t in (600, 602)]  # 594s silence => crash edge
    gap = mbb.find_last_gap(samples, gap_seconds=30.0)
    assert gap is not None and not gap["reboot"]
    assert gap["before"]["ts"] == 6 and gap["dt"] > 500


def test_find_last_gap_detects_reboot_via_boottime():
    samples = [{"ts": 0, "boottime": 100}, {"ts": 2, "boottime": 100},
               {"ts": 3, "boottime": 200}]  # small ts gap but boottime jumped => reboot
    gap = mbb.find_last_gap(samples, gap_seconds=30.0)
    assert gap is not None and gap["reboot"]


def test_find_last_gap_returns_none_when_continuous():
    samples = [{"ts": t, "boottime": 100} for t in range(0, 20, 2)]
    assert mbb.find_last_gap(samples, gap_seconds=30.0) is None


def test_find_last_gap_returns_the_MOST_RECENT_gap():
    samples = ([{"ts": t, "boottime": 100} for t in (0, 2)]
               + [{"ts": t, "boottime": 100} for t in (100, 102)]     # gap #1
               + [{"ts": t, "boottime": 100} for t in (900, 902)])    # gap #2 (most recent)
    gap = mbb.find_last_gap(samples, gap_seconds=30.0)
    assert gap["before"]["ts"] == 102  # the LAST gap's leading edge


def test_last_crash_report_window(tmp_path, monkeypatch):
    live = tmp_path / "memory_blackbox.jsonl"
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setattr(mbb, "_BLACKBOX", live)
    monkeypatch.setattr(mbb, "_ARCHIVE", archive)
    rows = []
    # a rising-used trajectory into a crash edge at ts=1000, then silence
    base = 1_000_000.0
    for i in range(10):
        rows.append({"ts": base + i * 2, "ts_iso": "x", "boottime": 100,
                     "used_gib": 60.0 + i * 6, "available_gib": 60.0 - i * 6, "pressure_level": 1})
    rows.append({"ts": base + 900, "ts_iso": "x", "boottime": 100, "used_gib": 30.0,
                 "available_gib": 90.0, "pressure_level": 1})  # after a long gap (post-crash)
    live.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rep = mbb.last_crash_report(minutes=10.0, gap_seconds=30.0)
    assert rep["status"] == "sampler_death_gap"
    assert rep["summary"]["peak_used_gib"] >= 110.0  # the trajectory captured the climb into the crash
    assert rep["summary"]["n_samples"] >= 5


def test_last_crash_report_no_samples(monkeypatch, tmp_path):
    monkeypatch.setattr(mbb, "_BLACKBOX", tmp_path / "none.jsonl")
    monkeypatch.setattr(mbb, "_ARCHIVE", tmp_path / "none_archive")
    rep = mbb.last_crash_report()
    assert rep["status"] == "no_samples"


def test_window_summary():
    w = [{"used_gib": 60, "available_gib": 40, "pressure_level": 1},
         {"used_gib": 100, "available_gib": 5, "pressure_level": 4}]
    s = mbb._window_summary(w)
    assert s["peak_used_gib"] == 100 and s["min_available_gib"] == 5 and s["max_pressure_level"] == 4


# ── singleton ───────────────────────────────────────────────────────────────────────────────────
def test_singleton_lock_is_exclusive(tmp_path, monkeypatch):
    lock = tmp_path / ".singleton.lock"
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", lock)
    fd1 = mbb._acquire_singleton()
    assert fd1 is not None
    fd2 = mbb._acquire_singleton()   # second acquire must fail (already held)
    assert fd2 is None
    import os
    os.close(fd1)


def test_run_daemon_exits_when_singleton_held(tmp_path, monkeypatch):
    lock = tmp_path / ".singleton.lock"
    monkeypatch.setattr(mbb, "_SINGLETON_LOCK", lock)
    held = mbb._acquire_singleton()
    assert held is not None
    rc = mbb.run_daemon(max_iterations=1, govern=False)  # cannot get the lock -> exits 0 immediately
    assert rc == 0
    import os
    os.close(held)
