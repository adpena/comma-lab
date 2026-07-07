# SPDX-License-Identifier: MIT
"""ADMISSION-RESERVATION TOCTOU tests (memory-safety whole-subsystem review, CRITICAL C + HIGH D).

The bug class: ``spawn_durable_daemon._do_start`` read the registry for its admission decision
with no lock and only reserved (via ``_register_daemon``) AFTER Popen — so two near-simultaneous
governed launches could BOTH pass admission before either was visible to the other (each
individually under the ceiling; their SUM over it = the 2026-07-02 crash shape). The fix is an
fcntl-locked {stale-pending sweep -> decision -> PENDING reservation write} critical section, a
governor read side that COUNTS fresh pending reservations as growth headroom, and a stale-pending
sweep so a crashed launcher can never leak a phantom reservation.

Everything here runs against a TMP registry (monkeypatched module paths) — the LIVE
``.omx/state/durable_daemons.json`` (which holds the running #205 mod32cap row) is never touched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import spawn_durable_daemon as sd  # noqa: E402
import system_memory_governor as gov  # noqa: E402
import safe_run as sr  # noqa: E402


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    """Hermetic registry: every module-level default path points at tmp (never the live one)."""
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    return reg


def _rows(reg: Path) -> list[dict]:
    return json.loads(reg.read_text()) if reg.exists() else []


# ── the reservation primitives ───────────────────────────────────────────────────────────────────
def test_write_pending_reservation_shape(tmp_registry):
    rec = sd._write_pending_reservation("job_a", 50.0)
    rows = _rows(tmp_registry)
    assert len(rows) == 1
    r = rows[0]
    assert r["label"] == "job_a"
    assert r["status"] == sd._PENDING_STATUS == "admitting"
    assert r["pid"] is None
    assert r["projected_peak_gib"] == 50.0
    assert isinstance(r["reserved_ts"], float)
    assert abs(r["reserved_ts"] - time.time()) < 10.0
    assert rec["label"] == "job_a"


def test_two_sequential_admits_see_each_others_pending_row(tmp_registry):
    """THE TOCTOU regression: launcher A writes its reservation inside the lock; launcher B's
    admission read (governor list_tracked_jobs over the same registry) then counts A's projected
    peak as growth headroom — B can no longer be admitted as if A did not exist."""
    if gov._mg is None:
        pytest.skip("memory_guard unavailable (fail-safe: no tracked jobs)")
    # launcher A: admitted -> pending reservation written (what _do_start does inside the lock).
    sd._write_pending_reservation("launcher_a", 50.0)
    # launcher B: reads the SAME registry for its own admission arithmetic.
    jobs = gov.list_tracked_jobs(samples={}, registry_rows=_rows(tmp_registry))
    headroom = gov.sum_active_growth_headroom_gib(jobs)
    assert headroom >= 50.0, "B's admission must see A's reserved projected peak"
    ceiling = gov.compute_adaptive_ceiling(total_gib=128.0, used_gib=20.0, tracked_current_gib=0.0)
    b = gov.admission_decision(projected_new_gib=60.0, system_used_gib=20.0,
                               active_growth_headroom_gib=headroom, ceiling=ceiling)
    assert not b.admit, "20 used + 50 reserved + 60 new must NOT fit under the ~101.6 ceiling"
    # and B in turn reserves; a third launcher sees BOTH.
    sd._write_pending_reservation("launcher_b", 60.0)
    jobs3 = gov.list_tracked_jobs(samples={}, registry_rows=_rows(tmp_registry))
    assert gov.sum_active_growth_headroom_gib(jobs3) >= 110.0


def test_register_daemon_promotes_pending_row_same_label(tmp_registry):
    """The upsert-by-label promote: after Popen the real running row REPLACES the reservation —
    exactly one row per label, no double count."""
    sd._write_pending_reservation("job_a", 50.0)
    sd._register_daemon({"label": "job_a", "pid": 12345, "pgid": 12345, "cmd": ["x"],
                         "log": "", "started_utc": "t", "cwd": ".", "status": "running",
                         "projected_peak_gib": 50.0})
    rows = [r for r in _rows(tmp_registry) if r["label"] == "job_a"]
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["pid"] == 12345


def test_clear_pending_reservation_only_drops_admitting(tmp_registry):
    sd._write_pending_reservation("job_a", 50.0)
    sd._register_daemon({"label": "job_b", "pid": 7, "pgid": 7, "cmd": [], "log": "",
                         "started_utc": "t", "cwd": ".", "status": "running"})
    sd._clear_pending_reservation("job_a")
    sd._clear_pending_reservation("job_b")   # running row must NOT be dropped
    labels = {r["label"]: r["status"] for r in _rows(tmp_registry)}
    assert labels == {"job_b": "running"}


def test_sweep_drops_stale_and_malformed_keeps_fresh_and_running():
    now = 1_000_000.0
    rows = [
        {"label": "fresh", "pid": None, "status": "admitting", "reserved_ts": now - 5.0},
        {"label": "stale", "pid": None, "status": "admitting", "reserved_ts": now - 999.0},
        {"label": "no_ts", "pid": None, "status": "admitting"},                    # malformed
        {"label": "bad_ts", "pid": None, "status": "admitting", "reserved_ts": "x"},
        {"label": "running", "pid": 42, "status": "running"},
        {"label": "stopped", "pid": 41, "status": "stopped"},
    ]
    kept = sd._sweep_stale_pending_rows(rows, now_ts=now)
    assert [r["label"] for r in kept] == ["fresh", "running", "stopped"]


# ── _do_start integration (gate monkeypatched ADMIT so the test never depends on live pressure) ──
def _start_args(**kw):
    base = dict(
        cmd=kw.pop("cmd"), log=kw.pop("log"), label=kw.pop("label"),
        skip_mem_preflight=True, min_free_gb=0.0, projected_gb=0.0,
        rss_cap_mb=None, walltime_cap_s=None, verify_s=kw.pop("verify_s", 1.0),
        projected_peak_gib=kw.pop("projected_peak_gib", 33.0),
        skip_admission_gate=False, skip_blackbox_autostart=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_do_start_promotes_reservation_no_pending_left(tmp_registry, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "_system_admission_gate", lambda a, cmd: None)  # deterministic ADMIT
    a = _start_args(cmd=["--", "/bin/sleep", "10"], log=str(tmp_path / "l.log"),
                    label="toctou_live_throwaway")
    pid = None
    try:
        assert sd._do_start(a) == 0
        rows = _rows(tmp_registry)
        mine = [r for r in rows if r["label"] == "toctou_live_throwaway"]
        assert len(mine) == 1
        assert mine[0]["status"] == "running"
        assert mine[0]["projected_peak_gib"] == 33.0
        pid = int(mine[0]["pid"])
        assert not [r for r in rows if r.get("status") == sd._PENDING_STATUS], \
            "the pending reservation must be PROMOTED, never left behind"
    finally:
        if pid:
            import contextlib
            import os
            import signal
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(pid, signal.SIGKILL)


def test_do_start_spawn_failure_releases_reservation(tmp_registry, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "_system_admission_gate", lambda a, cmd: None)
    a = _start_args(cmd=["--", "/nonexistent/binary_toctou_throwaway"],
                    log=str(tmp_path / "dead.log"), label="toctou_dead_throwaway")
    assert sd._do_start(a) == 4
    assert not [r for r in _rows(tmp_registry) if r.get("status") == sd._PENDING_STATUS], \
        "a failed spawn must not leak a phantom reservation"


def test_do_start_refusal_writes_no_reservation(tmp_registry, tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "_system_admission_gate", lambda a, cmd: 5)  # deterministic REFUSE
    a = _start_args(cmd=["--", "/bin/sleep", "10"], log=str(tmp_path / "r.log"),
                    label="toctou_refused_throwaway")
    assert sd._do_start(a) == 5
    assert _rows(tmp_registry) == []


# ── safe_run side (HIGH D: bare safe_run visibility) ─────────────────────────────────────────────
def test_safe_run_gate_and_reserve_skipped_for_daemon_wrapped():
    """--skip-admission-gate (the daemon-wrapped path) must neither gate nor reserve — the daemon
    is the single gate point and already owns the registry row."""
    ns = SimpleNamespace(skip_admission_gate=True, projected_gib=None, rss_mb=1000,
                         label="x", admission_override_rationale=None)
    rc, reservation = sr._gate_and_reserve(ns, ["/bin/true"])
    assert rc is None and reservation is None


def test_safe_run_gate_and_reserve_hermetic_under_pytest(monkeypatch):
    """Under pytest the reservation write is skipped (never touch the LIVE registry from a test);
    the gate itself still runs (monkeypatched here to avoid a live governor read)."""
    monkeypatch.setattr(sr, "_system_admission_gate", lambda ns, cmd: None)
    ns = SimpleNamespace(skip_admission_gate=False, projected_gib=5.0, rss_mb=1000,
                         label="x", admission_override_rationale=None)
    rc, reservation = sr._gate_and_reserve(ns, ["/bin/true"])
    assert rc is None and reservation is None  # gate ran, hermetic -> no registry write


def test_safe_run_activate_and_release_reservation(tmp_registry):
    """The bare-safe_run lifecycle against a TMP registry: pending -> running (real pid + projected
    peak visible to the governor) -> stopped on exit."""
    label = "saferun_test_throwaway_pid999"
    sd._write_pending_reservation(label, 12.0)
    sr._activate_reservation((sd, label, 12.0), 999, 999, ["/bin/true"])
    rows = [r for r in _rows(tmp_registry) if r["label"] == label]
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["pid"] == 999
    assert rows[0]["projected_peak_gib"] == 12.0
    sr._release_reservation((sd, label, 12.0), reason="safe_run_exit_ok")
    rows = [r for r in _rows(tmp_registry) if r["label"] == label]
    assert len(rows) == 1
    assert rows[0]["status"] == "stopped"
    assert rows[0]["stopped_reason"] == "safe_run_exit_ok"


def test_safe_run_release_drops_never_activated_reservation(tmp_registry):
    """Spawn failure path: a still-pending reservation is DROPPED (not marked stopped) so it can
    never linger as phantom growth headroom."""
    label = "saferun_dead_throwaway_pid998"
    sd._write_pending_reservation(label, 12.0)
    sr._release_reservation((sd, label, 12.0), reason="safe_run_spawn_failed")
    assert [r for r in _rows(tmp_registry) if r["label"] == label] == []


def test_safe_run_release_none_reservation_is_noop():
    sr._release_reservation(None, reason="x")
    sr._activate_reservation(None, 1, 1, [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
