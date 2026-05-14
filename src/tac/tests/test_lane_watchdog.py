# SPDX-License-Identifier: MIT
"""Smoke + alert-detection tests for scripts/lane_watchdog.py.

DX #9 (2026-04-26): the watchdog catches the SHIRAZ failure mode (16h of
GPU billed, no measurement). These tests verify the alert classifier
fires on the three documented failure cases:

    STALE_HEARTBEAT, IDLE_BURN, NO_PROGRESS
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "lane_watchdog.py"


def _load_watchdog():
    if "lane_watchdog" in sys.modules:
        return sys.modules["lane_watchdog"]
    spec = importlib.util.spec_from_file_location("lane_watchdog", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lane_watchdog"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_loads():
    mod = _load_watchdog()
    assert hasattr(mod, "InstanceState")
    assert hasattr(mod, "evaluate_instance")
    assert hasattr(mod, "poll_once")


def test_idle_burn_fires(monkeypatch):
    """A running instance with gpu_util=0 for > IDLE_GPU_WINDOW_SEC must
    yield an IDLE_BURN alert exactly once."""
    mod = _load_watchdog()
    # Skip the SSH probe in tests — return None so STALE_HEARTBEAT is silent.
    monkeypatch.setattr(mod, "_instance_remote_heartbeat_mtime", lambda inst: None)
    state = mod.InstanceState(instance_id="123", label="test")
    # First poll: gpu_util=0, but only seconds in. Should not fire.
    now = time.time()
    state.first_seen_at = now - 30
    state.last_gpu_busy_at = now - 30
    inst = {"id": 123, "actual_status": "running", "gpu_util": 0,
            "dph_total": 0.25}
    alerts = mod.evaluate_instance(inst, state, now)
    assert not any(a["kind"] == "IDLE_BURN" for a in alerts)
    # Second poll: still idle, now > IDLE_GPU_WINDOW_SEC ago. Must fire.
    later = now + mod.IDLE_GPU_WINDOW_SEC + 5
    alerts = mod.evaluate_instance(inst, state, later)
    assert any(a["kind"] == "IDLE_BURN" for a in alerts), \
        f"expected IDLE_BURN, got {[a['kind'] for a in alerts]}"
    # Third poll: same condition. Already-emitted, must not double-fire.
    alerts = mod.evaluate_instance(inst, state, later + 60)
    assert not any(a["kind"] == "IDLE_BURN" for a in alerts)


def test_no_progress_fires_on_dollar_threshold(monkeypatch):
    """Instance billed > $5 with no result yet must yield NO_PROGRESS."""
    mod = _load_watchdog()
    monkeypatch.setattr(mod, "_instance_remote_heartbeat_mtime", lambda inst: None)
    state = mod.InstanceState(instance_id="42")
    # $0.50/hr × 12h = $6 ⇒ over $5 threshold.
    now = time.time()
    state.first_seen_at = now - 12 * 3600
    state.last_gpu_busy_at = now  # not idle
    inst = {"id": 42, "actual_status": "running", "gpu_util": 50,
            "dph_total": 0.50}
    alerts = mod.evaluate_instance(inst, state, now)
    assert any(a["kind"] == "NO_PROGRESS" for a in alerts)


def test_stale_heartbeat_fires(monkeypatch):
    """Heartbeat older than STALE_HEARTBEAT_SEC must alert."""
    mod = _load_watchdog()
    now = time.time()
    monkeypatch.setattr(
        mod, "_instance_remote_heartbeat_mtime",
        lambda inst: now - mod.STALE_HEARTBEAT_SEC - 60,
    )
    state = mod.InstanceState(instance_id="99")
    state.first_seen_at = now - 60
    state.last_gpu_busy_at = now
    inst = {"id": 99, "actual_status": "running", "gpu_util": 80,
            "dph_total": 0.25}
    alerts = mod.evaluate_instance(inst, state, now)
    assert any(a["kind"] == "STALE_HEARTBEAT" for a in alerts), \
        f"expected STALE_HEARTBEAT, got {[a['kind'] for a in alerts]}"


def test_healthy_instance_emits_nothing(monkeypatch):
    """Active GPU + fresh heartbeat + low cost ⇒ silence."""
    mod = _load_watchdog()
    now = time.time()
    monkeypatch.setattr(
        mod, "_instance_remote_heartbeat_mtime", lambda inst: now - 30,
    )
    state = mod.InstanceState(instance_id="77")
    state.first_seen_at = now - 600
    state.last_gpu_busy_at = now
    inst = {"id": 77, "actual_status": "running", "gpu_util": 95,
            "dph_total": 0.25}
    alerts = mod.evaluate_instance(inst, state, now)
    assert alerts == [], f"expected no alerts, got {alerts}"
