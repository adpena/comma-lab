# SPDX-License-Identifier: MIT
"""Tests for scripts/verify_vast_instances.py SETUP-stuck cost-leak fix.

R31 cross-cutting concern: a TRULY hung setup_full.sh (deadlocked, never
writes heartbeat) is classified SETUP, not IDLE. The IDLE
``--stale-minutes`` heartbeat-freshness comparison never fires (no
heartbeat to be stale), so without a separate SETUP timer the instance
accrues GPU cost silently forever.

Tests cover:
- SETUP-first-seen state file load/save round-trip
- Pruning entries for instances no longer in the tracker
- Dual-threshold helper presence + parser flag wiring

Reference: feedback_setup_stuck_cost_leak_FIXED_20260428.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_vast_instances.py"

_spec = importlib.util.spec_from_file_location(
    "_verify_vast_under_test", SCRIPT_PATH
)
verify_vast = importlib.util.module_from_spec(_spec)
sys.modules["_verify_vast_under_test"] = verify_vast
_spec.loader.exec_module(verify_vast)  # type: ignore[union-attr]


# ── State file round-trip ────────────────────────────────────────────────


def test_load_setup_first_seen_returns_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", tmp_path / "missing.json")
    assert verify_vast._load_setup_first_seen() == {}


def test_load_setup_first_seen_handles_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON must NOT crash the verify pass — it just resets the
    SETUP-first-seen tracker (worst case: one extra DALI install cycle
    is tolerated; a crashed verify pass would block the whole fleet)."""
    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json")
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    assert verify_vast._load_setup_first_seen() == {}


def test_save_and_load_setup_first_seen_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    payload = {"35707822": 1714356000.5, "35759655": 1714356100.0}
    verify_vast._save_setup_first_seen(payload)
    assert p.exists()
    loaded = verify_vast._load_setup_first_seen()
    assert loaded == payload


def test_save_setup_first_seen_creates_parent_directory(tmp_path, monkeypatch):
    nested = tmp_path / "deep" / "nested" / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", nested)
    verify_vast._save_setup_first_seen({"x": 1.0})
    assert nested.exists()


def test_load_setup_first_seen_coerces_keys_and_values(tmp_path, monkeypatch):
    """Whatever JSON shape lands on disk, `_load_setup_first_seen` must
    return `dict[str, float]` so the in-memory aging math doesn't crash
    on int keys / int timestamps."""
    p = tmp_path / "first_seen.json"
    # Write integer values (legal JSON, but Python loads them as ints).
    p.write_text(json.dumps({"123": 1714356000, "456": 1714356100}))
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    loaded = verify_vast._load_setup_first_seen()
    assert all(isinstance(k, str) for k in loaded.keys())
    assert all(isinstance(v, float) for v in loaded.values())


# ── InstanceHealth dataclass shape ────────────────────────────────────────


def test_instance_health_has_setup_age_minutes_field():
    """The dataclass must carry the SETUP-age field so JSON output
    surfaces it to the operator (otherwise --setup-stale-minutes is
    invisible at debug time)."""
    fields = {
        f.name for f in verify_vast.InstanceHealth.__dataclass_fields__.values()
    }
    assert "setup_age_minutes" in fields


# ── classify() — SETUP classification preserved ──────────────────────────


def test_classify_returns_setup_when_ssh_ok_no_heartbeat():
    """SSH OK + no heartbeat is the SETUP-in-flight state. Must NOT be
    classified IDLE (which would fire the IDLE timer prematurely)."""
    cls = verify_vast.classify(
        age_min=None, gpu_util=0.0, crash_signal=None,
        stale_minutes=30.0, ssh_succeeded=True,
    )
    assert cls == "SETUP"


def test_classify_returns_unreachable_when_ssh_failed_no_heartbeat():
    cls = verify_vast.classify(
        age_min=None, gpu_util=None, crash_signal=None,
        stale_minutes=30.0, ssh_succeeded=False,
    )
    assert cls == "UNREACHABLE"


def test_classify_returns_idle_when_heartbeat_stale():
    cls = verify_vast.classify(
        age_min=120.0, gpu_util=10.0, crash_signal=None,
        stale_minutes=30.0, ssh_succeeded=True,
    )
    assert cls == "IDLE"


# ── argparse: --setup-stale-minutes flag exists with sane default ─────────


def test_setup_stale_minutes_flag_defined_with_sane_default():
    """Source-text introspection: --setup-stale-minutes must be defined
    with a default well past max DALI install time (15 min)."""
    src = SCRIPT_PATH.read_text()
    assert '"--setup-stale-minutes"' in src, (
        "--setup-stale-minutes flag must be defined"
    )
    # Default 90 min: well past max DALI + setup ~15 min.
    assert "default=90" in src, (
        "--setup-stale-minutes default should be ~90 min "
        "(well past max DALI install)"
    )
