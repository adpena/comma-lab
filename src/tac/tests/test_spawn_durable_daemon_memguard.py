"""NO-FAKE tests for the OOM launch-preflight wire-in + dead-registry reconcile
in tools/spawn_durable_daemon.py."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_SD_PATH = Path(__file__).resolve().parents[3] / "tools" / "spawn_durable_daemon.py"
_spec = importlib.util.spec_from_file_location("tac_spawn_durable_daemon_under_test", _SD_PATH)
sd = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["tac_spawn_durable_daemon_under_test"] = sd
_spec.loader.exec_module(sd)


def _args(**kw):
    base = dict(skip_mem_preflight=False, min_free_gb=30.0, projected_gb=25.0,
                rss_cap_mb=None, walltime_cap_s=None, label=None)
    base.update(kw)
    return SimpleNamespace(**base)


# --- launch preflight -------------------------------------------------------
def test_preflight_refuses_impossible_projection():
    # 999 GB projected exceeds any real machine's free memory → REFUSE (rc 3).
    assert sd._mem_preflight(_args(projected_gb=999.0)) == 3


def test_preflight_skip_bypasses_even_impossible_projection():
    assert sd._mem_preflight(_args(skip_mem_preflight=True, projected_gb=999.0)) is None


def test_preflight_ok_with_zero_floor_and_zero_projection():
    # floor 0, projected 0 → headroom == available >= 0 always → proceed (None).
    assert sd._mem_preflight(_args(min_free_gb=0.0, projected_gb=0.0)) is None


# --- reconcile --------------------------------------------------------------
def test_reconcile_marks_dead_running_rows_stopped(tmp_path, monkeypatch):
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    # pid 999999 is (essentially certainly) not alive → a dead "running" row.
    reg.write_text(json.dumps([
        {"label": "dead_arm", "pid": 999999, "pgid": 999999, "cmd": ["python", "x.py"], "status": "running"},
        {"label": "already_stopped", "pid": 999998, "pgid": 999998, "cmd": "y", "status": "stopped"},
    ]))
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    rc = sd._do_reconcile()
    assert rc == 0
    rows = json.loads(reg.read_text())
    by_label = {r["label"]: r for r in rows}
    assert by_label["dead_arm"]["status"] == "stopped"
    assert by_label["dead_arm"]["stopped_reason"] == "reconcile_dead_process"
    assert by_label["already_stopped"]["status"] == "stopped"


# --- D3: per-arm safe_run cap wrapping (layer 3) -----------------------------
TRAIN = [".venv/bin/python", "experiments/train_witness_realized_through_R_mlx.py", "--num-pairs", "600"]


def test_maybe_wrap_safe_run_wraps_when_rss_cap_set():
    wrapped = sd._maybe_wrap_safe_run(list(TRAIN), _args(rss_cap_mb=30000, label="n600"))
    joined = " ".join(wrapped)
    assert "safe_run.py" in joined
    assert "--rss-mb" in wrapped and "30000" in wrapped
    assert "--timeout" in wrapped  # walltime defaulted (~14d) so safe_run's 30s default never fires
    assert "--" in wrapped
    # the trainer script token MUST survive so the watchdog custody identity gate matches.
    assert any("train_witness_realized_through_R_mlx.py" in p for p in wrapped)


def test_maybe_wrap_safe_run_respects_explicit_walltime():
    wrapped = sd._maybe_wrap_safe_run(list(TRAIN), _args(rss_cap_mb=20000, walltime_cap_s=3600.0, label="x"))
    assert "3600.0" in wrapped


def test_maybe_wrap_safe_run_noop_when_no_cap():
    cmd = ["python", "x.py"]
    assert sd._maybe_wrap_safe_run(list(cmd), _args(rss_cap_mb=None, walltime_cap_s=None)) == cmd


# --- HIGH-1 make-or-break: external killpg(custody pgid) must CASCADE to the
#     wrapped inner arm (not leave it reparented-alive + uncapped). RUNTIME test.
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _find_child_sleep(wrapper_pid: int, deadline_s: float = 4.0) -> int | None:
    """Find the inner `sleep` spawned by the safe_run wrapper (its child).

    The wrapper spawns exactly one child (the sleep), so ppid == wrapper_pid
    uniquely identifies the inner arm."""
    end = time.time() + deadline_s
    while time.time() < end:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid, ppid = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if ppid == wrapper_pid and "sleep" in parts[2] and "ps -axo" not in parts[2]:
                return pid
        time.sleep(0.1)
    return None


def test_killpg_custody_pgid_cascades_to_wrapped_inner_arm(tmp_path, monkeypatch):
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    log = tmp_path / "d.log"
    # Launch a bounded NON-training inner child (sleep) WRAPPED in safe_run via the
    # canonical daemon launcher (--rss-cap-mb wraps it; --skip-mem-preflight so the
    # floor gate doesn't interfere). The custody record will hold the WRAPPER pgid.
    a = SimpleNamespace(
        cmd=["--", "/bin/sleep", "300"],
        log=str(log), label="cascade_test", skip_mem_preflight=True,
        min_free_gb=0.0, projected_gb=0.0, rss_cap_mb=4000, walltime_cap_s=120,
    )
    wrapper_pid = wrapper_pgid = inner = None
    try:
        rc = sd._do_start(a)
        assert rc == 0
        rows = json.loads(reg.read_text())
        rec = next(r for r in rows if r["label"] == "cascade_test")
        wrapper_pid, wrapper_pgid = int(rec["pid"]), int(rec["pgid"])
        # the recorded cmd MUST be the safe_run-WRAPPED command (layer 3 active).
        assert any("safe_run.py" in c for c in rec["cmd"])
        # find the inner sleep (child of the wrapper) and prove it is alive.
        inner = _find_child_sleep(wrapper_pid)
        assert inner is not None and _pid_alive(inner), "inner arm should be running"
        # SHED simulation: the watchdog kills the recorded custody pgid (wrapper).
        os.killpg(wrapper_pgid, signal.SIGTERM)
        # the cascade must kill the INNER arm (not leave it reparented-alive).
        end = time.time() + 8.0
        while time.time() < end and _pid_alive(inner):
            time.sleep(0.1)
        assert not _pid_alive(inner), (
            "HIGH-1: killpg(custody/wrapper pgid) did NOT cascade to the inner arm "
            "— it would reparent to launchd and keep running UNCAPPED"
        )
    finally:
        for p in (inner, wrapper_pid):
            if p:
                with __import__("contextlib").suppress(ProcessLookupError, PermissionError):
                    os.kill(p, signal.SIGKILL)


def test_reconcile_noop_when_clean(tmp_path, monkeypatch, capsys):
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    reg.write_text(json.dumps([
        {"label": "s", "pid": 999998, "pgid": 999998, "cmd": "y", "status": "stopped"},
    ]))
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    assert sd._do_reconcile() == 0
    assert "already accurate" in capsys.readouterr().out
