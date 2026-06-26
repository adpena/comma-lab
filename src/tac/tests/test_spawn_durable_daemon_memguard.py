"""NO-FAKE tests for the OOM launch-preflight wire-in + dead-registry reconcile
in tools/spawn_durable_daemon.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
