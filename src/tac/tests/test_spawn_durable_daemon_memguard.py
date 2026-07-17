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


def _start_args(**kw):
    """Full namespace for _do_start (start-mode)."""
    base = dict(
        cmd=["--", "sleep", "1"], log=None, label="test_job",
        skip_mem_preflight=True, min_free_gb=0.0, projected_gb=25.0,
        rss_cap_mb=None, walltime_cap_s=None,
        skip_admission_gate=False, projected_peak_gib=25.0, priority=None,
        admission_override_rationale=None, skip_readiness_gate=True,
        readiness_override_rationale=None, skip_blackbox_autostart=True, verify_s=0.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _admitting_rows(reg):
    return [r for r in json.loads(reg.read_text()) if r.get("status") == "admitting"]


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


# --- 2026-07-11 phantom-reservation fix: reconcile converges the pending store, idempotently -------
def _point_registry(monkeypatch, tmp_path):
    reg = tmp_path / "durable_daemons.json"
    lock = tmp_path / ".durable_daemons.lock"
    monkeypatch.setattr(sd, "_REGISTRY_PATH", reg)
    monkeypatch.setattr(sd, "_REGISTRY_LOCK", lock)
    return reg


def test_reconcile_drops_stale_pending_reservation(tmp_path, monkeypatch):
    """A crashed launcher leaves a STALE 'admitting' row (no pid, old reserved_ts). The admission
    gate counts FRESH pending rows as growth headroom, so a stale one is phantom growth. reconcile
    (which the admission path now calls) must DROP it — previously reconcile ignored pending rows,
    the exact reason active_jobs stayed high after --reconcile marked dead running rows."""
    now = time.time()
    reg = _point_registry(monkeypatch, tmp_path)
    reg.write_text(json.dumps([
        {"label": "crashed", "pid": None, "status": "admitting",
         "projected_peak_gib": 60.0, "reserved_ts": now - 9999.0},
        {"label": "fresh", "pid": None, "status": "admitting",
         "projected_peak_gib": 60.0, "reserved_ts": now - 1.0},
    ]))
    n = sd.reconcile_dead_daemons(verbose=False, now_ts=now)
    assert n == 1  # only the stale one
    labels = {r["label"]: r for r in json.loads(reg.read_text())}
    assert "crashed" not in labels          # stale pending DROPPED
    assert labels["fresh"]["status"] == "admitting"  # fresh in-flight reservation PRESERVED


def test_reconcile_is_idempotent(tmp_path, monkeypatch):
    """Running reconcile twice reconciles nothing the second time (ground-truth convergence)."""
    now = time.time()
    reg = _point_registry(monkeypatch, tmp_path)
    reg.write_text(json.dumps([
        {"label": "dead", "pid": 999999, "pgid": 999999, "cmd": ["python", "x.py"], "status": "running"},
        {"label": "stale_pending", "pid": None, "status": "admitting",
         "projected_peak_gib": 60.0, "reserved_ts": now - 9999.0},
    ]))
    first = sd.reconcile_dead_daemons(verbose=False, now_ts=now)
    second = sd.reconcile_dead_daemons(verbose=False, now_ts=now)
    assert first == 2 and second == 0


# --- 2026-07-11: a REFUSED admission leaves ZERO net reservation (no accumulation on retry) --------
def test_refused_admission_leaves_zero_net_reservation(tmp_path, monkeypatch):
    """The invariant the phantom-accumulation bug violated: when the admission gate REFUSES, the
    launcher must return before writing any pending reservation — so a refused (and re-refused)
    launch never leaves an 'admitting' row that would inflate the NEXT admission projection."""
    reg = _point_registry(monkeypatch, tmp_path)
    reg.write_text(json.dumps([]))
    # Gate refuses (rc=5) — the enforce-mode REFUSE path.
    monkeypatch.setattr(sd, "_system_admission_gate", lambda a, cmd: 5)
    log = tmp_path / "d.log"
    rc1 = sd._do_start(_start_args(log=str(log), label="witness_resume"))
    assert rc1 == 5
    assert _admitting_rows(reg) == []            # refused -> ZERO net reservation
    # retry: still zero (each refused retry must NOT make it worse).
    rc2 = sd._do_start(_start_args(log=str(log), label="witness_resume"))
    assert rc2 == 5
    assert _admitting_rows(reg) == []


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
        # skip_admission_gate: this test exercises killpg CUSTODY, not admission — the designed
        # infra bypass keeps it hermetic vs live machine memory state (the committed-basis gate
        # correctly refuses any spawn while a ~100 GiB bench runs).
        skip_admission_gate=True,
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


# --- observability SENSE-organ admission exemption (facet-2 observer-gap fix) -----
# The score-neutral #247 costate observer must be admission-exempt from the SUM-over-RAM
# gate: once a large trainer fills the box, the aggregate gate refuses EVERYTHING, which
# would silently orphan read-only observability that "defaults ON". Per-arm RSS cap +
# free-floor preflight still apply, so a genuine OOM is still impossible.
def test_protection_infra_exempts_costate_observer_loop():
    cmd = [sys.executable, "tools/costate_observer_loop.py", "--run-dir", "experiments/results/x"]
    assert sd._is_protection_infra_cmd(cmd) is True


def test_protection_infra_exempts_costate_shadow_report():
    cmd = [sys.executable, "tools/costate_shadow_report.py", "--run-dir", "experiments/results/x", "--write"]
    assert sd._is_protection_infra_cmd(cmd) is True


def test_protection_infra_still_exempts_memory_guard():
    assert sd._is_protection_infra_cmd([sys.executable, "tools/memory_guard.py"]) is True
    assert sd._is_protection_infra_cmd([sys.executable, "tools/system_memory_governor.py"]) is True


def test_protection_infra_does_NOT_exempt_the_trainer():
    # The witness trainer + its bash launch.sh wrapper must remain FULLY admission-gated —
    # over-exempting the heavy trainer would defeat the SUM-over-RAM crash guard.
    trainer = [".venv/bin/python", "experiments/train_levelset_witness_realized_through_R_mlx.py",
               "--num-pairs", "600", "--out-dir", "experiments/results/levelset_v752_baseline"]
    assert sd._is_protection_infra_cmd(trainer) is False
    launch_sh = ["bash", "experiments/results/levelset_v752_baseline/launch.sh"]
    assert sd._is_protection_infra_cmd(launch_sh) is False


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
