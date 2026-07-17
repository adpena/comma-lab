"""tools/witness_chain_watchdog.py — B4 composite chain-liveness verdicts
(p0_launcher_chain_durability_20260717). The phantom-death class: verdicts must come from
pid-tree x run-dir mtimes x receipt presence, never log tails / grep pipelines / registry
status."""
import json
import os
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "tools"))

import witness_chain_watchdog as W  # noqa: E402


def _registry(tmp_path, rows):
    p = tmp_path / "durable_daemons.json"
    p.write_text(json.dumps(rows))
    return p


def _run_dir(tmp_path, name="experiments/results/levelset_n600_witness_test", receipt=False):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "launch_manifest.json").write_text("{}")
    if receipt:
        (d / "dry_start_report.json").write_text('{"green": false}')
    return d


def _row(label, pid, run_dir=None, status="running"):
    cmd = ["bash", f"{run_dir}/dry_start/launch.sh"] if run_dir else ["python", "x.py"]
    return {"label": label, "pid": pid, "pgid": pid, "status": status, "cmd": cmd, "log": ""}


def test_dead_pid_no_receipt_is_the_alarm(tmp_path):
    d = _run_dir(tmp_path, receipt=False)
    reg = _registry(tmp_path, [_row("drystart_dead", 2, run_dir=d)])  # pid 2 not ours/alive
    # pid 2 exists on macOS (kernel_task) but os.kill(2,0) raises PermissionError -> alive.
    # Use an impossible pid instead.
    reg = _registry(tmp_path, [_row("drystart_dead", 2**22 + 12345, run_dir=d)])
    v = W.scan(registry_path=reg)
    assert len(v) == 1 and v[0]["verdict"] == "CHAIN_DEAD_NO_RECEIPT"
    assert "SILENT DEATH" in v[0]["note"]


def test_dead_pid_with_receipt_is_receipted(tmp_path):
    d = _run_dir(tmp_path, receipt=True)
    reg = _registry(tmp_path, [_row("drystart_done", 2**22 + 12346, run_dir=d)])
    v = W.scan(registry_path=reg)
    assert v[0]["verdict"] == "CHAIN_DEAD_RECEIPTED"


def test_alive_pid_quiet_vs_healthy(tmp_path, monkeypatch):
    d = _run_dir(tmp_path)
    (d / "fresh_file").write_text("x")  # newest mtime = now -> HEALTHY
    reg = _registry(tmp_path, [_row("witness_live", os.getpid(), run_dir=d)])
    # F3b: the cross-check reads the LIVE cmdline; pytest's own cmdline carries no chain
    # token, so stub it to the registered launch.sh (the cross-check itself is tested below).
    monkeypatch.setattr(W, "_live_cmdline", lambda pid: f"bash {d}/dry_start/launch.sh")
    v = W.scan(registry_path=reg, stale_s=900.0)
    assert v[0]["verdict"] == "RUNNING_HEALTHY" and v[0]["alive"] is True
    # stale threshold 0 -> everything is stale -> QUIET, loudly labeled ALIVE
    v2 = W.scan(registry_path=reg, stale_s=0.0)
    assert v2[0]["verdict"] == "RUNNING_QUIET"
    assert "ALIVE despite quiet" in v2[0]["note"]


def test_non_witness_labels_and_stopped_rows_out_of_scope(tmp_path):
    reg = _registry(tmp_path, [
        _row("memory_blackbox", os.getpid()),
        _row("drystart_x", os.getpid(), status="stopped"),
    ])
    assert W.scan(registry_path=reg) == []


def test_run_dir_requires_launch_artifacts_no_fuzzy_cache_match(tmp_path):
    """Live-fire finding: a cmd token like the gt-cache dir under experiments/results/ must
    NOT resolve as the chain's run dir (it lacks launch artifacts)."""
    cache = tmp_path / "experiments/results/mlx_fleet_gt_cache"
    cache.mkdir(parents=True)
    row = {"cmd": [f"--gt-cache {cache}/gt_n600.npz"], "log": ""}
    real = W._REPO
    try:
        W._REPO = tmp_path  # relative-fragment base
        assert W._run_dir_from_row(row) is None
        d = _run_dir(tmp_path)
        row2 = {"cmd": [f"bash {d}/dry_start/launch.sh"], "log": ""}
        assert W._run_dir_from_row(row2) == d
    finally:
        W._REPO = real


def test_descendants_walk():
    kids = {1: [2, 3], 2: [4], 4: [5]}
    assert sorted(W._descendants(1, kids)) == [2, 3, 4, 5]
    assert W._descendants(5, kids) == []


# ───────── F3 (independent review 2026-07-17): pid-reuse cross-check + chain manifest ─────────

def test_pid_reuse_reads_dead(monkeypatch):
    """F3b: a live pid whose cmdline matches NEITHER the registered tokens NOR any chain
    token is pid-REUSE -> dead for this chain (bare kill(pid,0) said alive forever)."""
    monkeypatch.setattr(W, "_live_cmdline", lambda pid: "/usr/libexec/some_unrelated_daemon")
    alive, live = W._pid_alive_cmd(os.getpid(), ["launch.sh"])
    assert alive is False and "unrelated" in live


def test_exec_chain_fallback_reads_alive(monkeypatch):
    """F3b nuance: an exec-chain replaces the registered argv (waiter bash execs safe_run);
    a live cmdline carrying a KNOWN chain token stays alive."""
    monkeypatch.setattr(W, "_live_cmdline",
                        lambda pid: ".venv/bin/python tools/safe_run.py --timeout 10800")
    alive, _ = W._pid_alive_cmd(os.getpid(), ["v3_delta_waiter.sh"])
    assert alive is True


def test_ambiguous_multi_artifact_dirs_resolve_none(tmp_path):
    """F3c: a v3-waiter-style argv carries PRIOR run dirs (delta-from + observer-evidence)
    that have launch artifacts AND green receipts — ambiguous resolution must be None, not
    a false CHAIN_DEAD_RECEIPTED."""
    d1 = _run_dir(tmp_path, "experiments/results/levelset_prior1", receipt=True)
    d2 = _run_dir(tmp_path, "experiments/results/levelset_prior2", receipt=True)
    row = {"cmd": [f"--dry-start-delta-from {d1}", f"--observer-cost-evidence {d2}"], "log": ""}
    assert W._run_dir_from_row(row) is None


def test_manifest_silent_launcher_death_alarms(tmp_path):
    """F3a: the launcher self-registers {pid, out_dir}; a dead launcher with no receipt in
    its out_dir ALARMS as CHAIN_DEAD_NO_RECEIPT (was: NO_RUN_DIR rc 0)."""
    out = tmp_path / "runs" / "levelset_x"
    out.mkdir(parents=True)
    man = tmp_path / "manifest.jsonl"
    man.write_text(json.dumps({"schema": "witness_chain_manifest.v1", "ts": "20260717T010000Z",
                               "launcher_pid": 2**22 + 4242, "out_dir": str(out),
                               "config": "c2_surgical_warm", "label": "t"}) + "\n")
    reg = _registry(tmp_path, [])
    v = W.scan(registry_path=reg, manifest_path=man)
    assert len(v) == 1 and v[0]["verdict"] == "CHAIN_DEAD_NO_RECEIPT"
    assert v[0]["source"] == "manifest"
    # with a receipt: RECEIPTED, no alarm
    (out / "dry_start_report.json").write_text('{"green": false}')
    v2 = W.scan(registry_path=reg, manifest_path=man)
    assert v2[0]["verdict"] == "CHAIN_DEAD_RECEIPTED"


def test_manifest_last_row_per_out_dir_wins(tmp_path):
    out = tmp_path / "runs" / "levelset_y"
    out.mkdir(parents=True)
    man = tmp_path / "manifest.jsonl"
    rows = [
        {"schema": "witness_chain_manifest.v1", "ts": "20260717T010000Z",
         "launcher_pid": 2**22 + 1, "out_dir": str(out), "label": "old"},
        {"schema": "witness_chain_manifest.v1", "ts": "20260717T020000Z",
         "launcher_pid": 2**22 + 2, "out_dir": str(out), "label": "new"},
    ]
    man.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = W._manifest_rows(man)
    assert len(got) == 1 and got[0]["label"] == "new"


def test_launcher_appends_chain_manifest(tmp_path):
    """The launcher side of F3a: _append_chain_manifest writes the row the watchdog reads."""
    import launch_witness_run as L
    man = tmp_path / "m.jsonl"
    L._append_chain_manifest(tmp_path / "outdir", "c2_surgical_warm", "lbl", manifest_path=man)
    row = json.loads(man.read_text().strip())
    assert row["launcher_pid"] == os.getpid() and row["config"] == "c2_surgical_warm"
    assert row["out_dir"].endswith("outdir")


def test_manifest_deleted_out_dir_reads_no_run_dir_not_alarm(tmp_path):
    """D1b: a self-registered out_dir that was DELETED (GC'd scratch proof) reads NO_RUN_DIR,
    never CHAIN_DEAD_NO_RECEIPT — a missing dir never held a real chain to die."""
    gone = tmp_path / "runs" / "deleted_scratch"  # never created
    man = tmp_path / "manifest.jsonl"
    man.write_text(json.dumps({"schema": "witness_chain_manifest.v1", "ts": "20260717T010000Z",
                               "launcher_pid": 2**22 + 9999, "out_dir": str(gone),
                               "label": "scratch_proof"}) + "\n")
    v = W.scan(registry_path=_registry(tmp_path, []), manifest_path=man)
    assert len(v) == 1 and v[0]["verdict"] == "NO_RUN_DIR"


def test_env_override_manifest_path_read(tmp_path, monkeypatch):
    """D1a: TAC_CHAIN_MANIFEST_PATH redirects the reader off the live manifest (proof
    isolation) even without an explicit manifest_path arg."""
    out = tmp_path / "runs" / "x"
    out.mkdir(parents=True)
    scratch = tmp_path / "scratch_manifest.jsonl"
    scratch.write_text(json.dumps({"schema": "witness_chain_manifest.v1",
                                   "ts": "20260717T010000Z", "launcher_pid": 2**22 + 5,
                                   "out_dir": str(out), "label": "p"}) + "\n")
    monkeypatch.setenv("TAC_CHAIN_MANIFEST_PATH", str(scratch))
    v = W.scan(registry_path=_registry(tmp_path, []))
    assert any(r.get("source") == "manifest" and r["run_dir"] == str(out) for r in v)


def test_env_override_manifest_path_write(tmp_path, monkeypatch):
    """D1a: an execution proof sets TAC_CHAIN_MANIFEST_PATH -> _append_chain_manifest writes
    the SCRATCH file, never the live one (even under PYTEST the env override wins)."""
    import launch_witness_run as L
    scratch = tmp_path / "scratch.jsonl"
    monkeypatch.setenv("TAC_CHAIN_MANIFEST_PATH", str(scratch))
    L._append_chain_manifest(tmp_path / "od", "c2_surgical_warm", "proof")
    assert scratch.exists()
    row = json.loads(scratch.read_text().strip())
    assert row["out_dir"].endswith("od")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
