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


def test_alive_pid_quiet_vs_healthy(tmp_path):
    d = _run_dir(tmp_path)
    (d / "fresh_file").write_text("x")  # newest mtime = now -> HEALTHY
    reg = _registry(tmp_path, [_row("witness_live", os.getpid(), run_dir=d)])
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
