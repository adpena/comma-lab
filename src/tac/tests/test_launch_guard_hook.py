"""Tests for the launch-guard PreToolUse hook (tools/launch_guard_hook.py)
and witness_checkin autodiscovery (tools/witness_checkin.py) — #338.

Covers the pure decide() surface (block raw trainer launches; allow governed
/ waived / merely-mentioning commands), the fail-open subprocess contract
(MANDATORY: malformed stdin must exit 0 with no block output), and the
checkin run-dir autodiscovery helpers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[3]
_HOOK = _REPO / "tools" / "launch_guard_hook.py"
_CHECKIN = _REPO / "tools" / "witness_checkin.py"

_ENV_CLEAN = {"TAC_LAUNCH_GUARD_OK": ""}  # explicit falsy env for pure calls


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _load(_HOOK, "launch_guard_hook")
checkin = _load(_CHECKIN, "witness_checkin")

_RAW_LAUNCH = (
    ".venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py "
    "--out-dir experiments/results/levelset_n600_witness_x --num-pairs 600"
)


def _decide(command: str, env: dict | None = None):
    return hook.decide(command, env={} if env is None else env)


# --- BLOCK cases: actual python execution of the trainer -----------------------------


def test_blocks_raw_venv_python_launch():
    allow, reason = _decide(_RAW_LAUNCH)
    assert allow is False
    assert "launch_witness_run" in reason  # points at the governed path


def test_blocks_absolute_python3_launch():
    allow, _ = _decide(
        "/usr/bin/python3 /Users/x/pact/experiments/train_levelset_witness_realized_through_R_mlx.py"
    )
    assert allow is False


def test_blocks_with_interpreter_options_and_env_prefix():
    allow, _ = _decide(
        "MLX_METAL_DEBUG=1 nohup python -u experiments/train_levelset_witness_realized_through_R_mlx.py"
    )
    assert allow is False


def test_blocks_bash_dash_c_wrapped_launch():
    allow, _ = _decide(f'bash -c "{_RAW_LAUNCH}"')
    assert allow is False


def test_blocks_second_segment_of_compound_command():
    allow, _ = _decide(f"cd /tmp && {_RAW_LAUNCH}")
    assert allow is False


def test_blocks_bash_dash_c_with_pipe_inside_payload():
    # Round-1 review catch: the coarse pipe-split breaks the quoted payload;
    # the whole-command pass must still catch this.
    allow, _ = _decide(f'bash -c "{_RAW_LAUNCH} | tee /tmp/run.log"')
    assert allow is False


# --- ALLOW cases: governed paths, waivers, and mere mentions --------------------------


def test_allows_safe_run_wrapped_launch():
    allow, _ = _decide(f"python tools/safe_run.py -- {_RAW_LAUNCH}")
    assert allow is True


def test_allows_governed_launcher():
    allow, _ = _decide(".venv/bin/python tools/launch_witness_run.py --clip 0")
    assert allow is True


def test_allows_skip_admission_gate_flag():
    allow, _ = _decide(f"{_RAW_LAUNCH} --skip-admission-gate")
    assert allow is True


def test_allows_inline_env_waiver_and_env_var():
    allow, _ = _decide(f"TAC_LAUNCH_GUARD_OK=1 {_RAW_LAUNCH}")
    assert allow is True
    allow, _ = _decide(_RAW_LAUNCH, env={"TAC_LAUNCH_GUARD_OK": "1"})
    assert allow is True


def test_allows_grep_cat_tail_of_trainer_file():
    for cmd in (
        "grep -n add_argument experiments/train_levelset_witness_realized_through_R_mlx.py",
        "cat experiments/train_levelset_witness_realized_through_R_mlx.py",
        "tail -50 experiments/train_levelset_witness_realized_through_R_mlx.py",
        "wc -l experiments/train_levelset_witness_realized_through_R_mlx.py",
    ):
        allow, _ = _decide(cmd)
        assert allow is True, cmd


def test_allows_python_dash_m_pytest_over_trainer():
    allow, _ = _decide(
        ".venv/bin/python -m pytest experiments/train_levelset_witness_realized_through_R_mlx.py -q"
    )
    assert allow is True


def test_allows_other_tool_mentioning_trainer_path_as_argument():
    allow, _ = _decide(
        "python tools/witness_memory_preflight.py --trainer "
        "experiments/train_levelset_witness_realized_through_R_mlx.py"
    )
    assert allow is True


def test_allows_empty_and_unrelated_commands():
    assert _decide("")[0] is True
    assert _decide("git status")[0] is True


# --- Subprocess contract (the hook binary itself) --------------------------------------


def _run_hook(stdin_text: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("TAC_LAUNCH_GUARD_OK", None)
    return subprocess.run(
        [sys.executable, str(_HOOK)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_hook_blocks_raw_launch_via_stdin_json():
    proc = _run_hook(json.dumps({"tool_name": "Bash", "tool_input": {"command": _RAW_LAUNCH}}))
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out["decision"] == "block"


def test_hook_allows_safe_run_via_stdin_json():
    proc = _run_hook(
        json.dumps({"tool_name": "Bash", "tool_input": {"command": f"python tools/safe_run.py -- {_RAW_LAUNCH}"}})
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_hook_fails_open_on_malformed_stdin():
    # MANDATORY fail-open proof: garbage stdin ⇒ exit 0, no block output.
    proc = _run_hook("this is { not json")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


# --- witness_checkin autodiscovery -----------------------------------------------------


def _mk_run(results: pathlib.Path, name: str, mtime: float) -> pathlib.Path:
    d = results / name
    d.mkdir(parents=True)
    best = d / "levelset_best.json"
    best.write_text(json.dumps({"d_seg": 0.005, "epoch": 1, "path": "x.npz", "ts": "t"}))
    os.utime(best, (mtime, mtime))
    return d


def test_newest_run_dir_picks_freshest_signal(tmp_path):
    old = _mk_run(tmp_path, "levelset_n600_witness_old", time.time() - 9000)
    new = _mk_run(tmp_path, "levelset_n600_witness_new", time.time() - 10)
    assert checkin.newest_run_dir(tmp_path) == new
    assert old.exists()


def test_pick_run_dir_prefers_live_trainer_out_dir(tmp_path):
    _mk_run(tmp_path, "levelset_n600_witness_newer", time.time() - 5)
    owned = _mk_run(tmp_path, "levelset_n600_witness_owned", time.time() - 5000)
    procs = [{"pid": 123, "out_dir": str(owned), "rss_bytes": 1 << 30}]
    run_dir, proc, how = checkin.pick_run_dir(procs, tmp_path)
    assert run_dir == owned
    assert proc["pid"] == 123
    assert "live-trainer" in how


def test_pick_run_dir_falls_back_to_newest_when_no_procs(tmp_path):
    new = _mk_run(tmp_path, "levelset_n600_witness_b", time.time() - 5)
    _mk_run(tmp_path, "levelset_n600_witness_a", time.time() - 5000)
    run_dir, proc, _ = checkin.pick_run_dir([], tmp_path)
    assert run_dir == new
    assert proc is None


def test_collect_status_flags_dead_and_stale(tmp_path):
    stale = _mk_run(tmp_path, "levelset_n600_witness_s", time.time() - 7200)
    dead = checkin.collect_status(stale, None, stale_after_s=1800.0)
    assert any("DEAD" in w for w in dead["warnings"])
    alive = checkin.collect_status(
        stale, {"pid": 1, "rss_bytes": 1 << 30}, stale_after_s=1800.0
    )
    assert any("STALE" in w for w in alive["warnings"])
    fresh_dir = _mk_run(tmp_path, "levelset_n600_witness_f", time.time() - 10)
    fresh = checkin.collect_status(
        fresh_dir, {"pid": 1, "rss_bytes": 1 << 30}, stale_after_s=1800.0
    )
    assert fresh["warnings"] == []
    assert "0.005" in checkin.human_line(fresh)
