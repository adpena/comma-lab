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


# --- F6 loud-escalation of the fail-open error log (round-2 review) ---------------------
# >= 3 errors within 24h ⇒ ONE stderr warning line; the hook STILL allows (fail-open).


def _patch_error_log(tmp_path, monkeypatch):
    log = tmp_path / "launch_guard_hook_errors.log"
    monkeypatch.setattr(hook, "_ERROR_LOG", log)
    return log


def test_escalation_does_not_fire_below_threshold(tmp_path, monkeypatch, capsys):
    _patch_error_log(tmp_path, monkeypatch)
    hook._log_error(ValueError("one"))
    hook._log_error(ValueError("two"))
    assert "WARNING" not in capsys.readouterr().err  # 2 < 3: silent append only


def test_escalation_fires_at_three_within_window_and_still_allows(tmp_path, monkeypatch, capsys):
    log = _patch_error_log(tmp_path, monkeypatch)
    for i in range(3):
        hook._log_error(ValueError(f"e{i}"))
    err = capsys.readouterr().err
    assert "WARNING" in err and str(log) in err
    # fail-open preserved: escalation is observability only — decide() still allows
    assert _decide("git status")[0] is True
    assert len(log.read_text().splitlines()) == 3  # all three errors durably appended


def test_escalation_ignores_stale_and_legacy_untimestamped_lines(tmp_path, monkeypatch, capsys):
    log = _patch_error_log(tmp_path, monkeypatch)
    log.parent.mkdir(parents=True, exist_ok=True)
    stale = time.time() - 25 * 3600
    log.write_text(f"{stale:.3f}\tValueError: old-beyond-24h\n"
                   "ValueError: legacy-line-without-timestamp\n")
    hook._log_error(ValueError("x"))
    hook._log_error(ValueError("y"))
    # 2 recent + 1 stale + 1 legacy => below threshold => no warning
    assert "WARNING" not in capsys.readouterr().err
    hook._log_error(ValueError("z"))  # third RECENT error => fires
    assert "WARNING" in capsys.readouterr().err


def test_recent_error_count_unreadable_log_returns_zero(tmp_path, monkeypatch):
    _patch_error_log(tmp_path, monkeypatch)  # file never created
    assert hook._recent_error_count(time.time()) == 0


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


# --- codex-arm spawn detachment (2026-08-04) ---------------------------------------
# Four arms spawned with `nohup ... & disown` were reaped together by one
# process-group signal. disown clears the shell's JOB TABLE; only setsid(2) leaves
# the GROUP. These pin the block so the killer form cannot come back.


def _guard():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools" / "launch_guard_hook.py"
    spec = importlib.util.spec_from_file_location("_lgh_codex", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocks_nohup_disown_codex_spawn_the_measured_killer():
    g = _guard()
    cmd = "nohup bash -c 'codex exec -m gpt-5.5 -o a.last.txt \"go\"' < /dev/null & disown"
    allow, reason = g.decide(cmd, {})
    assert allow is False
    assert "setsid" in reason  # the message must name the actual mechanism


def test_blocks_bare_foreground_codex_exec():
    assert _guard().decide("codex exec --skip-git-repo-check 'go'", {})[0] is False


def test_allows_inline_fork_setsid():
    g = _guard()
    cmd = "python3 -c 'import os; os.setsid()' codex exec -m gpt-5.5 'go'"
    assert g.decide(cmd, {})[0] is True


def test_allows_the_canonical_spawners():
    g = _guard()
    for cmd in (
        ".venv/bin/python tools/codex_arm_queue.py saturate --spawn",
        "tools/codex_companion_spawn.sh spawn lbl xhigh prompt.md",
    ):
        assert g.decide(cmd, {})[0] is True, cmd


def test_allows_read_only_process_inspection():
    """`ps`/`pgrep`/`grep` may name `codex exec` — they inspect, they don't spawn."""
    g = _guard()
    for cmd in ("ps -eo pid,command | grep 'codex exec'", "pgrep -af 'codex exec'"):
        assert g.decide(cmd, {})[0] is True, cmd


def test_codex_spawn_block_honours_the_escape_hatch():
    assert _guard().decide("codex exec 'go'", {"TAC_LAUNCH_GUARD_OK": "1"})[0] is True


def test_inline_escape_hatch_overrides_the_codex_spawn_block():
    """The block message advertises `set TAC_LAUNCH_GUARD_OK=1`; an inline env
    assignment reaches decide() only via the command string. Round-1 ordering
    checked codex-spawn BEFORE safe tokens, making the advertised hatch a
    no-op — caught by dogfooding (a python heredoc merely containing the
    words was blocked twice)."""
    allow, _ = hook.decide("TAC_LAUNCH_GUARD_OK=1 python3 edit_text_mentioning_codex_exec.py", {})
    assert allow is True


def test_heredoc_text_manipulation_mentioning_codex_exec_still_blocked_without_hatch():
    """Conservative default stands: without the hatch, a non-readonly command
    containing `codex exec` is still refused (false positives are the price;
    the hatch is the documented exception)."""
    allow, reason = hook.decide("python3 - <<'EOF'\nprint('codex exec')\nEOF", {})
    assert allow is False
    assert "KEEPER" in reason


def test_trainer_safe_tokens_do_not_bypass_the_codex_spawn_block():
    """Round-3 review finding (executed control 2026-08-04): after the hatch
    ordering fix, ALL of _SAFE_TOKENS preceded the codex check — so a
    hand-rolled spawn whose charter text merely MENTIONED a trainer tool
    (`nohup codex exec "review tools/launch_witness_run.py" & disown`, the
    exact killed-arm shape) sailed through. Only TAC_LAUNCH_GUARD_OK may
    override the codex block; trainer tokens scope the trainer gate alone."""
    for tok in ("safe_run", "launch_witness_run", "--skip-admission-gate"):
        cmd = f'nohup codex exec "review {tok} for bugs" & disown'
        allow, reason = hook.decide(cmd, {})
        assert allow is False, f"trainer token {tok!r} bypassed the codex block"
        assert "KEEPER" in reason


def test_trainer_safe_tokens_still_allow_governed_trainer_launches():
    """The re-ordering must not break the tokens' real job: a governed
    trainer invocation (launch_witness_run) still passes without a hatch."""
    allow, _ = hook.decide(".venv/bin/python tools/launch_witness_run.py --dry-run", {})
    assert allow is True


# --- #1121 waiter discipline: the STRUCTURAL half of the two-landing cure ------
#
# Landing 1 (5d4b1818f5) put WAITER_DISCIPLINE into every composed subagent
# prompt -- volitional. These pin the structural refusal. The FIRST four are the
# load-bearing ones: the canonical artifact-bound waiter and the live fleet
# watcher chain must keep working, or this guard breaks every arm on the box.


def test_waiter_guard_allows_the_canonical_artifact_bound_wait():
    """`until [ -f "$DONE" ]` is exactly what arms SHOULD write. Never refuse it."""
    for cmd in (
        'until [ -f "$DONE" ]; do sleep 30; done; cat "$DONE"',
        "while [ ! -f .omx/tmp/codex_runs/x.done ]; do sleep 30; done",
        'until test -f run/receipt.json; do sleep 10; done',
    ):
        allow, reason = hook.decide(cmd, _ENV_CLEAN, run_in_background=True)
        assert allow is True, f"refused the CORRECT pattern: {cmd!r} -> {reason}"


def test_waiter_guard_allows_the_canonical_launcher_and_live_watcher_chain():
    """The fleet-watcher chain is live on this box; refusing it severs notifications."""
    for cmd in (
        ".venv/bin/python tools/launch_detached_process.py --output-dir r "
        "--done-receipt n -- python x.py",
        'bash -c "exec .venv/bin/python tools/codex_arm_watch.py"',
        ".venv/bin/python tools/codex_arm_queue.py saturate --spawn",
    ):
        allow, reason = hook.decide(cmd, _ENV_CLEAN, run_in_background=True)
        assert allow is True, f"refused a canonical surface: {cmd!r} -> {reason}"


def test_waiter_guard_does_not_fire_on_commands_merely_describing_waiters():
    """grep/echo ABOUT a waiter is inspection, not a spawn (the _READONLY_HEADS rule)."""
    for cmd in (
        'grep -rn "while true; do sleep 60; done" scripts/',
        'echo "while true; do sleep 5; done"',
    ):
        allow, _ = hook.decide(cmd, _ENV_CLEAN, run_in_background=False)
        assert allow is True, f"false positive on inspection: {cmd!r}"


def test_waiter_guard_ignores_foreground_and_non_sleep_loops():
    allow, _ = hook.decide(
        'while read l; do echo "$l"; done < f.txt', _ENV_CLEAN, run_in_background=True
    )
    assert allow is True
    # A clock-bound wait in the FOREGROUND is the harness's problem, not this
    # guard's: it cannot orphan, because it dies with the shell.
    allow, _ = hook.decide("sleep 600", _ENV_CLEAN, run_in_background=False)
    assert allow is True


def test_waiter_guard_blocks_process_table_bound_background_waits():
    """A pgrep/kill -0 wait can fire when nothing happened -- it is not an instrument."""
    for cmd in (
        "while pgrep -f ddm_jg1 >/dev/null; do sleep 30; done",
        "while kill -0 12345 2>/dev/null; do sleep 5; done",
    ):
        allow, reason = hook.decide(cmd, _ENV_CLEAN, run_in_background=True)
        assert allow is False, f"orphan-prone waiter allowed: {cmd!r}"
        assert "artifact" in reason.lower()


def test_waiter_guard_blocks_bare_clock_waits_in_background():
    """The measured noise half: each expiry re-invokes MAIN with zero information."""
    for cmd in ("sleep 600", "sleep 300; echo done"):
        allow, reason = hook.decide(cmd, _ENV_CLEAN, run_in_background=True)
        assert allow is False, f"bare clock waiter allowed: {cmd!r}"
        assert "#1121" in reason


def test_waiter_guard_blocks_the_measured_latent_actuator_verbatim():
    """The 2026-08-18 ddm_iv1 shape, which launched a duplicate ~30 min late.

    Blocked in the FOREGROUND too: the harm is the duplicate launch over an
    adjudicated receipt, not the notification.
    """
    cmd = (
        "until ! pgrep -f route1_search.py; do sleep 60; done; "
        ".venv/bin/python route2.py"
    )
    for bg in (False, True):
        allow, reason = hook.decide(cmd, _ENV_CLEAN, run_in_background=bg)
        assert allow is False, "the measured latent actuator was allowed"
        assert "latent actuator" in reason.lower()
        assert "re-decide at fire time" in reason.lower()


def test_waiter_guard_honours_the_explicit_override():
    cmd = "while pgrep -f x; do sleep 30; done"
    allow, _ = hook.decide(cmd, {"TAC_LAUNCH_GUARD_OK": "1"}, run_in_background=True)
    assert allow is True


def test_trainer_safe_tokens_do_not_bypass_the_waiter_block():
    """Same lesson as the codex-spawn ordering bug: a trainer token must not buy
    a bypass of an orphan-class violation."""
    for tok in ("safe_run", "launch_witness_run", "--skip-admission-gate"):
        cmd = f"while pgrep -f {tok}; do sleep 30; done"
        allow, _ = hook.decide(cmd, _ENV_CLEAN, run_in_background=True)
        assert allow is False, f"trainer token {tok!r} bypassed the waiter block"


def test_waiter_guard_does_not_fire_on_a_sleep_outside_the_loop_body():
    """Review-pass-2 regression: the `sleep` must be INSIDE the loop.

    `git log | while read c; do echo $c; done && sleep 5` is an ordinary command
    with a trailing sleep. An earlier `while.*do.*sleep` pattern (no trailing
    `done`) matched it. On a PreToolUse Bash hook a false positive costs every
    arm, so the allow side is the load-bearing side.
    """
    allow, _ = hook.decide(
        "git log --oneline | while read c; do echo $c; done && sleep 5",
        _ENV_CLEAN,
        run_in_background=True,
    )
    assert allow is True


def test_waiter_guard_tail_split_is_word_boundary_aware():
    """`.done` / `--done-receipt` / 'abandoned' must not be split as the `done` keyword."""
    allow, _ = hook.decide(
        "until [ -f x.done ]; do sleep 30; done", _ENV_CLEAN, run_in_background=True
    )
    assert allow is True
