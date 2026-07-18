"""Tests for the canonical witness run.log filtered-monitor emitter.

Guards the two bugs the hand-built filter had (pose ``fired:false`` noise;
whole-file replay) plus full failure/alarm coverage (silence-is-not-success).
Fixtures are REAL run.log line shapes sampled from the live c2 run 2026-07-17.
"""

from __future__ import annotations

from tac.witness_run_monitor import (
    build_exclude_regex,
    build_include_regex,
    build_monitor_command,
    classify_line,
)

# --- real benign lines (must classify None: the ~95% per-epoch noise) ---
_BENIGN = [
    '{"schema": "witness_component_wallclock.v1", "stage": "witness_component_wallclock", "epoch": 886, "errors": [], "complete": true}',
    '{"stage": "loss_terms", "epoch": 886, "total": 33.9}',
    '{"stage": "grad_clip_activation", "epoch": 886}',
    '{"stage": "jacobian_basin", "epoch": 886, "sigma_min": 0.0015}',
    '{"stage": "pose_finish_conditioning_gate", "epoch": 886, "fired": false, "classification": "DEGENERATE_GUARD_TRIPPED"}',
    '{"stage": "lever_would_fire", "epoch": 886}',
    '{"stage": "mod_dim_dynamics", "epoch": 886}',
    '{"stage": "annulus_convergence", "epoch": 886}',
    'some line mentioning ErrorCode=0 which is benign',
]

# --- real genuine lines (must classify to the right category) ---
_GENUINE = [
    ('{"stage": "verdict", "epoch": 875, "d_seg": 0.003975, "ep_loss": 33.97}', "verdict"),
    ('{"stage": "verdict_async_done", "epoch": 875, "secs": 2947.7}', "verdict"),
    ('{"stage": "checkpoint", "epoch": 875, "path": "ckpt_ep875.npz"}', "checkpoint"),
    ('{"stage": "lever_engage", "epoch": 800, "lever": "phase_advection"}', "lever_engage"),
    ('{"stage": "pose_finish_conditioning_gate", "epoch": 1002, "fired": true}', "pose_gate_fired"),
    ('{"stage": "tie_locus", "epoch": 700}', "transition"),
    ('{"stage": "warm_start_weights_only", "epoch": 651}', "transition"),
]

# --- failure / confound-alarm lines (must ALWAYS surface, never excluded) ---
_FAILURES = [
    ('Traceback (most recent call last):', "failure"),
    ('RuntimeError: MLX metal OOM', "failure"),
    ('safe_run status=oom peak_rss=90300MiB', "failure"),
    ('Killed', "failure"),
    ('mlx jetsam event', "failure"),
    ('{"stage": "confound_alarm", "kind": "spike_deadlock"}', "confound_alarm"),
    ('{"alarm": "frozen_epoch", "ep_loss": 0.0}', "confound_alarm"),
    ('{"alarm": "term_domination", "frac": 0.55}', "confound_alarm"),
]


def test_benign_lines_are_silent():
    for ln in _BENIGN:
        assert classify_line(ln) is None, f"benign line surfaced: {ln[:80]}"


def test_genuine_lines_classify_correctly():
    for ln, cat in _GENUINE:
        assert classify_line(ln) == cat, f"{ln[:80]} -> {classify_line(ln)} != {cat}"


def test_failures_and_alarms_always_surface():
    for ln, cat in _FAILURES:
        assert classify_line(ln) == cat, f"MISSED FAILURE: {ln[:80]} -> {classify_line(ln)}"


def test_errors_empty_field_does_not_trip_Error_pattern():
    # the wallclock row carries `"errors": []` every epoch; case-sensitive Error
    # must not match it, and the exclude is belt-and-suspenders.
    assert classify_line('{"stage": "witness_component_wallclock", "errors": []}') is None


def test_pose_gate_false_excluded_true_kept():
    assert classify_line('{"stage": "pose_finish_conditioning_gate", "fired": false}') is None
    assert classify_line('{"stage": "pose_finish_conditioning_gate", "fired": true}') == "pose_gate_fired"


def test_command_defaults_to_tail_no_replay():
    cmd = build_monitor_command("/runs/foo")
    assert "tail -n0 -F" in cmd            # starts at tail, no whole-file replay
    assert "/runs/foo/run.log" in cmd
    assert "grep -E" in cmd and "grep -vE" in cmd


def test_command_from_start_opt_in():
    assert "tail -n +1 -F" in build_monitor_command("/runs/foo", from_start=True)


def test_regexes_are_nonempty_and_cover_failures():
    inc = build_include_regex()
    assert "verdict" in inc and "Traceback" in inc and "spike_deadlock" in inc
    exc = build_exclude_regex()
    assert "fired" in exc and "errors" in exc


def test_no_benign_line_passes_the_full_shell_semantics():
    # emulate include AND NOT exclude on the benign set
    import re
    inc, exc = re.compile(build_include_regex()), re.compile(build_exclude_regex())
    for ln in _BENIGN:
        passes = bool(inc.search(ln)) and not bool(exc.search(ln))
        assert not passes, f"benign line would pass the shell filter: {ln[:80]}"
