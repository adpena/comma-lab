"""Adversarial tests for the model-identity HARD-FAIL gate (operator 2026-07-21).

Attacks: sidechain spoof · synthetic-entry spoof · stale-early-turn masking · fresh
session · unreadable transcript · corrupt lines · huge-file tail seek · pin override ·
kill-switches · empty requirement · substring semantics (fable vs claude-fable-5) ·
case tricks. The gate must hard-fail (rc 2) ONLY on a genuine main-thread mismatch and
must never brick a session on unknowns.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GATE = REPO / "tools" / "model_identity_gate.py"

spec = importlib.util.spec_from_file_location("model_identity_gate", GATE)
mig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mig)


def _row(model, sidechain=False, typ="assistant"):
    return json.dumps({"type": typ, "isSidechain": sidechain, "message": {"model": model}})


def _transcript(tmp_path, rows):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def _run(transcript_path, env_extra=None, home=None):
    """Run the gate as the hook would: stdin JSON, capture rc/stdout/stderr."""
    import os
    env = dict(os.environ)
    env.pop("TAC_MODEL_GATE_ALLOW", None)
    if env_extra:
        env.update(env_extra)
    if home:
        env["HOME"] = str(home)
    payload = json.dumps({"transcript_path": transcript_path, "hook_event_name": "UserPromptSubmit"})
    return subprocess.run(
        [sys.executable, str(GATE)], input=payload, capture_output=True, text=True, env=env
    )


def _home_with_default(tmp_path, model):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps({"model": model}))
    return home


# --- core verdicts -------------------------------------------------------------------

def test_match_fable_substring_passes(tmp_path):
    t = _transcript(tmp_path, [_row("claude-fable-5")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 0 and "HARD FAIL" not in r.stderr


def test_mismatch_opus_hard_fails_rc2(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2
    assert "MODEL IDENTITY HARD FAIL" in r.stderr
    assert "claude-opus-4-8" in r.stderr


def test_most_recent_turn_wins_not_stale_early(tmp_path):
    # early fable turns must NOT mask a live opus reroute
    t = _transcript(tmp_path, [_row("claude-fable-5")] * 5 + [_row("claude-opus-4-8")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2


def test_recovery_after_restart_passes(tmp_path):
    # inverse: old opus turns must not block once fable is live again
    t = _transcript(tmp_path, [_row("claude-opus-4-8")] * 5 + [_row("claude-fable-5")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 0


# --- spoof resistance ----------------------------------------------------------------

def test_sidechain_fable_cannot_mask_mainthread_opus(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8"), _row("claude-fable-5", sidechain=True)])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2  # subagent turns are not the main thread


def test_synthetic_entry_ignored(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8"), _row("<synthetic>")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2


def test_non_assistant_rows_ignored(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8"), _row("claude-fable-5", typ="user")])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2


# --- never-brick unknowns ------------------------------------------------------------

def test_fresh_session_no_assistant_turns_warns_not_blocks(tmp_path):
    t = _transcript(tmp_path, [json.dumps({"type": "user"})])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 0 and "WARN" in r.stdout


def test_missing_transcript_warns_not_blocks(tmp_path):
    r = _run(str(tmp_path / "nope.jsonl"), home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 0 and "WARN" in r.stdout


def test_corrupt_lines_skipped(tmp_path):
    t = _transcript(tmp_path, ["{not json", _row("claude-opus-4-8"), "}{"])
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2  # corruption must not hide a real mismatch


def test_empty_requirement_warns_not_blocks(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8")])
    r = _run(t, home=_home_with_default(tmp_path, ""))
    assert r.returncode == 0 and "not enforced" in r.stdout


def test_malformed_stdin_does_not_crash(tmp_path):
    import os
    env = dict(os.environ); env.pop("TAC_MODEL_GATE_ALLOW", None)
    env["HOME"] = str(_home_with_default(tmp_path, "fable"))
    r = subprocess.run([sys.executable, str(GATE)], input="not json", capture_output=True, text=True, env=env)
    assert r.returncode == 0  # no transcript -> WARN path, never a crash-block


# --- kill-switch + tail-seek ---------------------------------------------------------

def test_env_kill_switch_bypasses_loudly(tmp_path):
    t = _transcript(tmp_path, [_row("claude-opus-4-8")])
    r = _run(t, env_extra={"TAC_MODEL_GATE_ALLOW": "1"}, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 0 and "BYPASS ACTIVE" in r.stdout


def test_tail_seek_on_large_transcript_sees_latest(tmp_path):
    filler = [_row("claude-fable-5")] * 10000  # push early rows past the 512KB tail window
    t = _transcript(tmp_path, filler + [_row("claude-opus-4-8")])
    assert Path(t).stat().st_size > 512 * 1024
    r = _run(t, home=_home_with_default(tmp_path, "fable"))
    assert r.returncode == 2


def test_full_id_requirement_also_matches(tmp_path):
    t = _transcript(tmp_path, [_row("claude-fable-5")])
    r = _run(t, home=_home_with_default(tmp_path, "claude-fable-5"))
    assert r.returncode == 0
