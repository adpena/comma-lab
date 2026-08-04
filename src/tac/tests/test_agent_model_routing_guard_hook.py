# SPDX-License-Identifier: MIT
"""Tests for the Agent-model routing guard PreToolUse hook.

The guard exists because on 2026-08-04 every subagent silently inherited the
parent session model (Fable-5) and burned weeks of rate limit in a day. The
decision surface is pure; these tests pin the exact allow/deny boundary plus
the fail-open contract.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_HOOK = _REPO / "tools" / "agent_model_routing_guard_hook.py"


def _load():
    spec = importlib.util.spec_from_file_location("_agent_model_routing_guard_hook", _HOOK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    return _load()


# --- the bug this guard extincts -------------------------------------------------


def test_omitted_model_is_blocked(hook):
    blocked, reason = hook.decide({"prompt": "do a thing"}, {})
    assert blocked is True
    assert "model" in reason


def test_blocked_reason_names_the_incident_and_the_fix(hook):
    _, reason = hook.decide({"prompt": "x"}, {})
    assert "2026-08-04" in reason
    assert 'model="opus"' in reason


def test_none_model_is_blocked(hook):
    blocked, _ = hook.decide({"model": None, "prompt": "x"}, {})
    assert blocked is True


def test_empty_string_model_is_blocked(hook):
    blocked, _ = hook.decide({"model": "", "prompt": "x"}, {})
    assert blocked is True


def test_unknown_model_name_is_blocked(hook):
    """A typo must not silently pass — it would fall back to inheritance."""
    blocked, _ = hook.decide({"model": "opuss", "prompt": "x"}, {})
    assert blocked is True


def test_fable_is_blocked_even_when_explicit(hook):
    """`fable` is not in ALLOWED_MODELS: naming the expensive default still refuses."""
    blocked, _ = hook.decide({"model": "fable", "prompt": "x"}, {})
    assert blocked is True


# --- the allow paths -------------------------------------------------------------


@pytest.mark.parametrize("model", ["opus", "sonnet", "haiku"])
def test_allowed_models_pass(hook, model):
    blocked, reason = hook.decide({"model": model, "prompt": "x"}, {})
    assert blocked is False
    assert reason == ""


def test_allowed_model_is_case_and_space_insensitive(hook):
    blocked, _ = hook.decide({"model": "  Opus "}, {})
    assert blocked is False


def test_fork_subagent_type_is_exempt(hook):
    """Forks inherit by tool contract — `model` is ignored, so refusing is unactionable."""
    blocked, _ = hook.decide({"subagent_type": "fork", "prompt": "x"}, {})
    assert blocked is False


def test_fork_exemption_is_case_insensitive(hook):
    blocked, _ = hook.decide({"subagent_type": "FORK"}, {})
    assert blocked is False


def test_non_fork_subagent_type_still_needs_a_model(hook):
    blocked, _ = hook.decide({"subagent_type": "general-purpose"}, {})
    assert blocked is True


def test_escape_hatch_env_allows(hook):
    blocked, _ = hook.decide({"prompt": "x"}, {"TAC_AGENT_MODEL_GUARD_OK": "1"})
    assert blocked is False


def test_escape_hatch_requires_exactly_one(hook):
    """A truthy-looking value that is not "1" must not open the hatch."""
    blocked, _ = hook.decide({"prompt": "x"}, {"TAC_AGENT_MODEL_GUARD_OK": "true"})
    assert blocked is True


# --- end-to-end process contract -------------------------------------------------


def _run(payload: object, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.pop("TAC_AGENT_MODEL_GUARD_OK", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(payload) if not isinstance(payload, str) else payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_process_denies_agent_spawn_without_model():
    proc = _run({"tool_name": "Agent", "tool_input": {"prompt": "x"}})
    assert proc.returncode == 0, proc.stderr
    emitted = json.loads(proc.stdout)
    assert emitted["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert emitted["decision"] == "block"
    # legacy + current shapes must carry the same reason
    assert emitted["reason"] == emitted["hookSpecificOutput"]["permissionDecisionReason"]


def test_process_allows_agent_spawn_with_model():
    proc = _run({"tool_name": "Agent", "tool_input": {"model": "opus", "prompt": "x"}})
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_ignores_other_tools():
    proc = _run({"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_garbage_stdin():
    proc = _run("this is not json")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_empty_stdin():
    proc = _run("")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_non_dict_tool_input():
    proc = _run({"tool_name": "Agent", "tool_input": ["not", "a", "dict"]})
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_honours_escape_hatch():
    proc = _run(
        {"tool_name": "Agent", "tool_input": {"prompt": "x"}},
        env_extra={"TAC_AGENT_MODEL_GUARD_OK": "1"},
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


# --- wiring ---------------------------------------------------------------------


def test_hook_is_wired_into_claude_settings():
    settings = json.loads((_REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))
    matchers = settings["hooks"]["PreToolUse"]
    agent_entries = [m for m in matchers if m.get("matcher") == "Agent"]
    assert agent_entries, "PreToolUse must carry an Agent matcher"
    commands = [h["command"] for entry in agent_entries for h in entry["hooks"]]
    assert any("agent_model_routing_guard_hook.py" in c for c in commands)
