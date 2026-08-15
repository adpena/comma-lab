# SPDX-License-Identifier: MIT
"""Tests for the Agent-model routing guard PreToolUse hook.

Live law (operator 2026-08-04): CODEX ARMS ONLY — no Claude subagent may be
spawned. ``ALLOWED_MODELS`` is empty, so every Agent spawn refuses regardless of
``model`` or ``subagent_type``. These tests pin that, plus the fail-open contract
(a PreToolUse hook must never brick a session) and the allow-set mechanism that a
future directive would use to re-open Claude arms.
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


# --- the live law: every Claude subagent spawn refuses ---------------------------


def test_allow_set_is_exactly_the_two_operator_grants(hook):
    """08-04 fable (convocation-class) + 08-15 opus (close-supervision bridge)."""
    assert hook.ALLOWED_MODELS == frozenset({"fable", "opus"})


def test_omitted_model_is_blocked(hook):
    blocked, reason = hook.decide({"prompt": "do a thing"}, {})
    assert blocked is True
    assert reason


@pytest.mark.parametrize("model", ["sonnet", "haiku", "opuss", ""])
def test_ungranted_named_models_are_blocked(hook, model):
    """Only the two explicit operator grants pass; near-misses (opuss) refuse."""
    blocked, _ = hook.decide({"model": model, "prompt": "x"}, {})
    assert blocked is True


@pytest.mark.parametrize("model", ["opus", "  Opus ", "fable"])
def test_granted_named_models_pass(hook, model):
    """Explicitly NAMING a granted model is the deliberate act the guard demands."""
    blocked, _ = hook.decide({"model": model, "prompt": "x"}, {})
    assert blocked is False


def test_none_model_is_blocked(hook):
    blocked, _ = hook.decide({"model": None, "prompt": "x"}, {})
    assert blocked is True


def test_fork_subagent_type_is_blocked(hook):
    """A fork inherits by tool contract but is still a Claude subagent on our quota."""
    blocked, _ = hook.decide({"subagent_type": "fork", "prompt": "x"}, {})
    assert blocked is True


def test_general_purpose_subagent_type_is_blocked(hook):
    blocked, _ = hook.decide({"subagent_type": "general-purpose"}, {})
    assert blocked is True


def test_blocked_reason_names_the_directive_and_the_alternative(hook):
    _, reason = hook.decide({"prompt": "x"}, {})
    assert "2026-08-04" in reason
    assert "2026-08-15" in reason  # the opus close-supervision grant is named
    assert "codex_arm_queue" in reason
    assert "EXPLICIT allowed model" in reason


def test_blocked_reason_has_no_unformatted_placeholders(hook):
    _, reason = hook.decide({"prompt": "x"}, {})
    assert "{" not in reason and "}" not in reason


# --- the escape hatch ------------------------------------------------------------


def test_escape_hatch_env_allows(hook):
    blocked, _ = hook.decide({"prompt": "x"}, {"TAC_AGENT_MODEL_GUARD_OK": "1"})
    assert blocked is False


def test_escape_hatch_requires_exactly_one(hook):
    blocked, _ = hook.decide({"prompt": "x"}, {"TAC_AGENT_MODEL_GUARD_OK": "true"})
    assert blocked is True


# --- the re-open mechanism (what a future directive would flip) ------------------


def test_nonempty_allow_set_would_permit_a_named_model(hook, monkeypatch):
    """If Claude arms re-open, the guard reverts to enforcing EXPLICIT routing."""
    monkeypatch.setattr(hook, "ALLOWED_MODELS", frozenset({"sonnet"}))
    assert hook.decide({"model": "sonnet"}, {})[0] is False
    assert hook.decide({"model": "  Sonnet "}, {})[0] is False  # case/space tolerant
    assert hook.decide({"prompt": "x"}, {})[0] is True  # omission still refuses
    assert hook.decide({"model": "opus"}, {})[0] is True  # not in the patched set


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
    assert emitted["reason"] == emitted["hookSpecificOutput"]["permissionDecisionReason"]


def test_process_allows_agent_spawn_with_model_opus():
    """The 08-15 grant: an explicit opus spawn (supervised bridge) passes silently."""
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
    """Regression: empty stdin once fell through to a DENY on nothing."""
    proc = _run("")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_empty_tool_input():
    proc = _run({"tool_name": "Agent", "tool_input": {}})
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_non_dict_tool_input():
    proc = _run({"tool_name": "Agent", "tool_input": ["not", "a", "dict"]})
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_process_fails_open_on_non_dict_payload():
    proc = _run(["not", "a", "dict"])
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
