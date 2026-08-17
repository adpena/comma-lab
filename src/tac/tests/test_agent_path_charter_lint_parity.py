# SPDX-License-Identifier: MIT
"""Charter-lint parity between the Agent spawn path and the codex spawn path.

WHY THIS FILE EXISTS (task #1082, ddm_cl3, 2026-08-17). Registry row
``charter_lint_is_spawn_path_conditional_20260817`` measured the defect: all
three charter-lint legs lived ONLY in ``tools/codex_arm_queue.py``, the CODEX
keeper path. Codex was WALLED from 2026-08-15 (#1079), so every arm spawned
through the Agent tool instead — past a hook that guarded model routing and
nothing else. Every charter written during the wall was UNLINTED.

The code fix closes the coverage gap. THIS FILE is what keeps the two paths from
drifting apart again, which is the actual deliverable: a fix without it just
moves the gap. Every assertion here compares the SAME charter text through BOTH
paths and demands the SAME verdict.

Two clauses inherited from the genus (a correct cure wired to a population that
stopped being live) are pinned as tests, not prose:
  1. ALLOW-LISTS FAIL OPEN — a spawn the hook cannot lint must announce itself,
     never pass silently as clean.
  2. A SEARCH THAT FINDS NOTHING MUST PROVE IT LOOKED — so this suite carries a
     POSITIVE CONTROL proving the lint can FAIL through the Agent path. A suite
     that only ever sees "no findings" cannot tell "clean" from "never ran".
     That is the exact bar two prior detectors failed.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_HOOK = _REPO / "tools" / "agent_model_routing_guard_hook.py"
_QUEUE = _REPO / "tools" / "codex_arm_queue.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def hook():
    return _load(_HOOK, "_agent_model_routing_guard_hook_parity")


@pytest.fixture(scope="module")
def queue():
    return _load(_QUEUE, "_codex_arm_queue_parity")


# Charter texts exercised through BOTH paths. Chosen so the set is NOT vacuous:
# `build_no_optimal_form` must trip the gating leg and `falsified_premise` must
# trip an advisory leg, both asserted below.
CHARTERS: dict[str, str] = {
    "build_no_optimal_form": "Build and measure a new codec arm.",
    "falsified_premise": "Proceed because charter recall is apparatus not volition.",
    "waived": "OPTIMAL_FORM_NA: pure analysis pass, no mechanism is built here.",
    "audit_only": "Read the ledger and report what is already owned.",
    "placeholder_waiver": "OPTIMAL_FORM_NA: <rationale>",
}


def _codex_optimal_form(queue, text: str) -> list[str]:
    """Run the codex path's gating leg on the same text the Agent path sees."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as fh:
        path = Path(fh.name)
        fh.write(text)
    try:
        return list(queue.lint_charter_optimal_form(str(path)))
    finally:
        path.unlink(missing_ok=True)


def _agent_findings(hook, text: str, env: dict | None = None):
    blocked, messages, status = hook.charter_lint({"prompt": text}, env or {})
    return blocked, messages, status


# --- PARITY: the two paths must agree on the verdict ----------------------------


@pytest.mark.parametrize("name", sorted(CHARTERS))
def test_gating_leg_verdict_is_identical_on_both_paths(hook, queue, name):
    """The Agent path's REFUSED/WARN lines must carry exactly the codex problems.

    Not similar — the same list, in the same order. Any divergence means one path
    started judging charters differently from the other, which is the drift this
    file exists to catch.
    """
    text = CHARTERS[name]
    codex_problems = _codex_optimal_form(queue, text)
    _, messages, status = _agent_findings(hook, text)
    assert status == "ran"
    agent_problems = [
        m.split(": ", 1)[1]
        for m in messages
        if m.startswith("charter-lint REFUSED: ") or m.startswith("charter-lint WARN: ")
    ]
    for problem in codex_problems:
        assert problem in agent_problems, (
            f"{name}: codex reports {problem!r}; the Agent path does not. "
            "The spawn paths have drifted."
        )


@pytest.mark.parametrize("name", sorted(CHARTERS))
def test_advisory_legs_verdict_is_identical_on_both_paths(hook, queue, name):
    """Both shared recall legs must produce the same advisories on both paths."""
    text = CHARTERS[name]
    codex_advisories = list(queue._lint_stale_numbers(text)) + list(
        queue._lint_falsified_premises(text)
    )
    _, messages, _ = _agent_findings(hook, text)
    agent_lines = [m.split(": ", 1)[1] for m in messages if m.startswith("charter-lint WARN: ")]
    for advisory in codex_advisories:
        assert advisory in agent_lines, (
            f"{name}: codex advisory {advisory[:80]!r} is missing from the Agent path."
        )


# --- POSITIVE CONTROLS: prove the instrument can fail ---------------------------


def test_positive_control_gating_leg_actually_fires(hook, queue):
    """Clause 2. Without this, every parity test above could pass on emptiness."""
    text = CHARTERS["build_no_optimal_form"]
    assert _codex_optimal_form(queue, text), "codex fixture produces no problem — vacuous suite"
    _, messages, status = _agent_findings(hook, text)
    assert status == "ran"
    assert any("OPTIMAL FORM" in m for m in messages), "Agent path found nothing — vacuous suite"


def test_positive_control_advisory_leg_actually_fires(hook, queue):
    """Clause 2 for the recall legs: a live registry row must be reachable."""
    text = CHARTERS["falsified_premise"]
    assert queue._lint_falsified_premises(text), "no live registry hit — advisory suite is vacuous"
    _, messages, _ = _agent_findings(hook, text)
    assert any("FALSIFIED premise" in m for m in messages)


def test_positive_control_lint_can_block_the_agent_path(hook):
    """The lint must be able to REFUSE a spawn, not merely narrate one."""
    blocked, messages, status = _agent_findings(
        hook, CHARTERS["build_no_optimal_form"], {"TAC_CHARTER_LINT_STRICT": "1"}
    )
    assert status == "ran"
    assert blocked is True
    assert any(m.startswith("charter-lint REFUSED: ") for m in messages)


def test_positive_control_block_reaches_the_process_boundary():
    """End-to-end: a real deny payload, not just a True in a tuple."""
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps(
            {
                "tool_name": "Agent",
                "tool_input": {"model": "opus", "prompt": CHARTERS["build_no_optimal_form"]},
            }
        ),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "TAC_CHARTER_LINT_STRICT": "1"},
    )
    assert proc.returncode == 0, "a PreToolUse hook must never exit non-zero"
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "charter lint" in payload["reason"]


# --- CONTRACT MIRROR: advisories never gate -------------------------------------


def test_advisories_never_escalate_to_a_block_even_under_strict(hook):
    """``codex_arm_queue`` keeps advisories warn-only forever; so must this path."""
    blocked, messages, status = _agent_findings(
        hook, CHARTERS["falsified_premise"], {"TAC_CHARTER_LINT_STRICT": "1"}
    )
    assert status == "ran"
    assert blocked is False, "an advisory must never refuse a spawn"
    assert any("FALSIFIED premise" in m for m in messages)


def test_strict_is_opt_in_on_the_agent_path(hook):
    """Warn by default per the #1082 charter (see memo for the divergence note)."""
    blocked, messages, _ = _agent_findings(hook, CHARTERS["build_no_optimal_form"], {})
    assert blocked is False
    assert any(m.startswith("charter-lint WARN: ") for m in messages)


def test_recall_lint_na_opt_out_matches_codex(hook, queue):
    """The Agent path must not be noisier than codex on identical text.

    Compares both paths on the SAME text, and first proves the fixture is
    non-vacuous by asserting the advisory DOES fire without the opt-out.
    """
    base = CHARTERS["falsified_premise"]
    text = base + "\nrecall_lint_na: sanctioned restatement of a known-falsified premise"

    # non-vacuity: without the opt-out both paths speak.
    assert queue._lint_falsified_premises(base)
    assert any("FALSIFIED premise" in m for m in _agent_findings(hook, base)[1])

    # with the opt-out both paths fall silent.
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as fh:
        path = Path(fh.name)
        fh.write(text)
    try:
        codex_advisories = queue.lint_charter_recall_advisories(str(path))
    finally:
        path.unlink(missing_ok=True)
    _, messages, status = _agent_findings(hook, text)
    assert status == "ran"
    assert codex_advisories == [], "codex honours recall_lint_na"
    assert not any("FALSIFIED premise" in m for m in messages), "so must the Agent path"


# --- CLAUSE 1: never report "clean" for input never examined ---------------------


@pytest.mark.parametrize(
    "tool_input",
    [
        {"model": "opus"},
        {"model": "opus", "prompt": ""},
        {"model": "opus", "prompt": "   "},
        {"model": "opus", "prompt": None},
        {"model": "opus", "prompt": 42},
    ],
)
def test_unlintable_spawn_reports_no_prompt_not_clean(hook, tool_input):
    blocked, messages, status = hook.charter_lint(tool_input, {})
    assert status == "no-prompt", "vacuity must never be reported as a pass"
    assert blocked is False
    assert messages == []


def test_no_prompt_is_announced_loudly_at_the_process_boundary():
    """A spawn the hook could not examine must SAY SO on stderr."""
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps({"tool_name": "Agent", "tool_input": {"model": "opus"}}),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "", "an unlintable spawn is allowed, not denied"
    assert "DID NOT RUN" in proc.stderr
    assert "no-prompt" in proc.stderr


def test_clean_charter_is_silent(hook):
    """A genuinely clean charter produces no noise — distinguishable from above."""
    blocked, messages, status = hook.charter_lint({"prompt": CHARTERS["waived"]}, {})
    assert status == "ran"
    assert blocked is False
    assert messages == []


# --- FAIL-OPEN: the hook gates every Agent spawn --------------------------------


def test_lint_fails_open_and_loudly_when_legs_unavailable(hook, monkeypatch):
    def _boom():
        raise ImportError("simulated missing codex_arm_queue")

    monkeypatch.setattr(hook, "_load_lint_legs", _boom)
    blocked, messages, status = hook.charter_lint({"prompt": "Build a thing."}, {"TAC_CHARTER_LINT_STRICT": "1"})
    assert blocked is False, "a lint bug must never brick a spawn"
    assert status.startswith("unavailable:")
    assert messages == []


def test_lint_never_blocks_when_model_guard_already_denies():
    """Ordering: a routing denial keeps its own reason; the lint does not overwrite it."""
    proc = subprocess.run(
        [sys.executable, str(_HOOK)],
        input=json.dumps({"tool_name": "Agent", "tool_input": {"prompt": "Build a thing."}}),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    payload = json.loads(proc.stdout)
    assert payload["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "EXPLICIT allowed model" in payload["reason"]


# --- ANTI-DRIFT: one implementation, not two ------------------------------------


def test_legs_are_imported_never_copied():
    """A second copy of a lint is a second thing to drift."""
    source = _HOOK.read_text(encoding="utf-8")
    assert "def _lint_stale_numbers" not in source
    assert "def _lint_falsified_premises" not in source
    assert "def lint_charter_optimal_form" not in source
    assert "_load_lint_legs" in source


def test_agent_matcher_is_still_the_only_hook_on_that_path():
    """If a second Agent hook appears, this binding needs re-examining."""
    settings = json.loads((_REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))
    commands = [
        h["command"]
        for entry in settings["hooks"]["PreToolUse"]
        if entry.get("matcher") == "Agent"
        for h in entry["hooks"]
    ]
    assert any("agent_model_routing_guard_hook.py" in c for c in commands)
