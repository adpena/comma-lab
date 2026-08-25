# SPDX-License-Identifier: MIT
"""Catalog #206 serializer auto-waiver: determinized checkpoint evidence.

2026-08-25 adjudication: the checkpoint discipline was practiced (307
compliant commits after the 2026-05-19 cutoff) then lapsed — the #936
adoption-decay genus. The cure lives at the ergonomic layer: the serializer
appends one honest class-specific `# CHECKPOINT_DISCIPLINE_WAIVED:<reason>`
line when the caller's message carries no checkpoint token, and stays out
of the way whenever the caller supplied its own evidence. These tests bind
the auto-line to the GATE'S OWN matchers (`_check_206_body_has_checkpoint_
signal` + `_CHECKPOINT_WAIVER_RE`), so a drift in either surface goes red
here rather than silently un-curing the gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tac.preflight import (
    _CHECKPOINT_WAIVER_RE,
    _check_206_body_has_checkpoint_signal,
)

_SERIALIZER_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "subagent_commit_serializer.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_serializer_checkpoint_discipline_line", _SERIALIZER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_module()
_ensure = _MOD._ensure_checkpoint_discipline_line


def test_anonymous_message_gets_waiver_that_satisfies_the_gate() -> None:
    out = _ensure("fix: some single-shot landing", "anonymous")

    assert out != "fix: some single-shot landing"
    assert _CHECKPOINT_WAIVER_RE.search(out) is not None
    assert _check_206_body_has_checkpoint_signal(out)


def test_labeled_arm_message_gets_keeper_custody_reason() -> None:
    out = _ensure("ddm_xy1: land verdict", "ddm_xy1_some_arm")

    assert "keeper-arm 'ddm_xy1_some_arm'" in out
    assert _CHECKPOINT_WAIVER_RE.search(out) is not None
    assert _check_206_body_has_checkpoint_signal(out)


def test_waiver_reason_is_non_empty_and_not_placeholder() -> None:
    out = _ensure("fix: thing", "anonymous")

    match = _CHECKPOINT_WAIVER_RE.search(out)
    assert match is not None
    reason = match.group(1).strip()
    assert reason
    assert not reason.startswith("<")


def test_suppressed_when_checkpoint_token_already_present() -> None:
    msg = "fix: thing\n\ncheckpoint discipline honored (10-step protocol)"

    assert _ensure(msg, "anonymous") == msg


def test_suppressed_when_caller_supplied_waiver_present() -> None:
    msg = "fix: thing\n\n# CHECKPOINT_DISCIPLINE_WAIVED:caller's own reason"

    assert _ensure(msg, "anonymous") == msg


def test_suppressed_on_bare_reasonless_waiver_caller_intent_wins() -> None:
    # A bare reason-less waiver is REJECTED by the gate — deliberately NOT
    # rescued here: auto-appending a second waiver would mask caller intent.
    msg = "fix: thing\n\nCHECKPOINT_DISCIPLINE_WAIVED"

    out = _ensure(msg, "anonymous")

    assert out == msg
    assert not _check_206_body_has_checkpoint_signal(out)


def test_token_match_is_case_insensitive_like_the_gate() -> None:
    msg = "fix: thing — uses Tools/Subagent_Checkpoint.PY protocol"

    assert _ensure(msg, "anonymous") == msg
