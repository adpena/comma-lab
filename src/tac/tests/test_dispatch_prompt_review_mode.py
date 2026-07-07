"""Tests for tools/dispatch_prompt.py --review mode (review_contract composition)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "dispatch_prompt", _REPO / "tools" / "dispatch_prompt.py"
)
dispatch_prompt = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("dispatch_prompt", dispatch_prompt)
_spec.loader.exec_module(dispatch_prompt)


def test_compose_review_default_is_report_only() -> None:
    out = dispatch_prompt.compose_review("Review commits A B.")
    assert out.startswith("Review commits A B.")
    assert "REPORT-ONLY DISPATCH" in out
    assert "subagent_commit_serializer" not in out
    assert "probability × blast-radius × SILENCE" in out  # manual §3
    assert "never blur them" in out  # CONFIRMED vs PLAUSIBLE
    assert "round-finished ≠ clean-pass" in out  # counter semantics


def test_compose_review_allow_commits_appends_serializer() -> None:
    out = dispatch_prompt.compose_review("Review + fix.", allow_commits=True)
    assert "subagent_commit_serializer" in out
    assert "REPORT-ONLY DISPATCH" not in out


def test_compose_review_counter_context_threaded() -> None:
    out = dispatch_prompt.compose_review(
        "Review X.", counter_context="surface X: counter 2/3"
    )
    assert "REVIEW-COUNTER CONTEXT: surface X: counter 2/3" in out


def test_build_mode_unchanged() -> None:
    out = dispatch_prompt.compose("Build X.")
    assert "subagent_commit_serializer" in out
    assert "REPORT-ONLY DISPATCH" not in out
    assert "Before reporting progress, audit each claim against a tool result" in out
