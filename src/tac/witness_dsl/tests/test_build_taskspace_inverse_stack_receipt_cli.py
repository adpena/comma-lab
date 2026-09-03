# SPDX-License-Identifier: MIT
"""The stack-receipt tool's ``--strict-source-reopen`` must be a REAL, wired flag.

ddm_ql1 found the tool hard-coded ``strict_source_reopen=True``, so when the
V9/PBR2 renderer manifest pin drifted the tool became unusable with no way to
ask for the declared degraded mode. ddm_ql2 exposed the flag. These tests pin
the two things that make a flag real rather than decorative, per CLAUDE.md's
never-invent-flags discipline: it EXISTS in argparse with the safe default, and
its value actually REACHES the consumer instead of being shadowed by a literal.

The build itself is expensive and is exercised elsewhere; here the consumer is
substituted so the WIRING is what is under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import build_taskspace_inverse_stack_receipt as cli

_RECEIPT = {
    "body_sha256": "0" * 64,
    "body": {
        "verdict": "REOPENED_EVIDENCE_STACK_BLOCKED_BEFORE_N600_CANDIDATE_AUTHORITY",
        "exact_blockers": ["receiver_consumption_custody_absent"],
        "source_reopen": {"teacher_census": "NOT_RUN", "prior_signal_harvest": "NOT_RUN"},
    },
}


def test_strict_source_reopen_is_a_real_flag_defaulting_to_strict() -> None:
    assert cli.parse_args([]).strict_source_reopen is True
    assert cli.parse_args(["--strict-source-reopen"]).strict_source_reopen is True
    assert cli.parse_args(["--no-strict-source-reopen"]).strict_source_reopen is False


def test_unknown_reopen_spelling_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--strict-source-reopen=maybe"])


@pytest.mark.parametrize("strict", [True, False])
def test_flag_value_reaches_the_builder(monkeypatch: pytest.MonkeyPatch, strict: bool, capsys) -> None:
    seen: dict[str, object] = {}

    def _build(*, repo_root: Path, strict_source_reopen: bool) -> dict:
        seen["repo_root"] = repo_root
        seen["strict_source_reopen"] = strict_source_reopen
        return _RECEIPT

    monkeypatch.setattr(cli, "build_stack_receipt", _build)
    monkeypatch.setattr(cli, "write_once_receipt", lambda *a, **k: None)
    argv = [] if strict else ["--no-strict-source-reopen"]
    assert cli.main(argv) == 0
    assert seen["strict_source_reopen"] is strict
    assert json.loads(capsys.readouterr().out)["strict_source_reopen"] is strict


def test_degraded_mode_never_publishes_and_says_so(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A NOT_RUN receipt is honest evidence but is not canonical bytes.

    ``write_once_receipt`` re-validates by rebuilding with the strict reopen, so
    publishing a degraded receipt is impossible anyway; the tool must decline
    up front rather than fail inside the publisher.
    """

    published: list[object] = []
    monkeypatch.setattr(cli, "build_stack_receipt", lambda **_: _RECEIPT)
    monkeypatch.setattr(cli, "write_once_receipt", lambda *a, **k: published.append(a))

    assert cli.main(["--no-strict-source-reopen"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert published == []
    assert report["published"] is False
    assert report["output"] is None
    assert report["source_reopen"]["teacher_census"] == "NOT_RUN"

    assert cli.main(["--dry-run"]) == 0
    assert published == []
    assert json.loads(capsys.readouterr().out)["published"] is False

    assert cli.main([]) == 0
    assert len(published) == 1
    assert json.loads(capsys.readouterr().out)["published"] is True


def test_report_never_claims_score_or_promotion(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_stack_receipt", lambda **_: _RECEIPT)
    monkeypatch.setattr(cli, "write_once_receipt", lambda *a, **k: None)
    assert cli.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["pointer_moved"] is False
