# SPDX-License-Identifier: MIT
"""Catalog #330 - Modal harvesters must record call-id terminal outcomes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_modal_harvesters_record_call_id_outcome


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_330_flags_function_call_get_without_ledger_outcome(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/bad_modal_harvester.py",
        """\
import modal

fc = modal.functions.FunctionCall.from_id("fc-demo")
result = fc.get(timeout=2)
print(result)
""",
    )

    violations = check_modal_harvesters_record_call_id_outcome(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "bad_modal_harvester.py" in violations[0]
    assert "canonical call_id ledger" in violations[0]


def test_check_330_accepts_canonical_harvest_outcome_helper(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/good_modal_harvester.py",
        """\
import modal
from tac.deploy.modal.harvest_outcomes import append_terminal_call_id_ledger_event

fc = modal.functions.FunctionCall.from_id("fc-demo")
result = fc.get(timeout=2)
append_terminal_call_id_ledger_event(
    repo_root=ROOT,
    metadata={"call_id": "fc-demo"},
    harvested=result,
    terminal_claim=None,
    agent="pytest",
)
""",
    )

    assert check_modal_harvesters_record_call_id_outcome(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    ) == []


def test_check_330_accepts_explicit_nonplaceholder_waiver(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/waived_modal_harvester.py",
        """\
import modal

# HARVESTER_LEDGER_WRITE_OK: read-only liveness probe never observes terminal state
fc = modal.functions.FunctionCall.from_id("fc-demo")
fc.get(timeout=0.1)
""",
    )

    assert check_modal_harvesters_record_call_id_outcome(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    ) == []


def test_check_330_rejects_placeholder_waiver(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/placeholder_waiver.py",
        """\
import modal

# HARVESTER_LEDGER_WRITE_OK:<rationale>
fc = modal.functions.FunctionCall.from_id("fc-demo")
fc.get(timeout=0.1)
""",
    )

    violations = check_modal_harvesters_record_call_id_outcome(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1


def test_check_330_strict_raises_on_unmirrored_harvester(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "experiments/bad_modal_harvester.py",
        """\
import modal

modal.functions.FunctionCall.from_id("fc-demo").get(timeout=2)
""",
    )

    with pytest.raises(PreflightError, match="check_modal_harvesters_record_call_id_outcome"):
        check_modal_harvesters_record_call_id_outcome(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_check_330_live_repo_has_no_unmirrored_modal_harvesters() -> None:
    assert check_modal_harvesters_record_call_id_outcome(strict=False, verbose=False) == []
