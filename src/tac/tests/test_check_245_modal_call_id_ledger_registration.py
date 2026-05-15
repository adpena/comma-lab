# SPDX-License-Identifier: MIT
"""Catalog #245 - Modal spawn call_ids must enter the canonical ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_modal_dispatches_register_call_id


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_check_245_flags_modal_spawn_without_call_id_ledger_registration(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "experiments/bad_modal_spawn.py",
        """\
import modal

app = modal.App("bad")
fn = modal.Function.lookup("app", "train")
call_id = fn.spawn("x")
print(call_id)
""",
    )

    violations = check_modal_dispatches_register_call_id(repo_root=tmp_path, strict=False, verbose=False)

    assert len(violations) == 1
    assert "bad_modal_spawn.py:5" in violations[0]
    assert "canonical ledger" in violations[0]


def test_check_245_accepts_spawn_with_nearby_canonical_registration(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "experiments/good_modal_spawn.py",
        """\
import modal
from tac.deploy.modal.call_id_ledger import register_dispatched_call_id

app = modal.App("good")
fn = modal.Function.lookup("app", "train")
call_id = fn.spawn("x")
register_dispatched_call_id(call_id=call_id, lane_id="lane_x", label="x")
""",
    )

    assert check_modal_dispatches_register_call_id(repo_root=tmp_path, strict=False, verbose=False) == []


def test_check_245_strict_raises_on_unregistered_modal_spawn(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "tools/bad_modal_spawn.py",
        """\
import modal

function = modal.Function.lookup("app", "train")
function.spawn()
""",
    )

    with pytest.raises(PreflightError, match="register the call_id"):
        check_modal_dispatches_register_call_id(repo_root=tmp_path, strict=True, verbose=False)


def test_check_245_live_repo_has_no_unregistered_modal_spawn_sites() -> None:
    assert check_modal_dispatches_register_call_id(strict=False, verbose=False) == []
