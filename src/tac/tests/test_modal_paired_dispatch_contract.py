# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.deploy.modal.paired_dispatch_contract import (
    paired_auth_eval_dispatch_command_blockers,
)


def test_paired_dispatch_command_blockers_accept_canonical_plan() -> None:
    blockers = paired_auth_eval_dispatch_command_blockers(
        paired_dispatch_tool="tools/dispatch_modal_paired_auth_eval.py",
        command_template=(
            ".venv/bin/python tools/dispatch_modal_paired_auth_eval.py "
            "--archive a.zip --expected-runtime-tree-sha256 auto "
            "--skip-axis-if-promotable-anchor-exists"
        ),
    )

    assert blockers == []


def test_paired_dispatch_command_blockers_reject_single_axis_targets() -> None:
    blockers = paired_auth_eval_dispatch_command_blockers(
        paired_dispatch_tool="tools/dispatch_modal_paired_auth_eval.py",
        command_template=".venv/bin/modal run experiments/modal_auth_eval.py --archive a.zip",
    )

    assert "single_axis_modal_entrypoint_leak" in blockers
    assert (
        "paired_dispatch_command_missing:tools/dispatch_modal_paired_auth_eval.py"
        in blockers
    )
    assert (
        "paired_dispatch_command_missing:--expected-runtime-tree-sha256 auto"
        in blockers
    )
    assert (
        "paired_dispatch_command_missing:--skip-axis-if-promotable-anchor-exists"
        in blockers
    )
