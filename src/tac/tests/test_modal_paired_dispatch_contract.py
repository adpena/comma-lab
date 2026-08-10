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


def test_axis_command_names_the_modal_entrypoint_explicitly() -> None:
    """``modal run <file>`` is ambiguous once the app has >1 local entrypoint.

    Measured 2026-08-10: both wrappers expose ``prove_env`` alongside ``main``,
    so the bare form died with "Specify a Modal Function or local entrypoint to
    run" and refused the lc2 exact-row dispatch. The wrapper target must always
    carry the ``::main`` suffix.
    """

    from tac.deploy.modal.paired_dispatch import (
        PAIRED_AUTH_EVAL_CPU_WRAPPER,
        PAIRED_AUTH_EVAL_CUDA_WRAPPER,
        PAIRED_AUTH_EVAL_ENTRYPOINT,
        paired_auth_eval_axis_command,
    )

    expected = {
        "contest_cuda": f"{PAIRED_AUTH_EVAL_CUDA_WRAPPER}::{PAIRED_AUTH_EVAL_ENTRYPOINT}",
        "contest_cpu": f"{PAIRED_AUTH_EVAL_CPU_WRAPPER}::{PAIRED_AUTH_EVAL_ENTRYPOINT}",
    }
    for axis, wrapper_target in expected.items():
        command = paired_auth_eval_axis_command(
            axis=axis,
            modal_bin=".venv/bin/modal",
            archive_path="/tmp/archive.zip",
            archive_sha256="ab" * 32,
            inflate_sh="inflate.sh",
            output_dir="/tmp/out",
            pair_group_id="pair_x",
            lane_id="lane_x",
            instance_job_id="job_x",
            claim_agent="MAIN",
            claim_notes="n",
        )
        # The wrapper target sits immediately after ``modal run --detach``.
        assert command[:3] == [".venv/bin/modal", "run", "--detach"]
        assert command[3] == wrapper_target
        # Positive control: the bare, ambiguous form must NOT appear anywhere.
        assert PAIRED_AUTH_EVAL_CUDA_WRAPPER not in command[4:]
        assert PAIRED_AUTH_EVAL_CPU_WRAPPER not in command[4:]
