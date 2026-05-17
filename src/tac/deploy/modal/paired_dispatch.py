# SPDX-License-Identifier: MIT
"""Canonical operator command templates for paired Modal auth-eval dispatch."""

from __future__ import annotations

from pathlib import Path

PAIRED_AUTH_EVAL_DISPATCH_TOOL = "tools/dispatch_modal_paired_auth_eval.py"
PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT = "codex:modal_paired_auth_eval"
PAIRED_AUTH_EVAL_CUDA_WRAPPER = "experiments/modal_auth_eval.py"
PAIRED_AUTH_EVAL_CPU_WRAPPER = "experiments/modal_auth_eval_cpu.py"
MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR = "/tmp/modal_auth_eval/submission_dir"
MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR = "/tmp/modal_auth_eval_cpu/submission_dir"


def paired_auth_eval_axis_command(
    *,
    axis: str,
    modal_bin: str,
    archive_path: str | Path,
    archive_sha256: str,
    inflate_sh: str,
    output_dir: str | Path,
    pair_group_id: str,
    lane_id: str,
    instance_job_id: str,
    claim_agent: str,
    claim_notes: str,
    submission_dir: str | Path | None = None,
    expected_runtime_tree_sha256: str = "",
    gpu: str = "T4",
) -> list[str]:
    """Build one detached Modal auth-eval wrapper command.

    This is the shared per-axis command shape used by paired dispatcher plans
    and L5 materialized work-unit ledgers. It deliberately does not execute
    anything; callers own lane claims, JSON persistence, and operator review.
    """

    if axis not in {"contest_cuda", "contest_cpu"}:
        raise ValueError(f"unsupported Modal auth-eval axis: {axis!r}")
    wrapper = (
        PAIRED_AUTH_EVAL_CUDA_WRAPPER
        if axis == "contest_cuda"
        else PAIRED_AUTH_EVAL_CPU_WRAPPER
    )
    command = [
        modal_bin,
        "run",
        "--detach",
        wrapper,
        "--archive",
        str(archive_path),
        "--expected-archive-sha256",
        str(archive_sha256),
        "--inflate-sh",
        inflate_sh,
        "--output-dir",
        str(output_dir),
    ]
    if axis == "contest_cuda":
        command.extend(["--gpu", gpu])
    command.extend(
        [
            "--detach",
            "--provider-detach-ack",
            "--pair-group-id",
            pair_group_id,
            "--lane-id",
            lane_id,
            "--instance-job-id",
            instance_job_id,
            "--claim-agent",
            claim_agent,
            "--claim-notes",
            claim_notes,
        ]
    )
    if submission_dir:
        command.extend(["--submission-dir", str(submission_dir)])
    if expected_runtime_tree_sha256:
        command.extend(
            ["--expected-runtime-tree-sha256", str(expected_runtime_tree_sha256)]
        )
    return command


def paired_auth_eval_dispatch_command_template(
    *,
    archive_path: str | Path,
    submission_dir: str | Path,
    lane_id_base: str,
    archive_sha256: str,
    execute: bool,
    label: str | None = None,
    run_id: str | None = None,
    inflate_sh: str = "inflate.sh",
    output_root: str | Path = "experiments/results",
    modal_bin: str = ".venv/bin/modal",
    gpu: str = "T4",
    claim_agent: str = PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT,
    claim_notes: str = "",
) -> list[str]:
    """Build the plan-only or execute form of the paired Modal dispatcher.

    This deliberately returns the operator-facing paired dispatcher command, not
    the underlying per-axis Modal wrapper commands. That keeps packet builders
    from reintroducing single-axis CPU/CUDA dispatch surfaces.
    """

    command = [
        ".venv/bin/python",
        PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        "--archive",
        str(archive_path),
        "--submission-dir",
        str(submission_dir),
        "--inflate-sh",
        inflate_sh,
        "--label",
        str(label or lane_id_base),
        "--expected-archive-sha256",
        str(archive_sha256),
    ]
    if run_id:
        command.extend(["--run-id", str(run_id)])
    command.extend(
        [
            "--pair-group-id",
            f"pair_{lane_id_base}_{str(archive_sha256)[:12]}",
            "--lane-id-base",
            lane_id_base,
            "--output-root",
            str(output_root),
            "--modal-bin",
            modal_bin,
            "--gpu",
            gpu,
            "--claim-agent",
            claim_agent,
        ]
    )
    if claim_notes:
        command.extend(["--claim-notes", claim_notes])
    command.extend(
        [
            "--expected-runtime-tree-sha256",
            "auto",
            "--skip-axis-if-promotable-anchor-exists",
        ]
    )
    if execute:
        command.append("--execute")
    return command


__all__ = [
    "MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR",
    "MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR",
    "PAIRED_AUTH_EVAL_CPU_WRAPPER",
    "PAIRED_AUTH_EVAL_CUDA_WRAPPER",
    "PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT",
    "PAIRED_AUTH_EVAL_DISPATCH_TOOL",
    "paired_auth_eval_axis_command",
    "paired_auth_eval_dispatch_command_template",
]
