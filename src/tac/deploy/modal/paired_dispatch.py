# SPDX-License-Identifier: MIT
"""Canonical operator command templates for paired Modal auth-eval dispatch."""

from __future__ import annotations

from pathlib import Path

PAIRED_AUTH_EVAL_DISPATCH_TOOL = "tools/dispatch_modal_paired_auth_eval.py"
PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT = "codex:modal_paired_auth_eval"


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
    "PAIRED_AUTH_EVAL_DEFAULT_CLAIM_AGENT",
    "PAIRED_AUTH_EVAL_DISPATCH_TOOL",
    "paired_auth_eval_dispatch_command_template",
]
