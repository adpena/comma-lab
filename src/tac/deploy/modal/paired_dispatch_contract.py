# SPDX-License-Identifier: MIT
"""Fail-closed command contract for paired Modal auth-eval targets."""

from __future__ import annotations

from tac.deploy.modal.paired_dispatch import PAIRED_AUTH_EVAL_DISPATCH_TOOL

PAIRED_AUTH_EVAL_REQUIRED_COMMAND_FRAGMENTS = (
    PAIRED_AUTH_EVAL_DISPATCH_TOOL,
    "--expected-runtime-tree-sha256 auto",
    "--skip-axis-if-promotable-anchor-exists",
)
PAIRED_AUTH_EVAL_FORBIDDEN_SINGLE_AXIS_ENTRYPOINTS = (
    "experiments/modal_auth_eval.py",
    "experiments/modal_auth_eval_cpu.py",
)
PAIRED_AUTH_EVAL_AXIS_RUNTIME_PLACEHOLDER = (
    "<AXIS_SPECIFIC_MODAL_UPLOADED_RUNTIME_TREE_SHA256>"
)


def paired_auth_eval_dispatch_command_blockers(
    *,
    paired_dispatch_tool: object,
    command_template: object,
    require_command: bool = False,
) -> list[str]:
    """Return false-authority blockers for an executable paired auth-eval command."""

    blockers: list[str] = []
    if paired_dispatch_tool != PAIRED_AUTH_EVAL_DISPATCH_TOOL:
        blockers.append("paired_dispatch_tool_not_canonical")

    command = str(command_template or "")
    if not command:
        if require_command:
            blockers.append("paired_dispatch_command_missing")
        return blockers

    if PAIRED_AUTH_EVAL_AXIS_RUNTIME_PLACEHOLDER in command:
        blockers.append("axis_specific_runtime_tree_placeholder_leak")
    if any(
        entrypoint in command
        for entrypoint in PAIRED_AUTH_EVAL_FORBIDDEN_SINGLE_AXIS_ENTRYPOINTS
    ):
        blockers.append("single_axis_modal_entrypoint_leak")
    for fragment in PAIRED_AUTH_EVAL_REQUIRED_COMMAND_FRAGMENTS:
        if fragment not in command:
            blockers.append(f"paired_dispatch_command_missing:{fragment}")
    return blockers


__all__ = [
    "PAIRED_AUTH_EVAL_AXIS_RUNTIME_PLACEHOLDER",
    "PAIRED_AUTH_EVAL_FORBIDDEN_SINGLE_AXIS_ENTRYPOINTS",
    "PAIRED_AUTH_EVAL_REQUIRED_COMMAND_FRAGMENTS",
    "paired_auth_eval_dispatch_command_blockers",
]
