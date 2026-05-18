# SPDX-License-Identifier: MIT
"""Strict validation checks for the canonical task-status ledger."""

from __future__ import annotations

from pathlib import Path

from .loader import load_canonical_task_status_strict


def canonical_task_status_violations(repo_root: str | Path | None = None) -> list[str]:
    """Return strict ledger violations not already covered by schema loading."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for row in load_canonical_task_status_strict(root):
        memo = root / row.source_design_memo
        if not memo.is_file():
            violations.append(
                f"{row.task_id}: source_design_memo missing: {row.source_design_memo}"
            )
    return violations


def check_canonical_task_status_no_dangling_transitions(
    *,
    repo_root: str | Path | None = None,
    strict: bool = True,
    verbose: bool = False,
) -> list[str]:
    """Validate canonical task-status schema, transitions, and memo pointers."""

    violations = canonical_task_status_violations(repo_root)
    if violations and strict:
        raise AssertionError(
            "canonical_task_status violations:\n" + "\n".join(f"- {v}" for v in violations)
        )
    if verbose:
        if violations:
            print(
                "  [canonical-task-status] WARN: "
                f"{len(violations)} violation(s)"
            )
        else:
            print("  [canonical-task-status] OK")
    return violations

