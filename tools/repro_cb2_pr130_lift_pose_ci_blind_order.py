#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce the #983 nested-package CI-blind over-selection order.

The fixed hook no longer emits the bare ``pose`` token for
``src/tac/pr130_lift/pose/__init__.py``.  This script preserves the legacy
selection path so MAIN can run the old multi-module order on a Metal host
without editing the hook back to the broken state.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import preflight_hook  # noqa: E402

STAGED_POSE_INIT = "src/tac/pr130_lift/pose/__init__.py"
PAIR_TARGETS = [
    "src/tac/tests/test_compact_renderer_mlx_spine_runner.py::"
    "test_hinerv_execute_runs_training_archive_and_receiver_proof",
    "src/tac/tests/test_levelset_micro_batch_loss.py",
]


def legacy_pose_token_targets() -> list[str]:
    """Targets selected by the pre-fix bare-token ``pose`` matcher."""

    selected: list[str] = []
    pattern = re.compile(r"\bpose\b")
    for path in preflight_hook._ci_blind_test_modules():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            selected.extend(preflight_hook._ci_blind_targets_for(path, text))
    return selected


def _run_pytest(targets: list[str]) -> int:
    if not targets:
        print("no targets selected")
        return 0
    cmd = [
        ".venv/bin/python",
        "-m",
        "pytest",
        *targets,
        "-q",
        "--no-header",
        "-m",
        "not slow",
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-current",
        action="store_true",
        help="Run the fixed hook selection for src/tac/pr130_lift/pose/__init__.py.",
    )
    parser.add_argument(
        "--run-legacy",
        action="store_true",
        help="Run the legacy broad bare-pose selection that triggered #983.",
    )
    parser.add_argument(
        "--run-pair",
        action="store_true",
        help="Run the minimal ordered pair extracted from the legacy target set.",
    )
    args = parser.parse_args(argv)

    current = preflight_hook._select_ci_blind_tests([STAGED_POSE_INIT])
    legacy = legacy_pose_token_targets()
    print(f"current_count={len(current)}")
    for target in current:
        print(f"current {target}")
    print(f"legacy_count={len(legacy)}")
    for target in legacy:
        print(f"legacy {target}")

    if args.run_pair:
        return _run_pytest(PAIR_TARGETS)
    if args.run_legacy:
        return _run_pytest(legacy)
    if args.run_current:
        return _run_pytest(current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
