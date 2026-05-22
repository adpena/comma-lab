# SPDX-License-Identifier: MIT
"""Regression tests for operator-facing frontier mirror docs."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_state_frontier_docs_mirror_canonical_cpu_pointer() -> None:
    pointer = json.loads(
        (REPO_ROOT / ".omx/state/canonical_frontier_pointer.json").read_text()
    )
    cpu = pointer["our_local_frontier_contest_cpu"]
    score = str(cpu["score"])
    archive_sha = str(cpu["archive_sha256"])
    lane = str(cpu["extra"]["architecture_class"])

    for rel_path in (
        ".omx/state/current_focus.md",
        ".omx/state/next_experiments.md",
    ):
        text = (REPO_ROOT / rel_path).read_text()
        assert score in text, rel_path
        assert archive_sha in text, rel_path
        assert lane in text, rel_path
        assert "this file is a mirror, not a\n  frontier source of truth" in text
