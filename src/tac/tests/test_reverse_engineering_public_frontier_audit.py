# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from comma_lab.reverse_engineering import (
    audit_reverse_engineering_tree,
    blocking_records,
)


def test_public_frontier_runtime_references_are_curated_forensic_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    runtime = (
        repo
        / "reverse_engineering/public_frontier/recovered_runtime/example_pr/runtime"
    )
    runtime.mkdir(parents=True)
    source = runtime / "inflate.py"
    source.write_text("print('external reference only')\n", encoding="utf-8")

    records = audit_reverse_engineering_tree(repo)

    assert blocking_records(records) == []
    record = next(item for item in records if item.relpath.endswith("inflate.py"))
    assert record.category == "public_frontier_runtime_reference"
    assert record.disposition == "track_in_git"


def test_anatomy_reviewed_pr_deconstruction_source_is_curated(
    tmp_path: Path,
) -> None:
    """A .py beside an ANATOMY.md review note is curated, non-blocking.

    Anchor: reverse_engineering/quantizr_pr55/{compress,inflate}.py were
    deliberately tracked with their anatomy note (9c59e3f4c1) yet hit the
    manual_review fallback (2026-08-25 #842 loop). The co-located
    ANATOMY.md is the explicit-review artifact the tree's charter demands.
    """
    repo = tmp_path
    intake = repo / "reverse_engineering/quantizr_pr55"
    intake.mkdir(parents=True)
    (intake / "ANATOMY.md").write_text("# anatomy review\n", encoding="utf-8")
    (intake / "compress.py").write_text("print('pr source')\n", encoding="utf-8")

    records = audit_reverse_engineering_tree(repo)

    assert blocking_records(records) == []
    record = next(item for item in records if item.relpath.endswith("compress.py"))
    assert record.category == "public_pr_deconstruction_source"
    assert record.disposition == "track_in_git"


def test_stray_py_without_anatomy_note_still_blocks(tmp_path: Path) -> None:
    """Acceptance unchanged: no review note -> manual_review blocker."""
    repo = tmp_path
    intake = repo / "reverse_engineering/mystery_dump"
    intake.mkdir(parents=True)
    (intake / "grabbed.py").write_text("print('unreviewed')\n", encoding="utf-8")

    records = audit_reverse_engineering_tree(repo)

    blockers = blocking_records(records)
    assert len(blockers) == 1
    assert blockers[0].disposition == "manual_review"
