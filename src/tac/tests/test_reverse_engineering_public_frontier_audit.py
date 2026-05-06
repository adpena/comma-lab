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
