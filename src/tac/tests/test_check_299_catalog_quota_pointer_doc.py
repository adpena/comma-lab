# SPDX-License-Identifier: MIT
"""Regression tests for Catalog #299 pointer-backed catalog parsing."""
from __future__ import annotations

from pathlib import Path

from tac.preflight import check_catalog_quota_under_400


def _write_pointer_catalog(root: Path, row: str) -> None:
    (root / "CLAUDE.md").write_text("See `docs/meta_bug_class_catalog.md`.\n")
    docs_dir = root / "docs"
    docs_dir.mkdir()
    (docs_dir / "meta_bug_class_catalog.md").write_text(row)


def test_catalog_299_reads_extracted_catalog_doc(tmp_path: Path) -> None:
    _write_pointer_catalog(
        tmp_path,
        "401. `check_future_over_quota_gate` - over quota row.\n",
    )

    violations = check_catalog_quota_under_400(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "Catalog #401" in violations[0]


def test_catalog_299_accepts_extracted_doc_rows_under_quota(tmp_path: Path) -> None:
    _write_pointer_catalog(
        tmp_path,
        "399. `check_future_under_quota_gate` - under quota row.\n",
    )

    violations = check_catalog_quota_under_400(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert violations == []
