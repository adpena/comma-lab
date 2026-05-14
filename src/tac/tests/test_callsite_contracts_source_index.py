# SPDX-License-Identifier: MIT
"""Focused source-index regression for callsite contract preflight."""
from __future__ import annotations

import textwrap
from pathlib import Path

from tac.preflight import check_callsite_contracts_satisfied
from tac.source_index import source_index_context


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


def test_callsite_contract_uses_source_index_substring_prefilter(tmp_path: Path) -> None:
    for i in range(40):
        _write(
            tmp_path,
            f"experiments/no_contract_{i}.py",
            """
            def unrelated(value):
                return value + 1
            """,
        )
    _write(
        tmp_path,
        "experiments/bad.py",
        """
        from tac.pose_gaussian_process import reconstruct_poses
        x = reconstruct_poses(model, 600)
        """,
    )

    with source_index_context(tmp_path) as index:
        violations = check_callsite_contracts_satisfied(
            strict=False,
            verbose=False,
            repo_root=tmp_path,
        )
        stats = index.stats()

    assert len(violations) == 1
    assert "experiments/bad.py" in violations[0]
    assert stats["facts_group_misses"] == 1
    assert stats["ast_cache_entries"] == 1
