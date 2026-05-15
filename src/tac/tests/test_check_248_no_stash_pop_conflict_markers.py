# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_stash_pop_conflict_markers_in_canonical_files,
)


def test_check_248_clean_tree_returns_empty(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.py").write_text("value = 1\n", encoding="utf-8")

    assert check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path
    ) == []


def test_check_248_detects_stash_pop_markers(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.py").write_text(
        "value = 1\n<<<<<<< Updated upstream\nvalue = 2\n>>>>>>> Stashed changes\n",
        encoding="utf-8",
    )

    violations = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path
    )

    assert len(violations) == 2
    assert "src/bad.py:2" in violations[0]
    assert "src/bad.py:4" in violations[1]


def test_check_248_strict_raises(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "bad.sh").write_text(
        "#!/usr/bin/env bash\n<<<<<<< HEAD\n",
        encoding="utf-8",
    )

    with pytest.raises(PreflightError, match="Catalog #248"):
        check_no_stash_pop_conflict_markers_in_canonical_files(
            repo_root=tmp_path,
            strict=True,
        )


def test_check_248_allows_explicit_same_line_documentation_waiver(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "runbook.md").write_text(
        "<<<<<<< HEAD # CONFLICT_MARKER_INTENTIONAL_OK: documents git marker syntax\n",
        encoding="utf-8",
    )

    assert check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path
    ) == []


# ----------------------------------------------------------------------------
# Extended coverage (GRAND-COUNCIL-MULTI-SISTER-MERGE-RESOLUTION 2026-05-15)
# ----------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write(tmp: Path, rel: str, body: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def test_check_248_live_repo_clean() -> None:
    """The live repo MUST be clean at landing per Strict-flip atomicity rule."""
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert v == [], f"unexpected residual conflict markers: {v[:5]}"


def test_check_248_generic_branch_marker_flagged(tmp_path: Path) -> None:
    body = "x = 1\n<<<<<<< feature-branch\ny = 2\n=======\ny = 3\n>>>>>>> main\n"
    _write(tmp_path, "src/bug.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    assert len(v) >= 2


def test_check_248_placeholder_rationale_waiver_rejected(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream  # CONFLICT_MARKER_INTENTIONAL_OK:<rationale>\n"
    _write(tmp_path, "src/bad.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_check_248_placeholder_reason_waiver_rejected(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream  # CONFLICT_MARKER_INTENTIONAL_OK:<reason>\n"
    _write(tmp_path, "src/bad.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_check_248_excluded_path_markers_skipped(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream\n=======\n>>>>>>> Stashed changes\n"
    _write(tmp_path, "experiments/results/foo.json", body)
    _write(tmp_path, "src/something_intake_bar.py", body)
    _write(tmp_path, "vendored/baz.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_248_yaml_and_json_also_scanned(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream\nfoo: bar\n=======\nfoo: baz\n>>>>>>> Stashed changes\n"
    _write(tmp_path, "configs/foo.yaml", body)
    _write(tmp_path, "configs/foo.json", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    # 2 markers per file × 2 files
    assert len(v) == 4


def test_check_248_unrelated_text_not_flagged(tmp_path: Path) -> None:
    body = (
        "# divider line below\n"
        "===============================================================\n"
        "x = 1\n"
        "# arrows in text: <<< or >>>\n"
        "y = 'abc <<< def'\n"
    )
    _write(tmp_path, "src/divider.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_check_248_string_repo_root_accepted(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream\n=======\n>>>>>>> Stashed changes\n"
    _write(tmp_path, "src/bug.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=str(tmp_path), strict=False
    )
    assert len(v) == 2


def test_check_248_multiple_violations_aggregated(tmp_path: Path) -> None:
    body = "<<<<<<< Updated upstream\n=======\n>>>>>>> Stashed changes\n"
    for i in range(3):
        _write(tmp_path, f"src/bug{i}.py", body)
    v = check_no_stash_pop_conflict_markers_in_canonical_files(
        repo_root=tmp_path, strict=False
    )
    # 2 markers per file × 3 files
    assert len(v) == 6
