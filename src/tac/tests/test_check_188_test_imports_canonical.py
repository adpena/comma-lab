"""Tests for Catalog #188 — check_test_imports_use_tac_not_src_tac.

FIX-WAVE-10 R10-1 silent-test-CI-evasion self-protect, 2026-05-13.

Bug class: test files using non-canonical ``from src.tac.*`` /
``import src.tac.*`` imports pass manually with
``PYTHONPATH=<repo> pytest <file>`` but fail default collection because
``from src.tac.X import Y`` raises ``ModuleNotFoundError: No module named
'src'`` when ``src/`` is not on ``sys.path``. Misplaced test files outside
configured testpaths are a sibling way for manual passes to diverge from
the default gate.

Sister of Catalog #25 (broken-import) + check_test_files_imports_resolve.
Together they extinct the silent-test-CI-evasion class bidirectionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_test_imports_use_tac_not_src_tac,
)

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_test_corpus(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a fake repo root with `src/tac/tests/` + `tests/` populated.

    Args:
        tmp_path: pytest tmp_path fixture root
        files: mapping of relative path -> file content. Paths starting with
            ``src/tac/tests/`` or ``tests/`` will land inside the scan dirs.
    """
    for rel, content in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # Ensure both scan dirs exist (even if empty) so the gate has stable
    # behavior under empty corpus.
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────
# Positive cases — gate should refuse
# ─────────────────────────────────────────────────────────────────────────


def test_from_src_tac_import_is_violation(tmp_path):
    """Top-level `from src.tac.X import Y` is a violation."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/test_bad.py": "from src.tac.foo import Bar\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1
    assert "test_bad.py" in violations[0]
    assert "from src.tac.foo import Bar" in violations[0]


def test_import_src_tac_is_violation(tmp_path):
    """Bare `import src.tac.X` is a violation."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/test_bad.py": "import src.tac.foo\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1
    assert "import src.tac.foo" in violations[0]


def test_from_src_tac_with_parens_is_violation(tmp_path):
    """`from src.tac.X import (` (multi-line import) is a violation."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_bad.py": (
                "from src.tac.foo import (\n"
                "    A,\n"
                "    B,\n"
                ")\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1
    # Only first line of the import flagged (which is correct — that's the
    # `from src.tac.X import (` statement)
    assert "from src.tac.foo import" in violations[0]


def test_indented_from_src_tac_is_violation(tmp_path):
    """Indented (inside function) `from src.tac.X import Y` is a violation."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_bad.py": (
                "def helper():\n    from src.tac.foo import Bar\n    return Bar\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1


def test_tests_dir_also_scanned(tmp_path):
    """Files in the top-level `tests/` dir are also scanned."""
    root = _make_test_corpus(
        tmp_path,
        {"tests/test_bad.py": "from src.tac.foo import Bar\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1
    assert "tests/test_bad.py" in violations[0]


def test_multiple_violations_in_one_file(tmp_path):
    """Multiple offending lines in one file each produce a violation."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_bad.py": (
                "from src.tac.foo import A\n"
                "from src.tac.bar import B\n"
                "import src.tac.baz\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 3


def test_multiple_files_each_violation_counted(tmp_path):
    """Violations across multiple files are aggregated."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_a.py": "from src.tac.foo import A\n",
            "src/tac/tests/test_b.py": "import src.tac.bar\n",
            "tests/test_c.py": "from src.tac.baz import C\n",
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 3


# ─────────────────────────────────────────────────────────────────────────
# Negative cases — gate should NOT refuse
# ─────────────────────────────────────────────────────────────────────────


def test_canonical_tac_import_is_clean(tmp_path):
    """Canonical `from tac.X import Y` is allowed (the whole point)."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/test_good.py": "from tac.foo import Bar\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_import_tac_is_clean(tmp_path):
    """Canonical `import tac` / `import tac.foo` is allowed."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_good.py": (
                "import tac\nimport tac.foo\nfrom tac.foo import Bar\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_docstring_mention_of_src_tac_is_clean(tmp_path):
    """Docstring/comment mentions of `src/tac/...` paths are NOT imports."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_good.py": (
                '"""Test for src.tac.foo module — references src/tac/foo.py."""\n'
                "# This test references src/tac/something.py as a fixture path\n"
                "from tac.foo import Bar\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_string_literal_src_tac_is_clean(tmp_path):
    """String literals containing `src/tac/...` paths are NOT imports."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_good.py": (
                "from tac.foo import Bar\n"
                'PATH = "src/tac/duplicate.py.pyc"\n'
                'COMMENT = "from src.tac is forbidden"\n'
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_empty_corpus_is_clean(tmp_path):
    """Empty test corpus produces no violations."""
    root = _make_test_corpus(tmp_path, {})
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_missing_test_dirs_are_clean(tmp_path):
    """Missing scan dirs don't raise; return [] cleanly."""
    # No test dirs at all.
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=tmp_path, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Waiver mechanism
# ─────────────────────────────────────────────────────────────────────────


def test_same_line_waiver_respected(tmp_path):
    """Same-line `# SRC_TAC_IMPORT_OK:<reason>` waives the violation."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_waived.py": (
                "from src.tac.foo import Bar  "
                "# SRC_TAC_IMPORT_OK:exercising-the-broken-import-error-path\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_waiver_placeholder_reason_rejected(tmp_path):
    """`# SRC_TAC_IMPORT_OK:<reason>` literal placeholder is NOT a valid waiver."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_bad_waiver.py": (
                "from src.tac.foo import Bar  # SRC_TAC_IMPORT_OK:<reason>\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1


def test_waiver_only_on_same_line_not_preceding(tmp_path):
    """Waiver on the previous line does NOT waive (must be same-line)."""
    root = _make_test_corpus(
        tmp_path,
        {
            "src/tac/tests/test_preceding_waiver.py": (
                "# SRC_TAC_IMPORT_OK:test\n"
                "from src.tac.foo import Bar\n"
            ),
        },
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1


# ─────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_on_violation(tmp_path):
    """`strict=True` raises PreflightError when violations exist."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/test_bad.py": "from src.tac.foo import Bar\n"},
    )
    with pytest.raises(PreflightError) as exc_info:
        check_test_imports_use_tac_not_src_tac(
            repo_root=root, strict=True, verbose=False
        )
    assert "check_test_imports_use_tac_not_src_tac" in str(exc_info.value)
    assert "ModuleNotFoundError" in str(exc_info.value)


def test_strict_mode_passes_when_clean(tmp_path):
    """`strict=True` returns [] silently when no violations exist."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/test_good.py": "from tac.foo import Bar\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, strict=True, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_clean():
    """The live repo must pass STRICT mode with 0 violations.

    This pins the FIX-WAVE-10 R10-1 fix: the 3 offending files
    (test_trainer_skeleton.py, test_categorical_substrate.py,
    test_anr_token_renderer.py) must NOT re-introduce the
    `from src.tac.*` pattern.
    """
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=repo_root, strict=True, verbose=False
    )
    assert violations == [], (
        f"Live repo has {len(violations)} `src.tac.*` import violation(s); "
        "the FIX-WAVE-10 R10-1 fix regressed."
    )


# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────


def test_pycache_is_excluded(tmp_path):
    """`__pycache__/` files are excluded from scan."""
    root = _make_test_corpus(tmp_path, {})
    pycache = root / "src" / "tac" / "tests" / "__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    # Even if a `.py` file appears in __pycache__ (rare but possible), skip.
    (pycache / "test_cached.py").write_text(
        "from src.tac.foo import Bar\n", encoding="utf-8"
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_nested_test_subdir_is_scanned(tmp_path):
    """Recursive scan: nested subdirs under `src/tac/tests/` are scanned."""
    root = _make_test_corpus(
        tmp_path,
        {"src/tac/tests/nested/sub/test_bad.py": "from src.tac.foo import A\n"},
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert len(violations) == 1


def test_non_py_files_ignored(tmp_path):
    """Non-`.py` files are not scanned (e.g. .md, .yaml)."""
    root = _make_test_corpus(tmp_path, {})
    (root / "src" / "tac" / "tests" / "README.md").write_text(
        "from src.tac.foo import Bar\n", encoding="utf-8"
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []


def test_outside_test_dirs_is_ignored(tmp_path):
    """Files OUTSIDE `src/tac/tests/` + `tests/` are NOT scanned.

    The gate is scoped to test corpus only — `src/tac/foo.py` legitimately
    has no `src.tac.*` import (it would be circular), but the gate must
    not extend its enforcement scope to production code surfaces.
    """
    root = _make_test_corpus(tmp_path, {})
    (root / "src" / "tac" / "foo.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "src" / "tac" / "foo.py").write_text(
        "from src.tac.bar import Baz  # would be circular but not our scope\n",
        encoding="utf-8",
    )
    violations = check_test_imports_use_tac_not_src_tac(
        repo_root=root, verbose=False
    )
    assert violations == []
