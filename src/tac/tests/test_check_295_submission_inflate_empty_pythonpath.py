# SPDX-License-Identifier: MIT
"""Catalog #295 - check_submission_inflate_works_with_empty_pythonpath tests.

Empirical anchor 2026-05-16: NSCS06 v5 Modal dispatch
``fc-01KRQMAQ7V41AFYMJH5HRK9P10`` failed at runtime because
``submissions/nscs06_carmack_hotz_strip_everything/inflate.py`` used the
PYTHONPATH-shim pattern ``sys.path.insert(0, str(HERE / "src"))`` then
``from tac.substrates...`` which worked during local development (operator's
working tree had the repo ``src/`` resolvable on sys.path) but FAILED on
the Modal worker because the ``src/tac/...`` tree was never vendored into
the submission directory. The v6 fix (commit ``90bca47ff``) vendored the
codec package alongside as ``submissions/<id>/_nscs06_codec/`` and removed
the PYTHONPATH shim.

This test file exercises:

* live-repo regression guard (live count bounded to current baseline 6)
* positive detection of bare ``from tac....`` import with no vendoring
  AND no sys.path.insert (NSCS06 v5 bug class via implicit working-tree
  dependency)
* positive detection of sys.path.insert + ``from tac....`` import where
  the ``tac`` package is NOT vendored alongside (the canonical bug class)
* positive detection of sys.path.insert pointing OUTSIDE submission
  directory (sibling-submission dependency or ``HERE.parent / ...``)
* negative acceptance: no sys.path.insert + no ``from tac.*`` imports
  (clean simple inflate)
* negative acceptance: sys.path.insert(HERE / "src") + no tac imports
  (NSCS06-v6-style vendored non-tac codec alongside)
* negative acceptance: sys.path.insert + ``from tac.*`` AND ``tac``
  package vendored alongside as ``<submission_dir>/src/tac/__init__.py``
* negative acceptance: sys.path.insert + ``from tac.*`` AND ``tac``
  package vendored as ``<submission_dir>/tac/__init__.py``
* same-line ``# SUBMISSION_PYTHONPATH_SHIM_OK:<rationale>`` waiver
  acceptance
* placeholder ``<rationale>`` / ``<reason>`` literal rejection
* one-level variable indirection (``SRC_DIR = HERE / "src"`` accepted)
* multi-level parent traversal rejected (``HERE.parent / ...``)
* ``submissions/exact_current/`` is exempt (pinned upstream snapshot)
* ``submissions/foo_intake_bar/`` is exempt (vendored intake clone)
* empty repo / no submissions/ dir returns 0 violations
* malformed Python tolerated (no raise)
* strict mode raises with Catalog #295 message
* docstring/comment mentions of the literal pattern are NOT flagged
  (AST-aware, not regex)
* orchestrator wire-in regression guard (warn-only at landing)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_submission_inflate_works_with_empty_pythonpath,
)


# ---------------------------------------------------------------------------
# Synthetic-fixture builders
# ---------------------------------------------------------------------------


def _make_submission(
    tmp_root: Path,
    name: str,
    inflate_body: str,
    vendored_dirs: dict[str, str] | None = None,
) -> Path:
    """Create a synthetic submission under ``tmp_root/submissions/<name>/``.

    ``vendored_dirs`` is a mapping of relative-path-to-contents — used to
    drop ``src/tac/__init__.py`` or sibling vendored packages alongside.
    """
    sub_dir = tmp_root / "submissions" / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "inflate.py").write_text(inflate_body)
    if vendored_dirs:
        for rel, contents in vendored_dirs.items():
            full = sub_dir / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(contents)
    return sub_dir


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_count_bounded():
    """At landing the live repo has 6 known violations (apogee_v2 / magic_codec /
    nscs01 / pr106_lrl1 / pr106_stacked / pr106_yshift). Strict-flip pending
    sister-subagent backfill. Ceiling is 20 to absorb in-flight changes."""
    violations = check_submission_inflate_works_with_empty_pythonpath(
        strict=False, verbose=False,
    )
    assert isinstance(violations, list)
    assert len(violations) <= 20, (
        f"live-repo regression: Catalog #295 count grew to {len(violations)}; "
        f"first violations: {violations[:3]}"
    )


# ---------------------------------------------------------------------------
# Positive detections (the bug class)
# ---------------------------------------------------------------------------


def test_bare_tac_import_without_sys_path_insert_flagged(tmp_path: Path):
    """`from tac.*` with no sys.path manipulation = silent dep on working tree."""
    _make_submission(
        tmp_path,
        "bare_tac",
        '"""bare-tac"""\nfrom tac.foo import bar  # noqa: E402\n',
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 1
    assert "bare_tac" in violations[0]
    assert "WITHOUT any vendored" in violations[0]


def test_sys_path_insert_plus_tac_import_no_vendor_flagged(tmp_path: Path):
    """sys.path.insert + from tac.* + no vendored tac/ = NSCS06 v5 bug class."""
    _make_submission(
        tmp_path,
        "nscs06_v5_repro",
        '"""nscs06-v5"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(HERE / "src"))\n'
        "from tac.substrates.foo.inflate import main_cli  # noqa: E402\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 1
    assert "nscs06_v5_repro" in violations[0]
    assert "NOT vendored" in violations[0]
    assert "NSCS06 v5 bug class" in violations[0]


def test_sys_path_insert_parent_traversal_flagged(tmp_path: Path):
    """sys.path.insert(HERE.parent / sibling / src) = depends on sibling shipping."""
    _make_submission(
        tmp_path,
        "sister_dep",
        '"""sister-dep"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SIBLING_SRC = HERE.parent / "apogee_intN" / "src"\n'
        "sys.path.insert(0, str(SIBLING_SRC))\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 1
    assert "OUTSIDE the submission" in violations[0]


# ---------------------------------------------------------------------------
# Negative acceptances (canonical patterns)
# ---------------------------------------------------------------------------


def test_no_sys_path_no_tac_import_passes(tmp_path: Path):
    """Simple inflate with no path manipulation + no tac imports = OK."""
    _make_submission(
        tmp_path,
        "simple",
        '"""simple"""\nimport os\nimport sys\nfrom pathlib import Path\n',
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_nscs06_v6_pattern_vendored_alongside_passes(tmp_path: Path):
    """NSCS06 v6 canonical pattern: sys.path.insert(0, str(HERE)) + sibling pkg."""
    _make_submission(
        tmp_path,
        "nscs06_v6",
        '"""nscs06-v6"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        "if str(HERE) not in sys.path:\n"
        "    sys.path.insert(0, str(HERE))\n"
        "from _nscs06_codec.inflate import inflate_one_video  # noqa: E402\n",
        vendored_dirs={"_nscs06_codec/__init__.py": ""},
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_src_dir_vendored_codec_no_tac_passes(tmp_path: Path):
    """PR106-style: sys.path.insert(0, str(SRC_DIR)) + vendored codec.py + model.py."""
    _make_submission(
        tmp_path,
        "pr106_style",
        '"""pr106-style"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SRC_DIR = HERE / "src"\n'
        "sys.path.insert(0, str(SRC_DIR))\n"
        "from codec import parse_packed_archive  # noqa: E402\n",
        vendored_dirs={"src/codec.py": "def parse_packed_archive(): pass\n"},
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_tac_vendored_in_src_subdir_passes(tmp_path: Path):
    """sys.path.insert(0, str(HERE/'src')) + from tac.* + vendored src/tac/__init__.py."""
    _make_submission(
        tmp_path,
        "tac_vendored",
        '"""tac-vendored"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(HERE / "src"))\n'
        "from tac.foo import bar  # noqa: E402\n",
        vendored_dirs={
            "src/tac/__init__.py": "",
            "src/tac/foo.py": "def bar(): pass\n",
        },
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_tac_vendored_directly_under_submission_passes(tmp_path: Path):
    """sys.path.insert(0, str(HERE)) + from tac.* + vendored tac/__init__.py."""
    _make_submission(
        tmp_path,
        "tac_direct",
        '"""tac-direct"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE))\n"
        "from tac.foo import bar  # noqa: E402\n",
        vendored_dirs={
            "tac/__init__.py": "",
            "tac/foo.py": "def bar(): pass\n",
        },
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_one_level_indirection_src_dir_assignment_passes(tmp_path: Path):
    """SRC_DIR = HERE / "src" then sys.path.insert(0, str(SRC_DIR)) accepted."""
    _make_submission(
        tmp_path,
        "indirect_src",
        '"""indirect"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SRC_DIR = HERE / "src"\n'
        "sys.path.insert(0, str(SRC_DIR))\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_same_line_waiver_with_rationale_accepted(tmp_path: Path):
    """# SUBMISSION_PYTHONPATH_SHIM_OK:<real rationale> waives the violation."""
    _make_submission(
        tmp_path,
        "waived",
        '"""waived"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SIBLING = HERE.parent / "apogee_intN" / "src"\n'
        'sys.path.insert(0, str(SIBLING))  # SUBMISSION_PYTHONPATH_SHIM_OK:sister-sub ships together with apogee_intN per dispatch packet manifest\n',
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_placeholder_rationale_rejected(tmp_path: Path):
    """# SUBMISSION_PYTHONPATH_SHIM_OK:<rationale> placeholder literal rejected."""
    _make_submission(
        tmp_path,
        "placeholder",
        '"""placeholder"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SIBLING = HERE.parent / "apogee_intN" / "src"\n'
        'sys.path.insert(0, str(SIBLING))  # SUBMISSION_PYTHONPATH_SHIM_OK:<rationale>\n',
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 1


def test_placeholder_reason_rejected(tmp_path: Path):
    """# SUBMISSION_PYTHONPATH_SHIM_OK:<reason> placeholder also rejected."""
    _make_submission(
        tmp_path,
        "placeholder2",
        '"""placeholder2"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'SIBLING = HERE.parent / "apogee_intN" / "src"\n'
        'sys.path.insert(0, str(SIBLING))  # SUBMISSION_PYTHONPATH_SHIM_OK:<reason>\n',
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Scope / exemption semantics
# ---------------------------------------------------------------------------


def test_exact_current_excluded(tmp_path: Path):
    """submissions/exact_current/ is the pinned upstream snapshot - exempt."""
    _make_submission(
        tmp_path,
        "exact_current",
        "from tac.foo import bar\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_intake_clones_excluded(tmp_path: Path):
    """submissions/foo_intake_bar/ vendored intake clones are exempt."""
    _make_submission(
        tmp_path,
        "public_pr_intake_codex",
        "from tac.foo import bar\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_no_submissions_dir_passes(tmp_path: Path):
    """Empty repo / no submissions/ dir returns 0 violations silently."""
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_malformed_python_tolerated(tmp_path: Path):
    """SyntaxError in inflate.py is silently skipped (per sister gate convention)."""
    _make_submission(
        tmp_path,
        "broken",
        "this is not valid python @@@\nfrom tac\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


def test_docstring_mention_not_flagged(tmp_path: Path):
    """AST-aware: docstring mentioning the pattern is NOT a real import."""
    _make_submission(
        tmp_path,
        "docstring_only",
        '"""\n'
        "This docstring mentions sys.path.insert(0, str(HERE)) and\n"
        "from tac.substrates.foo import bar but neither is a real import.\n"
        '"""\n'
        "import os\n",
    )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict-mode behavior
# ---------------------------------------------------------------------------


def test_strict_mode_raises_with_catalog_295_message(tmp_path: Path):
    """strict=True raises PreflightError citing Catalog #295."""
    _make_submission(
        tmp_path,
        "strict_test",
        '"""s"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(HERE / "src"))\n'
        "from tac.foo import bar\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_submission_inflate_works_with_empty_pythonpath(
            repo_root=tmp_path, strict=True,
        )
    msg = str(exc_info.value)
    assert "check_submission_inflate_works_with_empty_pythonpath" in msg
    assert "NSCS06 v5" in msg


def test_strict_mode_silent_on_clean_repo(tmp_path: Path):
    """strict=True returns [] without raising when no violations."""
    _make_submission(
        tmp_path,
        "clean",
        "from pathlib import Path\n",
    )
    out = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=True,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Multi-file aggregation
# ---------------------------------------------------------------------------


def test_multiple_submissions_aggregated(tmp_path: Path):
    """Multiple offending submissions all reported in one list."""
    for name in ("sub_a", "sub_b", "sub_c"):
        _make_submission(
            tmp_path,
            name,
            '"""x"""\nfrom tac.foo import bar  # noqa: E402\n',
        )
    violations = check_submission_inflate_works_with_empty_pythonpath(
        repo_root=tmp_path, strict=False,
    )
    assert len(violations) == 3
    flagged_names = {v.split("/")[1] for v in violations}
    assert flagged_names == {"sub_a", "sub_b", "sub_c"}


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_callsite_wires_warn_only():
    """preflight_all() must wire Catalog #295 with strict=False at landing."""
    import inspect

    from tac.preflight import preflight_all

    src = inspect.getsource(preflight_all)
    assert "check_submission_inflate_works_with_empty_pythonpath" in src, (
        "Catalog #295 must be wired into preflight_all()"
    )
    # Find the call and verify strict=False (warn-only at landing).
    lines = src.split("\n")
    found_call = False
    for i, line in enumerate(lines):
        if "check_submission_inflate_works_with_empty_pythonpath(" in line:
            # Check next 3 lines for strict= argument.
            window = "\n".join(lines[i : i + 4])
            assert "strict=False" in window, (
                "Catalog #295 wire-in must be strict=False at landing "
                "per CLAUDE.md 'Strict-flip atomicity rule' (live count > 0)"
            )
            found_call = True
            break
    assert found_call, "Could not locate Catalog #295 call site in preflight_all"


# ---------------------------------------------------------------------------
# Live regression: known affected substrates remain flagged
# ---------------------------------------------------------------------------


def test_nscs01_remains_flagged_until_vendored():
    """nscs01_nullspace_split_renderer is the canonical NSCS06-v5-class anchor.
    Until the trainer vendors the codec package alongside, this MUST stay flagged.
    """
    violations = check_submission_inflate_works_with_empty_pythonpath(
        strict=False, verbose=False,
    )
    nscs01_violations = [v for v in violations if "nscs01" in v]
    assert len(nscs01_violations) >= 1, (
        "nscs01_nullspace_split_renderer must remain flagged until the trainer "
        "vendors `tac.substrates.nscs01_nullspace_split_renderer` package "
        "alongside as `submissions/<id>/src/tac/...` "
        "(NSCS06 v6 pattern @ commit 90bca47ff)"
    )
