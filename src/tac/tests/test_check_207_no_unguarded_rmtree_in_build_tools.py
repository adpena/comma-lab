# SPDX-License-Identifier: MIT
"""Tests for Catalog #207 — no unguarded ``shutil.rmtree`` in build tools.

Empirical anchor 2026-05-14: codex finding HIGH —
``tools/build_pr101_nonlocal_sweep_packets.py`` lines 712-714 piped a
user-controlled ``--out-dir`` directly into ``shutil.rmtree`` when
``--force`` was set. A typo such as ``--out-dir . --force`` would have
recursively deleted the entire repo.

This gate refuses any ``tools/build_*.py`` / ``tools/promote_*.py`` whose
function body BOTH (a) references a ``--force``-style flag AND (b) calls
``shutil.rmtree`` without a recognized guard token (``_assert_rmtree_safe``,
``rmtree_within_namespace``, ``ensure_within_namespace``, etc.) in the same
function body. Same-line waiver: ``# RMTREE_UNGUARDED_OK:<reason>``.

Sister of Catalog #154 (`check_experiments_results_gc_helper_is_canonical`)
and Catalog #156 (`check_gc_helper_refuses_delete_on_tracked_paths`) —
together they extinct the user-controlled-recursive-deletion bug class
across the canonical GC surface, the public release prep surface, and the
build-tool surface.

Memory: feedback_codex_3_findings_fix_landed_20260514.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_207_iter_build_tool_files,
    check_no_unguarded_rmtree_in_build_tools,
)

# ─── Fixture: synthetic tools/ directory ────────────────────────────────


@pytest.fixture
def fake_tools_root(tmp_path):
    """Build a fake repo with a tools/ subdir."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    return repo


def _write_tool(repo: Path, name: str, body: str) -> Path:
    p = repo / "tools" / name
    p.write_text(body)
    return p


# ─── Positive: unguarded rmtree gated by --force is flagged ──────────────


def test_unguarded_rmtree_with_force_flag_is_flagged(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_unguarded.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "build_unguarded.py" in v[0]
    assert "main" in v[0]


def test_unguarded_promote_tool_also_flagged(fake_tools_root):
    body = (
        "import shutil\n"
        "def go(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.target_dir)\n"
    )
    _write_tool(fake_tools_root, "promote_widget.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 1
    assert "promote_widget.py" in v[0]


def test_unguarded_materialize_tool_also_flagged(fake_tools_root):
    body = (
        "import shutil\n"
        "def go(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.output_cache_dir)\n"
    )
    _write_tool(fake_tools_root, "materialize_bad.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 1
    assert "materialize_bad.py" in v[0]


# ─── Negative: rmtree without --force is OUT OF SCOPE ──────────────────


def test_rmtree_without_force_flag_is_out_of_scope(fake_tools_root):
    """If the function doesn't reference args.force, the bug class signature
    is not present and the gate doesn't fire (avoid false positives on
    legitimate tmp-dir cleanup)."""
    body = (
        "import shutil\n"
        "def cleanup_tmp_after_failure(tmp_dir):\n"
        "    shutil.rmtree(tmp_dir)\n"
    )
    _write_tool(fake_tools_root, "build_no_force.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


# ─── Negative: guard token in body satisfies ──────────────────────────


def test_rmtree_with_assert_safe_guard_passes(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        _assert_rmtree_safe(args.out_dir)\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_safe.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


def test_rmtree_with_namespace_guard_passes(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        rmtree_within_namespace(args.out_dir, allowed='experiments/results')\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_namespace.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


# ─── Same-line waiver ─────────────────────────────────────────────────


def test_same_line_waiver_with_reason_accepted(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)  # RMTREE_UNGUARDED_OK:legitimate cleanup of self-created tmp\n"
    )
    _write_tool(fake_tools_root, "build_waiver.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


def test_same_line_waiver_without_reason_REJECTED(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)  # RMTREE_UNGUARDED_OK:\n"
    )
    _write_tool(fake_tools_root, "build_naked.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 1


def test_placeholder_reason_rejected(fake_tools_root):
    """The literal `<reason>` placeholder cannot self-waive."""
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)  # RMTREE_UNGUARDED_OK:<reason>\n"
    )
    _write_tool(fake_tools_root, "build_placeholder.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 1


# ─── Strict mode raises ─────────────────────────────────────────────


def test_strict_mode_raises_preflight_error(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_dirty_strict.py", body)
    with pytest.raises(PreflightError, match=r"unguarded shutil\.rmtree"):
        check_no_unguarded_rmtree_in_build_tools(
            repo_root=fake_tools_root, strict=True
        )


def test_strict_mode_passes_when_clean(fake_tools_root):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        _assert_rmtree_safe(args.out_dir)\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_clean_strict.py", body)
    # Should NOT raise.
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=True
    )
    assert v == []


# ─── AST helper edge cases ──────────────────────────────────────────


def test_iter_build_tool_files_filters_correctly(fake_tools_root):
    _write_tool(fake_tools_root, "build_a.py", "import shutil\n")
    _write_tool(fake_tools_root, "materialize_a.py", "import shutil\n")
    _write_tool(fake_tools_root, "promote_b.py", "import shutil\n")
    _write_tool(fake_tools_root, "make_c.py", "import shutil\n")  # NOT matched
    _write_tool(fake_tools_root, "build_a.txt", "")  # NOT matched
    files = _check_207_iter_build_tool_files(fake_tools_root / "tools")
    rel = sorted(f.name for f in files)
    assert "build_a.py" in rel
    assert "materialize_a.py" in rel
    assert "promote_b.py" in rel
    assert "make_c.py" not in rel
    assert "build_a.txt" not in rel


def test_iter_build_tool_files_returns_empty_for_missing_dir(tmp_path):
    files = _check_207_iter_build_tool_files(tmp_path / "no_such_dir")
    assert files == []


def test_collect_violations_handles_syntax_error(fake_tools_root):
    """A malformed .py file should produce no violations (skip cleanly)."""
    _write_tool(fake_tools_root, "build_bad.py", "def x(\nthis is not python\n")
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


def test_collect_violations_recognizes_bare_rmtree_import(fake_tools_root):
    """from shutil import rmtree; rmtree(args.out_dir) is also flagged."""
    body = (
        "from shutil import rmtree\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_bare_import.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 1


# ─── Canonical helper allowlist ────────────────────────────────────


def test_canonical_helper_file_is_exempt(fake_tools_root):
    """The canonical helper file (which DEFINES the guard) is exempt."""
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)\n"  # raw call inside the helper
    )
    # Use the actual allowlisted name.
    _write_tool(
        fake_tools_root,
        "build_pr101_nonlocal_sweep_packets.py",
        body,
    )
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert v == []


# ─── Multiple violations in one file ───────────────────────────────


def test_multiple_violations_in_one_file_all_flagged(fake_tools_root):
    body = (
        "import shutil\n"
        "def go_one(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.dir1)\n"
        "def go_two(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.dir2)\n"
    )
    _write_tool(fake_tools_root, "build_multi.py", body)
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False
    )
    assert len(v) == 2
    funcs = sorted(s.split("function ", 1)[1].split("'", 2)[1] for s in v)
    assert funcs == ["go_one", "go_two"]


# ─── Verbose output ────────────────────────────────────────────────


def test_verbose_output_on_clean(fake_tools_root, capsys):
    check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "no-unguarded-rmtree-in-build-tools" in out
    assert "OK" in out


def test_verbose_output_on_dirty(fake_tools_root, capsys):
    body = (
        "import shutil\n"
        "def main(args):\n"
        "    if args.force:\n"
        "        shutil.rmtree(args.out_dir)\n"
    )
    _write_tool(fake_tools_root, "build_dirty.py", body)
    check_no_unguarded_rmtree_in_build_tools(
        repo_root=fake_tools_root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "WARN" in out


# ─── String repo_root accepted ────────────────────────────────────


def test_string_repo_root_accepted(fake_tools_root):
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=str(fake_tools_root), strict=False
    )
    assert v == []


# ─── Live-repo regression guard ──────────────────────────────────


def test_live_repo_count_is_zero():
    """Strict-flip atomicity: the live count in the actual repo MUST be 0."""
    from tac.preflight import REPO_ROOT
    v = check_no_unguarded_rmtree_in_build_tools(
        repo_root=REPO_ROOT, strict=False
    )
    assert v == [], (
        f"Live violation count in REPO_ROOT = {len(v)} (expected 0). "
        f"First 3:\n  " + "\n  ".join(v[:3])
    )


def test_orchestrator_callsite_is_strict_true():
    """The strict-flip MUST be wired in preflight_all() per Catalog #176."""
    import inspect

    from tac import preflight
    src = inspect.getsource(preflight.preflight_all)
    assert "check_no_unguarded_rmtree_in_build_tools" in src
    idx = src.index("check_no_unguarded_rmtree_in_build_tools")
    window = src[idx:idx + 600]
    assert "strict=True" in window, (
        "Expected strict=True wire-in for "
        "check_no_unguarded_rmtree_in_build_tools per Catalog #207"
    )
