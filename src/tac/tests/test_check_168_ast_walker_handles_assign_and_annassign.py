"""Tests for Catalog #168 ``check_ast_walker_handles_both_assign_and_annassign``.

META-CLASS gate that refuses any AST extractor function that filters by
``ast.Assign`` ONLY without also handling ``ast.AnnAssign``. The bug class
silently bypasses static analysis when the target code uses annotated
assignment syntax (``name: type = value``).

Bug-class anchor: 2026-05-12 META-CATALOG-152-FIX. The
``_check_151_extract_tier_manifests`` AST walker filtered ``ast.Assign``
only. Substrate trainers (sane_hnerv, balle, SIREN, Cool-Chic, VQ-VAE,
hybrid_*, self_compress, TCNeRV, BlockNeRV, FFNeRV, DSNeRV, HiNeRV) declare
the manifest as ``TIER_<N>_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]]
= {...}`` (``ast.AnnAssign``), so 12/12 substrate trainers silently returned
empty manifests -- making Catalog #151 + #152 STRICT modes structurally
false-OK across the entire substrate canvas.

Coverage:
- positive (Assign-only walk -> violation)
- positive (Assign-only with AugAssign sister but no AnnAssign -> violation)
- positive (Assign-only on attribute target -> violation)
- negative (dual handling via tuple -> clean)
- negative (dual handling via tuple in either order -> clean)
- negative (if/elif chain with sister AnnAssign branch -> clean)
- waiver (same-line ASSIGN_ONLY_OK with rationale -> clean)
- waiver (preceding-line ASSIGN_ONLY_OK with rationale -> clean)
- waiver (file-level CHECK_168_FILE_LEVEL_WAIVED with rationale -> clean)
- waiver placeholder reject (``<reason>`` literal does NOT waive)
- exempt path: own test file is exempt
- excluded path markers (experiments/results/, _intake_, .omx/oss_export/, vendored)
- non-isinstance comparisons are ignored
- isinstance with non-ast.Module references is ignored
- live repo state is 0 violations (regression guard)
- strict=True raises PreflightError on violations
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_ast_walker_handles_both_assign_and_annassign,
)


def _make_repo(tmp_path: Path) -> Path:
    """Create a fake repo with src/tac, tools, experiments dirs."""

    root = tmp_path / "fakerepo"
    (root / "src/tac").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / "experiments").mkdir(parents=True)
    return root


# ---------------------------------------------------------------------------
# Positive cases: Assign-only walks should be flagged
# ---------------------------------------------------------------------------


def test_bare_assign_only_walk_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/buggy.py").write_text(
        "import ast\n"
        "def extract(tree):\n"
        "    out = []\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.Assign):\n"
        "            out.append(node)\n"
        "    return out\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "src/tac/buggy.py" in violations[0]
    assert "ast.Assign" in violations[0]


def test_assign_with_augassign_sister_still_violation(tmp_path: Path) -> None:
    """AugAssign is `x += y`, NOT annotated assign. Sister branch with
    AugAssign does not satisfy AnnAssign coverage."""
    root = _make_repo(tmp_path)
    (root / "src/tac/buggy.py").write_text(
        "import ast\n"
        "def extract(tree):\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.Assign):\n"
        "            handle_assign(node)\n"
        "        elif isinstance(node, ast.AugAssign):\n"
        "            handle_augassign(node)\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_assign_only_on_attribute_target_violation(tmp_path: Path) -> None:
    """Even when the target is `args.X`, missing AnnAssign coverage misses
    `args.X: type = value`."""
    root = _make_repo(tmp_path)
    (root / "tools/buggy_attr.py").write_text(
        "import ast\n"
        "def collect(tree):\n"
        "    out = set()\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.Assign):\n"
        "            out.add(node.targets[0])\n"
        "    return out\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_assign_in_tuple_alone_is_violation(tmp_path: Path) -> None:
    """isinstance(x, (ast.Assign,)) -- single-element tuple still missing AnnAssign."""
    root = _make_repo(tmp_path)
    (root / "experiments/buggy_tuple.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, (ast.Assign,)):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Negative cases: correct dual handling should NOT be flagged
# ---------------------------------------------------------------------------


def test_dual_handling_via_tuple_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/correct.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, (ast.Assign, ast.AnnAssign)):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_dual_handling_reverse_order_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/correct_rev.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, (ast.AnnAssign, ast.Assign)):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_if_elif_chain_with_annassign_branch_clean(tmp_path: Path) -> None:
    """The common if/elif chain pattern where each AST node-kind is handled
    in its own branch should be accepted by the scope-aware filter."""
    root = _make_repo(tmp_path)
    (root / "src/tac/chain.py").write_text(
        "import ast\n"
        "def collect(tree):\n"
        "    out = set()\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.Assign):\n"
        "            out.add('a')\n"
        "        elif isinstance(node, ast.AnnAssign):\n"
        "            out.add('b')\n"
        "    return out\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver acceptance
# ---------------------------------------------------------------------------


def test_same_line_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/waived.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):  # ASSIGN_ONLY_OK:Annotated form impossible here\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_preceding_line_waiver_accepted(tmp_path: Path) -> None:
    """Multi-line isinstance() callsite indent style places waiver on the
    line above the isinstance line."""
    root = _make_repo(tmp_path)
    (root / "src/tac/waived_above.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    # ASSIGN_ONLY_OK:vendored upstream snapshot uses bare Assign\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_file_level_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/file_waived.py").write_text(
        "# CHECK_168_FILE_LEVEL_WAIVED:test fixture for the gate itself\n"
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_placeholder_reason_does_NOT_waive(tmp_path: Path) -> None:
    """A bare `# ASSIGN_ONLY_OK:<reason>` placeholder must NOT silently waive
    -- it must carry a real rationale or the gate fires."""
    root = _make_repo(tmp_path)
    (root / "src/tac/placeholder.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):  # ASSIGN_ONLY_OK:<reason>\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_file_level_placeholder_does_NOT_waive(tmp_path: Path) -> None:
    """File-level placeholder rejected for the same reason."""
    root = _make_repo(tmp_path)
    (root / "src/tac/file_placeholder.py").write_text(
        "# CHECK_168_FILE_LEVEL_WAIVED:<reason>\n"
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Path-based exclusions
# ---------------------------------------------------------------------------


def test_results_dir_excluded(tmp_path: Path) -> None:
    """experiments/results/ is DERIVED_OUTPUT (Catalog #113), out-of-scope."""
    root = _make_repo(tmp_path)
    (root / "experiments/results").mkdir(parents=True)
    (root / "experiments/results/buggy.py").write_text(
        "import ast\n"
        "if isinstance(x, ast.Assign): pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_intake_marker_excluded(tmp_path: Path) -> None:
    """Vendored PR intake clones are out-of-scope."""
    root = _make_repo(tmp_path)
    (root / "experiments/public_pr_intake_pr105").mkdir(parents=True)
    (root / "experiments/public_pr_intake_pr105/source.py").write_text(
        "import ast\n"
        "if isinstance(x, ast.Assign): pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_oss_export_mirror_excluded(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / ".omx/oss_export").mkdir(parents=True)
    # .omx is outside the scan dirs anyway, but ensure the excluded marker
    # logic still works for paths that would otherwise be scanned.
    (root / "experiments/.omx/oss_export").mkdir(parents=True)
    (root / "experiments/.omx/oss_export/buggy.py").write_text(
        "import ast\n"
        "if isinstance(x, ast.Assign): pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Non-target patterns are ignored
# ---------------------------------------------------------------------------


def test_non_isinstance_comparison_ignored(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/non_isinstance.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if type(node) == ast.Assign:\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_isinstance_without_ast_assign_ignored(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/other.py").write_text(
        "import ast\n"
        "def f(x):\n"
        "    if isinstance(x, str):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_isinstance_with_other_ast_kinds_ignored(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/other_ast.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict mode and live-repo regression guard
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/raises.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    with pytest.raises(PreflightError) as exc:
        check_ast_walker_handles_both_assign_and_annassign(
            repo_root=root, strict=True
        )
    assert "Catalog #168" in str(exc.value)


def test_strict_mode_passes_on_clean_repo(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/clean.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, (ast.Assign, ast.AnnAssign)):\n"
        "        pass\n"
    )
    # Should not raise.
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=True
    )
    assert violations == []


def test_live_repo_regression_zero_violations() -> None:
    """Regression guard. The post-fix repo state must remain at 0 violations.

    META-CATALOG-152-FIX 2026-05-12 fixed 12 candidate sites. If anyone
    adds a new Assign-only walk (or a sister gate doesn't catch it),
    this regression test will fire.
    """
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=repo_root, strict=False
    )
    assert violations == [], (
        f"Catalog #168 regression: {len(violations)} new Assign-only AST "
        f"walker site(s):\n  " + "\n  ".join(v[:240] for v in violations[:5])
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_repo_no_violations(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_syntax_error_skipped_silently(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "src/tac/broken.py").write_text(
        "import ast\n"
        "def f(:\n"  # syntax error
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert violations == []


def test_tools_dir_scanned(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "tools/in_tools.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "tools/in_tools.py" in violations[0]


def test_source_index_candidate_prefilter_preserves_detection(
    tmp_path: Path,
) -> None:
    """The fast SourceIndex path must still catch Assign-only walkers."""
    root = _make_repo(tmp_path)
    (root / "src/tac/indexed_bug.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )

    from tac.source_index import source_index_context

    with source_index_context(root):
        violations = check_ast_walker_handles_both_assign_and_annassign(
            repo_root=root,
            strict=False,
        )

    assert len(violations) == 1
    assert "src/tac/indexed_bug.py" in violations[0]


def test_source_index_candidate_prefilter_skips_rg(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SourceIndex-backed runs should not pay a redundant ripgrep subprocess."""
    root = _make_repo(tmp_path)
    (root / "src/tac/indexed_bug.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )

    def fail_rg(*args, **kwargs):
        raise AssertionError("SourceIndex path must not shell out to rg")

    monkeypatch.setattr("tac.preflight._rg_python_files_matching_regex", fail_rg)

    from tac.source_index import source_index_context

    with source_index_context(root):
        violations = check_ast_walker_handles_both_assign_and_annassign(
            repo_root=root,
            strict=False,
        )

    assert len(violations) == 1
    assert "src/tac/indexed_bug.py" in violations[0]


def test_experiments_dir_scanned(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    (root / "experiments/in_exp.py").write_text(
        "import ast\n"
        "def f(node):\n"
        "    if isinstance(node, ast.Assign):\n"
        "        pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_module_level_assign_only_with_annassign_elsewhere_clean(
    tmp_path: Path,
) -> None:
    """If a module has separate functions that handle Assign and AnnAssign
    DIFFERENTLY (different scopes), the scope-aware accept should NOT
    accept the Assign-only function from the OTHER function's AnnAssign
    handling. This proves the scope filter is per-function."""
    root = _make_repo(tmp_path)
    (root / "src/tac/separate_scopes.py").write_text(
        "import ast\n"
        "def buggy(tree):\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.Assign):\n"
        "            pass\n"
        "def correct(tree):\n"
        "    for node in ast.walk(tree):\n"
        "        if isinstance(node, ast.AnnAssign):\n"
        "            pass\n"
    )
    violations = check_ast_walker_handles_both_assign_and_annassign(
        repo_root=root, strict=False
    )
    # The buggy function should fire even though correct() handles AnnAssign
    # in a separate scope.
    assert len(violations) == 1
    assert "buggy" not in violations[0]  # only the line ref, not function name
    assert "src/tac/separate_scopes.py" in violations[0]
