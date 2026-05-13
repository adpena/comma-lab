"""Tests for Catalog #201 ``check_modal_sentinel_files_are_in_mount_set``.

STRICT preflight gate that refuses any string-literal sentinel path in
``tools/operator_authorize.py::_modal_sentinel_files`` that is outside the
canonical Modal mount set declared by
``src/tac/deploy/modal/mount_manifest.py::STRUCTURAL_MINIMUM_DIRS``
(``src/``, ``scripts/``, ``upstream/``, ``submissions/``, ``experiments/``,
``tools/``, or exactly ``pyproject.toml``).

Empirical anchor: PR95++ enhanced-curriculum Modal smoke
fc-01KRHNMT4SEB794HFPH5GNTHFP @ 2026-05-13 21:59:30 UTC returned rc=13
(``MISSING_WORKER``) because the recipe YAML at
``.omx/operator_authorize_recipes/<recipe>.yaml`` was appended to
``--sentinel-files``. Catalog #166's worker-side hash ledger refused the
dispatch because ``.omx/`` is NOT one of the canonical mount-set prefixes,
so the worker had no matching sentinel to hash.

Sister of Catalog #166 (worker-side hash check that surfaces the failure
rc=13) and Catalog #153 (canonical Modal mount builder that defines
``STRUCTURAL_MINIMUM_DIRS``).

Coverage:
- Positive: catches `.omx/` recipe YAML literals (the original bug class)
- Positive: catches `.ralph/`, `configs/`, `docs/`, `cuda/`, `runtime-rs/`
  literals (sister operator-side dirs that are equally unmounted)
- Negative: allows `experiments/`, `src/`, `scripts/`, `tools/`,
  `submissions/`, `upstream/` literals (canonical mount-set prefixes)
- Negative: allows `pyproject.toml` exact match
- Waiver: same-line ``# SENTINEL_OUTSIDE_MOUNT_OK:<reason>`` accepted; the
  literal placeholder ``<reason>`` is rejected (so the gate's own docstring
  cannot self-waive)
- Dynamic: ``raw_paths.append(recipe.remote_driver)`` /
  ``raw_paths.append(str(value))`` etc. are NOT flagged (only ast.Constant
  string literals are checked statically)
- Both AST shapes covered: seed-list ``raw_paths: list[str] = ["lit", ...]``
  AND ``raw_paths.append("lit")``
- Live-repo regression guard: actual ``tools/operator_authorize.py`` is
  clean (0 violations)
- Strict-mode raises ``PreflightError``; non-strict returns the list

The fixture rig writes a *fake* ``tools/operator_authorize.py`` under a
tmp_path repo root so we can exercise both positive and negative shapes
without touching the real file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_sentinel_files_are_in_mount_set,
)


def _make_repo(tmp_path: Path) -> Path:
    """Create a fake repo with the tools/operator_authorize.py target path."""
    root = tmp_path / "fakerepo"
    (root / "tools").mkdir(parents=True)
    return root


def _write_sentinel_function(root: Path, body: str) -> None:
    """Write a fake ``tools/operator_authorize.py`` containing a
    ``_modal_sentinel_files`` function whose body is ``body``.
    """
    text = (
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "class Recipe:\n"
        "    pass\n"
        "\n"
        "\n"
        "def _modal_sentinel_files(recipe):\n"
        f"{body}"
        "\n"
        "    return ','.join(raw_paths)\n"
    )
    (root / "tools" / "operator_authorize.py").write_text(text)


# ---------------------------------------------------------------------------
# Positive cases: literals outside the mount set are flagged
# ---------------------------------------------------------------------------


def test_omx_recipe_yaml_literal_is_flagged(tmp_path: Path) -> None:
    """The original 2026-05-13 bug class: appending the recipe YAML under
    `.omx/operator_authorize_recipes/`.
    """
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('.omx/operator_authorize_recipes/x.yaml')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert ".omx/operator_authorize_recipes/x.yaml" in violations[0]
    assert "tools/operator_authorize.py" in violations[0]


def test_ralph_literal_is_flagged(tmp_path: Path) -> None:
    """Sister operator-side dir `.ralph/` is also outside the mount set."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('.ralph/run_log.md')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert ".ralph/run_log.md" in violations[0]


def test_configs_literal_is_flagged(tmp_path: Path) -> None:
    """`configs/` is allowed-to-edit but not Modal-mounted."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('configs/some.yaml')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "configs/some.yaml" in violations[0]


def test_docs_literal_is_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('docs/paper/manifest.json')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "docs/paper/manifest.json" in violations[0]


def test_cuda_literal_is_flagged(tmp_path: Path) -> None:
    """`cuda/` is in the mutation frontier but not in STRUCTURAL_MINIMUM_DIRS."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('cuda/kernel.cu')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "cuda/kernel.cu" in violations[0]


def test_runtime_rs_literal_is_flagged(tmp_path: Path) -> None:
    """`runtime-rs/` is in the mutation frontier but not Modal-mounted."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('runtime-rs/Cargo.toml')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "runtime-rs/Cargo.toml" in violations[0]


def test_seed_list_literal_outside_mount_set_is_flagged(tmp_path: Path) -> None:
    """Check the AST-Assign seed-list shape: ``raw_paths = ['x']``."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = ['.omx/state/foo.json']\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert ".omx/state/foo.json" in violations[0]


def test_annotated_seed_list_literal_outside_mount_set_is_flagged(
    tmp_path: Path,
) -> None:
    """Check the AST-AnnAssign seed-list shape: ``raw_paths: list[str] = [...]``.
    Per Catalog #168, both Assign and AnnAssign must be covered.
    """
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths: list[str] = ['.omx/state/foo.json']\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert ".omx/state/foo.json" in violations[0]


def test_multiple_violations_in_seed_list(tmp_path: Path) -> None:
    """All flagged literals appear in the violations list."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths: list[str] = [\n"
        "        '.omx/a.yaml',\n"
        "        'docs/b.md',\n"
        "        'configs/c.yaml',\n"
        "    ]\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 3


def test_mixed_seed_list_and_append_literals(tmp_path: Path) -> None:
    """Seed-list + append shapes are both scanned in one pass."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths: list[str] = ['.omx/recipe.yaml']\n"
        "    raw_paths.append('docs/notes.md')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 2


# ---------------------------------------------------------------------------
# Negative cases: literals under the mount set are allowed
# ---------------------------------------------------------------------------


def test_experiments_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths: list[str] = ['experiments/modal_train_lane.py']\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_src_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths: list[str] = ['src/tac/deploy/modal/mount_manifest.py']\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_scripts_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('scripts/remote_archive_only_eval.sh')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_tools_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('tools/operator_authorize.py')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_submissions_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('submissions/robust_current/inflate.py')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_upstream_literal_is_allowed(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('upstream/evaluate.py')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_pyproject_toml_exact_match_is_allowed(tmp_path: Path) -> None:
    """`pyproject.toml` is an exact-match exception (not a prefix)."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('pyproject.toml')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_pyproject_substring_does_not_match_prefix(tmp_path: Path) -> None:
    """A made-up file `pyproject.toml.bak` is NOT under the mount set
    (the exact-match rule must not silently accept prefix-substrings)."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('pyproject.toml.bak')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1
    assert "pyproject.toml.bak" in violations[0]


# ---------------------------------------------------------------------------
# Dynamic expressions: NOT flagged (only string literals are checked)
# ---------------------------------------------------------------------------


def test_dynamic_attribute_append_is_not_flagged(tmp_path: Path) -> None:
    """`raw_paths.append(recipe.remote_driver)` cannot be statically resolved
    and is silently skipped."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append(recipe.remote_driver)\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_dynamic_str_call_append_is_not_flagged(tmp_path: Path) -> None:
    """`raw_paths.append(str(value))` cannot be statically resolved."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    value = recipe.raw.get('foo')\n"
        "    raw_paths.append(str(value))\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_dynamic_method_call_append_is_not_flagged(tmp_path: Path) -> None:
    """`raw_paths.append(entry.strip())` cannot be statically resolved."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    entry = 'experiments/foo.py'\n"
        "    raw_paths.append(entry.strip())\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_sameline_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """A `# SENTINEL_OUTSIDE_MOUNT_OK:<real-rationale>` waiver opts out."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('.omx/state/probe.json')  "
        "# SENTINEL_OUTSIDE_MOUNT_OK:operator-side-drift-probe-not-worker-mounted\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert violations == []


def test_sameline_waiver_placeholder_reason_rejected(tmp_path: Path) -> None:
    """The literal placeholder `<reason>` MUST NOT self-waive."""
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('.omx/state/probe.json')  "
        "# SENTINEL_OUTSIDE_MOUNT_OK:<reason>\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


def test_waiver_on_unrelated_line_does_not_protect(tmp_path: Path) -> None:
    """A waiver on a DIFFERENT line does NOT exempt the violation line."""
    root = _make_repo(tmp_path)
    body = (
        "    # SENTINEL_OUTSIDE_MOUNT_OK:nope-this-is-not-attached\n"
        "    raw_paths = []\n"
        "    raw_paths.append('.omx/state/probe.json')\n"
    )
    _write_sentinel_function(root, body)
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Strict mode + missing-target behavior + live-repo regression
# ---------------------------------------------------------------------------


def test_strict_mode_raises_preflight_error(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('.omx/recipe.yaml')\n"
    )
    _write_sentinel_function(root, body)
    with pytest.raises(PreflightError) as excinfo:
        check_modal_sentinel_files_are_in_mount_set(
            repo_root=root, strict=True
        )
    assert "Catalog #201" in str(excinfo.value)
    assert ".omx/recipe.yaml" in str(excinfo.value)


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    body = (
        "    raw_paths = []\n"
        "    raw_paths.append('experiments/modal_train_lane.py')\n"
    )
    _write_sentinel_function(root, body)
    # Should NOT raise
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=True
    )
    assert violations == []


def test_missing_target_file_is_silent(tmp_path: Path) -> None:
    """If `tools/operator_authorize.py` is absent (e.g. partial checkout),
    the gate returns an empty list without raising."""
    root = _make_repo(tmp_path)
    # Don't write any target file.
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=True
    )
    assert violations == []


def test_target_without_sentinel_function_is_silent(tmp_path: Path) -> None:
    """If `_modal_sentinel_files` is absent / renamed, the gate is a no-op."""
    root = _make_repo(tmp_path)
    (root / "tools" / "operator_authorize.py").write_text(
        "def some_other_function():\n"
        "    pass\n"
    )
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=True
    )
    assert violations == []


def test_syntax_error_target_is_silent(tmp_path: Path) -> None:
    """A syntactically broken target file is skipped (not the concern of
    this gate; other preflight checks catch parse errors)."""
    root = _make_repo(tmp_path)
    (root / "tools" / "operator_authorize.py").write_text(
        "def broken(:\n    pass\n"
    )
    violations = check_modal_sentinel_files_are_in_mount_set(
        repo_root=root, strict=True
    )
    assert violations == []


def test_live_repo_zero_violations() -> None:
    """Live-repo regression guard: the actual tools/operator_authorize.py
    in the repo must have zero violations after the Catalog #201 fix."""
    violations = check_modal_sentinel_files_are_in_mount_set(strict=False)
    assert violations == [], (
        f"Live repo has {len(violations)} Catalog #201 violation(s); "
        f"the canonical fix may have regressed: {violations}"
    )
