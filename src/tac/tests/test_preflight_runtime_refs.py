"""Tests for tac.preflight_runtime_refs.check_shell_script_runtime_refs_resolve."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight_runtime_refs import check_shell_script_runtime_refs_resolve


def _write_script(path: Path, body: str) -> None:
    path.write_text(body)


def test_resolves_existing_require_file(tmp_path):
    """require_file pointing at an existing file → no violation."""
    (tmp_path / "scripts").mkdir()
    real = tmp_path / "experiments"
    real.mkdir()
    (real / "foo.py").write_text("# foo")
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
require_file "$WORKSPACE/experiments/foo.py"
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_catches_missing_require_file(tmp_path):
    """require_file pointing at non-existent file → violation."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
require_file "$WORKSPACE/experiments/missing_helper.py"
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1
    assert "missing_helper.py" in violations[0]
    assert "lane.sh:2" in violations[0]


def test_catches_pybin_invocation_missing(tmp_path):
    """`$PYBIN -u <missing.py>` → violation."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
"$PYBIN" -u experiments/missing_train.py --epochs 100
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1
    assert "missing_train.py" in violations[0]


def test_catches_bash_invocation_missing(tmp_path):
    """`bash $WORKSPACE/<missing.sh>` → violation."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
bash "$WORKSPACE/scripts/missing_helper.sh"
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1
    assert "missing_helper.sh" in violations[0]


def test_ignores_comment_only_references(tmp_path):
    """File path in a comment line → no violation (false-positive class)."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
# Format example: REQUIRED_SOURCE_SHA256S='src/tac/foo.py=<sha>'
# See experiments/never_existed.py for the historical reference
echo "starting lane"
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_ignores_module_path_python_invocation(tmp_path):
    """`$PYBIN -m tac.experiments.foo` (module path) → no violation —
    resolved by Python import system, not file system."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
"$PYBIN" -u -m tac.experiments.train_renderer --batch-size 1
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_skip_block_marker_suppresses(tmp_path):
    """Reference in a block marked `# placeholder` → no violation."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
# Stage 3 is currently a placeholder until experiments/qat_omega.py lands.
# the call below is intentionally deferred:
"$PYBIN" -u experiments/qat_omega.py --foo bar
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_strict_mode_raises_on_violations(tmp_path):
    """strict=True raises RuntimeError on any violation."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
require_file "$WORKSPACE/experiments/missing.py"
''')
    with pytest.raises(RuntimeError, match="unresolved runtime references"):
        check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=True, verbose=False)


def test_strict_mode_passes_when_clean(tmp_path):
    """strict=True does NOT raise when there are zero violations."""
    (tmp_path / "scripts").mkdir()
    real = tmp_path / "experiments"
    real.mkdir()
    (real / "foo.py").write_text("# foo")
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
require_file "$WORKSPACE/experiments/foo.py"
''')
    # Should not raise
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=True, verbose=False)
    assert violations == []


def test_live_codebase_under_known_violation_threshold():
    """Sanity check against the live repo: this is the snapshot at the
    moment the check was introduced. As of 2026-05-04 there is exactly 1
    known violation: experiments/line_search_pose_refinement.py (lost in
    subagent worktree; can be rebuilt). Threshold is loose so future small
    drift doesn't break this test, but a hard-blow-up would surface a
    regression."""
    violations = check_shell_script_runtime_refs_resolve(strict=False, verbose=False)
    assert len(violations) <= 5, (
        f"shell-script runtime refs check violations climbed to {len(violations)} "
        f"(was 1 at introduction): {violations}"
    )


def test_pybin_brace_form_recognized(tmp_path):
    """`${PYBIN}` brace form should also be recognized."""
    (tmp_path / "scripts").mkdir()
    _write_script(tmp_path / "scripts" / "lane.sh", '''#!/bin/bash
${PYBIN} -u experiments/missing_x.py --x 1
''')
    violations = check_shell_script_runtime_refs_resolve(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1


# ============================================================
# check_test_imports_resolve_to_disk — sister check for test-file imports
# ============================================================

from tac.preflight_runtime_refs import check_test_imports_resolve_to_disk  # noqa: E402


def test_test_import_resolves_existing_module(tmp_path):
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "real_module.py").write_text("X = 1\n")
    (test_dir / "test_x.py").write_text(
        "from experiments.real_module import X\n"
    )
    violations = check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_test_import_catches_missing_experiments_module(tmp_path):
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    (test_dir / "test_x.py").write_text(
        "from experiments.missing_module import (foo, bar)\n"
    )
    violations = check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1
    assert "missing_module" in violations[0]
    assert "experiments/missing_module.py" in violations[0]


def test_test_import_catches_missing_tools_module(tmp_path):
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_y.py").write_text(
        "from tools.missing_helper import claim\n"
    )
    violations = check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=False, verbose=False)
    assert len(violations) == 1
    assert "tools/missing_helper.py" in violations[0]


def test_test_import_resolves_package_init(tmp_path):
    """`from experiments.foo import bar` resolves to experiments/foo/__init__.py
    when foo is a package, not a module."""
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (tmp_path / "experiments" / "foo").mkdir(parents=True)
    (tmp_path / "experiments" / "foo" / "__init__.py").write_text("bar = 42\n")
    (test_dir / "test_z.py").write_text(
        "from experiments.foo import bar\n"
    )
    violations = check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_test_import_resolves_namespace_package_dir(tmp_path):
    """`from submissions.robust_current import inflate_renderer` is valid when
    robust_current is a package/namespace directory containing the child
    module, even without a sibling `robust_current.py` file."""
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    runtime_dir = tmp_path / "submissions" / "robust_current"
    runtime_dir.mkdir(parents=True)
    (tmp_path / "submissions" / "__init__.py").write_text(
        '"""Submission packages."""\n'
    )
    (runtime_dir / "inflate_renderer.py").write_text("X = 1\n")
    (test_dir / "test_runtime.py").write_text(
        "from submissions.robust_current import inflate_renderer\n"
    )

    violations = check_test_imports_resolve_to_disk(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )

    assert violations == []


def test_test_import_ignores_stdlib_and_third_party(tmp_path):
    """stdlib + third-party imports must NOT be flagged."""
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_x.py").write_text(
        "from collections import OrderedDict\n"
        "from numpy import array\n"
        "from torch.nn import Module\n"
        "from tac.preflight_runtime_refs import check_test_imports_resolve_to_disk\n"
    )
    violations = check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=False, verbose=False)
    assert violations == []


def test_test_import_strict_mode_raises(tmp_path):
    test_dir = tmp_path / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True)
    (test_dir / "test_x.py").write_text(
        "from experiments.missing_x import foo\n"
    )
    with pytest.raises(RuntimeError, match="unresolved test-file imports"):
        check_test_imports_resolve_to_disk(repo_root=tmp_path, strict=True, verbose=False)


def test_test_import_live_codebase_under_known_threshold():
    """Sanity: as of 2026-05-04 (after rebuilding repack_quantizr_faithful_qzs3_archive
    + build_renderer_packed_payload_archive stubs) there are 0 known
    violations. Threshold is loose so future small drift doesn't break.

    2026-05-13: the static resolver now accepts package/namespace
    directories, matching Python's import behavior for
    ``submissions.robust_current`` contest-runtime imports.
    """
    violations = check_test_imports_resolve_to_disk(strict=False, verbose=False)
    assert violations == []
