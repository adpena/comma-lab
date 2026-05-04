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
