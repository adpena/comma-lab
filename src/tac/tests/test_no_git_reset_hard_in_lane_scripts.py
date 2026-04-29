"""Check 66 regression — `git reset --hard` in remote_lane_*.sh."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_no_git_reset_hard_in_remote_lane_scripts,
)


def _scripts_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    return repo


def test_clean_script_passes(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "remote_lane_clean.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "# CODE PARITY: launcher tarball is authoritative.\n"
        "$PYBIN -u -m pip install -e .\n"
    )
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_executable_git_reset_hard_detected(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "remote_lane_bad.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "git fetch origin main && git reset --hard origin/main\n"
        "$PYBIN -u -m pip install -e .\n"
    )
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "remote_lane_bad.sh:3" in violations[0]


def test_strict_raises(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "remote_lane_bad.sh").write_text(
        "git reset --hard origin/main\n"
    )
    with pytest.raises(MetaBugViolation, match="git reset --hard"):
        check_no_git_reset_hard_in_remote_lane_scripts(
            repo_root=repo, strict=True, verbose=False
        )


def test_comment_only_reference_ignored(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "remote_lane_commented.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "# CODE PARITY: do NOT git reset --hard — wipes local-only anchors.\n"
        "# memory: feedback_git_reset_nukes_anchors_20260429\n"
        "$PYBIN -u -m pip install -e .\n"
    )
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_indented_comment_ignored(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "remote_lane_indented.sh").write_text(
        "#!/bin/bash\n"
        "if [ x = y ]; then\n"
        "    # git reset --hard origin/main  (commented-out reminder)\n"
        "    true\n"
        "fi\n"
    )
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_only_remote_lane_prefix_scanned(tmp_path):
    repo = _scripts_dir(tmp_path)
    (repo / "scripts" / "other_helper.sh").write_text(
        "git reset --hard origin/main\n"
    )
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        repo_root=repo, strict=False, verbose=False
    )
    assert violations == []


def test_real_codebase_clean():
    """The actual pact codebase must be free of executable git reset --hard."""
    violations = check_no_git_reset_hard_in_remote_lane_scripts(
        strict=False, verbose=False
    )
    assert violations == [], f"unexpected violations: {violations}"
