# SPDX-License-Identifier: MIT
"""Tests for Catalog #174 — check_subagent_commit_serializer_always_uses_expected_content_sha256.

FIX-WAVE-1 R1 Medium #1 (2026-05-13). The 2026-05-12 8c9a5e7f commit-swap
incident proved Catalog #157's protection is asymmetric. This META-meta
gate makes ``--expected-content-sha256`` mandatory on every invocation
of the canonical serializer.

Memory: feedback_fix_wave_1_r1_findings_LANDED_20260513.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_subagent_commit_serializer_always_uses_expected_content_sha256,
)


def _mk(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_check_174_live_count_zero():
    """The check MUST have 0 live violations at landing (strict-flip atom)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=repo_root, strict=False, verbose=False
        )
    )
    assert violations == [], (
        f"Live violations should be 0, got: {violations}"
    )


def test_check_174_detects_python_subprocess_without_flag(tmp_path):
    _mk(
        tmp_path,
        "tools/bad_caller.py",
        "import subprocess\n"
        "subprocess.run(['.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        " '--message', 'x', '--files', 'a.py'])\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert any("bad_caller.py" in v for v in violations)


def test_check_174_strict_raises_preflight_error(tmp_path):
    _mk(
        tmp_path,
        "tools/bad_caller.py",
        "import subprocess\n"
        "subprocess.run(['.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        " '--message', 'x'])\n",
    )
    with pytest.raises(PreflightError):
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_174_accepts_caller_with_flag(tmp_path):
    _mk(
        tmp_path,
        "tools/good_caller.py",
        "import subprocess\n"
        "subprocess.run([\n"
        "    '.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        "    '--message', 'x',\n"
        "    '--files', 'a.py',\n"
        "    '--expected-content-sha256', 'a.py=deadbeef',\n"
        "])\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_accepts_shell_caller_with_flag(tmp_path):
    _mk(
        tmp_path,
        "scripts/good_caller.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SHA=$(shasum -a 256 a.py | awk '{print $1}')\n"
        ".venv/bin/python tools/subagent_commit_serializer.py \\\n"
        "    --message 'x' --files a.py \\\n"
        "    --expected-content-sha256 \"a.py=${SHA}\"\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_accepts_same_line_waiver(tmp_path):
    _mk(
        tmp_path,
        "tools/waived_caller.py",
        "import subprocess\n"
        "subprocess.run(['.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        " '--message', 'x'])  # COMMIT_SERIALIZER_NO_SHA_OK: state-only auto-commit\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_accepts_file_level_bypass_waiver(tmp_path):
    _mk(
        tmp_path,
        "tools/operator_helper.sh",
        "#!/usr/bin/env bash\n"
        "# COMMIT_SERIALIZER_BYPASS_OK_FILE: operator-only housekeeping\n"
        ".venv/bin/python tools/subagent_commit_serializer.py --message 'x'\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_ignores_docstring_references(tmp_path):
    _mk(
        tmp_path,
        "tools/docstring_ref.py",
        '"""Module docstring.\n\n'
        'All commits via tools/subagent_commit_serializer.py per Catalog #117.\n'
        '"""\n'
        'def main():\n'
        '    pass\n',
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_ignores_comment_references(tmp_path):
    _mk(
        tmp_path,
        "tools/comment_ref.py",
        "# Reference: tools/subagent_commit_serializer.py\n"
        "def main():\n"
        "    return 0\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_ignores_path_list_entry(tmp_path):
    _mk(
        tmp_path,
        "tools/path_list.py",
        "PATHS = [\n"
        "    'src/tac/**/*.py',\n"
        "    'tools/subagent_commit_serializer.py',\n"
        "    'tools/lane_maturity.py',\n"
        "]\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_exempts_canonical_serializer_self(tmp_path):
    # Place a fake serializer at the canonical path; the check exempts
    # the file by name.
    _mk(
        tmp_path,
        "tools/subagent_commit_serializer.py",
        "import subprocess\n"
        "subprocess.run(['git', 'commit', '-m', 'x'])\n"
        "# this file is its own implementation - exempt\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    # No violations even though we DON'T pass the flag - exempt by name.
    assert all(
        "tools/subagent_commit_serializer.py" not in v for v in violations
    )


def test_check_174_ignores_test_files(tmp_path):
    _mk(
        tmp_path,
        "src/tac/tests/test_bad.py",
        "import subprocess\n"
        "subprocess.run(['.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        " '--message', 'x'])\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []


def test_check_174_ignores_intake_clones(tmp_path):
    _mk(
        tmp_path,
        "experiments/results/public_pr_99_intake_codex/src/bad.py",
        "import subprocess\n"
        "subprocess.run(['.venv/bin/python', 'tools/subagent_commit_serializer.py',\n"
        " '--message', 'x'])\n",
    )
    violations = (
        check_subagent_commit_serializer_always_uses_expected_content_sha256(
            repo_root=tmp_path, strict=False, verbose=False
        )
    )
    assert violations == []
