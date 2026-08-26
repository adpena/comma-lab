# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tac.preflight as preflight
from tac.preflight import (
    PreflightError,
    check_no_clone_based_serializer_fallbacks,
)

REPO = Path(__file__).resolve().parents[3]


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_negative_control_current_landing_tools_are_clone_free() -> None:
    assert check_no_clone_based_serializer_fallbacks(repo_root=REPO) == []


def test_guard_is_strictly_wired_into_preflight_all() -> None:
    source = inspect.getsource(preflight.preflight_all)
    call_at = source.index('"check_no_clone_based_serializer_fallbacks"')
    call = source[call_at : call_at + 400]

    assert "check_no_clone_based_serializer_fallbacks(" in call
    assert "strict=True" in call


def test_positive_control_synthetic_serializer_clone_fires(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/subagent_commit_serializer.py",
        "import subprocess\n"
        "subprocess.run(\n"
        "    ['git', 'clone', str(repo), str(scratch)],\n"
        "    check=True,\n"
        ")\n",
    )

    violations = check_no_clone_based_serializer_fallbacks(repo_root=tmp_path)

    assert len(violations) == 1
    assert "clone-based landing fallback forbidden" in violations[0]
    with pytest.raises(PreflightError, match="CLONE_BASED_SERIALIZER_FALLBACK_FORBIDDEN"):
        check_no_clone_based_serializer_fallbacks(repo_root=tmp_path, strict=True)


def test_direct_sister_clone_command_fires(tmp_path: Path) -> None:
    _write(tmp_path, "tools/subagent_commit_serializer.py", "VALUE = 1\n")
    _write(
        tmp_path,
        "tools/codex_harvest_commit.py",
        "cmd = ['git', 'clone', source_repo, recovery_dir]\n",
    )

    violations = check_no_clone_based_serializer_fallbacks(repo_root=tmp_path)

    assert len(violations) == 1
    assert "tools/codex_harvest_commit.py:1" in violations[0]


def test_same_expression_waiver_requires_substantive_rationale(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tools/subagent_commit_serializer.py",
        "cmd = ['git', 'clone', source, dest]  # SERIALIZER_FALLBACK_CLONE_OK:<reason>\n",
    )
    assert len(check_no_clone_based_serializer_fallbacks(repo_root=tmp_path)) == 1

    _write(
        tmp_path,
        "tools/subagent_commit_serializer.py",
        "cmd = ['git', 'clone', source, dest]  # SERIALIZER_FALLBACK_CLONE_OK:remote public intake never targets the Pact checkout\n",
    )
    assert check_no_clone_based_serializer_fallbacks(repo_root=tmp_path) == []


def test_unscanned_public_intake_clone_is_outside_guard_scope(tmp_path: Path) -> None:
    _write(tmp_path, "tools/subagent_commit_serializer.py", "VALUE = 1\n")
    _write(
        tmp_path,
        "tools/fetch_all_public_pr_archives.py",
        "cmd = ['git', 'clone', '--depth', '1', remote, dest]\n",
    )

    assert check_no_clone_based_serializer_fallbacks(repo_root=tmp_path) == []


def test_shell_landing_sister_clone_fires_and_real_waiver_passes(tmp_path: Path) -> None:
    _write(tmp_path, "tools/subagent_commit_serializer.py", "VALUE = 1\n")
    _write(tmp_path, "tools/commit_autosha.sh", "git clone \"$source\" \"$dest\"\n")
    assert len(check_no_clone_based_serializer_fallbacks(repo_root=tmp_path)) == 1

    _write(
        tmp_path,
        "tools/commit_autosha.sh",
        "git clone \"$source\" \"$dest\"  # SERIALIZER_FALLBACK_CLONE_OK:remote public intake never targets the Pact checkout\n",
    )
    assert check_no_clone_based_serializer_fallbacks(repo_root=tmp_path) == []
