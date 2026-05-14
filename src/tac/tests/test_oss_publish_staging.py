# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.oss_publish_staging import selected_oss_paths, stage_oss_publish


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit_all(repo: Path) -> None:
    _git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_selected_oss_paths_uses_include_and_private_exclude_patterns() -> None:
    paths = [
        "src/tac/codec.py",
        "src/tac/tests/test_codec.py",
        "src/tac/__pycache__/codec.cpython-312.pyc",
        ".omx/state/private.json",
        "experiments/results/raw.json",
        "tools/oss_publish_staging.py",
        "tools/tool_bootstrap.py",
        "tools/audit_public_publish_links.py",
        "README.md",
    ]

    assert selected_oss_paths(paths) == [
        "src/tac/codec.py",
        "src/tac/tests/test_codec.py",
        "tools/audit_public_publish_links.py",
        "tools/oss_publish_staging.py",
        "tools/tool_bootstrap.py",
    ]


def test_stage_oss_publish_uses_committed_blobs_not_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "src/tac").mkdir(parents=True)
    (repo / "src/tac/codec.py").write_text("VALUE = 'clean'\n", encoding="utf-8")
    _commit_all(repo)
    (repo / "src/tac/codec.py").write_text("VALUE = '/Users/adpena/private-dirty'\n", encoding="utf-8")

    out = tmp_path / "stage"
    manifest = stage_oss_publish(out, repo, strict_hygiene=False)

    assert (out / "src/tac/codec.py").read_text(encoding="utf-8") == "VALUE = 'clean'\n"
    manifest_text = (out / "MANIFEST.json").read_text(encoding="utf-8")
    assert "/Users/" not in manifest_text
    assert str(repo) not in manifest_text
    assert str(out) not in manifest_text
    assert manifest["staging_root"] == "${TAC_OSS_STAGING_ROOT}"
    assert manifest["readme_source"] == "fallback (operator should provide --readme)"
    assert manifest["hygiene_violation_count"] == 0
    assert manifest["public_link_violation_count"] == 0
    assert json.loads(manifest_text) == manifest


def test_stage_oss_publish_refuses_in_repo_out_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "src/tac").mkdir(parents=True)
    (repo / "src/tac/codec.py").write_text("VALUE = 'clean'\n", encoding="utf-8")
    _commit_all(repo)

    with pytest.raises(SystemExit, match="outside the repository"):
        stage_oss_publish(repo / "stage", repo, strict_hygiene=False)


def test_stage_oss_publish_strict_hygiene_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "src/tac").mkdir(parents=True)
    (repo / "src/tac/codec.py").write_text("PATH = '/Users/adpena/private'\n", encoding="utf-8")
    _commit_all(repo)

    with pytest.raises(SystemExit, match="PUBLIC RELEASE HYGIENE"):
        stage_oss_publish(tmp_path / "stage", repo)
