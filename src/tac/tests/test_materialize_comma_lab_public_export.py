from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.audit_public_publish_links import audit_public_publish_links
from tools.materialize_comma_lab_public_export import (
    materialize_public_export,
    selected_export_paths,
)


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


def test_selected_export_paths_excludes_private_custody() -> None:
    paths = [
        "README.md",
        ".omx/state/lightning_batch_jobs.json",
        ".hypothesis/constants/x",
        "reports/raw/run.json",
        "reports/graphs/site/dashboard_data.json",
        "docs/community.md",
        "docs/superpowers/specs/local.md",
        "src/tac/codec.py",
        "tools/check_dispatch_cli_shell_hazards.py",
        "tools/materialize_comma_lab_public_export.py",
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/x.py",
    ]

    assert selected_export_paths(paths) == [
        "README.md",
        "docs/community.md",
        "tools/materialize_comma_lab_public_export.py",
    ]


def test_materialize_public_export_uses_tracked_blobs_and_omits_private_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("public\n", encoding="utf-8")
    (repo / ".omx/state").mkdir(parents=True)
    private_lightning_url = "https://lightning.ai/" + "private/team/studios/x/app"
    (repo / ".omx/state/lightning_batch_jobs.json").write_text(
        json.dumps({"url": private_lightning_url}) + "\n",
        encoding="utf-8",
    )
    (repo / "docs").mkdir()
    (repo / "docs/community.md").write_text("community record\n", encoding="utf-8")
    _commit_all(repo)

    out = tmp_path / "export"
    manifest = materialize_public_export(repo, out, strict_hygiene=True)

    assert (out / "README.md").read_text(encoding="utf-8") == "public\n"
    assert (out / "docs/community.md").is_file()
    assert not (out / ".omx").exists()
    assert manifest["copied_count"] == 2
    assert manifest["hygiene_violation_count"] == 0
    assert manifest["public_link_count"] == 0
    assert "${PUBLIC_EXPORT_ROOT}" in (out / "PUBLIC_EXPORT_MANIFEST.json").read_text(
        encoding="utf-8"
    )


def test_materialize_public_export_fails_on_private_repo_link(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "docs").mkdir()
    private_repo_url = "https://github.com/adpena/" + "comma-lab"
    (repo / "docs/link.md").write_text(
        f"methodology: {private_repo_url}\n",
        encoding="utf-8",
    )
    _commit_all(repo)

    with pytest.raises(SystemExit, match="PUBLIC LINK HYGIENE"):
        materialize_public_export(repo, tmp_path / "export", strict_hygiene=True)


def test_public_link_violations_allows_other_public_github_links(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "tooling: https://github.com/adpena/tac\n",
        encoding="utf-8",
    )

    payload = audit_public_publish_links([tmp_path], base_root=tmp_path)

    assert payload["violation_count"] == 0
    assert payload["link_count"] == 1
