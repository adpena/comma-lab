from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.audit_public_publish_links import audit_public_publish_links
from tools.materialize_comma_lab_public_export import (
    main,
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
        "src/tac/tests/test_audit_public_publish_links.py",
        "src/tac/tests/test_repo_io.py",
        "src/tac/tests/test_tool_bootstrap.py",
        "tools/check_dispatch_cli_shell_hazards.py",
        "tools/audit_public_publish_links.py",
        "tools/materialize_comma_lab_public_export.py",
        "tools/tool_bootstrap.py",
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/x.py",
    ]

    assert selected_export_paths(paths) == [
        "README.md",
        "docs/community.md",
        "src/tac/tests/test_audit_public_publish_links.py",
        "src/tac/tests/test_repo_io.py",
        "src/tac/tests/test_tool_bootstrap.py",
        "tools/audit_public_publish_links.py",
        "tools/materialize_comma_lab_public_export.py",
        "tools/tool_bootstrap.py",
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
    manifest_path = out / "PUBLIC_EXPORT_MANIFEST.json"

    assert (out / "README.md").read_text(encoding="utf-8") == "public\n"
    assert (out / "docs/community.md").is_file()
    assert not (out / ".omx").exists()
    assert manifest["copied_count"] == 2
    assert manifest["hygiene_violation_count"] == 0
    assert manifest["public_link_count"] == 0
    assert set(manifest) == {
        "schema_version",
        "produced_by",
        "source_ref",
        "source_head",
        "output",
        "include_patterns",
        "exclude_patterns",
        "copied_count",
        "copied_bytes",
        "copied",
        "publish_warning",
        "hygiene_violation_count",
        "public_link_violation_count",
        "public_link_count",
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["output"] == "${PUBLIC_EXPORT_ROOT}"
    assert all(set(item) == {"path", "bytes", "sha256"} for item in manifest["copied"])
    copied = {item["path"]: item for item in manifest["copied"]}
    assert copied["README.md"]["sha256"] == hashlib.sha256(b"public\n").hexdigest()


def test_main_force_refuses_to_delete_repo_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("public\n", encoding="utf-8")
    _commit_all(repo)

    with pytest.raises(SystemExit, match="repository root"):
        main(["--repo-root", str(repo), "--out-dir", str(repo), "--force"])

    assert (repo / "README.md").read_text(encoding="utf-8") == "public\n"


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
