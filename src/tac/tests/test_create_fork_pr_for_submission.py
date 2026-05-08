"""Unit tests for tools/create_fork_pr_for_submission.py.

Tests focus on the pure-function helpers (validation, URL parsing, manifest
shape, branch-name derivation). The end-to-end clone+push+gh-pr-create
workflow needs network + a fork repo, so it's exercised via integration
testing only when explicitly invoked, not in CI.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_module():
    path = REPO / "tools" / "create_fork_pr_for_submission.py"
    spec = importlib.util.spec_from_file_location("_create_fork_pr_for_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_submission_dir_accepts_dir_with_inflate_sh(tmp_path: Path) -> None:
    mod = _load_module()
    src = tmp_path / "submission_dir"
    src.mkdir()
    (src / "inflate.sh").write_text("#!/usr/bin/env bash\necho ok\n")
    # Should not raise / sys.exit.
    mod.validate_submission_dir(src)


def test_validate_submission_dir_refuses_missing_inflate_sh(tmp_path: Path) -> None:
    mod = _load_module()
    src = tmp_path / "submission_dir"
    src.mkdir()
    # No inflate.sh — should sys.exit(2).
    with pytest.raises(SystemExit) as excinfo:
        mod.validate_submission_dir(src)
    assert excinfo.value.code == 2


def test_validate_submission_dir_refuses_nonexistent(tmp_path: Path) -> None:
    mod = _load_module()
    src = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit) as excinfo:
        mod.validate_submission_dir(src)
    assert excinfo.value.code == 2


def test_validate_submission_dir_refuses_file_path(tmp_path: Path) -> None:
    mod = _load_module()
    src = tmp_path / "not_a_dir.txt"
    src.write_text("text")
    with pytest.raises(SystemExit) as excinfo:
        mod.validate_submission_dir(src)
    assert excinfo.value.code == 2


def test_parse_pr_number_from_url() -> None:
    mod = _load_module()
    assert mod.parse_pr_number_from_url("https://github.com/owner/repo/pull/42") == 42
    assert mod.parse_pr_number_from_url("https://github.com/owner/repo/pull/1") == 1
    assert mod.parse_pr_number_from_url("https://github.com/owner/repo/pull/12345") == 12345
    assert mod.parse_pr_number_from_url("https://github.com/owner/repo/pull/42/") == 42  # trailing slash


def test_parse_pr_number_from_url_invalid() -> None:
    mod = _load_module()
    assert mod.parse_pr_number_from_url("") is None
    assert mod.parse_pr_number_from_url("https://github.com/owner/repo/pull/notanumber") is None
    assert mod.parse_pr_number_from_url("singletoken") is None


def test_derive_branch_name_format() -> None:
    mod = _load_module()
    branch = mod.derive_branch_name("pr101_lossy_test", "20260508T143000Z")
    assert branch == "add-submission-pr101_lossy_test-20260508T143000Z"


def test_copy_submission_dir_preserves_contents(tmp_path: Path) -> None:
    mod = _load_module()
    src = tmp_path / "src_submission"
    src.mkdir()
    (src / "inflate.sh").write_text("#!/usr/bin/env bash\necho hi\n")
    (src / "README.md").write_text("readme content\n")
    nested = src / "scripts"
    nested.mkdir()
    (nested / "helper.py").write_text("# helper\n")

    clone_dir = tmp_path / "clone"
    (clone_dir / "submissions").mkdir(parents=True)

    target = mod.copy_submission_dir(src, clone_dir, "pr101_test")
    assert target == clone_dir / "submissions" / "pr101_test"
    assert (target / "inflate.sh").read_text() == "#!/usr/bin/env bash\necho hi\n"
    assert (target / "README.md").read_text() == "readme content\n"
    assert (target / "scripts" / "helper.py").read_text() == "# helper\n"


def test_copy_submission_dir_replaces_existing(tmp_path: Path) -> None:
    """If submissions/<name>/ already exists, it must be replaced cleanly."""
    mod = _load_module()
    src = tmp_path / "src_submission"
    src.mkdir()
    (src / "inflate.sh").write_text("new\n")

    clone_dir = tmp_path / "clone"
    existing = clone_dir / "submissions" / "pr101_test"
    existing.mkdir(parents=True)
    (existing / "OLD_FILE").write_text("stale\n")

    target = mod.copy_submission_dir(src, clone_dir, "pr101_test")
    # Old file gone, new file present
    assert not (target / "OLD_FILE").exists()
    assert (target / "inflate.sh").read_text() == "new\n"


def test_build_pr_body_minimal_without_archive() -> None:
    mod = _load_module()
    body = mod.build_pr_body("pr101_test", None, None)
    assert "submissions/pr101_test/" in body
    assert "GHA CPU auth eval workflow" in body
    assert "tools/create_fork_pr_for_submission.py" in body
    # No archive lines when path/sha are None
    assert "intended_archive:" not in body
    assert "archive_sha256:" not in body


def test_build_pr_body_includes_archive_provenance() -> None:
    mod = _load_module()
    body = mod.build_pr_body(
        "pr101_test",
        Path("experiments/results/.../archive.zip"),
        "abc123def456",
    )
    assert "intended_archive: archive.zip" in body
    assert "archive_sha256: abc123def456" in body


def test_write_manifest_produces_valid_json(tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "out"
    manifest_path = mod.write_manifest(
        out,
        fork_repo="adpena/comma_video_compression_challenge",
        submission_name="pr101_test",
        branch="add-submission-pr101_test-20260508T143000Z",
        pr_number=42,
        pr_url="https://github.com/adpena/comma_video_compression_challenge/pull/42",
        base_branch="master",
        submission_dir_src=Path("/tmp/foo"),
        archive_path=Path("/tmp/archive.zip"),
        archive_sha256="abc123",
        reused_existing_pr=False,
        draft=True,
    )
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["submission_name"] == "pr101_test"
    assert data["pr_number"] == 42
    assert data["pr_url"].endswith("/pull/42")
    assert data["base_branch"] == "master"
    assert data["draft"] is True
    assert data["reused_existing_pr"] is False
    assert data["intended_archive_sha256"] == "abc123"


def test_now_utc_compact_format() -> None:
    mod = _load_module()
    s = mod.now_utc_compact()
    # Format YYYYMMDDTHHMMSSZ
    assert len(s) == 16
    assert s[8] == "T"
    assert s.endswith("Z")
    # All digits except T and Z
    assert s[:8].isdigit()
    assert s[9:15].isdigit()


def test_now_utc_iso_format() -> None:
    mod = _load_module()
    s = mod.now_utc_iso()
    # Format YYYY-MM-DDTHH:MM:SSZ
    assert s[10] == "T"
    assert s.endswith("Z")
    assert "-" in s[:10]
    assert ":" in s[11:]
