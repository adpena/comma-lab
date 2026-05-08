"""Tests for B3 — build_manifest archive custody scanner.

Every build_manifest.json with archive_relpath + archive_sha256 must satisfy
ONE of: archive committed, verifier script references it, custody_status set.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_build_manifest_archive_custody_clean.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b3_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Reset lru_cache so each test gets a fresh git/verifier scan.
    module._git_tracked_paths.cache_clear()
    module._verifier_scripts_text.cache_clear()
    return module


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--allow-empty", "-q", "-m", "init"],
        cwd=str(repo), check=True,
    )


def _write_manifest(path: Path, archive_rel: str, sha: str, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"archive_relpath": archive_rel, "archive_sha256": sha, **extra}
    path.write_text(json.dumps(data, indent=2))


def test_b3_flags_orphan_manifest(tmp_path: Path) -> None:
    mod = _load_module()
    _init_git_repo(tmp_path)
    _write_manifest(
        tmp_path / "experiments" / "results" / "lane_x" / "build_manifest.json",
        "experiments/results/lane_x/archive.zip",
        "ab8a8a13c70b3d3b" * 4,
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b3_accepts_custody_status_published(tmp_path: Path) -> None:
    mod = _load_module()
    _init_git_repo(tmp_path)
    _write_manifest(
        tmp_path / "experiments" / "results" / "lane_x" / "build_manifest.json",
        "experiments/results/lane_x/archive.zip",
        "ab" * 32,
        custody_status="published",
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b3_accepts_custody_status_transient_allowed(tmp_path: Path) -> None:
    mod = _load_module()
    _init_git_repo(tmp_path)
    _write_manifest(
        tmp_path / "experiments" / "results" / "lane_x" / "build_manifest.json",
        "experiments/results/lane_x/archive.zip",
        "ab" * 32,
        custody_status="transient-allowed",
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b3_accepts_verifier_script_referencing_sha(tmp_path: Path) -> None:
    mod = _load_module()
    _init_git_repo(tmp_path)
    sha = "ab" * 32
    _write_manifest(
        tmp_path / "experiments" / "results" / "lane_x" / "build_manifest.json",
        "experiments/results/lane_x/archive.zip",
        sha,
    )
    verifier = tmp_path / "tools" / "verify_lane_x_archive_sha256.py"
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_text(f'EXPECTED_SHA = "{sha}"\n')
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b3_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _init_git_repo(tmp_path)
    _write_manifest(
        tmp_path / "experiments" / "results" / "lane_y" / "build_manifest.json",
        "experiments/results/lane_y/archive.zip",
        "cd" * 32,
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
