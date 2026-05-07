"""Tests for ``scripts/deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh``.

Bug-hunter v2 (new MEDIUM): the playbook previously warned but did not fail
closed when the lane claim was absent for a real-provider dispatch, violating
CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION non-negotiable". The post-fix
script must:

  - exit 6 on real-provider dispatch with claim missing
  - remain permissive in dry-run mode
  - not regress the existing dry-run / sha-check / bytes-check exit codes

These tests use the real script with a mocked archive + claim file under
``tmp_path``; nothing dispatches a remote job (the script's exec lines are
not reached because we use --dry-run or fail before the case statement).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "deferred_dispatch_playbook_pr103_pr106_standalone_20260507.sh"
ARCHIVE = (
    REPO_ROOT / "experiments" / "results" / "pr103_repack_pr106_standalone_20260507"
    / "archive.zip"
)


def _script_present() -> bool:
    return SCRIPT.is_file() and ARCHIVE.is_file()


pytestmark = pytest.mark.skipif(
    not _script_present(),
    reason="playbook script or candidate archive not present",
)


def _make_repo_root_copy(tmp_path: Path) -> Path:
    """Stage a minimal mirror of the repo so the playbook resolves paths."""
    work = tmp_path / "mirror"
    work.mkdir()
    # The playbook computes REPO_ROOT from BASH_SOURCE/.. so we must place
    # the script at scripts/<name>.sh under our staged tree.
    (work / "scripts").mkdir()
    staged_script = work / "scripts" / SCRIPT.name
    shutil.copy2(SCRIPT, staged_script)
    staged_script.chmod(0o755)
    # Stage the archive at the canonical relative path the script expects.
    archive_dir = work / "experiments" / "results" / "pr103_repack_pr106_standalone_20260507"
    archive_dir.mkdir(parents=True)
    shutil.copy2(ARCHIVE, archive_dir / "archive.zip")
    # Stage a minimal runtime tree so the directory check (if any) succeeds.
    runtime_dir = work / "submissions" / "pr103_pr106_final_runtime"
    runtime_dir.mkdir(parents=True)
    return work


def _run_playbook(work: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    cmd = ["bash", str(work / "scripts" / SCRIPT.name), *args]
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd, capture_output=True, text=True, env=full_env,
    )


def test_playbook_dry_run_permissive_with_claim_missing(tmp_path: Path) -> None:
    """Dry-run mode must not fail when the lane claim file exists but no
    matching row is present (claim_missing)."""
    work = _make_repo_root_copy(tmp_path)
    omx_dir = work / ".omx" / "state"
    omx_dir.mkdir(parents=True)
    # Claim file exists but the row matches a DIFFERENT lane.
    (omx_dir / "active_lane_dispatch_claims.md").write_text(
        "| some_other_lane | active | ttl=168h |\n"
    )
    proc = _run_playbook(work, "--dry-run")
    assert proc.returncode == 0, (
        f"dry-run with missing claim must exit 0; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "claim_missing" in proc.stdout
    # Permissive WARNING wording is preserved for dry-run.
    assert "dry-run permissive" in proc.stderr


def test_playbook_dry_run_permissive_with_claim_unknown(tmp_path: Path) -> None:
    """Dry-run mode must also tolerate the claim file being absent entirely
    (claim_unknown). Both states are "no claim" conditions."""
    work = _make_repo_root_copy(tmp_path)
    # NO claim file at all => LANE_STATE=claim_unknown.
    proc = _run_playbook(work, "--dry-run")
    assert proc.returncode == 0, (
        f"dry-run with claim_unknown must exit 0; stderr={proc.stderr!r}"
    )
    assert "claim_unknown" in proc.stdout
    assert "dry-run permissive" in proc.stderr


def test_playbook_real_provider_fails_closed_when_claim_missing(tmp_path: Path) -> None:
    """Bug-hunter v2 (new MEDIUM): real-provider dispatch with a missing
    lane claim must FAIL CLOSED with exit 6 (not warn-and-proceed).

    We invoke with --provider lightning but expect the script to abort BEFORE
    the case statement reaches its exec line."""
    work = _make_repo_root_copy(tmp_path)
    # No lane-claim file: claim_missing.
    proc = _run_playbook(work, "--provider", "lightning")
    assert proc.returncode == 6, (
        f"real-provider dispatch with missing claim must exit 6; got "
        f"{proc.returncode}; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "FATAL: lane claim" in proc.stderr
    assert "claim_unknown" in proc.stderr or "claim_missing" in proc.stderr
    assert "tools/claim_lane_dispatch.py" in proc.stderr


def test_playbook_real_provider_proceeds_when_claim_present(tmp_path: Path) -> None:
    """Sanity: with the claim file in place, the playbook should NOT exit 6
    (it may still fail later because it tries to exec a launcher binary that
    doesn't exist in the staged tree, but that's a separate exit code)."""
    work = _make_repo_root_copy(tmp_path)
    omx_dir = work / ".omx" / "state"
    omx_dir.mkdir(parents=True)
    (omx_dir / "active_lane_dispatch_claims.md").write_text(
        "| pr103_pr106_standalone | active | ttl=168h |\n"
    )
    proc = _run_playbook(work, "--provider", "lightning")
    # Must NOT be the claim-missing exit code.
    assert proc.returncode != 6, (
        f"claim-present must not trigger exit 6; got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
    # Either the script execs and the launcher fails, or the staged tree
    # produces a different exit code. We just assert claim_present was logged.
    assert "claim_present" in proc.stdout or "claim_present" in proc.stderr


def test_playbook_archive_bytes_drift_exits_3(tmp_path: Path) -> None:
    """Existing exit-code regression: archive bytes drift must still exit 3."""
    work = _make_repo_root_copy(tmp_path)
    archive_path = work / "experiments" / "results" / "pr103_repack_pr106_standalone_20260507" / "archive.zip"
    # Truncate the archive to force bytes drift.
    archive_path.write_bytes(b"\x00" * 100)
    proc = _run_playbook(work, "--dry-run")
    assert proc.returncode == 3, (
        f"bytes drift must exit 3; got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )


def test_playbook_archive_missing_exits_2(tmp_path: Path) -> None:
    """Existing exit-code regression: missing archive must still exit 2."""
    work = _make_repo_root_copy(tmp_path)
    archive_path = work / "experiments" / "results" / "pr103_repack_pr106_standalone_20260507" / "archive.zip"
    archive_path.unlink()
    proc = _run_playbook(work, "--dry-run")
    assert proc.returncode == 2, (
        f"missing archive must exit 2; got {proc.returncode}; "
        f"stderr={proc.stderr!r}"
    )
