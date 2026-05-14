# SPDX-License-Identifier: MIT
"""Tests for tools/audit_archive.py contest-compliance auditor."""
from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "audit_archive.py"

# Load the tool as a module without sys.path tricks.
_spec = importlib.util.spec_from_file_location("_audit_archive_under_test", TOOL_PATH)
audit_archive = importlib.util.module_from_spec(_spec)
sys.modules["_audit_archive_under_test"] = audit_archive
_spec.loader.exec_module(audit_archive)  # type: ignore[union-attr]


def _make_archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Helper: build a zip with the given (name → bytes) members."""
    p = tmp_path / "test_archive.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return p


def test_missing_archive_fails(tmp_path):
    res = audit_archive.audit(tmp_path / "does_not_exist.zip")
    assert not res.passed
    assert any("not found" in f for f in res.failures)


def test_passes_on_canonical_3_member_archive(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"ASYM" + b"\x00" * 1000,
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
    })
    res = audit_archive.audit(archive)
    assert res.passed, f"unexpected failures: {res.failures}"
    assert res.renderer_magic == b"ASYM"
    assert res.archive_bytes > 0
    assert res.rate_term > 0


def test_fails_on_missing_renderer(tmp_path):
    archive = _make_archive(tmp_path, {
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
    })
    res = audit_archive.audit(archive)
    assert not res.passed
    assert any("renderer.bin" in f for f in res.failures)


def test_missing_renderer_failure_message_is_not_duplicated(tmp_path):
    """R32 Finding 1: prior code appended `missing required member: renderer.bin`
    once via the dedicated check (line ~133) AND a second time via the
    required_members loop (line ~152). The fix skips renderer.bin in the
    required_members loop. Anchor: exactly ONE occurrence in res.failures.
    """
    archive = _make_archive(tmp_path, {
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
    })
    res = audit_archive.audit(archive)
    renderer_failures = [
        f for f in res.failures
        if "missing required member: renderer.bin" in f
    ]
    assert len(renderer_failures) == 1, (
        f"expected exactly 1 renderer.bin failure (no duplicate), "
        f"got {len(renderer_failures)}: {renderer_failures}"
    )


def test_fails_on_unknown_renderer_magic(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"BADX" + b"\x00" * 1000,  # unknown magic
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
    })
    res = audit_archive.audit(archive)
    assert not res.passed
    assert any("scorer-free allowlist" in f for f in res.failures)


def test_warns_on_unknown_extra_member(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"ASYM" + b"\x00" * 1000,
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
        "experiment_residuals.bin": b"\x00" * 1000,  # unknown!
    })
    res = audit_archive.audit(archive)
    assert res.passed  # warnings don't fail
    assert any("experiment_residuals.bin" in w for w in res.warnings)


def test_strict_mode_promotes_warnings_to_failures(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"ASYM" + b"\x00" * 1000,
        "masks.mkv": b"\x00" * 500,
        "optimized_poses.pt": b"\x00" * 200,
        "extra.bin": b"\x00" * 100,
    })
    res = audit_archive.audit(archive, strict=True)
    assert not res.passed
    assert res.warnings == []
    assert any("extra.bin" in f for f in res.failures)


def test_warns_on_oversize_archive(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"ASYM" + b"\x00" * 50000,
        "masks.mkv": b"\x00" * 50000,
        "optimized_poses.pt": b"\x00" * 200,
    })
    # Set max well below the actual size (compressed zeros are tiny so
    # we use a 1-byte cap to force the warning regardless of compression).
    res = audit_archive.audit(archive, max_archive_bytes=1)
    assert res.passed  # warnings don't fail
    assert any("soft cap" in w for w in res.warnings)


def test_fails_on_corrupt_zip(tmp_path):
    p = tmp_path / "corrupt.zip"
    p.write_bytes(b"NOT_A_ZIP_FILE_AT_ALL")
    res = audit_archive.audit(p)
    assert not res.passed
    assert any("not a valid zip" in f for f in res.failures)


def test_render_includes_status_line(tmp_path):
    archive = _make_archive(tmp_path, {
        "renderer.bin": b"ASYM" + b"\x00" * 100,
        "masks.mkv": b"\x00" * 50,
        "optimized_poses.pt": b"\x00" * 20,
    })
    res = audit_archive.audit(archive)
    out = audit_archive.render(res)
    assert "STATUS:" in out
    assert "rate_term:" in out
    assert "renderer.bin" in out
