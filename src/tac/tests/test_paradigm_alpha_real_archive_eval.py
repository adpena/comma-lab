"""Tests for Alpha real-archive evaluator custody guards."""
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "paradigm_alpha_real_archive_eval.py"
SPEC = importlib.util.spec_from_file_location("paradigm_alpha_real_archive_eval", MODULE_PATH)
assert SPEC is not None
alpha_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(alpha_eval)


def test_read_archive_member_records_member_custody(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", b"mask-bytes")

    data, meta = alpha_eval._read_archive_member(archive, "masks.mkv")

    assert data == b"mask-bytes"
    assert meta["archive_member_resolved"] == "masks.mkv"
    assert meta["archive_member_size_bytes"] == len(b"mask-bytes")
    assert meta["archive_member_sha256"] == alpha_eval._sha256_bytes(b"mask-bytes")


def test_read_archive_member_rejects_requested_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")

    with pytest.raises(ValueError, match="unsafe requested archive member"):
        alpha_eval._read_archive_member(archive, "../masks.mkv")


def test_read_archive_member_rejects_zip_slip_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../masks.mkv", b"mask-bytes")

    with pytest.raises(ValueError, match="unsafe archive member path"):
        alpha_eval._read_archive_member(archive, "masks.mkv")


def test_read_archive_member_rejects_hidden_sidecar(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("masks.mkv", b"mask-bytes")
        zf.writestr("._masks.mkv", b"sidecar")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        alpha_eval._read_archive_member(archive, "masks.mkv")


def test_default_archive_tracks_pfp16_final_deploy_bundle() -> None:
    assert (
        "lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip"
        in alpha_eval.DEFAULT_PFP16_A_PLUS_PLUS_ARCHIVE.as_posix()
    )
