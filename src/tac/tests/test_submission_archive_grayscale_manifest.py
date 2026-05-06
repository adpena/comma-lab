from __future__ import annotations

import zipfile
from pathlib import Path

from tac.submission_archive import (
    RENDERER_ALPHA4_COMPACT_MANIFEST,
    build_submission_archive,
    detect_pose_manifest,
    validate_archive,
)


def test_grayscale_manifest_build_validate_and_detect(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)
    archive = tmp_path / "archive.zip"

    result = build_submission_archive(
        archive,
        renderer_bin=renderer,
        grayscale_mkv=grayscale,
        optimized_poses_bin=poses,
        manifest=RENDERER_ALPHA4_COMPACT_MANIFEST,
    )

    assert result.valid
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {"renderer.bin", "grayscale.mkv", "optimized_poses.bin"}

    detected = detect_pose_manifest(archive)
    assert detected.grayscale_mkv is True
    assert detected.optimized_poses_bin is True
    assert validate_archive(archive, detected, strict=True).valid
