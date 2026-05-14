# SPDX-License-Identifier: MIT
from __future__ import annotations

import lzma
import zipfile
from pathlib import Path

from tac.submission_archive import (
    RENDERER_ALPHA4_COMPACT_MANIFEST,
    SEGMAP_ARITHMETIC_LCT_MANIFEST,
    SEGMAP_LCT_SUBMISSION_MANIFEST,
    build_submission_archive,
    detect_pose_manifest,
    validate_archive,
    write_deterministic_zip_member,
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


def test_segmap_lct_manifest_build_validate_and_detect(tmp_path: Path) -> None:
    weights = tmp_path / "segmap_weights.tar.xz"
    weights.write_bytes(lzma.compress(b"segmap weights"))
    grayscale = tmp_path / "grayscale.mkv"
    grayscale.write_bytes(b"g" * 197382)
    poses = tmp_path / "optimized_poses.pt"
    poses.write_bytes(b"p" * 7200)
    class_targets = tmp_path / "class_targets.fp16"
    class_targets.write_bytes(b"c" * 10)
    archive = tmp_path / "segmap.zip"

    result = build_submission_archive(
        archive,
        segmap_weights_tar_xz=weights,
        grayscale_mkv=grayscale,
        optimized_poses_pt=poses,
        class_targets_fp16=class_targets,
        manifest=SEGMAP_LCT_SUBMISSION_MANIFEST,
    )

    assert result.valid
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == {
            "segmap_weights.tar.xz",
            "grayscale.mkv",
            "optimized_poses.pt",
            "class_targets.fp16",
        }

    detected = detect_pose_manifest(archive)
    assert detected.segmap_weights_tar_xz is True
    assert detected.renderer_bin is False
    assert detected.grayscale_mkv is True
    assert detected.class_targets_fp16 is True
    assert validate_archive(archive, detected, strict=True).valid


def test_segmap_arithmetic_lct_manifest_validate_and_detect(tmp_path: Path) -> None:
    archive = tmp_path / "segmap_arithmetic.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        write_deterministic_zip_member(zf, "payload.bin", b"SHv1payload")
        write_deterministic_zip_member(zf, "grayscale.mkv", b"g" * 197382)
        write_deterministic_zip_member(zf, "optimized_poses.pt", b"p" * 7200)
        write_deterministic_zip_member(zf, "class_targets.fp16", b"c" * 10)

    detected = detect_pose_manifest(archive)
    assert detected.segmap_payload_bin is True
    assert detected.renderer_bin is False
    assert detected.grayscale_mkv is True
    assert detected.class_targets_fp16 is True
    assert validate_archive(archive, detected, strict=True).valid


def test_segmap_payload_bin_requires_shv1_magic(tmp_path: Path) -> None:
    archive = tmp_path / "bad_payload.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        write_deterministic_zip_member(zf, "payload.bin", b"BAD!payload")
        write_deterministic_zip_member(zf, "grayscale.mkv", b"g" * 197382)
        write_deterministic_zip_member(zf, "optimized_poses.pt", b"p" * 7200)
        write_deterministic_zip_member(zf, "class_targets.fp16", b"c" * 10)

    result = validate_archive(archive, SEGMAP_ARITHMETIC_LCT_MANIFEST, strict=True)

    assert not result.valid
    assert any("payload.bin has bad magic" in error for error in result.errors)


def test_segmap_lct_class_targets_size_must_be_five_fp16s(tmp_path: Path) -> None:
    archive = tmp_path / "bad_class_targets.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        write_deterministic_zip_member(zf, "segmap_weights.tar.xz", lzma.compress(b"x"))
        write_deterministic_zip_member(zf, "grayscale.mkv", b"g" * 197382)
        write_deterministic_zip_member(zf, "optimized_poses.pt", b"p" * 7200)
        write_deterministic_zip_member(zf, "class_targets.fp16", b"c" * 8)

    result = validate_archive(archive, SEGMAP_LCT_SUBMISSION_MANIFEST, strict=True)

    assert not result.valid
    assert any("class_targets.fp16 has 8 bytes" in error for error in result.errors)
