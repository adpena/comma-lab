from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from tac.submission_archive import (
    RENDERER_COMPACT_MANIFEST,
    TYPED_SIDECHANNEL_CONTRACT_MEMBER,
    TypedSidechannelMember,
    build_submission_archive,
    deterministic_zip_directory,
    validate_archive,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_submission_archive_is_stable_when_source_mtimes_change(tmp_path: Path) -> None:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"m" * 12000)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)

    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"

    build_submission_archive(
        archive_a,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    for i, path in enumerate((renderer, masks, poses), start=1):
        os.utime(path, (1_900_000_000 + i, 1_900_000_000 + i))

    build_submission_archive(
        archive_b,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    assert _sha256(archive_a) == _sha256(archive_b)
    with zipfile.ZipFile(archive_a) as zf:
        assert [info.date_time for info in zf.infolist()] == [
            (1980, 1, 1, 0, 0, 0),
            (1980, 1, 1, 0, 0, 0),
            (1980, 1, 1, 0, 0, 0),
        ]
        assert [info.external_attr >> 16 for info in zf.infolist()] == [
            0o644,
            0o644,
            0o644,
        ]


def test_build_submission_archive_adds_typed_sidechannel_fail_closed_contract(
    tmp_path: Path,
) -> None:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"m" * 12000)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)
    categorical = tmp_path / "categorical_payload.bin"
    categorical.write_bytes(b"class-prior-payload")
    archive = tmp_path / "archive.zip"

    build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        typed_sidechannels=[
            TypedSidechannelMember(
                kind="categorical_payload",
                member_name="categorical_payload.bin",
                source_path=categorical,
                score_affecting=True,
                consumed_by_runtime=False,
            )
        ],
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    result = validate_archive(archive, RENDERER_COMPACT_MANIFEST, strict=True)
    assert result.valid is True
    assert result.dispatch_ready is False
    assert result.dispatch_blockers == [
        "categorical_payload_score_affecting_member_not_consumed_by_runtime"
    ]
    with zipfile.ZipFile(archive) as zf:
        assert "categorical_payload.bin" in zf.namelist()
        contract = json.loads(zf.read(TYPED_SIDECHANNEL_CONTRACT_MEMBER).decode("utf-8"))
    assert contract["score_claim"] is False
    assert contract["dispatch_ready"] is False
    assert contract["members"][0]["sha256"] == hashlib.sha256(categorical.read_bytes()).hexdigest()


def test_typed_sidechannel_runtime_consumption_proof_clears_dispatch_contract(
    tmp_path: Path,
) -> None:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"m" * 12000)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)
    hdm3 = tmp_path / "hdm3.bin"
    hdm3.write_bytes(b"HDM3\x00")
    archive = tmp_path / "archive.zip"

    build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        typed_sidechannels=[
            {
                "kind": "hnerv_hdm3",
                "member_name": "hdm3.bin",
                "source_path": hdm3,
                "score_affecting": True,
                "consumed_by_runtime": True,
                "runtime_consumer": "submissions.robust_current.inflate_renderer",
                "runtime_consumption_proof_sha256": "a" * 64,
            }
        ],
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    result = validate_archive(archive, RENDERER_COMPACT_MANIFEST, strict=True)
    assert result.valid is True
    assert result.dispatch_ready is True
    assert result.dispatch_blockers == []


def test_validate_archive_rejects_uncontracted_sidechannel_member(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("renderer.bin", b"r" * 12000)
        zf.writestr("masks.mkv", b"m" * 12000)
        zf.writestr("optimized_poses.bin", b"p" * 7200)
        zf.writestr("jcsp.bin", b"uncontracted")

    result = validate_archive(archive, RENDERER_COMPACT_MANIFEST, strict=True)

    assert result.valid is False
    assert "jcsp.bin" in result.files_unexpected
    assert "Unexpected file in archive: jcsp.bin" in result.errors


def test_deterministic_zip_directory_sorts_members_and_ignores_source_mtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "b.bin").write_bytes(b"b" * 17)
    (source / "a.bin").write_bytes(b"a" * 13)

    archive_a = tmp_path / "dir_a.zip"
    archive_b = tmp_path / "dir_b.zip"
    members_a = deterministic_zip_directory(source, archive_a)

    os.utime(source / "a.bin", (1_900_000_100, 1_900_000_100))
    os.utime(source / "nested" / "b.bin", (1_900_000_200, 1_900_000_200))
    members_b = deterministic_zip_directory(source, archive_b)

    assert members_a == ["a.bin", "nested/b.bin"]
    assert members_b == ["a.bin", "nested/b.bin"]
    assert _sha256(archive_a) == _sha256(archive_b)


@pytest.mark.parametrize(
    "relpath",
    [
        ".DS_Store",
        "._renderer.bin",
        "__MACOSX/renderer.bin",
        "nested/.hidden",
    ],
)
def test_deterministic_zip_directory_flags_hidden_system_junk(
    tmp_path: Path,
    relpath: str,
) -> None:
    source = tmp_path / "source"
    junk = source / relpath
    junk.parent.mkdir(parents=True, exist_ok=True)
    junk.write_bytes(b"junk")

    with pytest.raises(ValueError, match="hidden/system archive member"):
        deterministic_zip_directory(source, tmp_path / "archive.zip")
