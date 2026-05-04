from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / "experiments" / "repack_fixedslice_qp1_to_outer_brotli.py"
UNPACK_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def _write_fixedslice_archive(
    tmp_path: Path,
    *,
    masks: bytes = b"\x12\x00\x0a\x0a" + b"mask-obu" * 64,
    renderer: bytes = b"QZS3" + b"renderer" * 96,
    pose_qp1: bytes | None = None,
) -> tuple[Path, Path, dict[str, bytes]]:
    if pose_qp1 is None:
        pose_qp1 = b"QP1" + struct.pack("<H", 5120) + b"\x02\x04"
    mask_br = brotli.compress(masks, quality=1, mode=brotli.MODE_GENERIC, lgwin=18)
    model_br = brotli.compress(renderer, quality=1, mode=brotli.MODE_GENERIC, lgwin=18)
    pose_br = brotli.compress(pose_qp1, quality=1, mode=brotli.MODE_GENERIC, lgwin=18)
    payload = mask_br + model_br + pose_br
    archive = tmp_path / "fixedslice.zip"
    _stored_zip(archive, {"p": payload})
    metadata = {
        "payload_format": "test_pr67_fixed_slices",
        "mask_br_bytes": len(mask_br),
        "mask_br_sha256": _sha256(mask_br),
        "model_br_bytes": len(model_br),
        "model_br_sha256": _sha256(model_br),
        "pose_br_bytes": len(pose_br),
        "pose_br_sha256": _sha256(pose_br),
        "score_claim": False,
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return archive, metadata_path, {
        "masks.mkv": masks,
        "renderer.bin": renderer,
        "optimized_poses.bin": pose_qp1,
    }


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_repack_fixedslice_emits_pr64_mask_first_outer_brotli(tmp_path: Path) -> None:
    repack = _load_module(SCRIPT_PATH, "_fixedslice_repack_test")
    unpacker = _load_module(UNPACK_PATH, "_fixedslice_unpack_test")
    source, metadata_path, expected = _write_fixedslice_archive(tmp_path)
    out = tmp_path / "out" / "archive.zip"
    manifest_path = tmp_path / "out" / "manifest.json"

    manifest = repack.repack_fixedslice_archive(
        source_archive=source,
        metadata_path=metadata_path,
        output_archive=out,
        manifest_json=manifest_path,
    )

    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["output"]["payload_format"] == "public_pr64_mask_first_len_table_outer_brotli"
    assert manifest["decoded_members"]["renderer.bin"]["wire_format"] == "QZS3"
    assert manifest["decoded_members"]["optimized_poses.bin"]["wire_format"] == "QP1"

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["p"]
        info = zf.getinfo("p")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        raw_payload = brotli.decompress(zf.read("p"))

    mask_len, renderer_len, pose_len = struct.unpack_from("<III", raw_payload, 0)
    assert (mask_len, renderer_len, pose_len) == (
        len(expected["masks.mkv"]),
        len(expected["renderer.bin"]),
        len(expected["optimized_poses.bin"]),
    )
    assert raw_payload[12:12 + mask_len] == expected["masks.mkv"]
    assert raw_payload[12 + mask_len:12 + mask_len + renderer_len] == expected["renderer.bin"]
    assert raw_payload[-pose_len:] == expected["optimized_poses.bin"]

    archive_dir = tmp_path / "inflate_dir"
    archive_dir.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)
    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert summary["payload_format"] == "public_pr64_mask_first_len_table"
    assert (archive_dir / "masks.mkv").read_bytes() == expected["masks.mkv"]
    assert (archive_dir / "renderer.bin").read_bytes() == expected["renderer.bin"]
    assert (archive_dir / "optimized_poses.bin").read_bytes()
    assert manifest_path.exists()


def test_repack_rejects_missing_metadata_field(tmp_path: Path) -> None:
    repack = _load_module(SCRIPT_PATH, "_fixedslice_repack_missing_meta_test")
    source, metadata_path, _expected = _write_fixedslice_archive(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("pose_br_bytes")
    metadata_path.write_text(json.dumps(metadata) + "\n")

    try:
        repack.repack_fixedslice_archive(
            source_archive=source,
            metadata_path=metadata_path,
            output_archive=tmp_path / "out.zip",
            manifest_json=tmp_path / "manifest.json",
        )
    except repack.RepackFixedSliceError as exc:
        assert "metadata missing required fields" in str(exc)
    else:
        raise AssertionError("missing pose_br_bytes was accepted")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"masks": b"bad-mask-obu"}, "invalid masks.mkv magic"),
        ({"renderer": b"NOPE" + b"renderer" * 96}, "invalid renderer.bin magic"),
        ({"pose_qp1": b"NO1" + struct.pack("<H", 5120)}, "invalid QP1 pose payload magic"),
    ],
)
def test_repack_rejects_invalid_member_magic(
    tmp_path: Path,
    override: dict[str, bytes],
    match: str,
) -> None:
    repack = _load_module(SCRIPT_PATH, "_fixedslice_repack_bad_qzs3_test")
    source, metadata_path, _expected = _write_fixedslice_archive(tmp_path, **override)

    try:
        repack.repack_fixedslice_archive(
            source_archive=source,
            metadata_path=metadata_path,
            output_archive=tmp_path / "out.zip",
            manifest_json=tmp_path / "manifest.json",
        )
    except repack.RepackFixedSliceError as exc:
        assert match in str(exc)
    else:
        raise AssertionError("invalid fixed-slice member magic was accepted")


def test_repack_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    repack = _load_module(SCRIPT_PATH, "_fixedslice_repack_unsafe_zip_test")
    source, metadata_path, _expected = _write_fixedslice_archive(tmp_path)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source) as zf:
        payload = zf.read("p")
    _stored_zip(unsafe, {"../p": payload})

    try:
        repack.repack_fixedslice_archive(
            source_archive=unsafe,
            metadata_path=metadata_path,
            output_archive=tmp_path / "out.zip",
            manifest_json=tmp_path / "manifest.json",
        )
    except repack.RepackFixedSliceError as exc:
        assert "expected single member" in str(exc) or "unsafe archive member path" in str(exc)
    else:
        raise AssertionError("unsafe member path was accepted")
