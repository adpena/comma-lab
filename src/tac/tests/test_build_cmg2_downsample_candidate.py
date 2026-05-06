from __future__ import annotations

import bz2
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_cmg2_downsample_candidate.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cmg2_payload_roundtrips_through_runtime_loader(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg2_builder_payload_test")
    inflate = _load(INFLATE_PATH, "_cmg2_inflate_payload_test")

    low = np.zeros((1, 192, 256), dtype=np.uint8)
    low[:, 96:, :] = 2
    low[:, 40:80, 100:120] = 4
    payload, header = builder.encode_cmg2_payload(low, scale_y=2, scale_x=2, compressor="bz2")
    path = tmp_path / "masks.cmg2"
    path.write_bytes(payload)

    masks = inflate._load_masks_from_cmg2(path, expected_frames=1)

    expected = np.repeat(np.repeat(low, 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(masks.numpy(), expected)
    assert header["body_sha256"] == hashlib.sha256(bz2.compress(low.tobytes(), compresslevel=9)).hexdigest()


def test_cmg2_runtime_rejects_tampered_body(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg2_builder_tamper_test")
    inflate = _load(INFLATE_PATH, "_cmg2_inflate_tamper_test")

    low = np.zeros((1, 192, 256), dtype=np.uint8)
    payload, _header = builder.encode_cmg2_payload(low, scale_y=2, scale_x=2, compressor="bz2")
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    path = tmp_path / "masks.cmg2"
    path.write_bytes(bytes(tampered))

    try:
        inflate._load_masks_from_cmg2(path, expected_frames=1)
    except ValueError as exc:
        assert "body SHA mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered CMG2 payload was accepted")


def test_downsample_block_mode_records_disagreement() -> None:
    builder = _load(BUILDER_PATH, "_cmg2_builder_downsample_test")

    arr = np.zeros((1, 4, 4), dtype=np.uint8)
    arr[:, 2:, :] = 2
    arr[:, 1, 1] = 4
    low, recon, disagreement = builder.downsample_block_mode(arr, scale_y=2, scale_x=2)

    assert low.shape == (1, 2, 2)
    assert recon.shape == arr.shape
    assert 0.0 < disagreement < 1.0


def test_cmg2_member_is_allowed_in_packed_payload(tmp_path: Path) -> None:
    packer = _load(REPO / "experiments" / "build_renderer_packed_payload_archive.py", "_cmg2_packer_test")
    unpacker = _load(REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py", "_cmg2_unpacker_test")

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"QZS3fake")
        zf.writestr("masks.cmg2", b"CMG2fake")
        zf.writestr("optimized_poses.bin", struct.pack("<" + "e" * 12, *([0.0] * 12)))

    archive = tmp_path / "archive.zip"
    result = packer.build_packed_archive(
        source,
        archive,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    assert result["score_claim"] is False
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["p"]
        payload = zf.read("p")
    archive_dir = tmp_path / "extracted"
    archive_dir.mkdir()
    (archive_dir / "p").write_bytes(payload)
    summary = unpacker.unpack_renderer_payload(archive_dir)
    assert sorted(member["name"] for member in summary["members"]) == [
        "masks.cmg2",
        "optimized_poses.bin",
        "renderer.bin",
    ]
