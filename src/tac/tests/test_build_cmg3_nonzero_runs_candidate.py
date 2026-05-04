from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_cmg3_nonzero_runs_candidate.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cmg3_nonzero_runs_payload_roundtrips_through_runtime_loader(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_runs_builder_payload_test")
    inflate = _load(INFLATE_PATH, "_cmg3_runs_inflate_payload_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[:, 20:80, 40:180] = 1
    source[:, 120:160, 220:260] = 3
    source[:, 200:260, 100:500] = 4
    stream, recon, stats = builder.encode_run_stream(source, max_runs_per_row=2)
    payload, header = builder.encode_cmg3_payload(
        stream,
        frame_count=1,
        max_runs_per_row=2,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        reconstructed_mask_sha256=hashlib.sha256(recon.tobytes()).hexdigest(),
        pixel_disagreement=stats["pixel_disagreement"],
        pixel_disagreement_count=stats["pixel_disagreement_count"],
        compressor="bz2",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    masks = inflate._load_masks_from_cmg3(path, expected_frames=1)

    np.testing.assert_array_equal(masks.numpy(), recon)
    assert header["mode"] == "nonzero_row_runs_topk_v1"
    assert header["record_struct"] == "u8_count_then_u8_class_u16_start_u16_end_le"
    assert stats["pixel_disagreement_count"] == 0


def test_cmg3_nonzero_runs_topk_records_lossy_disagreement() -> None:
    builder = _load(BUILDER_PATH, "_cmg3_runs_builder_lossy_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[:, 10:20, 5:25] = 1
    source[:, 10:20, 30:70] = 2
    source[:, 10:20, 90:200] = 4
    stream, recon, stats = builder.encode_run_stream(source, max_runs_per_row=1)

    assert stream
    assert stats["kept_runs"] < stats["total_nonzero_runs"]
    assert stats["pixel_disagreement_count"] > 0
    assert np.any(recon != source)


def test_cmg3_runtime_rejects_reconstructed_mask_sha_mismatch(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_runs_builder_recon_sha_test")
    inflate = _load(INFLATE_PATH, "_cmg3_runs_inflate_recon_sha_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    source[:, 20:80, 40:180] = 1
    stream, recon, stats = builder.encode_run_stream(source, max_runs_per_row=1)
    payload, _header = builder.encode_cmg3_payload(
        stream,
        frame_count=1,
        max_runs_per_row=1,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        reconstructed_mask_sha256="0" * 64,
        pixel_disagreement=stats["pixel_disagreement"],
        pixel_disagreement_count=stats["pixel_disagreement_count"],
        compressor="raw",
    )
    assert hashlib.sha256(recon.tobytes()).hexdigest() != "0" * 64
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    try:
        inflate._load_masks_from_cmg3(path, expected_frames=1)
    except ValueError as exc:
        assert "reconstructed mask SHA mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("CMG3 payload with wrong reconstructed tensor SHA was accepted")


def test_cmg3_nonzero_runs_runtime_rejects_out_of_order_runs(tmp_path: Path) -> None:
    builder = _load(BUILDER_PATH, "_cmg3_runs_builder_order_test")
    inflate = _load(INFLATE_PATH, "_cmg3_runs_inflate_order_test")

    source = np.zeros((1, 384, 512), dtype=np.uint8)
    stream = bytearray()
    stream.append(2)
    stream.append(1)
    stream += (20).to_bytes(2, "little")
    stream += (25).to_bytes(2, "little")
    stream.append(2)
    stream += (10).to_bytes(2, "little")
    stream += (15).to_bytes(2, "little")
    stream.extend(b"\x00" * (384 - 1))
    recon = np.zeros_like(source)
    payload, _header = builder.encode_cmg3_payload(
        bytes(stream),
        frame_count=1,
        max_runs_per_row=2,
        source_mask_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        reconstructed_mask_sha256=hashlib.sha256(recon.tobytes()).hexdigest(),
        pixel_disagreement=0.0,
        pixel_disagreement_count=0,
        compressor="raw",
    )
    path = tmp_path / "masks.cmg3"
    path.write_bytes(payload)

    try:
        inflate._load_masks_from_cmg3(path, expected_frames=1)
    except ValueError as exc:
        assert "strictly non-overlapping and sorted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("out-of-order CMG3 row runs were accepted")


def test_cmg3_nonzero_runs_member_is_allowed_in_packed_payload(tmp_path: Path) -> None:
    packer = _load(REPO / "experiments" / "build_renderer_packed_payload_archive.py", "_cmg3_runs_packer_test")
    unpacker = _load(REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py", "_cmg3_runs_unpacker_test")

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("renderer.bin", b"QZS3fake")
        zf.writestr("masks.cmg3", b"CMG3fake")
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
        payload = zf.read("p")
    archive_dir = tmp_path / "extracted"
    archive_dir.mkdir()
    (archive_dir / "p").write_bytes(payload)
    summary = unpacker.unpack_renderer_payload(archive_dir)
    assert sorted(member["name"] for member in summary["members"]) == [
        "masks.cmg3",
        "optimized_poses.bin",
        "renderer.bin",
    ]
