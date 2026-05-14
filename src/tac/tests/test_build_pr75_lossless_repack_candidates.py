# SPDX-License-Identifier: MIT
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
BUILDER_PATH = REPO / "experiments" / "build_pr75_lossless_repack_candidates.py"
UNPACK_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"
INFLATE_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stored_zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo("p", (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def _p3_payload() -> tuple[bytes, dict[str, bytes]]:
    decoded = {
        "masks.mkv": b"\x12\x00\x0a\x0a" + b"m" * 96,
        "renderer.bin": b"QZS3" + b"r" * 96,
        "seg_tile_actions.bin": (
            (33).to_bytes(2, "little") + bytes([109, 92])
            + (36).to_bytes(2, "little") + bytes([109, 93])
        ),
        "optimized_poses.bin": b"QP1" + (5120).to_bytes(2, "little") + b"p" * 8,
    }
    mask_br = brotli.compress(decoded["masks.mkv"], quality=0)
    model_br = brotli.compress(decoded["renderer.bin"], quality=0)
    actions_br = brotli.compress(decoded["seg_tile_actions.bin"], quality=0)
    pose_br = brotli.compress(decoded["optimized_poses.bin"], quality=0)
    payload = (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(model_br), len(actions_br))
        + mask_br
        + model_br
        + actions_br
        + pose_br
    )
    return payload, decoded


def test_build_lossless_candidates_emits_p6_with_decoded_stream_parity(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_pr75_lossless_builder_test")
    unpacker = _load_module(UNPACK_PATH, "_pr75_lossless_unpacker_test")
    source_payload, _expected_raw = _p3_payload()
    source_archive = tmp_path / "source.zip"
    _stored_zip(source_archive, source_payload)
    _source_header, expected = unpacker._parse_payload(source_payload)

    summary = builder.build_lossless_candidates(
        source_archive=source_archive,
        output_dir=tmp_path / "out",
        params=[(0, 0, 10, 0)],
        unpacker=unpacker,
    )

    p6 = next(
        row
        for row in summary["candidates"]
        if row["candidate_id"] == "c082_p6_delta_varint_actions_stream_resweep"
    )
    assert p6["score_claim"] is False
    assert p6["decoded_stream_parity"] is True
    with zipfile.ZipFile(p6["archive_path"]) as zf:
        header, decoded = unpacker._parse_payload(zf.read("p"))
    assert header["payload_format"] == "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    for name, data in expected.items():
        assert decoded[name] == data

    manifest = json.loads(Path(p6["manifest_path"]).read_text())
    assert manifest["decoded_stream_parity"] is True
    assert manifest["decoded_stream_parity_detail"]["status"] == "passed"
    assert manifest["noop"] is False
    assert manifest["noop_status"] == "not_noop_repacked_payload"
    assert manifest["source_preserving"] is True
    assert (
        manifest["source_preservation"]["status"]
        == "lossless_decoded_stream_preserving_repack"
    )
    assert manifest["source_preservation"]["decoded_streams_byte_identical"] is True
    assert manifest["source_preservation"]["payload_byte_identical_to_source"] is False
    assert manifest["actions_delta_varint"]["record_count"] == 2


def test_p6_delta_varint_action_parser_rejects_malformed_streams() -> None:
    unpacker = _load_module(UNPACK_PATH, "_pr75_lossless_unpacker_malformed_test")
    cases = [
        (b"\x80\x00\x01\x02", "noncanonical"),
        (b"\x80", "truncated"),
        (b"\x01\x02", "ended inside record"),
        (b"\x10\x01\x02\x00\x03\x04", "trailing bytes"),
        (b"\x90\x4e\x01\x02", "exceeds max"),
    ]
    for raw, message in cases:
        with pytest.raises(ValueError, match=message):
            unpacker._decode_delta_varint_seg_tile_actions(  # noqa: SLF001
                brotli.compress(raw, quality=0),
                record_count=1,
            )


def test_malformed_self_describing_p6_fails_with_specific_error() -> None:
    unpacker = _load_module(UNPACK_PATH, "_pr75_lossless_unpacker_full_malformed_test")
    mask_br = brotli.compress(b"\x12\x00\x0a\x0a" + b"m" * 8, quality=0)
    model_br = brotli.compress(b"QZS3" + b"r" * 8, quality=0)
    actions_br = brotli.compress(b"\x80\x00\x01\x02", quality=0)
    pose_br = brotli.compress(b"QP1" + (5120).to_bytes(2, "little"), quality=0)
    payload = (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(model_br), len(actions_br), 1)
        + mask_br
        + model_br
        + actions_br
        + pose_br
    )
    with pytest.raises(ValueError, match="invalid self-describing PR75 payload:.*noncanonical"):
        unpacker._parse_payload(payload)  # noqa: SLF001


def _uvarint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def test_runtime_seg_tile_action_loader_accepts_sg2_and_tagged_records(tmp_path: Path) -> None:
    runtime = _load_module(INFLATE_PATH, "_pr75_runtime_action_loader_test")
    # SG2 stores records grouped by tile with frame deltas. It should decode
    # to the same runtime by_pair table as expanded 4-byte records.
    sg2_raw = (
        b"SG2"
        + _uvarint(109)
        + _uvarint(2)
        + _uvarint(33)
        + bytes([92])
        + _uvarint(3)
        + bytes([93])
    )
    archive = tmp_path / "sg2"
    archive.mkdir()
    (archive / "seg_tile_actions.br").write_bytes(brotli.compress(sg2_raw, quality=0))

    state = runtime._load_seg_tile_actions_from_archive_dir(archive, "cpu")  # noqa: SLF001

    assert state is not None
    assert state["record_size"] == 4
    assert state["record_count"] == 2
    assert state["by_pair"] == {33: [(109, 92)], 36: [(109, 93)]}

    tagged = tmp_path / "tagged"
    tagged.mkdir()
    tagged_raw = (
        b"TA5"
        + (33).to_bytes(2, "little")
        + (109).to_bytes(2, "little")
        + bytes([92])
    )
    (tagged / "seg_tile_actions.br").write_bytes(brotli.compress(tagged_raw, quality=0))

    tagged_state = runtime._load_seg_tile_actions_from_archive_dir(tagged, "cpu")  # noqa: SLF001

    assert tagged_state is not None
    assert tagged_state["record_size"] == 5
    assert tagged_state["record_count"] == 1
    assert tagged_state["by_pair"] == {33: [(109, 92)]}


def test_runtime_seg_tile_action_loader_accepts_charged_grid_header(tmp_path: Path) -> None:
    runtime = _load_module(INFLATE_PATH, "_pr75_runtime_action_loader_grid_test")
    # Fine-grid PR79 search payloads must carry tile size explicitly. Without
    # this charged header, inflate would apply tile ids as 32x32 records even
    # when search optimized 16x16 or 8x8 geometry.
    tile16_raw = (
        b"TG1"
        + (16).to_bytes(2, "little")
        + b"SG2"
        + _uvarint(767)
        + _uvarint(1)
        + _uvarint(599)
        + bytes([92])
    )
    archive = tmp_path / "grid16"
    archive.mkdir()
    (archive / "seg_tile_actions.br").write_bytes(brotli.compress(tile16_raw, quality=0))

    state = runtime._load_seg_tile_actions_from_archive_dir(archive, "cpu")  # noqa: SLF001

    assert state is not None
    assert state["tile_size"] == 16
    assert state["record_size"] == 5
    assert state["record_count"] == 1
    assert state["by_pair"] == {599: [(767, 92)]}


def test_runtime_seg_tile_action_loader_resolves_raw4_raw5_length_collision(tmp_path: Path) -> None:
    runtime = _load_module(INFLATE_PATH, "_pr75_runtime_action_loader_collision_test")
    raw = b"".join(
        (pair).to_bytes(2, "little") + bytes([109, 92])
        for pair in range(40)
    )
    assert len(raw) == 160
    archive = tmp_path / "raw4_collision"
    archive.mkdir()
    (archive / "seg_tile_actions.bin").write_bytes(raw)

    state = runtime._load_seg_tile_actions_from_archive_dir(archive, "cpu")  # noqa: SLF001

    assert state is not None
    assert state["record_size"] == 4
    assert state["record_count"] == 40
    assert state["by_pair"][0] == [(109, 92)]
    assert state["by_pair"][39] == [(109, 92)]


def test_runtime_seg_tile_action_loader_rejects_truly_ambiguous_untagged_payload(
    tmp_path: Path,
) -> None:
    runtime = _load_module(INFLATE_PATH, "_pr75_runtime_action_loader_ambiguous_test")
    raw = b"\x00" * 20
    archive = tmp_path / "ambiguous"
    archive.mkdir()
    (archive / "seg_tile_actions.bin").write_bytes(raw)

    with pytest.raises(ValueError, match="ambiguous seg tile action payload length"):
        runtime._load_seg_tile_actions_from_archive_dir(archive, "cpu")  # noqa: SLF001
