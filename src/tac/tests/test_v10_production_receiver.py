"""Behavioral tests for the scorer-free V10 production archive/receiver."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl.v10_production_receiver import (
    DESCRIPTION_FRAME0_POLICY_ID,
    MEMBER_NAME,
    PREDICTOR_RESIDUAL_Y_CODEC_ID,
    PREFIX,
    RESIDUAL_RECORD,
    SECTION_LENGTH,
    ArchiveBuildResult,
    ProductionReceiverError,
    build_packet,
    build_production_archive,
    decode_y_plane_pair,
    decode_y_planes,
    inflate_archive,
    parse_packet,
    tree_sha256,
)

CAMERA_H = 8
CAMERA_W = 10
SCORER_H = 3
SCORER_W = 4


def _planes(pair_count: int = 4) -> np.ndarray:
    values = np.arange(pair_count * SCORER_H * SCORER_W * 3, dtype=np.uint16)
    return ((values * 37 + 11) % 256).astype(np.uint8).reshape(pair_count, SCORER_H, SCORER_W, 3)


def _build(tmp_path: Path, *, planes: np.ndarray | None = None) -> ArchiveBuildResult:
    archive_dir = tmp_path / "archive"
    return build_production_archive(
        _planes() if planes is None else planes,
        archive_path=archive_dir / "archive.zip",
        camera_height=CAMERA_H,
        camera_width=CAMERA_W,
    )


def _names(tmp_path: Path) -> Path:
    path = tmp_path / "video_names.txt"
    path.write_text("0.mkv\n", encoding="utf-8")
    return path


def _packet_from_archive(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as archive:
        assert archive.namelist() == [MEMBER_NAME]
        return archive.read(MEMBER_NAME)


def test_packet_roundtrip_binds_every_section_length_hash_and_total() -> None:
    packet_bytes = build_packet(_planes(), camera_height=CAMERA_H, camera_width=CAMERA_W)
    parsed = parse_packet(packet_bytes)
    assert parsed.header["packet_bytes"] == len(packet_bytes)
    assert parsed.header["section_count"] == 2
    assert [section.section_id for section in parsed.sections] == [
        "y_description",
        "frame0_policy",
    ]
    assert sum(len(section.payload) for section in parsed.sections) == parsed.header["counted_section_payload_bytes"]
    np.testing.assert_array_equal(decode_y_planes(parsed), _planes())


def test_parser_refuses_length_hash_truncation_and_trailing_bytes() -> None:
    packet = build_packet(_planes(), camera_height=CAMERA_H, camera_width=CAMERA_W)
    _magic, _version, header_length = PREFIX.unpack_from(packet)
    first_length_offset = PREFIX.size + header_length
    malformed_length = bytearray(packet)
    declared = SECTION_LENGTH.unpack_from(malformed_length, first_length_offset)[0]
    SECTION_LENGTH.pack_into(malformed_length, first_length_offset, declared + 1)
    with pytest.raises(ProductionReceiverError, match="framed/header length"):
        parse_packet(bytes(malformed_length))

    corrupt_payload = bytearray(packet)
    payload_offset = first_length_offset + SECTION_LENGTH.size
    corrupt_payload[payload_offset] ^= 1
    with pytest.raises(ProductionReceiverError, match="payload hash"):
        parse_packet(bytes(corrupt_payload))
    with pytest.raises(ProductionReceiverError, match=r"truncated|payload"):
        parse_packet(packet[:-1])
    with pytest.raises(ProductionReceiverError, match="trailing"):
        parse_packet(packet + b"x")


def test_witness_y_stub_is_typed_but_decode_refuses() -> None:
    packet = parse_packet(
        build_packet(
            _planes(),
            camera_height=CAMERA_H,
            camera_width=CAMERA_W,
            y_codec_id="witness-y-stub",
        )
    )
    with pytest.raises(ProductionReceiverError, match="typed refusal"):
        decode_y_planes(packet)


def test_brotli_y_codec_roundtrips_exact_decoded_bytes() -> None:
    packet = parse_packet(
        build_packet(
            _planes(),
            camera_height=CAMERA_H,
            camera_width=CAMERA_W,
            y_codec_id="brotli-y",
        )
    )
    assert packet.section("y_description").codec_id == "brotli-y"
    np.testing.assert_array_equal(decode_y_planes(packet), _planes())


def test_predictor_residual_codec_roundtrips_typed_two_plane_description() -> None:
    frame0 = np.flip(_planes(), axis=2).copy()
    frame1 = _planes()
    packet = parse_packet(
        build_packet(
            frame1,
            frame0_y_planes=frame0,
            camera_height=CAMERA_H,
            camera_width=CAMERA_W,
            y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
            predictor_modes=[
                "previous-plane-copy.v1",
                "affine6-q12.v1",
                "spatial-smooth-121.v1",
                "previous-plane-copy.v1",
            ],
        )
    )
    assert packet.header["frame0_policy_id"] == DESCRIPTION_FRAME0_POLICY_ID
    pair = decode_y_plane_pair(packet)
    np.testing.assert_array_equal(pair.frame0, frame0)
    np.testing.assert_array_equal(pair.frame1, frame1)
    # Legacy helper remains frame-1-only.
    np.testing.assert_array_equal(decode_y_planes(packet), frame1)
    section = packet.section("y_description")
    assert section.video_derived is True
    assert section.decoded_byte_length == frame0.nbytes + frame1.nbytes
    assert packet.header["video_derived_payload_bytes"] == sum(
        len(item.payload) for item in packet.sections if item.video_derived
    )


def test_legacy_codec_refuses_predictor_only_arguments() -> None:
    with pytest.raises(ProductionReceiverError, match="legacy y codecs refuse"):
        build_packet(
            _planes(),
            frame0_y_planes=_planes(),
            camera_height=CAMERA_H,
            camera_width=CAMERA_W,
        )
    with pytest.raises(ProductionReceiverError, match="requires frame0_y_planes"):
        build_packet(
            _planes(),
            camera_height=CAMERA_H,
            camera_width=CAMERA_W,
            y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        )


def test_archive_is_deterministic_single_member_and_manifest_is_write_once(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path)
    second = _build(tmp_path / "independent")
    assert first.archive_sha256 == second.archive_sha256
    assert first.packet_sha256 == second.packet_sha256
    assert first.manifest == second.manifest
    with zipfile.ZipFile(first.archive_path, "r") as archive:
        assert archive.namelist() == [MEMBER_NAME]
        assert archive.getinfo(MEMBER_NAME).compress_type == zipfile.ZIP_STORED
    with pytest.raises(ProductionReceiverError, match="write-once manifest"):
        _build(tmp_path)
    changed = _planes().copy()
    changed[0, 0, 0, 0] ^= 1
    collision = _build(tmp_path / "archive-collision")
    collision.manifest_path.unlink()
    with pytest.raises(ProductionReceiverError, match="drifted"):
        _build(tmp_path / "archive-collision", planes=changed)


def test_archive_reader_refuses_zip_compressed_member(tmp_path: Path) -> None:
    built = _build(tmp_path)
    packet = _packet_from_archive(built.archive_path)
    with zipfile.ZipFile(built.archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MEMBER_NAME, packet)
    with pytest.raises(ProductionReceiverError, match="stored without ZIP compression"):
        inflate_archive(built.archive_path.parent, tmp_path / "compressed-out", _names(tmp_path))


def test_double_decode_has_exact_expected_raw_and_identical_tree_hashes(
    tmp_path: Path,
) -> None:
    built = _build(tmp_path)
    names = _names(tmp_path)
    first = inflate_archive(built.archive_path.parent, tmp_path / "decode-a", names)
    second = inflate_archive(built.archive_path.parent, tmp_path / "decode-b", names)
    assert first.completed is second.completed is True
    assert first.raw_sha256 == second.raw_sha256
    assert first.tree_sha256 == second.tree_sha256
    assert first.tree_sha256 == tree_sha256(tmp_path / "decode-a")
    assert first.raw_path is not None
    raw = first.raw_path.read_bytes()
    frame_bytes = CAMERA_H * CAMERA_W * 3
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    planes = _planes()
    assert len(raw) == len(planes) * frame_bytes * 2
    for pair_index, y_plane in enumerate(planes):
        offset = pair_index * frame_bytes * 2
        frame0 = np.frombuffer(raw[offset : offset + frame_bytes], dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3)
        frame1 = np.frombuffer(raw[offset + frame_bytes : offset + 2 * frame_bytes], dtype=np.uint8).reshape(
            CAMERA_H, CAMERA_W, 3
        )
        np.testing.assert_array_equal(frame0, frame1)
        numerators, denominator = operator.apply_numerators(frame1)
        np.testing.assert_array_equal(numerators, y_plane.astype(np.int64) * denominator)


def test_predictor_archive_inflates_distinct_exact_frame0_and_frame1_planes(tmp_path: Path) -> None:
    frame0_planes = np.flip(_planes(), axis=1).copy()
    frame1_planes = _planes()
    archive_dir = tmp_path / "predictor-archive"
    built = build_production_archive(
        frame1_planes,
        frame0_y_planes=frame0_planes,
        archive_path=archive_dir / "archive.zip",
        camera_height=CAMERA_H,
        camera_width=CAMERA_W,
        y_codec_id=PREDICTOR_RESIDUAL_Y_CODEC_ID,
        predictor_modes="spatial-smooth-121.v1",
    )
    parsed = parse_packet(_packet_from_archive(built.archive_path))
    assert parsed.header["counted_section_payload_bytes"] == sum(len(section.payload) for section in parsed.sections)
    result = inflate_archive(archive_dir, tmp_path / "predictor-out", _names(tmp_path))
    assert result.raw_path is not None
    raw = result.raw_path.read_bytes()
    frame_bytes = CAMERA_H * CAMERA_W * 3
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    for pair_index in range(len(frame1_planes)):
        offset = pair_index * frame_bytes * 2
        frame0 = np.frombuffer(raw[offset : offset + frame_bytes], dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3)
        frame1 = np.frombuffer(raw[offset + frame_bytes : offset + 2 * frame_bytes], dtype=np.uint8).reshape(
            CAMERA_H, CAMERA_W, 3
        )
        numerator0, denominator0 = operator.apply_numerators(frame0)
        numerator1, denominator1 = operator.apply_numerators(frame1)
        assert denominator0 == denominator1
        np.testing.assert_array_equal(numerator0, frame0_planes[pair_index].astype(np.int64) * denominator0)
        np.testing.assert_array_equal(numerator1, frame1_planes[pair_index].astype(np.int64) * denominator1)


def test_interrupted_prefix_resumes_after_revalidating_stages(tmp_path: Path) -> None:
    built = _build(tmp_path)
    names = _names(tmp_path)
    output = tmp_path / "resume"
    interrupted = inflate_archive(built.archive_path.parent, output, names, stop_after_pairs=2)
    assert interrupted.completed is False
    assert interrupted.pair_stages_preserved == 2
    resumed = inflate_archive(built.archive_path.parent, output, names)
    assert resumed.completed is True
    assert resumed.pair_stages_preserved == len(_planes())


@pytest.mark.parametrize("leg", ["bin", "json"])
def test_edited_preserved_stage_or_state_refuses(tmp_path: Path, leg: str) -> None:
    built = _build(tmp_path)
    names = _names(tmp_path)
    output = tmp_path / f"edited-{leg}"
    inflate_archive(built.archive_path.parent, output, names, stop_after_pairs=1)
    state_root = output / ".v10-production-receiver" / "0"
    path = state_root / f"pair-000000.{leg}"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ProductionReceiverError, match="drifted"):
        inflate_archive(built.archive_path.parent, output, names)


def test_edited_archive_bytes_refuse_before_decode(tmp_path: Path) -> None:
    built = _build(tmp_path)
    names = _names(tmp_path)
    archive_bytes = bytearray(built.archive_path.read_bytes())
    archive_bytes[len(archive_bytes) // 2] ^= 1
    built.archive_path.write_bytes(archive_bytes)
    with pytest.raises(ProductionReceiverError):
        inflate_archive(built.archive_path.parent, tmp_path / "bad-archive", names)


def test_signed_residual_saturates_in_unowned_nullspace_coordinate(tmp_path: Path) -> None:
    planes = np.zeros((1, SCORER_H, SCORER_W, 3), dtype=np.uint8)
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_H,
        scorer_w=SCORER_W,
    )
    owned = np.zeros((CAMERA_H, CAMERA_W), dtype=bool)
    for row_support in operator.row_supports:
        for col_support in operator.col_supports:
            owned[np.ix_(row_support.indices, col_support.indices)] = True
    unowned_row, unowned_col = np.argwhere(~owned)[0]
    residual = np.zeros((1, CAMERA_H, CAMERA_W, 3), dtype="<i2")
    residual[0, unowned_row, unowned_col, 0] = 30_000
    archive_dir = tmp_path / "residual-archive"
    built = build_production_archive(
        planes,
        archive_path=archive_dir / "archive.zip",
        camera_height=CAMERA_H,
        camera_width=CAMERA_W,
        quotient_residual=residual,
    )
    parsed = parse_packet(_packet_from_archive(built.archive_path))
    assert len(parsed.section("quotient_residual").payload) == RESIDUAL_RECORD.size
    result = inflate_archive(built.archive_path.parent, tmp_path / "residual-out", _names(tmp_path))
    assert result.raw_path is not None
    frame = np.frombuffer(result.raw_path.read_bytes()[: CAMERA_H * CAMERA_W * 3], dtype=np.uint8).reshape(
        CAMERA_H, CAMERA_W, 3
    )
    assert frame[unowned_row, unowned_col, 0] == 255
    numerators, denominator = operator.apply_numerators(frame)
    np.testing.assert_array_equal(numerators, planes[0].astype(np.int64) * denominator)


def test_receiver_source_has_no_scorer_or_torch_import_path() -> None:
    source = Path(__file__).parents[1] / "witness_dsl" / "v10_production_receiver.py"
    lowered = source.read_text(encoding="utf-8").lower()
    assert "import torch" not in lowered
    assert "import segnet" not in lowered
    assert "import posenet" not in lowered
    assert "distortionnet" not in lowered


@pytest.mark.parametrize("video_name", ["../escape.mkv", "/absolute.mkv", " 0.mkv"])
def test_video_name_path_escape_refuses(tmp_path: Path, video_name: str) -> None:
    built = _build(tmp_path)
    names = tmp_path / "unsafe.txt"
    names.write_text(video_name + "\n", encoding="utf-8")
    with pytest.raises(ProductionReceiverError, match=r"video name|escape"):
        inflate_archive(built.archive_path.parent, tmp_path / "unsafe-out", names)


def test_header_unknown_field_and_non_integer_totals_refuse() -> None:
    packet = build_packet(_planes(), camera_height=CAMERA_H, camera_width=CAMERA_W)
    magic, version, header_length = PREFIX.unpack_from(packet)
    header_end = PREFIX.size + header_length
    header = json.loads(packet[PREFIX.size : header_end])
    header["counted_section_payload_bytes"] = float(header["counted_section_payload_bytes"])
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    rewritten = PREFIX.pack(magic, version, len(header_bytes)) + header_bytes + packet[header_end:]
    with pytest.raises(ProductionReceiverError, match=r"exact integer|packet total"):
        parse_packet(rewritten)
