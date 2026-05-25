# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from tac.optimization.family_agnostic_materializers import (
    materialize_renderer_payload_dfl1_candidate,
)
from tac.repo_io import sha256_bytes

REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACK_PATH = REPO / "submissions" / "robust_current" / "unpack_renderer_payload.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_renderer_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in {
            "renderer.bin": b"renderer" * 4096,
            "masks.mkv": b"mask" * 8192,
            "optimized_poses.bin": b"pose" * 1800,
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return path


def _write_renderer_archive_with_pose_values(path: Path) -> Path:
    pose_values = []
    for row in range(600):
        pose_values.extend([30.0 + row / 512.0, 1.0, -2.0, 0.5, -0.25, 0.125])
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in {
            "renderer.bin": b"renderer" * 4096,
            "masks.mkv": b"mask" * 8192,
            "optimized_poses.bin": pose_bytes,
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return path


def _write_renderer_archive_with_variable_pose_values(path: Path) -> Path:
    pose_values = []
    for row in range(600):
        pose_values.extend(
            [
                30.0 + row / 512.0,
                1.0,
                -2.0,
                0.5,
                -0.25,
                0.125,
            ]
        )
    pose_values[1] = 10.0
    pose_values[6 * 17 + 2] = -9.0
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in {
            "renderer.bin": b"renderer" * 4096,
            "masks.mkv": b"mask" * 8192,
            "optimized_poses.bin": pose_bytes,
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return path


def _write_public_pr64_like_renderer_archive(path: Path) -> Path:
    pose_values = []
    for row in range(600):
        pose_values.extend([30.0 + row / 512.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in {
            "renderer.bin": b"PK\x03\x04renderer" * 4096,
            "masks.mkv": b"\x12\x00\x0a\x0amask" * 8192,
            "optimized_poses.bin": pose_bytes,
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return path


def _write_renderer_archive_with_fractional_pose_values(path: Path) -> Path:
    pose_values = []
    for row in range(600):
        pose_values.extend(
            [
                30.003 + row / 512.0,
                0.3333,
                -1.234,
                0.127,
                -0.456,
                1.789,
            ]
        )
    pose_bytes = struct.pack("<" + "e" * len(pose_values), *pose_values)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, data in {
            "renderer.bin": b"renderer" * 4096,
            "masks.mkv": b"mask" * 8192,
            "optimized_poses.bin": pose_bytes,
        }.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    return path


def test_build_packed_archive_is_deterministic_and_single_member(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_test")
    source = _write_renderer_archive_with_pose_values(tmp_path / "source.zip")
    out1 = tmp_path / "out1.zip"
    out2 = tmp_path / "out2.zip"

    first = builder.build_packed_archive(source, out1)
    second = builder.build_packed_archive(source, out2)

    assert out1.read_bytes() == out2.read_bytes()
    assert first["output_archive_sha256"] == second["output_archive_sha256"]
    assert first["score_claim"] is False
    with zipfile.ZipFile(out1) as zf:
        assert zf.namelist() == ["renderer_payload.bin.br"]
        info = zf.infolist()[0]
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_STORED


def test_unpack_renderer_payload_restores_members_byte_exact(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_unpack_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"
    builder.build_packed_archive(source, packed)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)

    import brotli

    payload_br = archive_dir / "renderer_payload.bin.br"
    payload_bin = archive_dir / "renderer_payload.bin"
    payload_bin.write_bytes(brotli.decompress(payload_br.read_bytes()))
    payload_br.unlink()

    summary = unpacker.unpack_renderer_payload(archive_dir)
    with zipfile.ZipFile(source) as zf:
        expected = {name: zf.read(name) for name in zf.namelist()}
    assert {m["name"] for m in summary["members"]} == set(expected)
    for name, data in expected.items():
        assert (archive_dir / name).read_bytes() == data


def test_unpack_renderer_payload_rejects_ambiguous_containers(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_ambiguous_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_ambiguous_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"
    builder.build_packed_archive(source, packed)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    (archive_dir / "renderer_payload.bin").write_bytes(b"duplicate")

    with pytest.raises(ValueError, match="ambiguous renderer payload containers"):
        unpacker.unpack_renderer_payload(archive_dir)


def test_unpack_renderer_payload_rejects_sha_mismatch(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_sha_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_sha_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    members = builder.read_source_members(source)
    ordered = builder.ordered_runtime_members(members)
    payload, _ = builder.build_renderer_payload(
        ordered,
        source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pose_codec="raw",
    )
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "renderer_payload.bin").write_bytes(bytes(tampered))

    with pytest.raises(ValueError, match="SHA mismatch"):
        unpacker.unpack_renderer_payload(archive_dir)


def test_unpack_renderer_payload_rejects_unsafe_member_path(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_path_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_path_test")
    payload, _ = builder.build_renderer_payload(
        [("../escape.bin", b"bad")],
        source_archive_sha256="0" * 64,
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "renderer_payload.bin").write_bytes(payload)

    with pytest.raises(ValueError, match="unsafe"):
        unpacker.unpack_renderer_payload(archive_dir)


def test_submission_archive_detects_packed_payload_manifest(tmp_path: Path) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_manifest_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"
    builder.build_packed_archive(source, packed)

    manifest = detect_pose_manifest(packed)
    assert manifest.renderer_payload_bin_br is True
    result = validate_archive(packed, manifest, strict=True)
    assert result.valid


def test_short_p_payload_member_round_trips_and_validates(tmp_path: Path) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_short_member_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_short_member_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"
    result = builder.build_packed_archive(source, packed, payload_member_name="p")

    assert result["payload_member"] == "p"
    with zipfile.ZipFile(packed) as zf:
        assert zf.namelist() == ["p"]
    manifest = detect_pose_manifest(packed)
    assert manifest.renderer_payload_p is True
    assert validate_archive(packed, manifest, strict=True).valid

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    with zipfile.ZipFile(source) as zf:
        expected = {name: zf.read(name) for name in zf.namelist()}
    assert {m["name"] for m in summary["members"]} == set(expected)
    for name, data in expected.items():
        assert (archive_dir / name).read_bytes() == data


def test_native_renderer_payload_dfl1_round_trips_and_validates(tmp_path: Path) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_native_dfl1_test")
    source = tmp_path / "source.zip"
    payloads = {
        "renderer.bin": b"renderer" * 4096,
        "masks.mkv": b"mask" * 4096,
        "optimized_poses.pt": b"pose" * 2048,
    }
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in payloads.items():
            zf.writestr(name, payload)
    packed = tmp_path / "packed.zip"

    result = materialize_renderer_payload_dfl1_candidate(
        archive_path=source,
        output_archive=packed,
        runtime_consumption_proof_out=tmp_path / "proof.json",
        repo_root=REPO,
    )

    assert result["candidate_archive"]["bytes"] < result["source_archive"]["bytes"]
    assert validate_archive(packed, detect_pose_manifest(packed), strict=True).valid
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)
    assert summary["schema"] == "renderer_payload_fixed3_deflate_sequence_v1"
    assert summary["payload_format"] == "native_dfl1_fixed3"
    assert {m["name"]: m["sha256"] for m in summary["members"]} == {
        name: sha256_bytes(payload) for name, payload in payloads.items()
    }
    for name, payload in payloads.items():
        assert (archive_dir / name).read_bytes() == payload


def test_native_renderer_payload_dfl1_requires_complete_logical_members(
    tmp_path: Path,
) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    renderer_stream = compressor.compress(b"renderer" * 4096) + compressor.flush()
    archive = tmp_path / "bad_dfl1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"DFL1" + renderer_stream)

    result = validate_archive(archive, detect_pose_manifest(archive), strict=True)

    assert not result.valid
    assert any("renderer payload preflight failed" in error for error in result.errors)


def test_native_renderer_payload_dfl1_rejects_empty_required_member(
    tmp_path: Path,
) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    streams = []
    for payload in (b"renderer" * 4096, b"", b"pose" * 2048):
        compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
        streams.append(compressor.compress(payload) + compressor.flush())
    archive = tmp_path / "empty_mask_dfl1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("p", b"DFL1" + b"".join(streams))

    result = validate_archive(archive, detect_pose_manifest(archive), strict=True)

    assert not result.valid
    assert any("logical mask member is empty" in error for error in result.errors)


def test_build_submission_archive_accepts_renderer_payload_p_source(tmp_path: Path) -> None:
    from tac.submission_archive import (
        RENDERER_PACKED_PAYLOAD_SHORT_BROTLI_MANIFEST,
        build_submission_archive,
    )

    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer" * 4096)
        zf.writestr("masks.mkv", b"mask" * 4096)
        zf.writestr("optimized_poses.pt", b"pose" * 2048)
    packed = tmp_path / "packed.zip"
    materialize_renderer_payload_dfl1_candidate(
        archive_path=source,
        output_archive=packed,
        runtime_consumption_proof_out=tmp_path / "proof.json",
        repo_root=REPO,
    )
    payload_path = tmp_path / "p"
    with zipfile.ZipFile(packed) as zf:
        payload_path.write_bytes(zf.read("p"))

    result = build_submission_archive(
        tmp_path / "rebuilt.zip",
        renderer_payload_p=payload_path,
        manifest=RENDERER_PACKED_PAYLOAD_SHORT_BROTLI_MANIFEST,
    )

    assert result.valid
    assert result.files_found == {"p": payload_path.stat().st_size}


def test_submission_archive_detects_per_member_brotli_renderer_manifest(tmp_path: Path) -> None:
    import brotli

    from tac.submission_archive import detect_pose_manifest, validate_archive

    archive = tmp_path / "archive.zip"
    members = {
        "renderer.bin.br": brotli.compress(b"renderer" * 4096),
        "masks.mkv.br": brotli.compress(b"mask" * 8192),
        "optimized_poses.bin.br": brotli.compress(b"pose" * 1800),
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, data)

    manifest = detect_pose_manifest(archive)
    result = validate_archive(archive, manifest, strict=True)
    assert result.valid
    assert set(result.files_found) == set(members)


def test_lossless_pose_col_delta_codec_round_trips_exact_bytes(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_pose_codec_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_pose_codec_test")
    source = _write_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"
    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_FP16_COL_DELTA_CODEC,
    )
    assert result["pose_codec"] == builder.POSE_FP16_COL_DELTA_CODEC

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)

    import brotli

    payload_br = archive_dir / "renderer_payload.bin.br"
    payload_bin = archive_dir / "renderer_payload.bin"
    payload_bin.write_bytes(brotli.decompress(payload_br.read_bytes()))
    payload_br.unlink()
    unpacker.unpack_renderer_payload(archive_dir)

    with zipfile.ZipFile(source) as zf:
        assert (archive_dir / "optimized_poses.bin").read_bytes() == zf.read("optimized_poses.bin")


def test_velocity_only_pose_codec_decodes_declared_lossy_reconstruction(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_velocity_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_velocity_test")
    source = _write_renderer_archive_with_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_FP16_VELOCITY_ONLY_CODEC,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_FP16_VELOCITY_ONLY_CODEC
    assert pose_meta["lossy"] is True
    assert pose_meta["source_decoded_sha256"]
    assert pose_meta["decoded_sha256"] != pose_meta["source_decoded_sha256"]
    assert pose_meta["pose_error_stats"]["rows"] == 600

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)

    import brotli

    payload_br = archive_dir / "renderer_payload.bin.br"
    payload_bin = archive_dir / "renderer_payload.bin"
    payload_bin.write_bytes(brotli.decompress(payload_br.read_bytes()))
    payload_br.unlink()
    unpacker.unpack_renderer_payload(archive_dir)

    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]
    values = struct.unpack("<" + "e" * (len(decoded) // 2), decoded)
    assert len(values) == 600 * 6
    assert values[0] >= 20.0
    for row in range(600):
        assert values[row * 6 + 1 : row * 6 + 6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_qpose14_col_delta_pose_codec_quantizes_all_channels(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_qpose14_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_qpose14_test")
    source = _write_renderer_archive_with_fractional_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_QPOSE14_COL_DELTA_CODEC,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_QPOSE14_COL_DELTA_CODEC
    assert pose_meta["lossy"] is True
    assert pose_meta["source_decoded_sha256"]
    assert pose_meta["decoded_sha256"] != pose_meta["source_decoded_sha256"]
    stats = pose_meta["pose_error_stats"]
    assert stats["rows"] == 600
    assert stats["max_abs_by_dim"][0] <= 1.0 / 1024.0
    assert max(stats["max_abs_by_dim"][1:]) <= 1.0 / 4096.0

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)

    import brotli

    payload_br = archive_dir / "renderer_payload.bin.br"
    payload_bin = archive_dir / "renderer_payload.bin"
    payload_bin.write_bytes(brotli.decompress(payload_br.read_bytes()))
    payload_br.unlink()
    unpacker.unpack_renderer_payload(archive_dir)

    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]
    values = struct.unpack("<" + "e" * (len(decoded) // 2), decoded)
    assert len(values) == 600 * 6
    assert values[0] >= 20.0


def test_compact_fixed3_payload_round_trips_pose_delta_p_member(tmp_path: Path) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_compact_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_compact_test")
    source = _write_renderer_archive_with_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_FP16_COL_DELTA_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_RP2_FIXED3,
    )

    assert result["payload_format"] == builder.PAYLOAD_FORMAT_RP2_FIXED3
    assert result["payload_member"] == "p"
    with zipfile.ZipFile(packed) as zf:
        assert zf.namelist() == ["p"]
    assert validate_archive(packed, detect_pose_manifest(packed), strict=True).valid

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    unpacker.unpack_renderer_payload(archive_dir)

    with zipfile.ZipFile(source) as zf:
        expected = {name: zf.read(name) for name in zf.namelist()}
    for name, data in expected.items():
        assert (archive_dir / name).read_bytes() == data


def test_compact_fixed3_qpose14_payload_decodes_declared_reconstruction(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_compact_qpose_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_compact_qpose_test")
    source = _write_renderer_archive_with_fractional_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_QPOSE14_COL_DELTA_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_RP2_FIXED3,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_QPOSE14_COL_DELTA_CODEC
    assert pose_meta["lossy"] is True

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    unpacker.unpack_renderer_payload(archive_dir)

    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]


def test_compact_fixed3_qp1_payload_decodes_velocity_only_reconstruction(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_compact_qp1_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_compact_qp1_test")
    source = _write_renderer_archive_with_fractional_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_QP1_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_RP2_FIXED3,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_QP1_CODEC
    assert pose_meta["lossy"] is True
    assert pose_meta["bytes"] < 7208

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    unpacker.unpack_renderer_payload(archive_dir)

    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]
    values = struct.unpack("<" + "e" * (len(decoded) // 2), decoded)
    assert len(values) == 600 * 6
    assert values[0] >= 20.0
    for row in range(600):
        assert values[row * 6 + 1 : row * 6 + 6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_pr64_len_table_payload_round_trips_qpose14_p_member(tmp_path: Path) -> None:
    from tac.submission_archive import detect_pose_manifest, validate_archive

    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_pr64_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_pr64_test")
    source = _write_renderer_archive_with_fractional_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_QPOSE14_COL_DELTA_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_PR64_LEN_TABLE,
    )

    assert result["payload_format"] == builder.PAYLOAD_FORMAT_PR64_LEN_TABLE
    assert result["header"]["schema"] == builder.PR64_LEN_TABLE_SCHEMA
    with zipfile.ZipFile(packed) as zf:
        assert zf.namelist() == ["p"]
    assert validate_archive(packed, detect_pose_manifest(packed), strict=True).valid

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == builder.PR64_LEN_TABLE_SCHEMA
    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]


def test_pr64_len_table_renderer_first_bare_velocity_delta_pose(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_pr64_bare_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_pr64_bare_test")
    source = _write_public_pr64_like_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_PR64_LEN_TABLE,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC
    assert pose_meta["bytes"] == 1200

    import brotli

    with zipfile.ZipFile(packed) as zf:
        payload = brotli.decompress(zf.read("p"))
    renderer_len, mask_len, pose_len = struct.unpack_from("<III", payload, 0)
    assert (renderer_len, mask_len, pose_len) == (
        len(b"PK\x03\x04renderer" * 4096),
        len(b"\x12\x00\x0a\x0amask" * 8192),
        1200,
    )

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["payload_format"] == builder.PAYLOAD_FORMAT_PR64_LEN_TABLE
    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]


def test_public_pr64_mask_first_writer_emits_bare_velocity_delta_pose(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_public_pr64_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_public_pr64_writer_test")
    source = _write_public_pr64_like_renderer_archive(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC,
        payload_member_name="p",
        payload_format=builder.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
    )

    assert result["payload_format"] == builder.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE
    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC
    assert pose_meta["bytes"] == 1200
    assert pose_meta["pose_error_stats"]["max_abs_by_dim"] == [0.0] * 6

    import brotli

    with zipfile.ZipFile(packed) as zf:
        payload = brotli.decompress(zf.read("p"))
    mask_len, renderer_len, pose_len = struct.unpack_from("<III", payload, 0)
    assert (mask_len, renderer_len, pose_len) == (
        len(b"\x12\x00\x0a\x0amask" * 8192),
        len(b"PK\x03\x04renderer" * 4096),
        1200,
    )

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == builder.PR64_LEN_TABLE_SCHEMA
    assert summary["payload_format"] == builder.PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE
    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]


def test_public_pr64_velocity_delta_decode_uses_int32_cumsum_not_uint16_wrap() -> None:
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_public_pr64_cumsum_test")

    payload = struct.pack("<Hhh", 1000, -1100, 50)
    decoded = unpacker._decode_public_pr64_velocity_delta(payload)
    values = struct.unpack("<" + "e" * 18, decoded)

    expected_velocity = [
        1000 / 512.0 + 20.0,
        -100 / 512.0 + 20.0,
        -50 / 512.0 + 20.0,
    ]
    for row, expected in enumerate(expected_velocity):
        assert values[row * 6] == pytest.approx(expected, abs=1e-2)
        assert values[row * 6 + 1: row * 6 + 6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_public_pr63_qpose14_archive_round_trips_to_runtime_members(tmp_path: Path) -> None:
    archive = (
        REPO
        / "experiments/results/top_submission_current_floor_20260501/external_archives/"
        / "pr63_qpose14_archive.zip"
    )
    if not archive.exists():
        pytest.skip("public qpose14 archive fixture is not present")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_public_pr63_test")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert {m["name"]: m["bytes"] for m in summary["members"]} == {
        "renderer.bin": 91582,
        "masks.mkv": 223385,
        "optimized_poses.bin": 7200,
    }
    assert hashlib.sha256((archive_dir / "renderer.bin").read_bytes()).hexdigest() == (
        "d97849d15859ae013ec983de8c1e2f638e63f3876fef658a8b7781bcfaa16a5f"
    )
    values = struct.unpack("<" + "e" * 3600, (archive_dir / "optimized_poses.bin").read_bytes())
    assert len(values) == 600 * 6
    assert min(values[0::6]) > 20.0


def test_public_pr64_unified_brotli_archive_round_trips_to_runtime_members(tmp_path: Path) -> None:
    archive = (
        REPO
        / "experiments/results/top_submission_current_floor_20260501/external_archives/"
        / "pr64_unified_brotli_archive.zip"
    )
    if not archive.exists():
        pytest.skip("public unified_brotli archive fixture is not present")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_public_pr64_test")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert {m["name"]: m["bytes"] for m in summary["members"]} == {
        "renderer.bin": 91582,
        "masks.mkv": 223385,
        "optimized_poses.bin": 7200,
    }
    assert hashlib.sha256((archive_dir / "masks.mkv").read_bytes()).hexdigest() == (
        "a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb"
    )
    values = struct.unpack("<" + "e" * 3600, (archive_dir / "optimized_poses.bin").read_bytes())
    assert len(values) == 600 * 6
    assert values[1:6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_public_pr67_qzs3_qp1_archive_round_trips_to_runtime_members(tmp_path: Path) -> None:
    archive = REPO / "reports/raw/leaderboard_intel_20260501/pr67_archive.zip"
    if not archive.exists():
        pytest.skip("public PR67 qpose14_qzs3 archive fixture is not present")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_public_pr67_test")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert summary["payload_format"] == "public_pr67_qzs3_qp1_fixed_slices"
    assert {m["name"]: m["bytes"] for m in summary["members"]} == {
        "renderer.bin": 59288,
        "masks.mkv": 223385,
        "optimized_poses.bin": 7200,
    }
    renderer = (archive_dir / "renderer.bin").read_bytes()
    assert renderer.startswith(b"QZS3")
    from tac.quantizr_qzs3_codec import decode_qzs3_state_dict

    decoded = decode_qzs3_state_dict(renderer, device="cpu")
    assert len(decoded) > 0
    values = struct.unpack("<" + "e" * 3600, (archive_dir / "optimized_poses.bin").read_bytes())
    assert len(values) == 600 * 6
    assert values[1:6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_generated_pr67_fixed_slice_qzs3_qp1_archive_round_trips(tmp_path: Path) -> None:
    archive = (
        REPO
        / "experiments/results/public_floor_qzs3_qp1_packer_20260502/"
        / "pr63_qzs3_qp1_fixedslice/archive.zip"
    )
    if not archive.exists():
        pytest.skip("generated PR67 fixed-slice QZS3/QP1 archive fixture is not present")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_generated_pr67_test")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert summary["payload_format"] == "public_pr67_qzs3_qp1_fixed_slices"
    assert {m["name"]: m["bytes"] for m in summary["members"]} == {
        "renderer.bin": 59288,
        "masks.mkv": 223385,
        "optimized_poses.bin": 7200,
    }
    renderer = (archive_dir / "renderer.bin").read_bytes()
    assert renderer.startswith(b"QZS3")
    values = struct.unpack("<" + "e" * 3600, (archive_dir / "optimized_poses.bin").read_bytes())
    assert len(values) == 600 * 6
    assert values[1:6] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_generated_pr67_line_search_checkpoint_round_trips(tmp_path: Path) -> None:
    archive = (
        REPO
        / "experiments/results/vast_harvest/"
        / "line_search_qzs3_qp1_fixedslice_20260502T0049Z/"
        / "archive.accepted_latest.zip"
    )
    if not archive.exists():
        pytest.skip("generated PR67 line-search checkpoint fixture is not present")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_generated_pr67_ls_test")

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    assert summary["payload_format"] == "public_pr67_qzs3_qp1_fixed_slices"
    renderer = (archive_dir / "renderer.bin").read_bytes()
    poses = (archive_dir / "optimized_poses.bin").read_bytes()
    assert renderer.startswith(b"QZS3")
    assert len(poses) == 600 * 6 * 2


def test_velocity_residual_topk_pose_codec_preserves_charged_atoms(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_residual_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_residual_test")
    source = _write_renderer_archive_with_variable_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        pose_codec=builder.POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
        pose_residual_topk=2,
    )

    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC
    assert pose_meta["lossy"] is True
    assert pose_meta["pose_residual_topk"] == 2
    assert pose_meta["pose_error_stats"]["rows"] == 600

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        zf.extractall(archive_dir)

    import brotli

    payload_br = archive_dir / "renderer_payload.bin.br"
    payload_bin = archive_dir / "renderer_payload.bin"
    payload_bin.write_bytes(brotli.decompress(payload_br.read_bytes()))
    payload_br.unlink()
    unpacker.unpack_renderer_payload(archive_dir)

    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]
    values = struct.unpack("<" + "e" * (len(decoded) // 2), decoded)
    assert len(values) == 600 * 6
    assert values[1] == 10.0
    assert values[6 * 17 + 2] == -9.0
    assert values[0] >= 20.0


def test_pr64_len_table_velocity_residual_topk_pose_codec_round_trips(tmp_path: Path) -> None:
    builder = _load_module(BUILDER_PATH, "_renderer_payload_builder_pr64_residual_test")
    unpacker = _load_module(UNPACK_PATH, "_renderer_payload_unpack_pr64_residual_test")
    source = _write_renderer_archive_with_variable_pose_values(tmp_path / "source.zip")
    packed = tmp_path / "packed.zip"

    result = builder.build_packed_archive(
        source,
        packed,
        payload_format=builder.PAYLOAD_FORMAT_PR64_LEN_TABLE,
        payload_member_name="p",
        pose_codec=builder.POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
        pose_residual_topk=2,
    )

    assert result["payload_format"] == builder.PAYLOAD_FORMAT_PR64_LEN_TABLE
    assert result["payload_member"] == "p"
    pose_meta = next(
        m for m in result["header"]["members"] if m["name"] == "optimized_poses.bin"
    )
    assert pose_meta["codec"] == builder.POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC
    assert pose_meta["pose_residual_topk"] == 2

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    with zipfile.ZipFile(packed) as zf:
        assert zf.namelist() == ["p"]
        zf.extractall(archive_dir)
    summary = unpacker.unpack_renderer_payload(archive_dir)

    assert summary["schema"] == unpacker.PR64_LEN_TABLE_SCHEMA
    decoded = (archive_dir / "optimized_poses.bin").read_bytes()
    assert hashlib.sha256(decoded).hexdigest() == pose_meta["decoded_sha256"]
    values = struct.unpack("<" + "e" * (len(decoded) // 2), decoded)
    assert values[1] == 10.0
    assert values[6 * 17 + 2] == -9.0
