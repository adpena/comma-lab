from __future__ import annotations

import random
import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from tac.custom_binary_container import (
    PACT_MAGIC,
    pack_archive,
    pack_archive_minimal_zip,
    pack_archive_to_zip_compatible,
    unpack_archive,
)


LANE_A_ARCHIVE = Path("/Users/adpena/Projects/pact/experiments/results/lane_a_landed/archive_lane_a.zip")


def _lane_a_members() -> dict[str, bytes]:
    with zipfile.ZipFile(LANE_A_ARCHIVE, "r") as zf:
        return {info.filename: zf.read(info) for info in zf.infolist() if not info.is_dir()}


def _with_crc(body: bytes) -> bytes:
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def test_roundtrip_with_random_file_dict() -> None:
    rng = random.Random(20260429)
    file_dict = {
        "zero.bin": b"",
        "one.bin": rng.randbytes(1),
        "tiny.bin": rng.randbytes(17),
        "page.bin": rng.randbytes(4096),
        "blob.bin": rng.randbytes(65537),
    }

    packed = pack_archive(file_dict)

    assert unpack_archive(packed) == file_dict


def test_roundtrip_real_lane_a_archive_contents_bit_exact() -> None:
    members = _lane_a_members()

    unpacked = unpack_archive(pack_archive(members))

    assert unpacked == members
    assert set(unpacked) == {"renderer.bin", "masks.mkv", "optimized_poses.pt"}


def test_rejects_bad_magic() -> None:
    packed = bytearray(pack_archive({"renderer.bin": b"abc"}))
    packed[0] ^= 0xFF

    with pytest.raises(ValueError, match="bad magic"):
        unpack_archive(bytes(packed))


def test_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated header"):
        unpack_archive(PACT_MAGIC[:7])


def test_rejects_truncated_entry_header() -> None:
    body = PACT_MAGIC + b"\x00\x00\x00"

    with pytest.raises(ValueError, match="truncated entry"):
        unpack_archive(_with_crc(body))


def test_rejects_truncated_entry_payload() -> None:
    body = PACT_MAGIC + struct.pack(">II", 4, 10) + b"name" + b"abc"

    with pytest.raises(ValueError, match="truncated entry"):
        unpack_archive(_with_crc(body))


def test_rejects_crc_mismatch() -> None:
    packed = bytearray(pack_archive({"renderer.bin": b"abcdef"}))
    packed[-1] ^= 0x01

    with pytest.raises(ValueError, match="CRC mismatch"):
        unpack_archive(bytes(packed))


def test_empty_file_payload_roundtrip() -> None:
    file_dict = {"empty.dat": b""}

    assert unpack_archive(pack_archive(file_dict)) == file_dict


def test_single_byte_payload_roundtrip() -> None:
    file_dict = {"single.dat": b"x"}

    assert unpack_archive(pack_archive(file_dict)) == file_dict


def test_large_payload_roundtrip() -> None:
    payload = random.Random(42).randbytes(300 * 1024)
    file_dict = {"large.bin": payload}

    assert unpack_archive(pack_archive(file_dict)) == file_dict


def test_pack_archive_is_deterministic() -> None:
    file_dict = {"b.bin": b"bbb", "a.bin": b"aaa", "c.bin": b"ccc"}

    assert pack_archive(file_dict) == pack_archive(file_dict)


def test_zip_wrapped_container_is_valid_zip_with_single_stored_member() -> None:
    file_dict = {"renderer.bin": b"abc", "masks.mkv": b"def"}

    archive_zip = pack_archive_to_zip_compatible(file_dict)

    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        infos = zf.infolist()
        assert [info.filename for info in infos] == ["archive.pact"]
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert unpack_archive(zf.read("archive.pact")) == file_dict


def test_minimal_zip_roundtrip_preserves_members() -> None:
    file_dict = {"renderer.bin": b"a" * 1000, "masks.mkv": b"\x00\x01\x02" * 100, "optimized_poses.pt": b"poses"}

    archive_zip = pack_archive_minimal_zip(file_dict)

    with zipfile.ZipFile(io.BytesIO(archive_zip), "r") as zf:
        infos = {info.filename: info for info in zf.infolist()}
        assert zf.read("renderer.bin") == file_dict["renderer.bin"]
        assert zf.read("masks.mkv") == file_dict["masks.mkv"]
        assert zf.read("optimized_poses.pt") == file_dict["optimized_poses.pt"]
        assert infos["masks.mkv"].compress_type == zipfile.ZIP_STORED
        assert infos["renderer.bin"].compress_type == zipfile.ZIP_DEFLATED
        assert infos["optimized_poses.pt"].compress_type == zipfile.ZIP_DEFLATED


def test_lane_a_measurement_overhead_is_tiny_advisory_only() -> None:
    members = _lane_a_members()

    raw_name_payload_concat = b"".join(name.encode("utf-8") + payload for name, payload in sorted(members.items()))
    custom_container = pack_archive(members)
    baseline_zip_bytes = LANE_A_ARCHIVE.stat().st_size

    assert len(custom_container) <= len(raw_name_payload_concat) + 64
    assert baseline_zip_bytes == 694_045
    assert len(custom_container) > baseline_zip_bytes
