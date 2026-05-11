"""Tests for ``tac.packet_compiler.custom_binary_container``.

Mirrors the Rust sister-crate unit tests in
``runtime-rs/crates/tac-packet-compiler/src/custom_binary_container/mod.rs``
so any drift between the two oracles surfaces in CI.
"""

from __future__ import annotations

import pytest

from tac.packet_compiler.custom_binary_container import (
    TACP_MAGIC,
    TACP_VERSION,
    TacpRecord,
    decode_container,
    encode_container,
    section_savings_vs_zip,
)


def test_empty_container_roundtrips() -> None:
    blob = encode_container([])
    records = decode_container(blob)
    assert records == []


def test_single_record_roundtrips() -> None:
    recs = [TacpRecord(name="renderer.bin", body=bytes([1, 2, 3, 4, 5]))]
    blob = encode_container(recs)
    decoded = decode_container(blob)
    assert decoded == recs


def test_multi_record_roundtrips() -> None:
    recs = [
        TacpRecord(name="renderer.bin", body=bytes([0xAA] * 100)),
        TacpRecord(name="masks.mkv", body=bytes([0xBB] * 200)),
        TacpRecord(name="poses.pt", body=bytes([0xCC] * 50)),
    ]
    blob = encode_container(recs)
    decoded = decode_container(blob)
    assert decoded == recs


def test_corruption_in_body_fails_loud() -> None:
    recs = [TacpRecord(name="renderer.bin", body=bytes([1, 2, 3, 4, 5]))]
    blob = bytearray(encode_container(recs))
    mid = len(blob) // 2
    blob[mid] ^= 0xFF
    with pytest.raises(ValueError, match="SHA-256"):
        decode_container(bytes(blob))


def test_bad_magic_fails_loud() -> None:
    blob = bytearray(encode_container([]))
    blob[0] = ord("X")
    with pytest.raises(ValueError, match="magic"):
        decode_container(bytes(blob))


def test_savings_math_known_values() -> None:
    # Contest pattern: one 12-char record. ZIP = 22 + 76 + 24 = 122; TACP =
    # 40 + 6 + 12 = 58. Savings = 64.
    zip_ovh, tacp_ovh, savings = section_savings_vs_zip(["renderer.bin"])
    assert zip_ovh == 122
    assert tacp_ovh == 58
    assert savings == 64


def test_savings_grow_with_record_count() -> None:
    # Three 12-char records. ZIP = 22 + 3*(76 + 24) = 322; TACP = 40 +
    # 3*(6 + 12) = 94. Savings = 228.
    zip_ovh, tacp_ovh, savings = section_savings_vs_zip(
        ["renderer.bin", "masks.mkv___", "poses.pt____"]
    )
    assert zip_ovh == 322
    assert tacp_ovh == 94
    assert savings == 228


def test_magic_constant_matches_wire_format() -> None:
    # The Rust crate hard-codes b"TACP". This test pins the Python side.
    assert TACP_MAGIC == b"TACP"
    assert TACP_VERSION == 0x01


def test_round_trip_byte_stability_known_payload() -> None:
    """The same input always produces the same encoded bytes.

    This is the Python-side anchor for byte-for-byte parity against the
    Rust crate. The exact byte sequence is reproducible from this fixture.
    """
    recs = [
        TacpRecord(name="A", body=b"\x01\x02"),
        TacpRecord(name="BB", body=b"\x03"),
    ]
    blob1 = encode_container(recs)
    blob2 = encode_container(recs)
    assert blob1 == blob2
    # Header: magic + version + flags + n_records = b"TACP" + b"\x01" + b"\x00" + b"\x02\x00"
    assert blob1[:8] == b"TACP\x01\x00\x02\x00"


def test_name_too_long_refused() -> None:
    name = "x" * 256
    with pytest.raises(ValueError, match="255"):
        encode_container([TacpRecord(name=name, body=b"")])
