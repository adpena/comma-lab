"""Archive grammar tests for the PR95 LoRA/DoRA TRAILER codec.

Per HNeRV parity lesson 11 (no-op detector): tests verify that the trailer
bytes ARE consumed by the inflate path (mutate trailer byte -> changes apply
differently).
"""

from __future__ import annotations

import pytest
import torch

from tac.substrates.pr95_lora_dora.archive import (
    LORA_TRAILER_MAGIC,
    LORA_TRAILER_VERSION,
    build_lora_archive,
    decode_lora_trailer,
    encode_lora_trailer,
    parse_lora_archive,
)


def _make_lora_record(name: str = "blocks.0", rank: int = 8,
                      out_dim: int = 144, in_dim: int = 324) -> dict:
    return {
        "name": name, "kind": "lora", "rank": rank, "alpha": float(rank),
        "A": torch.randn(rank, in_dim),
        "B": torch.randn(out_dim, rank),
    }


def _make_dora_record(name: str = "blocks.0", rank: int = 8,
                      out_dim: int = 144, in_dim: int = 324) -> dict:
    return {
        "name": name, "kind": "dora", "rank": rank, "alpha": float(rank),
        "A": torch.randn(rank, in_dim),
        "B": torch.randn(out_dim, rank),
        "magnitude": torch.randn(out_dim).abs() + 0.5,
    }


def test_trailer_roundtrip_single_lora_record() -> None:
    torch.manual_seed(0)
    rec = _make_lora_record()
    blob = encode_lora_trailer([rec])
    out = decode_lora_trailer(blob)
    assert len(out) == 1
    assert out[0]["name"] == "blocks.0"
    assert out[0]["kind"] == "lora"
    assert out[0]["rank"] == 8
    assert out[0]["A"].shape == (8, 324)
    assert out[0]["B"].shape == (144, 8)


def test_trailer_roundtrip_dora_carries_magnitude() -> None:
    torch.manual_seed(1)
    rec = _make_dora_record()
    blob = encode_lora_trailer([rec])
    out = decode_lora_trailer(blob)
    assert out[0]["kind"] == "dora"
    assert "magnitude" in out[0]
    assert out[0]["magnitude"].shape == (144,)


def test_trailer_roundtrip_multiple_records() -> None:
    torch.manual_seed(2)
    recs = [
        _make_lora_record(name="blocks.0", rank=8, out_dim=144, in_dim=324),
        _make_lora_record(name="blocks.1", rank=8, out_dim=144, in_dim=324),
        _make_dora_record(name="blocks.2", rank=8, out_dim=108, in_dim=324),
    ]
    blob = encode_lora_trailer(recs)
    out = decode_lora_trailer(blob)
    assert len(out) == 3
    assert [r["name"] for r in out] == ["blocks.0", "blocks.1", "blocks.2"]
    assert [r["kind"] for r in out] == ["lora", "lora", "dora"]


def test_trailer_roundtrip_preserves_quantized_values() -> None:
    """INT8 quantization is lossy; check value fidelity at the per-tensor scale."""
    torch.manual_seed(3)
    A = torch.randn(4, 16) * 10.0  # large dynamic range
    B = torch.randn(8, 4) * 5.0
    rec = {"name": "blocks.0", "kind": "lora", "rank": 4, "alpha": 4.0, "A": A, "B": B}
    blob = encode_lora_trailer([rec])
    out = decode_lora_trailer(blob)
    # Reconstructed should be within ~1% (INT8 quantization)
    rel_err_A = ((out[0]["A"] - A).abs() / (A.abs() + 1e-6)).mean().item()
    rel_err_B = ((out[0]["B"] - B).abs() / (B.abs() + 1e-6)).mean().item()
    assert rel_err_A < 0.05, f"A relative error too high: {rel_err_A}"
    assert rel_err_B < 0.05, f"B relative error too high: {rel_err_B}"


def test_build_archive_appends_to_base() -> None:
    base = b"\x00" * 1000  # synthetic base
    rec = _make_lora_record()
    archive = build_lora_archive(base, [rec])
    assert archive.startswith(base)
    assert len(archive) > len(base)


def test_parse_archive_no_trailer_returns_empty_records() -> None:
    base = b"\x00" * 1000
    base_back, recs = parse_lora_archive(base)
    assert base_back == base
    assert recs == []


def test_parse_archive_with_trailer_returns_records() -> None:
    torch.manual_seed(4)
    base = b"PR95_MOCK_BYTES" * 100
    rec = _make_lora_record()
    archive = build_lora_archive(base, [rec])
    base_back, recs = parse_lora_archive(archive)
    assert base_back == base
    assert len(recs) == 1
    assert recs[0]["name"] == "blocks.0"


def test_parse_archive_rejects_bad_magic() -> None:
    # Construct a malformed trailer (wrong magic) — should fall back to no trailer
    base = b"PR95_MOCK_BYTES" * 100
    fake_trailer_payload = b"\xFF\xFF\xFF\xFF" + b"X" * 20
    suffix = fake_trailer_payload + (len(fake_trailer_payload)).to_bytes(4, "little")
    archive = base + suffix
    base_back, recs = parse_lora_archive(archive)
    # With wrong magic, parser should return (full, []) since magic check fails
    assert recs == []


def test_no_op_detector_mutate_trailer_byte_changes_decode() -> None:
    """No-op detector per HNeRV parity lesson 11. Mutating a trailer byte
    must cause a different decode result."""
    torch.manual_seed(5)
    rec = _make_lora_record()
    blob = encode_lora_trailer([rec])
    out_a = decode_lora_trailer(blob)

    # Mutate one byte in the middle of the payload (skip header)
    mid = len(blob) // 2
    mutated = bytearray(blob)
    mutated[mid] = (mutated[mid] + 1) % 256
    out_b = decode_lora_trailer(bytes(mutated))
    # At least one tensor entry should differ
    a_diff = (out_a[0]["A"] - out_b[0]["A"]).abs().sum().item()
    b_diff = (out_a[0]["B"] - out_b[0]["B"]).abs().sum().item()
    assert (a_diff + b_diff) > 0, "Trailer mutation should change decoded values"


def test_trailer_magic_constant_value() -> None:
    assert LORA_TRAILER_MAGIC == 0x4154524C  # 'L','R','T','A' little-endian
    assert LORA_TRAILER_VERSION == 1


def test_long_name_rejected() -> None:
    long_name = "x" * 300  # > 255 bytes
    rec = _make_lora_record(name=long_name)
    with pytest.raises(ValueError, match="Adapter name too long"):
        encode_lora_trailer([rec])


def test_bad_rank_rejected() -> None:
    rec = _make_lora_record()
    rec["rank"] = 256
    with pytest.raises(ValueError, match="rank must be"):
        encode_lora_trailer([rec])


def test_bad_kind_rejected() -> None:
    rec = _make_lora_record()
    rec["kind"] = "not_a_kind"
    with pytest.raises(ValueError, match="adapter kind must be"):
        encode_lora_trailer([rec])


def test_trailer_size_estimate_for_full_tier_c_at_r8() -> None:
    """Sanity-check the trailer size estimate from the deconstruction memo.

    6 Tier C adapters at r=8 + tier B/A FT weights ~ 22 KB raw."""
    torch.manual_seed(6)
    # Tier C 6 blocks at r=8
    records = [
        _make_lora_record(name="blocks.0", rank=8, out_dim=144, in_dim=324),
        _make_lora_record(name="blocks.1", rank=8, out_dim=144, in_dim=324),
        _make_lora_record(name="blocks.2", rank=8, out_dim=108, in_dim=324),
        _make_lora_record(name="blocks.3", rank=8, out_dim=80, in_dim=243),
        _make_lora_record(name="blocks.4", rank=8, out_dim=72, in_dim=180),
        _make_lora_record(name="blocks.5", rank=8, out_dim=72, in_dim=162),
    ]
    blob = encode_lora_trailer(records)
    # 17,416 LoRA params (per memo) at INT8 + ~120 B header per adapter +
    # ~16 B per scale/alpha/shape ~ 17,416 + 6*150 ≈ 18,316 bytes.
    assert len(blob) < 25_000, f"Trailer larger than expected: {len(blob)}"
    assert len(blob) > 15_000, f"Trailer smaller than expected: {len(blob)}"
