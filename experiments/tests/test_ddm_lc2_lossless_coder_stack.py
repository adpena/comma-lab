from __future__ import annotations

import struct
from pathlib import Path

import brotli
import pytest

from experiments import ddm_lc2_lossless_coder_stack as lc2


def test_cx2_reference_transform_roundtrip() -> None:
    semantic = bytes(range(256)) * 19 + b"short-tail"
    carrier = bytes(reversed(range(256))) * 3
    hpac = bytes((index * 37) & 0xFF for index in range(4099))
    transformed = lc2.receiver.encode_cx2_model_sections(semantic, carrier, hpac)
    decoded = lc2.receiver.decode_cx2_model_sections(*transformed)
    assert decoded == (semantic, carrier, hpac)


def test_split_archive_reconstructs_exact_loader_bytes() -> None:
    semantic = b"semantic" * 97
    carrier = b"carrier" * 53
    hpac = b"hpac" * 211
    tokens = struct.pack("<4I", 1, 2, 3, 4)
    transformed = lc2.receiver.encode_cx2_model_sections(semantic, carrier, hpac)
    streams = tuple(
        brotli.compress(payload, quality=quality) for payload, quality in zip(transformed, (10, 9, 10), strict=True)
    )
    archive = lc2.build_archive(
        streams,
        tokens,
        model_codec="split_brotli_cx2",
    )
    member = lc2.read_archive_member_bytes(archive)
    parts = lc2.receiver.split_payload(member)
    decoded = lc2.receiver.decode_models(parts.models, model_codec=parts.model_codec)
    expected = struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier + hpac
    assert decoded.raw == expected
    assert parts.tokens == tokens


def test_deterministic_zip_has_exact_100_byte_overhead() -> None:
    member = b"x" * 1234
    first = lc2.deterministic_zip(member)
    second = lc2.deterministic_zip(member)
    assert first == second
    assert len(first) == len(member) + 100


def test_persist_exact_refuses_divergent_existing_payload(tmp_path: Path) -> None:
    path = tmp_path / "retained.bin"
    first = lc2.persist_exact(path, b"first")
    assert first["bytes"] == 5
    assert lc2.persist_exact(path, b"first") == first
    with pytest.raises(RuntimeError, match="divergent retained payload"):
        lc2.persist_exact(path, b"second")


def test_timeout_partial_is_certified_before_atomic_cold_store(tmp_path: Path) -> None:
    decode_root = tmp_path / "decode"
    staging = decode_root / "staging" / "0.raw"
    lc2.atomic_bytes(staging, b"retained partial raw")
    timeout = {
        "schema": "ddm_lc2_decode_timeout.v1",
        "payloads_retained": True,
        "partial_raw": lc2.file_record(staging),
    }
    timeout_path = decode_root / "decode_timeout_receipt.json"
    lc2.atomic_json(timeout_path, timeout)

    receipt = lc2.preserve_certified_partial_raw(decode_root, staging)

    assert receipt is not None
    assert not staging.exists()
    cold = Path(receipt["cold_stored"]["path"])
    assert cold.read_bytes() == b"retained partial raw"
    assert receipt["original"]["sha256"] == receipt["cold_stored"]["sha256"]


def test_uncertified_partial_blocks_relaunch(tmp_path: Path) -> None:
    decode_root = tmp_path / "decode"
    staging = decode_root / "staging" / "0.raw"
    lc2.atomic_bytes(staging, b"unadjudicated")
    with pytest.raises(RuntimeError, match="uncertified retained staging raw"):
        lc2.preserve_certified_partial_raw(decode_root, staging)
    assert staging.read_bytes() == b"unadjudicated"
