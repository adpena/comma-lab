"""Lane SH arithmetic codec roundtrip + entropy regression tests."""
from __future__ import annotations

import json
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.arithmetic_qint_codec import (
    build_freq_table,
    decode_qints_arithmetic,
    encode_qints_arithmetic,
    profile_aqv1_container,
    profile_qints_arithmetic,
    unpack_arithmetic_payload,
)
from tac.repo_io import sha256_file

REPO = Path(__file__).resolve().parents[3]


def _write_shv1(tmp_path, meta: dict, records: list[bytes], trailing: bytes = b""):
    blob = bytearray()
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    blob.extend(b"SHv1")
    blob.extend(struct.pack("<H", 1))
    blob.extend(struct.pack("<H", len(records)))
    blob.extend(struct.pack("<I", len(meta_bytes)))
    blob.extend(meta_bytes)
    for record in records:
        blob.extend(record)
    blob.extend(trailing)
    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(blob))
    return path


def _sh_record(
    key: str,
    *,
    codec: int,
    data: bytes = b"x",
    shape: tuple[int, ...] = (),
    qmax: int = 0,
) -> bytes:
    key_bytes = key.encode("utf-8")
    out = bytearray()
    out.extend(struct.pack("<H", len(key_bytes)))
    out.extend(key_bytes)
    out.extend(struct.pack("<B", codec))
    out.extend(struct.pack("<Q", len(data)))
    out.extend(data)
    out.extend(struct.pack("<B", len(shape)))
    if shape:
        out.extend(struct.pack(f"<{len(shape)}i", *shape))
    out.extend(struct.pack("<B", qmax))
    return bytes(out)


def test_ternary_roundtrip_uniform():
    rng = np.random.default_rng(0)
    qints = rng.choice([-1, 0, 1], size=10_000, p=[0.3, 0.4, 0.3]).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_ternary_roundtrip_skewed_low_entropy():
    rng = np.random.default_rng(1)
    qints = rng.choice([-1, 0, 1], size=20_000, p=[0.05, 0.9, 0.05]).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)
    # Shannon bound for this distribution is ~0.57 bits/symbol; arithmetic
    # coder should land within 5% of the bound.
    bits_per_symbol = len(blob) * 8 / len(qints)
    assert bits_per_symbol < 1.0, f"expected < 1.0 bits/symbol, got {bits_per_symbol:.3f}"


def test_profile_qints_arithmetic_reports_entropy_gap_without_score_claim():
    rng = np.random.default_rng(11)
    qints = rng.choice([-1, 0, 1], size=4096, p=[0.05, 0.9, 0.05]).astype(np.int8)

    profile = profile_qints_arithmetic(qints, num_symbols=3, offset=1)

    assert profile["score_claim"] is False
    assert profile["dispatch_attempted"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["roundtrip_equal"] is True
    assert profile["num_symbols"] == 3
    assert profile["n_symbols"] == 4096
    assert profile["payload_bytes"] < profile["container_bytes"]
    assert profile["zero_order_entropy_bits_per_symbol"] < 1.0
    assert profile["payload_entropy_gap_bits_per_symbol"] >= -1e-9
    assert profile["container_entropy_gap_bits_per_symbol"] >= 0.0
    assert "zero_order_entropy_profile_not_score_evidence" in profile["dispatch_blockers"]


def test_profile_aqv1_container_rejects_trailing_bytes():
    qints = np.array([-1, 0, 1, 0], dtype=np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)

    with pytest.raises(ValueError, match="trailing bytes"):
        profile_aqv1_container(blob + b"junk")


def test_audit_arithmetic_qint_optimality_cli_records_tool_manifest(tmp_path):
    qints_path = tmp_path / "qints.npy"
    out = tmp_path / "aq_profile.json"
    np.save(qints_path, np.array([-1, 0, 0, 0, 1, 0, -1, 0], dtype=np.int8))

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_arithmetic_qint_optimality.py"),
            "--qints-npy",
            str(qints_path),
            "--num-symbols",
            "3",
            "--offset",
            "1",
            "--json-out",
            str(out),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    tool_run = payload["tool_run_manifest"]
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["roundtrip_equal"] is True
    assert tool_run["tool"] == "tools/audit_arithmetic_qint_optimality.py"
    assert tool_run["input_files"] == [
        {
            "path": qints_path.as_posix(),
            "bytes": qints_path.stat().st_size,
            "sha256": sha256_file(qints_path),
        }
    ]


def test_septenary_roundtrip():
    rng = np.random.default_rng(2)
    qints = rng.integers(-3, 4, size=5000).astype(np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=7, offset=3)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_single_symbol_stream_handles_gracefully():
    qints = np.zeros(1000, dtype=np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
    assert np.array_equal(qints, recovered)


def test_empty_stream_rejected():
    qints = np.zeros(0, dtype=np.int8)
    with pytest.raises(ValueError, match="nonempty"):
        encode_qints_arithmetic(qints, num_symbols=3, offset=1)


def test_offset_overflow_rejected():
    qints = np.array([5, 0, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="out of range"):
        encode_qints_arithmetic(qints, num_symbols=3, offset=1)


def test_freq_table_floor_is_one():
    qints = np.array([0, 0, 0, 1, 1], dtype=np.int8) + 1
    freq = build_freq_table(qints, num_symbols=3)
    # All counts must be >= 1 (defensive against zero-prob symbols at decode).
    assert (freq >= 1).all()
    # Counts: symbol 0 not present in input -> still 1. symbol 1 (originally 0)
    # appears 3 times. symbol 2 (originally 1) appears 2 times.
    assert int(freq[1]) == 3
    assert int(freq[2]) == 2


def test_freq_table_rejects_empty_symbol_stream():
    with pytest.raises(ValueError, match="nonempty"):
        build_freq_table(np.zeros(0, dtype=np.int8), num_symbols=3)


def test_decode_rejects_truncated_header():
    with pytest.raises(ValueError, match="truncated while reading magic"):
        decode_qints_arithmetic(b"AQ", expected_dtype=np.int8)


def test_decode_rejects_truncated_payload():
    qints = np.zeros(16, dtype=np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    with pytest.raises(ValueError, match="truncated while reading payload"):
        decode_qints_arithmetic(blob[:-1], expected_dtype=np.int8)


def test_decode_rejects_trailing_container_bytes():
    qints = np.array([-1, 0, 1, 0, 0], dtype=np.int8)
    blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_qints_arithmetic(blob + b"junk", expected_dtype=np.int8)


def test_short_stream_roundtrip():
    # Tiny streams stress the bit-flush path of the arithmetic coder.
    for n in [1, 2, 3, 7, 15, 16, 17]:
        rng = np.random.default_rng(n)
        qints = rng.choice([-1, 0, 1], size=n).astype(np.int8)
        blob = encode_qints_arithmetic(qints, num_symbols=3, offset=1)
        recovered = decode_qints_arithmetic(blob, expected_dtype=np.int8)
        assert np.array_equal(qints, recovered), f"failed at n={n}"


def test_unpack_shv1_rejects_trailing_bytes(tmp_path):
    payload = _write_shv1(tmp_path, {"keys": {}}, [], trailing=b"junk")

    with pytest.raises(ValueError, match="trailing bytes"):
        unpack_arithmetic_payload(str(payload))


def test_unpack_shv1_rejects_duplicate_record_keys(tmp_path):
    payload = _write_shv1(
        tmp_path,
        {"keys": {}},
        [
            _sh_record("layer.weight", codec=1),
            _sh_record("layer.weight", codec=1),
        ],
    )

    with pytest.raises(ValueError, match="duplicate SHv1 record key"):
        unpack_arithmetic_payload(str(payload))


def test_unpack_shv1_rejects_truncated_record_data(tmp_path):
    meta = {"keys": {}}
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    key = b"layer.weight"
    blob = bytearray()
    blob.extend(b"SHv1")
    blob.extend(struct.pack("<H", 1))
    blob.extend(struct.pack("<H", 1))
    blob.extend(struct.pack("<I", len(meta_bytes)))
    blob.extend(meta_bytes)
    blob.extend(struct.pack("<H", len(key)))
    blob.extend(key)
    blob.extend(struct.pack("<B", 1))
    blob.extend(struct.pack("<Q", 16))
    blob.extend(b"short")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(bytes(blob))

    with pytest.raises(ValueError, match="truncated while reading SHv1 record 'layer.weight' data"):
        unpack_arithmetic_payload(str(payload))


def test_unpack_shv1_rejects_unknown_record_codec(tmp_path):
    payload = _write_shv1(
        tmp_path,
        {"keys": {}},
        [_sh_record("layer.weight", codec=9)],
    )

    with pytest.raises(ValueError, match="unknown codec 9"):
        unpack_arithmetic_payload(str(payload))


def test_unpack_shv1_rejects_missing_meta_record(tmp_path):
    payload = _write_shv1(
        tmp_path,
        {"keys": {"layer.weight": {"codec": "block_fp_per_channel_v1"}}},
        [],
    )

    with pytest.raises(ValueError, match="missing qint record"):
        unpack_arithmetic_payload(str(payload))
