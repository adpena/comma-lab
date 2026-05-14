# SPDX-License-Identifier: MIT
"""Tests for the PR91 ``hpac_coder_hybrid`` packet-compiler primitives.

Covers universal AC wrapper round-trip, QM0/QH0 magic grammar, hi-lo split
byte permutation, golden-vector pinning, and representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    MAGIC_QH0,
    MAGIC_QM0,
    QMQHHeader,
    decode_categorical_stream,
    emit_qmqh_header,
    encode_categorical_stream,
    pack_hi_lo_split,
    parse_qmqh_header,
    unpack_hi_lo_split,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Magic constants ─────────────────────────────────────────────────────────


def test_pr91_magic_constants_match_source_pinning() -> None:
    assert MAGIC_QM0 == b"QM0"
    assert MAGIC_QH0 == b"QH0"


# ── Universal AC wrapper ────────────────────────────────────────────────────


def test_pr91_categorical_stream_round_trips_uniform_alphabet() -> None:
    rng = np.random.default_rng(seed=20260511)
    n_symbols = 100
    alphabet = 5
    probs = np.full((n_symbols, alphabet), 1.0 / alphabet, dtype=np.float64)
    symbols = rng.integers(0, alphabet, size=n_symbols, dtype=np.int32)
    payload = encode_categorical_stream(symbols, probs)
    decoded = decode_categorical_stream(payload, probs)
    np.testing.assert_array_equal(decoded, symbols)


def test_pr91_categorical_stream_round_trips_peaked_distribution() -> None:
    n_symbols = 200
    alphabet = 8
    probs = np.full((n_symbols, alphabet), 0.01, dtype=np.float64)
    probs[:, 0] = 0.93
    probs /= probs.sum(axis=1, keepdims=True)
    symbols = np.zeros(n_symbols, dtype=np.int32)
    symbols[::13] = 5  # rare event sprinkled in
    payload = encode_categorical_stream(symbols, probs)
    # Peaked + mostly-mode → range coder achieves strong compression.
    assert len(payload) < n_symbols
    decoded = decode_categorical_stream(payload, probs)
    np.testing.assert_array_equal(decoded, symbols)


def test_pr91_categorical_stream_floors_negative_probs() -> None:
    # Probability flooring is documented; verify it does not crash.
    n_symbols = 5
    alphabet = 3
    probs = np.full((n_symbols, alphabet), 1.0 / alphabet, dtype=np.float64)
    probs[0, 1] = 0.0  # would be illegal without floor
    probs[2, 2] = -1.0  # likewise
    symbols = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    payload = encode_categorical_stream(symbols, probs)
    decoded = decode_categorical_stream(payload, probs)
    np.testing.assert_array_equal(decoded, symbols)


def test_pr91_categorical_stream_rejects_symbol_count_mismatch() -> None:
    probs = np.ones((3, 4), dtype=np.float64) / 4.0
    symbols = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="symbol count"):
        encode_categorical_stream(symbols, probs)


def test_pr91_categorical_stream_rejects_out_of_range_symbol() -> None:
    probs = np.ones((2, 4), dtype=np.float64) / 4.0
    symbols = np.array([0, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        encode_categorical_stream(symbols, probs)


def test_pr91_categorical_stream_rejects_empty_stream() -> None:
    probs = np.zeros((0, 4), dtype=np.float64)
    symbols = np.zeros(0, dtype=np.int32)
    with pytest.raises(ValueError, match="at least one symbol"):
        encode_categorical_stream(symbols, probs)


def test_pr91_categorical_stream_rejects_non_2d_probs() -> None:
    probs = np.ones(5, dtype=np.float64) / 5.0
    symbols = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="2D"):
        encode_categorical_stream(symbols, probs)


def test_pr91_categorical_stream_decoder_rejects_non_multiple_of_4_payload() -> None:
    probs = np.ones((1, 4), dtype=np.float64) / 4.0
    with pytest.raises(ValueError, match="multiple of 4"):
        decode_categorical_stream(b"\x00\x01\x02", probs)


def test_pr91_categorical_stream_decoder_rejects_empty_probs() -> None:
    probs = np.zeros((0, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="at least one symbol"):
        decode_categorical_stream(b"\x00\x00\x00\x00", probs)


# ── QM0 / QH0 magic grammar ─────────────────────────────────────────────────


def test_pr91_qmqh_emit_qm0() -> None:
    header = emit_qmqh_header(hilo_split=False)
    assert header == MAGIC_QM0


def test_pr91_qmqh_emit_qh0() -> None:
    header = emit_qmqh_header(hilo_split=True)
    assert header == MAGIC_QH0


def test_pr91_qmqh_parse_qm0() -> None:
    parsed = parse_qmqh_header(MAGIC_QM0 + b"...body...")
    assert isinstance(parsed, QMQHHeader)
    assert parsed.magic == MAGIC_QM0
    assert parsed.hilo_split is False
    assert parsed.body_offset == 3


def test_pr91_qmqh_parse_qh0() -> None:
    parsed = parse_qmqh_header(MAGIC_QH0 + b"...body...")
    assert parsed.magic == MAGIC_QH0
    assert parsed.hilo_split is True
    assert parsed.body_offset == 3


def test_pr91_qmqh_parse_rejects_truncated() -> None:
    with pytest.raises(ValueError, match="too short"):
        parse_qmqh_header(b"QM")


def test_pr91_qmqh_parse_rejects_unknown_magic() -> None:
    with pytest.raises(ValueError, match="unknown QMQH magic"):
        parse_qmqh_header(b"XYZ" + b"body")


def test_pr91_qmqh_header_is_frozen() -> None:
    parsed = parse_qmqh_header(MAGIC_QM0)
    with pytest.raises((AttributeError, TypeError)):
        parsed.body_offset = 99  # type: ignore[misc]


# ── Hi-lo split byte permutation ────────────────────────────────────────────


def test_pr91_hi_lo_split_round_trips_typical_bytes() -> None:
    packed = bytes(range(64))  # 32 bytes of deterministic content
    split = pack_hi_lo_split(packed)
    assert len(split) == len(packed)
    recovered = unpack_hi_lo_split(split)
    assert recovered == packed


def test_pr91_hi_lo_split_round_trips_empty() -> None:
    split = pack_hi_lo_split(b"")
    assert split == b""
    recovered = unpack_hi_lo_split(b"")
    assert recovered == b""


def test_pr91_hi_lo_split_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="even"):
        pack_hi_lo_split(b"\x00\x01\x02")


def test_pr91_hi_lo_unpack_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="even"):
        unpack_hi_lo_split(b"\x00\x01\x02")


def test_pr91_hi_lo_split_changes_byte_order() -> None:
    # Construct a packed byte stream where hi-nibbles all equal 0xA and
    # lo-nibbles cycle 0..15. The split output should clearly partition
    # into "all 0xA" first then the increasing lo's — proving the
    # permutation truly happened.
    nibbles = np.zeros(32, dtype=np.uint8)
    nibbles[0::2] = 0xA  # hi-nibbles
    nibbles[1::2] = np.arange(16, dtype=np.uint8)  # lo-nibbles
    packed = ((nibbles[0::2] << 4) | nibbles[1::2]).astype(np.uint8).tobytes()
    split = pack_hi_lo_split(packed)
    # First half should pack the hi-nibbles (all 0xA), so each byte = 0xAA.
    half = len(split) // 2
    assert all(byte == 0xAA for byte in split[:half])
    recovered = unpack_hi_lo_split(split)
    assert recovered == packed


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR91GoldenVectors:
    def test_arithmetic_coder_constriction_golden_vector(self) -> None:
        # Build a deterministic 200-symbol stream over alphabet=8 with a
        # peaked-at-0 distribution that varies slightly across positions.
        rng = np.random.default_rng(seed=20260511)
        n_symbols = 200
        alphabet = 8
        probs = np.full((n_symbols, alphabet), 0.02, dtype=np.float64)
        peak = (np.arange(n_symbols) // 50).astype(np.int64) % alphabet
        for i in range(n_symbols):
            probs[i, peak[i]] = 1.0 - 0.02 * (alphabet - 1)
        # Sample from the cumulative distribution deterministically.
        cdf = np.cumsum(probs, axis=1)
        u = rng.random(n_symbols)
        symbols = np.array(
            [int(np.searchsorted(cdf[i], u[i])) for i in range(n_symbols)],
            dtype=np.int32,
        )
        payload = encode_categorical_stream(symbols, probs)
        digest = hashlib.sha256(payload).hexdigest()
        golden = GOLDEN_DIR / "pr91_arithmetic_coder_constriction_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR91 constriction AC byte stream changed; "
                "verify constriction version and regenerate if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "alphabet": alphabet,
                        "n_symbols": n_symbols,
                        "payload_len": len(payload),
                        "schema": "pr91_arithmetic_coder_constriction.v1",
                        "seed": 20260511,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_qmqh_grammar_golden_vector(self) -> None:
        # A small deterministic QH0 + body bytes; we pin both the magic
        # emission and the hi-lo permutation of a deterministic packed
        # body so a future Rust port matches byte-for-byte.
        body = bytes(range(64))
        split = pack_hi_lo_split(body)
        full = emit_qmqh_header(hilo_split=True) + split
        digest = hashlib.sha256(full).hexdigest()
        golden = GOLDEN_DIR / "pr91_qmqh_grammar_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR91 QH0 grammar byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "body_len": len(body),
                        "hilo_split": True,
                        "payload_len": len(full),
                        "schema": "pr91_qmqh_grammar.v1",
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
