# SPDX-License-Identifier: MIT
"""Tests for the PR84 adaptive-context mask packet-compiler primitive.

Covers round-trip identity over varying contexts, golden-vector pinning,
spec validation, and representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    AdaptiveContextSpec,
    decode_adaptive_context_stream,
    encode_adaptive_context_stream,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── AdaptiveContextSpec validation ──────────────────────────────────────────


def test_pr84_adaptive_context_spec_round_trip_construction() -> None:
    cdf = np.full((8, 4), 0.25, dtype=np.float64)
    spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
    assert spec.alphabet_size == 4
    assert spec.n_contexts == 8
    np.testing.assert_array_equal(spec.cdf_table, cdf)


def test_pr84_spec_rejects_too_small_alphabet() -> None:
    with pytest.raises(ValueError, match="alphabet_size"):
        AdaptiveContextSpec(
            alphabet_size=1, cdf_table=np.ones((2, 1), dtype=np.float64)
        )


def test_pr84_spec_rejects_non_2d_cdf() -> None:
    with pytest.raises(ValueError, match="2D"):
        AdaptiveContextSpec(
            alphabet_size=4, cdf_table=np.ones(4, dtype=np.float64)
        )


def test_pr84_spec_rejects_alphabet_mismatch() -> None:
    with pytest.raises(ValueError, match="alphabet_size"):
        AdaptiveContextSpec(
            alphabet_size=4, cdf_table=np.ones((2, 5), dtype=np.float64)
        )


def test_pr84_spec_rejects_zero_contexts() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        AdaptiveContextSpec(
            alphabet_size=4, cdf_table=np.zeros((0, 4), dtype=np.float64)
        )


def test_pr84_spec_is_frozen() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises((AttributeError, TypeError)):
        spec.alphabet_size = 999  # type: ignore[misc]


# ── Round-trip ──────────────────────────────────────────────────────────────


def test_pr84_adaptive_context_round_trips_two_contexts() -> None:
    # Context 0 is peaked on symbol 0; context 1 is peaked on symbol 2.
    cdf = np.array(
        [
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.85, 0.05],
        ],
        dtype=np.float64,
    )
    spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
    rng = np.random.default_rng(seed=20260511)
    n_symbols = 200
    context_ids = rng.integers(0, 2, size=n_symbols, dtype=np.int32)
    # Sample symbols matching each context's peaked distribution.
    symbols = np.where(context_ids == 0, 0, 2).astype(np.int32)
    payload = encode_adaptive_context_stream(symbols, context_ids, spec)
    decoded = decode_adaptive_context_stream(payload, context_ids, spec)
    np.testing.assert_array_equal(decoded, symbols)


def test_pr84_adaptive_context_compression_improves_over_uniform() -> None:
    # When the context model matches the source perfectly, the AC should
    # produce a payload meaningfully smaller than a uniform-bytes baseline.
    cdf = np.array(
        [
            [0.94, 0.02, 0.02, 0.02],
            [0.02, 0.94, 0.02, 0.02],
        ],
        dtype=np.float64,
    )
    spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
    n_symbols = 500
    rng = np.random.default_rng(seed=20260511)
    context_ids = rng.integers(0, 2, size=n_symbols, dtype=np.int32)
    symbols = np.where(context_ids == 0, 0, 1).astype(np.int32)
    payload = encode_adaptive_context_stream(symbols, context_ids, spec)
    # 500 symbols at ~0.34 bits each (log2 of 0.94) ≈ 170 bits ≈ 22 bytes.
    # Uniform 2-bit (alphabet=4) baseline would be 125 bytes. Anything well
    # below the uniform baseline counts as "actually compressed".
    assert len(payload) < 80, (
        f"adaptive-context AC failed to compress; payload_len={len(payload)}"
    )


def test_pr84_adaptive_context_round_trips_single_context() -> None:
    cdf = np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
    spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
    rng = np.random.default_rng(seed=42)
    n_symbols = 100
    symbols = rng.integers(0, 4, size=n_symbols, dtype=np.int32)
    context_ids = np.zeros(n_symbols, dtype=np.int32)
    payload = encode_adaptive_context_stream(symbols, context_ids, spec)
    decoded = decode_adaptive_context_stream(payload, context_ids, spec)
    np.testing.assert_array_equal(decoded, symbols)


def test_pr84_adaptive_context_floors_zero_probabilities() -> None:
    # Probability flooring is documented; verify it does not crash even
    # when the cdf has exact zeros.
    cdf = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    spec = AdaptiveContextSpec(alphabet_size=4, cdf_table=cdf)
    symbols = np.array([0, 0, 0], dtype=np.int32)
    context_ids = np.array([0, 0, 0], dtype=np.int32)
    payload = encode_adaptive_context_stream(symbols, context_ids, spec)
    decoded = decode_adaptive_context_stream(payload, context_ids, spec)
    np.testing.assert_array_equal(decoded, symbols)


# ── Failure modes ───────────────────────────────────────────────────────────


def test_pr84_encode_rejects_length_mismatch() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    symbols = np.array([0, 1, 2], dtype=np.int32)
    context_ids = np.array([0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="same length"):
        encode_adaptive_context_stream(symbols, context_ids, spec)


def test_pr84_encode_rejects_empty_stream() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises(ValueError, match="at least one symbol"):
        encode_adaptive_context_stream(
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            spec,
        )


def test_pr84_encode_rejects_out_of_range_symbol() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises(ValueError, match="symbols out of range"):
        encode_adaptive_context_stream(
            np.array([0, 4], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            spec,
        )


def test_pr84_encode_rejects_out_of_range_context_id() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises(ValueError, match="context_ids out of range"):
        encode_adaptive_context_stream(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 5], dtype=np.int32),
            spec,
        )


def test_pr84_decode_rejects_non_multiple_of_4_payload() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises(ValueError, match="multiple of 4"):
        decode_adaptive_context_stream(
            b"\x00\x01",
            np.array([0], dtype=np.int32),
            spec,
        )


def test_pr84_decode_rejects_empty_context_ids() -> None:
    spec = AdaptiveContextSpec(
        alphabet_size=4, cdf_table=np.ones((2, 4), dtype=np.float64) / 4.0
    )
    with pytest.raises(ValueError, match="at least one symbol"):
        decode_adaptive_context_stream(
            b"\x00\x00\x00\x00",
            np.zeros(0, dtype=np.int32),
            spec,
        )


# ── Golden vector ───────────────────────────────────────────────────────────


class TestPR84GoldenVectors:
    def test_adaptive_mask_context_golden_vector(self) -> None:
        # Build a deterministic 4-context × 5-class table (mask classes
        # 0..4 per the contest scoring path) and encode 256 symbols whose
        # contexts cycle in raster fashion. The vector pins the constriction
        # AC byte format for a non-trivial adaptive scenario.
        n_contexts = 4
        alphabet = 5
        cdf = np.full((n_contexts, alphabet), 0.05, dtype=np.float64)
        for ctx in range(n_contexts):
            cdf[ctx, ctx] = 0.8  # each context peaked on a distinct symbol
        cdf /= cdf.sum(axis=1, keepdims=True)
        spec = AdaptiveContextSpec(alphabet_size=alphabet, cdf_table=cdf)
        rng = np.random.default_rng(seed=20260511)
        n_symbols = 256
        context_ids = (np.arange(n_symbols) % n_contexts).astype(np.int32)
        u = rng.random(n_symbols)
        cum = np.cumsum(cdf, axis=1)
        symbols = np.array(
            [int(np.searchsorted(cum[context_ids[i]], u[i])) for i in range(n_symbols)],
            dtype=np.int32,
        )
        payload = encode_adaptive_context_stream(symbols, context_ids, spec)
        digest = hashlib.sha256(payload).hexdigest()
        golden = GOLDEN_DIR / "pr84_adaptive_mask_context_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR84 adaptive-context byte stream changed; "
                "verify constriction version and regenerate if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "alphabet": alphabet,
                        "n_contexts": n_contexts,
                        "n_symbols": n_symbols,
                        "payload_len": len(payload),
                        "schema": "pr84_adaptive_mask_context.v1",
                        "seed": 20260511,
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
