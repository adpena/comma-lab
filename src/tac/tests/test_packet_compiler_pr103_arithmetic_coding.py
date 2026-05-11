from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    AdaptiveBrotliResult,
    MergedRangeStream,
    WeightTensorACSpec,
    adaptive_brotli_param_search,
    decode_latent_hi_arithmetic,
    decode_merged_range_stream,
    encode_latent_hi_arithmetic,
    encode_merged_range_stream,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent
    / "packet_compiler"
    / "golden_vectors"
)


def _peaked_histogram(alphabet: int = 256, peak: int = 0) -> np.ndarray:
    """Histogram with most mass at ``peak`` (PR103-style int8 weight)."""
    h = np.full(alphabet, 1.0, dtype=np.float64)
    h[peak] = float(alphabet) * 50.0
    h /= h.sum()
    return h


def test_pr103_merged_range_stream_round_trips_tensor_sequence() -> None:
    hist = np.ones(256, dtype=np.float64)
    tensors = [
        np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8),
        np.array([255, 128, 0, 1], dtype=np.uint8),
    ]
    specs = [
        WeightTensorACSpec(name="w0", shape=tensors[0].shape, histogram=hist),
        WeightTensorACSpec(name="w1", shape=tensors[1].shape, histogram=hist),
    ]

    stream = encode_merged_range_stream(tensors, specs)
    decoded = decode_merged_range_stream(stream, specs)

    assert stream.word_count * 4 == len(stream.payload)
    assert stream.tensor_symbol_counts == (6, 4)
    np.testing.assert_array_equal(decoded[0], tensors[0])
    np.testing.assert_array_equal(decoded[1], tensors[1])


def test_pr103_latent_hi_arithmetic_round_trips_high_bytes() -> None:
    hist = np.ones(256, dtype=np.float64)
    latents = np.array([0, 1, 255, 256, 511, 65535], dtype=np.uint16)

    payload = encode_latent_hi_arithmetic(latents, histogram=hist)
    decoded_hi = decode_latent_hi_arithmetic(
        payload,
        histogram=hist,
        n_symbols=latents.size,
    )

    expected_hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.uint8)
    np.testing.assert_array_equal(decoded_hi, expected_hi)


def test_pr103_adaptive_brotli_search_returns_best_tested_candidate() -> None:
    result = adaptive_brotli_param_search(
        b"abc" * 100,
        time_budget_s=0.1,
        lgwin_range=(10, 11),
        quality_range=(4, 5),
        max_evaluations=4,
    )

    assert result.tested_count == 4
    assert len(result.payload) == min(size for _, _, size in result.explored)
    assert 10 <= result.lgwin <= 11
    assert 4 <= result.quality <= 5


def test_pr103_weight_spec_rejects_histogram_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="histogram"):
        WeightTensorACSpec(
            name="bad",
            shape=(2,),
            histogram=np.ones(4, dtype=np.float64),
            alphabet_size=256,
        )


# ── Extended WeightTensorACSpec validation ──────────────────────────────────


class TestWeightTensorACSpec:
    def test_construct_pr103_like(self) -> None:
        s = WeightTensorACSpec(
            name="blocks.0.weight",
            shape=(144, 36, 3, 3),
            histogram=_peaked_histogram(),
        )
        assert s.name == "blocks.0.weight"
        assert s.shape == (144, 36, 3, 3)
        assert s.alphabet_size == 256

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            WeightTensorACSpec(
                name="", shape=(4,), histogram=_peaked_histogram()
            )

    def test_rejects_zero_dim(self) -> None:
        with pytest.raises(ValueError, match="positive dims"):
            WeightTensorACSpec(
                name="x", shape=(0, 4), histogram=_peaked_histogram()
            )

    def test_rejects_small_alphabet(self) -> None:
        with pytest.raises(ValueError, match="alphabet_size"):
            WeightTensorACSpec(
                name="x",
                shape=(4,),
                histogram=np.array([1.0]),
                alphabet_size=1,
            )

    def test_frozen(self) -> None:
        s = WeightTensorACSpec(
            name="x", shape=(4,), histogram=_peaked_histogram()
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "y"  # type: ignore[misc]


# ── Extended merged range stream coverage ───────────────────────────────────


class TestMergedRangeStreamExtended:
    def _make(self, seed: int) -> tuple[list[np.ndarray], list[WeightTensorACSpec]]:
        rng = np.random.default_rng(seed)
        shapes = [(40,), (12, 8), (4, 4, 4)]
        specs: list[WeightTensorACSpec] = []
        tensors: list[np.ndarray] = []
        for i, sh in enumerate(shapes):
            size = int(np.prod(sh))
            raw = rng.integers(-32, 33, size=size).astype(np.int32)
            symbols = (raw + 128).astype(np.int32).reshape(sh)
            hist = np.bincount(
                symbols.reshape(-1), minlength=256
            ).astype(np.float64) + 1.0
            specs.append(
                WeightTensorACSpec(name=f"t{i}", shape=sh, histogram=hist)
            )
            tensors.append(symbols)
        return tensors, specs

    def test_round_trip_three_tensors(self) -> None:
        tensors, specs = self._make(seed=1)
        stream = encode_merged_range_stream(tensors, specs)
        decoded = decode_merged_range_stream(stream, specs)
        assert len(decoded) == len(tensors)
        for orig, recv in zip(tensors, decoded, strict=False):
            np.testing.assert_array_equal(orig.astype(np.int32), recv)

    def test_payload_smaller_than_raw_on_peaked_distribution(self) -> None:
        shape = (200,)
        raw = np.zeros(shape, dtype=np.int32)
        raw[10] = 1
        symbols = (raw + 128).astype(np.int32)
        hist = _peaked_histogram(peak=128)
        spec = WeightTensorACSpec(name="x", shape=shape, histogram=hist)
        stream = encode_merged_range_stream([symbols], [spec])
        assert len(stream.payload) < 200

    def test_rejects_count_mismatch(self) -> None:
        tensors, specs = self._make(seed=2)
        with pytest.raises(ValueError, match="spec count"):
            encode_merged_range_stream(tensors, specs[:-1])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            encode_merged_range_stream([], [])

    def test_rejects_shape_mismatch(self) -> None:
        tensors, specs = self._make(seed=3)
        bad_tensor = tensors[0].reshape(-1)[:-1]
        with pytest.raises(ValueError, match="shape"):
            encode_merged_range_stream([bad_tensor, *tensors[1:]], specs)

    def test_rejects_symbol_out_of_range(self) -> None:
        spec = WeightTensorACSpec(
            name="t", shape=(4,), histogram=_peaked_histogram()
        )
        bad = np.array([0, 256, 1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="out of range"):
            encode_merged_range_stream([bad], [spec])

    def test_decode_rejects_payload_size_mismatch(self) -> None:
        tensors, specs = self._make(seed=4)
        stream = encode_merged_range_stream(tensors, specs)
        tampered = MergedRangeStream(
            payload=stream.payload,
            tensor_symbol_counts=stream.tensor_symbol_counts,
            word_count=stream.word_count + 1,
        )
        with pytest.raises(ValueError, match="word count"):
            decode_merged_range_stream(tampered, specs)

    def test_decode_rejects_spec_count_mismatch(self) -> None:
        tensors, specs = self._make(seed=5)
        stream = encode_merged_range_stream(tensors, specs)
        with pytest.raises(ValueError, match="entries"):
            decode_merged_range_stream(stream, specs[:-1])

    def test_word_count_matches_payload(self) -> None:
        tensors, specs = self._make(seed=6)
        stream = encode_merged_range_stream(tensors, specs)
        assert len(stream.payload) == stream.word_count * 4


# ── Extended latent-hi coverage ─────────────────────────────────────────────


class TestLatentHiExtended:
    def test_round_trip_peaked(self) -> None:
        rng = np.random.default_rng(seed=7)
        n = 600 * 28
        latents = rng.integers(0, 8, size=n).astype(np.uint16)
        # Insert a few large values to exercise the hi-byte distribution.
        latents[::50] = 257
        hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.int32)
        hist = np.bincount(hi, minlength=256).astype(np.float64) + 1.0
        payload = encode_latent_hi_arithmetic(latents, histogram=hist)
        decoded_hi = decode_latent_hi_arithmetic(
            payload, histogram=hist, n_symbols=n
        )
        np.testing.assert_array_equal(decoded_hi, hi.astype(np.uint8))

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            encode_latent_hi_arithmetic(
                np.empty(0, dtype=np.uint16), histogram=_peaked_histogram()
            )

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            encode_latent_hi_arithmetic(
                np.zeros((2, 3), dtype=np.uint16),
                histogram=_peaked_histogram(),
            )

    def test_decode_rejects_zero_count(self) -> None:
        with pytest.raises(ValueError, match="n_symbols"):
            decode_latent_hi_arithmetic(
                b"\x00\x00\x00\x00",
                histogram=_peaked_histogram(),
                n_symbols=0,
            )


# ── Extended adaptive Brotli coverage ───────────────────────────────────────


class TestAdaptiveBrotliExtended:
    def test_returns_decompressible_payload(self) -> None:
        import brotli

        payload = b"hello world " * 500
        result = adaptive_brotli_param_search(
            payload, time_budget_s=10.0, max_evaluations=8
        )
        assert isinstance(result, AdaptiveBrotliResult)
        assert brotli.decompress(result.payload) == payload
        assert result.tested_count >= 1

    def test_respects_time_budget(self) -> None:
        payload = b"x" * 1000
        result = adaptive_brotli_param_search(
            payload, time_budget_s=0.01, max_evaluations=100
        )
        # At least one evaluation must happen.
        assert result.tested_count >= 1

    def test_respects_max_evaluations(self) -> None:
        payload = b"abcd" * 200
        result = adaptive_brotli_param_search(
            payload, time_budget_s=60.0, max_evaluations=3
        )
        assert result.tested_count == 3

    def test_rejects_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            adaptive_brotli_param_search(b"", time_budget_s=10.0)

    def test_rejects_zero_budget(self) -> None:
        with pytest.raises(ValueError, match="time_budget_s"):
            adaptive_brotli_param_search(b"abc", time_budget_s=0)

    def test_rejects_invalid_lgwin_range(self) -> None:
        with pytest.raises(ValueError, match="lgwin_range"):
            adaptive_brotli_param_search(
                b"abc", time_budget_s=10.0, lgwin_range=(5, 30)
            )

    def test_rejects_invalid_quality_range(self) -> None:
        with pytest.raises(ValueError, match="quality_range"):
            adaptive_brotli_param_search(
                b"abc", time_budget_s=10.0, quality_range=(0, 12)
            )

    def test_chosen_params_in_explored(self) -> None:
        payload = b"abcabc" * 100
        result = adaptive_brotli_param_search(
            payload, time_budget_s=10.0, max_evaluations=4
        )
        assert (result.lgwin, result.quality) in {
            (w, q) for (w, q, _) in result.explored
        }

    def test_chosen_payload_is_smallest_in_explored(self) -> None:
        payload = b"abc" * 100
        result = adaptive_brotli_param_search(
            payload,
            time_budget_s=10.0,
            max_evaluations=4,
            lgwin_range=(10, 11),
            quality_range=(4, 5),
        )
        assert len(result.payload) == min(s for _, _, s in result.explored)


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR103GoldenVectors:
    """Persist + verify byte-level conformance vectors for native ports."""

    def test_golden_dir_exists(self) -> None:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        assert GOLDEN_DIR.exists()

    def test_merged_range_stream_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260511)
        shapes = [(60,), (10, 8), (3, 3, 4)]
        tensors: list[np.ndarray] = []
        specs: list[WeightTensorACSpec] = []
        for i, sh in enumerate(shapes):
            size = int(np.prod(sh))
            raw = rng.integers(-12, 13, size=size).astype(np.int32)
            symbols = (raw + 128).astype(np.int32).reshape(sh)
            hist = np.bincount(
                symbols.reshape(-1), minlength=256
            ).astype(np.float64) + 1.0
            tensors.append(symbols)
            specs.append(
                WeightTensorACSpec(
                    name=f"golden_t{i}", shape=sh, histogram=hist
                )
            )
        stream = encode_merged_range_stream(tensors, specs)
        decoded = decode_merged_range_stream(stream, specs)
        for orig, recv in zip(tensors, decoded, strict=False):
            np.testing.assert_array_equal(orig.astype(np.int32), recv)

        digest = hashlib.sha256(stream.payload).hexdigest()
        golden = GOLDEN_DIR / "merged_range_stream_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "merged-range-stream payload changed; if constriction "
                "version differs the golden must be regenerated"
            )
        else:
            golden.write_text(
                json.dumps(
                    {
                        "schema": "merged_range_stream.v1",
                        "tensor_shapes": [list(sh) for sh in shapes],
                        "tensor_symbol_counts": list(stream.tensor_symbol_counts),
                        "word_count": stream.word_count,
                        "payload_len": len(stream.payload),
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_latent_hi_arithmetic_golden_vector(self) -> None:
        rng = np.random.default_rng(seed=20260511)
        n = 1000
        latents = rng.integers(0, 16, size=n).astype(np.uint16)
        latents[::50] = 300  # inject some high-byte values
        hi = ((latents.astype(np.int32) >> 8) & 0xFF).astype(np.int32)
        hist = np.bincount(hi, minlength=256).astype(np.float64) + 1.0
        payload = encode_latent_hi_arithmetic(latents, histogram=hist)
        decoded = decode_latent_hi_arithmetic(
            payload, histogram=hist, n_symbols=n
        )
        np.testing.assert_array_equal(decoded, hi.astype(np.uint8))
        digest = hashlib.sha256(payload).hexdigest()
        golden = GOLDEN_DIR / "latent_hi_arithmetic_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "latent-hi AC payload changed; if constriction differs the "
                "golden must be regenerated"
            )
        else:
            golden.write_text(
                json.dumps(
                    {
                        "schema": "latent_hi_arithmetic.v1",
                        "n_symbols": n,
                        "payload_len": len(payload),
                        "sha256": digest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
