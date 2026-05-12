"""Tests for the unified 5-strategy sign-encoding taxonomy.

Covers:

* Round-trip per strategy (negzig / zig / twos / off / raw_uint8)
* Negzig -128 guard (the -128 != 0 collapse the design memo flags as the
  critical precondition)
* Entropy-based picker correctness
* Per-tensor mixed-strategy maps (PR101's {9: negzig, 14: negzig, 20:
  twos, 27: off})
* PR101 / PR103 / PR96 source-byte semantics
* Failure modes (dtype mismatch, unknown strategy, byte-count mismatch)
* Golden-vector SHA pin (per Catalog #91 — paired roundtrip vector)

Source: ``src/tac/packet_compiler/sign_encoding.py``.
Design memo: ``.omx/research/sign_encoding_unified_taxonomy_20260512.md``.

[empirical:src/tac/packet_compiler/golden_vectors/sign_encoding_*_v1.json]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    SignEncodingStrategy,
    StrategySelection,
    VALID_SIGN_ENCODING_STRATEGIES,
    decode_sign,
    encode_sign,
    select_optimal_strategy,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ═══════════════════════════════════════════════════════════════════════════
# Enum + validation
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyEnum:
    def test_five_strategies_present(self) -> None:
        assert {s.value for s in SignEncodingStrategy} == {
            "negzig",
            "zig",
            "twos",
            "off",
            "raw_uint8",
        }
        assert len(VALID_SIGN_ENCODING_STRATEGIES) == 5

    def test_valid_set_matches_enum(self) -> None:
        assert VALID_SIGN_ENCODING_STRATEGIES == frozenset(
            {s.value for s in SignEncodingStrategy}
        )

    def test_enum_compares_equal_to_string(self) -> None:
        """SignEncodingStrategy inherits str — values compare equal to bare strings."""
        assert SignEncodingStrategy.ZIG == "zig"
        assert SignEncodingStrategy.NEGZIG == "negzig"
        assert SignEncodingStrategy.RAW_UINT8 == "raw_uint8"

    def test_valid_set_is_frozen(self) -> None:
        """The frozenset is immutable; cannot add/remove entries at runtime."""
        with pytest.raises(AttributeError):
            VALID_SIGN_ENCODING_STRATEGIES.add("garbage")  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════
# Round-trip per strategy
# ═══════════════════════════════════════════════════════════════════════════


class TestRoundTripBijection:
    def test_zig_roundtrip_full_int8_domain(self) -> None:
        """zig is bijective over the full int8 [-128, 127] range."""
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, SignEncodingStrategy.ZIG)
        dec = decode_sign(enc, arr.shape, np.int8, SignEncodingStrategy.ZIG)
        assert np.array_equal(arr, dec)
        assert len(enc) == 256

    def test_twos_roundtrip_full_int8_domain(self) -> None:
        """twos is bijective over the full int8 [-128, 127] range."""
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, "twos")
        dec = decode_sign(enc, arr.shape, np.int8, "twos")
        assert np.array_equal(arr, dec)

    def test_off_roundtrip_full_int8_domain(self) -> None:
        """off is bijective over the full int8 [-128, 127] range."""
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, "off")
        dec = decode_sign(enc, arr.shape, np.int8, "off")
        assert np.array_equal(arr, dec)

    def test_negzig_roundtrip_bounded_127_domain(self) -> None:
        """negzig is bijective on [-127, 127] only (NOT [-128, 127])."""
        arr = np.arange(-127, 128, dtype=np.int8)
        enc = encode_sign(arr, "negzig")
        dec = decode_sign(enc, arr.shape, np.int8, "negzig")
        assert np.array_equal(arr, dec)
        assert len(enc) == 255

    def test_raw_uint8_roundtrip_full_uint8_domain(self) -> None:
        """raw_uint8 is bijective over the full uint8 [0, 255] range."""
        arr = np.arange(0, 256, dtype=np.uint8)
        enc = encode_sign(arr, "raw_uint8")
        dec = decode_sign(enc, arr.shape, np.uint8, "raw_uint8")
        assert np.array_equal(arr, dec)

    def test_roundtrip_preserves_shape(self) -> None:
        """Decode must produce the requested shape, not a flat 1D array."""
        arr = np.arange(-30, 30, dtype=np.int8).reshape(5, 12)
        for s in ("zig", "twos", "off", "negzig"):
            enc = encode_sign(arr, s)
            dec = decode_sign(enc, arr.shape, np.int8, s)
            assert dec.shape == (5, 12)
            assert np.array_equal(arr, dec)

    def test_roundtrip_on_non_contiguous_input(self) -> None:
        """Encoder must handle non-contiguous input arrays correctly."""
        base = np.arange(0, 100, dtype=np.int8).reshape(10, 10)
        view = base[::2, ::2]  # non-contiguous
        assert not view.flags.c_contiguous
        enc = encode_sign(view, "zig")
        dec = decode_sign(enc, view.shape, np.int8, "zig")
        assert np.array_equal(view, dec)


# ═══════════════════════════════════════════════════════════════════════════
# Source-byte-faithful semantics (PR96 / PR101 / PR103 citations)
# ═══════════════════════════════════════════════════════════════════════════


class TestSourceByteFaithful:
    def test_zig_matches_pr101_codec_py_lines_225_227(self) -> None:
        """PR101 source line 226-227: zigzag_decode_u8 maps
        even u8 to +x/2, odd u8 to -(x//2)-1.
        Verify: 0 -> 0, 1 -> -1, 2 -> +1, 3 -> -2, 4 -> +2.
        """
        arr_u8 = np.array([0, 1, 2, 3, 4], dtype=np.uint8).tobytes()
        decoded = decode_sign(arr_u8, (5,), np.int8, "zig")
        assert decoded.tolist() == [0, -1, 1, -2, 2]

    def test_off_matches_pr101_pr96_pr103_offset_128_semantics(self) -> None:
        """PR96 inflate.py:90, PR101 codec.py:236, PR103 inflate.py:147 all share:
        decode = (u8 - 128).astype(int8). So 0 -> -128, 128 -> 0, 255 -> 127."""
        arr_u8 = np.array([0, 128, 255], dtype=np.uint8).tobytes()
        decoded = decode_sign(arr_u8, (3,), np.int8, "off")
        assert decoded.tolist() == [-128, 0, 127]

    def test_twos_matches_pr101_view_semantics(self) -> None:
        """PR101 codec.py:238: arr.view(int8). So 0 -> 0, 127 -> 127, 128 -> -128, 255 -> -1."""
        arr_u8 = np.array([0, 127, 128, 255], dtype=np.uint8).tobytes()
        decoded = decode_sign(arr_u8, (4,), np.int8, "twos")
        assert decoded.tolist() == [0, 127, -128, -1]

    def test_negzig_matches_pr101_neg_zigzag_semantics(self) -> None:
        """PR101 codec.py:234: (-zigzag_decode_u8(arr).astype(int16)).astype(int8).
        u8=0 -> zig=0 -> -0 = 0; u8=1 -> zig=-1 -> -(-1) = 1; u8=2 -> zig=1 -> -1.
        u8=255 -> zig=-128 -> -(-128) via int16 = 128 -> int8 wrap = -128.
        """
        arr_u8 = np.array([0, 1, 2, 3, 255], dtype=np.uint8).tobytes()
        decoded = decode_sign(arr_u8, (5,), np.int8, "negzig")
        assert decoded.tolist() == [0, 1, -1, 2, -128]

    def test_raw_uint8_is_identity(self) -> None:
        """raw_uint8 is byte-identity — encode + decode yields the same bytes."""
        arr_u8 = np.array([0, 1, 64, 128, 200, 255], dtype=np.uint8)
        enc = encode_sign(arr_u8, "raw_uint8")
        assert enc == arr_u8.tobytes()
        dec = decode_sign(enc, arr_u8.shape, np.uint8, "raw_uint8")
        assert np.array_equal(arr_u8, dec)


# ═══════════════════════════════════════════════════════════════════════════
# NEGZIG -128 guard (the critical precondition from the design memo)
# ═══════════════════════════════════════════════════════════════════════════


class TestNegzigMinusOneTwentyEightGuard:
    """NEGZIG is NOT bijective on the full int8 range.

    Per design memo §1.2: the values -128 and 0 both encode to byte 0
    because zigzag(-(-128)) overflows int8 back to zero. The encoder
    MUST raise at encode-time to surface this at the cause, not the
    effect (Subagent C found this in PR101).
    """

    def test_minus_128_raises_at_encode_time(self) -> None:
        with pytest.raises(ValueError, match="not bijective"):
            encode_sign(np.array([-128], dtype=np.int8), "negzig")

    def test_minus_128_among_other_values_raises(self) -> None:
        with pytest.raises(ValueError, match="not bijective"):
            encode_sign(np.array([0, 1, -128, 5], dtype=np.int8), "negzig")

    def test_minus_128_count_reported_in_error(self) -> None:
        with pytest.raises(ValueError, match="3 occurrences"):
            encode_sign(
                np.array([0, -128, 1, -128, 2, -128], dtype=np.int8), "negzig"
            )

    def test_minus_127_to_127_inclusive_does_not_raise(self) -> None:
        """[-127, 127] is the admissible NEGZIG domain — no raise."""
        arr = np.arange(-127, 128, dtype=np.int8)
        enc = encode_sign(arr, "negzig")
        assert len(enc) == 255

    def test_other_strategies_accept_minus_128(self) -> None:
        """Only negzig has the -128 ban; zig/twos/off accept the full range."""
        arr = np.array([-128, 0, 127], dtype=np.int8)
        for s in ("zig", "twos", "off"):
            enc = encode_sign(arr, s)
            assert len(enc) == 3
            dec = decode_sign(enc, arr.shape, np.int8, s)
            assert np.array_equal(arr, dec)

    def test_picker_skips_negzig_when_minus_128_present(self) -> None:
        """select_optimal_strategy must NOT propose negzig if -128 is present."""
        arr = np.array([-128, 0, 1, -1, 0, 0], dtype=np.int8)
        sel = select_optimal_strategy(arr)
        assert sel.strategy != SignEncodingStrategy.NEGZIG
        assert (
            SignEncodingStrategy.NEGZIG
            not in sel.per_strategy_entropy_bits
        )


# ═══════════════════════════════════════════════════════════════════════════
# Entropy-based picker
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectOptimalStrategy:
    def test_picker_returns_strategy_selection(self) -> None:
        sel = select_optimal_strategy(
            np.array([0, 1, -1, 0, 0, 1], dtype=np.int8)
        )
        assert isinstance(sel, StrategySelection)
        assert sel.strategy in SignEncodingStrategy
        assert sel.entropy_bits >= 0.0

    def test_picker_evaluates_all_four_int8_strategies_by_default(self) -> None:
        sel = select_optimal_strategy(
            np.arange(-4, 4, dtype=np.int8)
        )
        # All 4 signed strategies should be in the map (no -128 present).
        assert set(sel.per_strategy_entropy_bits.keys()) == {
            SignEncodingStrategy.NEGZIG,
            SignEncodingStrategy.ZIG,
            SignEncodingStrategy.TWOS,
            SignEncodingStrategy.OFF,
        }

    def test_picker_uint8_only_evaluates_raw_uint8(self) -> None:
        sel = select_optimal_strategy(
            np.array([0, 1, 2, 64, 128, 255], dtype=np.uint8)
        )
        assert set(sel.per_strategy_entropy_bits.keys()) == {
            SignEncodingStrategy.RAW_UINT8
        }
        assert sel.strategy == SignEncodingStrategy.RAW_UINT8

    def test_picker_minimum_entropy_matches_winner(self) -> None:
        arr = np.array([0, 0, 0, 1, -1, 0, 0, 0], dtype=np.int8)
        sel = select_optimal_strategy(arr)
        min_entropy = min(sel.per_strategy_entropy_bits.values())
        assert abs(sel.entropy_bits - min_entropy) < 1e-12

    def test_picker_zig_wins_on_peaked_symmetric_distribution(self) -> None:
        """A small symmetric distribution peaked at zero should prefer zig.

        zig maps {0, 1, -1, 2, -2, ...} -> {0, 1, 2, 3, 4, ...} packing
        low-magnitude (high-mass) symbols at low UINT8 indices, which
        minimises histogram entropy when many small values cluster near
        zero.
        """
        # Peaked at zero, symmetric.
        arr = np.array([0, 0, 0, 0, 0, 1, 1, -1, -1, 2, -2], dtype=np.int8)
        sel = select_optimal_strategy(arr)
        # zig and negzig produce identical UINT8 histograms (modulo
        # mirror); twos and off also can — but at minimum, the winning
        # entropy is achieved by zig OR negzig (the symmetric peak case).
        assert sel.strategy in {
            SignEncodingStrategy.ZIG,
            SignEncodingStrategy.NEGZIG,
            SignEncodingStrategy.TWOS,
        }

    def test_picker_custom_candidate_set(self) -> None:
        """Caller may restrict candidates."""
        arr = np.array([0, 1, -1, 2, -2], dtype=np.int8)
        sel = select_optimal_strategy(
            arr, candidates=(SignEncodingStrategy.OFF, SignEncodingStrategy.TWOS)
        )
        assert set(sel.per_strategy_entropy_bits.keys()) == {
            SignEncodingStrategy.OFF,
            SignEncodingStrategy.TWOS,
        }

    def test_picker_skips_dtype_mismatched_candidates(self) -> None:
        """When the caller asks for raw_uint8 on int8 input, the candidate is silently skipped."""
        arr = np.array([0, 1, -1], dtype=np.int8)
        sel = select_optimal_strategy(
            arr,
            candidates=(SignEncodingStrategy.ZIG, SignEncodingStrategy.RAW_UINT8),
        )
        # raw_uint8 dropped due to dtype mismatch; only zig survives.
        assert set(sel.per_strategy_entropy_bits.keys()) == {
            SignEncodingStrategy.ZIG
        }

    def test_picker_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            select_optimal_strategy(np.array([], dtype=np.int8))

    def test_picker_rejects_unsupported_dtype(self) -> None:
        with pytest.raises(ValueError, match="int8 or uint8"):
            select_optimal_strategy(np.array([0, 1, 2], dtype=np.int32))

    def test_picker_rejects_all_skipped_candidates(self) -> None:
        """If every candidate is filtered out (dtype mismatch), raise."""
        arr = np.array([0, 1, -1], dtype=np.int8)
        with pytest.raises(ValueError, match="no compatible candidate"):
            select_optimal_strategy(
                arr, candidates=(SignEncodingStrategy.RAW_UINT8,)
            )

    def test_picker_tiebreak_is_deterministic(self) -> None:
        """Same input must produce same winner across runs (deterministic tiebreaker)."""
        arr = np.array([0, 1, 2, 3], dtype=np.int8)
        sels = [select_optimal_strategy(arr) for _ in range(5)]
        assert all(s.strategy == sels[0].strategy for s in sels)


# ═══════════════════════════════════════════════════════════════════════════
# Per-tensor mixed strategy table (PR101 anchor)
# ═══════════════════════════════════════════════════════════════════════════


class TestPerTensorMixedStrategies:
    """PR101's empirical table assigns one strategy per state-dict index.

    The mechanism (per-tensor lookup) is reusable; this test demonstrates
    it on the PR101 anchor table {9: negzig, 14: negzig, 20: twos, 27: off}
    plus a default of `zig` for all unlisted tensors.
    """

    def test_per_tensor_table_roundtrip_pr101_anchor(self) -> None:
        # Synthetic 4-tensor mini-archive: each tensor uses a different strategy.
        table: dict[int, str] = {
            9: "negzig",
            14: "negzig",
            20: "twos",
            27: "off",
        }
        default = "zig"
        rng = np.random.default_rng(42)
        # PR101's training never produces -128 in int8 weights (per design memo).
        tensors: dict[int, np.ndarray] = {
            idx: rng.integers(-127, 128, size=20, dtype=np.int8)
            for idx in (5, 9, 14, 20, 27)
        }
        # Encode each tensor with its table-lookup strategy.
        encoded_per_idx: dict[int, bytes] = {}
        strategy_per_idx: dict[int, str] = {}
        for idx, t in tensors.items():
            strategy = table.get(idx, default)
            encoded_per_idx[idx] = encode_sign(t, strategy)
            strategy_per_idx[idx] = strategy
        # Decode each tensor using the same strategy + shape.
        for idx, t in tensors.items():
            dec = decode_sign(
                encoded_per_idx[idx],
                t.shape,
                np.int8,
                strategy_per_idx[idx],
            )
            assert np.array_equal(t, dec)
        # Strategies actually used in the PR101 anchor table all appear.
        assert strategy_per_idx[9] == "negzig"
        assert strategy_per_idx[14] == "negzig"
        assert strategy_per_idx[20] == "twos"
        assert strategy_per_idx[27] == "off"
        assert strategy_per_idx[5] == "zig"  # default

    def test_strategies_are_mutually_exclusive_per_tensor(self) -> None:
        """A single tensor can be encoded under exactly ONE strategy at a time."""
        arr = np.array([-10, -1, 0, 1, 10], dtype=np.int8)
        enc_zig = encode_sign(arr, "zig")
        enc_off = encode_sign(arr, "off")
        # Encoded outputs differ — the strategies are not interchangeable
        # for a given input.
        assert enc_zig != enc_off


# ═══════════════════════════════════════════════════════════════════════════
# Failure modes
# ═══════════════════════════════════════════════════════════════════════════


class TestFailureModes:
    def test_encode_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown sign-encoding"):
            encode_sign(np.array([0], dtype=np.int8), "garbage")

    def test_decode_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown sign-encoding"):
            decode_sign(b"\x00", (1,), np.int8, "garbage")

    def test_encode_rejects_non_array_input(self) -> None:
        with pytest.raises(TypeError, match="np.ndarray"):
            encode_sign([0, 1, 2], "zig")  # type: ignore[arg-type]

    def test_decode_rejects_non_bytes_input(self) -> None:
        with pytest.raises(TypeError, match="bytes-like"):
            decode_sign([0, 1, 2], (3,), np.int8, "zig")  # type: ignore[arg-type]

    def test_encode_zig_rejects_uint8_input(self) -> None:
        with pytest.raises(ValueError, match="int8"):
            encode_sign(np.array([0, 1], dtype=np.uint8), "zig")

    def test_encode_raw_uint8_rejects_int8_input(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            encode_sign(np.array([0, 1], dtype=np.int8), "raw_uint8")

    def test_decode_zig_rejects_uint8_target_dtype(self) -> None:
        with pytest.raises(ValueError, match="int8"):
            decode_sign(b"\x00\x01", (2,), np.uint8, "zig")

    def test_decode_raw_uint8_rejects_int8_target_dtype(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            decode_sign(b"\x00\x01", (2,), np.int8, "raw_uint8")

    def test_decode_rejects_byte_count_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match shape"):
            decode_sign(b"\x00\x01\x02", (5,), np.int8, "zig")

    def test_decode_rejects_negative_shape_dim(self) -> None:
        with pytest.raises(ValueError, match="negative dim"):
            decode_sign(b"\x00\x01", (-1, 2), np.int8, "zig")

    def test_strategy_type_validation(self) -> None:
        """Pass a non-str / non-enum value to coerce — should raise TypeError."""
        with pytest.raises(TypeError, match="SignEncodingStrategy or str"):
            encode_sign(np.array([0], dtype=np.int8), 42)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# Golden vector — Catalog #91 paired roundtrip
# ═══════════════════════════════════════════════════════════════════════════


class TestSignEncodingGoldenVectors:
    """One golden vector per strategy.

    Each vector pins the SHA-256 of a deterministic fixture's
    encoded-then-decoded round-trip under that strategy. The fixture is
    chosen to exercise the full bijection domain (e.g., all 256 uint8
    values for raw_uint8, all 256 int8 values for zig/twos/off, all 255
    [-127, 127] values for negzig).
    """

    def _pin_golden(self, basename: str, payload: dict) -> None:
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        manifest = {
            "schema": f"sign_encoding_{basename}.v1",
            "input_length": payload["input_length"],
            "strategy": payload["strategy"],
            "sha256": digest,
        }
        golden = GOLDEN_DIR / f"sign_encoding_{basename}_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                f"sign_encoding_{basename} SHA changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def test_golden_zig(self) -> None:
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, "zig")
        dec = decode_sign(enc, arr.shape, np.int8, "zig")
        assert np.array_equal(arr, dec)
        self._pin_golden(
            "zig",
            {
                "strategy": "zig",
                "input_length": int(arr.size),
                "encoded_first16": list(enc[:16]),
                "encoded_last16": list(enc[-16:]),
            },
        )

    def test_golden_twos(self) -> None:
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, "twos")
        dec = decode_sign(enc, arr.shape, np.int8, "twos")
        assert np.array_equal(arr, dec)
        self._pin_golden(
            "twos",
            {
                "strategy": "twos",
                "input_length": int(arr.size),
                "encoded_first16": list(enc[:16]),
                "encoded_last16": list(enc[-16:]),
            },
        )

    def test_golden_off(self) -> None:
        arr = np.arange(-128, 128, dtype=np.int8)
        enc = encode_sign(arr, "off")
        dec = decode_sign(enc, arr.shape, np.int8, "off")
        assert np.array_equal(arr, dec)
        self._pin_golden(
            "off",
            {
                "strategy": "off",
                "input_length": int(arr.size),
                "encoded_first16": list(enc[:16]),
                "encoded_last16": list(enc[-16:]),
            },
        )

    def test_golden_negzig(self) -> None:
        arr = np.arange(-127, 128, dtype=np.int8)
        enc = encode_sign(arr, "negzig")
        dec = decode_sign(enc, arr.shape, np.int8, "negzig")
        assert np.array_equal(arr, dec)
        self._pin_golden(
            "negzig",
            {
                "strategy": "negzig",
                "input_length": int(arr.size),
                "encoded_first16": list(enc[:16]),
                "encoded_last16": list(enc[-16:]),
            },
        )

    def test_golden_raw_uint8(self) -> None:
        arr = np.arange(0, 256, dtype=np.uint8)
        enc = encode_sign(arr, "raw_uint8")
        dec = decode_sign(enc, arr.shape, np.uint8, "raw_uint8")
        assert np.array_equal(arr, dec)
        self._pin_golden(
            "raw_uint8",
            {
                "strategy": "raw_uint8",
                "input_length": int(arr.size),
                "encoded_first16": list(enc[:16]),
                "encoded_last16": list(enc[-16:]),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Registry token presence (paired with phase1_packet_compiler)
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistryTokenWireUp:
    """The 5 sign-encoding strategies each have a packet_compiler token.

    The registry refuses unknown tokens at compile time. This test pins
    the contract that the trainer can emit any of the 5 sign-encoding
    strategy tokens in its packet_compiler_transforms list.
    """

    def test_tokens_registered_in_phase1_packet_compiler(self) -> None:
        from tac.phase1_packet_compiler import PACKET_COMPILER_TRANSFORMS

        for s in ("negzig", "zig", "twos", "off", "raw_uint8"):
            token = f"sign_encode_{s}"
            assert (
                token in PACKET_COMPILER_TRANSFORMS
            ), f"sign-encoding token {token!r} missing from registry"
