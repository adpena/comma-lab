"""Tests for the score-aware frame-conditional codec module.

Per HNeRV parity discipline lesson 1: the difficulty signal is score-aware.
Per CLAUDE.md non-arbitrariness rule: 3 quantization strategies tested.
Per gate #11 (no-op detector): roundtrip determinism + per-decile bytes
prove the encoded stream actually changes when bit budgets change.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from tac.codec.frame_conditional import (
    DEFAULT_BIT_BUDGET_PER_DECILE,
    FRAME_CONDITIONAL_CODEC_FORMAT,
    FRAME_CONDITIONAL_HEADER_MAGIC,
    FrameConditionalCodecConfig,
    MAX_Q_BITS,
    MIN_Q_BITS,
    N_DECILES,
    allocate_per_decile_bits,
    assign_frame_to_decile,
    decode_frame_conditional,
    encode_frame_conditional,
    estimate_encoded_bytes,
    sha256_hex,
)


def _make_difficulty_profile(n: int = 100, seed: int = 0) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=n).astype(np.float64)
    return {int(i): float(values[i]) for i in range(n)}


def _make_latents(n_frames: int = 100, latent_dim: int = 28, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_frames, latent_dim)).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Config validation


class TestConfig:
    def test_basic_construction(self):
        prof = _make_difficulty_profile(50)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        assert cfg.n_frames() == 50
        assert cfg.bit_budget_per_decile == DEFAULT_BIT_BUDGET_PER_DECILE

    def test_rejects_empty_profile(self):
        with pytest.raises(ValueError, match="non-empty"):
            FrameConditionalCodecConfig(difficulty_profile={})

    def test_rejects_non_dict_profile(self):
        with pytest.raises(TypeError, match="must be dict"):
            FrameConditionalCodecConfig(difficulty_profile=[0.0, 1.0])  # type: ignore[arg-type]

    def test_rejects_bad_decile_count(self):
        with pytest.raises(ValueError, match="N_DECILES|10 entries"):
            FrameConditionalCodecConfig(
                difficulty_profile=_make_difficulty_profile(10),
                bit_budget_per_decile=(4, 4, 4, 4, 4),  # type: ignore[arg-type]
            )

    def test_rejects_q_bits_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            FrameConditionalCodecConfig(
                difficulty_profile=_make_difficulty_profile(10),
                bit_budget_per_decile=(0, 4, 4, 4, 4, 4, 4, 4, 4, 4),
            )
        with pytest.raises(ValueError, match="out of range"):
            FrameConditionalCodecConfig(
                difficulty_profile=_make_difficulty_profile(10),
                bit_budget_per_decile=(4, 4, 4, 4, 4, 4, 4, 4, 4, 9),
            )

    def test_rejects_invalid_strategy(self):
        with pytest.raises(ValueError, match="quantization_strategy"):
            FrameConditionalCodecConfig(
                difficulty_profile=_make_difficulty_profile(10),
                quantization_strategy="bogus",  # type: ignore[arg-type]
            )

    def test_rejects_negative_byte_budget(self):
        with pytest.raises(ValueError, match="non-negative"):
            FrameConditionalCodecConfig(
                difficulty_profile=_make_difficulty_profile(10),
                total_byte_budget=-1,
            )

    def test_three_strategies_accepted(self):
        prof = _make_difficulty_profile(10)
        for s in ("uniform", "per-frame", "per-decile-tied"):
            cfg = FrameConditionalCodecConfig(difficulty_profile=prof, quantization_strategy=s)
            assert cfg.quantization_strategy == s


# ────────────────────────────────────────────────────────────────────────────
# Decile assignment


class TestDecileAssignment:
    def test_decile_partition_balanced(self):
        prof = _make_difficulty_profile(100)
        deciles = assign_frame_to_decile(prof)
        assert deciles.shape == (100,)
        assert deciles.dtype == np.int8
        assert deciles.min() == 0
        assert deciles.max() == N_DECILES - 1
        # 10 frames per decile (100 / 10).
        for d in range(N_DECILES):
            assert (deciles == d).sum() == 10, f"decile {d} should have 10 frames"

    def test_decile_assignment_reflects_difficulty_ordering(self):
        prof = {i: float(i) for i in range(50)}  # monotonic
        deciles = assign_frame_to_decile(prof)
        # Frame 0-4 should be decile 0; frames 45-49 should be decile 9.
        assert deciles[0] == 0
        assert deciles[49] == N_DECILES - 1
        # Monotonicity preserved.
        assert all(deciles[i] <= deciles[i + 1] for i in range(49))

    def test_rejects_non_contiguous_frame_indices(self):
        prof = {0: 1.0, 2: 2.0, 5: 3.0}  # not 0..n-1
        with pytest.raises(ValueError, match="contiguous"):
            assign_frame_to_decile(prof)

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            assign_frame_to_decile({})

    def test_handles_tied_difficulties(self):
        # All frames have same difficulty → stable sort should put them in
        # frame_idx order; deciles partition by index.
        prof = {i: 3.14 for i in range(20)}
        deciles = assign_frame_to_decile(prof)
        assert deciles.shape == (20,)
        # 2 frames per decile.
        for d in range(N_DECILES):
            assert (deciles == d).sum() == 2


# ────────────────────────────────────────────────────────────────────────────
# Per-decile allocation


class TestPerDecileAllocation:
    def test_uniform_strategy_constant_q_bits(self):
        prof = _make_difficulty_profile(100)
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof, quantization_strategy="uniform"
        )
        q = allocate_per_decile_bits(cfg)
        assert q.shape == (100,)
        assert q.dtype == np.uint8
        assert (q == q[0]).all(), "uniform strategy should give constant q-bits"
        # Should equal round(mean(DEFAULT_BIT_BUDGET_PER_DECILE)).
        expected = int(round(float(np.mean(DEFAULT_BIT_BUDGET_PER_DECILE))))
        assert q[0] == expected

    def test_per_decile_tied_uses_decile_budget(self):
        prof = {i: float(i) for i in range(100)}  # monotonic difficulty
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            quantization_strategy="per-decile-tied",
            bit_budget_per_decile=(1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
        )
        q = allocate_per_decile_bits(cfg)
        # Easy frames (low difficulty / low frame_idx) get budget 1; hard get 5.
        assert q[0] == 1
        assert q[99] == 5

    def test_per_frame_strategy_present(self):
        prof = _make_difficulty_profile(50)
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof, quantization_strategy="per-frame"
        )
        q = allocate_per_decile_bits(cfg)
        assert q.shape == (50,)
        # All q-bits in valid range.
        assert (q >= MIN_Q_BITS).all()
        assert (q <= MAX_Q_BITS).all()


# ────────────────────────────────────────────────────────────────────────────
# Encode / decode roundtrip


class TestRoundtrip:
    def test_basic_roundtrip(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        encoded = encode_frame_conditional(latents, cfg)
        decoded, meta = decode_frame_conditional(encoded)
        assert decoded.shape == (50, 28)
        assert decoded.dtype == np.float32
        assert meta["format"] == FRAME_CONDITIONAL_CODEC_FORMAT
        assert meta["strategy"] == "per-decile-tied"
        assert meta["n_frames"] == 50
        assert meta["latent_dim"] == 28

    def test_roundtrip_deterministic(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        encoded_a = encode_frame_conditional(latents, cfg)
        encoded_b = encode_frame_conditional(latents, cfg)
        assert encoded_a == encoded_b
        assert sha256_hex(encoded_a) == sha256_hex(encoded_b)

    def test_roundtrip_all_three_strategies(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        for strategy in ("uniform", "per-frame", "per-decile-tied"):
            cfg = FrameConditionalCodecConfig(
                difficulty_profile=prof, quantization_strategy=strategy
            )
            encoded = encode_frame_conditional(latents, cfg)
            decoded, meta = decode_frame_conditional(encoded)
            assert meta["strategy"] == strategy
            assert decoded.shape == latents.shape

    def test_roundtrip_quantization_loss_bounded(self):
        # With 8-bit q-bits, error should be ≤ 1/127 of dynamic range.
        prof = _make_difficulty_profile(20)
        latents = _make_latents(20, 28, seed=42)
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            bit_budget_per_decile=(8,) * 10,
        )
        encoded = encode_frame_conditional(latents, cfg)
        decoded, _ = decode_frame_conditional(encoded)
        # 8-bit symmetric quantization → max relative error ≈ 1/127.
        rel_err = np.abs(decoded - latents) / (np.abs(latents).max(axis=1, keepdims=True) + 1e-9)
        assert rel_err.max() < 0.02, f"8-bit max rel_err {rel_err.max()} exceeds 2%"

    def test_roundtrip_low_qbits_higher_error(self):
        prof = _make_difficulty_profile(20)
        latents = _make_latents(20, 28, seed=42)
        cfg_low = FrameConditionalCodecConfig(
            difficulty_profile=prof, bit_budget_per_decile=(2,) * 10
        )
        cfg_high = FrameConditionalCodecConfig(
            difficulty_profile=prof, bit_budget_per_decile=(8,) * 10
        )
        decoded_low, _ = decode_frame_conditional(encode_frame_conditional(latents, cfg_low))
        decoded_high, _ = decode_frame_conditional(encode_frame_conditional(latents, cfg_high))
        err_low = np.abs(decoded_low - latents).mean()
        err_high = np.abs(decoded_high - latents).mean()
        assert err_low > err_high, "lower q-bits must yield higher reconstruction error"

    def test_magic_byte_format(self):
        prof = _make_difficulty_profile(10)
        latents = _make_latents(10, 28)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        encoded = encode_frame_conditional(latents, cfg)
        assert encoded[:4] == FRAME_CONDITIONAL_HEADER_MAGIC

    def test_decode_rejects_bad_magic(self):
        with pytest.raises(ValueError, match="bad magic"):
            decode_frame_conditional(b"XXXX" + b"\x00" * 100)

    def test_decode_rejects_truncated_data(self):
        with pytest.raises(ValueError, match="too short"):
            decode_frame_conditional(b"FCC1\x00")

    def test_decode_rejects_unknown_strategy_byte(self):
        prof = _make_difficulty_profile(10)
        latents = _make_latents(10, 28)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        encoded = bytearray(encode_frame_conditional(latents, cfg))
        encoded[4] = 99  # invalid strategy byte
        with pytest.raises(ValueError, match="unknown strategy byte"):
            decode_frame_conditional(bytes(encoded))


# ────────────────────────────────────────────────────────────────────────────
# Byte-budget honoring


class TestByteBudget:
    def test_estimate_matches_actual_encoded_size(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
        estimated = estimate_encoded_bytes(cfg, latent_dim=28)
        actual = len(encode_frame_conditional(latents, cfg))
        assert estimated == actual

    def test_total_byte_budget_enforced(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            total_byte_budget=10,  # impossibly small
        )
        with pytest.raises(ValueError, match="total_byte_budget"):
            encode_frame_conditional(latents, cfg)

    def test_total_byte_budget_passes_when_fits(self):
        prof = _make_difficulty_profile(50)
        latents = _make_latents(50, 28)
        cfg = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            total_byte_budget=100_000,  # generous
        )
        # Should not raise.
        encoded = encode_frame_conditional(latents, cfg)
        assert len(encoded) <= 100_000


# ────────────────────────────────────────────────────────────────────────────
# Per-decile bytes change rendered output (no-op detector)


class TestNoOpDetector:
    def test_changing_per_decile_budget_changes_bytes(self):
        # Per HNeRV parity discipline lesson 11: prove the targeted bytes
        # changed AND were consumed by inflate.
        prof = _make_difficulty_profile(100)
        latents = _make_latents(100, 28)
        cfg_a = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            bit_budget_per_decile=(4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
            quantization_strategy="per-decile-tied",
        )
        cfg_b = FrameConditionalCodecConfig(
            difficulty_profile=prof,
            bit_budget_per_decile=(4, 4, 4, 4, 4, 4, 4, 4, 4, 8),  # last decile +4
            quantization_strategy="per-decile-tied",
        )
        encoded_a = encode_frame_conditional(latents, cfg_a)
        encoded_b = encode_frame_conditional(latents, cfg_b)
        # Bytes must differ.
        assert sha256_hex(encoded_a) != sha256_hex(encoded_b), "no-op detector: bytes did not change"
        # Decoded latents must differ for hardest-decile frames.
        decoded_a, _ = decode_frame_conditional(encoded_a)
        decoded_b, _ = decode_frame_conditional(encoded_b)
        # Find a hardest-decile frame.
        deciles = assign_frame_to_decile(prof)
        hardest = np.where(deciles == 9)[0][0]
        # Higher q-bits should give closer reconstruction for that frame.
        err_a = np.abs(decoded_a[hardest] - latents[hardest]).mean()
        err_b = np.abs(decoded_b[hardest] - latents[hardest]).mean()
        assert err_b < err_a, "hardest-decile reconstruction must improve with more bits"

    def test_changing_strategy_changes_bytes(self):
        prof = _make_difficulty_profile(100)
        latents = _make_latents(100, 28)
        cfg_uniform = FrameConditionalCodecConfig(
            difficulty_profile=prof, quantization_strategy="uniform"
        )
        cfg_tied = FrameConditionalCodecConfig(
            difficulty_profile=prof, quantization_strategy="per-decile-tied"
        )
        encoded_u = encode_frame_conditional(latents, cfg_uniform)
        encoded_t = encode_frame_conditional(latents, cfg_tied)
        assert sha256_hex(encoded_u) != sha256_hex(encoded_t)


# ────────────────────────────────────────────────────────────────────────────
# Integration with score-aware difficulty profile (sister tool format)


class TestScoreAwareProfileIntegration:
    def test_accepts_xray_tool_output_format(self):
        # Mimic the JSON output of tools/xray_per_frame_difficulty_profile.py
        # which is a list of {frame_idx, ..., combined_difficulty} dicts.
        rng = np.random.default_rng(0)
        n = 50
        xray_frames = [
            {
                "frame_idx": int(i),
                "segnet_entropy": float(rng.uniform(0, 2)),
                "posenet_variance": float(rng.uniform(0, 1e-4)),
                "combined_difficulty": float(rng.normal()),
                "percentile_rank": float(i * 100 / (n - 1)),
            }
            for i in range(n)
        ]
        # Convert to difficulty_profile.
        profile = {f["frame_idx"]: f["combined_difficulty"] for f in xray_frames}
        cfg = FrameConditionalCodecConfig(difficulty_profile=profile)
        deciles = assign_frame_to_decile(profile)
        # Should produce 10 deciles, 5 frames each.
        for d in range(N_DECILES):
            assert (deciles == d).sum() == 5

    def test_default_bit_budget_per_decile_is_score_aware_skewed(self):
        # The DEFAULT_BIT_BUDGET_PER_DECILE should give MORE bits to hardest deciles.
        assert DEFAULT_BIT_BUDGET_PER_DECILE[-1] > DEFAULT_BIT_BUDGET_PER_DECILE[0]
        # And be monotonically non-decreasing (no anti-pattern).
        for i in range(len(DEFAULT_BIT_BUDGET_PER_DECILE) - 1):
            assert DEFAULT_BIT_BUDGET_PER_DECILE[i] <= DEFAULT_BIT_BUDGET_PER_DECILE[i + 1]


# ────────────────────────────────────────────────────────────────────────────
# Misc / metadata


def test_metadata_contains_q_bits_and_abs_max():
    prof = _make_difficulty_profile(20)
    latents = _make_latents(20, 28)
    cfg = FrameConditionalCodecConfig(difficulty_profile=prof)
    encoded = encode_frame_conditional(latents, cfg)
    _, meta = decode_frame_conditional(encoded)
    assert "q_bits_per_frame" in meta
    assert len(meta["q_bits_per_frame"]) == 20
    assert "abs_max_per_frame" in meta
    assert len(meta["abs_max_per_frame"]) == 20


def test_constants_exposed():
    # Inflate-runtime LOC budget verification: count public symbols.
    from tac.codec import frame_conditional as fc

    public_attrs = [a for a in fc.__all__ if not a.startswith("_")]
    # Reasonable surface area.
    assert 8 <= len(public_attrs) <= 25, f"public surface {len(public_attrs)} feels off"
    assert "encode_frame_conditional" in public_attrs
    assert "decode_frame_conditional" in public_attrs


def test_inflate_runtime_loc_budget_under_100():
    # Per HNeRV parity discipline lesson 4: inflate.py ≤ 100 LOC.
    # The "inflate runtime" surface is just decode_frame_conditional +
    # _dequantize_codes + unpack_frame_conditional_q_bits dependencies.
    # We approximate by counting the LOC of decode_frame_conditional.
    import inspect

    from tac.codec.frame_conditional import _dequantize_codes, decode_frame_conditional

    decode_loc = len(inspect.getsource(decode_frame_conditional).split("\n"))
    deq_loc = len(inspect.getsource(_dequantize_codes).split("\n"))
    total_inflate_loc = decode_loc + deq_loc
    # ≤ 100 LOC for the inflate-time surface (rounded with imports).
    assert total_inflate_loc <= 100, f"inflate runtime LOC {total_inflate_loc} > 100 budget"
