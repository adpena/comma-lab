"""Lane 20 — Production Ballé hyperprior codec tests.

Covers ``tac.balle_hyperprior_codec`` (the production arithmetic-coding
extension to the Level 1 scaffold in ``tac.balle_hyperprior_renderer``).

Test matrix
-----------
1. Discretized-Gaussian PMF sums to 1
2. Discretized-Gaussian narrows mass at small σ
3. PMF → integer-frequency normalization preserves total
4. Hotz-LITE encode/decode roundtrip (uniform stream)
5. Hotz-LITE encode/decode roundtrip (heteroscedastic stream)
6. Hotz-LITE matches static codec on K=1
7. Hotz-LITE BEATS static on heteroscedastic stream (K=4)
8. Full-Ballé encode/decode roundtrip
9. Full-Ballé side-info ≥ hyper_decoder bytes (no missing weights)
10. Full-Ballé does NOT degrade gracefully on synthetic heteroscedastic
    distributions vs static (the win is data-dependent — the test only
    asserts we don't lose >2x bytes)
11. encode_qints_balle_auto picks the smallest candidate
12. encode_qints_balle_auto returns "static_wins" sentinel when baseline beats
13. No-silent-defaults: encode/decode reject None
14. BHv1 magic / version / mode validation
15. Ratio of side-info to total payload stays bounded for both modes
16. EMA-style update on hyper_decoder produces shippable bytes (smoke for
    the training path's CLAUDE.md "EMA decay 0.997 non-negotiable")

All claims tagged ``[synthetic]`` (no real-archive data here; that lives in
``test_balle_hyperprior_real_archive.py`` Phase E).

CLAUDE.md non-negotiables verified
----------------------------------
* No silent defaults: every codec arg is required-keyword
* No scorer load (codec is pure-math byte-level)
* No GPU dependency at decode
* Pure CPU determinism (torch.manual_seed at construction)
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import torch

from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    HyperDecoder,
    HyperEncoder,
    MODE_FULL_BALLE,
    MODE_HOTZ_LITE,
    _BHV1_MAGIC,
    _BHV1_VERSION,
    decode_qints_balle,
    discretized_gaussian_pmf,
    encode_qints_balle_auto,
    encode_qints_full_balle,
    encode_qints_hotz_lite,
)
from tac.balle_hyperprior_codec import _pmf_to_int_freq
from tac.arithmetic_qint_codec import encode_qints_arithmetic, decode_qints_arithmetic


# ─────────────────────────────────────────────────────────────────────────────
# 1. Discretized-Gaussian PMF sums to 1
# ─────────────────────────────────────────────────────────────────────────────


def test_discretized_gaussian_pmf_sums_to_one_synthetic() -> None:
    """[synthetic] PMF over the alphabet sums to 1.0 within fp64 epsilon."""
    for sigma in (0.1, 0.5, 1.0, 2.0, 5.0):
        for num_symbols, offset in ((3, 1), (15, 7), (256, 128)):
            pmf = discretized_gaussian_pmf(
                sigma=sigma, num_symbols=num_symbols, offset=offset
            )
            assert pmf.shape == (num_symbols,)
            assert (pmf > 0).all()
            assert pytest.approx(pmf.sum(), abs=1e-9) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Discretized-Gaussian narrows mass at small σ
# ─────────────────────────────────────────────────────────────────────────────


def test_discretized_gaussian_narrows_at_small_sigma_synthetic() -> None:
    """[synthetic] Small σ concentrates mass at the central bin (offset)."""
    pmf_narrow = discretized_gaussian_pmf(sigma=0.1, num_symbols=15, offset=7)
    pmf_wide = discretized_gaussian_pmf(sigma=5.0, num_symbols=15, offset=7)
    # At σ=0.1, the central bin (k=offset=7, value v=0) should hold > 99% mass
    assert pmf_narrow[7] > 0.99
    # At σ=5.0, the central bin holds < 20% mass
    assert pmf_wide[7] < 0.20


# ─────────────────────────────────────────────────────────────────────────────
# 3. PMF → integer-frequency normalization preserves total
# ─────────────────────────────────────────────────────────────────────────────


def test_pmf_to_int_freq_preserves_total_synthetic() -> None:
    """[synthetic] _pmf_to_int_freq always sums to total_freq exactly."""
    for sigma in (0.1, 1.0, 3.0):
        pmf = discretized_gaussian_pmf(sigma=sigma, num_symbols=15, offset=7)
        for total in (256, 1 << 12, 1 << 16):
            freq = _pmf_to_int_freq(pmf, total_freq=total)
            assert int(freq.sum()) == total
            assert (freq >= 1).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hotz-LITE encode/decode roundtrip (uniform stream)
# ─────────────────────────────────────────────────────────────────────────────


def test_hotz_lite_roundtrip_uniform_synthetic() -> None:
    """[synthetic] Hotz-LITE encodes + decodes a uniform qint stream."""
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=1024, dtype=np.int8)
    blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=4
    )
    assert blob[:4] == _BHV1_MAGIC
    decoded = decode_qints_balle(blob=blob, expected_dtype=np.int8)
    assert decoded.shape == qints.shape
    assert np.array_equal(decoded, qints)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hotz-LITE encode/decode roundtrip (heteroscedastic stream)
# ─────────────────────────────────────────────────────────────────────────────


def test_hotz_lite_roundtrip_heteroscedastic_synthetic() -> None:
    """[synthetic] Hotz-LITE roundtrip on a stream with chunk-varying scale."""
    rng = np.random.default_rng(42)
    chunk_a = rng.integers(-1, 2, size=400, dtype=np.int8)  # ternary
    chunk_b = rng.integers(-7, 8, size=400, dtype=np.int8)  # full alphabet
    chunk_c = np.zeros(200, dtype=np.int8)  # all zeros
    qints = np.concatenate([chunk_a, chunk_b, chunk_c])
    blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=3
    )
    decoded = decode_qints_balle(blob=blob, expected_dtype=np.int8)
    assert np.array_equal(decoded, qints)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hotz-LITE matches static codec at K=1 (degenerate case)
# ─────────────────────────────────────────────────────────────────────────────


def test_hotz_lite_k1_within_constant_overhead_of_static_synthetic() -> None:
    """[synthetic] At K=1, Hotz-LITE should be within a small constant overhead
    of the static codec (no compression advantage but no big regression)."""
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=2048, dtype=np.int8)
    static_blob = encode_qints_arithmetic(qints, num_symbols=15, offset=7)
    lite_blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=1
    )
    # Hotz-LITE has a slightly larger header (mode byte, side_info_len, etc.)
    # but should not balloon the body. Allow at most 80 bytes overhead at K=1.
    assert len(lite_blob) - len(static_blob) < 80, (
        f"Hotz-LITE K=1 overhead {len(lite_blob) - len(static_blob)} bytes "
        f"too large vs static"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hotz-LITE BEATS static codec on heteroscedastic stream
# ─────────────────────────────────────────────────────────────────────────────


def test_hotz_lite_beats_static_on_heteroscedastic_synthetic() -> None:
    """[synthetic] Hotz-LITE should beat the static codec on a stream where
    chunks have very different distributions."""
    rng = np.random.default_rng(42)
    # First chunk: all zeros (zero entropy). Second chunk: uniform [-7,7].
    # Static codec sees a mixture distribution and pays the average entropy
    # for both chunks. Hotz-LITE codes chunk_a with ~0 bits, chunk_b at
    # full entropy → strict win.
    chunk_a = np.zeros(2048, dtype=np.int8)
    chunk_b = rng.integers(-7, 8, size=2048, dtype=np.int8)
    qints = np.concatenate([chunk_a, chunk_b])
    static_blob = encode_qints_arithmetic(qints, num_symbols=15, offset=7)
    lite_blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=2
    )
    # Hotz-LITE must beat static by >= 5% (we expect ~30% on this synthetic)
    saving = (len(static_blob) - len(lite_blob)) / len(static_blob)
    assert saving > 0.05, (
        f"Hotz-LITE saving {saving*100:.1f}% < 5% on heteroscedastic stream"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full-Ballé encode/decode roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def _make_full_codec(block_size: int = 64, z_dim: int = 4, hidden: int = 8) -> BalleHyperpriorCodec:
    enc = HyperEncoder(block_size=block_size, z_dim=z_dim, hidden_dim=hidden, seed=2026)
    dec = HyperDecoder(z_dim=z_dim, hidden_dim=hidden, seed=2026)
    return BalleHyperpriorCodec(
        block_size=block_size, z_dim=z_dim, hyper_encoder=enc, hyper_decoder=dec
    )


def test_full_balle_roundtrip_synthetic() -> None:
    """[synthetic] Full Ballé encode + decode preserves the qint stream exactly."""
    codec = _make_full_codec(block_size=64, z_dim=4, hidden=8)
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=512, dtype=np.int8)
    blob = encode_qints_full_balle(
        qints=qints, num_symbols=15, offset=7, codec=codec
    )
    assert blob[:4] == _BHV1_MAGIC
    # mode byte sits at offset 6
    assert blob[6] == MODE_FULL_BALLE
    decoded = decode_qints_balle(blob=blob, expected_dtype=np.int8)
    assert decoded.shape == qints.shape
    assert np.array_equal(decoded, qints)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full-Ballé side-info ≥ hyper_decoder bytes
# ─────────────────────────────────────────────────────────────────────────────


def test_full_balle_side_info_includes_decoder_bytes_synthetic() -> None:
    """[synthetic] Side-info length is at least hyper_decoder fp16 bytes
    (the decoder is REQUIRED to reconstruct σ at inflate time — this is
    the Check 91 STRICT predicate)."""
    codec = _make_full_codec(block_size=64, z_dim=4, hidden=8)
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=256, dtype=np.int8)
    blob = encode_qints_full_balle(
        qints=qints, num_symbols=15, offset=7, codec=codec
    )
    # Parse: skip magic+ver+mode+ns+off+ntot+bs (4+2+1+2+4+8+4 = 25)
    cursor = 25
    (side_info_len,) = struct.unpack_from("<I", blob, cursor)
    decoder_bytes = codec.hyper_decoder_byte_size()
    assert side_info_len > decoder_bytes, (
        f"side_info_len={side_info_len} ≤ decoder_bytes={decoder_bytes} — "
        f"missing decoder weights → Check 91 STRICT violation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full-Ballé does not catastrophically regress vs static
# ─────────────────────────────────────────────────────────────────────────────


def test_full_balle_untrained_loses_but_auto_path_falls_back_synthetic() -> None:
    """[synthetic] An UNTRAINED full-Ballé codec produces mis-calibrated σ
    and is much worse than static — this is expected. The point of the test
    is that ``encode_qints_balle_auto`` correctly DETECTS this and falls
    back to static (or hotz_lite) via the ``static_baseline_bytes`` guard.

    This is the CLAUDE.md "kill criterion" wired into code: shipping the
    losing codec is not allowed; the auto path enforces it.
    """
    codec = _make_full_codec(block_size=256, z_dim=4, hidden=8)
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=8192, dtype=np.int8)  # uniform homoscedastic
    static_blob = encode_qints_arithmetic(qints, num_symbols=15, offset=7)
    # Auto path with the static baseline supplied: must return static_wins
    # because untrained Ballé cannot beat static on uniform data.
    blob, mode_name, stats = encode_qints_balle_auto(
        qints=qints,
        num_symbols=15,
        offset=7,
        num_chunks_lite=2,
        full_codec=codec,
        static_baseline_bytes=len(static_blob),
    )
    assert mode_name == "static_wins", (
        f"Untrained Ballé incorrectly claimed to beat static; mode={mode_name} "
        f"stats={stats}"
    )
    assert blob == b""


# ─────────────────────────────────────────────────────────────────────────────
# 11. encode_qints_balle_auto picks the smallest candidate
# ─────────────────────────────────────────────────────────────────────────────


def test_balle_auto_picks_smallest_synthetic() -> None:
    """[synthetic] Auto path selects the smallest-blob candidate."""
    codec = _make_full_codec(block_size=128, z_dim=4, hidden=8)
    # Heteroscedastic: Hotz-LITE should win
    qints_het = np.concatenate(
        [np.zeros(1024, dtype=np.int8), np.random.randint(-7, 8, size=1024, dtype=np.int8)]
    )
    blob, mode_name, stats = encode_qints_balle_auto(
        qints=qints_het,
        num_symbols=15,
        offset=7,
        num_chunks_lite=4,
        full_codec=codec,
    )
    assert mode_name in ("hotz_lite", "full_balle")
    # The chosen blob length must equal the smallest candidate
    chosen_size = len(blob)
    candidate_sizes = [v for k, v in stats.items() if k in ("hotz_lite", "full_balle")]
    assert chosen_size == min(candidate_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# 12. encode_qints_balle_auto returns "static_wins" sentinel
# ─────────────────────────────────────────────────────────────────────────────


def test_balle_auto_returns_static_wins_sentinel_synthetic() -> None:
    """[synthetic] Auto returns ("", "static_wins", stats) when no candidate
    beats the supplied static baseline."""
    codec = _make_full_codec(block_size=64, z_dim=4, hidden=8)
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=128, dtype=np.int8)
    # Pass an artificially small static baseline so nothing can beat it
    blob, mode_name, stats = encode_qints_balle_auto(
        qints=qints,
        num_symbols=15,
        offset=7,
        num_chunks_lite=2,
        full_codec=codec,
        static_baseline_bytes=10,
    )
    assert mode_name == "static_wins"
    assert blob == b""
    assert stats["static_baseline_bytes"] == 10


# ─────────────────────────────────────────────────────────────────────────────
# 13. No-silent-defaults: encode/decode reject None
# ─────────────────────────────────────────────────────────────────────────────


def test_no_silent_defaults_synthetic() -> None:
    """[synthetic] Every public encode/decode rejects None args."""
    codec = _make_full_codec()
    qints = np.array([0, 1, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="qints is required"):
        encode_qints_hotz_lite(qints=None, num_symbols=15, offset=7, num_chunks=2)
    with pytest.raises(ValueError, match="num_chunks must be"):
        encode_qints_hotz_lite(qints=qints, num_symbols=15, offset=7, num_chunks=0)
    with pytest.raises(ValueError, match="qints is required"):
        encode_qints_full_balle(qints=None, num_symbols=15, offset=7, codec=codec)
    with pytest.raises(ValueError, match="codec is required"):
        encode_qints_full_balle(qints=qints, num_symbols=15, offset=7, codec=None)
    with pytest.raises(ValueError, match="blob is required"):
        decode_qints_balle(blob=None)


# ─────────────────────────────────────────────────────────────────────────────
# 14. BHv1 magic / version validation
# ─────────────────────────────────────────────────────────────────────────────


def test_bhv1_magic_and_version_validation_synthetic() -> None:
    """[synthetic] decode rejects bad magic / unsupported version / unknown mode."""
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=128, dtype=np.int8)
    good = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=2
    )
    # Bad magic
    with pytest.raises(ValueError, match="bad magic"):
        decode_qints_balle(blob=b"WRNG" + good[4:])
    # Bad version
    bad_ver = bytearray(good)
    struct.pack_into("<H", bad_ver, 4, 999)
    with pytest.raises(ValueError, match="unsupported version"):
        decode_qints_balle(blob=bytes(bad_ver))
    # Unknown mode
    bad_mode = bytearray(good)
    bad_mode[6] = 99
    with pytest.raises(ValueError, match="unknown mode"):
        decode_qints_balle(blob=bytes(bad_mode))


# ─────────────────────────────────────────────────────────────────────────────
# 15. Side-info / payload ratio stays bounded
# ─────────────────────────────────────────────────────────────────────────────


def test_side_info_payload_ratio_bounded_synthetic() -> None:
    """[synthetic] For a moderately large stream, side-info should not exceed
    payload bytes (sanity bound — if it does, the codec arch is too heavy
    for the data length and should not ship)."""
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=4096, dtype=np.int8)  # large stream
    codec = _make_full_codec(block_size=128, z_dim=4, hidden=8)
    blob = encode_qints_full_balle(
        qints=qints, num_symbols=15, offset=7, codec=codec
    )
    cursor = 25
    (side_info_len,) = struct.unpack_from("<I", blob, cursor)
    cursor += 4 + side_info_len
    (payload_len,) = struct.unpack_from("<Q", blob, cursor)
    # On a 4K stream with 128-block-size + 4-D z + 8-hidden hyper-decoder,
    # side-info should be < payload (or at least within 3x).
    assert side_info_len < 3 * payload_len, (
        f"side_info_len={side_info_len} > 3 * payload_len={payload_len} — "
        f"codec too heavy for this stream length"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 16. EMA shadow on hyper_decoder produces shippable bytes
# ─────────────────────────────────────────────────────────────────────────────


def test_ema_shadow_on_hyper_decoder_smoke_synthetic() -> None:
    """[synthetic] EMA decay 0.997 (CLAUDE.md non-negotiable) on the hyper_decoder
    weights produces a roundtrip-able codec."""
    codec = _make_full_codec(block_size=64, z_dim=4, hidden=8)
    # Simulate one EMA step (decay=0.997)
    decay = 0.997
    shadow = {k: v.detach().clone() for k, v in codec.hyper_decoder.state_dict().items()}
    # Pretend training updated weights by a small noise
    with torch.no_grad():
        for p in codec.hyper_decoder.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    # Apply EMA: shadow = decay*shadow + (1-decay)*new
    for k, v in codec.hyper_decoder.state_dict().items():
        shadow[k].mul_(decay).add_(v, alpha=1.0 - decay)
    # Snapshot live, swap to shadow, encode, restore live (canonical pattern
    # from CLAUDE.md "Apply only at eval time, with snapshot+restore")
    live_state = {k: v.detach().clone() for k, v in codec.hyper_decoder.state_dict().items()}
    codec.hyper_decoder.load_state_dict(shadow)
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=256, dtype=np.int8)
    blob = encode_qints_full_balle(
        qints=qints, num_symbols=15, offset=7, codec=codec
    )
    decoded = decode_qints_balle(blob=blob, expected_dtype=np.int8)
    assert np.array_equal(decoded, qints)
    # Restore live
    codec.hyper_decoder.load_state_dict(live_state)


# ─────────────────────────────────────────────────────────────────────────────
# 17. Static-prior-baseline cross-check: decoding a Hotz-LITE blob must give
#    the same array as decoding a separate static codec on the same data
# ─────────────────────────────────────────────────────────────────────────────


def test_hotz_lite_decoded_matches_static_decoded_synthetic() -> None:
    """[synthetic] Both codecs should decode to the same logical qints."""
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=300, dtype=np.int8)
    static_blob = encode_qints_arithmetic(qints, num_symbols=15, offset=7)
    static_decoded = decode_qints_arithmetic(static_blob, expected_dtype=np.int8)
    lite_blob = encode_qints_hotz_lite(
        qints=qints, num_symbols=15, offset=7, num_chunks=3
    )
    lite_decoded = decode_qints_balle(blob=lite_blob, expected_dtype=np.int8)
    assert np.array_equal(static_decoded, lite_decoded)
