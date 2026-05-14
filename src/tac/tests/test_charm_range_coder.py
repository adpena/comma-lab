# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.codec.charm_range_coder import (
    ChARMBitStreamHeader,
    ChARMRangeDecoder,
    ChARMRangeEncoder,
    HEADER_SIZE,
    decode_symbols,
    encode_symbols,
    gaussian_pmf_int8,
    shannon_bits_for_pmf,
)

# Canonical full INT8 alphabet endpoints, named to make the
# `symbol - alphabet_lo → pmf-index` arithmetic explicit instead of magic.
DEFAULT_ALPHABET_LO = -128
DEFAULT_ALPHABET_HI = 127


def _pmfs_for(symbols: list[int]) -> list[np.ndarray]:
    return [
        gaussian_pmf_int8(float(symbol) * 0.25, 1.4, alphabet_lo=-3, alphabet_hi=3)
        for symbol in symbols
    ]


def test_charm_range_coder_roundtrips_dynamic_pmfs() -> None:
    symbols = [-3, -1, 0, 1, 2, -2, 3, 0, 1]
    pmfs = _pmfs_for(symbols)

    encoded = encode_symbols(symbols, pmfs, alphabet=(-3, 3), pmf_total_bits=12)
    decoded = decode_symbols(encoded, pmfs)

    assert decoded == symbols
    header = ChARMBitStreamHeader.from_bytes(encoded)
    assert header.magic == b"CHRC"
    assert header.version == 1
    assert header.alphabet_lo == -3
    assert header.alphabet_hi == 3
    assert header.num_symbols == len(symbols)
    assert header.pmf_total_bits == 12
    assert header.payload_len == len(encoded) - HEADER_SIZE


def test_charm_range_coder_streaming_api_roundtrips() -> None:
    symbols = [0, 1, -1, 2]
    pmfs = _pmfs_for(symbols)
    encoder = ChARMRangeEncoder(alphabet=(-3, 3), pmf_total_bits=10)
    for symbol, pmf in zip(symbols, pmfs):
        encoder.write_symbol(symbol, pmf)
    blob = encoder.finish()

    decoder = ChARMRangeDecoder(blob, alphabet=(-3, 3))
    restored = [decoder.read_symbol(pmf) for pmf in pmfs]

    assert restored == symbols
    assert decoder.num_symbols_read == len(symbols)
    with pytest.raises(RuntimeError, match="already read"):
        decoder.read_symbol(pmfs[-1])
    with pytest.raises(RuntimeError, match="finish"):
        encoder.finish()


def test_charm_range_coder_empty_stream_is_header_only() -> None:
    blob = encode_symbols([], [], alphabet=(-8, 7))

    assert len(blob) == HEADER_SIZE
    decoder = ChARMRangeDecoder(blob, alphabet=(-8, 7))
    assert decoder.num_symbols == 0
    assert decode_symbols(blob, []) == []


def test_charm_range_decoder_rejects_trailing_bytes() -> None:
    symbols = [0, 1]
    pmfs = _pmfs_for(symbols)
    blob = encode_symbols(symbols, pmfs, alphabet=(-3, 3))

    with pytest.raises(ValueError, match="trailing bytes"):
        ChARMRangeDecoder(blob + b"\x00")


def test_charm_range_decoder_rejects_truncated_payload() -> None:
    symbols = [0, 1, 2]
    pmfs = _pmfs_for(symbols)
    blob = encode_symbols(symbols, pmfs, alphabet=(-3, 3))

    with pytest.raises(ValueError, match="truncated"):
        ChARMRangeDecoder(blob[:-1])


def test_charm_range_coder_rejects_shape_and_symbol_mismatch() -> None:
    pmf = gaussian_pmf_int8(0.0, 1.0, alphabet_lo=-3, alphabet_hi=3)
    encoder = ChARMRangeEncoder(alphabet=(-3, 3))

    with pytest.raises(ValueError, match="out of alphabet"):
        encoder.write_symbol(4, pmf)
    with pytest.raises(ValueError, match="pmf shape"):
        encoder.write_symbol(0, pmf[:-1])
    with pytest.raises(ValueError, match="same length"):
        encode_symbols([0], [], alphabet=(-3, 3))
    blob = encode_symbols([0, 1], [pmf, pmf], alphabet=(-3, 3))
    with pytest.raises(ValueError, match="shorter than num_symbols"):
        decode_symbols(blob, [pmf])


def test_charm_range_coder_rejects_nonfinite_pmfs_and_invalid_precision() -> None:
    pmf = gaussian_pmf_int8(0.0, 1.0, alphabet_lo=-3, alphabet_hi=3)
    bad = pmf.copy()
    bad[0] = np.nan

    encoder = ChARMRangeEncoder(alphabet=(-3, 3), pmf_total_bits=3)
    with pytest.raises(ValueError, match="finite"):
        encoder.write_symbol(0, bad)

    blob = encode_symbols([0], [pmf], alphabet=(-3, 3), pmf_total_bits=3)
    decoder = ChARMRangeDecoder(blob, alphabet=(-3, 3))
    with pytest.raises(ValueError, match="finite"):
        decoder.read_symbol(bad)

    with pytest.raises(ValueError, match="exceeds PMF total"):
        ChARMRangeEncoder(alphabet=(-128, 127), pmf_total_bits=7)


def test_gaussian_pmf_rejects_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="mu must be finite"):
        gaussian_pmf_int8(float("nan"), 1.0)
    with pytest.raises(ValueError, match="sigma must be finite"):
        gaussian_pmf_int8(0.0, float("inf"))
    with pytest.raises(ValueError, match="floor_prob must be finite"):
        gaussian_pmf_int8(0.0, 1.0, floor_prob=float("nan"))


def test_gaussian_pmf_int8_is_normalized_positive_and_tail_closed() -> None:
    pmf = gaussian_pmf_int8(200.0, 4.0, alphabet_lo=-3, alphabet_hi=3)

    assert pmf.shape == (7,)
    assert np.all(pmf > 0.0)
    assert math.isclose(float(pmf.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert int(np.argmax(pmf)) == 6


def test_shannon_bits_for_pmf_matches_negative_log2() -> None:
    pmf = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    assert shannon_bits_for_pmf(pmf, 1) == 1.0


# ---------------------------------------------------------------------------
# Synthetic Gaussian roundtrip — 10 seeded test cases (operator spec)
# ---------------------------------------------------------------------------


def _gaussian_roundtrip_case(
    *,
    seed: int,
    n: int,
    mu_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> None:
    rng = np.random.default_rng(seed)
    mus = rng.uniform(mu_range[0], mu_range[1], size=n)
    sigmas = rng.uniform(sigma_range[0], sigma_range[1], size=n)
    raw = rng.normal(mus, sigmas)
    symbols = np.clip(np.round(raw).astype(np.int64), -128, 127).tolist()
    pmfs = [
        gaussian_pmf_int8(mu=float(mus[i]), sigma=float(sigmas[i]))
        for i in range(n)
    ]
    blob = encode_symbols(symbols, pmfs)
    recovered = decode_symbols(blob, pmfs)
    assert recovered == symbols, (
        f"roundtrip failed for seed={seed}, n={n}, mu_range={mu_range}, "
        f"sigma_range={sigma_range}"
    )


@pytest.mark.parametrize(
    "seed,n,mu_range,sigma_range",
    [
        (1, 64, (-2.0, 2.0), (0.5, 2.0)),
        (2, 128, (20.0, 30.0), (1.0, 5.0)),
        (3, 256, (-50.0, 50.0), (5.0, 30.0)),
        (4, 64, (0.0, 5.0), (0.5, 1.0)),
        (5, 2048, (-10.0, 10.0), (2.0, 8.0)),
        (6, 128, (-100.0, 100.0), (40.0, 80.0)),
        (7, 128, (-80.0, -40.0), (2.0, 6.0)),
        (8, 4, (0.0, 5.0), (1.0, 3.0)),
        (9, 128, (120.0, 130.0), (1.0, 3.0)),
        (10, 512, (-30.0, 30.0), (0.5, 50.0)),
    ],
)
def test_synthetic_gaussian_roundtrip(seed, n, mu_range, sigma_range) -> None:
    _gaussian_roundtrip_case(
        seed=seed, n=n, mu_range=mu_range, sigma_range=sigma_range
    )


# ---------------------------------------------------------------------------
# Edge cases — zero sigma rejected, single-symbol stream, near-uniform / near-
# degenerate PMFs
# ---------------------------------------------------------------------------


def test_gaussian_pmf_rejects_zero_or_negative_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_pmf_int8(0.0, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        gaussian_pmf_int8(0.0, -1.0)


def test_single_symbol_stream_roundtrips() -> None:
    pmf = gaussian_pmf_int8(5.0, 2.0)
    blob = encode_symbols([5], [pmf])
    assert decode_symbols(blob, [pmf]) == [5]

    pmf2 = gaussian_pmf_int8(-50.0, 3.0)
    blob2 = encode_symbols([-50], [pmf2])
    assert decode_symbols(blob2, [pmf2]) == [-50]


def test_near_uniform_pmf_within_one_percent_of_shannon() -> None:
    """A wide-Gaussian (interior bins approx-uniform) should code at ~Shannon entropy."""
    rng = np.random.default_rng(42)
    n = 4096
    # Use a sigma where the alphabet's INTERIOR is approximately uniform but
    # tails of the truncated Gaussian don't dominate. Pick mu near the centre
    # so neither boundary bin accumulates mass.
    mu = 0.0
    sigma = 60.0
    raw = rng.normal(mu, sigma, size=n)
    symbols = np.clip(np.round(raw).astype(np.int64), -128, 127).tolist()
    pmf = gaussian_pmf_int8(mu=mu, sigma=sigma)
    pmfs = [pmf for _ in range(n)]

    blob = encode_symbols(symbols, pmfs)
    recovered = decode_symbols(blob, pmfs)
    assert recovered == symbols

    coded_bits = (len(blob) - HEADER_SIZE) * 8
    shannon_bits = sum(shannon_bits_for_pmf(pmf, s - DEFAULT_ALPHABET_LO) for s in symbols)
    ratio = coded_bits / shannon_bits
    assert 0.99 < ratio < 1.01, f"ratio={ratio} outside ±1%"


def test_near_degenerate_pmf_yields_minimal_bits() -> None:
    """All-zero stream under a tight-sigma PMF should code in << 1 bit/symbol.

    Note that the Shannon floor is set by p(symbol=0): for sigma=0.15, the
    per-bin integral at symbol 0 is ~0.999, so Shannon ~0.0012 bits/symbol.
    The actual coded bits/symbol is dominated by the 32-bit finalisation
    overhead (~32/n bits) plus the PMF-quantisation floor.
    """
    n = 4096
    symbols = [0] * n
    pmf = gaussian_pmf_int8(mu=0.0, sigma=0.15)
    pmfs = [pmf for _ in range(n)]
    blob = encode_symbols(symbols, pmfs)
    assert decode_symbols(blob, pmfs) == symbols
    bits_per_symbol = (len(blob) - HEADER_SIZE) * 8 / n
    # Tight bound: sigma=0.15 gives p(0) ≈ 0.999 so Shannon ≈ 0.0012 bits/sym;
    # add ~32-bit finalisation overhead amortised over 4096 symbols (~0.008
    # bits/sym) and a small PMF-quantisation slack ⇒ bound at 0.05.
    assert bits_per_symbol < 0.05


def test_smaller_alphabet_roundtrips() -> None:
    rng = np.random.default_rng(123)
    symbols = rng.integers(-4, 5, size=64).tolist()
    pmfs = []
    for _ in symbols:
        mu = float(rng.uniform(-3, 3))
        sigma = float(rng.uniform(0.5, 2.0))
        pmfs.append(
            gaussian_pmf_int8(mu=mu, sigma=sigma, alphabet_lo=-4, alphabet_hi=4)
        )
    blob = encode_symbols(symbols, pmfs, alphabet=(-4, 4))
    recovered = decode_symbols(blob, pmfs)
    assert recovered == symbols


# ---------------------------------------------------------------------------
# Shannon-rate convergence (operator spec: within ~1% at convergence)
# ---------------------------------------------------------------------------


def test_rate_within_one_percent_long_matched_stream() -> None:
    rng = np.random.default_rng(2026)
    n = 8192
    mu = 0.0
    sigma = 5.0
    raw = rng.normal(mu, sigma, size=n)
    symbols = np.clip(np.round(raw).astype(np.int64), -128, 127).tolist()
    pmf = gaussian_pmf_int8(mu=mu, sigma=sigma)
    pmfs = [pmf for _ in range(n)]
    blob = encode_symbols(symbols, pmfs)
    coded_bits = (len(blob) - HEADER_SIZE) * 8
    shannon_bits = sum(shannon_bits_for_pmf(pmf, s - DEFAULT_ALPHABET_LO) for s in symbols)
    ratio = coded_bits / shannon_bits
    assert 0.99 < ratio < 1.01, f"ratio={ratio} outside ±1%"


def test_rate_within_one_percent_per_symbol_pmfs() -> None:
    rng = np.random.default_rng(7)
    n = 8192
    mus = rng.uniform(-20.0, 20.0, size=n)
    sigmas = rng.uniform(2.0, 10.0, size=n)
    raw = rng.normal(mus, sigmas)
    symbols = np.clip(np.round(raw).astype(np.int64), -128, 127).tolist()
    pmfs = [
        gaussian_pmf_int8(mu=float(mus[i]), sigma=float(sigmas[i]))
        for i in range(n)
    ]
    blob = encode_symbols(symbols, pmfs)
    coded_bits = (len(blob) - HEADER_SIZE) * 8
    shannon_bits = sum(
        shannon_bits_for_pmf(pmfs[i], symbols[i] - DEFAULT_ALPHABET_LO) for i in range(n)
    )
    ratio = coded_bits / shannon_bits
    assert 0.99 < ratio < 1.01, f"ratio={ratio} outside ±1%"


# ---------------------------------------------------------------------------
# Header parsing — bad magic / version / truncation
# ---------------------------------------------------------------------------


def test_header_roundtrip() -> None:
    h = ChARMBitStreamHeader(
        alphabet_lo=-128,
        alphabet_hi=127,
        num_symbols=12345,
        pmf_total_bits=15,
        payload_len=420,
    )
    raw = h.to_bytes()
    assert len(raw) == HEADER_SIZE
    rec = ChARMBitStreamHeader.from_bytes(raw)
    assert (rec.magic, rec.version) == (h.magic, h.version)
    assert (rec.alphabet_lo, rec.alphabet_hi) == (h.alphabet_lo, h.alphabet_hi)
    assert rec.num_symbols == h.num_symbols
    assert rec.pmf_total_bits == h.pmf_total_bits
    assert rec.payload_len == h.payload_len


def test_header_rejects_bad_magic() -> None:
    good = ChARMBitStreamHeader(num_symbols=1, payload_len=5).to_bytes()
    bad = b"XXXX" + good[4:]
    with pytest.raises(ValueError, match="magic"):
        ChARMBitStreamHeader.from_bytes(bad)


def test_header_rejects_truncation() -> None:
    with pytest.raises(ValueError, match="too short"):
        ChARMBitStreamHeader.from_bytes(b"CHRC\x01")


def test_decoder_rejects_bad_magic() -> None:
    pmf = gaussian_pmf_int8(0.0, 2.0)
    blob = encode_symbols([0], [pmf])
    bad = b"NOPE" + blob[4:]
    with pytest.raises(ValueError, match="magic"):
        ChARMRangeDecoder(bad)


def test_decoder_rejects_alphabet_mismatch() -> None:
    pmf = gaussian_pmf_int8(0.0, 2.0)
    blob = encode_symbols([0], [pmf])
    with pytest.raises(ValueError, match="alphabet mismatch"):
        ChARMRangeDecoder(blob, alphabet=(-64, 63))


# ---------------------------------------------------------------------------
# Integration smoke — int8 weight-residual stream similar to ChARM 2020 use
# ---------------------------------------------------------------------------


def test_int8_residual_stream_compresses_below_raw_int8() -> None:
    rng = np.random.default_rng(11)
    symbols = []
    pmfs = []
    n_chunks = 64
    chunk_size = 64
    for _ in range(n_chunks):
        mu = float(rng.uniform(-8, 8))
        sigma = float(rng.uniform(1.0, 6.0))
        chunk = rng.normal(mu, sigma, size=chunk_size)
        chunk = np.clip(np.round(chunk).astype(np.int64), -128, 127)
        symbols.extend(chunk.tolist())
        pmf = gaussian_pmf_int8(mu=mu, sigma=sigma)
        pmfs.extend([pmf for _ in range(chunk_size)])
    blob = encode_symbols(symbols, pmfs)
    assert decode_symbols(blob, pmfs) == symbols
    raw_int8_bytes = len(symbols)
    assert len(blob) < raw_int8_bytes, (
        f"coded {len(blob)} B not smaller than raw INT8 {raw_int8_bytes} B — "
        "ChARM PMFs not informative enough?"
    )


# Closes A4 review R3-1 (uint16 ceiling guard path was untested).
def test_header_to_bytes_rejects_payload_len_above_uint16_ceiling() -> None:
    """`ChARMBitStreamHeader.to_bytes` must raise when payload_len > 65535."""
    too_large = (1 << 16)  # 65536 — first invalid value
    h = ChARMBitStreamHeader(
        version=1,
        alphabet_lo=DEFAULT_ALPHABET_LO,
        alphabet_hi=DEFAULT_ALPHABET_HI,
        num_symbols=1,
        pmf_total_bits=15,
        payload_len=too_large,
    )
    with pytest.raises(ValueError, match="payload_len out of uint16 range"):
        h.to_bytes()


def test_header_to_bytes_accepts_max_uint16_payload_len() -> None:
    """Boundary: 65535 is the LAST valid payload_len (uint16 max)."""
    h = ChARMBitStreamHeader(
        version=1,
        alphabet_lo=DEFAULT_ALPHABET_LO,
        alphabet_hi=DEFAULT_ALPHABET_HI,
        num_symbols=1,
        pmf_total_bits=15,
        payload_len=(1 << 16) - 1,
    )
    # Must NOT raise.
    encoded = h.to_bytes()
    assert len(encoded) == HEADER_SIZE


# Closes A4 review R3-3 (decode_symbols silent-ignore behaviour was untested).
def test_decode_symbols_silently_ignores_pmfs_beyond_num_symbols() -> None:
    """`decode_symbols` may receive more PMFs than the encoded symbol count;
    extras must be silently ignored — only the first ``num_symbols`` consumed.
    """
    # Encode a known stream of 5 symbols on the small (-3..3) alphabet.
    alphabet = (-3, 3)
    pmf = gaussian_pmf_int8(0.0, 1.5, alphabet_lo=-3, alphabet_hi=3)
    symbols = [0, 1, -1, 2, -2]
    pmfs = [pmf for _ in symbols]
    blob = encode_symbols(symbols, pmfs, alphabet=alphabet)

    # Decode passing 5 + 4 extra "wrong" PMFs that must NOT be consumed.
    bogus_pmf = gaussian_pmf_int8(2.5, 0.1, alphabet_lo=-3, alphabet_hi=3)
    pmfs_with_extras = pmfs + [bogus_pmf, bogus_pmf, bogus_pmf, bogus_pmf]
    recovered = decode_symbols(blob, pmfs_with_extras)
    assert recovered == symbols, (
        "extras should not affect decoded symbols; got "
        f"{recovered!r} expected {symbols!r}"
    )


def test_decode_symbols_rejects_too_few_pmfs() -> None:
    """Sister test of the silent-ignore: passing FEWER PMFs than num_symbols
    must raise (per the docstring contract).
    """
    alphabet = (-3, 3)
    pmf = gaussian_pmf_int8(0.0, 1.5, alphabet_lo=-3, alphabet_hi=3)
    symbols = [0, 1, -1, 2, -2]
    pmfs = [pmf for _ in symbols]
    blob = encode_symbols(symbols, pmfs, alphabet=alphabet)

    too_few = pmfs[:-2]  # 3 PMFs for a 5-symbol stream
    with pytest.raises((ValueError, IndexError, AssertionError)):
        decode_symbols(blob, too_few)
