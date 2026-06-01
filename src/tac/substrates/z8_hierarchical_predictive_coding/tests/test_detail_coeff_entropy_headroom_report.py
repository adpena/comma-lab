# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the Z8 detail-coefficient entropy-headroom report.

Discipline (CLAUDE.md "NO FAKE IMPLEMENTATIONS"): every test exercises the REAL
measurement math — real brotli compression, real Shannon entropy, real quantize
round-trip — on real-shaped coefficient arrays. The report's empirical claim is on
the REAL byte-closed Z8 archive (the live-archive regression guard below); the small
Laplacian fixtures unit-test the math, not a fabricated empirical anchor.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.z8_hierarchical_predictive_coding import detail_entropy_headroom as T

_REPO_ROOT = Path(__file__).resolve().parents[5]


# --------------------------------------------------------------------------- #
# Shannon entropy math (REAL, not stubbed).
# --------------------------------------------------------------------------- #
def test_shannon_bits_uniform_four_symbols_is_two() -> None:
    counts = np.array([25, 25, 25, 25])
    h = T._shannon_bits_per_symbol(counts, 100)
    assert abs(h - 2.0) < 1e-9  # uniform over 4 symbols => log2(4) = 2 bits


def test_shannon_bits_single_symbol_is_zero() -> None:
    counts = np.array([100])
    assert T._shannon_bits_per_symbol(counts, 100) == 0.0


def test_shannon_bits_empty_is_zero() -> None:
    assert T._shannon_bits_per_symbol(np.array([], dtype=np.int64), 0) == 0.0


def test_shannon_bits_skewed_distribution_below_max() -> None:
    counts = np.array([90, 10])
    h = T._shannon_bits_per_symbol(counts, 100)
    # binary entropy of p=0.1
    expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert abs(h - expected) < 1e-9
    assert 0.0 < h < 1.0  # skewed => below the 1-bit uniform-binary max


# --------------------------------------------------------------------------- #
# Structured floor (occupancy + nonzero value entropy).
# --------------------------------------------------------------------------- #
def test_structured_floor_all_zero_is_zero() -> None:
    q = np.zeros(1000, dtype=np.int64)
    assert T._structured_floor_bits_per_coeff(q) == 0.0


def test_structured_floor_all_nonzero_same_value() -> None:
    # p_nonzero=1 => H_bin=0; single nonzero value => H_nonzero=0.
    q = np.full(1000, 3, dtype=np.int64)
    assert T._structured_floor_bits_per_coeff(q) == 0.0


def test_structured_floor_half_sparse_includes_occupancy_bit() -> None:
    # 50% nonzero, all nonzeros == 1 (no value entropy) => floor == H_bin(0.5) == 1.0
    q = np.array([0, 1] * 500, dtype=np.int64)
    floor = T._structured_floor_bits_per_coeff(q)
    assert abs(floor - 1.0) < 1e-9


def test_structured_floor_sparse_is_small() -> None:
    # 1% nonzero => floor dominated by tiny occupancy entropy.
    q = np.zeros(10000, dtype=np.int64)
    q[:100] = 1
    floor = T._structured_floor_bits_per_coeff(q)
    assert 0.0 < floor < 0.1  # << 1 bit/coeff for a sparse band


# --------------------------------------------------------------------------- #
# brotli is REAL and compresses repetitive input.
# --------------------------------------------------------------------------- #
def test_brotli_len_compresses_zeros() -> None:
    payload = bytes(10000)  # all zeros
    assert T._brotli_len(payload) < 200  # LZ77 crushes a zero run


def test_brotli_len_near_random_stays_large() -> None:
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 256, size=10000, dtype=np.uint8).tobytes()
    # incompressible random bytes => brotli output near the input size.
    assert T._brotli_len(payload) > 9000


# --------------------------------------------------------------------------- #
# measure_subband: REAL quantize + entropy-code on Laplacian-shaped coeffs.
# --------------------------------------------------------------------------- #
def _laplacian_coeffs(n: int, scale: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.laplace(loc=0.0, scale=scale, size=n).astype(np.float32)


def test_measure_subband_headroom_positive_and_round_trips() -> None:
    coeffs = _laplacian_coeffs(20000, scale=0.02)
    rep = T.measure_subband(
        coeffs,
        quant_steps=[0.0625, 0.125, 0.25],
        measure_static_range=False,
        static_range_sample_cap=20000,
        key="L0_hh",
    )
    assert rep.n_coeffs == 20000
    assert rep.raw_f32_bytes_per_coeff == 4.0
    # Tiny analog coeffs => raw-f32->brotli pays multiple bytes/coeff (the rate killer).
    assert rep.raw_f32_brotli_bytes_per_coeff > 2.0
    for m in rep.quant:
        # Quantize+entropy-code is a fraction of the current cost (positive headroom).
        assert m.payload_brotli_bytes_per_coeff < rep.raw_f32_brotli_bytes_per_coeff
        # Distortion is finite and non-negative.
        assert m.distortion_mse >= 0.0
        # Live codec picks a real named method.
        assert m.method_name in {
            "qi16_dense",
            "qi16_zero_rle",
            "qi16_static_range",
            "qi16_constriction_range",
            "zigzag_u16_byteplane",
        }


def test_measure_subband_distortion_monotone_in_step() -> None:
    coeffs = _laplacian_coeffs(20000, scale=0.05, seed=3)
    rep = T.measure_subband(
        coeffs,
        quant_steps=[0.0625, 0.25, 1.0],
        measure_static_range=False,
        static_range_sample_cap=20000,
        key="L1_hl",
    )
    dists = [m.distortion_mse for m in rep.quant]
    # Coarser quantization => more distortion (non-decreasing).
    assert dists[0] <= dists[1] <= dists[2]
    # Coarser quantization => fewer nonzeros (more sparse).
    nz = [m.nonzero_fraction for m in rep.quant]
    assert nz[0] >= nz[1] >= nz[2]


def test_measure_subband_v2_payload_is_decodable_round_trip() -> None:
    # The encoder the report consumes must round-trip bijectively (no signal loss
    # beyond the quantization step).
    from tac.substrates.z8_hierarchical_predictive_coding import (
        canonical_quadruple_binding as cqb,
    )

    coeffs = _laplacian_coeffs(4096, scale=0.03, seed=7).reshape(64, 64, 1)
    step = 0.0625
    method, payload = cqb._encode_quantized_detail_payload(coeffs, quantization_step=step)
    decoded = cqb._decode_quantized_detail_payload(
        method=method, payload=payload, shape=(64, 64, 1), quantization_step=step
    )
    # Decoded == quantize(coeffs) * step, bit-for-bit on the integer symbols.
    q = np.rint(coeffs / np.float32(step)).clip(-32768, 32767)
    np.testing.assert_array_equal(decoded, (q.astype(np.float32) * np.float32(step)))


# --------------------------------------------------------------------------- #
# Non-promotable custody markers (Catalog #127/#192/#317/#323/#341).
# --------------------------------------------------------------------------- #
def test_non_promotable_markers_present() -> None:
    m = T.NON_PROMOTABLE_MARKERS
    assert m["evidence_grade"] == "macOS-CPU-advisory"
    assert m["axis_tag"] == "[macOS-CPU advisory]"
    assert m["score_claim"] is False
    assert m["promotion_eligible"] is False
    assert m["promotable"] is False


def test_parse_quant_steps_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        T._parse_quant_steps("0.5,-1.0")
    with pytest.raises(ValueError):
        T._parse_quant_steps("")
    assert T._parse_quant_steps("0.5,1.0,2.0") == [0.5, 1.0, 2.0]


# --------------------------------------------------------------------------- #
# Live-archive regression guard (the REAL empirical surface). Skipped if absent.
# --------------------------------------------------------------------------- #
_LIVE_ARCHIVE = (
    _REPO_ROOT
    / "experiments/results/z8_joint_p18_p19_deadzone_rate_attack"
    / "baseline/byte_closed_archive/0.bin"
)


@pytest.mark.skipif(not _LIVE_ARCHIVE.is_file(), reason="real Z8 archive not present")
def test_live_archive_headroom_is_large() -> None:
    report = T.build_report(
        archive_path=_LIVE_ARCHIVE,
        num_pairs=2,
        quant_steps=[0.0625, 0.25],
        measure_static_range=False,
        static_range_sample_cap=20000,
    )
    assert report["score_claim"] is False
    assert report["promotable"] is False
    assert report["schema"] == "z8_detail_coeff_entropy_headroom_report.v1"
    assert report["archive_sha256"] == hashlib.sha256(_LIVE_ARCHIVE.read_bytes()).hexdigest()
    assert report["pairs_measured"] == 2
    assert report["total_detail_coeffs_measured"] > 0
    # The current raw-f32->brotli detail blob is dominated by headroom (>=80% at the
    # fidelity-preserving Δ=0.0625 end). This is the empirical "rate killer" finding.
    head = next(h for h in report["headline_by_quant_step"] if h["quant_step"] == 0.0625)
    assert head["headroom_fraction"] >= 0.80
    # The v2 codec is at/near the structured Shannon floor (within ~25% slack).
    assert head["v2_codec_detail_bytes"] <= head["structured_shannon_floor_detail_bytes"] * 1.25
