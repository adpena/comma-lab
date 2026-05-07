"""Tests for tac.score_geometry_shannon_floor — closed-form information floor."""
from __future__ import annotations

import math

import pytest

from tac.score_geometry_shannon_floor import (
    compute_shannon_floor,
    shannon_floor_pr101_default,
)


def test_pr101_uniform_floor_exceeds_empirical_archive() -> None:
    """The UNIFORM Shannon floor (assumes worst-case symbol distribution) must
    EXCEED PR101's empirical 162,164-byte decoder blob — because PR101's
    quantized-weight PMF is heavily skewed (clusters near zero) and brotli
    exploits that to compress below the uniform-distribution bound.

    This is the operational lesson: the uniform floor is an UPPER bound on
    the genuine hard limit. The actual hard limit is the empirical-entropy
    floor, which requires per-tensor symbol-frequency analysis (TODO).
    """
    report = shannon_floor_pr101_default()
    archive_overhead = 15_387 + 607 + 100
    decoder_floor = report.total_bytes_uniform_floor - archive_overhead
    # Empirical PR101 decoder is 162,164 B. Uniform floor must EXCEED it
    # since the PMF is non-uniform and brotli exploits that.
    assert decoder_floor > 162_164, (
        f"Uniform Shannon floor {decoder_floor} <= empirical {162_164}: "
        f"would imply uniform PMF, which contradicts brotli's measured efficiency."
    )
    # Sanity: uniform floor at log2(127) ~ 6.989 bits/elem times ~213,792
    # elements ~ 200KB-ish range
    assert 180_000 < decoder_floor < 250_000, (
        f"Uniform floor {decoder_floor} outside expected band [180k, 250k]"
    )


def test_uniform_floor_matches_log2_n_quant_calc() -> None:
    """Hand-check uniform-floor formula on a synthetic schema."""
    schema = (("w", (256, 256)), ("b", (256,)))  # 65,536 + 256 = 65,792 elements
    n_quant = 16  # 4 bits/element uniform
    report = compute_shannon_floor(
        schema=schema, n_quant=n_quant, schema_label="synthetic"
    )
    expected_bytes_no_overhead = math.ceil(65_792 * 4 / 8)  # = 32,896 bytes
    assert report.total_bytes_uniform_floor == expected_bytes_no_overhead
    assert report.total_elements == 65_792


def test_archive_overhead_added() -> None:
    """archive_overhead_bytes is added to the floor."""
    schema = (("w", (100, 100)),)  # 10,000 elements
    report1 = compute_shannon_floor(schema=schema, n_quant=127)
    report2 = compute_shannon_floor(
        schema=schema, n_quant=127, archive_overhead_bytes=1000
    )
    assert report2.total_bytes_uniform_floor == report1.total_bytes_uniform_floor + 1000


def test_empirical_floor_below_uniform() -> None:
    """When per-tensor empirical bits < uniform bits, empirical floor < uniform floor."""
    schema = (("w", (1000, 1000)),)  # 1M elements
    # n_quant=127 → uniform 6.989 bits/elem
    # Empirical 4.0 bits/elem → much lower
    report = compute_shannon_floor(
        schema=schema,
        n_quant=127,
        per_tensor_empirical_bits={"w": 4.0},
    )
    assert report.total_bytes_empirical_floor is not None
    assert report.total_bytes_empirical_floor < report.total_bytes_uniform_floor


def test_n_quant_1_zero_bits() -> None:
    """n_quant=1 means a single symbol → 0 bits/element → 0 bytes (just overhead)."""
    schema = (("w", (1000,)),)
    report = compute_shannon_floor(
        schema=schema, n_quant=1, archive_overhead_bytes=50
    )
    assert report.total_bytes_uniform_floor == 50  # overhead only


def test_zero_distortion_score_matches_information_floor() -> None:
    """The score at the Shannon floor at zero distortion equals the rate-only
    contribution from contest_score()."""
    schema = (("w", (10000,)),)
    report = compute_shannon_floor(schema=schema, n_quant=16)
    # Rate-only score = 25 * bytes / N_REF
    expected = 25.0 * report.total_bytes_uniform_floor / 37_545_489
    assert math.isclose(
        report.score_at_uniform_floor_zero_distortion, expected, rel_tol=1e-9
    )


def test_per_tensor_empirical_falls_back_to_uniform_for_missing() -> None:
    """If a tensor isn't in the empirical dict, it uses the uniform bound."""
    schema = (("w1", (1000,)), ("w2", (1000,)))
    report = compute_shannon_floor(
        schema=schema,
        n_quant=127,
        per_tensor_empirical_bits={"w1": 3.0},  # only w1 supplied
    )
    # w2 should fall back to uniform = log2(127) ~ 6.989
    w2_component = next(c for c in report.components if c.name == "w2")
    assert w2_component.bits_per_element_empirical == w2_component.bits_per_element_uniform


def test_negative_overhead_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_shannon_floor(
            schema=(("w", (10,)),), n_quant=2, archive_overhead_bytes=-1
        )


def test_invalid_n_quant_rejected() -> None:
    with pytest.raises(ValueError, match="n_quant"):
        compute_shannon_floor(schema=(("w", (10,)),), n_quant=0)


def test_pr101_uniform_floor_score_above_canonical_proves_pmf_skew() -> None:
    """The uniform-floor score AT ZERO DISTORTION must exceed PR101's canonical
    archive score floor (0.119 at 178,258 bytes) — this is the FORMAL PROOF that
    the empirical PMF is non-uniform.

    If uniform_floor_score were below canonical_score, that would mean PR101's
    encoder was somehow finding a smaller archive than even an idealized
    encoder on a uniform distribution — physically impossible.

    The fact that uniform_floor_score > canonical_score is direct empirical
    evidence that the symbol distribution is heavily skewed. The 'real' Shannon
    floor (with empirical PMF) is below canonical_score; brotli is approaching
    that empirical floor, not the uniform one."""
    pr101_report = shannon_floor_pr101_default()
    canonical_score = 25.0 * 178_258 / 37_545_489  # ~0.119
    assert pr101_report.score_at_uniform_floor_zero_distortion > canonical_score, (
        f"Uniform-floor score {pr101_report.score_at_uniform_floor_zero_distortion:.5f} "
        f"<= canonical {canonical_score:.5f}: would imply PR101's symbols ARE uniform, "
        f"contradicting empirical brotli compression efficiency."
    )
