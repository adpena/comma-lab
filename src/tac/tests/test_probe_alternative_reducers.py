# SPDX-License-Identifier: MIT
"""Unit tests for the alternative-reducer probe library.

Per T2 council Q1.4 + Catalog #308 META-pattern E remediation, the 4 alternative
reducers are the canonical reactivation criteria for v2 paradigm-class
disambiguation. These unit tests pin the reducers' correctness on synthetic
inputs BEFORE the expensive SegNet rendering is invoked.

Per CLAUDE.md "EVERY training path MUST use eval_roundtrip" precedent: tests
exist for the deterministic primitives so the rendering driver only has to
contribute decode + SegNet wall-clock, not algorithmic correctness.
"""

from __future__ import annotations

import math

import pytest

from tools.probe_alternative_reducers_latent_class_conditioning import (
    REDUCER_MEANINGFUL_THRESHOLD_BITS,
    compute_alternative_reducer_verdict,
    reduce_per_frame_argmax,
    reduce_per_pair_class_2_fraction,
    reduce_per_pixel_histogram,
    reduce_per_region_histogram,
)

# --- Reducer 1: per-pixel histogram ---


def test_per_pixel_histogram_all_one_class_maps_to_max_bin_for_that_class():
    """All-class-2 input maps to only the class-2 max-quant bin."""
    argmax = [2] * 1000
    fp = reduce_per_pixel_histogram(
        argmax_map=argmax,
        num_classes=5,
        bin_quant_levels=16,
    )
    # All 1000 pixels are class 2, so count[2] = 1000 and others = 0.
    # quantized[2] = floor(1.0 * 16), clamped to 15; others = 0.
    # fingerprint = 15 * 16**2 = 3840.
    expected = 15 * (16 ** 2)
    assert fp == expected


def test_per_pixel_histogram_uniform_distribution_maps_to_balanced_bins():
    """Uniform 5-class input gives each bin floor(0.2 * 16) = 3."""
    argmax = ([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200)
    fp = reduce_per_pixel_histogram(
        argmax_map=argmax,
        num_classes=5,
        bin_quant_levels=16,
    )
    # quantized = [3, 3, 3, 3, 3]; fp = 3 * (1 + 16 + 16^2 + 16^3 + 16^4)
    expected = sum(3 * (16 ** k) for k in range(5))
    assert fp == expected


def test_per_pixel_histogram_two_distinct_distributions_give_distinct_fingerprints():
    a = [2] * 800 + [3] * 200
    b = [2] * 200 + [3] * 800
    fa = reduce_per_pixel_histogram(argmax_map=a, num_classes=5, bin_quant_levels=16)
    fb = reduce_per_pixel_histogram(argmax_map=b, num_classes=5, bin_quant_levels=16)
    assert fa != fb


def test_per_pixel_histogram_rejects_out_of_range_class():
    with pytest.raises(ValueError, match="out of range"):
        reduce_per_pixel_histogram(argmax_map=[0, 1, 7], num_classes=5)


def test_per_pixel_histogram_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        reduce_per_pixel_histogram(argmax_map=[])


# --- Reducer 2: per-region histogram ---


def test_per_region_histogram_uniform_per_region_gives_distinct_fingerprint_vs_other():
    """Distinct per-region distributions give distinct fingerprints."""
    h, w = 8, 8  # 4x4 quadrants
    # Pattern: TL all class 0, TR all class 1, BL all class 2, BR all class 3.
    grid = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if y < 4 and x < 4:
                grid[y][x] = 0
            elif y < 4 and x >= 4:
                grid[y][x] = 1
            elif y >= 4 and x < 4:
                grid[y][x] = 2
            else:
                grid[y][x] = 3
    fp_a = reduce_per_region_histogram(
        argmax_map_2d=grid, num_classes=5, bin_quant_levels=8
    )
    # Alternate pattern: all 4 regions identical class 2 gives a distinct fingerprint.
    grid_b = [[2] * w for _ in range(h)]
    fp_b = reduce_per_region_histogram(
        argmax_map_2d=grid_b, num_classes=5, bin_quant_levels=8
    )
    assert fp_a != fp_b


def test_per_region_histogram_rejects_odd_dimensions():
    grid = [[2] * 7 for _ in range(8)]
    with pytest.raises(ValueError, match="even H and W"):
        reduce_per_region_histogram(argmax_map_2d=grid)


def test_per_region_histogram_rejects_invalid_num_regions():
    grid = [[2] * 8 for _ in range(8)]
    with pytest.raises(ValueError, match="unsupported"):
        reduce_per_region_histogram(argmax_map_2d=grid, num_regions=8)


# --- Reducer 3: per-pair class-2-fraction ---


def test_per_pair_class_2_fraction_all_class_2_maps_to_max_bucket():
    fp = reduce_per_pair_class_2_fraction(
        argmax_map=[2] * 1000, class_index=2, num_buckets=32
    )
    assert fp == 31  # fraction=1.0 gives floor(1.0*32)=32, clamped to 31


def test_per_pair_class_2_fraction_no_class_2_maps_to_bucket_0():
    fp = reduce_per_pair_class_2_fraction(
        argmax_map=[3] * 1000, class_index=2, num_buckets=32
    )
    assert fp == 0


def test_per_pair_class_2_fraction_varying_fractions_give_distinct_buckets():
    """Critical T2 council use case: even when every pair has dominant class 2,
    the per-pair class-2-fraction varies (60%-95% on dashcam) and produces
    distinct conditioning symbols.
    """
    a = [2] * 600 + [3] * 400  # 60% class 2
    b = [2] * 950 + [3] * 50  # 95% class 2
    fa = reduce_per_pair_class_2_fraction(argmax_map=a, class_index=2, num_buckets=32)
    fb = reduce_per_pair_class_2_fraction(argmax_map=b, class_index=2, num_buckets=32)
    assert fa != fb
    assert fa == int(0.6 * 32)  # 19
    assert fb == int(0.95 * 32)  # 30


# --- Reducer 4: per-frame argmax ---


def test_per_frame_argmax_both_frames_same_class_gives_diagonal():
    fp = reduce_per_frame_argmax(
        frame_0_argmax_map=[2] * 100,
        frame_1_argmax_map=[2] * 100,
        num_classes=5,
    )
    assert fp == 2 * 5 + 2  # 12


def test_per_frame_argmax_distinct_frames_give_off_diagonal():
    fp = reduce_per_frame_argmax(
        frame_0_argmax_map=[1] * 100,
        frame_1_argmax_map=[3] * 100,
        num_classes=5,
    )
    assert fp == 1 * 5 + 3  # 8


def test_per_frame_argmax_frame_0_none_degenerates_to_diagonal():
    fp = reduce_per_frame_argmax(
        frame_0_argmax_map=None,
        frame_1_argmax_map=[3] * 100,
        num_classes=5,
    )
    assert fp == 3 * 5 + 3  # 18


# --- compute_alternative_reducer_verdict ---


def test_verdict_independent_when_per_pair_reduced_class_is_degenerate():
    """All pairs map to the same per-pair-reduced-class, so MI = 0.

    This pins the existing v2 per-pair-dominant result behavior under the
    alternative-reducer probe framework: any reducer that collapses to a single
    fingerprint will still produce INDEPENDENT.
    """
    n_pairs = 600
    symbols_per_pair = 28
    latent_stream = bytes(i % 251 for i in range(n_pairs * symbols_per_pair))
    per_pair_reduced = [42] * n_pairs  # all same fingerprint
    v = compute_alternative_reducer_verdict(
        substrate_id="synthetic_degenerate",
        reducer_name="per_pixel_histogram",
        latent_stream=latent_stream,
        per_pair_reduced_class=per_pair_reduced,
        symbols_per_pair=symbols_per_pair,
    )
    assert v.verdict == "INDEPENDENT"
    assert v.mutual_information_bits < 1e-9
    assert v.num_unique_reduced_classes == 1


def test_verdict_meaningful_when_per_pair_class_perfectly_predicts_latent():
    """Latent symbol = per-pair-class repeated, so MI = H(class).

    Construct a deliberately-correlated stream where the reducer's per-pair
    output IS the latent symbol. MI(latent; class) = H(class) which for
    uniform-over-5-classes is log2(5) = 2.32 bits > meaningful threshold of 0.5.
    """
    n_pairs = 600
    symbols_per_pair = 28
    per_pair_reduced = [i % 5 for i in range(n_pairs)]
    latent_stream = bytes(per_pair_reduced[i // symbols_per_pair] for i in range(n_pairs * symbols_per_pair))
    v = compute_alternative_reducer_verdict(
        substrate_id="synthetic_correlated",
        reducer_name="per_pixel_histogram",
        latent_stream=latent_stream,
        per_pair_reduced_class=per_pair_reduced,
        symbols_per_pair=symbols_per_pair,
    )
    assert v.verdict == "MEANINGFUL_CONDITIONING"
    # I(latent; class) = H(latent) = log2(5) when latent is fully determined by class
    assert abs(v.mutual_information_bits - math.log2(5)) < 1e-9


def test_verdict_weak_when_partial_correlation():
    """Latent stream with NOISE + per-pair class signal gives WEAK_CONDITIONING.

    Half-random, half-correlated bytes per pair: MI is positive but small.
    """
    n_pairs = 600
    symbols_per_pair = 28
    per_pair_reduced = [i % 5 for i in range(n_pairs)]
    latent_bytes = bytearray()
    for p in range(n_pairs):
        for s in range(symbols_per_pair):
            if s < 4:
                latent_bytes.append(per_pair_reduced[p])  # correlated head
            else:
                latent_bytes.append((p * 31 + s * 13) % 251)  # noise tail
    v = compute_alternative_reducer_verdict(
        substrate_id="synthetic_partial",
        reducer_name="per_pixel_histogram",
        latent_stream=bytes(latent_bytes),
        per_pair_reduced_class=per_pair_reduced,
        symbols_per_pair=symbols_per_pair,
    )
    # With 4/28 = 14% perfectly correlated symbols, MI is about
    # 0.14 * log2(5), or about 0.33 bits, below the 0.5 threshold.
    assert v.verdict == "WEAK_CONDITIONING"
    assert 0.01 < v.mutual_information_bits < 0.5


def test_verdict_rejects_mismatched_stream_lengths():
    with pytest.raises(ValueError, match="symbols_per_pair"):
        compute_alternative_reducer_verdict(
            substrate_id="synthetic_mismatched",
            reducer_name="per_pair_class_2_fraction",
            latent_stream=b"abcdefgh",
            per_pair_reduced_class=[0, 1, 2],
            symbols_per_pair=4,  # 3 * 4 = 12 != 8 = len(latent_stream)
        )


def test_verdict_rejects_unknown_reducer():
    with pytest.raises(ValueError, match="not in canonical reducer set"):
        compute_alternative_reducer_verdict(
            substrate_id="synthetic_bad_reducer",
            reducer_name="nonsense_reducer",  # type: ignore[arg-type]
            latent_stream=b"abcd",
            per_pair_reduced_class=[0],
            symbols_per_pair=4,
        )


def test_verdict_carries_canonical_evidence_grade_and_score_claim():
    """Per CLAUDE.md Catalog #127/#192/#249: every probe verdict MUST carry
    diagnostic_cpu evidence grade + score_claim=false + explicit axis label.
    """
    v = compute_alternative_reducer_verdict(
        substrate_id="canonical_evidence_grade_check",
        reducer_name="per_pair_class_2_fraction",
        latent_stream=b"abcdefgh",
        per_pair_reduced_class=[0, 1],
        symbols_per_pair=4,
    )
    assert v.evidence_grade == "diagnostic_cpu"
    assert v.score_claim is False
    assert "[diagnostic-CPU" in v.axis_label
    assert "alternative-reducer" in v.axis_label


# --- Canonical thresholds pinned per T2 council Q1.4 ---


def test_canonical_thresholds_match_t2_council_q14():
    """The 4 reducer MI thresholds are pinned per T2 council Q1.4 sextet-pact
    binding verdict 2026-05-16. Future tightening requires a new council
    deliberation per CLAUDE.md "Design decisions - non-negotiable".
    """
    assert REDUCER_MEANINGFUL_THRESHOLD_BITS["per_pixel_histogram"] == 0.5
    assert REDUCER_MEANINGFUL_THRESHOLD_BITS["per_region_histogram"] == 1.0
    assert REDUCER_MEANINGFUL_THRESHOLD_BITS["per_pair_class_2_fraction"] == 0.2
    assert REDUCER_MEANINGFUL_THRESHOLD_BITS["per_frame_argmax"] == 0.2
