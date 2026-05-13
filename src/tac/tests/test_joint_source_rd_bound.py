"""Tests for joint-source R(D) lower bounds (Berger 1971 §4.5).

Eureka mechanism — Fields-medal council 2026-05-09. The module under test
operationalizes Shannon's joint-source-correlation reduction of the i.i.d.
Shannon floor identified in
``feedback_grand_council_fields_medal_phase2_floor_refinement_20260509.md``
§6 (Shannon position).
"""
from __future__ import annotations

import math

import pytest

from tac.joint_source_rd_bound import (
    JointSourceStream,
    compute_joint_source_floor,
    gauss_markov_bit_savings_per_symbol,
    joint_floor_from_shannon_report,
    per_pair_sqrt_n_budget,
    predicted_score_at_joint_floor,
)
from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    POSE_COEFFICIENT_INSIDE_SQRT,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
    contest_score,
)
from tac.score_geometry_shannon_floor import (
    ShannonFloorComponent,
    ShannonFloorReport,
)


# ---------------------------------------------------------------------------
# gauss_markov_bit_savings_per_symbol — closed-form correctness
# ---------------------------------------------------------------------------


def test_savings_zero_at_rho_zero() -> None:
    assert gauss_markov_bit_savings_per_symbol(0.0) == 0.0


def test_savings_symmetric_in_sign_of_rho() -> None:
    pos = gauss_markov_bit_savings_per_symbol(0.7)
    neg = gauss_markov_bit_savings_per_symbol(-0.7)
    assert pos == pytest.approx(neg, rel=1e-12)


@pytest.mark.parametrize(
    "rho,expected_bits",
    [
        # Closed-form: 0.5 * log2(1 / (1 - rho^2)).
        (0.5, 0.5 * math.log2(1.0 / 0.75)),
        (0.7, 0.5 * math.log2(1.0 / 0.51)),
        (0.85, 0.5 * math.log2(1.0 / 0.2775)),
        (0.95, 0.5 * math.log2(1.0 / 0.0975)),
    ],
)
def test_savings_matches_closed_form(rho: float, expected_bits: float) -> None:
    assert gauss_markov_bit_savings_per_symbol(rho) == pytest.approx(
        expected_bits, rel=1e-12
    )


def test_savings_strictly_increasing_in_abs_rho() -> None:
    rhos = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
    savings = [gauss_markov_bit_savings_per_symbol(r) for r in rhos]
    # The rho list is monotonically increasing in abs value, so successive
    # savings must be strictly increasing. Use `zip` over consecutive pairs
    # without strict= since `savings[1:]` is intentionally one shorter.
    for a, b in zip(savings, savings[1:]):
        assert b > a


def test_council_pose_savings_match_memo() -> None:
    """Shannon's memo: pose ρ ≈ 0.85 → savings ~0.92 bits/pose.

    Closed form: ``0.5 · log2(1/(1-0.85²)) = 0.5 · log2(1/0.2775) =
    0.5 · log2(3.6036) = 0.9247...``. The council memo cited 0.9214 which
    is a minor arithmetic typo in the memo; the implementation matches the
    closed form. The memo's qualitative claim "~0.92 bits/pose" stands.
    """
    s = gauss_markov_bit_savings_per_symbol(0.85)
    assert s == pytest.approx(0.9247, abs=5e-4)


def test_council_seg_savings_match_memo() -> None:
    """Shannon's memo: seg ρ ≈ 0.72 → savings ~0.53 bits/mask-pixel.

    Closed form: ``0.5 · log2(1/(1-0.72²)) = 0.5 · log2(1/0.4816) =
    0.5 · log2(2.0764) = 0.5266``. Memo's 0.53 rounded; implementation
    matches the closed form to 4 decimals.
    """
    s = gauss_markov_bit_savings_per_symbol(0.72)
    assert s == pytest.approx(0.5266, abs=5e-4)


@pytest.mark.parametrize("bad_rho", [1.0, -1.0, 1.5, -1.5, float("inf"), float("nan")])
def test_savings_rejects_out_of_range_rho(bad_rho: float) -> None:
    with pytest.raises(ValueError, match="rho"):
        gauss_markov_bit_savings_per_symbol(bad_rho)


def test_savings_rejects_bool() -> None:
    with pytest.raises(ValueError, match="rho"):
        gauss_markov_bit_savings_per_symbol(True)  # type: ignore[arg-type]


def test_savings_accepts_numeric_string() -> None:
    """Numeric strings are coerced via ``float()`` like other math fns."""
    assert gauss_markov_bit_savings_per_symbol("0.5") == pytest.approx(  # type: ignore[arg-type]
        gauss_markov_bit_savings_per_symbol(0.5), rel=1e-12
    )


def test_savings_rejects_non_numeric_string() -> None:
    with pytest.raises(ValueError, match="rho"):
        gauss_markov_bit_savings_per_symbol("not_a_number")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# JointSourceStream — invariants and field correctness
# ---------------------------------------------------------------------------


def _build_pose_stream(rho: float = 0.85) -> JointSourceStream:
    return JointSourceStream(
        name="pose",
        n_symbols_per_pair=6,
        n_pairs=600,
        bits_per_symbol_iid=8.0,
        rho=rho,
    )


def test_stream_total_symbols() -> None:
    s = _build_pose_stream()
    assert s.total_symbols == 6 * 600


def test_stream_bits_per_symbol_joint_decreases() -> None:
    s = _build_pose_stream(rho=0.85)
    assert s.bits_per_symbol_joint < s.bits_per_symbol_iid


def test_stream_bits_per_symbol_joint_clipped_at_zero() -> None:
    """If the per-symbol savings exceed the i.i.d. cost, joint floor is 0."""
    s = JointSourceStream(
        name="overdetermined",
        n_symbols_per_pair=1,
        n_pairs=10,
        bits_per_symbol_iid=0.1,
        rho=0.99,  # savings ≈ 2.84 bits/symbol > 0.1 i.i.d.
    )
    assert s.bits_per_symbol_joint == 0.0
    assert s.bytes_joint_floor == 0


def test_stream_bytes_iid_floor_matches_ceil() -> None:
    s = _build_pose_stream()
    expected = math.ceil(s.total_symbols * s.bits_per_symbol_iid / 8.0)
    assert s.bytes_iid_floor == expected


def test_stream_bytes_savings_non_negative() -> None:
    for rho in (-0.9, -0.5, 0.0, 0.5, 0.9):
        s = _build_pose_stream(rho=rho)
        assert s.bytes_savings >= 0


def test_stream_n_pairs_one_has_zero_savings() -> None:
    """Berger's bound is asymptotic; finite-block correction zeros out
    savings at n_pairs=1 (no temporal context to exploit)."""
    s = JointSourceStream(
        name="single",
        n_symbols_per_pair=10,
        n_pairs=1,
        bits_per_symbol_iid=2.0,
        rho=0.85,
    )
    assert s.bits_savings_per_symbol == 0.0
    assert s.bytes_iid_floor == s.bytes_joint_floor
    assert s.bytes_savings == 0


def test_stream_finite_block_correction_approaches_asymptote() -> None:
    """As n_pairs grows, finite-block correction → 1; per-symbol savings
    → asymptotic Berger value."""
    asymptotic = gauss_markov_bit_savings_per_symbol(0.85)
    for n in (10, 100, 600, 10_000):
        s = JointSourceStream(
            name=f"len_{n}",
            n_symbols_per_pair=1,
            n_pairs=n,
            bits_per_symbol_iid=8.0,
            rho=0.85,
        )
        ratio = s.bits_savings_per_symbol / asymptotic
        # (n-1)/n correction
        assert ratio == pytest.approx((n - 1) / n, rel=1e-12)
        # And approaches 1.0 monotonically.
        assert 0.0 < ratio <= 1.0


def test_stream_rejects_non_positive_symbols() -> None:
    with pytest.raises(ValueError, match="n_symbols_per_pair"):
        JointSourceStream(
            name="bad",
            n_symbols_per_pair=0,
            n_pairs=1,
            bits_per_symbol_iid=1.0,
            rho=0.0,
        )


def test_stream_rejects_non_positive_pairs() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        JointSourceStream(
            name="bad",
            n_symbols_per_pair=1,
            n_pairs=0,
            bits_per_symbol_iid=1.0,
            rho=0.0,
        )


def test_stream_rejects_negative_bits() -> None:
    with pytest.raises(ValueError, match="bits_per_symbol_iid"):
        JointSourceStream(
            name="bad",
            n_symbols_per_pair=1,
            n_pairs=1,
            bits_per_symbol_iid=-1.0,
            rho=0.0,
        )


def test_stream_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        JointSourceStream(
            name="",
            n_symbols_per_pair=1,
            n_pairs=1,
            bits_per_symbol_iid=1.0,
            rho=0.0,
        )


# ---------------------------------------------------------------------------
# compute_joint_source_floor — aggregation and score projection
# ---------------------------------------------------------------------------


def test_aggregate_pose_only() -> None:
    pose = _build_pose_stream(rho=0.85)
    r = compute_joint_source_floor([pose])
    assert r.total_symbols == pose.total_symbols
    assert r.total_bytes_iid_floor == pose.bytes_iid_floor
    assert r.total_bytes_joint_floor == pose.bytes_joint_floor
    assert r.total_bytes_savings == pose.bytes_savings


def test_aggregate_pose_and_mask() -> None:
    pose = _build_pose_stream(rho=0.85)
    mask = JointSourceStream(
        name="mask",
        n_symbols_per_pair=384 * 512,
        n_pairs=600,
        bits_per_symbol_iid=2.32,  # ~ log2(5)
        rho=0.72,
    )
    r = compute_joint_source_floor([pose, mask])
    assert r.total_symbols == pose.total_symbols + mask.total_symbols
    assert r.total_bytes_iid_floor == pose.bytes_iid_floor + mask.bytes_iid_floor
    assert (
        r.total_bytes_joint_floor
        == pose.bytes_joint_floor + mask.bytes_joint_floor
    )
    # Savings should be massive on the mask stream (huge symbol count).
    assert r.total_bytes_savings >= mask.bytes_savings


def test_aggregate_overhead_added_to_both_floors() -> None:
    pose = _build_pose_stream(rho=0.5)
    r_no_overhead = compute_joint_source_floor([pose], archive_overhead_bytes=0)
    r_overhead = compute_joint_source_floor([pose], archive_overhead_bytes=1024)
    assert (
        r_overhead.total_bytes_iid_floor
        == r_no_overhead.total_bytes_iid_floor + 1024
    )
    assert (
        r_overhead.total_bytes_joint_floor
        == r_no_overhead.total_bytes_joint_floor + 1024
    )
    # Savings unchanged because overhead is identical on both sides.
    assert r_overhead.total_bytes_savings == r_no_overhead.total_bytes_savings


def test_aggregate_zero_rho_savings_is_zero() -> None:
    pose = _build_pose_stream(rho=0.0)
    r = compute_joint_source_floor([pose])
    assert r.total_bytes_savings == 0
    assert r.total_bytes_iid_floor == r.total_bytes_joint_floor


def test_aggregate_score_projection_uses_rate_coefficient() -> None:
    pose = _build_pose_stream(rho=0.85)
    r = compute_joint_source_floor([pose])
    expected_iid = (
        RATE_COEFFICIENT * r.total_bytes_iid_floor / CONTEST_REFERENCE_BYTES
    )
    expected_joint = (
        RATE_COEFFICIENT * r.total_bytes_joint_floor / CONTEST_REFERENCE_BYTES
    )
    assert r.score_at_iid_floor_zero_distortion == pytest.approx(
        expected_iid, rel=1e-12
    )
    assert r.score_at_joint_floor_zero_distortion == pytest.approx(
        expected_joint, rel=1e-12
    )
    assert r.score_savings_zero_distortion == pytest.approx(
        expected_iid - expected_joint, rel=1e-12
    )


def test_aggregate_rejects_empty_streams() -> None:
    with pytest.raises(ValueError, match="at least one"):
        compute_joint_source_floor([])


def test_aggregate_rejects_negative_overhead() -> None:
    pose = _build_pose_stream()
    with pytest.raises(ValueError, match="archive_overhead_bytes"):
        compute_joint_source_floor([pose], archive_overhead_bytes=-1)


def test_aggregate_rejects_non_positive_reference_bytes() -> None:
    pose = _build_pose_stream()
    with pytest.raises(ValueError, match="reference_bytes"):
        compute_joint_source_floor([pose], reference_bytes=0)


def test_aggregate_rejects_non_stream_input() -> None:
    with pytest.raises(TypeError, match="JointSourceStream"):
        compute_joint_source_floor(["not_a_stream"])  # type: ignore[list-item]


def test_aggregate_notes_include_predicted_tag() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    joined = " | ".join(r.notes)
    assert "[predicted; joint-source theoretical floor" in joined


def test_aggregate_to_dict_roundtrip_keys() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    d = r.to_dict()
    assert {
        "streams",
        "total_symbols",
        "total_bytes_iid_floor",
        "total_bytes_joint_floor",
        "total_bytes_savings",
        "archive_overhead_bytes",
        "score_at_iid_floor_zero_distortion",
        "score_at_joint_floor_zero_distortion",
        "score_savings_zero_distortion",
        "notes",
    } <= set(d)


# ---------------------------------------------------------------------------
# joint_floor_from_shannon_report — bridge to existing module
# ---------------------------------------------------------------------------


def _build_shannon_report() -> ShannonFloorReport:
    components = [
        ShannonFloorComponent(
            name="pose",
            n_elements=6 * 600,
            n_quant=256,
            bits_per_element_uniform=8.0,
            bits_per_element_empirical=None,
            bytes_uniform_floor=math.ceil(6 * 600 * 8 / 8.0),
            bytes_empirical_floor=None,
        ),
        ShannonFloorComponent(
            name="mask",
            n_elements=384 * 512 * 600,
            n_quant=5,
            bits_per_element_uniform=math.log2(5),
            bits_per_element_empirical=None,
            bytes_uniform_floor=math.ceil(384 * 512 * 600 * math.log2(5) / 8.0),
            bytes_empirical_floor=None,
        ),
    ]
    total = sum(c.n_elements for c in components)
    return ShannonFloorReport(
        schema_label="test",
        total_elements=total,
        n_quant=256,
        components=components,
        total_bytes_uniform_floor=sum(c.bytes_uniform_floor for c in components),
        total_bytes_empirical_floor=None,
        score_at_uniform_floor_zero_distortion=0.0,
        score_at_empirical_floor_zero_distortion=None,
        notes=[],
    )


def test_bridge_uniform_rho() -> None:
    report = _build_shannon_report()
    r = joint_floor_from_shannon_report(report, rho_per_component=0.5, n_pairs=600)
    assert r.total_bytes_joint_floor < r.total_bytes_iid_floor
    # Each stream uses the same rho.
    assert {s.rho for s in r.streams} == {0.5}


def test_bridge_dict_rho_with_default_zero() -> None:
    report = _build_shannon_report()
    r = joint_floor_from_shannon_report(
        report,
        rho_per_component={"pose": 0.85},  # mask defaults to 0
        n_pairs=600,
    )
    rhos = {s.name: s.rho for s in r.streams}
    assert rhos == {"pose": 0.85, "mask": 0.0}


def test_bridge_rejects_unknown_component() -> None:
    report = _build_shannon_report()
    with pytest.raises(ValueError, match="not in report"):
        joint_floor_from_shannon_report(
            report,
            rho_per_component={"nonexistent": 0.5},
            n_pairs=600,
        )


def test_bridge_rejects_zero_n_pairs() -> None:
    report = _build_shannon_report()
    with pytest.raises(ValueError, match="n_pairs"):
        joint_floor_from_shannon_report(report, rho_per_component=0.5, n_pairs=0)


def test_bridge_uses_empirical_bits_when_present() -> None:
    components = [
        ShannonFloorComponent(
            name="custom",
            n_elements=600,
            n_quant=256,
            bits_per_element_uniform=8.0,
            bits_per_element_empirical=2.0,  # MUCH lower than uniform
            bytes_uniform_floor=600,
            bytes_empirical_floor=150,
        ),
    ]
    report = ShannonFloorReport(
        schema_label="custom",
        total_elements=600,
        n_quant=256,
        components=components,
        total_bytes_uniform_floor=600,
        total_bytes_empirical_floor=150,
        score_at_uniform_floor_zero_distortion=0.0,
        score_at_empirical_floor_zero_distortion=0.0,
        notes=[],
    )
    r = joint_floor_from_shannon_report(report, rho_per_component=0.0, n_pairs=600)
    # rho=0 → no joint savings; floor must equal empirical i.i.d.
    assert r.streams[0].bits_per_symbol_iid == 2.0


def test_bridge_handles_small_component_with_n_elements_below_n_pairs() -> None:
    components = [
        ShannonFloorComponent(
            name="tiny",
            n_elements=10,
            n_quant=256,
            bits_per_element_uniform=8.0,
            bits_per_element_empirical=None,
            bytes_uniform_floor=10,
            bytes_empirical_floor=None,
        ),
    ]
    report = ShannonFloorReport(
        schema_label="tiny",
        total_elements=10,
        n_quant=256,
        components=components,
        total_bytes_uniform_floor=10,
        total_bytes_empirical_floor=None,
        score_at_uniform_floor_zero_distortion=0.0,
        score_at_empirical_floor_zero_distortion=None,
        notes=[],
    )
    r = joint_floor_from_shannon_report(report, rho_per_component=0.5, n_pairs=600)
    s = r.streams[0]
    # n_elements < n_pairs → fold all into one pair.
    assert s.n_pairs == 1
    assert s.n_symbols_per_pair == 10


# ---------------------------------------------------------------------------
# predicted_score_at_joint_floor — score projection
# ---------------------------------------------------------------------------


def test_predicted_score_matches_contest_formula_at_zero_distortion() -> None:
    pose = _build_pose_stream(rho=0.85)
    r = compute_joint_source_floor([pose])
    s = predicted_score_at_joint_floor(r, d_seg=0.0, d_pose=0.0)
    expected = (
        RATE_COEFFICIENT * r.total_bytes_joint_floor / CONTEST_REFERENCE_BYTES
    )
    assert s == pytest.approx(expected, rel=1e-12)


def test_predicted_score_matches_contest_score_at_full_distortion() -> None:
    pose = _build_pose_stream(rho=0.85)
    r = compute_joint_source_floor([pose])
    d_seg = 5.6e-4
    d_pose = 3.4e-5
    predicted = predicted_score_at_joint_floor(r, d_seg=d_seg, d_pose=d_pose)
    expected = contest_score(d_seg, d_pose, r.total_bytes_joint_floor)
    assert predicted == pytest.approx(expected, rel=1e-12)


def test_predicted_score_strictly_below_iid_floor_at_zero_distortion() -> None:
    pose = _build_pose_stream(rho=0.85)
    r = compute_joint_source_floor([pose])
    score_joint = predicted_score_at_joint_floor(r, d_seg=0.0, d_pose=0.0)
    score_iid = (
        SEG_COEFFICIENT * 0.0
        + math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * 0.0)
        + RATE_COEFFICIENT * r.total_bytes_iid_floor / CONTEST_REFERENCE_BYTES
    )
    assert score_joint < score_iid


def test_predicted_score_rejects_negative_d_seg() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    with pytest.raises(ValueError, match="d_seg"):
        predicted_score_at_joint_floor(r, d_seg=-1.0, d_pose=0.0)


def test_predicted_score_rejects_negative_d_pose() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    with pytest.raises(ValueError, match="d_pose"):
        predicted_score_at_joint_floor(r, d_seg=0.0, d_pose=-1.0)


def test_predicted_score_rejects_nan_d_seg() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    with pytest.raises(ValueError, match="NaN"):
        predicted_score_at_joint_floor(r, d_seg=float("nan"), d_pose=0.0)


def test_predicted_score_rejects_nan_d_pose() -> None:
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    with pytest.raises(ValueError, match="NaN"):
        predicted_score_at_joint_floor(r, d_seg=0.0, d_pose=float("nan"))


def test_predicted_score_inf_propagates_to_inf() -> None:
    """Inf distortion is mathematically valid (unbounded distortion);
    return inf rather than raising."""
    pose = _build_pose_stream()
    r = compute_joint_source_floor([pose])
    assert math.isinf(predicted_score_at_joint_floor(
        r, d_seg=float("inf"), d_pose=0.0
    ))
    assert math.isinf(predicted_score_at_joint_floor(
        r, d_seg=0.0, d_pose=float("inf")
    ))


# ---------------------------------------------------------------------------
# Council-memo numeric anchor: comma-class video, full stream stack
# ---------------------------------------------------------------------------


def test_council_full_comma_stream_stack_matches_predicted_band() -> None:
    """End-to-end smoke matching the council §6 numeric prediction.

    Shannon's memo: ~5-15 KB additional byte savings beyond i.i.d. R(D) on
    the comma video, just from exploiting frame-to-frame correlation. This
    test fits the LOWER end of that band with conservative ρ values; the
    upper end requires more aggressive correlation estimates that the
    council deferred to empirical measurement.

    Stream taxonomy (matching comma's contest archive):
      - Pose: 6-DOF transmitted PER PAIR, 600 pairs.
      - Per-pair latent (HNeRV/Quantizr-style): a SMALL latent vector per
        pair (~64 dims at 4 bits = 32 bytes/pair raw) — this IS
        per-pair-correlated.
      - Mask: 600 mask frames at 384×512×log2(5) bits — per-pair correlated.
      - Renderer weights are SHIPPED ONCE (not in this joint-source bound;
        captured by the existing ShannonFloorReport for one-shot streams).
    """
    streams = [
        # Pose: 6-DOF × 600 pairs at 8 bits/symbol uniform, ρ=0.85.
        JointSourceStream(
            name="pose",
            n_symbols_per_pair=6,
            n_pairs=600,
            bits_per_symbol_iid=8.0,
            rho=0.85,
        ),
        # Per-pair latent: 64 elements × 600 pairs at 4 bits, ρ=0.5.
        JointSourceStream(
            name="per_pair_latent",
            n_symbols_per_pair=64,
            n_pairs=600,
            bits_per_symbol_iid=4.0,
            rho=0.5,
        ),
        # Mask: ~196,608 px × 600 pairs at 2.32 bits ≈ log2(5), ρ=0.72.
        JointSourceStream(
            name="mask",
            n_symbols_per_pair=384 * 512,
            n_pairs=600,
            bits_per_symbol_iid=math.log2(5),
            rho=0.72,
        ),
    ]
    r = compute_joint_source_floor(streams)
    # Lower-band check: at LEAST 5 KB of savings from joint-source coding
    # alone, conservative ρ values, no entropy coding cleverness. The mask
    # stream dominates (largest symbol count) — joint savings should easily
    # clear the 5 KB threshold.
    assert r.total_bytes_savings >= 5_000
    # And the score projection must be strictly below the i.i.d. equivalent.
    assert (
        r.score_at_joint_floor_zero_distortion
        < r.score_at_iid_floor_zero_distortion
    )


def test_council_pose_only_savings_around_415_bytes_at_rho_0_85() -> None:
    """Per-stream sanity: pose alone with ρ=0.85 gives ~415 bytes saved.

    Closed form (with finite-block correction):
      600 pairs × 6 symbols × 0.9247 bits/symbol × (599/600) ≈ 3323 bits
      → ceil(3323 / 8) = 416 bytes saved (asymptotic 0.9247 → 415-416 B
      after finite-block + per-stream ceil rounding).
    """
    pose = _build_pose_stream(rho=0.85)
    finite_block_factor = (pose.n_pairs - 1) / pose.n_pairs
    expected_bits_saved = (
        600 * 6 * gauss_markov_bit_savings_per_symbol(0.85) * finite_block_factor
    )
    expected_bytes_saved = math.ceil(expected_bits_saved / 8.0)
    # Allow ±2 byte slack due to per-stream ceil at 8-bit symbol boundary
    # vs the closed-form continuous bit count.
    assert pose.bytes_savings == pytest.approx(expected_bytes_saved, abs=2)


def test_joint_floor_per_pair_byte_savings_anchor_against_a1() -> None:
    """Anchor: with comma-class per-pair stream sizes, joint-source R(D)
    saves at least 600 B vs i.i.d. on the per-pair streams alone.

    A1: bytes=178,262 (CPU 0.192847577437, CUDA 0.226352023478). The
    per-pair streams in A1 are pose (6-DOF × 600 = 3.6 KB at 8 bits/symbol
    i.i.d.) plus a per-pair latent (~64 dims × 600 = 9.6 KB at 4 bits/symbol
    i.i.d.). Joint-source R(D) at ρ_pose=0.85 + ρ_latent=0.5 must reduce
    these floors materially.
    """
    streams = [
        JointSourceStream(
            name="pose",
            n_symbols_per_pair=6,
            n_pairs=600,
            bits_per_symbol_iid=8.0,
            rho=0.85,
        ),
        JointSourceStream(
            name="per_pair_latent",
            n_symbols_per_pair=64,
            n_pairs=600,
            bits_per_symbol_iid=4.0,
            rho=0.5,
        ),
    ]
    r = compute_joint_source_floor(streams)
    # Closed-form expected savings:
    # pose: 600*6*0.5*log2(1/(1-0.85^2)) bits ≈ 3329 bits ≈ 417 B
    # latent: 600*64*0.5*log2(1/(1-0.5^2)) bits ≈ 7918 bits ≈ 990 B
    # Total ≈ 1407 B; allow conservative lower bound of 600 B.
    assert r.total_bytes_savings >= 600
    # Sanity: i.i.d. floor matches the closed-form per-pair sum.
    expected_iid = math.ceil(600 * 6 * 8.0 / 8.0) + math.ceil(
        600 * 64 * 4.0 / 8.0
    )
    assert r.total_bytes_iid_floor == expected_iid


def test_joint_floor_at_a1_overhead_predicts_below_iid_score() -> None:
    """At A1's archive byte budget (overhead model), joint floor scores
    strictly below the i.i.d. floor at the same distortion.

    The contest-CPU axis distortion for A1 is approximately:
      d_seg ≈ 5.6e-4
      d_pose ≈ 3.45e-5  (CUDA value 1.74e-4 / R_pose=5.04 calibration)

    These are predictions derived from the public per-PR comments + the
    R_pose=5.04 HNeRV-cluster calibration (memory:
    feedback_cuda_cpu_axis_profile_learning_layer_20260508). Tagged
    [predicted; advisory only].
    """
    streams = [
        JointSourceStream(
            name="pose",
            n_symbols_per_pair=6,
            n_pairs=600,
            bits_per_symbol_iid=8.0,
            rho=0.85,
        ),
        JointSourceStream(
            name="per_pair_latent",
            n_symbols_per_pair=64,
            n_pairs=600,
            bits_per_symbol_iid=4.0,
            rho=0.5,
        ),
    ]
    # Overhead = 168,000 ≈ A1 archive (178,262) - per-pair i.i.d. floor.
    r = compute_joint_source_floor(streams, archive_overhead_bytes=168_000)
    # Distortions on the contest-CPU axis (R_pose-calibrated from CUDA).
    d_seg = 5.6e-4
    d_pose_cpu = 1.74e-4 / 5.04
    s_joint = predicted_score_at_joint_floor(r, d_seg=d_seg, d_pose=d_pose_cpu)
    # Equivalent score at i.i.d. floor:
    s_iid = (
        SEG_COEFFICIENT * d_seg
        + math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose_cpu)
        + RATE_COEFFICIENT * r.total_bytes_iid_floor / CONTEST_REFERENCE_BYTES
    )
    # The joint floor must be strictly LOWER than the i.i.d. floor at the
    # same distortion (this is the entire mechanism of the eureka).
    assert s_joint < s_iid
    # And the gap on the rate term alone must equal the per-pair savings
    # rate-term contribution.
    rate_gap = (
        RATE_COEFFICIENT
        * r.total_bytes_savings
        / CONTEST_REFERENCE_BYTES
    )
    assert (s_iid - s_joint) == pytest.approx(rate_gap, rel=1e-12)


# ---------------------------------------------------------------------------
# T13 — Fridrich √n per-pair undetectable embedding budget
# ---------------------------------------------------------------------------


def test_sqrt_n_basic_closed_form() -> None:
    # 600 pairs × 64 symbols/pair = 38400 total → √38400 ≈ 195.96.
    # Per-pair: √64 = 8.0.
    r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    assert r.n_total_symbols == 38400
    assert r.n_pairs == 600
    assert r.n_symbols_per_pair == 64
    assert r.alpha == 1.0
    assert r.undetectable_bits_per_pair == pytest.approx(8.0, rel=1e-9)
    assert r.undetectable_bits_total == pytest.approx(math.sqrt(38400), rel=1e-9)


def test_sqrt_n_alpha_scales_linearly() -> None:
    base = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    doubled = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64, alpha=2.0)
    assert doubled.undetectable_bits_per_pair == pytest.approx(
        2.0 * base.undetectable_bits_per_pair, rel=1e-12
    )
    assert doubled.undetectable_bits_total == pytest.approx(
        2.0 * base.undetectable_bits_total, rel=1e-12
    )


def test_sqrt_n_a1_substrate_28d_per_pair() -> None:
    # A1's per-pair latent stream — 28-D × 600 pairs.
    r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=28)
    # Per-pair budget √28 ≈ 5.29; total √16800 ≈ 129.6.
    assert r.undetectable_bits_per_pair == pytest.approx(math.sqrt(28), rel=1e-9)
    assert r.undetectable_bits_total == pytest.approx(math.sqrt(16800), rel=1e-9)


def test_sqrt_n_rejects_zero_pairs() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        per_pair_sqrt_n_budget(n_pairs=0, n_symbols_per_pair=64)


def test_sqrt_n_rejects_negative_pairs() -> None:
    with pytest.raises(ValueError, match="n_pairs"):
        per_pair_sqrt_n_budget(n_pairs=-1, n_symbols_per_pair=64)


def test_sqrt_n_rejects_zero_symbols() -> None:
    with pytest.raises(ValueError, match="n_symbols_per_pair"):
        per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=0)


def test_sqrt_n_rejects_non_int_pairs() -> None:
    # Floats not accepted to keep symbol counting unambiguous.
    with pytest.raises(ValueError, match="n_pairs"):
        per_pair_sqrt_n_budget(n_pairs=600.5, n_symbols_per_pair=64)  # type: ignore[arg-type]


def test_sqrt_n_rejects_bool_pairs() -> None:
    # bool subclasses int in Python; explicitly rejected to prevent True/False
    # being silently coerced to 1/0.
    with pytest.raises(ValueError, match="n_pairs"):
        per_pair_sqrt_n_budget(n_pairs=True, n_symbols_per_pair=64)  # type: ignore[arg-type]


def test_sqrt_n_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        per_pair_sqrt_n_budget(
            n_pairs=600, n_symbols_per_pair=64, alpha=float("nan")
        )
    with pytest.raises(ValueError, match="alpha"):
        per_pair_sqrt_n_budget(
            n_pairs=600, n_symbols_per_pair=64, alpha=float("inf")
        )
    with pytest.raises(ValueError, match="alpha"):
        per_pair_sqrt_n_budget(
            n_pairs=600, n_symbols_per_pair=64, alpha=-1.0
        )


def test_sqrt_n_to_dict_round_trips_fields() -> None:
    r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64, alpha=1.5)
    d = r.to_dict()
    assert d["n_total_symbols"] == 38400
    assert d["n_pairs"] == 600
    assert d["n_symbols_per_pair"] == 64
    assert d["alpha"] == 1.5
    assert d["undetectable_bits_per_pair"] == pytest.approx(1.5 * 8.0, rel=1e-9)
    # Notes propagate the [predicted; T13 ...] tag.
    assert any("T13" in s for s in d["notes"])
    assert any("Fridrich" in s for s in d["notes"])


def test_sqrt_n_per_pair_lt_total_per_pair_average() -> None:
    """Sanity: the per-pair fair-share is the per-pair sqrt; the total is
    sqrt of the joint length. With 600 independent pairs each of size n_p,
    total_bits / n_pairs < bits_per_pair (independent embeddings concentrate
    less than per-pair embeddings — Fridrich's law on joint vs marginal)."""
    r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    avg_total_per_pair = r.undetectable_bits_total / r.n_pairs
    # √(600·64)/600 = √64/√600 ≈ 8/24.5 ≈ 0.327; per-pair = 8.
    assert avg_total_per_pair < r.undetectable_bits_per_pair


def test_sqrt_n_dataclass_is_frozen() -> None:
    r = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    with pytest.raises(Exception):  # FrozenInstanceError
        r.alpha = 2.0  # type: ignore[misc]


def test_sqrt_n_a1_substrate_realloc_envelope() -> None:
    """A1's per-pair latent is currently ~3 bits/pair (per Fridrich council
    note). Bound says we could safely raise to ~√28 ≈ 5.3 bits/pair (28-D
    A1 substrate) or ~√64 ≈ 8 bits/pair (HNeRV 64-D). Verify the
    reallocation envelope matches the council-memo prediction."""
    r28 = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=28)
    r64 = per_pair_sqrt_n_budget(n_pairs=600, n_symbols_per_pair=64)
    assert 5.0 < r28.undetectable_bits_per_pair < 5.5
    assert 7.5 < r64.undetectable_bits_per_pair < 8.5
    # Headroom over A1's current 3 bits/pair.
    a1_current = 3.0
    assert r28.undetectable_bits_per_pair - a1_current > 2.0
    assert r64.undetectable_bits_per_pair - a1_current > 4.5
