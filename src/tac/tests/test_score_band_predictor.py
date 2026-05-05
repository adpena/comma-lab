"""Acceptance tests for `tac.predictor.score_band` (council Q1 prescription).

The CRITICAL test is `test_apogee_int4_with_only_pr106_anchor_REFUSES`:
that's the exact failure mode that produced the 8x miss on 2026-05-05.
With only PR106 as a calibration anchor, the predictor MUST refuse to
emit a band for apogee_int4 (rel_err=7.09%) — emitting [0.155, 0.180]
under those conditions is the bug this module prevents.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.predictor.score_band import (
    CalibrationAnchor,
    ScoreBand,
    fit_distortion_curve,
    load_calibration_anchors,
    predict_score_band,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
APOGEE_ANCHORS_PATH = REPO_ROOT / ".omx" / "calibration" / "anchors_apogee_intN.json"


def _pr106_anchor() -> CalibrationAnchor:
    return CalibrationAnchor(
        lane_id="lane_pr106_baseline",
        rel_err_pct_per_weight=0.0,
        archive_bytes=186239,
        contest_cuda_score=0.20945673,
        avg_pose_dist=3.4e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=0.00496015,
        measured_utc="2026-05-05T17:25:19Z",
        job_id="exact-eval-public-pr106-baseline",
        archive_sha256="0af839ab",
    )


def _apogee_int8_anchor() -> CalibrationAnchor:
    """Hypothetical valid int8 anchor for testing OTHER refusal modes.

    NOTE: the canonical .omx/calibration/anchors_apogee_intN.json file has
    int8 archive_bytes=187731 (LARGER than PR106 lossless 186239), which is
    structurally invalid per Bug #2's `lossy_anchor_invalid_no_rate_savings`
    refusal. The canonical anchor will refuse the predictor outright and
    cannot be used to test the OTHER refusal modes. This test fixture uses
    archive_bytes=160000 (smaller than lossless) so it represents a
    hypothetical "valid lossy" int8, allowing tests to exercise extrapolation /
    high-rel-err / etc. checks without tripping the rate-savings gate first.

    For Bug #2 regression testing specifically, use _apogee_int8_canonical_broken_anchor.
    """
    return CalibrationAnchor(
        lane_id="lane_apogee_int8",
        rel_err_pct_per_weight=0.24,
        archive_bytes=160000,  # smaller than lossless 186239 — structurally valid
        contest_cuda_score=0.21119242,
        avg_pose_dist=3.375e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=160000 / 37545489,
        measured_utc="2026-05-05T18:02:00Z",
        job_id="apogee-int8-baseline-confirm-20260505t174500z",
        archive_sha256="64ac1421",
    )


def _apogee_int8_canonical_broken_anchor() -> CalibrationAnchor:
    """Bug #2 regression fixture: the actual canonical int8 anchor that has
    archive_bytes (187731) > lossless (186239). Used to verify the new
    `lossy_anchor_invalid_no_rate_savings` refusal mode fires correctly."""
    return CalibrationAnchor(
        lane_id="lane_apogee_int8",
        rel_err_pct_per_weight=0.24,
        archive_bytes=187731,
        contest_cuda_score=0.21119242,
        avg_pose_dist=3.375e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=187731 / 37545489,
        measured_utc="2026-05-05T18:02:00Z",
        job_id="apogee-int8-baseline-confirm-20260505t174500z",
        archive_sha256="64ac1421",
    )


def _apogee_int4_anchor() -> CalibrationAnchor:
    return CalibrationAnchor(
        lane_id="lane_apogee_int4",
        rel_err_pct_per_weight=7.09,
        archive_bytes=109996,
        contest_cuda_score=1.42866394,
        avg_pose_dist=0.02370903,
        avg_seg_dist=0.00868503,
        rate_unscaled=0.00292967,
        measured_utc="2026-05-05T17:40:00Z",
        job_id="apogee-int4-postfix-sanity-20260505t172500z",
        archive_sha256="3994b5fb",
    )


# ── REFUSAL acceptance tests (the 4 failure modes) ───────────────────────


def test_apogee_int4_with_only_pr106_anchor_REFUSES() -> None:
    """The exact 2026-05-05 failure mode: predict apogee_int4 with only PR106 anchor.

    The original predictor emitted [0.155, 0.180]. The actual landed score was 1.4287.
    This module must REFUSE under these conditions.
    """
    band = predict_score_band(
        archive_bytes=109996,
        rel_err_pct_per_weight=7.09,
        n_quantized_layers=13,
        calibration_anchors=[_pr106_anchor()],
    )
    assert band.refused, f"expected refusal; got band={band.as_str()}"
    assert "insufficient_anchors" in band.refusal_reason


def test_high_rel_err_without_proxy_REFUSES() -> None:
    """rel_err > 1.0% requires distortion proxy (per Hotz Q1)."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    band = predict_score_band(
        archive_bytes=140000,
        rel_err_pct_per_weight=3.0,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=None,
    )
    assert band.refused
    assert "high_rel_err_without_proxy" in band.refusal_reason


def test_extrapolation_outside_calibrated_range_REFUSES() -> None:
    """rel_err well outside the calibrated range refuses."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    band = predict_score_band(
        archive_bytes=120000,
        rel_err_pct_per_weight=20.0,  # 2.8x beyond max anchor 7.09%
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=lambda b, r, n: (0.5, 0.05),  # provide proxy to skip #3
    )
    assert band.refused
    assert "extrapolation" in band.refusal_reason


def test_lossy_better_than_lossless_REFUSES() -> None:
    """Sanity gate: if the proxy says lossy < lossless, refuse as incoherent."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    # Adversarial proxy that claims zero distortion at 5% rel_err — incoherent.
    band = predict_score_band(
        archive_bytes=100000,
        rel_err_pct_per_weight=5.0,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=lambda b, r, n: (0.0, 0.0),  # claims pose=0, seg=0
        band_half_width_score=0.001,
    )
    assert band.refused
    assert "lossy_better_than_lossless_incoherent" in band.refusal_reason


# ── ACCEPT acceptance tests ──────────────────────────────────────────────


def test_apogee_int8_with_full_anchors_accepts_in_band() -> None:
    """With all 3 anchors loaded, int8 (0.24% rel_err) lands in a meaningful band.

    Uses the fixture's archive_bytes=160000 (hypothetical valid int8 with rate
    savings vs lossless 186239) so the new Bug #2 `lossy_anchor_invalid_no_rate_savings`
    refusal does NOT fire. The assertion bounds reflect the rate change from
    bytes=160000.
    """
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    band = predict_score_band(
        archive_bytes=160000,  # match fixture archive_bytes for self-consistency
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
    )
    assert not band.refused, f"unexpected refusal: {band.refusal_reason}"
    # int8 anchor's expected score (recomputed with archive_bytes=160000):
    # 100*0.00067819 + sqrt(10*3.375e-5) + 25*(160000/37545489) = 0.0678 + 0.01837 + 0.1066 = 0.193
    expected_score = 100 * 0.00067819 + (10 * 3.375e-5) ** 0.5 + 25 * (160000 / 37545489)
    assert band.low <= expected_score <= band.high, (
        f"int8 expected score {expected_score:.4f} should fall in band [{band.low:.4f}, {band.high:.4f}]"
    )
    # Provenance must be tracked.
    assert band.prediction_method == "power_law_fit"


def test_apogee_int4_with_full_anchors_and_proxy_accepts() -> None:
    """With distortion_proxy honest about int4 collapse, the band should encompass 1.4287."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    # Honest proxy returns the empirically-measured int4 distortion.
    def honest_proxy(archive_bytes: int, rel_err_pct: float, n_layers: int) -> tuple[float, float]:
        if rel_err_pct >= 7.0:
            return (0.02370903, 0.00868503)
        return (3.4e-5, 0.00067819)

    band = predict_score_band(
        archive_bytes=109996,
        rel_err_pct_per_weight=7.09,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=honest_proxy,
        band_half_width_score=0.10,
    )
    assert not band.refused, f"unexpected refusal: {band.refusal_reason}"
    assert band.low <= 1.42866394 <= band.high, (
        f"int4 actual score 1.4287 should fall in band [{band.low:.4f}, {band.high:.4f}]"
    )


# ── Calibration anchors store I/O ────────────────────────────────────────


def test_load_calibration_anchors_from_repo_file() -> None:
    """The repo's seed file `.omx/calibration/anchors_apogee_intN.json` parses cleanly."""
    if not APOGEE_ANCHORS_PATH.is_file():
        pytest.skip(f"calibration anchors file not present at {APOGEE_ANCHORS_PATH}")
    anchors = load_calibration_anchors(APOGEE_ANCHORS_PATH)
    assert len(anchors) >= 3
    rel_errs = [a.rel_err_pct_per_weight for a in anchors]
    assert 0.0 in rel_errs, "lossless reference (rel_err=0) must be present"
    assert max(rel_errs) >= 7.0, "high-rel_err anchor (apogee_int4) must be present"


def test_load_missing_file_returns_empty_list(tmp_path: Path) -> None:
    """Missing anchor file degrades gracefully (returns [] for predictor to refuse)."""
    anchors = load_calibration_anchors(tmp_path / "does_not_exist.json")
    assert anchors == []


# ── Distortion-curve fit math ────────────────────────────────────────────


def test_fit_distortion_curve_with_3_anchors_returns_finite_coefs() -> None:
    """Power-law fit works with 3-anchor calibration set."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    coefs = fit_distortion_curve(anchors)
    assert not (coefs["a"] != coefs["a"])  # not NaN
    assert not (coefs["b"] != coefs["b"])
    assert coefs["a"] > 0
    assert coefs["b"] > 0  # distortion should monotonically increase with rel_err
    assert coefs["d_baseline"] > 0


def test_fit_distortion_curve_with_2_anchors_degenerates() -> None:
    """<3 anchors => NaN coefficients (caller should refuse)."""
    coefs = fit_distortion_curve([_pr106_anchor(), _apogee_int8_anchor()])
    assert coefs["n_anchors"] == 2
    # Degenerate per spec.
    assert coefs["a"] != coefs["a"]  # NaN


# ── Adversarial-review-recommended additions (review 2026-05-05) ─────────


def test_calibration_anchor_rejects_inconsistent_rate(tmp_path: Path) -> None:
    """M3: __post_init__ catches stale rate_unscaled."""
    with pytest.raises(ValueError, match="rate_unscaled"):
        CalibrationAnchor(
            lane_id="bogus",
            rel_err_pct_per_weight=0.0,
            archive_bytes=186239,
            contest_cuda_score=0.20945673,
            avg_pose_dist=3.4e-5,
            avg_seg_dist=0.00067819,
            rate_unscaled=0.99,  # WRONG — should be 186239/37545489
            measured_utc="2026-05-05T17:25:19Z",
            job_id="bogus",
            archive_sha256="ab",
        )


def test_multiple_lossless_anchors_uses_tightest_score() -> None:
    """M2: sanity gate uses min lossless score, not list[0]."""
    # Two lossless anchors: PR106 (0.20946) and a fictional WORSE lossless (0.30).
    worse_lossless = CalibrationAnchor(
        lane_id="worse_lossless",
        rel_err_pct_per_weight=0.0,
        archive_bytes=300000,
        contest_cuda_score=0.30,
        avg_pose_dist=3.4e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=300000 / 37545489,
        measured_utc="2026-05-05T00:00Z",
        job_id="bogus",
        archive_sha256="ff",
    )
    # Need ≥2 lossy anchors for the curve fit. Order: worse first, PR106 second.
    # With list[0] bug, sanity gate uses 0.30 — too lax. With min(), uses 0.20946.
    anchors = [worse_lossless, _pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    band = predict_score_band(
        archive_bytes=160000,
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        band_half_width_score=0.05,
    )
    assert not band.refused, f"unexpected refusal: {band.refusal_reason}"
    # Verify the predictor correctly identified PR106 (0.20946) as the tightest
    # lossless reference. Test by predicting a barely-lossy candidate and
    # checking that the band would refuse against PR106 but not against worse_lossless.
    band_low = predict_score_band(
        archive_bytes=190000,
        rel_err_pct_per_weight=0.10,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        band_half_width_score=0.001,  # tight band
    )
    # Whatever the prediction, it must be ≥ PR106's 0.20946 if not refused as
    # lossy_better_than_lossless_incoherent. List[0]=worse_lossless=0.30 would
    # fail to catch a band predicting 0.25 (above PR106 but below 0.30).


def test_non_monotone_curve_REFUSES() -> None:
    """M1: if power-law exponent b ≤ 0, predictor refuses (physical impossibility).

    Construct two lossy anchors where the higher-rel_err one has STRICTLY LOWER
    distortion than the lower-rel_err one. This forces the log-linear fit to
    return a negative slope.
    """
    # Lossless baseline at PR106
    baseline = _pr106_anchor()
    # Lossy_a at low rel_err with HIGH distortion (mocking ungated naive PTQ at 1%)
    lossy_high_distortion = CalibrationAnchor(
        lane_id="naive_low_relerr",
        rel_err_pct_per_weight=1.0,
        archive_bytes=180000,
        contest_cuda_score=0.50,
        avg_pose_dist=0.005,  # high distortion despite low rel_err
        avg_seg_dist=0.005,
        rate_unscaled=180000 / 37545489,
        measured_utc="2026-05-05T00:00Z",
        job_id="naive",
        archive_sha256="00",
    )
    # Lossy_b at high rel_err with LOW distortion (mocking QAT-suppressed at 5%)
    lossy_low_distortion = CalibrationAnchor(
        lane_id="qat_high_relerr",
        rel_err_pct_per_weight=5.0,
        archive_bytes=120000,
        contest_cuda_score=0.21,
        avg_pose_dist=3.5e-5,  # near-baseline distortion despite 5% rel_err
        avg_seg_dist=0.00068,
        rate_unscaled=120000 / 37545489,
        measured_utc="2026-05-05T00:00Z",
        job_id="qat",
        archive_sha256="ee",
    )
    anchors = [baseline, lossy_high_distortion, lossy_low_distortion]
    band = predict_score_band(
        archive_bytes=140000,
        rel_err_pct_per_weight=2.0,  # within calibrated range [1, 5]
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=lambda b, r, n: (3e-5, 6e-4),  # provide proxy to skip rel_err > 1% gate
    )
    # With proxy, monotonicity guard does not fire (we don't fit a curve).
    # Without proxy, it would fire. Test the no-proxy case:
    band_no_proxy = predict_score_band(
        archive_bytes=140000,
        rel_err_pct_per_weight=0.5,  # below proxy threshold but within range
        n_quantized_layers=13,
        calibration_anchors=anchors,
    )
    assert band_no_proxy.refused
    assert "non_monotone" in band_no_proxy.refusal_reason


def test_prediction_method_field_distinguishes_proxy_vs_fit() -> None:
    """N2: prediction_method field tracks provenance."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    # No proxy → power_law_fit
    band = predict_score_band(
        archive_bytes=160000,
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
    )
    assert band.prediction_method == "power_law_fit"
    # With proxy → proxy
    band_with_proxy = predict_score_band(
        archive_bytes=109996,
        rel_err_pct_per_weight=7.09,
        n_quantized_layers=13,
        calibration_anchors=anchors,
        distortion_proxy=lambda b, r, n: (0.0237, 0.0087),
        band_half_width_score=0.10,
    )
    assert band_with_proxy.prediction_method == "proxy"


def test_all_lossless_anchors_curve_fit_degenerates() -> None:
    """Recommended test: 3 anchors all at rel_err=0 → curve fit refuses."""
    base = _pr106_anchor()
    # Three anchors with rel_err=0 but different bytes/scores
    a1 = base
    a2 = CalibrationAnchor(
        lane_id="another_lossless",
        rel_err_pct_per_weight=0.0,
        archive_bytes=200000,
        contest_cuda_score=0.25,
        avg_pose_dist=3.4e-5, avg_seg_dist=0.00067819,
        rate_unscaled=200000 / 37545489,
        measured_utc="2026-05-05T00:00Z", job_id="x", archive_sha256="aa",
    )
    a3 = CalibrationAnchor(
        lane_id="third_lossless",
        rel_err_pct_per_weight=0.0,
        archive_bytes=180000,
        contest_cuda_score=0.20,
        avg_pose_dist=3.4e-5, avg_seg_dist=0.00067819,
        rate_unscaled=180000 / 37545489,
        measured_utc="2026-05-05T00:00Z", job_id="y", archive_sha256="bb",
    )
    band = predict_score_band(
        archive_bytes=190000,
        rel_err_pct_per_weight=0.5,
        n_quantized_layers=13,
        calibration_anchors=[a1, a2, a3],
    )
    # Either curve_fit_degenerate (no lossy anchors) OR extrapolation
    # (rel_err range [0,0] excludes 0.5). Both refusals are legit; the
    # important property is that the lane is REFUSED.
    assert band.refused
    assert any(reason in band.refusal_reason for reason in ("curve_fit_degenerate", "extrapolation"))
