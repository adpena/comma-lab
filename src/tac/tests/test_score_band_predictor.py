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
    return CalibrationAnchor(
        lane_id="lane_apogee_int8",
        rel_err_pct_per_weight=0.24,
        archive_bytes=187731,
        contest_cuda_score=0.21119242,
        avg_pose_dist=3.375e-5,
        avg_seg_dist=0.00067819,
        rate_unscaled=0.00500009,
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
    """With all 3 anchors loaded, int8 (0.24% rel_err) lands in a sane band near 0.2112."""
    anchors = [_pr106_anchor(), _apogee_int8_anchor(), _apogee_int4_anchor()]
    band = predict_score_band(
        archive_bytes=187731,
        rel_err_pct_per_weight=0.24,
        n_quantized_layers=13,
        calibration_anchors=anchors,
    )
    assert not band.refused, f"unexpected refusal: {band.refusal_reason}"
    # int8 anchor itself should hit ≈ 0.2112; band is ±0.05.
    assert band.low <= 0.21119242 <= band.high, (
        f"int8 actual score 0.2112 should fall in band [{band.low:.4f}, {band.high:.4f}]"
    )


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
