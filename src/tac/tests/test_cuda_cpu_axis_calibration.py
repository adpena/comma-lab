"""Tests for the CUDA→CPU axis calibration helper."""
from __future__ import annotations

import math

import pytest

from tac.optimization.cuda_cpu_axis_calibration import (
    CUDA_POSE_PRECISION_FLOOR,
    KNOWN_ARCHITECTURE_CLASSES,
    R_POSE_HNERV,
    R_POSE_HNERV_STD,
    R_SEG_HNERV,
    R_SEG_HNERV_STD,
    SCORE_GAP_HNERV_CONSTANT,
    SCORE_GAP_HNERV_STD,
    CalibrationBand,
    CudaCpuCalibration,
    cpu_pose_floor_score_contribution,
    effective_pose_loss_for_cpu,
    is_at_pose_floor,
    normalize_architecture_class,
    predict_cpu_from_cuda,
    tag_predicted_band,
)
from tac.score_geometry import contest_score

# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------


def test_calibration_constants_match_empirical_anchors() -> None:
    """The pinned constants match the 2026-05-08 HNeRV cluster sweep."""
    assert R_POSE_HNERV == 5.04
    assert R_POSE_HNERV_STD == 0.10
    assert R_SEG_HNERV == 1.17
    assert R_SEG_HNERV_STD == 0.01
    assert SCORE_GAP_HNERV_CONSTANT == 0.033
    assert CUDA_POSE_PRECISION_FLOOR == 1.4e-4


def test_known_architecture_classes_includes_hnerv_and_unknown() -> None:
    assert "hnerv" in KNOWN_ARCHITECTURE_CLASSES
    assert "unknown" in KNOWN_ARCHITECTURE_CLASSES


# ---------------------------------------------------------------------------
# Constant-gap mode
# ---------------------------------------------------------------------------


def test_constant_gap_predict_cpu_from_cuda_subtracts_calibrated_gap() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    band = cal.predict_cpu_from_cuda(0.20)
    # Constant gap mode at hnerv anchor: point estimate = 0.20 - 0.033.
    assert math.isclose(band.score_point, 0.20 - SCORE_GAP_HNERV_CONSTANT, rel_tol=1e-9)
    # Sigma should equal SCORE_GAP_HNERV_STD on hnerv (no inflation).
    assert band.sigma == SCORE_GAP_HNERV_STD
    assert band.calibration_quality == "hnerv-anchored"


def test_constant_gap_inverse_round_trip_recovers_input() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    cuda_in = 0.20
    cpu_pred = cal.predict_cpu_from_cuda(cuda_in).score_point
    cuda_back = cal.predict_cuda_from_cpu(cpu_pred).score_point
    assert math.isclose(cuda_back, cuda_in, rel_tol=1e-9)


def test_unknown_architecture_inflates_sigma() -> None:
    cal_hnerv = CudaCpuCalibration(architecture_class="hnerv")
    cal_unknown = CudaCpuCalibration(architecture_class="unknown")
    band_hnerv = cal_hnerv.predict_cpu_from_cuda(0.20)
    band_unknown = cal_unknown.predict_cpu_from_cuda(0.20)
    assert band_unknown.sigma > band_hnerv.sigma
    assert band_unknown.calibration_quality == "extrapolated"


# ---------------------------------------------------------------------------
# Decomposed mode
# ---------------------------------------------------------------------------


def test_decomposed_mode_uses_per_axis_rebase() -> None:
    """When d_pose_cuda + d_seg_cuda + bytes provided, use per-axis rebase."""
    cal = CudaCpuCalibration(architecture_class="hnerv")
    # PR106 anchor (CUDA): d_pose ≈ 3.4e-5 * 5.04 ≈ 1.71e-4 (above floor)
    cuda_pose = 3.4e-5 * R_POSE_HNERV  # 1.71e-4
    cuda_seg = 6.7e-4 * R_SEG_HNERV    # ~7.84e-4
    archive_bytes = 178_258
    cuda_score = contest_score(d_seg=cuda_seg, d_pose=cuda_pose, archive_bytes=archive_bytes)
    band = cal.predict_cpu_from_cuda(
        cuda_score,
        d_pose_cuda=cuda_pose,
        d_seg_cuda=cuda_seg,
        archive_bytes=archive_bytes,
    )
    # CPU pose uses the measured per-axis ratio. The floor is an advisory
    # trust-region marker, not a subtraction term.
    cpu_pose_expected = cuda_pose / R_POSE_HNERV
    cpu_seg_expected = cuda_seg / R_SEG_HNERV
    expected_cpu_score = contest_score(
        d_seg=cpu_seg_expected, d_pose=cpu_pose_expected, archive_bytes=archive_bytes
    )
    assert math.isclose(band.score_point, expected_cpu_score, rel_tol=1e-9)
    assert "mode=decomposed" in band.notes


def test_decomposed_mode_inverse_round_trip_via_constant_gap_close() -> None:
    """Constant-gap round-trip is exact; decomposed forward+constant-gap reverse
    is the operational pattern (you measure CUDA, predict CPU, do not invert)."""
    cal = CudaCpuCalibration(architecture_class="hnerv")
    cuda_pose = 1.5e-3
    cuda_seg = 5e-3
    archive_bytes = 200_000
    cuda_score = contest_score(d_seg=cuda_seg, d_pose=cuda_pose, archive_bytes=archive_bytes)
    cpu_band = cal.predict_cpu_from_cuda(
        cuda_score,
        d_pose_cuda=cuda_pose,
        d_seg_cuda=cuda_seg,
        archive_bytes=archive_bytes,
    )
    # Constant-gap inverse is not a mathematical inverse of decomposed mode;
    # it should at least preserve the direction that CUDA score is higher.
    cuda_back = cal.predict_cuda_from_cpu(cpu_band.score_point)
    assert cuda_back.score_point > cpu_band.score_point


def test_decomposed_mode_pr106_anchor_pose_dominated_regime() -> None:
    """Sanity-check decomposed mode at the PR106 frontier."""
    cal = CudaCpuCalibration(architecture_class="hnerv")
    # PR106 anchor (CUDA): d_pose = 1.7e-4 (just above floor); d_seg = 7.84e-4.
    cuda_pose = 1.7e-4
    cuda_seg = 7.84e-4
    archive_bytes = 178_258
    cuda_score = contest_score(
        d_seg=cuda_seg, d_pose=cuda_pose, archive_bytes=archive_bytes
    )
    band = cal.predict_cpu_from_cuda(
        cuda_score,
        d_pose_cuda=cuda_pose,
        d_seg_cuda=cuda_seg,
        archive_bytes=archive_bytes,
    )
    # The CPU prediction should be lower than the CUDA score (pose got smaller).
    assert band.score_point < cuda_score


# ---------------------------------------------------------------------------
# Pose floor saturation
# ---------------------------------------------------------------------------


def test_is_at_pose_floor_saturates_at_threshold() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    # PR106 anchor d_pose=3.4e-5: well below the 1.4e-4 CUDA floor.
    assert cal.is_at_pose_floor(3.4e-5)
    # 5e-4 is above the floor.
    assert not cal.is_at_pose_floor(5e-4)
    # Exactly at the floor returns True (boundary inclusive).
    assert cal.is_at_pose_floor(CUDA_POSE_PRECISION_FLOOR)


def test_effective_pose_loss_uses_ratio_above_threshold() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    # Above the advisory floor: still use the measured axis ratio.
    cpu_pose = cal.effective_pose_loss_for_cpu(1e-3)
    assert math.isclose(cpu_pose, 1e-3 / R_POSE_HNERV, rel_tol=1e-9)


def test_effective_pose_loss_uses_ratio_below_floor() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    # At/below the advisory floor: still use the measured axis ratio.
    cpu_pose = cal.effective_pose_loss_for_cpu(1e-5)
    assert math.isclose(cpu_pose, 1e-5 / R_POSE_HNERV, rel_tol=1e-9)


def test_pose_floor_negative_input_raises() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    with pytest.raises(ValueError):
        cal.is_at_pose_floor(-0.1)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def test_module_level_predict_cpu_from_cuda_returns_band_tuple() -> None:
    band = predict_cpu_from_cuda(0.20, archive_class="hnerv")
    assert len(band) == 2
    score_lo, score_hi = band
    assert score_lo <= score_hi
    # Point estimate is roughly 0.167 (= 0.20 - 0.033). Band should bracket.
    assert score_lo <= 0.167 <= score_hi


def test_module_level_is_at_pose_floor() -> None:
    assert is_at_pose_floor(3.4e-5, archive_class="hnerv")
    assert not is_at_pose_floor(1e-3, archive_class="hnerv")


def test_module_level_effective_pose_loss_for_cpu() -> None:
    cpu_pose = effective_pose_loss_for_cpu(1e-3, archive_class="hnerv")
    assert math.isclose(cpu_pose, 1e-3 / R_POSE_HNERV, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# CPU floor + tagging
# ---------------------------------------------------------------------------


def test_cpu_pose_floor_score_contribution_default() -> None:
    """The default CPU pose floor (3.4e-5) maps to ~0.0184 score points."""
    contrib = cpu_pose_floor_score_contribution()
    # sqrt(10 * 3.4e-5) ≈ 0.01844
    assert math.isclose(contrib, math.sqrt(10.0 * 3.4e-5), rel_tol=1e-9)
    assert 0.018 <= contrib <= 0.020


def test_cpu_pose_floor_score_contribution_custom() -> None:
    contrib = cpu_pose_floor_score_contribution(d_pose_cpu_floor=1e-4)
    # sqrt(10 * 1e-4) = sqrt(0.001) ≈ 0.0316
    assert math.isclose(contrib, math.sqrt(0.001), rel_tol=1e-9)


def test_tag_predicted_band_emits_required_keys() -> None:
    tagged = tag_predicted_band(0.18, axis="cpu", calibration_quality="hnerv-anchored")
    assert tagged["score"] == 0.18
    assert tagged["axis"] == "cpu"
    assert tagged["calibration_quality"] == "hnerv-anchored"


def test_tag_predicted_band_rejects_invalid_axis() -> None:
    with pytest.raises(ValueError):
        tag_predicted_band(0.18, axis="gpu")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Architecture-class validation
# ---------------------------------------------------------------------------


def test_registry_architecture_class_aliases_normalize() -> None:
    assert normalize_architecture_class("hnerv_ft_microcodec") == "hnerv"
    assert normalize_architecture_class("hnerv_lc_v2") == "hnerv"
    assert normalize_architecture_class("qhnerv_ft") == "qhnerv"
    assert normalize_architecture_class("not-a-real-arch") == "unknown"


def test_hnerv_aliases_are_anchored_not_extrapolated() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv_ft_microcodec")
    assert cal.architecture_class == "hnerv"
    assert cal.predict_cpu_from_cuda(0.20).calibration_quality == "hnerv-anchored"


def test_unknown_architecture_class_falls_back_to_extrapolated_unknown() -> None:
    cal = CudaCpuCalibration(architecture_class="not-a-real-arch")
    assert cal.architecture_class == "unknown"
    assert cal.requested_architecture_class == "not-a-real-arch"
    assert cal.predict_cpu_from_cuda(0.20).calibration_quality == "extrapolated"


def test_negative_score_raises() -> None:
    cal = CudaCpuCalibration(architecture_class="hnerv")
    with pytest.raises(ValueError):
        cal.predict_cpu_from_cuda(-0.01)
    with pytest.raises(ValueError):
        cal.predict_cuda_from_cpu(-0.01)


# ---------------------------------------------------------------------------
# Sample prediction: PR106 frontier
# ---------------------------------------------------------------------------


def test_pr106_frontier_constant_gap_matches_empirical_band() -> None:
    """PR106 CUDA score ≈ 0.193; predicted CPU ≈ 0.160."""
    cal = CudaCpuCalibration(architecture_class="hnerv")
    band = cal.predict_cpu_from_cuda(0.193)
    # 0.193 - 0.033 = 0.160. Should bracket.
    assert band.score_lo <= 0.160 <= band.score_hi


def test_owv3_0120_predicted_cpu_band() -> None:
    """owv3_0120 is 0.9974 on CUDA; constant-gap CPU prediction ≈ 0.964."""
    cal = CudaCpuCalibration(architecture_class="hnerv")
    band = cal.predict_cpu_from_cuda(0.9974)
    expected_point = 0.9974 - SCORE_GAP_HNERV_CONSTANT
    assert math.isclose(band.score_point, expected_point, rel_tol=1e-9)
    assert band.score_lo <= expected_point <= band.score_hi


# ---------------------------------------------------------------------------
# Calibration band dataclass
# ---------------------------------------------------------------------------


def test_calibration_band_to_dict_includes_required_keys() -> None:
    band = CalibrationBand(
        score_lo=0.15,
        score_hi=0.17,
        score_point=0.16,
        calibration_quality="hnerv-anchored",
        sigma=0.005,
        notes=("mode=constant_gap",),
    )
    d = band.to_dict()
    assert d["score_lo"] == 0.15
    assert d["score_hi"] == 0.17
    assert d["score_point"] == 0.16
    assert d["calibration_quality"] == "hnerv-anchored"
    assert d["sigma"] == 0.005
    assert d["notes"] == ["mode=constant_gap"]
    assert d["score_claim"] is False
    assert d["promotion_eligible"] is False
    assert d["rank_or_kill_eligible"] is False
    assert d["ready_for_exact_eval_dispatch"] is False
