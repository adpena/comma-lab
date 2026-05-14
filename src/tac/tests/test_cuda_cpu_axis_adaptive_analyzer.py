# SPDX-License-Identifier: MIT
"""Tests for the adaptive-analyzer wrapper that bridges the profile registry
to cathedral_autopilot, meta-Lagrangian, and theoretical_floor solvers."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.cuda_cpu_axis_adaptive_analyzer import (
    detect_class_from_payload,
    explain_observed_pose_drift,
    get_registry,
    per_class_floor_band,
    per_class_lagrangian_weights,
    predict_cpu_score_band,
    reload_registry,
)
from tac.optimization.cuda_cpu_axis_profile_registry import (
    bootstrap_registry_from_hnerv_anchors,
    write_registry,
)


def test_get_registry_returns_bootstrap_when_no_disk_state(tmp_path: Path) -> None:
    nonexistent = tmp_path / "not_there.json"
    reg = get_registry(path=nonexistent, force_reload=True)
    assert "hnerv_ft_microcodec" in reg
    assert reg["hnerv_ft_microcodec"].n_anchors == 5


def test_get_registry_loads_from_disk_when_present(tmp_path: Path) -> None:
    p = tmp_path / "registry.json"
    seed = bootstrap_registry_from_hnerv_anchors()
    write_registry(seed, p)

    reg = get_registry(path=p, force_reload=True)
    assert reg["hnerv_ft_microcodec"].n_anchors == 5


def test_reload_registry_invalidates_cache(tmp_path: Path) -> None:
    p = tmp_path / "registry.json"
    seed = bootstrap_registry_from_hnerv_anchors()
    write_registry(seed, p)
    get_registry(path=p, force_reload=True)
    reg2 = reload_registry(path=p)
    # Same instance per get_registry's caching contract, but post-reload
    # we should get a fresh dict
    assert reg2 is not None
    assert "hnerv_ft_microcodec" in reg2


# ── cathedral_autopilot adapter ────────────────────────────────────────────
def test_predict_cpu_score_band_uses_classifier_when_class_omitted(tmp_path: Path) -> None:
    reg = get_registry(path=tmp_path / "no.json", force_reload=True)
    band = predict_cpu_score_band(
        cuda_score=0.228,
        archive_metadata={"inferred_kind": "ff_packed_brotli_hnerv"},
        registry=reg,
    )
    assert band["architecture_class_used"] == "hnerv_ft_microcodec"


def test_predict_cpu_score_band_low_calibration_class_widens(tmp_path: Path) -> None:
    reg = get_registry(path=tmp_path / "no.json", force_reload=True)
    band_hnerv = predict_cpu_score_band(
        cuda_score=0.228,
        architecture_class="hnerv_ft_microcodec",
        registry=reg,
    )
    band_qhnerv = predict_cpu_score_band(
        cuda_score=0.228,
        architecture_class="qhnerv_ft",
        registry=reg,
    )
    # qhnerv has 0 anchors → wider band
    assert band_qhnerv["score_gap_band_half"] >= band_hnerv["score_gap_band_half"]


# ── meta-Lagrangian adapter ────────────────────────────────────────────────
def test_per_class_lagrangian_weights_hnerv_yields_inverted_pose_emphasis() -> None:
    weights = per_class_lagrangian_weights("hnerv_ft_microcodec")
    # lambda_pose = 1 / R_pose ≈ 1/5.04 ≈ 0.198
    assert 0.18 < weights["lambda_pose"] < 0.22
    assert 0.83 < weights["lambda_seg"] < 0.88
    # rate is unaffected
    assert weights["lambda_rate"] == 1.0


def test_per_class_lagrangian_weights_unknown_class_falls_back_to_hnerv() -> None:
    weights = per_class_lagrangian_weights("totally_unknown")
    # Falls back to unknown_uncalibrated → HNeRV-default constants
    assert 0.15 < weights["lambda_pose"] < 0.25
    assert weights["confidence_label"] == "uncalibrated_default"


def test_per_class_lagrangian_weights_passes_evidence_grade() -> None:
    weights = per_class_lagrangian_weights("hnerv_ft_microcodec")
    assert "[CPU-prep planning-only" in weights["evidence_grade"]
    assert weights["score_claim"] is False
    assert weights["promotion_eligible"] is False
    assert weights["rank_or_kill_eligible"] is False
    assert weights["ready_for_exact_eval_dispatch"] is False


# ── theoretical_floor adapter ──────────────────────────────────────────────
def test_per_class_floor_band_hnerv_default_floor() -> None:
    band = per_class_floor_band(architecture_class="hnerv_ft_microcodec")
    assert band["cuda_pose_floor"] == pytest.approx(1.4e-4)
    # cpu_pose_floor = cuda_floor / R_pose ≈ 2.78e-5
    assert 2.0e-5 < band["cpu_pose_floor_implied"] < 3.0e-5


def test_per_class_floor_band_with_custom_cuda_floor() -> None:
    band = per_class_floor_band(
        architecture_class="hnerv_ft_microcodec",
        cuda_pose_floor=3.0e-4,
    )
    assert band["cuda_pose_floor"] == pytest.approx(3.0e-4)
    assert band["score_claim"] is False
    assert band["promotion_eligible"] is False
    assert band["rank_or_kill_eligible"] is False
    assert band["ready_for_exact_eval_dispatch"] is False


# ── Decomposition adapter ──────────────────────────────────────────────────
def test_explain_observed_pose_drift_uses_default_mechanism_prior() -> None:
    result = explain_observed_pose_drift(observed_r_pose=5.04)
    excess = 5.04 - 1.0  # 4.04
    # Default mechanism prior: 25% decoder + 75% network until measured.
    assert abs(result["decoder_contribution"] - 0.25 * excess) < 1e-6
    assert abs(result["network_contribution"] - 0.75 * excess) < 1e-6
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False


def test_explain_observed_pose_drift_uses_class_floor_estimate() -> None:
    result = explain_observed_pose_drift(observed_r_pose=5.04)
    assert result["pose_floor_estimate"] == pytest.approx(1.4e-4)


# ── Detect-from-payload ────────────────────────────────────────────────────
def test_detect_class_from_payload_recognizes_hnerv() -> None:
    payload = {"inferred_kind": "ff_packed_brotli_hnerv"}
    assert detect_class_from_payload(payload) == "hnerv_ft_microcodec"


def test_detect_class_from_payload_unknown_falls_back() -> None:
    payload = {"inferred_kind": "novel"}
    assert detect_class_from_payload(payload) == "unknown_uncalibrated"
