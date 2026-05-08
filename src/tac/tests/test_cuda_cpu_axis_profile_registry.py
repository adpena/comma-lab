"""Tests for the per-architecture-class CUDA/CPU drift calibration registry.

Covers:
1. Bootstrap from PR100/101/102/103/105 anchors → expected HNeRV defaults
2. Bayesian update with synthetic anchor → posterior moves correctly
3. Outlier rejection (3σ threshold) → flagged, not promoted
4. Auto-classifier (HNeRV / qhnerv / kitchen_sink / balle / unknown)
5. Adaptive analyzer band widening for low-anchor classes
6. Decoder-aware drift decomposition
7. Persistence roundtrip
8. Online-learning hook (harvest_new_anchor_and_update)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.cuda_cpu_axis_profile_registry import (
    ARCHITECTURE_CLASSES,
    DEFAULT_DECODER_POSE_DRIFT_FRACTION,
    DEFAULT_DECODER_PROFILE,
    DEFAULT_POSE_FLOOR_ESTIMATE,
    LOW_CALIBRATION_ANCHOR_THRESHOLD,
    LOW_CALIBRATION_BAND_WIDENING,
    OUTLIER_SIGMA_THRESHOLD,
    ArchitectureProfile,
    DecoderProfile,
    bootstrap_registry_from_hnerv_anchors,
    classify_archive_into_profile,
    confidence_aware_score_band,
    decompose_observed_drift,
    deserialize_registry,
    harvest_new_anchor_and_update,
    query_profile_for_archive_class,
    read_registry,
    serialize_registry,
    update_profile_from_anchor,
    write_registry,
)


# ── 1. Bootstrap tests ─────────────────────────────────────────────────────
def test_bootstrap_seeds_hnerv_cluster_with_5_anchors() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()

    assert "hnerv_ft_microcodec" in registry
    hnerv = registry["hnerv_ft_microcodec"]
    assert hnerv.n_anchors == 5  # PR100/101/102/103/105

    # Empirical mean R_pose for the cluster is ~5.04 ± 0.10 per the
    # public PR scorecard; allow 0.2 tolerance to account for any future
    # micro-edits to the bootstrap anchors.
    assert 4.9 < hnerv.r_pose_mean < 5.3
    assert hnerv.r_pose_std > 0.0

    # R_seg is tightly clustered around 1.17 across all 5 PRs
    assert 1.15 < hnerv.r_seg_mean < 1.20
    assert hnerv.r_seg_std > 0.0

    # Score gap is the (cuda - cpu) score difference, ~0.033
    assert 0.030 < hnerv.score_gap_mean < 0.036


def test_bootstrap_creates_uncalibrated_default_classes() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()

    # Every architecture class in ARCHITECTURE_CLASSES should have an entry
    for cls in ARCHITECTURE_CLASSES:
        assert cls in registry, f"missing class {cls}"

    # Non-HNeRV classes are uncalibrated (n_anchors == 0)
    for cls in ARCHITECTURE_CLASSES:
        if cls == "hnerv_ft_microcodec":
            continue
        assert registry[cls].n_anchors == 0
        assert registry[cls].confidence_label() == "uncalibrated_default"


def test_bootstrap_evidence_anchors_are_promoted() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    hnerv = registry["hnerv_ft_microcodec"]
    assert all(a.get("promoted", False) for a in hnerv.evidence_anchors)
    assert all(
        a.get("seeded_at_bootstrap", False)
        for a in hnerv.evidence_anchors
    )


# ── 2. Bayesian update tests ───────────────────────────────────────────────
def test_synthetic_anchor_with_known_r_updates_posterior() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    hnerv = registry["hnerv_ft_microcodec"]
    n_before = hnerv.n_anchors
    mean_before = hnerv.r_pose_mean

    # Synthetic anchor that sits exactly at the prior mean — posterior should
    # not move materially (sample-size shrinks std slightly).
    anchor = {
        "observed_r_pose": mean_before,
        "observed_r_seg": hnerv.r_seg_mean,
        "score_gap": hnerv.score_gap_mean,
        "source": "synthetic_test",
        "pr_num": 999,
    }

    update = update_profile_from_anchor(hnerv, anchor)

    assert update.accepted is True
    assert update.outlier_candidate is False
    assert hnerv.n_anchors == n_before + 1
    # Mean stays close (since anchor sits at the mean)
    assert abs(hnerv.r_pose_mean - mean_before) < 0.01


def test_synthetic_anchor_pulls_mean_toward_observation() -> None:
    profile = ArchitectureProfile(
        architecture_class="qhnerv_ft",
        n_anchors=0,
        r_pose_mean=5.04,
        r_pose_std=0.10,
        r_seg_mean=1.17,
        r_seg_std=0.01,
        score_gap_mean=0.033,
        score_gap_std=0.001,
    )

    # Two anchors at R_pose = 4.0 — posterior mean should move from 5.04
    # toward 4.0 since these are the only anchors backing this class.
    for i in range(2):
        anchor = {
            "observed_r_pose": 4.0,
            "observed_r_seg": 1.10,
            "score_gap": 0.025,
            "source": f"synthetic_{i}",
        }
        update = update_profile_from_anchor(profile, anchor)
        assert update.accepted is True

    assert profile.n_anchors == 2
    assert abs(profile.r_pose_mean - 4.0) < 0.01
    assert abs(profile.r_seg_mean - 1.10) < 0.01


# ── 3. Outlier rejection tests ─────────────────────────────────────────────
def test_outlier_anchor_with_3sigma_deviation_is_flagged() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    hnerv = registry["hnerv_ft_microcodec"]
    mean = hnerv.r_pose_mean
    std = hnerv.r_pose_std

    # Anchor 10× std away → should be flagged
    outlier_r_pose = mean + 10.0 * std
    anchor = {
        "observed_r_pose": outlier_r_pose,
        "observed_r_seg": hnerv.r_seg_mean,
        "score_gap": hnerv.score_gap_mean,
        "source": "synthetic_outlier",
    }

    update = update_profile_from_anchor(hnerv, anchor)

    assert update.accepted is False
    assert update.outlier_candidate is True
    assert "outlier_reason" in update.anchor
    # Posterior n_anchors should NOT increment (outlier excluded from posterior)
    assert hnerv.n_anchors == 5  # bootstrap count


def test_outlier_anchor_stays_in_audit_trail() -> None:
    """Per CLAUDE.md killing as last resort — outliers are NOT dropped, they
    stay in evidence_anchors with outlier_candidate=True so operators can
    review."""
    registry = bootstrap_registry_from_hnerv_anchors()
    hnerv = registry["hnerv_ft_microcodec"]
    n_evidence_before = len(hnerv.evidence_anchors)

    anchor = {
        "observed_r_pose": 50.0,  # extreme outlier
        "observed_r_seg": 1.17,
        "score_gap": 0.033,
        "source": "test_outlier_audit_trail",
    }
    update_profile_from_anchor(hnerv, anchor)

    # Audit trail grew, posterior count did not
    assert len(hnerv.evidence_anchors) == n_evidence_before + 1
    flagged = [
        a for a in hnerv.evidence_anchors
        if a.get("outlier_candidate", False)
    ]
    assert len(flagged) == 1
    assert flagged[0]["source"] == "test_outlier_audit_trail"


def test_outlier_check_skipped_for_low_anchor_classes() -> None:
    """Classes with < LOW_CALIBRATION_ANCHOR_THRESHOLD anchors should NOT
    flag outliers — there's not enough data to establish a prior."""
    profile = ArchitectureProfile(
        architecture_class="qhnerv_ft",
        n_anchors=0,
    )

    # Way-out-of-band anchor — should still be accepted because we have
    # no posterior to outlier against
    anchor = {
        "observed_r_pose": 50.0,
        "observed_r_seg": 1.17,
        "score_gap": 0.033,
        "source": "first_anchor_no_prior",
    }
    update = update_profile_from_anchor(profile, anchor)
    assert update.accepted is True


# ── 4. Auto-classifier tests ───────────────────────────────────────────────
def test_classifier_recognizes_hnerv_via_inferred_kind() -> None:
    md = {"inferred_kind": "ff_packed_brotli_hnerv"}
    assert classify_archive_into_profile(archive_metadata=md) == "hnerv_ft_microcodec"


def test_classifier_distinguishes_lc_v2_from_microcodec() -> None:
    md_lc = {"inferred_kind": "ff_packed_brotli_hnerv", "title": "hnerv_lc_v2"}
    md_micro = {"inferred_kind": "ff_packed_brotli_hnerv", "title": "hnerv ft microcodec"}
    assert classify_archive_into_profile(archive_metadata=md_lc) == "hnerv_lc_v2"
    assert classify_archive_into_profile(archive_metadata=md_micro) == "hnerv_ft_microcodec"


def test_classifier_recognizes_kitchen_sink() -> None:
    md = {"title": "kitchen_sink (0.197)"}
    assert classify_archive_into_profile(archive_metadata=md) == "kitchen_sink_ensemble"


def test_classifier_recognizes_balle_hyperprior() -> None:
    md = {"description": "Balle scale_hyperprior NN codec"}
    assert classify_archive_into_profile(archive_metadata=md) == "balle_scale_hyperprior"


def test_classifier_falls_back_to_unknown() -> None:
    md = {"inferred_kind": "wholly_novel_format", "title": "mystery"}
    assert classify_archive_into_profile(archive_metadata=md) == "unknown_uncalibrated"


def test_classifier_explicit_override_wins() -> None:
    md = {
        "inferred_kind": "ff_packed_brotli_hnerv",
        "architecture_class": "balle_scale_hyperprior",
    }
    assert classify_archive_into_profile(archive_metadata=md) == "balle_scale_hyperprior"


def test_classifier_recognizes_via_section_names() -> None:
    md = {
        "sections": [
            {"name": "decoder_packed_brotli", "bytes": 161891},
            {"name": "latents_and_sidecar_brotli", "bytes": 15854},
        ]
    }
    assert classify_archive_into_profile(archive_metadata=md) == "hnerv_ft_microcodec"


# ── 5. Adaptive analyzer band-widening tests ───────────────────────────────
def test_low_anchor_class_widens_predicted_band() -> None:
    """When n_anchors < 3, the predicted CPU band must widen by 1.5×."""
    registry = bootstrap_registry_from_hnerv_anchors()

    # HNeRV (5 anchors) → narrow band
    hnerv_band = confidence_aware_score_band(
        architecture_class="hnerv_ft_microcodec",
        cuda_score=0.228,
        registry=registry,
    )
    # qhnerv (0 anchors) → wide band
    qhnerv_band = confidence_aware_score_band(
        architecture_class="qhnerv_ft",
        cuda_score=0.228,
        registry=registry,
    )

    assert hnerv_band["confidence_label"] == "calibrated"
    assert qhnerv_band["confidence_label"] == "uncalibrated_default"

    # The qhnerv band uses the prior std (0.001) widened by 1.5 = 0.0015
    # The hnerv band uses the posterior std × 1.0
    # qhnerv's band_half should be wider
    assert qhnerv_band["score_gap_band_half"] >= hnerv_band["score_gap_band_half"]


def test_band_widening_factor_matches_constant() -> None:
    profile = ArchitectureProfile(
        architecture_class="balle_scale_hyperprior",
        n_anchors=0,
        score_gap_mean=0.033,
        score_gap_std=0.001,
    )
    band = profile.predict_cpu_score(cuda_score=0.228)
    expected_half = LOW_CALIBRATION_BAND_WIDENING * 0.001
    assert abs(band["score_gap_band_half"] - expected_half) < 1e-9


def test_predicted_band_carries_no_score_claim() -> None:
    profile = ArchitectureProfile(architecture_class="hnerv_ft_microcodec")
    band = profile.predict_cpu_score(cuda_score=0.228)
    assert band["score_claim"] is False
    assert band["promotion_eligible"] is False
    assert band["ready_for_exact_eval_dispatch"] is False
    assert "[predicted; learning-layer registry posterior]" in band["evidence_grade"]


# ── 6. Decoder-aware split tests ───────────────────────────────────────────
def test_decompose_observed_drift_default_25_percent_decoder() -> None:
    result = decompose_observed_drift(observed_r_pose=5.04)
    excess = 5.04 - 1.0
    expected_decoder = excess * DEFAULT_DECODER_POSE_DRIFT_FRACTION
    expected_network = excess * (1 - DEFAULT_DECODER_POSE_DRIFT_FRACTION)
    assert abs(result["decoder_contribution"] - expected_decoder) < 1e-9
    assert abs(result["network_contribution"] - expected_network) < 1e-9


def test_decompose_observed_drift_custom_decoder_profile() -> None:
    custom = DecoderProfile(
        decoder_pair=("DaliVideoDataset", "DaliVideoDataset"),
        pose_drift_fraction=0.0,  # same decoder both sides
        seg_drift_fraction=0.0,
    )
    result = decompose_observed_drift(observed_r_pose=5.04, decoder_profile=custom)
    assert result["decoder_contribution"] == 0.0
    assert result["network_contribution"] == 5.04 - 1.0


def test_decompose_observed_drift_rejects_negative_input() -> None:
    with pytest.raises(ValueError):
        decompose_observed_drift(observed_r_pose=-0.1)


def test_decompose_observed_drift_returns_no_score_claim() -> None:
    result = decompose_observed_drift(observed_r_pose=5.04)
    assert "[CPU-prep planning-only" in result["evidence_grade"]


# ── 7. Persistence roundtrip tests ─────────────────────────────────────────
def test_serialize_then_deserialize_preserves_posterior(tmp_path: Path) -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    hnerv_before = registry["hnerv_ft_microcodec"]
    mean_before = hnerv_before.r_pose_mean

    payload = serialize_registry(registry)
    restored = deserialize_registry(payload)
    hnerv_after = restored["hnerv_ft_microcodec"]

    assert hnerv_after.n_anchors == hnerv_before.n_anchors
    assert abs(hnerv_after.r_pose_mean - mean_before) < 1e-9
    assert hnerv_after.architecture_class == hnerv_before.architecture_class


def test_write_then_read_registry_roundtrip(tmp_path: Path) -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    path = tmp_path / "registry.json"

    written = write_registry(registry, path)
    assert written == path
    assert path.exists()

    restored = read_registry(path)
    assert "hnerv_ft_microcodec" in restored
    assert restored["hnerv_ft_microcodec"].n_anchors == 5


def test_read_registry_bootstraps_when_missing(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope.json"
    registry = read_registry(nonexistent, bootstrap_if_missing=True)
    assert "hnerv_ft_microcodec" in registry
    assert registry["hnerv_ft_microcodec"].n_anchors == 5


def test_read_registry_raises_when_missing_and_no_bootstrap(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        read_registry(nonexistent, bootstrap_if_missing=False)


def test_serialised_payload_carries_evidence_grade_metadata() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    payload = serialize_registry(registry)
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["schema"] == "cuda_cpu_axis_profile_registry.v1"


# ── 8. Online-learning hook tests ──────────────────────────────────────────
def _adjudicated_payload(
    *,
    cpu_pose: float,
    cpu_seg: float,
    cuda_pose: float,
    cuda_seg: float,
    archive_bytes: int = 178_258,
    architecture_class: str = "hnerv_ft_microcodec",
) -> dict:
    return {
        "archive_bytes": archive_bytes,
        "archive_sha256": "deadbeef",
        "architecture_class": architecture_class,
        "cpu": {"pose": cpu_pose, "seg": cpu_seg},
        "cuda": {"pose": cuda_pose, "seg": cuda_seg},
        "source": "test_adjudicated_json",
    }


def test_harvest_extracts_paired_anchor_and_updates_registry(tmp_path: Path) -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    n_before = registry["hnerv_ft_microcodec"].n_anchors

    payload = _adjudicated_payload(
        cpu_pose=3.4e-5, cpu_seg=5.7e-4,
        cuda_pose=1.7e-4, cuda_seg=6.7e-4,
    )

    audit_log = tmp_path / "audit.jsonl"
    update = harvest_new_anchor_and_update(
        payload, registry=registry, audit_log_path=audit_log,
    )

    assert update is not None
    assert update.accepted is True
    assert registry["hnerv_ft_microcodec"].n_anchors == n_before + 1
    assert audit_log.exists()

    # Audit log line is valid JSON
    lines = audit_log.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["architecture_class"] == "hnerv_ft_microcodec"
    assert record["accepted"] is True


def test_harvest_returns_none_for_unpaired_payload() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    # Only CUDA side present — should not produce an anchor
    payload = {
        "archive_bytes": 178_258,
        "cuda": {"pose": 1.7e-4, "seg": 6.7e-4},
    }
    update = harvest_new_anchor_and_update(payload, registry=registry)
    assert update is None


def test_harvest_auto_instantiates_unknown_class() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    # Remove a class to simulate a never-seen architecture
    payload = _adjudicated_payload(
        cpu_pose=5.0e-5, cpu_seg=4.0e-4,
        cuda_pose=2.5e-4, cuda_seg=4.7e-4,
        architecture_class="totally_new_paradigm_v2",
    )
    update = harvest_new_anchor_and_update(
        payload,
        registry=registry,
        architecture_class="totally_new_paradigm_v2",
        audit_log_path=None,
    )
    assert update is not None
    assert "totally_new_paradigm_v2" in registry
    assert registry["totally_new_paradigm_v2"].n_anchors == 1


# ── Auxiliary tests ────────────────────────────────────────────────────────
def test_query_profile_returns_hnerv_default_for_unknown_class() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    # Class not in ARCHITECTURE_CLASSES at all
    profile = query_profile_for_archive_class(
        "wholly_unknown_class", registry=registry
    )
    # Should fall back to ``unknown_uncalibrated`` profile
    assert profile.architecture_class == "unknown_uncalibrated"


def test_confidence_aware_score_band_returns_class_used() -> None:
    registry = bootstrap_registry_from_hnerv_anchors()
    band = confidence_aware_score_band(
        architecture_class="hnerv_ft_microcodec",
        cuda_score=0.228,
        registry=registry,
    )
    assert band["architecture_class_used"] == "hnerv_ft_microcodec"
    assert band["n_anchors_backing"] == 5


def test_score_gap_predicts_within_known_anchor_range() -> None:
    """The HNeRV cluster's score_gap_mean ≈ 0.033. Predicted CPU score for
    cuda=0.228 should be ~0.195 (the medal-band CPU score)."""
    registry = bootstrap_registry_from_hnerv_anchors()
    band = confidence_aware_score_band(
        architecture_class="hnerv_ft_microcodec",
        cuda_score=0.228,
        registry=registry,
    )
    assert 0.190 < band["predicted_cpu_score"] < 0.200
