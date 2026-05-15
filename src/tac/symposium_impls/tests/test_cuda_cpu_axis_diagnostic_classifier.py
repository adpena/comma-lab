# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.symposium_impls.cuda_cpu_axis_diagnostic_classifier`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.symposium_impls.cuda_cpu_axis_diagnostic_classifier import (
    DEFAULT_EMPIRICAL_ANCHORS,
    CodecClass,
    CodecClassificationResult,
    CudaCpuAxisClassifier,
    PairedAnchor,
    build_default_classifier,
    load_cached_classifier,
    save_classifier,
    update_from_anchor,
)


# ----- canonical empirical anchors --------------------------------------------------------------


def test_default_empirical_anchors_include_a1_pr102_pr106() -> None:
    ids = {a.substrate_id for a in DEFAULT_EMPIRICAL_ANCHORS}
    assert "A1" in ids
    assert "PR102" in ids
    assert "PR106_topk8" in ids


def test_default_anchors_a1_signed_gap_matches_symposium_eureka() -> None:
    a1 = next(a for a in DEFAULT_EMPIRICAL_ANCHORS if a.substrate_id == "A1")
    assert a1.signed_gap == pytest.approx(0.034, abs=0.001)


def test_default_anchors_pr106_topk8_signed_gap_reversed() -> None:
    pr106 = next(a for a in DEFAULT_EMPIRICAL_ANCHORS if a.substrate_id == "PR106_topk8")
    assert pr106.signed_gap < 0
    assert pr106.signed_gap == pytest.approx(-0.021, abs=0.001)


# ----- classifier fit + classify ----------------------------------------------------------------


def test_default_classifier_classifies_a1_as_within_class_refinement() -> None:
    classifier = build_default_classifier()
    result = classifier.classify(score_cuda=0.22635, score_cpu=0.1928)
    assert result.predicted_class == CodecClass.WITHIN_CLASS_REFINEMENT
    assert result.confidence > 0.5
    assert result.score_claim is False


def test_default_classifier_classifies_pr106_topk8_as_residual_sidecar() -> None:
    classifier = build_default_classifier()
    result = classifier.classify(score_cuda=0.205, score_cpu=0.226)
    assert result.predicted_class == CodecClass.RESIDUAL_SIDECAR_CROSS_CLASS
    assert result.confidence > 0.5


def test_default_classifier_classifies_pr102_correctly() -> None:
    classifier = build_default_classifier()
    result = classifier.classify(score_cuda=0.22839, score_cpu=0.19538)
    assert result.predicted_class == CodecClass.WITHIN_CLASS_REFINEMENT


def test_classifier_signed_gap_overrides_score_pair() -> None:
    classifier = build_default_classifier()
    result_a = classifier.classify(score_cuda=0.20, score_cpu=0.18)
    result_b = classifier.classify(signed_gap=0.02)
    assert result_a.signed_gap == pytest.approx(result_b.signed_gap, abs=1e-9)
    assert result_a.predicted_class == result_b.predicted_class


def test_classifier_missing_inputs_raises() -> None:
    classifier = build_default_classifier()
    with pytest.raises(ValueError):
        classifier.classify(score_cuda=0.20)
    with pytest.raises(ValueError):
        classifier.classify(score_cpu=0.18)


def test_empty_classifier_returns_unknown_with_zero_confidence() -> None:
    classifier = CudaCpuAxisClassifier.fit([])
    result = classifier.classify(signed_gap=0.05)
    assert result.predicted_class == CodecClass.UNKNOWN
    assert result.confidence == 0.0


def test_classifier_borderline_gap_returns_unknown_when_far_from_means() -> None:
    """A signed-gap far from any class mean is UNKNOWN-classified."""
    classifier = build_default_classifier()
    result = classifier.classify(signed_gap=0.5)  # well outside any observed class mean
    assert result.predicted_class == CodecClass.UNKNOWN


def test_class_distance_tuple_is_sorted_ascending() -> None:
    classifier = build_default_classifier()
    result = classifier.classify(signed_gap=0.034)
    distances = [d for _, d in result.class_distances]
    assert distances == sorted(distances)


def test_classifier_fit_pooled_std_positive() -> None:
    classifier = build_default_classifier()
    assert classifier.pooled_std > 0


def test_classifier_serialization_roundtrip() -> None:
    classifier = build_default_classifier()
    payload = classifier.to_dict()
    rebuilt = CudaCpuAxisClassifier.from_dict(payload)
    assert rebuilt.class_means == classifier.class_means
    assert rebuilt.pooled_std == pytest.approx(classifier.pooled_std)
    assert rebuilt.n_training_anchors == classifier.n_training_anchors


def test_classifier_from_dict_invalid_raises() -> None:
    with pytest.raises(ValueError):
        CudaCpuAxisClassifier.from_dict({"class_means": "not a mapping"})


# ----- save / load -----------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    classifier = build_default_classifier()
    path = tmp_path / "classifier.json"
    save_classifier(classifier, state_path=path)
    loaded = load_cached_classifier(state_path=path)
    assert loaded is not None
    assert loaded.class_means == classifier.class_means


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_cached_classifier(state_path=tmp_path / "absent.json") is None


# ----- continual-learning hook -----------------------------------------------------------------


def test_update_from_anchor_appends_and_refits(tmp_path: Path) -> None:
    state_path = tmp_path / "classifier.json"
    new_anchor = {
        "substrate_id": "DP1",
        "score_cuda": 0.20,
        "score_cpu": 0.18,
        "codec_class": "within_class_refinement",
    }
    classifier = update_from_anchor(new_anchor, state_path=state_path)
    assert classifier is not None
    assert classifier.n_training_anchors == len(DEFAULT_EMPIRICAL_ANCHORS) + 1
    assert state_path.is_file()


def test_update_from_anchor_missing_field_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "classifier.json"
    incomplete = {"substrate_id": "DP1"}
    assert update_from_anchor(incomplete, state_path=state_path) is None


def test_update_from_anchor_invalid_codec_class_returns_none(tmp_path: Path) -> None:
    state_path = tmp_path / "classifier.json"
    bad = {
        "substrate_id": "DP1",
        "score_cuda": 0.20,
        "score_cpu": 0.18,
        "codec_class": "not_a_class",
    }
    assert update_from_anchor(bad, state_path=state_path) is None


def test_update_from_anchor_accepts_codec_class_enum(tmp_path: Path) -> None:
    state_path = tmp_path / "classifier.json"
    anchor = {
        "substrate_id": "DP1",
        "score_cuda": 0.20,
        "score_cpu": 0.18,
        "codec_class": CodecClass.WITHIN_CLASS_REFINEMENT,
    }
    classifier = update_from_anchor(anchor, state_path=state_path)
    assert classifier is not None
