from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tac.optimization.ddm_g4_spatial_stationarity import (
    AXIS,
    HEIGHT,
    WIDTH,
    DdmG4SpatialStationarityConfigV1,
    StationarityError,
    boundary_mask,
    concentration_fractions,
    encode_sparse_rules,
    pose_homography,
    recurrence_histogram,
    sparse_rule_opportunity,
    transition_codes,
)


def _config(**overrides: object) -> DdmG4SpatialStationarityConfigV1:
    payload: dict[str, object] = {
        "schema": "DdmG4SpatialStationarityConfigV1",
        "run_id": "test",
        "g3_receipt_path": "g3.json",
        "g3_receipt_sha256": "a" * 64,
        "v12_receipt_path": "v12.json",
        "v12_receipt_sha256": "b" * 64,
        "output_directory": "/Volumes/VertigoDataTier/pact/test",
        "compact_receipt_directory": ".omx/research/test",
        "n_pairs": 600,
        "chunk_pairs": 16,
        "seed": 1234,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
    }
    payload.update(overrides)
    return DdmG4SpatialStationarityConfigV1.model_validate(payload)


def test_typed_config_is_hash_stable_and_fail_closed() -> None:
    assert _config().typed_hash() == _config().typed_hash()
    with pytest.raises(ValidationError):
        _config(score_claim=True)
    with pytest.raises(ValidationError):
        _config(extra_field="not allowed")


def test_transition_codes_are_exact_and_reject_invalid_cells() -> None:
    predicted = np.array([[0, 1], [3, 4]], dtype=np.uint8)
    target = np.array([[4, 1], [2, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(transition_codes(predicted, target), [[4, 6], [17, 20]])
    with pytest.raises(StationarityError, match="outside"):
        transition_codes(np.array([[5]], dtype=np.uint8), np.array([[0]], dtype=np.uint8))


def test_boundary_mask_marks_both_sides_of_each_transition() -> None:
    labels = np.array([[0, 0, 1], [0, 2, 2]], dtype=np.uint8)
    expected = np.array([[False, True, True], [True, True, True]])
    np.testing.assert_array_equal(boundary_mask(labels), expected)


def test_concentration_fractions_use_exact_ceil_pixel_budgets() -> None:
    frequency = np.zeros(100, dtype=np.uint16)
    frequency[0] = 40
    frequency[1:5] = 10
    result = concentration_fractions(frequency)
    assert result["total_flip_mass"] == 80
    assert result["top_1pct"] == {"pixels": 1, "flip_mass": 40, "fraction_of_flip_mass": 0.5}
    assert result["top_5pct"]["flip_mass"] == 80


def test_recurrence_histogram_counts_exact_transition_loci_and_event_mass() -> None:
    counts = np.zeros((25, HEIGHT, WIDTH), dtype=np.uint16)
    counts[1, 0, 0] = 1
    counts[1, 0, 1] = 2
    counts[7, 0, 2] = 2
    counts[7, 0, 3] = 5
    result = recurrence_histogram(counts)
    assert result["exact_k"] == [
        {"k": 1, "locus_count": 1, "flip_event_mass": 1},
        {"k": 2, "locus_count": 2, "flip_event_mass": 4},
        {"k": 5, "locus_count": 1, "flip_event_mass": 5},
    ]
    assert result["bands"][1]["flip_event_mass"] == 4


def test_zero_pose_gives_identity_homography() -> None:
    np.testing.assert_allclose(pose_homography(np.zeros(6)), np.eye(3), atol=1e-12)


def test_sparse_rule_encoding_requires_sorted_unique_indices() -> None:
    payload = encode_sparse_rules(np.array([1, 4, 12]), np.array([1, 7, 24], dtype=np.uint8))
    assert payload.startswith(b"G4SR")
    with pytest.raises(StationarityError, match="sorted unique"):
        encode_sparse_rules(np.array([4, 4]), np.array([1, 2], dtype=np.uint8))


def test_sparse_opportunity_subtracts_collateral_and_stays_advisory() -> None:
    counts = np.zeros((25, HEIGHT, WIDTH), dtype=np.uint16)
    counts[1, 10, 20] = 10  # predicted Road -> target Lane
    counts[0, 10, 20] = 3  # collateral if Road is always changed to Lane
    support = np.zeros((HEIGHT, WIDTH), dtype=bool)
    support[10, 20] = True
    result = sparse_rule_opportunity(counts, counts, support, "unit")
    assert result["net_cell_flips_fixed"] == 7
    assert result["parameterization"]["rule_count"] == 1
    assert result["receiver_realized_delta_d_seg"] is None
    assert result["evidence_axis"] == AXIS
    assert result["score_claim"] is False
