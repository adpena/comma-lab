# SPDX-License-Identifier: MIT
"""Tests for tac.domain_priors namespace + 3 canonical equations.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 Steps 4.1-4.2 + Catalog #344
canonical equations registry discipline.

Test coverage:
  * Each helper dataclass contract (frozen + __post_init__ invariants)
  * Builder smoke tests (build from synthetic input + verify shape)
  * Per-pair → per-frame projection correctness (adjacent + non_overlap)
  * Aggregator operator validation
  * Ego-motion atlas from pose-vector + affine-flow
  * Per-class statistical priors (5-class invariants)
  * Comma2k19 priors duck-typed cache + Catalog #213 invariants
  * Canonical equation registration via register_canonical_equation
  * Query via get_equation_by_id returns the equation
  * Relevance-token / consumer / producer auditability
  * Catalog #287 placeholder-rationale rejection (via dataclass invariants)
  * Catalog #185 sister-callable regression guard (gate-style)
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.canonical_equations import (
    CANONICAL_EQUATIONS_REGISTRY_PATH,
    CanonicalEquation,
    get_equation_by_id,
    query_equations_by_consumer,
    query_equations_by_producer,
    register_canonical_equation,
)
from tac.domain_priors import (
    DOMAIN_PRIORS_EQUATION_IDS,
    PER_FRAME_DIFFICULTY_AGGREGATOR_VALID,
    Comma2k19DashcamPriors,
    EgoMotionConcentrationAtlas,
    EgoMotionConcentrationEntry,
    PerClassPrior,
    PerClassStatisticalPriors,
    PerFrameDifficultyAtlas,
    PerFrameDifficultyEntry,
    SEGNET_CLASS_COUNT,
    SEGNET_CLASS_NAMES,
    build_all_domain_prior_equations,
    build_comma2k19_dashcam_priors_from_cache,
    build_ego_motion_concentration_from_affine_flow,
    build_ego_motion_concentration_from_pose_anchors,
    build_ego_motion_concentration_prior_v1,
    build_per_class_statistical_priors_from_scorer_output,
    build_per_frame_difficulty_atlas_v1,
    build_per_frame_difficulty_from_per_pair_atlas,
    build_per_segnet_class_chroma_priors_v1,
    register_domain_prior_canonical_equations,
)
from tac.provenance.builders import build_provenance_for_predicted


_HEX_64 = "0" * 60 + "abcd"  # canonical valid 64-char hex sha


def _predicted_prov(model_id: str = "test.predicted.v1"):
    return build_provenance_for_predicted(
        model_id=model_id,
        inputs_sha256=_HEX_64,
        measurement_axis="[predicted]",
        hardware_substrate="unknown",
    )


# ----------------------------------------------------------------------
# Per-frame difficulty atlas — Step 4.1 helper #1
# ----------------------------------------------------------------------


def test_per_frame_difficulty_aggregator_canonical_set():
    """Aggregator constants are pinned and frozen."""
    assert isinstance(PER_FRAME_DIFFICULTY_AGGREGATOR_VALID, frozenset)
    assert PER_FRAME_DIFFICULTY_AGGREGATOR_VALID == {
        "mean_over_incident_pairs",
        "max_over_incident_pairs",
        "sum_over_incident_pairs",
    }


def test_per_frame_difficulty_entry_invariants():
    """Entry dataclass rejects negative / NaN / non-int / empty-incident."""
    # Valid construction.
    e = PerFrameDifficultyEntry(
        frame_index=0,
        difficulty=1.5,
        incident_pair_indices=(0,),
        difficulty_rank=0,
    )
    assert e.frame_index == 0
    # Negative difficulty rejected.
    with pytest.raises(ValueError, match=r"must be >= 0"):
        PerFrameDifficultyEntry(
            frame_index=0,
            difficulty=-0.1,
            incident_pair_indices=(0,),
            difficulty_rank=0,
        )
    # NaN difficulty rejected.
    with pytest.raises(ValueError, match=r"must not be NaN"):
        PerFrameDifficultyEntry(
            frame_index=0,
            difficulty=float("nan"),
            incident_pair_indices=(0,),
            difficulty_rank=0,
        )
    # Empty incident rejected.
    with pytest.raises(ValueError, match=r"non-empty"):
        PerFrameDifficultyEntry(
            frame_index=0,
            difficulty=0.0,
            incident_pair_indices=(),
            difficulty_rank=0,
        )
    # Negative frame index rejected.
    with pytest.raises(ValueError, match=r"must be >= 0"):
        PerFrameDifficultyEntry(
            frame_index=-1,
            difficulty=0.0,
            incident_pair_indices=(0,),
            difficulty_rank=0,
        )


def test_per_frame_difficulty_atlas_adjacent_construction():
    """Adjacent construction: total_frames = total_pairs + 1."""
    per_pair = [1.0, 2.0, 3.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
        pair_to_frame_construction="adjacent",
    )
    assert atlas.total_pairs == 3
    assert atlas.total_frames == 4
    assert len(atlas.entries) == 4
    # Frame 0 incident = (0,); mean(1.0) = 1.0
    assert atlas.entries[0].incident_pair_indices == (0,)
    assert atlas.entries[0].difficulty == pytest.approx(1.0)
    # Frame 1 incident = (0, 1); mean(1.0, 2.0) = 1.5
    assert atlas.entries[1].incident_pair_indices == (0, 1)
    assert atlas.entries[1].difficulty == pytest.approx(1.5)
    # Frame 2 incident = (1, 2); mean(2.0, 3.0) = 2.5
    assert atlas.entries[2].incident_pair_indices == (1, 2)
    assert atlas.entries[2].difficulty == pytest.approx(2.5)
    # Frame 3 incident = (2,); mean(3.0) = 3.0
    assert atlas.entries[3].incident_pair_indices == (2,)
    assert atlas.entries[3].difficulty == pytest.approx(3.0)


def test_per_frame_difficulty_atlas_non_overlap_construction():
    """Non-overlap construction: total_frames = total_pairs * 2."""
    per_pair = [1.0, 2.0, 3.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[contest-CUDA]",
        provenance=_predicted_prov(),
        pair_to_frame_construction="non_overlap",
    )
    assert atlas.total_pairs == 3
    assert atlas.total_frames == 6
    # Frame 0 + 1 → pair 0 (val 1.0); frame 2 + 3 → pair 1 (val 2.0); etc.
    assert atlas.entries[0].incident_pair_indices == (0,)
    assert atlas.entries[0].difficulty == pytest.approx(1.0)
    assert atlas.entries[1].incident_pair_indices == (0,)
    assert atlas.entries[5].incident_pair_indices == (2,)
    assert atlas.entries[5].difficulty == pytest.approx(3.0)


def test_per_frame_difficulty_atlas_max_aggregator():
    """Max aggregator picks the worst-case per-pair."""
    per_pair = [1.0, 5.0, 2.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
        aggregator="max_over_incident_pairs",
        pair_to_frame_construction="adjacent",
    )
    # Frame 1 incident = (0, 1); max(1.0, 5.0) = 5.0
    assert atlas.entries[1].difficulty == pytest.approx(5.0)
    # Frame 2 incident = (1, 2); max(5.0, 2.0) = 5.0
    assert atlas.entries[2].difficulty == pytest.approx(5.0)


def test_per_frame_difficulty_atlas_sum_aggregator():
    """Sum aggregator returns additive incident contribution."""
    per_pair = [1.0, 2.0, 3.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
        aggregator="sum_over_incident_pairs",
    )
    # Frame 1 incident = (0, 1); sum = 3.0
    assert atlas.entries[1].difficulty == pytest.approx(3.0)
    # Frame 2 incident = (1, 2); sum = 5.0
    assert atlas.entries[2].difficulty == pytest.approx(5.0)


def test_per_frame_difficulty_atlas_invalid_aggregator_rejected():
    """Aggregator must be one of canonical set."""
    with pytest.raises(ValueError, match=r"must be one of"):
        build_per_frame_difficulty_from_per_pair_atlas(
            [1.0],
            archive_sha256=_HEX_64,
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
            aggregator="ad_hoc_invalid",
        )


def test_per_frame_difficulty_atlas_negative_values_rejected():
    """Per-pair difficulty must be non-negative (gradient norms are)."""
    with pytest.raises(ValueError, match=r"must be non-negative"):
        build_per_frame_difficulty_from_per_pair_atlas(
            [-1.0, 1.0],
            archive_sha256=_HEX_64,
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
        )


def test_per_frame_difficulty_atlas_ranking():
    """Top-K and bottom-K orderings are correct."""
    per_pair = [1.0, 5.0, 2.0]  # frame_diff: [1.0, 3.0, 3.5, 2.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
        top_k=2,
        bottom_k=2,
    )
    # Hardest = frame 2 (diff 3.5), then frame 1 (diff 3.0)
    assert atlas.top_k_hardest_frame_indices == (2, 1)
    # Easiest = frame 0 (diff 1.0), then frame 3 (diff 2.0)
    assert atlas.bottom_k_easiest_frame_indices == (0, 3)


def test_per_frame_difficulty_atlas_as_dict_roundtrip():
    """as_dict returns JSON-safe schema_v1 payload."""
    per_pair = [1.0, 2.0]
    atlas = build_per_frame_difficulty_from_per_pair_atlas(
        per_pair,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    payload = atlas.as_dict()
    assert payload["schema"] == "per_frame_difficulty_atlas_v1"
    assert payload["total_pairs"] == 2
    assert payload["total_frames"] == 3
    assert payload["aggregator"] == "mean_over_incident_pairs"
    assert payload["source_pair_atlas_archive_sha256"] == _HEX_64
    assert len(payload["entries"]) == 3
    assert payload["provenance"]["evidence_grade"] == "predicted"


def test_per_frame_difficulty_atlas_bad_sha_rejected():
    """Archive sha must be 64-char hex."""
    with pytest.raises(ValueError, match=r"64-char hex"):
        build_per_frame_difficulty_from_per_pair_atlas(
            [1.0],
            archive_sha256="not_64_chars",
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
        )


# ----------------------------------------------------------------------
# Ego-motion concentration atlas — Step 4.1 helper #2
# ----------------------------------------------------------------------


def test_ego_motion_entry_invariants():
    """Entry dataclass rejects invalid values."""
    e = EgoMotionConcentrationEntry(
        frame_index=0,
        pose_magnitude_l2=1.0,
        flow_concentration=0.5,
        ego_motion_score=0.75,
    )
    assert e.flow_concentration == 0.5
    # flow_concentration > 1 rejected
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EgoMotionConcentrationEntry(
            frame_index=0,
            pose_magnitude_l2=1.0,
            flow_concentration=1.5,
            ego_motion_score=0.0,
        )
    # Negative magnitude rejected
    with pytest.raises(ValueError, match=r"must be >= 0"):
        EgoMotionConcentrationEntry(
            frame_index=0,
            pose_magnitude_l2=-1.0,
            flow_concentration=0.0,
            ego_motion_score=0.0,
        )


def test_ego_motion_from_pose_anchors_canonical():
    """Pose-vector path: per-pair L2 → per-frame mean."""
    pose_vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # magnitude = 1.0
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # magnitude = 2.0
    ]
    atlas = build_ego_motion_concentration_from_pose_anchors(
        pose_vectors,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    assert atlas.total_frames == 3  # 2 pairs + 1
    assert atlas.source_anchor_kind == "pose_vector"
    # Frame 0 incident = (0,); magnitude = 1.0
    assert atlas.entries[0].pose_magnitude_l2 == pytest.approx(1.0)
    # Frame 1 incident = (0, 1); mean(1.0, 2.0) = 1.5
    assert atlas.entries[1].pose_magnitude_l2 == pytest.approx(1.5)
    # Frame 2 incident = (1,); magnitude = 2.0
    assert atlas.entries[2].pose_magnitude_l2 == pytest.approx(2.0)
    # Pose-only path: flow_concentration = 0.0
    for entry in atlas.entries:
        assert entry.flow_concentration == 0.0
    # ego_motion_score == pose_magnitude_l2 (flow_concentration=0)
    assert atlas.entries[1].ego_motion_score == pytest.approx(1.5)
    # atick_redlich_alignment_tag declared per Catalog #311
    assert "Atick-Redlich" in atlas.atick_redlich_alignment_tag


def test_ego_motion_from_pose_anchors_invalid_dof_rejected():
    """Pose vector must be 6-DOF."""
    with pytest.raises(ValueError, match=r"6-DOF"):
        build_ego_motion_concentration_from_pose_anchors(
            [[1.0, 2.0, 3.0]],  # only 3-DOF
            archive_sha256=_HEX_64,
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
        )


def test_ego_motion_from_affine_flow_concentration_metric():
    """Affine flow concentration: pure translation = 0; rotation+skew = high."""
    # Pure translation: a=1, b=0, tx=5, c=0, d=1, ty=5 → concentration 0
    pure_translation = [1.0, 0.0, 5.0, 0.0, 1.0, 5.0]
    # Pure rotation (45deg approx): a=0.7, b=-0.7, tx=0, c=0.7, d=0.7, ty=0
    # Translation magnitude = 0; rotation+skew dominates → concentration → 1
    pure_rotation = [0.7, -0.7, 0.0, 0.7, 0.7, 0.0]
    atlas = build_ego_motion_concentration_from_affine_flow(
        [pure_translation, pure_rotation],
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    # Frame 0 incident = (0,) pure_translation: con=0, mag=sqrt(50)≈7.07
    assert atlas.entries[0].flow_concentration == pytest.approx(0.0, abs=1e-6)
    assert atlas.entries[0].pose_magnitude_l2 == pytest.approx(math.sqrt(50.0))
    # Frame 2 incident = (1,) pure_rotation: con=1.0, mag=0
    assert atlas.entries[2].flow_concentration == pytest.approx(1.0, abs=1e-6)
    assert atlas.entries[2].pose_magnitude_l2 == pytest.approx(0.0)
    assert atlas.source_anchor_kind == "affine_flow"


def test_ego_motion_atlas_as_dict():
    """as_dict returns JSON-safe schema_v1 payload."""
    pose_vectors = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    atlas = build_ego_motion_concentration_from_pose_anchors(
        pose_vectors,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    payload = atlas.as_dict()
    assert payload["schema"] == "ego_motion_concentration_atlas_v1"
    assert payload["source_anchor_kind"] == "pose_vector"
    assert "Atick-Redlich" in payload["atick_redlich_alignment_tag"]


# ----------------------------------------------------------------------
# Per-class statistical priors — Step 4.1 helper #3
# ----------------------------------------------------------------------


def test_segnet_class_constants():
    """5 canonical SegNet classes; canonical names in operator-readable form."""
    assert SEGNET_CLASS_COUNT == 5
    assert len(SEGNET_CLASS_NAMES) == 5
    assert SEGNET_CLASS_NAMES[0] == "background_sky_road"
    assert SEGNET_CLASS_NAMES[1] == "vehicle"
    assert SEGNET_CLASS_NAMES[2] == "pedestrian"
    assert SEGNET_CLASS_NAMES[3] == "lane_marking"
    assert SEGNET_CLASS_NAMES[4] == "other_foreground"


def test_per_class_prior_invariants():
    """PerClassPrior rejects mismatched class_index/class_name."""
    p = PerClassPrior(
        class_index=1,
        class_name="vehicle",
        pixel_count_fraction=0.2,
        chroma_variance=0.5,
        motion_magnitude_mean=1.0,
    )
    assert p.class_index == 1
    # Mismatch rejected
    with pytest.raises(ValueError, match=r"must equal canonical"):
        PerClassPrior(
            class_index=1,
            class_name="background_sky_road",  # wrong name for index 1
            pixel_count_fraction=0.2,
            chroma_variance=0.5,
            motion_magnitude_mean=1.0,
        )
    # Out-of-range class_index
    with pytest.raises(ValueError, match=r"must be in"):
        PerClassPrior(
            class_index=5,  # out of [0, 5)
            class_name="vehicle",
            pixel_count_fraction=0.2,
            chroma_variance=0.5,
            motion_magnitude_mean=1.0,
        )


def test_per_class_statistical_priors_sum_invariant():
    """Sum of pixel_count_fractions must equal 1.0."""
    priors = build_per_class_statistical_priors_from_scorer_output(
        pixel_count_fractions=[0.6, 0.15, 0.05, 0.15, 0.05],  # sum = 1.0
        chroma_variances=[0.1, 0.2, 0.3, 0.4, 0.5],
        motion_magnitudes=[0.0, 1.0, 2.0, 0.5, 1.5],
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    assert len(priors.class_priors) == 5
    assert priors.class_priors[0].class_name == "background_sky_road"
    # Sum violation rejected
    with pytest.raises(ValueError, match=r"must equal 1\.0"):
        build_per_class_statistical_priors_from_scorer_output(
            pixel_count_fractions=[0.5, 0.5, 0.5, 0.5, 0.5],  # sum = 2.5
            chroma_variances=[0.0] * 5,
            motion_magnitudes=[0.0] * 5,
            archive_sha256=_HEX_64,
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
        )


def test_per_class_statistical_priors_length_invariant():
    """Vectors must be length 5."""
    with pytest.raises(ValueError, match=r"must equal SEGNET_CLASS_COUNT"):
        build_per_class_statistical_priors_from_scorer_output(
            pixel_count_fractions=[0.5, 0.5],  # only 2
            chroma_variances=[0.0] * 5,
            motion_magnitudes=[0.0] * 5,
            archive_sha256=_HEX_64,
            measurement_axis="[predicted]",
            provenance=_predicted_prov(),
        )


def test_per_class_statistical_priors_canonical_openpilot_cited_default():
    """Default citation of openpilot mask prior contract is True."""
    priors = build_per_class_statistical_priors_from_scorer_output(
        pixel_count_fractions=[1.0, 0.0, 0.0, 0.0, 0.0],
        chroma_variances=[0.0] * 5,
        motion_magnitudes=[0.0] * 5,
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    assert priors.canonical_openpilot_mask_prior_contract_cited is True


def test_per_class_statistical_priors_as_dict():
    """JSON-safe serialization."""
    priors = build_per_class_statistical_priors_from_scorer_output(
        pixel_count_fractions=[0.6, 0.15, 0.05, 0.15, 0.05],
        chroma_variances=[0.1, 0.2, 0.3, 0.4, 0.5],
        motion_magnitudes=[0.0, 1.0, 2.0, 0.5, 1.5],
        archive_sha256=_HEX_64,
        measurement_axis="[predicted]",
        provenance=_predicted_prov(),
    )
    payload = priors.as_dict()
    assert payload["schema"] == "per_class_statistical_priors_v1"
    assert len(payload["class_priors"]) == 5
    assert payload["class_priors"][1]["class_name"] == "vehicle"


# ----------------------------------------------------------------------
# Comma2k19 dashcam priors — Step 4.1 helper #4
# ----------------------------------------------------------------------


class _MockComma2k19Cache:
    """Duck-typed mock satisfying the canonical cache surface for tests."""

    DATASET_LICENSE = "MIT"
    CANONICAL_SOURCE_URL = "https://github.com/commaai/comma2k19"

    def __init__(self, cached_ids, manifest=None):
        self._cached = list(cached_ids)
        self.chunk_manifest = manifest or {}

    def list_cached_chunks(self):
        return list(self._cached)


def test_comma2k19_priors_from_mock_cache():
    """Build atlas from duck-typed cache; OOD-tagged + canonical-helper cited."""
    mock_entry = SimpleNamespace(size_bytes=37_491_754)
    cache = _MockComma2k19Cache(
        cached_ids=["example_1"],
        manifest={"example_1": mock_entry},
    )
    priors = build_comma2k19_dashcam_priors_from_cache(
        cache,
        provenance=_predicted_prov(),
    )
    assert priors.cached_chunk_ids == ("example_1",)
    assert priors.total_cached_bytes == 37_491_754
    assert priors.dataset_license_spdx == "MIT"
    assert "comma2k19" in priors.dataset_provenance
    # ALWAYS OOD per Catalog #209
    assert priors.is_ood_relative_to_contest_video is True
    # Canonical helper cited per Catalog #213
    assert "Comma2k19LocalCache" in priors.canonical_cache_helper_invocation


def test_comma2k19_priors_rejects_non_mit_license():
    """License MUST be MIT per Comma2k19 canonical."""

    class _BadLicenseCache:
        DATASET_LICENSE = "GPLv3"
        CANONICAL_SOURCE_URL = "https://github.com/commaai/comma2k19"
        chunk_manifest = {}

        def list_cached_chunks(self):
            return []

    with pytest.raises(ValueError, match=r"must be 'MIT'"):
        build_comma2k19_dashcam_priors_from_cache(
            _BadLicenseCache(),
            provenance=_predicted_prov(),
        )


def test_comma2k19_priors_rejects_non_comma2k19_provenance():
    """Provenance MUST reference comma2k19 canonical dataset."""

    class _BadProvCache:
        DATASET_LICENSE = "MIT"
        CANONICAL_SOURCE_URL = "https://github.com/somebody/different-dataset"
        chunk_manifest = {}

        def list_cached_chunks(self):
            return []

    with pytest.raises(ValueError, match=r"comma2k19"):
        build_comma2k19_dashcam_priors_from_cache(
            _BadProvCache(),
            provenance=_predicted_prov(),
        )


def test_comma2k19_priors_rejects_missing_canonical_attrs():
    """Cache missing canonical attrs is rejected."""
    incomplete_cache = SimpleNamespace(list_cached_chunks=lambda: [])
    with pytest.raises(TypeError, match=r"missing required attribute"):
        build_comma2k19_dashcam_priors_from_cache(
            incomplete_cache,
            provenance=_predicted_prov(),
        )


def test_comma2k19_priors_as_dict_round_trip():
    """JSON-safe serialization."""
    cache = _MockComma2k19Cache(cached_ids=[], manifest={})
    priors = build_comma2k19_dashcam_priors_from_cache(
        cache,
        provenance=_predicted_prov(),
    )
    payload = priors.as_dict()
    assert payload["schema"] == "comma2k19_dashcam_priors_v1"
    assert payload["is_ood_relative_to_contest_video"] is True
    assert payload["total_cached_bytes"] == 0


# ----------------------------------------------------------------------
# Canonical equations — Step 4.2
# ----------------------------------------------------------------------


def test_equation_ids_canonical_set():
    """3 equation IDs pinned in canonical order."""
    assert DOMAIN_PRIORS_EQUATION_IDS == (
        "per_frame_difficulty_atlas_v1",
        "ego_motion_concentration_prior_v1",
        "per_segnet_class_chroma_priors_v1",
    )


def test_build_per_frame_difficulty_atlas_v1_contract():
    """Equation 1: per-frame difficulty atlas — orphan-equation invariant."""
    eq = build_per_frame_difficulty_atlas_v1()
    assert eq.equation_id == "per_frame_difficulty_atlas_v1"
    assert isinstance(eq, CanonicalEquation)
    # Orphan-equation invariant: non-empty producers AND consumers
    assert len(eq.canonical_producers) >= 1
    assert len(eq.canonical_consumers) >= 1
    # Producer references the actual canonical helper
    assert any("per_frame_difficulty" in p for p in eq.canonical_producers)
    # Latex form is non-empty + cites the aggregation formula
    assert "D_{\\text{frame}}" in eq.latex_form or "D_" in eq.latex_form
    # Domain of validity declared
    assert "aggregator_operators" in eq.domain_of_validity
    assert "mean_over_incident_pairs" in eq.domain_of_validity["aggregator_operators"]
    # Initial empirical_anchors empty per design
    assert eq.empirical_anchors == ()


def test_build_ego_motion_concentration_prior_v1_contract():
    """Equation 2: ego-motion concentration prior."""
    eq = build_ego_motion_concentration_prior_v1()
    assert eq.equation_id == "ego_motion_concentration_prior_v1"
    # Cite-chain to Atick-Redlich / Rao-Ballard
    assert "atick_redlich_1990" in eq.domain_of_validity["predictive_coding_lineage"]
    assert "rao_ballard_1999" in eq.domain_of_validity["predictive_coding_lineage"]
    assert "tishby_zaslavsky_2015" in eq.domain_of_validity["predictive_coding_lineage"]
    # Orphan-equation invariant
    assert len(eq.canonical_producers) >= 1
    assert len(eq.canonical_consumers) >= 1
    # Initial anchors empty
    assert eq.empirical_anchors == ()


def test_build_per_segnet_class_chroma_priors_v1_contract():
    """Equation 3: per-SegNet-class chroma priors."""
    eq = build_per_segnet_class_chroma_priors_v1()
    assert eq.equation_id == "per_segnet_class_chroma_priors_v1"
    # 5-class scorer pinned
    assert eq.domain_of_validity["scorer_class_count"] == [5]
    # Canonical class names declared
    assert "vehicle" in eq.domain_of_validity["canonical_class_names"]
    assert "lane_marking" in eq.domain_of_validity["canonical_class_names"]
    # Sister consumer per Catalog #354 exploit #5
    assert any(
        "per_segnet_class_chroma_consumer" in c for c in eq.canonical_consumers
    )
    # Orphan-equation invariant
    assert len(eq.canonical_producers) >= 1
    assert eq.empirical_anchors == ()


def test_build_all_domain_prior_equations_returns_3():
    """build_all returns 3 equations in canonical order."""
    eqs = build_all_domain_prior_equations()
    assert len(eqs) == 3
    assert [e.equation_id for e in eqs] == list(DOMAIN_PRIORS_EQUATION_IDS)


def test_register_domain_prior_canonical_equations_persists(tmp_path: Path):
    """register_domain_prior_canonical_equations writes to JSONL registry."""
    test_path = tmp_path / "domain_priors_test_registry.jsonl"
    test_lock = tmp_path / "domain_priors_test_registry.lock"
    eqs = register_domain_prior_canonical_equations(
        path=test_path,
        lock_path=test_lock,
        agent="test_domain_priors",
        subagent_id="test_subagent_id",
    )
    assert len(eqs) == 3
    # Persistence: JSONL exists + has 3+ rows (one per registration).
    assert test_path.is_file()
    rows = test_path.read_text().strip().splitlines()
    assert len(rows) == 3
    # Each row references a canonical equation_id.
    for row in rows:
        assert any(eq_id in row for eq_id in DOMAIN_PRIORS_EQUATION_IDS)


def test_register_then_query_by_id_roundtrip(tmp_path: Path):
    """register + get_equation_by_id returns the registered equation."""
    test_path = tmp_path / "roundtrip.jsonl"
    test_lock = tmp_path / "roundtrip.lock"
    register_domain_prior_canonical_equations(path=test_path, lock_path=test_lock)
    for eq_id in DOMAIN_PRIORS_EQUATION_IDS:
        eq = get_equation_by_id(eq_id, path=test_path)
        assert eq is not None
        assert eq.equation_id == eq_id


def test_register_then_query_by_consumer(tmp_path: Path):
    """query_equations_by_consumer surfaces our equations to known consumers."""
    test_path = tmp_path / "consumer_query.jsonl"
    test_lock = tmp_path / "consumer_query.lock"
    register_domain_prior_canonical_equations(path=test_path, lock_path=test_lock)
    # Catalog #354 sister consumer per Catalog #335 contract — must surface
    # the per_segnet_class_chroma_priors_v1 equation.
    matches = query_equations_by_consumer(
        "tac.cathedral_consumers.per_segnet_class_chroma_consumer",
        path=test_path,
    )
    assert any(eq.equation_id == "per_segnet_class_chroma_priors_v1" for eq in matches)


def test_register_then_query_by_producer(tmp_path: Path):
    """query_equations_by_producer surfaces equations by helper module path."""
    test_path = tmp_path / "producer_query.jsonl"
    test_lock = tmp_path / "producer_query.lock"
    register_domain_prior_canonical_equations(path=test_path, lock_path=test_lock)
    matches = query_equations_by_producer(
        "tac.domain_priors.per_frame_difficulty:build_per_frame_difficulty_from_per_pair_atlas",
        path=test_path,
    )
    assert any(eq.equation_id == "per_frame_difficulty_atlas_v1" for eq in matches)


def test_canonical_equations_carry_provenance():
    """Every equation carries PREDICTED Provenance per Catalog #323."""
    from tac.provenance.contract import Provenance, ProvenanceEvidenceGrade

    for eq in build_all_domain_prior_equations():
        assert isinstance(eq.provenance, Provenance)
        # Design-only equations are PREDICTED until paired CUDA+CPU anchor lands
        assert eq.provenance.evidence_grade == ProvenanceEvidenceGrade.PREDICTED
        assert eq.provenance.promotion_eligible is False
        assert eq.provenance.score_claim_valid is False


def test_canonical_equations_idempotent_re_registration(tmp_path: Path):
    """Re-running register appends additional 'registered' events (APPEND-ONLY)."""
    test_path = tmp_path / "idempotent.jsonl"
    test_lock = tmp_path / "idempotent.lock"
    register_domain_prior_canonical_equations(path=test_path, lock_path=test_lock)
    register_domain_prior_canonical_equations(path=test_path, lock_path=test_lock)
    rows = test_path.read_text().strip().splitlines()
    # 3 equations × 2 registrations = 6 rows
    assert len(rows) == 6
    # Latest equation by equation_id still queries correctly (latest-wins)
    for eq_id in DOMAIN_PRIORS_EQUATION_IDS:
        assert get_equation_by_id(eq_id, path=test_path) is not None


# ----------------------------------------------------------------------
# Catalog #185 / Catalog #176 META-meta sister regression guards
# ----------------------------------------------------------------------


def test_module_is_importable_no_circular():
    """Live regression: importing tac.domain_priors does not trigger circular imports."""
    import importlib
    import tac.domain_priors

    importlib.reload(tac.domain_priors)
    assert hasattr(tac.domain_priors, "build_all_domain_prior_equations")


def test_all_public_api_surface_pinned():
    """__all__ pins the canonical public surface."""
    import tac.domain_priors as dp

    expected_subset = {
        "PerFrameDifficultyAtlas",
        "EgoMotionConcentrationAtlas",
        "PerClassStatisticalPriors",
        "Comma2k19DashcamPriors",
        "DOMAIN_PRIORS_EQUATION_IDS",
        "register_domain_prior_canonical_equations",
    }
    assert expected_subset.issubset(set(dp.__all__))
