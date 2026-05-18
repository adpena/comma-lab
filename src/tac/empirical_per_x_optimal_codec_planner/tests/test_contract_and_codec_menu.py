# SPDX-License-Identifier: MIT
"""Tests for PerXCodecAssignmentPlan contract + codec_menu canonical."""
from __future__ import annotations

import pytest

from tac.empirical_per_x_optimal_codec_planner import (
    CODEC_NAMES,
    PerXAssignmentRow,
    PerXCodecAssignmentPlan,
    PlannerError,
    X_GRANULARITY_VALUES,
    codec_bits_per_sample,
)
from tac.empirical_per_x_optimal_codec_planner.codec_menu import codec_bytes_for_n_samples


# ============================================================
# X_GRANULARITY_VALUES canonical set
# ============================================================

def test_x_granularity_values_canonical_11_kinds() -> None:
    """Canonical X granularities cover byte / bit / pixel / region / pair / frame / boundary / latent_index / channel / tensor / layer."""
    expected = {
        "byte", "bit", "pixel", "region", "pair", "frame", "boundary",
        "latent_index", "channel", "tensor", "layer",
    }
    assert X_GRANULARITY_VALUES == frozenset(expected)


# ============================================================
# CODEC_NAMES canonical set
# ============================================================

def test_codec_names_contains_canonical_bit_widths() -> None:
    """Canonical codec menu includes all the bit widths from the Fields-Medal design."""
    assert "fp16" in CODEC_NAMES
    assert "int8" in CODEC_NAMES
    assert "int6" in CODEC_NAMES
    assert "int4" in CODEC_NAMES
    # Variable-rate canonical entries
    assert "brotli" in CODEC_NAMES
    assert "magic_codec" in CODEC_NAMES


def test_codec_bits_per_sample_canonical() -> None:
    """Canonical bit-width lookup."""
    assert codec_bits_per_sample("fp16") == 16
    assert codec_bits_per_sample("int8") == 8
    assert codec_bits_per_sample("int6") == 6
    assert codec_bits_per_sample("int4") == 4
    assert codec_bits_per_sample("fp4") == 4
    # Variable-rate
    assert codec_bits_per_sample("brotli") is None
    assert codec_bits_per_sample("magic_codec") is None


def test_codec_bits_per_sample_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="not in canonical CODEC_NAMES"):
        codec_bits_per_sample("bogus_codec")


def test_codec_bytes_for_n_samples_canonical() -> None:
    """Per-byte → encoded bytes calculation."""
    # fp16: 16 bits/sample, 1 sample = 2 bytes
    assert codec_bytes_for_n_samples("fp16", 1) == 2
    assert codec_bytes_for_n_samples("fp16", 10) == 20
    # int8: 1 byte/sample
    assert codec_bytes_for_n_samples("int8", 100) == 100
    # int4: 4 bits/sample, 2 samples = 1 byte
    assert codec_bytes_for_n_samples("int4", 1) == 1  # (4+7)//8 = 1
    assert codec_bytes_for_n_samples("int4", 2) == 1  # (8+7)//8 = 1
    assert codec_bytes_for_n_samples("int4", 4) == 2  # (16+7)//8 = 2
    # int6: 6 bits/sample
    assert codec_bytes_for_n_samples("int6", 4) == 3  # (24+7)//8 = 3
    # Variable-rate: 1 byte/sample upper bound
    assert codec_bytes_for_n_samples("brotli", 100) == 100


# ============================================================
# PerXAssignmentRow contract
# ============================================================

def test_assignment_row_rejects_unknown_codec() -> None:
    with pytest.raises(PlannerError, match="not in canonical CODEC_NAMES"):
        PerXAssignmentRow(
            x_index=0,
            x_class="top_2pct",
            sensitivity_score=1.0,
            chosen_codec="bogus",
            chosen_codec_bits=8,
            predicted_score_delta=0.0,
            predicted_bytes_after_codec=1,
        )


def test_assignment_row_rejects_negative_index() -> None:
    with pytest.raises(PlannerError, match="x_index must be >= 0"):
        PerXAssignmentRow(
            x_index=-1,
            x_class="top_2pct",
            sensitivity_score=1.0,
            chosen_codec="fp16",
            chosen_codec_bits=16,
            predicted_score_delta=0.0,
            predicted_bytes_after_codec=2,
        )


def test_assignment_row_rejects_negative_bytes_after_codec() -> None:
    with pytest.raises(PlannerError, match="predicted_bytes_after_codec must be >= 0"):
        PerXAssignmentRow(
            x_index=0,
            x_class="top_2pct",
            sensitivity_score=1.0,
            chosen_codec="fp16",
            chosen_codec_bits=16,
            predicted_score_delta=0.0,
            predicted_bytes_after_codec=-1,
        )


# ============================================================
# PerXCodecAssignmentPlan contract
# ============================================================

def _valid_provenance() -> dict:
    return {
        "kind": "predicted_from_master_gradient",
        "source_artifact_path": "/path/to/grad.npy",
        "captured_at_utc": "2026-05-18T00:00:00+00:00",
        "score_claim": False,
        "evidence_grade": "predicted",
    }


def test_plan_rejects_unknown_granularity() -> None:
    with pytest.raises(PlannerError, match="not in"):
        PerXCodecAssignmentPlan(
            archive_sha256="sha",
            x_granularity="bogus",
            codec_menu=("fp16",),
            byte_budget=100,
            sensitivity_threshold_quantiles=(1.0,),
            assignments=(),
            total_predicted_score_delta=0.0,
            total_predicted_bytes=0,
            total_predicted_bytes_within_budget=True,
            operating_point={},
            measurement_axis="[predicted]",
            evidence_grade="predicted",
            provenance=_valid_provenance(),
            captured_at_utc="2026-05-18T00:00:00+00:00",
        )


def test_plan_rejects_unknown_codec_in_menu() -> None:
    with pytest.raises(PlannerError, match="contains unknown codec"):
        PerXCodecAssignmentPlan(
            archive_sha256="sha",
            x_granularity="byte",
            codec_menu=("fp16", "bogus_codec"),
            byte_budget=100,
            sensitivity_threshold_quantiles=(0.5, 1.0),
            assignments=(),
            total_predicted_score_delta=0.0,
            total_predicted_bytes=0,
            total_predicted_bytes_within_budget=True,
            operating_point={},
            measurement_axis="[predicted]",
            evidence_grade="predicted",
            provenance=_valid_provenance(),
            captured_at_utc="2026-05-18T00:00:00+00:00",
        )


def test_plan_rejects_non_predicted_evidence_grade() -> None:
    """Per Catalog #287 every newly-emitted plan must be 'predicted'."""
    with pytest.raises(PlannerError, match="must be 'predicted'"):
        PerXCodecAssignmentPlan(
            archive_sha256="sha",
            x_granularity="byte",
            codec_menu=("fp16",),
            byte_budget=100,
            sensitivity_threshold_quantiles=(1.0,),
            assignments=(),
            total_predicted_score_delta=0.0,
            total_predicted_bytes=0,
            total_predicted_bytes_within_budget=True,
            operating_point={},
            measurement_axis="[contest-CUDA]",
            evidence_grade="contest_cuda",  # FORBIDDEN at construction
            provenance=_valid_provenance(),
            captured_at_utc="2026-05-18T00:00:00+00:00",
        )


def test_plan_rejects_provenance_missing_required_fields() -> None:
    """Per Catalog #323 provenance must contain kind + source_artifact_path + captured_at_utc."""
    with pytest.raises(PlannerError, match="must contain"):
        PerXCodecAssignmentPlan(
            archive_sha256="sha",
            x_granularity="byte",
            codec_menu=("fp16",),
            byte_budget=100,
            sensitivity_threshold_quantiles=(1.0,),
            assignments=(),
            total_predicted_score_delta=0.0,
            total_predicted_bytes=0,
            total_predicted_bytes_within_budget=True,
            operating_point={},
            measurement_axis="[predicted]",
            evidence_grade="predicted",
            provenance={"incomplete": True},  # missing required fields
            captured_at_utc="2026-05-18T00:00:00+00:00",
        )


def test_plan_accepts_canonical_byte_plan() -> None:
    plan = PerXCodecAssignmentPlan(
        archive_sha256="sha",
        x_granularity="byte",
        codec_menu=("fp16", "int4"),
        byte_budget=100,
        sensitivity_threshold_quantiles=(0.5, 1.0),
        assignments=(
            PerXAssignmentRow(
                x_index=0, x_class="top_2pct", sensitivity_score=1.0,
                chosen_codec="fp16", chosen_codec_bits=16,
                predicted_score_delta=-0.001, predicted_bytes_after_codec=2,
            ),
            PerXAssignmentRow(
                x_index=1, x_class="tail", sensitivity_score=0.1,
                chosen_codec="int4", chosen_codec_bits=4,
                predicted_score_delta=-0.0005, predicted_bytes_after_codec=1,
            ),
        ),
        total_predicted_score_delta=-0.0015,
        total_predicted_bytes=3,
        total_predicted_bytes_within_budget=True,
        operating_point={"d_seg": 1e-3, "d_pose": 1e-4, "rate": 5e-3, "score": 0.2},
        measurement_axis="[predicted]",
        evidence_grade="predicted",
        provenance=_valid_provenance(),
        captured_at_utc="2026-05-18T00:00:00+00:00",
    )
    assert plan.x_granularity == "byte"
    assert len(plan.assignments) == 2

    # Class summary
    summary = plan.class_summary()
    assert "top_2pct" in summary
    assert summary["top_2pct"]["n_bytes"] == 1
    assert summary["tail"]["codec"] == "int4"

    # Serialization
    as_dict = plan.as_dict()
    assert as_dict["assignments_count"] == 2
    assert as_dict["x_granularity"] == "byte"
    assert as_dict["evidence_grade"] == "predicted"
