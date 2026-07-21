# SPDX-License-Identifier: MIT
"""Focused tests for the G2 real-seed input-domain audit."""

from __future__ import annotations

import pytest

from tools.measure_realization_g2_lattice import (
    SOURCE_CONTROL_STAGE_SCHEMA,
    RealizationAuditError,
    audit_prefix,
    summarize_source_control_prefix,
)


def _stage(pair: int) -> dict:
    return {
        "schema": "predict_project_pair_stage.v0",
        "pair_index": pair,
        "hard_oracle": {
            "schema": "predict_project_hard_oracle_pair.v0",
            "cell_exact": True,
            "pose_within_tube": True,
            "uint8_factor2_exact": False,
            "d_seg": 0.25,
            "d_pose": 0.0,
        },
    }


def test_audit_prefix_keeps_label_cell_exact_separate_from_rgb_lattice_exactness():
    constraints = [
        {"time": 0, "cell_id": 1, "stratum": "boundary_codim1"},
        {"time": 1, "cell_id": 0, "stratum": "critical_event"},
        {"time": 20, "cell_id": 4, "stratum": "cell_interior"},
    ]
    row = audit_prefix(prefix=16, stage_rows=[_stage(i) for i in range(16)], constraints=constraints)
    assert row["label_only_pair_count"] == 16
    assert row["rgb_projection_pair_count"] == 0
    assert row["uint8_factor2_exact_fraction"] is None
    assert row["declared_constraint_count"] == 2
    assert row["plane_level_cache_replay"]["cell_exact_pairs"] == 16
    assert row["status"] == "BLOCKED_INPUT_DOMAIN_LABEL_FIELD_IS_NOT_RGB_PLANE"


def test_audit_prefix_refuses_unaudited_rgb_fields_or_changed_settled_blocker():
    rows = [_stage(i) for i in range(16)]
    rows[0]["hard_oracle"]["projected_rgb_sha256"] = "f" * 64
    with pytest.raises(RealizationAuditError, match="unaudited RGB"):
        audit_prefix(prefix=16, stage_rows=rows, constraints=[])

    rows = [_stage(i) for i in range(16)]
    rows[0]["hard_oracle"]["uint8_factor2_exact"] = True
    with pytest.raises(RealizationAuditError, match="no longer matches"):
        audit_prefix(prefix=16, stage_rows=rows, constraints=[])


def _source_control_stage(pair: int, *, survives: bool) -> dict:
    return {
        "schema": SOURCE_CONTROL_STAGE_SCHEMA,
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "additional_seed_bytes": 1_179_648,
        "hard_oracle": {
            "d_seg_realized_vs_frozen_target": 0.001,
            "d_seg_description_vs_frozen_target": 0.343,
            "d_seg_realized_argmax_vs_description": 0.342,
            "realized_argmax_equals_description": False,
            "d_pose_realized_vs_frozen_target": 0.0001,
            "d_pose_realized_outside_declared_tube": 0.0,
            "pose_within_declared_tube": True,
        },
        "declared_write_survival": [
            {"class_id": 1, "stratum": "boundary_codim1", "survives": survives}
        ],
        "timings_seconds": {
            "source_cache_load": 0.1,
            "seed_cell_decode": 0.2,
            "source_plane_projection": 0.3,
            "lattice_double_decode": 0.4,
            "native_cpu_torch_hard_oracle": 0.5,
            "total": 1.5,
        },
    }


def test_source_plane_prefix_proves_lattice_but_refuses_zero_byte_semantic_promotion():
    rows = [_source_control_stage(pair, survives=pair % 2 == 0) for pair in range(16)]
    summary = summarize_source_control_prefix(16, rows)
    assert summary["uint8_factor2_exact_fraction"] == 1.0
    assert summary["double_decode_identical_pair_count"] == 16
    assert summary["semantic_cells_to_rgb_exact_pair_count"] == 0
    assert summary["additional_seed_bytes_total"] == 16 * 1_179_648
    assert summary["zero_added_seed_byte_target_met"] is False
    assert summary["by_class"] == [
        {
            "class_id": 1,
            "declared_writes": 16,
            "surviving_writes": 8,
            "dying_writes": 8,
            "survival_fraction": 0.5,
        }
    ]
    assert summary["status"] == "MEASURED_SOURCE_RGB_CONTROL_NOT_SEED_RECEIVER"


def test_source_plane_prefix_refuses_noncontiguous_or_noncanonical_ladder():
    rows = [_source_control_stage(pair, survives=True) for pair in range(16)]
    rows[5]["pair_index"] = 6
    with pytest.raises(RealizationAuditError, match="contiguous"):
        summarize_source_control_prefix(16, rows)
