# SPDX-License-Identifier: MIT
"""Focused tests for the G2 real-seed input-domain audit."""

from __future__ import annotations

import numpy as np
import pytest

from tools.measure_realization_g2_lattice import (
    INTERIOR_STAGE_SCHEMA,
    SOURCE_CONTROL_STAGE_SCHEMA,
    RealizationAuditError,
    apply_dying_write_exceptions,
    audit_prefix,
    encode_dying_write_exceptions,
    interior_rgb_plane,
    summarize_interior_prefix,
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


def test_zero_byte_interior_rungs_are_deterministic_and_behaviorally_distinct():
    yy, xx = np.indices((384, 512))
    cells = ((yy // 37 + xx // 41) % 5).astype(np.uint8)
    r1 = interior_rgb_plane(cells, "R1_FIXED_MAGNITUDE")
    r2 = interior_rgb_plane(cells, "R2_MAX_MARGIN")
    r3a = interior_rgb_plane(cells, "R3_HOPFIELD_MEMORY_PROX")
    r3b = interior_rgb_plane(cells, "R3_HOPFIELD_MEMORY_PROX")
    assert r1.shape == (384, 512, 3) and r1.dtype == np.uint8
    assert not np.array_equal(r1, r2)
    assert not np.array_equal(r2, r3a)
    assert not np.array_equal(r1, r3a)
    assert np.array_equal(r3a, r3b)


def test_r4_exception_stream_contains_only_dying_ordinals_and_roundtrips():
    constraints = [
        {"y": 3, "x": 4, "cell_id": 1, "stratum": "boundary_codim1"},
        {"y": 7, "x": 8, "cell_id": 0, "stratum": "critical_event"},
        {"y": 9, "x": 10, "cell_id": 1, "stratum": "boundary_codim1"},
    ]
    survival = [{"survives": False}, {"survives": True}, {"survives": False}]
    source = np.zeros((384, 512, 3), dtype=np.uint8)
    source[3, 4] = (11, 12, 13)
    source[9, 10] = (21, 22, 23)
    payload = encode_dying_write_exceptions(constraints, survival, source)
    assert len(payload) == 2 + 2 * 5
    base = np.full_like(source, 127)
    decoded, ordinals = apply_dying_write_exceptions(base, constraints, payload)
    assert ordinals == [0, 2]
    assert decoded[3, 4].tolist() == [11, 12, 13]
    assert decoded[9, 10].tolist() == [21, 22, 23]
    assert decoded[7, 8].tolist() == [127, 127, 127]


def _interior_stage(pair: int, *, survives: bool, rung: str = "R2_MAX_MARGIN") -> dict:
    return {
        "schema": INTERIOR_STAGE_SCHEMA,
        "rung_id": rung,
        "pair_index": pair,
        "uint8_factor2_exact": True,
        "double_decode_identical": True,
        "additional_seed_bytes": 0,
        "hard_oracle": {
            "d_seg_realized_vs_frozen_target": 0.2,
            "d_seg_description_vs_frozen_target": 0.343,
            "d_seg_realized_argmax_vs_description": 0.4,
            "realized_argmax_equals_description": False,
            "all_declared_writes_survive": survives,
            "d_pose_realized_vs_frozen_target": 0.01,
            "d_pose_realized_outside_declared_tube": 0.02,
            "pose_within_declared_tube": False,
        },
        "declared_write_survival": [
            {
                "ordinal": 0,
                "class_id": 1,
                "stratum": "boundary_codim1",
                "survives": survives,
                "target_logit_margin": 0.5 if survives else -0.5,
                "margin_bucket": "positive_le_1" if survives else "nonpositive",
            }
        ],
        "exception_stream": {
            "record_count": 0,
            "records": [],
        },
        "timings_seconds": {
            "source_cache_load_shared": 0.1,
            "seed_cell_decode_shared": 0.1,
            "rgb_plane_decode": 0.1,
            "lattice_double_decode": 0.2,
            "native_cpu_torch_hard_oracle": 0.3,
            "rung_total": 0.7,
            "pair_total_to_stage": 0.7,
        },
    }


def test_interior_prefix_reports_pair_and_write_exactness_separately():
    rows = [_interior_stage(pair, survives=pair % 2 == 0) for pair in range(16)]
    summary = summarize_interior_prefix(16, rows, rung_id="R2_MAX_MARGIN")
    assert summary["uint8_factor2_exact_fraction"] == 1.0
    assert summary["semantic_cells_to_rgb_exact_pair_count"] == 0
    assert summary["all_declared_writes_survive_pair_count"] == 8
    assert summary["declared_write_survival_fraction"] == 0.5
    assert summary["by_margin_bucket"] == [
        {
            "margin_bucket": "nonpositive",
            "declared_writes": 8,
            "surviving_writes": 0,
            "dying_writes": 8,
            "survival_fraction": 0.0,
        },
        {
            "margin_bucket": "positive_le_1",
            "declared_writes": 8,
            "surviving_writes": 8,
            "dying_writes": 0,
            "survival_fraction": 1.0,
        },
    ]
