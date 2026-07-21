# SPDX-License-Identifier: MIT
"""Focused tests for the G2 real-seed input-domain audit."""

from __future__ import annotations

import pytest

from tools.measure_realization_g2_lattice import RealizationAuditError, audit_prefix


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
