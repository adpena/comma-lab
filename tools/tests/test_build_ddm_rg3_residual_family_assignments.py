from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization.ddm_pf2_bucket_assignment import canonical_sha256
from tools.build_ddm_rg3_residual_family_assignments import (
    DEFAULT_BASE,
    DEFAULT_MARGIN,
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY,
    EXPECTED_FAMILY_COUNTS,
    RG3AssignmentError,
    build_assignment,
)
from tools.measure_ddm_ms6_receiver_support import _load_rg3_assignment


def test_real_rg2_residue_builds_exact_rg3_assignment() -> None:
    value = build_assignment(
        summary_path=DEFAULT_SUMMARY,
        base_path=DEFAULT_BASE,
        margin_path=DEFAULT_MARGIN,
    )
    payload = dict(value)
    claimed = payload.pop("assignment_content_sha256")
    assert claimed == canonical_sha256(payload)
    assert value["row_count"] == 36
    assert value["actuator_count"] == 62
    assert value["new_signed_probe_count"] == 124
    assert value["family_counts"] == EXPECTED_FAMILY_COUNTS
    assert all(
        row["score_units_per_byte_status"] == "OWED_NOT_ADMITTED"
        for row in value["rows"]
    )
    assert all(
        len(row["receiver_actuator_ids"])
        == (1 if "CLASS_BIRTH" in row["selected_coordinate_family"] else 2)
        for row in value["rows"]
    )


def test_assignment_fails_closed_on_wrong_margin_bytes(tmp_path: Path) -> None:
    margin = tmp_path / "margin.f16"
    margin.write_bytes(b"\0" * 32)
    with pytest.raises(RG3AssignmentError, match="Fisher-margin source custody"):
        build_assignment(
            summary_path=DEFAULT_SUMMARY,
            base_path=DEFAULT_BASE,
            margin_path=margin,
        )


def test_measurement_loader_accepts_only_the_sealed_rg3_vocabulary() -> None:
    value, actuator_ids = _load_rg3_assignment(DEFAULT_OUTPUT)
    assert value["row_count"] == 36
    assert len(actuator_ids) == 62
    assert len(set(actuator_ids)) == 62
