# SPDX-License-Identifier: MIT
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from tools.measure_ddm_dr1_realization_race_coding_gain import (
    AXIS,
    _nearest_integer_half_up,
    _nonredundancy_audit,
    _pad_zip_comment,
    _rate_audit,
    _select_dv1_rows,
    _undrivable_row,
)


@pytest.mark.parametrize(
    ("numerator", "expected"),
    [(0, 0), (127, 0), (128, 1), (255, 1), (384, 2), (-127, 0), (-128, -1), (-384, -2)],
)
def test_nearest_integer_half_up_is_signed_and_deterministic(
    numerator: int, expected: int
) -> None:
    assert _nearest_integer_half_up(numerator) == expected


def test_rate_audit_contains_all_three_binding_clauses() -> None:
    row = _rate_audit(
        descriptive_form="form",
        compact_dof="dof",
        coder_gain="gain",
        visibility="visible",
        tolerance="priced",
        admissible=True,
    )
    assert row["scorer_visibility"] == "visible"
    assert row["sensitivity_priced_tolerance"] == "priced"
    assert row["three_layer_decomposition"] == {
        "descriptive_form": "form",
        "inherently_compact_dof": "dof",
        "coder_gain": "gain",
    }
    assert row["composed_candidate_admissible"] is True


def test_nonredundancy_audit_contains_binding_clause_four() -> None:
    row = _nonredundancy_audit(
        owner="one owner",
        conditional_coding="B given A",
        pairwise_measurement={"redundancy_bytes": 0},
        dimension_home="clip",
        correction_delta_rule="delta only",
        admissible=False,
    )
    assert row["single_owner_fact_rule"] == "one owner"
    assert row["pairwise_redundancy_measurement"] == {"redundancy_bytes": 0}
    assert row["dimension_home"] == "clip"
    assert row["corrections_are_deltas"] == "delta only"
    assert row["composed_candidate_admissible"] is False


def test_zip_comment_padding_is_exact_and_preserves_members() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("member", b"payload")
    before = buffer.getvalue()
    after = _pad_zip_comment(before, len(before) + 3)
    assert len(after) == len(before) + 3
    with zipfile.ZipFile(io.BytesIO(after)) as archive:
        assert archive.read("member") == b"payload"
        assert archive.comment == b"\0\0\0"


def test_select_dv1_rows_fails_closed_on_ambiguity() -> None:
    primitive = {
        "primitive_id": "persistent_level_set_ground_partition",
        "measurement": {},
    }
    joint = {"candidate_id": "persistent_plus_events"}
    assert _select_dv1_rows([primitive, joint]) == (primitive, joint)
    with pytest.raises(Exception, match="not unique"):
        _select_dv1_rows([primitive, primitive, joint])


def test_undrivable_row_preserves_measurement_and_narrow_scope() -> None:
    source = type("Source", (), {"path": "ledger.jsonl", "sha256": "a" * 64})()
    measurement = {
        "per_stratum_errors_described": {"Undrivable": 80},
        "stationarity_of_described_errors": {
            "Undrivable": {"STATIC_IN_IMAGE": 72}
        },
        "per_stratum_described_fraction": {"Undrivable": 0.4},
        "net_errors_closed": 50,
        "counted_bytes": 10,
    }
    row = _undrivable_row(
        {"measurement": measurement},
        scope="standalone",
        source=source,
    )
    assert row["described_fraction"] == 0.4
    assert row["static_in_image_fraction_of_described"] == 0.9
    assert row["net_errors_closed_all_strata"] == 50
    assert row["evidence_axis"] == AXIS
    assert row["score_claim"] is False
    assert row["verdict_scope"].startswith("INSTANCE:")
    assert "first-rung" in row
    assert json.dumps(row, sort_keys=True)


def test_numpy_import_keeps_test_environment_honest() -> None:
    # The measurement module depends on exact integer array semantics.
    assert np.dtype(np.int16).itemsize == 2
