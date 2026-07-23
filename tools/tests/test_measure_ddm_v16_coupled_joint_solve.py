# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.optimization.direct_description_carrier_compose import (
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
)
from tools.measure_ddm_v16_coupled_joint_solve import (
    TARGET_ROLES,
    _bank_values,
    _byte_diff,
    _expanded_bank,
    _measurement_key,
    _program,
)


def test_target_roles_use_canonical_comma10k_class_order() -> None:
    assert TARGET_ROLES == {"Movable": 3, "Lane": 1}


def _bank() -> ScorerSolvedTemplateBankV1:
    return ScorerSolvedTemplateBankV1(
        (
            RowBandScorerTemplateV1("Lane", "inner_boundary", 0, 384, 1, 1, bytes((4, 5, 6))),
            RowBandScorerTemplateV1("Movable", "fill", 0, 384, 1, 1, bytes((1, 2, 3))),
        )
    )


def test_expanded_bank_is_uniform_2x2() -> None:
    expanded = _expanded_bank(_bank())
    values = _bank_values(expanded)
    assert values.shape == (2, 2, 2, 3)
    assert np.array_equal(values[1], np.tile(np.array([1, 2, 3], dtype=np.uint8), (2, 2, 1)))


def test_program_omits_zero_compensation_but_counts_placements() -> None:
    program = _program(
        (9,),
        2,
        {"9:0": (1, 0), "9:1": (0, 1)},
        (
            {"source_pair_id": 9, "frame_index": 1, "camera_y": 10, "camera_x": 11},
            {"source_pair_id": 9, "frame_index": 1, "camera_y": 12, "camera_x": 13},
        ),
        np.array([[0, 0, 0], [1, -2, 3]], dtype=np.int16),
    )
    assert len(program.placements) == 2
    assert len(program.compensations) == 1
    assert program.compensations[0].delta_rgb == (1, -2, 3)


def test_byte_diff_and_constraint_key_are_deterministic() -> None:
    assert _byte_diff(b"abc", b"axcd")["changed_positions_in_common_prefix"] == 1
    measurement = {
        "constraints": {
            "target": {"violations": 2, "debt": "1.25"},
            "protected": {"violations": 0},
            "pose_upper": {"violations": 1},
            "pose_lower": {"violations": 0},
        },
        "advisory_score_formula_value": "3.5",
    }
    assert _measurement_key(measurement) == (0, 1, 2, 1.25, 3.5)
