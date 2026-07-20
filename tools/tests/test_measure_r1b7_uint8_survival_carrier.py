from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tools.measure_r1b7_uint8_survival_carrier import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    STAGE_BUCKETS,
    TORCH_RESIZE_PARITY_TOLERANCE,
    _classify_stage,
    _exact_block_projection,
    _integer_block_proposals,
    _is_new_hard_crossing,
    _non_target_rival,
    _site_writes,
    _validate_histogram,
)


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=CAMERA_HEIGHT,
        camera_w=CAMERA_WIDTH,
        scorer_h=SCORER_HEIGHT,
        scorer_w=SCORER_WIDTH,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "camera_changed": False,
                "resize_max_abs": 1.0,
                "stem_max_abs": 1.0,
                "survived": True,
                "rival_changed": False,
                "collateral_count": 0,
            },
            "killed_at_uint8",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.0,
                "stem_max_abs": 1.0,
                "survived": True,
                "rival_changed": False,
                "collateral_count": 0,
            },
            "killed_at_resize_dilution",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.5,
                "stem_max_abs": 0.0,
                "survived": True,
                "rival_changed": False,
                "collateral_count": 0,
            },
            "killed_at_stem",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.5,
                "stem_max_abs": 0.5,
                "survived": False,
                "rival_changed": False,
                "collateral_count": 0,
            },
            "killed_at_head_same_rival",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.5,
                "stem_max_abs": 0.5,
                "survived": False,
                "rival_changed": True,
                "collateral_count": 0,
            },
            "killed_at_head_wrong_rival",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.5,
                "stem_max_abs": 0.5,
                "survived": True,
                "rival_changed": False,
                "collateral_count": 1,
            },
            "survived_but_collateral",
        ),
        (
            {
                "camera_changed": True,
                "resize_max_abs": 0.5,
                "stem_max_abs": 0.5,
                "survived": True,
                "rival_changed": False,
                "collateral_count": 0,
            },
            "survived_clean",
        ),
    ],
)
def test_stage_classifier_is_ordered_and_mutually_exclusive(kwargs: dict[str, object], expected: str) -> None:
    assert _classify_stage(**kwargs) == expected


def test_stage_histogram_is_exhaustive() -> None:
    histogram = dict.fromkeys(STAGE_BUCKETS, 0)
    histogram["killed_at_head_same_rival"] = 497
    histogram["survived_clean"] = 1
    _validate_histogram(histogram, 498)

    histogram["survived_clean"] = 0
    with pytest.raises(Exception, match="total"):
        _validate_histogram(histogram, 498)


def test_exact_projection_and_changed_only_replay_agree() -> None:
    operator = _operator()
    row, col = 189, 323
    frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 127, dtype=np.uint8)
    baseline_numerator, denominator = _exact_block_projection(operator, frame, row, col)
    rs = operator.row_supports[row]
    cs = operator.col_supports[col]
    block = frame[np.ix_(rs.indices, cs.indices, range(3))].copy()
    block[0, 0, 0] += 1
    changed = frame.copy()
    changed[np.ix_(rs.indices, cs.indices, range(3))] = block
    changed_numerator, changed_denominator = _exact_block_projection(operator, changed, row, col)
    writes = _site_writes(
        pair=14,
        operator=operator,
        row=row,
        col=col,
        block=block,
        reference=frame[np.ix_(rs.indices, cs.indices, range(3))],
    )

    assert changed_denominator == denominator
    assert changed_numerator[0] > baseline_numerator[0]
    assert np.array_equal(changed_numerator[1:], baseline_numerator[1:])
    assert len(writes) == 1
    assert writes[0].value == 128


def test_integer_proposals_are_exact_nonzero_uint8_lattice_moves() -> None:
    operator = _operator()
    frame = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 127, dtype=np.uint8)
    fisher_row = [
        14,
        189,
        323,
        2_849_603,
        1,
        0,
        0,
        4,
        9.72747802734375e-05,
        0.49999999881720214,
        3.953453779624277,
        2.4605012653690913e-05,
        True,
        1.089179277420044,
        [-0.5396262407302856, -0.8415757417678833, -0.02353091537952423],
    ]
    proposals = _integer_block_proposals(
        operator=operator,
        baseline_frame=frame,
        fisher_row=fisher_row,
        site_offset=0,
        multipliers=(1, 2, 4),
    )

    assert proposals
    for proposal in proposals:
        assert proposal.changed_camera_bytes > 0
        assert proposal.camera_l1 > 0
        assert any(value != 0.0 for value in proposal.projected_rgb_delta)
        modified = frame.copy()
        rs = operator.row_supports[189]
        cs = operator.col_supports[323]
        modified[np.ix_(rs.indices, cs.indices, range(3))] = proposal.block
        base_numerator, denominator = _exact_block_projection(operator, frame, 189, 323)
        proposal_numerator, proposal_denominator = _exact_block_projection(operator, modified, 189, 323)
        assert proposal_denominator == denominator
        assert np.allclose(
            (proposal_numerator - base_numerator) / denominator,
            proposal.projected_rgb_delta,
            atol=0.0,
            rtol=0.0,
        )


def test_non_target_rival_excludes_target_class() -> None:
    logits = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0])
    assert _non_target_rival(logits, 0) == 1
    assert _non_target_rival(logits, 1) == 0


def test_new_hard_crossing_rejects_already_correct_baseline() -> None:
    assert not _is_new_hard_crossing(
        baseline_predicted_class=0,
        proposal_predicted_class=0,
        target_class=0,
        baseline_margin=0.2,
        proposal_margin=0.3,
        margin_gate=0.0,
    )


def test_new_hard_crossing_requires_wrong_to_target_transition() -> None:
    assert _is_new_hard_crossing(
        baseline_predicted_class=1,
        proposal_predicted_class=0,
        target_class=0,
        baseline_margin=-0.2,
        proposal_margin=0.1,
        margin_gate=0.0,
    )
    assert not _is_new_hard_crossing(
        baseline_predicted_class=1,
        proposal_predicted_class=2,
        target_class=0,
        baseline_margin=-0.2,
        proposal_margin=-0.1,
        margin_gate=0.0,
    )


def test_torch_resize_parity_bound_is_power_of_two_sub_byte() -> None:
    measured_sealed_population_max = 0.003428141276041685
    assert TORCH_RESIZE_PARITY_TOLERANCE == 1.0 / 256.0
    assert measured_sealed_population_max < TORCH_RESIZE_PARITY_TOLERANCE
