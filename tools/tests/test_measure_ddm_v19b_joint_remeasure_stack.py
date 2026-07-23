# SPDX-License-Identifier: MIT
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.measure_ddm_v19b_joint_remeasure_stack import (
    FIRST_MOVE_ID,
    DDMV19BJointRemeasureStackConfigV1,
    _accepted_inventory,
    _apply_move,
    _control_with_errors,
    _g1_track_translation_bounds,
    _initial_stack,
    _nonadditivity,
    _realized_post_state,
)


def _config(**updates: object) -> DDMV19BJointRemeasureStackConfigV1:
    values: dict[str, object] = {
        "run_id": "fixture_ddm_v19b",
        "v19_config_path": "v19.json",
        "v19_config_sha256": "1" * 64,
        "v19_receipt_path": "v19_receipt.json",
        "v19_receipt_sha256": "2" * 64,
    }
    values.update(updates)
    return DDMV19BJointRemeasureStackConfigV1(**values)


def _problem() -> dict[str, object]:
    return {
        "initial_template_values_u8": [[[[254, 1, 2]]]],
        "initial_compensation_rgb_i8": [[126, -126, 0]],
        "initial_phases": {},
    }


def _ctx() -> dict[str, object]:
    candidates = []
    for label, template, compensation in (
        (FIRST_MOVE_ID, [[[[255, 0, 4]]]], [[127, -127, 2]]),
        ("post_r8", [[[[255, 0, 10]]]], [[127, -127, 8]]),
    ):
        candidates.append(
            {
                "label": label,
                "state": {
                    "template_values_u8": template,
                    "compensation_rgb_i8": compensation,
                },
            }
        )
    return {"v17_receipt": {"iterations": [{"solve_candidates": candidates}]}}


def _move(candidate_id: str, family: str) -> dict[str, object]:
    return {"candidate_id": candidate_id, "family": family}


def test_v19b_config_seals_false_authority_and_batch_geometry() -> None:
    config = _config()
    assert config.research_only is True
    assert config.execution_allowed is False
    assert config.score_claim is False
    assert config.scorer_batch_size == 16
    with pytest.raises(ValueError):
        _config(scorer_batch_size=8)


def test_stack_merge_is_additive_then_clipped_to_wire_domains() -> None:
    problem = _problem()
    ctx = _ctx()
    stack = _initial_stack(problem)
    stack = _apply_move(
        stack,
        _move(FIRST_MOVE_ID, "v17_rejected_class_neighborhood"),
        problem=problem,
        ctx=ctx,
    )
    stack = _apply_move(
        stack,
        _move("post_r8", "v17_rejected_class_neighborhood"),
        problem=problem,
        ctx=ctx,
    )
    realized = _realized_post_state(stack, problem)
    assert np.asarray(realized["template_values_u8"]).tolist() == [[[[255, 0, 12]]]]
    assert np.asarray(realized["compensation_rgb_i8"]).tolist() == [[127, -127, 10]]


def test_grammar_and_preuint8_moves_compose_in_separate_stages() -> None:
    problem = _problem()
    ctx = _ctx()
    stack = _initial_stack(problem)
    for move in (
        _move("worldsheet_joint_active_x_+1", "grammar_native"),
        _move("worldsheet_joint_active_y_-1", "grammar_native"),
        _move("preuint8_405_scale_q8_128", "preuint8_camera_q8"),
        _move("preuint8_405_scale_q8_256", "preuint8_camera_q8"),
    ):
        stack = _apply_move(stack, move, problem=problem, ctx=ctx)
    assert stack.grammar_move_ids == (
        "worldsheet_joint_active_x_+1",
        "worldsheet_joint_active_y_-1",
    )
    assert stack.preuint8_scale_q8 == 384


def test_nonadditivity_quantifies_degradation_and_amplification() -> None:
    degraded = _nonadditivity(
        {"joint_delta": -0.01},
        {"joint_delta": -0.004},
    )
    assert degraded["verdict"] == "SURVIVED_PARTIALLY_DEGRADED"
    assert degraded["survival_fraction"] == pytest.approx(0.4)
    rejected = _nonadditivity(
        {"joint_delta": -0.01},
        {"joint_delta": 0.001},
    )
    assert rejected["verdict"] == "DEGRADED_TO_REJECTION"
    amplified = _nonadditivity(
        {"joint_delta": -0.01},
        {"joint_delta": -0.02},
    )
    assert amplified["verdict"] == "SURVIVED_AND_AMPLIFIED"


def test_inventory_forces_405_then_orders_remaining_by_measured_joint_delta() -> None:
    rows = []
    for index in range(10):
        candidate_id = FIRST_MOVE_ID if index == 5 else f"candidate_{index:02d}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "archive": {"path": "x", "bytes": 1, "sha256": "0" * 64},
                "pure_priced_delta": {
                    "accepted": True,
                    "joint_delta": -float(index + 1),
                },
            }
        )
    receipt = {"proposal_sources": {"fixture": rows}}
    inventory = _accepted_inventory(receipt)
    assert inventory[0]["candidate_id"] == FIRST_MOVE_ID
    assert inventory[1]["candidate_id"] == "candidate_09"


def test_direct_g1_bounds_use_lift_records_without_joint_wrapper() -> None:
    template = SimpleNamespace(
        template_ref="shape",
        relative_vertices_xy=((-2, -3), (4, 5)),
    )
    knot = SimpleNamespace(
        template_ref="shape",
        center_x=10,
        center_y=20,
    )
    track = SimpleNamespace(knot_indices=(0,))
    lift = SimpleNamespace(templates=(template,), knots=(knot,), tracks=(track,))
    assert _g1_track_translation_bounds(lift, 0) == ((-8, 497), (-17, 358))


def test_v15_control_binder_maps_sealed_per_stratum_schema() -> None:
    receipt = {
        "producer_custody": [],
        "solved_template_ladder": [
            {
                "candidate": "v15_solved_templates",
                "archive_bytes": 100,
                "archive_sha256": "a" * 64,
                "d_seg": "0.1",
                "d_pose": "1.0",
                "errors": 10,
                "sites": 100,
                "per_stratum": {
                    "Lane": {"errors": 3, "sites": 10},
                    "Movable": {"errors": 4, "sites": 20},
                },
            }
        ],
    }
    control = _control_with_errors(receipt)
    assert control["per_role"] == {
        "Lane": {"errors": 3, "sites": 10},
        "Movable": {"errors": 4, "sites": 20},
    }
