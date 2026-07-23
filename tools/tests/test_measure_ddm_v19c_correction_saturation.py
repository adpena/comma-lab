# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import pytest

import tools.measure_ddm_v19c_correction_saturation as v19c
from tools.measure_ddm_v19c_correction_saturation import (
    FAMILIES,
    DDMV19CCorrectionSaturationConfigV1,
    SaturationState,
    _apply_proposal,
    _candidate_n600_batches,
    _family_summary,
    _interleave,
    _proposal_at,
    _restore_n600_decisions,
    _scope_scale,
)


def _config(**updates: object) -> DDMV19CCorrectionSaturationConfigV1:
    values: dict[str, object] = {
        "run_id": "fixture_ddm_v19c",
        "v19b_config_path": "v19b.json",
        "v19b_config_sha256": "1" * 64,
        "v19b_receipt_path": "v19b_receipt.json",
        "v19b_receipt_sha256": "2" * 64,
    }
    values.update(updates)
    return DDMV19CCorrectionSaturationConfigV1(**values)


def _state() -> SaturationState:
    return SaturationState(
        template_delta=np.zeros((2, 1, 1, 1), dtype=np.int16),
        compensation_delta=np.zeros((2, 1), dtype=np.int16),
        base_grammar_move_ids=("worldsheet_joint_active_x_+1",),
        base_preuint8_scale_q8=576,
        move_ids=tuple(f"v19b_{index}" for index in range(10)),
    )


def _problem() -> dict[str, object]:
    return {
        "initial_template_values_u8": [[[[10]]], [[[20]]]],
        "initial_compensation_rgb_i8": [[1], [2]],
    }


def _ctx() -> dict[str, object]:
    return {
        "v17_receipt": {
            "iterations": [
                {
                    "solve_candidates": [
                        {
                            "label": "row",
                            "state": {
                                "template_values_u8": [[[[12]]], [[[19]]]],
                                "compensation_rgb_i8": [[3], [1]],
                            },
                        }
                    ]
                }
            ]
        }
    }


def test_v19c_config_seals_false_authority_budget_and_stop() -> None:
    config = _config()
    assert config.score_claim is False
    assert config.execution_allowed is False
    assert config.correction_budget_bytes == 200_000
    assert config.consecutive_failure_limit == 64
    assert config.scorer_batch_size == 16
    with pytest.raises(ValueError):
        _config(q8_all_scales=(-31, 32))


def test_family_interleave_exposes_all_families_before_recursion() -> None:
    groups = [[{"base_id": family, "family": family}] for family in FAMILIES]
    inventory = _interleave(groups)
    assert [row["family"] for row in inventory] == list(FAMILIES)
    proposal = _proposal_at(inventory, len(inventory))
    assert proposal["cycle"] == 1
    assert proposal["proposal_index"] == len(inventory)
    assert proposal["candidate_id"].startswith("cycle_001_")


def test_q8_scope_sum_distinguishes_stage_kind_and_pair() -> None:
    directives = (
        ("all", 32),
        ("templates", 64),
        ("sparse", -16),
        ("pair:447", 128),
        ("pair:53", 256),
    )
    assert _scope_scale(directives, kind="templates", pair_id=447) == 224
    assert _scope_scale(directives, kind="sparse", pair_id=447) == 144
    assert _scope_scale(directives, kind="templates", pair_id=53) == 352


def test_rowband_direction_is_additive_and_q8_is_stage_separate() -> None:
    state = _state()
    row = {
        "candidate_id": "cycle_000_row",
        "family": "inverse_solved_rowband",
        "rowband_candidate_id": "row",
        "scale": 1,
    }
    after = _apply_proposal(state, row, problem=_problem(), ctx=_ctx())
    assert after.template_delta.tolist() == [[[[2]]], [[[-1]]]]
    assert after.compensation_delta.tolist() == [[2], [-1]]
    q8 = {
        "candidate_id": "cycle_000_q8",
        "family": "preuint8_q8_region",
        "scope": "pair:447",
        "scale_q8": -64,
    }
    after_q8 = _apply_proposal(after, q8, problem=_problem(), ctx=_ctx())
    assert after_q8.template_delta.tolist() == after.template_delta.tolist()
    assert after_q8.q8_directives == (("pair:447", -64),)


def test_track_events_accumulate_and_template_swap_is_stateful() -> None:
    state = _state()
    event = {
        "candidate_id": "cycle_000_track",
        "family": "worldsheet_track_event",
        "track_indices": [7],
        "axis": "y",
        "sign": -1,
    }
    state = _apply_proposal(state, event, problem=_problem(), ctx=_ctx())
    state = _apply_proposal(state, event, problem=_problem(), ctx=_ctx())
    assert state.track_translations == ((7, 0, -2),)
    swap = {
        "candidate_id": "cycle_000_swap",
        "family": "scorer_template_swap",
        "left": 0,
        "right": 1,
    }
    state = _apply_proposal(state, swap, problem=_problem(), ctx=_ctx())
    assert state.template_delta.tolist() == [[[[10]]], [[[-10]]]]


def test_resume_proposal_identity_is_cycle_stable() -> None:
    inventory = [{"base_id": f"candidate_{index}", "family": FAMILIES[index]} for index in range(len(FAMILIES))]
    first = [_proposal_at(inventory, index) for index in range(13)]
    replay = [_proposal_at(inventory, index) for index in range(13)]
    assert [row["candidate_id"] for row in replay] == [row["candidate_id"] for row in first]


def test_family_summary_counts_n600_compile_infeasibility() -> None:
    rows = [
        {
            "proposal": {"family": family},
            "accepted": False,
            "disposition": (
                "INFEASIBLE_N600_COMPILE"
                if family == "worldsheet_track_event"
                else "REJECTED_NONNEGATIVE_N600_JOINT_DELTA"
            ),
        }
        for family in FAMILIES
    ]
    summary = _family_summary(rows)
    assert summary["worldsheet_track_event"]["compile_infeasible"] == 1
    assert summary["preuint8_q8_region"]["compile_infeasible"] == 0


def test_n600_resume_restores_only_admitted_state(tmp_path) -> None:
    config_hash = "a" * 64
    inventory = [
        {
            "base_id": "q8_all_32",
            "family": "preuint8_q8_region",
            "scope": "all",
            "scale_q8": 32,
        },
        {
            "base_id": "q8_sparse_64",
            "family": "preuint8_q8_region",
            "scope": "sparse",
            "scale_q8": 64,
        },
    ]
    for index, accepted in enumerate((True, False)):
        proposal = _proposal_at(inventory, index)
        payload = {
            "typed_config_sha256": config_hash,
            "dev_admission_index": index,
            "proposal": proposal,
            "accepted": accepted,
            "current_after": {
                "archive_bytes": 100 + index,
                "archive_sha256": str(index) * 64,
            },
        }
        (tmp_path / f"candidate_{index:04d}.json").write_text(json.dumps(payload))
    rows, state, current = _restore_n600_decisions(
        decision_root=tmp_path,
        config_hash=config_hash,
        proposal_indices=[0, 1],
        inventory=inventory,
        state=_state(),
        problem=_problem(),
        ctx=_ctx(),
    )
    assert len(rows) == 2
    assert state.q8_directives == (("all", 32),)
    assert current == {"archive_bytes": 101, "archive_sha256": "1" * 64}


def test_n600_exact_camera_identity_reuses_scorer_row(tmp_path, monkeypatch) -> None:
    camera = np.zeros((2, 2, 3, 4, 3), dtype=np.uint8)

    class Receiver:
        def render_camera_pairs(self, pair_ids):
            assert tuple(pair_ids) == (0, 1)
            return camera

    def fail_forward(*_args, **_kwargs):
        raise AssertionError("byte-identical camera batch must not invoke scorer")

    monkeypatch.setattr(v19c, "_forward", fail_forward)
    monkeypatch.setattr(
        v19c,
        "_candidate_receiver_and_support",
        lambda **_kwargs: (
            Receiver(),
            set(),
            {"method": "fixture", "source_pair_count": 0},
        ),
    )
    monkeypatch.setattr(
        v19c,
        "_preuint8_byte_rows_and_custody",
        lambda _archive, _receiver: ([], {"fixture": True}),
    )
    current_batch = {
        "errors": 2,
        "sites": 12,
        "pose_squared_error_sum": "6.000000000000",
        "pose_coordinates": 6,
        "class_rows": {
            "Road": {"errors": 1, "sites": 3},
            "Lane": {"errors": 1, "sites": 3},
            "Undrivable": {"errors": 0, "sites": 2},
            "Movable": {"errors": 0, "sites": 2},
            "MyCar": {"errors": 0, "sites": 2},
        },
        "cells_sha256": "1" * 64,
        "pose6_sha256": "2" * 64,
        "camera_diff_vs_v15": {
            "changed_channel_values": 0,
            "changed_rgb_pixels": 0,
            "l1_channel_sum": 0,
            "candidate_camera_sha256": "3" * 64,
        },
    }
    measurement, rows, _receiver, _camera_cache = _candidate_n600_batches(
        name="fixture/candidate_0000",
        archive=b"candidate",
        current_archive=b"current",
        current_receiver=Receiver(),
        current_batches=[current_batch],
        current_camera_cache={},
        baseline_receiver=Receiver(),
        baseline_camera_cache={},
        source_pair_ids=(0, 1),
        local_pair_ids=(0, 1),
        root=tmp_path,
        config_hash="a" * 64,
        batch_size=2,
        labels_all=np.zeros((2, 2, 3), dtype=np.int64),
        poses_all=np.zeros((2, 6), dtype=np.float32),
        segnet=object(),
        posenet=object(),
    )
    assert rows[0]["camera_identity_reused_exact_scorer_row"] is True
    assert measurement["exact_camera_identity_reused_batch_count"] == 1
    assert measurement["changed_camera_rescored_batch_count"] == 0
    assert measurement["errors"] == 2
    assert measurement["d_pose"] == "1.000000000000"
