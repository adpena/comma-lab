# SPDX-License-Identifier: MIT
"""Typed contract and exact-all replay tests for G90 V2."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.taskspace_projected_population_costates_v1 import (
    ProjectedOperandRowV1,
)

TOOL = Path(__file__).resolve().parents[1] / "materialize_taskspace_exact_coarse_costates_v2.py"
SPEC = importlib.util.spec_from_file_location("g90_exact_coarse_v2_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

H = "a" * 64


def _identity() -> dict[str, object]:
    return {"path": "/does/not/exist", "bytes": 1, "sha256": H}


def _config() -> dict[str, object]:
    return {
        "schema": MODULE.CONFIG_SCHEMA,
        "output_root": "/Volumes/VertigoDataTier/pact/g90_v2_test",
        "v1_config": _identity(),
        "v1_terminal_receipt": _identity(),
        "seed": 1234,
        "num_threads": 4,
        "safety_reserve_bytes": 1,
        "batch_pairs_maximum": 16,
        "stage_pairs": 120,
        "stage_count": 5,
        "exact_replay_policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
        "pareto_pruning_allowed": False,
        "dense_costates_persisted": False,
        "research_only": True,
    }


def _row(index: int) -> ProjectedOperandRowV1:
    return ProjectedOperandRowV1(
        operand_id=f"row-{index}",
        family_id="G72",
        pair_ids=(0,),
        operand_member_bytes=1,
        operand_sha256=H,
        atom_count=1,
        changed_camera_values=1,
        pose_linearized_score_delta=float(index),
        seg_gap_directional_delta=float(-index),
    )


def test_config_requires_exact_all_and_refuses_pareto(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert MODULE.load_config(path).seed == 1234
    body = _config()
    body["pareto_pruning_allowed"] = True
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(MODULE.ExactCoarseCostateV2Error, match="contract differs"):
        MODULE.load_config(path)


def test_exact_replay_calls_every_deterministic_physical_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(_row(index) for index in range(12))
    realized = {row.operand_id: np.zeros((1, 1, 1, 1, 1), dtype=np.uint8) for row in rows}
    called: list[str] = []

    def replay(row: ProjectedOperandRowV1, **_: object) -> ProjectedOperandRowV1:
        called.append(row.operand_id)
        return row

    monkeypatch.setattr(MODULE, "exact_replay_projected_intervention", replay)
    output = MODULE._exact_replay_all(
        rows,
        expected_group_ids=tuple(row.operand_id for row in rows),
        realized=realized,
        target_cells=np.zeros((1, 1, 1), dtype=np.uint8),
        costates=None,
        posenet=None,
        segnet=None,
    )
    assert output == rows
    assert called == [row.operand_id for row in rows]

    with pytest.raises(
        MODULE.ExactCoarseCostateV2Error,
        match="physical-group set differs",
    ):
        MODULE._exact_replay_all(
            rows,
            expected_group_ids=tuple(row.operand_id for row in rows[:-1]),
            realized=realized,
            target_cells=np.zeros((1, 1, 1), dtype=np.uint8),
            costates=None,
            posenet=None,
            segnet=None,
        )
    seven = rows[:7]
    with pytest.raises(
        MODULE.ExactCoarseCostateV2Error,
        match="physical-group set differs",
    ):
        MODULE._exact_replay_all(
            seven,
            expected_group_ids=tuple(row.operand_id for row in seven),
            realized={row.operand_id: realized[row.operand_id] for row in seven},
            target_cells=np.zeros((1, 1, 1), dtype=np.uint8),
            costates=None,
            posenet=None,
            segnet=None,
        )


def test_stage_resume_never_skips_an_incomplete_stage(tmp_path: Path) -> None:
    assert MODULE._next_incomplete_stage(tmp_path) == 0
    stage0 = tmp_path / "20_stage_00_0000_0120" / "stage_receipt.json"
    stage0.parent.mkdir(parents=True)
    stage0.write_text("{}", encoding="utf-8")
    assert MODULE._next_incomplete_stage(tmp_path) == 1


def test_resume_frontier_refuses_self_consistent_incomplete_checkpoint(
    tmp_path: Path,
) -> None:
    actual_ids = tuple(f"row-{index}" for index in range(7))
    expected_ids = tuple(f"row-{index}" for index in range(8))
    body = {
        "schema": MODULE.BATCH_SCHEMA,
        "pair_range": [0, 16],
        "projection_coordinate_count": len(actual_ids),
        "expected_physical_group_count": len(actual_ids),
        "expected_physical_group_ids": list(actual_ids),
        "projection_rows": [
            {
                "operand_id": operand_id,
                "exact_seg_score_delta": 0.0,
                "exact_pose_score_delta": 0.0,
            }
            for operand_id in actual_ids
        ],
        "actuator_basis_groups": [{"group_id": operand_id} for operand_id in actual_ids],
        "exact_replay_state_custody": [{"operand_id": operand_id} for operand_id in actual_ids],
        "exact_replay_policy": "ALL_DETERMINISTIC_PHYSICAL_GROUPS",
        "all_deterministic_physical_groups_exact_replayed": True,
        "pareto_pruning_performed": False,
        "local_admission_performed": False,
    }
    forged = MODULE._seal(body, field="batch_checkpoint_sha256")
    batch_path = tmp_path / "20_stage_00_0000_0120" / "batches" / "batch_0000_0016.json"
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(
        MODULE.ExactCoarseCostateV2Error,
        match="differs from rederived physical groups",
    ):
        MODULE._immutable_resume_frontier(
            tmp_path,
            expected_group_ids_by_range={(0, 16): expected_ids},
        )
