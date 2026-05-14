# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr79_c102_pose_action_interactions.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_c102_interactions_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, *, samples: dict[int, tuple[float, float]], bytes_: int) -> None:
    payload = {
        "archive_size_bytes": bytes_,
        "score_recomputed_from_components": 0.314,
        "score_seg_contribution": sum(seg for seg, _pose in samples.values()),
        "score_pose_contribution": sum(pose for _seg, pose in samples.values()),
        "samples": [
            {
                "pair_index": pair,
                "frame_indices": [pair * 2, pair * 2 + 1],
                "video_name": "0.mkv",
                "score_seg_contribution_exact": seg,
                "score_pose_contribution_first_order": pose,
                "score_combined_contribution_first_order": seg + pose,
                "segnet_dist": seg / 100.0,
                "posenet_dist": pose / 100.0,
            }
            for pair, (seg, pose) in sorted(samples.items())
        ],
        "trace_inputs": {"archive_sha256": "a" * 64},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_action_csv(path: Path, rows: list[tuple[int, int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "pair", "tile", "action"])
        for idx, (pair, tile, action) in enumerate(rows):
            writer.writerow([idx, pair, tile, action])


def _write_diff_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "tile", "action", "count_c102", "count_pr75", "count_pr77", "count_pr79"])
        writer.writerow([1, 88, 7, 1, 1, 1, 1])
        writer.writerow([1, 89, 9, 0, 0, 0, 1])
        writer.writerow([2, 90, 10, 0, 0, 0, 1])


def _write_pose_diff_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "c102", "pr75", "pr77", "pr79", "all_equal"])
        writer.writerow([0, 100, 100, 100, 100, True])
        writer.writerow([1, 100, 101, 101, 101, False])
        writer.writerow([2, 120, 122, 122, 122, False])


def _write_family(path: Path) -> None:
    def archive(label: str, action_bytes: int, pose_bytes: int, records: int) -> dict[str, Any]:
        return {
            "label": label,
            "archive": {"bytes": 1000 + action_bytes + pose_bytes, "sha256": label * 16},
            "decoded_streams": {
                "seg_tile_actions.bin": {"charged_bytes": action_bytes},
                "optimized_poses.qp1": {"charged_bytes": pose_bytes},
            },
            "actions": {"record_count": records, "unique_pair_count": records},
        }

    payload = {
        "schema": "unit_family",
        "score_claim": False,
        "archives": [
            archive("c102", 10, 20, 1),
            archive("pr75", 10, 21, 1),
            archive("pr77", 11, 21, 2),
            archive("pr79", 14, 22, 3),
        ],
        "action_pairwise_diffs": {"c102_vs_pr79": {"right_only_record_count": 2}},
        "pose_diffs": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_plan_emits_non_dispatchable_rankings(tmp_path: Path) -> None:
    script = _load_script()
    pr79_trace = tmp_path / "pr79_trace.json"
    c102_trace = tmp_path / "c102_trace.json"
    family = tmp_path / "family.json"
    action_diff = tmp_path / "action_diff.csv"
    pr79_actions = tmp_path / "pr79_actions.csv"
    c102_actions = tmp_path / "c102_actions.csv"
    pose_diff = tmp_path / "pose_diff.csv"
    output = tmp_path / "out"
    _write_trace(pr79_trace, samples={1: (0.012, 0.030), 2: (0.008, 0.040)}, bytes_=1036)
    _write_trace(c102_trace, samples={1: (0.010, 0.020), 2: (0.010, 0.020)}, bytes_=1030)
    _write_family(family)
    _write_diff_csv(action_diff)
    _write_action_csv(pr79_actions, [(1, 88, 7), (1, 89, 9), (2, 90, 10)])
    _write_action_csv(c102_actions, [(1, 88, 7)])
    _write_pose_diff_csv(pose_diff)

    plan = script.build_plan(
        output_dir=output,
        pr79_trace_path=pr79_trace,
        c102_trace_path=c102_trace,
        family_profile_path=family,
        action_diff_csv=action_diff,
        pr79_action_csv=pr79_actions,
        c102_action_csv=c102_actions,
        pose_diff_csv=pose_diff,
        top_k=10,
    )

    assert plan["score_claim"] is False
    assert plan["dispatch_decision"]["exact_eval_justified"] is False
    assert plan["ranked_action_atoms"][0]["no_op"] is False
    assert plan["ranked_action_atoms"][0]["recommended_archive_builder_inputs"]["dispatch_after_build"] is False
    assert plan["ranked_pose_atoms"][0]["pose_atom_id"].startswith("pose_qp1_row")
    assert plan["ranked_interaction_atoms"]
    assert all(policy["dispatchable"] is False for policy in plan["ranked_policy_inputs"])
    assert (output / "ranked_pose_action_atoms.json").exists()
    assert (output / "ranked_action_atoms.csv").exists()


def test_break_even_keeps_pr79_target_gap() -> None:
    script = _load_script()

    be = script._break_even(-100.0, benefit_proxy=0.001)

    assert be["target_score"] == 0.31
    assert be["score_if_components_unchanged"] == pytest.approx(
        script.PR79_FRONTIER_SCORE - 100.0 * script.RATE_SCORE_PER_BYTE
    )
    assert be["benefit_minus_required_component_gain"] < 0.0


def test_action_atoms_mark_exact_c102_duplicate_as_noop() -> None:
    script = _load_script()
    pair_rows = {
        1: {
            "frame_indices": [2, 3],
            "component_repair_benefit_proxy": 0.01,
            "pose_repair_benefit_proxy": 0.005,
            "seg_repair_benefit_proxy": 0.005,
        }
    }
    records = [script.ActionRecord(index=0, pair_index=1, tile_id=88, action_id=7)]

    atoms = script._action_atom_rows(
        pr79_records=records,
        c102_records=records,
        action_diff={(1, 88, 7): {"count_c102": 1, "count_pr79": 1}},
        pair_rows=pair_rows,
        action_extra_byte_proxy=1.0,
    )

    assert atoms[0]["no_op"] is True
    assert atoms[0]["component_benefit_proxy"] == 0.0
    assert atoms[0]["recommended_archive_builder_inputs"] is None
