from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.plan_c067_reversed_base_cdo1_overlay_economics import (
    BaseCandidate,
    build_plan,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _mask(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.uint8, copy=False))
    return path


def _trust_plan(
    path: Path,
    *,
    pair: int = 0,
    class_id: int = 1,
) -> Path:
    return _write_json(
        path,
        {
            "schema": "unit_trust_plan",
            "score_claim": False,
            "budget_policies": [
                {
                    "policy_id": "budget4000",
                    "budget_payload_bytes": 4000,
                    "selected_atoms": [
                        {
                            "atom_id": f"pair{pair:04d}_class{class_id}",
                            "pair_indices": [pair],
                            "class_id": class_id,
                            "changed_pixels": 4,
                        }
                    ],
                }
            ],
        },
    )


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((4, 384, 512), dtype=np.uint8)
    base = target.copy()
    target[0:2, 10:14, 20:24] = 1
    target[2:4, 30:34, 40:44] = 2
    return base, target


def test_reversed_base_planner_prices_byte_and_geometry_gates(tmp_path: Path) -> None:
    base, target = _arrays()
    base_path = _mask(tmp_path / "base.npy", base)
    target_path = _mask(tmp_path / "target.npy", target)
    payload = tmp_path / "base.cmg3"
    payload.write_bytes(b"B" * 128)

    plan = build_plan(
        repo_root=tmp_path,
        target_mask_array=target_path,
        base_candidates=[BaseCandidate("base", base_path, payload)],
        trust_plan_json=_trust_plan(tmp_path / "trust.json"),
        max_residual_disagreement_fraction=1e-9,
    )

    full = next(row for row in plan["all_candidates"] if row["policy"]["policy_id"] == "full_repair_to_c067_decoded_mask")
    threshold = next(
        row
        for row in plan["all_candidates"]
        if row["policy"]["policy_id"] == "geometry_threshold_longest_runs_to_residual_gate"
    )
    partial = next(row for row in plan["all_candidates"] if row["policy"]["policy_id"] == "budget4000")
    assert full["gates"]["joint_sub0240_geometry_gate"] is True
    assert threshold["gates"]["residual_geometry_gate"] is True
    assert threshold["mask_disagreement"]["residual_vs_target_count_after_overlay"] == 0
    assert threshold["policy"]["source"] == "deterministic_largest_run_waterfill_lower_bound"
    assert partial["gates"]["byte_gate_sub0240_if_distortion_unchanged"] is True
    assert partial["gates"]["residual_geometry_gate"] is False
    assert partial["mask_disagreement"]["selected_overlay_pixels"] == 32
    assert partial["mask_disagreement"]["residual_vs_target_count_after_overlay"] == 32
    assert partial["cdo1_payload"]["run_count"] > 0
    assert plan["score_claim"] is False


def test_reversed_base_half_frame_pair_basis_does_not_double_pair_ids(
    tmp_path: Path,
) -> None:
    base = np.zeros((8, 384, 512), dtype=np.uint8)
    target = base.copy()
    target[3, 10:12, 20:22] = 2
    target[6, 30:32, 40:42] = 2
    target[7, 30:32, 44:46] = 2
    base_path = _mask(tmp_path / "base.npy", base)
    target_path = _mask(tmp_path / "target.npy", target)
    payload = tmp_path / "base.cmg3"
    payload.write_bytes(b"B" * 128)

    plan = build_plan(
        repo_root=tmp_path,
        target_mask_array=target_path,
        base_candidates=[BaseCandidate("base", base_path, payload)],
        trust_plan_json=_trust_plan(tmp_path / "trust.json", pair=3, class_id=2),
        pair_index_basis="half_frame_pair_index",
    )

    partial = next(row for row in plan["all_candidates"] if row["policy"]["policy_id"] == "budget4000")
    assert plan["gates"]["pair_index_basis"] == "half_frame_pair_index"
    assert partial["mask_disagreement"]["pair_index_basis"] == "half_frame_pair_index"
    assert partial["mask_disagreement"]["base_vs_target_pair_indices"] == [3, 6, 7]
    assert partial["mask_disagreement"]["selected_overlay_pair_indices"] == [3]
    assert partial["mask_disagreement"]["residual_vs_target_pair_indices_after_overlay"] == [6, 7]
    assert partial["cdo1_payload"]["run_count"] > 0
    assert partial["cdo1_payload"]["payload_header"]["pair_index_basis"] == "half_frame_pair_index"
    assert partial["cdo1_payload"]["payload_header"]["selected_pair_indices"] == [3]
    assert partial["mask_disagreement"]["selected_overlay_pixels"] == 4
    assert partial["mask_disagreement"]["residual_vs_target_count_after_overlay"] == 8


def test_reversed_base_planner_can_fail_geometry_safe_candidate_on_bytes(tmp_path: Path) -> None:
    base, target = _arrays()
    base_path = _mask(tmp_path / "base.npy", base)
    target_path = _mask(tmp_path / "target.npy", target)
    payload = tmp_path / "large_base.cmg3"
    payload.write_bytes(b"B" * 400_000)

    plan = build_plan(
        repo_root=tmp_path,
        target_mask_array=target_path,
        base_candidates=[BaseCandidate("large", base_path, payload)],
        trust_plan_json=_trust_plan(tmp_path / "trust.json"),
    )

    assert plan["joint_sub0300_geometry_count"] == 0
    assert plan["geometry_safe_byte_regressive_count"] >= 1
    assert plan["decision"] == "geometry_safe_but_byte_regressive"
