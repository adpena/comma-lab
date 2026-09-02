from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments import ddm_qbr1_born_fairform_burn_prep as qbr


def test_seeded_schedule_is_balanced_distinct_and_deterministic() -> None:
    receipts = [qbr.schedule_receipt(seed) for seed in qbr.SEEDS]
    assert all(row["chunk_update_counts"] == [2500, 2500] for row in receipts)
    assert len({row["sha256_u8"] for row in receipts}) == 3
    assert receipts[0] == qbr.schedule_receipt(qbr.SEEDS[0])


def test_config_identity_allows_resume_location_and_authority_changes() -> None:
    base = {
        "scientific": 1,
        "output": "/a",
        "resume_from": None,
        "launch_authorized": False,
        "scorer_lane": {"claimed": False},
        "metal_lane": {"claimed": False},
    }
    changed = dict(base)
    changed.update(
        output="/b",
        resume_from="/checkpoint",
        launch_authorized=True,
        scorer_lane={"claimed": True, "claim_id": "s"},
        metal_lane={"claimed": True, "claim_id": "m"},
    )
    assert qbr.config_identity(base) == qbr.config_identity(changed)


def test_zero_native_arm_removes_only_the_native_term() -> None:
    target = torch.tensor([[[0, 1], [3, 0]]], dtype=torch.long)
    realized_logits = torch.zeros((1, 5, 2, 2), dtype=torch.float32)
    realized_logits.scatter_(1, target[:, None], 2.0)
    native_a = realized_logits.permute(0, 2, 3, 1).clone()
    native_b = (-realized_logits).permute(0, 2, 3, 1).clone()
    pose = torch.zeros((1, 6), dtype=torch.float32)
    camera = torch.zeros((1, 2, 3, 1, 1), dtype=torch.float32)
    weights = torch.ones(1)
    common = {
        "camera": camera,
        "pose6": pose,
        "logits": realized_logits,
        "target_argmax": target,
        "target_pose6": pose,
        "tau": 0.1,
        "sample_weights": weights,
        "lambdas": {"Lane": 0.0, "Movable": 0.0},
    }
    treatment = {"objective": qbr.ARMS["treatment_zero_native"]}
    control = {"objective": qbr.ARMS["control_native100"]}
    treatment_a, _ = qbr.fairform_objective(treatment, {"class_logits": native_a}, **common)
    treatment_b, _ = qbr.fairform_objective(treatment, {"class_logits": native_b}, **common)
    control_a, _ = qbr.fairform_objective(control, {"class_logits": native_a}, **common)
    control_b, _ = qbr.fairform_objective(control, {"class_logits": native_b}, **common)
    assert torch.equal(treatment_a, treatment_b)
    assert not torch.equal(control_a, control_b)


def _result(path: Path, seed: int, arm: str, score: float, pose_pass: bool) -> Path:
    payload = {
        "schema": qbr.RESULT_SCHEMA,
        "complete": True,
        "cell_id": f"seed_{seed}_{arm}",
        "milestones": [{"S_hat": score, "pose_corner_pass": pose_pass}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_adjudication_is_mechanical(tmp_path: Path) -> None:
    paths = []
    for index, seed in enumerate(qbr.SEEDS):
        paths.append(_result(tmp_path / f"c{seed}.json", seed, "control_native100", 1.0, False))
        paths.append(
            _result(
                tmp_path / f"t{seed}.json",
                seed,
                "treatment_zero_native",
                0.9 if index < 2 else 1.1,
                index < 2,
            )
        )
    result = qbr.adjudicate(paths, tmp_path / "verdict.json")
    assert result["disposition"] == "OPTIMIZATION_LIVE_DISTORTION_ROUTE"
    assert result["treatment_wins"] == 2
    assert result["treatment_pose_corner_passes"] == 2
