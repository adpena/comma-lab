from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_scorer_weighted_pose_atoms.py"
SPEC = importlib.util.spec_from_file_location("plan_scorer_weighted_pose_atoms", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)


def _write_component_trace(
    path: Path,
    *,
    pose_by_pair: dict[int, float],
    seg_by_pair: dict[int, float] | None = None,
    cross_check: bool = True,
    score: float = 0.2,
    bytes_: int = 1234,
) -> None:
    samples = []
    pose_sum = 0.0
    seg_sum = 0.0
    seg_by_pair = seg_by_pair or {}
    for pair_index in range(600):
        pose = pose_by_pair.get(pair_index, 0.0001)
        seg = seg_by_pair.get(pair_index, 0.0001)
        pose_sum += pose
        seg_sum += seg
        samples.append(
            {
                "pair_index": pair_index,
                "video_name": "0.mkv",
                "video_pair_index": pair_index,
                "frame_indices": [2 * pair_index, 2 * pair_index + 1],
                "posenet_dist": pose,
                "segnet_dist": seg,
            }
        )
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": 600,
        "expected_contest_samples": 600,
        "avg_posenet_dist": pose_sum / 600.0,
        "avg_segnet_dist": seg_sum / 600.0,
        "archive_size_bytes": bytes_,
        "score_recomputed_from_components": score,
        "contest_auth_eval_cross_check": {
            "all_match": cross_check,
            "contest_auth_eval_json_sha256": "c" * 64,
        },
        "samples": samples,
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_contest_eval(path: Path, *, score: float = 0.2, bytes_: int = 1234) -> None:
    payload = {
        "score_recomputed_from_components": score,
        "archive_size_bytes": bytes_,
        "avg_posenet_dist": 0.001,
        "avg_segnet_dist": 0.001,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": "a" * 64,
            "archive_size_bytes": bytes_,
            "device": "cuda",
            "gpu_model": "Tesla T4",
            "gpu_t4_match": True,
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_active_metadata(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "empirical_metadata_for_line_search",
        "refinement": {
            "basis_kind": "pair_window",
            "basis_index": 0,
            "basis_signed_magnitude": 1,
            "basis_pair_indices": [3, 8, 13],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_atom_ledger(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "derived_diagnostic_trace_waterfill",
        "atom_allocation_table": {
            "byte_models": {
                "test": {
                    "pose_pair_bytes": 2,
                    "pose_pair_bytes_source": "test byte model",
                }
            },
            "ranked_atoms": [
                {
                    "family": "pose",
                    "pair_index": 8,
                    "atom_id": "ledger:pose:pair_0008",
                    "expected_score_saved": 0.02,
                    "expected_score_saved_per_byte": 0.01,
                    "waterfill_utility_score": 0.018,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def test_pose_atom_plan_is_deterministic_and_non_promotable(tmp_path: Path) -> None:
    target = tmp_path / "component_trace.json"
    contest = tmp_path / "contest_auth_eval.json"
    reference = tmp_path / "reference_trace.json"
    active = tmp_path / "active.json"
    ledger = tmp_path / "ledger.json"
    out1 = tmp_path / "plan1.json"
    out2 = tmp_path / "plan2.json"

    _write_component_trace(target, pose_by_pair={5: 0.03, 9: 0.02})
    _write_contest_eval(contest)
    _write_component_trace(reference, pose_by_pair={5: 0.001, 9: 0.019}, cross_check=False)
    _write_active_metadata(active)
    _write_atom_ledger(ledger)

    payload1 = planner.build_pose_atom_plan(
        component_trace_path=target,
        contest_auth_eval_path=contest,
        output_json=out1,
        frontier_label="C-TEST",
        reference_trace_paths=[reference],
        active_metadata_paths=[active],
        atom_ledger_paths=[ledger],
        max_atoms=10,
        policy_counts=(2, 4),
    )
    payload2 = planner.build_pose_atom_plan(
        component_trace_path=target,
        contest_auth_eval_path=contest,
        output_json=out2,
        frontier_label="C-TEST",
        reference_trace_paths=[reference],
        active_metadata_paths=[active],
        atom_ledger_paths=[ledger],
        max_atoms=10,
        policy_counts=(2, 4),
    )

    assert out1.read_bytes() == out2.read_bytes()
    assert payload1 == payload2
    assert payload1["score_claim"] is False
    assert payload1["promotion_eligible"] is False
    assert payload1["recommended_policies"][0]["score_claim"] is False
    assert payload1["recommended_policies"][0]["promotion_eligible"] is False
    assert payload1["required_promotion_eval"].endswith(
        "experiments/contest_auth_eval.py --device cuda"
    )


def test_pose_atoms_sort_by_expected_benefit_per_charged_byte(tmp_path: Path) -> None:
    target = tmp_path / "component_trace.json"
    contest = tmp_path / "contest_auth_eval.json"
    reference = tmp_path / "reference_trace.json"
    out = tmp_path / "plan.json"

    _write_component_trace(target, pose_by_pair={2: 0.06, 7: 0.04})
    _write_contest_eval(contest)
    _write_component_trace(reference, pose_by_pair={2: 0.01, 7: 0.039}, cross_check=False)

    payload = planner.build_pose_atom_plan(
        component_trace_path=target,
        contest_auth_eval_path=contest,
        output_json=out,
        frontier_label="C-TEST",
        reference_trace_paths=[reference],
        active_metadata_paths=[],
        atom_ledger_paths=[],
        charged_bytes_per_atom=4,
        max_atoms=5,
        policy_counts=(2,),
    )

    top_atoms = payload["top_atoms"]
    assert top_atoms[0]["pair_index"] == 2
    ratios = [atom["expected_score_saved_per_charged_byte"] for atom in top_atoms]
    assert ratios == sorted(ratios, reverse=True)
    assert payload["recommended_policies"][0]["selected_pair_indices"][0] == 2


def test_pose_atom_plan_uses_prior_when_reference_delta_is_unavailable(tmp_path: Path) -> None:
    target = tmp_path / "component_trace.json"
    contest = tmp_path / "contest_auth_eval.json"
    reference = tmp_path / "reference_trace.json"
    active = tmp_path / "active.json"
    out = tmp_path / "plan.json"

    _write_component_trace(target, pose_by_pair={8: 0.05, 12: 0.04})
    _write_contest_eval(contest)
    _write_component_trace(reference, pose_by_pair={8: 0.05, 12: 0.04}, cross_check=False)
    _write_active_metadata(active)

    payload = planner.build_pose_atom_plan(
        component_trace_path=target,
        contest_auth_eval_path=contest,
        output_json=out,
        frontier_label="C-TEST",
        reference_trace_paths=[reference],
        active_metadata_paths=[active],
        atom_ledger_paths=[],
        charged_bytes_per_atom=4,
        max_atoms=20,
        policy_counts=(4,),
    )

    prior_atoms = [atom for atom in payload["top_atoms"] if not atom["measured_delta_available"]]
    assert prior_atoms
    assert prior_atoms[0]["evidence_source"] == "hard_pair_active_subspace_prior"
    assert prior_atoms[0]["score_claim"] is False
    assert prior_atoms[0]["promotion_eligible"] is False
    assert "no_positive_reference_component_delta" in prior_atoms[0]["risk_reasons"]
