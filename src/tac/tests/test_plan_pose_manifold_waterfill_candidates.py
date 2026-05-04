from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "plan_pose_manifold_waterfill_candidates.py"
SPEC = importlib.util.spec_from_file_location("plan_pose_manifold_waterfill_candidates", MODULE_PATH)
assert SPEC is not None
planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(planner)


def _archive(path: Path, payload: bytes = b"archive-bytes") -> tuple[int, str]:
    path.write_bytes(payload)
    return len(payload), hashlib.sha256(payload).hexdigest()


def _write_eval(
    path: Path,
    *,
    archive_size_bytes: int,
    archive_sha256: str,
    score: float,
    pose: float,
    seg: float,
    gpu_model: str,
    gpu_t4_match: bool,
) -> None:
    payload = {
        "archive_size_bytes": archive_size_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "device": "cuda",
            "gpu_model": gpu_model,
            "gpu_t4_match": gpu_t4_match,
            "sys_argv": ["experiments/contest_auth_eval.py", "--device", "cuda"],
        },
        "score_recomputed_from_components": score,
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_component_trace(
    path: Path,
    *,
    archive_size_bytes: int,
    score: float,
    pose: float,
    seg: float,
) -> None:
    samples = []
    for pair_index in range(600):
        hard = pair_index in {7, 11, 19}
        samples.append(
            {
                "frame_indices": [2 * pair_index, 2 * pair_index + 1],
                "pair_index": pair_index,
                "posenet_dist": 0.01 if hard else 0.0001,
                "segnet_dist": 0.002 if hard else 0.0001,
                "video_name": "0.mkv",
            }
        )
    payload = {
        "archive_size_bytes": archive_size_bytes,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "contest_auth_eval_cross_check": {"all_match": True},
        "n_samples": 600,
        "samples": samples,
        "score_claim": False,
        "score_recomputed_from_components": score,
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_pose_atom_plan(path: Path) -> None:
    payload = {
        "promotion_eligible": False,
        "recommended_policies": [
            {
                "charged_bytes_estimate": 4.0,
                "expected_score_saved_sum": 0.01,
                "measured_delta_atom_count": 1,
                "policy_name": "c_test_pose_atoms_top002",
                "prior_atom_count": 1,
                "selected_pair_indices": [7, 22],
            },
            {
                "charged_bytes_estimate": 12.0,
                "expected_score_saved_sum": 0.002,
                "measured_delta_atom_count": 1,
                "policy_name": "c_test_pose_atoms_top006",
                "prior_atom_count": 5,
                "selected_pair_indices": [1, 2, 3, 4, 5, 6],
            },
        ],
        "score_claim": False,
        "top_atoms": [],
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_active_metadata(path: Path) -> None:
    payload = {
        "evidence_grade": "empirical_metadata_for_line_search",
        "refinement": {
            "basis_index": 3,
            "basis_kind": "pair_window",
            "basis_pair_indices": [7, 11, 19],
            "basis_signed_magnitude": -1,
        },
        "score_claim": False,
    }
    path.write_text(json.dumps(payload, sort_keys=True))


def test_plan_emits_t4_confirmation_and_macro_specs(tmp_path: Path) -> None:
    frontier_eval = tmp_path / "frontier_eval.json"
    frontier_trace = tmp_path / "frontier_trace.json"
    diagnostic_eval = tmp_path / "diagnostic_eval.json"
    diagnostic_archive = tmp_path / "archive.zip"
    atom_plan = tmp_path / "pose_atom_plan.json"
    active = tmp_path / "active.json"
    output_dir = tmp_path / "out"
    ledger = tmp_path / "ledger.md"

    diag_bytes, diag_sha = _archive(diagnostic_archive)
    _write_eval(
        frontier_eval,
        archive_size_bytes=100,
        archive_sha256="a" * 64,
        score=0.40,
        pose=0.001,
        seg=0.001,
        gpu_model="Tesla T4",
        gpu_t4_match=True,
    )
    _write_component_trace(
        frontier_trace,
        archive_size_bytes=100,
        score=0.40,
        pose=0.001,
        seg=0.001,
    )
    _write_eval(
        diagnostic_eval,
        archive_size_bytes=diag_bytes,
        archive_sha256=diag_sha,
        score=0.35,
        pose=0.0008,
        seg=0.001,
        gpu_model="NVIDIA H100 NVL",
        gpu_t4_match=False,
    )
    _write_pose_atom_plan(atom_plan)
    _write_active_metadata(active)

    payload = planner.build_pose_manifold_plan(
        frontier_contest_eval=frontier_eval,
        frontier_component_trace=frontier_trace,
        output_dir=output_dir,
        ledger_md=ledger,
        diagnostic_contest_eval=diagnostic_eval,
        diagnostic_archive=diagnostic_archive,
        diagnostic_label="diag_h100",
        pose_atom_plan_path=atom_plan,
        active_metadata_path=active,
        frontier_label="C-TEST",
    )

    top = payload["dispatch_recommendations"][0]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert top["candidate_id"] == "diag_h100"
    assert top["requires_t4_confirmation"] is True
    assert top["dispatch_priority"] == "highest"
    assert "--archive" in top["exact_eval_command"]
    assert str(diagnostic_archive) in top["exact_eval_command"]
    assert payload["macro_candidate_specs"][0]["selected_pair_indices"] == [7, 22]
    assert payload["macro_candidate_specs"][0]["active_subspace_pair_overlap"] == [7]
    assert (output_dir / "pose_manifold_waterfill_plan.json").exists()
    assert (output_dir / "exact_eval_recommendations.json").exists()
    assert (output_dir / "artifact_manifest.json").exists()
    assert ledger.exists()


def test_plan_rejects_diagnostic_archive_sha_mismatch(tmp_path: Path) -> None:
    frontier_eval = tmp_path / "frontier_eval.json"
    frontier_trace = tmp_path / "frontier_trace.json"
    diagnostic_eval = tmp_path / "diagnostic_eval.json"
    diagnostic_archive = tmp_path / "archive.zip"

    diag_bytes, _diag_sha = _archive(diagnostic_archive, payload=b"actual")
    _write_eval(
        frontier_eval,
        archive_size_bytes=100,
        archive_sha256="a" * 64,
        score=0.40,
        pose=0.001,
        seg=0.001,
        gpu_model="Tesla T4",
        gpu_t4_match=True,
    )
    _write_component_trace(
        frontier_trace,
        archive_size_bytes=100,
        score=0.40,
        pose=0.001,
        seg=0.001,
    )
    _write_eval(
        diagnostic_eval,
        archive_size_bytes=diag_bytes,
        archive_sha256="b" * 64,
        score=0.35,
        pose=0.0008,
        seg=0.001,
        gpu_model="NVIDIA H100 NVL",
        gpu_t4_match=False,
    )

    with pytest.raises(planner.PoseManifoldPlanError, match="archive SHA mismatch"):
        planner.build_pose_manifold_plan(
            frontier_contest_eval=frontier_eval,
            frontier_component_trace=frontier_trace,
            output_dir=tmp_path / "out",
            diagnostic_contest_eval=diagnostic_eval,
            diagnostic_archive=diagnostic_archive,
            frontier_label="C-TEST",
        )
