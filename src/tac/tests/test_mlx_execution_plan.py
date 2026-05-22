# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration import EVIDENCE_GRADE_MLX
from tac.local_acceleration.mlx_execution_plan import (
    MLXExecutionPlanError,
    build_mlx_scorer_response_execution_plan,
)
from tac.local_acceleration.mlx_profile_stability import build_profile_stability_manifest

REPO = Path(__file__).resolve().parents[3]


def _profile_with_gpu_rejected() -> dict:
    return {
        "schema_version": "mlx_scorer_response_profile.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "reference_cache_dir": "/tmp/reference-cache",
        "candidate_cache_dir": "/tmp/candidate-cache",
        "archive_size_bytes": 178417,
        "start_pair": 16,
        "max_pairs": 4,
        "rows": [
            {
                "device": "cpu",
                "batch_pairs": 1,
                "n_samples": 4,
                "pair_window": [16, 20],
                "canonical_score": 0.178195,
                "avg_posenet_dist": 0.000006,
                "avg_segnet_dist": 0.000515,
                "posenet_sha256": "p" * 64,
                "segnet_sha256": "s" * 64,
                "pairs_per_second": 1.0,
                "start_pair": 16,
            },
            {
                "device": "cpu",
                "batch_pairs": 2,
                "n_samples": 4,
                "pair_window": [16, 20],
                "canonical_score": 0.1781951,
                "avg_posenet_dist": 0.000006,
                "avg_segnet_dist": 0.000515,
                "posenet_sha256": "p" * 64,
                "segnet_sha256": "s" * 64,
                "pairs_per_second": 1.5,
                "start_pair": 16,
            },
            {
                "device": "gpu",
                "batch_pairs": 1,
                "n_samples": 4,
                "pair_window": [16, 20],
                "canonical_score": 0.180,
                "avg_posenet_dist": 0.000006,
                "avg_segnet_dist": 0.000530,
                "posenet_sha256": "g" * 64,
                "segnet_sha256": "h" * 64,
                "pairs_per_second": 12.0,
                "start_pair": 16,
            },
        ],
    }


def _stable_gpu_manifest() -> dict:
    profile = _profile_with_gpu_rejected()
    profile["rows"] = [
        {
            "device": "gpu",
            "batch_pairs": 1,
            "n_samples": 4,
            "pair_window": [16, 20],
            "canonical_score": 0.178195,
            "avg_posenet_dist": 0.000006,
            "avg_segnet_dist": 0.000515,
            "posenet_sha256": "p" * 64,
            "segnet_sha256": "s" * 64,
            "pairs_per_second": 12.0,
            "start_pair": 16,
        }
    ]
    return build_profile_stability_manifest(profile, baseline_device="gpu")


def test_execution_plan_uses_fastest_eligible_cpu_row_from_failed_profile() -> None:
    manifest = build_profile_stability_manifest(
        _profile_with_gpu_rejected(),
        baseline_device="cpu",
        baseline_batch_pairs=1,
    )

    assert manifest["passed"] is False
    plan = build_mlx_scorer_response_execution_plan(
        manifest,
        response_output="experiments/results/next_response.json",
    )

    execution = plan["recommended_execution"]
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["profile_full_pass_required"] is False
    assert execution["device"] == "cpu"
    assert execution["batch_pairs"] == 2
    assert execution["start_pair"] == 16
    assert execution["max_pairs"] == 4
    assert execution["archive_size_bytes"] == 178417
    assert "--allow-gpu-research-signal" not in execution["command_args"]
    assert any("source_profile_failed" in warning for warning in plan["warnings"])


def test_execution_plan_rejects_false_authority() -> None:
    manifest = build_profile_stability_manifest(_profile_with_gpu_rejected())
    manifest["score_claim"] = True

    try:
        build_mlx_scorer_response_execution_plan(manifest)
    except MLXExecutionPlanError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected false-authority rejection")


def test_execution_plan_requires_gpu_allowance_for_gpu_selection() -> None:
    manifest = _stable_gpu_manifest()

    try:
        build_mlx_scorer_response_execution_plan(manifest)
    except MLXExecutionPlanError as exc:
        assert "requires allow_gpu_research_signal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected GPU research-signal allowance rejection")

    plan = build_mlx_scorer_response_execution_plan(
        manifest,
        allow_gpu_research_signal=True,
    )
    assert plan["recommended_execution"]["device"] == "gpu"
    assert "--allow-gpu-research-signal" in plan["recommended_execution"]["command_args"]


def test_execution_plan_cli_and_ll_plan_attachment(tmp_path: Path) -> None:
    manifest = build_profile_stability_manifest(
        _profile_with_gpu_rejected(),
        baseline_device="cpu",
        baseline_batch_pairs=1,
    )
    stability_path = tmp_path / "stability.json"
    plan_path = tmp_path / "mlx_plan.json"
    ll_plan_path = tmp_path / "ll_plan.json"
    ll_plan_md_path = tmp_path / "ll_plan.md"
    dataset_path = tmp_path / "dataset.json"
    stability_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "summary": {}, "rows": []}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_scorer_response_execution.py"),
            "--stability-manifest",
            str(stability_path),
            "--output",
            str(plan_path),
            "--response-output",
            str(tmp_path / "response.json"),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    assert '"score_claim": false' in completed.stdout
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["recommended_execution"]["batch_pairs"] == 2

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_ll_scorer_response_next.py"),
            "--dataset",
            str(dataset_path),
            "--mlx-profile-stability",
            str(stability_path),
            "--mlx-response-output",
            str(tmp_path / "response.json"),
            "--json-out",
            str(ll_plan_path),
            "--md-out",
            str(ll_plan_md_path),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    ll_plan = json.loads(ll_plan_path.read_text(encoding="utf-8"))
    assert ll_plan["mlx_scorer_response_execution_plan"]["score_claim"] is False
    assert (
        ll_plan["mlx_scorer_response_execution_plan"]["recommended_execution"][
            "batch_pairs"
        ]
        == 2
    )
    assert "MLX Execution Recommendation" in ll_plan_md_path.read_text(encoding="utf-8")
