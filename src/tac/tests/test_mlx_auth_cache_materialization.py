# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_auth_cache_materialization import (
    READY_VERDICT,
    REQUIRED_VERDICT,
    build_mlx_auth_cache_materialization_plan,
)

REPO = Path(__file__).resolve().parents[3]
HASH_DOMAIN = "_array_sha256(dtype_string + json_shape + contiguous_bytes)"
ARRAY_HASHES = {
    "pair_indices": "1" * 64,
    "segnet_last_rgb": "2" * 64,
    "posenet_yuv6_pair": "3" * 64,
}


def test_materialization_plan_ready_for_passing_audit() -> None:
    plan = build_mlx_auth_cache_materialization_plan(_audit(passed=True))

    assert plan["passed"] is True
    assert plan["verdict"] == READY_VERDICT
    assert plan["score_claim"] is False
    assert plan["score_claim_valid"] is False
    assert plan["promotion_eligible"] is False
    assert plan["rank_or_kill_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["promotable"] is False
    assert plan["evidence_grade"] == EVIDENCE_GRADE_MLX
    assert plan["score_axis"] == EVIDENCE_TAG_MLX
    assert plan["next_materialization_action"] == "use_existing_cache_for_local_mlx_transfer_calibration"
    assert "local_mlx_training_transfer_calibration" in plan["allowed_use"]


def test_materialization_plan_classifies_same_archive_auth_surface_mismatch() -> None:
    audit = _audit(
        passed=False,
        blockers=[
            "inflated_outputs_aggregate_sha256_mismatch_or_missing",
            "raw_sha256_mismatch_or_missing",
            "scorer_input_array_sha256_mismatch:segnet_last_rgb",
            "scorer_input_array_sha256_mismatch:posenet_yuv6_pair",
        ],
        cache_raw="c" * 64,
        auth_raw="d" * 64,
        cache_aggregate="a" * 64,
        auth_aggregate="b" * 64,
        cache_hashes={
            "pair_indices": "1" * 64,
            "segnet_last_rgb": "4" * 64,
            "posenet_yuv6_pair": "5" * 64,
        },
    )

    plan = build_mlx_auth_cache_materialization_plan(audit)

    assert plan["passed"] is False
    assert plan["verdict"] == REQUIRED_VERDICT
    assert plan["next_materialization_action"] == (
        "materialize_auth_axis_tensor_cache_from_modal_linux_raw_or_export_linux_tensor_cache"
    )
    assert "do_not_use_for_auth_axis_transfer_calibration" in plan["allowed_use"]
    surfaces = plan["surface_classification"]
    assert surfaces["archive_identity"]["match"] is True
    assert (
        surfaces["decoded_raw_surface"]["classification"]
        == "same_archive_different_decoded_raw_surface"
    )
    assert (
        surfaces["scorer_input_surface"]["classification"]
        == "same_archive_different_scorer_input_tensors"
    )
    assert surfaces["scorer_input_surface"]["array_sha256"]["pair_indices"]["match"] is True
    assert surfaces["scorer_input_surface"]["array_sha256"]["segnet_last_rgb"]["match"] is False
    assert surfaces["scorer_input_surface"]["array_sha256"]["posenet_yuv6_pair"]["match"] is False
    assert plan["required_artifacts"][0]["expected_raw_sha256"] == "d" * 64
    recommended = "\n".join(plan["recommended_commands"])
    assert "experiments/modal_auth_eval_cpu.py" in recommended
    assert "modal volume get comma-auth-eval-cache-artifacts" in recommended
    assert "--scorer-input-cache-tensors-out-dir" in recommended
    assert "--allow-large-scorer-input-cache-tensor-export" in recommended
    assert "tools/build_mlx_scorer_input_cache.py --raw" in recommended


def test_materialization_plan_requests_hash_export_when_auth_hashes_missing() -> None:
    audit = _audit(
        passed=False,
        blockers=["scorer_input_array_sha256_mismatch:segnet_last_rgb"],
    )
    audit["auth_eval"]["scorer_input_hash_domain"] = None
    audit["auth_eval"]["scorer_input_array_sha256"] = {}

    plan = build_mlx_auth_cache_materialization_plan(audit)

    assert plan["next_materialization_action"] == "rerun_auth_eval_with_scorer_input_cache_hashes"
    assert (
        plan["surface_classification"]["scorer_input_surface"]["classification"]
        == "auth_scorer_input_hashes_missing"
    )


def test_materialization_plan_stops_on_wrong_archive() -> None:
    audit = _audit(passed=False, blockers=["archive_sha256_mismatch_or_missing"])
    audit["auth_eval"]["archive_sha256"] = "f" * 64

    plan = build_mlx_auth_cache_materialization_plan(audit)

    assert plan["next_materialization_action"] == "stop_wrong_archive"


def test_materialization_plan_cli_writes_json(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    out_path = tmp_path / "plan.json"
    audit_path.write_text(
        json.dumps(
            _audit(
                passed=False,
                blockers=[
                    "raw_sha256_mismatch_or_missing",
                    "scorer_input_array_sha256_mismatch:segnet_last_rgb",
                ],
                cache_raw="c" * 64,
                auth_raw="d" * 64,
                cache_hashes={
                    "pair_indices": "1" * 64,
                    "segnet_last_rgb": "4" * 64,
                    "posenet_yuv6_pair": "3" * 64,
                },
            )
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_mlx_auth_cache_materialization.py"),
            "--cache-auth-audit",
            str(audit_path),
            "--output",
            str(out_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "materialize_auth_axis_tensor_cache" in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["score_claim_valid"] is False


def _audit(
    *,
    passed: bool,
    blockers: list[str] | None = None,
    cache_raw: str = "r" * 64,
    auth_raw: str = "r" * 64,
    cache_aggregate: str = "a" * 64,
    auth_aggregate: str = "a" * 64,
    cache_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    array_hashes = dict(cache_hashes or ARRAY_HASHES)
    auth_hashes = dict(ARRAY_HASHES)
    return {
        "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
        "passed": passed,
        "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY" if passed else "FAIL_CACHE_AUTH_EVAL_IDENTITY",
        "blockers": [] if passed else list(blockers or ["raw_sha256_mismatch_or_missing"]),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": "a" * 64,
            "inflated_outputs_aggregate_sha256": cache_aggregate,
            "raw_sha256": cache_raw,
            "pair_count": 600,
            "hash_domain": HASH_DOMAIN,
            "array_sha256": array_hashes,
        },
        "auth_eval": {
            "archive_sha256": "a" * 64,
            "inflated_outputs_aggregate_sha256": auth_aggregate,
            "raw_file_sha256": auth_raw,
            "n_samples": 600,
            "score": 0.1920513168811056,
            "pose_avg": 0.00002943,
            "seg_avg": 0.00056029,
            "evidence_grade": "contest-CPU",
            "score_axis": "contest_cpu",
            "scorer_input_hash_domain": HASH_DOMAIN,
            "scorer_input_array_sha256": auth_hashes,
        },
        "canonical_equation": {
            "equation_id": "scorer_input_cache_hash_identity_v1",
            "identity_residual": 0 if passed else 1,
            "eligible_for_local_mlx_transfer_calibration": passed,
            "blockers": [] if passed else list(blockers or ["raw_sha256_mismatch_or_missing"]),
        },
    }
