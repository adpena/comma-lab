from __future__ import annotations

import json

from tac.auth_eval_result import recompute_contest_score_from_payload
from tools.run_modal_smoke_before_full import (
    _expected_auth_artifact_markers,
    _resolve_smoke_band,
    _validate_smoke_result,
)


def _auth_payload(*, score_axis: str = "contest_cuda") -> dict:
    payload = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.0004,
        "archive_size_bytes": 150_000,
        "score_axis": score_axis,
        "lane_tag": "[contest-CUDA]"
        if score_axis == "contest_cuda"
        else "[diagnostic-auth-eval]",
        "evidence_grade": "contest-CUDA" if score_axis == "contest_cuda" else "B",
        "exact_cuda_eval_complete": score_axis == "contest_cuda",
        "score_claim": score_axis == "contest_cuda",
        "score_claim_valid": score_axis == "contest_cuda",
    }
    payload["canonical_score"] = recompute_contest_score_from_payload(payload)
    return payload


def test_smoke_validation_accepts_canonical_contest_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(_auth_payload()),
            },
        }
    )

    assert green is True
    assert "SMOKE GREEN" in diagnostic


def test_smoke_validation_rejects_diagnostic_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload(score_axis="diagnostic_cuda")
                ),
            },
        }
    )

    assert green is False
    assert "did not contain any finite component-coherent contest-CUDA" in diagnostic


def test_smoke_validation_scans_all_auth_eval_artifacts() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "a_diagnostic_auth_eval_cuda.json": json.dumps(
                    _auth_payload(score_axis="diagnostic_cuda")
                ),
                "z_contest_auth_eval_cuda.json": json.dumps(_auth_payload()),
            },
        }
    )

    assert green is True
    assert "z_contest_auth_eval_cuda.json" in diagnostic


def test_smoke_validation_rejects_stale_auth_eval_artifact_when_marker_required() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
                "submissions/robust_current/auth_eval_renderer_fp4.json": json.dumps(
                    _auth_payload()
                ),
            },
        },
        required_artifact_markers=("lane_substrate_siren_results/output/",),
    )

    assert green is False
    assert "refusing stale evidence" in diagnostic


def test_smoke_validation_accepts_current_output_marker() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
                "lane_substrate_siren_results/output/contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload()
                ),
            },
        },
        required_artifact_markers=("lane_substrate_siren_results/output/",),
    )

    assert green is True
    assert "lane_substrate_siren_results/output/contest_auth_eval_cuda.json" in diagnostic


def test_expected_auth_artifact_markers_parse_workspace_output_dir() -> None:
    recipe = """
env_overrides:
  SIREN_OUTPUT_DIR: /workspace/pact/lane_substrate_siren_results/output
"""

    markers = _expected_auth_artifact_markers(recipe, instance_job_id="job123")

    assert "job123" in markers
    assert "results/job123/" in markers
    assert "lane_substrate_siren_results/output/" in markers


def test_resolve_smoke_band_reads_siren_recipe_prediction_band() -> None:
    lo, hi = _resolve_smoke_band("predicted_band: [0.130, 0.165]\n")

    assert lo == 0.1125
    assert hi == 0.1825


def test_resolve_smoke_band_prefers_explicit_smoke_score_band() -> None:
    lo, hi = _resolve_smoke_band(
        "predicted_band: [0.130, 0.165]\nsmoke_score_band: [0.050, 5.000]\n"
    )

    assert lo == 0.05
    assert hi == 5.0
