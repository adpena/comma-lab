from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from tac.auth_eval_result import recompute_contest_score_from_payload
from tools.run_modal_smoke_before_full import (
    _expected_auth_artifact_markers,
    _recipe_requests_smoke_only,
    _resolve_smoke_band,
    _smoke_validation_contract_from_recipe,
    _validate_smoke_result,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _archive_zip_artifact(payload: bytes) -> tuple[bytes, str, int]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", payload)
    archive = buf.getvalue()
    return archive, hashlib.sha256(archive).hexdigest(), len(archive)


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


def test_training_artifact_contract_accepts_current_false_authority_manifest() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {
            "training_mode": "smoke",
            "archive_bytes": len(payload),
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_zip_bytes": archive_zip_bytes,
            "archive_zip_sha256": archive_zip_sha,
        },
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        required_artifact_markers=("lane_research_results/output/",),
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is True
    assert "research-only training_artifact_v1" in diagnostic
    assert "archive.zip contains exactly 0.bin" in diagnostic
    assert "score/promotion/rank/readiness claims are false" in diagnostic


def test_training_artifact_contract_rejects_stale_manifest_marker() -> None:
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": 12,
        "archive_sha256": "a" * 64,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "experiments/results/old/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": b"PK\x03\x04payload",
            },
        },
        required_artifact_markers=("lane_research_results/output/",),
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "refusing stale evidence" in diagnostic


def test_training_artifact_contract_rejects_score_authority_manifest() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": True,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "false_authority_flags_not_false" in diagnostic


def test_training_artifact_contract_rejects_missing_archive_sha() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(payload)
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "archive_sha256_invalid" in diagnostic


def test_training_artifact_contract_rejects_archive_member_mismatch() -> None:
    payload = b"payload-0bin"
    archive_zip, archive_zip_sha, archive_zip_bytes = _archive_zip_artifact(b"wrong")
    manifest = {
        "lane_id": "lane_research_smoke",
        "archive_bytes": len(payload),
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_zip_bytes": archive_zip_bytes,
        "archive_zip_sha256": archive_zip_sha,
        "result": {"training_mode": "smoke"},
        "research_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "lane_research_results/output/manifest.json": json.dumps(manifest),
                "lane_research_results/output/archive.zip": archive_zip,
            },
        },
        validation_contract="training_artifact_v1",
        required_lane_id="lane_research_smoke",
    )

    assert green is False
    assert "member_0bin" in diagnostic


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


def test_recipe_requests_smoke_only_reads_top_level_flag() -> None:
    assert _recipe_requests_smoke_only("lane_id: lane_x\nsmoke_only: true\n") is True
    assert _recipe_requests_smoke_only("lane_id: lane_x\nsmoke_only: false\n") is False
    assert _recipe_requests_smoke_only("# smoke_only: true in comment\n") is False


def test_smoke_validation_contract_defaults_to_contest_cuda_auth_eval() -> None:
    assert (
        _smoke_validation_contract_from_recipe("lane_id: lane_x\n")
        == "contest_cuda_auth_eval_v1"
    )
    assert (
        _smoke_validation_contract_from_recipe(
            "smoke_validation_contract: training_artifact_v1\n"
        )
        == "training_artifact_v1"
    )


def test_main_rejects_training_artifact_contract_without_recipe_smoke_only(
    tmp_path: Path,
) -> None:
    recipe_dir = tmp_path / ".omx/operator_authorize_recipes"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "unsafe.yaml").write_text(
        "schema_version: 1\n"
        "name: unsafe\n"
        "lane_id: lane_unsafe\n"
        "smoke_validation_contract: training_artifact_v1\n",
        encoding="utf-8",
    )

    rc = main(["--repo-root", str(tmp_path), "--recipe", "unsafe", "--smoke-only"])

    assert rc == 8


def test_main_rejects_unknown_smoke_validation_contract(tmp_path: Path) -> None:
    recipe_dir = tmp_path / ".omx/operator_authorize_recipes"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "unknown.yaml").write_text(
        "schema_version: 1\n"
        "name: unknown\n"
        "lane_id: lane_unknown_contract\n"
        "smoke_only: true\n"
        "smoke_validation_contract: typo_contract\n",
        encoding="utf-8",
    )

    rc = main(["--repo-root", str(tmp_path), "--recipe", "unknown"])

    assert rc == 7


def test_s2sbs_recipe_is_smoke_only_scaffold_guard() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_s2sbs_byte_stuffing_modal_t4_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert 'S2SBS_SMOKE: "1"' in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_pretrained_driving_prior_recipe_is_smoke_only_scaffold_guard() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert "status: scaffold_only_no_full_dispatch" in text
    assert "contest_exact_eval" not in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_s2sbs_remote_script_does_not_emit_contest_cuda_claim() -> None:
    text = (
        REPO_ROOT / "scripts/remote_lane_substrate_s2sbs_byte_stuffing.sh"
    ).read_text(encoding="utf-8")

    assert "LANE_S2SBS_DONE [smoke-only-no-score-claim]" in text
    assert "TRAINER_SMOKE_ARGS+=(--smoke)" in text
    assert "LANE_S2SBS_DONE [contest-CUDA]" not in text


def test_dpp_remote_script_verifies_active_claim_and_false_authority() -> None:
    text = (
        REPO_ROOT / "scripts/remote_lane_substrate_pretrained_driving_prior.sh"
    ).read_text(encoding="utf-8")

    assert "stage_0b_dispatch_claim_verified" in text
    assert "claim_lane_dispatch.py\" summary" in text
    assert "validate_dispatch_required_inputs.py" in text
    assert "PYBIN not set or not executable after bootstrap" in text
    assert '"$PYBIN" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py"' in text
    assert '"$WORKSPACE/.venv/bin/python" "$WORKSPACE/experiments/train_substrate_pretrained_driving_prior.py"' not in text
    assert '"rank_or_kill_eligible": false' in text


def test_pr95plus_recipe_is_smoke_only_until_full_trainer_lands() -> None:
    text = (
        REPO_ROOT
        / ".omx/operator_authorize_recipes/substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert _recipe_requests_smoke_only(text) is True
    assert "PR95PLUS_SMOKE: \"1\"" in text
    assert _smoke_validation_contract_from_recipe(text) == "training_artifact_v1"


def test_smoke_wrapper_uses_explicit_noninteractive_authorization_flag() -> None:
    text = (REPO_ROOT / "tools/run_modal_smoke_before_full.py").read_text(encoding="utf-8")

    assert '"tools/operator_authorize.py",' in text
    assert '"--yes",' in text
