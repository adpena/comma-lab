from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "modal_t1_balle_endtoend.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_modal_t1_balle_endtoend_test", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    return SCRIPT.read_text()


def test_plan_cli_is_default_safe_and_writes_no_claim(tmp_path: Path) -> None:
    claim_file = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    before_claims = claim_file.read_text()
    json_out = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "plan",
            "--label",
            "unit-test-t1-plan",
            "--epochs",
            "7",
            "--batch-size",
            "3",
            "--timeout-hours",
            "24",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_out.read_text())
    assert json.loads(result.stdout) == payload
    assert payload["lane_id"] == "t1_balle_128k_endtoend"
    assert payload["instance_job_id"] == "unit-test-t1-plan"
    assert payload["dispatch_attempted"] is False
    assert payload["remote_or_gpu_eval_started"] is False
    assert payload["lane_claim_opened"] is False
    assert payload["modal_app_creation_not_attempted"] is True
    assert payload["dry_run_default"] is True
    assert payload["requires_execute_for_dispatch"] is True
    assert payload["modal_function_timeout_hours"] == 24.0
    assert payload["requested_timeout_hours"] == 24.0
    assert payload["estimated_cost_usd"] == 14.16
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert "--execute" in payload["dispatch_command"]
    assert "--device" not in payload["dispatch_command"]
    assert payload["params"]["device"] == "cuda"
    assert payload["params"]["enable_t13_sqrt_n_budget"] is True
    assert payload["params"]["enable_t19_adaptive_rho"] is True
    assert claim_file.read_text() == before_claims


def test_plan_cli_cost_cap_failure_still_opens_no_claim(tmp_path: Path) -> None:
    claim_file = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    before_claims = claim_file.read_text()
    json_out = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "plan",
            "--label",
            "too-expensive",
            "--timeout-hours",
            "24",
            "--cost-cap-usd",
            "1",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 2
    payload = json.loads(json_out.read_text())
    assert payload["ready_for_modal_dispatch_command"] is False
    assert payload["score_claim"] is False
    assert payload["validation_errors"] == ["estimated_cost_exceeds_cap:14.16>1.00"]
    assert claim_file.read_text() == before_claims


def test_plan_cli_rejects_timeout_that_does_not_match_modal_function(tmp_path: Path) -> None:
    claim_file = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    before_claims = claim_file.read_text()
    json_out = tmp_path / "plan.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "plan",
            "--label",
            "misleading-timeout",
            "--timeout-hours",
            "1.5",
            "--train-timeout-hours",
            "1",
            "--json-out",
            str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 2
    payload = json.loads(json_out.read_text())
    assert payload["ready_for_modal_dispatch_command"] is False
    assert payload["estimated_cost_usd"] == 14.16
    assert payload["modal_function_timeout_hours"] == 24.0
    assert payload["validation_errors"] == [
        "timeout_hours_must_match_modal_function_timeout:requested=1.50:actual=24.00"
    ]
    assert claim_file.read_text() == before_claims


def test_modal_t1_claims_before_spawn_and_passes_claim_ledger_to_remote() -> None:
    text = _source()
    main_src = text[text.index("@app.local_entrypoint()"):text.index("def _returncode_is_zero")]

    claim_idx = main_src.index("claim_rc = _claim_lane(")
    ledger_idx = main_src.index("claim_ledger_bytes = _read_claims_ledger_bytes()")
    spawn_idx = main_src.index("call = run_t1_balle_modal.spawn(")
    assert claim_idx < ledger_idx < spawn_idx
    assert "aborting before GPU spend" in main_src
    assert "claim_ledger_bytes=claim_ledger_bytes" in main_src


def test_modal_t1_remote_runs_existing_script_with_score_domain_exact_eval_env() -> None:
    text = _source()
    remote_src = text[text.index("def run_t1_balle_modal("):text.index("def _write_dispatch_metadata")]

    assert '"T1_ALLOW_SCORE_DOMAIN_TRAINING": "1"' in remote_src
    assert '"T1_RUN_CONTEST_CUDA_AUTH_EVAL": "1"' in remote_src
    assert '"LOCAL_CUDA_WORKER": "1"' in remote_src
    assert '"T1_DISPATCH_INSTANCE_JOB_ID": instance_job_id' in remote_src
    assert '"T1_DISPATCH_CLAIMS_PATH": str(claim_path)' in remote_src
    assert "remote_lane_t1_balle_endtoend.sh" in remote_src
    assert "SEGMENTATION_SURROGATE" in remote_src
    assert "sinkhorn" in remote_src


def test_recover_uses_auth_eval_schema_and_600_sample_blockers() -> None:
    text = _source()
    recover_src = text[text.index("def _contest_cuda_score_claim_from_result"):text.index("def recover(")]
    recover_body = text[text.index("def recover("):]

    assert "required_contest_cuda_evidence_blockers" in recover_src
    assert "expected_n_samples=600" in recover_src
    assert "t1_remote_adjudication_score_claim_not_true" in recover_src
    assert "t1_remote_summary_score_claim_not_true" in recover_src
    assert "completed_t1_contest_cuda_recovered" in recover_body
    assert "failed_t1_modal_recovered_no_score_claim" in recover_body
    assert "promotion_eligible\": False" in recover_body


def test_score_claim_helper_rejects_training_only_result() -> None:
    module = _load_module()

    score_claim, blockers, metrics = module._contest_cuda_score_claim_from_result(
        {
            "returncode": 0,
            "summary": {
                "score_claim": False,
                "stage": "completed_t1_score_domain_training_no_score_claim",
            },
            "eval_data": None,
            "auth_eval_adjudication": None,
        }
    )

    assert score_claim is False
    assert "t1_remote_adjudication_score_claim_not_true" in blockers
    assert "t1_remote_summary_score_claim_not_true" in blockers
    assert metrics["score"] is None


def test_score_claim_helper_uses_adjudicated_packet_bytes_not_self_reported_eval_size() -> None:
    module = _load_module()

    score_claim, blockers, metrics = module._contest_cuda_score_claim_from_result(
        {
            "returncode": 0,
            "summary": {"score_claim": True},
            "auth_eval_adjudication": {
                "score_claim": True,
                "packet_archive_sha256": "packet-sha",
                "packet_archive_size_bytes": 111,
                "blockers": [],
            },
            "eval_data": {
                "canonical_score": 0.2,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "score_rate_contribution": 0.01,
                "rate_unscaled": 0.001,
                "archive_size_bytes": 222,
                "n_samples": 600,
                "canonical_score_source": "score_recomputed_from_components",
                "lane_tag": "[contest-CUDA]",
                "score_axis": "contest_cuda",
                "evidence_semantics": "contest_cuda_exact_auth_eval",
                "score_claim_valid": True,
                "provenance": {
                    "device": "cuda",
                    "gpu_t4_match": True,
                    "archive_sha256": "eval-sha",
                },
            },
        }
    )

    assert score_claim is False
    assert metrics["archive_size_bytes"] == 222
    assert "archive_size_bytes_mismatch:manifest=222:actual=111" in blockers
    assert "t1_recover_archive_sha_mismatch_adjudication_vs_eval" in blockers
    assert "t1_recover_archive_size_mismatch_adjudication_vs_eval" in blockers


def test_score_claim_helper_requires_adjudicated_packet_custody_fields() -> None:
    module = _load_module()

    score_claim, blockers, _metrics = module._contest_cuda_score_claim_from_result(
        {
            "returncode": 0,
            "summary": {"score_claim": True},
            "auth_eval_adjudication": {"score_claim": True, "blockers": []},
            "eval_data": {
                "canonical_score": 0.2,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "score_rate_contribution": 0.01,
                "rate_unscaled": 0.001,
                "archive_size_bytes": 222,
                "n_samples": 600,
                "canonical_score_source": "score_recomputed_from_components",
                "lane_tag": "[contest-CUDA]",
                "score_axis": "contest_cuda",
                "evidence_semantics": "contest_cuda_exact_auth_eval",
                "score_claim_valid": True,
                "provenance": {"device": "cuda", "gpu_t4_match": True},
            },
        }
    )

    assert score_claim is False
    assert "t1_remote_adjudication_packet_archive_size_bytes_missing" in blockers
    assert "t1_remote_adjudication_packet_archive_sha256_missing" in blockers


def test_modal_t1_metadata_starts_as_no_score_claim(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "RESULT_ROOT", tmp_path)

    path = module._write_dispatch_metadata(
        instance_job_id="unit-t1-dispatch",
        call_id="fc-unit",
        params={"epochs": 1},
        estimated_cost_usd=0.59,
        predicted_eta_utc="2026-05-10T00:00:00Z",
    )

    payload = json.loads(path.read_text())
    assert payload["lane_id"] == "t1_balle_128k_endtoend"
    assert payload["call_id"] == "fc-unit"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["expected_n_samples"] == 600
    assert payload["canonical_path"].endswith("upstream/evaluate.py --device cuda")
    assert payload["mounted_code_snapshot"]["schema_version"] == "mounted_modal_code_snapshot_v1"
    assert "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py" in payload[
        "mounted_code_snapshot"
    ]["mounted_code_paths"]
    assert (tmp_path / "unit-t1-dispatch" / "code_snapshot" / "mounted_code_status.txt").is_file()


def test_modal_t1_records_mounted_code_snapshot_for_score_bearing_dispatches() -> None:
    text = _source()

    assert "MOUNTED_CODE_PATHS = (" in text
    assert "def _mounted_code_snapshot(" in text
    assert '"scripts/remote_lane_t1_balle_endtoend.sh"' in text
    assert '"tools/build_phase1_packet_compiler.py"' in text
    assert '"mounted_code_snapshot": code_snapshot' in text
    assert "mounted_code_worktree.patch" in text
