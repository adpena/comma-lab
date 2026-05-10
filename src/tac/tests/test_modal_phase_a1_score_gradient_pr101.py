import ast
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "experiments" / "modal_phase_a1_score_gradient_pr101.py"


def _source() -> str:
    return TOOL.read_text()


def _executable_source() -> str:
    text = _source()
    module = ast.parse(text)
    if not module.body:
        return text
    first = module.body[0]
    if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
        return text
    if not isinstance(first.value.value, str):
        return text
    lines = text.splitlines()
    start = first.lineno - 1
    end = first.end_lineno or first.lineno
    return "\n".join(lines[:start] + lines[end:])


def test_modal_phase_a1_dispatcher_is_cuda_auth_eval_only() -> None:
    text = _source()
    executable = _executable_source()

    assert "contest_auth_eval.py" in text
    assert '"--device", "cuda"' in text
    assert "AUTH_EVAL_DEVICE=cpu" not in executable
    assert "MODAL_AUTH_EVAL_ADVISORY_ONLY" not in executable
    assert '"auth_eval_advisory_only": False' in text


def test_modal_phase_a1_training_receives_pr101_source_dir() -> None:
    text = _source()
    train_cmd = text[
        text.index('str(REMOTE_REPO / "experiments/train_score_gradient_pr101_finetune.py")'):
        text.index('run = _run_logged_remote(\n            "stage1_train"')
    ]

    assert '"--pr101-source-dir", str(pr101_source_dir)' in train_cmd


def test_modal_phase_a1_uses_strict_contest_cuda_evidence_blockers() -> None:
    text = _source()

    assert "required_contest_cuda_evidence_blockers" in text
    assert "score_claim = eval_rc == 0 and not metric_blockers" in text
    assert '"evidence_grade": "[contest-CUDA]" if score_claim else "[exact-eval incomplete]"' in text
    assert '"score_claim_valid": score_claim' in text
    assert '"ready_for_exact_eval_dispatch": False' in text
    assert '"exact_cuda_eval_complete": score_claim' in text
    assert '"eval_archive_size_bytes": metrics["archive_size_bytes"]' in text
    assert '"n_samples": metrics["n_samples"]' in text


def test_modal_phase_a1_recover_is_fail_closed_for_score_claims() -> None:
    text = _source()
    recover = text[text.index("def recover("):]

    assert "def _recover_evidence_summary(" in text
    assert "auth_eval_blockers = required_contest_cuda_evidence_blockers(" in text
    assert "claim_blockers = _unique_strings(auth_eval_blockers + remote_claim_blockers)" in text
    assert "remote_build_manifest_score_claim_valid_not_true" in text
    assert 'if evidence["score_claim_valid"]:' in recover
    assert '"score_claim_valid": evidence["score_claim_valid"]' in recover
    assert '"exact_cuda_eval_complete": evidence["exact_cuda_eval_complete"]' in recover
    assert '"claim_blockers": evidence["claim_blockers"]' in recover
    assert "[exact-eval incomplete]" in recover
    assert 'score_claim_valid=bool(evidence["score_claim_valid"])' in recover
    assert 'return 0 if _returncode_is_zero(rc) and evidence["score_claim_valid"] else 1' in recover


def test_modal_phase_a1_dispatch_prints_forecast_not_cuda_evidence() -> None:
    text = _source()

    assert "[planning forecast; not evidence]" in text
    assert 'predicted band:  [{predicted_low}, {predicted_high}] [contest-CUDA]' not in text


def test_modal_phase_a1_claims_lane_before_spawn() -> None:
    text = _source()

    claim_idx = text.index("claim_rc = _claim_lane(")
    spawn_idx = text.index("call = run_phase_a1_t4.spawn(")
    assert claim_idx < spawn_idx
    assert "aborting before GPU spend" in text


def test_modal_phase_a1_records_mounted_code_snapshot_before_dispatch_metadata() -> None:
    text = _source()

    assert "MOUNTED_CODE_PATHS = (" in text
    assert "def _mounted_code_snapshot(" in text
    assert '"experiments/train_score_gradient_pr101_finetune.py"' in text
    assert '"tools/build_pr101_finetuned_archive.py"' in text
    assert '"mounted_code_snapshot": code_snapshot' in text
    metadata_src = text[text.index("def _write_dispatch_metadata("):text.index("@app.local_entrypoint()")]
    assert metadata_src.index("code_snapshot = _mounted_code_snapshot(out_dir)") < metadata_src.index(
        '"mounted_code_snapshot": code_snapshot'
    )
    assert "mounted_code_worktree.patch" in text
    assert "recorded patch artifacts or by rerunning from a clean commit" in text


def test_modal_phase_a1_requires_t4_dali_and_nvdec_preflight() -> None:
    text = _source()

    assert "nvidia.dali" in text
    assert "nvdec_probe_passed" in text
    assert "gpu_t4_match" in text
    assert "refusing CPU fallback" in text
    assert "continue_after_nvdec_failure" in text
    assert "completed_training_build_cuda_eval_skipped_nvdec_preflight" in text
    assert "score_claim_valid\": False" in text


def test_modal_phase_a1_disables_dali_nvml_on_modal() -> None:
    text = _source()

    runtime_text = (REPO_ROOT / "src/tac/deploy/modal/runtime.py").read_text()
    assert 'DALI_DISABLE_NVML_VALUE = "1"' in runtime_text
    assert 'PYTORCH_CUDA_ALLOC_CONF_VALUE = "expandable_segments:True"' in runtime_text
    assert "REMOTE_PYTHONPATH =" in text
    assert ".env(" in text
    assert '"DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE' in text
    assert '"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE' in text
    assert '"PYTHONPATH": REMOTE_PYTHONPATH' in text
    assert 'os.environ["DALI_DISABLE_NVML"] = env["DALI_DISABLE_NVML"]' in text
    assert (
        'os.environ["PYTORCH_CUDA_ALLOC_CONF"] = env["PYTORCH_CUDA_ALLOC_CONF"]'
        in text
    )
    run_image = text[text.index("run_image = ("):text.index("def _json_bytes")]
    assert run_image.index(".env(") < run_image.index(".add_local_")


def test_modal_phase_a1_remote_outputs_are_not_temp_score_evidence() -> None:
    text = _source()

    assert 'REMOTE_OUT_ROOT = Path("/tmp' not in text
    assert 'REMOTE_OUT_ROOT = REMOTE_REPO / "experiments/results/modal_phase_a1_remote"' in text
    assert "--allow-temp-work-dir" not in text


def test_modal_phase_a1_recover_preserves_harvest_command_and_ttl() -> None:
    text = _source()

    assert "Modal result-cache TTL" in text
    assert "tools/harvest_modal_calls.py" in text
    assert "recover --label <instance_job_id>" in text


def test_modal_phase_a1_recover_closes_terminal_claim_rows() -> None:
    text = _source()
    recover = text[text.index("def recover("):]

    assert "completed_modal_contest_cuda_recovered" in text
    assert "failed_modal_recovered_no_score_claim" in text
    assert "failed_modal_recovered" in text
    assert "failed_modal_result_cache_expired" in text
    assert "_returncode_is_zero(rc)" in recover
    assert recover.index("result = fc.get(timeout=2)") < recover.index(
        "_close_modal_recovery_claim("
    )


def test_modal_phase_a1_timeout_budget_is_not_understated() -> None:
    text = _source()

    assert "DEFAULT_TIMEOUT_HOURS = 6.0" in text
    assert "MODAL_TIMEOUT_SAFETY_SECONDS = 10 * 60" in text
    assert "def _modal_timeout_validation_errors(" in text
    assert "stage_timeouts_exceed_modal_budget" in text
    assert "timeout_hours_must_match_modal_function_timeout" in text
    assert "estimated_cost = HOURLY_RATE_T4_USD * DEFAULT_TIMEOUT_HOURS" in text


def test_modal_phase_a1_direct_plan_does_not_spawn_or_claim(tmp_path: Path) -> None:
    pr101_archive = tmp_path / "archive.zip"
    pr101_archive.write_bytes(b"not-a-real-archive-but-custody-bytes")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "codec.py").write_text("VALUE = 0\n")
    (source_dir / "model.py").write_text("VALUE = 1\n")
    video_path = tmp_path / "0.mkv"
    video_path.write_bytes(b"fake-video-bytes")
    json_out = tmp_path / "plan.json"
    claim_file = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    before_claims = claim_file.read_text() if claim_file.exists() else None

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "plan",
            "--pr101-archive", str(pr101_archive),
            "--pr101-source-dir", str(source_dir),
            "--video-path", str(video_path),
            "--label", "unit-test-plan",
            "--epochs", "3",
            "--checkpoint-selection", "best_proxy",
            "--continue-after-nvdec-failure",
            "--json-out", str(json_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(json_out.read_text())
    stdout_payload = json.loads(proc.stdout)
    assert stdout_payload == payload
    assert payload["instance_job_id"] == "unit-test-plan"
    assert payload["dispatch_attempted"] is False
    assert payload["remote_or_gpu_eval_started"] is False
    assert payload["lane_claim_opened"] is False
    assert payload["modal_app_creation_not_attempted"] is True
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_modal_dispatch_command"] is True
    assert payload["validation_errors"] == []
    assert payload["params"]["epochs"] == 3
    assert payload["params"]["checkpoint_selection"] == "best_proxy"
    assert payload["params"]["continue_after_nvdec_failure"] is True
    assert ".venv/bin/modal" in payload["dispatch_command"]
    assert "--detach" in payload["dispatch_command"]
    assert "--checkpoint-selection" in payload["dispatch_command"]
    assert "best_proxy" in payload["dispatch_command"]
    assert "--continue-after-nvdec-failure" in payload["dispatch_command"]
    after_claims = claim_file.read_text() if claim_file.exists() else None
    assert after_claims == before_claims


def test_modal_phase_a1_direct_plan_rejects_parent_pr101_source_dir(tmp_path: Path) -> None:
    pr101_archive = tmp_path / "archive.zip"
    pr101_archive.write_bytes(b"archive")
    parent = tmp_path / "source"
    nested = parent / "submissions" / "hnerv_ft_microcodec" / "src"
    nested.mkdir(parents=True)
    (nested / "codec.py").write_text("VALUE = 1\n")
    (nested / "model.py").write_text("VALUE = 2\n")
    video_path = tmp_path / "0.mkv"
    video_path.write_bytes(b"video")
    claim_file = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    before_claims = claim_file.read_text() if claim_file.exists() else None

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "plan",
            "--pr101-archive", str(pr101_archive),
            "--pr101-source-dir", str(parent),
            "--video-path", str(video_path),
            "--label", "bad-source-root",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert proc.returncode == 1
    assert "PR101 source dir must contain codec.py and model.py" in proc.stderr
    assert str(nested) in proc.stderr
    after_claims = claim_file.read_text() if claim_file.exists() else None
    assert after_claims == before_claims
