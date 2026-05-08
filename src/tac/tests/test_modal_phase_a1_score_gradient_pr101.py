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


def test_modal_phase_a1_claims_lane_before_spawn() -> None:
    text = _source()

    claim_idx = text.index("claim_rc = _claim_lane(")
    spawn_idx = text.index("call = run_phase_a1_t4.spawn(")
    assert claim_idx < spawn_idx
    assert "aborting before GPU spend" in text


def test_modal_phase_a1_requires_t4_dali_and_nvdec_preflight() -> None:
    text = _source()

    assert "nvidia.dali" in text
    assert "nvdec_probe_passed" in text
    assert "gpu_t4_match" in text
    assert "refusing CPU fallback" in text
    assert "continue_after_nvdec_failure" in text
    assert "completed_training_build_cuda_eval_skipped_nvdec_preflight" in text
    assert "score_claim_valid\": False" in text


def test_modal_phase_a1_recover_preserves_harvest_command_and_ttl() -> None:
    text = _source()

    assert "Modal result-cache TTL" in text
    assert "tools/harvest_modal_calls.py" in text
    assert "recover --label <instance_job_id>" in text


def test_modal_phase_a1_recover_closes_terminal_claim_rows() -> None:
    text = _source()
    recover = text[text.index("def recover("):]

    assert "completed_modal_recovered" in text
    assert "failed_modal_recovered" in text
    assert "failed_modal_result_cache_expired" in text
    assert "_returncode_is_zero(rc)" in recover
    assert recover.index("result = fc.get(timeout=2)") < recover.index(
        "_close_modal_recovery_claim("
    )


def test_modal_phase_a1_direct_plan_does_not_spawn_or_claim(tmp_path: Path) -> None:
    pr101_archive = tmp_path / "archive.zip"
    pr101_archive.write_bytes(b"not-a-real-archive-but-custody-bytes")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
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
    assert payload["params"]["continue_after_nvdec_failure"] is True
    assert ".venv/bin/modal" in payload["dispatch_command"]
    assert "--detach" in payload["dispatch_command"]
    assert "--continue-after-nvdec-failure" in payload["dispatch_command"]
    after_claims = claim_file.read_text() if claim_file.exists() else None
    assert after_claims == before_claims
