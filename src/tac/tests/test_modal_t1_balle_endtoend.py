from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "modal_t1_balle_endtoend.py"
REMOTE_SCRIPT = REPO_ROOT / "scripts" / "remote_lane_t1_balle_endtoend.sh"
RUNTIME_SCRIPT = REPO_ROOT / "src" / "tac" / "deploy" / "modal" / "runtime.py"


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
    assert payload["canonical_a1_payload"]["ready_for_modal_mount"] is True
    assert "archive" in payload["canonical_a1_payload"]["files"]
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


def test_modal_t1_spawn_exception_keeps_claim_nonterminal_for_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "RESULT_ROOT", tmp_path)
    claim_calls: list[dict[str, object]] = []

    def fake_claim_lane(**kwargs):
        claim_calls.append(kwargs)
        return 0

    monkeypatch.setattr(module, "_claim_lane", fake_claim_lane)

    try:
        raise RuntimeError("server call may already exist")
    except RuntimeError as exc:
        record_path = module._mark_modal_spawn_submission_unknown(
            instance_job_id="unit-spawn-ambiguous",
            predicted_eta_utc="2026-05-10T00:00:00Z",
            exc=exc,
        )

    assert claim_calls == [
        {
            "instance_job_id": "unit-spawn-ambiguous",
            "predicted_eta_utc": "2026-05-10T00:00:00Z",
            "notes": claim_calls[0]["notes"],
            "status": module.MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS,
            "force": True,
        }
    ]
    assert not module.MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS.startswith(
        ("completed_", "failed_", "refused_dispatch", "stopped_", "stale_")
    )
    assert "manual Modal call reconciliation" in str(claim_calls[0]["notes"])

    payload = json.loads(record_path.read_text())
    assert payload["call_id"] is None
    assert payload["terminal_claim_closed"] is False
    assert payload["server_side_call_existence"] == "unknown"
    assert payload["recovery_required"] is True
    assert payload["claim_update_rc"] == 0


def test_modal_t1_spawn_exception_path_does_not_terminal_close_claim() -> None:
    text = _source()
    main_src = text[text.index("@app.local_entrypoint()"):text.index("def _returncode_is_zero")]

    assert "failed_modal_spawn_submission" not in main_src
    assert "except Exception as exc:" in main_src
    assert "MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS" in main_src
    assert "_mark_modal_spawn_submission_unknown(" in main_src
    assert "Lane claim left open" in main_src


def test_modal_t1_remote_runs_existing_script_with_score_domain_exact_eval_env() -> None:
    text = _source()
    remote_src = text[text.index("def run_t1_balle_modal("):text.index("def _write_dispatch_metadata")]

    assert "t1_modal_import_probe" in remote_src
    assert "CONTEST_SCORER_IMPORT_PROBE_MODULES" in text
    assert "remote_import_probe_failed" in remote_src
    assert "missing_canonical_a1_payload" in remote_src
    assert "A1_CANONICAL_REMOTE_PATH" in remote_src
    assert "A1_DESIGNATION_REMOTE_PATH" in remote_src
    assert '"T1_ALLOW_SCORE_DOMAIN_TRAINING": "1"' in remote_src
    assert '"T1_RUN_CONTEST_CUDA_AUTH_EVAL": "1"' in remote_src
    assert '"LOCAL_CUDA_WORKER": "1"' in remote_src
    assert '"T1_DISPATCH_INSTANCE_JOB_ID": instance_job_id' in remote_src
    assert '"T1_DISPATCH_CLAIMS_PATH": str(claim_path)' in remote_src
    assert "remote_lane_t1_balle_endtoend.sh" in remote_src
    assert "SEGMENTATION_SURROGATE" in remote_src
    assert "sinkhorn" in remote_src
    assert '"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE' in remote_src
    assert '"T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head' in remote_src
    assert '"SINKHORN_MAX_POSITIONS_PER_CHUNK": str(int(sinkhorn_max_positions_per_chunk))' in remote_src


def _extract_remote_shell_function(name: str) -> str:
    text = REMOTE_SCRIPT.read_text()
    start = text.index(f"{name}() {{")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def test_remote_t1_git_probe_normalizes_no_git_and_multiline_output() -> None:
    functions = "\n\n".join(
        [
            _extract_remote_shell_function("normalize_git_probe_output"),
            _extract_remote_shell_function("read_git_probe_value"),
        ]
    )
    valid_head = "0123456789abcdef0123456789abcdef01234567"
    probe = f"""
set -euo pipefail
{functions}
test "$(normalize_git_probe_output head $'HEAD\\nunknown')" = unknown
test "$(normalize_git_probe_output branch $'HEAD\\nunknown')" = unknown
test "$(normalize_git_probe_output head 'not-a-sha')" = unknown
test "$(normalize_git_probe_output branch HEAD)" = unknown
test "$(normalize_git_probe_output head '{valid_head}')" = "{valid_head}"
test "$(normalize_git_probe_output branch main)" = main
"""
    result = subprocess.run(
        ["bash", "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_remote_t1_unknown_local_git_uses_declared_mounted_metadata() -> None:
    text = REMOTE_SCRIPT.read_text()

    assert 'LOCAL_BRANCH="$(read_git_probe_value branch git -C "$WORKSPACE" branch --show-current)"' in text
    assert 'LOCAL_HEAD="$(read_git_probe_value head git -C "$WORKSPACE" rev-parse HEAD)"' in text
    assert 'if [ "$LOCAL_HEAD" != "unknown" ] && [ "$LOCAL_HEAD" != "$DECLARED_MOUNTED_GIT_HEAD" ]; then' in text
    assert 'if [ "$LOCAL_BRANCH" != "main" ] && [ "$LOCAL_BRANCH" != "unknown" ]; then' in text
    assert '"declared_mounted_git_branch": "${DECLARED_MOUNTED_GIT_BRANCH}"' in text
    assert '"declared_mounted_git_head": "${DECLARED_MOUNTED_GIT_HEAD}"' in text


def test_modal_t1_image_installs_scorer_runtime_dependencies() -> None:
    text = _source()
    runtime_src = RUNTIME_SCRIPT.read_text()

    assert "build_contest_cuda_base_image(" in text
    assert '"safetensors"' in runtime_src
    assert '"segmentation-models-pytorch"' in runtime_src
    assert '"timm"' in runtime_src
    assert '"einops"' in runtime_src
    assert '"nvidia-dali-cuda120==1.52.0"' in runtime_src
    assert '"compressai==1.2.8"' in text
    assert 'PYTORCH_CUDA_ALLOC_CONF_VALUE = "expandable_segments:True"' in runtime_src
    assert '"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE' in text


def test_modal_t1_mounts_packet_compiler_bootstrap_dependency() -> None:
    text = _source()

    assert '"tools/build_phase1_packet_compiler.py"' in text
    assert '"tools/tool_bootstrap.py"' in text
    assert 'remote_path=str(REMOTE_REPO / "tools/tool_bootstrap.py")' in text


def test_modal_t1_guard_labels_require_bounded_guard_params() -> None:
    module = _load_module()

    payload, rc = module.build_local_plan(
        label="unit-guard-defaults",
        epochs=3000,
        batch_size=16,
        timeout_hours=24,
        cost_cap_usd=80,
        train_timeout_hours=module.DEFAULT_TRAIN_TIMEOUT_HOURS,
        max_target_pairs=None,
    )

    assert rc == 2
    assert payload["ready_for_modal_dispatch_command"] is False
    assert "guard_label_requires_epochs_lte_100:epochs=3000" in payload["validation_errors"]
    assert "guard_label_requires_batch_size_lte_1:batch_size=16" in payload["validation_errors"]
    assert (
        "guard_label_requires_max_target_pairs_lte_8:max_target_pairs=None"
        in payload["validation_errors"]
    )
    assert (
        "guard_label_requires_train_timeout_lte_3h:train_timeout_hours=22.50"
        in payload["validation_errors"]
    )


def test_modal_t1_guard_labels_accept_true_bounded_guard_params() -> None:
    module = _load_module()

    payload, rc = module.build_local_plan(
        label="unit-guard-bounded",
        epochs=50,
        batch_size=1,
        timeout_hours=24,
        cost_cap_usd=80,
        train_timeout_hours=2,
        max_target_pairs=8,
    )

    assert rc == 0
    assert payload["ready_for_modal_dispatch_command"] is True
    assert payload["params"]["epochs"] == 50
    assert payload["params"]["max_target_pairs"] == 8
    assert payload["params"]["sinkhorn_max_positions_per_chunk"] == 2048


def test_modal_t1_default_train_timeout_leaves_artifact_collection_buffer() -> None:
    module = _load_module()

    payload, rc = module.build_local_plan(
        label="unit-full-run",
        epochs=3000,
        batch_size=16,
        timeout_hours=24,
        cost_cap_usd=80,
        train_timeout_hours=module.DEFAULT_TRAIN_TIMEOUT_HOURS,
        max_target_pairs=None,
    )

    assert rc == 0
    assert payload["params"]["train_timeout_hours"] == module.DEFAULT_TRAIN_TIMEOUT_HOURS
    assert not any(
        err.startswith("train_timeout_leaves_no_modal_artifact_buffer")
        for err in payload["validation_errors"]
    )


def test_modal_t1_rejects_train_timeout_without_artifact_collection_buffer() -> None:
    module = _load_module()

    payload, rc = module.build_local_plan(
        label="unit-full-run",
        epochs=3000,
        batch_size=16,
        timeout_hours=24,
        cost_cap_usd=80,
        train_timeout_hours=23,
        max_target_pairs=None,
    )

    assert rc == 2
    assert (
        "train_timeout_leaves_no_modal_artifact_buffer:"
        "train_plus_eval_plus_artifact_buffer=24.25>24.00"
    ) in payload["validation_errors"]


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
    assert '"promotion_eligible": promotion_eligible' in recover_body
    assert '"promotion_blockers": promotion_blockers' in recover_body


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
                "canonical_score": 0.200147820688,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "score_rate_contribution": 0.000147820688,
                "rate_unscaled": 222 / 37_545_489,
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


def test_score_claim_helper_accepts_exact_evidence_when_promotion_ineligible() -> None:
    module = _load_module()

    score_claim, blockers, metrics = module._contest_cuda_score_claim_from_result(
        {
            "returncode": 0,
            "summary": {"score_claim": True},
            "auth_eval_adjudication": {
                "score_claim": True,
                "packet_archive_sha256": "packet-sha",
                "packet_archive_size_bytes": 222,
                "blockers": [],
                "promotion_eligible": False,
                "promotion_blockers": ["contest_cpu_eval_pending"],
            },
            "eval_data": {
                "canonical_score": 0.200147820688,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.001,
                "score_rate_contribution": 0.000147820688,
                "rate_unscaled": 222 / 37_545_489,
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
                    "archive_sha256": "packet-sha",
                },
            },
        }
    )

    assert score_claim is True
    assert metrics["score"] == 0.200147820688
    assert "t1_remote_adjudication_has_promotion_blockers" not in blockers
    assert "t1_remote_adjudication_promotion_eligible_false" not in blockers


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
    assert payload["canonical_a1_payload"]["ready_for_modal_mount"] is True
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


def test_modal_t1_plan_fails_closed_without_canonical_a1_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "A1_CANONICAL_LOCAL_PATH", tmp_path / "missing_a1")
    monkeypatch.setattr(module, "A1_DESIGNATION_LOCAL_PATH", tmp_path / "missing_memo.md")

    payload, rc = module.build_local_plan(
        label="missing-a1",
        epochs=1,
        batch_size=1,
        timeout_hours=24,
        cost_cap_usd=80,
        train_timeout_hours=2,
        max_target_pairs=4,
    )

    assert rc == 2
    assert payload["ready_for_modal_dispatch_command"] is False
    assert payload["canonical_a1_payload"]["ready_for_modal_mount"] is False
    assert any(
        item.startswith("canonical_a1_payload_canonical_dir_missing:")
        for item in payload["validation_errors"]
    )
    assert "canonical_a1_payload_archive_missing" in payload["validation_errors"]
