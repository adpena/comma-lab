from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
import datetime as dt
from pathlib import Path

import brotli
import pytest

from tac.hnerv_lowlevel_packer import sha256_bytes, write_stored_single_member_zip
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106SidecarPacket,
    emit_pr106_sidecar_packet,
)
from tac.repo_io import json_text

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "build_hnerv_lowlevel_exact_eval_packet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_hnerv_lowlevel_exact_eval_packet", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_refresh_dispatch_readiness_default_now_is_subprocess_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run_json_cmd(cmd, output):
        captured["cmd"] = cmd
        return {"ready_for_exact_eval_dispatch": True, "blockers": [], "stdout_tail": ""}

    monkeypatch.setattr(module, "_run_json_cmd", fake_run_json_cmd)
    result_dir = tmp_path / "packet"
    result_dir.mkdir()
    args = module.build_arg_parser().parse_args(
        [
            "--result-dir",
            str(result_dir),
            "--claims-path",
            str(tmp_path / "claims.md"),
            "--lane-id",
            "fixture_q10",
            "--job-name",
            "exact_eval_fixture_q10",
        ]
    )

    payload = module.refresh_dispatch_readiness(args)

    cmd = captured["cmd"]
    assert all(isinstance(item, str) for item in cmd)
    now_value = cmd[cmd.index("--now-utc") + 1]
    assert args.now_utc is None
    assert now_value.endswith("Z")
    dt.datetime.fromisoformat(now_value[:-1] + "+00:00")
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert any(row["code"] == "missing_active_lane_dispatch_claim" for row in payload["blockers"])


def test_lowlevel_exact_eval_packet_builds_static_release_surface(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "packet"
    packet_path = result_dir / "hnerv_lowlevel_exact_eval_packet.json"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--lane-id",
            "fixture_lgblock16",
            "--job-name",
            "exact_eval_fixture_lgblock16",
            "--claims-path",
            str(tmp_path / "claims.md"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
            "--json-out",
            str(packet_path),
        ]
    )

    packet = module.build_packet(args)

    assert packet["schema"] == "hnerv_lowlevel_exact_eval_operator_packet_v1"
    assert packet["score_claim"] is False
    assert packet["dispatch_attempted"] is False
    assert packet["remote_gpu_run"] is False
    assert "tools/build_hnerv_lowlevel_exact_eval_packet.py" in packet["code_paths"]
    assert "experiments/preflight_candidate_manifest_dispatch_readiness.py" in packet["code_paths"]
    assert "scripts/pre_submission_compliance_check.py" in packet["code_paths"]
    assert str(fixture["inflate_sh"]) in packet["source_paths"]
    assert str(fixture["runtime_py"]) in packet["source_paths"]
    assert packet["static_packet_ready"] is True
    assert packet["ready_for_exact_eval_dispatch_claim"] is True
    assert packet["ready_for_submit"] is False
    assert packet["static_blockers"] == []
    assert "missing_active_lane_dispatch_claim" in packet["submit_blockers"]
    assert "missing_operator_exact_cuda_approval" in packet["submit_blockers"]
    assert packet["score_blockers"] == [
        "exact_cuda_auth_eval_not_run_for_candidate",
        "contest_auth_eval_adjudication_not_run_for_candidate",
        "operator_score_claim_review_not_done",
    ]
    assert packet["byte_delta"] < 0
    assert packet["kkt_proof"]["status"] == "passed"
    assert packet["kkt_proof"]["proof_class"] == "discrete_rate_only_raw_equivalent_archive_repack"
    assert packet["kkt_proof"]["stationarity_residual"] == 0.0
    assert packet["kkt_proof"]["official_rate_score_delta"] == packet["expected_total_score_delta_rate_only"]

    release_surface = result_dir / "release_surface"
    assert _sha256(release_surface / "archive.zip") == fixture["candidate_archive_sha256"]
    assert os.access(release_surface / "inflate.sh", os.X_OK)
    report = (release_surface / "report.txt").read_text(encoding="utf-8")
    assert f"archive_sha256: {fixture['candidate_archive_sha256']}" in report
    assert "score_claim: false" in report
    assert "Lightning submit environment" in report
    assert "operator exact-CUDA approval" in report

    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "hnerv_lowlevel_exact_eval_candidate_manifest_v1"
    assert manifest["code_paths"] == packet["code_paths"]
    assert manifest["source_paths"] == packet["source_paths"]
    assert "src/tac/hnerv_lowlevel_packer.py" in manifest["code_paths"]
    assert manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"
    assert manifest["fixed_runtime_preflight"]["ready_for_fixed_runtime_exact_eval"] is True
    assert manifest["exact_eval_runtime_contract"]["ready_for_exact_eval_runtime"] is True
    runtime_tree = manifest["fixed_runtime_preflight"]["runtime_tree_sha256"]
    assert len(runtime_tree) == 64
    assert manifest["exact_eval_runtime_contract"]["runtime_tree_sha256"] == runtime_tree
    assert manifest["fixed_runtime_preflight"]["runtime_tree_source"].endswith("public_replay_preflight.json")
    assert manifest["exact_eval_runtime_contract"]["runtime_tree_source"].endswith("public_replay_preflight.json")
    assert manifest["kkt_proof"]["status"] == "passed"
    assert manifest["kkt_proof"]["byte_delta"] == packet["byte_delta"]

    claim_command = packet["commands"]["claim"]
    assert "PR106x lgblock16 1-byte" not in claim_command
    assert "fixture_lgblock16 HNeRV low-level Brotli exact CUDA eval" in claim_command
    assert f"byte_delta={packet['byte_delta']}" in claim_command

    payload_diff = json.loads((result_dir / "payload_section_diff_vs_source.json").read_text(encoding="utf-8"))
    assert payload_diff["changed_section_count"] == 2
    assert [row["name"] for row in payload_diff["sections"]] == [
        "packed_header_ff_len24",
        "decoder_packed_brotli",
    ]
    assert payload_diff["sections"][1]["byte_delta"] < 0
    assert payload_diff["blockers"] == []

    public_preflight = json.loads((result_dir / "public_replay_preflight.json").read_text(encoding="utf-8"))
    assert public_preflight["ready_for_exact_eval_dispatch"] is True
    compliance = json.loads((result_dir / "pre_submission_compliance.json").read_text(encoding="utf-8"))
    assert compliance["passed"] is True
    readiness = json.loads((result_dir / "dispatch_readiness_preflight.json").read_text(encoding="utf-8"))
    assert readiness["ready_for_exact_eval_dispatch"] is False
    assert readiness["stdout_tail"] == ""
    assert readiness["stdout_tail_disposition"].startswith("parsed_into_underlying_static_readiness_stdout")
    assert readiness["underlying_static_readiness_stdout"]["ready_for_exact_eval_dispatch"] is True
    assert readiness["lane_claim"]["active_claim_present"] is False
    assert any(
        row["code"] == "missing_active_lane_dispatch_claim"
        for row in readiness["blockers"]
    )
    assert (packet_path).is_file()


def test_lowlevel_release_surface_rebuild_clears_stale_custody_files(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "stale_packet"
    release_surface = result_dir / "release_surface"
    (release_surface / "stale_dir").mkdir(parents=True)
    (release_surface / "contest_auth_eval.json").write_text(
        '{"stale": true}\n',
        encoding="utf-8",
    )
    (release_surface / "stale_dir" / "old_runtime.py").write_text(
        "print('stale')\n",
        encoding="utf-8",
    )
    (release_surface / "old_runtime.py").write_text(
        "print('stale')\n",
        encoding="utf-8",
    )

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(release_surface),
            "--now-utc",
            "2026-05-07T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["static_packet_ready"] is True
    assert not (release_surface / "contest_auth_eval.json").exists()
    assert not (release_surface / "stale_dir").exists()
    assert not (release_surface / "old_runtime.py").exists()
    release_manifest = json.loads(
        (release_surface / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert release_manifest["contest_auth_eval"]["exists"] is False


def test_lowlevel_release_surface_refuses_repo_results_container(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "packet"
    result_dir.mkdir()

    with pytest.raises(ValueError, match="unsafe release-surface directory"):
        module._prepare_release_surface_dir(
            release_dir=REPO / "experiments" / "results",
            result_dir=result_dir,
            source_inflate=fixture["inflate_sh"],
        )


def test_lowlevel_release_surface_refuses_non_default_unowned_dir(
    tmp_path: Path,
) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "packet"
    release_dir = result_dir / "manual_surface"
    release_dir.mkdir(parents=True)
    (release_dir / "unrelated.txt").write_text("do not delete\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-default release-surface"):
        module._prepare_release_surface_dir(
            release_dir=release_dir,
            result_dir=result_dir,
            source_inflate=fixture["inflate_sh"],
        )
    assert (release_dir / "unrelated.txt").is_file()


def test_run_json_cmd_overwrites_stale_output_on_allowed_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "pre_submission_compliance.json"
    output.write_text('{"passed": true, "stale": true}\n', encoding="utf-8")

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["fixture"],
            returncode=7,
            stdout="old stdout",
            stderr="fresh failure",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    payload = module._run_json_cmd(
        [sys.executable, "scripts/pre_submission_compliance_check.py"],
        output,
        allow_failure=True,
    )

    assert payload["returncode"] == 7
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["schema"] == "subprocess_json_command_failure_v1"
    assert persisted["passed"] is False
    assert persisted["returncode"] == 7
    assert "fresh failure" in persisted["stderr_tail"]
    assert "stale" not in persisted


def test_lowlevel_exact_eval_packet_blocks_missing_raw_equivalence(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result = json.loads(fixture["result_json"].read_text(encoding="utf-8"))
    result["brotli_raw_equivalence"][0]["raw_equal"] = False
    fixture["result_json"].write_text(json_text(result), encoding="utf-8")
    result_dir = tmp_path / "blocked"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["static_packet_ready"] is False
    assert packet["kkt_proof"]["status"] == "blocked"
    assert "static_packet_not_ready" in packet["kkt_proof"]["blockers"]
    assert "raw_equivalence_missing:decoder_packed_brotli" in packet["kkt_proof"]["blockers"]
    assert "static_packet_not_ready" in packet["submit_blockers"]
    assert any(row["code"] == "payload_change_raw_equivalence_closed" for row in packet["static_blockers"])
    assert not (result_dir / "release_surface" / "archive.zip").exists()
    readiness_manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert readiness_manifest["static_packet_ready"] is False


def test_lowlevel_exact_eval_packet_records_operator_approval_without_dispatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in module.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "approved_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--lane-id",
            "fixture_q10",
            "--job-name",
            "exact_eval_fixture_q10",
            "--claims-path",
            str(tmp_path / "claims.md"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
            "--operator-approved-exact-cuda",
        ]
    )

    packet = module.build_packet(args)

    assert packet["operator_approved_exact_cuda"] is True
    assert packet["approved_exact_eval_target"] is True
    assert packet["ready_for_submit"] is False
    assert "missing_operator_exact_cuda_approval" not in packet["submit_blockers"]
    assert packet["submit_blockers"] == [
        "missing_lightning_environment",
        "missing_active_lane_dispatch_claim",
    ]
    assert packet["submit_blocker_disposition"]["method_failure"] is False
    assert (
        packet["submit_blocker_disposition"]["environment_status"]
        == "blocked_missing_lightning_environment"
    )
    assert packet["dispatch_attempted"] is False
    assert packet["remote_gpu_run"] is False
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["operator_approved_exact_cuda"] is True
    assert manifest["approved_exact_eval_target"] is True
    assert "requires_operator_exact_cuda_approval" not in manifest["submit_blockers_until_operator_action"]
    release_manifest = json.loads(
        (result_dir / "release_surface" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert release_manifest["operator_approved_exact_cuda"] is True
    assert release_manifest["approved_exact_eval_target"] is True
    assert "lightning_submit_environment" in release_manifest["remaining_required_for_score_or_dispatch"]
    assert "operator_exact_cuda_approval" not in release_manifest["remaining_required_for_score_or_dispatch"]
    report = (result_dir / "release_surface" / "report.txt").read_text(encoding="utf-8")
    assert "Lightning submit environment" in report
    assert "operator exact-CUDA approval" not in report


def test_lowlevel_exact_eval_packet_copies_valid_auth_eval_and_clears_eval_blockers(
    tmp_path: Path,
) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    auth_eval, score = _write_valid_cuda_auth_eval(tmp_path, fixture)
    result_dir = tmp_path / "auth_eval_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--auth-eval-json",
            str(auth_eval),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["static_packet_ready"] is True
    assert packet["auth_eval"]["valid_exact_cuda_candidate_eval"] is True
    assert packet["ready_for_submit"] is False
    assert "candidate_exact_cuda_auth_eval_already_present" in packet["submit_blockers"]
    assert packet["score_blockers"] == [
        "operator_score_claim_review_not_done",
        "contest_final_compliance_not_run",
        "successful_terminal_exact_cuda_claim_not_verified",
    ]
    release_auth_eval = result_dir / "release_surface" / "contest_auth_eval.json"
    assert release_auth_eval.is_file()
    assert _sha256(release_auth_eval) == _sha256(auth_eval)
    release_manifest = json.loads(
        (result_dir / "release_surface" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert release_manifest["auth_eval"]["valid_exact_cuda_candidate_eval"] is True
    assert release_manifest["contest_auth_eval"]["exists"] is True
    assert release_manifest["remaining_required_for_score_or_dispatch"] == [
        "operator_score_claim_review"
    ]
    report = (result_dir / "release_surface" / "report.txt").read_text(encoding="utf-8")
    assert "exact_cuda_eval_complete: true" in report
    assert f"exact_cuda_score: {score}" in report
    assert "remaining_for_score_or_dispatch: operator score-claim review" in report
    assert "Lightning submit environment" not in report


def test_lowlevel_exact_eval_packet_does_not_resubmit_after_valid_auth_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    auth_eval, _score = _write_valid_cuda_auth_eval(tmp_path, fixture)
    claims_path = tmp_path / "claims.md"
    claims_path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| 2026-05-07T15:42:13Z | codex:gpt-5.5 | fixture_q10 | lightning | "
        "exact_eval_fixture_q10 | 2026-05-07T16:00:00Z | claimed | active exact eval claim |\n",
        encoding="utf-8",
    )
    for name in module.REQUIRED_ENV:
        monkeypatch.setenv(name, f"fixture-{name.lower()}")
    result_dir = tmp_path / "auth_eval_no_resubmit_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--auth-eval-json",
            str(auth_eval),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--lane-id",
            "fixture_q10",
            "--job-name",
            "exact_eval_fixture_q10",
            "--claims-path",
            str(claims_path),
            "--now-utc",
            "2026-05-07T15:45:00Z",
            "--operator-approved-exact-cuda",
        ]
    )

    packet = module.build_packet(args)

    assert packet["auth_eval"]["valid_exact_cuda_candidate_eval"] is True
    assert packet["lane_claim_preflight"]["active_claim_present"] is True
    assert packet["missing_env"] == []
    assert packet["operator_approved_exact_cuda"] is True
    assert packet["ready_for_submit"] is False
    assert packet["submit_blockers"] == ["candidate_exact_cuda_auth_eval_already_present"]
    assert packet["score_blockers"] == [
        "operator_score_claim_review_not_done",
        "contest_final_compliance_not_run",
        "successful_terminal_exact_cuda_claim_not_verified",
    ]


def test_lowlevel_exact_eval_packet_rejects_auth_eval_formula_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    auth_eval, _score = _write_valid_cuda_auth_eval(tmp_path, fixture)
    payload = json.loads(auth_eval.read_text(encoding="utf-8"))
    payload["score_recomputed_from_components"] = 123.0
    payload["canonical_score"] = 123.0
    auth_eval.write_text(json_text(payload), encoding="utf-8")
    result_dir = tmp_path / "auth_eval_formula_blocked_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--auth-eval-json",
            str(auth_eval),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["auth_eval"]["valid_exact_cuda_candidate_eval"] is False
    assert "auth_eval_blocked:auth_eval_component_formula_mismatch" in packet["score_blockers"]


def test_lowlevel_exact_eval_packet_surfaces_terminal_env_refusal_as_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in module.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    fixture = _write_lowlevel_fixture(tmp_path)
    claims_path = tmp_path / "claims.md"
    claims_path.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance_job_id | predicted_eta_utc | status | notes |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| 2026-05-07T15:42:13Z | codex:gpt-5.5 | fixture_q10 | lightning | "
        "exact_eval_fixture_q10 |  | refused_dispatch_missing_lightning_env | "
        "operator env missing; no remote job submitted |\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "terminal_refusal_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--lane-id",
            "fixture_q10",
            "--job-name",
            "exact_eval_fixture_q10",
            "--claims-path",
            str(claims_path),
            "--now-utc",
            "2026-05-07T17:15:43Z",
            "--operator-approved-exact-cuda",
        ]
    )

    packet = module.build_packet(args)

    assert packet["ready_for_submit"] is False
    assert packet["lane_claim_preflight"]["active_claim_present"] is False
    assert packet["lane_claim_preflight"]["latest_matching_terminal_status"] == (
        "refused_dispatch_missing_lightning_env"
    )
    assert packet["lane_claim_preflight"]["matching_terminal_claims"][0]["claim_status"] == (
        "refused_dispatch_missing_lightning_env"
    )
    disposition = packet["submit_blocker_disposition"]
    assert disposition["method_failure"] is False
    assert disposition["latest_matching_terminal_status"] == "refused_dispatch_missing_lightning_env"
    assert "not method" in disposition["environment_disposition"]
    assert disposition["lane_claim_status"] == "missing_active_claim"
    readiness = json.loads((result_dir / "dispatch_readiness_preflight.json").read_text(encoding="utf-8"))
    assert readiness["lane_claim"]["latest_matching_terminal_status"] == (
        "refused_dispatch_missing_lightning_env"
    )


def test_lowlevel_exact_eval_packet_cli_materializes_packet(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path)
    result_dir = tmp_path / "cli_packet"
    packet_path = result_dir / "packet.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
            "--json-out",
            str(packet_path),
        ],
        check=True,
        cwd=REPO,
        text=True,
    )

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["static_packet_ready"] is True
    assert packet["artifacts"]["release_surface"].endswith("release_surface")


def test_lowlevel_exact_eval_packet_accepts_public_pr106_member_name(tmp_path: Path) -> None:
    fixture = _write_lowlevel_fixture(tmp_path, member_name="0.bin")
    result_dir = tmp_path / "public_member_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--now-utc",
            "2026-05-07T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["static_packet_ready"] is True
    compliance = json.loads((result_dir / "pre_submission_compliance.json").read_text(encoding="utf-8"))
    assert compliance["passed"] is True
    release_manifest = json.loads(
        (result_dir / "release_surface" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    assert "tools/build_hnerv_lowlevel_exact_eval_packet.py" in release_manifest["code_paths"]
    assert str(fixture["inflate_sh"]) in release_manifest["source_paths"]
    assert release_manifest["archive"]["path"] == "archive.zip"


def test_lowlevel_exact_eval_packet_accepts_pr106_sidecar_wrapper_payload(
    tmp_path: Path,
) -> None:
    fixture = _write_lowlevel_fixture(tmp_path, member_name="0.bin", sidecar_wrapper=True)
    result_dir = tmp_path / "pr106_sidecar_packet"

    args = module.build_arg_parser().parse_args(
        [
            "--candidate-result",
            str(fixture["result_json"]),
            "--archive",
            str(fixture["candidate_archive"]),
            "--baseline-json",
            str(fixture["baseline_json"]),
            "--inflate-sh",
            str(fixture["inflate_sh"]),
            "--upstream-dir",
            str(fixture["upstream_dir"]),
            "--result-dir",
            str(result_dir),
            "--release-surface-dir",
            str(result_dir / "release_surface"),
            "--lane-id",
            "pr106_sidecar_fixture",
            "--job-name",
            "exact_eval_pr106_sidecar_fixture",
            "--claims-path",
            str(tmp_path / "claims.md"),
            "--now-utc",
            "2026-05-13T12:00:00Z",
        ]
    )

    packet = module.build_packet(args)

    assert packet["static_packet_ready"] is True
    assert packet["static_blockers"] == []
    assert packet["archive_identity"]["sha256"] == fixture["candidate_archive_sha256"]
    assert packet["kkt_proof"]["status"] == "passed"
    release_surface = result_dir / "release_surface"
    assert _sha256(release_surface / "archive.zip") == fixture["candidate_archive_sha256"]
    public_preflight = json.loads((result_dir / "public_replay_preflight.json").read_text(encoding="utf-8"))
    assert public_preflight["ready_for_exact_eval_dispatch"] is True


def _write_lowlevel_fixture(
    root: Path,
    *,
    member_name: str = "x",
    sidecar_wrapper: bool = False,
) -> dict[str, object]:
    source_decoder_raw = (b"decoder-record-" * 3000) + b"source"
    latent_raw = b"latent-row-" * 2000
    source_decoder = brotli.compress(source_decoder_raw, quality=1)
    candidate_decoder = brotli.compress(source_decoder_raw, quality=11, lgblock=16)
    if len(candidate_decoder) >= len(source_decoder):
        candidate_decoder = brotli.compress(source_decoder_raw, quality=10, lgblock=16)
    assert len(candidate_decoder) < len(source_decoder)
    latents = brotli.compress(latent_raw, quality=5)
    source_inner_payload = _packed_payload(source_decoder, latents)
    candidate_inner_payload = _packed_payload(candidate_decoder, latents)
    if sidecar_wrapper:
        sidecar_payload = brotli.compress(b"fixture-sidecar-payload", quality=5)
        source_payload = emit_pr106_sidecar_packet(
            PR106SidecarPacket(
                format_id=PR106_SIDECAR_FORMAT_BROTLI,
                pr106_bytes=source_inner_payload,
                sidecar_payload=sidecar_payload,
            )
        )
        candidate_payload = emit_pr106_sidecar_packet(
            PR106SidecarPacket(
                format_id=PR106_SIDECAR_FORMAT_BROTLI,
                pr106_bytes=candidate_inner_payload,
                sidecar_payload=sidecar_payload,
            )
        )
    else:
        source_payload = source_inner_payload
        candidate_payload = candidate_inner_payload

    source_archive = root / "source.zip"
    candidate_archive = root / "candidate.zip"
    write_stored_single_member_zip(source_archive, member_name=member_name, payload=source_payload)
    write_stored_single_member_zip(candidate_archive, member_name=member_name, payload=candidate_payload)

    inflate_sh = _write_runtime(root / "runtime")
    upstream_dir = root / "upstream"
    upstream_dir.mkdir()
    (upstream_dir / "evaluate.py").write_text("print('fixture evaluate')\n", encoding="utf-8")
    baseline_json = root / "baseline.json"
    baseline_json.write_text("{}\n", encoding="utf-8")
    result_json = root / "result.json"
    result = _candidate_result(
        source_archive,
        candidate_archive,
        source_payload,
        candidate_payload,
        member_name=member_name,
        source_inner_payload=source_inner_payload,
        candidate_inner_payload=candidate_inner_payload,
    )
    result_json.write_text(json_text(result), encoding="utf-8")
    return {
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": _sha256(candidate_archive),
        "result_json": result_json,
        "inflate_sh": inflate_sh,
        "runtime_py": inflate_sh.parent / "inflate.py",
        "upstream_dir": upstream_dir,
        "baseline_json": baseline_json,
    }


def _write_runtime(root: Path) -> Path:
    root.mkdir(parents=True)
    inflate_sh = root / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'python "$(dirname "$0")/inflate.py" "$@"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)
    (root / "inflate.py").write_text("print('fixture inflate')\n", encoding="utf-8")
    return inflate_sh


def _write_valid_cuda_auth_eval(root: Path, fixture: dict[str, object]) -> tuple[Path, float]:
    auth_eval = root / "contest_auth_eval.json"
    archive_path = fixture["candidate_archive"]
    assert isinstance(archive_path, Path)
    archive_sha256 = fixture["candidate_archive_sha256"]
    assert isinstance(archive_sha256, str)
    archive_bytes = archive_path.stat().st_size
    seg = 0.0001
    pose = 0.0002
    rate = archive_bytes / 37_545_489
    score = 100.0 * seg + (10.0 * pose) ** 0.5 + 25.0 * rate
    auth_eval.write_text(
        json_text(
            {
                "provenance": {
                    "archive_sha256": archive_sha256,
                    "archive_size_bytes": archive_bytes,
                    "device": "cuda",
                    "gpu_model": "Tesla T4",
                    "gpu_t4_match": True,
                },
                "archive_size_bytes": archive_bytes,
                "score_claim_valid": True,
                "exact_cuda_eval_complete": True,
                "lane_tag": "[contest-CUDA]",
                "score_axis": "contest_cuda",
                "score_recomputed_from_components": score,
                "canonical_score": score,
                "canonical_score_source": "score_recomputed_from_components",
                "final_score": round(score, 2),
                "avg_segnet_dist": seg,
                "avg_posenet_dist": pose,
                "rate_unscaled": rate,
                "n_samples": 600,
            }
        ),
        encoding="utf-8",
    )
    return auth_eval, score


def _packed_payload(decoder: bytes, latents: bytes) -> bytes:
    return bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + latents


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_bytes(path: Path) -> int:
    return path.stat().st_size


def _candidate_result(
    source_archive: Path,
    candidate_archive: Path,
    source_payload: bytes,
    candidate_payload: bytes,
    *,
    member_name: str,
    source_inner_payload: bytes | None = None,
    candidate_inner_payload: bytes | None = None,
) -> dict:
    source_hnerv_payload = source_inner_payload or source_payload
    candidate_hnerv_payload = candidate_inner_payload or candidate_payload
    source_decoder_len = int.from_bytes(source_hnerv_payload[1:4], "little")
    candidate_decoder_len = int.from_bytes(candidate_hnerv_payload[1:4], "little")
    source_header = source_hnerv_payload[:4]
    candidate_header = candidate_hnerv_payload[:4]
    source_decoder = source_hnerv_payload[4 : 4 + source_decoder_len]
    candidate_decoder = candidate_hnerv_payload[4 : 4 + candidate_decoder_len]
    source_latents = source_hnerv_payload[4 + source_decoder_len :]
    candidate_latents = candidate_hnerv_payload[4 + candidate_decoder_len :]
    source_archive_sha = _sha256(source_archive)
    candidate_archive_sha = _sha256(candidate_archive)
    source_payload_sha = sha256_bytes(source_payload)
    candidate_payload_sha = sha256_bytes(candidate_payload)
    decoder_byte_delta = len(candidate_decoder) - len(source_decoder)
    archive_byte_delta = _archive_bytes(candidate_archive) - _archive_bytes(source_archive)
    with zipfile.ZipFile(source_archive) as zf:
        source_member_bytes = len(zf.read(member_name))
    with zipfile.ZipFile(candidate_archive) as zf:
        candidate_member_bytes = len(zf.read(member_name))
    return {
        "schema_version": 1,
        "tool": "tac.hnerv_lowlevel_packer.build_lowlevel_brotli_repack_candidate",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": True,
        "ready_for_exact_eval_dispatch": False,
        "source_label": "fixture-source",
        "source_archive_path": str(source_archive),
        "source_archive_sha256": source_archive_sha,
        "source_archive_bytes": _archive_bytes(source_archive),
        "source_member_name": member_name,
        "source_payload_sha256": source_payload_sha,
        "source_payload_bytes": source_member_bytes,
        "candidate_archive_path": str(candidate_archive),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": _archive_bytes(candidate_archive),
        "candidate_member_name": member_name,
        "candidate_payload_sha256": candidate_payload_sha,
        "candidate_payload_bytes": candidate_member_bytes,
        "brotli_raw_equivalence": [
            {
                "section_name": "decoder_packed_brotli",
                "raw_equal": True,
                "raw_bytes": len(brotli.decompress(source_decoder)),
                "source_raw_sha256": sha256_bytes(brotli.decompress(source_decoder)),
                "candidate_raw_sha256": sha256_bytes(brotli.decompress(candidate_decoder)),
            },
            {
                "section_name": "latents_and_sidecar_brotli",
                "raw_equal": True,
                "raw_bytes": len(brotli.decompress(source_latents)),
                "source_raw_sha256": sha256_bytes(brotli.decompress(source_latents)),
                "candidate_raw_sha256": sha256_bytes(brotli.decompress(candidate_latents)),
            },
        ],
        "candidate_diff_audit": {
            "blockers": [],
            "ready_for_archive_preflight": True,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "dispatch_attempted": False,
            "changed_section_count": 2,
            "total_byte_delta": archive_byte_delta,
            "sections": [
                {
                    "section_name": "packed_header_ff_len24",
                    "changed": True,
                    "optimization_role": "control_or_metadata",
                    "byte_delta": 0,
                    "source_bytes": 4,
                    "candidate_bytes": 4,
                    "source_section_sha256": sha256_bytes(source_header),
                    "candidate_section_sha256": sha256_bytes(candidate_header),
                    "score_claim": False,
                },
                {
                    "section_name": "decoder_packed_brotli",
                    "changed": True,
                    "optimization_role": "decoder_weight_stream",
                    "byte_delta": decoder_byte_delta,
                    "source_bytes": len(source_decoder),
                    "candidate_bytes": len(candidate_decoder),
                    "source_section_sha256": sha256_bytes(source_decoder),
                    "candidate_section_sha256": sha256_bytes(candidate_decoder),
                    "score_claim": False,
                },
            ],
        },
    }
