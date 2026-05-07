from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import sha256_bytes, write_stored_single_member_zip
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
    assert release_manifest["archive"]["path"] == "archive.zip"


def _write_lowlevel_fixture(root: Path, *, member_name: str = "x") -> dict[str, object]:
    source_decoder_raw = (b"decoder-record-" * 3000) + b"source"
    latent_raw = b"latent-row-" * 2000
    source_decoder = brotli.compress(source_decoder_raw, quality=1)
    candidate_decoder = brotli.compress(source_decoder_raw, quality=11, lgblock=16)
    if len(candidate_decoder) >= len(source_decoder):
        candidate_decoder = brotli.compress(source_decoder_raw, quality=10, lgblock=16)
    assert len(candidate_decoder) < len(source_decoder)
    latents = brotli.compress(latent_raw, quality=5)
    source_payload = _packed_payload(source_decoder, latents)
    candidate_payload = _packed_payload(candidate_decoder, latents)

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
    )
    result_json.write_text(json_text(result), encoding="utf-8")
    return {
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": _sha256(candidate_archive),
        "result_json": result_json,
        "inflate_sh": inflate_sh,
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
) -> dict:
    source_decoder_len = int.from_bytes(source_payload[1:4], "little")
    candidate_decoder_len = int.from_bytes(candidate_payload[1:4], "little")
    source_header = source_payload[:4]
    candidate_header = candidate_payload[:4]
    source_decoder = source_payload[4 : 4 + source_decoder_len]
    candidate_decoder = candidate_payload[4 : 4 + candidate_decoder_len]
    source_latents = source_payload[4 + source_decoder_len :]
    candidate_latents = candidate_payload[4 + candidate_decoder_len :]
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
