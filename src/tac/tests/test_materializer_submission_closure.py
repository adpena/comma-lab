# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from comma_lab.scheduler.materializer_chain_harvest import (
    run_exact_readiness_bridge_for_harvested_queue,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest
from tac.optimizer.materializer_submission_closure import (
    SUBMISSION_CLOSURE_REPORT_SCHEMA,
    build_materializer_submission_runtime_closure,
    build_materializer_submission_runtime_closures,
)
from tac.repo_io import tree_sha256

FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotable": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "score_claim_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "field_selection_ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "exact_cuda_auth_eval": False,
    "contest_cuda_auth_eval": False,
}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zip(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("payload.bin", payload)


def _runtime_consumption_probe() -> dict[str, object]:
    return {
        "schema": "family_agnostic_runtime_consumption_probe.v1",
        "passed": True,
        "blockers": [],
        "inflate_probe_passed": True,
    }


def test_materializer_submission_closure_clears_static_readiness_blockers(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    source_runtime = repo / "submissions" / "source_runtime"
    source_runtime.mkdir(parents=True)
    inflate_sh = source_runtime / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$2\"\n"
        "cp \"$1/payload.bin\" \"$2/0.raw\"\n",
        encoding="utf-8",
    )
    inflate_sh.chmod(inflate_sh.stat().st_mode | os.X_OK)
    (source_runtime / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_runtime / "report.txt").write_text("source report\n", encoding="utf-8")
    (source_runtime / "auth_eval_stale.json").write_text(
        json.dumps({"score_claim": True}),
        encoding="utf-8",
    )
    stale_run = source_runtime / "eval_runs" / "old_score"
    stale_run.mkdir(parents=True)
    (stale_run / "state.json").write_text(
        json.dumps({"score_claim": True}),
        encoding="utf-8",
    )

    source_archive = source_runtime / "source.zip"
    candidate_archive = repo / "artifacts" / "candidate.zip"
    candidate_archive.parent.mkdir(parents=True)
    _write_zip(source_archive, b"A" * 40)
    _write_zip(candidate_archive, b"B" * 24)
    candidate_sha = _sha256(candidate_archive)
    source_sha = _sha256(source_archive)
    proof_path = repo / "artifacts" / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "target_kind": "packet_member_zip_header_elide_v1",
                "materializer_id": "packet_member_zip_header_elide_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_consumption_probe": _runtime_consumption_probe(),
                "candidate_archive_sha256": candidate_sha,
                **FALSE_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue = {
        "schema": "optimizer_candidate_queue_v1",
        **FALSE_AUTHORITY,
        "n_candidates": 1,
        "top_k_count": 1,
        "dispatch_ready_count": 0,
        "dispatch_ready": [],
        "top_k": [
            {
                "schema": "packet_member_zip_header_elide_candidate.v1",
                **FALSE_AUTHORITY,
                "candidate_id": "zip_header_fixture",
                "lane_id": "lane_zip_header_fixture",
                "target_kind": "packet_member_zip_header_elide_v1",
                "materializer_id": "packet_member_zip_header_elide_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": True,
                "runtime_consumption_proof_required": True,
                "runtime_consumption_proof_status": "present",
                "runtime_consumption_proof_path": proof_path.relative_to(repo).as_posix(),
                "candidate_archive_path": candidate_archive.relative_to(repo).as_posix(),
                "candidate_archive_sha256": candidate_sha,
                "candidate_archive_bytes": candidate_archive.stat().st_size,
                "source_archive_path": source_archive.relative_to(repo).as_posix(),
                "source_archive_sha256": source_sha,
                "source_archive_bytes": source_archive.stat().st_size,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "score_affecting_change_proof": {
                    "archive_changed": True,
                    "byte_different": True,
                    "candidate_archive_sha256": candidate_sha,
                    "source_archive_sha256": source_sha,
                    "candidate_archive_bytes": candidate_archive.stat().st_size,
                    "source_archive_bytes": source_archive.stat().st_size,
                },
                "dispatch_blockers": [
                    "optimizer_candidate_queue_is_planning_only",
                    "requires_exact_eval_readiness_gate",
                    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                    "requires_non_proxy_score_evidence_before_promotion",
                    "materializer_candidate_is_not_dispatch_authorization",
                    "materialized_archive_runtime_custody_required",
                    "exact_readiness_promotion_required",
                    "exact_auth_eval_result_required_before_score_claim",
                ],
            }
        ],
    }
    source_queue_path.write_text(json.dumps(source_queue), encoding="utf-8")

    report = build_materializer_submission_runtime_closure(
        repo_root=repo,
        source_queue_path=source_queue_path,
        candidate_id="zip_header_fixture",
        submission_dir_out=repo / "closure" / "submission",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    assert report["schema"] == SUBMISSION_CLOSURE_REPORT_SCHEMA
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["source_runtime_adapter_ready"] is False
    assert (repo / "closure" / "submission" / "archive.zip").is_file()
    assert (repo / "closure" / "submission" / "inflate.sh").is_file()
    assert (repo / "closure" / "submission" / "archive_manifest.json").is_file()
    assert not (repo / "closure" / "submission" / "auth_eval_stale.json").exists()
    assert not (repo / "closure" / "submission" / "eval_runs").exists()

    closed_queue = json.loads(
        (repo / "closure" / "closed_source_queue.json").read_text(encoding="utf-8")
    )
    closed_row = closed_queue["top_k"][0]
    assert closed_row["candidate_archive_path"] == "closure/submission/archive.zip"
    assert closed_row["archive_manifest_path"] == (
        "closure/submission/archive_manifest.json"
    )
    assert closed_row["runtime_consumption_proof_path"] == (
        "closure/submission/runtime_consumption_proof.json"
    )
    assert closed_row["score_affecting_payload_changed"] is True
    assert closed_row["charged_bits_changed"] is True
    assert closed_row["ready_for_exact_eval_dispatch"] is False

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=repo / "closure" / "closed_source_queue.json",
        exact_readiness_out_dir=repo / "closure" / "exact_readiness",
        candidate_ids=("zip_header_fixture",),
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )
    assert bridge["ready_candidate_count"] == 1
    assert bridge["rows"][0]["blockers"] == []
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert Path(repo / bridge["rows"][0]["exact_ready_queue_path"]).is_file()


def test_materializer_submission_closure_discovers_source_packet_submission_dir(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    packet_dir = repo / "experiments" / "results" / "packet"
    source_runtime = packet_dir / "submission_dir"
    source_runtime.mkdir(parents=True)
    inflate_sh = source_runtime / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$2\"\n",
        encoding="utf-8",
    )
    inflate_sh.chmod(inflate_sh.stat().st_mode | os.X_OK)
    (source_runtime / "inflate.py").write_text("print('inflate')\n", encoding="utf-8")

    source_archive = packet_dir / "archive.zip"
    candidate_archive = repo / "artifacts" / "candidate.zip"
    candidate_archive.parent.mkdir(parents=True)
    _write_zip(source_archive, b"A" * 40)
    _write_zip(candidate_archive, b"B" * 24)
    candidate_sha = _sha256(candidate_archive)
    source_sha = _sha256(source_archive)
    packet_manifest = {
        "schema": "fixture_packet_manifest.v1",
        "runtime": {
            "path": source_runtime.relative_to(repo).as_posix(),
            "manifest": {"runtime_root": source_runtime.relative_to(repo).as_posix()},
        },
    }
    (packet_dir / "packet_manifest.json").write_text(
        json.dumps(packet_manifest),
        encoding="utf-8",
    )
    proof_path = repo / "artifacts" / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "target_kind": "packet_member_zip_header_elide_v1",
                "materializer_id": "packet_member_zip_header_elide_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_consumption_probe": _runtime_consumption_probe(),
                "candidate_archive_sha256": candidate_sha,
                **FALSE_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                **FALSE_AUTHORITY,
                "n_candidates": 1,
                "top_k_count": 1,
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": [
                    {
                        "schema": "packet_member_zip_header_elide_candidate.v1",
                        **FALSE_AUTHORITY,
                        "candidate_id": "zip_header_fixture",
                        "target_kind": "packet_member_zip_header_elide_v1",
                        "materializer_id": "packet_member_zip_header_elide_adapter",
                        "receiver_contract_kind": (
                            "family_agnostic_packet_member_zip_header_elide"
                        ),
                        "receiver_contract_satisfied": True,
                        "runtime_adapter_ready": True,
                        "runtime_consumption_proof_required": True,
                        "runtime_consumption_proof_status": "present",
                        "runtime_consumption_proof_path": proof_path.relative_to(
                            repo
                        ).as_posix(),
                        "candidate_archive_path": candidate_archive.relative_to(
                            repo
                        ).as_posix(),
                        "candidate_archive_sha256": candidate_sha,
                        "candidate_archive_bytes": candidate_archive.stat().st_size,
                        "source_archive_path": source_archive.relative_to(
                            repo
                        ).as_posix(),
                        "source_archive_sha256": source_sha,
                        "source_archive_bytes": source_archive.stat().st_size,
                        "score_affecting_payload_changed": True,
                        "charged_bits_changed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_materializer_submission_runtime_closure(
        repo_root=repo,
        source_queue_path=source_queue_path,
        submission_dir_out=repo / "closure" / "submission",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    assert report["schema"] == SUBMISSION_CLOSURE_REPORT_SCHEMA
    assert report["source_runtime_dir"] == "experiments/results/packet/submission_dir"
    assert report["source_runtime_adapter_ready"] is False
    assert (repo / "closure" / "submission" / "inflate.sh").is_file()
    assert (repo / "closure" / "submission" / "inflate.py").is_file()


def test_materializer_submission_closure_writes_refusal_when_runtime_missing(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    source_archive = repo / "artifacts" / "source.zip"
    candidate_archive = repo / "artifacts" / "candidate.zip"
    candidate_archive.parent.mkdir(parents=True)
    _write_zip(source_archive, b"A" * 40)
    _write_zip(candidate_archive, b"B" * 24)
    candidate_sha = _sha256(candidate_archive)
    source_sha = _sha256(source_archive)
    proof_path = repo / "artifacts" / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "target_kind": "archive_section_entropy_recode_v1",
                "materializer_id": "archive_section_entropy_recode_adapter",
                "receiver_contract_kind": (
                    "family_agnostic_archive_section_entropy_recode"
                ),
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "candidate_archive_sha256": candidate_sha,
                **FALSE_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                **FALSE_AUTHORITY,
                "n_candidates": 1,
                "top_k_count": 1,
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": [
                    {
                        "schema": "archive_section_entropy_recode_candidate.v1",
                        **FALSE_AUTHORITY,
                        "candidate_id": "archive_section_fixture",
                        "target_kind": "archive_section_entropy_recode_v1",
                        "materializer_id": "archive_section_entropy_recode_adapter",
                        "receiver_contract_kind": (
                            "family_agnostic_archive_section_entropy_recode"
                        ),
                        "receiver_contract_satisfied": True,
                        "runtime_adapter_ready": True,
                        "runtime_consumption_proof_required": True,
                        "runtime_consumption_proof_status": "present",
                        "runtime_consumption_proof_path": (
                            proof_path.relative_to(repo).as_posix()
                        ),
                        "candidate_archive_path": (
                            candidate_archive.relative_to(repo).as_posix()
                        ),
                        "candidate_archive_sha256": candidate_sha,
                        "candidate_archive_bytes": candidate_archive.stat().st_size,
                        "source_archive_path": (
                            source_archive.relative_to(repo).as_posix()
                        ),
                        "source_archive_sha256": source_sha,
                        "source_archive_bytes": source_archive.stat().st_size,
                        "score_affecting_payload_changed": True,
                        "charged_bits_changed": True,
                        "dispatch_blockers": [
                            "optimizer_candidate_queue_is_planning_only",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_materializer_submission_runtime_closure(
        repo_root=repo,
        source_queue_path=source_queue_path,
        candidate_id="archive_section_fixture",
        submission_dir_out=repo / "closure" / "submission",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    assert report["schema"] == SUBMISSION_CLOSURE_REPORT_SCHEMA
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["runtime_manifest"] is None
    assert "source_runtime_dir_missing_or_inflate_sh_missing" in (
        report["closure_blockers"]
    )
    assert (repo / "closure" / "submission" / "archive.zip").is_file()
    assert (repo / "closure" / "submission" / "archive_manifest.json").is_file()
    assert (repo / "closure" / "submission" / "runtime_consumption_proof.json").is_file()
    assert not (repo / "closure" / "submission" / "inflate.sh").exists()

    closed_queue = json.loads(
        (repo / "closure" / "closed_source_queue.json").read_text(encoding="utf-8")
    )
    closed_row = closed_queue["top_k"][0]
    assert closed_row["candidate_archive_path"] == "closure/submission/archive.zip"
    assert "source_runtime_dir_missing_or_inflate_sh_missing" in (
        closed_row["materializer_submission_closure_blockers"]
    )
    assert "submission_runtime_closure_refused_missing_runtime" in (
        closed_row["readiness_blockers"]
    )
    assert closed_row["score_claim"] is False
    assert closed_row["ready_for_exact_eval_dispatch"] is False


def test_materializer_submission_closure_uses_proof_backed_runtime_adapter(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    source_runtime = repo / "submissions" / "source_runtime"
    source_runtime.mkdir(parents=True)
    source_inflate = source_runtime / "inflate.sh"
    source_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo source-runtime > \"$2/source_marker.txt\"\n",
        encoding="utf-8",
    )
    source_inflate.chmod(source_inflate.stat().st_mode | os.X_OK)

    stale_runtime = repo / "artifacts" / "stale_candidate_runtime"
    stale_runtime.mkdir(parents=True)
    stale_inflate = stale_runtime / "inflate.sh"
    stale_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo stale-runtime > \"$2/stale_marker.txt\"\n",
        encoding="utf-8",
    )
    stale_inflate.chmod(stale_inflate.stat().st_mode | os.X_OK)
    stale_runtime_sha = tree_sha256(stale_runtime)

    adapter_runtime = repo / "artifacts" / "adapter_runtime"
    adapter_runtime.mkdir(parents=True)
    adapter_inflate = adapter_runtime / "inflate.sh"
    adapter_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo adapter-runtime > \"$2/adapter_marker.txt\"\n",
        encoding="utf-8",
    )
    adapter_inflate.chmod(adapter_inflate.stat().st_mode | os.X_OK)
    (adapter_runtime / "receiver.py").write_text("ADAPTER = True\n", encoding="utf-8")
    adapter_runtime_sha = tree_sha256(adapter_runtime)

    source_archive = source_runtime / "source.zip"
    candidate_archive = repo / "artifacts" / "candidate.zip"
    _write_zip(source_archive, b"A" * 40)
    _write_zip(candidate_archive, b"B" * 24)
    candidate_sha = _sha256(candidate_archive)
    source_sha = _sha256(source_archive)

    proof_path = repo / "artifacts" / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "target_kind": "packet_member_merge_v1",
                "materializer_id": "packet_member_merge_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_merge",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_consumption_probe": _runtime_consumption_probe(),
                "runtime_adapter_ready": True,
                "candidate_archive_sha256": candidate_sha,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": adapter_runtime.relative_to(repo).as_posix(),
                    "runtime_tree_sha256": adapter_runtime_sha,
                },
                **FALSE_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                **FALSE_AUTHORITY,
                "n_candidates": 1,
                "top_k_count": 1,
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": [
                    {
                        "schema": "packet_member_merge_candidate.v1",
                        **FALSE_AUTHORITY,
                        "candidate_id": "packet_member_merge_fixture",
                        "lane_id": "lane_packet_member_merge_fixture",
                        "target_kind": "packet_member_merge_v1",
                        "materializer_id": "packet_member_merge_adapter",
                        "receiver_contract_kind": (
                            "family_agnostic_packet_member_merge"
                        ),
                        "receiver_contract_satisfied": True,
                        "runtime_adapter_ready": True,
                        "runtime_consumption_proof_required": True,
                        "runtime_consumption_proof_status": "present",
                        "runtime_consumption_proof_path": proof_path.relative_to(
                            repo
                        ).as_posix(),
                        "candidate_runtime_dir": stale_runtime.relative_to(
                            repo
                        ).as_posix(),
                        "candidate_runtime_tree_sha256": stale_runtime_sha,
                        "packet_member_merge_receiver_runtime": {
                            "runtime_dir": adapter_runtime.relative_to(repo).as_posix(),
                            "runtime_tree_sha256": adapter_runtime_sha,
                        },
                        "candidate_archive_path": candidate_archive.relative_to(
                            repo
                        ).as_posix(),
                        "candidate_archive_sha256": candidate_sha,
                        "candidate_archive_bytes": candidate_archive.stat().st_size,
                        "source_archive_path": source_archive.relative_to(
                            repo
                        ).as_posix(),
                        "source_archive_sha256": source_sha,
                        "source_archive_bytes": source_archive.stat().st_size,
                        "score_affecting_payload_changed": True,
                        "charged_bits_changed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_materializer_submission_runtime_closure(
        repo_root=repo,
        source_queue_path=source_queue_path,
        candidate_id="packet_member_merge_fixture",
        source_runtime_dir=source_runtime,
        submission_dir_out=repo / "closure" / "submission",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    submission = repo / "closure" / "submission"
    assert report["source_runtime_adapter_ready"] is True
    assert (
        report["materializer_submission_closure_kind"]
        == "runtime_adapter_closure_with_candidate_archive"
    )
    assert (submission / "inflate.sh").read_text(encoding="utf-8") == (
        adapter_inflate.read_text(encoding="utf-8")
    )
    assert (submission / "receiver.py").is_file()
    closed_queue = json.loads(
        (repo / "closure" / "closed_source_queue.json").read_text(encoding="utf-8")
    )
    closed_row = closed_queue["top_k"][0]
    assert closed_row["runtime_source_dir"] == "artifacts/adapter_runtime"
    assert closed_row["candidate_runtime_dir"] == "artifacts/adapter_runtime"
    assert closed_row["candidate_runtime_tree_sha256"] == adapter_runtime_sha
    assert closed_row["adapter_runtime_tree_sha256"] == adapter_runtime_sha
    assert closed_row["submission_runtime_tree_sha256"] == closed_row[
        "runtime_tree_sha256"
    ]
    assert closed_row["packet_member_merge_receiver_runtime"]["runtime_dir"] == (
        "artifacts/adapter_runtime"
    )
    assert (
        closed_row["materializer_submission_closure_kind"]
        == "runtime_adapter_closure_with_candidate_archive"
    )
    assert closed_row["runtime_tree_sha256"] == report["runtime_manifest"][
        "runtime_tree_sha256"
    ]
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=repo / "closure" / "closed_source_queue.json",
        exact_readiness_out_dir=repo / "closure" / "exact_readiness",
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )
    readiness_report_path = repo / bridge["rows"][0]["exact_readiness_report_path"]
    readiness_report = json.loads(readiness_report_path.read_text(encoding="utf-8"))
    readiness_facts = readiness_report["facts"]
    assert readiness_facts["runtime_tree_sha256"] == closed_row["runtime_tree_sha256"]
    assert readiness_facts["submission_runtime_tree_sha256"] == closed_row[
        "runtime_tree_sha256"
    ]
    assert (
        readiness_facts["runtime_consumption_proof_runtime_tree_sha256"]
        == adapter_runtime_sha
    )
    assert readiness_facts["candidate_row_adapter_runtime_tree_sha256"] == (
        adapter_runtime_sha
    )


def test_materializer_submission_closure_refuses_adapter_without_expected_tree_sha(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    adapter_runtime = repo / "artifacts" / "adapter_runtime"
    adapter_runtime.mkdir(parents=True)
    adapter_inflate = adapter_runtime / "inflate.sh"
    adapter_inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "echo adapter-runtime > \"$2/adapter_marker.txt\"\n",
        encoding="utf-8",
    )
    adapter_inflate.chmod(adapter_inflate.stat().st_mode | os.X_OK)

    source_archive = repo / "artifacts" / "source.zip"
    candidate_archive = repo / "artifacts" / "candidate.zip"
    _write_zip(source_archive, b"A" * 40)
    _write_zip(candidate_archive, b"B" * 24)
    candidate_sha = _sha256(candidate_archive)
    source_sha = _sha256(source_archive)

    proof_path = repo / "artifacts" / "runtime_consumption_proof.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "family_agnostic_runtime_consumption_proof_v1",
                "target_kind": "packet_member_merge_v1",
                "materializer_id": "packet_member_merge_adapter",
                "receiver_contract_kind": "family_agnostic_packet_member_merge",
                "receiver_contract_satisfied": True,
                "runtime_consumption_proof_passed": True,
                "passed": True,
                "runtime_adapter_ready": True,
                "candidate_archive_sha256": candidate_sha,
                "runtime_adapter_manifest": {
                    "runtime_adapter_ready": True,
                    "runtime_dir": adapter_runtime.relative_to(repo).as_posix(),
                },
                **FALSE_AUTHORITY,
            }
        ),
        encoding="utf-8",
    )
    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                **FALSE_AUTHORITY,
                "n_candidates": 1,
                "top_k_count": 1,
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": [
                    {
                        "schema": "packet_member_merge_candidate.v1",
                        **FALSE_AUTHORITY,
                        "candidate_id": "packet_member_merge_fixture",
                        "target_kind": "packet_member_merge_v1",
                        "materializer_id": "packet_member_merge_adapter",
                        "receiver_contract_kind": (
                            "family_agnostic_packet_member_merge"
                        ),
                        "receiver_contract_satisfied": True,
                        "runtime_adapter_ready": True,
                        "runtime_consumption_proof_required": True,
                        "runtime_consumption_proof_status": "present",
                        "runtime_consumption_proof_path": (
                            proof_path.relative_to(repo).as_posix()
                        ),
                        "candidate_archive_path": (
                            candidate_archive.relative_to(repo).as_posix()
                        ),
                        "candidate_archive_sha256": candidate_sha,
                        "candidate_archive_bytes": candidate_archive.stat().st_size,
                        "source_archive_path": source_archive.relative_to(
                            repo
                        ).as_posix(),
                        "source_archive_sha256": source_sha,
                        "source_archive_bytes": source_archive.stat().st_size,
                        "score_affecting_payload_changed": True,
                        "charged_bits_changed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_materializer_submission_runtime_closure(
        repo_root=repo,
        source_queue_path=source_queue_path,
        candidate_id="packet_member_merge_fixture",
        submission_dir_out=repo / "closure" / "submission",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    assert report["source_runtime_adapter_ready"] is False
    assert "runtime_adapter_expected_tree_sha_missing" in report["closure_blockers"]
    assert not (repo / "closure" / "submission" / "inflate.sh").exists()
    closed_queue = json.loads(
        (repo / "closure" / "closed_source_queue.json").read_text(encoding="utf-8")
    )
    closed_row = closed_queue["top_k"][0]
    assert "runtime_adapter_expected_tree_sha_missing" in (
        closed_row["materializer_submission_closure_blockers"]
    )
    assert closed_row["ready_for_exact_eval_dispatch"] is False


def test_materializer_submission_closure_closes_all_source_queue_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path
    runtime = repo / "submissions" / "source_runtime"
    runtime.mkdir(parents=True)
    inflate_sh = runtime / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$2\"\n",
        encoding="utf-8",
    )
    inflate_sh.chmod(inflate_sh.stat().st_mode | os.X_OK)
    source_archive = runtime / "source.zip"
    _write_zip(source_archive, b"S" * 64)
    source_sha = _sha256(source_archive)

    rows = []
    for index, payload in enumerate((b"A" * 24, b"B" * 20), start=1):
        candidate_id = f"zip_header_fixture_{index}"
        candidate_archive = repo / "artifacts" / f"candidate_{index}.zip"
        candidate_archive.parent.mkdir(parents=True, exist_ok=True)
        _write_zip(candidate_archive, payload)
        candidate_sha = _sha256(candidate_archive)
        proof_path = repo / "artifacts" / f"proof_{index}.json"
        proof_path.write_text(
            json.dumps(
                {
                    "schema": "family_agnostic_runtime_consumption_proof_v1",
                    "target_kind": "packet_member_zip_header_elide_v1",
                    "materializer_id": "packet_member_zip_header_elide_adapter",
                    "receiver_contract_kind": (
                        "family_agnostic_packet_member_zip_header_elide"
                    ),
                    "receiver_contract_satisfied": True,
                    "runtime_consumption_proof_passed": True,
                    "passed": True,
                    "runtime_consumption_probe": _runtime_consumption_probe(),
                    "candidate_archive_sha256": candidate_sha,
                    **FALSE_AUTHORITY,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "schema": "packet_member_zip_header_elide_candidate.v1",
                **FALSE_AUTHORITY,
                "candidate_id": candidate_id,
                "lane_id": "lane_zip_header_fixture",
                "target_kind": "packet_member_zip_header_elide_v1",
                "materializer_id": "packet_member_zip_header_elide_adapter",
                "receiver_contract_kind": (
                    "family_agnostic_packet_member_zip_header_elide"
                ),
                "receiver_contract_satisfied": True,
                "runtime_adapter_ready": False,
                "runtime_consumption_proof_required": True,
                "runtime_consumption_proof_status": "present",
                "runtime_consumption_proof_path": proof_path.relative_to(
                    repo
                ).as_posix(),
                "candidate_archive_path": candidate_archive.relative_to(
                    repo
                ).as_posix(),
                "candidate_archive_sha256": candidate_sha,
                "candidate_archive_bytes": candidate_archive.stat().st_size,
                "source_archive_path": source_archive.relative_to(repo).as_posix(),
                "source_archive_sha256": source_sha,
                "source_archive_bytes": source_archive.stat().st_size,
                "score_affecting_payload_changed": True,
                "charged_bits_changed": True,
                "dispatch_blockers": [
                    "optimizer_candidate_queue_is_planning_only",
                    "requires_exact_eval_readiness_gate",
                    "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                    "requires_non_proxy_score_evidence_before_promotion",
                    "materializer_candidate_is_not_dispatch_authorization",
                    "materialized_archive_runtime_custody_required",
                    "exact_readiness_promotion_required",
                    "exact_auth_eval_result_required_before_score_claim",
                ],
            }
        )

    source_queue_path = repo / "artifacts" / "source_queue.json"
    source_queue_path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                **FALSE_AUTHORITY,
                "n_candidates": len(rows),
                "top_k_count": len(rows),
                "dispatch_ready_count": 0,
                "dispatch_ready": [],
                "top_k": rows,
            }
        ),
        encoding="utf-8",
    )

    report = build_materializer_submission_runtime_closures(
        repo_root=repo,
        source_queue_path=source_queue_path,
        submission_dir_out=repo / "closure" / "submissions",
        closed_source_queue_out=repo / "closure" / "closed_source_queue.json",
        closure_report_out=repo / "closure" / "submission_closure_report.json",
    )

    assert report["schema"] == SUBMISSION_CLOSURE_REPORT_SCHEMA
    assert report["candidate_count"] == 2
    closed_queue = json.loads(
        (repo / "closure" / "closed_source_queue.json").read_text(encoding="utf-8")
    )
    assert closed_queue["n_candidates"] == 2
    assert [row["candidate_id"] for row in closed_queue["top_k"]] == [
        "zip_header_fixture_1",
        "zip_header_fixture_2",
    ]
    for row in closed_queue["top_k"]:
        submission_dir = repo / row["submission_dir"]
        assert (submission_dir / "archive.zip").is_file()
        assert (submission_dir / "inflate.sh").is_file()
        assert not (submission_dir / "closed_source_queue.json").exists()
        assert not (submission_dir / "submission_closure_report.json").exists()
        assert row["candidate_archive_path"].endswith("/archive.zip")
        assert row["runtime_consumption_proof_path"].endswith(
            "/runtime_consumption_proof.json"
        )
        assert row["runtime_tree_sha256"] == runtime_dependency_manifest(
            submission_dir,
            repo,
        )["runtime_tree_sha256"]
        assert row["submission_runtime_tree_sha256"] == row["runtime_tree_sha256"]
    sidecar_root = repo / "closure" / "candidate_closure_sidecars"
    assert (sidecar_root / "zip_header_fixture_1" / "closed_source_queue.json").is_file()
    assert (
        sidecar_root / "zip_header_fixture_1" / "submission_closure_report.json"
    ).is_file()

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=repo / "closure" / "closed_source_queue.json",
        exact_readiness_out_dir=repo / "closure" / "exact_readiness",
        active_floor_archive_bytes=None,
        active_floor_score=None,
    )
    assert bridge["ready_candidate_count"] == 2
