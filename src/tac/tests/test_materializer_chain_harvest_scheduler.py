# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest

import comma_lab.scheduler.materializer_exact_eval_dispatch_plan as dispatch_plan_module
from comma_lab.scheduler.byte_shaving_campaign_queue import (
    MATERIALIZER_EXECUTION_STEP_ID,
    MATERIALIZER_WORK_QUEUE_SCHEMA,
)
from comma_lab.scheduler.experiment_queue import (
    QUEUE_SCHEMA,
    connect_state,
    initialize_queue_state,
    normalize_queue_definition,
)
from comma_lab.scheduler.materializer_chain_harvest import (
    EXACT_READINESS_BRIDGE_SCHEMA,
    HARVEST_SCHEMA,
    harvest_materializer_chain_manifests,
    run_exact_readiness_bridge_for_harvested_queue,
)
from comma_lab.scheduler.materializer_chain_harvest import (
    write_json as write_harvest_json,
)
from comma_lab.scheduler.materializer_exact_eval_dispatch_plan import (
    DISPATCH_PLAN_SCHEMA,
    build_materializer_exact_eval_dispatch_plan,
)
from comma_lab.scheduler.materializer_exact_eval_dispatch_plan import (
    write_json as write_dispatch_plan_json,
)
from tac.optimization.byte_range_entropy_recode_chain import (
    CHAIN_MANIFEST_NAME,
    CHAIN_SCHEMA,
)
from tac.optimization.family_agnostic_materializers import (
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
    RENDERER_PAYLOAD_DFL1_SCHEMA,
    TENSOR_FACTORIZE_SCHEMA,
    materialize_archive_section_entropy_recode_candidate,
    materialize_packet_member_recompress_candidate,
    materialize_renderer_payload_dfl1_candidate,
    materialize_tensor_factorize_candidate,
)
from tac.optimization.proxy_candidate_contract import truthy_authority_field_violations
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "harvest_materializer_chain_candidates.py"
DISPATCH_PLAN_TOOL = REPO_ROOT / "tools" / "build_materializer_exact_eval_dispatch_plan.py"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_bytes(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _write_zip_archive(path: Path, member: str, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(info, data, compress_type=zipfile.ZIP_STORED)
    return _artifact_record(path)


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }


def _chain_manifest(
    external_root: Path,
    *,
    authority_overrides: dict[str, object] | None = None,
) -> Path:
    source_archive = _write_bytes(
        external_root / "source" / "archive.zip",
        b"source archive bytes",
    )
    candidate_archive = _write_bytes(
        external_root / "candidate" / "archive.zip",
        b"candidate",
    )
    artifact = _write_json(
        external_root / "candidate" / "candidate_manifest.json",
        {"schema": "fixture_materializer_artifact_v1"},
    )
    artifact_record = _artifact_record(artifact)
    payload: dict[str, object] = {
        "schema": CHAIN_SCHEMA,
        "candidate_id": "external_materializer_candidate",
        "lane_id": "fixture_materializer_chain_harvest",
        "source_archive": source_archive,
        "source_archive_sha256": source_archive["sha256"],
        "source_archive_bytes": source_archive["bytes"],
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "serialized_archive_delta": build_serialized_archive_delta_contract(
            source_archive=source_archive,
            candidate_archive=candidate_archive,
            require_realized_saving=True,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_proof_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": ["exact_cuda_auth_eval_missing"],
        "dispatch_blockers": [
            "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
            "exact_cuda_auth_eval_missing",
        ],
        "artifacts": {"candidate_manifest": artifact_record},
        "chain_steps": [
            {
                "step_id": "materialize_candidate",
                "status": "succeeded",
                "artifact": artifact_record,
            }
        ],
        "next_required_gates": ["contest_auth_eval"],
        **_false_authority(),
    }
    if authority_overrides:
        payload.update(authority_overrides)
    return _write_json(external_root / CHAIN_MANIFEST_NAME, payload)


def _family_agnostic_manifest(external_root: Path) -> Path:
    source_archive = _write_bytes(
        external_root / "source" / "archive.zip",
        b"source archive bytes with extra padding",
    )
    candidate_archive = _write_bytes(
        external_root / "candidate" / "archive.zip",
        b"candidate archive bytes",
    )
    payload = {
        "schema": PACKET_MEMBER_RECOMPRESS_SCHEMA,
        "candidate_id": "family_agnostic_packet_member_candidate",
        "lane_id": "fixture_family_agnostic_harvest",
        "materializer_id": "packet_member_recompress_adapter",
        "target_kind": "packet_member_recompress_v1",
        "byte_closed_candidate_emitted": True,
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "receiver_contract_satisfied": False,
        "receiver_verification": {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": False,
            "proof_present": False,
            "proof_path": None,
            "blockers": ["runtime_consumption_proof_missing"],
        },
        "readiness_blockers": [
            "runtime_consumption_proof_missing",
            "packet_member_recompress_receiver_contract_not_satisfied",
        ],
        **_false_authority(),
    }
    return _write_json(external_root / "family_candidate.json", payload)


def _write_runtime_bound_pr101_proof(
    submission: Path,
    *,
    archive_sha: str,
    stale_inflate_sh_sha: bool = False,
) -> Path:
    inflate_sh_sha = hashlib.sha256((submission / "inflate.sh").read_bytes()).hexdigest()
    if stale_inflate_sh_sha:
        inflate_sh_sha = "0" * 64 if inflate_sh_sha != "0" * 64 else "1" * 64
    inflate_py_sha = hashlib.sha256((submission / "inflate.py").read_bytes()).hexdigest()
    manifest_path = _write_json(
        submission / "runtime_packet_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_packet_v1",
            "packet_dir": str(submission),
            "runtime_custody": {
                "runtime_files": [
                    {"relpath": "inflate.sh", "sha256": inflate_sh_sha},
                    {"relpath": "inflate.py", "sha256": inflate_py_sha},
                ],
            },
        },
    )
    return _write_json(
        submission / "runtime_consumption_proof.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
            "proof_kind": "fixture_runtime_bound_pr101_proof",
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "packet_dir": str(submission),
            "runtime_consumption_proven_for_supported_bias_params": True,
            "inflate_sh_routes_to_packet_inflate_py": True,
            "archive_unchanged_proof": {
                "archive_sha256": archive_sha,
            },
            "inflate_wrapper_route_proof": {
                "wrapper_invoked_packet_inflate_py": True,
                "inflate_sh_sha256": inflate_sh_sha,
                "packet_inflate_py_sha256": inflate_py_sha,
            },
            "inflate_static_bias_patch_proof": {
                "inflate_sha256": inflate_py_sha,
            },
            "inflate_runtime_bias_logic_proof": {
                "packet_inflate_function_executed": True,
                "inflate_py_sha256": inflate_py_sha,
            },
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
        },
    )


def _exact_ready_chain_manifest(
    repo: Path,
    *,
    stale_runtime_proof: bool = False,
) -> Path:
    source_archive = _write_zip_archive(
        repo / "source" / "archive.zip",
        "0.bin",
        b"source archive payload that is deliberately larger",
    )
    submission = repo / "submission"
    candidate_archive = _write_zip_archive(
        submission / "archive.zip",
        "0.bin",
        b"candidate",
    )
    inflate = submission / "inflate.sh"
    inflate_py = submission / "inflate.py"
    inflate_py.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(b'')\n",
        encoding="utf-8",
    )
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "python \"$SCRIPT_DIR/inflate.py\" \"$1\" \"$2\"\n",
        encoding="utf-8",
    )
    inflate.chmod(inflate.stat().st_mode | 0o100)
    (submission / "report.txt").write_text(
        "candidate archive ready for exact eval\n",
        encoding="utf-8",
    )
    _write_json(
        submission / "archive_manifest.json",
        {
            "candidate_archive_sha256": candidate_archive["sha256"],
            "candidate_archive_bytes": candidate_archive["bytes"],
            "candidate_archive": {"member_name": "0.bin"},
        },
    )
    runtime_proof = _write_runtime_bound_pr101_proof(
        submission,
        archive_sha=str(candidate_archive["sha256"]),
        stale_inflate_sh_sha=stale_runtime_proof,
    )
    (repo / "upstream").mkdir(parents=True, exist_ok=True)
    (repo / "upstream" / "evaluate.py").write_text("# fixture\n", encoding="utf-8")
    artifact = _write_json(
        submission / "candidate_manifest.json",
        {"schema": "fixture_materializer_artifact_v1"},
    )
    artifact_record = _artifact_record(artifact)
    payload: dict[str, object] = {
        "schema": CHAIN_SCHEMA,
        "candidate_id": "exact_ready_materializer_candidate",
        "lane_id": "fixture_materializer_chain_harvest",
        "source_archive": source_archive,
        "source_archive_sha256": source_archive["sha256"],
        "source_archive_bytes": source_archive["bytes"],
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "serialized_archive_delta": build_serialized_archive_delta_contract(
            source_archive=source_archive,
            candidate_archive=candidate_archive,
            require_realized_saving=True,
        ),
        "byte_closed_candidate_emitted": True,
        "runtime_adapter_ready": True,
        "receiver_proof_ready": True,
        "receiver_contract_satisfied": True,
        "candidate_runtime_adapter_blocker_cleared": True,
        "readiness_blockers": [],
        "dispatch_blockers": [
            "byte_range_entropy_recode_chain_is_not_dispatch_authorization",
            "exact_cuda_auth_eval_missing",
        ],
        "artifacts": {"candidate_manifest": artifact_record},
        "chain_steps": [
            {
                "step_id": "materialize_candidate",
                "status": "succeeded",
                "artifact": artifact_record,
            }
        ],
        "next_required_gates": ["contest_auth_eval"],
        "runtime_consumption_proof_required": True,
        "runtime_consumption_proof_status": "present",
        "runtime_consumption_proof_path": runtime_proof.relative_to(repo).as_posix(),
        **_false_authority(),
    }
    return _write_json(repo / "chain" / CHAIN_MANIFEST_NAME, payload)


def _work_queue(repo: Path, chain_manifest: Path) -> Path:
    payload = {
        "schema": MATERIALIZER_WORK_QUEUE_SCHEMA,
        "tool": "fixture",
        "row_count": 1,
        "executable_row_count": 1,
        "blocked_row_count": 0,
        "rows": [
            {
                "schema": "byte_shaving_materializer_work_row.v1",
                "work_id": "Materializer Work Fixture",
                "work_rank": 1,
                "backlog_key": "fixture",
                "unit_kind": "byte_range",
                "operation_family": "entropy_recode",
                "target_kind": "byte_range_entropy_recode_v1",
                "resource_kind": "local_cpu",
                "executable": True,
                "postconditions": [
                    {
                        "type": "materializer_chain_complete",
                        "path": str(chain_manifest),
                        "schema": CHAIN_SCHEMA,
                    }
                ],
                **_false_authority(),
            }
        ],
        **_false_authority(),
    }
    return _write_json(repo / "materializer_work_queue.json", payload)


def _state_path(repo: Path, *, status: str) -> Path:
    queue = normalize_queue_definition(
        {
            "schema": QUEUE_SCHEMA,
            "queue_id": "materializer_execution_queue",
            "controls": {"mode": "running", "max_concurrency": {"local_cpu": 1}},
            "experiments": [
                {
                    "id": "materializer_work_fixture",
                    "steps": [
                        {
                            "id": MATERIALIZER_EXECUTION_STEP_ID,
                            "command": [sys.executable, "-c", "print('fixture')"],
                            "resources": {"kind": "local_cpu"},
                        }
                    ],
                }
            ],
        }
    )
    state = repo / "queue.sqlite"
    with connect_state(state) as conn:
        initialize_queue_state(conn, queue)
        conn.execute(
            """
            UPDATE step_state
            SET status = ?, attempts = 1, last_event_json = ?
            WHERE queue_id = ? AND experiment_id = ? AND step_id = ?
            """,
            (
                status,
                json.dumps({"fixture": True}),
                "materializer_execution_queue",
                "materializer_work_fixture",
                MATERIALIZER_EXECUTION_STEP_ID,
            ),
        )
        conn.commit()
    return state


def test_harvest_work_queue_chain_manifest_into_source_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "VertigoDataTier"
    chain = _chain_manifest(external)
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="succeeded")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    report = result["report"]
    queue = result["source_queue"]
    row = queue["top_k"][0]
    assert report["schema"] == HARVEST_SCHEMA
    assert report["accepted_manifest_count"] == 1
    assert report["source_queue_candidate_count"] == 1
    assert report["source_queue_dispatch_ready_count"] == 0
    assert report["rows"][0]["state_rows"][0]["status"] == "succeeded"
    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "external_materializer_candidate"
    assert row["archive_candidate_verified"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert str(external) in row["candidate_archive_path"]


def test_harvest_work_queue_family_agnostic_candidate_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "VertigoDataTier"
    manifest = _family_agnostic_manifest(external)
    work_queue = _write_json(
        repo / "materializer_work_queue.json",
        {
            "schema": MATERIALIZER_WORK_QUEUE_SCHEMA,
            "tool": "fixture",
            "row_count": 1,
            "executable_row_count": 1,
            "blocked_row_count": 0,
            "rows": [
                {
                    "schema": "byte_shaving_materializer_work_row.v1",
                    "work_id": "Family Agnostic Work Fixture",
                    "work_rank": 1,
                    "backlog_key": "family_fixture",
                    "unit_kind": "packet_member",
                    "operation_family": "member_recompress",
                    "target_kind": "packet_member_recompress_v1",
                    "resource_kind": "local_cpu",
                    "executable": True,
                    "postconditions": [
                        {
                            "type": "json_completion_contract",
                            "path": str(manifest),
                            "required_equals": {
                                "schema": PACKET_MEMBER_RECOMPRESS_SCHEMA,
                            },
                            "required_true": ["byte_closed_candidate_emitted"],
                        }
                    ],
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
    )

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        require_succeeded_state=False,
    )

    report = result["report"]
    queue = result["source_queue"]
    row = queue["top_k"][0]
    assert report["accepted_manifest_count"] == 1
    assert report["rows"][0]["observed_schema"] == PACKET_MEMBER_RECOMPRESS_SCHEMA
    assert row["candidate_id"] == "family_agnostic_packet_member_candidate"
    assert row["candidate_family"] == "packet_member_recompress"
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["serialized_archive_delta"]["status"] == "realized_saving"
    assert "materializer_candidate_is_not_dispatch_authorization" in row[
        "dispatch_blockers"
    ]
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert row["runtime_consumption_proof_status"] == "missing"
    assert row["ready_for_exact_eval_dispatch"] is False


def test_harvest_family_agnostic_packet_recompress_payload_identity_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "family_candidate.json"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("payload.bin", b"A" * 4096)
    manifest = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="payload.bin",
        runtime_consumption_proof_out=proof,
        repo_root=repo,
    )
    manifest["candidate_id"] = "family_agnostic_packet_member_with_proof"
    _write_json(manifest_path, manifest)

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )

    queue = result["source_queue"]
    row = queue["top_k"][0]
    assert result["report"]["accepted_manifest_count"] == 1
    assert row["candidate_id"] == "family_agnostic_packet_member_with_proof"
    assert row["candidate_family"] == "packet_member_recompress"
    assert row["candidate_member_name"] == "payload.bin"
    assert row["candidate_member_sha256"] == manifest["candidate_member"]["sha256"]
    assert row["source_member_sha256"] == manifest["source_member"]["sha256"]
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["runtime_consumption_proof_path"] == str(proof)
    assert row["runtime_adapter_ready"] is True
    assert row["receiver_contract_satisfied"] is True
    assert row["candidate_runtime_adapter_blocker_cleared"] is True
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "family_agnostic_receiver_contract_not_satisfied" not in (
        row["dispatch_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_harvest_family_agnostic_renderer_payload_dfl1_native_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "renderer_dfl1_candidate.json"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer" * 4096)
        zf.writestr("masks.mkv", b"mask" * 4096)
        zf.writestr("optimized_poses.pt", b"pose" * 2048)
    manifest = materialize_renderer_payload_dfl1_candidate(
        archive_path=archive,
        output_archive=output,
        runtime_consumption_proof_out=proof,
        repo_root=REPO_ROOT,
    )
    manifest["candidate_id"] = "family_agnostic_renderer_payload_dfl1_with_proof"
    _write_json(manifest_path, manifest)

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )

    row = result["source_queue"]["top_k"][0]
    proof_payload = json.loads(proof.read_text(encoding="utf-8"))
    assert result["report"]["accepted_manifest_count"] == 1
    assert row["candidate_id"] == "family_agnostic_renderer_payload_dfl1_with_proof"
    assert row["schema"] == RENDERER_PAYLOAD_DFL1_SCHEMA
    assert row["candidate_family"] == "renderer_payload_dfl1"
    assert row["target_kind"] == "renderer_payload_dfl1_v1"
    assert row["materializer_id"] == "renderer_payload_dfl1_adapter"
    assert row["receiver_contract_kind"] == "source_runtime_native_renderer_payload_dfl1"
    assert row["candidate_member_name"] == "p"
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["runtime_consumption_proof_path"] == str(proof)
    assert row["runtime_adapter_ready"] is False
    assert row["receiver_contract_satisfied"] is False
    assert row["candidate_runtime_adapter_blocker_cleared"] is False
    assert row["renderer_payload_dfl1_anatomy_semantics"] == (
        "non_authoritative_planning_signal_only"
    )
    assert row["source_runtime_unpacker_parse_satisfied"] is True
    assert row["selected_member_names"] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
    ]
    assert row["payload_member_name"] == "p"
    assert row["selected_payload"] == manifest["selected_payload"]
    assert row["payload_table"] == proof_payload["payload_table"]
    assert row["reconstructed_member_sha256s"] == proof_payload[
        "reconstructed_member_sha256s"
    ]
    assert row["native_unpacker_member_sha256s"] == proof_payload[
        "runtime_consumption_probe"
    ]["native_unpacker_probe"]["member_sha256s"]
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "family_agnostic_receiver_contract_not_satisfied" in (
        row["dispatch_blockers"]
    )
    assert "renderer_payload_dfl1_full_frame_inflate_parity_missing" in (
        row["dispatch_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_exact_readiness_bridge_blocks_dfl1_without_full_frame_parity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "renderer_dfl1_candidate.json"
    source_queue_out = repo / "source_queue.json"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("renderer.bin", b"renderer" * 4096)
        zf.writestr("masks.mkv", b"mask" * 4096)
        zf.writestr("optimized_poses.pt", b"pose" * 2048)
    manifest = materialize_renderer_payload_dfl1_candidate(
        archive_path=archive,
        output_archive=output,
        runtime_consumption_proof_out=proof,
        repo_root=REPO_ROOT,
    )
    manifest["candidate_id"] = "family_agnostic_renderer_payload_dfl1_with_proof"
    _write_json(manifest_path, manifest)
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["ready_candidate_count"] == 0
    row = bridge["rows"][0]
    assert row["exact_ready_queue_written"] is False
    assert row["exact_ready_queue_path"] is None
    assert (
        "unknown_uncleared_source_dispatch_blocker:"
        "renderer_payload_dfl1_full_frame_inflate_parity_missing"
    ) in row["blockers"]
    assert bridge["score_claim"] is False
    assert bridge["ready_for_exact_eval_dispatch"] is False


def test_harvest_family_agnostic_packet_recompress_accepts_empty_member(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "family_candidate.json"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("empty.bin", b"")
    manifest = materialize_packet_member_recompress_candidate(
        archive_path=archive,
        output_archive=output,
        member_name="empty.bin",
        runtime_consumption_proof_out=proof,
        allow_size_regression=True,
        compression_methods=("stored",),
        repo_root=repo,
    )
    manifest["candidate_id"] = "family_agnostic_packet_member_empty_with_proof"
    _write_json(manifest_path, manifest)

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )

    row = result["source_queue"]["top_k"][0]
    assert row["candidate_id"] == "family_agnostic_packet_member_empty_with_proof"
    assert row["candidate_member_name"] == "empty.bin"
    assert row["candidate_member_bytes"] == 0
    assert row["source_member_bytes"] == 0
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["receiver_contract_satisfied"] is True


def test_harvest_family_agnostic_archive_section_raw_identity_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "family_candidate.json"
    raw = b"section-raw" * 512
    section = brotli.compress(raw, quality=0)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.raw", section)
    manifest = materialize_archive_section_entropy_recode_candidate(
        archive_path=archive,
        section_manifest={
            "schema": "fixture_section_manifest.v1",
            "member": {"name": "0.raw"},
            "sections": [
                {
                    "name": "section_a",
                    "index": 0,
                    "offset": 0,
                    "length": len(section),
                    "sha256": hashlib.sha256(section).hexdigest(),
                }
            ],
        },
        output_archive=output,
        section_names=("section_a",),
        brotli_qualities=(0,),
        runtime_consumption_proof_out=proof,
        allow_size_regression=True,
        repo_root=repo,
    )
    manifest["candidate_id"] = "family_agnostic_archive_section_with_proof"
    _write_json(manifest_path, manifest)

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )

    row = result["source_queue"]["top_k"][0]
    assert row["candidate_id"] == "family_agnostic_archive_section_with_proof"
    assert row["candidate_family"] == "archive_section_entropy_recode"
    assert row["candidate_member_name"] == "0.raw"
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["runtime_consumption_proof_path"] == str(proof)
    assert row["runtime_adapter_ready"] is True
    assert row["receiver_contract_satisfied"] is True
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "family_agnostic_receiver_contract_not_satisfied" not in (
        row["dispatch_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_harvest_family_agnostic_tensor_factorize_receiver_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    archive = repo / "source.zip"
    output = repo / "candidate.zip"
    proof = repo / "runtime_consumption_proof.json"
    manifest_path = repo / "family_candidate.json"
    tensor_path = repo / "weights.npy"
    vector_a = np.arange(128, dtype=np.float32)[:, None]
    vector_b = np.linspace(0.25, 2.0, 128, dtype=np.float32)[None, :]
    np.save(tensor_path, vector_a @ vector_b)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("weights.npy", tensor_path.read_bytes())
    manifest = materialize_tensor_factorize_candidate(
        archive_path=archive,
        tensor_manifest={"member_name": "weights.npy"},
        factorization_contract={
            "rank": 1,
            "cooperative_receiver_id": "fixture_tensor_factorize_receiver",
            "receiver_adapter_kind": "npz_svd_low_rank_v1",
            "max_abs_error_tolerance": 1.0e-3,
        },
        output_archive=output,
        runtime_consumption_proof_out=proof,
        repo_root=repo,
    )
    manifest["candidate_id"] = "family_agnostic_tensor_factorize_with_proof"
    _write_json(manifest_path, manifest)

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[manifest_path],
    )

    row = result["source_queue"]["top_k"][0]
    assert result["report"]["accepted_manifest_count"] == 1
    assert row["candidate_id"] == "family_agnostic_tensor_factorize_with_proof"
    assert row["candidate_family"] == "tensor_factorize"
    assert row["schema"] == TENSOR_FACTORIZE_SCHEMA
    assert row["candidate_member_name"] == "weights.npy"
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["runtime_consumption_proof_path"] == str(proof)
    assert row["runtime_adapter_ready"] is True
    assert row["receiver_contract_satisfied"] is True
    assert row["candidate_runtime_adapter_blocker_cleared"] is True
    assert "runtime_consumption_proof_missing" not in row["dispatch_blockers"]
    assert "family_agnostic_receiver_contract_not_satisfied" not in (
        row["dispatch_blockers"]
    )
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False


def test_harvest_requires_succeeded_state_for_work_queue_rows(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="failed")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    assert result["report"]["accepted_manifest_count"] == 0
    assert result["source_queue"]["n_candidates"] == 0
    assert result["report"]["rows"][0]["blockers"] == [
        "experiment_queue_state_not_succeeded:Materializer Work Fixture:failed"
    ]


def test_harvest_validates_manifest_even_when_state_succeeded(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(
        tmp_path / "external",
        authority_overrides={"score_claim": True},
    )
    work_queue = _work_queue(repo, chain)
    state = _state_path(repo, status="succeeded")

    result = harvest_materializer_chain_manifests(
        repo_root=repo,
        work_queue_path=work_queue,
        experiment_queue_state_path=state,
        experiment_queue_id="materializer_execution_queue",
    )

    assert result["report"]["accepted_manifest_count"] == 0
    assert result["source_queue"]["n_candidates"] == 0
    assert "score_claim=truthy" in result["report"]["rows"][0]["blockers"][0]


def test_harvest_cli_writes_report_and_source_queue(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    report_out = repo / "harvest_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo),
            "--chain-manifest",
            str(chain),
            "--source-queue-out",
            str(source_queue_out),
            "--report-out",
            str(report_out),
            "--require-accepted",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    source_queue = json.loads(source_queue_out.read_text(encoding="utf-8"))
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert source_queue["n_candidates"] == 1
    assert source_queue["dispatch_ready_count"] == 0
    assert report["accepted_manifest_count"] == 1
    assert report["ready_for_exact_eval_dispatch"] is False


def test_exact_readiness_bridge_promotes_valid_harvested_source_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["schema"] == EXACT_READINESS_BRIDGE_SCHEMA
    assert bridge["ready_candidate_count"] == 1
    assert bridge["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(bridge) == []
    row = bridge["rows"][0]
    assert row["candidate_id"] == "exact_ready_materializer_candidate"
    assert row["readiness_verdict"] == "exact_ready_queue_written"
    assert row["exact_ready_queue_written"] is True
    exact_ready_queue = json.loads(
        (repo / row["exact_ready_queue_path"]).read_text(encoding="utf-8")
    )
    assert exact_ready_queue["dispatch_ready_count"] == 1
    promoted = exact_ready_queue["dispatch_ready"][0]
    assert promoted["ready_for_exact_eval_dispatch"] is True
    assert promoted["score_claim"] is False
    assert promoted["dispatch_claim_required_before_gpu_or_remote_eval"] is True


def test_materializer_dispatch_plan_dry_run_does_not_write_claim(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["schema"] == DISPATCH_PLAN_SCHEMA
    assert plan["authorized_candidate_count"] == 1
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(plan) == []
    steps = dispatch_queue["experiments"][0]["steps"]
    claim_command = steps[0]["command"]
    dispatch_command = steps[1]["command"]
    assert steps[0]["id"] == "claim_lane_dispatch"
    assert "--dry-run" in claim_command
    assert "--dry-run" in dispatch_command
    assert "--claim-policy" not in dispatch_command


def test_materializer_dispatch_plan_dedupes_stable_archive_runtime_identity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    first_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    duplicate_payload = json.loads(first_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = duplicate_payload.get(key) or []
        for row in rows:
            row["candidate_id"] = "renamed_exact_ready_materializer_candidate"
    duplicate_queue = _write_json(
        repo / "renamed_duplicate.exact_ready_queue.json",
        duplicate_payload,
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[first_queue, duplicate_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 1
    assert plan["blocked_candidate_count"] == 1
    assert plan["duplicate_candidate_count"] == 1
    blocked = [row for row in plan["rows"] if row["blockers"]]
    assert blocked[0]["blockers"][0].startswith("duplicate_stable_identity:")
    assert blocked[0]["stable_identity"].startswith("archive=")
    assert ":runtime_tree=" in blocked[0]["stable_identity"]
    assert plan["rows"][0]["stable_identity"] == blocked[0]["stable_identity"]


def test_materializer_dispatch_plan_does_not_dedupe_different_runtime_tree(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    first_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    alternate_payload = json.loads(first_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = alternate_payload.get(key) or []
        for row in rows:
            row["candidate_id"] = "same_archive_alternate_runtime_tree"
            row["runtime_tree_sha256"] = "d" * 64
            runtime_manifest = row.get("runtime_manifest")
            if isinstance(runtime_manifest, dict):
                runtime_manifest["runtime_tree_sha256"] = "d" * 64
    alternate_queue = _write_json(
        repo / "alternate_runtime_tree.exact_ready_queue.json",
        alternate_payload,
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[first_queue, alternate_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["duplicate_candidate_count"] == 0
    identities = [row["stable_identity"] for row in plan["rows"]]
    assert len(set(identities)) == 2
    assert all(":runtime_tree=" in identity for identity in identities)


def test_materializer_dispatch_plan_serializes_same_lane_distinct_stable_identities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    first_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    alternate_payload = json.loads(first_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = alternate_payload.get(key) or []
        for row in rows:
            row["runtime_content_tree_sha256"] = "c" * 64
            row["runtime_tree_sha256"] = "d" * 64
            runtime_manifest = row.get("runtime_manifest")
            if isinstance(runtime_manifest, dict):
                runtime_manifest["runtime_content_tree_sha256"] = "c" * 64
                runtime_manifest["runtime_tree_sha256"] = "d" * 64
    alternate_queue = _write_json(
        repo / "same_lane_alternate_runtime_tree.exact_ready_queue.json",
        alternate_payload,
    )
    monkeypatch.setattr(
        dispatch_plan_module,
        "_exact_ready_queue_blockers",
        lambda **_: ([], {"audit_stale_ready_row_count": 0, "authority_source": "fixture"}),
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[first_queue, alternate_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 1
    assert plan["blocked_candidate_count"] == 1
    assert plan["duplicate_candidate_count"] == 0
    assert plan["serial_lane_blocked_candidate_count"] == 1
    blocked = next(row for row in plan["rows"] if row["blockers"])
    assert blocked["blockers"][0].startswith(
        "same_lane_dispatch_claim_serialization_required:"
    )
    assert blocked["dispatch_group_key"] == plan["rows"][0]["dispatch_group_key"]
    assert blocked["stable_identity"] != plan["rows"][0]["stable_identity"]


def test_materializer_dispatch_plan_blocked_rows_do_not_reserve_lane_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    first_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    alternate_payload = json.loads(first_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = alternate_payload.get(key) or []
        for row in rows:
            row["runtime_content_tree_sha256"] = "c" * 64
            row["runtime_tree_sha256"] = "d" * 64
            runtime_manifest = row.get("runtime_manifest")
            if isinstance(runtime_manifest, dict):
                runtime_manifest["runtime_content_tree_sha256"] = "c" * 64
                runtime_manifest["runtime_tree_sha256"] = "d" * 64
    alternate_queue = _write_json(
        repo / "same_lane_alternate_runtime_tree.exact_ready_queue.json",
        alternate_payload,
    )

    def fake_blockers(*, queue_path: Path, **_: object) -> tuple[list[str], dict[str, object]]:
        if Path(queue_path) == first_queue:
            return ["fixture_first_queue_blocked"], {}
        return [], {"audit_stale_ready_row_count": 0, "authority_source": "fixture"}

    monkeypatch.setattr(
        dispatch_plan_module,
        "_exact_ready_queue_blockers",
        fake_blockers,
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[first_queue, alternate_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 1
    assert plan["blocked_candidate_count"] == 1
    assert plan["serial_lane_blocked_candidate_count"] == 0
    assert any(row["blockers"] == ["fixture_first_queue_blocked"] for row in plan["rows"])
    authorized = next(row for row in plan["rows"] if row["authorized_for_dispatch_plan"])
    assert authorized["stable_identity"].endswith(f"runtime_tree={'d' * 64}:score_axis=contest_cuda")


def test_materializer_dispatch_plan_tightens_score_floor_from_frontier_scan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)
    monkeypatch.setattr(
        dispatch_plan_module,
        "_frontier_scan_active_floor_score",
        lambda _repo_root: (0.20533002902019143, "test_frontier_scan"),
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2063163866158099,
    )

    plan = result["plan"]
    assert plan["active_floor_score"] == 0.20533002902019143
    assert plan["active_floor_score_source"] == "test_frontier_scan"
    dispatch_command = result["experiment_queue"]["experiments"][0]["steps"][1][
        "command"
    ]
    score_index = dispatch_command.index("--active-floor-score") + 1
    assert dispatch_command[score_index] == "0.20533002902"


def test_materializer_dispatch_plan_blocks_archive_sha_alias_disagreement(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    ready_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    payload = json.loads(ready_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = payload.get(key) or []
        for row in rows:
            row["archive_sha256"] = "d" * 64
    stale_queue = _write_json(repo / "stale_alias.exact_ready_queue.json", payload)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[stale_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 0
    row = plan["rows"][0]
    assert any(
        blocker.startswith("archive_sha_alias_mismatch:")
        for blocker in row["blockers"]
    )


def test_materializer_dispatch_plan_requires_stable_archive_runtime_identity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    ready_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    payload = json.loads(ready_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = payload.get(key) or []
        for row in rows:
            row.pop("runtime_content_tree_sha256", None)
            runtime_manifest = row.get("runtime_manifest")
            if isinstance(runtime_manifest, dict):
                runtime_manifest.pop("runtime_content_tree_sha256", None)
    broken_queue = _write_json(
        repo / "broken_identity.exact_ready_queue.json",
        payload,
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[broken_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 0
    assert plan["blocked_candidate_count"] == 1
    assert plan["rows"][0]["blockers"] == [
        "stable_identity_runtime_content_tree_sha256_missing"
    ]
    assert plan["rows"][0]["score_claim_valid"] is False
    assert result["experiment_queue"]["controls"]["mode"] == "paused"


def test_materializer_dispatch_plan_blocks_unsupported_score_axis(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    ready_queue = repo / str(bridge["rows"][0]["exact_ready_queue_path"])
    payload = json.loads(ready_queue.read_text(encoding="utf-8"))
    for key in ("dispatch_ready", "top_k"):
        rows = payload.get(key) or []
        for row in rows:
            row["score_axis"] = "contest_cpu"
            row["target_score_axis"] = "contest_cpu"
    unsupported_queue = _write_json(
        repo / "unsupported_axis.exact_ready_queue.json",
        payload,
    )

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        exact_ready_queue_paths=[unsupported_queue],
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    assert plan["authorized_candidate_count"] == 0
    assert plan["blocked_candidate_count"] == 1
    assert plan["rows"][0]["blockers"] == [
        "stable_identity_score_axis_unsupported:contest_cpu"
    ]


def test_materializer_dispatch_plan_execute_requires_active_claim_for_dispatch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        dispatch_mode="execute",
        allow_paid_dispatch_queue=True,
        provider="lightning",
        label_prefix="fixture_materializer_exact_eval",
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["authorized_candidate_count"] == 1
    assert "execute_dispatch_queue_created_requires_operator_review" in plan["plan_blockers"]
    steps = dispatch_queue["experiments"][0]["steps"]
    claim_command = steps[0]["command"]
    dispatch_command = steps[1]["command"]
    job_id = plan["rows"][0]["dispatch_job_id"]
    assert "--dry-run" not in claim_command
    assert "--claim-policy" in dispatch_command
    assert "require_active_claim" in dispatch_command
    assert "--required-claim-platform" in dispatch_command
    assert "lightning" in dispatch_command
    assert "--required-claim-instance-job-id" in dispatch_command
    assert job_id in dispatch_command


def test_materializer_dispatch_plan_freezes_queue_when_cost_cap_exceeded(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
        estimated_cost_per_dispatch=0.30,
        max_total_cost=0.01,
    )

    plan = result["plan"]
    dispatch_queue = result["experiment_queue"]
    assert plan["authorized_candidate_count"] == 1
    assert plan["experiment_count"] == 0
    assert plan["hard_plan_blockers"] == ["estimated_total_cost_exceeds_cap:0.30>0.01"]
    assert dispatch_queue["experiments"][0]["id"] == (
        "frozen_materializer_exact_eval_dispatch"
    )
    assert dispatch_queue["experiments"][0]["steps"][0]["id"] == "noop"


def test_exact_readiness_bridge_writes_blocked_report_without_ready_queue(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["ready_candidate_count"] == 0
    assert bridge["blocked_candidate_count"] == 1
    row = bridge["rows"][0]
    assert row["exact_ready_queue_path"] is None
    assert row["blockers"]
    assert (repo / row["exact_readiness_report_path"]).is_file()


def test_exact_readiness_bridge_blocks_stale_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo, stale_runtime_proof=True)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )

    assert bridge["ready_candidate_count"] == 0
    blockers = bridge["rows"][0]["blockers"]
    assert "runtime_consumption_proof_inflate_sh_sha_mismatch" in blockers
    assert bridge["rows"][0]["exact_ready_queue_path"] is None


def test_exact_readiness_bridge_refuses_unallowlisted_source_blocker_clear(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])

    with pytest.raises(
        Exception,
        match="exact_readiness_extra_source_blocker_not_allowlisted:"
        "candidate_archive_missing",
    ):
        run_exact_readiness_bridge_for_harvested_queue(
            repo_root=repo,
            source_queue_path=source_queue_out,
            exact_readiness_out_dir=repo / "exact_readiness",
            allow_source_blockers=["candidate_archive_missing"],
            operator_override_reason="fixture",
            active_floor_archive_bytes=None,
        )


def test_exact_readiness_bridge_refuses_duplicate_candidate_ids(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    queue = harvest_result["source_queue"]
    queue["top_k"].append(dict(queue["top_k"][0]))
    _write_json(source_queue_out, queue)

    with pytest.raises(
        Exception,
        match="exact_readiness_duplicate_candidate_id:"
        "exact_ready_materializer_candidate",
    ):
        run_exact_readiness_bridge_for_harvested_queue(
            repo_root=repo,
            source_queue_path=source_queue_out,
            exact_readiness_out_dir=repo / "exact_readiness",
            active_floor_archive_bytes=None,
        )


def test_harvest_cli_can_run_explicit_exact_readiness_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    report_out = repo / "harvest_report.json"
    bridge_report_out = repo / "bridge_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo),
            "--chain-manifest",
            str(chain),
            "--source-queue-out",
            str(source_queue_out),
            "--report-out",
            str(report_out),
            "--exact-readiness-out-dir",
            str(repo / "exact_readiness"),
            "--exact-readiness-bridge-report-out",
            str(bridge_report_out),
            "--require-accepted",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    bridge = json.loads(bridge_report_out.read_text(encoding="utf-8"))
    assert bridge["schema"] == EXACT_READINESS_BRIDGE_SCHEMA
    assert bridge["candidate_count"] == 1
    assert bridge["ready_candidate_count"] == 0
    assert "exact-readiness bridge: ready=0/1" in completed.stdout


def test_harvest_cli_refuses_require_ready_without_bridge_dir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _chain_manifest(tmp_path / "external")
    source_queue_out = repo / "source_queue.json"
    report_out = repo / "harvest_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo-root",
            str(repo),
            "--chain-manifest",
            str(chain),
            "--source-queue-out",
            str(source_queue_out),
            "--report-out",
            str(report_out),
            "--exact-readiness-require-ready",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--exact-readiness-out-dir is required" in completed.stderr


def test_dispatch_plan_builds_claim_then_dry_run_queue_from_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)

    result = build_materializer_exact_eval_dispatch_plan(
        repo_root=repo,
        bridge_report_path=bridge_path,
        active_floor_archive_bytes=None,
    )

    plan = result["plan"]
    queue = result["experiment_queue"]
    assert plan["schema"] == DISPATCH_PLAN_SCHEMA
    assert plan["authorized_candidate_count"] == 1
    assert plan["blocked_candidate_count"] == 0
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert truthy_authority_field_violations(plan) == []
    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["controls"]["mode"] == "paused"
    experiment = queue["experiments"][0]
    assert experiment["metadata"]["candidate_id"] == "exact_ready_materializer_candidate"
    claim_step, dispatch_step = experiment["steps"]
    assert claim_step["id"] == "claim_lane_dispatch"
    assert "tools/claim_lane_dispatch.py" in claim_step["command"]
    assert dispatch_step["requires"] == ["claim_lane_dispatch"]
    assert dispatch_step["id"] == "dispatch_exact_eval_dry_run"
    assert "tools/parallel_dispatch_top_k.py" in dispatch_step["command"]
    assert "--dry-run" in dispatch_step["command"]


def test_dispatch_plan_execute_mode_requires_explicit_paid_queue_flag(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    exact_ready_queue = repo / "exact_ready_queue.json"
    _write_json(
        exact_ready_queue,
        {
            "schema": "optimizer_candidate_queue_v1",
            "dispatch_ready_count": 0,
            "dispatch_ready": [],
            "top_k": [],
        },
    )

    with pytest.raises(
        Exception,
        match="execute_dispatch_queue_requires_allow_paid_dispatch_queue",
    ):
        build_materializer_exact_eval_dispatch_plan(
            repo_root=repo,
            exact_ready_queue_paths=[exact_ready_queue],
            dispatch_mode="execute",
        )


def test_materializer_handoff_json_writes_refuse_overwrite_by_default(
    tmp_path: Path,
) -> None:
    harvest_path = tmp_path / "harvest.json"
    dispatch_path = tmp_path / "dispatch.json"

    write_harvest_json(harvest_path, {"value": 1})
    write_dispatch_plan_json(dispatch_path, {"value": 1})

    with pytest.raises(Exception, match="refusing_to_overwrite_json"):
        write_harvest_json(harvest_path, {"value": 2})
    with pytest.raises(Exception, match="refusing_to_overwrite_json"):
        write_dispatch_plan_json(dispatch_path, {"value": 2})

    write_harvest_json(harvest_path, {"value": 2}, overwrite=True)
    write_dispatch_plan_json(dispatch_path, {"value": 2}, overwrite=True)

    assert json.loads(harvest_path.read_text(encoding="utf-8"))["value"] == 2
    assert json.loads(dispatch_path.read_text(encoding="utf-8"))["value"] == 2


def test_dispatch_plan_cli_preserves_default_active_floor_guards(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_materializer_exact_eval_dispatch_plan_fixture",
        DISPATCH_PLAN_TOOL,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    def fake_build_materializer_exact_eval_dispatch_plan(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "plan": {
                "authorized_candidate_count": 0,
                "blocked_candidate_count": 0,
                "dispatch_mode": "dry_run",
            },
            "experiment_queue": {"schema": QUEUE_SCHEMA, "experiments": []},
        }

    module.build_materializer_exact_eval_dispatch_plan = (
        fake_build_materializer_exact_eval_dispatch_plan
    )
    module.write_json = lambda path, payload, **kwargs: None

    rc = module.main(
        [
            "--repo-root",
            str(tmp_path),
            "--exact-ready-queue",
            str(tmp_path / "exact_ready_queue.json"),
            "--dispatch-plan-out",
            str(tmp_path / "dispatch_plan.json"),
            "--experiment-queue-out",
            str(tmp_path / "dispatch_queue.json"),
        ]
    )

    assert rc == 0
    assert (
        captured["active_floor_archive_bytes"]
        == module.ACTIVE_FLOOR_ARCHIVE_BYTES
    )
    assert captured["active_floor_score"] == module.ACTIVE_FLOOR_SCORE


def test_dispatch_plan_cli_writes_paused_dry_run_queue_from_bridge(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    chain = _exact_ready_chain_manifest(repo)
    source_queue_out = repo / "source_queue.json"
    harvest_result = harvest_materializer_chain_manifests(
        repo_root=repo,
        chain_manifest_paths=[chain],
    )
    _write_json(source_queue_out, harvest_result["source_queue"])
    bridge = run_exact_readiness_bridge_for_harvested_queue(
        repo_root=repo,
        source_queue_path=source_queue_out,
        exact_readiness_out_dir=repo / "exact_readiness",
        active_floor_archive_bytes=None,
    )
    bridge_path = _write_json(repo / "bridge_report.json", bridge)
    plan_out = repo / "dispatch_plan.json"
    queue_out = repo / "dispatch_queue.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(DISPATCH_PLAN_TOOL),
            "--repo-root",
            str(repo),
            "--bridge-report",
            str(bridge_path),
            "--dispatch-plan-out",
            str(plan_out),
            "--experiment-queue-out",
            str(queue_out),
            "--active-floor-archive-bytes",
            "0",
            "--require-authorized",
            "--allow-above-active-floor-dispatch",
            "--operator-override-reason",
            "fixture",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    queue = json.loads(queue_out.read_text(encoding="utf-8"))
    assert plan["authorized_candidate_count"] == 1
    assert plan["dispatch_mode"] == "dry_run"
    assert queue["controls"]["mode"] == "paused"
    assert "authorized=1 blocked=0" in completed.stdout
