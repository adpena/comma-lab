# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tac.optimization.family_agnostic_materializers import (
    PACKET_MEMBER_MERGE_SCHEMA,
    PACKET_MEMBER_RECOMPRESS_SCHEMA,
)
from tac.optimization.local_cpu_contest_drift import EUREKA_FALSE_AUTHORITY_FIELDS
from tac.optimization.optimizer_training_signal_bridge import (
    build_optimizer_training_signal_wire_in,
    validate_optimizer_training_signal_wire_in,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate
from tac.optimization.serialized_archive_economics import (
    build_serialized_archive_delta_contract,
)
from tac.optimizer import candidate_queue as candidate_queue_module
from tac.optimizer.candidate_queue import QUEUE_SCHEMA, build_candidate_queue
from tac.optimizer.materializer_chain_harvest import MaterializerChainHarvestError


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _file_record(repo: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_bytes(repo: Path, path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _file_record(repo, path)


def _materializer_chain_manifest(
    repo: Path,
    *,
    authority_overrides: dict[str, object] | None = None,
    delta_overrides: dict[str, object] | None = None,
    artifact_overrides: dict[str, object] | None = None,
    payload_overrides: dict[str, object] | None = None,
) -> Path:
    source_archive = _write_bytes(
        repo, repo / "inputs/source.zip", b"source archive bytes"
    )
    candidate_archive = _write_bytes(repo, repo / "outputs/archive.zip", b"candidate")
    artifact = _write_json(
        repo / "outputs/candidate_manifest.json",
        {"schema": "fixture_materializer_artifact_v1"},
    )
    artifact_record = _file_record(repo, artifact)
    serialized_archive_delta = build_serialized_archive_delta_contract(
        source_archive=source_archive,
        candidate_archive=candidate_archive,
        require_realized_saving=True,
    )
    if delta_overrides:
        serialized_archive_delta.update(delta_overrides)
    artifact_payload = {
        "candidate_manifest": artifact_record,
        **(artifact_overrides or {}),
    }
    payload: dict[str, object] = {
        "schema": "byte_range_entropy_recode_chain_v1",
        "candidate_id": "fixture_materializer_candidate",
        "lane_id": "fixture_materializer_lane",
        "source_archive": source_archive,
        "source_archive_sha256": source_archive["sha256"],
        "source_archive_bytes": source_archive["bytes"],
        "candidate_archive": candidate_archive,
        "candidate_archive_sha256": candidate_archive["sha256"],
        "candidate_archive_bytes": candidate_archive["bytes"],
        "serialized_archive_delta": serialized_archive_delta,
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
        "artifacts": artifact_payload,
        "chain_steps": [
            {
                "step_id": "materialize_candidate",
                "status": "succeeded",
                "artifact": artifact_record,
            }
        ],
        "next_required_gates": ["contest_auth_eval"],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "local_cpu_anchor": {
            "score_axis": "[macOS-CPU advisory]",
            "score": 0.1928,
        },
    }
    if payload_overrides:
        payload.update(payload_overrides)
    if authority_overrides:
        payload.update(authority_overrides)
    return _write_json(repo / "outputs/chain_manifest.json", payload)


def _family_agnostic_materializer_manifest(repo: Path) -> Path:
    source_archive = _write_bytes(
        repo,
        repo / "inputs/source.zip",
        b"source archive bytes with padding",
    )
    candidate_archive = _write_bytes(
        repo,
        repo / "outputs/archive.zip",
        b"candidate archive bytes",
    )
    payload = {
        "schema": PACKET_MEMBER_RECOMPRESS_SCHEMA,
        "candidate_id": "fixture_family_materializer_candidate",
        "lane_id": "fixture_family_materializer_lane",
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "byte_closed_candidate_emitted": True,
        "receiver_contract_satisfied": False,
        "receiver_verification": {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": False,
            "proof_present": False,
            "proof_path": None,
            "blockers": ["runtime_consumption_proof_missing"],
        },
        "readiness_blockers": ["runtime_consumption_proof_missing"],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    return _write_json(repo / "outputs/family_materializer.json", payload)


def _packet_member_merge_materializer_manifest(repo: Path) -> Path:
    source_archive_path = repo / "inputs/source.zip"
    source_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive_path, "w") as archive:
        archive.writestr("renderer.bin", b"A" * 32)
        archive.writestr("masks.mkv", b"B" * 24)
        archive.writestr("optimized_poses.pt", b"C" * 16)
    source_archive = _file_record(repo, source_archive_path)
    candidate_archive_path = repo / "outputs/archive.zip"
    candidate_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidate_archive_path, "w") as archive:
        archive.writestr("__packet_member_merge_v1.bin", b"merged")
    candidate_archive = _file_record(repo, candidate_archive_path)
    proof = _write_json(
        repo / "outputs/runtime_consumption_proof.json",
        {
            "schema": "family_agnostic_runtime_consumption_proof_v1",
            "proof_kind": "packet_member_merge_runtime_adapter_consumption_proof.v1",
            "target_kind": "packet_member_merge_v1",
            "materializer_id": "packet_member_merge_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_merge",
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "candidate_archive_sha256": candidate_archive["sha256"],
            "runtime_adapter_ready": True,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "promotable": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        },
    )
    runtime = repo / "outputs" / "candidate.runtime"
    runtime.mkdir(parents=True)
    (runtime / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
    )
    payload = {
        "schema": PACKET_MEMBER_MERGE_SCHEMA,
        "candidate_id": "fixture_packet_member_merge_candidate",
        "lane_id": "fixture_packet_member_merge_lane",
        "source_archive": source_archive,
        "candidate_archive": candidate_archive,
        "candidate_member": {
            "name": "__packet_member_merge_v1.bin",
            "bytes": 6,
            "sha256": hashlib.sha256(b"merged").hexdigest(),
        },
        "byte_closed_candidate_emitted": True,
        "materializer_id": "packet_member_merge_adapter",
        "target_kind": "packet_member_merge_v1",
        "receiver_contract_kind": "family_agnostic_packet_member_merge",
        "receiver_contract_satisfied": True,
        "runtime_adapter_ready": True,
        "runtime_consumption_proof_path": proof.relative_to(repo).as_posix(),
        "receiver_verification": {
            "schema": "family_agnostic_runtime_consumption_proof_verification.v1",
            "receiver_contract_satisfied": True,
            "proof_present": True,
            "proof_path": proof.relative_to(repo).as_posix(),
            "runtime_adapter_ready": True,
            "runtime_adapter_sha256": "b" * 64,
            "blockers": [],
        },
        "packet_member_merge_receiver_runtime": {
            "schema": "packet_member_merge_receiver_runtime.v1",
            "runtime_adapter_ready": True,
            "runtime_dir": runtime.relative_to(repo).as_posix(),
            "runtime_manifest_path": "outputs/candidate.runtime_adapter.json",
            "runtime_tree_sha256": "b" * 64,
            "source_runtime_dir": "submissions/robust_current",
            "blockers": [],
        },
        "selected_member_names": [
            "renderer.bin",
            "masks.mkv",
            "optimized_poses.pt",
        ],
        "readiness_blockers": [],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
    }
    return _write_json(repo / "outputs/packet_member_merge_materializer.json", payload)


def test_a1_rollup_merges_m5_ranking_without_dispatch_overclaim(tmp_path: Path) -> None:
    repo = tmp_path
    manifest_a = _write_json(
        repo / "experiments/results/coord/v_a/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_a/submission_dir/archive.zip",
            "archive_sha256": "a" * 64,
            "archive_size_bytes": 178262,
            "archive_unchanged_from_a1": True,
            "inflate_py_sha256_new": "b" * 64,
            "runtime_smoke_checked": False,
        },
    )
    manifest_b = _write_json(
        repo / "experiments/results/coord/v_b/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_b/submission_dir/archive.zip",
            "archive_sha256": "c" * 64,
            "archive_size_bytes": 178262,
            "archive_unchanged_from_a1": True,
            "inflate_py_sha256_new": "d" * 64,
            "runtime_smoke_checked": False,
        },
    )
    rollup = _write_json(
        repo / "experiments/results/coord/rollup.json",
        {
            "schema_version": "constrained_coord_search_rollup_v1",
            "lane_id": "lane_pr101_bias_constrained_coord_search",
            "evidence_grade": "[predicted; constrained coord search on A1 substrate]",
            "variants": [
                {
                    "variant_id": "v_a",
                    "coords": {"c0_0": -1.0, "c0_2": -1.0, "c1_1": -1.0},
                    "build_manifest_relpath": manifest_a.relative_to(repo).as_posix(),
                    "submission_name": "v_a",
                    "inflate_py_sha256": "b" * 64,
                    "archive_sha256": "a" * 64,
                },
                {
                    "variant_id": "v_b",
                    "coords": {"c0_0": -1.0, "c0_2": 0.0, "c1_1": -1.0},
                    "build_manifest_relpath": manifest_b.relative_to(repo).as_posix(),
                    "submission_name": "v_b",
                    "inflate_py_sha256": "d" * 64,
                    "archive_sha256": "c" * 64,
                },
            ],
        },
    )
    m5 = _write_json(
        repo / "experiments/results/m5/sweep_manifest.json",
        {
            "schema_version": 1,
            "tool": "tools/sweep_m5max_hnerv_cluster.py",
            "summary": {
                "operator_decision_queue": [
                    {
                        "candidate_id": "v_a",
                        "macos_cpu_score": 0.1930,
                        "predicted_contest_cpu_gha": 0.19299,
                        "tag": "[macOS-CPU calibrated]",
                    },
                    {
                        "candidate_id": "v_b",
                        "macos_cpu_score": 0.1928,
                        "predicted_contest_cpu_gha": 0.19279,
                        "tag": "[macOS-CPU calibrated]",
                    },
                ]
            },
        },
    )

    queue = build_candidate_queue([rollup, m5], repo_root=repo, top_k=2)

    assert queue["schema"] == QUEUE_SCHEMA
    assert queue["dispatch_ready_count"] == 0
    assert [row["candidate_id"] for row in queue["top_k"]] == ["v_b", "v_a"]
    best = queue["top_k"][0]
    assert best["ready_for_exact_eval_dispatch"] is False
    assert best["score_claim"] is False
    assert (
        best["archive_path"]
        == "experiments/results/coord/v_b/submission_dir/archive.zip"
    )
    assert best["candidate_archive_sha256"] == "c" * 64
    assert best["predicted_contest_cpu_gha"] == 0.19279
    assert "macos_cpu_is_not_contest_cuda_evidence" in best["dispatch_blockers"]
    assert "requires_exact_eval_readiness_gate" in best["dispatch_blockers"]


def test_candidate_queue_identity_preserves_distinct_representation_families(
    tmp_path: Path,
) -> None:
    def manifest(
        *,
        path: Path,
        candidate_family: str,
        representation_family: str,
        substrate_family: str,
        score: float,
    ) -> Path:
        return _write_json(
            path,
            {
                "schema": "representation_training_probe_manifest_v1",
                "candidate_id": "seed17",
                "candidate_family": candidate_family,
                "representation_family": representation_family,
                "substrate_family": substrate_family,
                "param_schema": "representation_training_manifest_params_v1",
                "results": [
                    {
                        "stage_index": 1,
                        "stage_module": "smoke",
                        "epochs_run": 1,
                        "best_score": score,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )

    hnerv = manifest(
        path=tmp_path / "hnerv_manifest.json",
        candidate_family="hnerv_optimizer_probe",
        representation_family="hnerv",
        substrate_family="nerv_family",
        score=0.199,
    )
    siren = manifest(
        path=tmp_path / "siren_manifest.json",
        candidate_family="siren_optimizer_probe",
        representation_family="siren",
        substrate_family="non_nerv_learned",
        score=0.198,
    )

    queue = build_candidate_queue([hnerv, siren], repo_root=tmp_path, top_k=10)

    assert queue["n_candidates"] == 2
    assert queue["top_k_count"] == 2
    assert {row["candidate_family"] for row in queue["top_k"]} == {
        "hnerv_optimizer_probe",
        "siren_optimizer_probe",
    }


def test_candidate_queue_identity_preserves_same_family_distinct_params(
    tmp_path: Path,
) -> None:
    def manifest(*, path: Path, q: int, score: float) -> Path:
        return _write_json(
            path,
            {
                "schema": "representation_training_probe_manifest_v1",
                "candidate_id": "seed17",
                "candidate_family": "hnerv_optimizer_probe",
                "representation_family": "hnerv",
                "substrate_family": "nerv_family",
                "profile": "local_training",
                "lane_id": "offline_hnerv_training",
                "param_schema": "hnerv_training_params_v1",
                "candidate_params": {"quant": q, "temperature": 0.7},
                "results": [
                    {
                        "stage_index": 1,
                        "stage_module": "smoke",
                        "epochs_run": 1,
                        "best_score": score,
                    }
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
        )

    q5 = manifest(path=tmp_path / "q5_manifest.json", q=5, score=0.199)
    q7 = manifest(path=tmp_path / "q7_manifest.json", q=7, score=0.198)

    queue = build_candidate_queue([q5, q7], repo_root=tmp_path, top_k=10)

    assert queue["n_candidates"] == 2
    assert queue["top_k_count"] == 2
    assert [row["candidate_params"]["quant"] for row in queue["top_k"]] == [7, 5]


def test_merge_candidate_or_preserves_score_affecting_booleans() -> None:
    merged = candidate_queue_module._merge_candidate(
        {
            "candidate_id": "same",
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "score_affecting_runtime_changed": False,
        },
        {
            "candidate_id": "same",
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
            "score_affecting_runtime_changed": True,
        },
    )
    reversed_merge = candidate_queue_module._merge_candidate(
        merged,
        {
            "candidate_id": "same",
            "score_affecting_payload_changed": False,
            "charged_bits_changed": False,
            "score_affecting_runtime_changed": False,
        },
    )

    assert reversed_merge["score_affecting_payload_changed"] is True
    assert reversed_merge["charged_bits_changed"] is True
    assert reversed_merge["score_affecting_runtime_changed"] is True


def test_materializer_chain_manifest_becomes_custody_checked_planning_row(
    tmp_path: Path,
) -> None:
    manifest = _materializer_chain_manifest(tmp_path)

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["n_candidates"] == 1
    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "fixture_materializer_candidate"
    assert row["candidate_family"] == "byte_range_entropy_recode"
    assert row["archive_candidate_verified"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["serialized_archive_delta"]["status"] == "realized_saving"
    assert (
        row["serialized_archive_delta_validated"]["expected_status"]
        == "realized_saving"
    )
    assert row["score_affecting_change_proof"]["archive_changed"] is True
    assert row["local_advisory_axes"][0]["score_axis"] == "[macOS-CPU advisory]"
    assert (
        "materializer_chain_is_not_dispatch_authorization" in row["dispatch_blockers"]
    )
    assert (
        "exact_auth_eval_result_required_before_score_claim" in row["dispatch_blockers"]
    )


def test_family_agnostic_materializer_manifest_becomes_custody_checked_planning_row(
    tmp_path: Path,
) -> None:
    manifest = _family_agnostic_materializer_manifest(tmp_path)

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["n_candidates"] == 1
    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "fixture_family_materializer_candidate"
    assert row["candidate_family"] == "packet_member_recompress"
    assert row["archive_candidate_verified"] is True
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["runtime_adapter_ready"] is False
    assert row["receiver_contract_satisfied"] is False
    assert row["candidate_runtime_adapter_blocker_cleared"] is False
    assert row["serialized_archive_delta"]["status"] == "realized_saving"
    assert row["score_affecting_change_proof"]["archive_changed"] is True
    assert row["runtime_consumption_proof_status"] == "missing"
    assert "materializer_candidate_is_not_dispatch_authorization" in row[
        "dispatch_blockers"
    ]
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert "family_agnostic_receiver_contract_not_satisfied" in row[
        "dispatch_blockers"
    ]


def test_packet_member_merge_materializer_harvest_preserves_receiver_runtime(
    tmp_path: Path,
) -> None:
    manifest = _packet_member_merge_materializer_manifest(tmp_path)

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["n_candidates"] == 1
    assert row["candidate_id"] == "fixture_packet_member_merge_candidate"
    assert row["candidate_family"] == "packet_member_merge"
    assert row["receiver_contract_satisfied"] is True
    assert row["runtime_adapter_ready"] is True
    assert row["candidate_runtime_dir"] == "outputs/candidate.runtime"
    assert row["candidate_runtime_tree_sha256"] == "b" * 64
    assert row["packet_member_merge_runtime_dir"] == "outputs/candidate.runtime"
    assert row["packet_member_merge_receiver_runtime_tree_sha256"] == "b" * 64
    assert row["packet_member_merge_source_runtime_dir"] == (
        "submissions/robust_current"
    )
    assert row["selected_member_names"] == [
        "renderer.bin",
        "masks.mkv",
        "optimized_poses.pt",
    ]
    assert "family_agnostic_receiver_contract_not_satisfied" not in row[
        "dispatch_blockers"
    ]


def test_candidate_queue_merge_prefers_adapter_ready_runtime_contract(
    tmp_path: Path,
) -> None:
    ready_manifest = _packet_member_merge_materializer_manifest(tmp_path)
    stale_payload = json.loads(ready_manifest.read_text(encoding="utf-8"))
    stale_payload.update(
        {
            "runtime_adapter_ready": False,
            "receiver_contract_satisfied": False,
            "candidate_runtime_adapter_blocker_cleared": False,
            "candidate_runtime_dir": "outputs/stale_source_runtime",
            "candidate_runtime_tree_sha256": "a" * 64,
            "packet_member_merge_runtime_dir": "outputs/stale_source_runtime",
            "packet_member_merge_receiver_runtime_tree_sha256": "a" * 64,
            "packet_member_merge_receiver_runtime": {
                "runtime_dir": "outputs/stale_source_runtime",
                "runtime_tree_sha256": "a" * 64,
            },
        }
    )
    stale_manifest = _write_json(
        tmp_path / "outputs/stale_packet_member_merge_materializer.json",
        stale_payload,
    )

    queue = build_candidate_queue([stale_manifest, ready_manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["n_candidates"] == 1
    assert row["runtime_adapter_ready"] is True
    assert row["receiver_contract_satisfied"] is True
    assert row["candidate_runtime_adapter_blocker_cleared"] is True
    assert row["candidate_runtime_dir"] == "outputs/candidate.runtime"
    assert row["candidate_runtime_tree_sha256"] == "b" * 64
    assert row["packet_member_merge_receiver_runtime"]["runtime_dir"] == (
        "outputs/candidate.runtime"
    )
    assert row["packet_member_merge_receiver_runtime_tree_sha256"] == "b" * 64


def test_materializer_chain_harvest_preserves_runtime_context(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    (runtime_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
    )
    receiver_proof = _write_json(
        tmp_path / "outputs/receiver_proof.json",
        {"schema": "fixture_receiver_proof_v1"},
    )
    manifest = _materializer_chain_manifest(
        tmp_path,
        artifact_overrides={
            "receiver_proof": _file_record(tmp_path, receiver_proof),
        },
        payload_overrides={
            "schema": "inverse_scorer_cell_candidate_chain_v1",
            "source_runtime_dir": "runtime",
            "inflate_runtime_dir": "runtime",
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_family"] == "inverse_scorer_cell"
    assert row["source_runtime_dir"] == "runtime"
    assert row["inflate_runtime_dir"] == "runtime"
    assert row["runtime_consumption_proof_required"] is True
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["runtime_consumption_proof_path"] == "outputs/receiver_proof.json"


def test_materializer_chain_truthy_authority_fails_closed(tmp_path: Path) -> None:
    manifest = _materializer_chain_manifest(
        tmp_path,
        authority_overrides={"score_claim": True},
    )

    with pytest.raises(MaterializerChainHarvestError, match="score_claim=truthy"):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_materializer_chain_delta_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = _materializer_chain_manifest(
        tmp_path,
        delta_overrides={"candidate_archive_bytes": 999},
    )

    with pytest.raises(
        MaterializerChainHarvestError,
        match="serialized_archive_delta_candidate_bytes_mismatch",
    ):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_materializer_chain_delta_source_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = _materializer_chain_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_bytes = payload["candidate_archive"]["bytes"]
    false_source_bytes = 999
    payload["serialized_archive_delta"].update(
        {
            "source_archive_bytes": false_source_bytes,
            "archive_delta_bytes": candidate_bytes - false_source_bytes,
            "realized_saved_bytes": false_source_bytes - candidate_bytes,
            "savings_realized": True,
            "status": "realized_saving",
        }
    )
    _write_json(manifest, payload)

    with pytest.raises(
        MaterializerChainHarvestError,
        match="serialized_archive_delta_source_bytes_mismatch",
    ):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_materializer_chain_missing_delta_fails_closed(tmp_path: Path) -> None:
    manifest = _materializer_chain_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("serialized_archive_delta")
    _write_json(manifest, payload)

    with pytest.raises(
        MaterializerChainHarvestError,
        match="serialized_archive_delta_missing",
    ):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_materializer_chain_artifact_hash_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = _materializer_chain_manifest(
        tmp_path,
        artifact_overrides={
            "corrupt_artifact": {
                "path": "outputs/candidate_manifest.json",
                "bytes": 1,
                "sha256": "0" * 64,
            }
        },
    )

    with pytest.raises(
        MaterializerChainHarvestError, match=r"corrupt_artifact.*sha256"
    ):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_materializer_chain_symlink_artifact_fails_closed(tmp_path: Path) -> None:
    manifest = _materializer_chain_manifest(tmp_path)
    target = tmp_path / "outputs/candidate_manifest.json"
    link = tmp_path / "outputs/candidate_manifest_link.json"
    link.symlink_to(target)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    symlink_record = _file_record(tmp_path, target)
    symlink_record["path"] = link.relative_to(tmp_path).as_posix()
    payload["artifacts"]["symlink_artifact"] = symlink_record
    _write_json(manifest, payload)

    with pytest.raises(
        MaterializerChainHarvestError,
        match="artifact:symlink_artifact_file_is_symlink",
    ):
        build_candidate_queue([manifest], repo_root=tmp_path)


def test_codec_search_report_rows_are_payload_planning_not_archive_dispatch(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "optuna_search_report.json",
        {
            "schema": "codec_op_optuna_search_report_v1",
            "tool": "tools/codec_op_optuna_search.py",
            "op_module": "fixture.module",
            "op_class": "FixtureCodecOp",
            "evidence_grade": "[CPU-prep+optuna_tpe]",
            "evidence_semantics": "cpu_codec_op_search_forensic",
            "all_evaluations": [
                {
                    "eval_idx": 0,
                    "params": {"quality": 11},
                    "bytes_out": 120,
                    "reconstruction_rms": 0.0,
                    "fitness": 120.0,
                    "pareto_frontier": True,
                    "materialized_payload_path": "payloads/eval_00000.section",
                    "materialized_payload_sha256": "e" * 64,
                    "materialized_payload_bytes": 120,
                    "materialized_payload_contract": "raw_codecop_encode_blob",
                },
                {
                    "eval_idx": 1,
                    "params": {"quality": 1},
                    "bytes_out": -1,
                    "fitness": None,
                    "error": "RuntimeError: fixture",
                },
            ],
        },
    )

    queue = build_candidate_queue([report], repo_root=tmp_path, top_k=2)

    best = queue["top_k"][0]
    assert best["candidate_id"] == "fixturecodecop_eval_00000"
    assert best["candidate_substream_bytes"] == 120
    assert "archive_path" not in best
    assert best["ready_for_exact_eval_dispatch"] is False
    assert best["score_affecting_payload_changed"] is False
    assert "codec_op_payload_not_archive_zip" in best["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in best["dispatch_blockers"]

    failed = queue["top_k"][1]
    assert failed["candidate_id"] == "fixturecodecop_eval_00001"
    assert "optimizer_eval_failed" in failed["dispatch_blockers"]


def test_stale_archive_path_does_not_outrank_materialized_payload(
    tmp_path: Path,
) -> None:
    report = _write_json(
        tmp_path / "optuna_search_report.json",
        {
            "schema": "codec_op_optuna_search_report_v1",
            "tool": "tools/codec_op_optuna_search.py",
            "op_class": "FixtureCodecOp",
            "all_evaluations": [
                {
                    "eval_idx": 0,
                    "fitness": 100.0,
                    "materialized_payload_path": "payloads/eval_00000.section",
                    "materialized_payload_sha256": "e" * 64,
                    "materialized_payload_bytes": 100,
                }
            ],
        },
    )
    stale_archive = _write_json(
        tmp_path / "hnerv_manifest.json",
        {
            "schema": "hnerv_lowlevel_exact_eval_candidate_manifest_v1",
            "candidate_id": "stale_archive_candidate",
            "candidate_archive_sha256": "a" * 64,
            "candidate_archive_bytes": 123,
        },
    )

    queue = build_candidate_queue([stale_archive, report], repo_root=tmp_path, top_k=2)

    assert queue["top_k"][0]["candidate_id"] == "fixturecodecop_eval_00000"
    stale = next(
        row
        for row in queue["top_k"]
        if row["candidate_id"] == "stale_archive_candidate"
    )
    assert stale["archive_candidate_verified"] is False
    assert stale["candidate_archive_path_unverified"] is True
    assert "candidate_archive_path_unverified" in stale["dispatch_blockers"]


def test_archive_candidate_verification_hashes_archive_before_verified(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release_surface"
    release.mkdir()
    archive = release / "archive.zip"
    archive.write_bytes(b"real archive bytes")
    actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = _write_json(
        tmp_path / "hnerv_manifest.json",
        {
            "schema": "hnerv_lowlevel_exact_eval_candidate_manifest_v1",
            "candidate_id": "hash_mismatch_candidate",
            "candidate_archive_sha256": "f" * 64,
            "candidate_archive_bytes": archive.stat().st_size,
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path, top_k=1)
    row = queue["top_k"][0]

    assert row["archive_candidate_verified"] is False
    assert row["candidate_archive_sha256_observed"] == actual_sha
    assert "candidate_archive_sha256_mismatch" in row["dispatch_blockers"]


def test_predicted_param_sweep_manifest_is_forced_non_dispatchable(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "codec_sweep_manifest.json",
        {
            "schema_version": "codec_op_param_sweep_manifest.v1",
            "evidence_semantics": "cpu_substrate_predicted_band",
            "candidates": [
                {
                    "candidate_id": "unsafe_source_claim",
                    "predicted_score": 0.1,
                    "ready_for_exact_eval_dispatch": True,
                    "score_claim": True,
                    "rank_or_kill_eligible": True,
                    "exact_cuda_auth_eval": True,
                    "contest_cuda_auth_eval": True,
                    "field_selection_ready_for_exact_eval_dispatch": True,
                    "charged_bits_changed": True,
                    "op_params": {"q": 11},
                }
            ],
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["exact_cuda_auth_eval"] is False
    assert row["contest_cuda_auth_eval"] is False
    assert row["field_selection_ready_for_exact_eval_dispatch"] is False
    assert row["charged_bits_changed"] is False
    assert row["evidence_semantics"] == "cpu_substrate_predicted_band"
    assert "predicted_score_is_not_score_evidence" in row["dispatch_blockers"]
    assert validate_proxy_candidate(row) == []


def test_optimizer_guided_queue_schema_is_adapted_as_proxy_only(
    tmp_path: Path,
) -> None:
    solver_stack_wire_in = build_optimizer_training_signal_wire_in(
        candidate_id="bias_sidecar_cmaes_style_stdlib_anchor",
        profile_id="pr101_bias_sidecar",
        lane_id="offline_pr101_bias_sidecar_candidate_generation",
        lane_class="a1_pr101_bias_sidecar_prefilter",
        candidate_family="a1_pr101_runtime_bias_plus_sidecar_probe",
        param_schema="pr101_bias_sidecar_candidate_params_v1",
        candidate_params={
            "bias_b": -1.0,
            "bias_g": -1.0,
            "bias_r": -1.0,
            "sidecar_f1_r": 0.0,
        },
        source_anchor="fixture",
        score_lowering_hypothesis="fixture",
        dispatch_blockers=["sidecar_param_requires_archive_builder_support"],
    )
    queue_path = _write_json(
        tmp_path / "optimizer_guided_queue.json",
        {
            "schema": "optimizer_guided_candidate_queue_v1",
            "tool": "tools/build_optimizer_guided_candidate_queue.py",
            "profile": "pr101_bias_sidecar",
            "optimizer": "cmaes",
            "optimizer_status": "cmaes_style_stdlib",
            "profile_contract": {
                "dispatch_blockers": [
                    "sidecar_param_requires_archive_builder_support",
                    "runtime_consumption_proof_required",
                ]
            },
            "top_k": [
                {
                    "candidate_id": "bias_sidecar_cmaes_style_stdlib_anchor",
                    "lane_id": "offline_pr101_bias_sidecar_candidate_generation",
                    "lane_class": "a1_pr101_bias_sidecar_prefilter",
                    "candidate_family": "a1_pr101_runtime_bias_plus_sidecar_probe",
                    "candidate_params": {
                        "bias_b": -1.0,
                        "bias_g": -1.0,
                        "bias_r": -1.0,
                        "sidecar_f1_r": 0.0,
                    },
                    "proxy_objective": 0.19285,
                    "rank_score": 0.19285,
                    "solver_stack_wire_in": solver_stack_wire_in,
                    "score_claim": True,
                    "ready_for_exact_eval_dispatch": True,
                    "rank_or_kill_eligible": True,
                    "contest_cuda_auth_eval": True,
                    "charged_bits_changed": True,
                }
            ],
        },
    )

    queue = build_candidate_queue([queue_path], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == "bias_sidecar_cmaes_style_stdlib_anchor"
    assert row["profile"] == "pr101_bias_sidecar"
    assert row["optimizer_status"] == "cmaes_style_stdlib"
    assert row["op_params"] == row["candidate_params"]
    assert row["rank_score_field"] == "proxy_objective_not_score"
    assert "sidecar_param_requires_archive_builder_support" in row["dispatch_blockers"]
    assert (
        "optimizer_guided_queue_requires_archive_materialization"
        in row["dispatch_blockers"]
    )
    assert (
        "optimizer_guided_row_has_no_runtime_consumption_proof"
        in row["dispatch_blockers"]
    )
    assert validate_optimizer_training_signal_wire_in(row["solver_stack_wire_in"]) == []
    assert (
        row["solver_stack_wire_in"]["cathedral_autopilot_wire_in"]["dispatch_ready"]
        is False
    )
    assert row["solver_stack_wire_in"]["atom_wire_in"]["atom_kind"] == "meta_lagrangian"
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_accepts_mlx_dynamic_learned_sweep_plan_as_planning_only(
    tmp_path: Path,
) -> None:
    solver_stack_wire_in = build_optimizer_training_signal_wire_in(
        candidate_id="prefix_k032::mlx_local_response::micro",
        profile_id="mlx_dynamic_learned_sweep",
        lane_id="mlx_dynamic_learned_sweep_planning",
        lane_class="decoder_q_selective_dqs1",
        candidate_family="decoder_q_selective_dqs1",
        representation_family="decoder_q_selective_dqs1",
        substrate_family="[macOS-MLX research-signal]",
        training_signal_kind="mlx_dynamic_learned_sweep_proxy",
        param_schema="mlx_dynamic_learned_sweep_config_params_v1",
        candidate_params={
            "source_candidate_id": "prefix_k032",
            "sweep_config_id": "mlx_local_response",
            "optimization_pass_id": "micro",
        },
        source_anchor="fixture",
        score_lowering_hypothesis="fixture",
        dispatch_blockers=["score_claim_requires_exact_auth_eval_result"],
    )
    pairing_wire_in = build_optimizer_training_signal_wire_in(
        candidate_id=(
            "prefix_k032::mlx_local_response::micro::optimizer_scheduler::muon_adamw_cosine_representation"
        ),
        profile_id="mlx_dynamic_learned_sweep_optimizer_pairing",
        lane_id="mlx_dynamic_learned_sweep_planning",
        lane_class="decoder_q_selective_dqs1",
        candidate_family="decoder_q_selective_dqs1",
        representation_family="decoder_q_selective_dqs1",
        substrate_family="[macOS-MLX research-signal]",
        training_signal_kind="optimizer_scheduler_pairing_proxy",
        param_schema="mlx_dynamic_optimizer_scheduler_pairing_params_v1",
        candidate_params={
            "source_candidate_id": "prefix_k032",
            "sweep_config_id": "mlx_local_response",
            "optimization_pass_id": "micro",
            "optimizer_scheduler_descriptor_id": "muon_adamw_cosine_representation",
        },
        source_anchor="fixture",
        score_lowering_hypothesis="fixture",
        dispatch_blockers=["optimizer_scheduler_pairing_is_planning_only"],
        variant_axes=[
            "optimizer_scheduler_recipe",
            "parameter_group_lr_policy",
            "same_candidate_config_pass",
        ],
        paired_modes=[
            "same_candidate_config_pass_different_optimizer_scheduler",
            "same_optimizer_scheduler_different_candidate",
        ],
    )
    plan = _write_json(
        tmp_path / "mlx_dynamic_plan.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "tool": "tools/plan_mlx_dynamic_learned_sweep.py",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "optimizer_scheduler_candidates": [
                {
                    "schema": "optimizer_scheduler_descriptor.v1",
                    "descriptor_id": "muon_adamw_cosine_representation",
                    "optimizer": "tac.optimization.muon.MuonOptimizer+torch.optim.AdamW",
                    "scheduler": "cosine_warmup",
                    "config_sha256": "a" * 64,
                    "parameter_group_lr_policy_id": "embedding_theta1_hidden_muon_adamw",
                    "parameter_group_lr_policy": {
                        "schema": "parameter_group_lr_policy.v1",
                        "policy_id": "embedding_theta1_hidden_muon_adamw",
                    },
                    "allowed_axis_tags": ["[macOS-MLX research-signal]"],
                    "allowed_target_modes": ["mlx_research_signal"],
                    "rank_score_field": "planner_priority_not_score",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
            "ranked_sweep_rows": [
                {
                    "schema": "mlx_dynamic_learned_sweep_row.v1",
                    "candidate_id": "prefix_k032",
                    "sweep_config_id": "mlx_local_response",
                    "optimization_pass_id": "micro",
                    "family": "decoder_q_selective_dqs1",
                    "acquisition_value": 0.0125,
                    "component_axis_context": {"seg": 0.7, "pose": 0.3},
                    "canonical_equation_provenance": {
                        "canonical_equation_id": "pairset_component_marginal_score_decomposition_v1"
                    },
                    "master_gradient_provenance": {"anchor_count": 2},
                    "ready_for_local_sweep": True,
                    "exact_eval_candidate": False,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "solver_stack_wire_in": solver_stack_wire_in,
                }
            ],
            "optimizer_scheduler_pairings": [
                {
                    "schema": "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing.v1",
                    "queue_candidate_id": (
                        "prefix_k032::mlx_local_response::micro::optimizer_scheduler::muon_adamw_cosine_representation"
                    ),
                    "candidate_id": "prefix_k032",
                    "parent_queue_candidate_id": "prefix_k032::mlx_local_response::micro",
                    "sweep_config_id": "mlx_local_response",
                    "optimization_pass_id": "micro",
                    "optimization_scale": "micro",
                    "family": "decoder_q_selective_dqs1",
                    "optimizer_scheduler_descriptor_id": "muon_adamw_cosine_representation",
                    "optimizer_scheduler_config_sha256": "a" * 64,
                    "optimizer": "tac.optimization.muon.MuonOptimizer+torch.optim.AdamW",
                    "scheduler": "cosine_warmup",
                    "parameter_group_lr_policy_id": "embedding_theta1_hidden_muon_adamw",
                    "parameter_group_lr_policy_sha256": "b" * 64,
                    "rank_score": -0.012499999999,
                    "rank_score_field": (
                        "parent_negative_acquisition_value_plus_recipe_tiebreak_not_score"
                    ),
                    "paired_ablation_contract": {
                        "schema": "optimizer_scheduler_paired_ablation_contract.v1",
                        "tool_wiring": {
                            "schema": "optimizer_scheduler_pairing_tool_wiring.v1",
                            "ablation_surfaces": [
                                "src/tac/findings_lagrangian/phase_2_ablation/ablation_framework.py"
                            ],
                            "xray_surfaces": ["tools/master_gradient_xray.py"],
                            "atom_surfaces": ["src/tac/atom/ledger.py"],
                            "materialization_surfaces": [
                                "tools/materialize_decoder_q_selective_runtime_candidate.py"
                            ],
                            "freezing_surfaces": [
                                "src/tac/freezing/swa_checkpoint_averaging.py"
                            ],
                            "observation_surfaces": [
                                "tools/append_mlx_dynamic_sweep_observation.py"
                            ],
                            "score_claim": False,
                            "promotion_eligible": False,
                            "rank_or_kill_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
                        },
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    },
                    "solver_stack_wire_in": pairing_wire_in,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                    "gpu_launched": False,
                }
            ],
        },
    )

    queue = build_candidate_queue([plan], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == "prefix_k032::mlx_local_response::micro"
    assert row["source_candidate_id"] == "prefix_k032"
    assert row["rank_score"] == -0.0125
    assert row["rank_score_field"] == "negative_acquisition_value_proxy_not_score"
    assert row["component_axis_context"] == {"seg": 0.7, "pose": 0.3}
    assert row["canonical_equation_provenance"]["canonical_equation_id"].endswith("_v1")
    assert row["master_gradient_provenance"] == {"anchor_count": 2}
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert "mlx_dynamic_learned_sweep_plan_is_proxy_signal" in row["dispatch_blockers"]
    assert row["solver_stack_wire_in"]["candidate_id"] == row["candidate_id"]
    assert validate_optimizer_training_signal_wire_in(row["solver_stack_wire_in"]) == []
    assert row["consumer_payload"]["mlx_dynamic_learned_sweep"][
        "source_candidate_id"
    ] == ("prefix_k032")
    assert validate_proxy_candidate(row) == []
    recipe = next(
        item
        for item in queue["top_k"]
        if item["candidate_id"]
        == "optimizer_scheduler::muon_adamw_cosine_representation"
    )
    assert (
        recipe["parameter_group_lr_policy_id"] == "embedding_theta1_hidden_muon_adamw"
    )
    assert recipe["ready_for_exact_eval_dispatch"] is False
    assert recipe["score_claim"] is False
    assert (
        recipe["consumer_payload"]["optimizer_scheduler_recipe"]["config_sha256"]
        == "a" * 64
    )
    assert (
        "requires_training_telemetry_before_candidate_selection"
        in recipe["dispatch_blockers"]
    )
    assert validate_proxy_candidate(recipe) == []
    pairing = next(
        item
        for item in queue["top_k"]
        if item["candidate_family"] == "optimizer_scheduler_paired_sweep_recipe"
    )
    assert pairing["candidate_id"].endswith(
        "::optimizer_scheduler::muon_adamw_cosine_representation"
    )
    assert pairing["source_candidate_id"] == "prefix_k032"
    assert pairing["config_sha256"] == "a" * 64
    assert pairing["parameter_group_lr_policy_sha256"] == "b" * 64
    assert (
        pairing["consumer_payload"]["optimizer_scheduler_pairing"][
            "parent_queue_candidate_id"
        ]
        == "prefix_k032::mlx_local_response::micro"
    )
    assert pairing["consumer_payload"]["optimizer_scheduler_pairing"][
        "paired_ablation_contract"
    ]["tool_wiring"]["freezing_surfaces"] == [
        "src/tac/freezing/swa_checkpoint_averaging.py"
    ]
    assert (
        "requires_same_seed_local_ablation_before_recipe_posterior_update"
        in pairing["dispatch_blockers"]
    )
    assert (
        validate_optimizer_training_signal_wire_in(
            pairing["consumer_payload"]["optimizer_scheduler_pairing"][
                "solver_stack_wire_in"
            ]
        )
        == []
    )
    assert validate_proxy_candidate(pairing) == []


def test_candidate_queue_sorts_mixed_ranked_and_planning_only_rows(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "mixed_rows.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "optimizer_scheduler_candidates": [
                {
                    "schema": "optimizer_scheduler_descriptor.v1",
                    "descriptor_id": "adamw_cosine_micro",
                    "optimizer": "torch.optim.AdamW",
                    "scheduler": "cosine_warmup",
                    "config_sha256": "b" * 64,
                    "rank_score_field": "planner_priority_not_score",
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
            "ranked_sweep_rows": [
                {
                    "schema": "mlx_dynamic_learned_sweep_row.v1",
                    "candidate_id": "ranked",
                    "sweep_config_id": "local",
                    "optimization_pass_id": "micro",
                    "acquisition_value": 0.01,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    queue = build_candidate_queue([plan], repo_root=tmp_path)

    assert [row["candidate_id"] for row in queue["top_k"]] == [
        "ranked::local::micro",
        "optimizer_scheduler::adamw_cosine_micro",
    ]


def test_candidate_queue_rejects_mlx_dynamic_sweep_nested_authority(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "mlx_dynamic_plan_nested_authority.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ranked_sweep_rows": [
                {
                    "schema": "mlx_dynamic_learned_sweep_row.v1",
                    "candidate_id": "prefix_k032",
                    "sweep_config_id": "mlx_local_response",
                    "optimization_pass_id": "micro",
                    "family": "decoder_q_selective_dqs1",
                    "acquisition_value": 0.0125,
                    "waterbucket_context": {"promotable": True},
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match=r"waterbucket_context\.promotable=truthy"):
        build_candidate_queue([plan], repo_root=tmp_path)


def test_candidate_queue_rejects_mlx_dynamic_sweep_normalized_gain_mismatch(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "mlx_dynamic_plan_bad_normalized_gain.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ranked_sweep_rows": [
                {
                    "schema": "mlx_dynamic_learned_sweep_row.v1",
                    "candidate_id": "prefix_k032",
                    "sweep_config_id": "mlx_local_response",
                    "optimization_pass_id": "micro",
                    "family": "decoder_q_selective_dqs1",
                    "acquisition_value": 0.0125,
                    "non_authoritative_mlx_gain_sum": 0.03,
                    "non_authoritative_normalized_full_video_gain_sum": 18.0,
                    "non_authoritative_mlx_window_gain_sum": 18.0,
                    "full_video_denominator": 600,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    with pytest.raises(
        ValueError,
        match="non_authoritative_normalized_full_video_gain_sum mismatch",
    ):
        build_candidate_queue([plan], repo_root=tmp_path)


def test_candidate_queue_rejects_optimizer_pairing_without_accepted_parent(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "mlx_dynamic_plan_orphan_pairing.json",
        {
            "schema": "mlx_dynamic_learned_sweep_plan.v1",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "optimizer_scheduler_pairings": [
                {
                    "schema": "mlx_dynamic_learned_sweep_optimizer_scheduler_pairing.v1",
                    "queue_candidate_id": "prefix_k032::local::micro::optimizer_scheduler::adamw",
                    "parent_queue_candidate_id": "prefix_k032::local::micro",
                    "candidate_id": "prefix_k032",
                    "sweep_config_id": "local",
                    "optimization_pass_id": "micro",
                    "family": "decoder_q_selective_dqs1",
                    "optimizer_scheduler_descriptor_id": "adamw",
                    "rank_score": -0.01,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="parent_queue_candidate_id not accepted"):
        build_candidate_queue([plan], repo_root=tmp_path)


def test_candidate_queue_accepts_local_cpu_eureka_as_spend_triage_only(
    tmp_path: Path,
) -> None:
    eureka_path = _write_json(
        tmp_path / "local_cpu_eureka.json",
        {
            **dict.fromkeys(EUREKA_FALSE_AUTHORITY_FIELDS, False),
            "schema": "local_cpu_contest_drift_eureka_signal.v1",
            "candidate_id": "pairset_drop_one_rank999_pair0001",
            "candidate_archive_sha256": "a" * 64,
            "local_score": 0.1919,
            "projected_contest_score": 0.19189,
            "conservative_projected_contest_score": 0.191893,
            "auth_frontier_score": 0.1920,
            "eureka_margin": 0.000107,
            "eureka_trigger": True,
            "recommended_action": "dispatch_exact_auth_anchor",
            "source_artifact": "local_cpu_advisory.json",
            "candidate_trust_region_blockers": [],
        },
    )

    queue = build_candidate_queue([eureka_path], repo_root=tmp_path, top_k=1)
    row = queue["top_k"][0]

    assert queue["dispatch_ready_count"] == 0
    assert row["candidate_id"] == (
        "pairset_drop_one_rank999_pair0001::local_cpu_contest_drift_eureka"
    )
    assert row["exact_auth_anchor_requested"] is True
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert (
        row["rank_score_field"]
        == "conservative_projected_contest_score_false_authority"
    )
    assert (
        "positive_eureka_requires_manual_exact_auth_anchor_claim"
        in row["dispatch_blockers"]
    )
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_accepts_byte_shaving_campaign_plan_as_planning_only(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "byte_shaving_plan.json",
        {
            "schema": "byte_shaving_campaign_plan.v1",
            "tool": "tools/plan_byte_shaving_campaign.py",
            "campaign_id": "seed7_post_train_shave",
            "candidate_id": "seed7",
            "lane_id": "boostnerv_post_train_shave",
            "frontier_axis": "[macOS-MLX research-signal]",
            "source_signal_refs": [
                {"kind": "master_gradient_anchor", "archive_sha256": "a" * 64}
            ],
            "auth_eval_refs": [{"kind": "contest_cpu_anchor", "path": "auth.json"}],
            "mlx_calibration_refs": [
                {"kind": "strict_calibration", "path": "mlx_cal.json"}
            ],
            "scorer_response_refs": [
                {"kind": "scorer_response_dataset", "path": "rows.json"}
            ],
            "dispatch_blockers": [
                "requires_byte_closed_materialization_before_dispatch"
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "sweep_ladder": [
                {
                    "sweep_id": "top_0001",
                    "unit_count": 1,
                    "candidate_saved_bytes": 50,
                    "expected_delta_score": -0.00002,
                    "expected_score_gain": 0.00002,
                    "operation_families": ["entropy_recode"],
                    "selected_unit_ids": ["span_a"],
                    "selected_operations": [
                        {
                            "unit_id": "span_a",
                            "operation_id": "entropy_recode",
                            "operation_family": "entropy_recode",
                        }
                    ],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
            "operation_set_ladder": [
                {
                    "schema": "byte_shaving_coupled_operation_set.v1",
                    "operation_set_id": "opset_combo_0001",
                    "combo_id": "combo_0001",
                    "unit_count": 2,
                    "candidate_saved_bytes": 120,
                    "expected_delta_score": -0.00005,
                    "expected_score_gain": 0.00005,
                    "confidence": 0.6,
                    "confidence_adjusted_gain": 0.00003,
                    "operation_families": ["entropy_recode", "drop_pair"],
                    "selected_unit_ids": ["span_a", "pair0371"],
                    "selected_operations": [
                        {
                            "unit_id": "span_a",
                            "operation_id": "entropy_recode",
                            "operation_family": "entropy_recode",
                        },
                        {
                            "unit_id": "pair0371",
                            "operation_id": "drop_pair",
                            "operation_family": "drop_pair",
                        },
                    ],
                    "chosen_operation_sequence": [
                        {
                            "unit_id": "span_a",
                            "operation_id": "entropy_recode",
                            "operation_family": "entropy_recode",
                        },
                        {
                            "unit_id": "pair0371",
                            "operation_id": "drop_pair",
                            "operation_family": "drop_pair",
                        },
                    ],
                    "chosen_operation_sequence_sha256": "a" * 64,
                    "chosen_operation_sequence_is_permutation": True,
                    "chosen_operation_sequence_source": "bounded_permutation_ladder_rank_1",
                    "active_interactions": [{"interaction_id": "shared_header"}],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
            "combination_ladder": [
                {
                    "combo_id": "combo_0001",
                    "unit_count": 2,
                    "candidate_saved_bytes": 120,
                    "expected_delta_score": -0.00005,
                    "expected_score_gain": 0.00005,
                    "confidence": 0.6,
                    "confidence_adjusted_gain": 0.00003,
                    "operation_families": ["entropy_recode", "drop_pair"],
                    "selected_unit_ids": ["span_a", "pair0371"],
                    "selected_operations": [
                        {
                            "unit_id": "span_a",
                            "operation_id": "entropy_recode",
                            "operation_family": "entropy_recode",
                        },
                        {
                            "unit_id": "pair0371",
                            "operation_id": "drop_pair",
                            "operation_family": "drop_pair",
                        },
                    ],
                    "active_interactions": [{"interaction_id": "shared_header"}],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    queue = build_candidate_queue([plan], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == (
        "seed7_post_train_shave::operation_set::opset_combo_0001"
    )
    assert row["selection_kind"] == "operation_set"
    assert row["operation_set_id"] == "opset_combo_0001"
    assert row["param_schema"] == "byte_shaving_coupled_operation_set.v1"
    assert row["rank_score"] == -0.00005
    assert row["candidate_saved_bytes"] == 120
    assert row["predicted_saved_bytes"] == 120
    assert row["predicted_saved_bytes_semantics"] == (
        "planner_model_only_not_serialized_archive_delta"
    )
    assert row["serialized_archive_delta"]["status"] == "missing_archive_bytes"
    assert row["serialized_archive_delta"]["modeled_saved_bytes"] == 120
    assert "serialized_archive_delta_bytes_missing" in row["dispatch_blockers"]
    assert (
        "modeled_savings_without_realized_serialized_savings"
        in row["dispatch_blockers"]
    )
    assert row["source_signal_refs"][0]["kind"] == "master_gradient_anchor"
    assert row["mlx_calibration_refs"][0]["kind"] == "strict_calibration"
    assert row["consumer_payload"]["selected_unit_ids"] == [
        "span_a",
        "pair0371",
    ]
    assert row["consumer_payload"]["operation_set_id"] == "opset_combo_0001"
    assert row["consumer_payload"]["chosen_operation_sequence"][0]["unit_id"] == (
        "span_a"
    )
    assert row["consumer_payload"]["chosen_operation_sequence_sha256"] == "a" * 64
    assert row["candidate_params"]["chosen_operation_sequence_is_permutation"] is True
    assert "selected_operations_require_materializer" in row["dispatch_blockers"]
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["score_claim"] is False
    assert validate_proxy_candidate(row) == []


def test_candidate_queue_rejects_byte_shaving_campaign_nested_authority(
    tmp_path: Path,
) -> None:
    plan = _write_json(
        tmp_path / "unsafe_byte_shaving_plan.json",
        {
            "schema": "byte_shaving_campaign_plan.v1",
            "campaign_id": "unsafe",
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "combination_ladder": [
                {
                    "combo_id": "combo_0001",
                    "candidate_saved_bytes": 1,
                    "expected_delta_score": -0.1,
                    "selected_unit_ids": ["x"],
                    "selected_operations": [{"unit_id": "x", "promotable": True}],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            ],
        },
    )

    with pytest.raises(
        ValueError, match=r"selected_operations\[0\]\.promotable=truthy"
    ):
        build_candidate_queue([plan], repo_root=tmp_path)


def test_candidate_queue_rejects_truthy_local_cpu_eureka_authority(
    tmp_path: Path,
) -> None:
    eureka_path = _write_json(
        tmp_path / "local_cpu_eureka.json",
        {
            **dict.fromkeys(EUREKA_FALSE_AUTHORITY_FIELDS, False),
            "schema": "local_cpu_contest_drift_eureka_signal.v1",
            "candidate_id": "unsafe",
            "eureka_trigger": True,
            "recommended_action": "dispatch_exact_auth_anchor",
            "score_claim_valid": True,
        },
    )

    with pytest.raises(ValueError, match="score_claim_valid"):
        build_candidate_queue([eureka_path], repo_root=tmp_path, top_k=1)


def test_unknown_candidates_schema_is_not_ranked_by_generic_numeric_fields(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "unknown_candidate_rows.json",
        {
            "schema": "future_optimizer_rows_without_adapter_v1",
            "candidates": [
                {
                    "candidate_id": "looks_good_but_unknown",
                    "predicted_score": 0.0001,
                    "proxy_score": 0.0001,
                    "macos_cpu_score": 0.0001,
                    "score_claim": True,
                    "ready_for_exact_eval_dispatch": True,
                }
            ],
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)

    assert queue["n_candidates"] == 0
    assert queue["top_k"] == []
    assert queue["unsupported_sources"] == [
        {
            "path": "unknown_candidate_rows.json",
            "schema": "future_optimizer_rows_without_adapter_v1",
            "reason": (
                "unsupported_candidates_schema_requires_explicit_adapter_or_codec_op_param_sweep_manifest_v1"
            ),
        }
    ]
    assert queue["source_schemas"][0]["status"] == "unsupported"
    assert queue["source_schemas"][0]["extracted_candidate_count"] == 0


def test_sort_key_ignores_unadapted_proxy_score_fields() -> None:
    assert candidate_queue_module._candidate_sort_key(
        {
            "candidate_id": "bare_proxy_score",
            "proxy_score": 0.0001,
            "predicted_score": 0.0001,
            "macos_cpu_score": 0.0001,
        }
    ) == (4, 1, "bare_proxy_score")
    assert candidate_queue_module._candidate_sort_key(
        {
            "candidate_id": "adapted_proxy_score",
            "proxy_score": 0.0001,
            "rank_score": 0.0001,
        }
    ) == (4, 0, 0.0001, "adapted_proxy_score")


def test_kaggle_proxy_manifest_becomes_canonical_non_dispatchable_queue_row(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "proxy_sweep_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_sweep_v1",
            "optimizer": "cmaes",
            "optimizer_status": "cmaes_style_stdlib",
            "evidence_semantics": "kaggle_gpu_proxy_config_search_only_not_exact_auth_eval",
            "dispatch_blockers": [
                "kaggle_proxy_substrate_not_contest_exact_eval",
                "no_archive_zip_emitted",
            ],
            "best_candidate": {
                "candidate_id": "proxy_cmaes_0007",
                "trial_index": 7,
                "optimizer": "cmaes",
                "optimizer_status": "cmaes_style_stdlib",
                "params": {"delta_scale": 0.01, "bias_r": -1.0},
                "proxy_objective": 0.192851,
                "proxy_components": {"anchor_proximity": 0.0},
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "proxy_only": True,
            },
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == "proxy_cmaes_0007"
    assert row["rank_score"] == 0.192851
    assert row["rank_score_field"] == "proxy_objective"
    assert row["target_modes"] == ["contest_exact_eval_planning"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert "kaggle_proxy_substrate_not_contest_exact_eval" in row["dispatch_blockers"]
    assert (
        "kaggle_proxy_output_requires_archive_builder_promotion"
        in row["dispatch_blockers"]
    )
    assert validate_proxy_candidate(row) == []
    assert queue["dispatch_ready_count"] == 0


def test_kaggle_proxy_manifest_preserves_profile_lane_class_and_param_schema(
    tmp_path: Path,
) -> None:
    manifest = _write_json(
        tmp_path / "proxy_sweep_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_sweep_v1",
            "lane_id": "kaggle_pr101_bias_refine",
            "lane_class": "pr101_kaggle_bias_refine",
            "candidate_family": "pr101_runtime_consumed_bias_refinement",
            "param_schema": "pr101_kaggle_proxy_bias_runtime_params_v1",
            "optimizer": "cmaes",
            "optimizer_status": "cmaes_style_stdlib",
            "best_candidate": {
                "candidate_id": "bias_refine_cmaes_0017",
                "param_schema": "pr101_kaggle_proxy_bias_runtime_params_v1",
                "params": {
                    "bias_b": -0.998,
                    "bias_g": -1.003,
                    "bias_r": -0.997,
                },
                "proxy_components": {"anchor_proximity": 0.01},
                "proxy_objective": 0.19285,
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            "dispatch_blockers": ["kaggle_proxy_substrate_not_contest_exact_eval"],
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["lane_id"] == "kaggle_pr101_bias_refine"
    assert row["lane_class"] == "pr101_kaggle_bias_refine"
    assert row["candidate_family"] == "pr101_runtime_consumed_bias_refinement"
    assert row["param_schema"] == "pr101_kaggle_proxy_bias_runtime_params_v1"
    assert set(row["op_params"]) == {"bias_b", "bias_g", "bias_r"}
    assert row["proxy_only"] is True
    assert queue["dispatch_ready_count"] == 0


def test_pr101_kaggle_proxy_runtime_packet_becomes_byte_closed_planning_row(
    tmp_path: Path,
) -> None:
    packet_dir = tmp_path / "experiments/results/kaggle/packet"
    packet_dir.mkdir(parents=True)
    archive = packet_dir / "archive.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    manifest = _write_json(
        packet_dir / "runtime_packet_manifest.json",
        {
            "schema": "pr101_kaggle_proxy_runtime_packet_v1",
            "candidate_id": "proxy_cmaes_0037",
            "packet_dir": packet_dir.relative_to(tmp_path).as_posix(),
            "archive_changed": False,
            "packet_archive": {
                "relpath": "archive.zip",
                "bytes": archive.stat().st_size,
                "sha256": "a" * 64,
                "members": [],
            },
            "source_archive": {
                "bytes": archive.stat().st_size,
                "sha256": "a" * 64,
            },
            "runtime_custody": {"runtime_tree_sha256": "b" * 64},
            "runtime_consumed_params": {
                "bias_r": -1.0,
                "bias_g": -0.9,
                "bias_b": -0.8,
            },
            "runtime_patch": {"patched_file": "inflate.py"},
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == "proxy_cmaes_0037_pr101_proxy_runtime_packet"
    assert row["lane_id"] == "pr101_kaggle_proxy_runtime_packet_exact_eval"
    assert row["candidate_archive_path"].endswith("/archive.zip")
    assert row["score_affecting_runtime_changed"] is True
    assert row["score_affecting_payload_changed"] is False
    assert row["charged_bits_changed"] is False
    assert row["runtime_consumption_proof_required"] is True
    assert row["runtime_consumption_proof_status"] == "missing"
    assert row["runtime_consumption_proof_path"].endswith(
        "/runtime_consumption_proof.json"
    )
    assert "runtime_consumption_proof_missing" in row["dispatch_blockers"]
    assert "exact_cuda_auth_eval_missing" in row["dispatch_blockers"]
    assert "requires_exact_eval_readiness_gate" in row["dispatch_blockers"]
    assert queue["dispatch_ready_count"] == 0


def test_hnerv_q10_release_manifest_becomes_exact_eval_planning_row(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release_surface"
    release.mkdir()
    archive = release / "archive.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    manifest = _write_json(
        release / "archive_manifest.json",
        {
            "schema": "hnerv_lowlevel_release_surface_manifest_v1",
            "candidate_id": "pr106_q10_151byte_brotli",
            "lane_id": "pr106_q10_151byte_brotli",
            "candidate_archive_sha256": "a" * 64,
            "candidate_archive_bytes": archive.stat().st_size,
            "source_archive_sha256": "b" * 64,
            "source_archive_bytes": archive.stat().st_size + 151,
            "score_claim": False,
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["candidate_id"] == "pr106_q10_151byte_brotli"
    assert row["lane_id"] == "pr106_q10_151byte_brotli"
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["source_archive_bytes"] == archive.stat().st_size + 151
    assert "exact_cuda_auth_eval_missing" in row["dispatch_blockers"]


def test_pr103_hidden_gem_release_manifest_becomes_exact_eval_planning_row(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release_surface"
    release.mkdir()
    archive = release / "archive.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    manifest = _write_json(
        release / "archive_manifest.json",
        {
            "schema": "pr103_hidden_gem_release_surface_manifest_v1",
            "candidate_id": "pr103_ac_merged_range_drop_u32_at_160824_20260510_agent",
            "candidate_archive": {
                "path": "archive.zip",
                "archive_sha256": "c" * 64,
                "archive_bytes": archive.stat().st_size,
            },
            "source_archive": {
                "archive_sha256": "d" * 64,
                "archive_bytes": archive.stat().st_size + 4,
            },
            "section_sha256_proof": {"runtime_consumed_section_changed": True},
            "runtime_consumption_no_op_proof": {"state_dict_changed_vs_source": True},
            "score_claim": False,
        },
    )

    queue = build_candidate_queue([manifest], repo_root=tmp_path)
    row = queue["top_k"][0]

    assert row["lane_id"] == "pr103_ac_hidden_gem"
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["runtime_consumed_section_changed"] is True
    assert row["decoded_state_changed"] is True


def test_queue_sanitizes_non_finite_legacy_telemetry(tmp_path: Path) -> None:
    report = _write_json(
        tmp_path / "meta_lagrangian_report.json",
        {
            "schema": "meta_lagrangian_search_v1",
            "evidence_semantics": "local_proxy_prediction_forensic",
            "top_k_forensic": [
                {
                    "candidate_id": "legacy_proxy",
                    "archive_bytes": 159973,
                    "proxy_score": 0.1927,
                    "rank_key": float("inf"),
                    "rank_or_kill_eligible": True,
                    "contest_cuda_auth_eval": True,
                    "charged_bits_changed": True,
                }
            ],
        },
    )

    queue = build_candidate_queue([report], repo_root=tmp_path)

    assert queue["top_k"][0]["rank_key"] is None
    assert validate_proxy_candidate(queue["top_k"][0]) == []
    json.dumps(queue, allow_nan=False)


def test_concrete_eval_queue_outranks_lower_numeric_proxy_score(tmp_path: Path) -> None:
    rollup_manifest = _write_json(
        tmp_path / "experiments/results/coord/v_real/build_manifest.json",
        {
            "archive_path": "experiments/results/coord/v_real/submission_dir/archive.zip",
            "archive_sha256": "f" * 64,
            "archive_size_bytes": 178262,
            "inflate_py_sha256_new": "1" * 64,
        },
    )
    rollup = _write_json(
        tmp_path / "experiments/results/coord/rollup.json",
        {
            "schema_version": "constrained_coord_search_rollup_v1",
            "variants": [
                {
                    "variant_id": "v_real",
                    "build_manifest_relpath": rollup_manifest.relative_to(
                        tmp_path
                    ).as_posix(),
                }
            ],
        },
    )
    m5 = _write_json(
        tmp_path / "experiments/results/m5/sweep_manifest.json",
        {
            "tool": "tools/sweep_m5max_hnerv_cluster.py",
            "summary": {
                "operator_decision_queue": [
                    {
                        "candidate_id": "v_real",
                        "predicted_contest_cpu_gha": 0.193,
                        "macos_cpu_score": 0.193,
                    }
                ]
            },
        },
    )
    proxy = _write_json(
        tmp_path / "reports/meta_lagrangian.json",
        {
            "schema": "meta_lagrangian_search_v1",
            "evidence_semantics": "local_proxy_prediction_forensic",
            "top_k_forensic": [
                {
                    "candidate_id": "lower_proxy_only",
                    "proxy_score": 0.100,
                    "archive_path": None,
                }
            ],
        },
    )

    queue = build_candidate_queue([rollup, m5, proxy], repo_root=tmp_path)

    assert [row["candidate_id"] for row in queue["top_k"]] == [
        "v_real",
        "lower_proxy_only",
    ]
