# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import stat
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.hdm8_selector_cuda_gate import (
    HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA,
    SELECTOR_CUDA_TRANSFER_CALIBRATION_SCHEMA,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    MATERIALIZER_ID as BYTE_RANGE_MATERIALIZER_ID,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    RECEIVER_CONTRACT_ID as BYTE_RANGE_RECEIVER_CONTRACT_ID,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    RECEIVER_CONTRACT_KIND as BYTE_RANGE_RECEIVER_CONTRACT_KIND,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    RECEIVER_PROOF_SCHEMA as BYTE_RANGE_RECEIVER_PROOF_SCHEMA,
)
from tac.optimization.byte_range_entropy_recode_materializer import (
    TARGET_KIND as BYTE_RANGE_TARGET_KIND,
)
from tac.optimization.serialized_archive_economics import (
    CANDIDATE_ARCHIVE_LARGER_BLOCKER,
    MISSING_ARCHIVE_BYTES_BLOCKER,
    MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER,
    build_serialized_archive_delta_contract,
)
from tac.optimizer.exact_readiness import (
    ACTIVE_FLOOR_ARCHIVE_BYTES,
    ACTIVE_FLOOR_SCORE,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL,
    ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE,
    ACTIVE_RATE_ONLY_FLOOR_SCORE,
    ACTIVE_SCORE_FRONTIER_SCORE,
    promote_candidate_for_exact_eval,
    validate_runtime_consumption_proof,
)
from tac.optimizer.exact_ready_audit import audit_exact_ready_queue
from tac.packet_compiler.fp11_source_brotli_recode import (
    FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID,
    FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA,
    FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND,
    FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_active_floor_score_tracks_score_frontier_not_rate_only_anchor() -> None:
    assert ACTIVE_FLOOR_ARCHIVE_BYTES == 185_578
    assert ACTIVE_RATE_ONLY_FLOOR_SCORE == 0.2089810755823297
    assert ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE == 0.2063163866158099
    assert (
        ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL
        == "pr106_format0c_exact_radix_paired_20260515T0918Z_cuda"
    )
    assert ACTIVE_SCORE_FRONTIER_SCORE == 0.2063163866158099
    assert ACTIVE_FLOOR_SCORE == ACTIVE_SCORE_FRONTIER_SCORE


def test_family_agnostic_runtime_proof_requires_runtime_consumption_signal(
    tmp_path: Path,
) -> None:
    proof = _write_json(
        tmp_path / "receiver_only_proof.json",
        {
            "schema": "family_agnostic_runtime_consumption_proof_v1",
            "target_kind": "packet_member_zip_header_elide_v1",
            "materializer_id": "packet_member_zip_header_elide_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
            "receiver_contract_satisfied": True,
            "candidate_archive_sha256": "a" * 64,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    blockers, facts = validate_runtime_consumption_proof(
        {
            "target_kind": "packet_member_zip_header_elide_v1",
            "materializer_id": "packet_member_zip_header_elide_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_zip_header_elide",
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_status": "present",
            "runtime_consumption_proof_path": str(proof),
        },
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256="a" * 64,
    )

    assert facts["runtime_consumption_proof_schema"] == (
        "family_agnostic_runtime_consumption_proof_v1"
    )
    assert "runtime_consumption_proof_not_proven" in blockers


def test_runtime_adapter_ready_proof_binds_candidate_runtime_tree_sha(
    tmp_path: Path,
) -> None:
    proof = _write_json(
        tmp_path / "adapter_proof.json",
        {
            "schema": "family_agnostic_runtime_consumption_proof_v1",
            "target_kind": "packet_member_merge_v1",
            "materializer_id": "packet_member_merge_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_merge",
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
            "candidate_archive_sha256": "a" * 64,
            "runtime_adapter_manifest": {
                "runtime_adapter_ready": True,
                "runtime_tree_sha256": "b" * 64,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    blockers, facts = validate_runtime_consumption_proof(
        {
            "target_kind": "packet_member_merge_v1",
            "materializer_id": "packet_member_merge_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_merge",
            "runtime_adapter_ready": True,
            "candidate_runtime_tree_sha256": "c" * 64,
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_status": "present",
            "runtime_consumption_proof_path": str(proof),
        },
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256="a" * 64,
    )

    assert facts["runtime_consumption_proof_runtime_tree_sha256"] == "b" * 64
    assert facts["candidate_row_adapter_runtime_tree_sha256"] == "c" * 64
    assert facts["runtime_consumption_proof_expected_runtime_tree_sha256"] is None
    assert facts["candidate_row_expected_runtime_tree_sha256"] is None
    assert "runtime_consumption_proof_expected_runtime_tree_sha_missing" in blockers
    assert (
        "candidate_row_expected_runtime_tree_sha_missing_for_runtime_adapter"
        in blockers
    )
    assert "runtime_consumption_proof_runtime_tree_sha_mismatch" in blockers


def test_byte_range_runtime_proof_clears_receiver_schema_blocker(
    tmp_path: Path,
) -> None:
    proof = _write_json(
        tmp_path / "byte_range_receiver_proof.json",
        {
            "schema": BYTE_RANGE_RECEIVER_PROOF_SCHEMA,
            "target_kind": BYTE_RANGE_TARGET_KIND,
            "materializer_id": BYTE_RANGE_MATERIALIZER_ID,
            "receiver_contract_id": BYTE_RANGE_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": BYTE_RANGE_RECEIVER_CONTRACT_KIND,
            "runtime_consumption_proof_passed": True,
            "ready_for_exact_eval_runtime": True,
            "runtime_consumption_probe": {"passed": True},
            "decoder_state_parity_proof": {"passed": True},
            "candidate_archive_sha256": "a" * 64,
            "candidate_member_sha256": "b" * 64,
            "runtime_tree_sha256": "c" * 64,
            "expected_runtime_tree_sha256": "c" * 64,
            "archive_byte_ranges": [
                {
                    "archive_member_name": "x",
                    "section_name": "ac_histograms_brotli",
                    "candidate_start": 2,
                    "candidate_end": 10,
                }
            ],
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )

    blockers, facts = validate_runtime_consumption_proof(
        {
            "target_kind": BYTE_RANGE_TARGET_KIND,
            "materializer_id": BYTE_RANGE_MATERIALIZER_ID,
            "receiver_contract_id": BYTE_RANGE_RECEIVER_CONTRACT_ID,
            "receiver_contract_kind": BYTE_RANGE_RECEIVER_CONTRACT_KIND,
            "candidate_member_sha256": "b" * 64,
            "runtime_adapter_ready": True,
            "candidate_runtime_tree_sha256": "c" * 64,
            "expected_runtime_tree_sha256": "c" * 64,
            "runtime_consumption_proof_required": True,
            "runtime_consumption_proof_status": "present",
            "runtime_consumption_proof_path": str(proof),
        },
        repo_root=tmp_path,
        queue_dir=tmp_path,
        submission_dir=None,
        archive_sha256="a" * 64,
    )

    assert facts["runtime_consumption_proof_schema"] == BYTE_RANGE_RECEIVER_PROOF_SCHEMA
    assert facts["runtime_consumption_proof_archive_sha256"] == "a" * 64
    assert facts["runtime_consumption_proof_runtime_tree_sha256"] == "c" * 64
    assert facts["candidate_row_adapter_runtime_tree_sha256"] == "c" * 64
    assert facts["runtime_consumption_proof_expected_runtime_tree_sha256"] == "c" * 64
    assert facts["candidate_row_expected_runtime_tree_sha256"] == "c" * 64
    assert "runtime_consumption_proof_schema_unsupported" not in blockers
    assert blockers == []


def _load_parallel_dispatch_tool():
    path = REPO_ROOT / "tools" / "parallel_dispatch_top_k.py"
    spec = importlib.util.spec_from_file_location("parallel_dispatch_top_k_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path, member: str = "0.bin", payload: bytes = b"payload") -> tuple[int, str]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def _make_submission(repo: Path) -> tuple[Path, int, str]:
    submission = repo / "experiments/results/exact_ready_fixture"
    archive = submission / "archive.zip"
    archive_bytes, archive_sha = _write_archive(archive)
    inflate_py = submission / "inflate.py"
    inflate_py.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "Path(sys.argv[2]).write_bytes(b'')\n",
        encoding="utf-8",
    )
    inflate = submission / "inflate.sh"
    inflate.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
        "python \"$SCRIPT_DIR/inflate.py\" \"$1\" \"$2\"\n",
        encoding="utf-8",
    )
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (submission / "report.txt").write_text(
        f"archive.zip sha256={archive_sha} bytes={archive_bytes}\n",
        encoding="utf-8",
    )
    _write_json(
        submission / "archive_manifest.json",
        {
            "score_claim": False,
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": archive_bytes,
            "candidate_archive": {"member_name": "0.bin"},
        },
    )
    (repo / "upstream").mkdir(parents=True, exist_ok=True)
    (repo / "upstream/evaluate.py").write_text("# fixture\n", encoding="utf-8")
    return submission, archive_bytes, archive_sha


def _make_queue(repo: Path, submission: Path, archive_bytes: int, archive_sha: str) -> Path:
    proof_path = _write_pr101_runtime_proof(submission, archive_sha, proven=True)
    return _write_json(
        repo / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "fixture_candidate",
                    "lane_id": "fixture_lane",
                    "archive_path": (submission / "archive.zip").relative_to(repo).as_posix(),
                    "candidate_archive_sha256": archive_sha,
                    "candidate_archive_bytes": archive_bytes,
                    "ready_for_exact_eval_dispatch": False,
                    "score_claim": False,
                    "predicted_contest_cpu_gha": 0.1,
                    "score_affecting_payload_changed": True,
                    "charged_bits_changed": True,
                    "runtime_consumption_proof_required": True,
                    "runtime_consumption_proof_status": "present",
                    "runtime_consumption_proof_path": proof_path.relative_to(repo).as_posix(),
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "requires_exact_eval_readiness_gate",
                        "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )


def _mark_queue_row_as_inverse_scorer_chain(
    queue: Path,
    *,
    strict_full_frame_parity: bool,
    exact_auth_boundary: bool,
    proof_backed: bool = True,
) -> None:
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    proof_path = queue.parent / "inflate_parity_probe.json"
    proof_false_authority = {
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "gpu_launched": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }
    output_tree = {
        "exists": True,
        "file_count": 1,
        "total_bytes": 16,
        "tree_sha256": "1" * 64,
        "blockers": [],
        "files": [{"path": "0.raw", "bytes": 16, "sha256": "2" * 64}],
    }
    proof_payload = (
        json.dumps(
            {
                "schema": "inverse_scorer_cell_inflate_parity_probe_v1",
                "proof_scope": "full_frame_inflate_output_tree",
                "full_frame_inflate_output_parity_claim": strict_full_frame_parity,
                "output_bytes_identical": strict_full_frame_parity,
                "output_contract_nonempty": strict_full_frame_parity,
                "output_contract_paths_match": strict_full_frame_parity,
                "differing_path_count": 0,
                "blockers": []
                if strict_full_frame_parity
                else ["candidate_inflate_output_parity_missing"],
                "missing_from_candidate": [],
                "extra_in_candidate": [],
                "source_output_tree": output_tree,
                "candidate_output_tree": output_tree,
                **proof_false_authority,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if proof_backed:
        proof_path.write_bytes(proof_payload)
    row_false_authority = {
        key: value
        for key, value in proof_false_authority.items()
        if key not in {"score_affecting_payload_changed", "charged_bits_changed"}
    }
    row.update(
        {
            **row_false_authority,
            "schema": "inverse_scorer_cell_candidate_chain_v1",
            "kind": "inverse_scorer_cell_candidate_chain",
            "receiver_contract_satisfied": True,
            "inflate_parity_satisfied": True,
            "dispatch_blockers": [
                "inverse_scorer_cell_candidate_chain_is_not_dispatch_authorization",
                *(
                    ["exact_auth_eval_required_before_score_claim"]
                    if exact_auth_boundary
                    else []
                ),
            ],
            "chain_steps": [
                {
                    "step_id": "build_inflate_parity_probe",
                    "status": "succeeded",
                    "schema": "inverse_scorer_cell_inflate_parity_probe_v1",
                    "artifact": {
                        "path": proof_path.relative_to(queue.parent).as_posix(),
                        "bytes": len(proof_payload),
                        "sha256": hashlib.sha256(proof_payload).hexdigest(),
                    },
                    "full_frame_inflate_output_parity_claim": strict_full_frame_parity,
                    "blockers": []
                    if strict_full_frame_parity
                    else ["candidate_inflate_output_parity_missing"],
                }
            ],
            "score_claim_blockers": [
                "exact_auth_eval_required_before_score_claim"
            ]
            if exact_auth_boundary
            else [],
            "next_required_gates": ["contest_auth_eval"] if exact_auth_boundary else [],
        }
    )
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_source_archive_delta(
    queue: Path,
    *,
    archive_sha: str,
    source_archive_bytes: int,
    rate_only_control: bool = False,
) -> None:
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["source_archive_bytes"] = source_archive_bytes
    row["source_archive_sha256"] = "0" * 64
    row["score_affecting_change_proof"] = {
        "archive_changed": True,
        "byte_different": True,
        "source_archive_bytes": source_archive_bytes,
        "candidate_archive_bytes": row["candidate_archive_bytes"],
        "source_archive_sha256": "0" * 64,
        "candidate_archive_sha256": archive_sha,
    }
    if rate_only_control:
        row["rate_only_control"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _add_serialized_archive_delta(
    queue: Path,
    *,
    source_archive_bytes: int | None,
    candidate_archive_bytes: int | None,
    modeled_saved_bytes: int | None = None,
    require_realized_saving: bool = True,
    rate_only_control: bool = False,
) -> dict[str, object]:
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    contract = build_serialized_archive_delta_contract(
        source_archive_bytes=source_archive_bytes,
        candidate_archive_bytes=candidate_archive_bytes,
        modeled_saved_bytes=modeled_saved_bytes,
        require_realized_saving=require_realized_saving,
        rate_only_control=rate_only_control,
    )
    row["serialized_archive_delta"] = contract
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _hdm8_selector_gate(
    *,
    archive_sha: str,
    archive_bytes: int,
    passed: bool,
) -> dict[str, object]:
    return {
        "schema": HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA,
        "required": True,
        "passed": passed,
        "status": "passed_cuda_prefix_component_check" if passed else "blocked_mps_or_local_proxy_only",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": passed,
        "rank_or_kill_eligible": False,
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_bytes,
        "evidence_axis": "modal-t4-cuda-proxy-prefix" if passed else "local-mps-proxy-prefix",
        "component_deltas": {
            "pose_delta": -0.0001 if passed else 0.001,
            "seg_delta": 0.0,
            "score_delta": -0.0001 if passed else 0.01,
        },
        "blockers": [] if passed else ["mps_or_local_proxy_axis_requires_cuda_component_probe"],
    }


def _selector_cuda_transfer_calibration() -> dict[str, object]:
    return {
        "schema": SELECTOR_CUDA_TRANSFER_CALIBRATION_SCHEMA,
        "score_claim": False,
        "dispatch_attempted": False,
        "decision": {
            "calibration_status": "calibrated",
            "ready_for_broad_waterfill_dispatch": True,
            "ready_for_exact_eval_dispatch": False,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "blockers": [],
        },
    }


def _mark_submission_as_hdm8_selector(
    submission: Path,
    *,
    archive_sha: str,
    archive_bytes: int,
    gate_passed: bool,
) -> None:
    manifest_path = submission / "archive_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "schema": "hdm8_film_grain_sidecar_archive_manifest_v1",
            "selector_packed_in_archive": True,
            "cuda_component_risk_gate_required": True,
            "cuda_component_risk_gate": _hdm8_selector_gate(
                archive_sha=archive_sha,
                archive_bytes=archive_bytes,
                passed=gate_passed,
            ),
            "selector_cuda_transfer_calibration": _selector_cuda_transfer_calibration(),
        }
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_pr101_runtime_proof(
    submission: Path,
    archive_sha: str,
    *,
    proven: bool = True,
    stale_inflate_sh_sha: bool = False,
) -> Path:
    inflate_sh_sha = hashlib.sha256((submission / "inflate.sh").read_bytes()).hexdigest()
    if stale_inflate_sh_sha:
        inflate_sh_sha = "0" * 64 if inflate_sh_sha != "0" * 64 else "1" * 64
    inflate_py_sha = hashlib.sha256((submission / "inflate.py").read_bytes()).hexdigest()
    manifest = _write_json(
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
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "packet_dir": str(submission),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "inflate_sh_routes_to_packet_inflate_py": True,
            "runtime_consumption_proven_for_supported_bias_params": proven,
            "archive_unchanged_proof": {"archive_sha256": archive_sha},
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
        },
    )


def _write_family_agnostic_runtime_proof(
    submission: Path,
    archive_sha: str | None,
    *,
    proven: bool = True,
    extra_fields: Mapping[str, object] | None = None,
) -> Path:
    payload: dict[str, object] = {
        "schema": "family_agnostic_runtime_consumption_proof_v1",
        "proof_kind": "fixture_family_agnostic_receiver_proof",
        "target_kind": "packet_member_recompress_v1",
        "materializer_id": "packet_member_recompress_adapter",
        "receiver_contract_kind": "family_agnostic_packet_member_recompress",
        "receiver_contract_satisfied": proven,
        "runtime_consumption_proof_passed": proven,
        "passed": proven,
        "runtime_consumption_probe": {
            "passed": proven,
            "receiver_contract_probe_passed": proven,
        },
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
    }
    if archive_sha is not None:
        payload["candidate_archive_sha256"] = archive_sha
    if extra_fields:
        payload.update(extra_fields)
    return _write_json(
        submission / "runtime_consumption_proof.json",
        payload,
    )


def _add_required_runtime_proof_fields(
    queue: Path,
    submission: Path,
    repo: Path,
    *,
    status: str,
) -> None:
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["runtime_consumption_proof_required"] = True
    row["runtime_consumption_proof_status"] = status
    row["runtime_consumption_proof_path"] = (
        submission / "runtime_consumption_proof.json"
    ).relative_to(repo).as_posix()
    row["target_kind"] = "packet_member_recompress_v1"
    row["materializer_id"] = "packet_member_recompress_adapter"
    row["receiver_contract_kind"] = "family_agnostic_packet_member_recompress"
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_dfl1_shell_parity_proof(
    path: Path,
    *,
    source_archive_sha: str,
    candidate_archive_sha: str,
) -> Path:
    same_tree = "a" * 64
    same_output = "b" * 64
    payload = {
        "schema": "shell_inflate_parity_proof_v2",
        "generated_at_utc": "2026-05-25T00:00:00Z",
        "file_list_entries": ["0.mkv"],
        "file_list_entry_count": 1,
        "file_list_sha256": "c" * 64,
        "full_frame_file_list_source": "fixture_full_file_list",
        "expected_full_frame_file_list_sha256": "c" * 64,
        "expected_full_frame_entry_count": 1,
        "full_frame_file_list_sha256_match": True,
        "full_frame_entry_count_match": True,
        "parity_scope_kind": "declared_file_list",
        "contest_full_sample_claim": False,
        "contest_full_sample_parity_claim": False,
        "output_count": 1,
        "file_list_entry": "0.mkv",
        "output_basename": "0.raw",
        "python_bin": sys.executable,
        "left": {
            "label": "source",
            "archive": "source.zip",
            "archive_bytes": 100,
            "archive_sha256": source_archive_sha,
            "submission_dir": "runtime",
            "submission_tree_file_count": 1,
            "submission_tree_sha256": same_tree,
            "inflate_sh": "runtime/inflate.sh",
            "inflate_sh_sha256": "d" * 64,
            "output_count": 1,
            "output_manifest_sha256": same_output,
            "output_raw_bytes": 10,
            "output_raw_sha256": same_output,
            "outputs": [
                {
                    "file_list_entry": "0.mkv",
                    "output_basename": "0.raw",
                    "output_raw_bytes": 10,
                    "output_raw_sha256": same_output,
                }
            ],
            "inflate_seconds": 0.1,
        },
        "right": {
            "label": "candidate",
            "archive": "candidate.zip",
            "archive_bytes": 90,
            "archive_sha256": candidate_archive_sha,
            "submission_dir": "runtime",
            "submission_tree_file_count": 1,
            "submission_tree_sha256": same_tree,
            "inflate_sh": "runtime/inflate.sh",
            "inflate_sh_sha256": "d" * 64,
            "output_count": 1,
            "output_manifest_sha256": same_output,
            "output_raw_bytes": 10,
            "output_raw_sha256": same_output,
            "outputs": [
                {
                    "file_list_entry": "0.mkv",
                    "output_basename": "0.raw",
                    "output_raw_bytes": 10,
                    "output_raw_sha256": same_output,
                }
            ],
            "inflate_seconds": 0.1,
        },
        "output_bytes_match": True,
        "output_sha256_match": True,
        "output_manifest_sha256_match": True,
        "cmp_equal": True,
        "full_frame_file_list_claim": True,
        "full_frame_inflate_output_parity_claim": True,
        "blockers": [],
        "scratch_retained": False,
        "scratch_dir": "scratch",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
    }
    return _write_json(path, payload)


def _mark_queue_row_as_renderer_payload_dfl1(
    queue: Path,
    *,
    submission: Path,
    repo: Path,
    archive_sha: str,
    source_archive_sha: str = "e" * 64,
    proof_path: Path | None,
    self_assert_parity: bool = True,
) -> None:
    _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        proven=False,
        extra_fields={
            "target_kind": "renderer_payload_dfl1_v1",
            "materializer_id": "renderer_payload_dfl1_adapter",
            "receiver_contract_kind": "source_runtime_native_renderer_payload_dfl1",
        },
    )
    _add_required_runtime_proof_fields(queue, submission, repo, status="present")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["schema"] = "renderer_payload_dfl1_candidate.v1"
    row["candidate_family"] = "renderer_payload_dfl1"
    row["target_kind"] = "renderer_payload_dfl1_v1"
    row["materializer_id"] = "renderer_payload_dfl1_adapter"
    row["receiver_contract_kind"] = "source_runtime_native_renderer_payload_dfl1"
    row["source_archive_sha256"] = source_archive_sha
    row["full_frame_inflate_parity_proven"] = self_assert_parity
    row["renderer_payload_dfl1_full_frame_inflate_parity_satisfied"] = (
        self_assert_parity
    )
    if proof_path is not None:
        row["renderer_payload_dfl1_full_frame_inflate_parity_proof_path"] = (
            proof_path.relative_to(repo).as_posix()
        )
        row["renderer_payload_dfl1_full_frame_inflate_parity_proof_sha256"] = (
            hashlib.sha256(proof_path.read_bytes()).hexdigest()
        )
    row["dispatch_blockers"].extend(
        [
            "family_agnostic_receiver_contract_not_satisfied",
            "renderer_payload_dfl1_full_frame_inflate_parity_missing",
            "runtime_consumption_proof_not_passed",
            "renderer_payload_dfl1_receiver_contract_not_satisfied",
        ]
    )
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_promotes_byte_closed_candidate_without_score_claim(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    promoted = result["promoted_queue"]
    assert promoted["dispatch_ready_count"] == 1
    row = promoted["dispatch_ready"][0]
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["dispatch_packet_ready"] is True
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["target_modes"] == ["contest_exact_eval"]
    assert row["score_axis"] == "contest_cuda"
    assert row["target_score_axis"] == "contest_cuda"
    assert row["archive_sha256"] == archive_sha
    assert row["archive_bytes"] == archive_bytes
    assert "predicted_contest_cpu_gha" not in row
    assert row["dispatch_blockers"] == []
    assert row["runtime_tree_sha256"]
    assert row["runtime_content_tree_sha256"]
    assert row["score_affecting_payload_changed"] is True
    assert row["charged_bits_changed"] is True
    assert row["cpu_or_proxy_score_not_cuda_evidence"] is True
    assert row["cuda_gap_review_required_before_promotion"] is True
    assert promoted["evidence_boundary"]["cpu_or_proxy_score_not_cuda_evidence"] is True


def test_fp11_generic_full_frame_parity_clears_runtime_probe_gap(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    source_archive = tmp_path / "source_archive.zip"
    source_bytes, source_sha = _write_archive(source_archive, payload=b"source")
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    runtime_sha = "c" * 64
    proof = _write_json(
        submission / "runtime_consumption_proof.json",
        {
            "schema": FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA,
            "target_kind": FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
            "materializer_id": FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID,
            "receiver_contract_kind": FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND,
            "candidate_archive": {
                "path": "archive.zip",
                "bytes": archive_bytes,
                "sha256": archive_sha,
            },
            "candidate_archive_sha256": archive_sha,
            "candidate_member": {"name": "0.bin", "sha256": "d" * 64},
            "runtime_consumption_proof_passed": True,
            "passed": True,
            "receiver_contract_satisfied": True,
            "runtime_adapter_ready": True,
            "candidate_runtime_tree_sha256": runtime_sha,
            "expected_runtime_tree_sha256": runtime_sha,
            "decoder_raw_roundtrip_equal": True,
            "source_tail_unchanged": True,
            "selector_payload_unchanged": True,
            "dqs1_tail_unchanged": True,
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    parity = _write_json(
        submission / "full_frame_parity.json",
        {
            "schema": "shell_inflate_parity_proof_v2",
            "full_frame_file_list_claim": True,
            "full_frame_inflate_output_parity_claim": True,
            "parity_scope_kind": "contest_full_sample",
            "contest_full_sample_claim": True,
            "contest_full_sample_parity_claim": True,
            "full_frame_file_list_source": "fixture",
            "expected_full_frame_file_list_sha256": "a" * 64,
            "file_list_sha256": "a" * 64,
            "full_frame_file_list_sha256_match": True,
            "expected_full_frame_entry_count": 1,
            "file_list_entry_count": 1,
            "full_frame_entry_count_match": True,
            "output_count": 1,
            "output_bytes_match": True,
            "output_sha256_match": True,
            "output_manifest_sha256_match": True,
            "cmp_equal": True,
            "blockers": [],
            "left": {
                "label": "left",
                "archive_sha256": source_sha,
                "submission_tree_sha256": "1" * 64,
                "output_manifest_sha256": "b" * 64,
            },
            "right": {
                "label": "right",
                "archive_sha256": archive_sha,
                "submission_tree_sha256": "2" * 64,
                "output_manifest_sha256": "b" * 64,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.update(
        {
            "target_kind": FP11_SOURCE_BROTLI_RECODE_TARGET_KIND,
            "materializer_id": FP11_SOURCE_BROTLI_RECODE_MATERIALIZER_ID,
            "receiver_contract_kind": FP11_SOURCE_BROTLI_RECODE_RECEIVER_CONTRACT_KIND,
            "source_archive_sha256": source_sha,
            "source_archive_bytes": source_bytes,
            "candidate_member_sha256": "d" * 64,
            "runtime_adapter_ready": True,
            "candidate_runtime_tree_sha256": runtime_sha,
            "expected_runtime_tree_sha256": runtime_sha,
            "runtime_consumption_proof_path": proof.relative_to(tmp_path).as_posix(),
            "full_frame_inflate_parity_proven": True,
            "strict_full_frame_inflate_parity_satisfied": True,
            "full_frame_inflate_parity_proof_path": parity.relative_to(
                tmp_path
            ).as_posix(),
            "full_frame_inflate_parity_proof_sha256": hashlib.sha256(
                parity.read_bytes()
            ).hexdigest(),
        }
    )
    row["dispatch_blockers"].append("candidate_requires_exact_auth_eval_before_promotion")
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"]["dispatch_ready_count"] == 1
    facts = result["report"]["facts"]
    assert (
        facts["runtime_consumption_proof_schema"]
        == FP11_SOURCE_BROTLI_RECODE_PROOF_SCHEMA
    )
    promoted = result["promoted_queue"]["dispatch_ready"][0]
    assert promoted["runtime_consumption_proof_backed_by_full_frame_parity"] is True
    assert "runtime_consumption_proof_runtime_probe_missing" not in (
        result["report"]["blockers"]
    )


def test_refuses_serialized_archive_delta_missing_bytes_contract(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _add_serialized_archive_delta(
        queue,
        source_archive_bytes=None,
        candidate_archive_bytes=None,
        modeled_saved_bytes=128,
        require_realized_saving=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert MISSING_ARCHIVE_BYTES_BLOCKER in result["report"]["blockers"]
    assert MODELED_SAVINGS_WITHOUT_REALIZED_BLOCKER in result["report"]["blockers"]
    assert (
        result["report"]["facts"]["serialized_archive_delta"]["status"]
        == "missing_archive_bytes"
    )


def test_promotes_realized_serialized_archive_saving_and_preserves_contract(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.pop("score_affecting_payload_changed")
    row.pop("charged_bits_changed")
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = _add_serialized_archive_delta(
        queue,
        source_archive_bytes=archive_bytes + 17,
        candidate_archive_bytes=archive_bytes,
        modeled_saved_bytes=17,
        require_realized_saving=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    promoted_row = result["promoted_queue"]["dispatch_ready"][0]
    assert promoted_row["serialized_archive_delta"] == contract
    assert (
        result["report"]["facts"]["serialized_archive_delta"]["computed_realized_saved_bytes"]
        == 17
    )
    assert (
        "source_archive_bytes!=candidate_archive_bytes"
        in result["report"]["facts"]["score_affecting_change_proofs"]
    )


def test_refuses_required_serialized_archive_saving_when_candidate_is_larger(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _add_serialized_archive_delta(
        queue,
        source_archive_bytes=archive_bytes - 1,
        candidate_archive_bytes=archive_bytes,
        require_realized_saving=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert CANDIDATE_ARCHIVE_LARGER_BLOCKER in result["report"]["blockers"]
    assert (
        result["report"]["facts"]["serialized_archive_delta"]["expected_status"]
        == "realized_cost"
    )
    assert (
        result["report"]["facts"]["serialized_archive_delta"]["materializer_rate_outcome"]
        == "realized_cost"
    )
    assert result["report"]["facts"]["serialized_archive_delta"]["rate_positive"] is False
    assert (
        result["report"]["facts"]["serialized_archive_delta"]["signal_semantics"]
        == "successful_quality_spend_not_byte_saving_progress"
    )


def test_refuses_serialized_archive_delta_candidate_byte_mismatch(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _add_serialized_archive_delta(
        queue,
        source_archive_bytes=archive_bytes + 9,
        candidate_archive_bytes=archive_bytes + 1,
        require_realized_saving=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert (
        f"serialized_archive_delta_candidate_bytes_mismatch:"
        f"{archive_bytes + 1}!={archive_bytes}"
    ) in result["report"]["blockers"]


def test_refuses_inverse_scorer_chain_without_strict_full_frame_parity(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=False,
        exact_auth_boundary=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert (
        "inverse_scorer_cell_candidate_chain_strict_full_frame_inflate_parity_missing"
        in result["report"]["blockers"]
    )


def test_refuses_inverse_scorer_chain_without_exact_auth_score_boundary(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=False,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert (
        "inverse_scorer_cell_candidate_chain_exact_auth_eval_boundary_missing"
        in result["report"]["blockers"]
    )


def test_refuses_inverse_scorer_chain_with_self_asserted_unbacked_parity(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=True,
        proof_backed=False,
    )
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["full_frame_inflate_output_parity_claim"] = True
    payload["top_k"][0]["strict_full_frame_inflate_parity_satisfied"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert (
        "inverse_scorer_cell_candidate_chain_strict_full_frame_inflate_parity_missing"
        in result["report"]["blockers"]
    )


def test_refuses_inverse_scorer_chain_with_truthy_false_authority_fields(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=True,
    )
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["score_claim_valid"] = True
    payload["top_k"][0]["promotable"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "inverse_scorer_cell_candidate_chain_score_claim_valid_not_false" in result["report"]["blockers"]
    assert "inverse_scorer_cell_candidate_chain_promotable_not_false" in result["report"]["blockers"]


def test_promotes_inverse_scorer_chain_only_after_parity_and_auth_boundary(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["ready_for_exact_eval_dispatch"] is True
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["cuda_gap_review_required_before_promotion"] is True


def test_refuses_inverse_scorer_full_frame_parity_byte_increase_without_rate_only_control(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=True,
    )
    _add_source_archive_delta(
        queue,
        archive_sha=archive_sha,
        source_archive_bytes=archive_bytes - 1,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "inverse_scorer_full_frame_parity_byte_increase_without_rate_only_control"
        )
        for blocker in result["report"]["blockers"]
    )
    assert result["report"]["facts"]["realized_archive_byte_delta"] == 1


def test_allows_inverse_scorer_rate_only_control_with_full_frame_parity_byte_increase(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_inverse_scorer_chain(
        queue,
        strict_full_frame_parity=True,
        exact_auth_boundary=True,
    )
    _add_source_archive_delta(
        queue,
        archive_sha=archive_sha,
        source_archive_bytes=archive_bytes - 1,
        rate_only_control=True,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None
    assert result["report"]["facts"]["realized_archive_byte_delta"] == 1


def test_refuses_hdm8_selector_without_passing_cuda_component_gate(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    _mark_submission_as_hdm8_selector(
        submission,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        gate_passed=False,
    )
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "hdm8_selector_cuda_component_gate_not_passed" in result["report"]["blockers"]
    assert (
        "hdm8_selector_cuda_component_gate_has_blockers:"
        "mps_or_local_proxy_axis_requires_cuda_component_probe"
    ) in result["report"]["blockers"]
    assert result["report"]["facts"]["cuda_component_risk_gate_required"] is True
    assert (
        result["report"]["facts"]["cuda_component_risk_gate_status"]
        == "blocked_mps_or_local_proxy_only"
    )


def test_promotes_hdm8_selector_after_passing_cuda_component_gate(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    _mark_submission_as_hdm8_selector(
        submission,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        gate_passed=True,
    )
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["cuda_component_risk_gate_required"] is True
    assert row["cuda_component_risk_gate_status"] == "passed_cuda_prefix_component_check"
    assert row["cuda_component_risk_gate"]["passed"] is True


def test_exact_ready_audit_refuses_stale_hdm8_selector_gate_regression(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    _mark_submission_as_hdm8_selector(
        submission,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        gate_passed=True,
    )
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    exact_ready = tmp_path / "exact_ready_queue.json"
    exact_ready.write_text(json.dumps(result["promoted_queue"], indent=2, sort_keys=True))
    _mark_submission_as_hdm8_selector(
        submission,
        archive_sha=archive_sha,
        archive_bytes=archive_bytes,
        gate_passed=False,
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )

    audit = audit_exact_ready_queue(
        exact_ready,
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert audit["stale_ready_rows"]
    blockers = audit["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith("ready_row_hdm8_selector_cuda_component_gate_not_passed")
        for blocker in blockers
    )


def test_exact_ready_audit_refuses_stale_serialized_archive_delta_contract(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _add_serialized_archive_delta(
        queue,
        source_archive_bytes=archive_bytes + 5,
        candidate_archive_bytes=archive_bytes,
        require_realized_saving=True,
    )
    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    exact_ready = tmp_path / "exact_ready_queue.json"
    promoted = result["promoted_queue"]
    promoted["dispatch_ready"][0]["serialized_archive_delta"]["candidate_archive_bytes"] = (
        archive_bytes + 1
    )
    exact_ready.write_text(json.dumps(promoted, indent=2, sort_keys=True))
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )

    audit = audit_exact_ready_queue(
        exact_ready,
        repo_root=tmp_path,
        dispatch_claims_path=claims,
    )

    assert audit["stale_ready_rows"]
    blockers = audit["stale_ready_rows"][0]["blockers"]
    assert any(
        blocker.startswith(
            "ready_row_serialized_archive_delta_candidate_bytes_mismatch"
        )
        for blocker in blockers
    )


def test_refuses_pr101_runtime_packet_without_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    (submission / "runtime_consumption_proof.json").unlink()
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="missing")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_missing" in result["report"]["blockers"]
    assert "runtime_consumption_proof_file_missing" in result["report"]["blockers"]


def test_refuses_changed_byte_closed_candidate_without_default_runtime_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.pop("runtime_consumption_proof_required")
    row.pop("runtime_consumption_proof_status")
    row.pop("runtime_consumption_proof_path")
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_missing" in result["report"]["blockers"]


def test_refuses_pr101_runtime_packet_when_runtime_consumption_not_proven(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_pr101_runtime_proof(submission, archive_sha, proven=False)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_not_proven" in result["report"]["blockers"]


def test_refuses_pr101_runtime_packet_when_proof_not_bound_to_inflate_sh(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_pr101_runtime_proof(
        submission,
        archive_sha,
        proven=True,
        stale_inflate_sh_sha=True,
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert (
        "runtime_consumption_proof_inflate_sh_sha_mismatch"
        in result["report"]["blockers"]
    )


def test_promotes_pr101_runtime_packet_with_runtime_consumption_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    proof_path = _write_pr101_runtime_proof(submission, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["runtime_consumption_proof_path"] == proof_path.relative_to(tmp_path).as_posix()
    assert row["runtime_consumption_proof_sha256"]
    assert row["runtime_consumption_proof_schema"] == "pr101_kaggle_proxy_runtime_consumption_proof_v1"


def test_promotes_family_agnostic_candidate_with_receiver_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    proof_path = _write_family_agnostic_runtime_proof(submission, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["runtime_consumption_proof_path"] == proof_path.relative_to(
        tmp_path
    ).as_posix()
    assert (
        row["runtime_consumption_proof_schema"]
        == "family_agnostic_runtime_consumption_proof_v1"
    )
    assert row["runtime_consumption_proof_archive_sha256"] == archive_sha


def test_promotes_family_agnostic_packet_member_proof_with_member_binding(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    member_sha = "2" * 64
    proof_path = _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        extra_fields={
            "proof_kind": "packet_member_recompress_payload_identity_receiver_proof.v1",
            "candidate_member_sha256": member_sha,
        },
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["candidate_member_sha256"] = member_sha
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["runtime_consumption_proof_path"] == proof_path.relative_to(
        tmp_path
    ).as_posix()
    assert row["runtime_consumption_proof_candidate_member_sha256"] == member_sha
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False


def test_family_agnostic_runtime_proof_allows_truthy_change_evidence(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        extra_fields={
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
        },
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    blockers = result["report"]["blockers"]
    assert (
        "runtime_consumption_proof_false_authority_violation:"
        "score_affecting_payload_changed"
        not in blockers
    )
    assert (
        "runtime_consumption_proof_false_authority_violation:charged_bits_changed"
        not in blockers
    )
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is True


def test_promotes_dfl1_candidate_only_with_verified_shell_parity_artifact(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    source_archive_sha = "e" * 64
    parity = _write_dfl1_shell_parity_proof(
        tmp_path / "dfl1_shell_parity.json",
        source_archive_sha=source_archive_sha,
        candidate_archive_sha=archive_sha,
    )
    _mark_queue_row_as_renderer_payload_dfl1(
        queue,
        submission=submission,
        repo=tmp_path,
        archive_sha=archive_sha,
        source_archive_sha=source_archive_sha,
        proof_path=parity,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    row = result["promoted_queue"]["dispatch_ready"][0]
    assert row["source_candidate_fields"]["candidate_family"] == "renderer_payload_dfl1"
    assert row["runtime_consumption_proof_backed_by_full_frame_parity"] is True
    assert "renderer_payload_dfl1_strict_full_frame_inflate_parity_missing" not in (
        result["report"]["blockers"]
    )
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False


def test_refuses_self_asserted_dfl1_parity_without_verified_artifact(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _mark_queue_row_as_renderer_payload_dfl1(
        queue,
        submission=submission,
        repo=tmp_path,
        archive_sha=archive_sha,
        proof_path=None,
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "renderer_payload_dfl1_strict_full_frame_inflate_parity_missing" in (
        result["report"]["blockers"]
    )


def test_family_agnostic_packet_member_proof_rejects_member_sha_mismatch(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        extra_fields={
            "proof_kind": "packet_member_recompress_payload_identity_receiver_proof.v1",
            "candidate_member_sha256": "1" * 64,
        },
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["candidate_member_sha256"] = "2" * 64
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "runtime_consumption_proof_candidate_member_sha_mismatch" in (
        result["report"]["blockers"]
    )


def test_family_agnostic_runtime_proof_rejects_wrong_target_metadata(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        extra_fields={
            "target_kind": "packet_member_recompress_v1",
            "materializer_id": "packet_member_recompress_adapter",
            "receiver_contract_kind": "family_agnostic_packet_member_recompress",
        },
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["target_kind"] = "renderer_payload_dfl1_v1"
    payload["top_k"][0]["materializer_id"] = "renderer_payload_dfl1_adapter"
    payload["top_k"][0]["receiver_contract_kind"] = (
        "source_runtime_native_renderer_payload_dfl1"
    )
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    blockers = result["report"]["blockers"]
    assert "runtime_consumption_proof_target_kind_mismatch" in blockers
    assert "runtime_consumption_proof_materializer_id_mismatch" in blockers
    assert "runtime_consumption_proof_receiver_contract_kind_mismatch" in blockers


def test_renderer_payload_dfl1_exact_readiness_revalidates_full_frame_parity(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    source_archive_sha = "0" * 64
    candidate_member_sha = "3" * 64
    _write_family_agnostic_runtime_proof(
        submission,
        archive_sha,
        proven=False,
        extra_fields={
            "proof_kind": "renderer_payload_dfl1_native_unpacker_reconstruction_smoke.v1",
            "target_kind": "renderer_payload_dfl1_v1",
            "materializer_id": "renderer_payload_dfl1_adapter",
            "receiver_contract_kind": "source_runtime_native_renderer_payload_dfl1",
            "candidate_member_sha256": candidate_member_sha,
        },
    )
    parity_payload = {
        "schema": "shell_inflate_parity_proof_v2",
        "file_list_entries": ["0.mkv"],
        "file_list_entry_count": 1,
        "file_list_sha256": "4" * 64,
        "full_frame_file_list_source": "fixture_full_file_list",
        "expected_full_frame_file_list_sha256": "4" * 64,
        "expected_full_frame_entry_count": 1,
        "full_frame_file_list_sha256_match": True,
        "full_frame_entry_count_match": True,
        "parity_scope_kind": "declared_file_list",
        "contest_full_sample_claim": False,
        "contest_full_sample_parity_claim": False,
        "output_count": 1,
        "file_list_entry": "0.mkv",
        "output_basename": "0.raw",
        "python_bin": sys.executable,
        "left": {
            "label": "source",
            "archive_sha256": source_archive_sha,
            "submission_tree_sha256": "5" * 64,
            "output_manifest_sha256": "6" * 64,
        },
        "right": {
            "label": "candidate",
            "archive_sha256": archive_sha,
            "submission_tree_sha256": "5" * 64,
            "output_manifest_sha256": "6" * 64,
        },
        "output_bytes_match": True,
        "output_sha256_match": True,
        "output_manifest_sha256_match": True,
        "cmp_equal": True,
        "full_frame_file_list_claim": True,
        "full_frame_inflate_output_parity_claim": True,
        "blockers": [],
        "scratch_retained": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
    }
    parity_path = _write_json(tmp_path / "dfl1_parity.json", parity_payload)
    parity_sha = hashlib.sha256(parity_path.read_bytes()).hexdigest()
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.update(
        {
            "schema": "renderer_payload_dfl1_candidate.v1",
            "target_kind": "renderer_payload_dfl1_v1",
            "materializer_id": "renderer_payload_dfl1_adapter",
            "receiver_contract_kind": "source_runtime_native_renderer_payload_dfl1",
            "source_archive_sha256": source_archive_sha,
            "source_archive_bytes": archive_bytes + 128,
            "candidate_member_sha256": candidate_member_sha,
            "renderer_payload_dfl1_inflate_parity_satisfied": True,
            "renderer_payload_dfl1_inflate_parity_proof_path": parity_path.relative_to(
                tmp_path
            ).as_posix(),
            "renderer_payload_dfl1_inflate_parity_proof_sha256": parity_sha,
            "full_frame_inflate_parity_proven": True,
            "dispatch_blockers": [
                "optimizer_candidate_queue_is_planning_only",
                "requires_exact_eval_readiness_gate",
                "requires_lane_dispatch_claim_before_gpu_or_remote_eval",
                "renderer_payload_dfl1_full_frame_inflate_parity_missing",
                "renderer_payload_dfl1_receiver_contract_not_satisfied",
                "runtime_consumption_proof_not_passed",
                "family_agnostic_receiver_contract_not_satisfied",
            ],
        }
    )
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    promoted_row = result["promoted_queue"]["dispatch_ready"][0]
    assert promoted_row["ready_for_exact_eval_dispatch"] is True
    assert promoted_row["score_claim"] is False
    assert promoted_row["renderer_payload_dfl1_inflate_parity_satisfied"] is True
    assert promoted_row["renderer_payload_dfl1_inflate_parity_proof_sha256"] == parity_sha
    assert promoted_row["renderer_payload_dfl1_inflate_parity_proof_path"] == (
        parity_path.relative_to(tmp_path).as_posix()
    )


def test_family_agnostic_runtime_proof_requires_candidate_row_metadata(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_family_agnostic_runtime_proof(submission, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")
    payload = json.loads(queue.read_text(encoding="utf-8"))
    for field in ("target_kind", "materializer_id", "receiver_contract_kind"):
        payload["top_k"][0].pop(field, None)
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    blockers = result["report"]["blockers"]
    assert "candidate_row_target_kind_missing_for_family_agnostic_runtime_proof" in blockers
    assert "candidate_row_materializer_id_missing_for_family_agnostic_runtime_proof" in blockers
    assert (
        "candidate_row_receiver_contract_kind_missing_for_family_agnostic_runtime_proof"
        in blockers
    )


def test_promoted_family_agnostic_row_preserves_dispatch_contract_metadata(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    _write_family_agnostic_runtime_proof(submission, archive_sha)
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    promoted_row = result["promoted_queue"]["dispatch_ready"][0]
    assert promoted_row["target_kind"] == "packet_member_recompress_v1"
    assert promoted_row["materializer_id"] == "packet_member_recompress_adapter"
    assert (
        promoted_row["receiver_contract_kind"]
        == "family_agnostic_packet_member_recompress"
    )
    assert promoted_row["runtime_consumption_proof_schema"] == (
        "family_agnostic_runtime_consumption_proof_v1"
    )
    assert promoted_row["runtime_consumption_proof_archive_sha256"] == archive_sha


@pytest.mark.parametrize(
    ("proof_kwargs", "expected_blocker"),
    [
        ({"proven": False}, "runtime_consumption_proof_not_proven"),
        ({"archive_sha": None}, "runtime_consumption_proof_archive_sha_missing"),
        ({"archive_sha": "1" * 64}, "runtime_consumption_proof_archive_sha_mismatch"),
        (
            {"extra_fields": {"score_claim": True}},
            "runtime_consumption_proof_false_authority_violation:score_claim",
        ),
        (
            {"extra_fields": {"ready_for_exact_eval_dispatch": True}},
            "runtime_consumption_proof_false_authority_violation:ready_for_exact_eval_dispatch",
        ),
        (
            {"extra_fields": {"dispatch_attempted": True}},
            "runtime_consumption_proof_false_authority_violation:dispatch_attempted",
        ),
        (
            {"extra_fields": {"promotable": True}},
            "runtime_consumption_proof_false_authority_violation:promotable",
        ),
        (
            {"extra_fields": {"gpu_launched": True}},
            "runtime_consumption_proof_false_authority_violation:gpu_launched",
        ),
    ],
)
def test_family_agnostic_runtime_proof_fails_closed_on_invalid_evidence(
    tmp_path: Path,
    proof_kwargs: dict[str, object],
    expected_blocker: str,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    proof_archive_sha = proof_kwargs.pop("archive_sha", archive_sha)
    _write_family_agnostic_runtime_proof(
        submission,
        proof_archive_sha,  # type: ignore[arg-type]
        **proof_kwargs,
    )
    _add_required_runtime_proof_fields(queue, submission, tmp_path, status="present")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert expected_blocker in result["report"]["blockers"]


def test_promoted_queue_passes_existing_dispatcher_readiness_loader(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    promoted_path = tmp_path / "promoted.json"
    promoted_path.write_text(
        json.dumps(result["promoted_queue"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tool = _load_parallel_dispatch_tool()
    rows = tool._load_top_k(promoted_path, 1, active_floor_archive_bytes=None)

    assert [row["candidate_id"] for row in rows] == ["fixture_candidate"]


def _valid_contest_cuda_claim(score: float = 0.1) -> dict[str, object]:
    """Small auth-eval claim whose component recomputation equals ``score``."""

    return {
        "score_recomputed_from_components": score,
        "seg_dist": 0.0,
        "pose_dist": 0.0,
        "rate_unscaled": score / 25.0,
        "score_axis": "contest_cuda",
        "lane_tag": "[contest-CUDA]",
        "evidence_grade": "contest-CUDA",
        "score_claim": True,
        "score_claim_valid": True,
        "exact_cuda_eval_complete": True,
    }


def test_parallel_dispatch_harvest_requires_valid_contest_cuda_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_parallel_dispatch_tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_build_dispatch_cmd", lambda *args, **kwargs: [sys.executable, "-c", "pass"])

    def fake_run(*args, **kwargs):
        out = tmp_path / "experiments" / "results" / "lane_run_fixture_candidate"
        _write_json(
            out / "contest_auth_eval.json",
            {
                **_valid_contest_cuda_claim(),
                "score_axis": "contest_cpu",
                "exact_cuda_eval_complete": False,
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = tool._fire_one(
        {"candidate_id": "fixture_candidate"},
        provider="lightning",
        lane_script="scripts/remote.sh",
        label_prefix="run",
        estimated_cost=0.1,
        max_dph=0.1,
        timeout_seconds=30,
    )

    assert result.returncode == 0
    assert result.score_json_path is not None
    assert result.contest_cuda_score is None
    assert result.score_axis is None
    assert tool._harvest_tag(result) == "[dispatch-completed-no-contest-cuda-score]"


def test_parallel_dispatch_harvest_accepts_current_run_contest_cuda_claim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_parallel_dispatch_tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_build_dispatch_cmd", lambda *args, **kwargs: [sys.executable, "-c", "pass"])

    def fake_run(*args, **kwargs):
        out = tmp_path / "experiments" / "results" / "lane_run_fixture_candidate"
        _write_json(out / "contest_auth_eval.json", _valid_contest_cuda_claim(0.08))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = tool._fire_one(
        {"candidate_id": "fixture_candidate"},
        provider="lightning",
        lane_script="scripts/remote.sh",
        label_prefix="run",
        estimated_cost=0.1,
        max_dph=0.1,
        timeout_seconds=30,
    )

    assert result.contest_cuda_score == 0.08
    assert result.score_axis == "contest_cuda"
    assert result.score_claim_source_key == "score_recomputed_from_components"
    assert tool._harvest_tag(result) == "[contest-CUDA]"


def test_parallel_dispatch_refuses_required_hdm8_selector_gate_not_passed() -> None:
    tool = _load_parallel_dispatch_tool()
    archive_sha = "a" * 64

    blockers = tool._candidate_blockers(
        {
            "candidate_id": "hdm8_selector",
            "lane_id": "hnerv_hdm8_film_grain_sidecar_exact_eval",
            "ready_for_exact_eval_dispatch": True,
            "evidence_semantics": "byte_closed_archive_runtime_ready_for_exact_eval",
            "target_modes": ["contest_exact_eval"],
            "score_claim": False,
            "candidate_archive_sha256": archive_sha,
            "candidate_archive_bytes": 187_366,
            "cuda_component_risk_gate_required": True,
            "cuda_component_risk_gate": _hdm8_selector_gate(
                archive_sha=archive_sha,
                archive_bytes=187_366,
                passed=False,
            ),
        },
        active_floor_archive_bytes=None,
    )

    assert any(
        blocker.startswith(
            "hdm8_selector_cuda_component_gate:"
            "hdm8_selector_cuda_component_gate_not_passed"
        )
        for blocker in blockers
    )


def test_parallel_dispatch_harvest_ignores_stale_label_bound_auth_eval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = _load_parallel_dispatch_tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "_build_dispatch_cmd", lambda *args, **kwargs: [sys.executable, "-c", "pass"])
    stale = (
        tmp_path
        / "experiments"
        / "results"
        / "old_run_fixture_candidate"
        / "contest_auth_eval.json"
    )
    _write_json(stale, _valid_contest_cuda_claim(0.07))
    os.utime(stale, (1, 1))
    monkeypatch.setattr(
        tool.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = tool._fire_one(
        {"candidate_id": "fixture_candidate"},
        provider="lightning",
        lane_script="scripts/remote.sh",
        label_prefix="run",
        estimated_cost=0.1,
        max_dph=0.1,
        timeout_seconds=30,
    )

    assert result.score_json_path is None
    assert result.contest_cuda_score is None


def test_refuses_kaggle_proxy_row_without_archive(tmp_path: Path) -> None:
    queue = _write_json(
        tmp_path / "queue.json",
        {
            "schema": "optimizer_candidate_queue_v1",
            "top_k": [
                {
                    "candidate_id": "proxy_only",
                    "lane_id": "kaggle_proxy_sweep",
                    "proxy_only": True,
                    "score_claim": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_blockers": [
                        "optimizer_candidate_queue_is_planning_only",
                        "kaggle_proxy_output_requires_archive_builder_promotion",
                        "no_archive_zip_emitted",
                    ],
                }
            ],
            "dispatch_ready": [],
        },
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "proxy_only",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "source_row_proxy_only" in result["report"]["blockers"]
    assert "archive_path_missing" in result["report"]["blockers"]
    assert "score_affecting_change_proof_missing" in result["report"]["blockers"]


def test_exact_readiness_cli_import_does_not_require_repo_root_on_sys_path(
    tmp_path: Path, monkeypatch
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    original_path = list(sys.path)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in original_path if entry not in {"", str(REPO_ROOT)}],
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True


def test_refuses_archive_sha_mismatch(tmp_path: Path) -> None:
    submission, archive_bytes, _archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, "a" * 64)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "archive_sha256_mismatch" in result["report"]["blockers"]


def test_refuses_non_executable_inflate(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    inflate = submission / "inflate.sh"
    inflate.chmod(inflate.stat().st_mode & ~(
        stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    ))
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "inflate_sh_not_executable" in result["report"]["blockers"]


def test_refuses_above_active_floor_without_override(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=1,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("above_active_floor_archive_bytes_without_operator_override")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_cosmetic_candidate_without_change_proof(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0].pop("score_affecting_payload_changed")
    payload["top_k"][0].pop("charged_bits_changed")
    queue.write_text(json.dumps(payload), encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "score_affecting_change_proof_missing" in result["report"]["blockers"]


def test_refuses_unrelated_nested_provenance_diff_as_change_proof(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.pop("score_affecting_payload_changed")
    row.pop("charged_bits_changed")
    row["historical_provenance"] = {
        "old_archive_sha256": "0" * 64,
        "new_archive_sha256": "1" * 64,
        "score_affecting_payload_changed": True,
    }
    queue.write_text(json.dumps(payload), encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert "score_affecting_change_proof_missing" in result["report"]["blockers"]


def test_archive_manifest_rejects_generic_nested_hash_as_archive_authority(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    (submission / "archive_manifest.json").write_text(
        json.dumps(
            {
                "score_claim": False,
                "diagnostic_artifact": {
                    "sha256": archive_sha,
                    "bytes": archive_bytes,
                },
            }
        ),
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["promoted_queue"] is None
    assert f"archive_manifest_sha_mismatch:None!={archive_sha}" in result["report"]["blockers"]
    assert f"archive_manifest_size_mismatch:None!={archive_bytes}" in result["report"]["blockers"]


def test_accepts_named_score_affecting_change_proof_object(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row.pop("score_affecting_payload_changed")
    row.pop("charged_bits_changed")
    row["score_affecting_change_proof"] = {
        "source_archive_sha256": "0" * 64,
        "candidate_archive_sha256": archive_sha,
    }
    queue.write_text(json.dumps(payload), encoding="utf-8")

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None
    assert (
        "source_archive_sha256!=candidate_archive_sha256"
        in result["report"]["facts"]["score_affecting_change_proofs"]
    )


def test_refuses_same_lane_active_claim(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    timestamp = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| {timestamp} | test | fixture_lane | modal | job1 | 2026-05-10T01:00:00Z | active_dispatching | cost=$0.50 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_active_dispatch_claim")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_negative_for_same_archive(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_terminal_negative_for_same_archive")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_cuda_score_not_below_floor_for_same_archive(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=0.22650343150032118 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2089810755823297,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_not_below_active_floor_for_same_archive"
        )
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_cuda_without_score_for_same_archive(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score pending harvest |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2089810755823297,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_terminal_cuda_score_missing_for_same_archive")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_same_lane_terminal_cuda_score_already_below_floor_for_same_archive(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval | archive_sha={archive_sha}; score_recomputed=1.95e-1 |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        active_floor_score=0.2089810755823297,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "same_lane_terminal_score_already_below_active_floor_for_same_archive"
        )
        for blocker in result["report"]["blockers"]
    )


def test_allows_runtime_changed_candidate_after_different_runtime_terminal(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    initial = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    runtime_sha = initial["promoted_queue"]["dispatch_ready"][0]["runtime_tree_sha256"]
    old_runtime_sha = "0" * 64 if runtime_sha != "0" * 64 else "1" * 64
    payload = json.loads(queue.read_text(encoding="utf-8"))
    row = payload["top_k"][0]
    row["score_affecting_payload_changed"] = False
    row["charged_bits_changed"] = False
    row["score_affecting_runtime_changed"] = True
    queue.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:01:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={old_runtime_sha}; score_claim=false_until_modal_validation |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None


def test_refuses_runtime_mismatch_without_score_affecting_runtime_change(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    initial = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    runtime_sha = initial["promoted_queue"]["dispatch_ready"][0]["runtime_tree_sha256"]
    old_runtime_sha = "0" * 64 if runtime_sha != "0" * 64 else "1" * 64
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:01:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={old_runtime_sha}; score_claim=false_until_modal_validation |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith("same_lane_terminal_runtime_mismatch_for_same_archive")
        for blocker in result["report"]["blockers"]
    )


def test_refuses_runtime_content_mismatch_without_score_affecting_runtime_change(
    tmp_path: Path,
) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    initial = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
    )
    row = initial["promoted_queue"]["dispatch_ready"][0]
    runtime_sha = row["runtime_tree_sha256"]
    runtime_content_sha = row["runtime_content_tree_sha256"]
    old_runtime_content_sha = (
        "0" * 64 if runtime_content_sha != "0" * 64 else "1" * 64
    )
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:01:00Z | test | fixture_lane | modal | job1 |  | completed_contest_cuda_auth_eval_negative | archive_sha={archive_sha}; score_recomputed=41.35 |\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | active_dispatching | archive_sha={archive_sha}; runtime_tree_sha={runtime_sha}; runtime_content_tree_sha256={old_runtime_content_sha}; score_claim=false_until_modal_validation |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["promoted_queue"] is None
    assert any(
        blocker.startswith(
            "same_lane_terminal_runtime_content_mismatch_for_same_archive"
        )
        for blocker in result["report"]["blockers"]
    )


def test_allows_same_lane_terminal_infra_failure_for_same_archive(tmp_path: Path) -> None:
    submission, archive_bytes, archive_sha = _make_submission(tmp_path)
    queue = _make_queue(tmp_path, submission, archive_bytes, archive_sha)
    claims = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    claims.parent.mkdir(parents=True)
    claims.write_text(
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| 2026-05-10T00:00:00Z | test | fixture_lane | modal | job1 |  | failed_runtime_dependency_missing_constriction | archive_sha={archive_sha}; no score result |\n",
        encoding="utf-8",
    )

    result = promote_candidate_for_exact_eval(
        queue,
        "fixture_candidate",
        repo_root=tmp_path,
        active_floor_archive_bytes=None,
        dispatch_claims_path=claims,
    )

    assert result["report"]["ready_for_exact_eval_dispatch"] is True
    assert result["promoted_queue"] is not None
