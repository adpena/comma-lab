# SPDX-License-Identifier: MIT
"""Tests for queue-owned 5D canvas coverage acquisition."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA,
    PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA,
    PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA,
    PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA,
    PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA,
    PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA,
    PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA,
    build_coverage_acquisition_plan,
    build_coverage_followup_execution_queue,
    build_coverage_followup_input_binding_report,
    build_coverage_followup_readiness_report,
    build_pair_frame_5d_coverage_acquisition_queue,
)
from tac.optimization.pair_frame_scorer_geometry_lattice_5d_canvas_coverage import (
    COVERAGE_AUDIT_SCHEMA,
    WORK_ORDER_SCHEMA,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BUILD_TOOL = _REPO_ROOT / "tools" / "build_5d_canvas_coverage_acquisition_queue.py"
_EMIT_TOOL = _REPO_ROOT / "tools" / "emit_5d_canvas_coverage_acquisition_plan.py"
_AUDIT_TOOL = _REPO_ROOT / "tools" / "audit_5d_coverage_followup_requests.py"
_BIND_TOOL = _REPO_ROOT / "tools" / "bind_5d_coverage_followup_inputs.py"
_FOLLOWUP_QUEUE_TOOL = (
    _REPO_ROOT / "tools" / "build_5d_coverage_followup_execution_queue.py"
)


def _work_order(
    order_id: str,
    *,
    priority: int = 1,
    archive_sha256: str = "e" * 64,
) -> dict[str, object]:
    return {
        "schema": WORK_ORDER_SCHEMA,
        "id": order_id,
        "priority": priority,
        "reason": f"unit {order_id}",
        "consumer": "unit_consumer",
        "target": {"archive_sha256": archive_sha256},
        "suggested_next_tools": [],
        "allowed_use": "experiment_queue_v1_planning_input_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        "reproduction_claim": False,
        "reproduction_equivalence": False,
    }


def _audit(
    *work_orders: dict[str, object],
    archive_sha256: str = "e" * 64,
) -> dict[str, object]:
    return {
        "schema": COVERAGE_AUDIT_SCHEMA,
        "archive_sha256": archive_sha256,
        "verdict": "densification_required",
        "work_order_count": len(work_orders),
        "work_orders": list(work_orders),
        "blockers": ["unit_blocker"],
        "allowed_use": "local_planning_and_experiment_queue_acquisition_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_authority",
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        "reproduction_claim": False,
        "reproduction_equivalence": False,
    }


def _write_json(path: pathlib.Path, payload: object) -> pathlib.Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "score_claim_eligible": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_packet_ready": False,
        "reproduction_claim": False,
        "reproduction_equivalence": False,
    }


def _write_mlx_cache(
    cache_dir: pathlib.Path,
    *,
    archive_sha256: str = "e" * 64,
    local_advisory_identity: bool = False,
) -> pathlib.Path:
    cache_dir.mkdir()
    arrays = {
        "pair_indices": np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        "posenet_yuv6_pair": np.zeros((2, 12, 4, 4), dtype=np.float32),
        "segnet_last_rgb": np.zeros((2, 3, 4, 4), dtype=np.float32),
    }
    artifact_payload: dict[str, dict[str, object]] = {}
    array_sha256: dict[str, str] = {}
    filenames = {
        "pair_indices": "pair_indices.npy",
        "posenet_yuv6_pair": "posenet_yuv6_pair.npy",
        "segnet_last_rgb": "segnet_last_rgb.npy",
    }
    for key, array in arrays.items():
        path = cache_dir / filenames[key]
        np.save(path, array)
        digest = _sha256_file(path)
        array_sha256[key] = _array_sha256(array)
        artifact_payload[key] = {
            "path": filenames[key],
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    manifest = {
        "schema_version": "mlx_scorer_input_cache.v1",
        "archive_sha256": archive_sha256,
        "raw_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": "b" * 64,
        "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
        "source_kind": "raw",
        "pair_count": 2,
        "pair_indices_shape": [2, 2],
        "posenet_yuv6_pair_shape": [2, 12, 4, 4],
        "segnet_last_rgb_shape": [2, 3, 4, 4],
        "array_sha256": array_sha256,
        "artifacts": artifact_payload,
        "evidence_grade": "macOS-MLX-research-signal",
        "evidence_tag": "[macOS-MLX research-signal]",
        **_false_authority(),
    }
    if local_advisory_identity:
        audit = {
            "schema_version": "mlx_scorer_input_cache_local_cpu_advisory_audit.v1",
            "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            "passed": True,
            "identity_residual": 0,
            "cache": {
                "archive_sha256": manifest["archive_sha256"],
                "raw_sha256": manifest["raw_sha256"],
                "inflated_outputs_aggregate_sha256": (
                    manifest["inflated_outputs_aggregate_sha256"]
                ),
                "hash_domain": manifest["hash_domain"],
                "pair_count": manifest["pair_count"],
                "segnet_last_rgb_shape": manifest["segnet_last_rgb_shape"],
                "posenet_yuv6_pair_shape": manifest["posenet_yuv6_pair_shape"],
                "pair_indices_shape": manifest["pair_indices_shape"],
                "array_sha256": manifest["array_sha256"],
            },
            **_false_authority(),
        }
        audit_path = _write_json(cache_dir / "local_cpu_advisory_cache_audit.json", audit)
        manifest["eligible_for_local_mlx_local_advisory_debug"] = True
        manifest["local_cpu_advisory_cache_identity_audit"] = {
            "schema_version": audit["schema_version"],
            "path": audit_path.name,
            "sha256": _sha256_file(audit_path),
            "verdict": "PASS_CACHE_LOCAL_CPU_ADVISORY_IDENTITY",
            "passed": True,
            "identity_residual": 0,
            **_false_authority(),
        }
    _write_json(cache_dir / "manifest.json", manifest)
    return cache_dir


def _submission_bundle_payload(
    *,
    archive_sha256: str = "e" * 64,
    archive_bytes: int = 123,
    submission_dir: str | pathlib.Path = "experiments/results/unit_submission",
) -> dict[str, object]:
    submission_dir_ref = str(submission_dir)
    return {
        "schema_version": "submission_bundle_v1_20260526",
        "lane_id": "lane_unit",
        "substrate_id": "substrate_unit",
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "submission_dir": submission_dir_ref,
        "inflate_sh_path": str(pathlib.Path(submission_dir_ref) / "inflate.sh"),
        "inflate_py_path": str(pathlib.Path(submission_dir_ref) / "inflate.py"),
        "inflate_py_loc": 10,
        "inflate_py_loc_budget": 200,
        "inflate_py_loc_waiver_rationale": None,
        "readme_md_path": str(pathlib.Path(submission_dir_ref) / "README.md"),
        "report_txt_path": str(pathlib.Path(submission_dir_ref) / "report.txt"),
        "archive_manifest_path": str(
            pathlib.Path(submission_dir_ref) / "archive_manifest.json"
        ),
        "dependency_closure_manifest": {
            "declared_dependencies": ["numpy"],
            "dependency_budget": 2,
            "within_budget": True,
            "numpy_portable": True,
            "waiver_rationale": None,
        },
        "select_inflate_device_routing": "inline_with_waiver",
        "pythonpath_self_containment_status": "clean",
        "vendor_pythonpath_self_containment": False,
        "runtime_dep_closure": ["numpy"],
        "measurement_utc": "2026-05-27T00:00:00+00:00",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotable": False,
        "evidence_grade": "[predicted; submission-bundle-canonical]",
        "canonical_helper_invocation": "tac.submission_packet.build_submission_bundle",
        "canonical_equation_id": "submission_bundle_canonical_helper_consolidation_savings_v1",
        "canonical_equation_status": "FORMALIZATION_PENDING",
        "elapsed_seconds": 0.1,
        "canonical_provenance": {},
        "written_at_utc": "2026-05-27T00:00:00+00:00",
        "written_pid": 1,
        "written_host": "unit",
    }


def _write_submission_bundle_result(
    path: pathlib.Path,
    *,
    archive_bytes: bytes = b"unit submission archive\n",
) -> pathlib.Path:
    submission_dir = path.parent / f"{path.stem}_submission"
    submission_dir.mkdir(parents=True)
    archive_path = submission_dir / "archive.zip"
    archive_path.write_bytes(archive_bytes)
    (submission_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n",
        encoding="utf-8",
    )
    (submission_dir / "inflate.py").write_text(
        "import numpy as np\n",
        encoding="utf-8",
    )
    (submission_dir / "README.md").write_text("unit README\n", encoding="utf-8")
    (submission_dir / "report.txt").write_text("unit report\n", encoding="utf-8")
    _write_json(
        submission_dir / "archive_manifest.json",
        {"archive_sha256": _sha256_bytes(archive_bytes)},
    )
    return _write_json(
        path,
        _submission_bundle_payload(
            archive_sha256=_sha256_bytes(archive_bytes),
            archive_bytes=len(archive_bytes),
            submission_dir=submission_dir,
        ),
    )


def test_build_coverage_acquisition_plan_for_fail_closed_anchor_order() -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))

    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )

    assert plan["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA
    assert plan["executable_now"] is False
    assert plan["score_claim"] is False
    assert "requires_byte_closed_submission_bundle_for_paired_auth_eval" in (
        plan["blocking_conditions"]
    )
    request = plan["followup_lane_requests"][0]
    assert request["schema"] == PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA
    assert request["canonical_paired_dispatch_tool"] == (
        "tools/dispatch_modal_paired_auth_eval.py"
    )
    assert request["paired_dispatch_command_contract_blockers"] == []
    assert "--submission-bundle" not in " ".join(
        request["paired_dispatch_command_template"]
    )
    assert request["required_input_schema_version"] == "submission_bundle_v1_20260526"


def test_build_coverage_acquisition_plan_for_mlx_negative_delta_request() -> None:
    audit = _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout"))

    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
        output_root="experiments/results/unit_coverage_acquisition",
    )

    assert plan["executable_now"] is False
    assert plan["preferred_resource_kind"] == "local_mlx"
    request = plan["followup_lane_requests"][0]
    assert request["schema"] == PAIR_FRAME_5D_MLX_NEGATIVE_DELTA_REQUEST_SCHEMA
    assert request["canonical_execution_tool"] == "tools/run_mlx_scorer_response_cache.py"
    assert request["preferred_execution_queue_schema"] == (
        "mlx_scorer_response_execution_queue_plan.v1"
    )
    assert "--allow-gpu-research-signal" in request["command_template"]
    assert "--archive-size-bytes" in request["command_template"]
    assert request["score_claim"] is False


def test_build_followup_readiness_report_blocks_missing_inputs(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(
        _work_order("populate_missing_paired_cpu_cuda_axis_anchors", priority=1),
        _work_order("acquire_negative_delta_cells_before_operator_fanout", priority=2),
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    exact_path = _write_json(tmp_path / "exact_plan.json", exact_plan)
    mlx_path = _write_json(tmp_path / "mlx_plan.json", mlx_plan)

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[exact_path, mlx_path],
    )

    assert report["schema"] == PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA
    assert report["request_count"] == 2
    assert report["ready_request_count"] == 0
    assert report["blocked_request_count"] == 2
    blockers = {
        row["work_order_id"]: row["blockers"]
        for row in report["requests"]
    }
    assert blockers[
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ] == ["submission_bundle_result_path_missing"]
    assert blockers[
        "acquire_negative_delta_cells_before_operator_fanout"
    ] == [
        "reference_mlx_cache_dir_missing",
        "candidate_mlx_cache_dir_missing",
        "archive_size_bytes_missing_or_invalid",
    ]
    assert report["score_claim"] is False


def test_build_followup_readiness_report_materializes_ready_commands(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    exact_path = _write_json(tmp_path / "exact_plan.json", exact_plan)
    mlx_path = _write_json(tmp_path / "mlx_plan.json", mlx_plan)
    submission_bundle = _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    reference_cache = tmp_path / "reference_mlx_cache"
    candidate_cache = tmp_path / "candidate_mlx_cache"
    _write_mlx_cache(reference_cache, archive_sha256=archive_sha)
    _write_mlx_cache(candidate_cache, archive_sha256=archive_sha, local_advisory_identity=True)

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[exact_path, mlx_path],
        submission_bundle_path=submission_bundle,
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
        archive_size_bytes=len(archive_payload),
    )

    assert report["ready_request_count"] == 2
    assert report["blocked_request_count"] == 0
    commands = {
        row["work_order_id"]: row["materialized_command"]
        for row in report["requests"]
    }
    assert "tools/paired_auth_eval_cli.py" in commands[
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ]
    assert "tools/run_mlx_scorer_response_cache.py" in commands[
        "acquire_negative_delta_cells_before_operator_fanout"
    ]
    assert str(len(archive_payload)) in commands[
        "acquire_negative_delta_cells_before_operator_fanout"
    ]


def test_build_followup_input_binding_report_discovers_custody_checked_inputs(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    exact_path = _write_json(tmp_path / "exact_plan.json", exact_plan)
    mlx_path = _write_json(tmp_path / "mlx_plan.json", mlx_plan)
    submission_bundle = _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    reference_cache = _write_mlx_cache(
        tmp_path / "reference_mlx_cache",
        archive_sha256=archive_sha,
    )
    candidate_cache = _write_mlx_cache(
        tmp_path / "candidate_mlx_cache",
        archive_sha256=archive_sha,
        local_advisory_identity=True,
    )

    report = build_coverage_followup_input_binding_report(
        repo_root=_REPO_ROOT,
        plan_paths=[exact_path, mlx_path],
        search_roots=[tmp_path],
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
    )

    assert report["schema"] == PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA
    assert report["bound_request_count"] == 2
    assert report["blocked_request_count"] == 0
    assert report["selected_inputs"]["submission_bundle_path"] == str(
        submission_bundle.relative_to(_REPO_ROOT)
        if submission_bundle.is_relative_to(_REPO_ROOT)
        else submission_bundle
    )
    assert report["selected_inputs"]["reference_mlx_cache_dir"].endswith(
        "reference_mlx_cache"
    )
    assert report["selected_inputs"]["candidate_mlx_cache_dir"].endswith(
        "candidate_mlx_cache"
    )
    assert report["selected_inputs"]["archive_size_bytes"] == len(archive_payload)
    assert all(row["ready_for_readiness_refresh"] for row in report["requests"])


def test_build_followup_execution_queue_freezes_exact_and_queues_mlx(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    exact_path = _write_json(tmp_path / "exact_plan.json", exact_plan)
    mlx_path = _write_json(tmp_path / "mlx_plan.json", mlx_plan)
    submission_bundle = _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    reference_cache = tmp_path / "reference_mlx_cache"
    candidate_cache = tmp_path / "candidate_mlx_cache"
    _write_mlx_cache(reference_cache, archive_sha256=archive_sha)
    _write_mlx_cache(candidate_cache, archive_sha256=archive_sha, local_advisory_identity=True)
    readiness_path = _write_json(
        tmp_path / "followup_readiness_report.json",
        build_coverage_followup_readiness_report(
            repo_root=_REPO_ROOT,
            plan_paths=[exact_path, mlx_path],
            submission_bundle_path=submission_bundle,
            reference_mlx_cache_dir=reference_cache,
            candidate_mlx_cache_dir=candidate_cache,
            archive_size_bytes=len(archive_payload),
        ),
    )

    queue = build_coverage_followup_execution_queue(
        repo_root=_REPO_ROOT,
        readiness_report_path=readiness_path,
        queue_id="unit_followup_execution",
    )

    assert queue["metadata"]["schema"] == PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA
    assert queue["metadata"]["ready_request_count"] == 2
    exact, mlx = queue["experiments"]
    assert exact["status"] == "frozen"
    assert exact["metadata"]["operator_gated"] is True
    assert exact["steps"][0]["resources"]["kind"] == "local_cpu"
    assert mlx["status"] == "queued"
    assert mlx["steps"][0]["resources"]["kind"] == "local_mlx"
    assert any(
        condition.get("key") == "schema_version"
        and condition.get("equals") == "mlx_scorer_response.v1"
        for condition in mlx["steps"][0]["postconditions"]
    )


def test_build_followup_execution_queue_handles_no_ready_rows(
    tmp_path: pathlib.Path,
) -> None:
    readiness_path = _write_json(
        tmp_path / "blocked_readiness_report.json",
        {
            "schema": PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA,
            "plan_count": 1,
            "request_count": 1,
            "ready_request_count": 0,
            "blocked_request_count": 1,
            "requests": [
                {
                    "schema": "pair_frame_5d_canvas_followup_readiness_row.v1",
                    "plan_path": "unit_plan.json",
                    "work_order_id": "populate_missing_paired_cpu_cuda_axis_anchors",
                    "request_schema": PAIR_FRAME_5D_EXACT_AXIS_ANCHOR_REQUEST_SCHEMA,
                    "ready": False,
                    "blockers": ["submission_bundle_result_path_missing"],
                    "materialized_command": None,
                    "allowed_use": "queue_followup_readiness_routing_only",
                    "score_claim": False,
                    "score_claim_valid": False,
                    "score_claim_eligible": False,
                    "promotion_eligible": False,
                    "promotable": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_packet_ready": False,
                    "reproduction_claim": False,
                    "reproduction_equivalence": False,
                }
            ],
            "allowed_use": "queue_followup_readiness_routing_only",
            "forbidden_use": "score_claim_or_dispatch_authority",
            "score_claim": False,
            "score_claim_valid": False,
            "score_claim_eligible": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_packet_ready": False,
            "reproduction_claim": False,
            "reproduction_equivalence": False,
        },
    )

    queue = build_coverage_followup_execution_queue(
        repo_root=_REPO_ROOT,
        readiness_report_path=readiness_path,
        queue_id="unit_no_ready_followup_execution",
    )

    assert queue["metadata"]["selected_request_count"] == 0
    assert queue["experiments"][0]["id"] == "no_ready_followup_requests"
    assert queue["experiments"][0]["status"] == "disabled"


def test_followup_readiness_refuses_schema_only_submission_bundle(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "exact_plan.json", plan)
    schema_only_bundle = _write_json(
        tmp_path / "schema_only_bundle.json",
        {
            "schema_version": "submission_bundle_v1_20260526",
            "archive_sha256": "e" * 64,
        },
    )

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        submission_bundle_path=schema_only_bundle,
    )

    assert report["ready_request_count"] == 0
    assert report["blocked_request_count"] == 1
    assert report["requests"][0]["blockers"] == [
        "submission_bundle_result_contract_invalid:ValueError"
    ]
    assert report["requests"][0]["materialized_command"] is None


def test_followup_readiness_refuses_contract_only_missing_bundle_files(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "exact_plan.json", plan)
    contract_only_bundle = _write_json(
        tmp_path / "submission_bundle_result.json",
        _submission_bundle_payload(),
    )

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        submission_bundle_path=contract_only_bundle,
    )

    blockers = report["requests"][0]["blockers"]
    assert report["ready_request_count"] == 0
    assert "submission_bundle_submission_dir_not_found" in blockers
    assert "submission_bundle_archive_zip_missing" in blockers
    assert "submission_bundle_inflate_sh_path_not_found" in blockers
    assert report["requests"][0]["materialized_command"] is None


def test_followup_input_binding_refuses_contract_only_missing_bundle_files(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "exact_plan.json", plan)
    _write_json(
        tmp_path / "submission_bundle_result.json",
        _submission_bundle_payload(),
    )

    report = build_coverage_followup_input_binding_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        search_roots=[tmp_path],
    )

    blockers = report["requests"][0]["blockers"]
    assert report["bound_request_count"] == 0
    assert "submission_bundle_submission_dir_not_found" in blockers
    assert "submission_bundle_archive_zip_missing" in blockers
    assert "submission_bundle_inflate_sh_path_not_found" in blockers
    assert "submission_bundle_path" not in report["selected_inputs"]


def test_followup_readiness_refuses_manifest_only_mlx_cache(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "mlx_plan.json", plan)
    reference_cache = tmp_path / "reference_mlx_cache"
    candidate_cache = tmp_path / "candidate_mlx_cache"
    reference_cache.mkdir()
    candidate_cache.mkdir()
    _write_json(reference_cache / "manifest.json", {"schema": "unit"})
    _write_json(candidate_cache / "manifest.json", {"schema": "unit"})

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
        archive_size_bytes=123,
    )

    blockers = report["requests"][0]["blockers"]
    assert report["ready_request_count"] == 0
    assert "reference:mlx_cache_array_sha256_missing" in blockers
    assert "candidate:candidate_mlx_cache_identity_audit_missing_or_invalid" in blockers
    assert report["requests"][0]["materialized_command"] is None


def test_followup_readiness_refuses_spoofed_mlx_cache_authority(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "mlx_plan.json", plan)
    reference_cache = _write_mlx_cache(tmp_path / "reference_mlx_cache")
    candidate_cache = _write_mlx_cache(
        tmp_path / "candidate_mlx_cache",
        local_advisory_identity=True,
    )
    manifest_path = candidate_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unit.spoof"
    manifest["score_claim_eligible"] = True
    manifest["dispatch_packet_ready"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
        archive_size_bytes=123,
    )

    blockers = report["requests"][0]["blockers"]
    assert report["ready_request_count"] == 0
    assert (
        "candidate:mlx_cache_schema_version_not_mlx_scorer_input_cache_v1"
        in blockers
    )
    assert "candidate:mlx_cache_manifest_score_claim_eligible_truthy" in blockers
    assert "candidate:mlx_cache_manifest_dispatch_packet_ready_truthy" in blockers


def test_followup_input_binding_refuses_uncustodied_explicit_archive_size(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "mlx_plan.json", plan)
    reference_cache = _write_mlx_cache(tmp_path / "reference_mlx_cache")
    candidate_cache = _write_mlx_cache(
        tmp_path / "candidate_mlx_cache",
        local_advisory_identity=True,
    )

    report = build_coverage_followup_input_binding_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        search_roots=[tmp_path],
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
        archive_size_bytes=999,
    )

    assert report["bound_request_count"] == 0
    assert report["blocked_request_count"] == 1
    assert any(
        blocker.startswith("archive_size:submission_bundle_result_not_found")
        for blocker in report["requests"][0]["blockers"]
    )


def test_followup_input_binding_global_blockers_clear_row_readiness(
    tmp_path: pathlib.Path,
) -> None:
    first_archive = b"first selectable bundle\n"
    second_archive = b"second selectable bundle\n"
    first_sha = _sha256_bytes(first_archive)
    second_sha = _sha256_bytes(second_archive)
    first_audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            archive_sha256=first_sha,
        ),
        archive_sha256=first_sha,
    )
    second_audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            archive_sha256=second_sha,
        ),
        archive_sha256=second_sha,
    )
    first_plan = build_coverage_acquisition_plan(
        coverage_audit=first_audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit_a.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    second_plan = build_coverage_acquisition_plan(
        coverage_audit=second_audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit_b.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    first_path = _write_json(tmp_path / "first_plan.json", first_plan)
    second_path = _write_json(tmp_path / "second_plan.json", second_plan)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    _write_submission_bundle_result(
        first_root / "submission_bundle_result.json",
        archive_bytes=first_archive,
    )
    _write_submission_bundle_result(
        second_root / "submission_bundle_result.json",
        archive_bytes=second_archive,
    )

    report = build_coverage_followup_input_binding_report(
        repo_root=_REPO_ROOT,
        plan_paths=[first_path, second_path],
        search_roots=[tmp_path],
    )

    assert report["global_blockers"] == [
        "multiple_submission_bundles_selected",
        "multiple_archive_size_byte_values_selected",
    ]
    assert report["bound_request_count"] == 0
    assert all(
        not row["ready_for_readiness_refresh"] for row in report["requests"]
    )
    assert all(
        "global:multiple_submission_bundles_selected" in row["blockers"]
        for row in report["requests"]
    )


def test_build_followup_input_binding_report_selects_bundle_and_mlx_caches(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    exact_path = _write_json(plans_dir / "exact_acquisition_plan.json", exact_plan)
    mlx_path = _write_json(plans_dir / "mlx_acquisition_plan.json", mlx_plan)
    _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    _write_mlx_cache(tmp_path / "reference_mlx_cache", archive_sha256=archive_sha)
    _write_mlx_cache(
        tmp_path / "candidate_mlx_cache",
        archive_sha256=archive_sha,
        local_advisory_identity=True,
    )

    report = build_coverage_followup_input_binding_report(
        repo_root=_REPO_ROOT,
        plan_paths=[exact_path, mlx_path],
        search_roots=[tmp_path],
    )

    assert report["schema"] == PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA
    assert report["bound_request_count"] == 2
    assert report["blocked_request_count"] == 0
    selected = report["selected_inputs"]
    assert selected["archive_size_bytes"] == len(archive_payload)
    assert selected["submission_bundle_path"].endswith("submission_bundle_result.json")
    assert selected["reference_mlx_cache_dir"].endswith("reference_mlx_cache")
    assert selected["candidate_mlx_cache_dir"].endswith("candidate_mlx_cache")
    assert report["score_claim"] is False


def test_bind_followup_inputs_cli_emits_refreshed_readiness(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    _write_json(plans_dir / "exact_acquisition_plan.json", plan)
    _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    binding_path = tmp_path / "binding.json"
    readiness_path = tmp_path / "readiness.json"

    subprocess.run(
        [
            sys.executable,
            str(_BIND_TOOL),
            "--plans-dir",
            str(plans_dir),
            "--search-root",
            str(tmp_path),
            "--output",
            str(binding_path),
            "--refreshed-readiness-output",
            str(readiness_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    assert binding["schema"] == PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA
    assert binding["bound_request_count"] == 1
    assert readiness["schema"] == PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA
    assert readiness["ready_request_count"] == 1


def test_build_followup_execution_queue_from_ready_report(
    tmp_path: pathlib.Path,
) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
        output_root=tmp_path / "acquisition",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
        output_root=tmp_path / "acquisition",
    )
    exact_path = _write_json(tmp_path / "exact_plan.json", exact_plan)
    mlx_path = _write_json(tmp_path / "mlx_plan.json", mlx_plan)
    submission_bundle = _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    reference_cache = tmp_path / "reference_mlx_cache"
    candidate_cache = tmp_path / "candidate_mlx_cache"
    _write_mlx_cache(reference_cache, archive_sha256=archive_sha)
    _write_mlx_cache(candidate_cache, archive_sha256=archive_sha, local_advisory_identity=True)
    readiness = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[exact_path, mlx_path],
        submission_bundle_path=submission_bundle,
        reference_mlx_cache_dir=reference_cache,
        candidate_mlx_cache_dir=candidate_cache,
        archive_size_bytes=len(archive_payload),
    )
    readiness_path = _write_json(tmp_path / "readiness.json", readiness)

    queue = build_coverage_followup_execution_queue(
        repo_root=_REPO_ROOT,
        readiness_report_path=readiness_path,
        queue_id="unit_followup_execution",
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["metadata"]["schema"] == PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA
    assert queue["metadata"]["ready_request_count"] == 2
    assert queue["metadata"]["blocked_request_count"] == 0
    assert len(queue["experiments"]) == 2
    exact_exp, mlx_exp = queue["experiments"]
    assert exact_exp["steps"][0]["resources"]["kind"] == "local_cpu"
    assert exact_exp["status"] == "frozen"
    assert "--dry-run" in exact_exp["steps"][0]["command"]
    assert exact_exp["metadata"]["score_claim"] is False
    assert mlx_exp["steps"][0]["resources"]["kind"] == "local_mlx"
    assert [row["type"] for row in mlx_exp["steps"][0]["postconditions"]] == [
        "path_exists",
        "json_equals",
        "json_false_authority",
    ]


def test_build_followup_execution_queue_refuses_all_blocked_report(
    tmp_path: pathlib.Path,
) -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "exact_plan.json", plan)
    readiness = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
    )
    readiness_path = _write_json(tmp_path / "readiness.json", readiness)

    queue = build_coverage_followup_execution_queue(
        repo_root=_REPO_ROOT,
        readiness_report_path=readiness_path,
    )

    assert queue["metadata"]["ready_request_count"] == 0
    assert queue["metadata"]["blocked_request_count"] == 1
    assert len(queue["experiments"]) == 1
    assert queue["experiments"][0]["id"] == "no_ready_followup_requests"
    assert queue["experiments"][0]["status"] == "disabled"
    assert queue["experiments"][0]["metadata"]["score_claim"] is False


def test_build_followup_execution_queue_cli(tmp_path: pathlib.Path) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plan_path = _write_json(tmp_path / "exact_plan.json", plan)
    submission_bundle = _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    readiness = build_coverage_followup_readiness_report(
        repo_root=_REPO_ROOT,
        plan_paths=[plan_path],
        submission_bundle_path=submission_bundle,
    )
    readiness_path = _write_json(tmp_path / "readiness.json", readiness)
    queue_path = tmp_path / "followup_queue.json"

    subprocess.run(
        [
            sys.executable,
            str(_FOLLOWUP_QUEUE_TOOL),
            "--readiness-report",
            str(readiness_path),
            "--queue-out",
            str(queue_path),
            "--queue-id",
            "unit_followup_execution_cli",
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert queue["queue_id"] == "unit_followup_execution_cli"
    assert queue["metadata"]["schema"] == PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA
    assert len(queue["experiments"]) == 1


def test_emit_coverage_acquisition_plan_cli(tmp_path: pathlib.Path) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(_work_order("densify_pair_coverage_for_grouped_search")),
    )
    canvas_path = _write_json(
        tmp_path / "canvas.json",
        {
            "schema": "pair_frame_scorer_geometry_lattice_5d_canvas_populated_v1",
            "archive_sha256": "e" * 64,
            "cells": [],
        },
    )
    out_path = tmp_path / "plan.json"

    subprocess.run(
        [
            sys.executable,
            str(_EMIT_TOOL),
            "--coverage-audit",
            str(audit_path),
            "--work-order-id",
            "densify_pair_coverage_for_grouped_search",
            "--canvas-path",
            str(canvas_path),
            "--output",
            str(out_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_PLAN_SCHEMA
    assert payload["promotable"] is False


def test_audit_followup_requests_cli(tmp_path: pathlib.Path) -> None:
    plan = build_coverage_acquisition_plan(
        coverage_audit=_audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors")),
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    _write_json(plans_dir / "exact_acquisition_plan.json", plan)
    report_path = tmp_path / "followup_readiness_report.json"

    subprocess.run(
        [
            sys.executable,
            str(_AUDIT_TOOL),
            "--plans-dir",
            str(plans_dir),
            "--output",
            str(report_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA
    assert report["request_count"] == 1
    assert report["requests"][0]["ready"] is False
    assert report["requests"][0]["blockers"] == [
        "submission_bundle_result_path_missing"
    ]


def test_bind_followup_inputs_cli_refreshes_readiness(tmp_path: pathlib.Path) -> None:
    archive_payload = b"unit submission archive\n"
    archive_sha = _sha256_bytes(archive_payload)
    audit = _audit(
        _work_order(
            "populate_missing_paired_cpu_cuda_axis_anchors",
            priority=1,
            archive_sha256=archive_sha,
        ),
        _work_order(
            "acquire_negative_delta_cells_before_operator_fanout",
            priority=2,
            archive_sha256=archive_sha,
        ),
        archive_sha256=archive_sha,
    )
    exact_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    mlx_plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="acquire_negative_delta_cells_before_operator_fanout",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    _write_json(plans_dir / "exact_acquisition_plan.json", exact_plan)
    _write_json(plans_dir / "mlx_acquisition_plan.json", mlx_plan)
    _write_submission_bundle_result(
        tmp_path / "submission_bundle_result.json",
        archive_bytes=archive_payload,
    )
    reference_cache = _write_mlx_cache(tmp_path / "reference_mlx_cache")
    candidate_cache = _write_mlx_cache(
        tmp_path / "candidate_mlx_cache",
        archive_sha256=archive_sha,
        local_advisory_identity=True,
    )
    binding_report = tmp_path / "followup_input_binding_report.json"
    readiness_report = tmp_path / "followup_readiness_report.json"

    subprocess.run(
        [
            sys.executable,
            str(_BIND_TOOL),
            "--plans-dir",
            str(plans_dir),
            "--search-root",
            str(tmp_path),
            "--reference-mlx-cache-dir",
            str(reference_cache),
            "--candidate-mlx-cache-dir",
            str(candidate_cache),
            "--output",
            str(binding_report),
            "--refreshed-readiness-output",
            str(readiness_report),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    binding = json.loads(binding_report.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_report.read_text(encoding="utf-8"))
    assert binding["schema"] == PAIR_FRAME_5D_FOLLOWUP_INPUT_BINDING_REPORT_SCHEMA
    assert binding["bound_request_count"] == 2
    assert readiness["schema"] == PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA
    assert readiness["ready_request_count"] == 2


def test_build_pair_frame_5d_coverage_acquisition_queue_shape(
    tmp_path: pathlib.Path,
) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(
            _work_order("populate_missing_paired_cpu_cuda_axis_anchors", priority=1),
            _work_order("densify_frame_coverage_for_masked_and_feathered_search", priority=2),
        ),
    )

    queue = build_pair_frame_5d_coverage_acquisition_queue(
        repo_root=_REPO_ROOT,
        coverage_audit_path=audit_path,
        canvas_path=tmp_path / "canvas.json",
        output_root=tmp_path / "requests",
        queue_id="unit_coverage_acquisition",
        local_cpu_concurrency=2,
        followup_search_roots=[tmp_path / "existing_artifact_root"],
    )

    assert queue["schema"] == "experiment_queue.v1"
    assert queue["controls"]["max_concurrency"]["local_cpu"] == 2
    assert queue["metadata"]["blocked_work_order_ids"] == [
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ]
    assert queue["metadata"]["executable_work_order_ids"] == [
        "densify_frame_coverage_for_masked_and_feathered_search"
    ]
    assert len(queue["experiments"]) == 4
    first = queue["experiments"][0]
    assert first["metadata"]["schema"] == PAIR_FRAME_5D_COVERAGE_ACQUISITION_QUEUE_SCHEMA
    assert first["metadata"]["score_claim"] is False
    assert first["metadata"]["executable_now"] is False
    assert first["metadata"]["blocking_conditions"] == [
        "requires_byte_closed_submission_bundle_for_paired_auth_eval",
        "requires_dispatch_claim_before_paid_or_remote_exact_axis_work",
    ]
    assert first["steps"][0]["command"][1] == (
        "tools/emit_5d_canvas_coverage_acquisition_plan.py"
    )
    readiness = queue["experiments"][-2]
    assert readiness["id"] == "audit_blocked_followup_requests"
    assert readiness["metadata"]["followup_readiness_report_path"].endswith(
        "followup_readiness_report.json"
    )
    assert readiness["metadata"]["followup_execution_queue_path"].endswith(
        "followup_execution_queue.json"
    )
    assert readiness["metadata"]["followup_execution_worker_result_path"].endswith(
        "followup_execution_worker_result.json"
    )
    assert readiness["metadata"]["followup_input_binding_report_path"].endswith(
        "followup_input_binding_report.json"
    )
    assert readiness["metadata"]["followup_search_roots"][0].endswith("requests")
    assert readiness["metadata"]["followup_search_roots"][1].endswith(
        "existing_artifact_root"
    )
    assert readiness["steps"][0]["command"][1] == (
        "tools/bind_5d_coverage_followup_inputs.py"
    )
    search_root_indexes = [
        index
        for index, value in enumerate(readiness["steps"][0]["command"])
        if value == "--search-root"
    ]
    assert len(search_root_indexes) == 2
    assert readiness["steps"][0]["command"][search_root_indexes[0] + 1].endswith(
        "requests"
    )
    assert readiness["steps"][0]["command"][search_root_indexes[1] + 1].endswith(
        "existing_artifact_root"
    )
    assert readiness["steps"][0]["command"][
        readiness["steps"][0]["command"].index("--output") + 1
    ] == readiness["metadata"]["followup_input_binding_report_path"]
    assert readiness["steps"][0]["command"][
        readiness["steps"][0]["command"].index("--refreshed-readiness-output") + 1
    ] == readiness["metadata"]["followup_readiness_report_path"]
    assert readiness["steps"][1]["id"] == "emit_followup_execution_queue"
    assert readiness["steps"][1]["command"][1] == (
        "tools/build_5d_coverage_followup_execution_queue.py"
    )
    assert readiness["steps"][2]["id"] == "validate_followup_execution_queue"
    assert readiness["steps"][2]["command"] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        readiness["metadata"]["followup_execution_queue_path"],
        "validate",
    ]
    assert readiness["steps"][3]["id"] == "run_followup_execution_queue_bounded_local"
    assert readiness["steps"][3]["command"][:5] == [
        ".venv/bin/python",
        "tools/experiment_queue.py",
        "--queue",
        readiness["metadata"]["followup_execution_queue_path"],
        "run-worker",
    ]
    assert readiness["steps"][3]["command"][-2:] == [
        "--output",
        readiness["metadata"]["followup_execution_worker_result_path"],
    ]
    refresh = queue["experiments"][-1]
    assert refresh["id"] == "refresh_reaudit_and_refire_extended_operators"
    assert refresh["metadata"]["external_blocking_work_order_ids"] == [
        "populate_missing_paired_cpu_cuda_axis_anchors"
    ]


def test_build_coverage_acquisition_queue_cli(tmp_path: pathlib.Path) -> None:
    audit_path = _write_json(
        tmp_path / "audit.json",
        _audit(_work_order("acquire_negative_delta_cells_before_operator_fanout")),
    )
    queue_path = tmp_path / "queue.json"

    subprocess.run(
        [
            sys.executable,
            str(_BUILD_TOOL),
            "--coverage-audit",
            str(audit_path),
            "--canvas-path",
            str(tmp_path / "canvas.json"),
            "--output-root",
            str(tmp_path / "requests"),
            "--queue-out",
            str(queue_path),
            "--queue-id",
            "unit_coverage_acquisition_cli",
            "--local-cpu-concurrency",
            "3",
            "--followup-search-root",
            str(tmp_path / "artifact_root"),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["queue_id"] == "unit_coverage_acquisition_cli"
    assert payload["controls"]["max_concurrency"]["local_cpu"] == 3
    assert len(payload["experiments"]) == 3
    assert payload["experiments"][-2]["id"] == "audit_blocked_followup_requests"
    assert payload["experiments"][-2]["metadata"]["followup_search_roots"][1].endswith(
        "artifact_root"
    )


def test_audit_coverage_followup_requests_cli(tmp_path: pathlib.Path) -> None:
    audit = _audit(_work_order("populate_missing_paired_cpu_cuda_axis_anchors"))
    plan = build_coverage_acquisition_plan(
        coverage_audit=audit,
        work_order_id="populate_missing_paired_cpu_cuda_axis_anchors",
        coverage_audit_path="audit.json",
        repo_root=_REPO_ROOT,
        canvas_path="canvas.json",
    )
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    _write_json(plans_dir / "exact_plan.json", plan)
    out_path = tmp_path / "followup_readiness_report.json"

    subprocess.run(
        [
            sys.executable,
            str(_AUDIT_TOOL),
            "--plans-dir",
            str(plans_dir),
            "--output",
            str(out_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA
    assert payload["ready_request_count"] == 0
    assert payload["blocked_request_count"] == 1
    assert payload["requests"][0]["materialized_command"] is None


def test_build_followup_execution_queue_cli_no_ready(tmp_path: pathlib.Path) -> None:
    readiness_path = _write_json(
        tmp_path / "blocked_readiness_report.json",
        {
            "schema": PAIR_FRAME_5D_FOLLOWUP_READINESS_REPORT_SCHEMA,
            "plan_count": 0,
            "request_count": 0,
            "ready_request_count": 0,
            "blocked_request_count": 0,
            "requests": [],
            "allowed_use": "queue_followup_readiness_routing_only",
            "forbidden_use": "score_claim_or_dispatch_authority",
            "score_claim": False,
            "score_claim_valid": False,
            "score_claim_eligible": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_packet_ready": False,
            "reproduction_claim": False,
            "reproduction_equivalence": False,
        },
    )
    queue_path = tmp_path / "followup_execution_queue.json"

    subprocess.run(
        [
            sys.executable,
            str(_FOLLOWUP_QUEUE_TOOL),
            "--readiness-report",
            str(readiness_path),
            "--queue-out",
            str(queue_path),
        ],
        cwd=_REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["schema"] == PAIR_FRAME_5D_FOLLOWUP_EXECUTION_QUEUE_SCHEMA
    assert payload["experiments"][0]["status"] == "disabled"
