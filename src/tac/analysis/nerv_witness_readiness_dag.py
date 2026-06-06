# SPDX-License-Identifier: MIT
"""Evaluator-witness readiness DAG for HiNeRV/SNeRV long MLX training.

The current frontier question is not ordinary video fidelity.  A NeRV-family
candidate is launch-ready only when it can act as a compact witness compiler for
the frozen evaluator cells: SegNet last-frame argmax, PoseNet YUV6 pair output,
and exact archive bytes.  This module turns that doctrine into a planning-only
staircase DAG so partner findings become executable gates instead of prose.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from comma_lab.scheduler.staircase_dag import (
    build_staircase_dag_from_experiment_queue,
    plan_staircase_dispatch,
)
from tac.analysis.nerv_pair_local_distortion_servo import (
    PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA,
    pair_local_servo_static_contract,
)
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    apply_proxy_evidence_boundary,
    require_no_truthy_authority_fields,
)

NERV_WITNESS_READINESS_DAG_SCHEMA = "nerv_witness_long_training_readiness_dag.v1"
NERV_WITNESS_READINESS_NODE_SCHEMA = "nerv_witness_readiness_node.v1"
NERV_WITNESS_GATE_STATUS_SCHEMA = "nerv_witness_gate_status.v1"
DEFAULT_QUEUE_ID = "nerv_witness_long_training_readiness_dag.v1"
CONTEST_RATE_SCORE_PER_BYTE = 25.0 / 37_545_489.0


class NervWitnessReadinessDagError(ValueError):
    """Raised when witness-readiness DAG inputs are invalid."""


def build_nerv_witness_readiness_dag(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    source_boundary_audit_report: str | Path | None = None,
    hinerv_smoke_report: str | Path | None = None,
    snerv_authority_gate_report: str | Path | None = None,
    partner_source_refs: Sequence[str | Path] = (),
    dag_id: str = DEFAULT_QUEUE_ID,
    max_nodes: int = 8,
) -> dict[str, Any]:
    """Build a false-authority DAG for the next NeRV long-run gates.

    ``hinerv_smoke_report`` may be a bounded short-smoke report.  It is used
    only to classify the next gate; it does not grant score authority.
    """

    repo = Path(repo_root).expanduser().resolve(strict=False)
    out_root = Path(output_root).expanduser().resolve(strict=False)
    if not out_root.is_absolute():
        out_root = repo / out_root

    partner_refs = [_source_ref(path) for path in partner_source_refs]
    source_boundary_evidence = _source_boundary_evidence(source_boundary_audit_report)
    hinerv_evidence = _hinerv_smoke_evidence(
        hinerv_smoke_report,
        repo_root=repo,
        output_root=out_root,
    )
    snerv_evidence = _snerv_authority_gate_evidence(snerv_authority_gate_report)
    parseback = _parseback_contract_evidence(repo)
    pair_servo = _pair_local_servo_contract_evidence(repo)

    nodes = _node_specs(
        repo_root=repo,
        output_root=out_root,
        source_boundary_evidence=source_boundary_evidence,
        hinerv_evidence=hinerv_evidence,
        snerv_evidence=snerv_evidence,
        parseback_evidence=parseback,
        pair_servo_evidence=pair_servo,
    )
    queue = _queue_from_nodes(nodes, queue_id=dag_id)
    dag = build_staircase_dag_from_experiment_queue(
        queue,
        dag_id=dag_id,
        source_path=None,
        resource_pools=[
            {
                "id": "m5_max_local_mlx",
                "label": "M5 Max local MLX witness-readiness worker",
                "slots": {"local_cpu": 8, "local_mlx": 1},
                "memory_gb": 128,
                "disk_gb": 80,
                "tags": ["darwin", "arm64", "mlx", "witness_readiness"],
            }
        ],
    )
    status_map = _status_map(nodes)
    dispatch = plan_staircase_dispatch(
        dag,
        status_map=status_map,
        max_nodes=max_nodes,
    )
    blockers = _readiness_blockers(nodes)
    next_actions = _next_actions(nodes, dispatch)
    payload = apply_proxy_evidence_boundary(
        {
            "schema": NERV_WITNESS_READINESS_DAG_SCHEMA,
            "objective": {
                "description": (
                    "Find the shortest contest-compliant archive whose inflated "
                    "frames are witnesses inside the same SegNet argmax cells "
                    "and PoseNet output cells as the original video."
                ),
                "score_formula": (
                    "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489"
                ),
                "rate_score_per_byte": CONTEST_RATE_SCORE_PER_BYTE,
                "segnet_geometry": "last_frame_rgb_argmax_pixels",
                "posenet_geometry": "two_frame_official_yuv6_pair",
                "human_visual_fidelity_authority": False,
                "offline_compiler_eval_time_split": {
                    "offline_allowed": [
                        "original_video",
                        "frozen_scorers",
                        "oracle_caches",
                        "giant_teachers",
                        "witness_search",
                    ],
                    "eval_time_allowed": [
                        "charged_archive_payload",
                        "deterministic_receiver",
                        "raw_frame_emission_under_official_limit",
                    ],
                    "eval_time_forbidden": [
                        "training",
                        "uncharged_learned_sidecars",
                        "external_artifact_reads",
                    ],
                },
                "distortion_stage_order": [
                    "class_birth",
                    "target_region_margin_crossing",
                    "smooth_argmax_disagreement",
                    "fakequant_survival",
                    "archive_parseback_survival",
                    "entropy_coder_pressure",
                    "late_optimizer_polish",
                ],
                "rate_pressure_requires_receiver_visible_birth": True,
                "evaluator_capacity_channels": [
                    "segnet_boundary_channel",
                    "posenet_yuv6_channel",
                    "shared_latent_program_channel",
                    "sparse_sidecar_correction_channel",
                ],
            },
            "dag": dag,
            "status_map": status_map,
            "dispatch_plan": dispatch,
            "gate_nodes": nodes,
            "partner_source_refs": partner_refs,
            "source_boundary_evidence": source_boundary_evidence,
            "hinerv_smoke_evidence": hinerv_evidence,
            "snerv_authority_gate_evidence": snerv_evidence,
            "parseback_contract_evidence": parseback,
            "pair_local_servo_contract_evidence": pair_servo,
            "long_training_approved": False,
            "hinerv_long_training_approved": False,
            "snerv_long_training_approved": False,
            "actionable_blockers": blockers,
            "next_actions": next_actions,
            "axis_tag": "[planning/control]",
            **PROXY_FALSE_AUTHORITY_FIELDS,
        },
        dispatch_blockers=[
            "witness_readiness_dag_is_planning_only",
            "long_training_requires_all_gate_nodes_succeeded",
            "exact_score_claim_requires_receiver_closed_cpu_cuda_replay",
        ],
    )
    return payload


def check_witness_gate_status(
    *,
    node_id: str,
    source_boundary_audit_report: str | Path | None = None,
    hinerv_smoke_report: str | Path | None = None,
    snerv_authority_gate_report: str | Path | None = None,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return a machine-readable status for a DAG check command."""

    repo = Path(repo_root).expanduser().resolve(strict=False)
    out_root = repo / ".omx" / "research" / "nerv_witness_readiness"
    payload = build_nerv_witness_readiness_dag(
        repo_root=repo,
        output_root=out_root,
        source_boundary_audit_report=source_boundary_audit_report,
        hinerv_smoke_report=hinerv_smoke_report,
        snerv_authority_gate_report=snerv_authority_gate_report,
        max_nodes=1,
    )
    nodes = {
        str(node["node_id"]): node
        for node in payload.get("gate_nodes", [])
        if isinstance(node, Mapping)
    }
    if node_id not in nodes:
        raise NervWitnessReadinessDagError(f"unknown witness gate node: {node_id}")
    node = nodes[node_id]
    return apply_proxy_evidence_boundary(
        {
            "schema": NERV_WITNESS_GATE_STATUS_SCHEMA,
            "node_id": node_id,
            "status": node.get("status"),
            "satisfied": bool(node.get("satisfied")),
            "blockers": list(node.get("blockers") or []),
            "evidence": dict(node.get("evidence") or {}),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
        dispatch_blockers=["witness_gate_status_is_planning_only"],
    )


def _source_ref(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve(strict=False)
    exists = p.exists()
    return {
        "path": p.as_posix(),
        "exists": exists,
        "kind": "partner_burndown_or_first_principles_input",
    }


def _parseback_contract_evidence(repo: Path) -> dict[str, Any]:
    source = repo / "src/tac/training/long_training_canonical.py"
    tests = repo / "src/tac/tests/test_long_training_archive_selection.py"
    source_text = source.read_text(encoding="utf-8") if source.is_file() else ""
    test_text = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    required_needles = [
        "archive_selection_replay_required",
        "archive_replay_components",
        "archive_selection_replay_required_but_adapter_missing_archive_replay_components",
        "archive_selection_replay_required_but_archive_replay_components_returned_none",
    ]
    test_needles = [
        "test_archive_selection_required_parseback_fails_closed_without_hook",
        "archive_parseback_replay_proxy_false_authority",
    ]
    missing = [
        needle
        for needle in required_needles
        if needle not in source_text
    ] + [needle for needle in test_needles if needle not in test_text]
    return {
        "schema": "nerv_parseback_selection_static_contract_evidence.v1",
        "source_path": source.as_posix(),
        "test_path": tests.as_posix(),
        "implemented_contract_present": not missing,
        "missing_needles": missing,
        "recommended_validation_command": [
            "uv",
            "run",
            "pytest",
            "src/tac/tests/test_long_training_archive_selection.py",
            "-q",
        ],
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _pair_local_servo_contract_evidence(repo: Path) -> dict[str, Any]:
    source = repo / "src/tac/analysis/nerv_pair_local_distortion_servo.py"
    tests = repo / "src/tac/tests/test_nerv_pair_local_distortion_servo.py"
    source_text = source.read_text(encoding="utf-8") if source.is_file() else ""
    test_text = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    required_needles = [
        "admit_pair_local_distortion_action",
        "PairLocalSurfaceTrace",
        "PairLocalScoreState",
        "select_worst_scorer_debt_target",
        "exact_pair_local_score_delta",
        "PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA",
    ]
    test_needles = [
        "test_admits_frame1_pair_local_action_only_after_parseback_survival",
        "test_rejects_subquantum_float_update_even_when_score_numbers_improve",
        "test_rejects_live_argmax_motion_lost_by_fakequant_or_parseback",
        "test_rejects_when_exact_nonlinear_score_worsens_despite_seg_improvement",
        "test_frame0_pose_only_cannot_claim_segnet_mutation",
    ]
    missing = [
        needle for needle in required_needles if needle not in source_text
    ] + [needle for needle in test_needles if needle not in test_text]
    contract = pair_local_servo_static_contract()
    contract_schema_ok = (
        contract.get("schema") == PAIR_LOCAL_DISTORTION_SERVO_STATIC_CONTRACT_SCHEMA
    )
    if not contract_schema_ok:
        missing.append("pair_local_servo_static_contract_schema_mismatch")
    return {
        "schema": "nerv_pair_local_distortion_servo_static_evidence.v1",
        "source_path": source.as_posix(),
        "test_path": tests.as_posix(),
        "implemented_contract_present": not missing,
        "missing_needles": missing,
        "contract": contract,
        "recommended_validation_command": [
            "uv",
            "run",
            "pytest",
            "src/tac/tests/test_nerv_pair_local_distortion_servo.py",
            "-q",
        ],
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _source_boundary_evidence(report_path: str | Path | None) -> dict[str, Any]:
    if report_path is None:
        return {
            "schema": "nerv_witness_source_boundary_evidence.v1",
            "report_path": None,
            "report_loaded": False,
            "source_boundary_clean": False,
            "ready_for_witness_compile": False,
            "required_outputs": [
                "charged_vs_free_boundary.md",
                "conservative_mode_rules.md",
                "aggressive_mode_rules.md",
                "inflate_source_constant_audit.json",
            ],
            "blockers": ["nerv_witness_source_boundary_audit_missing"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    path = Path(report_path).expanduser().resolve(strict=False)
    payload = _read_json_or_none(path)
    if payload is None:
        return {
            "schema": "nerv_witness_source_boundary_evidence.v1",
            "report_path": path.as_posix(),
            "report_loaded": False,
            "source_boundary_clean": False,
            "ready_for_witness_compile": False,
            "blockers": ["nerv_witness_source_boundary_audit_invalid"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    require_no_truthy_authority_fields(payload, context="source_boundary_audit_report")
    clean = bool(payload.get("source_boundary_clean")) and bool(
        payload.get("ready_for_witness_compile")
    )
    blockers = [str(item) for item in payload.get("blockers") or []]
    if not clean and not blockers:
        blockers.append("nerv_witness_source_boundary_audit_not_clean")
    return {
        "schema": "nerv_witness_source_boundary_evidence.v1",
        "report_path": path.as_posix(),
        "report_loaded": True,
        "source_boundary_clean": clean,
        "ready_for_witness_compile": bool(payload.get("ready_for_witness_compile")),
        "mode": payload.get("mode"),
        "blockers": blockers,
        "source_report_count": len(payload.get("source_reports") or []),
        "issue_count": len(payload.get("issues") or []),
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _hinerv_smoke_evidence(
    report_path: str | Path | None,
    *,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if report_path is None:
        return {
            "schema": "hinerv_short_smoke_witness_evidence.v1",
            "report_path": None,
            "report_loaded": False,
            "status": "missing",
            "direct_smoke_rerun_argv": _default_hinerv_smoke_command(output_root),
            "blockers": ["hinerv_short_receiver_surface_smoke_missing"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    path = Path(report_path).expanduser().resolve(strict=False)
    payload = _read_json_or_none(path)
    if payload is None:
        return {
            "schema": "hinerv_short_smoke_witness_evidence.v1",
            "report_path": path.as_posix(),
            "report_loaded": False,
            "status": "missing_or_invalid_json",
            "direct_smoke_rerun_argv": _default_hinerv_smoke_command(output_root),
            "blockers": ["hinerv_short_receiver_surface_smoke_invalid"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    require_no_truthy_authority_fields(payload, context="hinerv_smoke_report")
    direct_argv = payload.get("direct_smoke_rerun_argv")
    if not isinstance(direct_argv, list) or not direct_argv:
        direct_argv = _default_hinerv_smoke_command(output_root)
    else:
        direct_argv = [str(item) for item in direct_argv]
    metrics = {
        key: _find_number(payload, key)
        for key in (
            "accepted_step_count",
            "hard_birth_argmax_progress_accepted_step_count",
            "hard_birth_argmax_progress_rejected_step_count",
            "receiver_quantum_attempt_count",
            "receiver_quantum_shrink_attempt_count",
            "receiver_quantum_growth_attempt_count",
            "max_candidate_segnet_worst_debt_reduction",
            "max_candidate_segnet_total_debt_reduction",
            "max_candidate_segnet_min_ratio_increase",
            "max_candidate_segnet_total_debt_spill_given_worst_improvement",
            "max_accepted_frame1_receiver_uint8_changed_count",
            "max_accepted_frame1_receiver_uint8_delta_abs",
            "max_candidate_frame1_receiver_uint8_changed_count",
            "max_candidate_frame1_receiver_uint8_delta_abs",
            "pose_exact_delta",
            "max_candidate_pose_exact_delta",
        )
    }
    trace_path = path.parent / "hi_nerv_mlx_training" / "nerv_crux_trace_rows.json"
    readiness_path = (
        path.parent
        / "hi_nerv_mlx_training"
        / "hi_nerv_short_scorer_smoke_readiness.json"
    )
    blockers: list[str] = []
    if metrics["receiver_quantum_attempt_count"] is None:
        blockers.append("hinerv_receiver_quantum_telemetry_missing")
    if (metrics["hard_birth_argmax_progress_accepted_step_count"] or 0.0) <= 0.0:
        blockers.append("hinerv_hard_birth_argmax_progress_not_accepted")
    if (
        (metrics["max_candidate_segnet_worst_debt_reduction"] or 0.0) > 0.0
        and (metrics["max_candidate_segnet_total_debt_spill_given_worst_improvement"] or 0.0) > 0.0
    ):
        blockers.append("hinerv_localized_projection_trust_region_missing")
    if (metrics["max_candidate_segnet_min_ratio_increase"] or 0.0) <= 0.0:
        blockers.append("hinerv_target_min_ratio_not_lifted")
    return {
        "schema": "hinerv_short_smoke_witness_evidence.v1",
        "report_path": path.as_posix(),
        "report_loaded": True,
        "status": "succeeded_with_blockers" if blockers else "succeeded",
        "direct_smoke_rerun_argv": direct_argv,
        "crux_trace_path": trace_path.as_posix(),
        "crux_trace_exists": trace_path.is_file(),
        "short_scorer_readiness_path": readiness_path.as_posix(),
        "short_scorer_readiness_exists": readiness_path.is_file(),
        "metrics": metrics,
        "blockers": blockers,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _snerv_authority_gate_evidence(report_path: str | Path | None) -> dict[str, Any]:
    if report_path is None:
        return {
            "schema": "snerv_official_source_forward_gate_evidence.v1",
            "report_path": None,
            "report_loaded": False,
            "status": "missing",
            "blockers": ["snerv_official_mfu_hfr_tub_authority_gate_missing"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    path = Path(report_path).expanduser().resolve(strict=False)
    payload = _read_json_or_none(path)
    if payload is None:
        return {
            "schema": "snerv_official_source_forward_gate_evidence.v1",
            "report_path": path.as_posix(),
            "report_loaded": False,
            "status": "missing_or_invalid_json",
            "blockers": ["snerv_official_mfu_hfr_tub_authority_gate_invalid"],
            **PROXY_FALSE_AUTHORITY_FIELDS,
        }
    require_no_truthy_authority_fields(payload, context="snerv_authority_gate_report")
    ready = bool(payload.get("official_tub_lf_hf_decoder_replacement_ready"))
    blockers = [str(item) for item in payload.get("queue_blockers") or payload.get("blockers") or []]
    if not ready and not blockers:
        blockers.append("snerv_official_tub_lf_hf_decoder_replacement_not_ready")
    return {
        "schema": "snerv_official_source_forward_gate_evidence.v1",
        "report_path": path.as_posix(),
        "report_loaded": True,
        "status": "succeeded" if ready else "blocked",
        "official_tub_lf_hf_decoder_replacement_ready": ready,
        "blockers": blockers,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _distortion_birth_stage_evidence(
    hinerv_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        key: _metric(hinerv_evidence, key)
        for key in (
            "receiver_quantum_attempt_count",
            "hard_birth_argmax_progress_accepted_step_count",
            "max_candidate_segnet_worst_debt_reduction",
            "max_candidate_segnet_min_ratio_increase",
            "max_candidate_segnet_total_debt_spill_given_worst_improvement",
            "max_accepted_frame1_receiver_uint8_changed_count",
            "max_accepted_frame1_receiver_uint8_delta_abs",
            "max_candidate_pose_exact_delta",
        )
    }
    blockers: list[str] = []
    if not bool(hinerv_evidence.get("report_loaded")):
        blockers.append("distortion_birth_smoke_report_missing")
    if metrics["receiver_quantum_attempt_count"] <= 0.0:
        blockers.append("receiver_quantum_attempt_telemetry_missing")
    if metrics["hard_birth_argmax_progress_accepted_step_count"] <= 0.0:
        blockers.append("receiver_visible_hard_birth_update_not_accepted")
    if metrics["max_candidate_segnet_worst_debt_reduction"] <= 0.0:
        blockers.append("worst_region_debt_reduction_missing")
    if metrics["max_candidate_segnet_min_ratio_increase"] <= 0.0:
        blockers.append("target_region_min_ratio_lift_missing")
    if metrics["max_candidate_segnet_total_debt_spill_given_worst_improvement"] > 0.0:
        blockers.append("worst_region_improvement_has_total_seg_spill")
    if metrics["max_accepted_frame1_receiver_uint8_changed_count"] <= 0.0:
        blockers.append("accepted_update_receiver_uint8_movement_missing")
    if metrics["max_accepted_frame1_receiver_uint8_delta_abs"] <= 0.0:
        blockers.append("accepted_update_receiver_uint8_delta_missing")
    satisfied = not blockers
    return {
        "schema": "nerv_distortion_birth_before_rate_pressure_evidence.v1",
        "source": "hinerv_short_receiver_surface_smoke",
        "distortion_birth_before_rate_pressure_satisfied": satisfied,
        "metrics": metrics,
        "blockers": blockers,
        "stage_order": [
            "class_birth",
            "target_region_margin_crossing",
            "smooth_argmax_disagreement",
            "fakequant_survival",
            "archive_parseback_survival",
            "entropy_coder_pressure",
            "late_optimizer_polish",
        ],
        "rate_pressure_controls_blocked_until_satisfied": [
            "coder_qat",
            "section_byte_duals",
            "c1a_entropy_pressure",
            "byte_compiler_selection",
            "muon_late_polish",
        ],
        "acceptance_policy": {
            "loss_only_improvement_is_not_enough": True,
            "coverage_without_target_min_ratio_lift_blocks": True,
            "receiver_uint8_movement_required": True,
            "total_seg_spill_blocks": True,
            "score_claim": False,
        },
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def _node_specs(
    *,
    repo_root: Path,
    output_root: Path,
    source_boundary_evidence: Mapping[str, Any],
    hinerv_evidence: Mapping[str, Any],
    snerv_evidence: Mapping[str, Any],
    parseback_evidence: Mapping[str, Any],
    pair_servo_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parseback_ok = bool(parseback_evidence.get("implemented_contract_present"))
    pair_servo_ok = bool(pair_servo_evidence.get("implemented_contract_present"))
    source_boundary_ok = bool(source_boundary_evidence.get("source_boundary_clean"))
    source_boundary_blockers = list(source_boundary_evidence.get("blockers") or [])
    oracle_cache = {
        "schema": "nerv_witness_exact_scorer_oracle_cache_evidence.v1",
        "required_outputs": [
            "seg_target_masks",
            "pose_target_vectors",
            "region_components",
            "pose_jacobian_sketches",
            "authority_hashes",
        ],
        "blockers": ["nerv_exact_scorer_oracle_cache_missing"],
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    hinerv_smoke_loaded = bool(hinerv_evidence.get("report_loaded"))
    hinerv_trace_ok = bool(hinerv_evidence.get("crux_trace_exists")) or (
        _metric(hinerv_evidence, "receiver_quantum_attempt_count") > 0.0
    )
    hinerv_projection_ok = not hinerv_evidence.get("blockers") and hinerv_smoke_loaded
    birth_stage_evidence = _distortion_birth_stage_evidence(hinerv_evidence)
    birth_stage_ok = bool(
        birth_stage_evidence.get("distortion_birth_before_rate_pressure_satisfied")
    )
    snerv_source_ok = bool(
        snerv_evidence.get("official_tub_lf_hf_decoder_replacement_ready")
    )
    nodes = [
        _node(
            node_id="shared.source_boundary_compliance_audit",
            family="shared",
            stage="source_boundary_compliance",
            priority=0,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "shared.source_boundary_compliance_audit",
            ],
            satisfied=source_boundary_ok,
            blockers=[] if source_boundary_ok else source_boundary_blockers,
            evidence=source_boundary_evidence,
            acceptance="charged learned payload never leaks into uncharged source",
        ),
        _node(
            node_id="shared.exact_scorer_oracle_cache",
            family="shared",
            stage="exact_scorer_oracle_cache",
            priority=1,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "shared.exact_scorer_oracle_cache",
            ],
            dependencies=["shared.source_boundary_compliance_audit"],
            satisfied=False,
            blockers=["nerv_exact_scorer_oracle_cache_missing"],
            evidence=oracle_cache,
            acceptance="source video scorer sufficient statistics are cached with authority hashes",
        ),
        _node(
            node_id="shared.parseback_selection_contract",
            family="shared",
            stage="parseback_selection_contract",
            priority=2,
            command=[
                "uv",
                "run",
                "pytest",
                "src/tac/tests/test_long_training_archive_selection.py",
                "-q",
            ],
            satisfied=parseback_ok,
            blockers=[] if parseback_ok else ["archive_parseback_selection_contract_missing"],
            evidence=parseback_evidence,
            acceptance="archive replay hook exists and required parse-back selection fails closed",
        ),
        _node(
            node_id="shared.distortion_trace_harness",
            family="shared",
            stage="distortion_trace",
            priority=3,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "shared.distortion_trace_harness",
            ],
            dependencies=["shared.parseback_selection_contract"],
            satisfied=hinerv_trace_ok,
            blockers=[] if hinerv_trace_ok else ["live_fakequant_receiver_trace_missing"],
            evidence=hinerv_evidence,
            acceptance="live/fakequant/receiver surface emits crux telemetry",
        ),
        _node(
            node_id="hinerv.short_receiver_surface_smoke",
            family="hi_nerv",
            stage="short_receiver_surface_smoke",
            priority=4,
            command=list(hinerv_evidence.get("direct_smoke_rerun_argv") or _default_hinerv_smoke_command(output_root)),
            dependencies=["shared.parseback_selection_contract"],
            resource_kind="local_mlx",
            satisfied=hinerv_smoke_loaded,
            blockers=[] if hinerv_smoke_loaded else ["hinerv_short_receiver_surface_smoke_missing"],
            evidence=hinerv_evidence,
            acceptance="bounded real-scorer MLX smoke artifact exists",
        ),
        _node(
            node_id="shared.distortion_birth_before_rate_pressure",
            family="shared",
            stage="distortion_birth_before_rate_pressure",
            priority=5,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "shared.distortion_birth_before_rate_pressure",
            ],
            dependencies=[
                "shared.distortion_trace_harness",
                "hinerv.short_receiver_surface_smoke",
            ],
            satisfied=birth_stage_ok,
            blockers=list(birth_stage_evidence.get("blockers") or []),
            evidence=birth_stage_evidence,
            acceptance=(
                "receiver-visible class birth and target-region debt movement "
                "are proven before QAT, byte duals, entropy pressure, or late "
                "optimizer polish can be treated as launch evidence"
            ),
        ),
        _node(
            node_id="shared.pair_local_distortion_servo_contract",
            family="shared",
            stage="pair_local_distortion_servo_contract",
            priority=6,
            command=[
                "uv",
                "run",
                "pytest",
                "src/tac/tests/test_nerv_pair_local_distortion_servo.py",
                "-q",
            ],
            dependencies=["shared.distortion_birth_before_rate_pressure"],
            satisfied=pair_servo_ok,
            blockers=(
                []
                if pair_servo_ok
                else ["pair_local_distortion_servo_contract_missing"]
            ),
            evidence=pair_servo_evidence,
            acceptance=(
                "pair-local actions are admitted only after uint8/preprocess/"
                "SegNet/Pose/fakequant/archive-parseback survival and exact "
                "nonlinear score improvement"
            ),
        ),
        _node(
            node_id="shared.joint_seg_pose_trust_region",
            family="shared",
            stage="joint_seg_pose_trust_region",
            priority=7,
            command=[
                "uv",
                "run",
                "pytest",
                "src/tac/substrates/hi_nerv/tests/test_short_scorer_readiness.py",
                "-q",
            ],
            dependencies=[
                "shared.distortion_birth_before_rate_pressure",
                "shared.pair_local_distortion_servo_contract",
            ],
            satisfied=False,
            blockers=["joint_seg_pose_exact_delta_admission_not_yet_proven_in_smoke"],
            evidence={
                "required_delta": (
                    "100*delta_d_seg + sqrt(10*d_pose_new) - "
                    "sqrt(10*d_pose_old) + rate_delta"
                ),
                "rate_pressure_precondition": (
                    "distortion birth must survive receiver uint8 surface first"
                ),
            },
            acceptance="accepted updates reduce exact nonlinear Seg/Pose/rate score units",
        ),
        _node(
            node_id="hinerv.localized_target_region_projection_actuator",
            family="hi_nerv",
            stage="localized_target_region_projection",
            priority=8,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "hinerv.localized_target_region_projection_actuator",
            ],
            dependencies=[
                "hinerv.short_receiver_surface_smoke",
                "shared.joint_seg_pose_trust_region",
            ],
            resource_kind="local_mlx",
            satisfied=hinerv_projection_ok,
            blockers=list(hinerv_evidence.get("blockers") or []),
            evidence=hinerv_evidence,
            acceptance=(
                "worst target-region debt or min-ratio improves without total Seg "
                "spill or Pose regression"
            ),
        ),
        _node(
            node_id="snerv.official_mfu_hfr_tub_source_forward",
            family="snerv",
            stage="official_mfu_hfr_tub_source_forward",
            priority=9,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_snerv_official_tub_lf_hf_replacement_authority_gate.py",
                "--output-root",
                (output_root / "snerv_official_tub_lf_hf_gate").as_posix(),
            ],
            dependencies=["shared.parseback_selection_contract"],
            satisfied=snerv_source_ok,
            blockers=list(snerv_evidence.get("blockers") or []),
            evidence=snerv_evidence,
            acceptance="official MFU/HFR/TUB source-forward parity binds train/export/runtime",
        ),
        _node(
            node_id="snerv.lf_hf_representation_collapse_smoke",
            family="snerv",
            stage="lf_hf_representation_collapse",
            priority=9,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "snerv.lf_hf_representation_collapse_smoke",
            ],
            dependencies=["snerv.official_mfu_hfr_tub_source_forward"],
            resource_kind="local_mlx",
            satisfied=False,
            blockers=["snerv_score_aware_lf_hf_replacement_smoke_missing"],
            evidence={},
            acceptance="LF/HF replacement lowers scorer distortion enough under measured bytes",
        ),
        _node(
            node_id="hinerv.long_mlx_training_launch_gate",
            family="hi_nerv",
            stage="long_mlx_training_launch_gate",
            priority=10,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "hinerv.long_mlx_training_launch_gate",
            ],
            dependencies=[
                "shared.source_boundary_compliance_audit",
                "shared.exact_scorer_oracle_cache",
                "shared.distortion_birth_before_rate_pressure",
                "hinerv.localized_target_region_projection_actuator",
                "shared.distortion_trace_harness",
            ],
            resource_kind="local_mlx",
            satisfied=False,
            blockers=["hinerv_localized_target_region_projection_gate_not_satisfied"],
            evidence={},
            acceptance="short smoke, receiver proof, section telemetry, full-video MLX replay clean",
        ),
        _node(
            node_id="snerv.long_mlx_training_launch_gate",
            family="snerv",
            stage="long_mlx_training_launch_gate",
            priority=11,
            command=[
                "uv",
                "run",
                "python",
                "tools/build_nerv_witness_readiness_dag.py",
                "check-evidence",
                "--node-id",
                "snerv.long_mlx_training_launch_gate",
            ],
            dependencies=[
                "shared.source_boundary_compliance_audit",
                "shared.exact_scorer_oracle_cache",
                "shared.distortion_birth_before_rate_pressure",
                "snerv.official_mfu_hfr_tub_source_forward",
                "snerv.lf_hf_representation_collapse_smoke",
            ],
            resource_kind="local_mlx",
            satisfied=False,
            blockers=["snerv_official_source_forward_and_lf_hf_gates_not_satisfied"],
            evidence={},
            acceptance="official runtime binding plus LF/HF collapse plus receiver replay clean",
        ),
    ]
    return nodes


def _node(
    *,
    node_id: str,
    family: str,
    stage: str,
    priority: int,
    command: Sequence[str],
    satisfied: bool,
    blockers: Sequence[str],
    evidence: Mapping[str, Any],
    acceptance: str,
    dependencies: Sequence[str] = (),
    resource_kind: str = "local_cpu",
) -> dict[str, Any]:
    status = "succeeded" if satisfied else ("blocked" if blockers else "queued")
    return apply_proxy_evidence_boundary(
        {
            "schema": NERV_WITNESS_READINESS_NODE_SCHEMA,
            "node_id": node_id,
            "status": status,
            "satisfied": bool(satisfied),
            "blockers": [str(item) for item in blockers if str(item)],
            "family": family,
            "stage": stage,
            "priority": int(priority),
            "resource_kind": resource_kind,
            "dependencies": [str(dep) for dep in dependencies],
            "command": [str(item) for item in command],
            "acceptance": acceptance,
            "evidence": dict(evidence),
        },
        dispatch_blockers=["witness_gate_node_is_planning_only"],
    )


def _queue_from_nodes(nodes: Sequence[Mapping[str, Any]], *, queue_id: str) -> dict[str, Any]:
    return {
        "schema": "experiment_queue.v1",
        "queue_id": queue_id,
        "controls": {"mode": "running", "max_concurrency": {"local_cpu": 4, "local_mlx": 1}},
        "experiments": [
            {
                "id": str(node["node_id"]).replace(".", "__"),
                "status": "queued",
                "priority": int(node.get("priority") or 100),
                "lane_id": str(node.get("family") or "nerv_witness"),
                "metadata": {
                    "schema": NERV_WITNESS_READINESS_NODE_SCHEMA,
                    "source_node_id": node["node_id"],
                    "stage": node.get("stage"),
                    "acceptance": node.get("acceptance"),
                    "blockers": list(node.get("blockers") or []),
                    "evidence": dict(node.get("evidence") or {}),
                    **PROXY_FALSE_AUTHORITY_FIELDS,
                },
                "steps": [
                    {
                        "id": "gate",
                        "kind": "command",
                        "command": list(node["command"]),
                        "requires": [
                            str(dep).replace(".", "__") + ".gate"
                            for dep in node.get("dependencies", [])
                        ],
                        "resources": {"kind": node.get("resource_kind", "local_cpu")},
                        "postconditions": [
                            {
                                "type": "false_authority_gate",
                                "node_id": node["node_id"],
                            }
                        ],
                    }
                ],
            }
            for node in nodes
        ],
    }


def _status_map(nodes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(node["node_id"]).replace(".", "__") + ".gate": str(node["status"])
        for node in nodes
        if str(node.get("status")) in {"succeeded", "blocked"}
    }


def _readiness_blockers(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for node in nodes:
        if node.get("satisfied") is True:
            continue
        node_id = str(node.get("node_id"))
        for blocker in node.get("blockers") or []:
            blockers.append(f"{node_id}:{blocker}")
    return _ordered_unique(blockers)


def _next_actions(
    nodes: Sequence[Mapping[str, Any]],
    dispatch: Mapping[str, Any],
) -> list[str]:
    actions: list[str] = []
    for node in nodes:
        if node.get("satisfied") is False and node.get("blockers"):
            actions.append(f"fix_or_prove:{node['node_id']}")
    for selected in dispatch.get("selected_nodes") or []:
        if isinstance(selected, Mapping):
            actions.append(f"run_ready_gate:{selected.get('node_id')}")
    return _ordered_unique(actions)


def _default_hinerv_smoke_command(output_root: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family",
        "hi_nerv",
        "--planner-row-id",
        "hi_nerv::witness_readiness_short_smoke",
        "--allow-bounded-planner-row-timing-smoke-waiver",
        "--source-video-path",
        "upstream/videos/0.mkv",
        "--output-dir",
        (output_root / "hinerv_witness_readiness_short_smoke").as_posix(),
        "--overwrite",
        "--num-pairs",
        "1",
        "--epochs",
        "1",
        "--batch-pairs",
        "1",
        "--scorer-domain-bootstrap",
        "--scorer-domain-bootstrap-steps",
        "2",
        "--scorer-domain-bootstrap-segnet-hard-birth-weight",
        "2.0",
        "--coder-aware-qat",
        "--receiver-cache-quality-mlx-scorer-response-device-type",
        "cpu",
    ]


def _read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _find_number(payload: Any, key: str) -> float | None:
    queue: deque[Any] = deque([payload])
    while queue:
        item = queue.popleft()
        if isinstance(item, Mapping):
            if key in item:
                value = item[key]
                if isinstance(value, int | float) and not isinstance(value, bool):
                    value = float(value)
                    return value if math.isfinite(value) else None
            queue.extend(item.values())
        elif isinstance(item, list | tuple):
            queue.extend(item)
    return None


def _metric(evidence: Mapping[str, Any], key: str) -> float:
    metrics = evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    value = metrics.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else 0.0
    return 0.0


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "CONTEST_RATE_SCORE_PER_BYTE",
    "DEFAULT_QUEUE_ID",
    "NERV_WITNESS_GATE_STATUS_SCHEMA",
    "NERV_WITNESS_READINESS_DAG_SCHEMA",
    "NervWitnessReadinessDagError",
    "build_nerv_witness_readiness_dag",
    "check_witness_gate_status",
]
