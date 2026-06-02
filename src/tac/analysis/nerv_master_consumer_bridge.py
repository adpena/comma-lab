# SPDX-License-Identifier: MIT
"""Normalize NeRV stack control artifacts for master consumers.

This bridge is the thin abstraction above the fragmented NeRV control surfaces:
Cathedral Autopilot, final-rate-attack planners, bit allocators, sensitivity
maps, and probe-disambiguators should consume this packet instead of scraping
SNeRV/HiNeRV memos independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY as HPRC_FALSE_AUTHORITY,
)

SCHEMA = "nerv_master_consumer_bridge.v1"
AXIS_TAG = "[planning/control]"
CONTROL_INVENTORY_SCHEMA = "nerv_control_inventory.v1"
FALSE_AUTHORITY = {
    **HPRC_FALSE_AUTHORITY,
    "frontier_score_claim": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
}

MASTER_CONSUMER_ROUTES = (
    {
        "consumer": "cathedral_autopilot",
        "hook": "CATHEDRAL_AUTOPILOT_DISPATCH",
        "uses": [
            "rank NeRV implementation work after fail-closed blockers",
            "prevent local sketch surfaces from entering dispatch lanes",
        ],
    },
    {
        "consumer": "final_rate_attack",
        "hook": "BIT_ALLOCATOR",
        "uses": [
            "reuse modelsize byte caps and mixed precision grammar blockers",
            "route bit-level/section entropy opportunities to receiver work",
        ],
    },
    {
        "consumer": "sensitivity_map",
        "hook": "SENSITIVITY_MAP",
        "uses": [
            "map SegNet/PoseNet/master-gradient signals to decoder atoms",
            "select protected int8/fp16 atoms vs int4/int2/zero/RLE atoms",
        ],
    },
    {
        "consumer": "probe_disambiguator",
        "hook": "PROBE_DISAMBIGUATOR",
        "uses": [
            "arbitrate SR-NeRV resolution-axis claims",
            "arbitrate RNeRV/FFNeRV/BoostNeRV enhancer ablations",
        ],
    },
    {
        "consumer": "continual_learning_posterior",
        "hook": "CONTINUAL_LEARNING_POSTERIOR",
        "uses": [
            "preserve negative implementation/config findings",
            "feed future stack selection without chat-only signal loss",
        ],
    },
)


class NervMasterConsumerBridgeError(ValueError):
    """Raised when a NeRV master-consumer bridge cannot be constructed."""


def build_nerv_master_consumer_bridge(
    *,
    seam: Mapping[str, Any],
    control_inventory: Mapping[str, Any],
    implementation_sweep: Mapping[str, Any],
    modelsize_curve: Mapping[str, Any] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build one normalized no-authority packet for master consumers."""

    _require_mapping_schema(seam, "nerv_top_priority_stack_seam.v1", "seam")
    _require_mapping_schema(
        control_inventory,
        CONTROL_INVENTORY_SCHEMA,
        "control_inventory",
    )
    _require_mapping_schema(
        implementation_sweep,
        "nerv_implementation_design_sweep.v1",
        "implementation_sweep",
    )
    if modelsize_curve is not None:
        _require_mapping_schema(
            modelsize_curve,
            "nerv_modelsize_archive_curve.v1",
            "modelsize_curve",
        )

    generated = generated_utc or datetime.now(UTC).isoformat()
    blockers = _unique(
        [
            *_string_list(seam.get("blockers")),
            *_string_list(control_inventory.get("blockers")),
            *_control_inventory_blockers(control_inventory),
            *_string_list(implementation_sweep.get("blockers")),
            *_string_list((modelsize_curve or {}).get("blockers")),
        ]
    )
    units = _stack_units(implementation_sweep) + _control_inventory_units(
        control_inventory
    )
    units.extend(_memo_ref_units(implementation_sweep))
    if modelsize_curve is not None:
        units.extend(_modelsize_units(modelsize_curve))
    memo_refs = _mapping_list(implementation_sweep.get("related_omx_design_memo_refs"))

    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "verdict": (
            "GO_MASTER_CONSUMER_ROUTING__NO_GO_SCORE_PROMOTION_OR_EXACT_DISPATCH"
        ),
        "top_priority_carriers": list(seam.get("top_priority_carriers") or []),
        "baseline_to_beat": seam.get("baseline_to_beat"),
        "producer_surfaces": [
            _surface_ref("top_priority_stack_seam", seam),
            _surface_ref("control_inventory", control_inventory),
            _surface_ref("implementation_design_sweep", implementation_sweep),
            _surface_ref("modelsize_archive_curve", modelsize_curve)
            if modelsize_curve is not None
            else {
                "surface_id": "modelsize_archive_curve",
                "present": False,
                "schema": None,
                "blocker": "modelsize_archive_curve_payload_missing",
            },
        ],
        "master_consumer_routes": list(MASTER_CONSUMER_ROUTES),
        "master_consumer_units": units,
        "related_omx_design_memo_ref_count": len(memo_refs),
        "related_omx_design_memo_refs": list(memo_refs),
        "normalized_candidate": {
            "schema": SCHEMA,
            "candidate_id": "nerv_snerv_hinerv_top_priority_stack",
            "substrate_ids": ["snerv", "hinerv"],
            "baseline_to_beat": seam.get("baseline_to_beat"),
            "axis_tag": AXIS_TAG,
            "promotable": False,
            "predicted_delta_adjustment": 0.0,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "blockers": blockers,
        },
        "abstraction_policy": {
            "purpose": (
                "single master-consumer packet for Cathedral/final-rate-attack/"
                "sensitivity/probe consumers"
            ),
            "not_a_new_orchestrator": True,
            "extends_existing_cathedral_consumer_contract": True,
            "prevents_duplicate_glue": True,
            "new_master_consumers_should_accept": [
                "normalized_candidate",
                "master_consumer_units",
                "producer_surfaces",
                "axis_tag",
                "blockers",
                "false_authority_flags",
            ],
        },
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _stack_units(implementation_sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for row in _mapping_list(implementation_sweep.get("stack_sweeps")):
        stack_id = str(row.get("stack_id") or "unknown")
        blockers = _string_list(row.get("production_blockers"))
        units.append(
            {
                "unit_id": f"{stack_id}_implementation_design_gate",
                "unit_type": "implementation_design_gate",
                "stack_id": stack_id,
                "target_consumers": [
                    "cathedral_autopilot",
                    "continual_learning_posterior",
                    "probe_disambiguator",
                ],
                "planner_action": (
                    "close_source_parity_receiver_and_proof_blockers"
                    if blockers
                    else "eligible_for_next_local_training_gate"
                ),
                "blockers": blockers,
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    return units


def _control_inventory_units(control_inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    return (
        _control_row_units(control_inventory)
        + _binding_surface_units(control_inventory)
        + _control_inventory_evidence_units(control_inventory)
    )


def _control_row_units(control_inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for row in _mapping_list(control_inventory.get("control_rows")):
        control_id = str(row.get("control_id") or "unknown")
        applies_to = str(row.get("applies_to") or "unknown")
        missing = _string_list(row.get("missing_bindings"))
        status = str(row.get("binding_status") or "unknown")
        units.append(
            {
                "unit_id": f"{control_id}_master_consumer_route",
                "unit_type": "nerv_control_row_route",
                "control_id": control_id,
                "applies_to": applies_to,
                "binding_status": status,
                "target_consumers": _target_consumers_for_hook(control_id),
                "planner_action": (
                    "repair_missing_control_bindings"
                    if missing
                    else "reuse_existing_control_before_new_code"
                ),
                "missing_bindings": missing,
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    return units


def _binding_surface_units(control_inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    surfaces = control_inventory.get("local_binding_surfaces")
    if not isinstance(surfaces, Mapping):
        return []
    units = []
    for surface_id, paths in sorted(surfaces.items()):
        path_list = _string_list(paths)
        units.append(
            {
                "unit_id": f"{surface_id}_binding_surface",
                "unit_type": "local_binding_surface_route",
                "surface_id": str(surface_id),
                "path_count": len(path_list),
                "paths": path_list,
                "target_consumers": _target_consumers_for_hook(str(surface_id)),
                "planner_action": "reuse_binding_surface_before_new_code",
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    return units


def _control_inventory_evidence_units(
    control_inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for family, report in _mapping_items(
        control_inventory.get("decoder_weight_waterfill_reports")
    ):
        rows = _mapping_list(report.get("waterfill_rows"))
        for index, row in enumerate(rows):
            row_id = str(row.get("row_id") or f"waterfill_row_{index:04d}")
            replay_command_argv = _string_list(
                row.get("archive_ladder_replay_command_argv")
            )
            replay_output_dir = row.get("archive_ladder_replay_output_dir")
            blockers = _string_list(report.get("blockers")) + _string_list(
                row.get("blockers")
            )
            if replay_command_argv and not replay_output_dir:
                blockers.append("decoder_weight_waterfill_replay_output_dir_missing")
            if replay_output_dir and not replay_command_argv:
                blockers.append("decoder_weight_waterfill_replay_command_missing")
            units.append(
                {
                    "unit_id": f"{family}_{row_id}_decoder_weight_waterfill",
                    "unit_type": "decoder_weight_waterfill_report",
                    "family": str(family),
                    "row_id": row_id,
                    "report_path": report.get("report_path"),
                    "archive_bytes": row.get("archive_bytes"),
                    "archive_sha256": row.get("archive_sha256"),
                    "state_npz_artifact_sha256": row.get(
                        "state_npz_artifact_sha256"
                    ),
                    "waterfill_summary": row.get("waterfill_summary") or {},
                    "archive_ladder_replay_command_axis_tag": row.get(
                        "archive_ladder_replay_command_axis_tag"
                    ),
                    "archive_ladder_replay_command_argv": replay_command_argv,
                    "archive_ladder_replay_command_hint": row.get(
                        "archive_ladder_replay_command_hint"
                    ),
                    "archive_ladder_replay_output_dir": replay_output_dir,
                    "target_consumers": [
                        "final_rate_attack",
                        "bit_allocator",
                        "cathedral_autopilot",
                    ],
                    "planner_action": (
                        "run_decoder_weight_waterfill_replay_against_archive_ladder"
                    ),
                    "blockers": _unique(blockers),
                    "predicted_delta_adjustment": 0.0,
                    **FALSE_AUTHORITY,
                }
            )
    for family, report in _mapping_items(
        control_inventory.get("archive_ladder_replay_actuator_reports")
    ):
        rows = _mapping_list(report.get("replay_rows"))
        for index, row in enumerate(rows):
            row_id = str(row.get("row_id") or f"replay_row_{index:04d}")
            blockers = _string_list(report.get("blockers")) + _string_list(
                row.get("blockers")
            )
            if row.get("receiver_proof_ready") is not True:
                blockers.append("receiver_proof_not_ready_for_archive_replay_row")
            if not row.get("archive_path"):
                blockers.append("archive_path_missing_for_archive_replay_row")
            if not row.get("submission_dir"):
                blockers.append("submission_dir_missing_for_archive_replay_row")
            if not row.get("archive_bytes"):
                blockers.append("archive_bytes_missing_for_archive_replay_row")
            units.append(
                {
                    "unit_id": f"{family}_{row_id}_archive_replay_result",
                    "unit_type": "archive_ladder_replay_result",
                    "family": str(family),
                    "row_id": row_id,
                    "report_path": report.get("report_path"),
                    "status": row.get("status"),
                    "archive_bytes": row.get("archive_bytes"),
                    "archive_sha256": row.get("archive_sha256"),
                    "archive_path": row.get("archive_path"),
                    "submission_dir": row.get("submission_dir"),
                    "spine_manifest_path": row.get("spine_manifest_path"),
                    "receiver_proof_path": row.get("receiver_proof_path"),
                    "decoder_weight_waterfill_plan_path": row.get(
                        "decoder_weight_waterfill_plan_path"
                    ),
                    "replay_report_path": row.get("replay_report_path"),
                    "replay_report_sha256": row.get("replay_report_sha256"),
                    "receiver_proof_ready": bool(row.get("receiver_proof_ready")),
                    "archive_export_backend_counts": dict(
                        row.get("archive_export_backend_counts") or {}
                    ),
                    "target_consumers": [
                        "final_rate_attack",
                        "bit_allocator",
                        "sensitivity_map",
                        "cathedral_autopilot",
                    ],
                    "planner_action": (
                        "run_full_video_mlx_scorer_replay_for_archive_row"
                    ),
                    "blockers": _unique(blockers),
                    "predicted_delta_adjustment": 0.0,
                    **FALSE_AUTHORITY,
                }
            )
    for family, report in _mapping_items(
        control_inventory.get("decoder_weight_saliency_replays")
    ):
        blockers = _string_list(report.get("blockers"))
        if not bool(report.get("full_video_coverage")):
            blockers.append("full_video_decoder_weight_saliency_replay_missing")
        units.append(
            {
                "unit_id": f"{family}_decoder_weight_saliency_replay",
                "unit_type": "decoder_weight_saliency_replay",
                "family": family,
                "row_count": int(report.get("row_count", 0) or 0),
                "full_video_coverage": bool(report.get("full_video_coverage")),
                "saliency_group_count": int(
                    report.get("saliency_group_count", 0) or 0
                ),
                "target_consumers": [
                    "bit_allocator",
                    "sensitivity_map",
                    "final_rate_attack",
                    "cathedral_autopilot",
                ],
                "planner_action": "route_decoder_weight_saliency_to_waterfill",
                "blockers": _unique(blockers),
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    for family, report in _mapping_items(
        control_inventory.get("decoder_mode_assignment_reports")
    ):
        for index, row in enumerate(_mapping_list(report.get("assignment_rows"))):
            row_id = str(row.get("row_id") or f"mode_assignment_{index:04d}")
            modes = _mode_histogram_keys(row.get("mode_histogram"))
            blockers = _string_list(report.get("blockers")) + _string_list(
                row.get("blockers")
            )
            probe_command_argv = _string_list(row.get("probe_command_argv"))
            probe_receiver_packet_dir = row.get("probe_receiver_packet_dir")
            if bool(row.get("ready_for_local_advisory_probe")) and (
                not probe_command_argv or not probe_receiver_packet_dir
            ):
                blockers.append("decoder_mode_probe_command_missing")
            if not bool(row.get("ready_for_receiver_mode_export")):
                blockers.append("receiver_visible_decoder_mode_export_missing")
            units.append(
                {
                    "unit_id": f"{family}_{row_id}_decoder_mode_assignment",
                    "unit_type": "decoder_mode_assignment_route",
                    "family": family,
                    "row_id": row_id,
                    "decoder_payload_schema": row.get("decoder_payload_schema"),
                    "mode_plan_cli_arg": row.get("mode_plan_cli_arg"),
                    "mode_histogram": dict(row.get("mode_histogram") or {}),
                    "ready_for_local_advisory_probe": bool(
                        row.get("ready_for_local_advisory_probe")
                    ),
                    "ready_for_receiver_mode_export": bool(
                        row.get("ready_for_receiver_mode_export")
                    ),
                    "probe_command_axis_tag": row.get("probe_command_axis_tag"),
                    "probe_command_argv": probe_command_argv,
                    "probe_command_hint": row.get("probe_command_hint"),
                    "probe_receiver_packet_dir": probe_receiver_packet_dir,
                    "receiver_precision_modes": modes,
                    "target_consumers": [
                        "final_rate_attack",
                        "bit_allocator",
                        "probe_disambiguator",
                    ],
                    "planner_action": (
                        "compile_decoder_modes_to_receiver_visible_grammar"
                    ),
                    "blockers": _unique(blockers),
                    "predicted_delta_adjustment": 0.0,
                    **FALSE_AUTHORITY,
                }
            )
    for family, report in _mapping_items(
        control_inventory.get("decoder_mode_probe_reports")
    ):
        blockers = _string_list(report.get("blockers"))
        candidate_rows = _mapping_list(report.get("candidate_rows"))
        if not candidate_rows:
            candidate_rows = _mapping_list(report.get("candidates"))
        best_label = str(report.get("best_plan_label") or "")
        best_candidate = _first_matching_candidate(candidate_rows, best_label)
        candidate_count = int(
            report.get("candidate_count")
            or len(candidate_rows)
            or report.get("mode_plan_count")
            or 0
        )
        units.append(
            {
                "unit_id": f"{family}_{best_label or 'unknown'}_decoder_mode_probe",
                "unit_type": "decoder_mode_probe_result",
                "family": family,
                "best_plan_label": best_label,
                "best_plan_score_linf_advisory": report.get(
                    "best_plan_score_linf_advisory"
                ),
                "candidate_count": candidate_count,
                "best_candidate": dict(best_candidate or {}),
                "target_consumers": [
                    "final_rate_attack",
                    "bit_allocator",
                    "probe_disambiguator",
                    "cathedral_autopilot",
                ],
                "planner_action": (
                    "rerun_best_decoder_mode_plan_with_pair_robust_pose_gate"
                ),
                "blockers": _unique(blockers),
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    return units


def _memo_ref_units(implementation_sweep: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = _mapping_list(implementation_sweep.get("related_omx_design_memo_refs"))
    if not refs:
        return [
            {
                "unit_id": "omx_design_memo_reference_index",
                "unit_type": "design_memo_anchor_index",
                "memo_count": 0,
                "topic_counts": {},
                "target_consumers": [
                    "cathedral_autopilot",
                    "continual_learning_posterior",
                ],
                "planner_action": "repair_missing_omx_design_memo_index",
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        ]
    topic_counts: dict[str, int] = {}
    for ref in refs:
        for topic in _string_list(ref.get("topics")):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    return [
        {
            "unit_id": "omx_design_memo_reference_index",
            "unit_type": "design_memo_anchor_index",
            "memo_count": len(refs),
            "topic_counts": dict(sorted(topic_counts.items())),
            "target_consumers": [
                "cathedral_autopilot",
                "continual_learning_posterior",
                "probe_disambiguator",
            ],
            "planner_action": "consume_memo_refs_before_stack_implementation_changes",
            "predicted_delta_adjustment": 0.0,
            **FALSE_AUTHORITY,
        }
    ]


def _modelsize_units(modelsize_curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for index, row in enumerate(_mapping_list(modelsize_curve.get("curve_rows"))):
        budget = row.get("solved_budget") if isinstance(row.get("solved_budget"), Mapping) else {}
        family = str(budget.get("family") or "unknown")
        cap = int(row.get("target_archive_byte_cap") or 0)
        ideal = budget.get("ideal_quant_payload") if isinstance(budget.get("ideal_quant_payload"), Mapping) else {}
        units.append(
            {
                "unit_id": f"modelsize_{family}_{cap}_byte_cap_{index}",
                "unit_type": "modelsize_byte_cap_lower_bound",
                "family": family,
                "target_archive_byte_cap": cap,
                "modelsize_mparams": budget.get("modelsize_mparams"),
                "ideal_packed_payload_bytes": ideal.get("ideal_packed_payload_bytes"),
                "target_consumers": ["final_rate_attack", "bit_allocator"],
                "planner_action": "measure_trained_receiver_archive_bytes",
                "predicted_delta_adjustment": 0.0,
                **FALSE_AUTHORITY,
            }
        )
    return units


def _target_consumers_for_hook(hook_id: str) -> list[str]:
    if "mlx" in hook_id or "auth" in hook_id:
        return ["cathedral_autopilot", "probe_disambiguator"]
    if "xray" in hook_id or "master_gradient" in hook_id or "sensitivity" in hook_id:
        return ["sensitivity_map", "cathedral_autopilot", "probe_disambiguator"]
    if "bit" in hook_id or "rate" in hook_id or "archive" in hook_id:
        return ["final_rate_attack", "bit_allocator", "cathedral_autopilot"]
    if "inverse" in hook_id or "steg" in hook_id:
        return ["probe_disambiguator", "sensitivity_map", "final_rate_attack"]
    return ["cathedral_autopilot"]


def _surface_ref(surface_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    binding_gap_count = len(_mapping_list(payload.get("binding_gap_rows")))
    nested_sweep = payload.get("implementation_sweep")
    nested_blockers = (
        _string_list(nested_sweep.get("blockers"))
        if isinstance(nested_sweep, Mapping)
        else []
    )
    return {
        "surface_id": surface_id,
        "present": True,
        "schema": payload.get("schema"),
        "axis_tag": payload.get("axis_tag"),
        "verdict": payload.get("verdict") or payload.get("go_no_go_verdict"),
        "blocker_count": (
            len(_string_list(payload.get("blockers")))
            + binding_gap_count
            + len(nested_blockers)
        ),
        "binding_gap_count": binding_gap_count,
        "score_claim": bool(payload.get("score_claim", False)),
        "promotion_eligible": bool(payload.get("promotion_eligible", False)),
        "ready_for_exact_eval_dispatch": bool(
            payload.get("ready_for_exact_eval_dispatch", False)
        ),
    }


def _control_inventory_blockers(control_inventory: Mapping[str, Any]) -> list[str]:
    blockers = []
    for row in _mapping_list(control_inventory.get("binding_gap_rows")):
        gap_id = str(row.get("gap_id") or row.get("control_id") or "unknown")
        blockers.append(f"nerv_control_binding_gap:{gap_id}")
    nested = control_inventory.get("implementation_sweep")
    if isinstance(nested, Mapping):
        blockers.extend(_string_list(nested.get("blockers")))
    return blockers


def _require_mapping_schema(
    payload: Mapping[str, Any],
    expected_schema: str,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise NervMasterConsumerBridgeError(f"{name} must be a mapping")
    if payload.get("schema") != expected_schema:
        raise NervMasterConsumerBridgeError(
            f"{name} schema must be {expected_schema}, got {payload.get('schema')}"
        )


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _mapping_items(value: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if not isinstance(value, Mapping):
        return []
    return [
        (str(key), item)
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        if isinstance(item, Mapping)
    ]


def _mode_histogram_keys(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    return _unique([str(key) for key in value if str(key)])


def _first_matching_candidate(
    candidates: Sequence[Mapping[str, Any]],
    label: str,
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("label") or "") == label:
            return candidate
    return candidates[0] if candidates else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "AXIS_TAG",
    "MASTER_CONSUMER_ROUTES",
    "SCHEMA",
    "NervMasterConsumerBridgeError",
    "build_nerv_master_consumer_bridge",
]
