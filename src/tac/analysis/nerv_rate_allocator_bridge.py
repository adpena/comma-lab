# SPDX-License-Identifier: MIT
"""Rate/bit-allocation work orders from the NeRV master-consumer bridge.

This module is deliberately no-authority. It turns the normalized
``nerv_master_consumer_bridge.v1`` packet into queue-ready work orders for
final-rate attack, bit allocators, and sensitivity consumers without inventing
another NeRV control inventory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY as HPRC_FALSE_AUTHORITY,
)

SCHEMA = "nerv_rate_allocator_bridge.v1"
AXIS_TAG = "[planning/control]"
FALSE_AUTHORITY = {
    **HPRC_FALSE_AUTHORITY,
    "frontier_score_claim": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
}

RECEIVER_PRECISION_MODES = (
    {
        "mode": "fp16_protected",
        "intended_atoms": "PoseNet/SegNet critical decoder weights or scales",
        "requires": ["measured_atom_sensitivity", "receiver_fp16_grammar"],
    },
    {
        "mode": "int8_protected",
        "intended_atoms": "high-leverage but brotli-friendly decoder groups",
        "requires": ["measured_atom_sensitivity", "receiver_int8_grammar"],
    },
    {
        "mode": "int4",
        "intended_atoms": "ordinary decoder groups after QAT",
        "requires": ["qat_roundtrip", "receiver_int4_grammar"],
    },
    {
        "mode": "int2",
        "intended_atoms": "low-sensitivity decoder groups",
        "requires": ["qat_roundtrip", "receiver_int2_grammar"],
    },
    {
        "mode": "zero",
        "intended_atoms": "measured dead or prunable decoder groups",
        "requires": ["zero_mask_receiver_grammar", "post_prune_scorer_replay"],
    },
    {
        "mode": "rle_only",
        "intended_atoms": "long zero/constant masks and sparse selectors",
        "requires": ["run_length_receiver_grammar", "charged_header_accounting"],
    },
)


class NervRateAllocatorBridgeError(ValueError):
    """Raised when a rate/allocator bridge payload cannot be built."""


def build_nerv_rate_allocator_bridge(
    *,
    master_bridge: Mapping[str, Any],
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build no-authority final-rate/bit-allocator work orders."""

    if not isinstance(master_bridge, Mapping):
        raise NervRateAllocatorBridgeError("master_bridge must be a mapping")
    if master_bridge.get("schema") != "nerv_master_consumer_bridge.v1":
        raise NervRateAllocatorBridgeError(
            "master_bridge schema must be nerv_master_consumer_bridge.v1, "
            f"got {master_bridge.get('schema')}"
        )

    generated = generated_utc or datetime.now(UTC).isoformat()
    source_blockers = _string_list(master_bridge.get("blockers"))
    units = _mapping_list(master_bridge.get("master_consumer_units"))
    work_orders = _dedupe_work_orders(
        [
            *_modelsize_work_orders(units),
            *_control_row_work_orders(units),
            *_evidence_work_orders(units),
            *_implementation_gate_work_orders(units),
        ]
    )
    blocking_work_orders = [
        order["work_order_id"] for order in work_orders if order["blockers"]
    ]

    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "verdict": (
            "GO_RATE_ALLOCATOR_ROUTING__NO_GO_SCORE_PROMOTION_OR_EXACT_DISPATCH"
        ),
        "source_schema": master_bridge.get("schema"),
        "source_candidate_id": _nested_candidate_id(master_bridge),
        "baseline_to_beat": master_bridge.get("baseline_to_beat"),
        "top_priority_carriers": list(master_bridge.get("top_priority_carriers") or []),
        "receiver_precision_mode_policy": list(RECEIVER_PRECISION_MODES),
        "rate_allocator_work_orders": work_orders,
        "work_order_count": len(work_orders),
        "blocking_work_order_count": len(blocking_work_orders),
        "blocking_work_order_ids": blocking_work_orders,
        "final_rate_attack_ingest_contract": {
            "consumer": "final_rate_attack",
            "allowed_use": "local_planning_and_receiver_grammar_work_order_selection",
            "forbidden_use": "score_claim_or_rank_or_exact_dispatch_authority",
            "work_order_fields": [
                "work_order_id",
                "work_order_type",
                "target_consumers",
                "planner_action",
                "receiver_precision_modes",
                "blockers",
            ],
        },
        "bit_allocator_ingest_contract": {
            "consumer": "bit_allocator",
            "allowed_use": "allocate_measurement_and_receiver_grammar_followups",
            "forbidden_use": "assign_real_bits_without_measured_sensitivity_and_receiver_proof",
            "requires_before_real_bit_assignment": [
                "per_atom_or_per_group_sensitivity",
                "receiver_decoded_byte_accounting",
                "scorer_replay_under_axis_label",
            ],
        },
        "blockers": _dedupe_strings(
            [
                *source_blockers,
                "rate_allocator_bridge_is_false_authority",
                "real_bit_assignment_requires_measured_sensitivity_and_receiver_proof",
            ]
        ),
        **FALSE_AUTHORITY,
    }


def _modelsize_work_orders(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    orders = []
    for unit in units:
        if unit.get("unit_type") != "modelsize_byte_cap_lower_bound":
            continue
        family = str(unit.get("family") or "unknown")
        cap = int(unit.get("target_archive_byte_cap") or 0)
        unit_blockers = _string_list(unit.get("blockers"))
        orders.append(
            _work_order(
                work_order_id=f"measure_{family}_{cap}_modelsize_archive_ladder",
                work_order_type="measured_modelsize_budget_ladder",
                target_consumers=["final_rate_attack", "bit_allocator"],
                planner_action="measure_trained_receiver_archive_bytes",
                source_unit_id=str(unit.get("unit_id") or ""),
                priority=20 if family in {"snerv", "hinerv", "hnerv"} else 50,
                receiver_precision_modes=["int8_protected", "int4", "int2", "zero"],
                rationale=(
                    "convert source-grounded modelsize lower-bound planning into "
                    "measured archive bytes, entropy bytes, metadata bytes, and "
                    "non-rate component deltas"
                ),
                payload={
                    "family": family,
                    "target_archive_byte_cap": cap,
                    "modelsize_mparams": unit.get("modelsize_mparams"),
                    "ideal_packed_payload_bytes": unit.get(
                        "ideal_packed_payload_bytes"
                    ),
                },
                blockers=[
                    *unit_blockers,
                    "trained_receiver_archive_bytes_missing",
                    "nonrate_component_deltas_missing",
                ],
            )
        )
    return orders


def _control_row_work_orders(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    orders = []
    for unit in units:
        if unit.get("unit_type") not in {
            "nerv_control_row_route",
            "local_binding_surface_route",
        }:
            continue
        control_id = str(unit.get("control_id") or unit.get("surface_id") or "unknown")
        target_consumers = _string_list(unit.get("target_consumers"))
        if not _is_rate_relevant(control_id, target_consumers):
            continue
        missing = _string_list(unit.get("missing_bindings"))
        modes = _precision_modes_for_control(control_id)
        orders.append(
            _work_order(
                work_order_id=f"route_{control_id}_to_rate_allocator",
                work_order_type="rate_allocator_control_binding",
                target_consumers=_dedupe_strings(
                    [*target_consumers, "final_rate_attack", "bit_allocator"]
                ),
                planner_action=str(
                    unit.get("planner_action")
                    or "reuse_existing_control_before_new_code"
                ),
                source_unit_id=str(unit.get("unit_id") or ""),
                priority=_priority_for_control(control_id),
                receiver_precision_modes=modes,
                rationale=(
                    "reuse the canonical NeRV control inventory route before "
                    "creating new rate/allocator glue"
                ),
                payload={
                    "control_id": control_id,
                    "applies_to": unit.get("applies_to"),
                    "binding_status": unit.get("binding_status"),
                    "path_count": unit.get("path_count"),
                    "paths": unit.get("paths") or [],
                },
                blockers=missing,
            )
        )
    return orders


def _implementation_gate_work_orders(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    orders = []
    for unit in units:
        if unit.get("unit_type") != "implementation_design_gate":
            continue
        stack_id = str(unit.get("stack_id") or "unknown")
        blockers = _string_list(unit.get("blockers"))
        rate_blockers = [
            blocker
            for blocker in blockers
            if any(
                token in blocker
                for token in (
                    "receiver",
                    "byte",
                    "mixed_precision",
                    "full600",
                    "CPU_CUDA",
                    "source_faithful",
                )
            )
        ]
        if not rate_blockers:
            continue
        orders.append(
            _work_order(
                work_order_id=f"close_{stack_id}_receiver_rate_promotion_gates",
                work_order_type="receiver_rate_promotion_gate",
                target_consumers=[
                    "final_rate_attack",
                    "bit_allocator",
                    "cathedral_autopilot",
                ],
                planner_action="close_receiver_byte_and_paired_axis_gates",
                source_unit_id=str(unit.get("unit_id") or ""),
                priority=10,
                receiver_precision_modes=[
                    "fp16_protected",
                    "int8_protected",
                    "int4",
                    "int2",
                    "zero",
                    "rle_only",
                ],
                rationale=(
                    "promotion remains blocked until the carrier has a "
                    "source-faithful receiver, byte accounting, and paired axes"
                ),
                payload={"stack_id": stack_id},
                blockers=rate_blockers,
            )
        )
    return orders


def _evidence_work_orders(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for unit in units:
        unit_type = str(unit.get("unit_type") or "")
        family = str(unit.get("family") or "unknown")
        blockers = _string_list(unit.get("blockers"))
        if unit_type == "decoder_weight_saliency_replay":
            orders.append(
                _work_order(
                    work_order_id=(
                        f"bind_{family}_decoder_weight_saliency_to_waterfill"
                    ),
                    work_order_type="decoder_weight_saliency_allocator_binding",
                    target_consumers=[
                        "bit_allocator",
                        "sensitivity_map",
                        "final_rate_attack",
                    ],
                    planner_action="route_decoder_weight_saliency_to_waterfill",
                    source_unit_id=str(unit.get("unit_id") or ""),
                    priority=15 if blockers else 8,
                    receiver_precision_modes=[
                        "fp16_protected",
                        "int8_protected",
                        "int4",
                        "int2",
                        "zero",
                    ],
                    rationale=(
                        "feed measured decoder-weight saliency into waterfill "
                        "before assigning protected or low-bit atoms"
                    ),
                    payload={
                        "family": family,
                        "row_count": unit.get("row_count"),
                        "full_video_coverage": unit.get("full_video_coverage"),
                        "saliency_group_count": unit.get("saliency_group_count"),
                    },
                    blockers=blockers,
                )
            )
            continue
        if unit_type == "decoder_mode_assignment_route":
            row_id = str(unit.get("row_id") or "unknown")
            modes = _receiver_precision_modes_from_unit(unit)
            orders.append(
                _work_order(
                    work_order_id=(
                        f"compile_{family}_{row_id}_decoder_modes_to_receiver"
                    ),
                    work_order_type="receiver_visible_decoder_mode_assignment",
                    target_consumers=[
                        "final_rate_attack",
                        "bit_allocator",
                        "probe_disambiguator",
                    ],
                    planner_action=(
                        "emit_receiver_visible_mixed_precision_decoder_grammar"
                    ),
                    source_unit_id=str(unit.get("unit_id") or ""),
                    priority=8,
                    receiver_precision_modes=modes,
                    rationale=(
                        "turn waterfilled decoder atom mode choices into the "
                        "receiver-visible mixed-precision grammar before any "
                        "rate claim"
                    ),
                    payload={
                        "family": family,
                        "row_id": row_id,
                        "decoder_payload_schema": unit.get(
                            "decoder_payload_schema"
                        ),
                        "mode_plan_cli_arg": unit.get("mode_plan_cli_arg"),
                        "mode_histogram": unit.get("mode_histogram") or {},
                        "ready_for_local_advisory_probe": unit.get(
                            "ready_for_local_advisory_probe"
                        ),
                        "ready_for_receiver_mode_export": unit.get(
                            "ready_for_receiver_mode_export"
                        ),
                        "probe_command_axis_tag": unit.get(
                            "probe_command_axis_tag"
                        ),
                        "probe_command_argv": _string_list(
                            unit.get("probe_command_argv")
                        ),
                        "probe_command_hint": unit.get("probe_command_hint"),
                        "probe_receiver_packet_dir": unit.get(
                            "probe_receiver_packet_dir"
                        ),
                    },
                    blockers=[
                        *blockers,
                        "receiver_decoded_byte_accounting_required",
                        "full600_byte_closed_receiver_proof_missing",
                        "paired_contest_cpu_cuda_auth_eval_missing",
                    ],
                )
            )
            continue
        if unit_type == "decoder_mode_probe_result":
            label = str(unit.get("best_plan_label") or "unknown")
            best_candidate = (
                unit.get("best_candidate")
                if isinstance(unit.get("best_candidate"), Mapping)
                else {}
            )
            modes = _receiver_precision_modes_from_unit(best_candidate or unit)
            orders.append(
                _work_order(
                    work_order_id=(
                        f"replay_{family}_{label}_decoder_mode_plan_pair_robust"
                    ),
                    work_order_type="decoder_mode_pair_robust_probe_followup",
                    target_consumers=[
                        "final_rate_attack",
                        "bit_allocator",
                        "probe_disambiguator",
                        "cathedral_autopilot",
                    ],
                    planner_action=(
                        "rerun_best_decoder_mode_plan_with_stratified_pairs_pose_guard"
                    ),
                    source_unit_id=str(unit.get("unit_id") or ""),
                    priority=12,
                    receiver_precision_modes=modes,
                    rationale=(
                        "local advisory mode-probe wins must survive pair-robust "
                        "PoseNet-guarded replay before driving training or export"
                    ),
                    payload={
                        "family": family,
                        "best_plan_label": label,
                        "best_plan_score_linf_advisory": unit.get(
                            "best_plan_score_linf_advisory"
                        ),
                        "candidate_count": unit.get("candidate_count"),
                        "receiver_archive_packet_path": best_candidate.get(
                            "receiver_archive_packet_path"
                        ),
                        "receiver_archive_packet_bytes": best_candidate.get(
                            "receiver_archive_packet_bytes"
                        ),
                        "receiver_archive_packet_sha256": best_candidate.get(
                            "receiver_archive_packet_sha256"
                        ),
                        "receiver_archive_replay_verified": best_candidate.get(
                            "receiver_archive_replay_verified"
                        )
                        is True,
                        "receiver_archive_packet_is_contest_archive_zip": (
                            best_candidate.get(
                                "receiver_archive_packet_is_contest_archive_zip"
                            )
                            is True
                        ),
                    },
                    blockers=[
                        *blockers,
                        "stratified_pair_pose_guard_replay_missing",
                        "full600_byte_closed_receiver_proof_missing",
                        "paired_contest_cpu_cuda_auth_eval_missing",
                    ],
                )
            )
    return orders


def _work_order(
    *,
    work_order_id: str,
    work_order_type: str,
    target_consumers: Sequence[str],
    planner_action: str,
    source_unit_id: str,
    priority: int,
    receiver_precision_modes: Sequence[str],
    rationale: str,
    payload: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    return {
        "work_order_id": work_order_id,
        "work_order_type": work_order_type,
        "target_consumers": _dedupe_strings(target_consumers),
        "planner_action": planner_action,
        "source_unit_id": source_unit_id,
        "priority": int(priority),
        "receiver_precision_modes": list(receiver_precision_modes),
        "rationale": rationale,
        "payload": dict(payload),
        "blockers": _dedupe_strings(blockers),
        "score_claim": False,
        "score_claim_valid": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "predicted_delta_adjustment": 0.0,
    }


def _precision_modes_for_control(control_id: str) -> list[str]:
    lowered = control_id.lower()
    modes: list[str] = []
    if any(token in lowered for token in ("saliency", "master_gradient", "vjp")):
        modes.extend(["fp16_protected", "int8_protected", "int4", "int2", "zero"])
    if any(token in lowered for token in ("bitmask", "zero", "packing", "rle")):
        modes.extend(["zero", "rle_only", "int2", "int4"])
    if any(token in lowered for token in ("bitstream", "quant", "modelsize")):
        modes.extend(["int8_protected", "int4", "int2", "zero"])
    if any(token in lowered for token in ("frequency", "lf", "hfr", "wavelet")):
        modes.extend(["fp16_protected", "int8_protected", "int4", "int2", "zero"])
    if any(token in lowered for token in ("sr", "resolution")):
        modes.extend(["zero", "rle_only", "int2"])
    return _dedupe_strings(modes or ["int8_protected", "int4", "int2", "zero"])


def _receiver_precision_modes_from_unit(unit: Mapping[str, Any]) -> list[str]:
    modes = _string_list(unit.get("receiver_precision_modes"))
    histogram = unit.get("mode_histogram")
    if isinstance(histogram, Mapping):
        modes.extend(str(mode) for mode in histogram if str(mode))
    modes.extend(_string_list(unit.get("modes")))
    normalized = [
        _receiver_precision_mode_from_raw(mode)
        for mode in modes
        if _receiver_precision_mode_from_raw(mode)
    ]
    return _dedupe_strings(
        normalized or ["fp16_protected", "int8_protected", "int4", "int2", "zero"]
    )


def _receiver_precision_mode_from_raw(mode: str) -> str:
    lowered = str(mode).strip().lower()
    if lowered in {"fp16", "float16", "half", "fp16_protected"}:
        return "fp16_protected"
    if lowered in {"int8", "i8", "uint8", "int8_protected"}:
        return "int8_protected"
    if lowered in {"int4", "i4", "uint4"}:
        return "int4"
    if lowered in {"int2", "i2", "uint2"}:
        return "int2"
    if lowered in {"zero", "zeros", "pruned"}:
        return "zero"
    if lowered in {"rle", "rle_only", "run_length"}:
        return "rle_only"
    return ""


def _priority_for_control(control_id: str) -> int:
    lowered = control_id.lower()
    if any(token in lowered for token in ("receiver", "bitmask", "zero", "bitstream")):
        return 10
    if any(token in lowered for token in ("modelsize", "saliency", "master_gradient")):
        return 20
    if any(token in lowered for token in ("sr", "resolution", "frequency")):
        return 30
    return 60


def _is_rate_relevant(control_id: str, target_consumers: Sequence[str]) -> bool:
    lowered = control_id.lower()
    if any(
        consumer in {"final_rate_attack", "bit_allocator"}
        for consumer in target_consumers
    ):
        return True
    return any(
        token in lowered
        for token in (
            "bit",
            "byte",
            "rate",
            "archive",
            "modelsize",
            "quant",
            "zero",
            "packing",
            "rle",
            "saliency",
            "master_gradient",
            "vjp",
            "frequency",
            "lf",
            "sr",
            "receiver",
        )
    )


def _nested_candidate_id(master_bridge: Mapping[str, Any]) -> str | None:
    candidate = master_bridge.get("normalized_candidate")
    if isinstance(candidate, Mapping):
        value = candidate.get("candidate_id")
        return str(value) if value else None
    return None


def _dedupe_work_orders(orders: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for order in sorted(
        orders,
        key=lambda row: (
            int(row.get("priority") or 999),
            str(row.get("work_order_id") or ""),
        ),
    ):
        work_order_id = str(order.get("work_order_id") or "")
        if work_order_id in seen:
            continue
        seen.add(work_order_id)
        out.append(dict(order))
    return out


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "AXIS_TAG",
    "FALSE_AUTHORITY",
    "RECEIVER_PRECISION_MODES",
    "SCHEMA",
    "NervRateAllocatorBridgeError",
    "build_nerv_rate_allocator_bridge",
]
