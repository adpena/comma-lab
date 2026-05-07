"""Planning manifests for HNeRV HDC2 combined entropy-overhead reduction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_hdc2_combined_entropy"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS = [
    "hdc2_runtime_decoder_contract_with_inflate_consumer",
    "hdc2_archive_candidate_manifest_with_decoder_stream_consumed",
]


class Hdc2CombinedEntropyError(ValueError):
    """Raised when HDC2 combined entropy inputs are malformed."""


def build_hdc2_combined_entropy_manifest(
    frontier_ranking: Mapping[str, Any],
    entropy_audit: Mapping[str, Any],
    hdc2_work_product: Mapping[str, Any],
    *,
    active_floor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed byte-accounted plan for the HDC2 section recode."""

    frontier = _mapping(frontier_ranking.get("current_frontier"), "current_frontier")
    entropy_action = _mapping(
        frontier_ranking.get("next_entropy_research_action"),
        "next_entropy_research_action",
    )
    target_label = _nonempty_str(entropy_action.get("target_label"), "target_label")
    target_section = _nonempty_str(entropy_action.get("target_section"), "target_section")
    group = _matching_combined_group(frontier_ranking, target_label, target_section)
    model_target = _matching_entropy_target(
        entropy_audit,
        target_label=target_label,
        target_kind="known_model_overhead",
    )
    payload_target = _matching_entropy_target(
        entropy_audit,
        target_label=target_label,
        target_kind="known_payload_entropy_gap",
    )
    source_stream = _mapping(
        _mapping(hdc2_work_product.get("source_stream_section_manifest"), "source_stream_section_manifest").get("stream"),
        "source_stream_section_manifest.stream",
    )
    roundtrip = _mapping(
        hdc2_work_product.get("roundtrip_decode_validation_manifest"),
        "roundtrip_decode_validation_manifest",
    )
    decoded_equiv = _mapping(
        hdc2_work_product.get("decoded_output_equivalence_report"),
        "decoded_output_equivalence_report",
    )

    frontier_section_bytes = _positive_int(group.get("frontier_section_bytes"), "frontier_section_bytes")
    actual_replacement_bytes = _positive_int(group.get("actual_replacement_bytes"), "actual_replacement_bytes")
    frontier_archive_bytes = _positive_int(frontier.get("archive_bytes"), "frontier_archive_bytes")
    model_overhead_bytes = _positive_int(model_target.get("target_bytes"), "model_overhead_target_bytes")
    payload_gap_bytes = _positive_int(payload_target.get("target_bytes"), "payload_gap_target_bytes")
    known_target_bytes = model_overhead_bytes + payload_gap_bytes
    projected_bytes_after_model_only = actual_replacement_bytes - model_overhead_bytes
    projected_bytes_after_combined = actual_replacement_bytes - known_target_bytes
    net_after_model_only = projected_bytes_after_model_only - frontier_section_bytes
    net_after_combined = projected_bytes_after_combined - frontier_section_bytes
    bounded_candidate = _best_bounded_candidate(
        hdc2_work_product,
        actual_replacement_bytes=actual_replacement_bytes,
        frontier_section_bytes=frontier_section_bytes,
        model_overhead_bytes=model_overhead_bytes,
    )

    roundtrip_ready = (
        roundtrip.get("roundtrip_valid") is True
        and roundtrip.get("raw_equal") is True
        and roundtrip.get("q_roundtrip_equal") is True
        and roundtrip.get("scale_roundtrip_equal") is True
        and decoded_equiv.get("old_new_sha256_equal") is True
        and decoded_equiv.get("decoded_output_equal") is True
    )
    blockers = []
    if not roundtrip_ready:
        blockers.append("hdc2_stream_roundtrip_or_decoded_equivalence_not_closed")
    if net_after_model_only <= 0:
        blockers.append("unexpected_model_overhead_alone_rate_positive_recheck_accounting")
    if net_after_combined >= 0:
        blockers.append("combined_known_targets_not_rate_positive")

    byte_accounting = {
        "actual_hdc2_replacement_bytes": actual_replacement_bytes,
        "current_frontier_section_bytes": frontier_section_bytes,
        "net_byte_delta_now": actual_replacement_bytes - frontier_section_bytes,
        "model_overhead_target_bytes": model_overhead_bytes,
        "payload_entropy_gap_target_bytes": payload_gap_bytes,
        "combined_known_target_bytes": known_target_bytes,
        "projected_bytes_after_model_overhead_only": projected_bytes_after_model_only,
        "net_byte_delta_after_model_overhead_only": net_after_model_only,
        "projected_bytes_after_combined_targets": projected_bytes_after_combined,
        "net_byte_delta_after_combined_targets": net_after_combined,
        "projected_rate_score_delta_after_combined_targets": (
            net_after_combined * RATE_SCORE_PER_BYTE
        ),
        "minimum_payload_reduction_needed_after_zero_model_overhead_bytes": (
            max(0, net_after_model_only + 1)
        ),
    }
    if bounded_candidate is not None:
        remaining_after_bounded = bounded_candidate[
            "remaining_reduction_to_beat_frontier_section_bytes"
        ]
        if bounded_candidate["archive_build_gate"] is not True:
            blockers.append("bounded_candidate_archive_build_gate_not_passed")
        bounded_break_even = {
            "current_best_variant": bounded_candidate["variant"],
            "current_best_bytes": bounded_candidate["bytes"],
            "frontier_section_bytes": frontier_section_bytes,
            "remaining_reduction_to_beat_frontier_section_bytes": remaining_after_bounded,
            "required_next_reduction_sources": [
                "reduce_range_payload_bytes_with_better_context_or_escape_policy",
                "reduce_raw_payload_bytes_with_secondary_entropy_code_or_run_model",
                "elide_static_context_header_bytes_without_reintroducing_self_describing_schema",
            ],
            "archive_build_gate": bounded_candidate["archive_build_gate"],
            "archive_build_gate_rule": (
                "candidate_stream_bytes_must_be_less_than_current_frontier_section_bytes_"
                "and_raw_equality_closed"
            ),
            "archive_build_gate_blockers": list(bounded_candidate["archive_build_gate_blockers"]),
        }
        byte_accounting.update(
            {
                "best_bounded_candidate_bytes": bounded_candidate["bytes"],
                "bounded_candidate_reduction_vs_hdc2_bytes": bounded_candidate[
                    "byte_reduction_vs_hdc2_bytes"
                ],
                "net_byte_delta_after_best_bounded_candidate": bounded_candidate[
                    "net_byte_delta_vs_frontier_section"
                ],
                "remaining_reduction_to_beat_frontier_after_best_bounded_candidate_bytes": remaining_after_bounded,
                "best_bounded_candidate_archive_build_gate": bounded_candidate[
                    "archive_build_gate"
                ],
            }
        )
    else:
        bounded_break_even = None
    active_floor_gate = _active_floor_gate(
        active_floor,
        frontier_archive_bytes=frontier_archive_bytes,
        direct_hdc2_archive_delta=actual_replacement_bytes - frontier_section_bytes,
        bounded_candidate=bounded_candidate,
    )
    if (
        isinstance(active_floor_gate, Mapping)
        and active_floor_gate.get("best_bounded_candidate_below_active_floor") is False
    ):
        blockers.append("best_bounded_candidate_not_below_active_archive_floor")

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": [
            "combined_entropy_manifest_is_planning_only",
            "hdc2_direct_archive_runtime_contract_missing",
            "requires_actual_decoder_runtime_implementation",
            "requires_candidate_archive_manifest",
            "requires_strict_pre_submission_compliance",
            "requires_lane_dispatch_claim_before_gpu",
            "requires_exact_cuda_auth_eval",
        ],
        "blockers": blockers,
        "target": {
            "label": target_label,
            "section": target_section,
            "frontier_archive_sha256": frontier.get("archive_sha256"),
            "frontier_archive_bytes": frontier_archive_bytes,
            "frontier_score": frontier.get("score"),
            "frontier_section_bytes": frontier_section_bytes,
            "source_stream_sha256": source_stream.get("sha256"),
            "source_stream_decoded_raw_sha256": source_stream.get("decoded_raw_sha256"),
        },
        "byte_accounting": byte_accounting,
        "bounded_candidate": bounded_candidate,
        "bounded_candidate_break_even_plan": bounded_break_even,
        "active_floor_gate": active_floor_gate,
        "hdc2_direct_archive_runtime_gate": {
            "status": "fail_closed",
            "score_claim": False,
            "dispatch_attempted": False,
            "ready_for_archive_preflight": False,
            "ready_for_exact_eval_dispatch": False,
            "candidate_variant": "range_prev_symbol_global_q_streams_plus_raw_scales",
            "candidate_stream_bytes": actual_replacement_bytes,
            "projected_archive_bytes_if_direct_hdc2_replaced_section": (
                frontier_archive_bytes + actual_replacement_bytes - frontier_section_bytes
            ),
            "missing_artifacts": list(HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS),
            "blockers": [
                "hdc2_runtime_decoder_contract_with_inflate_consumer_missing",
                "hdc2_archive_candidate_manifest_with_decoder_stream_consumed_missing",
            ],
        },
        "closed_stream_evidence": {
            "roundtrip_valid": roundtrip.get("roundtrip_valid") is True,
            "raw_equal": roundtrip.get("raw_equal") is True,
            "q_roundtrip_equal": roundtrip.get("q_roundtrip_equal") is True,
            "scale_roundtrip_equal": roundtrip.get("scale_roundtrip_equal") is True,
            "decoded_output_equal": decoded_equiv.get("decoded_output_equal") is True,
            "old_new_sha256_equal": decoded_equiv.get("old_new_sha256_equal") is True,
            "candidate_stream_sha256": roundtrip.get("candidate_stream_sha256"),
            "candidate_stream_bytes": roundtrip.get("candidate_stream_bytes"),
        },
        "next_required_artifacts": [
            "actual_static_model_context_elision_or_shared_codebook_implementation",
            "actual_payload_entropy_gap_reduction_implementation",
            *HDC2_DIRECT_ARCHIVE_RUNTIME_REQUIREMENTS,
            "candidate_archive_manifest_with_member_sha256s",
            "runtime_tree_parity_manifest",
            "strict_pre_submission_compliance_json",
            "meta_lagrangian_atom_json_with_byte_delta_and_interaction_assumptions",
        ],
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render the combined entropy manifest as compact markdown."""

    target = _mapping(manifest.get("target"), "target")
    accounting = _mapping(manifest.get("byte_accounting"), "byte_accounting")
    evidence = _mapping(manifest.get("closed_stream_evidence"), "closed_stream_evidence")
    candidate = manifest.get("bounded_candidate")
    lines = [
        "# HDC2 Combined Entropy Reduction Manifest",
        "",
        f"- planning_only: `{_bool_text(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool_text(manifest.get('dispatch_attempted') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        "",
        "## Target",
        "",
        f"- label: `{target.get('label')}`",
        f"- section: `{target.get('section')}`",
        f"- frontier_archive_sha256: `{target.get('frontier_archive_sha256')}`",
        f"- frontier_section_bytes: `{target.get('frontier_section_bytes')}`",
        "",
        "## Byte Accounting",
        "",
        f"- actual_hdc2_replacement_bytes: `{accounting.get('actual_hdc2_replacement_bytes')}`",
        f"- net_byte_delta_now: `{accounting.get('net_byte_delta_now')}`",
        f"- model_overhead_target_bytes: `{accounting.get('model_overhead_target_bytes')}`",
        f"- payload_entropy_gap_target_bytes: `{accounting.get('payload_entropy_gap_target_bytes')}`",
        f"- net_byte_delta_after_model_overhead_only: `{accounting.get('net_byte_delta_after_model_overhead_only')}`",
        f"- net_byte_delta_after_combined_targets: `{accounting.get('net_byte_delta_after_combined_targets')}`",
        f"- projected_rate_score_delta_after_combined_targets: `{accounting.get('projected_rate_score_delta_after_combined_targets')}`",
        "",
    ]
    if isinstance(candidate, Mapping):
        break_even = manifest.get("bounded_candidate_break_even_plan")
        lines.extend(
            [
                "## Bounded Candidate",
                "",
                f"- variant: `{candidate.get('variant')}`",
                f"- bytes: `{candidate.get('bytes')}`",
                f"- byte_reduction_vs_hdc2: `{candidate.get('byte_reduction_vs_hdc2_bytes')}`",
                f"- net_byte_delta_vs_frontier_section: `{candidate.get('net_byte_delta_vs_frontier_section')}`",
                f"- static_context_header_reduction_vs_hdc2: `{candidate.get('static_context_header_reduction_vs_hdc2_bytes')}`",
                f"- payload_delta_vs_hdc2: `{candidate.get('payload_delta_vs_hdc2_bytes')}`",
                f"- archive_build_gate: `{_bool_text(candidate.get('archive_build_gate') is True)}`",
                f"- ready_for_exact_eval_dispatch: `{_bool_text(candidate.get('ready_for_exact_eval_dispatch') is True)}`",
                "",
            ]
        )
        if isinstance(break_even, Mapping):
            lines.extend(
                [
                    "## Break-Even Gate",
                    "",
                    f"- remaining_reduction_to_beat_frontier_section_bytes: `{break_even.get('remaining_reduction_to_beat_frontier_section_bytes')}`",
                    f"- archive_build_gate: `{_bool_text(break_even.get('archive_build_gate') is True)}`",
                    f"- archive_build_gate_rule: `{break_even.get('archive_build_gate_rule')}`",
                    "",
                ]
            )
    active_floor = manifest.get("active_floor_gate")
    if isinstance(active_floor, Mapping):
        lines.extend(
            [
                "## Active Archive Floor Gate",
                "",
                f"- active_floor_label: `{active_floor.get('active_floor_label')}`",
                f"- active_floor_archive_bytes: `{active_floor.get('active_floor_archive_bytes')}`",
                f"- direct_hdc2_projected_archive_bytes: `{active_floor.get('direct_hdc2_projected_archive_bytes')}`",
                f"- best_bounded_candidate_projected_archive_bytes: `{active_floor.get('best_bounded_candidate_projected_archive_bytes')}`",
                f"- best_bounded_candidate_byte_delta_vs_active_floor: `{active_floor.get('best_bounded_candidate_byte_delta_vs_active_floor')}`",
                f"- best_bounded_candidate_below_active_floor: `{_bool_text(active_floor.get('best_bounded_candidate_below_active_floor') is True)}`",
                f"- blockers: `{', '.join(str(item) for item in active_floor.get('blockers') or [])}`",
                "",
            ]
        )
    hdc2_runtime_gate = manifest.get("hdc2_direct_archive_runtime_gate")
    if isinstance(hdc2_runtime_gate, Mapping):
        lines.extend(
            [
                "## HDC2 Direct Runtime/Archive Gate",
                "",
                f"- status: `{hdc2_runtime_gate.get('status')}`",
                f"- projected_archive_bytes_if_direct_hdc2_replaced_section: `{hdc2_runtime_gate.get('projected_archive_bytes_if_direct_hdc2_replaced_section')}`",
                f"- missing_artifacts: `{', '.join(str(item) for item in hdc2_runtime_gate.get('missing_artifacts') or [])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Stream Evidence",
            "",
            f"- roundtrip_valid: `{_bool_text(evidence.get('roundtrip_valid') is True)}`",
            f"- raw_equal: `{_bool_text(evidence.get('raw_equal') is True)}`",
            f"- decoded_output_equal: `{_bool_text(evidence.get('decoded_output_equal') is True)}`",
            "",
            "Interpretation: HDC2 model overhead alone is still byte-negative.",
            "The next implementation must reduce both static context overhead and",
            "range-payload entropy gap before archive construction is rational.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_bounded_candidate(
    hdc2_work_product: Mapping[str, Any],
    *,
    actual_replacement_bytes: int,
    frontier_section_bytes: int,
    model_overhead_bytes: int,
) -> dict[str, Any] | None:
    rows = hdc2_work_product.get("bounded_hdc2_recode_variants")
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise Hdc2CombinedEntropyError("bounded_hdc2_recode_variants must be a list")

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_equality_closed = (
            row.get("raw_equal") is True
            and row.get("q_roundtrip_equal") is True
            and row.get("scale_roundtrip_equal") is True
        )
        if not raw_equality_closed:
            continue
        candidate_bytes = _positive_int(row.get("bytes"), "bounded_candidate_bytes")
        candidate_header = _optional_positive_int(row.get("header_bytes"), "candidate_header_bytes")
        raw_scale_bytes = _optional_nonnegative_int(row.get("raw_scale_bytes"), "raw_scale_bytes")
        mixed_payload = _optional_positive_int(row.get("mixed_payload_bytes"), "mixed_payload_bytes")
        q_brotli_bytes = _optional_positive_int(row.get("q_brotli_bytes"), "q_brotli_bytes")
        net_delta_vs_frontier = candidate_bytes - frontier_section_bytes
        archive_build_gate_blockers = []
        if candidate_bytes >= frontier_section_bytes:
            archive_build_gate_blockers.append(
                "candidate_stream_bytes_not_less_than_current_frontier_section_bytes"
            )
        if not raw_equality_closed:
            archive_build_gate_blockers.append("candidate_raw_equality_not_closed")
        candidate = {
            "variant": _nonempty_str(row.get("variant"), "bounded_candidate_variant"),
            "codec": row.get("codec"),
            "score_claim": False,
            "planning_only": True,
            "archive_ready": row.get("archive_ready") is True,
            "archive_build_gate": not archive_build_gate_blockers,
            "archive_build_gate_rule": (
                "candidate_stream_bytes_must_be_less_than_current_frontier_section_bytes_"
                "and_raw_equality_closed"
            ),
            "archive_build_gate_blockers": archive_build_gate_blockers,
            "raw_equality_closed": raw_equality_closed,
            "ready_for_exact_eval_dispatch": False,
            "bytes": candidate_bytes,
            "byte_delta_vs_hdc2_bytes": candidate_bytes - actual_replacement_bytes,
            "byte_reduction_vs_hdc2_bytes": actual_replacement_bytes - candidate_bytes,
            "net_byte_delta_vs_frontier_section": net_delta_vs_frontier,
            "remaining_reduction_to_beat_frontier_section_bytes": max(
                0,
                net_delta_vs_frontier + 1,
            ),
            "rate_score_delta_vs_frontier_section_if_runtime_supported": (
                net_delta_vs_frontier * RATE_SCORE_PER_BYTE
            ),
        }
        if candidate_header is not None:
            candidate["header_bytes"] = candidate_header
            candidate["static_context_header_reduction_vs_hdc2_bytes"] = (
                model_overhead_bytes - candidate_header
            )
        if raw_scale_bytes is not None:
            candidate["raw_scale_bytes"] = raw_scale_bytes
        payload_for_hdc2_comparison = mixed_payload if mixed_payload is not None else q_brotli_bytes
        if payload_for_hdc2_comparison is not None and raw_scale_bytes is not None:
            baseline_payload = actual_replacement_bytes - model_overhead_bytes - raw_scale_bytes
            candidate["payload_delta_vs_hdc2_bytes"] = (
                payload_for_hdc2_comparison - baseline_payload
            )
        if mixed_payload is not None:
            candidate["mixed_payload_bytes"] = mixed_payload
        if q_brotli_bytes is not None:
            candidate["q_brotli_bytes"] = q_brotli_bytes
        for key in ("q_stream_bytes", "brotli_quality"):
            value = _optional_positive_int(row.get(key), key)
            if value is not None:
                candidate[key] = value
        for key in (
            "raw_context_count",
            "range_context_count",
            "raw_payload_bytes",
            "range_payload_bytes",
            "schema_metadata_elided_vs_hdc2_bytes",
        ):
            value = _optional_nonnegative_int(row.get(key), key)
            if value is not None:
                candidate[key] = value
        stream_file = row.get("candidate_stream_file")
        if isinstance(stream_file, Mapping):
            candidate["candidate_stream_file"] = dict(stream_file)
        candidates.append(candidate)

    if not candidates:
        return None
    return min(candidates, key=lambda row: (int(row["bytes"]), str(row["variant"])))


def _active_floor_gate(
    active_floor: Mapping[str, Any] | None,
    *,
    frontier_archive_bytes: int,
    direct_hdc2_archive_delta: int,
    bounded_candidate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if active_floor is None:
        return None
    floor_bytes = _positive_int(active_floor.get("archive_bytes"), "active_floor.archive_bytes")
    floor_label = str(active_floor.get("label") or "")
    direct_archive_bytes = frontier_archive_bytes + direct_hdc2_archive_delta
    gate: dict[str, Any] = {
        "schema": "hnerv_hdc2_active_archive_floor_gate_v1",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "active_floor_label": floor_label,
        "active_floor_archive_bytes": floor_bytes,
        "active_floor_archive_sha256": active_floor.get("archive_sha256"),
        "active_floor_score": active_floor.get("score"),
        "source_frontier_archive_bytes": frontier_archive_bytes,
        "direct_hdc2_projected_archive_bytes": direct_archive_bytes,
        "direct_hdc2_byte_delta_vs_active_floor": direct_archive_bytes - floor_bytes,
        "best_bounded_candidate_projected_archive_bytes": None,
        "best_bounded_candidate_byte_delta_vs_active_floor": None,
        "best_bounded_candidate_below_active_floor": False,
        "blockers": [],
    }
    if bounded_candidate is not None:
        net_delta = _optional_int(
            bounded_candidate.get("net_byte_delta_vs_frontier_section"),
            "net_byte_delta_vs_frontier_section",
        )
        if net_delta is not None:
            projected = frontier_archive_bytes + net_delta
            delta_vs_floor = projected - floor_bytes
            gate.update(
                {
                    "best_bounded_candidate_variant": bounded_candidate.get("variant"),
                    "best_bounded_candidate_projected_archive_bytes": projected,
                    "best_bounded_candidate_byte_delta_vs_active_floor": delta_vs_floor,
                    "best_bounded_candidate_below_active_floor": delta_vs_floor < 0,
                }
            )
    if gate["best_bounded_candidate_below_active_floor"] is not True:
        gate["blockers"] = [
            f"not_below_active_archive_floor:{floor_bytes}",
            "requires_candidate_archive_below_active_floor_before_exact_eval",
        ]
    return gate


def _matching_combined_group(
    frontier_ranking: Mapping[str, Any],
    target_label: str,
    target_section: str,
) -> Mapping[str, Any]:
    groups = frontier_ranking.get("combined_entropy_gap_groups")
    if not isinstance(groups, list):
        raise Hdc2CombinedEntropyError("combined_entropy_gap_groups must be a list")
    for row in groups:
        if (
            isinstance(row, Mapping)
            and row.get("target_label") == target_label
            and row.get("frontier_section") == target_section
        ):
            return row
    raise Hdc2CombinedEntropyError("matching combined entropy group not found")


def _matching_entropy_target(
    entropy_audit: Mapping[str, Any],
    *,
    target_label: str,
    target_kind: str,
) -> Mapping[str, Any]:
    rows = entropy_audit.get("entropy_overhead_target_ranking")
    if not isinstance(rows, list):
        raise Hdc2CombinedEntropyError("entropy_overhead_target_ranking must be a list")
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("label") == target_label
            and row.get("target_kind") == target_kind
        ):
            return row
    raise Hdc2CombinedEntropyError(f"entropy target not found: {target_kind}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Hdc2CombinedEntropyError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Hdc2CombinedEntropyError(f"{label} must be a positive integer")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Hdc2CombinedEntropyError(f"{label} must be a nonnegative integer")
    return value


def _optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise Hdc2CombinedEntropyError(f"{label} must be an integer")
    return value


def _nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Hdc2CombinedEntropyError(f"{label} must be a nonempty string")
    return value


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
