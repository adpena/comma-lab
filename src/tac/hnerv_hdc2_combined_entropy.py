"""Planning manifests for HNeRV HDC2 combined entropy-overhead reduction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_hdc2_combined_entropy"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489


class Hdc2CombinedEntropyError(ValueError):
    """Raised when HDC2 combined entropy inputs are malformed."""


def build_hdc2_combined_entropy_manifest(
    frontier_ranking: Mapping[str, Any],
    entropy_audit: Mapping[str, Any],
    hdc2_work_product: Mapping[str, Any],
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
    model_overhead_bytes = _positive_int(model_target.get("target_bytes"), "model_overhead_target_bytes")
    payload_gap_bytes = _positive_int(payload_target.get("target_bytes"), "payload_gap_target_bytes")
    known_target_bytes = model_overhead_bytes + payload_gap_bytes
    projected_bytes_after_model_only = actual_replacement_bytes - model_overhead_bytes
    projected_bytes_after_combined = actual_replacement_bytes - known_target_bytes
    net_after_model_only = projected_bytes_after_model_only - frontier_section_bytes
    net_after_combined = projected_bytes_after_combined - frontier_section_bytes

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
            "frontier_archive_bytes": frontier.get("archive_bytes"),
            "frontier_score": frontier.get("score"),
            "frontier_section_bytes": frontier_section_bytes,
            "source_stream_sha256": source_stream.get("sha256"),
            "source_stream_decoded_raw_sha256": source_stream.get("decoded_raw_sha256"),
        },
        "byte_accounting": {
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
    return "\n".join(lines)


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


def _nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Hdc2CombinedEntropyError(f"{label} must be a nonempty string")
    return value


def _bool_text(value: bool) -> str:
    return "true" if value else "false"

