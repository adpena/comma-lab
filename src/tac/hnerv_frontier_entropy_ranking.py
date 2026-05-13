"""Rank HNeRV frontier byte mass against entropy-gap planning targets.

This module is a local planning surface. It joins the exact-replay scorecard,
lossless low-level repack manifests, and entropy-gap audits so the next
rate-only action can be selected from byte custody rather than prose notes. It
does not build archives, dispatch GPU work, or claim new scores.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
TOOL_NAME = "tac.hnerv_frontier_entropy_ranking"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES

DISPATCH_BLOCKERS = [
    "frontier_entropy_ranking_is_planning_only",
    "requires_byte_different_candidate_archive_before_new_exact_eval",
    "requires_runtime_parity_or_lossless_raw_equivalence_proof",
    "requires_archive_manifest_preflight",
    "requires_lane_dispatch_claim_before_gpu",
    "requires_exact_cuda_auth_eval",
]


class HnervFrontierEntropyRankingError(ValueError):
    """Raised when frontier entropy-ranking inputs are malformed."""


def build_frontier_entropy_gap_ranking(
    scorecard: Mapping[str, Any],
    *,
    entropy_audits: Sequence[Mapping[str, Any]] = (),
    candidate_manifests: Sequence[Mapping[str, Any]] = (),
    frontier_mode: str = "canonical",
) -> dict[str, Any]:
    """Build a deterministic non-dispatch ranking for HNeRV rate-only work."""

    rows = _scorecard_rows(scorecard)
    frontier = _selected_frontier_row(scorecard, rows, frontier_mode=frontier_mode)
    sections = _frontier_sections(frontier)
    candidates = [dict(item) for item in candidate_manifests if isinstance(item, Mapping)]

    byte_mass = _frontier_byte_mass_ranking(frontier, sections)
    exact_controls = _exact_control_actions(frontier, candidates)
    entropy_rows = _entropy_gap_ranking(frontier, sections, entropy_audits, candidates)
    combined_entropy = _combined_entropy_groups(entropy_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "planning_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "gpu_required": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
        "frontier_mode": frontier_mode,
        "current_frontier": _frontier_summary(frontier),
        "next_rate_only_action": _next_rate_only_action(
            exact_controls=exact_controls,
            entropy_groups=combined_entropy,
            byte_mass=byte_mass,
        ),
        "next_entropy_research_action": _next_entropy_research_action(combined_entropy, entropy_rows),
        "exact_lossless_control_actions": exact_controls,
        "frontier_byte_mass_ranking": byte_mass,
        "entropy_gap_ranking": entropy_rows,
        "combined_entropy_gap_groups": combined_entropy,
    }


def render_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a frontier entropy-gap ranking as compact markdown."""

    frontier = _as_mapping(manifest.get("current_frontier"), "current_frontier")
    next_action = _as_mapping(manifest.get("next_rate_only_action"), "next_rate_only_action")
    lines = [
        "# HNeRV Frontier Entropy Gap Ranking",
        "",
        f"- planning_only: `{_bool_text(manifest.get('planning_only') is True)}`",
        f"- score_claim: `{_bool_text(manifest.get('score_claim') is True)}`",
        f"- dispatch_attempted: `{_bool_text(manifest.get('dispatch_attempted') is True)}`",
        f"- ready_for_exact_eval_dispatch: `{_bool_text(manifest.get('ready_for_exact_eval_dispatch') is True)}`",
        f"- frontier_mode: `{manifest.get('frontier_mode')}`",
        "",
        "## Selected Frontier",
        "",
        f"- label: `{frontier.get('label')}`",
        f"- score: `{frontier.get('score')}`",
        f"- archive_bytes: `{frontier.get('archive_bytes')}`",
        f"- archive_sha256: `{frontier.get('archive_sha256')}`",
        f"- eval_artifact: `{frontier.get('eval_artifact')}`",
        "",
        "## Next Rate-Only Action",
        "",
        f"- action_id: `{next_action.get('action_id')}`",
        f"- target_label: `{next_action.get('target_label')}`",
        f"- target_section: `{next_action.get('target_section')}`",
        f"- required_next_artifact: `{next_action.get('required_next_artifact')}`",
        f"- dispatch_allowed: `{_bool_text(next_action.get('dispatch_allowed') is True)}`",
        f"- rationale: {next_action.get('rationale')}",
    ]

    entropy_action = manifest.get("next_entropy_research_action")
    if isinstance(entropy_action, Mapping):
        lines.extend(
            [
                "",
                "## Next Entropy Research Action",
                "",
                f"- action_id: `{entropy_action.get('action_id')}`",
                f"- target_label: `{entropy_action.get('target_label')}`",
                f"- target_section: `{entropy_action.get('target_section')}`",
                f"- required_next_artifact: `{entropy_action.get('required_next_artifact')}`",
                f"- minimum_section_bytes_to_beat: `{entropy_action.get('minimum_section_bytes_to_beat')}`",
                f"- dispatch_allowed: `{_bool_text(entropy_action.get('dispatch_allowed') is True)}`",
            ]
        )

    controls = _as_sequence(manifest.get("exact_lossless_control_actions"))
    if controls:
        lines.extend(
            [
                "",
                "## Exact Lossless Controls",
                "",
                "| label | source bytes | candidate bytes | byte delta | review status |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in controls:
            item = _as_mapping(row, "exact_lossless_control_actions[]")
            lines.append(
                "| {label} | {source} | {candidate} | {delta} | `{status}` |".format(
                    label=item.get("target_label"),
                    source=item.get("source_archive_bytes"),
                    candidate=item.get("candidate_archive_bytes"),
                    delta=item.get("total_byte_delta"),
                    status=item.get("review_status"),
                )
            )

    groups = _as_sequence(manifest.get("combined_entropy_gap_groups"))
    if groups:
        lines.extend(
            [
                "",
                "## Combined Entropy Gap Groups",
                "",
                "| target | section | HDC2 bytes | current section bytes | known target bytes | net now | net after known targets | verdict |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in groups:
            item = _as_mapping(row, "combined_entropy_gap_groups[]")
            lines.append(
                "| {target} | `{section}` | {actual} | {frontier_bytes} | {target_bytes} | {net_now} | {net_after} | `{verdict}` |".format(
                    target=item.get("target_label"),
                    section=item.get("frontier_section"),
                    actual=item.get("actual_replacement_bytes"),
                    frontier_bytes=item.get("frontier_section_bytes"),
                    target_bytes=item.get("known_target_bytes"),
                    net_now=item.get("net_byte_delta_vs_current_frontier_section"),
                    net_after=item.get("net_byte_delta_after_known_targets_vs_current_frontier_section"),
                    verdict=item.get("verdict"),
                )
            )

    entropy_rows = _as_sequence(manifest.get("entropy_gap_ranking"))
    if entropy_rows:
        lines.extend(
            [
                "",
                "## Entropy Target Ranking",
                "",
                "| rank | target | kind | target bytes | section | net now | net after target | required artifact |",
                "|---:|---|---|---:|---|---:|---:|---|",
            ]
        )
        for rank, row in enumerate(entropy_rows[:12], start=1):
            item = _as_mapping(row, "entropy_gap_ranking[]")
            lines.append(
                "| {rank} | {target} | `{kind}` | {target_bytes} | `{section}` | {net_now} | {net_after} | `{artifact}` |".format(
                    rank=rank,
                    target=item.get("target_label"),
                    kind=item.get("target_kind"),
                    target_bytes=item.get("target_bytes"),
                    section=item.get("frontier_section"),
                    net_now=item.get("net_byte_delta_vs_current_frontier_section"),
                    net_after=item.get("net_byte_delta_after_target_vs_current_frontier_section"),
                    artifact=item.get("required_next_artifact"),
                )
            )

    byte_mass = _as_sequence(manifest.get("frontier_byte_mass_ranking"))
    if byte_mass:
        lines.extend(
            [
                "",
                "## Current Frontier Byte Mass",
                "",
                "| rank | section | role | bytes | entropy b/B | rate mass |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(byte_mass[:8], start=1):
            item = _as_mapping(row, "frontier_byte_mass_ranking[]")
            lines.append(
                "| {rank} | `{section}` | `{role}` | {bytes_} | {entropy} | {rate_mass} |".format(
                    rank=rank,
                    section=item.get("section"),
                    role=item.get("optimization_role"),
                    bytes_=item.get("section_bytes"),
                    entropy=item.get("entropy_bits_per_byte"),
                    rate_mass=item.get("rate_mass_score_if_removed"),
                )
            )

    lines.extend(
        [
            "",
            "Interpretation: this manifest ranks rate-only work. It is not a new",
            "score claim, not a candidate archive manifest, and not a dispatch",
            "authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def _scorecard_rows(scorecard: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = scorecard.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise HnervFrontierEntropyRankingError("scorecard rows must be a nonempty list")
    out = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not out:
        raise HnervFrontierEntropyRankingError("scorecard rows must contain objects")
    return out


def _selected_frontier_row(
    scorecard: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    frontier_mode: str,
) -> dict[str, Any]:
    if frontier_mode == "canonical":
        return _current_frontier_row(scorecard, rows)
    if frontier_mode == "score_lowering":
        return _score_lowering_frontier_row(scorecard, rows)
    raise HnervFrontierEntropyRankingError(f"unknown frontier_mode: {frontier_mode}")


def _current_frontier_row(scorecard: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current = scorecard.get("current_frontier")
    if isinstance(current, Mapping) and current.get("label"):
        label = str(current["label"])
        for row in rows:
            if str(row.get("label") or "") == label:
                return dict(row)
    eligible = [
        dict(row)
        for row in rows
        if row.get("canonical_frontier_eligible") is True and _is_number(row.get("score"))
    ]
    if not eligible:
        raise HnervFrontierEntropyRankingError("scorecard has no canonical frontier-eligible rows")
    return min(
        eligible,
        key=lambda row: (
            float(row["score"]),
            int(row.get("archive_bytes") or 10**18),
            str(row.get("label") or ""),
        ),
    )


def _score_lowering_frontier_row(
    scorecard: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frontier = scorecard.get("score_lowering_frontier")
    if not isinstance(frontier, Mapping) or not frontier.get("label"):
        raise HnervFrontierEntropyRankingError("scorecard has no score_lowering_frontier")
    label = str(frontier["label"])
    for row in rows:
        if str(row.get("label") or "") == label:
            return dict(row)
    raise HnervFrontierEntropyRankingError(
        f"score_lowering_frontier row missing from scorecard rows: {label}"
    )


def _frontier_sections(frontier: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sections = frontier.get("payload_sections") or frontier.get("top_payload_sections")
    if not isinstance(raw_sections, Sequence) or isinstance(raw_sections, (str, bytes, bytearray)):
        raise HnervFrontierEntropyRankingError("current frontier row is missing payload sections")
    sections = [dict(section) for section in raw_sections if isinstance(section, Mapping)]
    if not sections:
        raise HnervFrontierEntropyRankingError("current frontier row has no payload section objects")
    return sections


def _frontier_summary(frontier: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": frontier.get("label"),
        "score": frontier.get("score"),
        "archive_bytes": frontier.get("archive_bytes"),
        "archive_sha256": frontier.get("archive_sha256"),
        "payload_sha256": frontier.get("payload_sha256"),
        "runtime_tree_sha256": frontier.get("runtime_tree_sha256"),
        "frontier_scope": frontier.get("frontier_scope"),
        "evidence_grade": frontier.get("evidence_grade"),
        "eval_artifact": frontier.get("eval_artifact"),
    }


def _frontier_byte_mass_ranking(
    frontier: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for section in sections:
        section_name = str(section.get("name") or "")
        section_bytes = section.get("bytes")
        if not section_name or not isinstance(section_bytes, int) or section_bytes <= 0:
            continue
        role = _section_optimization_role(section_name)
        rows.append(
            {
                "label": frontier.get("label"),
                "archive_sha256": frontier.get("archive_sha256"),
                "payload_sha256": frontier.get("payload_sha256"),
                "section": section_name,
                "optimization_role": role,
                "section_bytes": section_bytes,
                "section_sha256": section.get("sha256"),
                "entropy_bits_per_byte": section.get("entropy_bits_per_byte"),
                "rate_mass_score_if_removed": _rate_score(section_bytes),
                "score_claim": False,
                "dispatch_attempted": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            1 if row["optimization_role"] == "control_or_metadata" else 0,
            -int(row["section_bytes"]),
            str(row["section"]),
        ),
    )


def _exact_control_actions(
    frontier: Mapping[str, Any],
    candidate_manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frontier_sha = frontier.get("archive_sha256")
    controls = []
    for manifest in candidate_manifests:
        if manifest.get("candidate_archive_sha256") != frontier_sha:
            continue
        raw_equivalence = manifest.get("brotli_raw_equivalence")
        raw_equivalence_closed = (
            isinstance(raw_equivalence, Sequence)
            and not isinstance(raw_equivalence, (str, bytes, bytearray))
            and bool(raw_equivalence)
            and all(isinstance(row, Mapping) and row.get("raw_equal") is True for row in raw_equivalence)
        )
        audit = manifest.get("candidate_diff_audit")
        audit_blockers = (
            list(audit.get("blockers") or []) if isinstance(audit, Mapping) else []
        )
        ready = raw_equivalence_closed and not audit_blockers
        controls.append(
            {
                "action_id": "review_existing_exact_lossless_repack_control",
                "target_label": frontier.get("label"),
                "candidate_manifest": manifest.get("candidate_manifest_path", ""),
                "source_label": manifest.get("source_label"),
                "source_archive_sha256": manifest.get("source_archive_sha256"),
                "candidate_archive_sha256": manifest.get("candidate_archive_sha256"),
                "source_archive_bytes": manifest.get("source_archive_bytes"),
                "candidate_archive_bytes": manifest.get("candidate_archive_bytes"),
                "total_byte_delta": (
                    audit.get("total_byte_delta") if isinstance(audit, Mapping) else None
                ),
                "rate_score_delta_if_components_equal": (
                    audit.get("rate_score_delta_if_components_equal")
                    if isinstance(audit, Mapping)
                    else None
                ),
                "raw_equivalence_closed": raw_equivalence_closed,
                "review_status": (
                    "ready_for_promotion_review_existing_exact_custody"
                    if ready
                    else "blocked_pending_candidate_diff_review"
                ),
                "required_next_artifact": "operator_promotion_review_note_for_existing_exact_archive_sha",
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_allowed": False,
                "ready_for_exact_eval_dispatch": False,
                "blockers": audit_blockers,
            }
        )
    return sorted(
        controls,
        key=lambda row: (
            0 if row["review_status"] == "ready_for_promotion_review_existing_exact_custody" else 1,
            str(row.get("target_label") or ""),
        ),
    )


def _entropy_gap_ranking(
    frontier: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
    entropy_audits: Sequence[Mapping[str, Any]],
    candidate_manifests: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for audit in entropy_audits:
        if not isinstance(audit, Mapping):
            continue
        streams = {
            str(stream.get("label") or ""): stream
            for stream in audit.get("streams") or []
            if isinstance(stream, Mapping)
        }
        for target in audit.get("entropy_overhead_target_ranking") or []:
            if not isinstance(target, Mapping):
                continue
            target_label = str(target.get("label") or "")
            stream = streams.get(target_label, {})
            actual_bytes = _first_int(target.get("actual_bytes"), stream.get("actual_bytes"))
            target_bytes = _first_int(target.get("target_bytes"))
            source_decoder_sha = _first_str(
                target.get("source_decoder_section_sha256"),
                audit.get("source_decoder_section_sha256"),
                stream.get("source_decoder_section_sha256"),
            )
            anchor = _anchor_frontier_section(source_decoder_sha, sections, candidate_manifests)
            frontier_section = anchor.get("section") if anchor else {}
            frontier_section_bytes = (
                frontier_section.get("bytes") if isinstance(frontier_section, Mapping) else None
            )
            net_now = _int_delta(actual_bytes, frontier_section_bytes)
            net_after_target = (
                actual_bytes - target_bytes - frontier_section_bytes
                if isinstance(actual_bytes, int)
                and isinstance(target_bytes, int)
                and isinstance(frontier_section_bytes, int)
                else None
            )
            source_section_bytes = _first_int(
                target.get("source_decoder_section_bytes"),
                stream.get("source_decoder_section_bytes"),
            )
            net_source = _int_delta(actual_bytes, source_section_bytes)
            minimum_additional = (
                max(0, net_after_target + 1) if isinstance(net_after_target, int) else None
            )
            rows.append(
                {
                    "target_label": target_label,
                    "target_kind": target.get("target_kind"),
                    "target_action": target.get("target_action"),
                    "target_bytes": target_bytes,
                    "target_bytes_field": target.get("target_bytes_field"),
                    "required_next_artifact": target.get("required_next_artifact"),
                    "audit_rank": target.get("rank"),
                    "audit_source_label": audit.get("source_label"),
                    "source_archive_sha256": _first_str(
                        target.get("source_archive_sha256"),
                        audit.get("source_archive_sha256"),
                        stream.get("source_archive_sha256"),
                    ),
                    "source_decoder_section_sha256": source_decoder_sha,
                    "source_decoder_section_bytes": source_section_bytes,
                    "frontier_label": frontier.get("label"),
                    "frontier_archive_sha256": frontier.get("archive_sha256"),
                    "frontier_section": (
                        frontier_section.get("name") if isinstance(frontier_section, Mapping) else None
                    ),
                    "frontier_section_sha256": (
                        frontier_section.get("sha256") if isinstance(frontier_section, Mapping) else None
                    ),
                    "frontier_section_bytes": frontier_section_bytes,
                    "frontier_anchor_source": anchor.get("match_source") if anchor else "unmatched",
                    "actual_replacement_bytes": actual_bytes,
                    "net_byte_delta_vs_source_section": net_source,
                    "net_byte_delta_vs_current_frontier_section": net_now,
                    "net_byte_delta_after_target_vs_current_frontier_section": net_after_target,
                    "rate_score_delta_if_replacement_now": _rate_score(net_now),
                    "rate_score_delta_after_target": _rate_score(net_after_target),
                    "minimum_additional_reduction_after_target_to_beat_current_bytes": minimum_additional,
                    "readiness_stage": _entropy_readiness_stage(net_now, net_after_target),
                    "byte_equivalence_blockers": list(target.get("byte_equivalence_blockers") or []),
                    "dispatch_blockers": list(target.get("dispatch_blockers") or DISPATCH_BLOCKERS),
                    "score_claim": False,
                    "dispatch_attempted": False,
                    "dispatch_allowed": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            0 if _is_negative_int(row.get("net_byte_delta_after_target_vs_current_frontier_section")) else 1,
            _optional_int(row.get("minimum_additional_reduction_after_target_to_beat_current_bytes")),
            -_optional_int(row.get("target_bytes")),
            _optional_int(row.get("audit_rank")),
            str(row.get("target_label") or ""),
            str(row.get("target_kind") or ""),
        ),
    )


def _combined_entropy_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("target_label") or ""),
            str(row.get("frontier_section") or ""),
            str(row.get("source_decoder_section_sha256") or ""),
        )
        grouped[key].append(row)

    out = []
    for (_label, _section, _sha), group in grouped.items():
        first = group[0]
        actual = _first_int(first.get("actual_replacement_bytes"))
        frontier_bytes = _first_int(first.get("frontier_section_bytes"))
        target_sum = sum(int(row.get("target_bytes")) for row in group if isinstance(row.get("target_bytes"), int))
        net_now = _int_delta(actual, frontier_bytes)
        net_after = (
            actual - target_sum - frontier_bytes
            if isinstance(actual, int) and isinstance(frontier_bytes, int)
            else None
        )
        out.append(
            {
                "target_label": first.get("target_label"),
                "frontier_label": first.get("frontier_label"),
                "frontier_section": first.get("frontier_section"),
                "frontier_section_sha256": first.get("frontier_section_sha256"),
                "frontier_section_bytes": frontier_bytes,
                "actual_replacement_bytes": actual,
                "known_target_bytes": target_sum,
                "target_kinds": sorted(str(row.get("target_kind") or "") for row in group),
                "required_next_artifacts": sorted(
                    str(row.get("required_next_artifact") or "") for row in group
                ),
                "net_byte_delta_vs_current_frontier_section": net_now,
                "net_byte_delta_after_known_targets_vs_current_frontier_section": net_after,
                "minimum_section_bytes_to_beat": (
                    frontier_bytes - 1 if isinstance(frontier_bytes, int) else None
                ),
                "rate_score_delta_if_replacement_now": _rate_score(net_now),
                "rate_score_delta_after_known_targets": _rate_score(net_after),
                "verdict": _combined_verdict(net_now, net_after),
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            0 if _is_negative_int(row.get("net_byte_delta_after_known_targets_vs_current_frontier_section")) else 1,
            _optional_int(row.get("net_byte_delta_after_known_targets_vs_current_frontier_section")),
            -_optional_int(row.get("known_target_bytes")),
            str(row.get("target_label") or ""),
        ),
    )


def _next_rate_only_action(
    *,
    exact_controls: Sequence[Mapping[str, Any]],
    entropy_groups: Sequence[Mapping[str, Any]],
    byte_mass: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for control in exact_controls:
        if control.get("review_status") == "ready_for_promotion_review_existing_exact_custody":
            return {
                "action_id": "review_current_exact_lossless_brotli_control_before_promotion",
                "target_label": control.get("target_label"),
                "target_section": "decoder_packed_brotli",
                "required_next_artifact": control.get("required_next_artifact"),
                "rationale": (
                    "existing exact CUDA custody already covers the byte-different "
                    "lossless repack; review/promotion is the next rate-only step, "
                    "not another dispatch"
                ),
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
    entropy_action = _next_entropy_research_action(entropy_groups, ())
    if entropy_action:
        return entropy_action
    if byte_mass:
        top = byte_mass[0]
        return {
            "action_id": "build_byte_different_recode_for_largest_current_frontier_section",
            "target_label": top.get("label"),
            "target_section": top.get("section"),
            "required_next_artifact": "old_new_section_sha256_and_charged_byte_diff_manifest",
            "rationale": "largest current exact-frontier section is the deterministic rate target",
            "score_claim": False,
            "dispatch_attempted": False,
            "dispatch_allowed": False,
            "ready_for_exact_eval_dispatch": False,
        }
    raise HnervFrontierEntropyRankingError("no rate-only action could be selected")


def _next_entropy_research_action(
    groups: Sequence[Mapping[str, Any]],
    entropy_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    for group in groups:
        if group.get("verdict") == "combined_targets_can_cross_rate_positive_if_byte_equivalent":
            artifacts = [
                item
                for item in group.get("required_next_artifacts") or []
                if isinstance(item, str) and item
            ]
            return {
                "action_id": "build_combined_entropy_overhead_reduction_manifest",
                "target_label": group.get("target_label"),
                "target_section": group.get("frontier_section"),
                "required_next_artifact": artifacts[0] if artifacts else "combined_entropy_reduction_manifest",
                "minimum_section_bytes_to_beat": group.get("minimum_section_bytes_to_beat"),
                "rationale": (
                    "the current entropy candidate is byte-negative, but the known "
                    "combined header/payload targets cross the current section only "
                    "if proven together"
                ),
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
    for row in entropy_rows:
        if row.get("readiness_stage") == "target_would_cross_rate_positive_if_byte_equivalent":
            return {
                "action_id": "build_entropy_target_byte_equivalence_manifest",
                "target_label": row.get("target_label"),
                "target_section": row.get("frontier_section"),
                "required_next_artifact": row.get("required_next_artifact"),
                "minimum_section_bytes_to_beat": (
                    int(row["frontier_section_bytes"]) - 1
                    if isinstance(row.get("frontier_section_bytes"), int)
                    else None
                ),
                "rationale": "selected entropy target would be rate-positive if byte-equivalent",
                "score_claim": False,
                "dispatch_attempted": False,
                "dispatch_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
    return None


def _anchor_frontier_section(
    source_decoder_sha: str | None,
    sections: Sequence[Mapping[str, Any]],
    candidate_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not source_decoder_sha:
        return None
    by_sha = {
        str(section.get("sha256")): section
        for section in sections
        if isinstance(section.get("sha256"), str)
    }
    if source_decoder_sha in by_sha:
        return {"section": by_sha[source_decoder_sha], "match_source": "direct_section_sha256"}
    for manifest in candidate_manifests:
        audit = manifest.get("candidate_diff_audit")
        if not isinstance(audit, Mapping):
            continue
        for row in audit.get("sections") or []:
            if not isinstance(row, Mapping):
                continue
            if row.get("source_section_sha256") != source_decoder_sha:
                continue
            candidate_sha = row.get("candidate_section_sha256")
            if isinstance(candidate_sha, str) and candidate_sha in by_sha:
                return {
                    "section": by_sha[candidate_sha],
                    "match_source": "candidate_diff_source_to_current_section_sha256",
                }
    return None


def _section_optimization_role(section_name: str) -> str:
    lowered = section_name.lower()
    if "header" in lowered or "meta" in lowered or "scale" in lowered:
        return "control_or_metadata"
    if "decoder" in lowered or "weight" in lowered:
        return "decoder_weight_stream"
    if "latent" in lowered:
        return "latent_stream"
    if "sidecar" in lowered or "correction" in lowered:
        return "sidecar_or_correction_stream"
    if "hist" in lowered or "range" in lowered or "ac_" in lowered or "arithmetic" in lowered:
        return "entropy_model_or_range_stream"
    return "opaque_payload_stream"


def _entropy_readiness_stage(net_now: int | None, net_after_target: int | None) -> str:
    if net_now is None:
        return "unmatched_to_current_frontier_section"
    if net_now < 0:
        return "replacement_is_rate_positive_if_byte_equivalent"
    if net_after_target is None:
        return "current_candidate_byte_negative_vs_frontier_section"
    if net_after_target < 0:
        return "target_would_cross_rate_positive_if_byte_equivalent"
    return "target_alone_insufficient_for_rate_positive_archive"


def _combined_verdict(net_now: int | None, net_after: int | None) -> str:
    if net_now is None:
        return "unmatched_to_current_frontier_section"
    if net_now < 0:
        return "current_replacement_already_rate_positive_if_byte_equivalent"
    if net_after is not None and net_after < 0:
        return "combined_targets_can_cross_rate_positive_if_byte_equivalent"
    return "known_targets_still_byte_negative_or_incomplete"


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HnervFrontierEntropyRankingError(f"{context} must be an object")
    return value


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _int_delta(left: Any, right: Any) -> int | None:
    if isinstance(left, int) and isinstance(right, int):
        return left - right
    return None


def _optional_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return 10**18


def _is_negative_int(value: Any) -> bool:
    return isinstance(value, int) and value < 0


def _rate_score(byte_delta: Any) -> float | None:
    if not isinstance(byte_delta, int):
        return None
    return round(byte_delta * RATE_SCORE_PER_BYTE, 12)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


__all__ = [
    "HnervFrontierEntropyRankingError",
    "build_frontier_entropy_gap_ranking",
    "render_markdown",
]
