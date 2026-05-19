# SPDX-License-Identifier: MIT
"""Build packet-valid operator rows for master-gradient score-response probes.

The useful replacement for a raw ``(N_archive_bytes, 3)`` derivative is a
manifest of packet-valid mutation operators. Each row names a parser-proven
logical section, a mutation family, the score-response columns to measure, and
the packet proofs required before the row can become a probe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from tac.master_gradient_feasibility import AxisLabel, audit_master_gradient_probe_plan

RESPONSE_MATRIX_COLUMNS: tuple[str, str, str] = (
    "seg_dist_delta",
    "pose_dist_delta",
    "rate_bytes_delta",
)

PACKET_PROOFS: tuple[str, ...] = (
    "repacked_archive",
    "updated_zip_headers",
    "updated_zip_crc",
    "inflate_success_proof",
    "byte_consumption_noop_detector",
)

_SKIP_ROLES: frozenset[str] = frozenset(
    {
        "internal_length_header",
        "wire_format_magic",
    }
)


@dataclass(frozen=True)
class MasterGradientOperatorRow:
    """One valid mutation-operator response row."""

    operator_id: str
    section_name: str
    section_role: str
    section_offset: int | None
    section_len: int | None
    section_sha256: str | None
    mutation_grain: str
    mutation_operator: str
    response_matrix_columns: tuple[str, ...]
    axis_label: str
    required_proofs: tuple[str, ...]
    operator_response_valid: bool
    ready_for_operator_probe: bool
    ready_for_provider_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    dispatch_attempted: bool
    feasibility_verdict: str
    blockers: tuple[str, ...]
    rationale: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateModificationSpec:
    """Packet-valid modification candidate for a score-response row.

    This is the explicit replacement for raw ``{byte_idx: delta}`` maps. A
    spec names a logical packet section and grammar-aware mutation operator;
    it never carries archive-byte coordinates. Packet proofs stay explicit so
    an operator-row manifest can be useful for planning without becoming a
    dispatchable score claim.
    """

    spec_id: str
    source_archive_path: str | None
    source_archive_sha256: str | None
    source_archive_bytes: int | None
    operator_id: str
    section_name: str
    section_role: str
    mutation_grain: str
    mutation_operator: str
    axis_label: str
    response_matrix_columns: tuple[str, ...]
    packet_proofs_required: tuple[str, ...]
    packet_proofs_available: bool
    ready_for_operator_probe: bool
    ready_for_provider_dispatch: bool
    score_claim: bool
    promotion_eligible: bool
    rank_or_kill_eligible: bool
    ready_for_exact_eval_dispatch: bool
    dispatch_attempted: bool
    coordinate_system: str
    raw_archive_byte_coordinates_allowed: bool
    blockers: tuple[str, ...]
    rationale: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def build_master_gradient_operator_plan(
    layout_manifest: dict[str, Any],
    *,
    axis_label: AxisLabel | str = "paired_contest_cpu_cuda",
    packet_proofs_available: bool = False,
) -> dict[str, Any]:
    """Return an operator-row manifest from a parser-proven layout manifest.

    ``packet_proofs_available`` is intentionally explicit. The default is a
    fail-closed planning manifest: rows are visible, but not probe-ready until a
    concrete mutation builder proves repack, ZIP metadata, CRC, inflate success,
    and byte-consumption closure.
    """

    logical = layout_manifest.get("logical_layout")
    rows: list[MasterGradientOperatorRow] = []
    skipped_sections: list[dict[str, object]] = []
    top_level_blockers: list[str] = []

    if not isinstance(logical, dict):
        top_level_blockers.append("logical_layout_missing")
        sections: list[dict[str, Any]] = []
        grammar = None
    else:
        grammar = logical.get("grammar")
        raw_sections = logical.get("sections")
        if not isinstance(raw_sections, list):
            top_level_blockers.append("logical_sections_missing")
            sections = []
        else:
            sections = [section for section in raw_sections if isinstance(section, dict)]

    for section in sections:
        section_name = str(section.get("name", "unknown_section"))
        section_role = str(section.get("role", "unknown_role"))
        if _should_skip_section(section_name, section_role):
            skipped_sections.append(
                {
                    "section_name": section_name,
                    "section_role": section_role,
                    "reason": "grammar_header_or_magic_not_score_lowering_operator",
                }
            )
            continue
        rows.append(
            _build_row(
                section,
                axis_label=str(axis_label),
                packet_proofs_available=packet_proofs_available,
            )
        )

    if logical and not rows:
        top_level_blockers.append("no_score_lowering_logical_sections")

    row_blockers = _unique(
        blocker for row in rows for blocker in row.blockers if blocker != "missing_axis_label"
    )
    if any(row.blockers for row in rows):
        top_level_blockers.append("operator_rows_blocked_until_packet_proofs_land")
    candidate_specs = [
        _candidate_spec_from_row(
            row,
            layout_manifest=layout_manifest,
            packet_proofs_available=packet_proofs_available,
        )
        for row in rows
    ]

    return {
        "schema": "tac_master_gradient_operator_plan_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "axis_label": str(axis_label),
        "raw_byte_gradient_valid": False,
        "raw_archive_byte_rows_emitted": 0,
        "operator_response_columns": RESPONSE_MATRIX_COLUMNS,
        "packet_proofs_available": bool(packet_proofs_available),
        "required_packet_proofs": PACKET_PROOFS,
        "source_manifest_schema": layout_manifest.get("schema"),
        "source_archive_path": layout_manifest.get("archive_path"),
        "source_archive_sha256": layout_manifest.get("archive_sha256"),
        "source_archive_bytes": layout_manifest.get("archive_bytes"),
        "logical_grammar": grammar,
        "operator_row_count": len(rows),
        "candidate_modification_spec_count": len(candidate_specs),
        "skipped_section_count": len(skipped_sections),
        "rows": [row.to_manifest() for row in rows],
        "candidate_modification_specs": [
            spec.to_manifest() for spec in candidate_specs
        ],
        "skipped_sections": skipped_sections,
        "blockers": _unique((*top_level_blockers, *row_blockers)),
        "next_step": (
            "implement one grammar-aware mutation builder for the highest-EV row, "
            "then rerun with packet_proofs_available=true only after repack, CRC, "
            "inflate, and byte-consumption proofs exist"
        ),
    }


def build_master_gradient_operator_plan_payload(
    layout_manifest_or_batch: dict[str, Any],
    *,
    axis_label: AxisLabel | str = "paired_contest_cpu_cuda",
    packet_proofs_available: bool = False,
) -> dict[str, Any]:
    """Build a single-plan or batch-plan payload from layout JSON."""

    runs = layout_manifest_or_batch.get("runs")
    if layout_manifest_or_batch.get("schema") != "tac_frontier_archive_layout_batch_v1":
        return build_master_gradient_operator_plan(
            layout_manifest_or_batch,
            axis_label=axis_label,
            packet_proofs_available=packet_proofs_available,
        )
    if not isinstance(runs, list):
        single = build_master_gradient_operator_plan(
            {},
            axis_label=axis_label,
            packet_proofs_available=packet_proofs_available,
        )
        return {
            "schema": "tac_master_gradient_operator_plan_batch_v1",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ready_for_provider_dispatch": False,
            "dispatch_attempted": False,
            "axis_label": str(axis_label),
            "packet_proofs_available": bool(packet_proofs_available),
            "plan_count": 0,
            "operator_row_count": 0,
            "candidate_modification_spec_count": 0,
            "raw_archive_byte_rows_emitted": 0,
            "plans": [],
            "blockers": _unique(("batch_runs_missing", *single["blockers"])),
        }

    plans = [
        build_master_gradient_operator_plan(
            run,
            axis_label=axis_label,
            packet_proofs_available=packet_proofs_available,
        )
        for run in runs
        if isinstance(run, dict)
    ]
    return {
        "schema": "tac_master_gradient_operator_plan_batch_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "axis_label": str(axis_label),
        "packet_proofs_available": bool(packet_proofs_available),
        "plan_count": len(plans),
        "operator_row_count": sum(int(plan["operator_row_count"]) for plan in plans),
        "candidate_modification_spec_count": sum(
            int(plan["candidate_modification_spec_count"]) for plan in plans
        ),
        "raw_archive_byte_rows_emitted": sum(
            int(plan["raw_archive_byte_rows_emitted"]) for plan in plans
        ),
        "plans": plans,
        "blockers": _unique(blocker for plan in plans for blocker in plan["blockers"]),
    }


def build_pose_axis_operator_candidates(
    selector_payload: dict[str, Any],
    layout_manifest: dict[str, Any],
    *,
    packet_proofs_available: bool = False,
) -> dict[str, Any]:
    """Resolve pose-axis diagnostic byte ranks into grammar-aware operator specs.

    ``select_pose_axis_dominant_bytes`` intentionally emits diagnostic
    gradient-subject byte indices, not archive-byte mutation authority. This
    helper is the next lowering pass: it maps each diagnostic index through a
    parser-proven logical layout section and returns typed
    ``CandidateModificationSpec`` rows. The rows remain non-dispatchable until
    packet proofs are available.
    """

    selected = _selected_pose_axis_entries(selector_payload)
    sections = _layout_sections(layout_manifest)
    archive_sha = str(
        layout_manifest.get("archive_sha256")
        or selector_payload.get("archive_sha256")
        or ""
    )
    archive_short = archive_sha[:12] if archive_sha else "unknown"
    resolved: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    specs: list[CandidateModificationSpec] = []
    blockers: list[str] = []

    if not selected:
        blockers.append("pose_axis_selector_entries_missing")
    if not sections:
        blockers.append("logical_layout_sections_missing")

    for entry in selected:
        rank = _optional_int(entry.get("rank")) or len(resolved) + len(unresolved) + 1
        index = _optional_int(entry.get("diagnostic_gradient_subject_byte_index"))
        if index is None or index < 0:
            unresolved.append(
                {
                    "rank": rank,
                    "diagnostic_gradient_subject_byte_index": index,
                    "reason": "diagnostic_index_missing_or_invalid",
                }
            )
            blockers.append("diagnostic_index_missing_or_invalid")
            continue

        section = _find_section_for_index(sections, index)
        if section is None:
            unresolved.append(
                {
                    "rank": rank,
                    "diagnostic_gradient_subject_byte_index": index,
                    "reason": "diagnostic_index_outside_layout_sections",
                }
            )
            blockers.append("diagnostic_index_outside_layout_sections")
            continue

        section_name = str(section.get("name", "unknown_section"))
        section_role = _section_role(section)
        if _should_skip_section(section_name, section_role):
            unresolved.append(
                {
                    "rank": rank,
                    "diagnostic_gradient_subject_byte_index": index,
                    "section_name": section_name,
                    "section_role": section_role,
                    "reason": "diagnostic_index_hits_non_mutation_header_section",
                }
            )
            blockers.append("diagnostic_index_hits_non_mutation_header_section")
            continue

        section_offset, section_len = _section_span(section)
        row = _build_row(
            {
                "name": section_name,
                "role": section_role,
                "offset": section_offset,
                "len": section_len,
                "sha256": section.get("sha256"),
            },
            axis_label="diagnostic",
            packet_proofs_available=packet_proofs_available,
        )
        base_spec = _candidate_spec_from_row(
            row,
            layout_manifest=layout_manifest,
            packet_proofs_available=packet_proofs_available,
        )
        spec_blockers = base_spec.blockers
        if not packet_proofs_available:
            spec_blockers = _unique((*spec_blockers, "packet_proofs_missing"))
        relative_offset = index - int(section_offset or 0)
        spec = replace(
            base_spec,
            spec_id=(
                f"pose_axis_candidate::{archive_short}::{rank:04d}::"
                f"{_slug(section_name)}"
            ),
            operator_id=f"{row.operator_id}::pose_axis_rank_{rank:04d}",
            axis_label="pose",
            blockers=spec_blockers,
            rationale=(
                "resolved diagnostic pose-axis byte into parser-proven logical section",
                f"diagnostic_gradient_subject_byte_index={index}",
                f"section_relative_offset={relative_offset}",
                f"pose_axis_share={entry.get('pose_axis_share')}",
                f"pose_axis_abs_score_contribution={entry.get('pose_axis_abs_score_contribution')}",
                *base_spec.rationale,
            ),
        )
        specs.append(spec)
        resolved.append(
            {
                "rank": rank,
                "diagnostic_gradient_subject_byte_index": index,
                "section_name": section_name,
                "section_role": section_role,
                "section_offset": section_offset,
                "section_len": section_len,
                "section_relative_offset": relative_offset,
                "mutation_operator": row.mutation_operator,
                "spec_id": spec.spec_id,
                "ready_for_operator_probe": spec.ready_for_operator_probe,
            }
        )
        blockers.extend(spec.blockers)

    return {
        "schema": "tac_pose_axis_master_gradient_operator_candidates_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "packet_proofs_available": bool(packet_proofs_available),
        "raw_archive_byte_rows_emitted": 0,
        "raw_archive_byte_coordinates_allowed": False,
        "coordinate_system": "grammar_aware_operator_response",
        "source_selector_schema": selector_payload.get("schema"),
        "source_layout_schema": layout_manifest.get("schema"),
        "source_archive_path": layout_manifest.get("archive_path"),
        "source_archive_sha256": layout_manifest.get("archive_sha256") or selector_payload.get("archive_sha256"),
        "source_archive_bytes": layout_manifest.get("archive_bytes"),
        "logical_grammar": _layout_grammar(layout_manifest),
        "selected_count": len(selected),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "resolved_pose_axis_candidates": resolved,
        "unresolved_pose_axis_entries": unresolved,
        "candidate_modification_spec_count": len(specs),
        "candidate_modification_specs": [spec.to_manifest() for spec in specs],
        "blockers": _unique(blockers),
        "next_step": (
            "run a grammar-specific mutation builder for one resolved spec and "
            "prove repack, ZIP metadata, CRC, inflate success, and byte-consumption "
            "closure before operator probe or provider dispatch"
        ),
    }


def _build_row(
    section: dict[str, Any],
    *,
    axis_label: str,
    packet_proofs_available: bool,
) -> MasterGradientOperatorRow:
    section_name = str(section.get("name", "unknown_section"))
    section_role = str(section.get("role", "unknown_role"))
    feasibility = audit_master_gradient_probe_plan(
        mutation_grain="grammar_aware_operator",
        axis_label=axis_label,
        updates_zip_headers=packet_proofs_available,
        updates_crc=packet_proofs_available,
        repacks_archive=packet_proofs_available,
        proves_inflate_success=packet_proofs_available,
    )
    required = _unique((*PACKET_PROOFS, *feasibility.required_proofs))
    return MasterGradientOperatorRow(
        operator_id=f"{_slug(section_name)}::{_mutation_operator(section_name, section_role)}",
        section_name=section_name,
        section_role=section_role,
        section_offset=_optional_int(section.get("offset")),
        section_len=_optional_int(section.get("len")),
        section_sha256=None if section.get("sha256") is None else str(section["sha256"]),
        mutation_grain="grammar_aware_operator",
        mutation_operator=_mutation_operator(section_name, section_role),
        response_matrix_columns=RESPONSE_MATRIX_COLUMNS,
        axis_label=axis_label,
        required_proofs=required,
        operator_response_valid=feasibility.operator_response_valid,
        ready_for_operator_probe=feasibility.ready_for_operator_probe,
        ready_for_provider_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        dispatch_attempted=False,
        feasibility_verdict=feasibility.verdict,
        blockers=feasibility.blockers,
        rationale=(
            *_operator_rationale(section_name, section_role),
            *feasibility.rationale,
        ),
    )


def _mutation_operator(section_name: str, section_role: str) -> str:
    haystack = f"{section_name} {section_role}".lower()
    if "byte_map" in haystack:
        return "byte_map_metadata_recode"
    if "decoder" in haystack or "op1_inner" in haystack:
        return "decoder_codec_coordinate_response"
    if "latent" in haystack and "sidecar" in haystack:
        return "latent_sidecar_stream_entropy_tournament"
    if "latent" in haystack:
        return "latent_conditioning_codebook_sweep"
    if "sidecar" in haystack or "selector" in haystack:
        return "selector_or_sidecar_table_recode"
    return "logical_section_entropy_tournament"


def _candidate_spec_from_row(
    row: MasterGradientOperatorRow,
    *,
    layout_manifest: dict[str, Any],
    packet_proofs_available: bool,
) -> CandidateModificationSpec:
    archive_bytes = _optional_int(layout_manifest.get("archive_bytes"))
    archive_sha = layout_manifest.get("archive_sha256")
    source_archive_sha256 = None if archive_sha is None else str(archive_sha)
    archive_path = layout_manifest.get("archive_path")
    source_archive_path = None if archive_path is None else str(archive_path)
    return CandidateModificationSpec(
        spec_id=f"candidate_modification::{row.operator_id}",
        source_archive_path=source_archive_path,
        source_archive_sha256=source_archive_sha256,
        source_archive_bytes=archive_bytes,
        operator_id=row.operator_id,
        section_name=row.section_name,
        section_role=row.section_role,
        mutation_grain="grammar_aware_operator",
        mutation_operator=row.mutation_operator,
        axis_label=row.axis_label,
        response_matrix_columns=RESPONSE_MATRIX_COLUMNS,
        packet_proofs_required=PACKET_PROOFS,
        packet_proofs_available=bool(packet_proofs_available),
        ready_for_operator_probe=row.ready_for_operator_probe,
        ready_for_provider_dispatch=False,
        score_claim=False,
        promotion_eligible=False,
        rank_or_kill_eligible=False,
        ready_for_exact_eval_dispatch=False,
        dispatch_attempted=False,
        coordinate_system="grammar_aware_operator_response",
        raw_archive_byte_coordinates_allowed=False,
        blockers=row.blockers,
        rationale=(
            "candidate modification specs are grammar-coordinate rows, not raw archive-byte deltas",
            *row.rationale,
        ),
    )


def _selected_pose_axis_entries(selector_payload: dict[str, Any]) -> list[dict[str, Any]]:
    dominance = selector_payload.get("score_axis_dominance")
    if not isinstance(dominance, dict):
        return []
    selected = dominance.get("selected")
    if not isinstance(selected, list):
        return []
    return [entry for entry in selected if isinstance(entry, dict)]


def _layout_sections(layout_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    logical = layout_manifest.get("logical_layout")
    if isinstance(logical, dict):
        sections = logical.get("sections")
        if isinstance(sections, list):
            return [section for section in sections if isinstance(section, dict)]
    sections = layout_manifest.get("sections")
    if isinstance(sections, list):
        return [section for section in sections if isinstance(section, dict)]
    return []


def _layout_grammar(layout_manifest: dict[str, Any]) -> object:
    logical = layout_manifest.get("logical_layout")
    if isinstance(logical, dict):
        return logical.get("grammar")
    return layout_manifest.get("grammar_name")


def _section_role(section: dict[str, Any]) -> str:
    role = section.get("role")
    if role is not None:
        return str(role)
    codec = section.get("codec")
    if codec is not None:
        return str(codec)
    return "unknown_role"


def _section_span(section: dict[str, Any]) -> tuple[int | None, int | None]:
    offset = _optional_int(section.get("offset"))
    length = _optional_int(section.get("len"))
    if length is None:
        length = _optional_int(section.get("length"))
    if length is None and offset is not None:
        end = _optional_int(section.get("end_offset"))
        if end is not None:
            length = max(0, end - offset)
    return offset, length


def _find_section_for_index(
    sections: list[dict[str, Any]],
    index: int,
) -> dict[str, Any] | None:
    for section in sections:
        offset, length = _section_span(section)
        if offset is None or length is None:
            continue
        if offset <= index < offset + length:
            return section
    return None


def _operator_rationale(section_name: str, section_role: str) -> tuple[str, ...]:
    operator = _mutation_operator(section_name, section_role)
    if operator == "decoder_codec_coordinate_response":
        return ("decoder-like sections can move scorer components and rate; probe as codec coordinates",)
    if operator == "latent_sidecar_stream_entropy_tournament":
        return ("combined latent/sidecar streams need section-conditioned coder tournaments",)
    if operator == "latent_conditioning_codebook_sweep":
        return ("latent sections can test VQ/codebook and quantization trust-region moves",)
    if operator == "selector_or_sidecar_table_recode":
        return ("sidecar or selector bytes need byte-consumption proof before rate wins count",)
    if operator == "byte_map_metadata_recode":
        return ("byte-map metadata can change decode semantics and must be probed as grammar data",)
    return ("unknown logical sections require explicit operator probes rather than raw byte flips",)


def _should_skip_section(section_name: str, section_role: str) -> bool:
    role = section_role.lower()
    name = section_name.lower()
    return (
        role in _SKIP_ROLES
        or role.startswith("raw_")
        or name.endswith("_magic")
        or name.endswith("_u32le")
        or "_len_le_" in name
        or name in {"format_magic", "format_id", "extra_framing_meta"}
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    collapsed = "".join(chars).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed or "unknown_section"


def _unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "PACKET_PROOFS",
    "RESPONSE_MATRIX_COLUMNS",
    "CandidateModificationSpec",
    "MasterGradientOperatorRow",
    "build_master_gradient_operator_plan",
    "build_master_gradient_operator_plan_payload",
    "build_pose_axis_operator_candidates",
]
