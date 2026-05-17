# SPDX-License-Identifier: MIT
"""Build packet-valid operator rows for master-gradient score-response probes.

The useful replacement for a raw ``(N_archive_bytes, 3)`` derivative is a
manifest of packet-valid mutation operators. Each row names a parser-proven
logical section, a mutation family, the score-response columns to measure, and
the packet proofs required before the row can become a probe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        "skipped_section_count": len(skipped_sections),
        "rows": [row.to_manifest() for row in rows],
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
        "raw_archive_byte_rows_emitted": sum(
            int(plan["raw_archive_byte_rows_emitted"]) for plan in plans
        ),
        "plans": plans,
        "blockers": _unique(blocker for plan in plans for blocker in plan["blockers"]),
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
    return role in _SKIP_ROLES or name.endswith("_magic") or name.endswith("_u32le")


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
    "MasterGradientOperatorRow",
    "build_master_gradient_operator_plan",
    "build_master_gradient_operator_plan_payload",
]
