# SPDX-License-Identifier: MIT
"""Archive/entropy substrate coverage matrix for repair byte transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import (
    ordered_unique,
    require_no_truthy_authority_fields,
)

REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA = (
    "repair_archive_entropy_substrate_coverage.v1"
)

_SUBSTRATE_ORDER: tuple[str, ...] = (
    "fec_variants",
    "zip_member_ordering",
    "header_elision_or_rewrite",
    "selector_streams",
    "range_coding",
    "ans_coding",
    "huffman_coding",
    "pre_coder_distribution_shaping",
    "coder_boundary_recoding",
    "post_coder_legal_repack",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        text = str(value).strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _variant_kinds(variants: Sequence[Mapping[str, Any]]) -> list[str]:
    return ordered_unique(
        str(variant.get("archive_native_transform_kind") or "").strip()
        for variant in variants
        if str(variant.get("archive_native_transform_kind") or "").strip()
    )


def _materialized_kinds(variants: Sequence[Mapping[str, Any]]) -> list[str]:
    return ordered_unique(
        str(variant.get("archive_native_transform_kind") or "").strip()
        for variant in variants
        if variant.get("materialized") is True
        and str(variant.get("archive_native_transform_kind") or "").strip()
    )


def _selected_kind(selected_candidate: Mapping[str, Any]) -> str:
    return str(selected_candidate.get("archive_native_transform_kind") or "").strip()


def _fec_families(detected_families: Sequence[str]) -> list[str]:
    return [
        family
        for family in detected_families
        if family.startswith(("fec", "fes")) or "huffman" in family
    ]


def _row(
    *,
    substrate: str,
    status: str,
    compiler_positions: Sequence[str],
    implemented_transform_kinds: Sequence[str] = (),
    selected_transform_kind: str | None = None,
    detected_archive_families: Sequence[str] = (),
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    row = {
        "schema": "repair_archive_entropy_substrate_coverage_row.v1",
        "substrate": substrate,
        "coverage_status": status,
        "compiler_positions": ordered_unique(compiler_positions),
        "implemented_transform_kinds": ordered_unique(implemented_transform_kinds),
        "selected_transform_kind": selected_transform_kind,
        "detected_archive_families": ordered_unique(detected_archive_families),
        "blockers": ordered_unique(blockers),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "budget_spend_allowed": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        row,
        context=f"repair_archive_entropy_substrate_coverage_row:{substrate}",
    )
    return row


def build_repair_archive_entropy_substrate_coverage(
    *,
    archive_family_probe: Mapping[str, Any],
    candidate_archive_transform_variants: Sequence[Mapping[str, Any]],
    selected_candidate_archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fail-closed coverage matrix over archive entropy substrates."""

    detected = _string_list(archive_family_probe.get("detected_archive_families"))
    variants = [
        variant
        for variant in candidate_archive_transform_variants
        if isinstance(variant, Mapping)
    ]
    kinds = _variant_kinds(variants)
    materialized = _materialized_kinds(variants)
    selected = _selected_kind(selected_candidate_archive)
    fec_detected = _fec_families(detected)
    selector_transform_kinds = [
        kind
        for kind in kinds
        if kind
        in {
            "fec6_selector_payload_mutation",
            "fp11_selector_payload_mutation",
            "psv4_selector_payload_mutation",
        }
    ]
    selector_materialized = [
        kind for kind in materialized if kind in set(selector_transform_kinds)
    ]
    rows = [
        _row(
            substrate="fec_variants",
            status=(
                "materialized"
                if selector_materialized and fec_detected
                else "detected_not_materialized"
                if fec_detected
                else "not_detected"
            ),
            compiler_positions=["before_coder"],
            implemented_transform_kinds=selector_transform_kinds,
            selected_transform_kind=selected if selected in selector_transform_kinds else None,
            detected_archive_families=fec_detected,
            blockers=[] if selector_materialized else ["fec_selector_variant_materialization_missing"],
        ),
        _row(
            substrate="zip_member_ordering",
            status=(
                "materialized_legal_repack"
                if "zip_repack_payload_identity" in materialized
                else "available_not_materialized"
            ),
            compiler_positions=["after_coder"],
            implemented_transform_kinds=["zip_repack_payload_identity"]
            if "zip_repack_payload_identity" in kinds
            else [],
            selected_transform_kind=selected
            if selected == "zip_repack_payload_identity"
            else None,
            detected_archive_families=["zip_container"],
            blockers=[]
            if "zip_repack_payload_identity" in materialized
            else ["zip_repack_payload_identity_not_materialized"],
        ),
        _row(
            substrate="header_elision_or_rewrite",
            status=(
                "materialized_packet_header_rewrite"
                if selected == "psv4_selector_payload_mutation"
                else "available_through_packet_selector_mutation"
                if "psv4_selector_payload_mutation" in kinds
                else "not_detected"
            ),
            compiler_positions=["before_coder", "at_coder_boundary"],
            implemented_transform_kinds=["psv4_selector_payload_mutation"]
            if "psv4_selector_payload_mutation" in kinds
            else [],
            selected_transform_kind=selected
            if selected == "psv4_selector_payload_mutation"
            else None,
            detected_archive_families=[
                family for family in detected if family == "pact_nerv_selector_v4_packet"
            ],
            blockers=[]
            if "psv4_selector_payload_mutation" in materialized
            else ["packet_header_rewrite_materialization_missing"],
        ),
        _row(
            substrate="selector_streams",
            status="materialized" if selector_materialized else "not_materialized",
            compiler_positions=["before_coder"],
            implemented_transform_kinds=selector_transform_kinds,
            selected_transform_kind=selected if selected in selector_transform_kinds else None,
            detected_archive_families=[
                family
                for family in detected
                if "selector" in family or family.startswith(("fec", "fes"))
            ],
            blockers=[]
            if selector_materialized
            else ["selector_stream_materializer_not_selected_or_missing"],
        ),
        _row(
            substrate="range_coding",
            status="not_materialized",
            compiler_positions=["at_coder_boundary"],
            blockers=["range_coder_materializer_missing"],
        ),
        _row(
            substrate="ans_coding",
            status="not_materialized",
            compiler_positions=["at_coder_boundary"],
            blockers=["ans_coder_materializer_missing"],
        ),
        _row(
            substrate="huffman_coding",
            status="materialized_selector_huffman_variant"
            if any("huffman" in family for family in detected) and selector_materialized
            else "detected_not_materialized"
            if any("huffman" in family for family in detected)
            else "not_detected",
            compiler_positions=["before_coder", "at_coder_boundary"],
            implemented_transform_kinds=selector_transform_kinds,
            selected_transform_kind=selected if selected in selector_transform_kinds else None,
            detected_archive_families=[
                family for family in detected if "huffman" in family
            ],
            blockers=[]
            if any("huffman" in family for family in detected) and selector_materialized
            else ["huffman_selector_materializer_missing_or_not_selected"],
        ),
        _row(
            substrate="pre_coder_distribution_shaping",
            status="materialized" if selector_materialized else "not_materialized",
            compiler_positions=["before_coder"],
            implemented_transform_kinds=selector_transform_kinds,
            selected_transform_kind=selected if selected in selector_transform_kinds else None,
            blockers=[]
            if selector_materialized
            else ["pre_coder_distribution_shaping_transform_missing"],
        ),
        _row(
            substrate="coder_boundary_recoding",
            status="materialized"
            if "packet_member_entropy_boundary_recompress" in materialized
            else "available_not_materialized"
            if "packet_member_entropy_boundary_recompress" in kinds
            else "not_available",
            compiler_positions=["at_coder_boundary"],
            implemented_transform_kinds=["packet_member_entropy_boundary_recompress"]
            if "packet_member_entropy_boundary_recompress" in kinds
            else [],
            selected_transform_kind=selected
            if selected == "packet_member_entropy_boundary_recompress"
            else None,
            blockers=[]
            if "packet_member_entropy_boundary_recompress" in materialized
            else ["packet_member_entropy_boundary_recompress_not_materialized"],
        ),
        _row(
            substrate="post_coder_legal_repack",
            status="materialized"
            if "zip_repack_payload_identity" in materialized
            else "available_not_materialized",
            compiler_positions=["after_coder"],
            implemented_transform_kinds=["zip_repack_payload_identity"]
            if "zip_repack_payload_identity" in kinds
            else [],
            selected_transform_kind=selected
            if selected == "zip_repack_payload_identity"
            else None,
            blockers=[]
            if "zip_repack_payload_identity" in materialized
            else ["post_coder_zip_repack_not_materialized"],
        ),
    ]
    by_name = {row["substrate"]: row for row in rows}
    ordered_rows = [by_name[name] for name in _SUBSTRATE_ORDER]
    materialized_substrates = [
        row["substrate"]
        for row in ordered_rows
        if str(row.get("coverage_status") or "").startswith("materialized")
    ]
    coverage = {
        "schema": REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA,
        "substrate_order": list(_SUBSTRATE_ORDER),
        "coverage_row_count": len(ordered_rows),
        "materialized_substrate_count": len(materialized_substrates),
        "materialized_substrates": materialized_substrates,
        "implemented_archive_transform_kinds": kinds,
        "materialized_archive_transform_kinds": materialized,
        "selected_archive_transform_kind": selected or None,
        "detected_archive_families": detected,
        "compiler_position_coverage": {
            "before_coder": any(
                "before_coder" in row["compiler_positions"]
                and str(row.get("coverage_status") or "").startswith("materialized")
                for row in ordered_rows
            ),
            "at_coder_boundary": any(
                "at_coder_boundary" in row["compiler_positions"]
                and str(row.get("coverage_status") or "").startswith("materialized")
                for row in ordered_rows
            ),
            "after_coder": any(
                "after_coder" in row["compiler_positions"]
                and str(row.get("coverage_status") or "").startswith("materialized")
                for row in ordered_rows
            ),
        },
        "rows": ordered_rows,
        "blockers": ordered_unique(
            blocker for row in ordered_rows for blocker in _string_list(row.get("blockers"))
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "budget_spend_allowed": False,
        "allowed_use": "repair_archive_entropy_substrate_planning_coverage_only",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        coverage,
        context="repair_archive_entropy_substrate_coverage",
    )
    return coverage


__all__ = [
    "REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA",
    "build_repair_archive_entropy_substrate_coverage",
]
