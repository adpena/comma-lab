# SPDX-License-Identifier: MIT
"""Scalable-layer admission accounting for receiver-visible SNeRV packets.

This module does not implement a new SNeRV architecture and does not grant
score authority.  It turns a real SNAR1 packet into base/enhancement byte rows
and applies the contest byte price to decide what scorer evidence is missing
before a scalable-layer fork deserves its own lane.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.analysis.nerv_modelsize_budget import RATE_SCORE_PER_BYTE
from tac.analysis.snerv_binary_profile import (
    DEFAULT_FRONTIER_BYTES,
    build_snerv_binary_profile,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "snerv_scalable_layer_admission.v1"
AXIS_TAG = "[planning/control:false-authority]"
SNERV_SCALABLE_LAYER_ADMISSION_PROOF = (
    "snerv_scalable_layer_sections_priced_against_contest_byte_waterline"
)

BASE_LAYER_ID = "snerv_base_lf_layer"
HF_LAYER_ID = "snerv_enhancement_hf_decoder_layer"
STEP_LAYER_ID = "snerv_enhancement_linf_step_map_layer"
HEADER_LAYER_ID = "snerv_packet_header_overhead"


class SnervScalableLayerAdmissionError(ValueError):
    """Raised when scalable-layer admission inputs are invalid."""


def build_snerv_scalable_layer_admission_report(
    *,
    input_path: str | Path,
    layer_nonrate_deltas: Mapping[str, Any] | None = None,
    full_video_coverage: bool = False,
    frontier_bytes: int = DEFAULT_FRONTIER_BYTES,
) -> dict[str, Any]:
    """Return layer byte-price rows for a real SNAR1 packet or archive.zip.

    ``layer_nonrate_deltas`` maps a layer id to the measured non-rate score
    increase caused by removing that layer.  Positive values mean the layer is
    scorer-useful.  Missing values keep the row blocked and non-authoritative.
    """

    profile = build_snerv_binary_profile(
        input_path=input_path,
        frontier_bytes=int(frontier_bytes),
    )
    section_bytes = {
        str(row["section"]): int(row["bytes"])
        for row in profile.get("section_rows", ())
    }
    deltas = _normalize_layer_nonrate_deltas(layer_nonrate_deltas or {})
    layer_rows = [
        _layer_row(
            layer_id=BASE_LAYER_ID,
            layer_role="base_layer_required_for_receiver_decode",
            sections=("metadata_payload", "lf_payload"),
            section_bytes=section_bytes,
            deltas=deltas,
            optional=False,
        ),
        _layer_row(
            layer_id=HF_LAYER_ID,
            layer_role="enhancement_layer_hf_restoration",
            sections=("decoder_payload",),
            section_bytes=section_bytes,
            deltas=deltas,
            optional=True,
        ),
        _layer_row(
            layer_id=STEP_LAYER_ID,
            layer_role="enhancement_layer_linf_step_map",
            sections=("step_map_packet",),
            section_bytes=section_bytes,
            deltas=deltas,
            optional=True,
        ),
        _header_row(profile=profile, deltas=deltas),
    ]
    section_value_rows = _section_value_rows_from_layers(
        layer_rows,
        profile=profile,
        full_video_coverage=bool(full_video_coverage),
    )
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": SCHEMA,
            "family": "snerv",
            "candidate_id": profile.get("snar1_packet_sha256"),
            "axis_tag": AXIS_TAG,
            "section_value_rows": section_value_rows,
            "blockers": [
                "snerv_scalable_layer_byte_price_plan_false_authority",
                "paired_contest_cpu_cuda_auth_eval_missing",
            ],
            **FALSE_AUTHORITY,
        }
    )
    optional_rows = [row for row in layer_rows if row["optional_layer"]]
    evidence_attached = all(
        row["measured_nonrate_score_increase_if_removed"] is not None
        for row in optional_rows
    )
    admitted_optional = [
        row
        for row in optional_rows
        if row["admission_decision"] == "admit_layer_bytes_are_scorer_justified"
    ]
    cut_optional = [
        row
        for row in optional_rows
        if row["admission_decision"] == "cut_or_receiver_generate_layer_candidate"
    ]
    blockers = _ordered_unique(
        [
            "snerv_scalable_layer_admission_false_authority_no_score_claim",
            *(
                []
                if full_video_coverage
                else ["snerv_scalable_layer_full_video_coverage_missing"]
            ),
            *(
                []
                if evidence_attached
                else ["snerv_scalable_layer_section_value_profile_missing"]
            ),
            "paired_contest_cpu_cuda_auth_eval_missing",
        ]
    )
    separate_lane = bool(full_video_coverage and evidence_attached and admitted_optional)
    verdict = (
        "SPLIT_SCALABLE_LAYER_LANE_CANDIDATE__BYTE_PRICED_AND_SCORER_USEFUL"
        if separate_lane
        else (
            "KEEP_AS_SNERV_BITSTREAM_POLICY__CUT_OR_GENERATE_OPTIONAL_LAYERS"
            if full_video_coverage and evidence_attached and cut_optional
            else "KEEP_AS_SNERV_BITSTREAM_POLICY__SECTION_VALUE_EVIDENCE_MISSING"
        )
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "axis_tag": AXIS_TAG,
        "proof": SNERV_SCALABLE_LAYER_ADMISSION_PROOF,
        "family": "snerv",
        "input_path": str(Path(input_path).expanduser().resolve(strict=False)),
        "input_kind": profile["input_kind"],
        "snar1_packet_bytes": int(profile["snar1_packet_bytes"]),
        "charged_archive_bytes": int(profile["charged_archive_bytes"]),
        "frontier_bytes": int(frontier_bytes),
        "rate_score_per_byte": float(RATE_SCORE_PER_BYTE),
        "full_video_coverage": bool(full_video_coverage),
        "section_value_profile_attached": bool(evidence_attached),
        "layer_rows": layer_rows,
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "admitted_optional_layer_count": len(admitted_optional),
        "cut_or_generate_optional_layer_count": len(cut_optional),
        "deserves_separate_scalable_layer_lane": separate_lane,
        "verdict": verdict,
        "next_actions": _next_actions(verdict),
        "source_alignment": {
            "scalable_layer_paper": "base/enhancement layer truncation",
            "contest_adaptation": (
                "layer admission is scorer-only and byte-priced; human quality is "
                "not an authority"
            ),
            "current_snerv_mapping": {
                BASE_LAYER_ID: ["metadata_payload", "lf_payload"],
                HF_LAYER_ID: ["decoder_payload"],
                STEP_LAYER_ID: ["step_map_packet"],
            },
        },
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def write_snerv_scalable_layer_admission_report(
    *,
    input_path: str | Path,
    output_path: str | Path,
    layer_nonrate_deltas: Mapping[str, Any] | None = None,
    full_video_coverage: bool = False,
    frontier_bytes: int = DEFAULT_FRONTIER_BYTES,
) -> dict[str, Any]:
    """Write a scalable-layer admission report."""

    report = build_snerv_scalable_layer_admission_report(
        input_path=input_path,
        layer_nonrate_deltas=layer_nonrate_deltas,
        full_video_coverage=bool(full_video_coverage),
        frontier_bytes=int(frontier_bytes),
    )
    out = Path(output_path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_snerv_scalable_layer_admission_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing report."""

    lines = [
        "# SNeRV Scalable-Layer Admission",
        "",
        f"Schema: `{report['schema']}`",
        f"Verdict: `{report['verdict']}`",
        f"Separate lane: `{report['deserves_separate_scalable_layer_lane']}`",
        "",
        "| layer | bytes | price | measured removal delta | decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.get("layer_rows", ()):
        delta = row.get("measured_nonrate_score_increase_if_removed")
        lines.append(
            "| {layer} | {bytes} | {price:.6f} | {delta} | {decision} |".format(
                layer=row["layer_id"],
                bytes=int(row["layer_bytes"]),
                price=float(row["byte_price_score"]),
                delta="missing" if delta is None else f"{float(delta):.6f}",
                decision=row["admission_decision"],
            )
        )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or [])
    lines.extend(f"- `{blocker}`" for blocker in blockers)
    lines.append("")
    return "\n".join(lines)


def _layer_row(
    *,
    layer_id: str,
    layer_role: str,
    sections: tuple[str, ...],
    section_bytes: Mapping[str, int],
    deltas: Mapping[str, float],
    optional: bool,
) -> dict[str, Any]:
    missing = [section for section in sections if section not in section_bytes]
    if missing:
        raise SnervScalableLayerAdmissionError(
            f"SNAR1 profile missing sections for {layer_id}: {missing}"
        )
    bytes_total = int(sum(int(section_bytes[section]) for section in sections))
    price = float(bytes_total * RATE_SCORE_PER_BYTE)
    measured = deltas.get(layer_id)
    blockers: list[str] = []
    if measured is None and optional:
        blockers.append("scalable_layer_section_value_profile_missing")
    if optional and measured is not None:
        decision = (
            "admit_layer_bytes_are_scorer_justified"
            if float(measured) > price
            else "cut_or_receiver_generate_layer_candidate"
        )
    elif optional:
        decision = "needs_scorer_section_value_profile"
    else:
        decision = "base_layer_required_price_for_training_budget"
    return {
        "layer_id": layer_id,
        "layer_role": layer_role,
        "sections": list(sections),
        "layer_bytes": bytes_total,
        "byte_price_score": price,
        "measured_nonrate_score_increase_if_removed": measured,
        "net_score_saved_if_removed": (
            None if measured is None else float(price - float(measured))
        ),
        "optional_layer": bool(optional),
        "admission_decision": decision,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _header_row(
    *,
    profile: Mapping[str, Any],
    deltas: Mapping[str, float],
) -> dict[str, Any]:
    bytes_total = int(profile.get("snar1_header_bytes") or 0)
    price = float(bytes_total * RATE_SCORE_PER_BYTE)
    measured = deltas.get(HEADER_LAYER_ID)
    return {
        "layer_id": HEADER_LAYER_ID,
        "layer_role": "mandatory_packet_overhead",
        "sections": [],
        "layer_bytes": bytes_total,
        "byte_price_score": price,
        "measured_nonrate_score_increase_if_removed": measured,
        "net_score_saved_if_removed": None,
        "optional_layer": False,
        "admission_decision": "minimize_header_but_not_scalable_content_layer",
        "blockers": [],
        **FALSE_AUTHORITY,
    }


def _section_value_rows_from_layers(
    layer_rows: list[Mapping[str, Any]],
    *,
    profile: Mapping[str, Any],
    full_video_coverage: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    packet_sha = str(profile.get("snar1_packet_sha256") or "")
    archive_sha = str(profile.get("input_sha256") or packet_sha)
    for row in layer_rows:
        if not bool(row.get("optional_layer")):
            continue
        layer_id = str(row.get("layer_id") or "snerv_layer")
        layer_bytes = int(row.get("layer_bytes") or 0)
        measured = row.get("measured_nonrate_score_increase_if_removed")
        blockers = [
            *[str(blocker) for blocker in row.get("blockers") or ()],
            "snerv_scalable_layer_cut_receiver_variant_not_materialized",
            "snerv_scalable_layer_exact_axis_replay_missing",
        ]
        if not full_video_coverage:
            blockers.append("snerv_scalable_layer_full_video_coverage_missing")
        out.append(
            {
                "row_id": f"snerv_scalable_layer_remove_{layer_id}",
                "section_id": layer_id,
                "family": "snerv",
                "scope": "snerv_scalable_layer_existing_section_removal",
                "row_kind": "existing_section_cut",
                "layer_role": row.get("layer_role"),
                "sections": list(row.get("sections") or []),
                "baseline_packet_sha256": packet_sha,
                "archive_sha256": archive_sha,
                "section_bytes": layer_bytes,
                "bytes_removed": layer_bytes,
                "byte_delta": -layer_bytes,
                "delta_nonrate_score": measured,
                "measured_nonrate_score_increase_if_removed": measured,
                "axis_tag": AXIS_TAG,
                "receiver_proof_status": (
                    "baseline_receiver_packet_profiled_variant_cut_missing"
                ),
                "full_video_coverage": bool(full_video_coverage),
                "blockers": _ordered_unique(blockers),
                **FALSE_AUTHORITY,
            }
        )
    return out


def _normalize_layer_nonrate_deltas(values: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            value = (
                value.get("measured_nonrate_score_increase_if_removed")
                or value.get("nonrate_score_increase_if_removed")
                or value.get("delta_nonrate_score_if_removed")
                or value.get("delta_nonrate_score")
            )
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0.0:
            out[str(key)] = number
    return out


def _next_actions(verdict: str) -> list[str]:
    if verdict.endswith("SCORER_USEFUL"):
        return [
            "create_bounded_scalable_layer_lane_with_receiver_closed_layer_sections",
            "train_layer_order_with_section_value_profile_in_loop",
            "run_full600_receiver_replay_then_exact_axis_gate",
        ]
    if "CUT_OR_GENERATE" in verdict:
        return [
            "remove_or_receiver_generate_unjustified_enhancement_sections",
            "rerun_snerv_full600_receiver_archive_and_layer_admission",
            "attach_exact_axis_replay_only_after_nonrate_survives",
        ]
    return [
        "run_full600_section_value_profile_for_hf_decoder_and_step_map_layers",
        "keep_scalable_layer_as_snerv_bitstream_policy_not_separate_lane",
        "price_layer_bytes_inside_scorer_aware_training_export_loop",
    ]


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "BASE_LAYER_ID",
    "HF_LAYER_ID",
    "SCHEMA",
    "SNERV_SCALABLE_LAYER_ADMISSION_PROOF",
    "STEP_LAYER_ID",
    "build_snerv_scalable_layer_admission_report",
    "render_snerv_scalable_layer_admission_markdown",
    "write_snerv_scalable_layer_admission_report",
]
