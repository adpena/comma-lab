# SPDX-License-Identifier: MIT
"""Acquisition rows for the HPRC representation spine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import contest_rate_term
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.bitstream_grammar import (
    build_optimal_bitstream_grammar_plan,
)
from tac.substrates.hprc.representation_spine import HprcRepresentationFamily
from tac.substrates.hprc.resolution_contract import CONTEST_PAIR_COUNT

HPRC_SPINE_ACQUISITION_REPORT_SCHEMA = "hprc_spine_acquisition_report.v1"
DEFAULT_BASE_RENDERER_BYTE_CEILINGS: tuple[int, ...] = (178_000, 216_000, 285_000)
_PRIMARY_CARRIER_FAMILIES = frozenset(
    {
        HprcRepresentationFamily.PR95_HNERV.value,
        HprcRepresentationFamily.HNERV_PACKED.value,
        HprcRepresentationFamily.RNERV.value,
        HprcRepresentationFamily.BOOST_NERV.value,
        HprcRepresentationFamily.PACT_NERV.value,
        HprcRepresentationFamily.SR_NERV.value,
    }
)
_STACK_ROLES: dict[str, dict[str, Any]] = {
    HprcRepresentationFamily.PR95_HNERV.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": "control_floor_tiny_decoder_plus_latents",
    },
    HprcRepresentationFamily.HNERV_PACKED.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": "control_floor_tiny_decoder_plus_latents",
    },
    HprcRepresentationFamily.RNERV.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": "sweep_under_hard_ceiling_before_residual_sidecar",
    },
    HprcRepresentationFamily.BOOST_NERV.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": (
            "boosted_compact_base_must_beat_pr95_scale_byte_value_before_residuals"
        ),
    },
    HprcRepresentationFamily.PACT_NERV.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": "sweep_under_hard_ceiling_before_residual_sidecar",
    },
    HprcRepresentationFamily.PACT_NERV_VQ.value: {
        "position": "latent_codebook_base_or_residual_codec",
        "byte_policy": "admit_codebook_index_bytes_only_when_replay_value_per_byte_wins",
    },
    HprcRepresentationFamily.TREE_NERV.value: {
        "position": "temporal_layout_policy",
        "byte_policy": "nonuniform_temporal_spend_selected_by_value_per_byte",
    },
    HprcRepresentationFamily.HI_NERV.value: {
        "position": "hierarchical_feature_policy",
        "byte_policy": "multi_level_features_selected_by_value_per_byte",
    },
    HprcRepresentationFamily.SR_NERV.value: {
        "position": "primary_learned_receiver_carrier",
        "byte_policy": "lowres_carrier_must_pay_for_upsampler_and_latents",
    },
    HprcRepresentationFamily.VQ_NERV.value: {
        "position": "latent_codebook_base_or_residual_codec",
        "byte_policy": "indices_and_codebooks_charged_against_replay_gain",
    },
    HprcRepresentationFamily.PVQ_NERV.value: {
        "position": "latent_codebook_base_or_residual_codec",
        "byte_policy": "gain_shape_codebook_indices_charged_against_replay_gain",
    },
    HprcRepresentationFamily.RT_VQ_NERV.value: {
        "position": "residual_token_vq_base_or_residual_codec",
        "byte_policy": (
            "residual_tokens_admitted_only_by_full_video_value_per_byte"
        ),
    },
    HprcRepresentationFamily.SIREN_IMPLICIT.value: {
        "position": "implicit_residual_or_procedural_atom",
        "byte_policy": "atom_bytes_admitted_only_as_measured_residual_value",
    },
    HprcRepresentationFamily.FINER_IMPLICIT.value: {
        "position": "implicit_residual_or_procedural_atom",
        "byte_policy": "atom_bytes_admitted_only_as_measured_residual_value",
    },
    HprcRepresentationFamily.C3_COOL_CHIC.value: {
        "position": "latent_codebook_codec",
        "byte_policy": "entropy_coded_latent_grid_competes_with_decoder_weight_bytes",
    },
    HprcRepresentationFamily.PROCEDURAL_DRIVING_PRIOR.value: {
        "position": "charged_procedural_driving_prior",
        "byte_policy": "replace_latents_only_when_all_constants_state_are_archive_charged",
    },
}


def build_spine_acquisition_report(
    *,
    projection_manifest_paths: Sequence[str | Path],
    hard_byte_ceilings: Sequence[int] = DEFAULT_BASE_RENDERER_BYTE_CEILINGS,
) -> dict[str, Any]:
    """Build contract-first acquisition rows from spine projections."""

    ceilings = tuple(sorted({int(value) for value in hard_byte_ceilings if int(value) > 0}))
    if not ceilings:
        raise ValueError("at least one positive hard byte ceiling is required")
    rows = []
    for raw_path in projection_manifest_paths:
        path = Path(raw_path).expanduser().resolve(strict=False)
        projection = _load_json_object(path)
        rows.append(_projection_row(path=path, projection=projection, ceilings=ceilings))
    rows = sorted(rows, key=lambda row: (row["effective_archive_bytes"], row["family"]))
    optimal_grammar = build_optimal_bitstream_grammar_plan(
        acquisition_rows=rows,
        hard_byte_ceilings=ceilings,
    )
    return {
        "schema": HPRC_SPINE_ACQUISITION_REPORT_SCHEMA,
        "hard_byte_ceilings": list(ceilings),
        "row_count": len(rows),
        "rows": rows,
        "optimal_bitstream_grammar": optimal_grammar,
        "best_under_each_ceiling": {
            str(ceiling): _best_under_ceiling(rows, ceiling) for ceiling in ceilings
        },
        "residual_admission_rule": {
            "schema": "hprc_residual_admission_rule.v1",
            "rule": "admit residual bytes only when measured_delta_nonrate + rate_cost < 0",
            "missing_measurement_action": "block_or_route_to_mlx_replay_not_exact_promotion",
            "exact_axis_required_for_promotion": True,
        },
        "stack_policy": {
            "schema": "hprc_optimal_stack_policy.v1",
            "primary_carrier_rule": (
                "prefer PR95/HNeRV-scale, RNeRV, or PACT-NeRV learned receivers "
                "as the base because explicit residual fields are rate-heavy"
            ),
            "temporal_layout_rule": (
                "Tree/Hi/SR/VQ-NeRV modes are swappable charged policies, not "
                "separate custody surfaces"
            ),
            "codebook_rule": (
                "C3/Cool-Chic/VQ sections compete on charged codebook plus index "
                "bytes and replayed scorer value"
            ),
            "implicit_atom_rule": (
                "SIREN/FINER/WIRE/BACON-style atoms are residual/procedural "
                "candidates until byte-value profiles prove full-carrier status"
            ),
            "procedural_prior_rule": (
                "ego-motion/lane/road priors may replace latents only when every "
                "constant and state byte is archive-charged"
            ),
            "residual_sidecar_rule": (
                "HPRC/Z8 residual tokens are admitted only when full-video P18/P19 "
                "non-rate improvement exceeds contest rate cost"
            ),
        },
        **FALSE_AUTHORITY,
    }


def _projection_row(
    *,
    path: Path,
    projection: dict[str, Any],
    ceilings: tuple[int, ...],
) -> dict[str, Any]:
    projection_body = projection.get("projection")
    body = projection_body if isinstance(projection_body, dict) else projection
    family = str(body.get("family") or projection.get("family") or "unknown")
    stack_role = _STACK_ROLES.get(family, {"position": "unknown", "byte_policy": "fail_closed"})
    manifest = body.get("manifest") if isinstance(body.get("manifest"), dict) else {}
    spine_manifest = (
        manifest.get("representation_spine")
        if isinstance(manifest.get("representation_spine"), dict)
        else {}
    )
    source = spine_manifest.get("source") if isinstance(spine_manifest.get("source"), dict) else {}
    hprc_bytes = int(body.get("hprc_bin_bytes") or 0)
    source_bytes = _positive_int(source.get("bytes"))
    effective_bytes = int(source_bytes or hprc_bytes)
    sections = spine_manifest.get("sections") if isinstance(spine_manifest.get("sections"), list) else []
    section_rows = [_section_row(raw) for raw in sections if isinstance(raw, dict)]
    residual_rows = [row for row in section_rows if row["name"] == "residual_rc"]
    manifest_extra = (
        spine_manifest.get("manifest_extra")
        if isinstance(spine_manifest.get("manifest_extra"), dict)
        else {}
    )
    coverage = _coverage_row(manifest_extra)
    smallest_ceiling = min(ceilings)
    largest_ceiling = max(ceilings)
    return {
        "schema": "hprc_spine_acquisition_row.v1",
        "projection_manifest_path": path.as_posix(),
        "family": family,
        "representation_source_payload_kind": manifest_extra.get("source_payload_kind"),
        "representation_side_channel_kind": manifest_extra.get("side_channel_kind"),
        "representation_payload_magic": manifest_extra.get("payload_magic"),
        "representation_emitter": manifest_extra.get("emitted_by"),
        "stack_role": stack_role,
        "effective_archive_bytes_source": "source_archive" if source_bytes else "hprc_projection",
        "effective_archive_bytes": effective_bytes,
        "effective_rate_term": contest_rate_term(effective_bytes),
        "hprc_projection_bytes": hprc_bytes,
        "source_archive": source,
        "coverage": coverage,
        "capacity_control_surface": _capacity_control_surface(
            family=family,
            manifest_extra=manifest_extra,
            section_rows=section_rows,
            effective_archive_bytes=effective_bytes,
        ),
        "ceiling_results": [
            _ceiling_result(effective_bytes=effective_bytes, ceiling=ceiling)
            for ceiling in ceilings
        ],
        "recommended_next_action": _recommended_next_action(
            effective_bytes=effective_bytes,
            family=family,
            coverage_valid_for_base_comparison=coverage["valid_for_base_comparison"],
            smallest_ceiling=smallest_ceiling,
            largest_ceiling=largest_ceiling,
        ),
        "section_rows": section_rows,
        "residual_section_admission": [
            {
                "section": row["name"],
                "bytes": row["bytes"],
                "rate_cost": row["rate_cost"],
                "required_measured_nonrate_improvement": -row["rate_cost"],
                "status": "measurement_required_before_admission",
            }
            for row in residual_rows
        ],
        **FALSE_AUTHORITY,
    }


def _section_row(raw: dict[str, Any]) -> dict[str, Any]:
    byte_count = int(raw.get("bytes") or 0)
    return {
        "schema": "hprc_spine_section_acquisition_row.v1",
        "name": str(raw.get("name") or ""),
        "role": str(raw.get("role") or ""),
        "bytes": byte_count,
        "sha256": raw.get("sha256"),
        "rate_cost": contest_rate_term(byte_count),
        "requires_value_measurement": True,
    }


def _capacity_control_surface(
    *,
    family: str,
    manifest_extra: dict[str, Any],
    section_rows: list[dict[str, Any]],
    effective_archive_bytes: int,
) -> dict[str, Any]:
    """Expose archive-rate controls for learned receiver capacity sweeps."""

    knob_keys = (
        "modelsize",
        "model_size",
        "latent_dim",
        "latent_dim_coarse",
        "latent_dim_mid",
        "latent_dim_fine",
        "embed_dim",
        "codebook_size",
        "decoder_channel",
        "decoder_channels",
        "num_layers",
        "depth",
        "width",
    )
    declared_knobs = {
        key: manifest_extra[key]
        for key in knob_keys
        if key in manifest_extra
    }
    section_byte_controls = {
        row["name"]: {
            "bytes": row["bytes"],
            "rate_cost": row["rate_cost"],
            "role": row["role"],
        }
        for row in section_rows
        if row["name"] in {"decoder_qw", "latents_rc", "codebooks_q", "selectors_rc"}
    }
    if declared_knobs:
        status = "declared_capacity_knobs_ready_for_hard_ceiling_sweep"
        next_action = "sweep_declared_capacity_knobs_under_hard_byte_ceilings"
    elif section_byte_controls:
        status = "archive_section_byte_controls_only_modelsize_knob_not_declared"
        next_action = (
            "recover_or_add_modelsize_capacity_knob_before_family_level_verdict"
        )
    else:
        status = "capacity_control_surface_missing"
        next_action = "add_archive_charged_capacity_controls_before_budget_spend"
    return {
        "schema": "hprc_capacity_control_surface.v1",
        "family": family,
        "effective_archive_bytes": int(effective_archive_bytes),
        "declared_capacity_knobs": declared_knobs,
        "declared_capacity_knob_count": len(declared_knobs),
        "section_byte_controls": section_byte_controls,
        "section_byte_control_count": len(section_byte_controls),
        "hard_byte_ceiling_sweep_required": True,
        "rate_variable": "archive.zip bytes under contest score term 25*bytes/N",
        "status": status,
        "next_action": next_action,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _ceiling_result(*, effective_bytes: int, ceiling: int) -> dict[str, Any]:
    over = int(effective_bytes) - int(ceiling)
    return {
        "schema": "hprc_hard_byte_ceiling_result.v1",
        "ceiling_bytes": int(ceiling),
        "fits": over <= 0,
        "excess_bytes": max(0, over),
        "slack_bytes": max(0, -over),
        "rate_term_at_ceiling": contest_rate_term(int(ceiling)),
    }


def _coverage_row(manifest_extra: dict[str, Any]) -> dict[str, Any]:
    num_pairs = _positive_int(manifest_extra.get("num_pairs"))
    if num_pairs is not None and num_pairs < CONTEST_PAIR_COUNT:
        return {
            "schema": "hprc_representation_coverage_gate.v1",
            "declared_pairs": num_pairs,
            "required_pairs": CONTEST_PAIR_COUNT,
            "valid_for_base_comparison": False,
            "promotion_blocker": "declared_pair_coverage_below_full_video",
        }
    return {
        "schema": "hprc_representation_coverage_gate.v1",
        "declared_pairs": num_pairs,
        "required_pairs": CONTEST_PAIR_COUNT,
        "valid_for_base_comparison": True,
        "promotion_blocker": None,
    }


def _recommended_next_action(
    *,
    effective_bytes: int,
    family: str,
    coverage_valid_for_base_comparison: bool,
    smallest_ceiling: int,
    largest_ceiling: int,
) -> str:
    if not coverage_valid_for_base_comparison:
        return "scale_or_train_to_full_600_pair_coverage_before_base_byte_comparison"
    if effective_bytes <= smallest_ceiling:
        if family not in _PRIMARY_CARRIER_FAMILIES:
            return "run_full_replay_then_admit_only_if_value_per_byte_beats_primary_carrier"
        return "run_full_replay_then_exact_gate_before_residual_bytes"
    if effective_bytes <= largest_ceiling:
        return "train_or_recode_under_stricter_frontier_byte_ceiling"
    return "shrink_base_renderer_before_any_residual_sidecar"


def _best_under_ceiling(rows: list[dict[str, Any]], ceiling: int) -> dict[str, Any] | None:
    passing = [
        row
        for row in rows
        if int(row["effective_archive_bytes"]) <= int(ceiling)
        and bool(row["coverage"]["valid_for_base_comparison"])
    ]
    if not passing:
        return None
    best = min(passing, key=lambda row: (int(row["effective_archive_bytes"]), row["family"]))
    return {
        "family": best["family"],
        "projection_manifest_path": best["projection_manifest_path"],
        "effective_archive_bytes": best["effective_archive_bytes"],
        "effective_rate_term": best["effective_rate_term"],
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = [
    "DEFAULT_BASE_RENDERER_BYTE_CEILINGS",
    "HPRC_SPINE_ACQUISITION_REPORT_SCHEMA",
    "build_spine_acquisition_report",
]
