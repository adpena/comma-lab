# SPDX-License-Identifier: MIT
"""L5 v2 lattice measurement schedule.

This is a planning artifact, not score evidence. It turns the C1/Z5/TT5L
staircase into a first-match measurement lattice so operators can see which
exact probe closes the next blocker without letting additive score-band
language regain rank authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import validate_exact_eval_evidence
from tac.optimization.l5_v2_probe_disambiguator import L5V2_CANDIDATES

L5V2_MEASUREMENT_SCHEDULE_SCHEMA = "l5_v2_lattice_measurement_schedule_v1"
L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH = (
    "tools/build_l5_v2_lattice_measurement_schedule.py"
)
L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json"
)
L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH = (
    ".omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md"
)
L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA = "l5_v2_sideinfo_effect_curve_v1"
L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH = (
    ".omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json"
)
L5V2_SIDEINFO_EFFECT_CURVE_TOOL_PATH = (
    "tools/build_l5_v2_sideinfo_effect_curve.py"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA = "tt5l_sideinfo_variant_packets_v1"
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH = (
    "tools/build_tt5l_sideinfo_variant_packets.py"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH = (
    ".omx/research/"
    "l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.md"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT = (
    "experiments/results/time_traveler_l5_v2/"
    "tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SOURCE_ARCHIVE_PATH = (
    "experiments/results/time_traveler_l5_v2/"
    "tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/"
    "archive.zip"
)
L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR = (
    "experiments/results/time_traveler_l5_v2/"
    "tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/"
    "submission_dir"
)
L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES = ("contest_cpu", "contest_cuda")
L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS = (
    "zero",
    "random_lsb",
    "shuffled",
    "trained",
    "ablated",
)
L5V2_SIDEINFO_EFFECT_CURVE_NONZERO_SIDEINFO_VARIANTS = frozenset(
    {"random_lsb", "shuffled", "trained"}
)


def _candidate_rows_from_intake(intake: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(intake, Mapping):
        return {}
    verdict = intake.get("verdict")
    if not isinstance(verdict, Mapping):
        return {}
    rows = verdict.get("evaluated_observations")
    if not isinstance(rows, list):
        return {}
    out: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            out[candidate_id] = row
    return out


def _eligible_candidate_ids(intake: Mapping[str, Any] | None) -> set[str]:
    out: set[str] = set()
    for candidate_id, row in _candidate_rows_from_intake(intake).items():
        if row.get("eligible_for_architecture_lock") is True:
            out.add(candidate_id)
    return out


def _candidate_blockers(intake: Mapping[str, Any] | None, candidate_id: str) -> list[str]:
    row = _candidate_rows_from_intake(intake).get(candidate_id)
    blockers = row.get("blockers") if isinstance(row, Mapping) else None
    if not isinstance(blockers, list):
        return ["l5_v2_probe_observation_missing"]
    return [str(item) for item in blockers]


def _as_text_set(value: object) -> set[str]:
    if not isinstance(value, list | tuple | set):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _sideinfo_effect_curve_rows(curve: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(curve, Mapping):
        return []
    for key in ("observed_cells", "rows", "measurements"):
        rows = curve.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    return []


def _looks_sha256(value: object) -> bool:
    text = str(value or "").strip()
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _sha_field(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value).strip().lower() if _looks_sha256(value) else ""


def _text_field(row: Mapping[str, Any], key: str) -> str:
    return str(row.get(key) or "").strip()


def _sideinfo_variant_pair_identity_blockers(
    rows: list[Mapping[str, Any]],
) -> list[str]:
    """Return blockers when CPU/CUDA cells for one variant are not paired.

    The Modal CPU and CUDA wrappers may hash the runtime tree differently
    because the uploaded submission root differs. In that case the runtime
    content tree is the pairing identity. If neither the tree nor content-tree
    identity can prove the two axes used one runtime contract, fail closed.
    """

    blockers: list[str] = []
    by_variant_axis: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        axis = str(row.get("axis") or row.get("device_axis") or "").strip()
        variant = str(row.get("variant") or row.get("sideinfo_variant") or "").strip()
        if not axis or not variant:
            continue
        by_variant_axis.setdefault(variant, {})[axis] = row

    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        axis_rows = by_variant_axis.get(variant, {})
        if any(axis not in axis_rows for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES):
            continue
        for identity_key in ("pair_group_id", "run_id"):
            identities = {
                axis: _text_field(axis_rows[axis], identity_key)
                for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
            }
            if any(not value for value in identities.values()):
                blockers.append(
                    "tt5l_sideinfo_effect_curve_variant_"
                    f"{identity_key}_missing:{variant}"
                )
            elif len(set(identities.values())) != 1:
                blockers.append(
                    "tt5l_sideinfo_effect_curve_variant_"
                    f"{identity_key}_mismatch:{variant}"
                )

        archive_shas = {
            axis: _sha_field(axis_rows[axis], "archive_sha256")
            for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        }
        if any(not digest for digest in archive_shas.values()):
            blockers.append(
                f"tt5l_sideinfo_effect_curve_variant_archive_sha_missing:{variant}"
            )
        elif len(set(archive_shas.values())) != 1:
            blockers.append(
                f"tt5l_sideinfo_effect_curve_variant_archive_sha_mismatch:{variant}"
            )

        runtime_trees = {
            axis: _sha_field(axis_rows[axis], "runtime_tree_sha256")
            for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        }
        runtime_content_trees = {
            axis: _sha_field(axis_rows[axis], "runtime_content_tree_sha256")
            for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        }
        if all(runtime_content_trees.values()):
            if len(set(runtime_content_trees.values())) != 1:
                blockers.append(
                    "tt5l_sideinfo_effect_curve_variant_"
                    f"runtime_content_tree_mismatch:{variant}"
                )
        elif all(runtime_trees.values()) and len(set(runtime_trees.values())) == 1:
            continue
        else:
            blockers.append(
                "tt5l_sideinfo_effect_curve_variant_"
                f"runtime_identity_unpaired:{variant}"
            )
    return blockers


def _positive_int_field(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _nonnegative_int_field(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _sideinfo_liveness_blockers(row: Mapping[str, Any]) -> list[str]:
    axis = str(row.get("axis") or row.get("device_axis") or "").strip()
    variant = str(row.get("variant") or row.get("sideinfo_variant") or "").strip()
    cell = f"{axis or '?'}:{variant or '?'}"
    liveness = row.get("sideinfo_liveness") or row.get("per_pair_side_info_liveness")
    if not isinstance(liveness, Mapping):
        return [f"tt5l_sideinfo_effect_curve_cell_sideinfo_liveness_missing:{cell}"]
    blockers: list[str] = []
    if liveness.get("checked") is not True:
        blockers.append(
            f"tt5l_sideinfo_effect_curve_cell_sideinfo_liveness_unchecked:{cell}"
        )
    total_values = _positive_int_field(liveness, "total_values")
    nonzero_values = _nonnegative_int_field(liveness, "nonzero_values")
    if total_values is None:
        blockers.append(
            f"tt5l_sideinfo_effect_curve_cell_sideinfo_liveness_empty:{cell}"
        )
    if nonzero_values is None:
        blockers.append(
            f"tt5l_sideinfo_effect_curve_cell_sideinfo_nonzero_count_missing:{cell}"
        )
    elif (
        variant in L5V2_SIDEINFO_EFFECT_CURVE_NONZERO_SIDEINFO_VARIANTS
        and nonzero_values <= 0
    ):
        blockers.append(
            f"tt5l_sideinfo_effect_curve_cell_sideinfo_nonzero_missing:{cell}"
        )
    return blockers


def _sideinfo_exact_eval_custody_blockers(
    row: Mapping[str, Any],
    *,
    artifact_base_dir: Path,
) -> list[str]:
    axis = str(row.get("axis") or row.get("device_axis") or "").strip()
    variant = str(row.get("variant") or row.get("sideinfo_variant") or "").strip()
    cell = f"{axis or '?'}:{variant or '?'}"
    validation = validate_exact_eval_evidence(
        row,
        expected_axis=(
            axis if axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES else None
        ),
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_devices=True,
        require_artifact_sha256=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        artifact_base_dir=artifact_base_dir,
    )
    return [
        f"tt5l_sideinfo_effect_curve_cell_exact_eval_{blocker}:{cell}"
        for blocker in validation.blockers
    ]


def _sideinfo_effect_curve_blockers(
    curve: Mapping[str, Any] | None,
    *,
    artifact_base_dir: Path | None = None,
) -> list[str]:
    """Return blockers for the paired TT5L side-info effect-curve contract."""

    if not isinstance(curve, Mapping):
        return ["tt5l_sideinfo_effect_curve_missing"]

    base_dir = artifact_base_dir or Path.cwd()
    blockers: list[str] = []
    if curve.get("schema") != L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA:
        blockers.append("tt5l_sideinfo_effect_curve_schema_mismatch")
    for field in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
        if curve.get(field) is True:
            blockers.append(f"tt5l_sideinfo_effect_curve_{field}_true")
    if curve.get("predicate_passed") is not True:
        blockers.append("tt5l_sideinfo_effect_curve_predicate_not_passed")

    effect_blockers = curve.get("effect_blockers")
    if isinstance(effect_blockers, list):
        for blocker in effect_blockers:
            text = str(blocker).strip()
            if text:
                blockers.append(
                    "tt5l_sideinfo_effect_curve_effect_blocked:" + text
                )
    elif effect_blockers is not None:
        blockers.append("tt5l_sideinfo_effect_curve_effect_blockers_not_list")

    axis_effects = curve.get("axis_effects")
    if not isinstance(axis_effects, Mapping):
        blockers.append("tt5l_sideinfo_effect_curve_axis_effects_missing")
    else:
        required_axis_set = set(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
        observed_axis_effects = {
            str(axis).strip()
            for axis in axis_effects
            if str(axis).strip()
        }
        extra_axis_effects = sorted(observed_axis_effects - required_axis_set)
        if extra_axis_effects:
            blockers.append(
                "tt5l_sideinfo_effect_curve_axis_effects_extra:"
                + ",".join(extra_axis_effects)
            )
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            effect = axis_effects.get(axis)
            if not isinstance(effect, Mapping):
                blockers.append(
                    f"tt5l_sideinfo_effect_curve_axis_effect_missing:{axis}"
                )
            elif effect.get("trained_beats_or_ties_best_control") is not True:
                blockers.append(
                    f"tt5l_sideinfo_effect_curve_trained_not_best_or_tied:{axis}"
                )

    required_axes = set(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES)
    axes = _as_text_set(curve.get("required_axes")) or _as_text_set(
        curve.get("paired_axes_evaluated")
    )
    missing_axes = sorted(required_axes - axes)
    extra_axes = sorted(axes - required_axes)
    if missing_axes:
        blockers.append(
            "tt5l_sideinfo_effect_curve_axes_missing:" + ",".join(missing_axes)
        )
    if extra_axes:
        blockers.append(
            "tt5l_sideinfo_effect_curve_axes_extra:" + ",".join(extra_axes)
        )

    required_variants = set(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
    variants = _as_text_set(curve.get("required_variants"))
    missing_variants = sorted(required_variants - variants)
    extra_variants = sorted(variants - required_variants)
    if missing_variants:
        blockers.append(
            "tt5l_sideinfo_effect_curve_variants_missing:"
            + ",".join(missing_variants)
        )
    if extra_variants:
        blockers.append(
            "tt5l_sideinfo_effect_curve_variants_extra:"
            + ",".join(extra_variants)
        )

    rows = _sideinfo_effect_curve_rows(curve)
    malformed_cells: list[str] = []
    duplicate_cells: list[str] = []
    seen_cells: set[tuple[str, str]] = set()
    for idx, row in enumerate(rows):
        axis = str(row.get("axis") or row.get("device_axis") or "").strip()
        variant = str(row.get("variant") or row.get("sideinfo_variant") or "").strip()
        if not axis or not variant:
            malformed_cells.append(f"{idx}:{axis or '?'}:{variant or '?'}")
            continue
        key = (axis, variant)
        if key in seen_cells:
            duplicate_cells.append(f"{axis}/{variant}")
        seen_cells.add(key)
    if malformed_cells:
        blockers.append(
            "tt5l_sideinfo_effect_curve_cells_malformed:"
            + ",".join(malformed_cells)
        )
    if duplicate_cells:
        blockers.append(
            "tt5l_sideinfo_effect_curve_cells_duplicate:"
            + ",".join(sorted(duplicate_cells))
        )
    observed = {
        (
            str(row.get("axis") or row.get("device_axis") or "").strip(),
            str(row.get("variant") or row.get("sideinfo_variant") or "").strip(),
        )
        for row in rows
    }
    observed_axes = {axis for axis, _variant in observed if axis}
    observed_variants = {variant for _axis, variant in observed if variant}
    extra_observed_axes = sorted(observed_axes - required_axes)
    extra_observed_variants = sorted(observed_variants - required_variants)
    if extra_observed_axes:
        blockers.append(
            "tt5l_sideinfo_effect_curve_observed_axes_extra:"
            + ",".join(extra_observed_axes)
        )
    if extra_observed_variants:
        blockers.append(
            "tt5l_sideinfo_effect_curve_observed_variants_extra:"
            + ",".join(extra_observed_variants)
        )
    missing_cells = sorted(
        f"{axis}/{variant}"
        for axis in required_axes
        for variant in required_variants
        if (axis, variant) not in observed
    )
    if missing_cells:
        blockers.append(
            "tt5l_sideinfo_effect_curve_cells_missing:" + ",".join(missing_cells)
        )
    blockers.extend(_sideinfo_variant_pair_identity_blockers(rows))
    for row in rows:
        axis = str(row.get("axis") or row.get("device_axis") or "").strip()
        variant = str(row.get("variant") or row.get("sideinfo_variant") or "").strip()
        cell = f"{axis or '?'}:{variant or '?'}"
        for field in ("score_claim", "promotion_eligible", "ready_for_exact_eval_dispatch"):
            if row.get(field) is True:
                blockers.append(f"tt5l_sideinfo_effect_curve_cell_{field}_true:{cell}")
        row_blockers = row.get("blockers")
        if isinstance(row_blockers, list) and row_blockers:
            blockers.append(f"tt5l_sideinfo_effect_curve_cell_blocked:{cell}")
            blockers.extend(
                f"tt5l_sideinfo_effect_curve_cell_blocked:{cell}:{blocker}"
                for blocker in row_blockers
                if str(blocker)
            )
        blockers.extend(
            _sideinfo_exact_eval_custody_blockers(row, artifact_base_dir=base_dir)
        )
        blockers.extend(_sideinfo_liveness_blockers(row))
    return list(dict.fromkeys(blockers))


def _sideinfo_effect_curve_required_cells() -> list[dict[str, str]]:
    return [
        {"axis": axis, "variant": variant}
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    ]


def validate_l5_v2_sideinfo_effect_curve(
    curve: Mapping[str, Any] | None,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return blockers for the public TT5L side-info effect-curve contract."""

    base_dir = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    return _sideinfo_effect_curve_blockers(curve, artifact_base_dir=base_dir)


def _measurement(
    *,
    measurement_id: str,
    candidate_id: str,
    purpose: str,
    estimated_cost_usd: float,
    expected_information_gain_nats: float,
    required_axes: tuple[str, ...] = ("contest_cpu", "contest_cuda"),
    output_artifact: str,
    blockers: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "measurement_id": measurement_id,
        "candidate_id": candidate_id,
        "purpose": purpose,
        "estimated_cost_usd": estimated_cost_usd,
        "expected_information_gain_nats": expected_information_gain_nats,
        "required_axes": list(required_axes),
        "output_artifact": output_artifact,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers or [],
    }
    if extra:
        row.update(dict(extra))
    return row


def _probe_measurement_ids_for_missing(candidate_ids: list[str]) -> list[str]:
    measurement_by_candidate = {
        "c1_world_model_foveation": "measure_c1_world_model_foveation_paired_exact",
        "z5_predictive_coding_world_model": "measure_z5_predictive_coding_paired_exact",
        "time_traveler_l5_autonomy": "measure_tt5l_autonomy_paired_exact",
    }
    return [
        measurement_by_candidate[candidate_id]
        for candidate_id in candidate_ids
        if candidate_id in measurement_by_candidate
    ]


def build_l5_v2_lattice_measurement_schedule(
    *,
    probe_intake: Mapping[str, Any] | None = None,
    sideinfo_effect_curve: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the first-match L5 v2 measurement schedule.

    The active rule is computed from probe intake only. Missing or incomplete
    intake routes to paired C1/Z5/TT5L probe filling; fully eligible intake
    routes to the side-info causal effect curve before paired anchor promotion.
    """

    eligible = _eligible_candidate_ids(probe_intake)
    missing_or_blocked = [
        candidate_id for candidate_id in L5V2_CANDIDATES if candidate_id not in eligible
    ]
    base_dir = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    sideinfo_curve_blockers = _sideinfo_effect_curve_blockers(
        sideinfo_effect_curve,
        artifact_base_dir=base_dir,
    )
    sideinfo_curve_valid = not sideinfo_curve_blockers
    measurements = [
        _measurement(
            measurement_id="measure_c1_world_model_foveation_paired_exact",
            candidate_id="c1_world_model_foveation",
            purpose="fill C1 paired CPU/CUDA exact probe observation",
            estimated_cost_usd=2.0,
            expected_information_gain_nats=0.25,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "c1_world_model_foveation_paired_exact.json"
            ),
            blockers=_candidate_blockers(probe_intake, "c1_world_model_foveation"),
        ),
        _measurement(
            measurement_id="measure_z5_predictive_coding_paired_exact",
            candidate_id="z5_predictive_coding_world_model",
            purpose="fill Z5 paired CPU/CUDA exact probe observation",
            estimated_cost_usd=5.0,
            expected_information_gain_nats=0.35,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "z5_predictive_coding_world_model_paired_exact.json"
            ),
            blockers=_candidate_blockers(
                probe_intake, "z5_predictive_coding_world_model"
            ),
        ),
        _measurement(
            measurement_id="measure_tt5l_autonomy_paired_exact",
            candidate_id="time_traveler_l5_autonomy",
            purpose="fill TT5L paired CPU/CUDA exact probe observation",
            estimated_cost_usd=7.5,
            expected_information_gain_nats=0.55,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "time_traveler_l5_autonomy_paired_exact.json"
            ),
            blockers=_candidate_blockers(probe_intake, "time_traveler_l5_autonomy"),
        ),
        _measurement(
            measurement_id="measure_tt5l_sideinfo_effect_curve",
            candidate_id="time_traveler_l5_autonomy",
            purpose=(
                "separate side-info consumption from causal usefulness with "
                "paired CPU/CUDA zero, random-LSB, shuffled, trained, and "
                "ablated side-info"
            ),
            estimated_cost_usd=1.0,
            expected_information_gain_nats=0.40,
            output_artifact=(
                L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
            ),
            extra={
                "sideinfo_effect_curve_builder_tool": (
                    L5V2_SIDEINFO_EFFECT_CURVE_TOOL_PATH
                ),
                "sideinfo_variant_packet_schema": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA
                ),
                "sideinfo_variant_packet_builder_tool": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_TOOL_PATH
                ),
                "sideinfo_variant_packet_manifest_artifact": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_ARTIFACT_PATH
                ),
                "sideinfo_variant_packet_report_artifact": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_REPORT_PATH
                ),
                "sideinfo_variant_packet_output_root": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_OUTPUT_ROOT
                ),
                "sideinfo_variant_packet_source_archive": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SOURCE_ARCHIVE_PATH
                ),
                "sideinfo_variant_packet_submission_dir": (
                    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SUBMISSION_DIR
                ),
                "sideinfo_effect_curve_dispatch_variants": list(
                    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
                ),
                "sideinfo_effect_curve_required_cells": (
                    _sideinfo_effect_curve_required_cells()
                ),
                "sideinfo_effect_curve_aggregate_output_artifact": (
                    L5V2_SIDEINFO_EFFECT_CURVE_ARTIFACT_PATH
                ),
            },
            blockers=(
                []
                if sideinfo_curve_valid
                else list(
                    dict.fromkeys([
                        *sideinfo_curve_blockers,
                        "consumption_proof_is_not_yet_usefulness_proof",
                        (
                            "requires_paired_cpu_cuda_sideinfo_effect_curve_"
                            "before_architecture_lock"
                        ),
                    ])
                )
            ),
        ),
        _measurement(
            measurement_id="prepare_l5_v2_paired_anchor_packet",
            candidate_id="time_traveler_l5_autonomy",
            purpose=(
                "materialize paired-axis anchor packet only after C1/Z5/TT5L "
                "probe observations and side-info effect curve are present"
            ),
            estimated_cost_usd=0.0,
            expected_information_gain_nats=0.10,
            output_artifact=(
                "experiments/results/l5_v2_probe/"
                "tt5l_paired_anchor_packet_manifest.json"
            ),
            blockers=[
                "requires_probe_disambiguator_architecture_lock",
                "requires_terminal_claim_templates",
            ],
        ),
    ]

    rules = [
        {
            "rule_id": "fill_missing_c1_z5_tt5l_probe_observations",
            "condition": "any required candidate lacks paired exact probe eligibility",
            "matches": bool(missing_or_blocked),
            "measurement_ids": _probe_measurement_ids_for_missing(missing_or_blocked),
            "missing_or_blocked_candidates": missing_or_blocked,
        },
        {
            "rule_id": "measure_tt5l_sideinfo_effect_curve",
            "condition": "all required probes eligible but TT5L causal usefulness curve missing",
            "matches": not missing_or_blocked and not sideinfo_curve_valid,
            "measurement_ids": ["measure_tt5l_sideinfo_effect_curve"],
            "missing_or_blocked_candidates": [],
            "sideinfo_effect_curve_blockers": sideinfo_curve_blockers,
        },
        {
            "rule_id": "prepare_paired_anchor_packet",
            "condition": "probe lock and side-info effect curve are both present",
            "matches": not missing_or_blocked and sideinfo_curve_valid,
            "measurement_ids": ["prepare_l5_v2_paired_anchor_packet"],
            "missing_or_blocked_candidates": [],
            "sideinfo_effect_curve_blockers": [],
        },
    ]
    active_rule = next((rule for rule in rules if rule["matches"]), rules[-1])
    return {
        "schema": L5V2_MEASUREMENT_SCHEDULE_SCHEMA,
        "tool": L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH,
        "first_match_wins": True,
        "active_rule_id": active_rule["rule_id"],
        "active_measurement_ids": list(active_rule["measurement_ids"]),
        "required_candidates": list(L5V2_CANDIDATES),
        "eligible_candidates": sorted(eligible),
        "sideinfo_effect_curve_valid": sideinfo_curve_valid,
        "sideinfo_effect_curve_blockers": sideinfo_curve_blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_reward_allowed": False,
        "rules": rules,
        "measurements": measurements,
    }


def render_l5_v2_lattice_measurement_schedule_markdown(
    schedule: Mapping[str, Any],
) -> str:
    """Render a compact operator-facing schedule report."""

    lines = [
        "# L5 v2 lattice measurement schedule",
        "",
        f"- schema: `{schedule.get('schema')}`",
        f"- active_rule_id: `{schedule.get('active_rule_id')}`",
        f"- active_measurement_ids: `{schedule.get('active_measurement_ids')}`",
        f"- sideinfo_effect_curve_valid: `{schedule.get('sideinfo_effect_curve_valid')}`",
        (
            "- sideinfo_effect_curve_blockers: "
            f"`{schedule.get('sideinfo_effect_curve_blockers')}`"
        ),
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "",
        "## Measurements",
    ]
    measurements = schedule.get("measurements")
    if isinstance(measurements, list):
        for row in measurements:
            if not isinstance(row, Mapping):
                continue
            lines.extend(
                [
                    "",
                    f"### {row.get('measurement_id')}",
                    "",
                    f"- candidate_id: `{row.get('candidate_id')}`",
                    f"- purpose: {row.get('purpose')}",
                    f"- required_axes: `{row.get('required_axes')}`",
                    f"- estimated_cost_usd: `{row.get('estimated_cost_usd')}`",
                    "- evidence authority: planning-only until paired exact artifacts land",
                ]
            )
            variants = row.get("sideinfo_effect_curve_dispatch_variants")
            cells = row.get("sideinfo_effect_curve_required_cells")
            if isinstance(variants, list):
                lines.append(f"- sideinfo_effect_curve_dispatch_variants: `{variants}`")
            if isinstance(cells, list):
                lines.append(
                    f"- sideinfo_effect_curve_required_cell_count: `{len(cells)}`"
                )
    return "\n".join(lines) + "\n"


def schedule_json(schedule: Mapping[str, Any]) -> str:
    """Return canonical JSON text for durable artifacts."""

    return json.dumps(schedule, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "L5V2_MEASUREMENT_SCHEDULE_ARTIFACT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_REPORT_PATH",
    "L5V2_MEASUREMENT_SCHEDULE_SCHEMA",
    "L5V2_MEASUREMENT_SCHEDULE_TOOL_PATH",
    "L5V2_SIDEINFO_EFFECT_CURVE_NONZERO_SIDEINFO_VARIANTS",
    "build_l5_v2_lattice_measurement_schedule",
    "render_l5_v2_lattice_measurement_schedule_markdown",
    "schedule_json",
]
