# SPDX-License-Identifier: MIT
"""Build the TT5L L5-v2 side-info effect-curve artifact.

The side-info effect curve is a planning/custody artifact, not a score claim.
It answers one narrow architecture-lock question: did the trained TT5L
side-info variant beat the zero/random/shuffled/ablated controls on both
contest CPU and contest CUDA axes, with byte-closed exact-eval custody for
every cell?
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.exact_eval_custody import (
    extract_expected_runtime_tree_sha256,
    extract_observed_runtime_content_tree_sha256,
    validate_exact_eval_evidence,
)
from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_NONZERO_SIDEINFO_VARIANTS,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA,
    validate_l5_v2_sideinfo_effect_curve,
)

L5V2_SIDEINFO_EFFECT_CURVE_PREDICATE_ID = (
    "tt5l_paired_sideinfo_effect_curve_v1"
)
L5V2_SIDEINFO_EFFECT_CURVE_MEASUREMENT_ID = "measure_tt5l_sideinfo_effect_curve"
L5V2_SIDEINFO_EFFECT_CURVE_CONTROL_VARIANTS = tuple(
    variant
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    if variant != "trained"
)
def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cell_evidence(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = cell.get("evidence")
    if isinstance(evidence, Mapping):
        return evidence
    return cell


def _score_from_validation(
    validation_score: float | None,
    evidence: Mapping[str, Any],
) -> float | None:
    if validation_score is not None:
        return validation_score
    value = evidence.get("score")
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _runtime_content_tree_sha256(evidence: Mapping[str, Any]) -> str:
    return extract_observed_runtime_content_tree_sha256(evidence)


def _sideinfo_liveness_for_cell(
    cell: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    for key in (
        "sideinfo_liveness",
        "side_info_liveness",
        "per_pair_side_info_liveness",
        "export_sideinfo_liveness",
    ):
        value = cell.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        value = evidence.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    for outer_key in ("provenance", "runtime_custody", "archive_custody"):
        nested = evidence.get(outer_key)
        if not isinstance(nested, Mapping):
            continue
        value = nested.get("per_pair_side_info_liveness")
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _normalize_cell(
    cell: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    evidence = _cell_evidence(cell)
    axis = str(cell.get("axis") or evidence.get("axis") or "").strip()
    variant = str(cell.get("variant") or cell.get("sideinfo_variant") or "").strip()
    blockers: list[str] = []
    if axis not in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
        blockers.append(f"axis_unrecognized:{axis or '<missing>'}")
    if variant not in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        blockers.append(f"variant_unrecognized:{variant or '<missing>'}")

    validation = validate_exact_eval_evidence(
        evidence,
        expected_axis=axis if axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES else None,
        require_artifact_path=True,
        require_hardware=True,
        require_auth_eval_command=True,
        require_log_path=True,
        require_devices=True,
        require_inflated_outputs_manifest=True,
        require_raw_output_aggregate_sha256=True,
        expected_runtime_tree_sha256=extract_expected_runtime_tree_sha256(evidence),
        artifact_base_dir=repo_root,
    )
    blockers.extend(f"exact_eval_{blocker}" for blocker in validation.blockers)

    return {
        "axis": axis,
        "variant": variant,
        "score": _score_from_validation(validation.score, evidence),
        "seg_dist": validation.seg_dist,
        "pose_dist": validation.pose_dist,
        "archive_bytes": validation.archive_bytes,
        "n_samples": validation.n_samples,
        "archive_sha256": validation.archive_sha256,
        "runtime_tree_sha256": validation.runtime_tree_sha256,
        "expected_runtime_tree_sha256": extract_expected_runtime_tree_sha256(evidence),
        "runtime_content_tree_sha256": _runtime_content_tree_sha256(evidence),
        "hardware": str(evidence.get("hardware") or ""),
        "inflate_device": str(evidence.get("inflate_device") or ""),
        "eval_device": str(evidence.get("eval_device") or ""),
        "auth_eval_command": str(evidence.get("auth_eval_command") or ""),
        "sideinfo_liveness": _sideinfo_liveness_for_cell(cell, evidence),
        "raw_output_aggregate_sha256": str(
            evidence.get("raw_output_aggregate_sha256")
            or evidence.get("inflated_raw_output_aggregate_sha256")
            or ""
        ),
        "artifact_path": str(evidence.get("artifact_path") or ""),
        "log_path": str(evidence.get("log_path") or ""),
        "inflated_outputs_manifest_path": str(
            evidence.get("inflated_outputs_manifest_path")
            or evidence.get("inflated_output_manifest_path")
            or ""
        ),
        "inflated_outputs_manifest_sha256": str(
            evidence.get("inflated_outputs_manifest_sha256")
            or evidence.get("inflated_output_manifest_sha256")
            or ""
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _axis_effects(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    by_axis_variant = {
        (str(row.get("axis") or ""), str(row.get("variant") or "")): row
        for row in rows
    }
    for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
        trained = by_axis_variant.get((axis, "trained"))
        trained_score = trained.get("score") if isinstance(trained, Mapping) else None
        control_scores: dict[str, float] = {}
        for variant in L5V2_SIDEINFO_EFFECT_CURVE_CONTROL_VARIANTS:
            row = by_axis_variant.get((axis, variant))
            value = row.get("score") if isinstance(row, Mapping) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                control_scores[variant] = float(value)
        if isinstance(trained_score, int | float) and not isinstance(
            trained_score,
            bool,
        ) and control_scores:
            best_control_variant, best_control_score = min(
                control_scores.items(),
                key=lambda item: item[1],
            )
            delta = best_control_score - float(trained_score)
            out[axis] = {
                "trained_score": float(trained_score),
                "best_control_variant": best_control_variant,
                "best_control_score": best_control_score,
                "delta_vs_best_control": delta,
                "trained_beats_or_ties_best_control": delta >= -1e-9,
            }
        else:
            out[axis] = {
                "trained_score": trained_score,
                "best_control_variant": "",
                "best_control_score": None,
                "delta_vs_best_control": None,
                "trained_beats_or_ties_best_control": False,
            }
    return out


def build_l5_v2_sideinfo_effect_curve(
    cells: Iterable[Mapping[str, Any]],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return a fail-closed side-info effect-curve artifact."""

    root = Path(repo_root).resolve()
    observed_cells = [
        _normalize_cell(_as_mapping(cell), repo_root=root)
        for cell in cells
    ]
    seen: set[tuple[str, str]] = set()
    duplicate_cells: list[str] = []
    for row in observed_cells:
        key = (str(row.get("axis") or ""), str(row.get("variant") or ""))
        if key in seen:
            duplicate_cells.append(f"{key[0]}/{key[1]}")
        seen.add(key)

    axis_effects = _axis_effects(observed_cells)
    effect_blockers = [
        f"trained_not_best_or_tied:{axis}"
        for axis, effect in axis_effects.items()
        if effect.get("trained_beats_or_ties_best_control") is not True
    ]
    if duplicate_cells:
        effect_blockers.append("duplicate_cells:" + ",".join(sorted(duplicate_cells)))

    payload: dict[str, Any] = {
        "schema": L5V2_SIDEINFO_EFFECT_CURVE_SCHEMA,
        "measurement_id": L5V2_SIDEINFO_EFFECT_CURVE_MEASUREMENT_ID,
        "predicate_id": L5V2_SIDEINFO_EFFECT_CURVE_PREDICATE_ID,
        "predicate_passed": False,
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "observed_cells": observed_cells,
        "axis_effects": axis_effects,
        "effect_blockers": effect_blockers,
    }
    contract_blockers = validate_l5_v2_sideinfo_effect_curve(payload, repo_root=root)
    non_predicate_contract_blockers = [
        blocker
        for blocker in contract_blockers
        if blocker != "tt5l_sideinfo_effect_curve_predicate_not_passed"
    ]
    payload["contract_blockers"] = contract_blockers
    payload["predicate_passed"] = (
        not non_predicate_contract_blockers and not effect_blockers
    )
    if payload["predicate_passed"]:
        payload["contract_blockers"] = validate_l5_v2_sideinfo_effect_curve(
            payload,
            repo_root=root,
        )
        payload["predicate_passed"] = not payload["contract_blockers"]
    return payload


def sideinfo_effect_curve_json(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON text for an effect-curve artifact."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


__all__ = [
    "L5V2_SIDEINFO_EFFECT_CURVE_CONTROL_VARIANTS",
    "L5V2_SIDEINFO_EFFECT_CURVE_MEASUREMENT_ID",
    "L5V2_SIDEINFO_EFFECT_CURVE_NONZERO_SIDEINFO_VARIANTS",
    "L5V2_SIDEINFO_EFFECT_CURVE_PREDICATE_ID",
    "build_l5_v2_sideinfo_effect_curve",
    "sideinfo_effect_curve_json",
]
