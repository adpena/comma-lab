#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan PR85 scorer-gradient atom opportunities from existing artifacts.

This tool is compression-time only. It consumes exact auth-eval JSON plus
optional diagnostic component traces and profile JSONs, then emits a
planning-only atom ranking. It does not build archives, load scorers, patch
inflate code, or dispatch GPU/remote work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "experiments/plan_pr85_scorer_gradient_atoms.py"
SCHEMA = "pr85_scorer_gradient_atom_opportunity_v1"
CONTEST_ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_VIDEO_BYTES
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments/results/pr85_scorer_gradient_atoms_20260504_worker/plan.json"
)
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr85_scorer_gradient_atoms_worker_20260504.md"


class PR85GradientPlannerError(ValueError):
    """Raised for malformed explicit planning inputs."""


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PR85GradientPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PR85GradientPlannerError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"stable_plan_digest_sha256"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rel(path: Path, *, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PR85GradientPlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PR85GradientPlannerError(f"{field} must be finite")
    return out


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PR85GradientPlannerError(f"{field} must be an integer")
    return int(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _path_meta(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(path, root=root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def score_from_components(*, seg_dist: float, pose_dist: float, archive_bytes: int) -> float:
    """Recompute the contest score from exact component distances and bytes."""

    if pose_dist < 0.0:
        raise PR85GradientPlannerError("pose_dist must be non-negative")
    return (
        100.0 * seg_dist
        + math.sqrt(10.0 * pose_dist)
        + RATE_SCORE_PER_BYTE * float(archive_bytes)
    )


def score_derivatives(*, pose_dist: float) -> dict[str, float | str]:
    """Return exact local score derivatives at the eval anchor."""

    if pose_dist <= 0.0:
        return {
            "dscore_dseg_dist": 100.0,
            "dscore_dpose_dist": None,
            "dscore_dpose_dist_limit": "positive_infinity",
            "dscore_darchive_byte": RATE_SCORE_PER_BYTE,
            "pose_derivative_status": "undefined_at_nonpositive_pose_dist",
            "derivation": (
                "score = 100*seg + sqrt(10*pose) + 25*bytes/37545489"
            ),
        }
    return {
        "dscore_dseg_dist": 100.0,
        "dscore_dpose_dist": 5.0 / math.sqrt(10.0 * pose_dist),
        "dscore_darchive_byte": RATE_SCORE_PER_BYTE,
        "pose_derivative_status": "finite_at_positive_pose_dist",
        "derivation": "score = 100*seg + sqrt(10*pose) + 25*bytes/37545489",
    }


def _extract_exact_eval(path: Path, *, root: Path) -> dict[str, Any]:
    payload = _read_json(path)
    seg = _finite_float(payload.get("avg_segnet_dist"), field=f"{path}:avg_segnet_dist")
    pose = _finite_float(payload.get("avg_posenet_dist"), field=f"{path}:avg_posenet_dist")
    archive_bytes = _int_value(
        payload.get("archive_size_bytes"),
        field=f"{path}:archive_size_bytes",
    )
    n_samples = _int_value(payload.get("n_samples"), field=f"{path}:n_samples")
    recomputed = score_from_components(
        seg_dist=seg,
        pose_dist=pose,
        archive_bytes=archive_bytes,
    )
    reported = _optional_float(payload.get("score_recomputed_from_components"))
    if reported is None:
        reported = _optional_float(payload.get("canonical_score"))
    if reported is None:
        reported = _optional_float(payload.get("final_score"))
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    return {
        **_path_meta(path, root=root),
        "schema_version": payload.get("schema_version"),
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_bytes,
        "n_samples": n_samples,
        "reported_score": reported,
        "score_recomputed_by_planner": recomputed,
        "formula_abs_error_vs_reported": (
            abs(recomputed - reported) if reported is not None else None
        ),
        "provenance": {
            "tool": provenance.get("tool"),
            "device": provenance.get("device"),
            "gpu_model": provenance.get("gpu_model"),
            "gpu_t4_match": provenance.get("gpu_t4_match"),
            "archive_sha256": provenance.get("archive_sha256"),
            "archive_size_bytes": provenance.get("archive_size_bytes"),
            "runtime_tree_sha256": (
                provenance.get("inflate_runtime_manifest", {}) or {}
            ).get("runtime_tree_sha256"),
        },
    }


def _discover_sibling_trace(exact_eval_path: Path) -> list[Path]:
    sibling = exact_eval_path.parent / "component_trace.json"
    return [sibling] if sibling.exists() else []


def _collect_input_paths(
    paths: Iterable[Path] | None,
    *,
    root: Path,
    exact_eval_path: Path,
    auto_discover: bool,
    pattern: str | None = None,
) -> list[Path]:
    out = []
    if paths:
        out.extend(Path(path) if Path(path).is_absolute() else root / path for path in paths)
    if auto_discover:
        if pattern is None:
            out.extend(_discover_sibling_trace(exact_eval_path))
        else:
            out.extend(root.glob(pattern))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen and path.exists() and path.is_file():
            seen.add(resolved)
            unique.append(path)
    return sorted(unique, key=lambda p: str(p))


def _trace_label(path: Path, payload: dict[str, Any]) -> str:
    trace_inputs = payload.get("trace_inputs")
    if isinstance(trace_inputs, dict):
        label = trace_inputs.get("label") or trace_inputs.get("run_id")
        if isinstance(label, str) and label:
            return label
    return path.parent.name


def _validate_trace(path: Path, payload: dict[str, Any], exact_eval: dict[str, Any]) -> None:
    if payload.get("score_claim") is not False:
        raise PR85GradientPlannerError(f"{path}: component trace must have score_claim=false")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise PR85GradientPlannerError(f"{path}: samples must be a list")
    n_samples = _int_value(payload.get("n_samples"), field=f"{path}:n_samples")
    if n_samples != len(samples):
        raise PR85GradientPlannerError(f"{path}: n_samples does not match samples length")
    if n_samples != exact_eval["n_samples"]:
        raise PR85GradientPlannerError(f"{path}: n_samples differs from exact eval")
    trace_bytes = _int_value(payload.get("archive_size_bytes"), field=f"{path}:archive_size_bytes")
    if trace_bytes != exact_eval["archive_size_bytes"]:
        raise PR85GradientPlannerError(f"{path}: archive bytes differ from exact eval")


def _component_opportunity(
    *,
    pose_dist: float,
    seg_dist: float,
    n_samples: int,
    derivatives: dict[str, Any],
) -> dict[str, Any]:
    pose_grad = derivatives["dscore_dpose_dist"]
    pose = 0.0
    if isinstance(pose_grad, (int, float)) and math.isfinite(float(pose_grad)):
        pose = float(pose_grad) * pose_dist / n_samples
    seg = float(derivatives["dscore_dseg_dist"]) * seg_dist / n_samples
    combined = pose + seg
    return {
        "pose_score_opportunity_to_zero_sample": pose,
        "seg_score_opportunity_to_zero_sample": seg,
        "combined_score_opportunity_to_zero_sample": combined,
    }


def _byte_break_even(score_value: float | None) -> dict[str, Any]:
    if score_value is None:
        return {
            "max_charged_bytes_for_zero_net_change": None,
            "dscore_darchive_byte": RATE_SCORE_PER_BYTE,
            "status": "missing_score_benefit",
        }
    return {
        "max_charged_bytes_for_zero_net_change": max(0.0, score_value) / RATE_SCORE_PER_BYTE,
        "dscore_darchive_byte": RATE_SCORE_PER_BYTE,
        "status": "formula_only_break_even",
    }


def _sensitivity_placeholders() -> dict[str, Any]:
    return {
        "posenet": {
            "status": "placeholder_unresolved",
            "needed_field": "dpose_dist_datom",
            "acceptable_sources": [
                "exact component-response trace",
                "calibrated CUDA finite-difference sensitivity artifact",
            ],
        },
        "segnet": {
            "status": "placeholder_unresolved",
            "needed_field": "dseg_dist_datom",
            "acceptable_sources": [
                "exact component-response trace",
                "calibrated CUDA finite-difference sensitivity artifact",
            ],
        },
    }


def _atom_dispatch_gate(*, trace_cross_checked: bool, has_measured_sensitivity: bool) -> dict[str, Any]:
    blockers = [
        "planner_output_is_not_an_archive",
        "no_remote_or_gpu_dispatch_from_this_tool",
        "closed_archive_exact_cuda_auth_eval_required",
        "raw_output_or_payload_change_must_be_proven_non_noop",
        "posenet_segnet_sensitivity_placeholders_must_be_resolved",
    ]
    if not trace_cross_checked:
        blockers.append("component_trace_missing_or_not_cross_checked")
    if not has_measured_sensitivity:
        blockers.append("no_measured_atom_component_response")
    return {
        "dispatchable": False,
        "status": "blocked_planning_only",
        "blockers": blockers,
    }


def _load_component_trace_atoms(
    path: Path,
    *,
    root: Path,
    exact_eval: dict[str, Any],
    derivatives: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    _validate_trace(path, payload, exact_eval)
    samples = payload["samples"]
    cross = payload.get("contest_auth_eval_cross_check")
    cross_checked = isinstance(cross, dict) and cross.get("all_match") is True
    label = _trace_label(path, payload)
    n_samples = int(payload["n_samples"])
    atoms: list[dict[str, Any]] = []
    for row, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise PR85GradientPlannerError(f"{path}: samples[{row}] must be an object")
        pair_index = _int_value(sample.get("pair_index"), field=f"{path}:samples[{row}].pair_index")
        pose_dist = _finite_float(
            sample.get("posenet_dist"),
            field=f"{path}:samples[{row}].posenet_dist",
        )
        seg_dist = _finite_float(
            sample.get("segnet_dist"),
            field=f"{path}:samples[{row}].segnet_dist",
        )
        frame_indices = sample.get("frame_indices")
        if not isinstance(frame_indices, list) or not all(isinstance(item, int) for item in frame_indices):
            frame_start = sample.get("frame_start")
            if isinstance(frame_start, int):
                frame_indices = [frame_start, frame_start + 1]
            else:
                frame_indices = [2 * pair_index, 2 * pair_index + 1]
        opportunity = _component_opportunity(
            pose_dist=pose_dist,
            seg_dist=seg_dist,
            n_samples=n_samples,
            derivatives=derivatives,
        )
        combined = opportunity["combined_score_opportunity_to_zero_sample"]
        atoms.append(
            {
                "atom_id": f"{label}:pair_{pair_index:04d}",
                "atom_kind": "component_trace_pair_opportunity",
                "source_kind": "component_trace",
                "source_path": _rel(path, root=root),
                "source_sha256": _sha256_file(path),
                "pair_index": pair_index,
                "frame_indices": [int(item) for item in frame_indices],
                "video_name": sample.get("video_name"),
                "video_pair_index": sample.get("video_pair_index"),
                "posenet_dist": pose_dist,
                "segnet_dist": seg_dist,
                **opportunity,
                "byte_break_even": {
                    "combined": _byte_break_even(combined),
                    "pose_only": _byte_break_even(
                        opportunity["pose_score_opportunity_to_zero_sample"]
                    ),
                    "seg_only": _byte_break_even(
                        opportunity["seg_score_opportunity_to_zero_sample"]
                    ),
                },
                "sensitivity": _sensitivity_placeholders(),
                "trust_region_membership": {
                    "anchor": "exact_eval_component_trace_pair",
                    "trace_cross_checked_to_exact_eval": cross_checked,
                    "local_linearization_only": True,
                    "quantitative_radius": None,
                    "radius_status": "not_established_by_this_planner",
                },
                "dispatch_gate": _atom_dispatch_gate(
                    trace_cross_checked=cross_checked,
                    has_measured_sensitivity=False,
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "ranking_score": combined,
            }
        )
    summary = {
        **_path_meta(path, root=root),
        "label": label,
        "n_samples": n_samples,
        "score_claim": payload.get("score_claim"),
        "evidence_grade": payload.get("evidence_grade"),
        "trace_cross_checked_to_exact_eval": cross_checked,
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
    }
    return summary, atoms


def _extract_candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = []
    for key in (
        "ranked_atoms",
        "atoms",
        "opportunities",
        "opportunity_ranking",
        "candidates",
        "top_atoms",
        "atom_ranking",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            containers.append(value)
    allocation = payload.get("atom_allocation_table")
    if isinstance(allocation, dict):
        ranked = allocation.get("ranked_atoms")
        if isinstance(ranked, list):
            containers.append(ranked)
    rows: list[dict[str, Any]] = []
    for container in containers:
        for item in container:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _first_number(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _optional_float(row.get(name))
        if value is not None:
            return value
    return None


def _profile_atom_score(row: dict[str, Any], derivatives: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    explicit = _first_number(
        row,
        (
            "expected_score_saved",
            "score_opportunity",
            "estimated_score_saved",
            "waterfill_utility_score",
        ),
    )
    pose_delta = _first_number(
        row,
        (
            "expected_pose_dist_saved",
            "estimated_pose_dist_saved",
            "delta_pose_dist_saved",
            "pose_dist_saved",
        ),
    )
    seg_delta = _first_number(
        row,
        (
            "expected_seg_dist_saved",
            "estimated_seg_dist_saved",
            "delta_seg_dist_saved",
            "seg_dist_saved",
        ),
    )
    pose_score = None
    pose_gradient = derivatives["dscore_dpose_dist"]
    if (
        pose_delta is not None
        and isinstance(pose_gradient, (int, float))
        and math.isfinite(float(pose_gradient))
    ):
        pose_score = float(pose_gradient) * pose_delta
    seg_score = None
    if seg_delta is not None:
        seg_score = float(derivatives["dscore_dseg_dist"]) * seg_delta
    derived_parts = [part for part in (pose_score, seg_score) if part is not None]
    derived = sum(derived_parts) if derived_parts else None
    score = explicit if explicit is not None else derived
    return score, {
        "explicit_score_saved": explicit,
        "pose_dist_saved": pose_delta,
        "seg_dist_saved": seg_delta,
        "pose_score_saved_from_derivative": pose_score,
        "seg_score_saved_from_derivative": seg_score,
    }


def _load_profile_atoms(
    paths: Iterable[Path],
    *,
    root: Path,
    derivatives: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profile_summaries: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        rows = _extract_candidate_rows(payload)
        profile_summaries.append(
            {
                **_path_meta(path, root=root),
                "schema_version": payload.get("schema_version"),
                "score_claim": payload.get("score_claim"),
                "planning_only": payload.get("planning_only"),
                "candidate_rows_found": len(rows),
            }
        )
        for index, row in enumerate(rows):
            score, score_detail = _profile_atom_score(row, derivatives)
            byte_cost = _first_number(
                row,
                (
                    "charged_bytes",
                    "estimated_charged_bytes",
                    "bytes_at_stake",
                    "byte_cost",
                    "estimated_bytes",
                ),
            )
            byte_delta = _first_number(
                row,
                (
                    "byte_delta_vs_source_archive",
                    "estimated_byte_delta",
                    "delta_bytes",
                    "bytes_saved",
                    "estimated_bytes_saved",
                ),
            )
            atom_id = row.get("atom_id") or row.get("policy_id") or row.get("route_id")
            if not isinstance(atom_id, str) or not atom_id:
                atom_id = f"{path.stem}:profile_atom_{index:04d}"
            has_measured_sensitivity = score_detail["pose_dist_saved"] is not None or score_detail[
                "seg_dist_saved"
            ] is not None
            atoms.append(
                {
                    "atom_id": str(atom_id),
                    "atom_kind": str(row.get("atom_kind") or row.get("family") or "profile_atom"),
                    "source_kind": "profile",
                    "source_path": _rel(path, root=root),
                    "source_sha256": _sha256_file(path),
                    "pair_index": _optional_int(row.get("pair_index")),
                    "frame_indices": row.get("frame_indices")
                    if isinstance(row.get("frame_indices"), list)
                    else None,
                    "profile_score_detail": score_detail,
                    "estimated_score_saved": score,
                    "estimated_charged_bytes": byte_cost,
                    "estimated_byte_delta": byte_delta,
                    "byte_break_even": {
                        "combined": _byte_break_even(score),
                        "declared_byte_cost": byte_cost,
                    },
                    "sensitivity": (
                        {
                            "posenet": {
                                "status": "provided_or_derivable"
                                if score_detail["pose_dist_saved"] is not None
                                else "placeholder_unresolved",
                                "value": score_detail["pose_dist_saved"],
                            },
                            "segnet": {
                                "status": "provided_or_derivable"
                                if score_detail["seg_dist_saved"] is not None
                                else "placeholder_unresolved",
                                "value": score_detail["seg_dist_saved"],
                            },
                        }
                    ),
                    "trust_region_membership": {
                        "anchor": "external_profile_artifact",
                        "local_linearization_only": True,
                        "quantitative_radius": None,
                        "radius_status": "must_be_supplied_by_profile_or_response_curve",
                    },
                    "dispatch_gate": _atom_dispatch_gate(
                        trace_cross_checked=False,
                        has_measured_sensitivity=has_measured_sensitivity,
                    ),
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ranking_score": score if score is not None else float("-inf"),
                }
            )
    return profile_summaries, atoms


def _rank_atoms(atoms: list[dict[str, Any]], *, max_atoms: int) -> list[dict[str, Any]]:
    ranked = sorted(
        atoms,
        key=lambda atom: (
            float(atom.get("ranking_score", float("-inf"))),
            str(atom.get("source_kind")),
            str(atom.get("atom_id")),
        ),
        reverse=True,
    )
    return ranked[:max_atoms]


def _overall_dispatch_gates(*, trace_count: int, profile_count: int) -> dict[str, Any]:
    blockers = [
        "planning_only_schema",
        "compression_time_only",
        "no_archive_built",
        "no_remote_or_gpu_dispatch_performed",
        "exact_cuda_auth_eval_required_for_any_candidate_archive",
        "inflate_time_scorer_load_forbidden",
        "component_gate_review_required_before_promotion",
    ]
    if trace_count == 0:
        blockers.append("missing_component_trace_planner_uses_formula_and_profiles_only")
    if profile_count == 0:
        blockers.append("missing_profile_artifacts_sensitivity_placeholders_unresolved")
    return {
        "dispatchable": False,
        "status": "blocked_planning_only",
        "blockers": blockers,
    }


def _trust_regions(exact_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_anchor": {
            "avg_posenet_dist": exact_eval["avg_posenet_dist"],
            "avg_segnet_dist": exact_eval["avg_segnet_dist"],
            "archive_size_bytes": exact_eval["archive_size_bytes"],
            "n_samples": exact_eval["n_samples"],
        },
        "allowed_use": [
            "compression-time atom ranking",
            "byte break-even accounting",
            "candidate-builder prioritization",
        ],
        "forbidden_use": [
            "score claim",
            "rank claim",
            "promotion",
            "retirement",
            "remote dispatch without a separate closed-archive gate",
            "inflate-time scorer load",
        ],
        "linearization": {
            "kind": "first_order_score_gradient_at_exact_eval_anchor",
            "quantitative_radius": None,
            "radius_status": "not_established_by_this_planner",
            "required_to_establish_radius": [
                "exact component-response curve",
                "CUDA finite-difference sensitivity artifact",
                "closed-archive exact eval for stacked candidate",
            ],
        },
    }


def build_plan(
    *,
    exact_eval_json: Path,
    component_trace_jsons: Iterable[Path] = (),
    profile_jsons: Iterable[Path] = (),
    root: Path = REPO_ROOT,
    auto_discover_sibling_trace: bool = True,
    auto_discover_pr85_profiles: bool = False,
    max_atoms: int = 64,
) -> dict[str, Any]:
    """Build a deterministic planning-only scorer-gradient atom plan."""

    if max_atoms <= 0:
        raise PR85GradientPlannerError("max_atoms must be positive")
    exact_eval_path = exact_eval_json if exact_eval_json.is_absolute() else root / exact_eval_json
    exact_eval = _extract_exact_eval(exact_eval_path, root=root)
    derivatives = score_derivatives(pose_dist=exact_eval["avg_posenet_dist"])
    trace_paths = _collect_input_paths(
        component_trace_jsons,
        root=root,
        exact_eval_path=exact_eval_path,
        auto_discover=auto_discover_sibling_trace,
    )
    profile_paths = _collect_input_paths(
        profile_jsons,
        root=root,
        exact_eval_path=exact_eval_path,
        auto_discover=auto_discover_pr85_profiles,
        pattern="experiments/results/**/profile_pr85*.json",
    )

    trace_summaries: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    for path in trace_paths:
        summary, trace_atoms = _load_component_trace_atoms(
            path,
            root=root,
            exact_eval=exact_eval,
            derivatives=derivatives,
        )
        trace_summaries.append(summary)
        atoms.extend(trace_atoms)
    profile_summaries, profile_atoms = _load_profile_atoms(
        profile_paths,
        root=root,
        derivatives=derivatives,
    )
    atoms.extend(profile_atoms)
    ranked_atoms = _rank_atoms(atoms, max_atoms=max_atoms)
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": PRODUCER,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "compression_time_only": True,
        "inflate_time_scorer_load_allowed": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "exact_eval": exact_eval,
        "formula_checks": {
            "score_formula": "100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
            "score_recomputed_by_planner": exact_eval["score_recomputed_by_planner"],
            "reported_score": exact_eval["reported_score"],
            "abs_error_vs_reported": exact_eval["formula_abs_error_vs_reported"],
            "dscore_dseg_expected": 100.0,
            "dscore_dpose_expected": "5/sqrt(10*pose_dist)",
            "dscore_dbyte_expected": RATE_SCORE_PER_BYTE,
        },
        "score_derivatives": derivatives,
        "input_artifacts": {
            "component_traces": trace_summaries,
            "profiles": profile_summaries,
        },
        "trust_regions": _trust_regions(exact_eval),
        "dispatch_gates": _overall_dispatch_gates(
            trace_count=len(trace_summaries),
            profile_count=len(profile_summaries),
        ),
        "atom_ranking": ranked_atoms,
        "atom_ranking_schema": {
            "ranking_score": "first-order score opportunity before charged-byte cost",
            "byte_break_even": (
                "max charged bytes an atom can spend before its first-order score benefit "
                "is fully consumed by the exact rate derivative"
            ),
            "sensitivity": "PoseNet/SegNet component-response fields or explicit placeholders",
            "dispatch_gate": "always blocked in this planner; builder/eval gate must reopen separately",
        },
        "planning_state": (
            "component_trace_ranked"
            if trace_summaries
            else "planning_only_no_component_trace"
        ),
    }
    plan["stable_plan_digest_sha256"] = _stable_digest(plan)
    return plan


def write_plan(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    exact = payload["exact_eval"]
    derivatives = payload["score_derivatives"]
    top = payload["atom_ranking"][:8]
    lines = [
        "# PR85 Scorer-Gradient Atom Opportunity Profiler",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- compression_time_only: true",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        "- inflate_time_scorer_load_allowed: false",
        "",
        "## Formula Checks",
        "",
        "- score = 100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
        f"- dscore/dseg = {derivatives['dscore_dseg_dist']}",
        f"- dscore/dpose = {derivatives['dscore_dpose_dist']}",
        f"- dscore/dbyte = {derivatives['dscore_darchive_byte']}",
        f"- recomputed_score = {exact['score_recomputed_by_planner']}",
        f"- reported_score = {exact['reported_score']}",
        f"- abs_error_vs_reported = {exact['formula_abs_error_vs_reported']}",
        "",
        "## Inputs",
        "",
        f"- exact_eval_json: {exact['path']}",
        f"- component_traces: {len(payload['input_artifacts']['component_traces'])}",
        f"- profiles: {len(payload['input_artifacts']['profiles'])}",
        "",
        "## Top Atoms",
        "",
    ]
    if not top:
        lines.append("- none; planner is formula/profile-only until a component trace is supplied")
    for atom in top:
        combined = atom.get("combined_score_opportunity_to_zero_sample")
        if combined is None:
            combined = atom.get("estimated_score_saved")
        break_even = atom.get("byte_break_even", {}).get("combined", {})
        lines.append(
            "- "
            f"{atom['atom_id']}: score_opportunity={combined}, "
            f"break_even_bytes={break_even.get('max_charged_bytes_for_zero_net_change')}, "
            f"gate={atom['dispatch_gate']['status']}"
        )
    lines.extend(
        [
            "",
            "## Dispatch Gates",
            "",
            f"- status: {payload['dispatch_gates']['status']}",
            "- required next proof: build a closed archive candidate, prove non-noop payload/raw-output change, then run exact CUDA auth eval.",
            "",
        ]
    )
    return "\n".join(lines)


def write_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(payload), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-eval-json", type=Path, required=True)
    parser.add_argument("--component-trace-json", type=Path, action="append", default=[])
    parser.add_argument("--profile-json", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument(
        "--no-auto-sibling-trace",
        action="store_true",
        help="Do not automatically read component_trace.json beside the exact eval JSON.",
    )
    parser.add_argument(
        "--auto-discover-pr85-profiles",
        action="store_true",
        help="Read profile_pr85*.json artifacts under experiments/results.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write the JSON plan to stdout instead of --output-json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = build_plan(
            exact_eval_json=args.exact_eval_json,
            component_trace_jsons=args.component_trace_json,
            profile_jsons=args.profile_json,
            auto_discover_sibling_trace=not args.no_auto_sibling_trace,
            auto_discover_pr85_profiles=args.auto_discover_pr85_profiles,
            max_atoms=args.max_atoms,
        )
        if args.stdout:
            sys.stdout.write(_json_dumps(plan))
        else:
            write_plan(args.output_json, plan)
        write_ledger(args.ledger_md, plan)
    except PR85GradientPlannerError as exc:
        print(f"{PRODUCER}: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
