#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build C067 atom-response taxonomy tables from existing artifacts.

This is a deterministic, non-GPU synthesis tool. It reads already-produced
exact-eval JSON, diagnostic component traces, and planner JSON files, then
emits no-score-claim JSON/Markdown tables for deciding the next high-EV eval.
Missing component measurements stay unknown; planner estimates are never
upgraded into score evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
CONTEST_UNCOMPRESSED_BYTES = 37_545_489
RATE_SLOPE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES

DEFAULT_BASELINE_JSON = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/"
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_DIR = Path("experiments/results/c067_atom_response_table_20260503")

DEFAULT_EXACT_LABEL_FRAGMENTS = (
    "exact_eval_c063_fixedslice_equiv",
    "c067",
    "sjkl",
    "pmg",
    "cmg",
    "micro_mask",
    "postdecode",
    "decoded",
    "hotspot",
    "multimask",
    "imp_c067",
    "qzs3_b512",
    "fixedslice_pr67",
)

DEFAULT_PLANNER_GLOBS = (
    "experiments/results/c067_decoded_delta_overlay_mask_topology_20260502/*.json",
    "experiments/results/c067_reversed_base_cdo1_overlay_economics_20260502/*.json",
    "experiments/results/c067_micro_mask_reencode_plan_20260502/*.json",
    "experiments/results/c067_postdecode_mask_repair_candidate_20260502/*plan.json",
    "experiments/results/c067_cmg3a_body200_atom_field_20260502/*.json",
    "experiments/results/c067_yousfi_fridrich_field_equations_20260502/*.json",
    "experiments/results/sjkl_trace_benefit_allocator_20260502/*.json",
)


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + RATE_SLOPE_SCORE_PER_BYTE * archive_bytes


def _score_terms(seg: float | None, pose: float | None, archive_bytes: int | None) -> dict[str, float | None]:
    if seg is None or pose is None or archive_bytes is None:
        return {
            "score_recomputed_from_components": None,
            "score_seg_contribution": None,
            "score_pose_contribution": None,
            "score_rate_contribution": None,
        }
    return {
        "score_recomputed_from_components": _score(seg, pose, archive_bytes),
        "score_seg_contribution": 100.0 * seg,
        "score_pose_contribution": math.sqrt(10.0 * pose),
        "score_rate_contribution": RATE_SLOPE_SCORE_PER_BYTE * archive_bytes,
    }


def classify_family(label: str, path: str) -> str:
    text = f"{label} {path}".lower()
    if "sjkl" in text:
        return "SJ-KL residual"
    if "cdo1" in text or "decoded_delta" in text or "overlay" in text:
        return "CDO1 decoded-mask overlay"
    if "postdecode" in text:
        return "postdecode mask repair"
    if "micro_mask" in text or "micro-mask" in text:
        return "micro mask reencode"
    if "pmg" in text or "predictive_mask_grammar" in text:
        return "PMG hotspot mask grammar"
    if "cmg" in text or "yousfi_fridrich" in text or "yf_field" in text:
        return "CMG mask grammar"
    if "multimask" in text:
        return "multimask reconciliation"
    if "imp_c067" in text or "qzs3_b512" in text or "blockfp" in text:
        return "renderer compression"
    if "fixedslice" in text:
        return "fixed-slice segment mix"
    if "hotspot" in text:
        return "hotspot geometry"
    return "other"


def _discover_exact_eval_paths(root: Path, *, baseline_json: Path) -> list[Path]:
    candidates: dict[Path, Path] = {}
    search_roots = [
        root / "experiments/results/lightning_batch",
        root / "experiments/results",
    ]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.glob("*/contest_auth_eval.adjudicated.json")):
            parent_text = path.parent.as_posix().lower()
            if any(fragment in parent_text for fragment in DEFAULT_EXACT_LABEL_FRAGMENTS):
                candidates[path.parent.resolve()] = path
        for path in sorted(search_root.glob("*/contest_auth_eval.json")):
            parent_text = path.parent.as_posix().lower()
            if not any(fragment in parent_text for fragment in DEFAULT_EXACT_LABEL_FRAGMENTS):
                continue
            candidates.setdefault(path.parent.resolve(), path)

    resolved_baseline = baseline_json.resolve()
    candidates.setdefault(resolved_baseline.parent.resolve(), resolved_baseline)
    return sorted(candidates.values(), key=lambda p: _rel(p, root=root))


def _discover_planner_paths(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in DEFAULT_PLANNER_GLOBS:
        for path in root.glob(pattern):
            if path.name.endswith(".policy.json"):
                continue
            if path.is_file():
                paths.add(path)
    return sorted(paths, key=lambda p: _rel(p, root=root))


def _archive_sha_from_payload(payload: dict[str, Any]) -> str | None:
    prov = payload.get("provenance") or {}
    trace_inputs = payload.get("trace_inputs") or {}
    return (
        prov.get("archive_sha256")
        or trace_inputs.get("archive_sha256")
        or payload.get("archive_sha256")
    )


def _load_adjudication(path: Path) -> dict[str, Any]:
    adj_path = path.with_name("adjudication_provenance.json")
    if adj_path.exists():
        try:
            payload = _json_load(adj_path)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def load_exact_eval(path: Path, *, root: Path) -> dict[str, Any]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    prov = payload.get("provenance") or {}
    archive_bytes = _maybe_int(payload.get("archive_size_bytes") or prov.get("archive_size_bytes"))
    seg = _finite_float(payload.get("avg_segnet_dist"))
    pose = _finite_float(payload.get("avg_posenet_dist"))
    terms = _score_terms(seg, pose, archive_bytes)
    score = _finite_float(payload.get("score_recomputed_from_components"))
    if score is None:
        score = terms["score_recomputed_from_components"]
    label = path.parent.name
    component_trace_path = path.with_name("component_trace.json")
    adjudication = _load_adjudication(path)
    runtime_manifest = prov.get("inflate_runtime_manifest") or {}
    return {
        "artifact_kind": "exact_eval",
        "label": label,
        "family": classify_family(label, _rel(path, root=root)),
        "artifact_path": _rel(path, root=root),
        "artifact_sha256": _sha256_file(path),
        "component_trace_path": (
            _rel(component_trace_path, root=root) if component_trace_path.exists() else None
        ),
        "component_trace_sha256": (
            _sha256_file(component_trace_path) if component_trace_path.exists() else None
        ),
        "archive_bytes": archive_bytes,
        "archive_sha256": _archive_sha_from_payload(payload),
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "n_samples": _maybe_int(payload.get("n_samples")),
        "score_recomputed_from_components": score,
        "score_recomputed_by_table": terms["score_recomputed_from_components"],
        "score_recompute_abs_error": (
            abs(score - terms["score_recomputed_from_components"])
            if score is not None and terms["score_recomputed_from_components"] is not None
            else None
        ),
        "score_seg_contribution": terms["score_seg_contribution"],
        "score_pose_contribution": terms["score_pose_contribution"],
        "score_rate_contribution": terms["score_rate_contribution"],
        "device": prov.get("device"),
        "gpu_model": prov.get("gpu_model"),
        "gpu_t4_match": prov.get("gpu_t4_match"),
        "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
        "source_evidence_grade": adjudication.get("evidence_grade"),
        "source_promotion_eligible": adjudication.get("promotion_eligible"),
    }


def _delta(a: float | int | None, b: float | int | None) -> float | int | None:
    if a is None or b is None:
        return None
    return a - b


def _byte_delta_class(delta_bytes: int | None) -> str:
    if delta_bytes is None:
        return "unknown"
    if delta_bytes < 0:
        return "byte_saving"
    if delta_bytes > 0:
        return "byte_regressive"
    return "byte_neutral"


def _component_collapse(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    collapse_ratio: float,
    pose_abs_threshold: float,
    seg_abs_threshold: float,
) -> list[str] | None:
    pose = candidate.get("avg_posenet_dist")
    seg = candidate.get("avg_segnet_dist")
    base_pose = baseline.get("avg_posenet_dist")
    base_seg = baseline.get("avg_segnet_dist")
    if pose is None or seg is None or base_pose is None or base_seg is None:
        return None
    collapsed: list[str] = []
    if pose > base_pose and pose >= max(pose_abs_threshold, base_pose * collapse_ratio):
        collapsed.append("PoseNet")
    if seg > base_seg and seg >= max(seg_abs_threshold, base_seg * collapse_ratio):
        collapsed.append("SegNet")
    return collapsed


def _component_status(row: dict[str, Any]) -> str:
    if row.get("collapse_components") is None:
        return "unknown"
    if row.get("collapse_components"):
        return "collapse"
    nonrate = row.get("delta_nonrate_score")
    if nonrate is None:
        return "unknown"
    if nonrate < -1e-12:
        return "component_positive"
    if nonrate > 1e-12:
        return "component_negative"
    return "component_neutral"


def _exact_evidence_grade(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    if row.get("device") != "cuda":
        return "invalid"
    if row.get("archive_sha256") == baseline.get("archive_sha256") and row.get("gpu_t4_match") is True:
        return "A++"
    if row.get("gpu_t4_match") is True:
        if row.get("score_delta_vs_baseline") is not None and row["score_delta_vs_baseline"] > 1e-12:
            return "A-negative"
        if row.get("collapse_components"):
            return "A-negative"
        return "A++"
    return "B"


def _exact_next_action(row: dict[str, Any]) -> str:
    if row.get("is_baseline_reference"):
        return "anchor_reference_only"
    if row.get("component_status") == "unknown":
        return "do_not_rank_missing_component_fields"
    if row.get("collapse_components"):
        if row.get("family") in {"PMG hotspot mask grammar", "CMG mask grammar", "micro mask reencode", "postdecode mask repair"}:
            return "do_not_dispatch_without_geometry_escape_and_component_guard"
        return "do_not_repeat_before_collapse_review"
    score_delta = row.get("score_delta_vs_baseline")
    if score_delta is not None and score_delta < -1e-12:
        if row.get("gpu_t4_match") is True:
            return "preserve_custody_candidate_already_t4_exact"
        return "queue_t4_confirmation_only_if_identical_archive_custody_holds"
    if row.get("component_status") == "component_positive" and row.get("byte_delta_class") == "byte_regressive":
        if row.get("family") == "SJ-KL residual":
            return "shrink_payload_or_coefficient_only_then_exact_eval"
        return "reduce_rate_cost_before_more_exact_eval"
    if row.get("byte_delta_class") == "byte_saving" and row.get("component_status") != "component_positive":
        return "treat_as_exact_negative_redesign_geometry"
    return "do_not_repeat_configuration_without_new_atom_rationale"


def enrich_exact_eval(
    record: dict[str, Any],
    *,
    baseline: dict[str, Any],
    collapse_ratio: float,
    pose_abs_threshold: float,
    seg_abs_threshold: float,
) -> dict[str, Any]:
    row = dict(record)
    delta_bytes = _delta(row.get("archive_bytes"), baseline.get("archive_bytes"))
    delta_seg = _delta(row.get("avg_segnet_dist"), baseline.get("avg_segnet_dist"))
    delta_pose = _delta(row.get("avg_posenet_dist"), baseline.get("avg_posenet_dist"))
    delta_score = _delta(
        row.get("score_recomputed_from_components"),
        baseline.get("score_recomputed_from_components"),
    )
    delta_rate = (
        RATE_SLOPE_SCORE_PER_BYTE * delta_bytes if isinstance(delta_bytes, int | float) else None
    )
    delta_seg_score = 100.0 * delta_seg if delta_seg is not None else None
    delta_pose_score = _delta(row.get("score_pose_contribution"), baseline.get("score_pose_contribution"))
    delta_nonrate = (
        delta_seg_score + delta_pose_score
        if delta_seg_score is not None and delta_pose_score is not None
        else None
    )
    row.update(
        {
            "is_baseline_reference": (
                row.get("archive_sha256") is not None
                and row.get("archive_sha256") == baseline.get("archive_sha256")
                and row.get("archive_bytes") == baseline.get("archive_bytes")
            ),
            "byte_delta_vs_baseline": delta_bytes,
            "byte_delta_class": _byte_delta_class(delta_bytes if isinstance(delta_bytes, int) else None),
            "avg_segnet_delta_vs_baseline": delta_seg,
            "avg_posenet_delta_vs_baseline": delta_pose,
            "score_delta_vs_baseline": delta_score,
            "delta_rate_score": delta_rate,
            "delta_seg_score": delta_seg_score,
            "delta_pose_score": delta_pose_score,
            "delta_nonrate_score": delta_nonrate,
            "collapse_components": _component_collapse(
                candidate=row,
                baseline=baseline,
                collapse_ratio=collapse_ratio,
                pose_abs_threshold=pose_abs_threshold,
                seg_abs_threshold=seg_abs_threshold,
            ),
            "score_claim": False,
            "promotion_eligible": False,
        }
    )
    row["component_status"] = _component_status(row)
    row["evidence_grade"] = _exact_evidence_grade(row, baseline)
    row["next_action"] = _exact_next_action(row)
    return row


def _planner_source_grade(payload: dict[str, Any]) -> str:
    grade = payload.get("evidence_grade")
    if isinstance(grade, str) and grade:
        return grade
    if payload.get("score_claim") is False:
        return "empirical"
    return "unknown"


def _normalized_planner_grade(source_grade: str) -> str:
    grade = source_grade.lower()
    if "planning" in grade or "empirical" in grade:
        return "empirical"
    if grade == "unknown":
        return "unknown"
    return source_grade


def _first_list_item(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = payload.get(key)
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return None


def _planner_byte_info(payload: dict[str, Any], *, baseline: dict[str, Any]) -> dict[str, Any]:
    best = _first_list_item(payload, "best_candidates") or _first_list_item(payload, "all_candidates")
    if best:
        estimated = best.get("estimated_archive") or {}
        delta = _maybe_int(estimated.get("estimated_delta_vs_c067"))
        if delta is not None:
            return {
                "byte_delta_vs_baseline": delta,
                "byte_delta_class": _byte_delta_class(delta),
                "byte_delta_kind": "estimated_archive_delta",
                "archive_bytes": _maybe_int(estimated.get("estimated_archive_bytes")),
                "byte_note": "planner estimated archive bytes; component response unknown",
            }

    screen = _first_list_item(payload, "local_byte_screens")
    if screen:
        delta = _maybe_int(screen.get("delta_bytes") or screen.get("archive_delta_bytes"))
        if delta is not None:
            return {
                "byte_delta_vs_baseline": delta,
                "byte_delta_class": _byte_delta_class(delta),
                "byte_delta_kind": "empirical_local_archive_delta",
                "archive_bytes": _maybe_int(screen.get("archive_bytes")),
                "byte_note": "local byte-screen archive delta; component response unknown",
            }

    measured = _first_list_item(payload, "measured_candidate_byte_screen")
    if measured:
        archive_bytes = _maybe_int(measured.get("archive_size_bytes"))
        base_bytes = _maybe_int(baseline.get("archive_bytes"))
        if archive_bytes is not None and base_bytes is not None:
            delta = archive_bytes - base_bytes
            return {
                "byte_delta_vs_baseline": delta,
                "byte_delta_class": _byte_delta_class(delta),
                "byte_delta_kind": "empirical_local_archive_delta",
                "archive_bytes": archive_bytes,
                "byte_note": "local byte-screen archive bytes; component response unknown",
            }

    config = _first_list_item(payload, "candidate_configs")
    if config:
        byte_screen = config.get("byte_screen") or {}
        target = _maybe_int(byte_screen.get("target_archive_bytes"))
        base = _maybe_int(byte_screen.get("baseline_archive_bytes") or baseline.get("archive_bytes"))
        if target is not None and base is not None:
            delta = target - base
            return {
                "byte_delta_vs_baseline": delta,
                "byte_delta_class": _byte_delta_class(delta),
                "byte_delta_kind": "planner_target_archive_delta",
                "archive_bytes": target,
                "byte_note": "planner target archive bytes; component response unknown",
            }

    budget = _first_list_item(payload, "budget_policies")
    if budget:
        return {
            "byte_delta_vs_baseline": None,
            "byte_delta_class": "unknown",
            "byte_delta_kind": "estimated_payload_bytes_only",
            "archive_bytes": None,
            "estimated_payload_bytes": _finite_float(budget.get("estimated_payload_bytes")),
            "byte_note": "payload estimate only; archive byte delta unknown",
        }

    return {
        "byte_delta_vs_baseline": None,
        "byte_delta_class": "unknown",
        "byte_delta_kind": "unknown",
        "archive_bytes": None,
        "byte_note": "no archive byte delta found",
    }


def _planner_next_action(row: dict[str, Any], payload: dict[str, Any]) -> str:
    decision = str(payload.get("decision") or "").lower()
    family = row["family"]
    if family == "CDO1 decoded-mask overlay":
        if "geometry_blocked" in decision or "no_safe" in decision:
            return "do_not_dispatch_until_residual_geometry_gate_and_runtime_closure"
        return "build_byte_closed_runtime_then_exact_cuda_before_claim"
    if family == "SJ-KL residual":
        return "use_positive_trace_pairs_for_payload_shrink_then_exact_eval"
    if family == "micro mask reencode":
        return "local_trust_region_byte_build_only_until_protected_agreement"
    if family == "postdecode mask repair":
        return "build_selected_policy_locally_then_exact_cuda_required"
    if family == "CMG mask grammar":
        return "build_and_byte_screen_no_gpu_from_planner_alone"
    return "keep_planning_only_until_closed_archive_exact_eval"


def load_planner(path: Path, *, root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    label = path.stem
    source_grade = _planner_source_grade(payload)
    row = {
        "artifact_kind": "planner",
        "label": label,
        "family": classify_family(label, _rel(path, root=root)),
        "artifact_path": _rel(path, root=root),
        "artifact_sha256": _sha256_file(path),
        "score_claim": False,
        "promotion_eligible": False,
        "source_score_claim": payload.get("score_claim"),
        "source_promotion_eligible": payload.get("promotion_eligible"),
        "source_evidence_grade": source_grade,
        "evidence_grade": _normalized_planner_grade(source_grade),
        "component_status": "unknown",
        "collapse_components": None,
        "avg_segnet_delta_vs_baseline": None,
        "avg_posenet_delta_vs_baseline": None,
        "score_delta_vs_baseline": None,
        "delta_nonrate_score": None,
        "decision": payload.get("decision"),
        "required_next_steps": payload.get("required_next_steps")
        or payload.get("required_before_remote_eval")
        or payload.get("required_before_remote_eval"),
        "candidate_count": payload.get("candidate_count"),
        "dispatchable_candidate_count": payload.get("dispatchable_candidate_count"),
        "remote_jobs_dispatched": payload.get("remote_jobs_dispatched"),
        "cuda_jobs_launched": payload.get("cuda_jobs_launched"),
    }
    row.update(_planner_byte_info(payload, baseline=baseline))
    row["next_action"] = _planner_next_action(row, payload)
    return row


def _grade_sort_key(grade: str) -> tuple[int, str]:
    order = {
        "A++": 0,
        "A": 1,
        "A-negative": 2,
        "B": 3,
        "empirical": 4,
        "derivation": 5,
        "prediction": 6,
        "external": 7,
        "invalid": 8,
        "unknown": 9,
    }
    return (order.get(grade, 10), grade)


def _best_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get(key) is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row[key]), str(row.get("label"))))


def _family_next_action(family: str, rows: Sequence[dict[str, Any]], status: str) -> str:
    if family == "SJ-KL residual" and status == "component_positive_byte_regressive":
        return "highest_ev_next: shrink_sjkl_payload_or_coefficient_only_then_exact_eval"
    if family in {
        "PMG hotspot mask grammar",
        "CMG mask grammar",
        "micro mask reencode",
        "postdecode mask repair",
        "multimask reconciliation",
        "renderer compression",
        "hotspot geometry",
    } and "collapse" in status:
        return "do_not_dispatch_same_family_without_geometry_escape_proof"
    if family == "CDO1 decoded-mask overlay":
        return "do_not_dispatch_until_residual_geometry_gate_runtime_closure_and_byte_screen_pass"
    if status == "score_positive_measured":
        return "preserve_custody_and_confirm_on_t4_if_needed"
    if status == "diagnostic_score_positive_needs_t4":
        return "do_not_promote_without_matching_t4_confirmation"
    if status == "planning_only":
        return "build_byte_closed_candidate_before_exact_eval"
    if any(row.get("component_status") == "component_positive" for row in rows):
        return "use_component_positive_atoms_but_fix_rate_before_eval"
    return "lower_ev_until_new_component_or_geometry_evidence"


def build_family_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)

    family_rows: list[dict[str, Any]] = []
    for family in sorted(grouped):
        items = sorted(grouped[family], key=lambda row: (row["artifact_kind"], row["label"]))
        exact = [row for row in items if row["artifact_kind"] == "exact_eval" and not row.get("is_baseline_reference")]
        planner = [row for row in items if row["artifact_kind"] == "planner"]
        best_score = _best_by(exact, "score_delta_vs_baseline")
        best_bytes = _best_by(items, "byte_delta_vs_baseline")
        best_nonrate = _best_by(exact, "delta_nonrate_score")
        collapse_count = sum(1 for row in exact if row.get("collapse_components"))
        component_positive_byte_regressive = [
            row
            for row in exact
            if row.get("component_status") == "component_positive"
            and row.get("byte_delta_class") == "byte_regressive"
        ]
        score_positive_t4 = [
            row
            for row in exact
            if row.get("gpu_t4_match") is True
            and row.get("score_delta_vs_baseline") is not None
            and row["score_delta_vs_baseline"] < 0
        ]
        score_positive_diag = [
            row
            for row in exact
            if row.get("gpu_t4_match") is not True
            and row.get("score_delta_vs_baseline") is not None
            and row["score_delta_vs_baseline"] < 0
        ]
        if score_positive_t4:
            status = "score_positive_measured"
        elif score_positive_diag:
            status = "diagnostic_score_positive_needs_t4"
        elif component_positive_byte_regressive:
            status = "component_positive_byte_regressive"
        elif collapse_count:
            status = "collapse_or_exact_negative"
        elif exact:
            status = "exact_negative_or_neutral"
        else:
            status = "planning_only"
        grades = sorted({str(row.get("evidence_grade") or "unknown") for row in items}, key=_grade_sort_key)
        exact_count = len(exact)
        best_score_delta = best_score.get("score_delta_vs_baseline") if best_score else None
        best_byte_delta = best_bytes.get("byte_delta_vs_baseline") if best_bytes else None
        best_nonrate_delta = best_nonrate.get("delta_nonrate_score") if best_nonrate else None
        evidence_grade = ",".join(grades)
        family_rows.append(
            {
                "family": family,
                "artifact_count": len(items),
                "exact_eval_count": exact_count,
                "exact_count": exact_count,
                "planner_count": len(planner),
                "status": status,
                "classification": status,
                "collapse_artifact_count": collapse_count,
                "collapse_count": collapse_count,
                "unknown_component_artifact_count": sum(
                    1 for row in items if row.get("component_status") == "unknown"
                ),
                "evidence_grades": grades,
                "evidence_grade": evidence_grade,
                "best_score_delta_vs_baseline": best_score_delta,
                "best_score_delta": best_score_delta,
                "best_score_delta_label": best_score.get("label") if best_score else None,
                "best_byte_delta_vs_baseline": best_byte_delta,
                "best_byte_delta": best_byte_delta,
                "best_byte_delta_label": best_bytes.get("label") if best_bytes else None,
                "best_nonrate_delta_vs_baseline": best_nonrate_delta,
                "best_nonrate_delta": best_nonrate_delta,
                "best_nonrate_delta_label": best_nonrate.get("label") if best_nonrate else None,
                "next_action": _family_next_action(family, items, status),
            }
        )
    return family_rows


def build_c067_atom_response_table(
    *,
    root: Path,
    baseline_json: Path,
    exact_jsons: Sequence[Path] | None = None,
    planner_jsons: Sequence[Path] | None = None,
    include_default_exact_scan: bool = True,
    include_default_planner_scan: bool = True,
    collapse_ratio: float = 10.0,
    pose_abs_threshold: float = 0.01,
    seg_abs_threshold: float = 0.01,
) -> dict[str, Any]:
    root = root.resolve()
    baseline_json = _resolve_path(baseline_json, root=root)
    baseline = load_exact_eval(baseline_json, root=root)

    exact_paths: list[Path] = []
    if include_default_exact_scan:
        exact_paths.extend(_discover_exact_eval_paths(root, baseline_json=baseline_json))
    if exact_jsons:
        exact_paths.extend(_resolve_path(path, root=root) for path in exact_jsons)
    exact_paths.append(baseline_json)
    exact_paths = sorted({path.resolve(): path for path in exact_paths if path.exists()}.values(), key=lambda p: _rel(p, root=root))

    planner_paths: list[Path] = []
    if include_default_planner_scan:
        planner_paths.extend(_discover_planner_paths(root))
    if planner_jsons:
        planner_paths.extend(_resolve_path(path, root=root) for path in planner_jsons)
    planner_paths = sorted({path.resolve(): path for path in planner_paths if path.exists()}.values(), key=lambda p: _rel(p, root=root))

    exact_rows = [
        enrich_exact_eval(
            load_exact_eval(path, root=root),
            baseline=baseline,
            collapse_ratio=collapse_ratio,
            pose_abs_threshold=pose_abs_threshold,
            seg_abs_threshold=seg_abs_threshold,
        )
        for path in exact_paths
    ]
    planner_rows = [load_planner(path, root=root, baseline=baseline) for path in planner_paths]
    artifact_rows = sorted(
        exact_rows + planner_rows,
        key=lambda row: (row["family"], row["artifact_kind"], row["label"], row["artifact_path"]),
    )
    return {
        "schema": "c067_atom_response_table_v1",
        "schema_version": SCHEMA_VERSION,
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_score_source_required": "exact CUDA auth eval JSON remains the only score authority",
        "notes": [
            "This table does not dispatch jobs and makes no score claim.",
            "Planner-only byte estimates never supply SegNet/PoseNet component deltas.",
            "Positive byte_delta_vs_baseline means larger than the C067/Apogee baseline.",
        ],
        "baseline": {
            key: baseline.get(key)
            for key in (
                "label",
                "artifact_path",
                "artifact_sha256",
                "archive_bytes",
                "archive_sha256",
                "avg_segnet_dist",
                "avg_posenet_dist",
                "score_recomputed_from_components",
                "n_samples",
                "device",
                "gpu_model",
                "gpu_t4_match",
                "runtime_tree_sha256",
            )
        },
        "classification_thresholds": {
            "collapse_ratio_vs_baseline": collapse_ratio,
            "pose_abs_threshold": pose_abs_threshold,
            "seg_abs_threshold": seg_abs_threshold,
            "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        },
        "inputs": {
            "baseline_json": _rel(baseline_json, root=root),
            "exact_jsons": [_rel(path, root=root) for path in exact_paths],
            "planner_jsons": [_rel(path, root=root) for path in planner_paths],
        },
        "family_rows": build_family_rows(artifact_rows),
        "artifact_rows": artifact_rows,
    }


def _fmt_value(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 0.001 or abs(value) >= 1000:
            return f"{value:.{digits}e}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else "none"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(table: dict[str, Any]) -> str:
    baseline = table["baseline"]
    lines = [
        "# C067 Atom Response Taxonomy",
        "",
        "No score claim. Exact CUDA auth eval JSON remains the score authority.",
        "",
        "## Baseline",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| label | {_fmt_value(baseline.get('label'))} |",
        f"| archive_bytes | {_fmt_value(baseline.get('archive_bytes'))} |",
        f"| archive_sha256 | {_fmt_value(baseline.get('archive_sha256'))} |",
        f"| score_recomputed_from_components | {_fmt_value(baseline.get('score_recomputed_from_components'), digits=12)} |",
        f"| avg_segnet_dist | {_fmt_value(baseline.get('avg_segnet_dist'), digits=12)} |",
        f"| avg_posenet_dist | {_fmt_value(baseline.get('avg_posenet_dist'), digits=12)} |",
        f"| gpu_model | {_fmt_value(baseline.get('gpu_model'))} |",
        "",
        "## Family Table",
        "",
        "| Family | Status | Exact | Planner | Best Byte Delta | Best Nonrate Delta | Collapse Count | Evidence | Next Action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in table["family_rows"]:
        lines.append(
            "| {family} | {status} | {exact} | {planner} | {byte_delta} | {nonrate_delta} | {collapse} | {grades} | {action} |".format(
                family=_fmt_value(row["family"]),
                status=_fmt_value(row["status"]),
                exact=_fmt_value(row["exact_eval_count"]),
                planner=_fmt_value(row["planner_count"]),
                byte_delta=_fmt_value(row["best_byte_delta_vs_baseline"]),
                nonrate_delta=_fmt_value(row["best_nonrate_delta_vs_baseline"]),
                collapse=_fmt_value(row["collapse_artifact_count"]),
                grades=_fmt_value(row["evidence_grades"]),
                action=_fmt_value(row["next_action"]),
            )
        )

    exact_rows = [row for row in table["artifact_rows"] if row["artifact_kind"] == "exact_eval"]
    lines.extend(
        [
            "",
            "## Exact Eval Rows",
            "",
            "| Label | Family | Evidence | Byte Delta | Seg Delta | Pose Delta | Score Delta | Component | Collapse | Next Action | Artifact |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in exact_rows:
        lines.append(
            "| {label} | {family} | {grade} | {byte_delta} | {seg_delta} | {pose_delta} | {score_delta} | {component} | {collapse} | {action} | {artifact} |".format(
                label=_fmt_value(row["label"]),
                family=_fmt_value(row["family"]),
                grade=_fmt_value(row["evidence_grade"]),
                byte_delta=_fmt_value(row["byte_delta_vs_baseline"]),
                seg_delta=_fmt_value(row["avg_segnet_delta_vs_baseline"], digits=6),
                pose_delta=_fmt_value(row["avg_posenet_delta_vs_baseline"], digits=6),
                score_delta=_fmt_value(row["score_delta_vs_baseline"], digits=6),
                component=_fmt_value(row["component_status"]),
                collapse=_fmt_value(row["collapse_components"]),
                action=_fmt_value(row["next_action"]),
                artifact=_fmt_value(row["artifact_path"]),
            )
        )

    planner_rows = [row for row in table["artifact_rows"] if row["artifact_kind"] == "planner"]
    lines.extend(
        [
            "",
            "## Planner Rows",
            "",
            "| Label | Family | Evidence | Byte Delta | Byte Kind | Component | Decision | Next Action | Artifact |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for row in planner_rows:
        lines.append(
            "| {label} | {family} | {grade} | {byte_delta} | {kind} | {component} | {decision} | {action} | {artifact} |".format(
                label=_fmt_value(row["label"]),
                family=_fmt_value(row["family"]),
                grade=_fmt_value(row["evidence_grade"]),
                byte_delta=_fmt_value(row["byte_delta_vs_baseline"]),
                kind=_fmt_value(row["byte_delta_kind"]),
                component=_fmt_value(row["component_status"]),
                decision=_fmt_value(row.get("decision")),
                action=_fmt_value(row["next_action"]),
                artifact=_fmt_value(row["artifact_path"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(table: dict[str, Any], *, output_dir: Path, root: Path) -> dict[str, str]:
    output_dir = _resolve_path(output_dir, root=root)
    json_path = output_dir / "c067_atom_response_table.json"
    md_path = output_dir / "c067_atom_response_table.md"
    _json_dump(json_path, table)
    md_path.write_text(render_markdown(table), encoding="utf-8")
    return {
        "json": _rel(json_path, root=root),
        "markdown": _rel(md_path, root=root),
    }


def _paths_from_args(values: Iterable[str]) -> list[Path]:
    return [Path(value) for value in values]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--exact-json", action="append", default=[], help="Exact eval JSON path; repeatable.")
    parser.add_argument("--planner-json", action="append", default=[], help="Planner JSON path; repeatable.")
    parser.add_argument("--no-default-exact-scan", action="store_true")
    parser.add_argument("--no-default-planner-scan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collapse-ratio", type=float, default=10.0)
    parser.add_argument("--pose-collapse-abs", type=float, default=0.01)
    parser.add_argument("--seg-collapse-abs", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    table = build_c067_atom_response_table(
        root=root,
        baseline_json=args.baseline_json,
        exact_jsons=_paths_from_args(args.exact_json),
        planner_jsons=_paths_from_args(args.planner_json),
        include_default_exact_scan=not args.no_default_exact_scan,
        include_default_planner_scan=not args.no_default_planner_scan,
        collapse_ratio=args.collapse_ratio,
        pose_abs_threshold=args.pose_collapse_abs,
        seg_abs_threshold=args.seg_collapse_abs,
    )
    outputs = write_outputs(table, output_dir=args.output_dir, root=root)
    print(
        json.dumps(
            {
                "score_claim": False,
                "promotion_eligible": False,
                "outputs": outputs,
                "family_count": len(table["family_rows"]),
                "artifact_count": len(table["artifact_rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
