#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed Lane 12 L2 unblock readiness planner.

This tool audits existing Lane 12 / Alpha-Geo evidence and reports whether the
repository already has enough custody to justify creating the L2 retraining
clearance packet. It never launches work. It writes the clearance packet only
when explicitly requested and all local evidence gates pass.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lane12_l2_unblock_readiness_v1"
REPORT_ID = "lane12_l2_unblock_readiness_20260502"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane12_l2_unblock_readiness_20260502"
    / "lane12_l2_unblock_readiness.json"
)
DEFAULT_CLEARANCE = REPO_ROOT / ".omx" / "state" / "lane12_nerv_l2_clearance.json"
DEFAULT_GEOMETRY_GLOBS = (
    "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/alpha_geo*.json",
)
DEFAULT_PRIMITIVE_CONTRACT_GLOBS = (
    "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/*.primitive_contract.json",
)
DEFAULT_EXACT_EVIDENCE_GLOBS = (
    "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json",
    "experiments/results/lane_12_nerv_20260430_codex_jsonfix40/adjudicated_contest_auth_eval.json",
)
DEFAULT_POSE_REGEN_PROVENANCE_GLOBS = (
    "experiments/results/lightning_batch/alpha_geo0_pose_regen*/alpha_geo0_summary.json",
    "experiments/results/modal_alpha_geo0_pose_regen/*/modal_alpha_geo0_result_summary.json",
)
DEFAULT_CLEARANCE_LANE_ID = "lane_12_nerv_mask_codec"
ALLOWED_LANE_IDS = {"lane_12_nerv_mask_codec", "lane_12_nerv"}
EXPECTED_SHAPE = [1200, 384, 512]
PROMOTION_THRESHOLD_PRESET = "promotion"
PROMOTION_THRESHOLDS = {
    "global_disagreement_max": 0.001,
    "boundary_band_disagreement_max": {"1": 0.002, "2": 0.002, "3": 0.002, "5": 0.002},
    "stable_region_false_flip_rate_max": 0.002,
    "pair_transition_disagreement_max": 0.002,
    "pair_transition_f1_min": None,
    "class_recall_min": {"1": 0.999, "2": 0.999},
    "tiny_speckle_rate_max": 0.0001,
    "max_component_centroid_jump_px": 1.0,
    "missing_component_rate_max": 0.0,
}
CURRENT_FRONTIER = {
    "label": "C067",
    "score": 0.31561703078448233,
    "archive_size_bytes": 276214,
    "archive_sha256": "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a",
    "evidence_grade": "A++",
    "source": "operator mission context 2026-05-02",
}
CUDA_AUTH_EVAL_SOURCE = "archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda"


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _path_record(path: Path, repo_root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path, repo_root),
        "exists": path.exists(),
        "is_file": path.is_file(),
    }
    if path.is_file():
        record.update(
            {
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return record


def _load_json_file(path: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = _path_record(path, repo_root)
    if not path.exists():
        record["json_error"] = "missing"
        return record, None
    if not path.is_file():
        record["json_error"] = "not a file"
        return record, None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        record["json_error"] = str(exc)
        return record, None
    if not isinstance(payload, dict):
        record["json_error"] = "JSON root must be an object"
        return record, None
    record["json_object"] = True
    record["diagnostic"] = payload.get("diagnostic")
    record["schema"] = payload.get("schema")
    record["schema_version"] = payload.get("schema_version")
    return record, payload


def _expand_paths(
    *,
    repo_root: Path,
    explicit_paths: list[Path],
    patterns: tuple[str, ...],
) -> list[Path]:
    found: dict[str, Path] = {}
    for path in explicit_paths:
        resolved = _resolve_path(path, repo_root)
        found[str(resolved.resolve()) if resolved.exists() else str(resolved)] = resolved
    for pattern in patterns:
        if Path(pattern).is_absolute():
            matches = [Path(p) for p in glob.glob(pattern)]
        else:
            matches = list(repo_root.glob(pattern))
        for match in sorted(matches, key=lambda p: _display_path(p, repo_root)):
            found[str(match.resolve()) if match.exists() else str(match)] = match
    return [found[key] for key in sorted(found)]


def _is_int_not_bool(value: Any) -> bool:
    return type(value) is int


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def _clearance_evidence_items(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    evidence = payload.get("evidence")
    violations: list[str] = []
    if isinstance(evidence, str):
        items = [evidence] if evidence.strip() else []
    elif isinstance(evidence, list):
        items = []
        for idx, item in enumerate(evidence):
            if isinstance(item, str) and item.strip():
                items.append(item)
            else:
                violations.append(f"evidence[{idx}] must be a non-empty string path")
    else:
        items = []
        violations.append("evidence must be a non-empty string path or list of string paths")
    if not items and not violations:
        violations.append("evidence must cite at least one Lane 12 L2 artifact path")
    return items, violations


def validate_clearance_packet(path: Path, repo_root: Path) -> dict[str, Any]:
    record, payload = _load_json_file(path, repo_root)
    checks: dict[str, Any] = {}
    violations: list[str] = []
    evidence_path_records: list[dict[str, Any]] = []

    if payload is None:
        violation = f"missing or unreadable Lane 12 L2 clearance packet: {record['path']}"
        violations.append(violation)
        return {
            "path": record,
            "passed": False,
            "checks": checks,
            "evidence_paths": evidence_path_records,
            "violations": violations,
        }

    lane_id = payload.get("lane_id")
    checks["lane_id_allowed"] = {
        "passed": lane_id in ALLOWED_LANE_IDS,
        "expected": sorted(ALLOWED_LANE_IDS),
        "observed": lane_id,
    }
    if not checks["lane_id_allowed"]["passed"]:
        violations.append("lane_id must be lane_12_nerv_mask_codec or lane_12_nerv")

    bool_fields = ("cleared_for_retraining_unblock", "lane12_l2", "geometry_gate_passed")
    for field in bool_fields:
        passed = payload.get(field) is True
        checks[field] = {"passed": passed, "expected": True, "observed": payload.get(field)}
        if not passed:
            violations.append(f"{field} must be true")

    clean_passes = payload.get("grand_council_clean_passes")
    passes_ok = _is_int_not_bool(clean_passes) and clean_passes >= 3
    checks["grand_council_clean_passes"] = {
        "passed": passes_ok,
        "expected": "integer >= 3",
        "observed": clean_passes,
    }
    if not passes_ok:
        violations.append("grand_council_clean_passes must be an integer >= 3")

    evidence_items, evidence_violations = _clearance_evidence_items(payload)
    violations.extend(evidence_violations)
    for item in evidence_items:
        evidence_path = _resolve_path(item, repo_root)
        item_record = _path_record(evidence_path, repo_root)
        item_record["cited_as"] = item
        evidence_path_records.append(item_record)
        if not item_record["is_file"]:
            violations.append(f"evidence path does not exist as a file: {item}")
    checks["evidence_paths"] = {
        "passed": bool(evidence_items) and all(item["is_file"] for item in evidence_path_records),
        "expected": "one or more existing local evidence files",
        "observed": evidence_items,
    }

    return {
        "path": record,
        "passed": not violations,
        "checks": checks,
        "evidence_paths": evidence_path_records,
        "violations": violations,
    }


def _shape_from_geometry(payload: dict[str, Any]) -> list[int] | None:
    shape = payload.get("shape")
    if isinstance(shape, dict) and {"frames", "height", "width"}.issubset(shape):
        return [int(shape["frames"]), int(shape["height"]), int(shape["width"])]
    inputs = payload.get("inputs")
    if isinstance(inputs, dict):
        candidate_source = inputs.get("candidate_source")
        if isinstance(candidate_source, dict):
            decoded_shape = candidate_source.get("decoded_mask_shape")
            if isinstance(decoded_shape, list) and len(decoded_shape) == 3:
                return [int(v) for v in decoded_shape]
    return None


def _input_archive_sha256(source: Any) -> str | None:
    if not isinstance(source, dict):
        return None
    for key in ("source_sha256", "archive_sha256", "sha256"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _input_archive_member(source: Any) -> str | None:
    if not isinstance(source, dict):
        return None
    for key in ("archive_member_resolved", "archive_member", "member"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _false_check_names(pass_fail: Any) -> list[str]:
    if not isinstance(pass_fail, dict):
        return []
    checks = pass_fail.get("checks")
    if not isinstance(checks, dict):
        return []
    names = []
    for name, row in checks.items():
        if isinstance(row, dict) and row.get("passed") is False:
            names.append(str(name))
    return sorted(names)


def _canonical_thresholds(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical_thresholds(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical_thresholds(v) for v in value]
    return value


def _geometry_uses_promotion_thresholds(payload: dict[str, Any]) -> tuple[bool, str | None]:
    diagnostic_config = payload.get("diagnostic_config")
    if isinstance(diagnostic_config, dict):
        preset = diagnostic_config.get("threshold_preset")
        if preset != PROMOTION_THRESHOLD_PRESET:
            return False, "diagnostic_config.threshold_preset must be promotion"
        configured = diagnostic_config.get("thresholds")
        if _canonical_thresholds(configured) != PROMOTION_THRESHOLDS:
            return False, "diagnostic_config.thresholds must match promotion preset"
        return True, None

    pass_fail = payload.get("pass_fail")
    configured = pass_fail.get("thresholds") if isinstance(pass_fail, dict) else None
    if _canonical_thresholds(configured) != PROMOTION_THRESHOLDS:
        return False, "pass_fail.thresholds must match promotion preset"
    return True, None


def summarize_geometry_json(
    path: Path,
    *,
    repo_root: Path,
    expected_shape: list[int],
) -> dict[str, Any]:
    record, payload = _load_json_file(path, repo_root)
    if payload is None:
        return {
            "kind": "geometry_json",
            "path": record,
            "geometry_gate_passed": False,
            "blockers": [record.get("json_error", "unreadable JSON")],
        }

    pass_fail = payload.get("pass_fail")
    shape = _shape_from_geometry(payload)
    blockers: list[str] = []
    if payload.get("diagnostic") != "alpha_geo_0_nerv_geometry":
        blockers.append("diagnostic is not alpha_geo_0_nerv_geometry")
    if payload.get("score_evidence_grade") != "empirical":
        blockers.append("score_evidence_grade must be empirical")
    if payload.get("scorer_proxy") is not False:
        blockers.append("scorer_proxy must be false")
    if shape != expected_shape:
        blockers.append(f"decoded shape must be {expected_shape}, observed {shape}")
    promotion_thresholds_ok, promotion_thresholds_blocker = _geometry_uses_promotion_thresholds(payload)
    if not promotion_thresholds_ok and promotion_thresholds_blocker is not None:
        blockers.append(promotion_thresholds_blocker)
    if not isinstance(pass_fail, dict) or pass_fail.get("overall_pass") is not True:
        blockers.append("pass_fail.overall_pass must be true")
    blockers.extend(f"failed check: {name}" for name in _false_check_names(pass_fail))

    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    candidate_source = inputs.get("candidate_source") if isinstance(inputs, dict) else None
    baseline_source = inputs.get("baseline_source") if isinstance(inputs, dict) else None
    baseline_archive_sha256 = _input_archive_sha256(baseline_source)
    candidate_archive_sha256 = _input_archive_sha256(candidate_source)
    candidate_member = _input_archive_member(candidate_source)
    if not _is_sha256(baseline_archive_sha256):
        blockers.append("inputs.baseline_source source SHA-256 must be recorded")
    if not _is_sha256(candidate_archive_sha256):
        blockers.append("inputs.candidate_source source SHA-256 must be recorded")
    if candidate_member != "masks.nrv":
        blockers.append("inputs.candidate_source archive member must resolve to masks.nrv")

    return {
        "kind": "geometry_json",
        "path": record,
        "diagnostic": payload.get("diagnostic"),
        "score_evidence_grade": payload.get("score_evidence_grade"),
        "device": payload.get("device"),
        "scorer_proxy": payload.get("scorer_proxy"),
        "shape": shape,
        "expected_shape": expected_shape,
        "geometry_gate_passed": not blockers,
        "overall_pass": pass_fail.get("overall_pass") if isinstance(pass_fail, dict) else None,
        "failed_checks": _false_check_names(pass_fail),
        "thresholds": pass_fail.get("thresholds") if isinstance(pass_fail, dict) else None,
        "threshold_preset": payload.get("diagnostic_config", {}).get("threshold_preset")
        if isinstance(payload.get("diagnostic_config"), dict)
        else None,
        "promotion_thresholds_required": True,
        "promotion_thresholds_passed": promotion_thresholds_ok,
        "global_disagreement": payload.get("global", {}).get("global_disagreement")
        if isinstance(payload.get("global"), dict)
        else None,
        "boundary_2px_disagreement": payload.get("boundary_bands", {})
        .get("2", {})
        .get("disagreement_rate")
        if isinstance(payload.get("boundary_bands"), dict)
        else None,
        "pair_transition_disagreement": payload.get("temporal", {})
        .get("pair_transition", {})
        .get("disagreement_rate")
        if isinstance(payload.get("temporal"), dict)
        else None,
        "baseline_archive_sha256": baseline_archive_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "candidate_member": candidate_member,
        "blockers": sorted(set(blockers)),
    }


def _contract_shape(payload: dict[str, Any]) -> list[int] | None:
    source = payload.get("source")
    if isinstance(source, dict):
        baseline = source.get("baseline")
        if isinstance(baseline, dict):
            shape = baseline.get("decoded_mask_shape")
            if isinstance(shape, list) and len(shape) == 3:
                return [int(v) for v in shape]
    return None


def summarize_primitive_contract(
    path: Path,
    *,
    repo_root: Path,
    expected_shape: list[int],
) -> dict[str, Any]:
    record, payload = _load_json_file(path, repo_root)
    if payload is None:
        return {
            "kind": "primitive_contract",
            "path": record,
            "usable_for_decoded_baseline_training": False,
            "blockers": [record.get("json_error", "unreadable JSON")],
        }

    blockers: list[str] = []
    shape = _contract_shape(payload)
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    baseline = source.get("baseline") if isinstance(source, dict) else {}
    if not isinstance(baseline, dict):
        baseline = {}
    if payload.get("diagnostic") != "alpha_geo_primitive_contract_v1":
        blockers.append("diagnostic must be alpha_geo_primitive_contract_v1")
    if payload.get("score_evidence_grade") != "empirical":
        blockers.append("score_evidence_grade must be empirical")
    for field in ("promotion_eligible", "score_claim_eligible", "exact_eval_claim"):
        if payload.get(field) is not False:
            blockers.append(f"{field} must be false")
    if shape != expected_shape:
        blockers.append(f"baseline decoded shape must be {expected_shape}, observed {shape}")
    decoded_sha256 = baseline.get("decoded_mask_sha256")
    if not _is_sha256(decoded_sha256):
        blockers.append("source.baseline.decoded_mask_sha256 must be a SHA-256 hex digest")
    if baseline.get("decoded_mask_sha256_algo") != "sha256(shape,dtype,contiguous-raw-bytes)":
        blockers.append(
            "source.baseline.decoded_mask_sha256_algo must be "
            "sha256(shape,dtype,contiguous-raw-bytes)"
        )
    if baseline.get("decoded_mask_dtype") != "torch.uint8":
        blockers.append("source.baseline.decoded_mask_dtype must be torch.uint8")
    baseline_member = baseline.get("archive_member") or baseline.get("archive_member_resolved")
    if baseline_member != "masks.mkv":
        blockers.append("source.baseline archive member must resolve to masks.mkv")
    if not _is_sha256(baseline.get("archive_sha256")):
        blockers.append("source.baseline.archive_sha256 must be a SHA-256 hex digest")

    threshold_gates = payload.get("threshold_gates")
    if not isinstance(threshold_gates, dict):
        threshold_gates = {}
    gate_summary: dict[str, Any] = {}
    for name in ("exploratory_retrain_gate", "exact_eval_spend_gate"):
        row = threshold_gates.get(name)
        if isinstance(row, dict):
            gate_summary[name] = {
                "passed": row.get("passed"),
                "blockers": row.get("blockers", []),
                "observed": row.get("observed", {}),
                "thresholds": row.get("thresholds", {}),
            }

    return {
        "kind": "primitive_contract",
        "path": record,
        "diagnostic": payload.get("diagnostic"),
        "score_evidence_grade": payload.get("score_evidence_grade"),
        "shape": shape,
        "expected_shape": expected_shape,
        "usable_for_decoded_baseline_training": not blockers,
        "promotion_eligible": payload.get("promotion_eligible"),
        "score_claim_eligible": payload.get("score_claim_eligible"),
        "exact_eval_claim": payload.get("exact_eval_claim"),
        "baseline_member": baseline_member,
        "baseline_archive_sha256": baseline.get("archive_sha256"),
        "decoded_mask_sha256": decoded_sha256,
        "decoded_mask_sha256_algo": baseline.get("decoded_mask_sha256_algo"),
        "decoded_mask_dtype": baseline.get("decoded_mask_dtype"),
        "threshold_gates": gate_summary,
        "blockers": sorted(set(blockers)),
    }


def summarize_exact_evidence(path: Path, *, repo_root: Path) -> dict[str, Any]:
    record, payload = _load_json_file(path, repo_root)
    if payload is None:
        return {
            "kind": "exact_evidence_json",
            "path": record,
            "exact_cuda": False,
            "blockers": [record.get("json_error", "unreadable JSON")],
        }
    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    device = provenance.get("device")
    tool = provenance.get("tool")
    score = payload.get("score_recomputed_from_components")
    score_delta = None
    outcome = "unknown"
    if isinstance(score, (int, float)):
        score_delta = float(score) - float(CURRENT_FRONTIER["score"])
        outcome = "worse_than_current_frontier" if score_delta > 0 else "at_or_better_than_current_frontier"
    exact_cuda = tool == "experiments/contest_auth_eval.py" and device == "cuda"
    return {
        "kind": "exact_evidence_json",
        "path": record,
        "exact_cuda": exact_cuda,
        "canonical_tool": tool == "experiments/contest_auth_eval.py",
        "device": device,
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "archive_sha256": provenance.get("archive_sha256"),
        "archive_size_bytes": payload.get("archive_size_bytes"),
        "n_samples": payload.get("n_samples"),
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "score_recomputed_from_components": score,
        "score_delta_vs_current_frontier": score_delta,
        "outcome_vs_current_frontier": outcome,
        "allowed_use": ["negative_evidence", "forensic", "no_clearance_by_itself", "no_promotion"],
        "score_claim_source": CUDA_AUTH_EVAL_SOURCE if exact_cuda else None,
    }


def _command_names(payload: dict[str, Any]) -> set[str]:
    commands = payload.get("commands")
    if not isinstance(commands, list):
        return set()
    names = set()
    for row in commands:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            names.add(row["name"])
    return names


def summarize_pose_provenance(path: Path, *, repo_root: Path) -> dict[str, Any]:
    record, payload = _load_json_file(path, repo_root)
    blockers: list[str] = []
    if payload is None:
        blockers.append(record.get("json_error", "unreadable JSON"))
        return {
            "kind": "pose_regeneration_provenance",
            "path": record,
            "usable_for_exact_eval_dispatch": False,
            "diagnostic": None,
            "schema": None,
            "tool": None,
            "blockers": blockers,
        }

    tool = payload.get("tool")
    schema = payload.get("schema")
    stage = payload.get("stage")
    passed = payload.get("passed")
    score_claim = payload.get("score_claim")
    archive_sha256 = payload.get("archive_sha256")
    archive_size_bytes = payload.get("archive_size_bytes")
    command_names = _command_names(payload)
    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    candidate_archive = inputs.get("candidate_archive") if isinstance(inputs, dict) else None
    baseline_archive = inputs.get("baseline_archive") if isinstance(inputs, dict) else None
    candidate_archive_sha256 = (
        _input_archive_sha256(candidate_archive)
        or payload.get("candidate_archive_sha256")
    )
    baseline_archive_sha256 = (
        _input_archive_sha256(baseline_archive)
        or payload.get("baseline_archive_sha256")
    )

    if tool == "experiments/alpha_geo0_pose_regen.py":
        if passed is not True:
            blockers.append("alpha_geo0_pose_regen summary passed must be true")
        if stage != "done":
            blockers.append("alpha_geo0_pose_regen stage must be done")
        if score_claim is not True:
            blockers.append("alpha_geo0_pose_regen score_claim must be true after exact CUDA eval")
        if not _is_sha256(archive_sha256):
            blockers.append("alpha_geo0_pose_regen archive_sha256 must be a SHA-256 hex digest")
        if not isinstance(archive_size_bytes, int) or archive_size_bytes <= 0:
            blockers.append("alpha_geo0_pose_regen archive_size_bytes must be a positive integer")
        if not _is_sha256(candidate_archive_sha256):
            blockers.append(
                "alpha_geo0_pose_regen inputs.candidate_archive.sha256 must be a SHA-256 hex digest"
            )
        required_commands = {
            "diagnose_nerv_geometry",
            "optimize_poses",
            "contest_auth_eval",
            "adjudicate_contest_auth_eval",
        }
        missing_commands = sorted(required_commands - command_names)
        if missing_commands:
            blockers.append(
                "alpha_geo0_pose_regen commands missing: " + ", ".join(missing_commands)
            )
    elif tool == "experiments/modal_alpha_geo0_pose_regen.py":
        if passed is not True:
            blockers.append("modal_alpha_geo0_pose_regen summary passed must be true")
        if stage != "done":
            blockers.append("modal_alpha_geo0_pose_regen stage must be done")
        if score_claim is not True:
            blockers.append("modal_alpha_geo0_pose_regen score_claim must be true after exact CUDA eval")
        if payload.get("promotion_eligible") is not True:
            blockers.append("modal_alpha_geo0_pose_regen promotion_eligible must be true")
        validation_errors = payload.get("validation_errors")
        if isinstance(validation_errors, list) and validation_errors:
            blockers.extend(f"modal validation error: {item}" for item in validation_errors)
        if passed is True:
            if not _is_sha256(archive_sha256):
                blockers.append("modal_alpha_geo0_pose_regen archive_sha256 must be a SHA-256 hex digest")
            if not isinstance(archive_size_bytes, int) or archive_size_bytes <= 0:
                blockers.append("modal_alpha_geo0_pose_regen archive_size_bytes must be a positive integer")
            if not _is_sha256(candidate_archive_sha256):
                blockers.append(
                    "modal_alpha_geo0_pose_regen inputs.candidate_archive.sha256 must be a SHA-256 hex digest"
                )
    elif schema in {"pose_regen_v1", "alpha_geo0_pose_regen_v1"}:
        if not _is_sha256(candidate_archive_sha256):
            blockers.append("candidate_archive_sha256 must be a SHA-256 hex digest")
        if not _is_sha256(payload.get("optimized_poses_sha256")):
            blockers.append("optimized_poses_sha256 must be a SHA-256 hex digest")
        if payload.get("eval_roundtrip") is not True:
            blockers.append("eval_roundtrip must be true")
        if payload.get("score_claim") is not False:
            blockers.append("standalone pose provenance score_claim must be false")
    else:
        blockers.append(
            "pose provenance must be alpha_geo0_pose_regen.py summary or schema pose_regen_v1"
        )

    return {
        "kind": "pose_regeneration_provenance",
        "path": record,
        "usable_for_exact_eval_dispatch": not blockers,
        "diagnostic": payload.get("diagnostic") if payload else None,
        "schema": schema,
        "schema_version": payload.get("schema_version"),
        "tool": tool,
        "stage": stage,
        "passed": passed,
        "score_claim": score_claim,
        "promotion_eligible": payload.get("promotion_eligible"),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "candidate_archive_sha256": candidate_archive_sha256,
        "baseline_archive_sha256": baseline_archive_sha256,
        "command_names": sorted(command_names),
        "blockers": sorted(set(blockers)),
    }


def match_alpha_geo_pose_provenance(
    geometry_records: list[dict[str, Any]],
    pose_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Match passing Alpha-Geo records to pose provenance by candidate SHA."""

    passing_geometry = [
        row for row in geometry_records if row.get("geometry_gate_passed") is True
    ]
    usable_pose = [
        row for row in pose_records if row.get("usable_for_exact_eval_dispatch") is True
    ]
    pose_by_candidate_sha = {
        row.get("candidate_archive_sha256"): row
        for row in usable_pose
        if _is_sha256(row.get("candidate_archive_sha256"))
    }
    matches: list[dict[str, Any]] = []
    unmatched_geometry: list[dict[str, Any]] = []
    for geometry in passing_geometry:
        candidate_sha = geometry.get("candidate_archive_sha256")
        if not _is_sha256(candidate_sha):
            unmatched_geometry.append(
                {
                    "geometry_path": geometry["path"]["path"],
                    "candidate_archive_sha256": candidate_sha,
                    "reason": "geometry_candidate_sha_missing_or_invalid",
                }
            )
            continue
        pose = pose_by_candidate_sha.get(candidate_sha)
        if pose is None:
            unmatched_geometry.append(
                {
                    "geometry_path": geometry["path"]["path"],
                    "candidate_archive_sha256": candidate_sha,
                    "reason": "no_usable_pose_regeneration_for_candidate_sha",
                }
            )
            continue
        matches.append(
            {
                "candidate_archive_sha256": candidate_sha,
                "geometry_path": geometry["path"]["path"],
                "pose_provenance_path": pose["path"]["path"],
                "pose_output_archive_sha256": pose.get("archive_sha256"),
                "match_key": "candidate_archive_sha256",
            }
        )
    return {
        "passed": bool(matches),
        "match_count": len(matches),
        "matches": matches,
        "unmatched_passing_geometry": unmatched_geometry,
        "usable_pose_candidate_archive_sha256": sorted(pose_by_candidate_sha),
    }


def _missing(code: str, severity: str, detail: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "detail": detail}


def _evidence_records(
    *,
    repo_root: Path,
    evidence_items: list[Path],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    cited: list[str] = []
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for item in evidence_items:
        path = _resolve_path(item, repo_root)
        cited_item = _display_path(path, repo_root)
        cited.append(cited_item)
        record = _path_record(path, repo_root)
        record["cited_as"] = cited_item
        records.append(record)
        if not record["is_file"]:
            blockers.append(f"clearance evidence path does not exist as a file: {cited_item}")
    if not cited:
        blockers.append("write-clearance requires at least one --clearance-evidence path")
    return cited, records, blockers


def _write_l2_clearance_packet(
    *,
    path: Path,
    repo_root: Path,
    lane_id: str,
    evidence_items: list[str],
    grand_council_clean_passes: int,
    geometry_records: list[dict[str, Any]],
    contract_records: list[dict[str, Any]],
    command: list[str] | None,
    force: bool,
) -> dict[str, Any]:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to overwrite")
    passing_geometry = [row for row in geometry_records if row.get("geometry_gate_passed") is True]
    usable_contracts = [
        row for row in contract_records if row.get("usable_for_decoded_baseline_training") is True
    ]
    packet = {
        "schema": "lane12_nerv_l2_clearance_v1",
        "lane_id": lane_id,
        "cleared_for_retraining_unblock": True,
        "lane12_l2": True,
        "geometry_gate_passed": True,
        "grand_council_clean_passes": int(grand_council_clean_passes),
        "evidence": evidence_items,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_job_launched": False,
        "clearance_scope": "build-only Lane 12 NeRV retraining unblock; exact eval still requires candidate archive custody and pose/geometry provenance",
        "geometry_evidence": [
            {
                "path": row["path"]["path"],
                "sha256": row["path"].get("sha256"),
                "global_disagreement": row.get("global_disagreement"),
                "boundary_2px_disagreement": row.get("boundary_2px_disagreement"),
                "pair_transition_disagreement": row.get("pair_transition_disagreement"),
                "threshold_preset": row.get("threshold_preset"),
            }
            for row in passing_geometry
        ],
        "primitive_contract_evidence": [
            {
                "path": row["path"]["path"],
                "sha256": row["path"].get("sha256"),
                "decoded_mask_sha256": row.get("decoded_mask_sha256"),
                "baseline_archive_sha256": row.get("baseline_archive_sha256"),
            }
            for row in usable_contracts
        ],
        "provenance": {
            "tool": "experiments/plan_lane12_l2_unblock.py",
            "command": command or ["experiments/plan_lane12_l2_unblock.py"],
            "deterministic_json": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return _path_record(path, repo_root)


def build_redesign_recipe(expected_shape: list[int]) -> dict[str, Any]:
    return {
        "recipe_id": "lane12_alpha_geo1_decoded_baseline_l2_unblock_v1",
        "score_claim": False,
        "remote_dispatch_now": False,
        "dispatch_claim_required_before_remote_job": {
            "tool": "tools/claim_lane_dispatch.py claim",
            "lane_id": "lane_12_nerv_mask_codec",
            "required": True,
        },
        "training_target": {
            "gt_masks_source": "decoded-baseline",
            "decoded_baseline_path": "experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip",
            "decoded_baseline_member": "masks.mkv",
            "expected_shape": expected_shape,
            "segnet_target_default_allowed": False,
            "segnet_forensic_escape_hatch": {
                "remote_env": "ALLOW_RETIRED_SEGNET_TARGET=1",
                "trainer_flag": "--allow-forensic-segnet-target",
                "allowed_use": ["forensic_debug", "no_clearance", "no_promotion"],
                "required_note": "Only for documented reproduction of the retired jsonfix40 target path.",
            },
        },
        "primitive_contract": {
            "required_diagnostic": "alpha_geo_primitive_contract_v1",
            "required_for_production_retraining": True,
            "required_fields": {
                "promotion_eligible": False,
                "score_claim_eligible": False,
                "exact_eval_claim": False,
                "source.baseline.decoded_mask_shape": expected_shape,
            },
            "trainer_consumes": "--alpha-primitive-contract",
        },
        "geometry_gate": {
            "diagnostic_tool": "experiments/diagnose_nerv_geometry.py",
            "threshold_preset": "promotion",
            "required_overall_pass": True,
            "thresholds": {
                "global_disagreement_max": 0.001,
                "boundary_band_disagreement_max": {"1": 0.002, "2": 0.002, "3": 0.002, "5": 0.002},
                "stable_region_false_flip_rate_max": 0.002,
                "pair_transition_disagreement_max": 0.002,
                "class_recall_min": {"1": 0.999, "2": 0.999},
                "tiny_speckle_rate_max": 0.0001,
                "max_component_centroid_jump_px": 1.0,
                "missing_component_rate_max": 0.0,
            },
            "required_command_shape": {
                "--baseline-member": "masks.mkv",
                "--candidate-member": "masks.nrv",
                "--num-frames": expected_shape[0],
                "--height": expected_shape[1],
                "--width": expected_shape[2],
            },
        },
        "training_custody": {
            "script": "scripts/remote_lane_nerv.sh",
            "run_auth_eval_default": "RUN_AUTH_EVAL=0",
            "required_env": {
                "GT_MASKS_SOURCE": "decoded-baseline",
                "DECODED_BASELINE_MEMBER": "masks.mkv",
                "ALPHA_PRIMITIVE_CONTRACT": "<alpha_geo_primitive_contract_v1.json>",
                "L2_CLEARANCE_PATH": ".omx/state/lane12_nerv_l2_clearance.json",
            },
            "trainer_command_core": [
                "experiments/train_nerv_mask.py",
                "--device",
                "cuda",
                "--gt-masks-source",
                "decoded-baseline",
                "--decoded-baseline-member",
                "masks.mkv",
                "--alpha-primitive-contract",
                "<alpha_geo_primitive_contract_v1.json>",
            ],
            "provenance_must_record": [
                "git_hash",
                "gpu_name",
                "profile",
                "gt_masks_source=decoded-baseline",
                "target_mask_sha256",
                "alpha_primitive_contract.sha256",
                "masks.nrv bytes and sha256",
            ],
        },
        "eval_custody": {
            "exact_score_source": CUDA_AUTH_EVAL_SOURCE,
            "run_auth_eval_requires": [
                "POSE_REGEN_PROVENANCE for the candidate mask stream",
                "ALPHA_GEO_PROVENANCE with diagnostic_config.threshold_preset=promotion and pass_fail.overall_pass=true for the exact archive",
                "archive SHA match between Alpha-Geo provenance and built archive",
                "runtime tree hash from contest_auth_eval.py",
            ],
            "promotion_requires": [
                "exact CUDA auth eval JSON",
                "component gates not collapsed",
                "archive bytes and SHA-256 recorded",
                "recomputed score from components",
            ],
        },
    }


def plan_lane12_l2_unblock(
    *,
    repo_root: Path = REPO_ROOT,
    clearance_json: Path = DEFAULT_CLEARANCE,
    geometry_jsons: list[Path] | None = None,
    primitive_contract_jsons: list[Path] | None = None,
    exact_evidence_jsons: list[Path] | None = None,
    pose_regeneration_provenance: list[Path] | None = None,
    use_default_artifact_globs: bool = True,
    output_json: Path | None = DEFAULT_OUTPUT,
    force: bool = False,
    write_clearance_packet: bool = False,
    clearance_evidence: list[Path] | None = None,
    grand_council_clean_passes: int = 0,
    clearance_lane_id: str = DEFAULT_CLEARANCE_LANE_ID,
    command: list[str] | None = None,
    expected_shape: list[int] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    expected = expected_shape or list(EXPECTED_SHAPE)
    geometry_patterns = DEFAULT_GEOMETRY_GLOBS if use_default_artifact_globs else ()
    contract_patterns = DEFAULT_PRIMITIVE_CONTRACT_GLOBS if use_default_artifact_globs else ()
    exact_patterns = DEFAULT_EXACT_EVIDENCE_GLOBS if use_default_artifact_globs else ()
    geometry_paths = _expand_paths(
        repo_root=repo_root,
        explicit_paths=geometry_jsons or [],
        patterns=geometry_patterns,
    )
    contract_paths = _expand_paths(
        repo_root=repo_root,
        explicit_paths=primitive_contract_jsons or [],
        patterns=contract_patterns,
    )
    exact_paths = _expand_paths(
        repo_root=repo_root,
        explicit_paths=exact_evidence_jsons or [],
        patterns=exact_patterns,
    )
    pose_patterns = DEFAULT_POSE_REGEN_PROVENANCE_GLOBS if use_default_artifact_globs else ()
    pose_paths = _expand_paths(
        repo_root=repo_root,
        explicit_paths=pose_regeneration_provenance or [],
        patterns=pose_patterns,
    )

    clearance_path = _resolve_path(clearance_json, repo_root)
    clearance = validate_clearance_packet(clearance_path, repo_root)
    geometry_records = [
        summarize_geometry_json(path, repo_root=repo_root, expected_shape=expected)
        for path in geometry_paths
    ]
    contract_records = [
        summarize_primitive_contract(path, repo_root=repo_root, expected_shape=expected)
        for path in contract_paths
    ]
    exact_records = [summarize_exact_evidence(path, repo_root=repo_root) for path in exact_paths]
    pose_records = [summarize_pose_provenance(path, repo_root=repo_root) for path in pose_paths]

    passing_geometry = [row for row in geometry_records if row.get("geometry_gate_passed") is True]
    usable_contracts = [
        row for row in contract_records if row.get("usable_for_decoded_baseline_training") is True
    ]
    usable_pose = [row for row in pose_records if row.get("usable_for_exact_eval_dispatch") is True]
    alpha_geo_pose_match = match_alpha_geo_pose_provenance(geometry_records, pose_records)
    cited_evidence, cited_evidence_records, write_evidence_blockers = _evidence_records(
        repo_root=repo_root,
        evidence_items=clearance_evidence or [],
    )
    clean_passes_ok = _is_int_not_bool(grand_council_clean_passes) and grand_council_clean_passes >= 3
    clearance_lane_id_ok = clearance_lane_id in ALLOWED_LANE_IDS
    eligible_to_create_clearance = (
        bool(passing_geometry)
        and bool(usable_contracts)
        and clean_passes_ok
        and clearance_lane_id_ok
        and not write_evidence_blockers
    )
    clearance_write_record: dict[str, Any] | None = None

    missing: list[dict[str, Any]] = []
    if not passing_geometry:
        missing.append(
            _missing(
                "no_passing_alpha_geo_geometry",
                "blocks_retraining_unblock",
                "Need Alpha-Geo geometry JSON with diagnostic=alpha_geo_0_nerv_geometry, empirical non-proxy evidence, full 1200x384x512 shape, diagnostic_config.threshold_preset=promotion, and pass_fail.overall_pass=true.",
            )
        )
    if not usable_contracts:
        missing.append(
            _missing(
                "no_usable_alpha_geo_primitive_contract_v1",
                "blocks_production_retraining",
                "Need alpha_geo_primitive_contract_v1 with non-claim flags false and decoded baseline shape custody.",
            )
        )
    if not usable_pose:
        pose_detail = "RUN_AUTH_EVAL=1 requires completed candidate pose-regeneration provenance for the candidate mask stream."
        if pose_records:
            pose_detail = (
                "Pose-regeneration provenance exists, but no record passed completion/custody gates. "
                "Inspect evidence_buckets.empirical_evidence.pose_regeneration_provenance."
            )
        missing.append(
            _missing(
                "pose_regeneration_provenance_missing",
                "blocks_exact_eval_dispatch",
                pose_detail,
            )
        )
    elif passing_geometry and not alpha_geo_pose_match["passed"]:
        missing.append(
            _missing(
                "alpha_geo_pose_regen_candidate_mismatch",
                "blocks_exact_eval_dispatch",
                "Need a usable POSE_REGEN provenance record whose candidate_archive_sha256 matches a passing Alpha-Geo geometry record.",
            )
        )

    if write_clearance_packet:
        if not clearance_lane_id_ok:
            missing.append(
                _missing(
                    "clearance_lane_id_invalid",
                    "blocks_clearance_write",
                    f"clearance_lane_id must be one of {sorted(ALLOWED_LANE_IDS)}, observed {clearance_lane_id!r}",
                )
            )
        if not clean_passes_ok:
            missing.append(
                _missing(
                    "grand_council_clean_passes_insufficient_for_clearance_write",
                    "blocks_clearance_write",
                    "write-clearance requires --grand-council-clean-passes >= 3",
                )
            )
        for blocker in write_evidence_blockers:
            missing.append(_missing("clearance_evidence_invalid", "blocks_clearance_write", blocker))
        if eligible_to_create_clearance:
            clearance_write_record = _write_l2_clearance_packet(
                path=clearance_path,
                repo_root=repo_root,
                lane_id=clearance_lane_id,
                evidence_items=cited_evidence,
                grand_council_clean_passes=grand_council_clean_passes,
                geometry_records=geometry_records,
                contract_records=contract_records,
                command=command,
                force=force,
            )
            clearance = validate_clearance_packet(clearance_path, repo_root)

    if not clearance["passed"]:
        missing.append(
            _missing(
                "clearance_packet_invalid_or_missing",
                "blocks_retraining_unblock",
                "; ".join(clearance["violations"]),
            )
        )

    ready_for_retraining = clearance["passed"] and bool(passing_geometry) and bool(usable_contracts)
    ready_for_exact_eval = ready_for_retraining and bool(alpha_geo_pose_match["passed"])
    blockers = [
        item["detail"] for item in missing if item["severity"] in {"blocks_retraining_unblock", "blocks_production_retraining"}
    ]
    exact_eval_dispatch_blockers = [
        item["detail"] for item in missing if item["severity"] == "blocks_exact_eval_dispatch"
    ]

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "report_id": REPORT_ID,
        "deterministic_report": True,
        "score_claim": False,
        "promotion_eligible": False,
        "training_performed": False,
        "remote_job_launched": False,
        "clearance_state_written": clearance_write_record is not None,
        "clearance_packet_path": ".omx/state/lane12_nerv_l2_clearance.json",
        "current_frontier_context": CURRENT_FRONTIER,
        "launcher_criteria": clearance,
        "inputs": {
            "repo_root": str(repo_root),
            "geometry_jsons": [_display_path(path, repo_root) for path in geometry_paths],
            "primitive_contract_jsons": [_display_path(path, repo_root) for path in contract_paths],
            "exact_evidence_jsons": [_display_path(path, repo_root) for path in exact_paths],
            "pose_regeneration_provenance": [_display_path(path, repo_root) for path in pose_paths],
        },
        "evidence_buckets": {
            "exact_evidence": exact_records,
            "empirical_evidence": {
                "geometry_jsons": geometry_records,
                "primitive_contracts": contract_records,
                "pose_regeneration_provenance": pose_records,
            },
            "provenance_closure": {
                "alpha_geo_pose_candidate_match": alpha_geo_pose_match,
            },
            "missing_prerequisites": missing,
        },
        "clearance_write_request": {
            "requested": bool(write_clearance_packet),
            "eligible_to_create_clearance_packet": eligible_to_create_clearance,
            "state_write_performed": clearance_write_record is not None,
            "clearance_lane_id": clearance_lane_id,
            "grand_council_clean_passes": grand_council_clean_passes,
            "clean_passes_ok": clean_passes_ok,
            "evidence_paths": cited_evidence_records,
            "written_packet": clearance_write_record,
        },
        "readiness_summary": {
            "ready_for_retraining_unblock": ready_for_retraining,
            "ready_for_exact_eval_dispatch": ready_for_exact_eval,
            "eligible_to_create_clearance_packet": eligible_to_create_clearance,
            "write_clearance_packet": bool(write_clearance_packet),
            "state_write_performed": clearance_write_record is not None,
            "passing_geometry_count": len(passing_geometry),
            "usable_primitive_contract_count": len(usable_contracts),
            "usable_pose_regeneration_provenance_count": len(usable_pose),
            "matched_alpha_geo_pose_candidate_count": alpha_geo_pose_match["match_count"],
            "exact_cuda_evidence_count": sum(1 for row in exact_records if row.get("exact_cuda") is True),
            "blockers": blockers,
            "exact_eval_dispatch_blockers": exact_eval_dispatch_blockers,
        },
        "redesign_recipe": build_redesign_recipe(expected),
        "provenance": {
            "tool": "experiments/plan_lane12_l2_unblock.py",
            "command": command or ["experiments/plan_lane12_l2_unblock.py"],
        },
    }

    if output_json is not None:
        output_path = _resolve_path(output_json, repo_root)
        if output_path.exists() and not force:
            raise FileExistsError(f"{output_path} exists; use --force to overwrite")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--clearance-json", type=Path, default=DEFAULT_CLEARANCE)
    parser.add_argument("--geometry-json", type=Path, action="append", default=[])
    parser.add_argument("--primitive-contract-json", type=Path, action="append", default=[])
    parser.add_argument("--exact-evidence-json", type=Path, action="append", default=[])
    parser.add_argument("--pose-regeneration-provenance", type=Path, action="append", default=[])
    parser.add_argument(
        "--no-default-artifact-globs",
        action="store_true",
        help="Use only explicitly provided artifact paths.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-clearance-packet",
        action="store_true",
        help="Write the L2 clearance packet only when all local evidence gates pass.",
    )
    parser.add_argument(
        "--clearance-evidence",
        type=Path,
        action="append",
        default=[],
        help="Existing local evidence file to cite in a written clearance packet.",
    )
    parser.add_argument(
        "--grand-council-clean-passes",
        type=int,
        default=0,
        help="Clean-pass count for a written clearance packet; must be >=3.",
    )
    parser.add_argument("--clearance-lane-id", default=DEFAULT_CLEARANCE_LANE_ID)
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_SHAPE[0])
    parser.add_argument("--expected-height", type=int, default=EXPECTED_SHAPE[1])
    parser.add_argument("--expected-width", type=int, default=EXPECTED_SHAPE[2])
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    expected_shape = [args.expected_frames, args.expected_height, args.expected_width]
    report = plan_lane12_l2_unblock(
        repo_root=args.repo_root,
        clearance_json=args.clearance_json,
        geometry_jsons=args.geometry_json,
        primitive_contract_jsons=args.primitive_contract_json,
        exact_evidence_jsons=args.exact_evidence_json,
        pose_regeneration_provenance=args.pose_regeneration_provenance,
        use_default_artifact_globs=not args.no_default_artifact_globs,
        output_json=args.output_json,
        force=args.force,
        write_clearance_packet=args.write_clearance_packet,
        clearance_evidence=args.clearance_evidence,
        grand_council_clean_passes=args.grand_council_clean_passes,
        clearance_lane_id=args.clearance_lane_id,
        expected_shape=expected_shape,
        command=["experiments/plan_lane12_l2_unblock.py", *(argv if argv is not None else sys.argv[1:])],
    )
    summary = report["readiness_summary"]
    print(
        "[lane12-l2-unblock] "
        f"wrote {args.output_json} "
        f"ready_for_retraining_unblock={summary['ready_for_retraining_unblock']} "
        f"ready_for_exact_eval_dispatch={summary['ready_for_exact_eval_dispatch']} "
        f"clearance_state_written={report['clearance_state_written']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
