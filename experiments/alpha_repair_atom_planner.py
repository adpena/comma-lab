#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank Alpha AMR1 sparse-repair atoms by byte cost and hard-pair priors.

This planner is intentionally non-promotable. It decodes an Alpha candidate
AMR1 residual payload, partitions residual runs into deterministic atoms, and
measures exact payload bytes for each atom under the same AMR1 encoder and
compression options used by the archive builder. It does not run scorers and
does not make score claims.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from tac.repo_io import read_json, sha256_bytes, sha256_file, write_json


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_BUILDER_PATH = REPO_ROOT / "experiments" / "build_alpha_mask_replacement_archive.py"
PRODUCER = "experiments/alpha_repair_atom_planner.py"
SCHEMA = "alpha_repair_atom_plan_v1"
EVIDENCE_GRADE = "empirical"
EXPECTED_PAIR_COUNT = 600
DEFAULT_COMPONENT_TRACE_TOP_K = 100
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SCORE_CLAIM_WARNING = (
    "No score claim is made by this planner. Atom rankings are byte/prior "
    "planning signals only; every selected archive requires exact CUDA auth "
    "eval before ranking, promotion, retirement, or paper claims."
)


class AlphaRepairAtomPlannerError(ValueError):
    """Raised when planner inputs are invalid or over-claimed."""


def _load_archive_builder() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_alpha_mask_replacement_archive_for_atom_planner",
        ARCHIVE_BUILDER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load archive builder from {ARCHIVE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise AlphaRepairAtomPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AlphaRepairAtomPlannerError(f"{path} must contain a JSON object")
    return payload


def _file_meta(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlphaRepairAtomPlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise AlphaRepairAtomPlannerError(f"{field} must be finite")
    return out


def _float_list(value: Any, *, field: str, expected_len: int) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != expected_len:
        raise AlphaRepairAtomPlannerError(f"{field} must be a list of length {expected_len}")
    return [_finite_number(item, field=f"{field}[{idx}]") for idx, item in enumerate(value)]


def _validate_pair_index_list(
    values: Any,
    *,
    field: str,
    expected_len: int | None = None,
) -> list[int]:
    if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
        raise AlphaRepairAtomPlannerError(f"{field} must be integer list")
    if expected_len is not None and len(values) != expected_len:
        raise AlphaRepairAtomPlannerError(f"{field} must contain {expected_len} entries")
    if any(v < 0 or v >= EXPECTED_PAIR_COUNT for v in values):
        raise AlphaRepairAtomPlannerError(f"{field} outside [0,{EXPECTED_PAIR_COUNT - 1}]")
    if len(set(values)) != len(values):
        raise AlphaRepairAtomPlannerError(f"{field} contains duplicates")
    return [int(v) for v in values]


def _load_component_trace_pair_meta(
    path: Path,
    payload: dict[str, Any],
    *,
    pair_signal_top_k: int,
) -> dict[str, Any]:
    if pair_signal_top_k <= 0:
        raise AlphaRepairAtomPlannerError("pair_signal_top_k must be positive")
    if payload.get("score_claim") is not False:
        raise AlphaRepairAtomPlannerError(f"{path}: component trace must have score_claim=false")
    if payload.get("evidence_grade") != "diagnostic_component_trace":
        raise AlphaRepairAtomPlannerError(
            f"{path}: component trace evidence_grade must be diagnostic_component_trace"
        )
    n_samples = payload.get("n_samples")
    if n_samples != EXPECTED_PAIR_COUNT:
        raise AlphaRepairAtomPlannerError(
            f"{path}: component trace n_samples must be {EXPECTED_PAIR_COUNT}, got {n_samples!r}"
        )
    cross = payload.get("contest_auth_eval_cross_check")
    if not isinstance(cross, dict) or cross.get("all_match") is not True:
        raise AlphaRepairAtomPlannerError(
            f"{path}: component trace must cross-check against contest_auth_eval.json"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_PAIR_COUNT:
        raise AlphaRepairAtomPlannerError(
            f"{path}: component trace samples must contain {EXPECTED_PAIR_COUNT} records"
        )

    pose = [0.0] * EXPECTED_PAIR_COUNT
    seg = [0.0] * EXPECTED_PAIR_COUNT
    combined = [0.0] * EXPECTED_PAIR_COUNT
    seen: set[int] = set()
    avg_pose = _finite_number(payload.get("avg_posenet_dist"), field="avg_posenet_dist")
    for idx, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise AlphaRepairAtomPlannerError(f"{path}: samples[{idx}] must be an object")
        pair_index = raw.get("pair_index")
        if not isinstance(pair_index, int):
            raise AlphaRepairAtomPlannerError(f"{path}: samples[{idx}].pair_index must be int")
        if pair_index < 0 or pair_index >= EXPECTED_PAIR_COUNT:
            raise AlphaRepairAtomPlannerError(f"{path}: samples[{idx}].pair_index out of range")
        if pair_index in seen:
            raise AlphaRepairAtomPlannerError(f"{path}: duplicate pair_index={pair_index}")
        seen.add(pair_index)
        pose_val = _finite_number(raw.get("posenet_dist"), field=f"samples[{idx}].posenet_dist")
        seg_val = _finite_number(raw.get("segnet_dist"), field=f"samples[{idx}].segnet_dist")
        signal = raw.get("score_combined_contribution_first_order")
        if signal is None:
            seg_signal = 100.0 * seg_val / EXPECTED_PAIR_COUNT
            pose_signal = 0.0
            if avg_pose > 0.0:
                pose_signal = (5.0 / math.sqrt(10.0 * avg_pose)) * (pose_val / EXPECTED_PAIR_COUNT)
            signal = seg_signal + pose_signal
        pose[pair_index] = pose_val
        seg[pair_index] = seg_val
        combined[pair_index] = _finite_number(
            signal,
            field=f"samples[{idx}].score_combined_contribution_first_order",
        )
    if seen != set(range(EXPECTED_PAIR_COUNT)):
        missing = sorted(set(range(EXPECTED_PAIR_COUNT)) - seen)[:8]
        raise AlphaRepairAtomPlannerError(f"{path}: missing component trace pairs: {missing}")

    ranked_pairs = sorted(
        range(EXPECTED_PAIR_COUNT),
        key=lambda pair: (combined[pair], pose[pair], seg[pair], -pair),
        reverse=True,
    )
    hardest = ranked_pairs[: min(pair_signal_top_k, EXPECTED_PAIR_COUNT)]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_schema": "diagnostic_component_trace",
        "signal_source": "score_combined_contribution_first_order",
        "pair_signal_top_k": min(pair_signal_top_k, EXPECTED_PAIR_COUNT),
        "hardest_pair_indices": hardest,
        "hardest_pair_set": set(hardest),
        "per_pair_pose_dist": pose,
        "per_pair_seg_dist": seg,
        "per_pair_combined_score_signal": combined,
        "stats": {
            "avg_posenet_dist": avg_pose,
            "avg_segnet_dist": _finite_number(payload.get("avg_segnet_dist"), field="avg_segnet_dist"),
            "score_recomputed_from_components": _finite_number(
                payload.get("score_recomputed_from_components"),
                field="score_recomputed_from_components",
            ),
            "archive_size_bytes": int(
                _finite_number(payload.get("archive_size_bytes"), field="archive_size_bytes")
            ),
            "contest_auth_eval_cross_check_sha256": cross.get("contest_auth_eval_json_sha256"),
            "top_signal_pair_indices": hardest,
        },
        "lane": "component_trace",
        "mode": "cuda_cross_checked_first_order_signal",
    }


def _load_pair_meta(
    path: Path | None,
    *,
    pair_signal_top_k: int = DEFAULT_COMPONENT_TRACE_TOP_K,
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _read_json(path)
    if payload.get("evidence_grade") == "diagnostic_component_trace" or "samples" in payload:
        return _load_component_trace_pair_meta(
            path,
            payload,
            pair_signal_top_k=pair_signal_top_k,
        )
    n_pairs = payload.get("n_pairs")
    if n_pairs != EXPECTED_PAIR_COUNT:
        raise AlphaRepairAtomPlannerError(
            f"{path}: n_pairs must be {EXPECTED_PAIR_COUNT}, got {n_pairs!r}"
        )
    hardest = _validate_pair_index_list(
        payload.get("hardest_pair_indices", []),
        field=f"{path}: hardest_pair_indices",
    )
    pose = _float_list(
        payload.get("per_pair_pose_dist"),
        field="per_pair_pose_dist",
        expected_len=EXPECTED_PAIR_COUNT,
    )
    seg = _float_list(
        payload.get("per_pair_seg_dist"),
        field="per_pair_seg_dist",
        expected_len=EXPECTED_PAIR_COUNT,
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_schema": payload.get("schema") or payload.get("schema_version"),
        "signal_source": "legacy_pose_seg_formula",
        "pair_signal_top_k": None,
        "hardest_pair_indices": hardest,
        "hardest_pair_set": set(hardest),
        "per_pair_pose_dist": pose,
        "per_pair_seg_dist": seg,
        "per_pair_combined_score_signal": None,
        "stats": payload.get("stats", {}),
        "lane": payload.get("lane"),
        "mode": payload.get("mode"),
    }


def _contest_components(path: Path, *, label: str) -> dict[str, Any]:
    payload = _read_json(path)
    score = payload.get("score_recomputed_from_components", payload.get("final_score"))
    archive_bytes = payload.get("archive_size_bytes")
    return {
        "file": _file_meta(path),
        "score": _finite_number(score, field=f"{label}.score"),
        "archive_size_bytes": int(_finite_number(archive_bytes, field=f"{label}.archive_size_bytes")),
        "avg_posenet_dist": _finite_number(
            payload.get("avg_posenet_dist"),
            field=f"{label}.avg_posenet_dist",
        ),
        "avg_segnet_dist": _finite_number(
            payload.get("avg_segnet_dist"),
            field=f"{label}.avg_segnet_dist",
        ),
        "n_samples": payload.get("n_samples"),
        "archive_sha256": (payload.get("provenance") or {}).get("archive_sha256"),
        "device": (payload.get("provenance") or {}).get("device"),
    }


def _geometry_basin_check(
    *,
    baseline_contest_json: Path | None,
    candidate_contest_json: Path | None,
    max_posenet_relative: float,
    max_segnet_relative: float,
) -> dict[str, Any] | None:
    if baseline_contest_json is None and candidate_contest_json is None:
        return None
    if baseline_contest_json is None or candidate_contest_json is None:
        raise AlphaRepairAtomPlannerError(
            "--baseline-contest-json and --candidate-contest-json must be provided together"
        )
    if max_posenet_relative <= 0 or max_segnet_relative <= 0:
        raise AlphaRepairAtomPlannerError("component basin relative limits must be positive")
    baseline = _contest_components(baseline_contest_json.resolve(), label="baseline")
    candidate = _contest_components(candidate_contest_json.resolve(), label="candidate")

    def ratio(metric: str) -> float:
        reference = baseline[metric]
        if reference <= 0:
            raise AlphaRepairAtomPlannerError(f"baseline {metric} must be positive")
        return candidate[metric] / reference

    pose_ratio = ratio("avg_posenet_dist")
    seg_ratio = ratio("avg_segnet_dist")
    violations = []
    if pose_ratio > max_posenet_relative:
        violations.append(
            {
                "component": "posenet",
                "metric": "avg_posenet_dist",
                "observed": candidate["avg_posenet_dist"],
                "reference": baseline["avg_posenet_dist"],
                "relative_to_reference": pose_ratio,
                "max_relative": max_posenet_relative,
            }
        )
    if seg_ratio > max_segnet_relative:
        violations.append(
            {
                "component": "segnet",
                "metric": "avg_segnet_dist",
                "observed": candidate["avg_segnet_dist"],
                "reference": baseline["avg_segnet_dist"],
                "relative_to_reference": seg_ratio,
                "max_relative": max_segnet_relative,
            }
        )
    return {
        "schema": "alpha_repair_geometry_basin_check_v1",
        "passed": not violations,
        "water_filling_allowed": not violations,
        "baseline": baseline,
        "candidate": candidate,
        "limits": {
            "max_posenet_relative": max_posenet_relative,
            "max_segnet_relative": max_segnet_relative,
        },
        "ratios": {
            "posenet_relative": round(pose_ratio, 12),
            "segnet_relative": round(seg_ratio, 12),
        },
        "violations": violations,
    }


def _atom_key(run: Any, atom_kind: str) -> tuple[Any, ...]:
    frame_index = int(run.frame_index)
    pair_index = frame_index // 2
    class_id = int(run.class_id)
    if atom_kind == "pair":
        return (pair_index,)
    if atom_kind == "frame":
        return (frame_index,)
    if atom_kind == "pair_class":
        return (pair_index, class_id)
    if atom_kind == "frame_class":
        return (frame_index, class_id)
    raise AssertionError(f"unexpected atom kind {atom_kind!r}")


def _atom_identity(key: tuple[Any, ...], atom_kind: str) -> dict[str, Any]:
    if atom_kind == "pair":
        pair = int(key[0])
        return {"pair_index": pair, "frame_indices": [pair * 2, pair * 2 + 1]}
    if atom_kind == "frame":
        frame = int(key[0])
        return {"frame_index": frame, "pair_index": frame // 2}
    if atom_kind == "pair_class":
        pair = int(key[0])
        return {
            "pair_index": pair,
            "frame_indices": [pair * 2, pair * 2 + 1],
            "class_id": int(key[1]),
        }
    if atom_kind == "frame_class":
        frame = int(key[0])
        return {"frame_index": frame, "pair_index": frame // 2, "class_id": int(key[1])}
    raise AssertionError(f"unexpected atom kind {atom_kind!r}")


def _pairs_for_runs(runs: Iterable[Any]) -> list[int]:
    return sorted({int(run.frame_index) // 2 for run in runs})


def _frames_for_runs(runs: Iterable[Any]) -> list[int]:
    return sorted({int(run.frame_index) for run in runs})


def _classes_for_runs(runs: Iterable[Any]) -> list[int]:
    return sorted({int(run.class_id) for run in runs})


def _prior_for_pairs(pairs: list[int], pair_meta: dict[str, Any] | None) -> dict[str, Any]:
    if pair_meta is None:
        return {
            "hard_pair_count": None,
            "pose_prior_sum": None,
            "seg_prior_sum": None,
            "formula_prior_sum": None,
        }
    hardest = pair_meta["hardest_pair_set"]
    pose = pair_meta.get("per_pair_pose_dist")
    seg = pair_meta.get("per_pair_seg_dist")
    score_signal = pair_meta.get("per_pair_combined_score_signal")
    pose_sum = sum(pose[i] for i in pairs) if pose is not None else None
    seg_sum = sum(seg[i] for i in pairs) if seg is not None else None
    score_signal_sum = sum(score_signal[i] for i in pairs) if score_signal is not None else None
    formula_sum = None
    if pose is not None and seg is not None:
        formula_sum = sum(100.0 * seg[i] + math.sqrt(10.0 * pose[i]) for i in pairs)
    elif pose is not None:
        formula_sum = sum(math.sqrt(10.0 * pose[i]) for i in pairs)
    return {
        "hard_pair_count": sum(1 for i in pairs if i in hardest),
        "hard_pair_indices": [i for i in pairs if i in hardest],
        "pose_prior_sum": round(pose_sum, 12) if pose_sum is not None else None,
        "seg_prior_sum": round(seg_sum, 12) if seg_sum is not None else None,
        "score_signal_prior_sum": round(score_signal_sum, 12)
        if score_signal_sum is not None
        else None,
        "formula_prior_sum": round(formula_sum, 12) if formula_sum is not None else None,
    }


def _encode_atom_payload(
    *,
    archive_builder: Any,
    alpha_builder: Any,
    runs: list[Any],
    full_header: dict[str, Any],
    atom_kind: str,
    identity: dict[str, Any],
) -> bytes:
    selected_pixels = sum(int(run.length) for run in runs)
    selection_meta = {
        "strategy": "alpha_repair_atom_plan_selected_atom_v1",
        "policy_kind": atom_kind,
        "policy_details": identity,
        "total_residual_pixels": int(full_header.get("selection", {}).get("total_residual_pixels", 0)),
        "selected_repair_pixels": int(selected_pixels),
        "selected_repair_runs": int(len(runs)),
        "partial_repair": True,
        "fail_on_partial_repair": False,
        "source_manifest_selection": full_header.get("selection", {}),
    }
    return alpha_builder._encode_repair_payload(
        runs,
        shape=tuple(int(v) for v in full_header["shape"]),
        source_mask_sha256=str(full_header["source_mask_u8_sha256"]),
        candidate_mask_sha256=str(full_header["candidate_mask_u8_sha256"]),
        selection_meta=selection_meta,
    )


def build_atom_plan(
    *,
    candidate_manifest_path: Path,
    output_json: Path,
    pair_weights_meta: Path | None,
    atom_kind: str,
    compressor: str,
    max_atoms: int,
    top_policy_counts: tuple[int, ...],
    pair_signal_top_k: int = DEFAULT_COMPONENT_TRACE_TOP_K,
    baseline_contest_json: Path | None = None,
    candidate_contest_json: Path | None = None,
    max_posenet_relative: float = 1.25,
    max_segnet_relative: float = 1.25,
) -> dict[str, Any]:
    if atom_kind not in {"pair", "frame", "pair_class", "frame_class"}:
        raise AlphaRepairAtomPlannerError(f"unsupported atom kind {atom_kind!r}")
    if max_atoms <= 0:
        raise AlphaRepairAtomPlannerError("max_atoms must be positive")
    if any(count <= 0 for count in top_policy_counts):
        raise AlphaRepairAtomPlannerError("top_policy_counts must be positive")

    archive_builder = _load_archive_builder()
    alpha_builder = archive_builder._load_alpha_builder_module()
    candidate_manifest_path = candidate_manifest_path.resolve()
    candidate_manifest = archive_builder._load_candidate_manifest(candidate_manifest_path)
    repair_record = archive_builder._find_repair_artifact(
        candidate_manifest,
        manifest_dir=candidate_manifest_path.parent,
    )
    full_payload = Path(repair_record["resolved_path"]).read_bytes()
    full_header, full_runs = alpha_builder._decode_repair_payload(full_payload)
    total_pixels = int(full_header.get("selection", {}).get("total_residual_pixels", 0))
    pair_meta = _load_pair_meta(pair_weights_meta, pair_signal_top_k=pair_signal_top_k)
    geometry_check = _geometry_basin_check(
        baseline_contest_json=baseline_contest_json,
        candidate_contest_json=candidate_contest_json,
        max_posenet_relative=max_posenet_relative,
        max_segnet_relative=max_segnet_relative,
    )
    water_filling_allowed = geometry_check is None or bool(geometry_check["water_filling_allowed"])

    grouped: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for run in full_runs:
        grouped[_atom_key(run, atom_kind)].append(run)

    atoms: list[dict[str, Any]] = []
    for key, runs in grouped.items():
        identity = _atom_identity(key, atom_kind)
        pairs = _pairs_for_runs(runs)
        frames = _frames_for_runs(runs)
        classes = _classes_for_runs(runs)
        raw_payload = _encode_atom_payload(
            archive_builder=archive_builder,
            alpha_builder=alpha_builder,
            runs=runs,
            full_header=full_header,
            atom_kind=atom_kind,
            identity=identity,
        )
        member_name, compressed = archive_builder._compress_repair_payload(raw_payload, compressor)
        selected_pixels = sum(int(run.length) for run in runs)
        prior = _prior_for_pairs(pairs, pair_meta)
        compressed_bytes = len(compressed)
        score_signal_prior = prior.get("score_signal_prior_sum")
        formula_prior = prior.get("formula_prior_sum")
        hard_pair_count = prior.get("hard_pair_count")
        atom = {
            "atom_kind": atom_kind,
            "identity": identity,
            "pair_indices": pairs,
            "frame_indices": frames,
            "class_ids": classes,
            "selected_repair_pixels": int(selected_pixels),
            "selected_repair_runs": int(len(runs)),
            "residual_pixel_coverage": round(0.0 if total_pixels == 0 else selected_pixels / total_pixels, 12),
            "raw_amr1_bytes": len(raw_payload),
            "raw_amr1_sha256": sha256_bytes(raw_payload),
            "compressed_member_name": member_name,
            "compressed_bytes": compressed_bytes,
            "compressed_sha256": sha256_bytes(compressed),
            "rate_term_cost": round(25.0 * compressed_bytes / 37_545_489, 12),
            "pixels_per_compressed_byte": round(selected_pixels / compressed_bytes, 12)
            if compressed_bytes
            else None,
            "prior": prior,
            "score_signal_prior_per_compressed_byte": round(score_signal_prior / compressed_bytes, 12)
            if isinstance(score_signal_prior, (int, float)) and compressed_bytes
            else None,
            "proxy_formula_prior_per_compressed_byte": round(formula_prior / compressed_bytes, 12)
            if isinstance(formula_prior, (int, float)) and compressed_bytes
            else None,
            "hard_pair_count_per_compressed_byte": round(hard_pair_count / compressed_bytes, 12)
            if isinstance(hard_pair_count, int) and compressed_bytes
            else None,
        }
        atoms.append(atom)

    def sort_key(atom: dict[str, Any]) -> tuple[float, float, float, float, int]:
        signal = atom.get("score_signal_prior_per_compressed_byte")
        proxy = atom.get("proxy_formula_prior_per_compressed_byte")
        hard = atom.get("hard_pair_count_per_compressed_byte")
        pixels = atom.get("pixels_per_compressed_byte")
        return (
            float(signal) if signal is not None else -1.0,
            float(proxy) if proxy is not None else -1.0,
            float(hard) if hard is not None else -1.0,
            float(pixels) if pixels is not None else -1.0,
            -int(atom["compressed_bytes"]),
        )

    atoms_sorted = sorted(atoms, key=sort_key, reverse=True)
    top_atoms = atoms_sorted[:max_atoms]

    recommendations: list[dict[str, Any]] = []
    if atom_kind == "pair" and water_filling_allowed:
        for count in top_policy_counts:
            selected = top_atoms[:count]
            pair_indices = [int(atom["identity"]["pair_index"]) for atom in selected]
            if pair_indices:
                recommendations.append(
                    {
                        "policy_kind": "pair_indices",
                        "policy_name": "pair_indices_" + "_".join(str(i) for i in pair_indices),
                        "selected_atom_count": len(pair_indices),
                        "pair_indices": pair_indices,
                        "estimated_atom_compressed_bytes_sum": sum(
                            int(atom["compressed_bytes"]) for atom in selected
                        ),
                        "estimated_atom_rate_term_sum": round(
                            sum(float(atom["rate_term_cost"]) for atom in selected), 12
                        ),
                        "note": (
                            "Byte sum is atom-local only. Build a concrete archive "
                            "because combined AMR1 compression is not additive."
                        ),
                    }
                )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "producer": PRODUCER,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "score_claim_warning": SCORE_CLAIM_WARNING,
        "water_filling_allowed": water_filling_allowed,
        "water_filling_blockers": []
        if water_filling_allowed
        else [
            (
                "geometry_basin_check_failed: exact candidate components exceed "
                "baseline relative limits, so atom-local residual byte/prior "
                "rankings are not valid archive policies for this base."
            )
        ],
        "geometry_basin_check": geometry_check,
        "atom_kind": atom_kind,
        "compressor": compressor,
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "repair_artifact": {
            "path": str(repair_record["resolved_path"]),
            "sha256": sha256_file(Path(repair_record["resolved_path"])),
            "bytes": int(Path(repair_record["resolved_path"]).stat().st_size),
            "manifest_record": repair_record,
        },
        "repair_header": {
            "shape": full_header.get("shape"),
            "source_mask_u8_sha256": full_header.get("source_mask_u8_sha256"),
            "candidate_mask_u8_sha256": full_header.get("candidate_mask_u8_sha256"),
            "total_residual_pixels": total_pixels,
            "total_residual_runs": len(full_runs),
        },
        "pair_weights_meta": None
        if pair_meta is None
        else {
            "path": pair_meta["path"],
            "sha256": pair_meta["sha256"],
            "source_schema": pair_meta.get("source_schema"),
            "signal_source": pair_meta.get("signal_source"),
            "pair_signal_top_k": pair_meta.get("pair_signal_top_k"),
            "lane": pair_meta.get("lane"),
            "mode": pair_meta.get("mode"),
            "hardest_pair_indices": pair_meta["hardest_pair_indices"],
            "stats": pair_meta.get("stats"),
        },
        "atom_count": len(atoms),
        "top_atoms": top_atoms,
        "recommended_archive_policies": recommendations,
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        },
    }
    _write_json(output_json, payload)
    return payload


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(token) for token in value.split(",") if token.strip())
    if not counts:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("counts must be positive")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pair-weights-meta", type=Path, default=None)
    parser.add_argument("--baseline-contest-json", type=Path, default=None)
    parser.add_argument("--candidate-contest-json", type=Path, default=None)
    parser.add_argument("--max-posenet-relative", type=float, default=1.25)
    parser.add_argument("--max-segnet-relative", type=float, default=1.25)
    parser.add_argument(
        "--atom-kind",
        choices=("pair", "frame", "pair_class", "frame_class"),
        default="pair",
    )
    parser.add_argument(
        "--compressor",
        choices=("raw", "zlib", "lzma_xz", "brotli"),
        default="lzma_xz",
    )
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--pair-signal-top-k", type=int, default=DEFAULT_COMPONENT_TRACE_TOP_K)
    parser.add_argument("--top-policy-counts", type=_parse_counts, default=(10, 20, 30))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_atom_plan(
        candidate_manifest_path=args.candidate_manifest,
        output_json=args.output_json,
        pair_weights_meta=args.pair_weights_meta,
        baseline_contest_json=args.baseline_contest_json,
        candidate_contest_json=args.candidate_contest_json,
        max_posenet_relative=args.max_posenet_relative,
        max_segnet_relative=args.max_segnet_relative,
        atom_kind=args.atom_kind,
        compressor=args.compressor,
        max_atoms=args.max_atoms,
        pair_signal_top_k=args.pair_signal_top_k,
        top_policy_counts=tuple(args.top_policy_counts),
    )
    print(
        "[alpha-repair-atom-planner] wrote "
        f"{args.output_json} atoms={payload['atom_count']} top={len(payload['top_atoms'])} "
        f"score_claim={payload['score_claim']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
