#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local-only Lane 12 geometry-gate repair atoms.

The planner consumes existing Alpha-Geo geometry packets, primitive-contract
metadata, and optionally predecoded mask-cache tensors. It emits deterministic
atom/policy JSON only: no archive is built, no scorer is run, and no promotion
or score claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lane12_geometry_gate_repair_atom_plan_v1"
PRODUCER = "experiments/plan_lane12_geometry_gate_repair_atoms.py"
DEFAULT_GEOMETRY_JSON = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_12_nerv_20260430_codex_jsonfix40"
    / "alpha_geo_0_vs_lane_g_v3_codex_current_20260501.json"
)
DEFAULT_PRIMITIVE_CONTRACT_JSON = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_12_nerv_20260430_codex_jsonfix40"
    / "alpha_geo_1_vs_pfp16_repair_regions_20260501T080036Z.primitive_contract.json"
)
DEFAULT_MASK_CACHE_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_12_nerv_20260430_codex_jsonfix40"
    / "predecoded_mask_cache"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane12_geometry_gate_repair_atoms_20260503"
    / "lane12_geometry_gate_repair_atoms.json"
)
EXPECTED_SHAPE = [1200, 384, 512]
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
CUDA_AUTH_EVAL_SOURCE = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NO_SCORE_WARNING = (
    "This is a local planning/profiling artifact only. It does not build a "
    "byte-closed archive, does not run CUDA auth eval, and cannot promote, "
    "rank, retire, or support a paper score claim."
)
RATE_DENOMINATOR_BYTES = 37_545_489
DEFAULT_POLICY_BUDGETS = (512, 1024, 2048, 4096, 8192, 16384)


class Lane12GeometryRepairPlannerError(ValueError):
    """Raised for invalid planner inputs."""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Lane12GeometryRepairPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise Lane12GeometryRepairPlannerError(f"{path} JSON root must be an object")
    return payload


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_meta(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
    if path.is_file():
        record.update({"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    return record


def _finite_float(value: Any, *, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lane12GeometryRepairPlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise Lane12GeometryRepairPlannerError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise Lane12GeometryRepairPlannerError(f"{field} must be int")
    return int(value)


def _box_area(box: list[Any]) -> int:
    if len(box) != 4:
        raise Lane12GeometryRepairPlannerError("box_xyxy must have four coordinates")
    x0, y0, x1, y1 = (_int_value(v, field="box_xyxy") for v in box)
    if x1 < x0 or y1 < y0:
        raise Lane12GeometryRepairPlannerError(f"invalid box_xyxy={box!r}")
    return max(0, x1 - x0) * max(0, y1 - y0)


def _normalize_shape(value: Any) -> list[int] | None:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, dict):
        keys = ("frames", "height", "width")
        if all(k in value for k in keys):
            return [int(value[k]) for k in keys]
    return None


def _rate_score_cost(byte_count: int) -> float:
    return 25.0 * float(byte_count) / float(RATE_DENOMINATOR_BYTES)


def _load_tensor(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
        raise Lane12GeometryRepairPlannerError("torch is required to load decoded mask tensors") from exc
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for key in ("masks", "mask", "decoded_masks", "tensor"):
            if key in tensor:
                tensor = tensor[key]
                break
    if not hasattr(tensor, "ndim"):
        raise Lane12GeometryRepairPlannerError(f"{path} did not contain a tensor-like object")
    if tensor.ndim == 4 and int(tensor.shape[1]) == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise Lane12GeometryRepairPlannerError(f"{path} tensor must be THW; got {tuple(tensor.shape)}")
    return tensor.cpu().to(dtype=__import__("torch").uint8).contiguous()


def _mask_cache_records(mask_cache_dir: Path) -> list[dict[str, Any]]:
    if not mask_cache_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for meta_path in sorted(mask_cache_dir.glob("*.json")):
        payload = _read_json(meta_path)
        tensor_file = payload.get("tensor_file")
        tensor_path = meta_path.with_name(str(tensor_file)) if tensor_file else None
        fingerprint = payload.get("fingerprint") if isinstance(payload.get("fingerprint"), dict) else {}
        records.append(
            {
                "metadata_path": meta_path,
                "metadata_sha256": _sha256_file(meta_path),
                "payload": payload,
                "tensor_path": tensor_path,
                "tensor_exists": bool(tensor_path and tensor_path.is_file()),
                "archive_member": fingerprint.get("archive_member_resolved"),
                "source_sha256": fingerprint.get("source_sha256"),
                "decoded_mask_sha256": payload.get("decoded_mask_sha256"),
                "decoded_mask_shape": payload.get("decoded_mask_shape"),
            }
        )
    return records


def _select_mask_cache_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    baseline = None
    candidate = None
    for record in records:
        member = str(record.get("archive_member") or "")
        if member.endswith(".mkv") and baseline is None:
            baseline = record
        if member.endswith(".nrv") and candidate is None:
            candidate = record
    return baseline, candidate


def _tensor_region_stats(
    baseline: Any | None,
    candidate: Any | None,
    *,
    frame: int,
    box_xyxy: list[Any],
) -> dict[str, Any] | None:
    if baseline is None or candidate is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in box_xyxy)
    if frame < 0 or frame >= int(baseline.shape[0]):
        raise Lane12GeometryRepairPlannerError(f"frame {frame} outside tensor shape {tuple(baseline.shape)}")
    crop_base = baseline[frame, y0:y1, x0:x1]
    crop_cand = candidate[frame, y0:y1, x0:x1]
    if tuple(crop_base.shape) != tuple(crop_cand.shape):
        raise Lane12GeometryRepairPlannerError("baseline/candidate crop shape mismatch")
    diff = crop_base != crop_cand
    changed = int(diff.sum().item())
    row_runs = 0
    classes: dict[int, int] = {}
    if changed:
        import torch

        changed_rows = diff.nonzero(as_tuple=False)
        for y in torch.unique(changed_rows[:, 0], sorted=True):
            xs = changed_rows[changed_rows[:, 0] == y][:, 1].tolist()
            prev = None
            for x in xs:
                if prev is None or int(x) != int(prev) + 1:
                    row_runs += 1
                prev = int(x)
        values, counts = torch.unique(crop_base[diff], return_counts=True)
        classes = {int(v.item()): int(c.item()) for v, c in zip(values, counts, strict=True)}
    return {
        "tensor_measured": True,
        "changed_pixels": changed,
        "row_run_count": row_runs,
        "baseline_class_hist": {str(k): v for k, v in sorted(classes.items())},
        "crop_shape": list(crop_base.shape),
    }


def _estimate_patch_cost(*, area_px: int, changed_pixels: int, row_runs: int | None) -> dict[str, Any]:
    changed_pixels = max(0, int(changed_pixels))
    area_px = max(0, int(area_px))
    if row_runs is None:
        row_runs = max(1, int(math.ceil(max(changed_pixels, area_px) / 12.0))) if max(changed_pixels, area_px) else 0
        model = "estimated_from_area"
    else:
        model = "tensor_measured_row_runs"
    header_bytes = 12
    side_info_bytes = 18
    run_payload_bytes = row_runs * 8
    bitmap_payload_bytes = int(math.ceil(max(area_px, changed_pixels) / 8.0)) + changed_pixels
    estimated_payload_bytes = min(run_payload_bytes, bitmap_payload_bytes) + header_bytes
    charged = estimated_payload_bytes + side_info_bytes
    return {
        "model": model,
        "area_px": area_px,
        "changed_pixels": changed_pixels,
        "row_run_count": row_runs,
        "estimated_payload_bytes": int(estimated_payload_bytes),
        "estimated_side_info_bytes": side_info_bytes,
        "estimated_charged_bytes": int(charged),
        "estimated_rate_score_cost": _rate_score_cost(int(charged)),
        "charge_status": "planning_estimate_not_archive_measured",
    }


def _promotion_gap(geometry: dict[str, Any]) -> dict[str, Any]:
    global_metrics = geometry.get("global") if isinstance(geometry.get("global"), dict) else {}
    temporal = geometry.get("temporal") if isinstance(geometry.get("temporal"), dict) else {}
    pair_transition = temporal.get("pair_transition") if isinstance(temporal.get("pair_transition"), dict) else {}
    stable = temporal.get("stable_region") if isinstance(temporal.get("stable_region"), dict) else {}
    components = geometry.get("components") if isinstance(geometry.get("components"), dict) else {}
    centroid = components.get("centroid") if isinstance(components.get("centroid"), dict) else {}
    observed = {
        "global_disagreement": global_metrics.get("global_disagreement"),
        "pair_transition_disagreement": pair_transition.get("disagreement_rate"),
        "stable_region_false_flip_rate": stable.get("false_flip_rate"),
        "max_component_centroid_jump_px": centroid.get("max_matched_jump_px"),
        "missing_component_rate": centroid.get("missing_component_rate"),
    }
    boundary = geometry.get("boundary_bands") if isinstance(geometry.get("boundary_bands"), dict) else {}
    observed["boundary_band_disagreement"] = {
        str(radius): values.get("disagreement_rate")
        for radius, values in sorted(boundary.items())
        if isinstance(values, dict)
    }
    return {"observed": observed, "required": PROMOTION_THRESHOLDS}


def _atom_sort_key(atom: dict[str, Any]) -> tuple[float, float, int, str]:
    score = float(atom["priority"]["planning_score_per_byte"])
    signal = float(atom["priority"]["planning_signal"])
    rank = int(atom["identity"].get("rank", 10_000))
    return (-score, -signal, rank, str(atom["atom_id"]))


def _source_record(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    record = _path_meta(path)
    record["diagnostic"] = payload.get("diagnostic")
    record["schema_version"] = payload.get("schema_version")
    return {"path": record, "payload": payload}


def _region_atoms(
    geometry: dict[str, Any],
    *,
    baseline_tensor: Any | None,
    candidate_tensor: Any | None,
    max_atoms: int,
) -> list[dict[str, Any]]:
    ranking = geometry.get("residual_region_ranking")
    if not isinstance(ranking, dict) or not isinstance(ranking.get("regions"), list):
        return []
    atoms: list[dict[str, Any]] = []
    for raw in ranking["regions"][:max_atoms]:
        if not isinstance(raw, dict):
            continue
        frame = _int_value(raw.get("frame"), field="region.frame")
        box = raw.get("box_xyxy")
        if not isinstance(box, list):
            continue
        area = _int_value(raw.get("area_px"), field="region.area_px", default=_box_area(box))
        tensor_stats = _tensor_region_stats(baseline_tensor, candidate_tensor, frame=frame, box_xyxy=box)
        changed = int(tensor_stats["changed_pixels"]) if tensor_stats else area
        row_runs = int(tensor_stats["row_run_count"]) if tensor_stats else None
        cost = _estimate_patch_cost(area_px=area, changed_pixels=changed, row_runs=row_runs)
        critical = _finite_float(raw.get("critical_class_pixels"), field="critical_class_pixels", default=0.0)
        boundary = _finite_float(raw.get("boundary_band_pixels"), field="boundary_band_pixels", default=0.0)
        transition = _finite_float(
            raw.get("temporal_transition_disagreement_pixels"),
            field="temporal_transition_disagreement_pixels",
            default=0.0,
        )
        signal = critical * 3.0 + boundary * 1.5 + transition * 2.0 + changed
        charged = max(1, int(cost["estimated_charged_bytes"]))
        atom_id = f"lane12_region_{int(raw.get('rank', len(atoms) + 1)):04d}_{raw.get('residual_region_id', frame)}"
        atom = {
            "atom_id": atom_id,
            "atom_kind": "residual_region_patch",
            "identity": {
                "rank": int(raw.get("rank", len(atoms) + 1)),
                "frame": frame,
                "box_xyxy": [int(v) for v in box],
                "residual_region_id": raw.get("residual_region_id"),
                "priority_bucket": raw.get("priority_bucket"),
                "priority_label": raw.get("priority_label"),
                "suggested_repair": raw.get("suggested_repair"),
            },
            "geometry_signal": {
                "area_px": area,
                "critical_class_pixels": int(critical),
                "boundary_band_pixels": int(boundary),
                "temporal_transition_disagreement_pixels": int(transition),
                "dominant_baseline_class": raw.get("dominant_baseline_class"),
                "dominant_candidate_class": raw.get("dominant_candidate_class"),
                "confusion_pairs": raw.get("confusion_pairs", [])[:6],
            },
            "tensor_measurement": tensor_stats,
            "cost_model": cost,
            "priority": {
                "planning_signal": float(signal),
                "planning_score_per_byte": float(signal) / float(charged),
                "score_proxy": "geometry_pixels_weighted_not_scorer",
            },
            "score_claim": False,
            "promotion_eligible": False,
            "exact_eval_claim": False,
        }
        atoms.append(atom)
    return atoms


def _critical_box_atoms(
    contract: dict[str, Any] | None,
    *,
    baseline_tensor: Any | None,
    candidate_tensor: Any | None,
    max_atoms: int,
) -> list[dict[str, Any]]:
    if contract is None or not isinstance(contract.get("ranked_critical_boxes"), list):
        return []
    atoms: list[dict[str, Any]] = []
    for raw in contract["ranked_critical_boxes"][:max_atoms]:
        if not isinstance(raw, dict):
            continue
        box = raw.get("box_xyxy")
        if not isinstance(box, list):
            continue
        frame = _int_value(raw.get("frame"), field="critical_box.frame")
        area = _int_value(raw.get("area_px"), field="critical_box.area_px", default=_box_area(box))
        tensor_stats = _tensor_region_stats(baseline_tensor, candidate_tensor, frame=frame, box_xyxy=box)
        changed = int(tensor_stats["changed_pixels"]) if tensor_stats else area
        row_runs = int(tensor_stats["row_run_count"]) if tensor_stats else None
        cost = _estimate_patch_cost(area_px=area, changed_pixels=changed, row_runs=row_runs)
        class_id = _int_value(raw.get("class_id"), field="critical_box.class_id", default=-1)
        pose_bonus = 1.5 if raw.get("pose_sensitive") is True else 1.0
        signal = float(area) * (3.5 if class_id in {1, 2} else 1.0) * pose_bonus + float(changed)
        charged = max(1, int(cost["estimated_charged_bytes"]))
        atom = {
            "atom_id": f"lane12_critical_box_{int(raw.get('rank', len(atoms) + 1)):04d}_f{frame:04d}_c{class_id}",
            "atom_kind": "critical_component_box_patch",
            "identity": {
                "rank": int(raw.get("rank", len(atoms) + 1)),
                "frame": frame,
                "box_xyxy": [int(v) for v in box],
                "class_id": class_id,
                "class_name": raw.get("class_name"),
                "failure_type": raw.get("failure_type"),
                "pose_sensitive": bool(raw.get("pose_sensitive")),
            },
            "geometry_signal": {
                "area_px": area,
                "mask_iou": raw.get("mask_iou"),
                "box_iou": raw.get("box_iou"),
                "centroid_jump_px": raw.get("centroid_jump_px"),
            },
            "tensor_measurement": tensor_stats,
            "cost_model": cost,
            "priority": {
                "planning_signal": float(signal),
                "planning_score_per_byte": float(signal) / float(charged),
                "score_proxy": "critical_component_geometry_not_scorer",
            },
            "score_claim": False,
            "promotion_eligible": False,
            "exact_eval_claim": False,
        }
        atoms.append(atom)
    return atoms


def _transition_atoms(geometry: dict[str, Any], contract: dict[str, Any] | None, *, max_atoms: int) -> list[dict[str, Any]]:
    source = None
    if contract and isinstance(contract.get("worst_transition_pairs"), list):
        source = contract["worst_transition_pairs"]
    else:
        temporal = geometry.get("temporal") if isinstance(geometry.get("temporal"), dict) else {}
        source = temporal.get("worst_frame_pairs") if isinstance(temporal.get("worst_frame_pairs"), list) else []
    atoms: list[dict[str, Any]] = []
    for raw in source[:max_atoms]:
        if not isinstance(raw, dict):
            continue
        pair_index = _int_value(raw.get("pair_index"), field="transition.pair_index")
        trans_pixels = _int_value(
            raw.get("transition_disagreement_pixels"),
            field="transition.transition_disagreement_pixels",
            default=0,
        )
        stable_pixels = _int_value(raw.get("stable_false_flip_pixels"), field="transition.stable_false_flip_pixels", default=0)
        cost = _estimate_patch_cost(area_px=trans_pixels + stable_pixels, changed_pixels=trans_pixels, row_runs=None)
        signal = trans_pixels * 2.5 + stable_pixels * 1.25
        charged = max(1, int(cost["estimated_charged_bytes"]))
        atoms.append(
            {
                "atom_id": f"lane12_transition_pair_{pair_index:04d}",
                "atom_kind": "transition_pair_focus",
                "identity": {
                    "rank": int(raw.get("rank", len(atoms) + 1)),
                    "pair_index": pair_index,
                    "frames": raw.get("frames"),
                },
                "geometry_signal": {
                    "transition_disagreement_pixels": trans_pixels,
                    "transition_disagreement_rate": raw.get("transition_disagreement_rate"),
                    "stable_false_flip_pixels": stable_pixels,
                    "stable_false_flip_rate": raw.get("stable_false_flip_rate"),
                    "pair_frame_disagreement_rate": raw.get("pair_frame_disagreement_rate"),
                },
                "tensor_measurement": None,
                "cost_model": cost,
                "priority": {
                    "planning_signal": float(signal),
                    "planning_score_per_byte": float(signal) / float(charged),
                    "score_proxy": "temporal_geometry_pixels_not_scorer",
                },
                "score_claim": False,
                "promotion_eligible": False,
                "exact_eval_claim": False,
            }
        )
    return atoms


def _policy_table(atoms: list[dict[str, Any]], *, budgets: tuple[int, ...]) -> list[dict[str, Any]]:
    ordered = sorted(atoms, key=_atom_sort_key)
    policies: list[dict[str, Any]] = []
    for budget in budgets:
        selected: list[dict[str, Any]] = []
        total_bytes = 0
        total_signal = 0.0
        for atom in ordered:
            charged = int(atom["cost_model"]["estimated_charged_bytes"])
            if selected and total_bytes + charged > budget:
                continue
            if not selected and charged > budget:
                continue
            selected.append(atom)
            total_bytes += charged
            total_signal += float(atom["priority"]["planning_signal"])
        policies.append(
            {
                "policy_id": f"lane12_geometry_gate_budget_{budget}b",
                "policy_kind": "planning_atom_selection",
                "budget_estimated_charged_bytes": budget,
                "selected_atom_ids": [atom["atom_id"] for atom in selected],
                "selected_atom_count": len(selected),
                "estimated_charged_bytes": total_bytes,
                "estimated_rate_score_cost": _rate_score_cost(total_bytes),
                "planning_signal_total": total_signal,
                "score_claim": False,
                "promotion_eligible": False,
                "exact_eval_claim": False,
                "dispatch_allowed": False,
                "builder_status": "not_byte_closed_no_archive_emitted",
            }
        )
    return policies


def build_geometry_repair_atom_plan(
    *,
    geometry_json: Path,
    primitive_contract_json: Path | None,
    mask_cache_dir: Path | None,
    output_json: Path,
    max_region_atoms: int = 20,
    max_critical_box_atoms: int = 32,
    max_transition_atoms: int = 10,
    load_tensors: bool = True,
    policy_budgets: tuple[int, ...] = DEFAULT_POLICY_BUDGETS,
) -> dict[str, Any]:
    geometry_record = _source_record(geometry_json)
    geometry = geometry_record["payload"]
    if geometry.get("diagnostic") != "alpha_geo_0_nerv_geometry":
        raise Lane12GeometryRepairPlannerError("geometry_json must be alpha_geo_0_nerv_geometry")
    shape = _normalize_shape(geometry.get("shape"))
    if shape != EXPECTED_SHAPE:
        raise Lane12GeometryRepairPlannerError(f"geometry shape must be {EXPECTED_SHAPE}, got {shape}")

    contract = None
    contract_record = None
    if primitive_contract_json is not None:
        contract_record = _source_record(primitive_contract_json)
        contract = contract_record["payload"]
        if contract.get("diagnostic") != "alpha_geo_primitive_contract_v1":
            raise Lane12GeometryRepairPlannerError("primitive_contract_json must be alpha_geo_primitive_contract_v1")

    cache_records = _mask_cache_records(mask_cache_dir) if mask_cache_dir is not None else []
    baseline_cache, candidate_cache = _select_mask_cache_records(cache_records)
    baseline_tensor = None
    candidate_tensor = None
    tensor_status = {
        "requested": bool(load_tensors),
        "loaded": False,
        "reason": "not_requested" if not load_tensors else None,
        "baseline": None,
        "candidate": None,
    }
    if load_tensors:
        if baseline_cache and candidate_cache and baseline_cache.get("tensor_path") and candidate_cache.get("tensor_path"):
            baseline_tensor = _load_tensor(Path(baseline_cache["tensor_path"]))
            candidate_tensor = _load_tensor(Path(candidate_cache["tensor_path"]))
            if list(baseline_tensor.shape) != EXPECTED_SHAPE or list(candidate_tensor.shape) != EXPECTED_SHAPE:
                raise Lane12GeometryRepairPlannerError("decoded mask tensor shape mismatch")
            tensor_status.update(
                {
                    "loaded": True,
                    "reason": None,
                    "baseline": {
                        "metadata_path": str(baseline_cache["metadata_path"]),
                        "tensor_path": str(baseline_cache["tensor_path"]),
                        "decoded_mask_sha256": baseline_cache.get("decoded_mask_sha256"),
                    },
                    "candidate": {
                        "metadata_path": str(candidate_cache["metadata_path"]),
                        "tensor_path": str(candidate_cache["tensor_path"]),
                        "decoded_mask_sha256": candidate_cache.get("decoded_mask_sha256"),
                    },
                }
            )
        else:
            tensor_status["reason"] = "baseline_or_candidate_cache_tensor_missing"

    atoms = []
    atoms.extend(
        _region_atoms(
            geometry,
            baseline_tensor=baseline_tensor,
            candidate_tensor=candidate_tensor,
            max_atoms=max_region_atoms,
        )
    )
    atoms.extend(
        _critical_box_atoms(
            contract,
            baseline_tensor=baseline_tensor,
            candidate_tensor=candidate_tensor,
            max_atoms=max_critical_box_atoms,
        )
    )
    atoms.extend(_transition_atoms(geometry, contract, max_atoms=max_transition_atoms))
    atoms = sorted(atoms, key=_atom_sort_key)

    policies = _policy_table(atoms, budgets=policy_budgets)
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "producer": PRODUCER,
        "deterministic": True,
        "repo_root": str(REPO_ROOT),
        "platform": {"python": sys.version.split()[0], "platform": platform.platform()},
        "score_claim": False,
        "promotion_eligible": False,
        "exact_eval_claim": False,
        "remote_jobs_dispatched": False,
        "byte_closed_exact_eval_candidate_created": False,
        "evidence_grade": "empirical_planning_only",
        "exact_score_source_required": CUDA_AUTH_EVAL_SOURCE,
        "warning": NO_SCORE_WARNING,
        "inputs": {
            "geometry_json": geometry_record["path"],
            "primitive_contract_json": contract_record["path"] if contract_record else None,
            "mask_cache_dir": str(mask_cache_dir) if mask_cache_dir is not None else None,
            "mask_cache_records": [
                {
                    "metadata_path": str(record["metadata_path"]),
                    "metadata_sha256": record["metadata_sha256"],
                    "tensor_path": str(record["tensor_path"]) if record.get("tensor_path") else None,
                    "tensor_exists": record["tensor_exists"],
                    "archive_member": record.get("archive_member"),
                    "decoded_mask_sha256": record.get("decoded_mask_sha256"),
                    "decoded_mask_shape": record.get("decoded_mask_shape"),
                }
                for record in cache_records
            ],
            "tensor_status": tensor_status,
        },
        "geometry_gate_state": {
            "current_pass_fail": geometry.get("pass_fail"),
            "promotion_gap": _promotion_gap(geometry),
            "blocked_for_retraining_unblock": True,
            "blocked_for_exact_eval_dispatch": True,
        },
        "atom_count": len(atoms),
        "atoms": atoms,
        "candidate_policies": policies,
        "policy_limitations": [
            "byte costs are planning estimates, not measured archive deltas",
            "policies are not dispatchable and must be converted into a charged builder before eval",
            "geometry/proxy improvement must be remeasured against promotion thresholds before retraining unblock",
            "exact CUDA auth eval is still required for any eventual archive score claim",
        ],
    }
    _write_json(output_json, payload)
    return payload


def _parse_budgets(values: list[str]) -> tuple[int, ...]:
    if not values:
        return DEFAULT_POLICY_BUDGETS
    out: list[int] = []
    for value in values:
        item = int(value)
        if item <= 0:
            raise argparse.ArgumentTypeError("policy budgets must be positive")
        out.append(item)
    return tuple(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_JSON)
    parser.add_argument("--primitive-contract-json", type=Path, default=DEFAULT_PRIMITIVE_CONTRACT_JSON)
    parser.add_argument("--mask-cache-dir", type=Path, default=DEFAULT_MASK_CACHE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-region-atoms", type=int, default=20)
    parser.add_argument("--max-critical-box-atoms", type=int, default=32)
    parser.add_argument("--max-transition-atoms", type=int, default=10)
    parser.add_argument("--no-load-tensors", action="store_true")
    parser.add_argument("--policy-budget", action="append", default=[], help="estimated charged-byte budget")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_geometry_repair_atom_plan(
        geometry_json=args.geometry_json,
        primitive_contract_json=args.primitive_contract_json,
        mask_cache_dir=args.mask_cache_dir,
        output_json=args.output_json,
        max_region_atoms=args.max_region_atoms,
        max_critical_box_atoms=args.max_critical_box_atoms,
        max_transition_atoms=args.max_transition_atoms,
        load_tensors=not args.no_load_tensors,
        policy_budgets=_parse_budgets(args.policy_budget),
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "atom_count": payload["atom_count"],
                "policy_count": len(payload["candidate_policies"]),
                "byte_closed_exact_eval_candidate_created": False,
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
