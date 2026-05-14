#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan scorer-weighted pose residual atoms for a frontier archive.

The planner is intentionally non-promotable. It reads CUDA-cross-checked
component traces, exact auth-eval custody JSON, optional public/reference
traces, optional active-subspace metadata, and optional frontier atom ledgers.
It emits deterministic JSON policies that rank charged pose-residual pair atoms
by expected scorer benefit per byte. No archive is built and no score claim is
made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "experiments/plan_scorer_weighted_pose_atoms.py"
SCHEMA_VERSION = 1
EXPECTED_SAMPLES = 600
CONTEST_ARCHIVE_BYTES_DENOMINATOR = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ARCHIVE_BYTES_DENOMINATOR
DEFAULT_C059_DIR = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z"
)
DEFAULT_REFERENCE_TRACES = (
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "archive_eval_pr67_public_qpose14_qzs3_filmq9g_slsb1_r55_20260502T0213Z/"
    "component_trace.json",
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr63_qpose14_trace_l40s_20260501T2149Z/component_trace.json",
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_public_pr64_unified_trace_l40s_20260501T2150Z/component_trace.json",
)
DEFAULT_ACTIVE_METADATA = (
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z_latest/"
    "archive.accepted_latest.json",
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z/"
    "metadata.json",
)
DEFAULT_ATOM_LEDGERS = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260502T0206Z/"
    "pr65_pr67_atom_allocation_ledger.json",
    REPO_ROOT / "experiments/results/frontier_atom_ledger_20260501/frontier_atom_ledger.json",
)
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
NON_PROMOTABLE_WARNING = (
    "Planner output is a non-promotable proposal policy only. Build a closed "
    "archive and run exact CUDA auth eval on the identical bytes before any "
    "score, rank, promotion, retirement, or paper claim."
)


class PoseAtomPlannerError(ValueError):
    """Raised when planner inputs are invalid or over-claimed."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PoseAtomPlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PoseAtomPlannerError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _finite_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PoseAtomPlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PoseAtomPlannerError(f"{field} must be finite")
    return out


def _int_value(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PoseAtomPlannerError(f"{field} must be an integer")
    return int(value)


def _file_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _pair_score_from_components(
    *,
    pose_dist: float,
    seg_dist: float,
    target_avg_pose: float,
    n_samples: int,
) -> dict[str, float]:
    pose_weight = 0.0
    if target_avg_pose > 0.0:
        pose_weight = 5.0 / math.sqrt(10.0 * target_avg_pose)
    pose_score = pose_weight * pose_dist / n_samples
    seg_score = 100.0 * seg_dist / n_samples
    return {
        "pose_score": pose_score,
        "seg_score": seg_score,
        "combined_score": pose_score + seg_score,
        "pose_weight": pose_weight,
    }


def _load_component_trace(path: Path, *, require_cross_check: bool) -> dict[str, Any]:
    payload = _read_json(path)
    n_samples = _int_value(payload.get("n_samples"), field=f"{path}:n_samples")
    if n_samples != EXPECTED_SAMPLES:
        raise PoseAtomPlannerError(
            f"{path}: expected {EXPECTED_SAMPLES} samples, got {n_samples}"
        )
    if payload.get("score_claim") is not False:
        raise PoseAtomPlannerError(f"{path}: component trace must have score_claim=false")
    if require_cross_check:
        cross = payload.get("contest_auth_eval_cross_check")
        if not isinstance(cross, dict) or cross.get("all_match") is not True:
            raise PoseAtomPlannerError(
                f"{path}: target component trace must cross-check contest_auth_eval"
            )
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != n_samples:
        raise PoseAtomPlannerError(f"{path}: samples must contain {n_samples} records")

    avg_pose = _finite_float(payload.get("avg_posenet_dist"), field=f"{path}:avg_posenet_dist")
    pairs: dict[int, dict[str, Any]] = {}
    for row, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise PoseAtomPlannerError(f"{path}: samples[{row}] must be an object")
        pair_index = _int_value(sample.get("pair_index"), field=f"{path}:samples[{row}].pair_index")
        if pair_index < 0 or pair_index >= n_samples:
            raise PoseAtomPlannerError(f"{path}: pair_index {pair_index} out of range")
        if pair_index in pairs:
            raise PoseAtomPlannerError(f"{path}: duplicate pair_index {pair_index}")
        pose_dist = _finite_float(
            sample.get("posenet_dist"),
            field=f"{path}:samples[{row}].posenet_dist",
        )
        seg_dist = _finite_float(
            sample.get("segnet_dist"),
            field=f"{path}:samples[{row}].segnet_dist",
        )
        frame_indices = sample.get("frame_indices")
        if not isinstance(frame_indices, list) or not all(
            isinstance(item, int) for item in frame_indices
        ):
            frame_indices = [2 * pair_index, 2 * pair_index + 1]
        score = _pair_score_from_components(
            pose_dist=pose_dist,
            seg_dist=seg_dist,
            target_avg_pose=avg_pose,
            n_samples=n_samples,
        )
        pairs[pair_index] = {
            "pair_index": pair_index,
            "video_name": sample.get("video_name"),
            "video_pair_index": sample.get("video_pair_index"),
            "frame_indices": [int(item) for item in frame_indices],
            "pose_dist": pose_dist,
            "seg_dist": seg_dist,
            "pose_score": score["pose_score"],
            "seg_score": score["seg_score"],
            "combined_score": score["combined_score"],
        }
    missing = sorted(set(range(n_samples)) - set(pairs))
    if missing:
        raise PoseAtomPlannerError(f"{path}: missing pair indices {missing[:8]}")
    return {
        "path": path,
        "payload": payload,
        "n_samples": n_samples,
        "avg_pose": avg_pose,
        "avg_seg": _finite_float(payload.get("avg_segnet_dist"), field=f"{path}:avg_segnet_dist"),
        "score": _finite_float(
            payload.get("score_recomputed_from_components"),
            field=f"{path}:score_recomputed_from_components",
        ),
        "archive_size_bytes": _int_value(
            payload.get("archive_size_bytes"),
            field=f"{path}:archive_size_bytes",
        ),
        "pairs": pairs,
        "meta": _file_meta(path),
    }


def _load_contest_auth_eval(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    n_samples = _int_value(payload.get("n_samples"), field=f"{path}:n_samples")
    if n_samples != EXPECTED_SAMPLES:
        raise PoseAtomPlannerError(f"{path}: expected {EXPECTED_SAMPLES} samples, got {n_samples}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise PoseAtomPlannerError(f"{path}: missing provenance object")
    if provenance.get("device") != "cuda":
        raise PoseAtomPlannerError(f"{path}: contest auth eval provenance.device must be cuda")
    archive_sha = provenance.get("archive_sha256")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise PoseAtomPlannerError(f"{path}: provenance.archive_sha256 must be a sha256")
    return {
        "path": path,
        "payload": payload,
        "score": _finite_float(
            payload.get("score_recomputed_from_components"),
            field=f"{path}:score_recomputed_from_components",
        ),
        "archive_size_bytes": _int_value(
            payload.get("archive_size_bytes"),
            field=f"{path}:archive_size_bytes",
        ),
        "avg_pose": _finite_float(payload.get("avg_posenet_dist"), field=f"{path}:avg_posenet_dist"),
        "avg_seg": _finite_float(payload.get("avg_segnet_dist"), field=f"{path}:avg_segnet_dist"),
        "archive_sha256": archive_sha,
        "gpu_model": provenance.get("gpu_model"),
        "gpu_t4_match": provenance.get("gpu_t4_match"),
        "sys_argv": provenance.get("sys_argv"),
        "meta": _file_meta(path),
    }


def _reference_label(path: Path, payload: dict[str, Any]) -> str:
    trace_inputs = payload.get("trace_inputs")
    if isinstance(trace_inputs, dict):
        label = trace_inputs.get("label") or trace_inputs.get("run_id")
        if isinstance(label, str) and label:
            return label
    return path.parent.name


def _ranked_pair_indices_from_trace(trace: dict[str, Any], *, top_k: int) -> list[int]:
    pairs = trace["pairs"]
    return sorted(
        pairs,
        key=lambda idx: (
            pairs[idx]["combined_score"],
            pairs[idx]["pose_score"],
            pairs[idx]["seg_score"],
            -idx,
        ),
        reverse=True,
    )[:top_k]


def _collect_active_subspace(paths: Iterable[Path]) -> dict[str, Any]:
    counts: dict[int, int] = {}
    sources: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        payload = _read_json(path)
        refinement = payload.get("refinement")
        if not isinstance(refinement, dict):
            refinement = payload
        pair_indices = refinement.get("basis_pair_indices")
        if not isinstance(pair_indices, list):
            continue
        clean: list[int] = []
        for item in pair_indices:
            if isinstance(item, int) and 0 <= item < EXPECTED_SAMPLES:
                clean.append(int(item))
                counts[int(item)] = counts.get(int(item), 0) + 1
        sources.append(
            {
                **_file_meta(path),
                "basis_kind": refinement.get("basis_kind"),
                "basis_index": refinement.get("basis_index"),
                "basis_signed_magnitude": refinement.get("basis_signed_magnitude"),
                "basis_pair_indices": clean,
                "evidence_grade": payload.get("evidence_grade"),
                "score_claim": payload.get("score_claim"),
            }
        )
    max_count = max(counts.values(), default=0)
    return {
        "sources": sources,
        "pair_counts": counts,
        "max_count": max_count,
    }


def _extract_ledger_pose_priors(paths: Iterable[Path]) -> dict[str, Any]:
    priors: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    charged_byte_candidates: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        payload = _read_json(path)
        sources.append(
            {
                **_file_meta(path),
                "evidence_grade": payload.get("evidence_grade"),
                "score_claim": payload.get("score_claim"),
                "active_frontier_label": payload.get("active_frontier_label"),
            }
        )
        allocation = payload.get("atom_allocation_table")
        if not isinstance(allocation, dict):
            allocation = payload
        byte_models = allocation.get("byte_models")
        if isinstance(byte_models, dict):
            for model in byte_models.values():
                if isinstance(model, dict) and isinstance(model.get("pose_pair_bytes"), (int, float)):
                    charged_byte_candidates.append(float(model["pose_pair_bytes"]))
        ranked = allocation.get("ranked_atoms")
        if not isinstance(ranked, list):
            continue
        for item in ranked:
            if not isinstance(item, dict) or item.get("family") != "pose":
                continue
            pair_index = item.get("pair_index")
            if not isinstance(pair_index, int) or not 0 <= pair_index < EXPECTED_SAMPLES:
                continue
            score_saved = item.get("expected_score_saved")
            utility = item.get("waterfill_utility_score")
            per_byte = item.get("expected_score_saved_per_byte")
            record = priors.setdefault(
                pair_index,
                {
                    "pair_index": pair_index,
                    "hits": 0,
                    "best_expected_score_saved": None,
                    "best_expected_score_saved_per_byte": None,
                    "best_waterfill_utility_score": None,
                    "source_atom_ids": [],
                },
            )
            record["hits"] += 1
            if isinstance(item.get("atom_id"), str):
                record["source_atom_ids"].append(item["atom_id"])
            if isinstance(score_saved, (int, float)) and (
                record["best_expected_score_saved"] is None
                or float(score_saved) > float(record["best_expected_score_saved"])
            ):
                record["best_expected_score_saved"] = float(score_saved)
            if isinstance(per_byte, (int, float)) and (
                record["best_expected_score_saved_per_byte"] is None
                or float(per_byte) > float(record["best_expected_score_saved_per_byte"])
            ):
                record["best_expected_score_saved_per_byte"] = float(per_byte)
            if isinstance(utility, (int, float)) and (
                record["best_waterfill_utility_score"] is None
                or float(utility) > float(record["best_waterfill_utility_score"])
            ):
                record["best_waterfill_utility_score"] = float(utility)
    charged_bytes = None
    if charged_byte_candidates:
        charged_bytes = min(charged_byte_candidates)
    return {
        "sources": sources,
        "pose_pair_priors": priors,
        "inferred_pose_pair_bytes": charged_bytes,
    }


def _best_reference_delta(
    *,
    pair_index: int,
    target_pair: dict[str, Any],
    reference_traces: list[dict[str, Any]],
    target_avg_pose: float,
    n_samples: int,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for trace in reference_traces:
        ref_pair = trace["pairs"].get(pair_index)
        if ref_pair is None:
            continue
        ref_score = _pair_score_from_components(
            pose_dist=ref_pair["pose_dist"],
            seg_dist=ref_pair["seg_dist"],
            target_avg_pose=target_avg_pose,
            n_samples=n_samples,
        )
        pose_saved = target_pair["pose_score"] - ref_score["pose_score"]
        seg_saved = target_pair["seg_score"] - ref_score["seg_score"]
        combined_saved = pose_saved + seg_saved
        candidate = {
            "source": "reference_component_delta",
            "reference_label": trace["label"],
            "reference_trace_path": str(trace["path"]),
            "reference_trace_sha256": trace["meta"]["sha256"],
            "reference_pose_dist": ref_pair["pose_dist"],
            "reference_seg_dist": ref_pair["seg_dist"],
            "expected_pose_score_saved": pose_saved,
            "expected_seg_score_saved": seg_saved,
            "expected_score_saved": combined_saved,
            "raw_component_delta_score": combined_saved,
            "measured_delta_available": True,
        }
        if best is None or (
            candidate["expected_score_saved"],
            candidate["expected_pose_score_saved"],
            -pair_index,
        ) > (
            best["expected_score_saved"],
            best["expected_pose_score_saved"],
            -pair_index,
        ):
            best = candidate
    return best


def _prior_delta(
    *,
    pair_index: int,
    target_pair: dict[str, Any],
    hard_pair_ranks: dict[int, int],
    active_subspace: dict[str, Any],
    ledger_priors: dict[int, dict[str, Any]],
    prior_savings_fraction: float,
) -> dict[str, Any]:
    hard_rank = hard_pair_ranks.get(pair_index)
    hard_weight = 0.0
    if hard_rank is not None:
        hard_weight = 1.0 / float(hard_rank + 1)
    active_count = int(active_subspace["pair_counts"].get(pair_index, 0))
    active_weight = 0.0
    if active_subspace["max_count"]:
        active_weight = active_count / float(active_subspace["max_count"])
    ledger = ledger_priors.get(pair_index)
    ledger_weight = 1.0 if ledger is not None else 0.0
    confidence = max(0.10, min(1.0, 0.58 * hard_weight + 0.32 * active_weight + 0.10 * ledger_weight))
    expected_pose_saved = target_pair["pose_score"] * prior_savings_fraction * (0.25 + 0.75 * confidence)
    return {
        "source": "hard_pair_active_subspace_prior",
        "reference_label": None,
        "reference_trace_path": None,
        "reference_trace_sha256": None,
        "expected_pose_score_saved": expected_pose_saved,
        "expected_seg_score_saved": 0.0,
        "expected_score_saved": expected_pose_saved,
        "raw_component_delta_score": None,
        "measured_delta_available": False,
        "prior": {
            "hard_pair_rank": hard_rank,
            "hard_pair_weight": hard_weight,
            "active_subspace_count": active_count,
            "active_subspace_weight": active_weight,
            "ledger_hit_count": None if ledger is None else ledger["hits"],
            "ledger_best_expected_score_saved": None
            if ledger is None
            else ledger["best_expected_score_saved"],
            "confidence": confidence,
            "prior_savings_fraction": prior_savings_fraction,
        },
    }


def _policy_name(label: str, count: int) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    return f"{safe}_pose_atoms_top{count:03d}"


def build_pose_atom_plan(
    *,
    component_trace_path: Path,
    contest_auth_eval_path: Path,
    output_json: Path | None,
    frontier_label: str = "C-059",
    expected_archive_sha256: str | None = None,
    reference_trace_paths: Iterable[Path] = (),
    active_metadata_paths: Iterable[Path] = (),
    atom_ledger_paths: Iterable[Path] = (),
    charged_bytes_per_atom: float | None = None,
    hard_pair_top_k: int = 128,
    max_atoms: int = 256,
    policy_counts: Iterable[int] = (16, 32, 64, 128),
    prior_savings_fraction: float = 0.25,
) -> dict[str, Any]:
    if hard_pair_top_k <= 0:
        raise PoseAtomPlannerError("hard_pair_top_k must be positive")
    if max_atoms <= 0:
        raise PoseAtomPlannerError("max_atoms must be positive")
    policy_counts_tuple = tuple(int(item) for item in policy_counts)
    if any(item <= 0 for item in policy_counts_tuple):
        raise PoseAtomPlannerError("policy_counts must be positive")
    if not 0.0 <= prior_savings_fraction <= 1.0:
        raise PoseAtomPlannerError("prior_savings_fraction must be in [0, 1]")

    target_trace = _load_component_trace(component_trace_path, require_cross_check=True)
    contest_eval = _load_contest_auth_eval(contest_auth_eval_path)
    if target_trace["archive_size_bytes"] != contest_eval["archive_size_bytes"]:
        raise PoseAtomPlannerError("component trace and contest auth eval archive bytes disagree")
    if abs(target_trace["score"] - contest_eval["score"]) > 1e-5:
        raise PoseAtomPlannerError("component trace and contest auth eval scores disagree")
    if expected_archive_sha256 is not None and contest_eval["archive_sha256"] != expected_archive_sha256:
        raise PoseAtomPlannerError(
            "contest auth eval archive sha does not match expected "
            f"{expected_archive_sha256}"
        )

    references: list[dict[str, Any]] = []
    for path in reference_trace_paths:
        if not path.exists():
            continue
        trace = _load_component_trace(path, require_cross_check=False)
        trace["label"] = _reference_label(path, trace["payload"])
        references.append(trace)
    references.sort(key=lambda item: (item["label"], str(item["path"])))

    active_subspace = _collect_active_subspace(active_metadata_paths)
    ledger = _extract_ledger_pose_priors(atom_ledger_paths)
    inferred_charged_bytes = ledger["inferred_pose_pair_bytes"]
    if charged_bytes_per_atom is None:
        charged_bytes_per_atom = float(inferred_charged_bytes or 4.0)
        charged_bytes_source = (
            "minimum pose_pair_bytes from atom ledger"
            if inferred_charged_bytes is not None
            else "PVR1 residual atom raw wire bytes: uint16 key + fp16 value"
        )
    else:
        charged_bytes_per_atom = float(charged_bytes_per_atom)
        charged_bytes_source = "operator override"
    if charged_bytes_per_atom <= 0.0 or not math.isfinite(charged_bytes_per_atom):
        raise PoseAtomPlannerError("charged_bytes_per_atom must be finite and positive")

    hard_pairs = _ranked_pair_indices_from_trace(target_trace, top_k=hard_pair_top_k)
    hard_pair_ranks = {pair_index: rank for rank, pair_index in enumerate(hard_pairs)}
    ledger_priors = ledger["pose_pair_priors"]

    atoms: list[dict[str, Any]] = []
    for pair_index in range(target_trace["n_samples"]):
        target_pair = target_trace["pairs"][pair_index]
        measured = _best_reference_delta(
            pair_index=pair_index,
            target_pair=target_pair,
            reference_traces=references,
            target_avg_pose=target_trace["avg_pose"],
            n_samples=target_trace["n_samples"],
        )
        if measured is not None and measured["expected_score_saved"] > 0.0:
            estimate = measured
            prior_detail = None
        else:
            estimate = _prior_delta(
                pair_index=pair_index,
                target_pair=target_pair,
                hard_pair_ranks=hard_pair_ranks,
                active_subspace=active_subspace,
                ledger_priors=ledger_priors,
                prior_savings_fraction=prior_savings_fraction,
            )
            prior_detail = estimate.pop("prior")
            if measured is not None:
                estimate["best_nonpositive_reference_delta"] = measured

        expected_score_saved = float(estimate["expected_score_saved"])
        rate_score_cost = charged_bytes_per_atom * RATE_SCORE_PER_BYTE
        net_score_utility = expected_score_saved - rate_score_cost
        risk_reasons: list[str] = []
        if estimate["expected_seg_score_saved"] < 0.0:
            risk_reasons.append("reference_delta_has_segnet_antagonism")
        if not estimate["measured_delta_available"]:
            risk_reasons.append("no_positive_reference_component_delta")
        if net_score_utility <= 0.0:
            risk_reasons.append("formula_only_rate_cost_exceeds_expected_saved")
        atom = {
            "atom_id": f"{frontier_label}:pose_residual_pair_{pair_index:04d}",
            "atom_kind": "pvr1_pair_pose_residual_candidate",
            "pair_index": pair_index,
            "frame_indices": target_pair["frame_indices"],
            "video_name": target_pair["video_name"],
            "charged_bytes": charged_bytes_per_atom,
            "charged_bytes_source": charged_bytes_source,
            "target_pose_dist": target_pair["pose_dist"],
            "target_seg_dist": target_pair["seg_dist"],
            "target_pose_score_contribution": target_pair["pose_score"],
            "target_seg_score_contribution": target_pair["seg_score"],
            "target_combined_score_contribution": target_pair["combined_score"],
            "expected_pose_score_saved": float(estimate["expected_pose_score_saved"]),
            "expected_seg_score_saved": float(estimate["expected_seg_score_saved"]),
            "expected_score_saved": expected_score_saved,
            "expected_score_saved_per_charged_byte": expected_score_saved / charged_bytes_per_atom,
            "rate_score_cost": rate_score_cost,
            "net_score_utility_formula_only": net_score_utility,
            "net_score_utility_per_charged_byte": net_score_utility / charged_bytes_per_atom,
            "evidence_source": estimate["source"],
            "reference_label": estimate["reference_label"],
            "reference_trace_path": estimate["reference_trace_path"],
            "reference_trace_sha256": estimate["reference_trace_sha256"],
            "measured_delta_available": estimate["measured_delta_available"],
            "raw_component_delta_score": estimate["raw_component_delta_score"],
            "prior": prior_detail,
            "risk_reasons": risk_reasons,
            "requires_exact_cuda_stack_eval": True,
            "score_claim": False,
            "promotion_eligible": False,
        }
        if "best_nonpositive_reference_delta" in estimate:
            atom["best_nonpositive_reference_delta"] = estimate["best_nonpositive_reference_delta"]
        atoms.append(atom)

    atoms.sort(
        key=lambda item: (
            item["expected_score_saved_per_charged_byte"],
            item["net_score_utility_per_charged_byte"],
            item["expected_score_saved"],
            -item["pair_index"],
        ),
        reverse=True,
    )
    for rank, atom in enumerate(atoms, start=1):
        atom["rank"] = rank
    top_atoms = atoms[:max_atoms]

    policies: list[dict[str, Any]] = []
    for count in sorted(set(policy_counts_tuple)):
        selected = [atom for atom in top_atoms[: min(count, len(top_atoms))]]
        if not selected:
            continue
        expected_saved_sum = sum(float(atom["expected_score_saved"]) for atom in selected)
        charged_bytes_sum = sum(float(atom["charged_bytes"]) for atom in selected)
        policies.append(
            {
                "policy_name": _policy_name(frontier_label, len(selected)),
                "policy_kind": "pose_residual_pair_indices",
                "frontier_label": frontier_label,
                "selected_count": len(selected),
                "selected_pair_indices": [int(atom["pair_index"]) for atom in selected],
                "selected_frame_indices": [atom["frame_indices"] for atom in selected],
                "selected_atom_ids": [atom["atom_id"] for atom in selected],
                "charged_bytes_estimate": charged_bytes_sum,
                "expected_score_saved_sum": expected_saved_sum,
                "rate_score_cost_sum": charged_bytes_sum * RATE_SCORE_PER_BYTE,
                "net_score_utility_formula_only": expected_saved_sum
                - charged_bytes_sum * RATE_SCORE_PER_BYTE,
                "measured_delta_atom_count": sum(
                    1 for atom in selected if atom["measured_delta_available"]
                ),
                "prior_atom_count": sum(1 for atom in selected if not atom["measured_delta_available"]),
                "atom_codec_contract": (
                    "PVR1/QP residual candidate policy; selected payload bits must be "
                    "charged inside archive.zip by a later archive builder."
                ),
                "requires_exact_cuda_stack_eval": True,
                "score_claim": False,
                "promotion_eligible": False,
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "frontier_label": frontier_label,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "diagnostic_planning_non_promotable",
        "non_promotable_warning": NON_PROMOTABLE_WARNING,
        "required_promotion_eval": CUDA_AUTH_EVAL_REQUIRED,
        "determinism": {
            "wall_clock_fields": False,
            "sort_keys": [
                "expected_score_saved_per_charged_byte",
                "net_score_utility_per_charged_byte",
                "expected_score_saved",
                "pair_index",
            ],
        },
        "contest_formula": {
            "score": "100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489",
            "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        },
        "inputs": {
            "component_trace": target_trace["meta"],
            "contest_auth_eval": contest_eval["meta"],
            "reference_traces": [
                {**trace["meta"], "label": trace["label"]} for trace in references
            ],
            "active_subspace_metadata": active_subspace["sources"],
            "atom_ledgers": ledger["sources"],
        },
        "frontier": {
            "archive_sha256": contest_eval["archive_sha256"],
            "archive_size_bytes": contest_eval["archive_size_bytes"],
            "score_recomputed_from_components": contest_eval["score"],
            "avg_posenet_dist": contest_eval["avg_pose"],
            "avg_segnet_dist": contest_eval["avg_seg"],
            "n_samples": EXPECTED_SAMPLES,
            "gpu_model": contest_eval["gpu_model"],
            "gpu_t4_match": contest_eval["gpu_t4_match"],
        },
        "planning_parameters": {
            "charged_bytes_per_atom": charged_bytes_per_atom,
            "charged_bytes_source": charged_bytes_source,
            "hard_pair_top_k": hard_pair_top_k,
            "max_atoms": max_atoms,
            "policy_counts": list(policy_counts_tuple),
            "prior_savings_fraction": prior_savings_fraction,
        },
        "hard_pair_prior": {
            "top_pair_indices": hard_pairs,
            "source": "target component trace scorer contributions",
        },
        "active_subspace_prior": {
            "pair_counts": {
                str(pair_index): active_subspace["pair_counts"][pair_index]
                for pair_index in sorted(active_subspace["pair_counts"])
            },
            "max_count": active_subspace["max_count"],
        },
        "ledger_pose_prior_summary": {
            "pair_count": len(ledger_priors),
            "inferred_pose_pair_bytes": inferred_charged_bytes,
        },
        "atom_count": len(top_atoms),
        "candidate_pair_count": len(atoms),
        "top_atoms": top_atoms,
        "recommended_policies": policies,
    }
    if output_json is not None:
        _write_json(output_json, payload)
    return payload


def _parse_policy_counts(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("policy counts must be comma-separated integers") from exc


def _existing_defaults(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-trace",
        type=Path,
        default=DEFAULT_C059_DIR / "component_trace.json",
        help="target frontier component_trace.json",
    )
    parser.add_argument(
        "--contest-auth-eval",
        type=Path,
        default=DEFAULT_C059_DIR / "contest_auth_eval.json",
        help="target frontier contest_auth_eval.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--frontier-label", default="C-059")
    parser.add_argument("--expected-archive-sha256", default=None)
    parser.add_argument(
        "--reference-trace",
        type=Path,
        action="append",
        default=None,
        help="optional reference/public component trace; repeatable",
    )
    parser.add_argument(
        "--no-default-reference-traces",
        action="store_true",
        help="do not auto-load known PR63/PR64/PR67 public traces",
    )
    parser.add_argument(
        "--active-subspace-metadata",
        type=Path,
        action="append",
        default=None,
        help="optional line-search metadata with basis_pair_indices; repeatable",
    )
    parser.add_argument(
        "--atom-ledger",
        type=Path,
        action="append",
        default=None,
        help="optional Popper/frontier atom ledger; repeatable",
    )
    parser.add_argument("--charged-bytes-per-atom", type=float, default=None)
    parser.add_argument("--hard-pair-top-k", type=int, default=128)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--policy-counts", type=_parse_policy_counts, default=(16, 32, 64, 128))
    parser.add_argument("--prior-savings-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)

    reference_traces = list(args.reference_trace or [])
    if not args.no_default_reference_traces:
        reference_traces.extend(_existing_defaults(DEFAULT_REFERENCE_TRACES))
    active_metadata = list(args.active_subspace_metadata or [])
    active_metadata.extend(_existing_defaults(DEFAULT_ACTIVE_METADATA))
    atom_ledgers = list(args.atom_ledger or [])
    atom_ledgers.extend(_existing_defaults(DEFAULT_ATOM_LEDGERS))

    payload = build_pose_atom_plan(
        component_trace_path=args.component_trace,
        contest_auth_eval_path=args.contest_auth_eval,
        output_json=args.output_json,
        frontier_label=args.frontier_label,
        expected_archive_sha256=args.expected_archive_sha256,
        reference_trace_paths=reference_traces,
        active_metadata_paths=active_metadata,
        atom_ledger_paths=atom_ledgers,
        charged_bytes_per_atom=args.charged_bytes_per_atom,
        hard_pair_top_k=args.hard_pair_top_k,
        max_atoms=args.max_atoms,
        policy_counts=args.policy_counts,
        prior_savings_fraction=args.prior_savings_fraction,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "frontier_label": payload["frontier_label"],
                "atom_count": payload["atom_count"],
                "policy_count": len(payload["recommended_policies"]),
                "top_pair_index": payload["top_atoms"][0]["pair_index"]
                if payload["top_atoms"]
                else None,
                "score_claim": payload["score_claim"],
                "promotion_eligible": payload["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
