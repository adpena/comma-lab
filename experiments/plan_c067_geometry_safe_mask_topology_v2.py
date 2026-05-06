#!/usr/bin/env python3
"""Fail-closed C067 geometry-safe mask/topology planner.

This tool is a non-training, no-dispatch gate for C067 mask/topology variants
after the PMG, CMG3A multimask, micro-mask, and AMR1 postdecode exact
negatives.  It consumes exact CUDA component traces as profile feedback and
refuses candidates that repeat an already-measured archive SHA, reuse a known
collapsing global mask-topology base, or touch pairs that were catastrophic in
same-family exact-negative traces.

It does not launch GPU jobs and does not make score claims.  A candidate marked
dispatchable here still needs a fresh dispatch claim and exact CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_c067_geometry_safe_mask_topology_v2.py"
SCHEMA = "c067_geometry_safe_mask_topology_v2"

ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
C067_FRONTIER_SCORE = 0.31561703078448233
C067_FRONTIER_ARCHIVE_BYTES = 276_214
C067_FRONTIER_ARCHIVE_SHA256 = (
    "226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a"
)
C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE = 252_760
CUDA_AUTH_EVAL_PATH = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/c067_geometry_safe_mask_topology_v2_20260502/"
    "c067_geometry_safe_mask_topology_v2_plan.json"
)
DEFAULT_TRIAGE_JSON = (
    REPO_ROOT
    / "experiments/results/c067_bigmove_nontrain_candidate_triage_20260502/"
    "c067_bigmove_nontrain_candidate_triage.json"
)
DEFAULT_FRONTIER_TRACE_JSON = (
    REPO_ROOT
    / "experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/"
    "component_trace.json"
)
DEFAULT_POSESAFE_POLICY_JSON = (
    REPO_ROOT
    / "experiments/results/c067_hotspot_mask_geometry_compiler_20260502/"
    "next_pose_safe_plan_after_extra065_072_negatives.json"
)
DEFAULT_ACTIVE_CLAIMS_MD = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"

POSE_CATASTROPHE_ABS = 0.05
COMBINED_PAIR_EXCESS_CATASTROPHE = 0.0010
POSE_RATIO_CATASTROPHE = 100.0
MAX_TOPOLOGY_PIXEL_DISAGREEMENT_FOR_PREFLIGHT = 0.0010


@dataclass(frozen=True)
class ExactNegativeSpec:
    negative_id: str
    family_group: str
    contest_auth_eval_json: Path
    component_trace_json: Path


DEFAULT_NEGATIVE_SPECS: tuple[ExactNegativeSpec, ...] = (
    ExactNegativeSpec(
        "c067_cmg3_nonzero_top1_t4",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3_nonzero_top2_t4",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top2_t4_20260502T1100Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3_nonzero_top2_t4_20260502T1100Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3a_body200_l40s",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3a_body200_l40s_20260502T114231Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_cmg3a_body200_l40s_20260502T114231Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "pmg_hotspot_c067_t4",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "pmg_hotspot_atomtop4068_l40s",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_hotspot_geometry_top0128_l40s",
        "mask_topology_global_replacement",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_hotspot_geometry_top0128_l40sdiag_20260502T1733Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_hotspot_geometry_top0128_l40sdiag_20260502T1733Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_multimask_extra065_l40s",
        "multimask_reconciler",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_multimask_reconciler_extra065k_fix1_l40sdiag_20260502T1903Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_multimask_reconciler_extra065k_fix1_l40sdiag_20260502T1903Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_multimask_extra072_l40s",
        "multimask_reconciler",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_multimask_reconciler_extra072k_fix1_l40sdiag_20260502T1910Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_multimask_reconciler_extra072k_fix1_l40sdiag_20260502T1910Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_micro_mask_save12k_l40s",
        "micro_mask_reencode",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_micro_mask_save12k_l40sdiag_20260502T2034Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_micro_mask_save12k_l40sdiag_20260502T2034Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_postdecode_repair_top10_l40s",
        "postdecode_mask_repair",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_top10_l40sdiag_20260502T2054Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_postdecode_repair_budget8000_l40s",
        "postdecode_mask_repair",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_budget8000_l40sdiag_20260502T2101Z/"
        "component_trace.json",
    ),
    ExactNegativeSpec(
        "c067_postdecode_repair_pairwaterfill4k_l40s",
        "postdecode_mask_repair",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z/"
        "contest_auth_eval.json",
        REPO_ROOT
        / "experiments/results/lightning_batch/exact_eval_c067_postdecode_repair_save12k_pairwaterfill4k_l40sdiag_20260502T2114Z/"
        "component_trace.json",
    ),
)

TERMINAL_STATUS_PREFIXES = (
    "completed",
    "failed",
    "cancelled",
    "stopped",
    "stale",
    "closed",
    "refused",
    "preempted",
)


class PlannerError(ValueError):
    """Raised for malformed planner inputs."""


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path | str | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(repo_root.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def _finite_float(value: Any, *, field: str, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlannerError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise PlannerError(f"{field} must be finite")
    return out


def _score_from_auth_eval(payload: dict[str, Any]) -> float:
    for key in ("score_recomputed_from_components", "final_score", "score"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    raise PlannerError("contest auth eval lacks score field")


def _archive_sha_from_auth_or_trace(
    auth: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> str | None:
    provenance = auth.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("archive_sha256"), str):
        return provenance["archive_sha256"]
    if trace is not None:
        trace_inputs = trace.get("trace_inputs")
        if isinstance(trace_inputs, dict) and isinstance(trace_inputs.get("archive_sha256"), str):
            return trace_inputs["archive_sha256"]
    return None


def _archive_bytes_from_auth_or_trace(
    auth: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> int | None:
    for source in (auth.get("provenance"), auth, trace):
        if isinstance(source, dict):
            for key in ("archive_size_bytes", "archive_bytes"):
                value = source.get(key)
                if isinstance(value, int):
                    return int(value)
    return None


def _combined_sample_score(sample: dict[str, Any]) -> float:
    value = sample.get("score_combined_contribution_first_order")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    seg = _finite_float(sample.get("score_seg_contribution_exact"), field="sample.seg", default=0.0)
    pose = _finite_float(
        sample.get("score_pose_contribution_first_order"),
        field="sample.pose",
        default=0.0,
    )
    return seg + pose


def _trace_samples_by_pair(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    samples = trace.get("samples")
    if not isinstance(samples, list):
        raise PlannerError("component trace lacks samples list")
    out: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise PlannerError("component trace samples must be objects")
        pair_index = sample.get("pair_index")
        if not isinstance(pair_index, int):
            raise PlannerError("component trace sample lacks integer pair_index")
        out[int(pair_index)] = sample
    return out


def _is_catastrophic_pair(sample: dict[str, Any], baseline: dict[str, Any] | None) -> bool:
    pose = _finite_float(sample.get("posenet_dist"), field="sample.posenet_dist")
    if pose >= POSE_CATASTROPHE_ABS:
        return True
    combined = _combined_sample_score(sample)
    baseline_combined = _combined_sample_score(baseline) if baseline is not None else 0.0
    if combined - baseline_combined >= COMBINED_PAIR_EXCESS_CATASTROPHE:
        return True
    if baseline is not None:
        base_pose = max(
            _finite_float(baseline.get("posenet_dist"), field="baseline.posenet_dist"),
            1.0e-9,
        )
        if pose / base_pose >= POSE_RATIO_CATASTROPHE and pose >= 0.01:
            return True
    return False


def summarize_exact_negative(
    spec: ExactNegativeSpec,
    *,
    frontier_pairs: dict[int, dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any] | None:
    if not spec.contest_auth_eval_json.exists() or not spec.component_trace_json.exists():
        return None
    auth = _read_json(spec.contest_auth_eval_json)
    trace = _read_json(spec.component_trace_json)
    samples = _trace_samples_by_pair(trace)
    catastrophic: list[dict[str, Any]] = []
    for pair_index, sample in samples.items():
        baseline = frontier_pairs.get(pair_index)
        if not _is_catastrophic_pair(sample, baseline):
            continue
        catastrophic.append(
            {
                "pair_index": pair_index,
                "frame_indices": sample.get("frame_indices"),
                "posenet_dist": round(
                    _finite_float(sample.get("posenet_dist"), field="posenet_dist"),
                    10,
                ),
                "segnet_dist": round(
                    _finite_float(sample.get("segnet_dist"), field="segnet_dist"),
                    10,
                ),
                "score_excess_vs_c067_pair": round(
                    _combined_sample_score(sample)
                    - (_combined_sample_score(baseline) if baseline is not None else 0.0),
                    12,
                ),
            }
        )
    catastrophic.sort(
        key=lambda row: (row["score_excess_vs_c067_pair"], row["posenet_dist"]),
        reverse=True,
    )
    score = _score_from_auth_eval(auth)
    return {
        "negative_id": spec.negative_id,
        "family_group": spec.family_group,
        "contest_auth_eval_json": _display_path(spec.contest_auth_eval_json, repo_root),
        "component_trace_json": _display_path(spec.component_trace_json, repo_root),
        "contest_auth_eval_sha256": _sha256_file(spec.contest_auth_eval_json),
        "component_trace_sha256": _sha256_file(spec.component_trace_json),
        "archive_sha256": _archive_sha_from_auth_or_trace(auth, trace),
        "archive_bytes": _archive_bytes_from_auth_or_trace(auth, trace),
        "score_recomputed_from_components": round(score, 12),
        "score_delta_vs_c067": round(score - C067_FRONTIER_SCORE, 12),
        "avg_posenet_dist": round(_finite_float(auth.get("avg_posenet_dist"), field="avg_posenet_dist"), 10),
        "avg_segnet_dist": round(_finite_float(auth.get("avg_segnet_dist"), field="avg_segnet_dist"), 10),
        "gpu_model": (auth.get("provenance") or {}).get("gpu_model")
        if isinstance(auth.get("provenance"), dict)
        else None,
        "gpu_t4_match": (auth.get("provenance") or {}).get("gpu_t4_match")
        if isinstance(auth.get("provenance"), dict)
        else None,
        "catastrophic_pair_count": len(catastrophic),
        "catastrophic_pair_indices": [int(row["pair_index"]) for row in catastrophic],
        "top_catastrophic_pairs": catastrophic[:24],
    }


def _family_group(candidate_family: str, candidate_id: str) -> str:
    text = f"{candidate_family} {candidate_id}".lower()
    if "multimask" in text or "reconciler" in text:
        return "multimask_reconciler"
    if "micro" in text:
        return "micro_mask_reencode"
    if "postdecode" in text or "amr1" in text:
        return "postdecode_mask_repair"
    if "pmg" in text or "hotspot" in text or "cmg3" in text or "poseguard" in text:
        return "mask_topology_global_replacement"
    return "unknown_or_new_geometry_delta"


def _parse_pair_indices_from_builder_command(command: Any) -> list[int]:
    if not isinstance(command, list):
        return []
    out: list[int] = []
    for index, token in enumerate(command):
        if token == "--pair-indices" and index + 1 < len(command):
            for part in str(command[index + 1]).split(","):
                stripped = part.strip()
                if stripped:
                    out.append(int(stripped, 10))
    return sorted(set(out))


def _candidate_from_triage(record: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(record.get("candidate_id"))
    archive = record.get("archive")
    archive_record = archive if isinstance(archive, dict) else None
    builder_command = record.get("builder_command_if_materialization_needed")
    family = str(record.get("family") or "unknown")
    selected_pairs = set(_parse_pair_indices_from_builder_command(builder_command))
    support = record.get("support")
    if isinstance(support, dict):
        for key in ("top_pair_indices_by_selected_atom_count", "pair_indices"):
            values = support.get(key)
            if isinstance(values, list):
                selected_pairs.update(int(value) for value in values if isinstance(value, int))
    return {
        "source": "bigmove_triage",
        "candidate_id": candidate_id,
        "policy_id": record.get("policy_id"),
        "family": family,
        "family_group": _family_group(family, candidate_id),
        "lane": record.get("lane"),
        "archive": archive_record,
        "archive_path": archive_record.get("path") if archive_record else None,
        "archive_sha256": archive_record.get("sha256") if archive_record else None,
        "archive_bytes": archive_record.get("bytes") if archive_record else None,
        "selected_pair_indices": sorted(selected_pairs),
        "builder_command_if_materialization_needed": builder_command
        if isinstance(builder_command, list)
        else [],
        "input_evidence_grade": record.get("evidence_grade"),
        "input_exact_eval": record.get("exact_eval"),
        "score_claim": False,
    }


def _candidate_from_poseguard_policy(policy: dict[str, Any], policy_json: Path, repo_root: Path) -> dict[str, Any]:
    policy_id = str(policy.get("policy_id"))
    selected_pairs: set[int] = set()
    support = policy.get("support")
    if isinstance(support, dict):
        values = support.get("top_pair_indices_by_selected_atom_count")
        if isinstance(values, list):
            selected_pairs.update(int(value) for value in values if isinstance(value, int))
    atoms = policy.get("selected_row_run_atoms")
    if isinstance(atoms, list):
        for atom in atoms:
            if not isinstance(atom, dict):
                continue
            frame_index = atom.get("frame_index")
            if isinstance(frame_index, int):
                selected_pairs.add(int(frame_index) // 2)
    builder = policy.get("builder")
    return {
        "source": "poseguard_policy",
        "candidate_id": policy_id,
        "policy_id": policy_id,
        "family": "poseguard_topology_after_exact_negatives",
        "family_group": "mask_topology_global_replacement",
        "lane": "scorer_weighted_mask_topology_repair_atoms",
        "archive": None,
        "archive_path": None,
        "archive_sha256": None,
        "archive_bytes": None,
        "selected_pair_indices": sorted(selected_pairs),
        "selected_atom_count": policy.get("selected_atom_count"),
        "selected_residual_pixels": (policy.get("estimated_proxy") or {}).get(
            "selected_residual_pixels"
        )
        if isinstance(policy.get("estimated_proxy"), dict)
        else None,
        "builder_command_if_materialization_needed": [str(builder)] if isinstance(builder, str) else [],
        "field_policy_json": _display_path(policy_json, repo_root),
        "input_evidence_grade": policy.get("evidence_grade"),
        "score_claim": False,
    }


def load_candidates(
    *,
    triage_json: Path,
    poseguard_policy_json: Path | None,
    repo_root: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if triage_json.exists():
        triage = _read_json(triage_json)
        ranked = triage.get("ranked_candidates")
        if isinstance(ranked, list):
            for record in ranked:
                if isinstance(record, dict):
                    candidates.append(_candidate_from_triage(record))
    if poseguard_policy_json is not None and poseguard_policy_json.exists():
        policies_payload = _read_json(poseguard_policy_json)
        policies = policies_payload.get("candidate_policies")
        if isinstance(policies, list):
            for policy in policies:
                if isinstance(policy, dict):
                    candidates.append(_candidate_from_poseguard_policy(policy, poseguard_policy_json, repo_root))
    return candidates


def parse_active_claims(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    terminal_keys: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped or "timestamp_utc" in stripped:
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 8:
            continue
        lane_id = parts[2]
        instance_job_id = parts[4]
        status = parts[6]
        if status.lower().startswith(TERMINAL_STATUS_PREFIXES):
            terminal_keys.add((lane_id, instance_job_id))
            continue
        if (lane_id, instance_job_id) in terminal_keys:
            continue
        rows.append(
            {
                "timestamp_utc": parts[0],
                "agent": parts[1],
                "lane_id": lane_id,
                "platform": parts[3],
                "instance_job_id": instance_job_id,
                "predicted_eta_utc": parts[5],
                "status": status,
                "notes": parts[7],
            }
        )
    return rows


def _candidate_lane_id(candidate: dict[str, Any]) -> str:
    raw = str(candidate.get("candidate_id") or candidate.get("policy_id") or "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")[:120] or "unknown"


def _active_conflicts(candidate: dict[str, Any], active_claims: list[dict[str, str]]) -> list[dict[str, str]]:
    candidate_lane = _candidate_lane_id(candidate)
    family_group = str(candidate.get("family_group"))
    matches: list[dict[str, str]] = []
    for claim in active_claims:
        lane_id = claim.get("lane_id", "")
        if lane_id == candidate_lane or candidate_lane in lane_id or family_group in lane_id:
            matches.append(claim)
    return matches


def _gate_candidate(
    candidate: dict[str, Any],
    *,
    negative_summaries: list[dict[str, Any]],
    negative_by_sha: dict[str, dict[str, Any]],
    catastrophic_pairs_by_group: dict[str, set[int]],
    active_claims: list[dict[str, str]],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    archive_sha = candidate.get("archive_sha256")
    archive_bytes = candidate.get("archive_bytes")
    family_group = str(candidate.get("family_group"))
    selected_pairs = set(int(value) for value in candidate.get("selected_pair_indices") or [])

    if archive_sha in negative_by_sha:
        neg = negative_by_sha[str(archive_sha)]
        blockers.append(
            f"identical archive SHA already has exact CUDA negative evidence: {neg['negative_id']}"
        )

    exact_eval = candidate.get("input_exact_eval")
    if isinstance(exact_eval, dict) and exact_eval.get("status") == "exact_negative":
        blockers.append("triage record already carries exact CUDA negative evidence")

    same_group_collapse = [
        neg
        for neg in negative_summaries
        if neg["family_group"] == family_group and neg["score_delta_vs_c067"] > 0.01
    ]
    topology_family = family_group in {
        "mask_topology_global_replacement",
        "multimask_reconciler",
        "micro_mask_reencode",
        "postdecode_mask_repair",
    }
    if same_group_collapse and topology_family:
        warnings.append(
            f"{family_group} has {len(same_group_collapse)} recent exact-negative trace(s)"
        )

    same_group_cat_pairs = set(catastrophic_pairs_by_group.get(family_group, set()))
    if family_group == "mask_topology_global_replacement":
        same_group_cat_pairs.update(catastrophic_pairs_by_group.get("multimask_reconciler", set()))
    overlap = sorted(selected_pairs & same_group_cat_pairs)
    if overlap:
        blockers.append(
            "selected pairs overlap same-family catastrophic exact-negative pairs: "
            + ",".join(str(value) for value in overlap[:24])
        )

    if topology_family and same_group_collapse and not selected_pairs:
        blockers.append(
            "same-family topology exact negatives exist but candidate exposes no pair/atom selection for pose-safety review"
        )

    if family_group == "mask_topology_global_replacement":
        blockers.append(
            "global CMG3/PMG topology replacement base is exact-negative; require a decoded-baseline delta/overlay or passing pose-regenerated geometry proof before exact eval"
        )

    if archive_bytes is None:
        blockers.append("candidate is not byte-closed yet; build and provenance-screen before exact CUDA eval")
    elif not isinstance(archive_bytes, int):
        blockers.append("candidate archive bytes are malformed")
    else:
        if archive_bytes > C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE:
            blockers.append(
                f"archive bytes {archive_bytes} exceed unchanged-distortion sub-0.300 gate {C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE}"
            )

    pixel_disagreement = candidate.get("pixel_disagreement_vs_source")
    if isinstance(pixel_disagreement, (int, float)) and pixel_disagreement > MAX_TOPOLOGY_PIXEL_DISAGREEMENT_FOR_PREFLIGHT:
        blockers.append(
            f"pixel disagreement {pixel_disagreement:.6f} exceeds geometry preflight trust-region {MAX_TOPOLOGY_PIXEL_DISAGREEMENT_FOR_PREFLIGHT:.6f}"
        )

    active_conflicts = _active_conflicts(candidate, active_claims)
    if active_conflicts:
        blockers.append("active dispatch claim conflict exists for this lane/family")

    byte_gate_passed = isinstance(archive_bytes, int) and archive_bytes <= C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE
    dispatchable = not blockers and byte_gate_passed
    return {
        **candidate,
        "lane_id_suggestion": _candidate_lane_id(candidate),
        "geometry_pose_safety_gate": {
            "status": "pass_dispatchable_after_claim" if dispatchable else "fail_closed",
            "dispatchable": dispatchable,
            "byte_gate_passed": byte_gate_passed,
            "blockers": blockers,
            "warnings": warnings,
            "selected_pair_count": len(selected_pairs),
            "selected_pair_indices": sorted(selected_pairs),
            "same_family_catastrophic_pair_overlap": overlap,
            "active_claim_conflicts": active_conflicts,
            "score_claim": False,
        },
        "required_before_remote_eval": [
            "fresh tools/claim_lane_dispatch.py claim row",
            "byte-closed archive/provenance manifest",
            "exact CUDA diagnostic only; T4/equivalent replay only if components survive",
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_plan(
    *,
    repo_root: Path,
    frontier_trace_json: Path,
    triage_json: Path,
    poseguard_policy_json: Path | None,
    active_claims_md: Path,
    negative_specs: Iterable[ExactNegativeSpec] = DEFAULT_NEGATIVE_SPECS,
) -> dict[str, Any]:
    if not frontier_trace_json.exists():
        raise PlannerError(f"frontier trace JSON missing: {frontier_trace_json}")
    frontier_trace = _read_json(frontier_trace_json)
    frontier_pairs = _trace_samples_by_pair(frontier_trace)

    negative_summaries: list[dict[str, Any]] = []
    missing_negative_inputs: list[dict[str, str]] = []
    for spec in negative_specs:
        summary = summarize_exact_negative(spec, frontier_pairs=frontier_pairs, repo_root=repo_root)
        if summary is None:
            missing_negative_inputs.append(
                {
                    "negative_id": spec.negative_id,
                    "contest_auth_eval_json": _display_path(spec.contest_auth_eval_json, repo_root),
                    "component_trace_json": _display_path(spec.component_trace_json, repo_root),
                }
            )
            continue
        negative_summaries.append(summary)

    negative_by_sha = {
        str(summary["archive_sha256"]): summary
        for summary in negative_summaries
        if isinstance(summary.get("archive_sha256"), str)
    }
    catastrophic_pairs_by_group: dict[str, set[int]] = {}
    for summary in negative_summaries:
        catastrophic_pairs_by_group.setdefault(summary["family_group"], set()).update(
            int(pair) for pair in summary["catastrophic_pair_indices"]
        )

    active_claims = parse_active_claims(active_claims_md)
    raw_candidates = load_candidates(
        triage_json=triage_json,
        poseguard_policy_json=poseguard_policy_json,
        repo_root=repo_root,
    )
    gated = [
        _gate_candidate(
            candidate,
            negative_summaries=negative_summaries,
            negative_by_sha=negative_by_sha,
            catastrophic_pairs_by_group=catastrophic_pairs_by_group,
            active_claims=active_claims,
        )
        for candidate in raw_candidates
    ]
    gated.sort(
        key=lambda row: (
            not row["geometry_pose_safety_gate"]["dispatchable"],
            -int(row.get("archive_bytes") or 0),
            str(row.get("candidate_id")),
        )
    )
    dispatchable = [row for row in gated if row["geometry_pose_safety_gate"]["dispatchable"]]

    return {
        "schema": SCHEMA,
        "producer": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_jobs_dispatched": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_PATH,
        "frontier": {
            "archive_bytes": C067_FRONTIER_ARCHIVE_BYTES,
            "archive_sha256": C067_FRONTIER_ARCHIVE_SHA256,
            "score": C067_FRONTIER_SCORE,
            "unchanged_distortion_sub0300_byte_gate": C067_UNCHANGED_DISTORTION_SUB0300_BYTE_GATE,
            "component_trace_json": _display_path(frontier_trace_json, repo_root),
            "component_trace_sha256": _sha256_file(frontier_trace_json),
        },
        "geometry_pose_safety_policy": {
            "pose_catastrophe_abs": POSE_CATASTROPHE_ABS,
            "combined_pair_excess_catastrophe": COMBINED_PAIR_EXCESS_CATASTROPHE,
            "pose_ratio_catastrophe": POSE_RATIO_CATASTROPHE,
            "max_topology_pixel_disagreement_for_preflight": MAX_TOPOLOGY_PIXEL_DISAGREEMENT_FOR_PREFLIGHT,
            "rule": (
                "Fail closed on identical exact-negative archives, global topology "
                "bases with exact-negative evidence, selected-pair overlap with "
                "catastrophic exact-negative traces, missing byte closure, active "
                "dispatch conflicts, or bytes above the unchanged-distortion sub-0.300 gate."
            ),
        },
        "exact_negative_inputs": negative_summaries,
        "missing_negative_inputs": missing_negative_inputs,
        "active_nonterminal_claims_observed": active_claims,
        "candidate_count": len(gated),
        "dispatchable_candidate_count": len(dispatchable),
        "dispatchable_candidates": dispatchable,
        "gated_candidates": gated,
        "grand_council_priority_design": {
            "decision": "no_same_family_remote_dispatch"
            if not dispatchable
            else "dispatch_only_after_lane_claim",
            "rationale": (
                "Recent exact CUDA negatives show the current non-training mask/topology "
                "families collapse PoseNet through global geometry changes. The next "
                "allowed design must be a decoded-baseline delta/overlay, pose-regenerated "
                "geometry proof, or learned topology export with L2 clearance; byte-only "
                "PMG/multimask/AMR1 variants are not enough."
            ),
            "next_nontraining_candidate_shape": (
                "A geometry-safe overlay over the C067 decoded mask stream: selected atoms "
                "must avoid catastrophic exact-negative pairs, preserve global topology, "
                "record pixel-disagreement <= trust region, emit byte-closed archive "
                "custody, then run exact CUDA diagnostic."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--frontier-trace-json", type=Path, default=DEFAULT_FRONTIER_TRACE_JSON)
    parser.add_argument("--triage-json", type=Path, default=DEFAULT_TRIAGE_JSON)
    parser.add_argument("--poseguard-policy-json", type=Path, default=DEFAULT_POSESAFE_POLICY_JSON)
    parser.add_argument("--active-claims-md", type=Path, default=DEFAULT_ACTIVE_CLAIMS_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json
    if output_json.exists() and not args.force:
        raise SystemExit(f"{output_json} exists; pass --force to overwrite")
    plan = build_plan(
        repo_root=args.repo_root,
        frontier_trace_json=args.frontier_trace_json,
        triage_json=args.triage_json,
        poseguard_policy_json=args.poseguard_policy_json,
        active_claims_md=args.active_claims_md,
    )
    _write_json(output_json, plan)
    print(
        json.dumps(
            {
                "output_json": _display_path(output_json, args.repo_root),
                "dispatchable_candidate_count": plan["dispatchable_candidate_count"],
                "candidate_count": plan["candidate_count"],
                "decision": plan["grand_council_priority_design"]["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
