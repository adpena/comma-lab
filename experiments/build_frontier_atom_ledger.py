#!/usr/bin/env python3
"""Build a deterministic atom ledger for current frontier packer work.

The ledger converts exact CUDA candidates, byte-only packer provenance, and
top-submission archive anatomy into one water-fill-friendly JSON artifact.
It does not create a score claim; exact score authority remains the referenced
``contest_auth_eval*.json`` files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SLOPE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

DEFAULT_EXACT_CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "C-044",
        "experiments/results/lightning_batch/"
        "exact_eval_owv3_0120_stack_t4_20260501T150652Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "C-049",
        "experiments/results/lightning_batch/"
        "exact_eval_renderer_payload_posecd_c044_t4fix1_20260501T1754Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "C-050",
        "experiments/results/lightning_batch/"
        "exact_eval_renderer_payload_posecd_rp2_palias_allowp_c044_t4_20260501T1823Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "C-051",
        "experiments/results/lightning_batch/"
        "exact_eval_renderer_payload_qpose14_rp2_palias_allowp_c044_t4_20260501T1823Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "C-063",
        "experiments/results/lightning_batch/"
        "exact_eval_lossless_repack_c059_brotli_t4_20260502T0537Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "velres_top512_A_negative",
        "experiments/results/lightning_batch/"
        "exact_eval_renderer_payload_velres_top512_c044_g6e_20260501T1748Z/"
        "contest_auth_eval.adjudicated.json",
    ),
    (
        "velres_top1024_A_negative",
        "experiments/results/lightning_batch/"
        "exact_eval_renderer_payload_velres_top1024_c044_g6e_20260501T1748Z/"
        "contest_auth_eval.adjudicated.json",
    ),
)

DEFAULT_PACKER_PROVENANCE: tuple[str, ...] = (
    "experiments/results/renderer_packed_payload_c044_posecd_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_posecd_rp2_palias_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_qpose14_rp2_palias_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_qp1_rp2_palias_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_velonly_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_velres_top512_20260501/"
    "packed_renderer_payload_provenance.json",
    "experiments/results/renderer_packed_payload_c044_velres_top1024_20260501/"
    "packed_renderer_payload_provenance.json",
)

DEFAULT_TOP_ANATOMY = (
    "experiments/results/top_submission_reverse_engineering_20260502T0206Z/"
    "archive_anatomy.json"
)

DEFAULT_TRACE_COMPARISONS: tuple[str, ...] = (
    "experiments/results/top_submission_reverse_engineering_20260502T0206Z/"
    "basis_vs_pr67_component_delta.json",
    "experiments/results/top_submission_reverse_engineering_20260502T0206Z/"
    "basis_h100_vs_pr67_recut_top120.json",
    "experiments/results/top_submission_reverse_engineering_20260502T0206Z/"
    "c057_vs_pr67_recut_top120.json",
)

DEFAULT_OUTPUT = (
    "experiments/results/frontier_atom_ledger_20260501/"
    "frontier_atom_ledger.json"
)


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


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


def _score(seg: float, pose: float, archive_bytes: int) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + RATE_SLOPE_SCORE_PER_BYTE * archive_bytes


def _load_exact_candidate(label: str, path: Path, *, root: Path) -> dict[str, Any]:
    data = _json_load(path)
    prov = data.get("provenance") or {}
    archive_bytes = int(data["archive_size_bytes"])
    pose = float(data["avg_posenet_dist"])
    seg = float(data["avg_segnet_dist"])
    recomputed = _score(seg, pose, archive_bytes)
    score_value = float(data["score_recomputed_from_components"])
    adj_path = path.with_name("adjudication_provenance.json")
    adjudication = _json_load(adj_path) if adj_path.exists() else {}
    return {
        "label": label,
        "artifact": _rel(path, root=root),
        "artifact_sha256": _sha256_file(path),
        "adjudication": _rel(adj_path, root=root) if adj_path.exists() else None,
        "archive_bytes": archive_bytes,
        "archive_sha256": prov.get("archive_sha256"),
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "score_recomputed_from_components": score_value,
        "score_recomputed_by_ledger": recomputed,
        "score_recompute_abs_error": abs(score_value - recomputed),
        "score_pose_contribution": math.sqrt(10.0 * pose),
        "score_seg_contribution": 100.0 * seg,
        "score_rate_contribution": RATE_SLOPE_SCORE_PER_BYTE * archive_bytes,
        "device": prov.get("device"),
        "gpu_model": prov.get("gpu_model"),
        "gpu_t4_match": prov.get("gpu_t4_match"),
        "n_samples": data.get("n_samples"),
        "promotion_eligible": adjudication.get("promotion_eligible"),
        "evidence_grade": adjudication.get("evidence_grade"),
        "component_gate_triggered": adjudication.get("component_gate_triggered"),
    }


def _transition(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    relation: str,
) -> dict[str, Any]:
    delta_bytes = int(target["archive_bytes"]) - int(source["archive_bytes"])
    bytes_saved = -delta_bytes
    delta_score = (
        float(target["score_recomputed_from_components"])
        - float(source["score_recomputed_from_components"])
    )
    score_saved = -delta_score
    delta_seg = float(target["avg_segnet_dist"]) - float(source["avg_segnet_dist"])
    delta_pose = float(target["avg_posenet_dist"]) - float(source["avg_posenet_dist"])
    delta_rate_score = RATE_SLOPE_SCORE_PER_BYTE * delta_bytes
    delta_pose_score = (
        float(target["score_pose_contribution"])
        - float(source["score_pose_contribution"])
    )
    delta_seg_score = 100.0 * delta_seg
    nonrate_delta = delta_score - delta_rate_score
    if bytes_saved > 0:
        saved_per_byte = score_saved / bytes_saved
        excess = saved_per_byte - RATE_SLOPE_SCORE_PER_BYTE
    else:
        saved_per_byte = None
        excess = None
    return {
        "relation": relation,
        "from_label": source["label"],
        "to_label": target["label"],
        "delta_bytes": delta_bytes,
        "bytes_saved": bytes_saved,
        "delta_score": delta_score,
        "score_saved": score_saved,
        "delta_rate_score": delta_rate_score,
        "delta_nonrate_score": nonrate_delta,
        "delta_seg_dist": delta_seg,
        "delta_posenet_dist": delta_pose,
        "delta_seg_score": delta_seg_score,
        "delta_pose_score": delta_pose_score,
        "score_saved_per_byte": saved_per_byte,
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "excess_score_saved_per_byte_vs_rate_slope": excess,
        "accepted_by_exact_score": delta_score < 0.0,
    }


def _load_packer(path: Path, *, root: Path) -> dict[str, Any]:
    data = _json_load(path)
    members = data.get("header", {}).get("members", [])
    member_atoms = []
    for member in members:
        atom = {
            "name": member.get("name"),
            "codec": member.get("codec"),
            "encoded_bytes": member.get("bytes"),
            "sha256": member.get("sha256"),
        }
        for key in (
            "decoded_bytes",
            "decoded_sha256",
            "source_decoded_sha256",
            "lossy",
            "pose_error_stats",
            "pose_residual_topk",
        ):
            if key in member:
                atom[key] = member[key]
        member_atoms.append(atom)
    return {
        "artifact": _rel(path, root=root),
        "artifact_sha256": _sha256_file(path),
        "score_claim": data.get("score_claim"),
        "evidence_grade": data.get("evidence_grade"),
        "source_archive_bytes": data.get("source_archive_bytes"),
        "source_archive_sha256": data.get("source_archive_sha256"),
        "output_archive": data.get("output_archive"),
        "output_archive_bytes": data.get("output_archive_bytes"),
        "output_archive_sha256": data.get("output_archive_sha256"),
        "savings_bytes": data.get("savings_bytes"),
        "formula_only_rate_delta": data.get("formula_only_rate_delta"),
        "payload_format": data.get("payload_format", "rpk1_json"),
        "payload_member": data.get("payload_member"),
        "payload_raw_bytes": data.get("payload_raw_bytes"),
        "payload_compressed_bytes": data.get("payload_compressed_bytes"),
        "pose_codec": data.get("pose_codec"),
        "pose_residual_topk": data.get("pose_residual_topk"),
        "member_atoms": member_atoms,
    }


def _top_submission_atoms(top_anatomy_path: Path, *, root: Path) -> dict[str, Any]:
    data = _json_load(top_anatomy_path)
    items = []
    for item in data.get("items", []):
        archive = item.get("archive") or {}
        container = item.get("container") or {}
        segments = []
        for segment in container.get("decoded_segments", []):
            compressed = int(segment.get("compressed_bytes", 0))
            raw = int(segment.get("raw_bytes", 0))
            segments.append(
                {
                    "name": segment.get("name"),
                    "compressed_bytes": compressed,
                    "raw_bytes": raw,
                    "compression_ratio": compressed / raw if raw else None,
                    "compressed_sha256": segment.get("compressed_sha256"),
                    "raw_sha256": segment.get("raw_sha256"),
                }
            )
        items.append(
            {
                "label": item.get("label"),
                "pr_number": item.get("pr_number"),
                "archive_bytes": archive.get("bytes"),
                "archive_sha256": archive.get("sha256"),
                "container_member": container.get("member"),
                "container_bytes": container.get("bytes"),
                "container_sha256": container.get("sha256"),
                "segmentation": container.get("segmentation"),
                "zip_members": item.get("zip_members", []),
                "segments": segments,
                "first_32_24bit_lengths": container.get("first_32_24bit_lengths"),
                "qzs3_decode_validation": container.get("qzs3_decode_validation"),
                "source": item.get("source"),
            }
        )
    return {
        "artifact": _rel(top_anatomy_path, root=root),
        "artifact_sha256": _sha256_file(top_anatomy_path),
        "schema_version": data.get("schema_version"),
        "score_claim": data.get("score_claim"),
        "evidence_grade": data.get("evidence_grade"),
        "determinism": data.get("determinism"),
        "local_jointframegenerator_reference": data.get(
            "local_jointframegenerator_reference"
        ),
        "items": items,
    }


def _indexed_top_items(top_submissions: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if top_submissions is None:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in top_submissions.get("items", []):
        pr_number = item.get("pr_number")
        if isinstance(pr_number, int):
            out[pr_number] = item
    return out


def _segment_bytes(item: dict[str, Any] | None, name: str) -> int | None:
    if item is None:
        return None
    for segment in item.get("segments") or []:
        if segment.get("name") == name and isinstance(segment.get("compressed_bytes"), int):
            return int(segment["compressed_bytes"])
    return None


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _score_float(sample: dict[str, Any], key: str) -> float:
    value = sample.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _risk_profile(
    *,
    family: str,
    pose_score_excess: float,
    seg_score_excess: float,
    expected_score_saved: float,
) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    if expected_score_saved <= 0.0:
        return "high", 0.50, ["no_positive_reference_relative_benefit"]
    if family == "pose" and seg_score_excess < 0.0:
        reasons.append("pose_atom_conflicts_with_candidate_segnet_advantage")
    if family == "mask" and pose_score_excess < 0.0:
        reasons.append("mask_atom_conflicts_with_candidate_posenet_advantage")
    if family == "postprocess" and pose_score_excess * seg_score_excess < 0.0:
        reasons.append("component_sign_conflict_inside_pair")
    if family == "postprocess" and expected_score_saved < 4.0 * RATE_SLOPE_SCORE_PER_BYTE:
        reasons.append("tiny_pair_signal")
    if not reasons:
        return "low", 0.0, []
    if len(reasons) == 1:
        return "medium", 0.15, reasons
    return "high", 0.35, reasons


def _atom_from_trace_sample(
    *,
    sample: dict[str, Any],
    family: str,
    charged_bytes: int,
    charged_bytes_source: str,
    comparison_label: str,
    candidate_label: str,
    reference_label: str,
    artifact: str,
) -> dict[str, Any] | None:
    pair_index = sample.get("pair_index")
    if not isinstance(pair_index, int):
        return None
    pose_score_excess = _score_float(sample, "score_pose_excess_first_order_at_candidate")
    seg_score_excess = _score_float(sample, "score_seg_excess_exact")
    if family == "pose":
        expected_score_saved = max(0.0, pose_score_excess)
        atom_kind = "qpose_pair_residual"
    elif family == "mask":
        expected_score_saved = max(0.0, seg_score_excess)
        atom_kind = "mask_pair_residual"
    else:
        expected_score_saved = max(0.0, _score_float(sample, "score_combined_excess_first_order"))
        atom_kind = "component_safe_postprocess_pair"
    charged_bytes = max(1, int(charged_bytes))
    rate_cost = RATE_SLOPE_SCORE_PER_BYTE * charged_bytes
    risk_level, risk_factor, risk_reasons = _risk_profile(
        family=family,
        pose_score_excess=pose_score_excess,
        seg_score_excess=seg_score_excess,
        expected_score_saved=expected_score_saved,
    )
    risk_penalty = expected_score_saved * risk_factor
    utility = expected_score_saved - rate_cost - risk_penalty
    return {
        "atom_id": (
            f"{comparison_label}:{candidate_label}:vs:{reference_label}:"
            f"{family}:pair_{pair_index:04d}"
        ),
        "atom_kind": atom_kind,
        "family": family,
        "comparison_label": comparison_label,
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "source_artifact": artifact,
        "pair_index": pair_index,
        "frame_indices": sample.get("frame_indices"),
        "video_name": sample.get("video_name"),
        "charged_bytes": charged_bytes,
        "charged_bytes_source": charged_bytes_source,
        "expected_score_saved": expected_score_saved,
        "expected_pose_score_saved": max(0.0, pose_score_excess),
        "expected_seg_score_saved": max(0.0, seg_score_excess),
        "raw_pose_score_excess": pose_score_excess,
        "raw_seg_score_excess": seg_score_excess,
        "raw_combined_score_excess": _score_float(sample, "score_combined_excess_first_order"),
        "rate_score_cost": rate_cost,
        "interaction_risk": risk_level,
        "interaction_risk_reasons": risk_reasons,
        "risk_penalty_score": risk_penalty,
        "waterfill_utility_score": utility,
        "expected_score_saved_per_byte": expected_score_saved / charged_bytes,
        "risk_adjusted_utility_per_byte": utility / charged_bytes,
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "waterfill_positive_ev": utility > 0.0,
        "score_claim": False,
        "requires_exact_cuda_stack_eval": True,
    }


def _allocation_sort_key(atom: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(atom["risk_adjusted_utility_per_byte"]),
        -float(atom["expected_score_saved_per_byte"]),
        {"low": 0, "medium": 1, "high": 2}.get(str(atom["interaction_risk"]), 9),
        str(atom["atom_id"]),
    )


def _default_pipeline_flags(family: str) -> dict[str, list[str]]:
    flags = {
        "postprocess": {
            "synergizes_with": ["hard_pair_trace_selection", "multipass_selector"],
            "antagonizes_with": ["archive_byte_budget", "runtime_decoder_branching"],
        },
        "pose": {
            "synergizes_with": ["renderer_pose_conditioning", "qpose_grid_training"],
            "antagonizes_with": ["stale_mask_geometry", "untrained_pose_contract"],
        },
        "mask": {
            "synergizes_with": ["mask_grammar", "segnet_hard_pair_selection"],
            "antagonizes_with": ["pose_sensitive_boundary_shift", "stale_pose_stream"],
        },
        "renderer_quantization": {
            "synergizes_with": ["qzs3_grouped_quantization", "scorer_path_qat"],
            "antagonizes_with": ["texture_collapse", "geometry_collapse"],
        },
        "archive_overhead": {
            "synergizes_with": ["single_member_pack", "fixed_length_table"],
            "antagonizes_with": ["debug_payloads", "duplicate_zip_members"],
        },
        "runtime_simplification": {
            "synergizes_with": ["fixed_segment_decoder", "inflate_budget"],
            "antagonizes_with": ["multipass_runtime_branching", "large_python_sidecars"],
        },
        "selection_policy": {
            "synergizes_with": ["rl_bandit_search", "component_trace_feedback"],
            "antagonizes_with": ["uncharged_selection_sidecar", "proxy_score_overfit"],
        },
    }
    return flags.get(
        family,
        {
            "synergizes_with": ["exact_trace_feedback"],
            "antagonizes_with": ["unvalidated_additive_delta_assumption"],
        },
    )


def _exact_eval_stack_gate_recommendations() -> list[dict[str, Any]]:
    return [
        {
            "gate": "closed_archive_payload",
            "recommendation": (
                "All selected postprocess, pose, mask, renderer, selection, and runtime "
                "payload bits must be inside archive.zip or fixed contest code."
            ),
        },
        {
            "gate": "canonical_cuda_auth_eval",
            "recommendation": (
                "Every proposed stack needs its own archive.zip -> inflate.sh -> "
                "upstream/evaluate.py CUDA auth eval before any score, rank, or kill claim."
            ),
        },
        {
            "gate": "component_antagonism",
            "recommendation": (
                "Compare exact PoseNet and SegNet deltas against the planner's "
                "synergy/antagonism flags; split or reject stacks with component collapse."
            ),
        },
        {
            "gate": "runtime_budget",
            "recommendation": (
                "Runtime simplification atoms can promote only with deterministic inflate "
                "closure and budget proof on contest-equivalent hardware."
            ),
        },
        {
            "gate": "t4_equivalent_promotion",
            "recommendation": (
                "Fast-chip traces are diagnostic; only identical bytes confirmed on "
                "T4/equivalent custody can promote."
            ),
        },
    ]


def _pipeline_opportunity_atom(
    *,
    atom_id: str,
    family: str,
    stage: str,
    description: str,
    charged_bytes: int | None,
    charged_bytes_source: str,
    evidence_grade: str,
    expected_score_saved: float | None = None,
    calibration_required: bool = True,
) -> dict[str, Any]:
    flags = _default_pipeline_flags(family)
    rate_cost = (
        RATE_SLOPE_SCORE_PER_BYTE * charged_bytes
        if charged_bytes is not None
        else None
    )
    utility = (
        expected_score_saved - rate_cost
        if expected_score_saved is not None and rate_cost is not None
        else None
    )
    return {
        "atom_id": atom_id,
        "family": family,
        "pipeline_stage": stage,
        "description": description,
        "charged_bytes": charged_bytes,
        "charged_bytes_source": charged_bytes_source,
        "expected_score_saved": expected_score_saved,
        "rate_score_cost": rate_cost,
        "waterfill_utility_score": utility,
        "synergizes_with": flags["synergizes_with"],
        "antagonizes_with": flags["antagonizes_with"],
        "calibration_required": calibration_required,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "requires_exact_cuda_stack_eval": True,
    }


def _full_pipeline_opportunity_atoms(
    top_submissions: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    by_pr = _indexed_top_items(top_submissions)
    pr67 = by_pr.get(67)
    pr65 = by_pr.get(65)
    pr67_archive_bytes = pr67.get("archive_bytes") if pr67 else None
    pr65_archive_bytes = pr65.get("archive_bytes") if pr65 else None
    pr65_extra = (
        int(pr65_archive_bytes) - int(pr67_archive_bytes)
        if isinstance(pr65_archive_bytes, int) and isinstance(pr67_archive_bytes, int)
        else None
    )
    mask_bytes = _segment_bytes(pr67, "mask_obu_br")
    pose_bytes = _segment_bytes(pr67, "pose_qp1_br")
    model_bytes = _segment_bytes(pr67, "model_qzs3_br")
    atoms = [
        _pipeline_opportunity_atom(
            atom_id="pipeline:pr65_postprocess_residual_table",
            family="postprocess",
            stage="postprocess",
            description=(
                "Use PR65-style charged residual/postprocess slots only where "
                "reference-relative component trace utility clears byte cost."
            ),
            charged_bytes=max(1, pr65_extra) if pr65_extra is not None else None,
            charged_bytes_source="pr65 archive bytes - pr67 archive bytes",
            evidence_grade="external_plus_diagnostic_trace_prior",
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:pr67_qp1_pose_stream",
            family="pose",
            stage="prediction",
            description=(
                "Treat QP1/qpose pose bytes as a trained-renderer contract atom; "
                "do not transplant onto untrained renderer geometry without exact eval."
            ),
            charged_bytes=pose_bytes,
            charged_bytes_source="pr67 pose_qp1_br compressed bytes",
            evidence_grade="external_plus_empirical_byte_anatomy",
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:mask_obu_grammar_stream",
            family="mask",
            stage="representation",
            description=(
                "Promote mask grammar, pair/class/tile, and boundary atoms through "
                "charged mask-stream edits with pose-sensitive gates."
            ),
            charged_bytes=mask_bytes,
            charged_bytes_source="pr67 mask_obu_br compressed bytes",
            evidence_grade="external_plus_empirical_byte_anatomy",
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:qzs3_renderer_quant_groups",
            family="renderer_quantization",
            stage="quantization",
            description=(
                "Allocate renderer QZS3/FP4/grouped quantization bits by component "
                "response, not byte reduction alone."
            ),
            charged_bytes=model_bytes,
            charged_bytes_source="pr67 model_qzs3_br compressed bytes",
            evidence_grade="external_plus_empirical_byte_anatomy",
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:single_member_archive_overhead",
            family="archive_overhead",
            stage="pack",
            description=(
                "Use deterministic single-member layout, fixed names, and length-table "
                "packing only when strict ZIP custody remains valid."
            ),
            charged_bytes=100,
            charged_bytes_source="observed PR67 archive bytes minus container bytes",
            expected_score_saved=100 * RATE_SLOPE_SCORE_PER_BYTE,
            evidence_grade="external_plus_empirical_byte_anatomy",
            calibration_required=False,
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:runtime_simplification_fixed_decoder",
            family="runtime_simplification",
            stage="runtime",
            description=(
                "Prefer fixed-segment deterministic decode paths when they reduce "
                "inflate risk without hiding score-affecting side information."
            ),
            charged_bytes=None,
            charged_bytes_source="runtime-only atom; byte value requires built archive",
            evidence_grade="derived_runtime_planning_prior",
        ),
        _pipeline_opportunity_atom(
            atom_id="pipeline:rl_bandit_multipass_selector",
            family="selection_policy",
            stage="optimizer",
            description=(
                "Use RL/bandit/multipass search only as offline proposal policy; final "
                "selected atoms and selector payload bits must be charged in the archive."
            ),
            charged_bytes=None,
            charged_bytes_source="selected payload dependent",
            evidence_grade="derived_optimizer_planning_prior",
        ),
    ]
    return atoms


def _byte_model_from_top_submissions(
    top_submissions: dict[str, Any] | None,
    *,
    n_samples: int,
    top_sample_count: int,
) -> dict[str, Any]:
    by_pr = _indexed_top_items(top_submissions)
    pr67 = by_pr.get(67)
    pr65 = by_pr.get(65)
    pr67_archive_bytes = pr67.get("archive_bytes") if pr67 else None
    pr65_archive_bytes = pr65.get("archive_bytes") if pr65 else None
    pose_total = _segment_bytes(pr67, "pose_qp1_br")
    mask_total = _segment_bytes(pr67, "mask_obu_br")
    pr65_extra = (
        int(pr65_archive_bytes) - int(pr67_archive_bytes)
        if isinstance(pr65_archive_bytes, int) and isinstance(pr67_archive_bytes, int)
        else None
    )
    return {
        "mask_pair_bytes": max(1, _ceil_div(mask_total or n_samples, n_samples)),
        "mask_pair_bytes_source": "ceil(pr67 mask_obu_br compressed bytes / n_samples)",
        "pose_pair_bytes": max(1, _ceil_div(pose_total or n_samples, n_samples)),
        "pose_pair_bytes_source": "ceil(pr67 pose_qp1_br compressed bytes / n_samples)",
        "postprocess_pair_bytes": max(
            1,
            _ceil_div(max(1, pr65_extra or top_sample_count), max(1, top_sample_count)),
        ),
        "postprocess_pair_bytes_source": (
            "ceil((pr65 archive bytes - pr67 archive bytes) / consumed top-excess samples)"
        ),
        "pr65_minus_pr67_archive_bytes": pr65_extra,
        "pr67_mask_obu_br_bytes": mask_total,
        "pr67_pose_qp1_br_bytes": pose_total,
    }


def _trace_comparison_allocation(
    trace_paths: list[Path],
    *,
    root: Path,
    top_submissions: dict[str, Any] | None,
) -> dict[str, Any]:
    atoms_by_id: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    untrusted_inputs: list[dict[str, Any]] = []
    byte_models: dict[str, Any] = {}
    for path in trace_paths:
        if not path.exists():
            missing_inputs.append(_rel(path, root=root))
            continue
        data = _json_load(path)
        allocator_use_allowed = data.get("allocator_use_allowed") is True
        source_files.append(
            {
                "artifact": _rel(path, root=root),
                "artifact_sha256": _sha256_file(path),
                "evidence_grade": data.get("evidence_grade"),
                "score_claim": data.get("score_claim"),
                "allocator_use_allowed": allocator_use_allowed,
                "hardware_statuses": data.get("hardware_statuses"),
            }
        )
        if not allocator_use_allowed:
            untrusted_inputs.append(
                {
                    "artifact": _rel(path, root=root),
                    "evidence_grade": data.get("evidence_grade"),
                    "hardware_statuses": data.get("hardware_statuses"),
                    "reason": (
                        "Trace comparison did not declare allocator_use_allowed=true. "
                        "Ranked repair atoms require same-hardware component traces; "
                        "cross-hardware or legacy comparisons remain diagnostic only."
                    ),
                }
            )
            continue
        candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
        candidate_label = str(candidate.get("label", path.stem))
        comparison_label = path.stem
        references = data.get("references") if isinstance(data.get("references"), list) else []
        for reference_block in references:
            if not isinstance(reference_block, dict):
                continue
            reference = reference_block.get("reference")
            reference_label = (
                str(reference.get("label"))
                if isinstance(reference, dict) and reference.get("label") is not None
                else str(data.get("best_reference_label", "reference"))
            )
            top_combined = [
                item
                for item in reference_block.get("top_excess_combined_samples", [])
                if isinstance(item, dict)
            ]
            top_pose = [
                item
                for item in reference_block.get("top_excess_pose_samples", [])
                if isinstance(item, dict)
            ]
            top_seg = [
                item
                for item in reference_block.get("top_excess_seg_samples", [])
                if isinstance(item, dict)
            ]
            n_samples = int(candidate.get("n_samples", 0) or 0) or len(
                reference_block.get("pair_deltas", [])
            ) or 600
            top_sample_count = max(len(top_combined), len(top_pose), len(top_seg), 1)
            byte_model = _byte_model_from_top_submissions(
                top_submissions,
                n_samples=n_samples,
                top_sample_count=top_sample_count,
            )
            byte_models[f"{comparison_label}:vs:{reference_label}"] = byte_model
            artifact = _rel(path, root=root)
            for family, samples, bytes_key, source_key in (
                (
                    "postprocess",
                    top_combined,
                    "postprocess_pair_bytes",
                    "postprocess_pair_bytes_source",
                ),
                ("pose", top_pose, "pose_pair_bytes", "pose_pair_bytes_source"),
                ("mask", top_seg, "mask_pair_bytes", "mask_pair_bytes_source"),
            ):
                for sample in samples:
                    atom = _atom_from_trace_sample(
                        sample=sample,
                        family=family,
                        charged_bytes=int(byte_model[bytes_key]),
                        charged_bytes_source=str(byte_model[source_key]),
                        comparison_label=comparison_label,
                        candidate_label=candidate_label,
                        reference_label=reference_label,
                        artifact=artifact,
                    )
                    if atom is not None:
                        atoms_by_id[atom["atom_id"]] = atom
    atoms = sorted(atoms_by_id.values(), key=_allocation_sort_key)
    for rank, atom in enumerate(atoms, start=1):
        atom["rank"] = rank
    family_summary: dict[str, Any] = {}
    for family in ("postprocess", "pose", "mask"):
        family_atoms = [atom for atom in atoms if atom["family"] == family]
        family_summary[family] = {
            "atom_count": len(family_atoms),
            "positive_ev_count": sum(1 for atom in family_atoms if atom["waterfill_positive_ev"]),
            "top_atom_ids": [atom["atom_id"] for atom in family_atoms[:10]],
        }
    return {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": (
            "derived_diagnostic_trace_waterfill"
            if not untrusted_inputs
            else "derived_diagnostic_trace_waterfill_with_blocked_untrusted_inputs"
        ),
        "method": (
            "reference-relative first-order component excess minus charged byte "
            "rate cost and deterministic interaction-risk penalty"
        ),
        "allocator_use_allowed": not untrusted_inputs,
        "allocator_blocked_untrusted_inputs": untrusted_inputs,
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "exact_eval_stack_gate_recommendations": _exact_eval_stack_gate_recommendations(),
        "byte_models": byte_models,
        "source_files": source_files,
        "family_summary": family_summary,
        "full_pipeline_opportunity_atoms": _full_pipeline_opportunity_atoms(top_submissions),
        "ranked_atoms": atoms,
        "missing_inputs": sorted(missing_inputs),
    }


def build_frontier_atom_ledger(
    *,
    root: Path,
    exact_candidates: list[tuple[str, Path]],
    packer_paths: list[Path],
    top_anatomy_path: Path | None,
    trace_comparison_paths: list[Path] | None = None,
) -> dict[str, Any]:
    missing_inputs: list[str] = []
    exact: list[dict[str, Any]] = []
    for label, path in exact_candidates:
        if path.exists():
            exact.append(_load_exact_candidate(label, path, root=root))
        else:
            missing_inputs.append(_rel(path, root=root))

    by_label = {item["label"]: item for item in exact}
    transitions: list[dict[str, Any]] = []
    # Only the accepted frontier sequence is a chain. Diagnostic failures stay
    # available for vs-anchor analysis, but must not become predecessor links.
    chain = [label for label in ("C-044", "C-049", "C-050", "C-051") if label in by_label]
    for left, right in zip(chain, chain[1:]):
        transitions.append(
            _transition(by_label[left], by_label[right], relation="chain_step")
        )
    if "C-044" in by_label:
        for label in chain:
            if label != "C-044":
                transitions.append(
                    _transition(by_label["C-044"], by_label[label], relation="vs_C-044")
                )
    if "C-050" in by_label and "C-051" in by_label:
        transitions.append(
            _transition(by_label["C-050"], by_label["C-051"], relation="qpose14_vs_posecd_rp2")
        )

    packers: list[dict[str, Any]] = []
    for path in packer_paths:
        if path.exists():
            packers.append(_load_packer(path, root=root))
        else:
            missing_inputs.append(_rel(path, root=root))

    top_submissions = None
    if top_anatomy_path is not None:
        if top_anatomy_path.exists():
            top_submissions = _top_submission_atoms(top_anatomy_path, root=root)
        else:
            missing_inputs.append(_rel(top_anatomy_path, root=root))

    atom_allocation = _trace_comparison_allocation(
        trace_comparison_paths or [],
        root=root,
        top_submissions=top_submissions,
    )
    missing_inputs.extend(atom_allocation["missing_inputs"])

    frontier_preference = (
        "C-063",
        "C-059",
        "C-058",
        "C-057",
        "C-056",
        "C-054",
        "C-053",
        "C-052",
        "C-051",
    )
    active_frontier = next(
        (by_label[label] for label in frontier_preference if label in by_label),
        exact[-1] if exact else None,
    )
    external_byte_gaps: list[dict[str, Any]] = []
    if active_frontier and top_submissions:
        frontier_bytes = int(active_frontier["archive_bytes"])
        for item in top_submissions["items"]:
            archive_bytes = item.get("archive_bytes")
            if isinstance(archive_bytes, int):
                delta = archive_bytes - frontier_bytes
                external_byte_gaps.append(
                    {
                        "external_label": item.get("label"),
                        "external_archive_bytes": archive_bytes,
                        "frontier_label": active_frontier["label"],
                        "frontier_archive_bytes": frontier_bytes,
                        "delta_bytes_external_minus_frontier": delta,
                        "rate_score_delta_external_minus_frontier": (
                            RATE_SLOPE_SCORE_PER_BYTE * delta
                        ),
                    }
                )

    return {
        "schema_version": 1,
        "tool": "experiments/build_frontier_atom_ledger.py",
        "score_claim": False,
        "evidence_grade": "derived_control_plane",
        "original_video_bytes": ORIGINAL_VIDEO_BYTES,
        "rate_slope_score_per_byte": RATE_SLOPE_SCORE_PER_BYTE,
        "active_frontier_label": active_frontier["label"] if active_frontier else None,
        "active_frontier_archive_sha256": (
            active_frontier["archive_sha256"] if active_frontier else None
        ),
        "frontier_chain_labels": chain,
        "exact_candidates": exact,
        "exact_transitions": transitions,
        "packer_atoms": packers,
        "top_submission_atoms": top_submissions,
        "atom_allocation_table": atom_allocation,
        "external_byte_gaps_vs_frontier": external_byte_gaps,
        "missing_inputs": sorted(missing_inputs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--top-anatomy",
        type=Path,
        default=Path(DEFAULT_TOP_ANATOMY),
        help="canonical top-submission anatomy JSON, or omit with --no-top-anatomy",
    )
    parser.add_argument("--no-top-anatomy", action="store_true")
    parser.add_argument(
        "--trace-comparison",
        type=Path,
        action="append",
        default=None,
        help="diagnostic trace comparison JSON to convert into ranked atoms",
    )
    parser.add_argument("--no-trace-comparisons", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    exact_candidates = [
        (label, root / relpath) for label, relpath in DEFAULT_EXACT_CANDIDATES
    ]
    packer_paths = [root / relpath for relpath in DEFAULT_PACKER_PROVENANCE]
    top_anatomy = None if args.no_top_anatomy else root / args.top_anatomy
    if args.no_trace_comparisons:
        trace_comparisons: list[Path] = []
    elif args.trace_comparison:
        trace_comparisons = [
            path if path.is_absolute() else root / path for path in args.trace_comparison
        ]
    else:
        trace_comparisons = [root / relpath for relpath in DEFAULT_TRACE_COMPARISONS]

    ledger = build_frontier_atom_ledger(
        root=root,
        exact_candidates=exact_candidates,
        packer_paths=packer_paths,
        top_anatomy_path=top_anatomy,
        trace_comparison_paths=trace_comparisons,
    )

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": _rel(output, root=root), **ledger}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
