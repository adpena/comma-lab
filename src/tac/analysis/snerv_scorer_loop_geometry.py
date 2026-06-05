# SPDX-License-Identifier: MIT
"""Geometry analysis for SNeRV scorer-loop QAT result packets.

This module decomposes local SNeRV scorer-loop moves into the exact contest
Lagrangian components:

    100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / N

It is deliberately false-authority.  Its job is to explain local descent
directions and produce planner units for full-600 receiver-proof follow-up, not
to claim score, rank, promotion, or exact CPU/CUDA parity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.repo_io import read_json, sha256_file
from tac.score_geometry import (
    CONTEST_REFERENCE_BYTES,
    RATE_COEFFICIENT,
    SEG_COEFFICIENT,
    operating_regime,
)
from tac.substrates._shared.mlx_score_aware.nerv_byte_price_controller import (
    build_nerv_byte_price_plan,
)

SCHEMA = "snerv_scorer_loop_geometry.v1"
AUTHORITY = "false_authority_macos_cpu_snerv_scorer_loop_geometry_no_score_claim"
DEFAULT_PUBLIC_FRONTIER_REFERENCE = 0.192
BYTE_PRICE = RATE_COEFFICIENT / CONTEST_REFERENCE_BYTES

FALSE_AUTHORITY = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class SnervScorerLoopGeometryError(ValueError):
    """Raised when a scorer-loop packet cannot be analyzed."""


def build_snerv_scorer_loop_geometry_report(
    result_json_paths: Sequence[str | Path],
    *,
    label: str = "snerv_scorer_loop_geometry",
    frontier_reference_score: float = DEFAULT_PUBLIC_FRONTIER_REFERENCE,
) -> dict[str, Any]:
    """Build a machine-readable geometry report from one or more result JSONs."""

    if not result_json_paths:
        raise SnervScorerLoopGeometryError("at least one result JSON path is required")
    if frontier_reference_score <= 0.0:
        raise SnervScorerLoopGeometryError("frontier_reference_score must be > 0")

    reports = [_analyze_one(Path(path)) for path in result_json_paths]
    best_descent = min(reports, key=lambda row: float(row["score_delta_linf"]))
    lowest_local_score = min(reports, key=lambda row: float(row["best_score_linf"]))
    aggregate = _aggregate_reports(
        reports,
        frontier_reference_score=float(frontier_reference_score),
    )
    blockers = [
        "snerv_scorer_loop_geometry_is_false_authority",
        "paired_contest_cpu_cuda_auth_eval_missing",
    ]
    if any(int(row.get("n_pairs") or 0) != 600 for row in reports):
        blockers.append("full600_receiver_proof_required")
    if any(row.get("receiver_contract_satisfied") is not True for row in reports):
        blockers.append("receiver_contract_not_satisfied_for_some_inputs")
    if any(int(row.get("rejected_score_descent_count") or 0) > 0 for row in reports):
        blockers.append("snerv_rejected_scorer_descent_admission_repair_required")

    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "label": str(label),
        "input_count": len(reports),
        "byte_price": BYTE_PRICE,
        "frontier_reference_score": float(frontier_reference_score),
        "best_input_path": best_descent["input_path"],
        "best_score_linf": best_descent["best_score_linf"],
        "best_score_delta_linf": best_descent["score_delta_linf"],
        "best_descent_input_path": best_descent["input_path"],
        "best_descent_score_linf": best_descent["best_score_linf"],
        "best_descent_score_delta_linf": best_descent["score_delta_linf"],
        "lowest_local_score_input_path": lowest_local_score["input_path"],
        "lowest_local_score_linf": lowest_local_score["best_score_linf"],
        "lowest_local_score_delta_linf": lowest_local_score["score_delta_linf"],
        "reports": reports,
        "aggregate": aggregate,
        "allocator_units": [_allocator_unit(row) for row in reports],
        "recommended_next_actions": _recommended_next_actions(reports, aggregate),
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def render_snerv_scorer_loop_geometry_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for operator review."""

    aggregate = report.get("aggregate") or {}
    lines = [
        "# SNeRV scorer-loop geometry",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Inputs: `{report.get('input_count')}`",
        f"Best descent delta: `{report.get('best_descent_score_delta_linf')}`",
        f"Best descent input: `{report.get('best_descent_input_path')}`",
        f"Lowest local score: `{report.get('lowest_local_score_linf')}`",
        f"Lowest local score input: `{report.get('lowest_local_score_input_path')}`",
        "",
        "## Aggregate",
        "",
        f"- Best search mode: `{aggregate.get('best_search_mode')}`",
        f"- Dominant lowering axis: `{aggregate.get('dominant_lowering_axis')}`",
        f"- Accepted trial count: `{aggregate.get('accepted_trial_count')}`",
        f"- Evaluated trial count: `{aggregate.get('evaluated_trial_count')}`",
        f"- Rate is current descent driver: `{aggregate.get('rate_is_current_descent_driver')}`",
        "",
        "## Inputs",
        "",
    ]
    for row in report.get("reports") or ():
        best = row.get("best_contribution") or {}
        lines.extend(
            [
                f"### `{Path(str(row.get('input_path'))).parent.name}`",
                "",
                f"- Search: `{row.get('search_mode')}`",
                f"- Pairs: `{row.get('n_pairs')}`",
                f"- Score: `{row.get('baseline_score_linf')}` -> `{row.get('best_score_linf')}`",
                f"- Delta: `{row.get('score_delta_linf')}`",
                f"- Seg contribution delta: `{best.get('delta_seg_term')}`",
                f"- Pose contribution delta: `{best.get('delta_pose_term')}`",
                f"- Rate contribution delta: `{best.get('delta_rate_term')}`",
                f"- Rejected score descents: `{row.get('rejected_score_descent_count')}`",
                f"- Best rejected score descent: `{row.get('best_rejected_score_descent')}`",
                f"- Geometry verdicts: `{row.get('geometry_verdicts')}`",
                "",
            ]
        )
    lines.extend(["## Blockers", ""])
    for blocker in report.get("blockers") or ():
        lines.append(f"- `{blocker}`")
    lines.append("")
    return "\n".join(lines)


def _analyze_one(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    result = _result_payload(payload)
    best_packet_materialization = _best_packet_materialization(payload)
    baseline = _required_mapping(result, "baseline")
    best = _required_mapping(result, "best")
    evaluations = [_as_mapping(row) for row in result.get("evaluations") or []]
    if not evaluations:
        raise SnervScorerLoopGeometryError(f"{path} has no evaluations")

    baseline_contribution = _contribution(
        candidate=baseline,
        reference=baseline,
        label="baseline",
    )
    move_contributions = [
        _contribution(
            candidate=row,
            reference=baseline,
            label=str(row.get("label") or f"trial_{idx}"),
        )
        for idx, row in enumerate(evaluations)
    ]
    accepted_trials = [
        row
        for row in move_contributions
        if row["accepted"] is True and row["label"] != str(baseline.get("label"))
    ]
    rejected_score_descents = [
        row
        for row in move_contributions
        if row["accepted"] is not True
        and row["label"] != str(baseline.get("label"))
        and float(row.get("score_delta_linf") or 0.0) < 0.0
        and "nes_probe_only_not_candidate" not in set(row.get("blockers") or ())
    ]
    best_rejected_score_descent = (
        min(rejected_score_descents, key=lambda row: float(row["score_delta_linf"]))
        if rejected_score_descents
        else None
    )
    best_contribution = _contribution(
        candidate=best,
        reference=baseline,
        label=str(best.get("label") or "best"),
    )
    axis_tag = result.get("axis_tag", payload.get("axis_tag"))
    n_pairs = int(result.get("n_pairs") or payload.get("n_pairs") or 0)
    receiver_contract_satisfied = bool(result.get("receiver_contract_satisfied"))
    section_value_rows = _section_value_rows_for_contributions(
        move_contributions,
        axis_tag=axis_tag,
        n_pairs=n_pairs,
        receiver_contract_satisfied=receiver_contract_satisfied,
    )
    byte_price_plan = build_nerv_byte_price_plan(
        {
            "schema": f"{SCHEMA}.section_value_rows",
            "family": "snerv",
            "candidate_id": f"snerv_scorer_loop:{path.parent.name}",
            "axis_tag": axis_tag,
            "receiver_proof_status": (
                "satisfied" if receiver_contract_satisfied else "missing"
            ),
            "full_video_coverage": n_pairs >= 600,
            "section_value_rows": section_value_rows,
        }
    )
    regime = operating_regime(float(baseline.get("d_pose_linf") or 0.0))
    return {
        "unit_type": "snerv_scorer_loop_geometry_result",
        "family": "snerv",
        "input_path": path.as_posix(),
        "input_sha256": sha256_file(path),
        "axis_tag": axis_tag,
        "schema": result.get("schema"),
        "n_pairs": n_pairs,
        "levels": result.get("levels"),
        "wavelet": result.get("wavelet"),
        "qat_bits": result.get("qat_bits"),
        "search_mode": result.get("search_mode"),
        "component_guard_mode": result.get("component_guard_mode"),
        "scorer_loop_evaluations": int(result.get("scorer_loop_evaluations") or 0),
        "baseline_archive_bytes": _required_int(baseline, "archive_bytes"),
        "best_archive_bytes": _required_int(best, "archive_bytes"),
        "baseline_score_linf": _required_float(baseline, "score_linf"),
        "best_score_linf": _required_float(best, "score_linf"),
        "score_delta_linf": _required_float(best, "score_linf")
        - _required_float(baseline, "score_linf"),
        "baseline_contribution": baseline_contribution,
        "best_contribution": best_contribution,
        "move_contributions": move_contributions,
        "section_value_rows": section_value_rows,
        "byte_price_plan": byte_price_plan,
        "accepted_trial_count": len(accepted_trials),
        "rejected_trial_count": max(len(move_contributions) - 1 - len(accepted_trials), 0),
        "rejected_score_descent_count": len(rejected_score_descents),
        "best_rejected_score_descent": best_rejected_score_descent,
        "accepted_trials": accepted_trials,
        "best_packet_materialized": best_packet_materialization["materialized"],
        "best_packet_path": best_packet_materialization["path"],
        "best_packet_bytes": best_packet_materialization["bytes"],
        "best_packet_sha256": best_packet_materialization["sha256"],
        "best_packet_materialization": best_packet_materialization,
        "best_pair_deltas": list(result.get("best_pair_deltas") or ()),
        "operating_regime": {
            "d_pose": regime.d_pose,
            "flip_threshold": regime.flip_threshold,
            "seg_dominates": regime.seg_dominates,
            "pose_dominates": regime.pose_dominates,
            "marginal_ratio_seg_over_pose": regime.marginal_ratio_seg_over_pose,
            "advice": regime.advice,
        },
        "accepted_improvement": bool(result.get("accepted_improvement")),
        "ready_for_pose_guard_gate": bool(result.get("ready_for_pose_guard_gate")),
        "receiver_contract_satisfied": receiver_contract_satisfied,
        "result_sha256": sha256_file(path),
        "blockers": list(result.get("blockers") or payload.get("blockers") or ()),
        "geometry_verdicts": _geometry_verdicts(best_contribution, accepted_trials),
        **FALSE_AUTHORITY,
    }


def _contribution(
    *,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    cand_seg = _required_float(candidate, "d_seg_linf")
    ref_seg = _required_float(reference, "d_seg_linf")
    cand_pose = _required_float(candidate, "d_pose_linf")
    ref_pose = _required_float(reference, "d_pose_linf")
    cand_bytes = _required_int(candidate, "archive_bytes")
    ref_bytes = _required_int(reference, "archive_bytes")
    cand_score = _required_float(candidate, "score_linf")
    ref_score = _required_float(reference, "score_linf")
    cand_rate = _optional_float(candidate, "rate_term", BYTE_PRICE * cand_bytes)
    ref_rate = _optional_float(reference, "rate_term", BYTE_PRICE * ref_bytes)
    delta_seg_term = SEG_COEFFICIENT * (cand_seg - ref_seg)
    delta_pose_term = _pose_term(cand_pose) - _pose_term(ref_pose)
    delta_rate_term = cand_rate - ref_rate
    reconstructed = delta_seg_term + delta_pose_term + delta_rate_term
    non_rate_delta = delta_seg_term + delta_pose_term
    max_added_bytes_from_non_rate_gain = (
        math.floor(-non_rate_delta / BYTE_PRICE) if non_rate_delta < 0.0 else 0
    )
    return {
        "label": str(label),
        "accepted": bool(candidate.get("accepted")),
        "archive_bytes": cand_bytes,
        "archive_sha256": str(candidate.get("archive_sha256") or ""),
        "byte_delta": cand_bytes - ref_bytes,
        "score_linf": cand_score,
        "score_delta_linf": cand_score - ref_score,
        "d_seg_linf": cand_seg,
        "d_seg_delta_linf": cand_seg - ref_seg,
        "d_pose_linf": cand_pose,
        "d_pose_delta_linf": cand_pose - ref_pose,
        "delta_seg_term": delta_seg_term,
        "delta_pose_term": delta_pose_term,
        "delta_rate_term": delta_rate_term,
        "delta_non_rate_term": non_rate_delta,
        "reconstructed_score_delta": reconstructed,
        "score_delta_reconstruction_residual": (cand_score - ref_score) - reconstructed,
        "max_added_bytes_from_non_rate_gain": max_added_bytes_from_non_rate_gain,
        "dominant_lowering_axis": _dominant_axis(
            {
                "seg": delta_seg_term,
                "pose": delta_pose_term,
                "rate": delta_rate_term,
            }
        ),
        "component_tradeoff": _component_tradeoff(delta_seg_term, delta_pose_term),
        "blockers": list(candidate.get("blockers") or ()),
    }


def _section_value_rows_for_contributions(
    contributions: Sequence[Mapping[str, Any]],
    *,
    axis_tag: Any,
    n_pairs: int,
    receiver_contract_satisfied: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_video = int(n_pairs) >= 600
    receiver_status = "satisfied" if receiver_contract_satisfied else "missing"
    for contrib in contributions:
        label = str(contrib.get("label") or "")
        if _is_baseline_contribution(contrib):
            continue
        byte_delta = int(contrib.get("byte_delta") or 0)
        rows.append(
            {
                "row_id": f"snerv_scorer_loop_move:{_safe_row_label(label)}",
                "section_id": f"snerv_scorer_loop_move:{_safe_row_label(label)}",
                "row_kind": (
                    "new_residual_or_sidecar"
                    if byte_delta > 0
                    else "existing_section_cut"
                ),
                "family": "snerv",
                "scope": "scorer_loop_decoder_candidate",
                "byte_delta": byte_delta,
                "section_bytes": abs(byte_delta)
                or int(contrib.get("archive_bytes") or 0),
                "delta_nonrate_score": float(
                    contrib.get("delta_non_rate_term") or 0.0
                ),
                "axis_tag": str(axis_tag or ""),
                "receiver_proof_status": receiver_status,
                "full_video_coverage": full_video,
                "archive_sha256": str(contrib.get("archive_sha256") or "") or None,
                "accepted": bool(contrib.get("accepted")),
                "score_delta_linf": contrib.get("score_delta_linf"),
                "delta_seg_term": contrib.get("delta_seg_term"),
                "delta_pose_term": contrib.get("delta_pose_term"),
                "delta_rate_term": contrib.get("delta_rate_term"),
                "dominant_lowering_axis": contrib.get("dominant_lowering_axis"),
                "blockers": list(contrib.get("blockers") or ()),
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _is_baseline_contribution(contrib: Mapping[str, Any]) -> bool:
    return (
        int(contrib.get("byte_delta") or 0) == 0
        and abs(float(contrib.get("score_delta_linf") or 0.0)) <= 1e-15
        and abs(float(contrib.get("delta_non_rate_term") or 0.0)) <= 1e-15
    )


def _safe_row_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    return safe or "candidate"


def _aggregate_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    frontier_reference_score: float,
) -> dict[str, Any]:
    best_descent = min(reports, key=lambda row: float(row["score_delta_linf"]))
    lowest_local_score = min(reports, key=lambda row: float(row["best_score_linf"]))
    by_mode: dict[str, list[float]] = defaultdict(list)
    accepted = 0
    evaluated = 0
    for row in reports:
        by_mode[str(row.get("search_mode"))].append(float(row["score_delta_linf"]))
        accepted += int(row.get("accepted_trial_count") or 0)
        evaluated += max(int(row.get("scorer_loop_evaluations") or 0) - 1, 0)
    mode_summary = {
        mode: {
            "run_count": len(values),
            "best_score_delta_linf": min(values),
            "mean_score_delta_linf": sum(values) / len(values),
        }
        for mode, values in sorted(by_mode.items())
    }
    best_mode = min(
        mode_summary,
        key=lambda mode: float(mode_summary[mode]["best_score_delta_linf"]),
    )
    best_contrib = best_descent.get("best_contribution") or {}
    non_rate = abs(float(best_contrib.get("delta_non_rate_term") or 0.0))
    rate = abs(float(best_contrib.get("delta_rate_term") or 0.0))
    return {
        "best_search_mode": best_mode,
        "search_mode_summary": mode_summary,
        "accepted_trial_count": accepted,
        "evaluated_trial_count": evaluated,
        "accepted_fraction": accepted / evaluated if evaluated else 0.0,
        "dominant_lowering_axis": best_contrib.get("dominant_lowering_axis"),
        "rate_is_current_descent_driver": rate > non_rate,
        "best_descent_score_vs_frontier_reference_gap": (
            float(best_descent["best_score_linf"]) - float(frontier_reference_score)
        ),
        "best_descent_input_path": best_descent.get("input_path"),
        "best_descent_local_false_authority_score": best_descent.get("best_score_linf"),
        "best_descent_score_delta_linf": best_descent.get("score_delta_linf"),
        "lowest_local_score_input_path": lowest_local_score.get("input_path"),
        "lowest_local_false_authority_score": lowest_local_score.get("best_score_linf"),
        "best_geometry_verdicts": best_descent.get("geometry_verdicts"),
    }


def _allocator_unit(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unit_type": "snerv_scorer_loop_qat_result",
        "family": "snerv",
        "unit_id": f"snerv_scorer_loop_qat:{Path(str(row.get('input_path'))).parent.name}",
        "report_path": row.get("input_path"),
        "axis_tag": row.get("axis_tag"),
        "n_pairs": row.get("n_pairs"),
        "levels": row.get("levels"),
        "wavelet": row.get("wavelet"),
        "qat_bits": row.get("qat_bits"),
        "search_mode": row.get("search_mode"),
        "component_guard_mode": row.get("component_guard_mode"),
        "scorer_loop_evaluations": row.get("scorer_loop_evaluations"),
        "history_count": len(row.get("move_contributions") or ()),
        "selection_policy": "score_primary_lagrangian_geometry",
        "baseline_archive_bytes": row.get("baseline_archive_bytes"),
        "best_archive_bytes": row.get("best_archive_bytes"),
        "baseline_score_linf": row.get("baseline_score_linf"),
        "best_score_linf": row.get("best_score_linf"),
        "score_delta_linf": row.get("score_delta_linf"),
        "score_delta_fraction": (
            float(row["score_delta_linf"]) / float(row["baseline_score_linf"])
            if float(row.get("baseline_score_linf") or 0.0) != 0.0
            else None
        ),
        "candidate_count": max(int(row.get("scorer_loop_evaluations") or 0) - 1, 0),
        "accepted_candidate_count": row.get("accepted_trial_count"),
        "rejected_candidate_count": row.get("rejected_trial_count"),
        "rejected_score_descent_count": row.get("rejected_score_descent_count"),
        "best_rejected_score_descent": row.get("best_rejected_score_descent"),
        "best_packet_materialized": row.get("best_packet_materialized"),
        "best_packet_path": row.get("best_packet_path"),
        "best_packet_bytes": row.get("best_packet_bytes"),
        "best_packet_sha256": row.get("best_packet_sha256"),
        "best_packet_materialization": row.get("best_packet_materialization"),
        "best_pair_deltas": row.get("best_pair_deltas"),
        "section_value_rows": row.get("section_value_rows") or [],
        "byte_price_plan": row.get("byte_price_plan") or {},
        "accepted_improvement": row.get("accepted_improvement"),
        "ready_for_pose_guard_gate": row.get("ready_for_pose_guard_gate"),
        "receiver_contract_satisfied": row.get("receiver_contract_satisfied"),
        "result_sha256": row.get("result_sha256"),
        **FALSE_AUTHORITY,
    }


def _recommended_next_actions(
    reports: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    best_mode = str(aggregate.get("best_search_mode"))
    accepted_trial_count = int(aggregate.get("accepted_trial_count") or 0)
    best_packet_materialized = any(
        row.get("best_packet_materialized") is True for row in reports
    )
    actions = [
        {
            "id": "replace_random_directions_with_decoder_weight_vjp",
            "priority": 9,
            "why": (
                "finite-difference probes expose descent but are CPU-expensive; "
                "PR95-grade grandfather needs scorer-gradient training, not "
                "post-hoc random search"
            ),
            "blockers": [
                "snerv_decoder_weight_vjp_binding_missing",
                "eval_roundtrip_gradient_reachability_proof_required",
            ],
        },
    ]
    if accepted_trial_count > 0:
        actions.insert(
            0,
            {
                "id": "scale_score_primary_random_subspace_batch",
                "priority": 10 if best_mode == "learned_random_subspace" else 8,
                "why": (
                    "current local geometry shows accepted scorer-primary descent; "
                    "scale to more pairs/trials before exact spend"
                ),
                "blockers": [
                    "full600_receiver_proof_required",
                    "mlx_or_cuda_batched_scorer_loop_needed_for_velocity",
                ],
            },
        )
        actions.append(
            {
                "id": "bind_archive_codec_to_descent_step",
                "priority": 8,
                "why": (
                    "accepted local moves are distortion-driven; next carrier must "
                    f"{'price' if best_packet_materialized else 'materialize'} "
                    "the best decoder with PR95-style byte maps/stream splits"
                ),
                "blockers": [
                    "mixed_precision_decoder_payload_grammar_not_byte_optimized",
                    *(
                        []
                        if best_packet_materialized
                        else ["best_decoder_packet_materialization_missing"]
                    ),
                ],
            },
        )
    if any((row.get("best_contribution") or {}).get("component_tradeoff") for row in reports):
        actions.insert(
            1,
            {
                "id": "keep_score_primary_component_tradeoff_default",
                "priority": 10,
                "why": (
                    "at least one accepted move lowers total score while one "
                    "component drifts; hard component gates hide valid scorer "
                    "directions"
                ),
                "blockers": ["paired_contest_cpu_cuda_replay_missing"],
            },
        )
    if any(int(row.get("rejected_score_descent_count") or 0) > 0 for row in reports):
        actions.insert(
            0,
            {
                "id": "repair_rejected_scorer_descent_admission",
                "priority": 10,
                "why": (
                    "at least one receiver-replayed decoder candidate lowered "
                    "local score but failed byte or pair admission; repair the "
                    "training/search guard before full600 replay"
                ),
                "blockers": [
                    "snerv_rejected_scorer_descent_admission_repair_required",
                    "paired_contest_cpu_cuda_replay_missing",
                ],
            },
        )
    return actions


def _geometry_verdicts(
    best: Mapping[str, Any],
    accepted_trials: Sequence[Mapping[str, Any]],
) -> list[str]:
    verdicts = []
    if float(best.get("score_delta_linf") or 0.0) < 0.0:
        verdicts.append("score_primary_found_local_descent")
    if best.get("dominant_lowering_axis") == "pose":
        verdicts.append("pose_geometry_primary_current_descent")
    if best.get("dominant_lowering_axis") == "seg":
        verdicts.append("seg_boundary_primary_current_descent")
    if best.get("dominant_lowering_axis") == "rate":
        verdicts.append("rate_primary_current_descent")
    if best.get("component_tradeoff"):
        verdicts.append("component_tradeoff_admitted_by_lagrangian")
    if accepted_trials:
        verdicts.append("receiver_replayed_accepted_candidate_exists")
    if abs(float(best.get("delta_rate_term") or 0.0)) < abs(
        float(best.get("delta_non_rate_term") or 0.0)
    ):
        verdicts.append("rate_not_current_descent_driver")
    return verdicts


def _result_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result = payload.get("result")
    if isinstance(result, Mapping):
        return result
    return payload


def _best_packet_materialization(payload: Mapping[str, Any]) -> dict[str, Any]:
    materialization = payload.get("best_packet_materialization")
    if isinstance(materialization, Mapping):
        return {
            "materialized": materialization.get("materialized") is True,
            "path": materialization.get("best_packet_path"),
            "bytes": materialization.get("best_packet_bytes"),
            "sha256": materialization.get("best_packet_sha256"),
            "source": "best_packet_materialization",
        }
    return {
        "materialized": payload.get("best_packet_materialized") is True,
        "path": payload.get("best_packet_path"),
        "bytes": payload.get("best_packet_bytes"),
        "sha256": payload.get("best_packet_sha256"),
        "source": "top_level_fields",
    }


def _pose_term(d_pose: float) -> float:
    return math.sqrt(10.0 * max(float(d_pose), 0.0))


def _dominant_axis(contributions: Mapping[str, float]) -> str | None:
    lowering = {axis: value for axis, value in contributions.items() if value < 0.0}
    if not lowering:
        return None
    return min(lowering, key=lambda axis: lowering[axis])


def _component_tradeoff(delta_seg_term: float, delta_pose_term: float) -> str | None:
    if delta_seg_term > 0.0 and delta_pose_term < 0.0:
        return "seg_worsens_pose_improves"
    if delta_seg_term < 0.0 and delta_pose_term > 0.0:
        return "seg_improves_pose_worsens"
    return None


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise SnervScorerLoopGeometryError(f"missing mapping: {key}")
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnervScorerLoopGeometryError("evaluation row is not a mapping")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise SnervScorerLoopGeometryError(f"missing float: {key}")
    return float(value)


def _optional_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    return float(default if value is None else value)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise SnervScorerLoopGeometryError(f"missing int: {key}")
    return int(value)


__all__ = [
    "AUTHORITY",
    "BYTE_PRICE",
    "DEFAULT_PUBLIC_FRONTIER_REFERENCE",
    "FALSE_AUTHORITY",
    "SCHEMA",
    "SnervScorerLoopGeometryError",
    "build_snerv_scorer_loop_geometry_report",
    "render_snerv_scorer_loop_geometry_markdown",
]
