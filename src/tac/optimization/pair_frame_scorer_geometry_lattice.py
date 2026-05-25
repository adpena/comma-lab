# SPDX-License-Identifier: MIT
"""Pair/frame scorer-geometry lattice for DQS1 rate/distortion starts.

This module binds DQS1 pairset acquisition rows to the scorer topology that
matters for the contest: SegNet sees the last frame of each pair, PoseNet sees
both frames, and rate savings can be reinvested into repair.  It is a
planning-only surface; its executable requests are restricted to DQS1 pairset
drops that the existing local materializer queue already knows how to run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from tac.optimization.decoder_q_selective_runtime_packet import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

SCHEMA = "pair_frame_scorer_geometry_lattice.v1"
ROW_SCHEMA = "pair_frame_scorer_geometry_lattice_row.v1"
REQUEST_SCHEMA = "pair_frame_geometry_queue_executable_drop_request.v1"
TOOL = "tac.optimization.pair_frame_scorer_geometry_lattice"

FALSE_GEOMETRY_AUTHORITY: dict[str, bool] = {
    **FALSE_AUTHORITY,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


class PairFrameScorerGeometryLatticeError(ValueError):
    """Raised when scorer-geometry inputs cannot be fused safely."""


def dumps_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PairFrameScorerGeometryLatticeError(f"{path}: expected JSON object")
    return payload


def _require_false_authority(payload: Mapping[str, Any], *, label: str) -> None:
    try:
        require_no_truthy_authority_fields(payload, context=label)
    except ValueError as exc:
        raise PairFrameScorerGeometryLatticeError(str(exc)) from exc


def _finite_float(value: Any, *, label: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PairFrameScorerGeometryLatticeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise PairFrameScorerGeometryLatticeError(f"{label} must be finite")
    return result


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise PairFrameScorerGeometryLatticeError(f"{label} must be a sequence")
    out: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise PairFrameScorerGeometryLatticeError(
                f"{label}[{index}] must be an integer"
            )
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise PairFrameScorerGeometryLatticeError(
                f"{label}[{index}] must be an integer"
            ) from exc
        if parsed != item and not (isinstance(item, str) and str(parsed) == item):
            raise PairFrameScorerGeometryLatticeError(
                f"{label}[{index}] must be integral"
            )
        out.append(parsed)
    return out


def _unique_pairs(values: Sequence[int], *, label: str) -> list[int]:
    pairs = [int(value) for value in values]
    if not pairs:
        raise PairFrameScorerGeometryLatticeError(f"{label} must not be empty")
    if len(set(pairs)) != len(pairs):
        raise PairFrameScorerGeometryLatticeError(f"{label} contains duplicates")
    return pairs


def _parse_drop_counts(values: Sequence[int] | None, *, max_count: int) -> list[int]:
    raw = values if values is not None else (3, 4, 6, 8, 12, 16)
    counts = sorted({int(value) for value in raw if 1 < int(value) < max_count})
    return counts


def _best_pair_order(acquisition_plan: Mapping[str, Any]) -> list[int]:
    summary = acquisition_plan.get("source_selector_summary")
    if not isinstance(summary, Mapping):
        raise PairFrameScorerGeometryLatticeError(
            "pairset acquisition missing source_selector_summary"
        )
    raw_order = summary.get("best_rank_order_pair_indices")
    if raw_order is None:
        raw_order = summary.get("best_selected_pair_indices")
    pairs = _int_list(raw_order, label="best_rank_order_pair_indices")
    return _unique_pairs(pairs, label="best_rank_order_pair_indices")


def _pair_row_map(frame_pair_curriculum: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    if frame_pair_curriculum is None:
        return {}
    _require_false_authority(frame_pair_curriculum, label="frame_pair_curriculum")
    rows = frame_pair_curriculum.get("pair_rows")
    if not isinstance(rows, list):
        raise PairFrameScorerGeometryLatticeError(
            "frame_pair_curriculum pair_rows[] missing"
        )
    out: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PairFrameScorerGeometryLatticeError(
                f"frame_pair_curriculum pair_rows[{index}] must be object"
            )
        pair_index = int(row.get("pair_index"))
        out[pair_index] = dict(row)
    return out


def _xray_row_map(pair_component_xrays: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for payload_index, payload in enumerate(pair_component_xrays):
        _require_false_authority(payload, label=f"pair_component_xray[{payload_index}]")
        if payload.get("schema") != "pair_component_error_xray_v1":
            raise PairFrameScorerGeometryLatticeError(
                f"pair_component_xray[{payload_index}] schema mismatch"
            )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise PairFrameScorerGeometryLatticeError(
                f"pair_component_xray[{payload_index}] rows[] missing"
            )
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise PairFrameScorerGeometryLatticeError(
                    f"pair_component_xray[{payload_index}].rows[{row_index}] must be object"
                )
            pair_index = int(row.get("pair_idx"))
            if pair_index not in out:
                out[pair_index] = dict(row)
    return out


def _normalize(values: Mapping[int, float]) -> dict[int, float]:
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {}
    lo = min(finite)
    hi = max(finite)
    if math.isclose(lo, hi, abs_tol=1.0e-18):
        return dict.fromkeys(values, 0.0)
    span = hi - lo
    return {key: (value - lo) / span for key, value in values.items()}


def _frame_pair_cost(pair_row: Mapping[str, Any] | None) -> float | None:
    if not isinstance(pair_row, Mapping):
        return None
    total = _finite_float_or_none(pair_row.get("total_l1"))
    if total is not None:
        return max(0.0, total)
    seg = _finite_float_or_none(pair_row.get("seg_l1")) or 0.0
    pose = _finite_float_or_none(pair_row.get("pose_l1")) or 0.0
    rate = _finite_float_or_none(pair_row.get("rate_l1")) or 0.0
    return max(0.0, seg + pose + rate)


def _xray_cost(xray_row: Mapping[str, Any] | None) -> float | None:
    if not isinstance(xray_row, Mapping):
        return None
    score = _finite_float_or_none(xray_row.get("component_score_no_rate"))
    if score is not None:
        return max(0.0, score)
    pose = _finite_float_or_none(xray_row.get("pose_score_contribution")) or 0.0
    seg = _finite_float_or_none(xray_row.get("seg_score_contribution")) or 0.0
    return max(0.0, pose + seg)


def _axis_mix(pair_row: Mapping[str, Any] | None, xray_row: Mapping[str, Any] | None) -> dict[str, float | None]:
    if isinstance(pair_row, Mapping) and isinstance(pair_row.get("axis_mix"), Mapping):
        mix = pair_row["axis_mix"]
        return {
            "seg_share": _finite_float_or_none(mix.get("seg_share")),
            "pose_share": _finite_float_or_none(mix.get("pose_share")),
            "rate_share": _finite_float_or_none(mix.get("rate_share")),
        }
    if isinstance(xray_row, Mapping):
        pose = _finite_float_or_none(xray_row.get("pose_score_contribution")) or 0.0
        seg = _finite_float_or_none(xray_row.get("seg_score_contribution")) or 0.0
        total = pose + seg
        if total > 0.0:
            return {
                "seg_share": seg / total,
                "pose_share": pose / total,
                "rate_share": 0.0,
            }
    return {"seg_share": None, "pose_share": None, "rate_share": None}


def _request_id(dropped_pairs: Sequence[int]) -> str:
    pairs = sorted(int(pair) for pair in dropped_pairs)
    digest = sha256(",".join(str(pair) for pair in pairs).encode("utf-8")).hexdigest()[:10]
    return f"pairset_geometry_lowimpact_k{len(pairs):03d}_h{digest}"


def _drop_requests(
    *,
    best_order: Sequence[int],
    rows_by_pair: Mapping[int, Mapping[str, Any]],
    drop_counts: Sequence[int],
    max_requests: int,
) -> list[dict[str, Any]]:
    if max_requests <= 0:
        return []
    ordered_rows = sorted(
        (rows_by_pair[pair] for pair in best_order),
        key=lambda row: (
            -float(row["low_impact_drop_priority"]),
            int(row["dqs1_rank"]),
            int(row["pair_index"]),
        ),
    )
    best_pair_set = {int(pair) for pair in best_order}
    requests: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for count in drop_counts:
        if len(requests) >= max_requests:
            break
        if count <= 1 or count >= len(best_order):
            continue
        dropped = tuple(sorted(int(row["pair_index"]) for row in ordered_rows[:count]))
        if len(dropped) != count or dropped in seen:
            continue
        seen.add(dropped)
        selected = sorted(best_pair_set.difference(dropped))
        covered = sum(
            1
            for pair in dropped
            if rows_by_pair[pair]["scorer_geometry_status"] != "rank_only"
        )
        requests.append(
            {
                "schema": REQUEST_SCHEMA,
                "candidate_id": _request_id(dropped),
                "selector_kind": "pair_frame_geometry_low_impact_drop_many",
                "dropped_pair_indices": list(dropped),
                "selected_pair_indices": selected,
                "selected_pair_count": len(selected),
                "geometry_covered_dropped_pair_count": covered,
                "geometry_coverage": covered / float(count),
                "queue_executable": True,
                "queue_family": "dqs1_pairset_local_first",
                "operator_next_action": "materialize_pairset_archive_and_run_local_controls",
                "generation_policy": (
                    "drop lowest scorer-geometry-cost DQS1 pairs and let local queue "
                    "measure the real SegNet/PoseNet trade"
                ),
                "allowed_use": "queue_executable_local_dqs1_pairset_drop_probe_only",
                "forbidden_use": "score_claim_or_dispatch_or_promotion_authority",
                **FALSE_GEOMETRY_AUTHORITY,
            }
        )
    return requests


def build_pair_frame_scorer_geometry_lattice(
    pairset_acquisition: Mapping[str, Any],
    *,
    frame_pair_curriculum: Mapping[str, Any] | None = None,
    pair_component_xrays: Sequence[Mapping[str, Any]] = (),
    drop_counts: Sequence[int] | None = None,
    max_requests: int = 32,
) -> dict[str, Any]:
    """Build a planning-only pair/frame lattice plus executable DQS1 drop requests."""

    if pairset_acquisition.get("schema") != "decoder_q_pairset_acquisition.v1":
        raise PairFrameScorerGeometryLatticeError("pairset acquisition schema mismatch")
    _require_false_authority(pairset_acquisition, label="pairset_acquisition")
    best_order = _best_pair_order(pairset_acquisition)
    if isinstance(max_requests, bool) or int(max_requests) < 0:
        raise PairFrameScorerGeometryLatticeError("max_requests must be non-negative")
    counts = _parse_drop_counts(drop_counts, max_count=len(best_order))
    curriculum_rows = _pair_row_map(frame_pair_curriculum)
    xray_rows = _xray_row_map(pair_component_xrays)
    frame_costs = {
        pair: cost
        for pair in best_order
        if (cost := _frame_pair_cost(curriculum_rows.get(pair))) is not None
    }
    xray_costs = {
        pair: cost
        for pair in best_order
        if (cost := _xray_cost(xray_rows.get(pair))) is not None
    }
    norm_frame = _normalize(frame_costs)
    norm_xray = _normalize(xray_costs)
    rank_den = max(1, len(best_order) - 1)
    rows: list[dict[str, Any]] = []
    for rank, pair in enumerate(best_order, start=1):
        pair_row = curriculum_rows.get(pair)
        xray_row = xray_rows.get(pair)
        rank_tail_priority = (rank - 1) / float(rank_den)
        sources: list[str] = []
        if pair in norm_frame:
            sources.append("frame_pair_curriculum")
        if pair in norm_xray:
            sources.append("pair_component_xray")
        if pair in norm_frame and pair in norm_xray:
            normalized_cost = 0.5 * norm_frame[pair] + 0.5 * norm_xray[pair]
        elif pair in norm_frame:
            normalized_cost = norm_frame[pair]
        elif pair in norm_xray:
            normalized_cost = norm_xray[pair]
        else:
            normalized_cost = 0.5
        low_impact_priority = 0.55 * (1.0 - normalized_cost) + 0.45 * rank_tail_priority
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "pair_index": pair,
                "first_frame": pair * 2,
                "last_frame": pair * 2 + 1,
                "dqs1_rank": rank,
                "rank_tail_priority": rank_tail_priority,
                "normalized_scorer_geometry_cost": normalized_cost,
                "low_impact_drop_priority": low_impact_priority,
                "scorer_geometry_status": (
                    "fused_frame_pair_and_component_xray"
                    if len(sources) == 2
                    else (sources[0] if sources else "rank_only")
                ),
                "source_surfaces": sources,
                "frame_pair_cost": frame_costs.get(pair),
                "component_xray_cost": xray_costs.get(pair),
                "axis_mix": _axis_mix(pair_row, xray_row),
                "recommended_receiver_scope": {
                    "drop_probe": "dqs1_pairset_descriptor",
                    "segnet_repair_frame": pair * 2 + 1,
                    "posenet_repair_frames": [pair * 2, pair * 2 + 1],
                    "masked_or_feathered_variant": "blocked_until_receiver_materializer_exists",
                    **FALSE_GEOMETRY_AUTHORITY,
                },
                **FALSE_GEOMETRY_AUTHORITY,
            }
        )
    rows_by_pair = {int(row["pair_index"]): row for row in rows}
    requests = _drop_requests(
        best_order=best_order,
        rows_by_pair=rows_by_pair,
        drop_counts=counts,
        max_requests=int(max_requests),
    )
    geometry_covered = sum(1 for row in rows if row["scorer_geometry_status"] != "rank_only")
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": TOOL,
        "evidence_grade": "planning-only pair-frame scorer geometry",
        "allowed_use": "local_pairset_start_selection_and_repair_budget_planning_only",
        "forbidden_use": "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority",
        "source_pairset_acquisition_summary": {
            "candidate_count": pairset_acquisition.get("summary", {}).get("candidate_count")
            if isinstance(pairset_acquisition.get("summary"), Mapping)
            else None,
            "best_selected_pair_count": len(best_order),
            "best_rank_order_pair_indices": list(best_order),
        },
        "source_surfaces": {
            "frame_pair_curriculum": frame_pair_curriculum is not None,
            "pair_component_xray_count": len(pair_component_xrays),
        },
        "coverage": {
            "best_pair_count": len(best_order),
            "geometry_covered_pair_count": geometry_covered,
            "geometry_coverage": geometry_covered / float(len(best_order)),
            "rank_only_pair_count": len(best_order) - geometry_covered,
        },
        "summary": {
            "row_count": len(rows),
            "queue_executable_request_count": len(requests),
            "drop_counts": counts,
            "top_request_ids": [row["candidate_id"] for row in requests[:8]],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "rows": rows,
        "queue_executable_pairset_drop_requests": requests,
        "blocked_family_requests": [
            {
                "family": "within_selected_set_mask_feather_probe",
                "blocker": "requires receiver/materializer support for non-pair-drop mask semantics",
                **FALSE_GEOMETRY_AUTHORITY,
            },
            {
                "family": "inverse_scorer_null_direction_masked_variant",
                "blocker": "requires inverse-scorer action cell to runtime materializer binding",
                **FALSE_GEOMETRY_AUTHORITY,
            },
        ],
        **FALSE_GEOMETRY_AUTHORITY,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    coverage = payload.get("coverage") if isinstance(payload.get("coverage"), Mapping) else {}
    lines = [
        "# Pair-Frame Scorer Geometry Lattice",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Queue-executable requests: `{summary.get('queue_executable_request_count')}`",
        f"- Geometry coverage: `{coverage.get('geometry_covered_pair_count')}/{coverage.get('best_pair_count')}`",
        f"- Score claim: `{payload.get('score_claim')}`",
        f"- Ready for exact eval dispatch: `{payload.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Executable Drop Requests",
        "",
    ]
    requests = payload.get("queue_executable_pairset_drop_requests")
    if isinstance(requests, list):
        for request in requests[:16]:
            if not isinstance(request, Mapping):
                continue
            lines.append(
                f"- `{request.get('candidate_id')}` drops={request.get('dropped_pair_indices')} "
                f"coverage={request.get('geometry_coverage')}"
            )
    lines.extend(["", "## Lowest Geometry-Cost Pairs", ""])
    rows = payload.get("rows")
    if isinstance(rows, list):
        ranked = sorted(
            (row for row in rows if isinstance(row, Mapping)),
            key=lambda row: -float(row.get("low_impact_drop_priority") or 0.0),
        )
        for row in ranked[:16]:
            lines.append(
                f"- pair `{row.get('pair_index')}` frames=({row.get('first_frame')},{row.get('last_frame')}) "
                f"priority={float(row.get('low_impact_drop_priority') or 0.0):.6g} "
                f"status=`{row.get('scorer_geometry_status')}`"
            )
    lines.extend(
        [
            "",
            "Authority: planning/local queue starts only; no score, promotion, rank/kill, or dispatch authority.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "FALSE_GEOMETRY_AUTHORITY",
    "REQUEST_SCHEMA",
    "ROW_SCHEMA",
    "SCHEMA",
    "TOOL",
    "PairFrameScorerGeometryLatticeError",
    "build_pair_frame_scorer_geometry_lattice",
    "dumps_json",
    "load_json_object",
    "render_markdown",
    "write_json",
]
