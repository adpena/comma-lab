# SPDX-License-Identifier: MIT
"""Model-size budget planner for compact NeRV-family carriers.

This is a false-authority planning primitive. It turns a measured ladder of
archive-byte / non-rate-score rows into the discrete waterfilling decision the
contest score implies: spend the next model-size byte only when its measured
non-rate-score improvement beats the fixed byte price.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from typing import Any

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score

MODEL_SIZE_BUDGET_PLAN_SCHEMA = "compact_carrier_modelsize_budget_plan.v2"
CONTEST_BYTE_PRICE_SCORE = 25.0 / float(ORIGINAL_VIDEO_BYTES)
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

_RECEIVER_CLOSED_PROOF_KEYS = (
    "receiver_closed",
    "receiver_proof_passed",
    "receiver_archive_replay_verified",
    "receiver_contract_satisfied",
    "byte_closed_receiver_proof",
)
_RECEIVER_PROOF_PATH_KEYS = (
    "receiver_proof_path",
    "receiver_proof_report_path",
    "receiver_closed_proof_path",
)
_RECEIVER_PROOF_SHA_KEYS = (
    "receiver_proof_sha256",
    "receiver_proof_report_sha256",
    "receiver_closed_proof_sha256",
)
_ARCHIVE_SHA_KEYS = (
    "archive_sha256",
    "candidate_archive_sha256",
    "receiver_archive_sha256",
    "source_archive_sha256",
    "archive_zip_sha256",
)
_AXIS_TAG_KEYS = (
    "axis_tag",
    "score_axis_tag",
    "measured_score_axis_tag",
    "receiver_proof_axis_tag",
)
_SAMPLE_COUNT_KEYS = (
    "sample_pair_count",
    "sample_pairs",
    "n_pairs",
    "num_pairs",
    "pair_count",
)
_MEASURED_ARCHIVE_BYTE_KEYS = (
    "measured_archive_bytes",
    "archive_bytes",
    "archive_zip_bytes",
    "candidate_archive_bytes",
)
_PROJECTED_ARCHIVE_BYTE_KEYS = ("projected_archive_bytes_600pair",)
_LOWER_BOUND_MARKER_KEYS = (
    "lower_bound_only",
    "fit_is_lower_bound_only",
    "modelsize_curve_is_ideal_packed_lower_bound",
)
_SOURCE_BOUND_CAPACITY_PATHS = (
    ("modelsize_mparams",),
    ("fc_dim",),
    ("official_controls", "--modelsize"),
    ("official_controls", "fc_dim"),
    ("solved_budget", "modelsize_mparams"),
    ("solved_budget", "fc_dim"),
    ("solved_budget", "official_controls", "--modelsize"),
    ("solved_budget", "official_controls", "fc_dim"),
)


class ModelSizeBudgetPlanError(ValueError):
    """Raised when a model-size ladder cannot be interpreted."""


@dataclass(frozen=True)
class ModelSizeBudgetPoint:
    row_id: str
    archive_bytes: int
    nonrate_score: float
    archive_bytes_key: str
    evidence_kind: str
    receiver_closed_bytes: bool
    source_bound_capacity_control: bool
    evidence_blockers: tuple[str, ...]
    source: dict[str, Any]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "archive_bytes": int(self.archive_bytes),
            "archive_bytes_key": self.archive_bytes_key,
            "nonrate_score": float(self.nonrate_score),
            "evidence_kind": self.evidence_kind,
            "receiver_closed_bytes": bool(self.receiver_closed_bytes),
            "source_bound_capacity_control": bool(
                self.source_bound_capacity_control
            ),
            "evidence_blockers": list(self.evidence_blockers),
            "rate_score_at_contest_price": float(
                self.archive_bytes * CONTEST_BYTE_PRICE_SCORE
            ),
            "total_score_at_contest_price": float(
                self.nonrate_score + self.archive_bytes * CONTEST_BYTE_PRICE_SCORE
            ),
            "source": self.source,
        }


def build_modelsize_budget_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    carrier_id: str = "unknown",
    baseline_id: str = "pr95_hnerv",
) -> dict[str, Any]:
    """Choose a measured model-size budget from contest byte-price math."""

    points = _parse_points(rows)
    receiver_closed_points = [
        point for point in points if point.receiver_closed_bytes
    ]
    decision_points = (
        receiver_closed_points if len(receiver_closed_points) >= 2 else points
    )
    if len(points) < 2:
        return {
            "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
            "carrier_id": str(carrier_id),
            "baseline_id": str(baseline_id),
            "status": "insufficient_modelsize_ladder",
            "decision_basis": "all_rows",
            "contest_byte_price_score": CONTEST_BYTE_PRICE_SCORE,
            "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
            "points": [point.as_jsonable() for point in points],
            "measured_points": [
                point.as_jsonable() for point in receiver_closed_points
            ],
            "receiver_closed_points": [
                point.as_jsonable() for point in receiver_closed_points
            ],
            "point_count_by_evidence": _point_count_by_evidence(points),
            "marginal_steps": [],
            "selected_point": points[0].as_jsonable() if points else None,
            "selected_archive_bytes": int(points[0].archive_bytes) if points else None,
            "receiver_closed_selected_point": None,
            "receiver_closed_selected_archive_bytes": None,
            "recommended_next_actions": [
                "run_receiver_closed_modelsize_ladder_before_long_budget_spend",
                "include_archive_bytes_and_nonrate_score_for_each_size_point",
            ],
            "blockers": _ordered_unique(
                [
                    "modelsize_budget_ladder_has_fewer_than_two_points",
                    "receiver_closed_modelsize_ladder_has_fewer_than_two_points",
                    *_point_evidence_blockers(points),
                ]
            ),
            **FALSE_AUTHORITY,
        }

    marginal_steps = _marginal_steps(decision_points)
    selected = min(
        decision_points,
        key=lambda point: point.nonrate_score
        + point.archive_bytes * CONTEST_BYTE_PRICE_SCORE,
    )
    receiver_closed_selected = (
        selected if len(receiver_closed_points) >= 2 else None
    )
    last_value_step = next(
        (
            step
            for step in reversed(marginal_steps)
            if step["spend_rule"] == "spend_modelsize_byte"
        ),
        None,
    )
    receiver_closed_ladder_ready = len(receiver_closed_points) >= 2
    status = (
        "receiver_closed_modelsize_budget_selected"
        if receiver_closed_ladder_ready
        else "advisory_or_projected_modelsize_budget_selected"
    )
    decision_basis = (
        "receiver_closed_rows"
        if receiver_closed_ladder_ready
        else "all_rows_advisory_planning_only"
    )
    return {
        "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
        "carrier_id": str(carrier_id),
        "baseline_id": str(baseline_id),
        "status": status,
        "decision_basis": decision_basis,
        "contest_byte_price_score": CONTEST_BYTE_PRICE_SCORE,
        "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
        "selection_rule": (
            "minimize nonrate_score + archive_bytes * (25 / original_video_bytes); "
            "use receiver-closed rows when at least two are available, otherwise "
            "emit advisory planning only; spend a size step only when marginal "
            "nonrate improvement per byte exceeds byte price"
        ),
        "points": [point.as_jsonable() for point in points],
        "measured_points": [
            point.as_jsonable() for point in receiver_closed_points
        ],
        "receiver_closed_points": [
            point.as_jsonable() for point in receiver_closed_points
        ],
        "point_count_by_evidence": _point_count_by_evidence(points),
        "marginal_steps": marginal_steps,
        "selected_point": selected.as_jsonable(),
        "selected_archive_bytes": int(selected.archive_bytes),
        "receiver_closed_selected_point": (
            receiver_closed_selected.as_jsonable()
            if receiver_closed_selected is not None
            else None
        ),
        "receiver_closed_selected_archive_bytes": (
            int(receiver_closed_selected.archive_bytes)
            if receiver_closed_selected is not None
            else None
        ),
        "last_marginally_worthwhile_step": last_value_step,
        "recommended_next_actions": _recommended_next_actions(
            decision_points,
            selected,
            marginal_steps,
            receiver_closed_ladder_ready=receiver_closed_ladder_ready,
        ),
        "blockers": _ordered_unique(
            [
                "modelsize_budget_plan_is_false_authority",
                *(
                    []
                    if receiver_closed_ladder_ready
                    else [
                        "receiver_closed_modelsize_ladder_has_fewer_than_two_points",
                        "modelsize_budget_selection_is_advisory_or_projected",
                    ]
                ),
                *_point_evidence_blockers(points),
            ]
        ),
        **FALSE_AUTHORITY,
    }


def build_modelsize_budget_plan_from_iterable(
    rows: Iterable[Mapping[str, Any]],
    *,
    carrier_id: str = "unknown",
    baseline_id: str = "pr95_hnerv",
) -> dict[str, Any]:
    """Choose a measured model-size budget from an arbitrary row iterable."""

    return build_modelsize_budget_plan(
        list(rows),
        carrier_id=carrier_id,
        baseline_id=baseline_id,
    )


def _parse_points(rows: Sequence[Mapping[str, Any]]) -> list[ModelSizeBudgetPoint]:
    points = []
    for index, row in enumerate(rows):
        archive_bytes, archive_bytes_key = _extract_archive_bytes(row)
        nonrate_score = _extract_nonrate_score(row, archive_bytes=archive_bytes)
        row_id = str(row.get("row_id") or row.get("id") or f"modelsize_{index}")
        (
            evidence_kind,
            receiver_closed_bytes,
            source_bound_capacity_control,
            evidence_blockers,
        ) = _classify_point_evidence(row, archive_bytes_key=archive_bytes_key)
        points.append(
            ModelSizeBudgetPoint(
                row_id=row_id,
                archive_bytes=archive_bytes,
                nonrate_score=nonrate_score,
                archive_bytes_key=archive_bytes_key,
                evidence_kind=evidence_kind,
                receiver_closed_bytes=receiver_closed_bytes,
                source_bound_capacity_control=source_bound_capacity_control,
                evidence_blockers=tuple(evidence_blockers),
                source=dict(row),
            )
        )
    points.sort(key=lambda point: (point.archive_bytes, point.nonrate_score))
    deduped = []
    seen: set[int] = set()
    for point in points:
        if point.archive_bytes in seen:
            continue
        seen.add(point.archive_bytes)
        deduped.append(point)
    return deduped


def _extract_archive_bytes(row: Mapping[str, Any]) -> tuple[int, str]:
    for key in (*_MEASURED_ARCHIVE_BYTE_KEYS, *_PROJECTED_ARCHIVE_BYTE_KEYS):
        value = row.get(key)
        if value is None:
            continue
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out, key
    raise ModelSizeBudgetPlanError(f"modelsize row missing positive archive bytes: {row}")


def _extract_nonrate_score(row: Mapping[str, Any], *, archive_bytes: int) -> float:
    for key in ("nonrate_score", "nonrate_score_value", "nonrate_score_advisory"):
        value = _finite_float_or_none(row.get(key))
        if value is not None:
            return value
    seg = _finite_float_or_none(row.get("avg_segnet_dist", row.get("d_seg")))
    pose = _finite_float_or_none(row.get("avg_posenet_dist", row.get("d_pose")))
    if seg is not None and pose is not None:
        return float(contest_formula_score(seg_dist=seg, pose_dist=pose, archive_bytes=0))
    for key in ("canonical_score", "advisory_score", "score"):
        value = _finite_float_or_none(row.get(key))
        if value is not None:
            return float(value - archive_bytes * CONTEST_BYTE_PRICE_SCORE)
    raise ModelSizeBudgetPlanError(f"modelsize row missing nonrate score: {row}")


def _marginal_steps(points: Sequence[ModelSizeBudgetPoint]) -> list[dict[str, Any]]:
    steps = []
    for low, high in pairwise(points):
        bytes_added = int(high.archive_bytes - low.archive_bytes)
        nonrate_improvement = float(low.nonrate_score - high.nonrate_score)
        improvement_per_byte = (
            nonrate_improvement / bytes_added if bytes_added > 0 else float("-inf")
        )
        spend = improvement_per_byte > CONTEST_BYTE_PRICE_SCORE
        steps.append(
            {
                "from_row_id": low.row_id,
                "to_row_id": high.row_id,
                "from_archive_bytes": int(low.archive_bytes),
                "to_archive_bytes": int(high.archive_bytes),
                "bytes_added": bytes_added,
                "nonrate_score_improvement": nonrate_improvement,
                "marginal_improvement_per_byte": improvement_per_byte,
                "contest_byte_price_score": CONTEST_BYTE_PRICE_SCORE,
                "net_score_delta_if_spent": float(
                    high.nonrate_score
                    + high.archive_bytes * CONTEST_BYTE_PRICE_SCORE
                    - (
                        low.nonrate_score
                        + low.archive_bytes * CONTEST_BYTE_PRICE_SCORE
                    )
                ),
                "spend_rule": (
                    "spend_modelsize_byte"
                    if spend
                    else "stop_or_reallocate_modelsize_byte"
                ),
            }
        )
    return steps


def _recommended_next_actions(
    points: Sequence[ModelSizeBudgetPoint],
    selected: ModelSizeBudgetPoint,
    marginal_steps: Sequence[Mapping[str, Any]],
    *,
    receiver_closed_ladder_ready: bool,
) -> list[str]:
    actions = [
        "train_selected_budget_with_score_aware_decoder_weight_objective",
        "preserve_smaller_and_larger_budget_rows_as_rd_curve_evidence",
    ]
    if any(not point.source_bound_capacity_control for point in points):
        actions.insert(0, "emit_source_bound_modelsize_mparams_or_fc_dim_for_budget_points")
    if not receiver_closed_ladder_ready:
        actions.insert(0, "replace_projected_rows_with_receiver_closed_archive_ladder")
        actions.insert(1, "run_receiver_inflate_proof_for_each_modelsize_point")
    if selected.archive_bytes == points[0].archive_bytes:
        actions.append("larger_modelsize_steps_do_not_pay_measured_byte_price")
    if selected.archive_bytes == points[-1].archive_bytes and all(
        step.get("spend_rule") == "spend_modelsize_byte" for step in marginal_steps
    ):
        actions.append("extend_budget_ladder_until_marginal_step_stops_paying")
    return actions


def _classify_point_evidence(
    row: Mapping[str, Any],
    *,
    archive_bytes_key: str,
) -> tuple[str, bool, bool, list[str]]:
    blockers: list[str] = []
    proof_flag_present = any(
        _truthy(row.get(key)) for key in _RECEIVER_CLOSED_PROOF_KEYS
    )
    proof_identity_blockers = _receiver_closed_identity_blockers(row)
    proof_present = proof_flag_present and not proof_identity_blockers
    lower_bound = any(_truthy(row.get(key)) for key in _LOWER_BOUND_MARKER_KEYS)
    projected = archive_bytes_key in _PROJECTED_ARCHIVE_BYTE_KEYS or lower_bound
    source_bound_capacity = _has_source_bound_capacity_control(row)

    if projected:
        blockers.append("projected_or_lower_bound_archive_bytes_not_receiver_closed")
    if not proof_flag_present:
        blockers.append("receiver_closed_byte_proof_missing")
    blockers.extend(proof_identity_blockers)
    if archive_bytes_key not in _MEASURED_ARCHIVE_BYTE_KEYS:
        blockers.append("measured_archive_byte_field_missing")
    if not source_bound_capacity:
        blockers.append("source_bound_modelsize_or_fc_dim_missing")

    receiver_closed = (
        proof_present
        and not projected
        and archive_bytes_key in _MEASURED_ARCHIVE_BYTE_KEYS
    )
    if receiver_closed:
        return "receiver_closed_measured_bytes", True, source_bound_capacity, blockers
    if projected:
        return "projected_or_lower_bound_bytes", False, source_bound_capacity, blockers
    return (
        "advisory_measured_bytes_without_receiver_proof",
        False,
        source_bound_capacity,
        blockers,
    )


def _receiver_closed_identity_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _first_present_str(row, _RECEIVER_PROOF_PATH_KEYS):
        blockers.append("receiver_proof_path_missing")
    proof_sha = _first_present_str(row, _RECEIVER_PROOF_SHA_KEYS)
    if not _is_sha256_hex(proof_sha):
        blockers.append("receiver_proof_sha256_missing_or_invalid")
    archive_sha = _first_present_str(row, _ARCHIVE_SHA_KEYS)
    if not _is_sha256_hex(archive_sha):
        blockers.append("archive_sha256_missing_or_invalid")
    if not _first_present_str(row, _AXIS_TAG_KEYS):
        blockers.append("receiver_proof_axis_tag_missing")
    sample_count = _first_present_int(row, _SAMPLE_COUNT_KEYS)
    full_video = bool(
        _truthy(row.get("full_video_coverage"))
        or _truthy(row.get("full600_coverage"))
        or _truthy(row.get("full_sample_coverage"))
    )
    if sample_count is None and not full_video:
        blockers.append("receiver_proof_full_sample_count_missing")
    elif sample_count is not None and sample_count < 600 and not full_video:
        blockers.append("receiver_proof_full_sample_count_incomplete")
    return blockers


def _first_present_str(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_present_int(row: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_sha256_hex(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _has_source_bound_capacity_control(row: Mapping[str, Any]) -> bool:
    return any(
        _lookup_path(row, path) is not None
        for path in _SOURCE_BOUND_CAPACITY_PATHS
    )


def _lookup_path(row: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _point_evidence_blockers(points: Sequence[ModelSizeBudgetPoint]) -> list[str]:
    return _ordered_unique(
        blocker
        for point in points
        for blocker in point.evidence_blockers
    )


def _point_count_by_evidence(
    points: Sequence[ModelSizeBudgetPoint],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for point in points:
        counts[point.evidence_kind] = counts.get(point.evidence_kind, 0) + 1
    return counts


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(out):
        return None
    return out


__all__ = [
    "CONTEST_BYTE_PRICE_SCORE",
    "MODEL_SIZE_BUDGET_PLAN_SCHEMA",
    "ModelSizeBudgetPlanError",
    "build_modelsize_budget_plan",
    "build_modelsize_budget_plan_from_iterable",
]
