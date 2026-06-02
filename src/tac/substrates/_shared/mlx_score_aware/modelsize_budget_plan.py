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
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

MODEL_SIZE_BUDGET_PLAN_SCHEMA = "compact_carrier_modelsize_budget_plan.v1"
CONTEST_BYTE_PRICE_SCORE = 25.0 / float(ORIGINAL_VIDEO_BYTES)


class ModelSizeBudgetPlanError(ValueError):
    """Raised when a model-size ladder cannot be interpreted."""


@dataclass(frozen=True)
class ModelSizeBudgetPoint:
    row_id: str
    archive_bytes: int
    nonrate_score: float
    source: dict[str, Any]

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "archive_bytes": int(self.archive_bytes),
            "nonrate_score": float(self.nonrate_score),
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
    if len(points) < 2:
        return {
            "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
            "carrier_id": str(carrier_id),
            "baseline_id": str(baseline_id),
            "status": "insufficient_modelsize_ladder",
            "contest_byte_price_score": CONTEST_BYTE_PRICE_SCORE,
            "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
            "measured_points": [point.as_jsonable() for point in points],
            "marginal_steps": [],
            "selected_point": points[0].as_jsonable() if points else None,
            "selected_archive_bytes": int(points[0].archive_bytes) if points else None,
            "recommended_next_actions": [
                "run_measured_modelsize_ladder_before_long_budget_spend",
                "include_archive_bytes_and_nonrate_score_for_each_size_point",
            ],
            "blockers": ["modelsize_budget_ladder_has_fewer_than_two_points"],
            **FALSE_AUTHORITY,
        }

    marginal_steps = _marginal_steps(points)
    selected = min(
        points,
        key=lambda point: point.nonrate_score
        + point.archive_bytes * CONTEST_BYTE_PRICE_SCORE,
    )
    last_value_step = next(
        (
            step
            for step in reversed(marginal_steps)
            if step["spend_rule"] == "spend_modelsize_byte"
        ),
        None,
    )
    return {
        "schema": MODEL_SIZE_BUDGET_PLAN_SCHEMA,
        "carrier_id": str(carrier_id),
        "baseline_id": str(baseline_id),
        "status": "measured_modelsize_budget_selected",
        "contest_byte_price_score": CONTEST_BYTE_PRICE_SCORE,
        "original_video_bytes": int(ORIGINAL_VIDEO_BYTES),
        "selection_rule": (
            "minimize measured nonrate_score + archive_bytes * (25 / original_video_bytes); "
            "spend a size step only when marginal nonrate improvement per byte exceeds byte price"
        ),
        "measured_points": [point.as_jsonable() for point in points],
        "marginal_steps": marginal_steps,
        "selected_point": selected.as_jsonable(),
        "selected_archive_bytes": int(selected.archive_bytes),
        "last_marginally_worthwhile_step": last_value_step,
        "recommended_next_actions": _recommended_next_actions(points, selected, marginal_steps),
        "blockers": ["modelsize_budget_plan_is_false_authority"],
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
        archive_bytes = _extract_archive_bytes(row)
        nonrate_score = _extract_nonrate_score(row, archive_bytes=archive_bytes)
        row_id = str(row.get("row_id") or row.get("id") or f"modelsize_{index}")
        points.append(
            ModelSizeBudgetPoint(
                row_id=row_id,
                archive_bytes=archive_bytes,
                nonrate_score=nonrate_score,
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


def _extract_archive_bytes(row: Mapping[str, Any]) -> int:
    for key in (
        "archive_bytes",
        "archive_zip_bytes",
        "candidate_archive_bytes",
        "projected_archive_bytes_600pair",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            out = int(value)
        except (TypeError, ValueError):
            continue
        if out > 0:
            return out
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
) -> list[str]:
    actions = [
        "train_selected_budget_with_score_aware_decoder_weight_objective",
        "preserve_smaller_and_larger_budget_rows_as_rd_curve_evidence",
    ]
    if selected.archive_bytes == points[0].archive_bytes:
        actions.append("larger_modelsize_steps_do_not_pay_measured_byte_price")
    if selected.archive_bytes == points[-1].archive_bytes and all(
        step.get("spend_rule") == "spend_modelsize_byte" for step in marginal_steps
    ):
        actions.append("extend_budget_ladder_until_marginal_step_stops_paying")
    return actions


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
