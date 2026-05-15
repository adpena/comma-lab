#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a D1 per-pair overlay sign mask from pair-component xray reports.

This is a diagnostic-to-packet bridge. It does not score, dispatch, or promote
anything. It consumes baseline and D1 overlay per-pair component measurements
from ``tools/xray_pair_component_errors.py`` and emits a compact selector input
for ``tools/build_d1_overlay_policy_candidates.py --sign-policies pair_mask``.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CONTEST_ARCHIVE_DENOMINATOR_BYTES = 37_545_489


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rows_by_pair(path: Path) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain non-empty rows[]")
    out: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} rows[{idx}] is not an object")
        pair_idx = row.get("pair_idx", row.get("pair_index"))
        if not isinstance(pair_idx, int):
            raise ValueError(f"{path} rows[{idx}] missing integer pair_idx")
        if pair_idx in out:
            raise ValueError(f"{path} duplicate pair_idx={pair_idx}")
        out[pair_idx] = row
    return out


def _pose_seg(row: dict[str, Any]) -> tuple[float, float]:
    return float(row["pose_dist"]), float(row["seg_dist"])


def _component_from_means(mean_pose: float, mean_seg: float) -> float:
    return math.sqrt(10.0 * max(0.0, mean_pose)) + 100.0 * mean_seg


def _rate_penalty_from_bytes(rate_cost_bytes: int) -> float:
    if rate_cost_bytes < 0:
        raise ValueError(f"rate_cost_bytes must be >= 0; got {rate_cost_bytes}")
    return 25.0 * float(rate_cost_bytes) / float(_CONTEST_ARCHIVE_DENOMINATOR_BYTES)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-xray", type=Path, required=True)
    parser.add_argument(
        "--positive-xray",
        type=Path,
        required=True,
        help="Pair xray for the +payload D1 overlay candidate.",
    )
    parser.add_argument(
        "--negative-xray",
        type=Path,
        help="Optional pair xray for the negated D1 overlay candidate.",
    )
    parser.add_argument(
        "--improvement-guard",
        type=float,
        default=0.0,
        help="Minimum per-pair component-score improvement required to enable a pair.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("waterfill_prefix", "independent"),
        default="waterfill_prefix",
        help=(
            "waterfill_prefix sorts pair actions by linearized improvement and "
            "chooses the prefix that minimizes the actual global component score; "
            "independent keeps every pair above --improvement-guard."
        ),
    )
    parser.add_argument(
        "--rate-cost-bytes",
        type=int,
        default=0,
        help=(
            "Fixed archive-byte overhead for carrying the pair mask. The selector "
            "converts it through the contest rate term and emits an all-zero mask "
            "when no nonzero prefix pays for the bytes."
        ),
    )
    parser.add_argument(
        "--max-active-pairs",
        type=int,
        help="Optional cap on active pairs after sorting by improvement.",
    )
    parser.add_argument(
        "--min-net-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum positive score improvement required after rate penalty. "
            "The default accepts any strictly score-lowering nonzero mask."
        ),
    )
    parser.add_argument(
        "--output-n-pairs",
        type=int,
        help=(
            "Optional final mask length. When larger than the xray pair count, "
            "pads unmeasured pairs as disabled zeros."
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def build_pair_sign_mask(
    *,
    baseline_rows: dict[int, dict[str, Any]],
    positive_rows: dict[int, dict[str, Any]],
    negative_rows: dict[int, dict[str, Any]] | None = None,
    improvement_guard: float = 0.0,
    selection_mode: str = "waterfill_prefix",
    rate_cost_bytes: int = 0,
    max_active_pairs: int | None = None,
    min_net_improvement: float = 0.0,
    output_n_pairs: int | None = None,
) -> dict[str, Any]:
    pair_indices = sorted(baseline_rows)
    if pair_indices != list(range(len(pair_indices))):
        raise ValueError("baseline pair rows must be contiguous and zero-based")
    if sorted(positive_rows) != pair_indices:
        raise ValueError("positive xray pair rows do not match baseline")
    if negative_rows is not None and sorted(negative_rows) != pair_indices:
        raise ValueError("negative xray pair rows do not match baseline")

    n_measured = len(pair_indices)
    baseline_mean_pose = (
        sum(_pose_seg(baseline_rows[pair_idx])[0] for pair_idx in pair_indices)
        / n_measured
    )
    baseline_mean_seg = (
        sum(_pose_seg(baseline_rows[pair_idx])[1] for pair_idx in pair_indices)
        / n_measured
    )
    if baseline_mean_pose <= 0.0:
        raise ValueError("baseline mean pose distance must be > 0")
    pose_weight = 5.0 / math.sqrt(10.0 * baseline_mean_pose)
    seg_weight = 100.0
    baseline_pose_sum = sum(
        _pose_seg(baseline_rows[pair_idx])[0] for pair_idx in pair_indices
    )
    baseline_seg_sum = sum(
        _pose_seg(baseline_rows[pair_idx])[1] for pair_idx in pair_indices
    )
    baseline_component = _component_from_means(baseline_mean_pose, baseline_mean_seg)
    rate_penalty_score = _rate_penalty_from_bytes(int(rate_cost_bytes))
    if max_active_pairs is not None and max_active_pairs < 0:
        raise ValueError(f"max_active_pairs must be >= 0; got {max_active_pairs}")
    if min_net_improvement < 0:
        raise ValueError(
            f"min_net_improvement must be >= 0; got {min_net_improvement}"
        )

    guard = float(improvement_guard)
    potential_rows: list[dict[str, Any]] = []
    for pair_idx in pair_indices:
        base_pose, base_seg = _pose_seg(baseline_rows[pair_idx])
        base_objective = pose_weight * base_pose + seg_weight * base_seg
        choices = [(0, base_objective, baseline_rows[pair_idx])]
        pos_pose, pos_seg = _pose_seg(positive_rows[pair_idx])
        choices.append(
            (
                1,
                pose_weight * pos_pose + seg_weight * pos_seg,
                positive_rows[pair_idx],
            )
        )
        if negative_rows is not None:
            neg_pose, neg_seg = _pose_seg(negative_rows[pair_idx])
            choices.append(
                (
                    -1,
                    pose_weight * neg_pose + seg_weight * neg_seg,
                    negative_rows[pair_idx],
                )
            )
        sign, chosen_objective, row = min(
            choices, key=lambda item: (item[1], abs(item[0]))
        )
        improvement = base_objective - chosen_objective
        if sign != 0 and improvement > guard:
            selected_pose, selected_seg = _pose_seg(row)
            potential_rows.append(
                {
                    "pair_idx": pair_idx,
                    "sign": sign,
                    "baseline_objective": base_objective,
                    "selected_objective": chosen_objective,
                    "linearized_objective_improvement": improvement,
                    "selected_pose_dist": selected_pose,
                    "selected_seg_dist": selected_seg,
                    "pose_delta": selected_pose - base_pose,
                    "seg_delta": selected_seg - base_seg,
                }
            )

    mode = str(selection_mode)
    if mode not in {"waterfill_prefix", "independent"}:
        raise ValueError(
            f"selection_mode={selection_mode!r} must be waterfill_prefix or independent"
        )
    if max_active_pairs is not None:
        potential_rows = sorted(
            potential_rows,
            key=lambda row: (
                -float(row["linearized_objective_improvement"]),
                int(row["pair_idx"]),
            ),
        )[: int(max_active_pairs)]

    selected_by_pair: dict[int, dict[str, Any]] = {}
    selected_component = baseline_component
    selected_mean_pose = baseline_mean_pose
    selected_mean_seg = baseline_mean_seg
    best_prefix_size = 0
    best_component_prefix_size = 0
    best_component_no_rate_delta = 0.0
    best_net_delta = 0.0
    if mode == "independent":
        selected_by_pair = {int(row["pair_idx"]): row for row in potential_rows}
        selected_pose_sum = baseline_pose_sum + sum(
            float(row["pose_delta"]) for row in selected_by_pair.values()
        )
        selected_seg_sum = baseline_seg_sum + sum(
            float(row["seg_delta"]) for row in selected_by_pair.values()
        )
        selected_mean_pose = selected_pose_sum / n_measured
        selected_mean_seg = selected_seg_sum / n_measured
        selected_component = _component_from_means(selected_mean_pose, selected_mean_seg)
        best_component_no_rate_delta = selected_component - baseline_component
        best_component_prefix_size = len(selected_by_pair)
        best_net_delta = best_component_no_rate_delta + (
            rate_penalty_score if selected_by_pair else 0.0
        )
        if best_net_delta >= -float(min_net_improvement):
            selected_by_pair = {}
            selected_component = baseline_component
            selected_mean_pose = baseline_mean_pose
            selected_mean_seg = baseline_mean_seg
            best_net_delta = 0.0
    else:
        ranked = sorted(
            potential_rows,
            key=lambda row: (
                -float(row["linearized_objective_improvement"]),
                int(row["pair_idx"]),
            ),
        )
        pose_sum = baseline_pose_sum
        seg_sum = baseline_seg_sum
        best_rows: list[dict[str, Any]] = []
        best_component = baseline_component
        best_pose_mean = baseline_mean_pose
        best_seg_mean = baseline_mean_seg
        best_delta = 0.0
        for prefix_size, row in enumerate(ranked, start=1):
            pose_sum += float(row["pose_delta"])
            seg_sum += float(row["seg_delta"])
            pose_mean = pose_sum / n_measured
            seg_mean = seg_sum / n_measured
            component = _component_from_means(pose_mean, seg_mean)
            component_only_delta = component - baseline_component
            net_delta = component_only_delta + rate_penalty_score
            if component_only_delta < best_component_no_rate_delta:
                best_component_no_rate_delta = component_only_delta
                best_component_prefix_size = prefix_size
            if net_delta < best_delta:
                best_delta = net_delta
                best_rows = ranked[:prefix_size]
                best_component = component
                best_pose_mean = pose_mean
                best_seg_mean = seg_mean
                best_prefix_size = prefix_size
        if best_delta < -float(min_net_improvement):
            selected_by_pair = {int(row["pair_idx"]): row for row in best_rows}
            selected_component = best_component
            selected_mean_pose = best_pose_mean
            selected_mean_seg = best_seg_mean
            best_net_delta = best_delta

    signs: list[int] = [0] * n_measured
    selected_rows: list[dict[str, Any]] = []
    ranked_selected = sorted(
        selected_by_pair.values(),
        key=lambda row: (
            -float(row["linearized_objective_improvement"]),
            int(row["pair_idx"]),
        ),
    )
    rank_by_pair = {
        int(row["pair_idx"]): rank for rank, row in enumerate(ranked_selected, start=1)
    }
    for pair_idx, row in selected_by_pair.items():
        signs[pair_idx] = int(row["sign"])
    for pair_idx in sorted(selected_by_pair):
        row = dict(selected_by_pair[pair_idx])
        row["selection_rank"] = rank_by_pair[pair_idx]
        selected_rows.append(row)
    measured_pairs = len(signs)
    if output_n_pairs is not None:
        if output_n_pairs < measured_pairs:
            raise ValueError(
                f"output_n_pairs={output_n_pairs} is smaller than measured "
                f"pair count {measured_pairs}"
            )
        signs.extend([0] * (int(output_n_pairs) - measured_pairs))

    active_pairs = sum(1 for value in signs if value != 0)
    component_delta = selected_component - baseline_component
    total_delta_with_rate = component_delta + (
        rate_penalty_score if active_pairs else 0.0
    )
    return {
        "schema": "d1_pair_sign_mask_from_xray_v1",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objective": "contest_score_linearized_at_baseline_mean_pose_v1",
        "selection_mode": mode,
        "pair_signs": signs,
        "n_pairs": len(signs),
        "measured_pairs": measured_pairs,
        "active_pairs": active_pairs,
        "positive_pairs": sum(1 for value in signs if value > 0),
        "negative_pairs": sum(1 for value in signs if value < 0),
        "potential_pairs": len(potential_rows),
        "best_prefix_size": (
            best_prefix_size if mode == "waterfill_prefix" else active_pairs
        ),
        "best_component_prefix_size": (
            best_component_prefix_size
            if mode == "waterfill_prefix"
            else active_pairs
        ),
        "best_component_no_rate_delta": best_component_no_rate_delta,
        "max_active_pairs": max_active_pairs,
        "improvement_guard": guard,
        "rate_cost_bytes": int(rate_cost_bytes),
        "rate_penalty_score": rate_penalty_score,
        "min_net_improvement": float(min_net_improvement),
        "baseline_mean_pose_dist": baseline_mean_pose,
        "baseline_mean_seg_dist": baseline_mean_seg,
        "selected_mean_pose_dist": selected_mean_pose,
        "selected_mean_seg_dist": selected_mean_seg,
        "pose_weight": pose_weight,
        "seg_weight": seg_weight,
        "predicted_component_no_rate_baseline": baseline_component,
        "predicted_component_no_rate_selected": selected_component,
        "predicted_component_no_rate_delta": component_delta,
        "predicted_total_delta_with_rate": total_delta_with_rate,
        "predicted_score_lowering_after_rate": total_delta_with_rate < 0.0,
        "selected_pairs": selected_rows,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_pair_sign_mask(
        baseline_rows=_rows_by_pair(args.baseline_xray),
        positive_rows=_rows_by_pair(args.positive_xray),
        negative_rows=(
            _rows_by_pair(args.negative_xray)
            if args.negative_xray is not None
            else None
        ),
        improvement_guard=float(args.improvement_guard),
        selection_mode=str(args.selection_mode),
        rate_cost_bytes=int(args.rate_cost_bytes),
        max_active_pairs=args.max_active_pairs,
        min_net_improvement=float(args.min_net_improvement),
        output_n_pairs=args.output_n_pairs,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "output_json": str(args.output_json),
                "objective": result["objective"],
                "n_pairs": result["n_pairs"],
                "measured_pairs": result["measured_pairs"],
                "active_pairs": result["active_pairs"],
                "positive_pairs": result["positive_pairs"],
                "negative_pairs": result["negative_pairs"],
                "potential_pairs": result["potential_pairs"],
                "selection_mode": result["selection_mode"],
                "best_prefix_size": result["best_prefix_size"],
                "best_component_prefix_size": result["best_component_prefix_size"],
                "improvement_guard": result["improvement_guard"],
                "rate_cost_bytes": result["rate_cost_bytes"],
                "predicted_component_no_rate_delta": result[
                    "predicted_component_no_rate_delta"
                ],
                "predicted_total_delta_with_rate": result[
                    "predicted_total_delta_with_rate"
                ],
                "predicted_score_lowering_after_rate": result[
                    "predicted_score_lowering_after_rate"
                ],
                "score_claim": False,
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
