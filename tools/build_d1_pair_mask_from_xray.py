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

    signs: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    selected_pose_sum = 0.0
    selected_seg_sum = 0.0
    guard = float(improvement_guard)
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
            signs.append(sign)
            selected_pose, selected_seg = _pose_seg(row)
            selected_rows.append(
                {
                    "pair_idx": pair_idx,
                    "sign": sign,
                    "baseline_objective": base_objective,
                    "selected_objective": chosen_objective,
                    "objective_improvement": improvement,
                    "selected_pose_dist": selected_pose,
                    "selected_seg_dist": selected_seg,
                }
            )
            selected_pose_sum += selected_pose
            selected_seg_sum += selected_seg
        else:
            signs.append(0)
            selected_pose_sum += base_pose
            selected_seg_sum += base_seg
    measured_pairs = len(signs)
    if output_n_pairs is not None:
        if output_n_pairs < measured_pairs:
            raise ValueError(
                f"output_n_pairs={output_n_pairs} is smaller than measured "
                f"pair count {measured_pairs}"
            )
        signs.extend([0] * (int(output_n_pairs) - measured_pairs))

    selected_mean_pose = selected_pose_sum / measured_pairs
    selected_mean_seg = selected_seg_sum / measured_pairs
    baseline_component = _component_from_means(baseline_mean_pose, baseline_mean_seg)
    selected_component = _component_from_means(selected_mean_pose, selected_mean_seg)
    return {
        "schema": "d1_pair_sign_mask_from_xray_v1",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objective": "contest_score_linearized_at_baseline_mean_pose_v1",
        "pair_signs": signs,
        "n_pairs": len(signs),
        "measured_pairs": measured_pairs,
        "active_pairs": sum(1 for value in signs if value != 0),
        "positive_pairs": sum(1 for value in signs if value > 0),
        "negative_pairs": sum(1 for value in signs if value < 0),
        "improvement_guard": guard,
        "baseline_mean_pose_dist": baseline_mean_pose,
        "baseline_mean_seg_dist": baseline_mean_seg,
        "selected_mean_pose_dist": selected_mean_pose,
        "selected_mean_seg_dist": selected_mean_seg,
        "pose_weight": pose_weight,
        "seg_weight": seg_weight,
        "predicted_component_no_rate_baseline": baseline_component,
        "predicted_component_no_rate_selected": selected_component,
        "predicted_component_no_rate_delta": selected_component - baseline_component,
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
                "improvement_guard": result["improvement_guard"],
                "predicted_component_no_rate_delta": result[
                    "predicted_component_no_rate_delta"
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
