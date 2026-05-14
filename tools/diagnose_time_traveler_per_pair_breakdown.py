#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-pair sensitivity diagnostic for the Time-Traveler L5 Autonomy substrate.

PAIR T directive 2026-05-13: this tool ranks the 600 contest pairs by their
predicted contribution to (seg, pose, rate) score components for the
time-traveler substrate. The output identifies high-leverage pairs (those
that move the score most per byte of per-pair side info) so the substrate
trainer / archive packer can allocate the 45 B/pair budget non-uniformly.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: this is a
PLANNING-ONLY tool. Output is tagged ``planning_only=true`` and
``[macOS-CPU advisory]`` per Catalog #192 when the input ranking is derived
from a macOS-CPU smoke eval. The ranker NEVER claims promotability.

Methodology (per the design memo):
  * Pose contribution at PR106 frontier operating point (pose_avg ~ 3.4e-5)
    is 2.71x SegNet's MARGINAL value (CLAUDE.md "SegNet vs PoseNet importance").
  * SegNet marginal weight is constant (d(seg_total)/d(seg_avg) = 100).
  * Rate marginal is constant (d(rate)/d(bytes) = 25 / 37545489).
  * Per-pair sensitivity is therefore proportional to:
      sensitivity(pair_i) = w_seg * |d_seg_i_residual| +
                            w_pose * |d_pose_i_residual| / sqrt(10 * pose_avg) +
                            w_rate * side_info_bytes_i

The tool outputs:
  * Ranked list of pairs by predicted absolute sensitivity.
  * Top-K and bottom-K pairs for non-uniform allocation hints.
  * Aggregate budget plan: total per-pair bytes vs. 45 B/pair budget envelope.

Inputs:
  * --observations: JSON or JSONL with per-pair (d_seg_residual, d_pose_residual,
    side_info_bytes) for some or all 600 pairs. Missing pairs treated as
    average. The harness output schema is documented below.
  * --operating-point-pose-avg: pose_avg at the target operating point
    (default 3.4e-5 per PR106 r2 anchor).
  * --output: where to write the ranked breakdown JSON.

Output schema:
  {
    "lane_id": "lane_time_traveler_smoke_harness_20260513",
    "operating_point_pose_avg": <float>,
    "operating_point_seg_avg": <float>,
    "marginal_weights": {"seg": <float>, "pose": <float>, "rate": <float>},
    "per_pair_ranking": [
      {"pair_index": <int>, "sensitivity": <float>,
       "d_seg_residual": <float>, "d_pose_residual": <float>,
       "side_info_bytes": <int>, "leverage_class": "high"|"med"|"low"},
      ...
    ],
    "top_k_high_leverage": [...],
    "bottom_k_low_leverage": [...],
    "budget_plan": {...},
    "planning_only": true,
    "evidence_grade": "macOS-CPU-advisory" | "synthetic-prior",
    "score_claim": false,
    "promotion_eligible": false,
    "ready_for_exact_eval_dispatch": false
  }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
# PR106 frontier operating point uses pose_avg = 3.4e-5, seg_avg = 6.7e-4.
PR106_R2_POSE_AVG: float = 3.4e-5
PR106_R2_SEG_AVG: float = 6.7e-4

# Per src/tac/score_geometry.py: contest score coefficients.
SEG_COEFFICIENT: float = 100.0
POSE_COEFFICIENT_INSIDE_SQRT: float = 10.0
RATE_COEFFICIENT: float = 25.0
CONTEST_REFERENCE_BYTES: int = 37_545_489

# Per design memo: time-traveler architecture targets 600 pairs at
# 45 B/pair side info budget.
NUM_PAIRS: int = 600
PER_PAIR_SIDE_INFO_TARGET_BYTES: int = 45

LANE_ID: str = "lane_time_traveler_smoke_harness_20260513"


def _marginal_weights(pose_avg: float) -> dict[str, float]:
    """Compute the seg/pose/rate marginal weights at the operating point.

    Returns dict with constant per-unit coefficients suitable for ranking.
    """
    # d(contest_score) / d(seg_avg) = SEG_COEFFICIENT  (constant)
    w_seg = SEG_COEFFICIENT
    # d(contest_score) / d(pose_avg) = 0.5 * sqrt(POSE_COEFFICIENT_INSIDE_SQRT)
    # / sqrt(pose_avg) -- diverges as pose_avg -> 0.
    # We clip to 1e-12 to avoid NaN on operator-overridden pose_avg=0 input.
    safe_pose_avg = max(pose_avg, 1e-12)
    w_pose = 0.5 * math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT / safe_pose_avg)
    # d(contest_score) / d(bytes) = RATE_COEFFICIENT / CONTEST_REFERENCE_BYTES
    w_rate = RATE_COEFFICIENT / CONTEST_REFERENCE_BYTES
    return {"seg": w_seg, "pose": w_pose, "rate": w_rate}


def _load_observations(path: Path) -> list[dict[str, Any]]:
    """Load per-pair observations from JSON or JSONL."""
    if not path.is_file():
        raise FileNotFoundError(f"observations file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, dict):
            raw_rows = payload.get("per_pair") or payload.get("rows") or payload.get("observations")
            if not isinstance(raw_rows, list):
                raise ValueError(
                    f"{path}: dict input must contain per_pair[] / rows[] / observations[]"
                )
            rows = raw_rows
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"{path}: expected list or dict")
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError(f"{path} row {i} is not an object")
        out.append(dict(r))
    return out


def _synthetic_prior_rows(num_pairs: int = NUM_PAIRS) -> list[dict[str, Any]]:
    """Generate a uniform synthetic prior: every pair has identical residuals.

    Used when no real observations are available — the ranker still emits
    a structured table with ``evidence_grade="synthetic-prior"`` so the
    consumer (autopilot / operator) sees that nothing has been measured.
    """
    return [
        {
            "pair_index": i,
            "d_seg_residual": 1.0 / num_pairs,
            "d_pose_residual": 1.0 / num_pairs,
            "side_info_bytes": PER_PAIR_SIDE_INFO_TARGET_BYTES,
        }
        for i in range(num_pairs)
    ]


def _normalize_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Extract canonical fields from a row; raise on malformed input."""
    pair_index = row.get("pair_index", index)
    if not isinstance(pair_index, int) or pair_index < 0:
        raise ValueError(f"row {index}: pair_index must be non-negative int")
    d_seg_residual = row.get("d_seg_residual")
    d_pose_residual = row.get("d_pose_residual")
    side_info_bytes = row.get("side_info_bytes", PER_PAIR_SIDE_INFO_TARGET_BYTES)
    if d_seg_residual is None and d_pose_residual is None:
        raise ValueError(f"row {index}: must supply d_seg_residual or d_pose_residual")
    if d_seg_residual is not None:
        d_seg_residual = float(d_seg_residual)
        if d_seg_residual < 0.0:
            raise ValueError(f"row {index}: d_seg_residual must be non-negative")
    if d_pose_residual is not None:
        d_pose_residual = float(d_pose_residual)
        if d_pose_residual < 0.0:
            raise ValueError(f"row {index}: d_pose_residual must be non-negative")
    if not isinstance(side_info_bytes, int) or side_info_bytes <= 0:
        raise ValueError(f"row {index}: side_info_bytes must be positive int")
    return {
        "pair_index": pair_index,
        "d_seg_residual": d_seg_residual if d_seg_residual is not None else 0.0,
        "d_pose_residual": d_pose_residual if d_pose_residual is not None else 0.0,
        "side_info_bytes": side_info_bytes,
    }


def compute_per_pair_sensitivity(
    rows: list[dict[str, Any]],
    *,
    pose_avg: float,
) -> list[dict[str, Any]]:
    """For each row, compute the predicted sensitivity to total contest score.

    sensitivity = w_seg * d_seg_residual + w_pose * d_pose_residual +
                  w_rate * side_info_bytes
    """
    weights = _marginal_weights(pose_avg)
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(rows):
        r = _normalize_row(raw, index=i)
        seg_contribution = weights["seg"] * r["d_seg_residual"]
        pose_contribution = weights["pose"] * r["d_pose_residual"]
        rate_contribution = weights["rate"] * r["side_info_bytes"]
        sensitivity = seg_contribution + pose_contribution + rate_contribution
        out.append(
            {
                "pair_index": r["pair_index"],
                "d_seg_residual": r["d_seg_residual"],
                "d_pose_residual": r["d_pose_residual"],
                "side_info_bytes": r["side_info_bytes"],
                "seg_contribution": seg_contribution,
                "pose_contribution": pose_contribution,
                "rate_contribution": rate_contribution,
                "sensitivity": sensitivity,
            }
        )
    return out


def _classify_leverage(sorted_rows: list[dict[str, Any]]) -> None:
    """Tag each row with leverage_class (high / med / low) by terciles."""
    n = len(sorted_rows)
    if n == 0:
        return
    # Sorted descending by sensitivity: top tercile = high, middle = med, bottom = low.
    third = max(1, n // 3)
    for i, row in enumerate(sorted_rows):
        if i < third:
            row["leverage_class"] = "high"
        elif i < 2 * third:
            row["leverage_class"] = "med"
        else:
            row["leverage_class"] = "low"


def _budget_plan(
    ranked_rows: list[dict[str, Any]],
    *,
    target_bytes_per_pair: int,
) -> dict[str, Any]:
    """Compute aggregate budget plan + uneven-allocation hint.

    The plan reflects (a) total side-info bytes vs. the target envelope and
    (b) a suggested non-uniform allocation: high-leverage pairs get
    ``int(target_bytes_per_pair * 1.5)`` bytes; low-leverage get ``int(target *
    0.5)``. The hint is PLANNING-ONLY — the actual allocation is the
    substrate trainer's responsibility.
    """
    total_actual = sum(r["side_info_bytes"] for r in ranked_rows)
    envelope = target_bytes_per_pair * len(ranked_rows)
    hint_allocation: list[int] = []
    for r in ranked_rows:
        cls = r.get("leverage_class", "med")
        if cls == "high":
            hint_allocation.append(int(target_bytes_per_pair * 1.5))
        elif cls == "low":
            hint_allocation.append(int(target_bytes_per_pair * 0.5))
        else:
            hint_allocation.append(target_bytes_per_pair)
    hint_total = sum(hint_allocation)
    return {
        "num_pairs": len(ranked_rows),
        "target_bytes_per_pair": target_bytes_per_pair,
        "envelope_bytes_total": envelope,
        "actual_bytes_total": total_actual,
        "actual_vs_envelope_pct": (
            None if envelope == 0 else 100.0 * total_actual / envelope
        ),
        "hint_allocation_bytes_total": hint_total,
        "hint_allocation_vs_envelope_pct": (
            None if envelope == 0 else 100.0 * hint_total / envelope
        ),
        "hint_strategy": (
            "high_leverage_pairs_1_5x_low_leverage_pairs_0_5x_med_unchanged"
        ),
        "planning_only": True,
    }


def diagnose(
    rows: list[dict[str, Any]],
    *,
    pose_avg: float,
    top_k: int,
    bottom_k: int,
    evidence_grade: str,
    target_bytes_per_pair: int = PER_PAIR_SIDE_INFO_TARGET_BYTES,
) -> dict[str, Any]:
    """Run the full per-pair diagnostic.

    Returns the structured output schema described in the module docstring.
    """
    weights = _marginal_weights(pose_avg)
    enriched = compute_per_pair_sensitivity(rows, pose_avg=pose_avg)
    enriched.sort(key=lambda r: r["sensitivity"], reverse=True)
    _classify_leverage(enriched)
    top_k = max(0, min(top_k, len(enriched)))
    bottom_k = max(0, min(bottom_k, len(enriched)))
    budget_plan = _budget_plan(enriched, target_bytes_per_pair=target_bytes_per_pair)
    return {
        "lane_id": LANE_ID,
        "operating_point_pose_avg": pose_avg,
        "operating_point_seg_avg": PR106_R2_SEG_AVG,
        "marginal_weights": weights,
        "per_pair_ranking": enriched,
        "top_k_high_leverage": enriched[:top_k],
        "bottom_k_low_leverage": enriched[-bottom_k:] if bottom_k else [],
        "budget_plan": budget_plan,
        "evidence_grade": evidence_grade,
        "evidence_tag": (
            "[macOS-CPU advisory only]"
            if evidence_grade == "macOS-CPU-advisory"
            else "[synthetic-prior; planning-only]"
        ),
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ranking_only": True,
        "dispatch_blockers": [
            "per_pair_breakdown_is_planning_only_not_score_evidence",
            "promotion_requires_paired_contest_cpu_gha_linux_x86_64",
        ],
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help=(
            "Path to per-pair observations JSON or JSONL. When omitted, the "
            "tool emits a synthetic-prior ranking with evidence_grade="
            "'synthetic-prior' so the operator sees that nothing has been "
            "measured yet."
        ),
    )
    parser.add_argument(
        "--operating-point-pose-avg",
        type=float,
        default=PR106_R2_POSE_AVG,
        help=(
            f"Pose_avg at the target operating point (default {PR106_R2_POSE_AVG} "
            "= PR106 r2 frontier). At this operating point, pose marginal "
            "value is 2.71x SegNet per CLAUDE.md 'operating-point dependent'."
        ),
    )
    parser.add_argument(
        "--target-bytes-per-pair",
        type=int,
        default=PER_PAIR_SIDE_INFO_TARGET_BYTES,
        help=(
            f"Per-pair side info byte budget (default {PER_PAIR_SIDE_INFO_TARGET_BYTES} "
            "per design memo)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of high-leverage pairs to surface (default 20).",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=20,
        help="Number of low-leverage pairs to surface (default 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Refused under /tmp per CLAUDE.md. When omitted, "
            "result is printed to stdout only."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.observations is not None:
        rows = _load_observations(args.observations)
        evidence_grade = "macOS-CPU-advisory"
    else:
        rows = _synthetic_prior_rows()
        evidence_grade = "synthetic-prior"

    summary = diagnose(
        rows,
        pose_avg=args.operating_point_pose_avg,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        evidence_grade=evidence_grade,
        target_bytes_per_pair=args.target_bytes_per_pair,
    )

    if args.output is not None:
        out_str = str(args.output)
        if (
            out_str.startswith("/tmp/")
            or "/private/tmp/" in out_str
            or "/var/tmp/" in out_str
        ):
            print(
                f"[diagnose-per-pair] FATAL: refusing /tmp output path: {out_str!r}",
                file=sys.stderr,
            )
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"[diagnose-per-pair] wrote {args.output}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    raise SystemExit(main())
