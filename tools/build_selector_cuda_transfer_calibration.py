#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the selector CPU/proxy-to-CUDA transfer calibration surface.

This is a planning and guardrail tool. It does not dispatch jobs, evaluate
archives, or claim score. It consolidates exact CUDA selector controls and the
PR101 FEC6 paired CPU/CUDA drift xray into one deterministic decision packet so
water-fill and film-grain selectors cannot silently route from proxy positives
after measured CUDA regressions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONTEST_ORIGINAL_BYTES = 37_545_489
DEFAULT_SELECTOR_REVIEWS = (
    REPO / ".omx/research/hdm8_cuda_selector_sparse_top001_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_cuda_selector_sparse_budget128_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_even_frame_selector_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_film_grain_selector_charged_mps_aggressive_v2_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_fixed_even_grain_chroma_1_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_fixed_even_rgb_bias_0_p05_m05_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_fixed_even_rgb_bias_m05_p05_0_exact_cuda_result_review_20260515_codex.json",
    REPO / ".omx/research/hdm8_fixed_even_rgb_bias_m1_p05_p05_exact_cuda_result_review_20260515_codex.json",
)
DEFAULT_PAIRED_AXIS_XRAY = (
    REPO / "experiments/results/xray_paired_cpu_cuda_axis_delta_pr101_fec6_20260515_codex/paired_axis_delta.json"
)
DEFAULT_JSON_OUT = REPO / ".omx/research/selector_cuda_transfer_calibration_20260515_codex.json"
DEFAULT_MD_OUT = REPO / ".omx/research/selector_cuda_transfer_calibration_20260515_codex.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def byte_equivalent(score_delta: float) -> int:
    return round(score_delta * CONTEST_ORIGINAL_BYTES / 25.0)


def selector_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    score = float(payload["canonical_score"])
    baseline = float(payload["baseline_score"])
    delta = score - baseline
    recomputation = payload.get("score_recomputation") or {}
    score_axis = payload.get("score_axis")
    exact_cuda_flag = payload.get("exact_cuda_evidence") is True
    if exact_cuda_flag and score_axis != "contest_cuda":
        outcome = "non_cuda_exact_flag_ignored"
    elif not exact_cuda_flag:
        outcome = "not_exact_cuda_evidence"
    elif delta <= 0.0:
        outcome = "cuda_improved_or_neutral"
    else:
        outcome = "cuda_regression"
    return {
        "path": repo_rel(path),
        "technique": payload.get("technique") or path.stem,
        "score_axis": score_axis,
        "exact_cuda_evidence_flag": exact_cuda_flag,
        "exact_cuda_evidence": exact_cuda_flag and score_axis == "contest_cuda",
        "canonical_score": score,
        "baseline_score": baseline,
        "score_delta_vs_baseline": delta,
        "byte_equivalent_delta": byte_equivalent(delta),
        "archive_bytes": recomputation.get("archive_bytes"),
        "archive_sha256": (payload.get("custody") or {}).get("archive_sha256"),
        "avg_segnet_dist": recomputation.get("avg_segnet_dist"),
        "avg_posenet_dist": recomputation.get("avg_posenet_dist"),
        "failure_class": payload.get("failure_class"),
        "measured_config_status": payload.get("measured_config_status"),
        "promotion_eligible": payload.get("promotion_eligible") is True,
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch") is True,
        "outcome": outcome,
    }


def paired_axis_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    components = payload.get("components") or {}
    target_gaps = payload.get("target_gaps") or {}
    raw = payload.get("raw_output_comparison") or {}
    return {
        "path": repo_rel(path),
        "classification": payload.get("classification"),
        "dominant_score_delta_component": components.get("dominant_score_delta_component"),
        "score_delta_byte_equivalent": components.get("score_delta_byte_equivalent"),
        "contest_cpu_score": (components.get("contest_cpu") or {}).get("score"),
        "contest_cuda_score": (components.get("contest_cuda") or {}).get("score"),
        "cuda_minus_cpu_score_delta": (
            components.get("delta_cuda_minus_cpu") or {}
        ).get("score_delta_cuda_minus_cpu"),
        "target_gaps": target_gaps,
        "raw_output_aggregate_sha256_match": raw.get("aggregate_sha256_match"),
        "cpu_aggregate_sha256": raw.get("cpu_aggregate_sha256"),
        "cuda_aggregate_sha256": raw.get("cuda_aggregate_sha256"),
    }


def calibration_decision(rows: list[dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    exact_rows = [row for row in rows if row["exact_cuda_evidence"]]
    non_cuda_exact_flag_rows = [
        row
        for row in rows
        if row.get("exact_cuda_evidence_flag") is True and row.get("score_axis") != "contest_cuda"
    ]
    neutral_or_positive = [row for row in exact_rows if row["score_delta_vs_baseline"] <= 0.0]
    regressions = [row for row in exact_rows if row["score_delta_vs_baseline"] > 0.0]
    raw_match = paired.get("raw_output_aggregate_sha256_match")
    blockers: list[str] = []
    if non_cuda_exact_flag_rows:
        blockers.append("non_cuda_selector_rows_marked_exact_cuda")
    if not neutral_or_positive:
        blockers.append("exact_cuda_positive_or_neutral_selector_control_missing")
    if regressions:
        blockers.append("measured_selector_controls_transfer_negative_on_cuda")
    if raw_match is False:
        blockers.append("pr101_cpu_cuda_inflated_output_aggregate_mismatch")
    elif raw_match is not True:
        blockers.append("pr101_cpu_cuda_inflated_output_aggregate_match_missing")
    if paired.get("dominant_score_delta_component") in {"pose", "seg"}:
        blockers.append("cpu_cuda_gap_component_dominated_not_rate_limited")
    calibration_status = "blocked" if blockers else "calibrated"
    ready = False
    if calibration_status == "calibrated":
        ready = bool(neutral_or_positive) and raw_match is True
    return {
        "ready_for_broad_waterfill_dispatch": ready,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "calibration_status": calibration_status,
        "blockers": blockers,
        "exact_cuda_selector_rows": len(exact_rows),
        "non_cuda_exact_flag_rows": len(non_cuda_exact_flag_rows),
        "selector_regression_rows": len(regressions),
        "selector_positive_or_neutral_rows": len(neutral_or_positive),
        "best_score_delta_vs_baseline": min(
            (row["score_delta_vs_baseline"] for row in exact_rows),
            default=None,
        ),
        "worst_score_delta_vs_baseline": max(
            (row["score_delta_vs_baseline"] for row in exact_rows),
            default=None,
        ),
        "policy": (
            "Do not route broad proxy-ranked film-grain/selector/water-fill "
            "dispatch until CUDA-in-loop selector controls are positive or "
            "neutral with no exact-CUDA regressions, and paired CPU/CUDA raw "
            "aggregate custody is explicit."
        ),
    }


def build_calibration(selector_paths: list[Path], paired_axis_path: Path) -> dict[str, Any]:
    rows = [selector_row(path) for path in selector_paths]
    paired = paired_axis_summary(paired_axis_path)
    decision = calibration_decision(rows, paired)
    input_paths = [*selector_paths, paired_axis_path]
    return {
        "schema": "selector_cuda_transfer_calibration_v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "tool": "tools/build_selector_cuda_transfer_calibration.py",
        "score_claim": False,
        "dispatch_attempted": False,
        "tool_run_manifest": {
            "input_files": [
                {
                    "path": repo_rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in input_paths
            ],
            "score_claim": False,
            "dispatch_attempted": False,
        },
        "decision": decision,
        "paired_axis_xray": paired,
        "selector_rows": sorted(
            rows,
            key=lambda row: (
                float(row["score_delta_vs_baseline"]),
                str(row["technique"]),
            ),
        ),
        "next_actions": [
            "Build a CUDA-in-loop selector objective before broad water-fill dispatch.",
            "Continue PR101 CPU-only selector work only when the candidate changes components by more than the charged byte cost.",
            "Treat rate-only PR101 FEC6 polishing as insufficient for CUDA while the paired aggregate hashes differ.",
            "Use the latest exact-reviewed PR106 PacketIR reference (currently format0C) and do not redispatch identical archives.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    paired = payload["paired_axis_xray"]
    lines = [
        "# Selector CUDA Transfer Calibration",
        "",
        f"- score_claim: `{str(payload['score_claim']).lower()}`",
        f"- dispatch_attempted: `{str(payload['dispatch_attempted']).lower()}`",
        f"- calibration_status: `{decision['calibration_status']}`",
        f"- ready_for_broad_waterfill_dispatch: `{str(decision['ready_for_broad_waterfill_dispatch']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(decision['ready_for_exact_eval_dispatch']).lower()}`",
        "",
        "## Decision",
        "",
        decision["policy"],
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in decision["blockers"])
    lines.extend(
        [
            "",
            "## PR101 FEC6 CPU/CUDA Drift",
            "",
            f"- classification: `{paired['classification']}`",
            f"- dominant score-delta component: `{paired['dominant_score_delta_component']}`",
            f"- score-delta byte equivalent: `{paired['score_delta_byte_equivalent']}`",
            f"- raw aggregate match: `{str(paired['raw_output_aggregate_sha256_match']).lower()}`",
            "",
            "## Exact-CUDA Selector Rows",
            "",
            "| technique | axis | score delta | byte equivalent | outcome |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in payload["selector_rows"]:
        lines.append(
            "| `{technique}` | `{axis}` | {delta:.12f} | {bytes_} | `{outcome}` |".format(
                technique=row["technique"],
                axis=row["score_axis"],
                delta=float(row["score_delta_vs_baseline"]),
                bytes_=row["byte_equivalent_delta"],
                outcome=row["outcome"],
            )
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in payload["next_actions"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector-review",
        action="append",
        type=Path,
        help="Exact-CUDA selector result-review JSON. Defaults to known HDM8 selector/control rows.",
    )
    parser.add_argument("--paired-axis-xray", type=Path, default=DEFAULT_PAIRED_AXIS_XRAY)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--strict-blocked-exit",
        action="store_true",
        help="Exit nonzero when the calibration decision is blocked.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_paths = args.selector_review or list(DEFAULT_SELECTOR_REVIEWS)
    payload = build_calibration(selector_paths, args.paired_axis_xray)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    if args.strict_blocked_exit and payload["decision"]["calibration_status"] != "calibrated":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
