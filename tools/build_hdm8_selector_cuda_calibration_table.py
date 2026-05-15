#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed exact-CUDA calibration table for HDM8 selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.hdm8_cuda_selector_probe_plan import (  # noqa: E402
    build_hdm8_cuda_selector_probe_plan,
)
from tac.repo_io import read_json, repo_relative, write_json  # noqa: E402

SCHEMA = "hdm8_selector_cuda_calibration_table_v1"


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_delta(payload: dict[str, Any]) -> float | None:
    baseline = _float_or_none(payload.get("baseline_score"))
    candidate = _float_or_none(payload.get("canonical_score"))
    if baseline is None or candidate is None:
        return None
    return candidate - baseline


def _mode_from_technique(technique: str) -> str | None:
    mapping = {
        "hdm8_fixed_even_rgb_bias_m1_p05_p05_positive_control_exact_cuda_review": (
            "even_rgb_bias:-1,0.5,0.5"
        ),
        "hdm8_fixed_even_rgb_bias_0_p05_m05_positive_control_exact_cuda_review": (
            "even_rgb_bias:0,0.5,-0.5"
        ),
        "hdm8_fixed_even_rgb_bias_m05_p05_0_positive_control_exact_cuda_review": (
            "even_rgb_bias:-0.5,0.5,0"
        ),
        "hdm8_fixed_even_grain_chroma_1_positive_control_exact_cuda_review": (
            "even_grain_chroma:1"
        ),
    }
    return mapping.get(technique)


def _candidate_kind(technique: str) -> str:
    if "sparse_top" in technique or "sparse_budget" in technique:
        return "selector_sparse"
    if "fixed_" in technique or "positive_control" in technique:
        return "fixed_mode"
    if "selector" in technique:
        return "selector"
    return "unknown"


def _review_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    technique = str(payload.get("technique") or path.stem)
    score_delta = _score_delta(payload)
    score_recomputation = payload.get("score_recomputation")
    if not isinstance(score_recomputation, dict):
        score_recomputation = {}
    custody = payload.get("custody")
    if not isinstance(custody, dict):
        custody = {}
    row = {
        "path": repo_relative(path, REPO_ROOT),
        "technique": technique,
        "candidate_kind": _candidate_kind(technique),
        "mode": _mode_from_technique(technique),
        "score_axis": payload.get("score_axis"),
        "exact_cuda_evidence": payload.get("exact_cuda_evidence") is True,
        "baseline_score": payload.get("baseline_score"),
        "canonical_score": payload.get("canonical_score"),
        "score_delta_vs_baseline": score_delta,
        "avg_posenet_dist": score_recomputation.get("avg_posenet_dist"),
        "avg_segnet_dist": score_recomputation.get("avg_segnet_dist"),
        "archive_bytes": score_recomputation.get("archive_bytes")
        or custody.get("archive_bytes"),
        "archive_sha256": custody.get("archive_sha256"),
        "measured_config_status": payload.get("measured_config_status"),
        "promotion_eligible": payload.get("promotion_eligible") is True,
        "ready_for_exact_eval_dispatch": payload.get("ready_for_exact_eval_dispatch")
        is True,
    }
    if score_delta is None:
        row["calibration_outcome"] = "missing_baseline_delta"
    elif score_delta <= 0.0:
        row["calibration_outcome"] = "exact_cuda_nonregression"
    else:
        row["calibration_outcome"] = "exact_cuda_regression"
    return row


def _load_review_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError(f"review JSON is not an object: {path}")
        rows.append(_review_row(path, payload))
    return rows


def _build_blockers(rows: list[dict[str, Any]]) -> list[str]:
    exact_rows = [row for row in rows if row["exact_cuda_evidence"]]
    scored_rows = [
        row for row in exact_rows if row.get("score_delta_vs_baseline") is not None
    ]
    nonregression_rows = [
        row for row in scored_rows if float(row["score_delta_vs_baseline"]) <= 0.0
    ]
    regression_rows = [
        row for row in scored_rows if float(row["score_delta_vs_baseline"]) > 0.0
    ]
    blockers: list[str] = []
    if not exact_rows:
        blockers.append("exact_cuda_calibration_rows_missing")
    if not nonregression_rows:
        blockers.append("exact_cuda_positive_or_neutral_control_missing")
    if regression_rows:
        blockers.append("proxy_positive_calibration_rows_transferred_negative")
    if regression_rows and not nonregression_rows:
        blockers.append("broad_waterfill_selector_blocked_until_transfer_model")
    return blockers


def build_calibration_table(
    sweep: dict[str, Any],
    *,
    sweep_path: Path,
    review_paths: list[Path],
    max_atoms: int,
) -> dict[str, Any]:
    """Return a fail-closed selector calibration table."""

    plan = build_hdm8_cuda_selector_probe_plan(
        sweep,
        evidence_source_path=repo_relative(sweep_path, REPO_ROOT),
        max_atoms=max_atoms,
        prefix_sizes=[1],
    )
    review_rows = _load_review_rows(review_paths)
    blockers = _build_blockers(review_rows)
    scored_exact_rows = [
        row
        for row in review_rows
        if row["exact_cuda_evidence"]
        and row.get("score_delta_vs_baseline") is not None
    ]
    positive_or_neutral = [
        row for row in scored_exact_rows if float(row["score_delta_vs_baseline"]) <= 0.0
    ]
    regression_rows = [
        row for row in scored_exact_rows if float(row["score_delta_vs_baseline"]) > 0.0
    ]
    return {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_broad_waterfill_dispatch": False,
        "calibration_status": "blocked" if blockers else "calibrated",
        "blockers": blockers,
        "sweep": {
            "path": repo_relative(sweep_path, REPO_ROOT),
            "axis": sweep.get("axis"),
            "n_pairs": sweep.get("n_pairs"),
            "archive_bytes": sweep.get("archive_bytes"),
            "archive_sha256": sweep.get("archive_sha256"),
            "candidate_atom_count": plan.get("candidate_atom_count"),
        },
        "exact_cuda_summary": {
            "review_count": len(review_rows),
            "scored_exact_count": len(scored_exact_rows),
            "positive_or_neutral_count": len(positive_or_neutral),
            "regression_count": len(regression_rows),
            "best_score_delta_vs_baseline": min(
                (float(row["score_delta_vs_baseline"]) for row in scored_exact_rows),
                default=None,
            ),
            "worst_score_delta_vs_baseline": max(
                (float(row["score_delta_vs_baseline"]) for row in scored_exact_rows),
                default=None,
            ),
        },
        "exact_cuda_reviews": review_rows,
        "top_proxy_atoms": plan.get("top_atoms", []),
        "selector_dispatch_policy": {
            "broad_proxy_waterfill_allowed": False,
            "requires_exact_cuda_positive_control": True,
            "requires_transfer_model_before_more_broad_selectors": bool(regression_rows),
            "cpu_or_mps_rows_are_authority": False,
        },
    }


def _render_markdown(table: dict[str, Any]) -> str:
    lines = [
        "# HDM8 Selector CUDA Calibration Table",
        "",
        f"- score_claim: `{str(table['score_claim']).lower()}`",
        f"- calibration_status: `{table['calibration_status']}`",
        f"- ready_for_broad_waterfill_dispatch: `{str(table['ready_for_broad_waterfill_dispatch']).lower()}`",
        "",
        "## Exact-CUDA Rows",
        "",
        "| technique | kind | mode | score delta | outcome |",
        "|---|---|---|---:|---|",
    ]
    for row in table["exact_cuda_reviews"]:
        lines.append(
            f"| `{row['technique']}` | `{row['candidate_kind']}` | "
            f"`{row.get('mode') or ''}` | {row.get('score_delta_vs_baseline')} | "
            f"`{row['calibration_outcome']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    for blocker in table["blockers"]:
        lines.append(f"- `{blocker}`")
    lines.extend(["", "## Policy", ""])
    lines.append(
        "Broad proxy-ranked selector and waterfill dispatch stays blocked until "
        "an exact-CUDA positive/neutral control exists or a calibrated transfer "
        "model explains why the current proxy-positive rows regress."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument(
        "--exact-review-json",
        type=Path,
        action="append",
        default=[],
        help="Exact-CUDA result review JSON. Pass once per reviewed candidate.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--max-atoms", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sweep = read_json(args.sweep_json)
    if not isinstance(sweep, dict):
        raise TypeError(f"sweep JSON is not an object: {args.sweep_json}")
    table = build_calibration_table(
        sweep,
        sweep_path=args.sweep_json,
        review_paths=args.exact_review_json,
        max_atoms=args.max_atoms,
    )
    write_json(args.output_json, table)
    output_md = args.output_md or args.output_json.with_suffix(".md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(table), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(output_md),
                "calibration_status": table["calibration_status"],
                "blockers": table["blockers"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
