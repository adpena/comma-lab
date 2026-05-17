#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare baseline/candidate exact-eval artifacts for scorer-visible response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac.scorer_response_probe import (  # noqa: E402
    VERDICT_BLOCKED_CONTROL_MISMATCH,
    VERDICT_BLOCKED_CUSTODY,
    compare_score_response,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _render_markdown(report: dict[str, Any], *, title: str) -> str:
    baseline = report.get("baseline") or {}
    candidate = report.get("candidate") or {}
    deltas = report.get("deltas") or {}
    thresholds = report.get("thresholds") or {}
    lines = [
        f"# {title}",
        "",
        "Authority:",
        "- score_claim: false",
        "- promotion_eligible: false",
        "- ready_for_exact_eval_dispatch: false",
        "- dispatch_attempted: false",
        "",
        f"Verdict: `{report['verdict']}`",
        f"Mode: `{report['mode']}`",
        "",
        "## Thresholds",
        "",
        f"- min_total_improvement: `{thresholds.get('min_total_improvement')}`",
        f"- min_scorer_term_improvement: `{thresholds.get('min_scorer_term_improvement')}`",
        "",
        "## Evidence",
        "",
        "| side | axis | score | seg_term | pose_term | rate_term | bytes | runtime_tree_sha256 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for side, evidence in (("baseline", baseline), ("candidate", candidate)):
        if not evidence:
            lines.append(f"| {side} | missing |  |  |  |  |  |  |")
            continue
        runtime_sha = str(evidence.get("runtime_tree_sha256") or "")
        lines.append(
            f"| {side} | {evidence.get('axis')} | {float(evidence.get('score')):.9f} | "
            f"{float(evidence.get('seg_term')):.9f} | "
            f"{float(evidence.get('pose_term')):.9f} | "
            f"{float(evidence.get('rate_term')):.9f} | "
            f"{int(evidence.get('archive_bytes'))} | `{runtime_sha[:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "Negative deltas improve the contest score.",
            "",
            f"- total_delta: `{deltas.get('total_delta')}`",
            f"- scorer_term_delta: `{deltas.get('scorer_term_delta')}`",
            f"- seg_term_delta: `{deltas.get('seg_term_delta')}`",
            f"- pose_term_delta: `{deltas.get('pose_term_delta')}`",
            f"- rate_term_delta: `{deltas.get('rate_term_delta')}`",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This probe tests score response, not byte liveness. A positive result "
            "requires official-score component movement under matched controls; "
            "a rate-only improvement is not evidence that the substrate's "
            "distinguishing feature is scorer-visible.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--title", default="Substrate Score-Response Probe")
    parser.add_argument("--axis", choices=("contest_cpu", "contest_cuda"))
    parser.add_argument("--mode", choices=("ablation", "candidate"), default="ablation")
    parser.add_argument("--min-total-improvement", type=float, default=0.001)
    parser.add_argument("--min-scorer-term-improvement", type=float, default=0.0005)
    parser.add_argument(
        "--relaxed-custody",
        action="store_true",
        help="Only validate axis/formula fields; do not require hardware/device/log metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = _load_json(args.baseline_json)
    candidate = _load_json(args.candidate_json)
    report = compare_score_response(
        baseline=baseline,
        candidate=candidate,
        expected_axis=args.axis,
        mode=args.mode,
        min_total_improvement=args.min_total_improvement,
        min_scorer_term_improvement=args.min_scorer_term_improvement,
        strict_exact_custody=not args.relaxed_custody,
    )
    payload = report.to_json_dict()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            _render_markdown(payload, title=args.title),
            encoding="utf-8",
        )
    print(f"[score-response-probe] {payload['verdict']} wrote {args.output_json}")
    if payload["verdict"] in {
        VERDICT_BLOCKED_CUSTODY,
        VERDICT_BLOCKED_CONTROL_MISMATCH,
    }:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
