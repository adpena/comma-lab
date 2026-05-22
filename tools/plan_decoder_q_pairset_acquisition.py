#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local-only DQS1 decoder-q pair-set acquisition candidates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.decoder_q_pairset_acquisition import (
    DecoderQPairsetAcquisitionError,
    build_decoder_q_pairset_acquisition_plan,
    load_json_object,
    write_json,
)


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-pareto", type=Path, required=True)
    parser.add_argument("--frame-policy", default="pair_all_frames")
    parser.add_argument("--prefix-ks", help="Comma-separated prefix sizes from the best selector.")
    parser.add_argument("--diversity-ks", help="Comma-separated diversity-spaced pair-set sizes.")
    parser.add_argument("--no-drop-one", action="store_true")
    parser.add_argument("--max-drop-two", type=int, default=128)
    parser.add_argument("--max-swap-in", type=int, default=32)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    return parser.parse_args(argv)


def _render_markdown(plan: dict[str, object]) -> str:
    summary = plan.get("summary", {})
    candidates = plan.get("candidates", [])
    lines = [
        "# Decoder-Q Pair-Set Acquisition Plan",
        "",
        f"- Candidate count: `{summary.get('candidate_count') if isinstance(summary, dict) else None}`",
        (
            "- Recommended acquisition: "
            f"`{summary.get('recommended_acquisition_id') if isinstance(summary, dict) else None}`"
        ),
        f"- Score claim: `{plan.get('score_claim')}`",
        f"- Promotion eligible: `{plan.get('promotion_eligible')}`",
        f"- Ready for exact eval dispatch: `{plan.get('ready_for_exact_eval_dispatch')}`",
        f"- Dispatch attempted: `{plan.get('dispatch_attempted')}`",
        "",
        "| rank | acquisition | kind | pairs | payload bytes | acquisition score | diversity | predicted mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(candidates, list):
        for row in candidates[:20]:
            if not isinstance(row, dict):
                continue
            predicted = row.get("predicted_score_mean")
            lines.append(
                "| "
                f"{row.get('acquisition_rank')} | `{row.get('acquisition_id')}` | "
                f"`{row.get('selector_kind')}` | {row.get('selected_pair_count')} | "
                f"{row.get('payload_bytes')} | `{row.get('acquisition_score')}` | "
                f"`{row.get('diversity_score')}` | `{predicted}` |"
            )
    lines.extend(
        [
            "",
            "Authority: planning-only local pair-set acquisition. It does not dispatch, "
            "materialize an archive, claim a score, promote a candidate, or rank/kill a lane.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selector_pareto = load_json_object(args.selector_pareto)
        plan = build_decoder_q_pairset_acquisition_plan(
            selector_pareto,
            frame_policy=args.frame_policy,
            prefix_ks=_parse_csv_ints(args.prefix_ks),
            diversity_ks=_parse_csv_ints(args.diversity_ks),
            include_drop_one=not args.no_drop_one,
            max_drop_two=args.max_drop_two,
            max_swap_in=args.max_swap_in,
            diversity_weight=args.diversity_weight,
        )
        write_json(args.json_out, plan)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(_render_markdown(plan), encoding="utf-8")
    except (OSError, json.JSONDecodeError, DecoderQPairsetAcquisitionError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "candidate_count": plan["summary"]["candidate_count"],
                "recommended_acquisition_id": plan["summary"]["recommended_acquisition_id"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
