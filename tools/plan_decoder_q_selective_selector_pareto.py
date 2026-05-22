#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan compact DQS1 selector Pareto candidates and optional packet plans."""

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

from tac.optimization.decoder_q_selective_runtime_packet import (
    FALSE_AUTHORITY,
    DecoderQSelectiveRuntimePacketError,
    build_decoder_q_selective_runtime_packet_plan,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    dumps_json as dumps_packet_json,
)
from tac.optimization.decoder_q_selective_selector_pareto import (
    DecoderQSelectiveSelectorParetoError,
    build_selector_pareto_plan,
    load_json_object,
    write_json,
)

DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "submission_dir/archive.zip"
)


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values or None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-plan", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--frame-policy", default="pair_all_frames")
    parser.add_argument("--prefix-ks", help="Comma-separated top-rank prefix sizes.")
    parser.add_argument("--no-drop-one", action="store_true")
    parser.add_argument("--no-singletons", action="store_true")
    parser.add_argument("--base-score", type=float)
    parser.add_argument("--reference-score", type=float)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--packet-plan-dir", type=Path)
    parser.add_argument("--emit-packet-plan-limit", type=int, default=0)
    return parser.parse_args(argv)


def _write_packet_plans(
    *,
    bridge_plan: dict[str, object],
    pareto_plan: dict[str, object],
    base_archive: Path,
    packet_plan_dir: Path,
    limit: int,
    frame_policy: str,
) -> list[dict[str, object]]:
    emitted: list[dict[str, object]] = []
    candidates = pareto_plan.get("candidates")
    if not isinstance(candidates, list):
        raise DecoderQSelectiveSelectorParetoError("pareto candidates missing")
    packet_plan_dir.mkdir(parents=True, exist_ok=True)
    frontier_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("pareto_frontier") is True
    ]
    emit_candidates = (frontier_candidates or candidates)[: max(0, int(limit))]
    for candidate in emit_candidates:
        if not isinstance(candidate, dict):
            continue
        selector_id = str(candidate["selector_id"])
        packet_plan = build_decoder_q_selective_runtime_packet_plan(
            bridge_plan,
            base_archive=base_archive,
            repo_root=REPO_ROOT,
            frame_policy=frame_policy,
            selected_pair_indices=candidate["selected_pair_indices"],
        )
        path = packet_plan_dir / f"{selector_id}.json"
        path.write_text(dumps_packet_json(packet_plan), encoding="utf-8")
        emitted.append(
            {
                "selector_id": selector_id,
                "path": str(path.resolve()),
                "payload_bytes": packet_plan["selective_packet"]["payload_bytes"],
                "selected_pair_count": packet_plan["selective_packet"]["selected_pair_count"],
                **FALSE_AUTHORITY,
            }
        )
    return emitted


def _render_markdown(plan: dict[str, object]) -> str:
    summary = plan.get("summary", {})
    candidates = plan.get("candidates", [])
    lines = [
        "# Decoder-Q Selective Selector Pareto Plan",
        "",
        f"- Candidate count: `{summary.get('candidate_count') if isinstance(summary, dict) else None}`",
        f"- Recommended selector: `{summary.get('recommended_selector_id') if isinstance(summary, dict) else None}`",
        f"- Pareto-frontier candidates: `{summary.get('pareto_frontier_candidate_count') if isinstance(summary, dict) else None}`",
        f"- Score claim: `{plan.get('score_claim')}`",
        f"- Promotion eligible: `{plan.get('promotion_eligible')}`",
        "",
        "| rank | selector | kind | pairs | payload bytes | estimated CPU score |",
        "|---:|---|---|---:|---:|---:|",
    ]
    if isinstance(candidates, list):
        for row in candidates[:16]:
            if not isinstance(row, dict):
                continue
            estimate = row.get("exact_cpu_calibrated_estimate")
            predicted = (
                estimate.get("predicted_score")
                if isinstance(estimate, dict)
                else None
            )
            lines.append(
                "| "
                f"{row.get('pareto_rank')} | `{row.get('selector_id')}` | "
                f"`{row.get('selector_kind')}` | {row.get('selected_pair_count')} | "
                f"{row.get('payload_bytes')} | `{predicted}` |"
            )
    lines.extend(
        [
            "",
            "Authority: selector planning only. Materialization, locality controls, "
            "and exact auth eval are required before any score claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bridge_plan = load_json_object(args.bridge_plan)
        plan = build_selector_pareto_plan(
            bridge_plan,
            frame_policy=args.frame_policy,
            prefix_ks=_parse_csv_ints(args.prefix_ks),
            include_drop_one=not args.no_drop_one,
            include_singletons=not args.no_singletons,
            base_score=args.base_score,
            reference_score=args.reference_score,
        )
        emitted = []
        if args.packet_plan_dir is not None and args.emit_packet_plan_limit > 0:
            emitted = _write_packet_plans(
                bridge_plan=bridge_plan,
                pareto_plan=plan,
                base_archive=args.base_archive,
                packet_plan_dir=args.packet_plan_dir,
                limit=args.emit_packet_plan_limit,
                frame_policy=args.frame_policy,
            )
        plan["emitted_packet_plans"] = emitted
        write_json(args.json_out, plan)
        if args.md_out is not None:
            args.md_out.parent.mkdir(parents=True, exist_ok=True)
            args.md_out.write_text(_render_markdown(plan), encoding="utf-8")
    except (
        OSError,
        json.JSONDecodeError,
        DecoderQSelectiveSelectorParetoError,
        DecoderQSelectiveRuntimePacketError,
    ) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "json_out": str(args.json_out),
                "candidate_count": plan["summary"]["candidate_count"],
                "recommended_selector_id": plan["summary"]["recommended_selector_id"],
                "emitted_packet_plan_count": len(plan["emitted_packet_plans"]),
                "score_claim": False,
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
