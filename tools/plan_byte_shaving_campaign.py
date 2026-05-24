#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan generalized byte-shaving sweeps from score/byte signal surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.byte_shaving_campaign import (  # noqa: E402
    PLAN_SCHEMA,
    SIGNAL_SURFACE_SCHEMA,
    ByteShavingCampaignError,
    build_byte_shaving_campaign_plan,
    build_signal_surface_from_candidate_queue,
    build_signal_surface_from_inverse_action_functional,
    build_signal_surface_from_master_gradient_anchor,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        f"# Byte-Shaving Campaign Plan: {plan['campaign_id']}",
        "",
        f"- schema: `{PLAN_SCHEMA}`",
        f"- generated_at_utc: `{plan['generated_at_utc']}`",
        f"- lane_id: `{plan['lane_id']}`",
        f"- frontier_axis: `{plan['frontier_axis']}`",
        f"- planning_only: `{plan['evidence_boundary']['planning_only']}`",
        "",
        "## Recommended Prefix",
    ]
    prefix = plan.get("recommended_prefix")
    if isinstance(prefix, dict):
        lines.extend([
            (
                f"- `{prefix['sweep_id']}` units=`{prefix['unit_count']}` "
                f"saved_bytes=`{prefix['candidate_saved_bytes']}` "
                f"expected_delta=`{prefix['expected_delta_score']}`"
            ),
            f"- units: `{','.join(prefix['selected_unit_ids'])}`",
        ])
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Combination"])
    combo = plan.get("recommended_combination")
    if isinstance(combo, dict):
        lines.extend([
            (
                f"- `{combo['combo_id']}` units=`{combo['unit_count']}` "
                f"saved_bytes=`{combo['candidate_saved_bytes']}` "
                f"expected_delta=`{combo['expected_delta_score']}`"
            ),
            f"- units: `{','.join(combo['selected_unit_ids'])}`",
            f"- operations: `{','.join(combo['operation_families'])}`",
        ])
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Top Combination Ladder",
        "| rank | combo | units | saved bytes | expected delta | operations |",
        "|---:|---|---:|---:|---:|---|",
    ])
    for rank, row in enumerate(plan.get("combination_ladder", [])[:12], start=1):
        lines.append(
            "| {rank} | `{combo}` | {units} | {bytes_} | `{delta}` | `{ops}` |".format(
                rank=rank,
                combo=row["combo_id"],
                units=row["unit_count"],
                bytes_=row["candidate_saved_bytes"],
                delta=row["expected_delta_score"],
                ops=",".join(row["operation_families"]),
            )
        )
    lines.extend([
        "",
        "## Bounded Permutation Priors",
        "| rank | combo | top sequence | inversion penalty |",
        "|---:|---|---|---:|",
    ])
    for rank, row in enumerate(plan.get("permutation_ladder", [])[:8], start=1):
        permutations = row.get("permutations") or []
        top = permutations[0] if permutations else {}
        sequence = " -> ".join(
            item.get("operation_family", "")
            for item in top.get("operation_sequence", [])
        )
        lines.append(
            "| {rank} | `{combo}` | `{sequence}` | {penalty} |".format(
                rank=rank,
                combo=row.get("combo_id"),
                sequence=sequence,
                penalty=top.get("prior_order_inversion_count", ""),
            )
        )
    policy = plan.get("search_space_policy") or {}
    lines.extend([
        "",
        "## Search Policy",
        f"- combination_search: `{policy.get('combination_search')}`",
        f"- permutation_search: `{policy.get('permutation_search')}`",
        f"- non_bruteforce_principle: `{policy.get('non_bruteforce_principle')}`",
    ])
    lines.extend([
        "",
        "## Authority Boundary",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- next_gate: `{plan['evidence_boundary']['next_gate']}`",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument(
        "--from-candidate-queue",
        action="store_true",
        help="convert optimizer_candidate_queue_v1 saved-byte rows into a signal surface first",
    )
    parser.add_argument(
        "--from-inverse-action-functional",
        action="store_true",
        help=(
            "convert inverse_steganalysis_discrete_action_functional.v1 "
            "water buckets into a byte-shaving signal surface first"
        ),
    )
    parser.add_argument(
        "--allow-leaf-inverse-cell-candidates",
        action="store_true",
        help=(
            "legacy/diagnostic mode: allow bare inverse-action cells to become "
            "IAS1 descriptor candidates. By default, bare cells become a "
            "high-level operation-compiler gap unless source operation-set "
            "provenance can be rehydrated into real materializers."
        ),
    )
    parser.add_argument("--campaign-id", default="byte_shaving_campaign")
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--from-master-gradient-archive-sha",
        default=None,
        help="build a planning-only byte-range surface from the latest usable master-gradient anchor",
    )
    parser.add_argument("--master-gradient-ledger", type=Path, default=None)
    parser.add_argument("--master-gradient-axis", default=None)
    parser.add_argument("--master-gradient-low-quantile", type=float, default=0.05)
    parser.add_argument("--master-gradient-max-units", type=int, default=32)
    parser.add_argument("--master-gradient-max-span-bytes", type=int, default=4096)
    parser.add_argument("--master-gradient-quality-cost-multiplier", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.max_k is not None and args.max_k < 1:
        raise SystemExit("--max-k must be >= 1 when provided")
    conversion_modes = sum(
        bool(value)
        for value in (
            args.from_candidate_queue,
            args.from_inverse_action_functional,
            args.from_master_gradient_archive_sha,
        )
    )
    if conversion_modes > 1:
        raise SystemExit(
            "choose at most one of --from-candidate-queue, "
            "--from-inverse-action-functional, or --from-master-gradient-archive-sha"
        )
    if args.from_master_gradient_archive_sha:
        if args.source is not None:
            raise SystemExit("--source is not used with --from-master-gradient-archive-sha")
        payload = build_signal_surface_from_master_gradient_anchor(
            archive_sha256=args.from_master_gradient_archive_sha,
            repo_root=args.repo_root,
            ledger_path=args.master_gradient_ledger,
            axis=args.master_gradient_axis,
            campaign_id=args.campaign_id,
            low_sensitivity_quantile=args.master_gradient_low_quantile,
            max_units=args.master_gradient_max_units,
            max_span_bytes=args.master_gradient_max_span_bytes,
            quality_cost_multiplier=args.master_gradient_quality_cost_multiplier,
        )
    elif args.source is None:
        raise SystemExit("--source is required unless --from-master-gradient-archive-sha is provided")
    else:
        if not args.source.is_file():
            raise SystemExit(f"source path does not exist: {args.source}")
        payload = _load_json(args.source)
        if not isinstance(payload, dict):
            raise SystemExit(f"{args.source}: expected object")
        if args.from_candidate_queue:
            payload = build_signal_surface_from_candidate_queue(
                payload,
                campaign_id=args.campaign_id,
            )
        elif args.from_inverse_action_functional:
            payload = build_signal_surface_from_inverse_action_functional(
                payload,
                campaign_id=args.campaign_id,
                allow_leaf_cell_candidates=(
                    args.allow_leaf_inverse_cell_candidates
                ),
            )
        elif payload.get("schema") != SIGNAL_SURFACE_SCHEMA:
            raise SystemExit(f"{args.source}: expected schema {SIGNAL_SURFACE_SCHEMA}")

    try:
        plan = build_byte_shaving_campaign_plan(
            payload,
            source_path=args.source,
            repo_root=args.repo_root,
            max_k=args.max_k,
        )
    except ByteShavingCampaignError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_markdown(plan), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"(units={len(plan['ranked_units'])}, "
        f"prefixes={len(plan['sweep_ladder'])}, "
        f"combinations={len(plan['combination_ladder'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
