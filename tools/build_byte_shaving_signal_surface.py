#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a canonical byte-shaving signal surface from reusable signal sources."""

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

from tac.optimization.byte_shaving_campaign import ByteShavingCampaignError  # noqa: E402
from tac.optimization.byte_shaving_signal_surface_builder import (  # noqa: E402
    build_byte_shaving_signal_surface,
)


def _render_markdown(surface: dict[str, Any]) -> str:
    lines = [
        f"# Byte-Shaving Signal Surface: {surface['campaign_id']}",
        "",
        f"- schema: `{surface['schema']}`",
        f"- lane_id: `{surface.get('lane_id')}`",
        f"- frontier_axis: `{surface.get('frontier_axis')}`",
        f"- units: `{len(surface.get('units') or [])}`",
        f"- source_signal_refs: `{len(surface.get('source_signal_refs') or [])}`",
        f"- auth_eval_refs: `{len(surface.get('auth_eval_refs') or [])}`",
        f"- mlx_calibration_refs: `{len(surface.get('mlx_calibration_refs') or [])}`",
        f"- scorer_response_refs: `{len(surface.get('scorer_response_refs') or [])}`",
        f"- inverse_scorer_surface_refs: `{len(surface.get('inverse_scorer_surface_refs') or [])}`",
        f"- engineered_correction_refs: `{len(surface.get('engineered_correction_refs') or [])}`",
        f"- inverse_action_functional_refs: `{len(surface.get('inverse_action_functional_refs') or [])}`",
        f"- xray_refs: `{len(surface.get('xray_refs') or [])}`",
        f"- canonical_equation_refs: `{len(surface.get('canonical_equation_refs') or [])}`",
        f"- atom_refs: `{len(surface.get('atom_refs') or [])}`",
        "",
        "## Authority Boundary",
        "- score_claim: `false`",
        "- score_claim_valid: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        f"- next_gate: `{surface['evidence_boundary']['next_gate']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--campaign-id", default="byte_shaving_signal_surface")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--lane-id", default="byte_shaving_signal_surface")
    parser.add_argument("--frontier-axis", default="[planning-only]")
    parser.add_argument("--candidate-queue", action="append", default=[])
    parser.add_argument("--engineered-correction-targeting", action="append", default=[])
    parser.add_argument("--engineered-correction-max-targets", type=int, default=None)
    parser.add_argument(
        "--engineered-correction-default-delta",
        type=float,
        default=0.0,
        help=(
            "Planning-only predicted score delta for each correction target. "
            "Use 0 until an empirical calibration row exists."
        ),
    )
    parser.add_argument("--inverse-action-functional", action="append", default=[])
    parser.add_argument("--master-gradient-archive-sha", action="append", default=[])
    parser.add_argument("--master-gradient-ledger", type=Path, default=None)
    parser.add_argument("--master-gradient-axis", default=None)
    parser.add_argument("--master-gradient-low-quantile", type=float, default=0.05)
    parser.add_argument("--master-gradient-max-units", type=int, default=32)
    parser.add_argument("--master-gradient-max-span-bytes", type=int, default=4096)
    parser.add_argument("--master-gradient-quality-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--auth-eval", action="append", default=[])
    parser.add_argument("--mlx-calibration", action="append", default=[])
    parser.add_argument("--scorer-response", action="append", default=[])
    parser.add_argument("--inverse-scorer-response", action="append", default=[])
    parser.add_argument("--inverse-scorer-max-units", type=int, default=16)
    parser.add_argument("--inverse-scorer-null-delta-epsilon", type=float, default=1e-6)
    parser.add_argument("--inverse-scorer-fragile-delta-threshold", type=float, default=0.0)
    parser.add_argument(
        "--inverse-scorer-allow-native-mlx-window-objective",
        action="store_true",
        help=(
            "Allow MLX scorer-response rows without normalized full-video fields as "
            "planning-only native-window inverse-surface samples with an explicit blocker."
        ),
    )
    parser.add_argument("--xray-hook", action="append", default=[])
    parser.add_argument("--canonical-equation-domain", action="append", default=[])
    parser.add_argument("--canonical-equation-consumer", action="append", default=[])
    parser.add_argument("--canonical-equation-registry", type=Path, default=None)
    parser.add_argument("--atom-id", action="append", default=[])
    parser.add_argument("--atom-min-predicted-impact", type=float, default=None)
    parser.add_argument("--atom-ledger", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        surface = build_byte_shaving_signal_surface(
            repo_root=args.repo_root,
            campaign_id=args.campaign_id,
            candidate_id=args.candidate_id,
            lane_id=args.lane_id,
            frontier_axis=args.frontier_axis,
            candidate_queue_paths=args.candidate_queue,
            engineered_correction_targeting_paths=args.engineered_correction_targeting,
            engineered_correction_max_targets=args.engineered_correction_max_targets,
            engineered_correction_default_predicted_quality_score_delta=(
                args.engineered_correction_default_delta
            ),
            inverse_action_functional_paths=args.inverse_action_functional,
            master_gradient_archive_sha256s=args.master_gradient_archive_sha,
            master_gradient_ledger_path=args.master_gradient_ledger,
            master_gradient_axis=args.master_gradient_axis,
            master_gradient_low_sensitivity_quantile=args.master_gradient_low_quantile,
            master_gradient_max_units=args.master_gradient_max_units,
            master_gradient_max_span_bytes=args.master_gradient_max_span_bytes,
            master_gradient_quality_cost_multiplier=args.master_gradient_quality_cost_multiplier,
            auth_eval_paths=args.auth_eval,
            mlx_calibration_paths=args.mlx_calibration,
            scorer_response_paths=args.scorer_response,
            inverse_scorer_response_paths=args.inverse_scorer_response,
            inverse_scorer_max_units=args.inverse_scorer_max_units,
            inverse_scorer_null_delta_epsilon=args.inverse_scorer_null_delta_epsilon,
            inverse_scorer_fragile_delta_threshold=args.inverse_scorer_fragile_delta_threshold,
            inverse_scorer_allow_native_mlx_window_objective=(
                args.inverse_scorer_allow_native_mlx_window_objective
            ),
            xray_hooks=args.xray_hook,
            canonical_equation_domains=args.canonical_equation_domain,
            canonical_equation_consumers=args.canonical_equation_consumer,
            canonical_equation_registry_path=args.canonical_equation_registry,
            atom_ids=args.atom_id,
            atom_min_predicted_impact=args.atom_min_predicted_impact,
            atom_ledger_path=args.atom_ledger,
        )
    except ByteShavingCampaignError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(surface, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(_render_markdown(surface), encoding="utf-8")
    print(
        f"wrote {args.output} "
        f"(units={len(surface['units'])}, "
        f"refs={len(surface['source_signal_refs']) + len(surface['auth_eval_refs']) + len(surface['mlx_calibration_refs']) + len(surface['scorer_response_refs']) + len(surface['inverse_scorer_surface_refs']) + len(surface['inverse_action_functional_refs'])})"
    )
    print(
        "score_claim=false promotion_eligible=false "
        "rank_or_kill_eligible=false ready_for_exact_eval_dispatch=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
