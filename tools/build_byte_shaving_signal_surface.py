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
        f"- pairset_acquisition_refs: `{len(surface.get('pairset_acquisition_refs') or [])}`",
        f"- pair_frame_geometry_outcome_refs: `{len(surface.get('pair_frame_geometry_outcome_refs') or [])}`",
        f"- inverse_scorer_surface_refs: `{len(surface.get('inverse_scorer_surface_refs') or [])}`",
        f"- engineered_correction_refs: `{len(surface.get('engineered_correction_refs') or [])}`",
        f"- inverse_action_functional_refs: `{len(surface.get('inverse_action_functional_refs') or [])}`",
        f"- inverse_action_materialization_portfolios: `{len(surface.get('inverse_action_materialization_portfolios') or [])}`",
        f"- materializer_registry_portfolio_refs: `{len(surface.get('materializer_registry_portfolio_refs') or [])}`",
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


def _materializer_registry_portfolio(
    *,
    target_kinds: list[str],
    planning_saved_bytes: int,
    executable_confidence: float,
    contract_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from comma_lab.scheduler.byte_shaving_materializer_registry import (
        registry_manifest,
    )

    if planning_saved_bytes < 1:
        raise ByteShavingCampaignError(
            "--materializer-registry-planning-saved-bytes must be >= 1"
        )
    targets = {str(item).strip() for item in target_kinds if str(item).strip()}
    manifest = registry_manifest()
    adapters = [
        item
        for item in manifest.get("adapters", [])
        if isinstance(item, dict)
        and (not targets or str(item.get("target_kind") or "") in targets)
    ]
    if targets:
        found = {str(item.get("target_kind") or "") for item in adapters}
        missing = sorted(targets - found)
        if missing:
            raise ByteShavingCampaignError(
                "materializer registry target kind(s) not found: " + ",".join(missing)
            )

    units: list[dict[str, Any]] = []
    target_kind_counts: dict[str, int] = {}
    executable_count = 0
    candidate_archive_count = 0
    blocker_counts: dict[str, int] = {}
    for index, adapter in enumerate(adapters):
        target_kind = str(adapter.get("target_kind") or "")
        materializer_id = str(adapter.get("materializer_id") or "")
        executable = (
            adapter.get("executable") is True
            and adapter.get("emits_candidate_archive") is True
            and adapter.get("planning_only") is not True
        )
        if adapter.get("executable") is True:
            executable_count += 1
        if adapter.get("emits_candidate_archive") is True:
            candidate_archive_count += 1
        required_context_fields = [
            str(item)
            for item in adapter.get("required_context_fields", [])
            if str(item).strip()
        ]
        blockers = [
            "materializer_registry_portfolio_unit_is_planning_only",
            "materializer_registry_portfolio_requires_concrete_artifact_context",
            "requires_receiver_runtime_consumption_proof_before_exact_eval",
            "requires_exact_auth_eval_before_score_claim",
        ]
        if required_context_fields:
            blockers.append("materializer_registry_required_context_fields_missing")
        if adapter.get("cooperative_receiver_required") is True:
            blockers.append("requires_cooperative_receiver_runtime_adapter")
        if adapter.get("executable") is not True:
            blockers.append("materializer_registry_adapter_not_executable")
        if adapter.get("emits_candidate_archive") is not True or adapter.get("planning_only") is True:
            blockers.append(
                "materializer_registry_planning_only_adapter_not_candidate_archive"
            )
        for blocker in blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        target_kind_counts[target_kind] = target_kind_counts.get(target_kind, 0) + 1
        operation = {
            "operation_id": f"registry_portfolio_{target_kind or index}",
            "operation_family": adapter.get("operation_family"),
            "candidate_saved_bytes": planning_saved_bytes,
            "predicted_quality_score_delta": 0.0,
            "confidence": (
                executable_confidence if executable else contract_confidence
            ),
            "materializer": materializer_id,
            "target_kind": target_kind,
            "params": {
                "materializer_registry_schema": manifest.get("schema"),
                "materializer_registry_target_kind": target_kind,
                "materializer_id": materializer_id,
                "required_context_fields": required_context_fields,
                "implementation_module": adapter.get("implementation_module"),
                "materialize_function": adapter.get("materialize_function"),
                "receiver_proof_function": adapter.get("receiver_proof_function"),
                "receiver_verify_function": adapter.get("receiver_verify_function"),
            },
            "receiver_contract_kind": adapter.get("receiver_contract_kind"),
            "materializer_executable": bool(adapter.get("executable")),
            "materializer_adapter_registered": True,
            "executable_work_ready": False,
            "materializer_execution_status": (
                "registered_executable_requires_artifact_context"
                if executable
                else "registered_contract_not_executable"
            ),
            "required_context_fields": required_context_fields,
            "operation_portability": (
                "family_agnostic"
                if str(adapter.get("unit_kind") or "") in {"archive_section", "tensor", "packet_member"}
                else "adapter_contract_specific"
            ),
            "blockers": blockers,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        }
        unit_id = (
            f"materializer_registry_{adapter.get('unit_kind')}_"
            f"{adapter.get('operation_family')}_{target_kind or index}"
        )
        units.append(
            {
                "unit_id": unit_id,
                "unit_kind": adapter.get("unit_kind"),
                "source_index": index,
                "candidate_saved_bytes": planning_saved_bytes,
                "predicted_quality_score_cost": 0.0,
                "confidence": operation["confidence"],
                "operation_families": [adapter.get("operation_family")],
                "operations": [operation],
                "score_axis": "[planning-only materializer portfolio]",
                "evidence_grade": "[planning-only registry contract]",
                "evidence_semantics": (
                    "materializer_registry_portfolio_seed_for_autonomous_many_op_search"
                ),
                "source_candidate_id": materializer_id,
                "materializer_registry_signal": {
                    "schema": "byte_shaving_materializer_registry_portfolio_unit.v1",
                    "registry_schema": manifest.get("schema"),
                    "materializer_id": materializer_id,
                    "target_kind": target_kind,
                    "receiver_contract_id": adapter.get("receiver_contract_id"),
                    "receiver_contract_kind": adapter.get("receiver_contract_kind"),
                    "cooperative_receiver_required": adapter.get(
                        "cooperative_receiver_required"
                    ),
                    "materialization_resource_kind": adapter.get(
                        "materialization_resource_kind"
                    ),
                    "required_context_fields": required_context_fields,
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": blockers,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
                "gpu_launched": False,
            }
        )

    refs = [
        {
            "kind": "materializer_registry_portfolio",
            "schema": "byte_shaving_materializer_registry_portfolio_ref.v1",
            "registry_schema": manifest.get("schema"),
            "adapter_count": len(adapters),
            "registry_adapter_count": len(manifest.get("adapters", [])),
            "executable_adapter_count": executable_count,
            "candidate_archive_adapter_count": candidate_archive_count,
            "target_kind_filter": sorted(targets),
            "target_kind_counts": dict(sorted(target_kind_counts.items())),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "cooperative_receiver_grammar_registry": manifest.get(
                "cooperative_receiver_grammar_registry"
            ),
            "allowed_use": "planning_only_many_materializer_portfolio_seed",
            "forbidden_use": (
                "score_claim_or_promotion_or_rank_kill_or_paid_dispatch_authority"
            ),
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "dispatch_attempted": False,
            "gpu_launched": False,
        }
    ]
    return units, refs


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
    parser.add_argument(
        "--pairset-acquisition",
        action="append",
        default=[],
        help=(
            "decoder_q_pairset_acquisition.v1 JSON; active rate-savings repair "
            "budget rows become planning-only byte-shaving units."
        ),
    )
    parser.add_argument(
        "--dqs1-observation-jsonl",
        action="append",
        default=[],
        help=(
            "mlx_dynamic_sweep_observation.v1 JSONL from DQS1 local-first "
            "harvests; byte-saving rows become empirical planning anchors."
        ),
    )
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
    parser.add_argument(
        "--include-materializer-registry-portfolio",
        action="store_true",
        help=(
            "seed the signal surface with every registered materializer/receiver "
            "contract so campaign planning explores many operations, not one leaf"
        ),
    )
    parser.add_argument(
        "--materializer-registry-target-kind",
        action="append",
        default=[],
        help="optional target-kind filter for --include-materializer-registry-portfolio",
    )
    parser.add_argument(
        "--materializer-registry-planning-saved-bytes",
        type=int,
        default=1,
        help="planning-only saved-byte prior assigned to each registry portfolio unit",
    )
    parser.add_argument(
        "--materializer-registry-executable-confidence",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--materializer-registry-contract-confidence",
        type=float,
        default=0.25,
    )
    args = parser.parse_args(argv)

    try:
        registry_units: list[dict[str, Any]] = []
        registry_refs: list[dict[str, Any]] = []
        if args.include_materializer_registry_portfolio:
            registry_units, registry_refs = _materializer_registry_portfolio(
                target_kinds=args.materializer_registry_target_kind,
                planning_saved_bytes=args.materializer_registry_planning_saved_bytes,
                executable_confidence=(
                    args.materializer_registry_executable_confidence
                ),
                contract_confidence=args.materializer_registry_contract_confidence,
            )
        surface = build_byte_shaving_signal_surface(
            repo_root=args.repo_root,
            campaign_id=args.campaign_id,
            candidate_id=args.candidate_id,
            lane_id=args.lane_id,
            frontier_axis=args.frontier_axis,
            candidate_queue_paths=args.candidate_queue,
            pairset_acquisition_paths=args.pairset_acquisition,
            dqs1_observation_paths=args.dqs1_observation_jsonl,
            engineered_correction_targeting_paths=args.engineered_correction_targeting,
            engineered_correction_max_targets=args.engineered_correction_max_targets,
            engineered_correction_default_predicted_quality_score_delta=(
                args.engineered_correction_default_delta
            ),
            inverse_action_functional_paths=args.inverse_action_functional,
            allow_inverse_action_leaf_cell_candidates=(
                args.allow_leaf_inverse_cell_candidates
            ),
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
            materializer_registry_portfolio_units=registry_units,
            materializer_registry_portfolio_refs=registry_refs,
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
        f"refs={len(surface['source_signal_refs']) + len(surface['auth_eval_refs']) + len(surface['mlx_calibration_refs']) + len(surface['scorer_response_refs']) + len(surface['pairset_acquisition_refs']) + len(surface['pair_frame_geometry_outcome_refs']) + len(surface['inverse_scorer_surface_refs']) + len(surface['inverse_action_functional_refs']) + len(surface.get('materializer_registry_portfolio_refs') or [])})"
    )
    print(
        "score_claim=false promotion_eligible=false "
        "rank_or_kill_eligible=false ready_for_exact_eval_dispatch=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
