# SPDX-License-Identifier: MIT
"""Strict consolidation audit for the frontier final-rate compiler stack.

The historical name is "final-rate attack", but the owned implementation surface
already spans scorer actions, byte grammar, materializer queues, feedback loops,
and exact-readiness handoff. This audit keeps that surface coherent by refusing
parallel compiler-shaped scaffolds and by checking that the existing registry
covers the score-program layers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .byte_shaving_materializer_registry import registry_manifest

FRONTIER_RATE_ATTACK_CONSOLIDATION_SCHEMA = (
    "frontier_rate_attack_consolidation_audit.v1"
)

REQUIRED_SOURCE_SURFACES: tuple[tuple[str, str], ...] = (
    (
        "queue_autoloop",
        "src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py",
    ),
    (
        "feedback_cycle",
        "src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py",
    ),
    (
        "feedback_builders",
        "src/comma_lab/scheduler/frontier_rate_attack_feedback.py",
    ),
    (
        "frontier_bootstrap",
        "src/comma_lab/scheduler/frontier_rate_attack_bootstrap.py",
    ),
    (
        "materializer_registry",
        "src/comma_lab/scheduler/byte_shaving_materializer_registry.py",
    ),
    (
        "materializer_queue",
        "src/comma_lab/scheduler/byte_shaving_campaign_queue.py",
    ),
    (
        "campaign_planner",
        "src/tac/optimization/byte_shaving_campaign.py",
    ),
    (
        "operation_set_compiler",
        "src/tac/optimization/inverse_steganalysis_operation_set_compiler.py",
    ),
    (
        "queue_builder_cli",
        "tools/build_frontier_final_rate_attack_queue.py",
    ),
    (
        "feedback_cycle_cli",
        "tools/run_frontier_rate_attack_feedback_cycle.py",
    ),
    (
        "materializer_campaign_cli",
        "tools/run_byte_shaving_materializer_campaign.py",
    ),
    (
        "exact_eval_consumer_cli",
        "tools/build_materializer_exact_eval_consumer.py",
    ),
    (
        "exact_eval_dispatch_plan_cli",
        "tools/build_materializer_exact_eval_dispatch_plan.py",
    ),
    (
        "operator_briefing",
        "tools/operator_briefing.py",
    ),
    (
        "all_lanes_preflight",
        "tools/all_lanes_preflight.py",
    ),
)

SCORE_PROGRAM_LAYER_TARGET_KINDS: dict[str, tuple[str, ...]] = {
    "action_candidates": (
        "dqs1_pairset_drop_pair",
        "inverse_scorer_cell_candidate_v1",
        "inverse_scorer_action_functional_v1",
        "inverse_steganalysis_high_level_operation_set_v1",
    ),
    "entropy_grammar": (
        "byte_range_entropy_recode_v1",
        "selector_stream_context_recode_v1",
        "fp11_source_brotli_recode_v1",
        "archive_section_entropy_recode_v1",
        "archive_zip_repack_v1",
        "packet_member_recompress_v1",
        "packet_member_merge_v1",
        "packet_member_zip_header_elide_v1",
    ),
    "payload_and_residual_basis": (
        "renderer_payload_dfl1_v1",
        "tensor_factorize_v1",
        "tensor_prune_v1",
        "tensor_quantize_v1",
        "tensor_shared_codebook_v1",
        "z8_hpc1_detail_entropy_delta_v1",
    ),
}

SCORE_PROGRAM_COMPILER_DAG_EDGES: tuple[tuple[str, str], ...] = (
    ("oracle_target_fiber", "action_candidates"),
    ("base_witness_archive", "action_candidates"),
    ("action_candidates", "payload_and_residual_basis"),
    ("action_candidates", "entropy_grammar"),
    ("payload_and_residual_basis", "entropy_grammar"),
    ("entropy_grammar", "receiver_parseback_replay"),
    ("receiver_parseback_replay", "exact_eval_handoff"),
)

REQUIRED_STATE_GLOBS: tuple[tuple[str, str], ...] = (
    (
        "frontier_final_rate_attack_queues",
        ".omx/state/experiment_queue_frontier_final_rate_attack*.sqlite",
    ),
    (
        "post_feedback_chain_compiler_queues",
        ".omx/state/experiment_queue_frontier_final_rate_attack*_post_execute_feedback_chain_compiler.sqlite",
    ),
    (
        "post_feedback_repair_budget_queues",
        ".omx/state/experiment_queue_frontier_final_rate_attack*_post_execute_feedback_repair_budget_waterfill.sqlite",
    ),
)

REQUIRED_RESEARCH_GLOBS: tuple[tuple[str, str], ...] = (
    (
        "frontier_rate_attack_research_artifacts",
        ".omx/research/*frontier*rate*attack*",
    ),
    (
        "byte_shaving_research_artifacts",
        ".omx/research/*byte*shaving*",
    ),
    (
        "materializer_research_artifacts",
        ".omx/research/*materializer*",
    ),
)

FORBIDDEN_PARALLEL_SURFACE_GLOBS: tuple[str, ...] = (
    "tools/score_program_compiler",
    "tools/score_program_compiler/**",
    "tools/*score_program_compiler*.py",
    "tools/*action_byte*.py",
    "tools/*action_atlas*.py",
    "src/tac/*score_program_compiler*.py",
    "src/tac/*action_byte*.py",
    "src/tac/*action_atlas*.py",
    "src/tac/**/*score_program_compiler*.py",
    "src/tac/**/*action_byte*.py",
    "src/tac/**/*action_atlas*.py",
)


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _glob_paths(repo_root: Path, pattern: str) -> list[str]:
    return sorted(
        _repo_rel(path, repo_root)
        for path in repo_root.glob(pattern)
        if path.exists()
    )


def _required_surface_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface_id, rel_path in REQUIRED_SOURCE_SURFACES:
        path = repo_root / rel_path
        rows.append(
            {
                "surface_id": surface_id,
                "path": rel_path,
                "exists": path.is_file(),
            }
        )
    return rows


def _glob_count_rows(
    repo_root: Path,
    specs: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for surface_id, pattern in specs:
        paths = _glob_paths(repo_root, pattern)
        rows.append(
            {
                "surface_id": surface_id,
                "glob": pattern,
                "count": len(paths),
                "sample_paths": paths[:10],
            }
        )
    return rows


def _adapters_by_target_kind(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for item in manifest.get("adapters") or []:
        if not isinstance(item, Mapping):
            continue
        target_kind = str(item.get("target_kind") or "").strip()
        if target_kind:
            rows[target_kind] = item
    return rows


def _bool_field(row: Mapping[str, Any], key: str) -> bool:
    return row.get(key) is True


def _adapter_bound_function(row: Mapping[str, Any], key: str) -> bool:
    return bool(str(row.get(key) or "").strip())


def _layer_adapter_rows(
    *,
    target_kinds: Iterable[str],
    adapters_by_target_kind: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_kind in target_kinds:
        adapter = adapters_by_target_kind.get(target_kind)
        if adapter is None:
            rows.append(
                {
                    "target_kind": target_kind,
                    "adapter_registered": False,
                    "executable_candidate_archive": False,
                    "receiver_contract_bound": False,
                }
            )
            continue
        executable = _bool_field(adapter, "executable")
        emits_candidate_archive = _bool_field(adapter, "emits_candidate_archive")
        planning_only = _bool_field(adapter, "planning_only")
        rows.append(
            {
                "target_kind": target_kind,
                "adapter_registered": True,
                "materializer_id": adapter.get("materializer_id"),
                "unit_kind": adapter.get("unit_kind"),
                "operation_family": adapter.get("operation_family"),
                "executable": executable,
                "emits_candidate_archive": emits_candidate_archive,
                "planning_only": planning_only,
                "executable_candidate_archive": (
                    executable and emits_candidate_archive and not planning_only
                ),
                "receiver_contract_bound": bool(
                    str(adapter.get("receiver_contract_id") or "").strip()
                ),
                "receiver_contract_id": adapter.get("receiver_contract_id"),
                "receiver_contract_kind": adapter.get("receiver_contract_kind"),
                "cooperative_receiver_required": _bool_field(
                    adapter,
                    "cooperative_receiver_required",
                ),
                "materialize_function_bound": _adapter_bound_function(
                    adapter,
                    "materialize_function",
                ),
                "receiver_proof_function_bound": _adapter_bound_function(
                    adapter,
                    "receiver_proof_function",
                ),
                "receiver_verify_function_bound": _adapter_bound_function(
                    adapter,
                    "receiver_verify_function",
                ),
                "required_context_fields": list(
                    adapter.get("required_context_fields") or []
                ),
            }
        )
    return rows


def _layer_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    known_target_kinds = {
        str(item) for item in manifest.get("known_target_kinds") or []
    }
    adapters_by_target_kind = _adapters_by_target_kind(manifest)
    rows: list[dict[str, Any]] = []
    for layer_id, target_kinds in SCORE_PROGRAM_LAYER_TARGET_KINDS.items():
        missing = [target for target in target_kinds if target not in known_target_kinds]
        adapter_rows = _layer_adapter_rows(
            target_kinds=target_kinds,
            adapters_by_target_kind=adapters_by_target_kind,
        )
        executable_candidate_archive_count = sum(
            1 for row in adapter_rows if row.get("executable_candidate_archive") is True
        )
        receiver_contract_count = sum(
            1 for row in adapter_rows if row.get("receiver_contract_bound") is True
        )
        rows.append(
            {
                "layer_id": layer_id,
                "target_kinds": list(target_kinds),
                "missing_target_kinds": missing,
                "adapter_rows": adapter_rows,
                "adapter_count": sum(
                    1 for row in adapter_rows if row.get("adapter_registered") is True
                ),
                "executable_candidate_archive_count": (
                    executable_candidate_archive_count
                ),
                "planning_only_count": sum(
                    1 for row in adapter_rows if row.get("planning_only") is True
                ),
                "receiver_contract_count": receiver_contract_count,
                "receiver_proof_hook_count": sum(
                    1
                    for row in adapter_rows
                    if row.get("receiver_proof_function_bound") is True
                ),
                "receiver_verify_hook_count": sum(
                    1
                    for row in adapter_rows
                    if row.get("receiver_verify_function_bound") is True
                ),
                "covered": (
                    not missing
                    and len(adapter_rows) == len(target_kinds)
                    and executable_candidate_archive_count > 0
                    and receiver_contract_count == len(target_kinds)
                ),
            }
        )
    return rows


def _adapter_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    adapters = [row for row in manifest.get("adapters") or [] if isinstance(row, Mapping)]
    executable = [row for row in adapters if row.get("executable") is True]
    planning_only = [row for row in adapters if row.get("planning_only") is True]
    candidate_emitters = [
        row for row in adapters if row.get("emits_candidate_archive") is True
    ]
    return {
        "adapter_count": len(adapters),
        "executable_count": len(executable),
        "planning_only_count": len(planning_only),
        "candidate_archive_emitter_count": len(candidate_emitters),
        "target_kind_count": len(manifest.get("known_target_kinds") or []),
    }


def _forbidden_parallel_surfaces(repo_root: Path) -> list[str]:
    paths: list[str] = []
    for pattern in FORBIDDEN_PARALLEL_SURFACE_GLOBS:
        paths.extend(_glob_paths(repo_root, pattern))
    return sorted(dict.fromkeys(paths))


def build_frontier_rate_attack_consolidation_audit(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return a fail-closed audit of the existing final-rate compiler stack."""

    root = Path(repo_root).resolve(strict=False)
    manifest = registry_manifest()
    required_sources = _required_surface_rows(root)
    state_rows = _glob_count_rows(root, REQUIRED_STATE_GLOBS)
    research_rows = _glob_count_rows(root, REQUIRED_RESEARCH_GLOBS)
    layer_rows = _layer_rows(manifest)
    forbidden_surfaces = _forbidden_parallel_surfaces(root)

    blockers: list[str] = []
    blockers.extend(
        f"required_source_surface_missing:{row['surface_id']}:{row['path']}"
        for row in required_sources
        if row["exists"] is not True
    )
    blockers.extend(
        f"required_state_surface_missing:{row['surface_id']}:{row['glob']}"
        for row in state_rows
        if row["count"] == 0
    )
    blockers.extend(
        f"required_research_surface_missing:{row['surface_id']}:{row['glob']}"
        for row in research_rows
        if row["count"] == 0
    )
    for row in layer_rows:
        blockers.extend(
            f"score_program_layer_target_missing:{row['layer_id']}:{target}"
            for target in row["missing_target_kinds"]
        )
        if row["executable_candidate_archive_count"] == 0:
            blockers.append(
                f"score_program_layer_no_executable_candidate_archive:{row['layer_id']}"
            )
        for adapter in row.get("adapter_rows") or []:
            if not isinstance(adapter, Mapping):
                continue
            if adapter.get("adapter_registered") is not True:
                blockers.append(
                    "score_program_target_adapter_missing:"
                    f"{row['layer_id']}:{adapter.get('target_kind')}"
                )
            elif adapter.get("receiver_contract_bound") is not True:
                blockers.append(
                    "score_program_target_receiver_contract_missing:"
                    f"{row['layer_id']}:{adapter.get('target_kind')}"
                )
    blockers.extend(
        f"parallel_score_program_compiler_surface_forbidden:{path}"
        for path in forbidden_surfaces
    )

    return {
        "schema": FRONTIER_RATE_ATTACK_CONSOLIDATION_SCHEMA,
        "status": "PASS" if not blockers else "FAIL",
        "canonical_surface": "frontier_final_rate_attack_materializer_stack",
        "canonical_rule": (
            "Extend the existing final-rate/byte-shaving/materializer/"
            "inverse-steganalysis stack; do not add parallel score-program, "
            "action-byte, or action-atlas compiler trees."
        ),
        "legacy_name": "frontier_final_rate_attack",
        "formal_name": "score_program_compiler_over_frozen_evaluator_quotient",
        "required_sources": required_sources,
        "registry": {
            "schema": manifest.get("schema"),
            **_adapter_summary(manifest),
        },
        "score_program_layers": layer_rows,
        "score_program_dag": {
            "nodes": sorted(
                set().union(*(set(edge) for edge in SCORE_PROGRAM_COMPILER_DAG_EDGES))
            ),
            "edges": [
                {"from": source, "to": target}
                for source, target in SCORE_PROGRAM_COMPILER_DAG_EDGES
            ],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "state_surfaces": state_rows,
        "research_surfaces": research_rows,
        "forbidden_parallel_surface_globs": list(FORBIDDEN_PARALLEL_SURFACE_GLOBS),
        "forbidden_parallel_surfaces": forbidden_surfaces,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def render_frontier_rate_attack_consolidation_audit(
    audit: Mapping[str, Any],
) -> str:
    """Render a compact operator-readable audit summary."""

    lines = [
        f"schema: {audit.get('schema')}",
        f"status: {audit.get('status')}",
        f"canonical_surface: {audit.get('canonical_surface')}",
        f"formal_name: {audit.get('formal_name')}",
        f"rule: {audit.get('canonical_rule')}",
    ]
    registry = audit.get("registry") if isinstance(audit.get("registry"), Mapping) else {}
    lines.append(
        "registry: "
        f"{registry.get('adapter_count', 0)} adapter(s), "
        f"{registry.get('executable_count', 0)} executable, "
        f"{registry.get('planning_only_count', 0)} planning-only"
    )
    lines.append("layers:")
    for row in audit.get("score_program_layers") or []:
        if not isinstance(row, Mapping):
            continue
        missing = row.get("missing_target_kinds") or []
        suffix = "covered" if not missing else f"missing={','.join(map(str, missing))}"
        lines.append(
            f"  - {row.get('layer_id')}: {suffix}; "
            f"exec_archive={row.get('executable_candidate_archive_count', 0)}; "
            f"planning_only={row.get('planning_only_count', 0)}; "
            f"receiver_contracts={row.get('receiver_contract_count', 0)}; "
            f"receiver_proofs={row.get('receiver_proof_hook_count', 0)}"
        )
    dag = audit.get("score_program_dag")
    if isinstance(dag, Mapping):
        lines.append(
            "dag: "
            f"{len(dag.get('nodes') or [])} node(s), "
            f"{len(dag.get('edges') or [])} edge(s)"
        )
    lines.append("state:")
    for row in audit.get("state_surfaces") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(f"  - {row.get('surface_id')}: {row.get('count', 0)}")
    forbidden = audit.get("forbidden_parallel_surfaces") or []
    if forbidden:
        lines.append("forbidden_parallel_surfaces:")
        lines.extend(f"  - {path}" for path in forbidden)
    blockers = audit.get("blockers") or []
    if blockers:
        lines.append("blockers:")
        lines.extend(f"  - {blocker}" for blocker in blockers)
    return "\n".join(lines) + "\n"


__all__ = [
    "FRONTIER_RATE_ATTACK_CONSOLIDATION_SCHEMA",
    "SCORE_PROGRAM_COMPILER_DAG_EDGES",
    "build_frontier_rate_attack_consolidation_audit",
    "render_frontier_rate_attack_consolidation_audit",
]
