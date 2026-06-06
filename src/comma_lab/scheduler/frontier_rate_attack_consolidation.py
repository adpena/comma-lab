# SPDX-License-Identifier: MIT
"""Strict consolidation audit for the frontier final-rate compiler stack.

The historical name is "final-rate attack", but the owned implementation surface
already spans scorer actions, byte grammar, materializer queues, feedback loops,
and exact-readiness handoff. This audit keeps that surface coherent by refusing
parallel compiler-shaped scaffolds and by checking that the existing registry
covers the score-program layers.
"""

from __future__ import annotations

import json
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
    "tools/*mlx_score_program*.py",
    "tools/*vcm_oracle*.py",
    "tools/*qrepro_plus*.py",
    "tools/*qrepro++*.py",
    "tools/*action_byte*.py",
    "tools/*action_atlas*.py",
    "src/tac/*score_program_compiler*.py",
    "src/tac/*mlx_score_program*.py",
    "src/tac/*vcm_oracle*.py",
    "src/tac/*qrepro_plus*.py",
    "src/tac/*action_byte*.py",
    "src/tac/*action_atlas*.py",
    "src/tac/**/*score_program_compiler*.py",
    "src/tac/**/*mlx_score_program*.py",
    "src/tac/**/*vcm_oracle*.py",
    "src/tac/**/*qrepro_plus*.py",
    "src/tac/**/*action_byte*.py",
    "src/tac/**/*action_atlas*.py",
    "src/comma_lab/*score_program_compiler*.py",
    "src/comma_lab/*mlx_score_program*.py",
    "src/comma_lab/*vcm_oracle*.py",
    "src/comma_lab/*qrepro_plus*.py",
    "src/comma_lab/*action_byte*.py",
    "src/comma_lab/*action_atlas*.py",
    "src/comma_lab/**/*score_program_compiler*.py",
    "src/comma_lab/**/*mlx_score_program*.py",
    "src/comma_lab/**/*vcm_oracle*.py",
    "src/comma_lab/**/*qrepro_plus*.py",
    "src/comma_lab/**/*action_byte*.py",
    "src/comma_lab/**/*action_atlas*.py",
)

MACHINE_VISION_SOURCE_CODE_LINEAGE: tuple[dict[str, Any], ...] = (
    {
        "lineage_id": "quantizr_pr55_pose_conditioned_witness_renderer",
        "source_label": "Quantizr/PR55",
        "discovery": (
            "pose-conditioned single-mask witness renderer with evaluator "
            "rounding/resampling in the training surface"
        ),
        "compiler_layers": (
            "payload_and_residual_basis",
            "action_candidates",
        ),
        "artifact_globs": (
            ".omx/research/*quantizr*",
            "src/tac/*quantizr*.py",
            "src/tac/training_curriculum/quantizr_5_stage_staircase.py",
            "src/tac/packet_compiler/pr81_quantizr.py",
        ),
        "canonical_consumer_source_ids": (
            "feedback_builders",
            "operation_set_compiler",
            "materializer_registry",
        ),
        "consumer_globs": (
            "src/tac/quantizr_faithful_renderer.py",
            "src/tac/quantizr_faithful_export.py",
            "src/comma_lab/scheduler/frontier_rate_attack_feedback.py",
        ),
        "text_refs": (
            {
                "path": "src/comma_lab/scheduler/frontier_rate_attack_feedback.py",
                "patterns": ("Quantizr_TTO_scorer_informed_embedding",),
            },
            {
                "path": "src/tac/quantizr_faithful_renderer.py",
                "patterns": ("build_quantizr_faithful_renderer",),
            },
        ),
    },
    {
        "lineage_id": "qrepro_pr90_semantic_pose_qrgb_program",
        "source_label": "qrepro/PR90",
        "discovery": (
            "semantic mask program plus explicit pose controls and sparse "
            "low-frequency scorer-facing QRGB edits"
        ),
        "compiler_layers": (
            "action_candidates",
            "entropy_grammar",
            "payload_and_residual_basis",
        ),
        "artifact_globs": (
            ".omx/research/public_pr90_qrepro_intake_*.md",
            ".omx/research/pr90_qma9_mask_prior_transfer_worker_*.md",
            ".omx/research/pr85_qrgb*",
            ".omx/research/pr85_stbm1br*",
            "src/tac/stbm1br_mask_codec.py",
            "src/tac/stbm1br_rust_bridge.py",
        ),
        "canonical_consumer_source_ids": (
            "feedback_builders",
            "materializer_registry",
            "operation_set_compiler",
        ),
        "consumer_globs": (
            "src/tac/stbm1br_mask_codec.py",
            "src/tac/tests/test_build_pr85_stbm1br_qrgb_randmulti_stack_candidate.py",
            "src/tac/tests/test_plan_pr85_qrgb_transfer_actions.py",
            "tools/build_public_pr_mining_expansion_backlog.py",
        ),
        "text_refs": (
            {
                "path": "src/tac/stbm1br_mask_codec.py",
                "patterns": ("PR90", "qrepro", "STBM1BR"),
            },
            {
                "path": "src/tac/tests/test_build_pr85_stbm1br_qrgb_randmulti_stack_candidate.py",
                "patterns": ("QRGB", "STBM1BR"),
            },
        ),
    },
    {
        "lineage_id": "pr95_hnerv_distortion_servo_parseback_curriculum",
        "source_label": "PR95",
        "discovery": (
            "HNeRV/Muon control arm with staged scorer curriculum, "
            "QAT/parse-back discipline, and receiver-consumed archive export"
        ),
        "compiler_layers": (
            "payload_and_residual_basis",
            "action_candidates",
        ),
        "artifact_globs": (
            ".omx/research/pr95_*",
            ".omx/research/*pr95*",
            "src/tac/local_acceleration/pr95_hnerv_mlx*.py",
            "src/tac/substrates/hprc/pr95_adapter.py",
            "tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py",
        ),
        "canonical_consumer_source_ids": (
            "feedback_builders",
            "materializer_queue",
            "campaign_planner",
        ),
        "consumer_globs": (
            "src/tac/local_acceleration/pr95_hnerv_mlx.py",
            "src/tac/local_acceleration/pr95_hnerv_mlx_training.py",
            "tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py",
            "src/comma_lab/scheduler/experiment_queue_observer.py",
        ),
        "text_refs": (
            {
                "path": "tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py",
                "patterns": ("PR95", "contest archive"),
            },
            {
                "path": "src/tac/local_acceleration/pr95_hnerv_mlx.py",
                "patterns": ("PR95",),
            },
        ),
    },
    {
        "lineage_id": "pr110_selector_entropy_action_algebra",
        "source_label": "PR110",
        "discovery": (
            "entropy-coded selector/action algebra over scorer free axes, "
            "including frame0/PoseNet and inverse-scorer repair surfaces"
        ),
        "compiler_layers": (
            "action_candidates",
            "entropy_grammar",
        ),
        "artifact_globs": (
            ".omx/research/pr110_current_frontier_final_rate_attack_*",
            ".omx/research/pr110_opt*",
            ".omx/research/*pr110*",
            "src/tac/substrates/pr110_opt*",
            "src/tac/composition/pr110_opt_*",
            "tools/build_frame_exploit_selector_packet.py",
        ),
        "canonical_consumer_source_ids": (
            "feedback_builders",
            "materializer_registry",
            "operation_set_compiler",
            "materializer_queue",
        ),
        "consumer_globs": (
            "src/comma_lab/scheduler/frontier_rate_attack_feedback.py",
            "src/comma_lab/scheduler/byte_shaving_materializer_registry.py",
            "src/tac/optimization/inverse_steganalysis_operation_set_compiler.py",
            "tools/build_frame_exploit_selector_packet.py",
        ),
        "text_refs": (
            {
                "path": "src/comma_lab/scheduler/frontier_rate_attack_feedback.py",
                "patterns": ("PR110", "inverse_scorer", "selector"),
            },
            {
                "path": "src/comma_lab/scheduler/byte_shaving_materializer_registry.py",
                "patterns": ("selector_stream", "inverse_scorer"),
            },
        ),
    },
)

LEGACY_RATE_ATTACK_ADVISORY_SURFACES: tuple[dict[str, Any], ...] = (
    {
        "surface_id": "rate_op1_stable_orbit_packet_diet",
        "source_label": "RATE-OP-1 stable-orbit packet diet",
        "producer_tool": "tools/build_rate_attack_op1_stable_orbit_packet_diet_xray.py",
        "producer_module": "src/tac/contest_exploits/stable_orbit_packet_diet.py",
        "test_path": "src/tac/contest_exploits/tests/test_stable_orbit_packet_diet.py",
        "compiler_layers": ("entropy_grammar",),
        "artifact_globs": (
            "experiments/results/rate_attack_op1_*",
            "reports/rate_attack_op1_*",
        ),
        "text_refs": (
            {
                "path": "src/tac/contest_exploits/stable_orbit_packet_diet.py",
                "patterns": (
                    "cathedral_autopilot_rows",
                    "score_claim",
                    "ready_for_exact_eval_dispatch",
                ),
            },
            {
                "path": "tools/build_rate_attack_op1_stable_orbit_packet_diet_xray.py",
                "patterns": ("build_stable_orbit_packet_diet_xray",),
            },
        ),
    },
    {
        "surface_id": "rate_op2_tropical_argmax_boundary",
        "source_label": "RATE-OP-2 tropical argmax boundary",
        "producer_tool": "tools/build_rate_attack_op2_tropical_argmax_boundary_grammar.py",
        "producer_module": "src/tac/contest_exploits/tropical_argmax_boundary_grammar.py",
        "test_path": "src/tac/contest_exploits/tests/test_tropical_argmax_boundary_grammar.py",
        "compiler_layers": ("action_candidates", "entropy_grammar"),
        "artifact_globs": (
            "experiments/results/rate_attack_op2_*",
            "reports/rate_attack_op2_*",
        ),
        "text_refs": (
            {
                "path": "src/tac/contest_exploits/tropical_argmax_boundary_grammar.py",
                "patterns": (
                    "cathedral_autopilot_rows",
                    "score_claim",
                    "ready_for_exact_eval_dispatch",
                ),
            },
            {
                "path": "tools/build_rate_attack_op2_tropical_argmax_boundary_grammar.py",
                "patterns": ("build_tropical_argmax_boundary_feasibility",),
            },
        ),
    },
    {
        "surface_id": "rate_op3_decoy_mosaic_residual_basis",
        "source_label": "RATE-OP-3 decoy/mosaic residual basis",
        "producer_tool": "tools/build_rate_attack_op3_decoy_mosaic_residual_basis_probe.py",
        "producer_module": "src/tac/contest_exploits/decoy_mosaic_residual_basis.py",
        "test_path": "src/tac/contest_exploits/tests/test_decoy_mosaic_residual_basis.py",
        "compiler_layers": ("payload_and_residual_basis", "entropy_grammar"),
        "artifact_globs": (
            "experiments/results/rate_attack_op3_*",
            "reports/rate_attack_op3_*",
        ),
        "text_refs": (
            {
                "path": "src/tac/contest_exploits/decoy_mosaic_residual_basis.py",
                "patterns": (
                    "cathedral_autopilot_rows",
                    "score_claim",
                    "ready_for_exact_eval_dispatch",
                ),
            },
            {
                "path": "tools/build_rate_attack_op3_decoy_mosaic_residual_basis_probe.py",
                "patterns": ("build_decoy_mosaic_residual_basis_probe",),
            },
        ),
    },
    {
        "surface_id": "rate_attack_autopilot_feature_matrix",
        "source_label": "RATE ACH autopilot feature matrix",
        "producer_tool": "tools/build_rate_attack_autopilot_feature_matrix.py",
        "producer_module": "src/tac/contest_exploits/rate_attack_autopilot_features.py",
        "test_path": "src/tac/contest_exploits/tests/test_rate_attack_autopilot_features.py",
        "compiler_layers": ("action_candidates",),
        "artifact_globs": (
            "reports/rate_attack_autopilot_feature_matrix_*.json",
            "reports/rate_attack_autopilot_feature_matrix_*_cathedral_autopilot_candidates.jsonl",
        ),
        "text_refs": (
            {
                "path": "src/tac/contest_exploits/rate_attack_autopilot_features.py",
                "patterns": (
                    "canonical_consumer",
                    "score_claim",
                    "ready_for_exact_eval_dispatch",
                ),
            },
            {
                "path": "tools/build_rate_attack_autopilot_feature_matrix.py",
                "patterns": ("build_rate_attack_autopilot_feature_matrix",),
            },
        ),
    },
)

LEGACY_RATE_ATTACK_CANONICAL_CONSUMER_PATHS: tuple[str, ...] = (
    "src/tac/contest_exploits/rate_attack_autopilot_features.py",
    "tools/build_rate_attack_autopilot_feature_matrix.py",
    "tools/cathedral_autopilot_autonomous_loop.py",
)

LEGACY_RATE_ATTACK_LANE_ID_TOKENS: tuple[str, ...] = (
    "rate_attack",
    "final_rate",
    "rate_allocator",
    "frontier_rate",
)

REFUSED_DUPLICATE_LANE_CLASSES = frozenset(
    {
        "refused_duplicate_scaffold",
        "stale_duplicate_placeholder",
    }
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


def _unique_paths(paths: Iterable[str]) -> list[str]:
    return sorted(dict.fromkeys(paths))


def _read_text_or_empty(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _required_source_exists_by_id(repo_root: Path) -> dict[str, bool]:
    return {
        row["surface_id"]: bool(row["exists"])
        for row in _required_surface_rows(repo_root)
    }


def _lineage_text_ref_rows(
    repo_root: Path,
    refs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ref in refs:
        rel_path = str(ref.get("path") or "").strip()
        patterns = tuple(str(pattern) for pattern in ref.get("patterns") or ())
        path = repo_root / rel_path
        text = _read_text_or_empty(path)
        present_patterns = [pattern for pattern in patterns if pattern in text]
        rows.append(
            {
                "path": rel_path,
                "exists": path.is_file(),
                "patterns": list(patterns),
                "present_patterns": present_patterns,
                "missing_patterns": [
                    pattern for pattern in patterns if pattern not in present_patterns
                ],
                "covered": path.is_file() and len(present_patterns) == len(patterns),
            }
        )
    return rows


def _machine_vision_lineage_rows(repo_root: Path) -> list[dict[str, Any]]:
    required_source_exists = _required_source_exists_by_id(repo_root)
    known_layers = set(SCORE_PROGRAM_LAYER_TARGET_KINDS)
    rows: list[dict[str, Any]] = []
    for spec in MACHINE_VISION_SOURCE_CODE_LINEAGE:
        artifact_paths = _unique_paths(
            path
            for pattern in spec.get("artifact_globs") or ()
            for path in _glob_paths(repo_root, str(pattern))
        )
        consumer_paths = _unique_paths(
            path
            for pattern in spec.get("consumer_globs") or ()
            for path in _glob_paths(repo_root, str(pattern))
        )
        canonical_source_ids = tuple(
            str(source_id)
            for source_id in spec.get("canonical_consumer_source_ids") or ()
        )
        missing_canonical_sources = [
            source_id
            for source_id in canonical_source_ids
            if required_source_exists.get(source_id) is not True
        ]
        compiler_layers = tuple(
            str(layer_id) for layer_id in spec.get("compiler_layers") or ()
        )
        missing_layers = [
            layer_id for layer_id in compiler_layers if layer_id not in known_layers
        ]
        text_ref_rows = _lineage_text_ref_rows(repo_root, spec.get("text_refs") or ())
        missing_text_refs = [
            row for row in text_ref_rows if row.get("covered") is not True
        ]
        blockers: list[str] = []
        if not artifact_paths:
            blockers.append("lineage_artifact_surface_missing")
        if not consumer_paths:
            blockers.append("lineage_consumer_surface_missing")
        blockers.extend(
            f"canonical_consumer_source_missing:{source_id}"
            for source_id in missing_canonical_sources
        )
        blockers.extend(f"compiler_layer_unknown:{layer_id}" for layer_id in missing_layers)
        blockers.extend(
            "text_ref_missing:"
            f"{row.get('path')}:{','.join(map(str, row.get('missing_patterns') or []))}"
            for row in missing_text_refs
        )
        rows.append(
            {
                "lineage_id": spec.get("lineage_id"),
                "source_label": spec.get("source_label"),
                "discovery": spec.get("discovery"),
                "compiler_layers": list(compiler_layers),
                "artifact_globs": list(spec.get("artifact_globs") or ()),
                "artifact_count": len(artifact_paths),
                "artifact_sample_paths": artifact_paths[:20],
                "canonical_consumer_source_ids": list(canonical_source_ids),
                "missing_canonical_consumer_source_ids": missing_canonical_sources,
                "consumer_globs": list(spec.get("consumer_globs") or ()),
                "consumer_count": len(consumer_paths),
                "consumer_sample_paths": consumer_paths[:20],
                "text_ref_rows": text_ref_rows,
                "status": "consumed_by_canonical_stack" if not blockers else "blocked",
                "blockers": blockers,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return rows


def _machine_vision_lineage_blockers(
    lineage_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for row in lineage_rows:
        lineage_id = row.get("lineage_id")
        blockers.extend(
            f"machine_vision_lineage_blocked:{lineage_id}:{blocker}"
            for blocker in row.get("blockers") or []
        )
    return blockers


def _legacy_rate_attack_advisory_rows(repo_root: Path) -> list[dict[str, Any]]:
    known_layers = set(SCORE_PROGRAM_LAYER_TARGET_KINDS)
    consumer_rows = [
        {
            "path": rel_path,
            "exists": (repo_root / rel_path).is_file(),
        }
        for rel_path in LEGACY_RATE_ATTACK_CANONICAL_CONSUMER_PATHS
    ]
    missing_consumers = [
        str(row["path"]) for row in consumer_rows if row["exists"] is not True
    ]
    rows: list[dict[str, Any]] = []
    for spec in LEGACY_RATE_ATTACK_ADVISORY_SURFACES:
        producer_tool = str(spec.get("producer_tool") or "")
        producer_module = str(spec.get("producer_module") or "")
        test_path = str(spec.get("test_path") or "")
        compiler_layers = tuple(
            str(layer_id) for layer_id in spec.get("compiler_layers") or ()
        )
        missing_layers = [
            layer_id for layer_id in compiler_layers if layer_id not in known_layers
        ]
        artifact_paths = _unique_paths(
            path
            for pattern in spec.get("artifact_globs") or ()
            for path in _glob_paths(repo_root, str(pattern))
        )
        text_ref_rows = _lineage_text_ref_rows(repo_root, spec.get("text_refs") or ())
        missing_text_refs = [
            row for row in text_ref_rows if row.get("covered") is not True
        ]
        path_rows = [
            {"role": "producer_tool", "path": producer_tool},
            {"role": "producer_module", "path": producer_module},
            {"role": "test_path", "path": test_path},
        ]
        path_rows = [
            {
                **row,
                "exists": bool(row["path"]) and (repo_root / row["path"]).is_file(),
            }
            for row in path_rows
        ]
        blockers: list[str] = []
        blockers.extend(
            f"advisory_surface_path_missing:{row['role']}:{row['path']}"
            for row in path_rows
            if row["exists"] is not True
        )
        blockers.extend(
            f"canonical_consumer_missing:{consumer}" for consumer in missing_consumers
        )
        blockers.extend(f"compiler_layer_unknown:{layer_id}" for layer_id in missing_layers)
        blockers.extend(
            "text_ref_missing:"
            f"{row.get('path')}:{','.join(map(str, row.get('missing_patterns') or []))}"
            for row in missing_text_refs
        )
        rows.append(
            {
                "surface_id": spec.get("surface_id"),
                "source_label": spec.get("source_label"),
                "status": (
                    "planning_only_consumed_by_canonical_stack"
                    if not blockers
                    else "blocked"
                ),
                "producer_tool": producer_tool,
                "producer_module": producer_module,
                "test_path": test_path,
                "path_rows": path_rows,
                "compiler_layers": list(compiler_layers),
                "artifact_globs": list(spec.get("artifact_globs") or ()),
                "artifact_count": len(artifact_paths),
                "artifact_sample_paths": artifact_paths[:20],
                "canonical_consumer_paths": list(
                    LEGACY_RATE_ATTACK_CANONICAL_CONSUMER_PATHS
                ),
                "canonical_consumer_rows": consumer_rows,
                "text_ref_rows": text_ref_rows,
                "blockers": blockers,
                "research_only": True,
                "planning_only": True,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        )
    return rows


def _legacy_rate_attack_advisory_blockers(
    advisory_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for row in advisory_rows:
        surface_id = row.get("surface_id")
        blockers.extend(
            f"legacy_rate_attack_advisory_blocked:{surface_id}:{blocker}"
            for blocker in row.get("blockers") or []
        )
    return blockers


def _legacy_rate_attack_lane_registry_rows(repo_root: Path) -> list[dict[str, Any]]:
    registry = _read_json_mapping(repo_root / ".omx/state/lane_registry.json")
    lanes = registry.get("lanes")
    if not isinstance(lanes, list):
        return []
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        if not isinstance(lane, Mapping):
            continue
        lane_id = str(lane.get("id") or "")
        if not any(token in lane_id for token in LEGACY_RATE_ATTACK_LANE_ID_TOKENS):
            continue
        gates = lane.get("gates")
        gate_rows = gates if isinstance(gates, Mapping) else {}
        true_gates = sorted(
            str(gate_id)
            for gate_id, gate_payload in gate_rows.items()
            if isinstance(gate_payload, Mapping)
            and gate_payload.get("status") is True
        )
        lane_class = str(lane.get("lane_class") or "").strip()
        research_only = lane.get("research_only") is True
        notes = str(lane.get("notes") or "")
        note_lc = notes.lower()
        duplicate_noted = "duplicate" in note_lc or "superseded" in note_lc
        blockers: list[str] = []
        if not true_gates and not lane_class and not research_only:
            blockers.append("zero_gate_legacy_lane_unclassified")
        if duplicate_noted and lane_class not in REFUSED_DUPLICATE_LANE_CLASSES:
            blockers.append("duplicate_or_superseded_lane_not_refused")
        if lane_class in REFUSED_DUPLICATE_LANE_CLASSES and not research_only:
            blockers.append("refused_duplicate_lane_not_research_only")
        if blockers:
            status = "blocked"
        elif lane_class in REFUSED_DUPLICATE_LANE_CLASSES:
            status = "refused_duplicate_scaffold"
        elif true_gates:
            status = "evidence_recorded"
        else:
            status = "classified_research_only"
        rows.append(
            {
                "lane_id": lane_id,
                "name": lane.get("name"),
                "level": lane.get("level"),
                "phase": lane.get("phase"),
                "lane_class": lane_class,
                "research_only": research_only,
                "true_gates": true_gates,
                "notes": notes,
                "status": status,
                "blockers": blockers,
            }
        )
    return rows


def _legacy_rate_attack_lane_registry_blockers(
    lane_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    for row in lane_rows:
        lane_id = row.get("lane_id")
        blockers.extend(
            f"legacy_rate_attack_lane_registry_blocked:{lane_id}:{blocker}"
            for blocker in row.get("blockers") or []
        )
    return blockers


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
        registered_coverage = (
            not missing
            and len(adapter_rows) == len(target_kinds)
            and all(row.get("adapter_registered") is True for row in adapter_rows)
        )
        candidate_archive_coverage = bool(
            registered_coverage
            and all(
                row.get("executable_candidate_archive") is True
                for row in adapter_rows
            )
        )
        receiver_contract_coverage = bool(
            registered_coverage and receiver_contract_count == len(target_kinds)
        )
        receiver_proof_coverage = bool(
            registered_coverage
            and all(
                (
                    row.get("receiver_proof_function_bound") is True
                    and row.get("receiver_verify_function_bound") is True
                )
                for row in adapter_rows
                if row.get("executable_candidate_archive") is True
            )
        )
        production_action_coverage = bool(
            registered_coverage
            and candidate_archive_coverage
            and receiver_contract_coverage
            and receiver_proof_coverage
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
                "registered_coverage": registered_coverage,
                "candidate_archive_coverage": candidate_archive_coverage,
                "receiver_contract_coverage": receiver_contract_coverage,
                "receiver_proof_coverage": receiver_proof_coverage,
                "production_action_coverage": production_action_coverage,
                "covered": production_action_coverage,
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
    lineage_rows = _machine_vision_lineage_rows(root)
    lineage_blockers = _machine_vision_lineage_blockers(lineage_rows)
    advisory_rows = _legacy_rate_attack_advisory_rows(root)
    advisory_blockers = _legacy_rate_attack_advisory_blockers(advisory_rows)
    legacy_lane_rows = _legacy_rate_attack_lane_registry_rows(root)
    legacy_lane_blockers = _legacy_rate_attack_lane_registry_blockers(
        legacy_lane_rows
    )

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
    production_action_blockers: list[str] = []
    for row in layer_rows:
        if row.get("candidate_archive_coverage") is not True:
            production_action_blockers.append(
                f"score_program_layer_has_non_executable_targets:{row['layer_id']}"
            )
        if row.get("receiver_proof_coverage") is not True:
            production_action_blockers.append(
                f"score_program_layer_receiver_proof_coverage_incomplete:{row['layer_id']}"
            )
        for adapter in row.get("adapter_rows") or []:
            if not isinstance(adapter, Mapping):
                continue
            target = adapter.get("target_kind")
            materializer = adapter.get("materializer_id")
            if adapter.get("planning_only") is True:
                production_action_blockers.append(
                    "score_program_target_planning_only:"
                    f"{row['layer_id']}:{target}:{materializer}"
                )
            if adapter.get("executable_candidate_archive") is not True:
                production_action_blockers.append(
                    "score_program_target_not_executable_candidate_archive:"
                    f"{row['layer_id']}:{target}:{materializer}"
                )
            if (
                adapter.get("executable_candidate_archive") is True
                and adapter.get("materialize_function_bound") is not True
            ):
                production_action_blockers.append(
                    "score_program_target_materialize_function_missing:"
                    f"{row['layer_id']}:{target}:{materializer}"
                )
            if (
                adapter.get("executable_candidate_archive") is True
                and (
                    adapter.get("receiver_proof_function_bound") is not True
                    or adapter.get("receiver_verify_function_bound") is not True
                )
            ):
                production_action_blockers.append(
                    "score_program_target_receiver_proof_or_verify_missing:"
                    f"{row['layer_id']}:{target}:{materializer}"
                )
    blockers.extend(
        f"parallel_score_program_compiler_surface_forbidden:{path}"
        for path in forbidden_surfaces
    )
    blockers.extend(lineage_blockers)
    blockers.extend(advisory_blockers)
    blockers.extend(legacy_lane_blockers)
    production_action_ready = not production_action_blockers

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
        "production_action_ready": production_action_ready,
        "production_action_blockers": production_action_blockers,
        "machine_vision_source_code_lineage": {
            "schema": "machine_vision_source_code_lineage_consumption.v1",
            "hidden_object": (
                "compressed evaluator-equivalent witness program whose RGB "
                "frames carry the exact SegNet/PoseNet messages"
            ),
            "rows": lineage_rows,
            "blockers": lineage_blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "legacy_rate_attack_advisory_surfaces": {
            "schema": "legacy_rate_attack_advisory_surface_consumption.v1",
            "canonical_consumer_paths": list(
                LEGACY_RATE_ATTACK_CANONICAL_CONSUMER_PATHS
            ),
            "rows": advisory_rows,
            "blockers": advisory_blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "legacy_rate_attack_lane_registry": {
            "schema": "legacy_rate_attack_lane_registry_canonicalization.v1",
            "id_tokens": list(LEGACY_RATE_ATTACK_LANE_ID_TOKENS),
            "rows": legacy_lane_rows,
            "blockers": legacy_lane_blockers,
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
        if missing:
            suffix = f"missing={','.join(map(str, missing))}"
        elif row.get("production_action_coverage") is True:
            suffix = "production-covered"
        elif row.get("registered_coverage") is True:
            suffix = "registered-only"
        else:
            suffix = "not-covered"
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
    production_blockers = audit.get("production_action_blockers") or []
    lines.append(
        "production_action: "
        f"{'ready' if audit.get('production_action_ready') is True else 'blocked'}; "
        f"blockers={len(production_blockers)}"
    )
    lineage = audit.get("machine_vision_source_code_lineage")
    if isinstance(lineage, Mapping):
        rows = [row for row in lineage.get("rows") or [] if isinstance(row, Mapping)]
        blockers = lineage.get("blockers") or []
        lines.append(
            "machine_vision_source_code_lineage: "
            f"{len(rows)} signal(s); blockers={len(blockers)}"
        )
        for row in rows:
            lines.append(
                f"  - {row.get('source_label')}: {row.get('status')}; "
                f"artifacts={row.get('artifact_count', 0)}; "
                f"consumers={row.get('consumer_count', 0)}; "
                f"layers={','.join(map(str, row.get('compiler_layers') or []))}"
            )
    advisory = audit.get("legacy_rate_attack_advisory_surfaces")
    if isinstance(advisory, Mapping):
        rows = [row for row in advisory.get("rows") or [] if isinstance(row, Mapping)]
        blockers = advisory.get("blockers") or []
        lines.append(
            "legacy_rate_attack_advisory_surfaces: "
            f"{len(rows)} signal(s); blockers={len(blockers)}"
        )
        for row in rows:
            lines.append(
                f"  - {row.get('source_label')}: {row.get('status')}; "
                f"artifacts={row.get('artifact_count', 0)}; "
                f"layers={','.join(map(str, row.get('compiler_layers') or []))}"
            )
    lane_registry = audit.get("legacy_rate_attack_lane_registry")
    if isinstance(lane_registry, Mapping):
        rows = [
            row
            for row in lane_registry.get("rows") or []
            if isinstance(row, Mapping)
        ]
        blockers = lane_registry.get("blockers") or []
        zero_gate_count = sum(1 for row in rows if not row.get("true_gates"))
        refused_count = sum(
            1
            for row in rows
            if row.get("status") == "refused_duplicate_scaffold"
        )
        lines.append(
            "legacy_rate_attack_lane_registry: "
            f"{len(rows)} lane(s); blockers={len(blockers)}; "
            f"zero_gate={zero_gate_count}; refused_duplicate={refused_count}"
        )
        for row in rows:
            lines.append(
                f"  - {row.get('lane_id')}: {row.get('status')}; "
                f"level={row.get('level')}; "
                f"class={row.get('lane_class') or 'untyped'}; "
                f"gates={','.join(map(str, row.get('true_gates') or [])) or 'none'}"
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
