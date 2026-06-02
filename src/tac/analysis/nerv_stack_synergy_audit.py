# SPDX-License-Identifier: MIT
"""HiNeRV/SNeRV full-stack synergy audit.

This is a planner-facing audit, not a score artifact. It binds three sources of
truth into one reusable payload:

* local implementation surfaces and partial/stub markers;
* dated `.omx/research` design/result memos;
* upstream NeRV-family controls and byte-budget candidate planners.

The goal is to prevent arbitrary launch choices: the compact runner should know
which knobs are real, which are report-only, and which exact blockers remain
before a carrier can spend contest auth evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tac.analysis.nerv_modelsize_budget import (
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    official_nerv_oss_flag_audit,
)
from tac.analysis.pr95_stack_binding_requirements import (
    build_pr95_stack_binding_evidence,
    build_pr95_stack_binding_requirements,
)

SCHEMA = "nerv_stack_synergy_audit.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

PARTIAL_MARKERS = (
    "L0 SKETCH",
    "SCAFFOLD",
    "NotImplementedError",
    "not implemented",
    "TODO",
    "FIXME",
    "missing",
    "pending",
    "blocker",
)

HI_NERV_SURFACES = (
    "src/tac/substrates/hi_nerv/architecture.py",
    "src/tac/substrates/hi_nerv/mlx_renderer.py",
    "src/tac/substrates/hi_nerv/archive.py",
    "src/tac/substrates/hi_nerv/archive_candidate.py",
    "src/tac/substrates/hi_nerv/inflate.py",
    "src/tac/substrates/hi_nerv/score_aware_loss.py",
    "src/tac/analysis/hinerv_latent_linf_allocation.py",
    "tools/run_compact_renderer_mlx_spine_runner.py",
    "experiments/train_hinerv_as_renderer.py",
)

SNERV_SURFACES = (
    "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/dwt.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/allocation.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/archive.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/archive_candidate.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/inflate.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
    "src/tac/analysis/snerv_rate_adjudication.py",
    "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
    "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
    "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
    "tools/run_snerv_inverse_steg_advisory.py",
    "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
    "tools/prove_snerv_receiver_archive.py",
)

SHARED_SYNERGY_SURFACES = (
    "src/tac/substrates/_shared/mlx_score_aware/harness.py",
    "src/tac/substrates/_shared/mlx_score_aware/loss.py",
    "src/tac/substrates/_shared/mlx_score_aware/coder_qat.py",
    "src/tac/substrates/_shared/mlx_score_aware/pr95_faithful_curriculum.py",
    "src/tac/substrates/_shared/mlx_score_aware/portability.py",
    "src/tac/substrates/_shared/decoder_state_codec.py",
    "src/tac/analysis/score_exact_saliency.py",
    "src/tac/analysis/nerv_top_priority_stack_seam.py",
    "src/tac/analysis/nerv_modelsize_budget.py",
    "src/tac/analysis/nerv_candidate_curriculum.py",
    "tools/run_mlx_scorer_response_cache.py",
    "tools/profile_mlx_scorer_response_cache.py",
)


def build_nerv_stack_synergy_audit(
    *,
    repo_root: str | Path,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    memo_limit_per_stack: int = 40,
    marker_limit_per_stack: int = 80,
) -> dict[str, Any]:
    """Build a fail-closed audit for HiNeRV and SNeRV stack readiness."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"repo_root is not a directory: {root}")
    oss = official_nerv_oss_flag_audit()
    shared = [_file_record(root, rel) for rel in SHARED_SYNERGY_SURFACES]
    stacks = [
        _hi_nerv_stack_audit(
            root=root,
            hard_byte_ceilings=hard_byte_ceilings,
            num_pairs=num_pairs,
            oss=oss,
            memo_limit=memo_limit_per_stack,
            marker_limit=marker_limit_per_stack,
        ),
        _snerv_stack_audit(
            root=root,
            hard_byte_ceilings=hard_byte_ceilings,
            num_pairs=num_pairs,
            oss=oss,
            memo_limit=memo_limit_per_stack,
            marker_limit=marker_limit_per_stack,
        ),
    ]
    blockers = sorted({blocker for row in stacks for blocker in row["blockers"]})
    return {
        "schema": SCHEMA,
        "audit_kind": "hinerv_snerv_full_stack_synergy",
        "repo_root": root.as_posix(),
        "num_pairs": int(num_pairs),
        "hard_byte_ceilings": [int(value) for value in hard_byte_ceilings],
        "upstream_oss_sources": dict(oss["sources"]),
        "shared_synergy_surfaces": shared,
        "shared_synergy_surface_count": sum(1 for row in shared if row["present"]),
        "stacks": stacks,
        "blockers": blockers,
        "planner_policy": {
            "planner_and_curriculum_are_coupled": True,
            "launch_capacity_from_byte_budget_candidates": True,
            "curriculum_must_consume_scorer_and_coder_feedback": True,
            "preserve_over_ceiling_structural_blockers": True,
            "exact_auth_only_after_byte_closed_local_winner": True,
        },
        "next_required_integration": [
            "bind --execute-family hi_nerv modelsize candidate to measured trained archive byte oracle",
            "bind --execute-family snerv modelsize candidate to measured SNAR1 archive byte oracle",
            "write candidate curriculum byte-feedback rows into planner posteriors",
            "use replay scorer deltas to update candidate curriculum schedules",
            "write every local replay result back into planner posteriors",
        ],
        **FALSE_AUTHORITY,
    }


def _hi_nerv_stack_audit(
    *,
    root: Path,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    oss: dict[str, Any],
    memo_limit: int,
    marker_limit: int,
) -> dict[str, Any]:
    files = [_file_record(root, rel) for rel in HI_NERV_SURFACES]
    memos = _related_memos(
        root,
        keywords=("hinerv", "hi_nerv", "hnerv", "pr95", "srnerv", "rnerv", "nerv"),
        limit=memo_limit,
    )
    markers = _partial_markers(root, [row["rel_path"] for row in files], limit=marker_limit)
    budget = build_hinerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=num_pairs,
        per_ceiling_limit=6,
    )
    pr95_binding = build_pr95_stack_binding_requirements(
        family="hi_nerv",
        evidence=build_pr95_stack_binding_evidence(
            differentiable_pose_preprocess=True,
            eval_roundtrip_ste=True,
            ema_archive_selection=True,
            qat_forward=True,
            coder_aware_regularizer=True,
        ),
    )
    blockers = [
        "hinerv_modelsize_candidate_consumption_requires_trained_archive_byte_oracle",
        "hinerv_local_architecture_not_source_faithful_upstream_hinerv_feature_grid",
        "hinerv_official_convnext_feature_grid_path_missing",
        "hinerv_official_trilinear_feature_interpolation_path_missing",
        "hinerv_upstream_grid_depth_prune_bitstream_controls_not_executable_atoms",
        "hinerv_official_pruning_control_not_bound_to_planner",
        "hinerv_official_quantnoise_control_not_bound_to_mlx_trainer",
        "hinerv_torchac_style_bitstream_pipeline_missing",
        "hinerv_decoder_weight_saliency_waterfill_not_in_trainer",
        "hinerv_pr95_c1a_muon_ema_archive_schedule_not_fully_bound_to_real_trainer",
        "hinerv_legacy_full_trainer_archive_is_not_hermetic",
        "hinerv_receiver_load_strict_false_schema_drift_risk",
        "hinerv_legacy_trainer_rate_proxy_not_codec_matched",
        "hinerv_exact_cpu_cuda_eval_missing_for_current_candidates",
        *pr95_binding["blockers"],
    ]
    if markers:
        blockers.append("hinerv_partial_or_stub_markers_present")
    return {
        "schema": "nerv_stack_synergy_stack_row.v1",
        "stack_id": "hi_nerv",
        "priority": "top_priority_carrier",
        "local_status": "mlx_train_export_adapter_present_but_not_full_upstream_control_surface",
        "source_faithfulness": {
            "source_faithful_upstream_hinerv": False,
            "local_role": "contest_adapter_with_hierarchical_latent_pyramid",
            "missing_upstream_axes": [
                "hierarchical feature-grid encoding",
                "official ConvNeXt-style feature-grid path",
                "official trilinear feature interpolation path",
                "patch/frame unified training and eval",
                "adaptive pruning schedule",
                "QuantNoise-controlled source-faithful training path",
                "QAT bitstream-q torchac-style entropy closure",
            ],
        },
        "upstream_controls": {
            "hnerv_flags": list(oss["hnerv_high_ev_flags"]),
            "hinerv_flags": list(oss["hinerv_high_ev_flags"]),
            "cross_variant_priors": _priors_for_stack(oss, stack_id="hi_nerv"),
        },
        "local_surface_files": files,
        "related_memos": memos,
        "partial_markers": markers,
        "modelsize_budget": budget,
        "pr95_stack_binding": pr95_binding,
        "planner_curriculum_links": [
            "byte ceiling selects capacity candidate before launch",
            "PR95-faithful staged curriculum is required for non-smoke runs",
            "real SegNet/PoseNet teachers must both be positive for frontier-targeted runs",
            "PR95 eval-roundtrip STE is attached in the shared MLX harness for HiNeRV execution",
            "Pose student consumes canonical differentiable PR95 YUV6 preprocessing in HiNeRV execution",
            "final live-vs-EMA archive selection is manifest-backed in the shared long-training helper",
            "decoder fake-quant forward is wired; exact byte oracle still chooses int2/int4/int8/fp16 per candidate",
            "coder-aware regularization and recon_pixel_weight must be candidate-specific",
            "decoder-weight saliency/waterfill must feed the real MLX trainer, not only posthoc export",
            "PR95 C1a/Muon/EMA/archive-eval stages remain mandatory control-arm bindings, not optional polish",
        ],
        "anti_arbitrariness_requirements": [
            "no arbitrary latent_dim/embed_dim/decoder_channel launch without budget row",
            "no uniform int2/int4/int8 decision without scorer replay and archive byte oracle",
            "no exact spend without full-video MLX prefilter plus local CPU replay",
        ],
        "adversarial_review_findings": [
            {
                "severity": "P0",
                "finding": "legacy train_substrate_hi_nerv archive is non-hermetic",
                "evidence": "experiments/train_substrate_hi_nerv.py writes inflate.py imports from HERE/src/tac but zips only 0.bin/inflate.sh/inflate.py",
                "required_gate": "clean-temp unzip inflate smoke with no repo PYTHONPATH",
            },
            {
                "severity": "P1",
                "finding": "receiver load uses strict=False and can hide schema drift",
                "evidence": "src/tac/substrates/hi_nerv/inflate.py loads decoder_state with strict=False",
                "required_gate": "missing or extra decoder key must fail receiver proof",
            },
            {
                "severity": "P1",
                "finding": "legacy training rate proxy is not codec matched",
                "evidence": "legacy trainer uses closed-form fp16/int16 estimate while export path can use int8/int4/int2 codecs",
                "required_gate": "rate proxy versus packed archive byte oracle drift report",
            },
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _snerv_stack_audit(
    *,
    root: Path,
    hard_byte_ceilings: tuple[int, ...],
    num_pairs: int,
    oss: dict[str, Any],
    memo_limit: int,
    marker_limit: int,
) -> dict[str, Any]:
    files = [_file_record(root, rel) for rel in SNERV_SURFACES]
    memos = _related_memos(
        root,
        keywords=("snerv", "srnerv", "sr-nerv", "rnerv", "ffnerv", "boostnerv", "nerv"),
        limit=memo_limit,
    )
    markers = _partial_markers(root, [row["rel_path"] for row in files], limit=marker_limit)
    budget = build_snerv_modelsize_budget_report(
        hard_byte_ceilings=hard_byte_ceilings,
        num_pairs=num_pairs,
        per_ceiling_limit=6,
    )
    pr95_binding = build_pr95_stack_binding_requirements(
        family="snerv",
        evidence=build_pr95_stack_binding_evidence(),
    )
    blockers = [
        "snerv_mlx_native_train_export_adapter_missing",
        "snerv_execute_family_uses_cpu_advisory_not_mlx_native_training",
        "snerv_modelsize_candidate_consumption_requires_real_snar1_archive_byte_oracle",
        "snerv_local_carrier_not_source_faithful_official_snerv_multilayer_stack",
        "snerv_official_mfu_block_missing",
        "snerv_official_hfr_block_missing",
        "snerv_official_snerv_t_temporal_path_missing",
        "snerv_official_haar_j1_parity_missing",
        "snerv_receiver_dwt_custody_missing",
        "snerv_fc_dim_modelsize_control_not_bound_to_planner",
        "snerv_scorer_loop_qat_is_bounded_smoke_not_production_trainer",
        "snerv_l2_linf_receiver_packet_rate_accounting_not_separated",
        "snerv_learned_scorer_preserving_lf_hf_generation_missing",
        "snerv_gate_defaults_are_not_frontier_provenance_driven",
        "snerv_work_orders_are_textual_not_queue_actuated",
        "snerv_exact_cpu_cuda_eval_missing_for_current_candidates",
        *pr95_binding["blockers"],
    ]
    if markers:
        blockers.append("snerv_partial_or_stub_markers_present")
    return {
        "schema": "nerv_stack_synergy_stack_row.v1",
        "stack_id": "snerv",
        "priority": "top_priority_carrier",
        "local_status": "receiver_bound_advisory_export_present_mlx_native_train_missing",
        "source_faithfulness": {
            "source_faithful_official_snerv": False,
            "local_role": "contest_adapter_store_lf_generate_hf_snar1_packet",
            "missing_upstream_axes": [
                "official multi-layer scalable neural representation",
                "MFU/HFR/TUB-style blocks",
                "temporal SNeRV-T/SNeRV-T-2D training path",
                "official Haar/J=1 mode parity",
                "receiver DWT custody and source/runtime hash proof",
                "fc_dim/modelsize capacity control",
                "source-faithful MLX train/export/archive adapter",
            ],
        },
        "upstream_controls": {
            "snerv_flags": list(oss["snerv_high_ev_flags"]),
            "cross_variant_priors": _priors_for_stack(oss, stack_id="snerv"),
        },
        "local_surface_files": files,
        "related_memos": memos,
        "partial_markers": markers,
        "modelsize_budget": budget,
        "pr95_stack_binding": pr95_binding,
        "planner_curriculum_links": [
            "LF level and precision select rate before advisory launch",
            "HF decoder QAT must be trained under PoseNet hard guard",
            "step-map precision must be priced as receiver-visible payload",
            "L2 and L-infinity receiver-packet bytes must be accounted separately",
            "source-faithful Haar/J=1 and official MFU/HFR/SNeRV-T controls must arbitrate before method demotion",
            "mixed decoder modes stay alive until local scorer replay arbitrates",
        ],
        "anti_arbitrariness_requirements": [
            "no fixed levels=3 bits=2.5 launch without budget candidate",
            "no posthoc byte-saving demotion when bytes may compose with later training",
            "no method-negative verdict until official controls and separate L2/L-infinity packet bytes are profiled",
            "no promotion from CPU advisory SNAR1 packet without contest archive/local replay",
        ],
        "adversarial_review_findings": [
            {
                "severity": "P0",
                "finding": "scorer-loop/QAT is a bounded advisory smoke, not production trainer",
                "evidence": "scorer_loop_decoder_qat.py defaults to tiny pair/trial counts and perturbation search",
                "required_gate": "real MLX-native scorer-loop trainer with receiver export proof",
            },
            {
                "severity": "P2",
                "finding": "rate/pose/seg gate defaults are hard-coded",
                "evidence": "rate adjudication, pose gate, work order, and mode probe carry local constants",
                "required_gate": "current frontier/planner config input with manifest provenance",
            },
            {
                "severity": "P2",
                "finding": "work orders and pose gates are fail-closed prose-like outputs",
                "evidence": "recommended command strings are emitted but not queue-actuated",
                "required_gate": "experiment_queue.v1 work rows with byte/SHA/runtime blockers",
            },
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _priors_for_stack(oss: dict[str, Any], *, stack_id: str) -> list[dict[str, Any]]:
    key = "transfer_to_hinerv" if stack_id == "hi_nerv" else "transfer_to_snerv"
    out = []
    for row in oss.get("cross_variant_design_priors", []):
        if isinstance(row, dict) and row.get(key):
            out.append(
                {
                    "variant": row.get("variant"),
                    "role": row.get("role"),
                    "transfer": list(row.get(key, [])),
                }
            )
    return out


def _file_record(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    present = path.is_file()
    record: dict[str, Any] = {
        "rel_path": rel_path,
        "present": present,
    }
    if present:
        data = path.read_bytes()
        record.update(
            {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return record


def _related_memos(
    root: Path,
    *,
    keywords: tuple[str, ...],
    limit: int,
) -> list[dict[str, Any]]:
    research = root / ".omx" / "research"
    if not research.is_dir():
        return []
    lowered = tuple(keyword.lower() for keyword in keywords)
    rows = []
    for path in sorted(research.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(keyword in name for keyword in lowered):
            rows.append(_file_record(root, path.relative_to(root).as_posix()))
    return rows[-max(int(limit), 0) :]


def _partial_markers(root: Path, rel_paths: Iterable[str], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        path = root / rel_path
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            marker = _line_marker(line)
            if marker is None:
                continue
            rows.append(
                {
                    "rel_path": rel_path,
                    "line": lineno,
                    "marker": marker,
                    "excerpt": line.strip()[:180],
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def _line_marker(line: str) -> str | None:
    lowered = line.lower()
    for marker in PARTIAL_MARKERS:
        if marker.lower() in lowered:
            return marker
    return None


__all__ = [
    "SCHEMA",
    "build_nerv_stack_synergy_audit",
]
