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
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.analysis.nerv_modelsize_budget import (
    MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS,
    build_hinerv_modelsize_budget_report,
    build_snerv_modelsize_budget_report,
    official_nerv_oss_flag_audit,
)
from tac.analysis.pr95_stack_binding_requirements import (
    build_pr95_stack_binding_evidence,
    build_pr95_stack_binding_requirements,
)
from tac.analysis.snerv_official_primitive_replay import (
    build_snerv_official_primitive_replay_binding,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_adapter_contract import (
    build_snerv_mlx_native_adapter_contract,
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
    "src/tac/substrates/hi_nerv/bitstream.py",
    "src/tac/substrates/hi_nerv/official_grid.py",
    "src/tac/substrates/hi_nerv/inflate.py",
    "src/tac/substrates/hi_nerv/score_aware_loss.py",
    "src/tac/analysis/nerv_decoder_weight_waterfill.py",
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
    "src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_adapter_contract.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/official_hfr.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py",
    "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
    "src/tac/analysis/snerv_rate_adjudication.py",
    "src/tac/analysis/nerv_modelsize_budget.py",
    "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
    "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
    "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
    "tools/build_nerv_modelsize_budget.py",
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
    hinerv_official_source_audit: Mapping[str, Any] | None = None,
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
            hinerv_official_source_audit=hinerv_official_source_audit,
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
    blockers = sorted(
        {
            blocker
            for row in stacks
            for blocker in row["blockers"]
            if str(blocker)
        }
    )
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
    hinerv_official_source_audit: Mapping[str, Any] | None,
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
    modelsize_budget_binding = _hi_nerv_modelsize_budget_binding(budget)
    quantnoise_binding = _hi_nerv_quantnoise_control_binding(root)
    official_grid_binding = _hi_nerv_official_grid_binding(root)
    official_feature_grid_convnext_binding = (
        _hi_nerv_official_feature_grid_convnext_binding(root)
    )
    official_patch_binding = _hi_nerv_official_patch_binding(root)
    official_source_audit_binding = _hi_nerv_official_source_audit_binding(
        hinerv_official_source_audit
    )
    strict_receiver_binding = _hi_nerv_strict_receiver_load_binding(root)
    archive_candidate_binding = _hi_nerv_archive_candidate_binding(root)
    pr95_binding = build_pr95_stack_binding_requirements(
        family="hi_nerv",
        evidence=build_pr95_stack_binding_evidence(
            differentiable_pose_preprocess=True,
            eval_roundtrip_ste=True,
            ema_archive_selection=True,
            qat_forward=True,
            coder_aware_regularizer=True,
            modelsize_archive_budget=modelsize_budget_binding[
                "modelsize_archive_budget_bound"
            ],
            byte_closed_archive_export=archive_candidate_binding[
                "byte_closed_archive_export_bound"
            ],
            receiver_proof=archive_candidate_binding["receiver_proof_bound"],
        ),
    )
    blockers = [
        "hinerv_modelsize_candidate_consumption_requires_trained_archive_byte_oracle",
        "hinerv_local_architecture_not_source_faithful_upstream_hinerv_feature_grid",
        (
            ""
            if official_feature_grid_convnext_binding["bound"]
            else "hinerv_official_convnext_feature_grid_path_missing"
        ),
        (
            ""
            if official_patch_binding["bound"]
            else "hinerv_official_patch_index_path_missing"
        ),
        (
            ""
            if official_grid_binding["bound"]
            else "hinerv_official_trilinear_feature_interpolation_path_missing"
        ),
        "hinerv_upstream_grid_depth_prune_bitstream_controls_not_executable_atoms",
        "hinerv_official_pruning_control_not_bound_to_planner",
        (
            ""
            if quantnoise_binding["bound"]
            else "hinerv_official_quantnoise_control_not_bound_to_mlx_trainer"
        ),
        "hinerv_torchac_style_bitstream_pipeline_missing",
        "hinerv_decoder_weight_saliency_waterfill_not_in_trainer",
        "hinerv_pr95_c1a_muon_ema_archive_schedule_not_fully_bound_to_real_trainer",
        "hinerv_legacy_full_trainer_archive_is_not_hermetic",
        (
            ""
            if strict_receiver_binding["bound"]
            else "hinerv_receiver_load_strict_false_schema_drift_risk"
        ),
        "hinerv_legacy_trainer_rate_proxy_not_codec_matched",
        "hinerv_exact_cpu_cuda_eval_missing_for_current_candidates",
        *pr95_binding["blockers"],
    ]
    if markers:
        blockers.append("hinerv_partial_or_stub_markers_present")
    blockers = _ordered_unique_nonempty(blockers)
    return {
        "schema": "nerv_stack_synergy_stack_row.v1",
        "stack_id": "hi_nerv",
        "priority": "top_priority_carrier",
        "local_status": "mlx_train_export_adapter_present_but_not_full_upstream_control_surface",
        "source_faithfulness": {
            "source_faithful_upstream_hinerv": False,
            "local_role": "contest_adapter_with_hierarchical_latent_pyramid",
            "missing_upstream_axes": [
                (
                    "full official hierarchical feature-grid source-forward replay artifact"
                    if official_feature_grid_convnext_binding["bound"]
                    else "hierarchical feature-grid encoding"
                ),
                (
                    "full official ConvNeXt source-forward replay artifact"
                    if official_feature_grid_convnext_binding["bound"]
                    else "official ConvNeXt-style feature-grid path"
                ),
                (
                    "full official GridTrilinear3D forward replay artifact"
                    if official_grid_binding["bound"]
                    else "official trilinear feature interpolation path"
                ),
                (
                    "full official patch/frame equivalence replay artifact"
                    if official_patch_binding["bound"]
                    else "patch/frame unified training and eval"
                ),
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
        "modelsize_budget_binding": modelsize_budget_binding,
        "quantnoise_control_binding": quantnoise_binding,
        "official_grid_trilinear_binding": official_grid_binding,
        "official_feature_grid_convnext_binding": official_feature_grid_convnext_binding,
        "official_patch_index_binding": official_patch_binding,
        "official_source_audit_binding": official_source_audit_binding,
        "strict_receiver_load_binding": strict_receiver_binding,
        "archive_candidate_binding": archive_candidate_binding,
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
            "archive_candidate export is byte-closed and receiver-proven, but trainer eval still needs in-loop archive bytes",
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
                "finding": "legacy train_substrate_hi_nerv archive is non-hermetic; archive_candidate is the receiver-proven export route",
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


def _hi_nerv_quantnoise_control_binding(root: Path) -> dict[str, Any]:
    """Return whether official 6/7-bit QuantNoise controls are actually wired."""

    marker_sources = {
        "bitstream": root / "src/tac/substrates/hi_nerv/bitstream.py",
        "mlx_renderer": root / "src/tac/substrates/hi_nerv/mlx_renderer.py",
        "waterfill": root / "src/tac/analysis/nerv_decoder_weight_waterfill.py",
        "runner": root / "tools/run_compact_renderer_mlx_spine_runner.py",
    }
    required_markers = {
        "bitstream": (
            "HI_NERV_QUANT_NOISE_BITS",
            "HI_NERV_DECODER_WATERFILL_ACTION_BITS",
            "2, 4, 6, 7, 8",
        ),
        "mlx_renderer": ("HI_NERV_DECODER_FAKE_QUANT_ACTION_BITS", "6", "7"),
        "waterfill": ("DEFAULT_ACTION_BITS", "6", "7"),
        "runner": ("NERV_DECODER_WEIGHT_WATERFILL_ACTION_BITS", "6", "7"),
    }
    rows: list[dict[str, Any]] = []
    for source_id, path in marker_sources.items():
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        missing = [marker for marker in required_markers[source_id] if marker not in text]
        rows.append(
            {
                "source_id": source_id,
                "rel_path": (
                    path.relative_to(root).as_posix()
                    if path.is_relative_to(root)
                    else path.as_posix()
                ),
                "present": path.is_file(),
                "required_markers": list(required_markers[source_id]),
                "missing_markers": missing,
            }
        )
    blockers = [
        f"hinerv_quantnoise_binding_marker_missing:{row['source_id']}:{marker}"
        for row in rows
        for marker in row["missing_markers"]
    ]
    return {
        "schema": "hinerv_quantnoise_control_binding.v1",
        "bound": not blockers,
        "official_quant_levels_6_7_executable": not blockers,
        "source_rows": rows,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _hi_nerv_official_grid_binding(root: Path) -> dict[str, Any]:
    """Return whether official HiNeRV temporal-only GridTrilinear3D is bound."""

    path = root / "src/tac/substrates/hi_nerv/official_grid.py"
    test_path = root / "src/tac/substrates/hi_nerv/tests/test_official_grid.py"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    test_text = (
        test_path.read_text(encoding="utf-8", errors="replace")
        if test_path.is_file()
        else ""
    )
    required_markers = (
        "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF",
        "OfficialGridTrilinear3D",
        "official_grid_trilinear3d_forward",
        "mode=\"bilinear\"",
        "only supports temporal scaling",
    )
    required_test_markers = (
        "test_official_grid_trilinear3d_matches_torch_interpolate",
        "F.interpolate",
        "align_corners",
    )
    missing = [marker for marker in required_markers if marker not in text]
    test_missing = [marker for marker in required_test_markers if marker not in test_text]
    return {
        "schema": "hinerv_official_grid_trilinear_binding.v1",
        "rel_path": "src/tac/substrates/hi_nerv/official_grid.py",
        "test_rel_path": "src/tac/substrates/hi_nerv/tests/test_official_grid.py",
        "present": path.is_file(),
        "test_present": test_path.is_file(),
        "required_markers": list(required_markers),
        "missing_markers": missing,
        "missing_test_markers": test_missing,
        "bound": path.is_file() and test_path.is_file() and not missing and not test_missing,
        "authority": "false_authority_component_binding_no_full_forward_parity_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_official_feature_grid_convnext_binding(root: Path) -> dict[str, Any]:
    """Return whether local HiNeRV feature-grid/ConvNeXt surfaces are bound."""

    source_specs = {
        "architecture": (
            "src/tac/substrates/hi_nerv/architecture.py",
            (
                "HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF",
                "class HierarchicalFeatureGrid",
                "class ConvNeXtBlock",
                "trilinear_upsample",
            ),
        ),
        "mlx_renderer": (
            "src/tac/substrates/hi_nerv/mlx_renderer.py",
            ("class ConvNeXtBlockMLX", "trilinear_upsample_mlx"),
        ),
        "archive_roundtrip_tests": (
            "src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py",
            (
                "test_official_feature_grid_convnext_mode_is_receiver_visible",
                "test_official_feature_grid_convnext_archive_roundtrip_preserves_forward",
            ),
        ),
    }
    rows = _marker_binding_rows(root, source_specs)
    blockers = [
        f"hinerv_feature_grid_convnext_binding_marker_missing:{row['source_id']}:{marker}"
        for row in rows
        for marker in row["missing_markers"]
    ]
    return {
        "schema": "hinerv_official_feature_grid_convnext_binding.v1",
        "bound": not blockers,
        "full_upstream_source_forward_replay_proven": False,
        "source_rows": rows,
        "blockers": blockers,
        "authority": "false_authority_receiver_binding_no_full_forward_parity_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_official_patch_binding(root: Path) -> dict[str, Any]:
    """Return whether official HiNeRV patch/index NumPy primitives are bound."""

    source_specs = {
        "official_patch": (
            "src/tac/substrates/hi_nerv/official_patch.py",
            (
                "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
                "official_video_to_patch",
                "official_patch_to_video",
                "official_vidx_to_pidx",
                "official_compute_pixel_idx_3d",
                "official_flat_patch_index_to_thw",
            ),
        ),
        "official_patch_tests": (
            "src/tac/substrates/hi_nerv/tests/test_official_patch.py",
            (
                "test_official_vidx_to_pidx_expands_child_patch_grid",
                "test_official_compute_pixel_idx_3d_matches_padding_and_clipping_contract",
                "test_official_flat_patch_index_to_thw_matches_dataset_mapping",
                "test_official_patch_contract_is_false_authority",
            ),
        ),
    }
    rows = _marker_binding_rows(root, source_specs)
    blockers = [
        f"hinerv_patch_binding_marker_missing:{row['source_id']}:{marker}"
        for row in rows
        for marker in row["missing_markers"]
    ]
    return {
        "schema": "hinerv_official_patch_index_binding.v1",
        "bound": not blockers,
        "full_patch_frame_equivalence_replay_proven": False,
        "source_rows": rows,
        "blockers": blockers,
        "authority": "false_authority_receiver_binding_no_full_patch_replay_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_official_source_audit_binding(
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind optional official HiNeRV source-forward audit evidence."""

    if not isinstance(audit, Mapping):
        return {
            "schema": "hinerv_official_source_audit_stack_binding.v1",
            "artifact_supplied": False,
            "audit_schema_valid": False,
            "official_forward_replay_ran": False,
            "official_forward_parity_proven": False,
            "official_forward_parity_falsified": False,
            "falsification_accepted": False,
            "full_upstream_source_forward_replay_proven": False,
            "blockers": [
                "hinerv_official_source_audit_artifact_not_supplied",
            ],
            "authority": "false_authority_optional_stack_binding_no_score_claim",
            **FALSE_AUTHORITY,
        }
    schema_valid = audit.get("schema") == "hinerv_official_source_parity_audit.v1"
    forward_row = audit.get("official_forward_parity_artifact_row")
    if not isinstance(forward_row, Mapping):
        forward_row = {}
    component_rows = [
        row
        for row in audit.get("component_state_rows") or ()
        if isinstance(row, Mapping)
    ]
    core_row = next(
        (
            row
            for row in component_rows
            if row.get("component_id") == "core_hierarchical_renderer"
        ),
        {},
    )
    replay = core_row.get("official_source_forward_replay")
    if not isinstance(replay, Mapping):
        replay = {}
    blockers = []
    if not schema_valid:
        blockers.append("hinerv_official_source_audit_schema_invalid")
    if forward_row.get("status") != "present":
        blockers.append("hinerv_official_forward_parity_artifact_not_present")
    if replay.get("replay_ran") is not True:
        blockers.append("hinerv_official_torch_forward_replay_not_ran")
    if (
        forward_row.get("parity_passed") is not True
        and forward_row.get("falsification_accepted") is not True
    ):
        blockers.append("hinerv_official_forward_falsification_not_accepted")
    return {
        "schema": "hinerv_official_source_audit_stack_binding.v1",
        "artifact_supplied": True,
        "audit_schema_valid": schema_valid,
        "audit_authority": audit.get("authority"),
        "official_forward_artifact_status": forward_row.get("status"),
        "official_forward_artifact_path": forward_row.get("path"),
        "official_forward_artifact_sha256": forward_row.get("sha256"),
        "official_forward_artifact_bytes": forward_row.get("bytes"),
        "official_forward_replay_ran": bool(replay.get("replay_ran")),
        "official_forward_replay_backend": replay.get("backend"),
        "official_forward_input_bundle_sha256": replay.get("input_bundle_sha256"),
        "official_forward_output_sha256": replay.get("official_output_sha256"),
        "official_weight_sha256": replay.get("official_weight_sha256"),
        "official_forward_parity_proven": bool(
            audit.get("official_forward_parity_proven")
        ),
        "official_forward_parity_falsified": bool(forward_row.get("parity_falsified")),
        "falsification_accepted": bool(forward_row.get("falsification_accepted")),
        "full_upstream_source_forward_replay_proven": bool(
            audit.get("official_forward_parity_proven")
        ),
        "remaining_blockers": [
            "hinerv_local_portable_full_forward_adapter_missing",
            "hinerv_official_forward_replay_is_source_only",
        ],
        "blockers": blockers,
        "authority": "false_authority_official_source_forward_evidence_no_score_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_strict_receiver_load_binding(root: Path) -> dict[str, Any]:
    """Return whether the HiNeRV receiver fails closed on archive schema drift."""

    source_specs = {
        "inflate": (
            "src/tac/substrates/hi_nerv/inflate.py",
            (
                "full_state = dict(arc.decoder_state_dict)",
                "full_state[\"latents_coarse\"]",
                "full_state[\"latents_mid\"]",
                "full_state[\"latents_fine\"]",
                "model.load_state_dict(full_state, strict=True)",
            ),
        ),
        "receiver_tests": (
            "src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py",
            (
                "test_receiver_loads_complete_archive_state_strictly",
                "observed[\"strict\"] is True",
                "LATENT_STATE_KEYS",
            ),
        ),
    }
    rows = _marker_binding_rows(root, source_specs)
    inflate_path = root / "src/tac/substrates/hi_nerv/inflate.py"
    inflate_text = (
        inflate_path.read_text(encoding="utf-8", errors="replace")
        if inflate_path.is_file()
        else ""
    )
    blockers = [
        f"hinerv_strict_receiver_load_marker_missing:{row['source_id']}:{marker}"
        for row in rows
        for marker in row["missing_markers"]
    ]
    if "strict=False" in inflate_text:
        blockers.append("hinerv_receiver_inflate_still_uses_strict_false")
    return {
        "schema": "hinerv_strict_receiver_load_binding.v1",
        "bound": not blockers,
        "strict_receiver_load": not blockers,
        "source_rows": rows,
        "blockers": blockers,
        "authority": "false_authority_receiver_schema_guard_no_score_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_modelsize_budget_binding(budget: Mapping[str, Any]) -> dict[str, Any]:
    """Return whether HiNeRV capacity choices are bound to byte ceilings."""

    selected = [
        row
        for row in budget.get("selected_candidates") or ()
        if isinstance(row, Mapping)
    ]
    blockers: list[str] = []
    if budget.get("schema") != "nerv_modelsize_budget.v1":
        blockers.append("hinerv_modelsize_budget_schema_not_bound")
    if not budget.get("hard_byte_ceilings"):
        blockers.append("hinerv_modelsize_budget_hard_byte_ceilings_missing")
    if not selected:
        blockers.append("hinerv_modelsize_budget_selected_candidates_missing")
    contract_rows: list[dict[str, Any]] = []
    for row in selected:
        candidate_id = str(row.get("candidate_id") or "unknown")
        contract = row.get("modelsize_control_contract")
        hard_ceiling = row.get("hard_byte_ceiling")
        nominal = row.get("nominal_total_payload_bytes")
        contract_missing: list[str] = []
        if not isinstance(contract, Mapping):
            contract_missing.append("modelsize_control_contract")
        else:
            for key in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS:
                if contract.get(key) is not True:
                    contract_missing.append(str(key))
        if not isinstance(hard_ceiling, int) or hard_ceiling <= 0:
            contract_missing.append("hard_byte_ceiling")
        if not isinstance(nominal, int) or nominal <= 0:
            contract_missing.append("nominal_total_payload_bytes")
        if contract_missing:
            blockers.append(
                "hinerv_modelsize_budget_candidate_contract_missing:"
                f"{candidate_id}:{','.join(contract_missing)}"
            )
        contract_rows.append(
            {
                "candidate_id": candidate_id,
                "hard_byte_ceiling": hard_ceiling,
                "nominal_total_payload_bytes": nominal,
                "nominal_under_ceiling": row.get("nominal_under_ceiling"),
                "decoder_codec": row.get("decoder_codec"),
                "missing_contract_fields": contract_missing,
            }
        )
    return {
        "schema": "hinerv_modelsize_budget_binding.v1",
        "bound": not blockers,
        "modelsize_archive_budget_bound": not blockers,
        "trained_archive_byte_oracle_bound": False,
        "selected_candidate_count": len(selected),
        "hard_byte_ceilings": list(budget.get("hard_byte_ceilings") or []),
        "contract_required_true_fields": list(
            MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
        ),
        "candidate_contract_rows": contract_rows,
        "blockers": blockers,
        "remaining_train_loop_blockers": [
            "hinerv_modelsize_candidates_require_trained_archive_byte_oracle",
            "hinerv_modelsize_candidates_require_full_video_mlx_prefilter",
        ],
        "authority": "false_authority_modelsize_budget_no_trained_archive_claim",
        **FALSE_AUTHORITY,
    }


def _hi_nerv_archive_candidate_binding(root: Path) -> dict[str, Any]:
    """Return whether HiNeRV MLX export is byte-closed and receiver-proven."""

    source_specs = {
        "archive_candidate": (
            "src/tac/substrates/hi_nerv/archive_candidate.py",
            (
                "export_hi_nerv_mlx_archive",
                "_write_and_reload_exported_state_via_numpy_bridge",
                "write_contest_runtime(",
                "build_archive_zip(",
                "\"archive_bytes_are_authority_for_rate\": True",
                "write_representation_spine_projection",
                "emit_archive_bound_candidate_runtime_package(",
                "proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA",
                "expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg)",
            ),
        ),
        "archive_candidate_tests": (
            "src/tac/substrates/hi_nerv/tests/test_hi_nerv_mlx_renderer_and_archive_candidate.py",
            (
                "test_archive_export_emits_receiver_proof_and_hprc_spine",
                "assert archive_bytes == archive_path.stat().st_size",
                "assert proof[\"runtime_consumption_proof_ready\"] is True",
                "assert package[\"receiver_proof\"][\"receiver_contract_satisfied\"] is True",
                "assert spine_extra[\"hi_nerv_bitstream_preparation\"] == bitstream_report",
                "assert portability[\"canonical_npz_bridge_used\"] is True",
            ),
        ),
        "runtime_bridge": (
            "src/tac/optimization/archive_bound_candidate_runtime_bridge.py",
            (
                "run_generated_inflate_receiver_proof",
                "\"receiver_contract_satisfied\": passed",
                "expected_receiver_output_bytes",
                "build_archive_bound_candidate_runtime_package",
            ),
        ),
    }
    rows = _marker_binding_rows(root, source_specs)
    blockers = [
        f"hinerv_archive_candidate_binding_marker_missing:{row['source_id']}:{marker}"
        for row in rows
        for marker in row["missing_markers"]
    ]
    bound = not blockers
    return {
        "schema": "hinerv_archive_candidate_binding.v1",
        "bound": bound,
        "byte_closed_archive_export_bound": bound,
        "receiver_proof_bound": bound,
        "archive_in_loop_byte_oracle_bound": False,
        "source_rows": rows,
        "blockers": blockers,
        "remaining_train_loop_blockers": [
            "hinerv_archive_candidate_is_export_time_not_trainer_in_loop_byte_oracle",
        ],
        "authority": "false_authority_archive_runtime_binding_no_score_claim",
        **FALSE_AUTHORITY,
    }


def _marker_binding_rows(
    root: Path,
    source_specs: dict[str, tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_id, (rel_path, required_markers) in source_specs.items():
        path = root / rel_path
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        missing = [marker for marker in required_markers if marker not in text]
        rows.append(
            {
                "source_id": source_id,
                "rel_path": rel_path,
                "present": path.is_file(),
                "required_markers": list(required_markers),
                "missing_markers": missing,
            }
        )
    return rows


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
    modelsize_budget_binding = _snerv_modelsize_budget_binding(budget)
    pr95_binding = build_pr95_stack_binding_requirements(
        family="snerv",
        evidence=build_pr95_stack_binding_evidence(
            modelsize_archive_budget=modelsize_budget_binding[
                "modelsize_archive_budget_bound"
            ],
        ),
    )
    native_contract = build_snerv_mlx_native_adapter_contract()
    official_primitive_replay = build_snerv_official_primitive_replay_binding(
        repo_root=root,
    )
    native_adapter_blockers = list(native_contract.get("blockers") or [])
    native_adapter_blockers.append(
        "snerv_mlx_native_adapter_surfaces_present_but_unproven"
        if native_contract.get("surfaces_ready")
        else "snerv_mlx_native_train_export_adapter_missing"
    )
    official_primitive_blockers = _snerv_official_mfu_hfr_tub_stack_blockers(
        official_primitive_replay
    )
    blockers = [
        *native_adapter_blockers,
        "snerv_execute_family_uses_cpu_advisory_not_mlx_native_training",
        "snerv_modelsize_candidate_consumption_requires_real_snar1_archive_byte_oracle",
        "snerv_local_carrier_not_source_faithful_official_snerv_multilayer_stack",
        *official_primitive_blockers,
        "snerv_official_haar_j1_parity_missing",
        "snerv_receiver_dwt_custody_missing",
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
    blockers = _ordered_unique_nonempty(blockers)
    return {
        "schema": "nerv_stack_synergy_stack_row.v1",
        "stack_id": "snerv",
        "priority": "top_priority_carrier",
        "local_status": "receiver_bound_advisory_export_present_mlx_native_surfaces_unproven",
        "snerv_mlx_native_adapter_contract": native_contract,
        "official_mfu_hfr_tub_primitive_replay_binding": official_primitive_replay,
        "source_faithfulness": {
            "source_faithful_official_snerv": False,
            "local_role": "contest_adapter_store_lf_generate_hf_snar1_packet",
            "missing_upstream_axes": [
                "official multi-layer scalable neural representation",
                (
                    "MFU/HFR/TUB primitive replay is proven but not receiver/export-bound"
                    if official_primitive_replay.get(
                        "all_primitive_source_replay_proven"
                    )
                    else "MFU/HFR/TUB-style blocks"
                ),
                "full temporal SNeRV-T/SNeRV-T-2D source-forward training path beyond the proven Haar DWT1D primitive",
                "official Haar/J=1 mode parity",
                "receiver DWT custody and source/runtime hash proof",
                "measured SNAR1 archive-byte curve for official modelsize candidates",
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
        "modelsize_budget_binding": modelsize_budget_binding,
        "pr95_stack_binding": pr95_binding,
        "planner_curriculum_links": [
            "LF level and precision select rate before advisory launch",
            "official --modelsize/fc_dim candidates are source-bound but still require real SNAR1 archive-byte replay",
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


def _snerv_modelsize_budget_binding(budget: Mapping[str, Any]) -> dict[str, Any]:
    """Return whether SNeRV fc_dim/modelsize choices are byte-budget bound."""

    selected = [
        row
        for row in budget.get("selected_candidates") or ()
        if isinstance(row, Mapping)
    ]
    blockers: list[str] = []
    if budget.get("schema") != "snerv_modelsize_budget.v1":
        blockers.append("snerv_modelsize_budget_schema_not_bound")
    if not budget.get("hard_byte_ceilings"):
        blockers.append("snerv_modelsize_budget_hard_byte_ceilings_missing")
    if not selected:
        blockers.append("snerv_modelsize_budget_selected_candidates_missing")
    contract_rows: list[dict[str, Any]] = []
    for row in selected:
        candidate_id = str(row.get("candidate_id") or "unknown")
        contract = row.get("modelsize_control_contract")
        hard_ceiling = row.get("hard_byte_ceiling")
        nominal = row.get("nominal_total_payload_bytes")
        contract_missing: list[str] = []
        if not isinstance(contract, Mapping):
            contract_missing.append("modelsize_control_contract")
        else:
            for key in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS:
                if contract.get(key) is not True:
                    contract_missing.append(str(key))
        if not isinstance(hard_ceiling, int) or hard_ceiling <= 0:
            contract_missing.append("hard_byte_ceiling")
        if not isinstance(nominal, int) or nominal <= 0:
            contract_missing.append("nominal_total_payload_bytes")
        if contract_missing:
            blockers.append(
                "snerv_modelsize_budget_candidate_contract_missing:"
                f"{candidate_id}:{','.join(contract_missing)}"
            )
        contract_rows.append(
            {
                "candidate_id": candidate_id,
                "hard_byte_ceiling": hard_ceiling,
                "nominal_total_payload_bytes": nominal,
                "nominal_under_ceiling": row.get("nominal_under_ceiling"),
                "fc_dim": row.get("fc_dim"),
                "modelsize_mparams": row.get("modelsize_mparams"),
                "snerv_model_size_adapter": row.get("snerv_model_size_adapter"),
                "missing_contract_fields": contract_missing,
            }
        )
    under_ceiling_count = sum(
        1 for row in selected if row.get("nominal_under_ceiling") is True
    )
    return {
        "schema": "snerv_modelsize_budget_binding.v1",
        "bound": not blockers,
        "modelsize_archive_budget_bound": not blockers,
        "real_snar1_archive_byte_oracle_bound": False,
        "selected_candidate_count": len(selected),
        "selected_nominal_under_ceiling_count": under_ceiling_count,
        "hard_byte_ceilings": list(budget.get("hard_byte_ceilings") or []),
        "contract_required_true_fields": list(
            MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS
        ),
        "candidate_contract_rows": contract_rows,
        "blockers": blockers,
        "remaining_train_loop_blockers": [
            "snerv_modelsize_candidates_require_real_snar1_archive_byte_oracle",
            "snerv_modelsize_candidates_require_full_video_mlx_prefilter",
        ],
        "authority": "false_authority_snerv_modelsize_budget_no_snar1_rate_claim",
        **FALSE_AUTHORITY,
    }


def _snerv_official_mfu_hfr_tub_stack_blockers(
    official_primitive_replay: Mapping[str, Any],
) -> list[str]:
    """Return the exact remaining MFU/HFR/TUB blocker names for SNeRV."""

    rows = [
        row
        for row in official_primitive_replay.get("component_rows") or ()
        if isinstance(row, Mapping)
    ]
    component_ids = {str(row.get("component_id")) for row in rows}
    blockers: list[str] = []
    if official_primitive_replay.get("all_primitive_source_replay_proven") is not True:
        blocker_by_component = {
            "mfu": "snerv_official_mfu_source_forward_replay_missing",
            "hfr": "snerv_official_hfr_source_forward_replay_missing",
            "tub": "snerv_official_snerv_t_full_tub_path_not_source_forward_parity",
        }
        for component_id, blocker in blocker_by_component.items():
            row = next((item for item in rows if item.get("component_id") == component_id), None)
            if row is None or row.get("primitive_source_replay_proven") is not True:
                blockers.append(blocker)
        for component_id in sorted({"mfu", "hfr", "tub"} - component_ids):
            blockers.append(f"snerv_official_{component_id}_primitive_replay_row_missing")
        return blockers
    if official_primitive_replay.get("full_stack_source_forward_replay_proven") is not True:
        blockers.append("snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing")
    if official_primitive_replay.get("receiver_export_bound") is not True:
        blockers.append("snerv_official_mfu_hfr_tub_receiver_export_not_bound")
    return blockers


def _ordered_unique_nonempty(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


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
