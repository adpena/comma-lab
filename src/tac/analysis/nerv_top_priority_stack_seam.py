# SPDX-License-Identifier: MIT
"""Fail-closed orchestration contract for the top-priority NeRV stacks.

SNeRV and HiNeRV are the two active carrier stacks. PR95/HNeRV is the upstream
baseline/control to beat. This module turns that operating rule into a typed
artifact so queue builders, future agents, and operator briefings inherit the
same fail-closed gates instead of re-deriving priority from chat.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "nerv_top_priority_stack_seam.v1"
AXIS_TAG = "[planning/control]"
DEFAULT_LANE_ID = "lane_nerv_top_priority_stack_seam_20260602"
PR95_PR_NUMBER = 95
PR95_PR_URL = "https://github.com/commaai/comma_video_compression_challenge/pull/95"
PR95_SUBMISSION = "hnerv_muon"
PR101_LANE_ID = "lane_pr101_storage_order_len24_exact_cpu_20260601"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "ready_for_exact_eval_dispatch": False,
    "exact_or_full_video_launched": False,
}

TOP_PRIORITY_CARRIERS = ("snerv", "hinerv")
FULL_STACK_COMPONENTS = (
    "architecture",
    "optimizer_qat",
    "allocator",
    "archive_grammar",
    "receiver_proof",
    "eval_control",
)
TERMINAL_STATUS_PREFIXES = (
    "completed_",
    "failed_",
    "timed_out",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
    "stop_attempt_timeout_duplicate_after_primary_negative",
)
ACTIVE_STATUS_TOKENS = (
    "active",
    "pending",
    "spawned",
    "spawning",
    "running",
    "eval",
    "training",
    "dispatching",
)
EXACT_OR_FULL_VIDEO_TOKENS = (
    "auth_eval",
    "contest_cpu",
    "contest_cuda",
    "exact",
    "full_video",
    "full-video",
    "paired_cuda",
    "paired-cuda",
    "modal_auth",
    "cuda_ratification",
)
REMOTE_EVAL_PLATFORMS = ("lightning", "modal", "vast", "vastai", "azure", "aws", "gcp")

OFFICIAL_OSS_SOURCES: dict[str, dict[str, Any]] = {
    "snerv": {
        "repo_url": "https://github.com/qwertja/SNeRV.git",
        "paper": "SNeRV: Spectra-preserving Neural Representation for Video",
        "paper_url": "https://arxiv.org/abs/2501.01681",
        "required_files": [
            "train_snerv.py",
            "train_snerv_t.py",
            "model/snerv.py",
            "model/snerv_t.py",
            "model/layers.py",
        ],
        "required_features": [
            "official_encoder_decoder_stride_stack",
            "haar_dwt_idwt_low_high_frequency_reconstruction",
            "multi_resolution_fusion_blocks",
            "high_frequency_restoration_heads",
            "temporal_extension_snerv_t_or_documented_no_go",
            "modelsize_or_fc_dim_budget_binding",
            "quant_model_and_embedding_payload_accounting",
        ],
    },
    "hinerv": {
        "repo_url": "https://github.com/hmkx/HiNeRV.git",
        "paper": (
            "HiNeRV: Video Compression with Hierarchical Encoding-based "
            "Neural Representation"
        ),
        "paper_url": "https://arxiv.org/abs/2306.09818",
        "required_files": [
            "hinerv_main.py",
            "hinerv_compress.py",
            "models/hinerv.py",
            "models/encoding.py",
            "models/patch_utils.py",
            "compression/quant_utils.py",
            "compression/prune_utils.py",
            "compression/codec_utils.py",
        ],
        "required_features": [
            "hierarchical_feature_grid_encoding",
            "patch_mode_and_frame_mode_equivalence",
            "3d_trilinear_or_nearest_hierarchical_upsampling",
            "official_config_family_size_sweeps",
            "pruning_parametrization",
            "quant_noise_and_quant_ste_training_controls",
            "torchac_or_equivalent_integer_bitstream_codec",
        ],
    },
    "hnerv_pr95_control": {
        "repo_url": "https://github.com/haochen-rye/HNeRV.git",
        "paper": "HNeRV: A Hybrid Neural Representation for Videos",
        "paper_url": "https://arxiv.org/abs/2304.02633",
        "required_files": [
            "train_nerv_all.py",
            "model_all.py",
            "hnerv_utils.py",
            "efficient_nvloader.py",
        ],
        "required_features": [
            "modelsize_flag_controls_decoder_and_embedding_budget",
            "ks_reduce_lower_width_parameter_balance",
            "convnext_encoder_pshuffel_decoder_path",
            "quant_model_bit_and_quant_embed_bit_export",
            "source_runtime_replay_for_pr95_control",
        ],
    },
}

LOCAL_SOURCE_FAITHFULNESS_AUDIT: dict[str, dict[str, Any]] = {
    "snerv": {
        "status": "simplified_contest_adapter_not_source_faithful",
        "local_surfaces": [
            "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
            "src/tac/analysis/nerv_modelsize_budget.py",
            "tools/build_nerv_modelsize_budget.py",
            "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
        ],
        "implemented_features": [
            "haar_like_multilevel_dwt_idwt",
            "lf_storage_with_generated_hf_detail",
            "linear_3x3_hf_predictor",
            "official_modelsize_fc_dim_parameter_budget_solver",
            "invalid_official_modelsize_rows_preserved_false_authority",
            "source_bound_modelsize_candidate_ids",
            "local_scorer_loop_decoder_qat_smoke",
            "receiver_archive_proof_surface",
        ],
        "missing_source_features": [
            "official_SNeRV_encoder_decoder_stride_stack",
            "official_MFU_multi_resolution_fusion_blocks",
            "official_HFR_high_frequency_restoration_heads",
            "official_SNeRV_T_temporal_neighbor_path",
            "official_quantized_checkpoint_payload_replay",
        ],
    },
    "hinerv": {
        "status": "l0_sketch_not_source_faithful",
        "local_surfaces": [
            "src/tac/substrates/hi_nerv/architecture.py",
            "src/tac/substrates/hi_nerv/archive.py",
            "src/tac/substrates/hi_nerv/bitstream.py",
            "src/tac/substrates/hi_nerv/mlx_renderer.py",
            "src/tac/substrates/hi_nerv/score_aware_loss.py",
            "src/tac/analysis/nerv_decoder_weight_waterfill.py",
            "tools/run_compact_renderer_mlx_spine_runner.py",
        ],
        "implemented_features": [
            "three_scale_latent_pyramid_sketch",
            "local_archive_and_receiver_smoke",
            "score_aware_loss_bridge",
            "mlx_prefilter_path",
            "decoder_waterfill_action_lattice_0_2_4_6_7_8_16_32",
            "quant_noise_action_lattice_2_4_6_7_8",
            "receiver_visible_decoder_waterfill_actuation",
            "train_time_mlx_fake_quant_per_tensor_actions",
        ],
        "missing_source_features": [
            "official_hierarchical_feature_grid_encoding",
            "official_patch_mode_frame_mode_equivalence",
            "official_fast_3d_hierarchical_upsampling",
            "official_config_family_size_sweep_parity",
            "official_prune_quant_ste_torchac_pipeline_parity",
            "official_bitstream_compress_decompress_roundtrip",
        ],
    },
}


class NervTopPriorityStackSeamError(ValueError):
    """Raised when the orchestration contract inputs are malformed."""


def build_nerv_top_priority_stack_seam(
    *,
    repo_root: str | Path,
    upstream_repo_dir: str | Path | None = None,
    pr95_intake_root: str | Path | None = None,
    active_claims_path: str | Path | None = None,
    pr95_pr_metadata: Mapping[str, Any] | None = None,
    oss_source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    generated_utc: str | None = None,
    lane_id: str = DEFAULT_LANE_ID,
) -> dict[str, Any]:
    """Build the shared top-priority stack contract.

    The returned payload is intentionally not a score artifact. It grants local
    implementation authority only and blocks exact/full-video dispatch while
    active exact-eval claims remain pending.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NervTopPriorityStackSeamError(f"repo_root is not a directory: {root}")

    generated = generated_utc or datetime.now(UTC).isoformat()
    baseline = discover_pr95_baseline(
        repo_root=root,
        upstream_repo_dir=upstream_repo_dir,
        pr95_intake_root=pr95_intake_root,
        pr95_pr_metadata=pr95_pr_metadata,
    )
    source_faithfulness = build_source_faithfulness_matrix(oss_source_metadata)
    dispatch_blockers = discover_dispatch_blockers(
        active_claims_path,
        now_utc=generated,
    )
    blockers = _unique(
        list(baseline["blockers"])
        + list(source_faithfulness["blockers"])
        + dispatch_blockers
        + [
            "full_600_byte_closed_receiver_proof_missing_for_snerv_and_hinerv",
            "paired_contest_cpu_cuda_pass_missing_for_winner",
            "pr95_same_axis_control_replay_required_before_beat_claim",
        ]
    )
    exact_blocked = bool(dispatch_blockers)
    scorer_domain_control_policy = _scorer_domain_control_policy()
    optimal_modelsize_control_policy = _optimal_modelsize_control_policy()
    blockers = _unique(
        [
            *blockers,
            *scorer_domain_control_policy["production_blockers"],
            *optimal_modelsize_control_policy["production_blockers"],
        ]
    )

    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "lane_id": lane_id,
        "axis_tag": AXIS_TAG,
        "go_no_go_verdict": (
            "GO_LOCAL_STACK_OPTIMIZATION__NO_GO_PRODUCTION_HARDENED_OR_EXACT_CLAIM"
        ),
        "baseline_to_beat": "pr95_hnerv_muon",
        "top_priority_carriers": list(TOP_PRIORITY_CARRIERS),
        "priority_policy": {
            "carrier_stacks": list(TOP_PRIORITY_CARRIERS),
            "baseline_control": "pr95_hnerv_muon",
            "individually_fractally_optimized_full_stacks": True,
            "shared_synergy_surfaces_do_not_collapse_carrier_specific_work": True,
            "enhancers_are_not_standalone_carrier_stacks": True,
            "compare_under_same_archive_runtime_eval_axis": True,
            "do_not_launch_new_full_video_or_exact_while_dispatch_blockers_active": True,
            "no_fake_implementations_allowed": True,
            "paper_and_oss_parity_required_before_production_claim": True,
            "bad_current_scores_are_config_or_wiring_bug_signals_until_parity": True,
        },
        "source_faithfulness": source_faithfulness,
        "modelsize_archive_budget_policy": _modelsize_archive_budget_policy(),
        "scorer_domain_control_policy": scorer_domain_control_policy,
        "optimal_modelsize_control_policy": optimal_modelsize_control_policy,
        "full_stack_priority": _full_stack_priority(),
        "baseline": baseline,
        "carrier_stacks": [_snerv_stack(), _hinerv_stack()],
        "fractal_work_orders": _fractal_work_orders(),
        "synergy_enhancers": _synergy_enhancers(),
        "shared_promotion_gates": _shared_promotion_gates(),
        "blocked_dispatch": exact_blocked,
        "dispatch_blockers": dispatch_blockers,
        "next_local_actions": _next_local_actions(
            exact_blocked=exact_blocked,
            upstream_repo_dir=baseline["upstream_repo_dir"].get("path"),
        ),
        "forbidden_actions": [
            "claim_pr95_beat_without_same_axis_pr95_control",
            "promote_local_mlx_or_macos_cpu_advisory_score",
            "dispatch_exact_or_full_video_while_pr101_cpu_pending",
            "rerun_closed_form_snerv_scalar_hf_sweeps_as_promotion_evidence",
            "treat_sr_nerv_zero_parameter_interpolation_as_promotable",
            "call_simplified_snerv_adapter_source_faithful",
            "call_l0_hinerv_sketch_source_faithful",
            "retire_snerv_or_hinerv_from_current_bad_advisory_scores",
        ],
        "blockers": blockers,
        "operator_truth": {
            "main_is_source_of_truth": True,
            "dirty_shared_worktree_is_not_absorbed": True,
            "upstream_pr95_is_forensic_baseline_source": True,
            "large_artifacts_policy": (
                "this artifact is metadata-only; future training/eval outputs must "
                "spill to SSD and preserve deterministic custody before cleanup"
            ),
        },
        **FALSE_AUTHORITY,
    }


def build_source_faithfulness_matrix(
    oss_source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the paper/OSS parity gate for production-hardened NeRV claims.

    Bad advisory scores from a sketch are bug/config/wiring signals, not method
    negatives. This matrix makes that operational: a stack remains local-only
    until official source features, contest adaptations, receiver bytes, and
    same-axis control replay are all present.
    """

    metadata = {
        str(key): dict(value)
        for key, value in (oss_source_metadata or {}).items()
    }
    source_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for stack_id in ("snerv", "hinerv", "hnerv_pr95_control"):
        official = OFFICIAL_OSS_SOURCES[stack_id]
        source_meta = dict(metadata.get(stack_id, {}))
        source_rows.append(
            {
                "stack_id": stack_id,
                "role": (
                    "baseline_control"
                    if stack_id == "hnerv_pr95_control"
                    else "top_priority_carrier"
                ),
                "official": official,
                "observed_source": {
                    "repo_url": source_meta.get("repo_url", official["repo_url"]),
                    "head_sha": source_meta.get("head_sha"),
                    "audit_root": source_meta.get("audit_root"),
                    "source_snapshot_claim": bool(source_meta.get("head_sha")),
                },
                "parity_status": (
                    "oss_snapshot_observed_not_yet_contest_parity_proven"
                    if source_meta.get("head_sha")
                    else "oss_snapshot_missing"
                ),
                "production_authority": False,
                "blockers": [
                    f"{stack_id}_contest_config_parity_missing",
                    f"{stack_id}_architecture_symbol_parity_missing",
                    f"{stack_id}_tiny_forward_parity_missing",
                    f"{stack_id}_contest_receiver_byte_grammar_missing",
                    f"{stack_id}_same_axis_pr95_control_missing",
                ],
            }
        )
        if not source_meta.get("head_sha"):
            blockers.append(f"{stack_id}_official_oss_snapshot_missing")
    for stack_id, audit in LOCAL_SOURCE_FAITHFULNESS_AUDIT.items():
        blockers.append(f"{stack_id}_{audit['status']}")

    return {
        "schema": "nerv_source_faithfulness_matrix.v1",
        "verdict": (
            "NO_GO_PRODUCTION_HARDENED_CLAIM_UNTIL_PAPER_OSS_PARITY_AND_"
            "CONTEST_RECEIVER_PROOFS_PASS"
        ),
        "policy": {
            "no_fake_implementations_allowed": True,
            "minimal_or_sketch_adapters_are_local_only": True,
            "bad_scores_from_non_source_faithful_stacks_are_bug_signals": True,
            "production_hardened_requires_official_feature_parity": True,
            "journal_grade_requires_source_snapshot_and_replayable_commands": True,
        },
        "official_sources": source_rows,
        "local_implementation_audit": [
            {
                "stack_id": stack_id,
                **audit,
                "method_verdict_authority": False,
                "production_authority": False,
            }
            for stack_id, audit in LOCAL_SOURCE_FAITHFULNESS_AUDIT.items()
        ],
        "required_gates": [
            {
                "gate": "official_oss_snapshot_custody",
                "required": True,
                "evidence": "repo_url, head_sha, audit_root, required file manifest",
            },
            {
                "gate": "paper_feature_parity",
                "required": True,
                "evidence": "architecture symbol map from official feature to local module",
            },
            {
                "gate": "official_config_parity_smoke",
                "required": True,
                "evidence": "tiny deterministic official config and local adapted config",
            },
            {
                "gate": "contest_adaptation_binding",
                "required": True,
                "evidence": "1164x874 output, scorer downsample path, 600-pair filelist",
            },
            {
                "gate": "byte_closed_receiver_grammar",
                "required": True,
                "evidence": "archive bytes, member hashes, receiver-only inflate proof",
            },
            {
                "gate": "same_axis_pr95_control",
                "required": True,
                "evidence": "PR95 source runtime replay on matching CPU/CUDA axis",
            },
        ],
        "blockers": _unique(blockers),
        "production_hardened_claim": False,
        "source_faithful_stack_claim": False,
    }


def _modelsize_archive_budget_policy() -> dict[str, Any]:
    """Describe the archive-size inversion required before production claims."""

    return {
        "schema": "nerv_modelsize_archive_budget_policy.v1",
        "verdict": "NO_GO_FULL_LEVERAGE_UNTIL_MODEL_SIZE_TO_ARCHIVE_BYTES_CURVE_EXISTS",
        "why_it_matters": (
            "Official HNeRV/SNeRV expose a parameter-budget knob that can be "
            "inverted into an archive-byte budget once quantization, entropy "
            "coding, and metadata overhead are measured. SNeRV now has a "
            "source-bound official --modelsize/fc_dim solver, but neither "
            "top-priority stack has the measured modelsize-to-archive-bytes "
            "score/byte Pareto curve needed for production authority."
        ),
        "official_controls_to_bind": [
            "--modelsize",
            "--ks",
            "--reduce",
            "--lower_width",
            "--enc_dim",
            "--fc_hw",
            "--enc_strds",
            "--dec_strds",
            "--quant_model_bit",
            "--quant_embed_bit",
            "--quant_axis",
            "HiNeRV config family xs/s/m/l/xl/xxl",
            "HiNeRV prune-ratio/prune-weight",
            "HiNeRV quant-level/quant-noise/quant-ste",
        ],
        "contest_inversion_target": {
            "objective": (
                "choose architecture widths, embedding dimensions, quant bits, "
                "and entropy grammar to minimize SegNet/PoseNet distortion plus "
                "25*archive_bytes/raw_bytes under explicit byte caps"
            ),
            "byte_caps_to_sweep": [36_000, 72_000, 120_000, 150_000, 178_417],
            "control_baseline_bytes": 178_417,
            "rate_formula": "contest_rate_term = 25 * archive_zip_bytes / raw_video_bytes",
            "requires_measured_curve": True,
        },
        "required_measurements": [
            "official_config_tiny_forward_parity",
            "params_split_encoder_decoder_embedding",
            "quantized_tensor_payload_bytes",
            "entropy_coded_payload_bytes",
            "metadata_and_receiver_runtime_bytes",
            "brotli_or_zip_member_bytes",
            "receiver_inflate_parity",
            "macos_or_mlx_prefilter_component_deltas",
            "same_axis_pr95_control_replay_before_beat_claim",
        ],
        "current_gap": {
            "hi_nerv": (
                "local compact runner exposes latent_dim/embed_dim/"
                "decoder_channel plus receiver-visible 2/4/6/7/8 quant noise "
                "and 0/2/4/6/7/8/16/32 decoder-waterfill actions, but not the "
                "official HiNeRV config-family/prune/QuantNoise/bitstream "
                "receiver curve"
            ),
            "snerv": (
                "official SNeRV --modelsize/fc_dim solving is bound and invalid "
                "official controls are preserved fail-closed, but MFU/HFR/SNeRV_T "
                "parity and measured SNAR1 archive-byte replay are still missing"
            ),
        },
        "production_blockers": [
            "nerv_modelsize_to_archive_bytes_curve_missing",
            "official_modelsize_flag_not_receiver_closed_under_contest_byte_caps",
            "quant_bits_not_jointly_optimized_with_score_sensitivity",
            "modelsize_sweep_not_replayed_through_receiver_archive_bytes",
        ],
        "production_hardened_claim": False,
    }


def _scorer_domain_control_policy() -> dict[str, Any]:
    """Bind contest-scorer anatomy to concrete SNeRV/HiNeRV control points."""

    return {
        "schema": "nerv_scorer_domain_control_policy.v1",
        "verdict": (
            "REQUIRED_TRAIN_EXPORT_BITSTREAM_CONTROL__NOT_FULLY_BOUND_TO_SNERV_HINERV"
        ),
        "why_it_matters": (
            "The contest scorer is a known rate-distortion Lagrangian, not a "
            "human-fidelity objective. SegNet sees only frame 1 of each pair, "
            "PoseNet sees frames 0 and 1, and the byte price is fixed by eval.py. "
            "SNeRV and HiNeRV controls must exploit that pair/frame asymmetry "
            "at training, representation, quantization, and archive admission."
        ),
        "scorer_domains": {
            "segnet": {
                "frame_indices_per_pair": [1],
                "frame_0_score_leverage": "none_direct",
                "frame_1_score_leverage": "segnet_and_posenet",
                "input_size_hw": [384, 512],
                "class_count": 5,
                "decision_surface": "last-frame 5-class argmax/logit-margin boundary",
                "score_derivative": 100.0,
                "control_implication": (
                    "Protect, refine, or spend bytes on frame-1 boundary/logit "
                    "regions before visually plausible but scorer-flat pixels."
                ),
            },
            "posenet": {
                "frame_indices_per_pair": [0, 1],
                "input_size_hw": [384, 512],
                "input_channel_contract": "two RGB frames converted to YUV6 pair tensor",
                "scored_pose_dims": [0, 1, 2, 3, 4, 5],
                "score_derivative_symbolic": "5 / sqrt(10 * d_pose)",
                "training_path_requirement": (
                    "use differentiable eval-roundtrip/YUV6 replacement for "
                    "PoseNet Jacobian and scorer-aware training"
                ),
                "control_implication": (
                    "Frame 0 can spend only against pose; frame 1 spends against "
                    "pose plus SegNet. Pair samplers and allocators must encode "
                    "that asymmetry."
                ),
            },
            "rate": {
                "byte_price_symbolic": "25 / raw_uncompressed_total_bytes",
                "water_level_fixed_by_contest": True,
                "no_lambda_search_required": True,
                "admission_rule": (
                    "admit a parameter, atom, packet section, or quantization "
                    "upgrade only when measured expected non-rate score drop per "
                    "charged byte exceeds the fixed byte price"
                ),
            },
        },
        "required_bindings_by_stack": {
            "snerv": [
                "pair/frame scorer-domain sampler for LF/HF and temporal controls",
                "wavelet-group saliency over LF/HF/step-map receiver payloads",
                "frame-1 SegNet boundary protection for generated HF/SR paths",
                "PoseNet pair guard for frame-0 and frame-1 temporal residuals",
                "score-priced intN/zero/RLE/receiver-generation mode assignment",
                "SNAR1 section-value replay before exact dispatch",
            ],
            "hinerv": [
                "decoder-weight and hierarchical-grid saliency inside the trainer",
                "frame-1 SegNet margin loss plus pair PoseNet loss in the same schedule",
                "per-tensor/per-level QuantNoise and pruning driven by scorer deltas",
                "master-gradient-informed pair sampler and recon-pixel weights",
                "receiver-visible waterfill actions for int2/int4/int6/int7/int8/fp16/zero",
                "trained byte-section replay before exact dispatch",
            ],
        },
        "reusable_surfaces_to_consume": [
            "src/tac/master_gradient.py",
            "src/tac/master_gradient_consumers.py",
            "src/tac/master_gradient_wire_in.py",
            "src/tac/optimization/recon_pixel_weight_surface.py",
            "src/tac/analysis/score_exact_saliency.py",
            "src/tac/analysis/nerv_decoder_weight_waterfill.py",
            "src/tac/analysis/hinerv_latent_linf_allocation.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/allocation.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
            "src/tac/cathedral_consumers/pareto_carrier_fit_consumer/__init__.py",
            "src/tac/cathedral_consumers/pact_nerv_ultimate_composition_selector_consumer/__init__.py",
            "src/tac/cathedral_consumers/venn_risk_composition_consumer/__init__.py",
            "src/tac/cathedral_consumers/per_pixel_inverse_steganalysis_real_video_mlx_consumer/__init__.py",
            "src/tac/xray/segnet_margin_polytope.py",
            "src/tac/xray/posenet_se3_lie_algebra.py",
            "src/tac/xray/bilinear_resize_nullspace.py",
            "src/tac/atom/unified_action_bridge.py",
            "tools/master_gradient_xray.py",
            "tools/xray_hardpair_hitlist.py",
            "tools/build_joint_recon_pixel_weight_surface.py",
            "tools/cathedral_autopilot.py",
        ],
        "train_time_control_points": [
            "loss component weights and schedules",
            "hard-pair and frame-domain sampling",
            "joint P18/P19 recon-pixel-weight manifests",
            "decoder-weight saliency/waterfill masks",
            "QAT noise/action bit lattice",
            "pruning and ablation candidate masks",
            "EMA/archive candidate selection by scorer deltas",
        ],
        "export_and_bitstream_control_points": [
            "receiver-visible tensor section manifests",
            "mixed intN/fp16/zero/RLE action packets",
            "wavelet LF/HF group action maps",
            "packed-zero and entropy-coded byte sections",
            "archive-byte oracle rows consumed before queue launch",
        ],
        "production_blockers": [
            "nerv_scorer_domain_controls_not_bound_to_both_snerv_and_hinerv_train_export",
            "master_gradient_to_nerv_pair_frame_sampler_not_bound",
            "segnet_frame1_posenet_pair_asymmetry_not_receiver_priced",
            "nerv_section_value_replay_missing_for_scorer_domain_controls",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _optimal_modelsize_control_policy() -> dict[str, Any]:
    """Describe a contest-optimal modelsize control beyond off-the-shelf flags."""

    return {
        "schema": "nerv_optimal_modelsize_control_policy.v1",
        "verdict": (
            "MODEL_SIZE_IS_OUTER_CAPACITY_CONTROL__OPTIMALITY_REQUIRES_INNER_"
            "SCORER_WATERFILL"
        ),
        "core_answer": (
            "A hard modelsize flag only limits parameter count. For this contest, "
            "a useful modelsize control must also decide which parameters exist, "
            "which are pruned, which precision they receive, and which receiver "
            "grammar stores or regenerates them, all under the fixed scorer byte "
            "price and measured SegNet/PoseNet deltas."
        ),
        "shared_optimality_conditions": [
            "outer loop sweeps byte caps and architecture capacity, not raw params only",
            "inner loop trains against SegNet frame-1 and PoseNet pair response, not human fidelity",
            "every kept parameter or packet section has measured marginal score value above byte price",
            "every pruned or zero-coded region has measured marginal score value below byte price",
            "quantization bits are selected per tensor/group/atom from scorer deltas and charged bytes",
            "receiver archive bytes, metadata, and entropy overhead are part of the modelsize decision",
            "same-axis PR95 control replay is required before any beat claim",
        ],
        "same_for_snerv_and_hinerv": {
            "objective": (
                "minimize SegNet/PoseNet distortion plus 25*archive_bytes/raw_bytes "
                "with modelsize as a priced capacity variable"
            ),
            "control_loop": [
                "propose capacity candidate from byte cap",
                "train with scorer-domain and coder-aware losses",
                "apply saliency/prune/quant/zero waterfill",
                "export receiver-visible archive",
                "measure section bytes and scorer deltas",
                "update Pareto/cathedral/admission surfaces",
            ],
            "not_sufficient": [
                "parameter count alone",
                "uniform int8/int4",
                "visual reconstruction PSNR",
                "MLX advisory score without receiver archive proof",
            ],
        },
        "fine_grained_component_budget_controls": {
            "principle": (
                "Global modelsize is only the outer byte cap; the controller must "
                "allocate that cap across archive/blob sections and then recurse "
                "inside each section until every atom has a measured score-per-byte "
                "reason to exist."
            ),
            "shared_sections": [
                "decoder_weight_sections",
                "latent_or_feature_sections",
                "quant_scale_zero_point_sections",
                "packed_zero_or_rle_sections",
                "entropy_codebooks_and_headers",
                "receiver_runtime_metadata",
            ],
            "snerv_sections": [
                "lf_plane_payload",
                "hf_generator_or_decoder_payload",
                "wavelet_group_step_maps",
                "temporal_context_payload",
                "mfu_hfr_tub_control_payload",
                "snar1_section_manifest",
            ],
            "hinerv_sections": [
                "hierarchical_feature_grids",
                "convnext_decoder_blocks",
                "patch_frame_geometry_tables",
                "pruning_masks",
                "quantnoise_level_maps",
                "torchac_style_bitstream_sections",
            ],
            "per_section_actions": [
                "delete_or_receiver_generate",
                "zero_or_run_length",
                "int1_int2_int4_int6_int7_int8",
                "fp16_protect",
                "entropy_recode",
                "increase_capacity",
            ],
            "admission_metric": (
                "expected non-rate score drop from section action divided by "
                "charged archive-byte delta; admit only above contest byte price"
            ),
        },
        "different_for_snerv": {
            "primary_capacity_axes": [
                "fc_dim/emb_size official modelsize adapter",
                "MFU/HFR/TUB/SNeRV-T source controls",
                "DWT level and LF/HF representation split",
                "step-map and decoder payload grammar",
                "learned LF/HF/SR receiver-side generation capacity",
            ],
            "main_risk": (
                "explicit LF storage can dominate rate; modelsize must choose "
                "stored-versus-generated LF/HF structure, not merely decoder width"
            ),
            "highest_value_control": (
                "score-preserving receiver-side LF/HF/SR generator with wavelet-group "
                "waterfill and SNAR1 section-value replay"
            ),
        },
        "different_for_hinerv": {
            "primary_capacity_axes": [
                "hierarchical feature-grid depth and resolution",
                "patch/frame geometry and interpolation mode",
                "ConvNeXt/depthwise-MLP width",
                "adaptive pruning ratio and prune weight",
                "QuantNoise/STE levels and torchac-style bitstream sections",
            ],
            "main_risk": (
                "decoder/grid weights may carry most score leverage; modelsize must "
                "rank and price decoder-weight sections rather than assume latent "
                "tweaks will fix distortion"
            ),
            "highest_value_control": (
                "decoder-weight and feature-grid saliency waterfill inside the real "
                "trainer, followed by measured quantized byte-section export"
            ),
        },
        "smarter_than_off_the_shelf_controls": [
            "solve modelsize from target archive bytes rather than only target parameters",
            "split capacity by scorer domain: frame-1 SegNet boundary, pair PoseNet, and rate",
            "learn per-section marginal utility curves during training",
            "adapt quant/prune/zero modes recursively after each scorer replay",
            "bind modelsize rows into cathedral/Pareto/atom consumers for portfolio selection",
        ],
        "production_blockers": [
            "contest_optimal_modelsize_controller_not_yet_bound_to_training_loop",
            "modelsize_candidate_marginal_utility_curves_missing",
            "snerv_modelsize_lf_hf_generation_tradeoff_not_receiver_replayed",
            "hinerv_modelsize_decoder_grid_section_value_not_receiver_replayed",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def discover_pr95_baseline(
    *,
    repo_root: Path,
    upstream_repo_dir: str | Path | None,
    pr95_intake_root: str | Path | None,
    pr95_pr_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Discover the PR95/HNeRV baseline without mutating upstream or caches."""

    metadata = dict(pr95_pr_metadata or {})
    blockers: list[str] = []
    upstream = _path_record(upstream_repo_dir)
    if upstream["present"]:
        upstream_git = _git_info(Path(str(upstream["path"])))
        upstream["git"] = upstream_git
        if not upstream_git.get("head"):
            blockers.append("upstream_repo_git_head_missing")
        if not upstream_git.get("remote_origin"):
            blockers.append("upstream_repo_git_remote_origin_missing")
        elif not _remote_origin_matches_pr95(upstream_git.get("remote_origin")):
            blockers.append("upstream_repo_git_remote_origin_not_pr95_source")

    intake = _path_record(pr95_intake_root)
    archive = None
    runtime_files: list[dict[str, Any]] = []
    if intake["present"]:
        intake_root = Path(str(intake["path"]))
        archive_path = intake_root / "archive.zip"
        if archive_path.is_file():
            archive = _file_manifest(archive_path)
        else:
            blockers.append("pr95_public_archive_zip_missing")
        submission_root = intake_root / "source" / "submissions" / PR95_SUBMISSION
        for rel in (
            "inflate.sh",
            "inflate.py",
            "src/model.py",
            "src/codec.py",
            "src/optim.py",
            "src/stages/stage8_muon_finetune.py",
        ):
            path = submission_root / rel
            if path.is_file():
                runtime_files.append(_file_manifest(path))
            else:
                blockers.append(f"pr95_runtime_file_missing:{rel}")
    else:
        blockers.append("pr95_public_intake_root_missing")

    if not upstream["present"]:
        blockers.append("upstream_repo_dir_missing")

    metadata_url = metadata.get("url")
    metadata_state = metadata.get("state")
    metadata_head_sha = metadata.get("headRefOid")
    metadata_head_ref = metadata.get("headRefName")
    if not metadata:
        blockers.append("pr95_pr_metadata_missing")
    if not metadata_url:
        blockers.append("pr95_pr_url_missing")
    elif str(metadata_url) != PR95_PR_URL:
        blockers.append("pr95_pr_url_mismatch")
    if not metadata_state:
        blockers.append("pr95_pr_state_missing")
    elif str(metadata_state).upper() != "MERGED":
        blockers.append(f"pr95_pr_state_not_merged:{_blocker_slug(metadata_state)}")
    if not metadata_head_sha:
        blockers.append("pr95_pr_head_sha_missing")
    if not metadata_head_ref:
        blockers.append("pr95_pr_head_ref_missing")
    if (
        upstream["present"]
        and metadata_head_sha
        and upstream.get("git", {}).get("head")
        and str(upstream["git"]["head"]) != str(metadata_head_sha)
    ):
        blockers.append("upstream_repo_head_sha_mismatch_pr95_metadata")

    proof_tool = repo_root / "tools" / "prove_pr95_public_archive_runtime_consumption.py"
    if not proof_tool.is_file():
        blockers.append("pr95_runtime_consumption_proof_tool_missing")

    return {
        "schema": "pr95_hnerv_baseline_control_pointer.v1",
        "source_pr": PR95_PR_NUMBER,
        "source_url": metadata_url or PR95_PR_URL,
        "title": metadata.get("title", "hnerv_muon submission (0.20)"),
        "state": metadata_state,
        "head_sha": metadata_head_sha,
        "head_ref": metadata_head_ref,
        "submission": PR95_SUBMISSION,
        "role": "baseline_control_to_beat",
        "upstream_repo_dir": upstream,
        "intake_root": intake,
        "archive": archive,
        "runtime_files": runtime_files,
        "existing_proof_tools": [
            "tools/prove_pr95_public_archive_runtime_consumption.py",
            "tools/prove_pr95_public_archive_full_frame_parity.py",
            "tools/run_pr95_stage8_from_public_archive.py",
            "tools/run_pr95_hnerv_linf_carrier.py",
        ],
        "same_axis_requirement": (
            "SNeRV/HiNeRV winner must be compared against PR95 on the same "
            "archive, runtime, inflate, and eval hardware axis before any beat claim"
        ),
        "blockers": _unique(blockers),
    }


def discover_dispatch_blockers(
    active_claims_path: str | Path | None,
    *,
    now_utc: str | datetime | None = None,
    ttl_hours: float = 24.0,
) -> list[str]:
    """Return active exact/full-video blockers from the lane claim table."""

    if active_claims_path is None:
        return ["active_claims_table_not_supplied"]
    path = Path(active_claims_path)
    if not path.is_file():
        return ["active_claims_table_missing"]
    now = _parse_utc(now_utc) or datetime.now(UTC)

    blockers: list[str] = []
    rows = _latest_claim_rows_by_job(_parse_claim_rows(path.read_text(encoding="utf-8")))
    for claim in rows:
        status = str(claim.get("status", ""))
        if _status_is_terminal(status):
            continue
        if not _claim_is_active(claim):
            continue
        lane_id = str(claim.get("lane_id", ""))
        if lane_id == PR101_LANE_ID:
            blockers.append("pr101_cpu_recovery_pending_blocks_new_exact_or_full_video")
            continue
        if lane_id.startswith("lane_z5_rao_ballard_paired_cuda_ratification_wave2a_20260531"):
            blockers.append("z5_rao_ballard_modal_claims_still_need_terminal_adjudication")
            continue
        if _claim_blocks_exact_or_full_video(claim) and _claim_within_active_window(
            claim,
            now=now,
            ttl_hours=ttl_hours,
        ):
            blockers.append(f"active_exact_or_full_video_claim:{_blocker_slug(lane_id)}")
    return _unique(blockers)


def _snerv_stack() -> dict[str, Any]:
    return {
        "stack_id": "snerv",
        "role": "top_priority_carrier",
        "status": "local_stack_optimization_active",
        "current_read": (
            "linear/closed-form coordinate and scalar HF sweeps are not promotion "
            "evidence; explicit mixed decoder modes and scorer-loop QAT are the "
            "next useful local surfaces"
        ),
        "required_components": [
            "architecture",
            "learned_or_nonlinear_decoder_qat",
            "pose_guarded_scorer_loop_fit",
            "linf_oracle_allocator_inside_decoder_weight_training",
            "mixed_precision_decoder_grammar",
            "byte_closed_receiver_proof",
            "paired_pr95_same_axis_control",
        ],
        "existing_surfaces": [
            "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
            "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
            "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
            "tools/run_snerv_scorer_loop_decoder_qat_smoke.py",
            "tools/probe_snerv_decoder_mode_assignments.py",
            "tools/prove_snerv_receiver_archive.py",
        ],
        "next_gate": (
            "pair-robust scorer-loop or NES decoder-QAT continuation with PoseNet "
            "as hard guard and receiver-decoded mixed precision bytes preserved"
        ),
        "optimization_scope": "carrier-specific full stack; not a shared HNeRV clone",
    }


def _hinerv_stack() -> dict[str, Any]:
    return {
        "stack_id": "hinerv",
        "role": "top_priority_carrier",
        "status": "local_stack_optimization_active",
        "current_read": (
            "recent full-600 MLX prefilter was byte-closed but distortion-bad; "
            "rate plumbing is real enough, fit must move upstream into longer "
            "score-aware/coder-aware decoder-weight training"
        ),
        "required_components": [
            "hierarchical_architecture",
            "coder_aware_qat",
            "joint_p18_p19_oracle_weighting",
            "dense_decoder_vjp_linf_allocator",
            "pr95_faithful_curriculum_control",
            "byte_closed_receiver_proof",
            "full_video_mlx_prefilter_then_paired_cpu_cuda",
        ],
        "existing_surfaces": [
            "src/tac/analysis/hinerv_latent_linf_allocation.py",
            "src/tac/substrates/hi_nerv/score_aware_loss.py",
            "tools/run_compact_renderer_mlx_spine_runner.py",
            ".omx/research/codex_findings_hinerv_coder_aware_qat_wired_20260601T224653Z_codex.md",
            ".omx/research/codex_findings_hinerv_600pair_joint_p18_p19_full_prefilter_20260602T021050Z_codex.md",
        ],
        "next_gate": (
            "longer staged real-teacher SegNet/PoseNet training with coder-aware "
            "QAT and local MLX prefilter; no exact spend until prefilter enters a "
            "plausible replay band"
        ),
        "optimization_scope": "carrier-specific full stack; not a SNeRV codec wrapper",
    }


def _full_stack_priority() -> dict[str, Any]:
    return {
        "schema": "nerv_individual_fractal_full_stack_priority.v1",
        "top_priority_carriers": list(TOP_PRIORITY_CARRIERS),
        "components": list(FULL_STACK_COMPONENTS),
        "policy": (
            "Optimize SNeRV and HiNeRV as separate end-to-end stacks at every "
            "component boundary, then compose shared enhancers only through "
            "explicit byte/eval gates."
        ),
        "do_not_average_stacks": True,
        "do_not_promote_shared_enhancer_as_carrier": True,
        "pr95_control_required_for_any_beat_claim": True,
        "fractality_rule": (
            "Each stack component needs its own hypothesis, local command, byte "
            "accounting surface, guard, and promotion blocker; a global stack "
            "memo alone is not implementation authority."
        ),
    }


def _fractal_work_orders() -> list[dict[str, Any]]:
    return [
        {
            "stack_id": "snerv",
            "priority": "top_carrier",
            "work_order": _component_work_orders(
                {
                    "architecture": (
                        "learn nonlinear decoder/HF restoration against real "
                        "SegNet/PoseNet response, not scalar coordinate sweeps"
                    ),
                    "optimizer_qat": (
                        "run pair-robust scorer-loop or NES decoder-QAT with "
                        "PoseNet hard guard and per-pair deltas"
                    ),
                    "allocator": (
                        "move L-infinity allocation into decoder-weight fit; "
                        "latents are diagnostic only until leverage reappears"
                    ),
                    "archive_grammar": (
                        "replace fp32/fake-quant receiver payload with explicit "
                        "mixed decoder modes, int planes, or decoder-delta packing"
                    ),
                    "receiver_proof": (
                        "prove receiver-decoded byte accounting before any "
                        "full-600 authority"
                    ),
                    "eval_control": (
                        "compare against PR95 only after same-axis archive/runtime "
                        "control replay"
                    ),
                },
                local_command_ids=(
                    "snerv_pair_robust_decoder_qat_continuation",
                    "snerv_explicit_decoder_mode_triage",
                ),
            ),
        },
        {
            "stack_id": "hinerv",
            "priority": "top_carrier",
            "work_order": _component_work_orders(
                {
                    "architecture": (
                        "continue hierarchical NeRV carrier training; fit is the "
                        "active blocker, not proof that the rate knob is fake"
                    ),
                    "optimizer_qat": (
                        "stage real-teacher SegNet/PoseNet loss, coder-aware QAT, "
                        "and PR95/Muon curriculum controls"
                    ),
                    "allocator": (
                        "use dense decoder VJP L-infinity allocator with joint "
                        "P18/P19 weighting inside decoder-weight optimization"
                    ),
                    "archive_grammar": (
                        "keep byte budget parameterized and close the quantized "
                        "receiver grammar before exact replay"
                    ),
                    "receiver_proof": (
                        "prove full-600 archive/runtime consumption with no MLX "
                        "or advisory shortcut"
                    ),
                    "eval_control": (
                        "prefilter locally on MLX, then replay paired CPU/CUDA "
                        "only after blocker claims terminalize"
                    ),
                },
                local_command_ids=("hinerv_real_teacher_qat_continuation",),
            ),
        },
    ]


def _component_work_orders(
    descriptions: Mapping[str, str],
    *,
    local_command_ids: Sequence[str],
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    for component in FULL_STACK_COMPONENTS:
        orders.append(
            {
                "component": component,
                "next_action": descriptions[component],
                "local_command_ids": list(local_command_ids),
                "requires_receiver_byte_accounting": component
                in {"archive_grammar", "receiver_proof", "eval_control"},
                "promotion_authority": False,
            }
        )
    return orders


def _synergy_enhancers() -> list[dict[str, Any]]:
    return [
        {
            "enhancer_id": "sr_nerv_trained_scorer_aware",
            "priority": "highest_enhancer",
            "not_a_standalone_carrier_stack": True,
            "policy": (
                "low-internal-resolution carrier plus trained/scorer-aware SR; "
                "zero-parameter interpolation remains no-go"
            ),
        },
        {
            "enhancer_id": "rnerv_per_video_config_optimizer",
            "priority": "winner_optimizer",
            "not_a_standalone_carrier_stack": True,
            "policy": "optimize the winning SNeRV/HiNeRV carrier configuration per video",
        },
        {
            "enhancer_id": "ffnerv_flow_pose_channel",
            "priority": "pose_channel_enhancer",
            "not_a_standalone_carrier_stack": True,
            "policy": "flow-conditioning is a pose-channel bolt-on after carrier winner emerges",
        },
        {
            "enhancer_id": "boostnerv_decoder_temporal_affine",
            "priority": "cheap_synergy_multiplier",
            "not_a_standalone_carrier_stack": True,
            "policy": "conditional decoder/temporal-affine bolt-on for the winner",
        },
    ]


def _shared_promotion_gates() -> list[dict[str, Any]]:
    return [
        {
            "gate": "pr101_cpu_recovery_terminal",
            "required": True,
            "authority_after_pass": "may consider exact/full-video queue again",
        },
        {
            "gate": "byte_closed_full_600_receiver_proof",
            "required": True,
            "authority_after_pass": "candidate may enter paired auth-eval planning only",
        },
        {
            "gate": "pr95_same_axis_control",
            "required": True,
            "authority_after_pass": "beat/no-beat statement may be made only on that axis",
        },
        {
            "gate": "paired_contest_cpu_cuda_auth_eval",
            "required": True,
            "authority_after_pass": "promotion discussion may begin",
        },
    ]


def _next_local_actions(
    *,
    exact_blocked: bool,
    upstream_repo_dir: str | None,
) -> list[dict[str, Any]]:
    dispatch_note = (
        "blocked while active exact/full-video claims remain nonterminal"
        if exact_blocked
        else "allowed only after lane claim and byte-closed proof"
    )
    upstream_arg = upstream_repo_dir or "<UPSTREAM_REPO_DIR>"
    return [
        {
            "id": "poll_pr101_cpu_recovery",
            "axis": "[contest-CPU recovery]",
            "command": (
                ".venv/bin/python tools/recover_modal_auth_eval.py "
                "--call-id fc-01KT2BZT54G6CXPMD94SY43MMH "
                "--output-dir experiments/results/modal_auth_eval_cpu/"
                "pr101_storage_order_len24_cpu_20260601T1955Z"
            ),
            "allowed_now": True,
            "authority_after_pass": "terminalize claim only; no score shortcut",
        },
        {
            "id": "snerv_pair_robust_decoder_qat_continuation",
            "axis": "[macOS-CPU advisory]",
            "command": (
                ".venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py "
                "--n-pairs 4 --levels 3 --target-bits-per-coeff 2.0 "
                "--pair-stride 8 --search-mode nes_pair_robust --max-trials 2 "
                "--byte-pressure-multiplier 8.0 --max-archive-byte-growth 0 "
                "--pose-slack 0.0 --seg-slack 0.00005 "
                "--pair-guard-min-score-improved-fraction 0.75 "
                "--pair-guard-max-pose-worsened-fraction 0.0 "
                "--out .omx/research/snerv_scorer_loop_decoder_qat_next_<UTC>.json"
            ),
            "allowed_now": True,
            "authority_after_pass": "local triage only",
        },
        {
            "id": "snerv_explicit_decoder_mode_triage",
            "axis": "[macOS-CPU advisory]",
            "command": (
                ".venv/bin/python tools/probe_snerv_decoder_mode_assignments.py "
                "--n-pairs 2 --levels 1 --mode-plan magnitude_heuristic "
                f"--mode-plan fp16,int4,int4 --upstream-dir {upstream_arg}"
            ),
            "allowed_now": True,
            "authority_after_pass": "local receiver-decoded rate/fit triage only",
        },
        {
            "id": "hinerv_real_teacher_qat_continuation",
            "axis": "[macOS-MLX research-signal]",
            "command": (
                ".venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py "
                "--execute-family hi_nerv --coder-aware-qat "
                "--real-teacher-segnet-posenet --num-pairs 32 --epochs 8"
            ),
            "allowed_now": True,
            "authority_after_pass": "local MLX prefilter only",
        },
        {
            "id": "paired_exact_eval_or_full_video",
            "axis": "[contest-CPU]/[contest-CUDA]",
            "command": "deferred",
            "allowed_now": False,
            "blocked_reason": dispatch_note,
        },
    ]


def _path_record(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False}
    p = Path(path).resolve()
    return {"path": p.as_posix(), "present": p.exists(), "is_dir": p.is_dir()}


def _file_manifest(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _git_info(path: Path) -> dict[str, Any]:
    return {
        "head": _git_output(path, "rev-parse", "HEAD"),
        "branch": _git_output(path, "branch", "--show-current"),
        "remote_origin": _git_output(path, "remote", "get-url", "origin"),
    }


def _remote_origin_matches_pr95(remote_origin: object) -> bool:
    text = str(remote_origin or "").strip().lower()
    return "commaai/comma_video_compression_challenge" in text


def _git_output(path: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", path.as_posix(), *args],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _parse_utc(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_claim_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        if cells[0] == "timestamp_utc" or set(cells[0]) <= {"-"}:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _latest_claim_rows_by_job(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        claim = dict(row)
        key = (
            str(claim.get("lane_id", "")),
            str(claim.get("instance_job_id", "")),
        )
        previous = latest.get(key)
        if previous is None or _claim_row_is_newer(claim, previous):
            latest[key] = claim
    return list(latest.values())


def _claim_row_is_newer(
    candidate: Mapping[str, str],
    previous: Mapping[str, str],
) -> bool:
    candidate_ts = _parse_utc(candidate.get("timestamp_utc", ""))
    previous_ts = _parse_utc(previous.get("timestamp_utc", ""))
    if previous_ts is None:
        return candidate_ts is not None
    if candidate_ts is None:
        return False
    return candidate_ts > previous_ts


def _status_is_terminal(status: str) -> bool:
    lowered = status.lower()
    return any(lowered.startswith(prefix) for prefix in TERMINAL_STATUS_PREFIXES)


def _claim_is_active(claim: Mapping[str, str]) -> bool:
    status = claim.get("status", "").lower()
    lane_id = claim.get("lane_id", "").lower()
    notes = claim.get("notes", "").lower()
    haystack = " ".join((status, lane_id, notes))
    return any(token in haystack for token in ACTIVE_STATUS_TOKENS)


def _claim_blocks_exact_or_full_video(claim: Mapping[str, str]) -> bool:
    status = claim.get("status", "").lower()
    lane_id = claim.get("lane_id", "").lower()
    platform = claim.get("platform", "").lower()
    notes = claim.get("notes", "").lower()
    instance = claim.get("instance_job_id", "").lower()
    haystack = " ".join((lane_id, status, platform, notes, instance))
    if any(token in haystack for token in EXACT_OR_FULL_VIDEO_TOKENS):
        return True
    return platform in REMOTE_EVAL_PLATFORMS and any(
        token in status for token in ("eval", "dispatch", "spawn")
    )


def _claim_within_active_window(
    claim: Mapping[str, str],
    *,
    now: datetime,
    ttl_hours: float,
) -> bool:
    ttl_seconds = max(float(ttl_hours), 0.0) * 3600.0
    timestamp = _parse_utc(claim.get("timestamp_utc", ""))
    if timestamp is not None and (now - timestamp).total_seconds() <= ttl_seconds:
        return True
    predicted_eta = _parse_utc(claim.get("predicted_eta_utc", ""))
    return predicted_eta is not None and predicted_eta >= now


def _blocker_slug(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum() or char in ("_", "-"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "unknown_lane"


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "AXIS_TAG",
    "DEFAULT_LANE_ID",
    "FALSE_AUTHORITY",
    "FULL_STACK_COMPONENTS",
    "PR95_PR_NUMBER",
    "PR95_PR_URL",
    "PR101_LANE_ID",
    "SCHEMA",
    "TERMINAL_STATUS_PREFIXES",
    "TOP_PRIORITY_CARRIERS",
    "NervTopPriorityStackSeamError",
    "build_nerv_top_priority_stack_seam",
    "build_source_faithfulness_matrix",
    "discover_dispatch_blockers",
    "discover_pr95_baseline",
]
