# SPDX-License-Identifier: MIT
"""SubstrateContract for the cascade_c_prime_frame_1_segnet_waterfill substrate.

Per Catalog #241/#242 + ``src/tac/substrate_registry/contract.py`` 36-field
canonical schema. Per Catalog #240 ``research_only=True`` + ``dispatch_enabled
=False`` until per-substrate symposium per Catalog #325 lands a PROCEED
verdict within the 14-day window (2026-05-26 -> 2026-06-09).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": this substrate
implements Atick-Redlich asymmetric scorer channel theory (1990 cooperative-
receiver) — frame-0 perturbations have STRUCTURAL zero SegNet cost (SegNet's
``x[:,-1,...]`` slice; sister to Catalog #220 frame-0/frame-1 asymmetry) while
frame-1 perturbations cost M SegNet bytes + N' PoseNet bytes. The per-pair
Lagrangian dual routing decision per ``tac.findings_lagrangian`` Phase 1-3 wire-in
(Catalog #355) selects min over the joint frame-0 + frame-1 mode menu.

Per Cascade C' synthesis (commit ``2d5337f27``) MLX-LOCAL prediction:
``predicted_delta = -0.058820`` [macOS-MLX research-signal; paired-CUDA-pending]
at PR106 frontier operating point pose_avg=3.4e-5. 25.2% per-pair frame-1
routing matches sister #1324 PoseNet-null 22.3% within 3pp.
"""

from __future__ import annotations

from tac.substrate_registry import SubstrateContract

__all__ = ["CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT"]


CASCADE_C_PRIME_FRAME_1_SEGNET_WATERFILL_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="cascade_c_prime_frame_1_segnet_waterfill",
    lane_id="lane_cascade_c_prime_option_a_build_scaffold_20260526",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        # Cascade C' parent synthesis landing commit 2d5337f27 (subagent
        # aa563bbb31adadfd6) + Cascade C' Modal validation DEFERRED-pending-
        # substrate-scaffold verdict commit aa1a9cf32 (subagent a1d16a40f4a722e26)
        # operator-routable Option A.
        "feedback_cascade_c_prime_paired_cuda_validation_deferred_pending_substrate_scaffold_20260526.md#option_a"
    ),
    # 2.2 Architecture & runtime (Catalog #124 + HNeRV parity L1-L13)
    archive_grammar=(
        "CH-CCP-FRAME1-WATERFILL monolithic single-file 0.bin: header "
        "(magic=CCPF, version=1) + routing-decision sidecar (1-bit-per-pair "
        "packed + brotli compressed; ~79 bytes for 600 pairs per Cascade C' "
        "synthesis Option B empirical) + frame-0 menu index per pair (4-bit "
        "K=16 Huffman-coded) + frame-1 menu index per pair (3-bit K=8 Huffman-"
        "coded) + per-pair pose deltas (uint8 quantized; sister of PR110 grammar)."
    ),
    parser_section_manifest={
        "header": "CCPF_magic_and_version_byte",
        "routing_decision_sidecar": "1bit_per_pair_packed_brotli_compressed",
        "frame_0_menu_index_stream": "huffman_coded_4bit_per_pair",
        "frame_1_menu_index_stream": "huffman_coded_3bit_per_pair",
        "pose_stream": "uint8_quantized_pose_deltas_per_pair",
    },
    inflate_runtime_loc_budget=200,  # HNeRV parity L4 budget
    runtime_dep_closure=("numpy>=1.24", "brotli>=1.0"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=900,  # substrate_engineering exception per L7
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (Catalog #220)
    # SCAFFOLD_DEFERRED_INTEGRATION: routing-decision sidecar is ~79 bytes
    # (well under 1 KB Catalog #220 anti-pattern threshold). Per-substrate
    # symposium per Catalog #325 gates promotion to OPERATIONAL.
    archive_bytes_added=(
        "routing-decision sidecar: ~79 bytes (brotli-compressed 1-bit-per-pair "
        "packed flags for 600 pairs). Frame-1 menu index stream adds ~225 bytes "
        "(3-bit per pair for ~150 frame-1-routed pairs); frame-0 menu index "
        "stream offset by removing the corresponding frame-0-only-baseline bytes. "
        "Per Cascade C' synthesis (Option B) net rate-axis overhead is dominated "
        "by 79-byte sidecar (= 5.26e-5 score-axis rate term)."
    ),
    score_improvement_mechanism_status="SCAFFOLD_DEFERRED_INTEGRATION",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band
    cost_band_epochs=1,  # No training; epochs=1 means one compress pass
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.50,  # SCAFFOLD smoke; per-substrate symposium gates dispatch
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="atick_redlich_asymmetric_scorer_channel_lagrangian_routing",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=(
        "tools/probe_cascade_c_prime_atick_redlich_asymmetric_channel_routing_disambiguator.py"
    ),
    # 2.7 + 2.8
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_244_canonical_nvml_block_inherited",
        "catalog_270_dispatch_optimization_protocol_complete",
        "catalog_295_submission_inflate_self_contained_via_local_helpers",
        "catalog_324_predicted_band_validation_pending_post_training",
        "catalog_325_per_substrate_symposium_pending",
        "catalog_344_canonical_equation_pending_formalization",
    ),
    hook_not_applicable_rationale={
        "hook_bit_allocator_class": (
            "the substrate IS the bit allocator — per-pair Lagrangian dual "
            "routing decision selects min over joint frame-0 + frame-1 mode "
            "menu in single argmin pass; no separate ibps/lsq/uniform allocator "
            "applies. Closed-form routing per Atick-Redlich asymmetric channel."
        ),
    },
)
