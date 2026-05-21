# SPDX-License-Identifier: MIT
"""SubstrateContract for the nscs06 v8 chroma-LUT substrate.

Per Catalog #241/#242 + ``src/tac/substrate_registry/contract.py`` 36-field
canonical schema. Per Catalog #240 ``research_only=True`` + ``dispatch_enabled
=False`` until per-substrate symposium per Catalog #325 lands a PROCEED
verdict within the 14-day window (2026-05-21 -> 2026-06-04).
"""

from __future__ import annotations

from tac.substrate_registry import SubstrateContract

__all__ = ["NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT"]


NSCS06_V8_CHROMA_LUT_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="nscs06_v8_chroma_lut",
    lane_id="lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        # CASCADE COMPRESSION symposium commit d125af6c3 PRIORITY 3 + HONEST
        # CASCADE-MORTALITY ASSESSMENT commit d884dd6aa Rank 2 + NSCS06 v6 ->
        # v7 cargo-cult-unwind methodology rescue path commit 4292c8ce2.
        "feedback_t3_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520.md#priority-3"
    ),
    # 2.2 Architecture & runtime (Catalog #124)
    archive_grammar=(
        "CH08 v1/v2 monolithic single-file 0.bin: header (magic=CH08, "
        "version=1 inline LUT or version=2 procedural seed) + LUT payload "
        "(4096-byte dense (levels, classes, 3) uint8 inline OR 32-byte PCG64 "
        "seed) + per-pair pose deltas (uint8 quantized) + raw uint8 quantized "
        "grayscale stream (low-res). NO neural weights. NO PyTorch at inflate. "
        "Per canonical equation #26 _INCLUDED_CONTEXTS nscs06_v8_chroma_lut "
        "context: v2 procedural seed predicted ΔS = -0.002706 [prediction] "
        "vs v1 inline LUT baseline."
    ),
    parser_section_manifest={
        "header": "CH08_v1_or_v2_magic_and_version",
        "lut_payload": "v1_dense_uint8_lut_or_v2_pcg64_seed",
        "pose_stream": "uint8_quantized_pose_deltas",
        "grayscale_stream": "raw_uint8_quantized_grayscale_lowres",
    },
    inflate_runtime_loc_budget=200,  # substrate_engineering exception per L7
    runtime_dep_closure=("numpy>=1.24", "Pillow>=10"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=900,  # substrate_engineering exception per L7
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (Catalog #220)
    # SCAFFOLD_DEFERRED_INTEGRATION: the canonical CH08 v2 archive bytes are
    # 32-byte seed; well under the 1 KB threshold that triggers the Catalog
    # #220 forbidden anti-pattern. Per-substrate symposium per Catalog #325
    # gates promotion to OPERATIONAL.
    archive_bytes_added=(
        "v2 procedural seed: 32 bytes (replaces v1 inline 4096-byte LUT). "
        "Per canonical equation #26 _NSCS06_V8_BYTES_SAVED = 4064 bytes saved."
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
    cost_band_epochs=1,  # NO training; epochs=1 means one compress pass
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.50,  # SCAFFOLD smoke; per-substrate symposium gates dispatch
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token="procedural_chroma_lut_replacement",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=(
        "tools/probe_nscs06_v8_chroma_lut_canonical_equation_26_in_domain_disambiguator.py"
    ),
    # 2.7 + 2.8
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "v8 is a hand-rolled codec sister to v7; per-byte sensitivity is "
            "captured directly by hook_pareto_constraint=rate_distortion_v1 "
            "(the closed-form chroma-LUT allocation IS the sensitivity surface)"
        ),
        "hook_bit_allocator_class": (
            "the substrate IS the bit allocator (closed-form (level, class) "
            "chroma-LUT derivation); no separate ibps/lsq/uniform allocator applies"
        ),
    },
)
