# SPDX-License-Identifier: MIT
"""ATW Codec V2 SubstrateContract registration (META layer Catalog #241/#242)."""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)

ATW_CODEC_V2_CONTRACT = SubstrateContract(
    id="atw_codec_v2",
    lane_id="lane_atw_codec_v2_substrate_build_20260516",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=(
        ".omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md"
    ),
    archive_grammar=(
        "atw2_monolithic_wyner_ziv_g1_distill_b3_cdf_table"
    ),
    parser_section_manifest={
        "header": "ATW2 48-byte header (magic ATW2 + version + variant + 6 u16 dims + 8 u32 lengths)",
        "encoder_blob": "brotli(fp16 encoder state)",
        "decoder_blob": "brotli(fp16 decoder state)",
        "wz_head_blob": "brotli(fp16 WZ side-info head state)",
        "distill_head_blob": "brotli(fp16 G1 scorer-class distill head state)",
        "latent_residual_blob": "int8 Wyner-Ziv residual payload",
        "class_prior_table_blob": "fp16 per-pair scorer-class prior side-info",
        "cdf_table_blob": "fp16 B3 scorer-conditional CDF table",
        "meta_blob": "sorted json provenance with atw_v2_codec_meta tag",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "numpy"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=600,
    no_op_detector_planned=True,
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=False,
    recipe_research_only=True,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="lane_atw_codec_design_v1_20260515",
    cost_band_epochs=200,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=10.00,
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token="atw_codec_v2_cooperative_receiver",
    hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,
    hook_probe_disambiguator=(
        "tools/run_atw_v2_d4_probe_from_a1.py"
    ),
    catalog_compliance_declarations=(
        "catalog_124_archive_grammar_8_fields_declared",
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_215_min_smoke_gpu_consistent",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_244_modal_nvml_env_block_auto_emitted",
    ),
    hook_not_applicable_rationale={
        "hook_continual_learning_anchor_kind": (
            "No empirical paired-axis anchor exists at scaffold landing. The "
            "2026-05-16 D4 probe at "
            ".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json "
            "returned INDEPENDENT (MI=0.006385502752 bits/symbol), so the "
            "measured A1-latent/class-conditioning surface is deferred and "
            "ATW v2 remains research_only. The posterior update fires only "
            "after a richer side-info reactivation probe plus Phase 2 council "
            "approval and paired CPU+CUDA exact eval custody land."
        ),
    },
)


@register_substrate(ATW_CODEC_V2_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Forwarding entrypoint for substrate registry consumers.

    The canonical executable entry is
    ``experiments/train_substrate_atw_codec_v2.py``; this function exists
    only to satisfy the @register_substrate decorator contract.
    """
    raise NotImplementedError(
        "ATW Codec V2 trainer entry point lives at "
        "experiments/train_substrate_atw_codec_v2.py. The registered_substrate "
        "module exists only for Catalog #241/#242 META layer registration; "
        "invoke the trainer module directly for smoke / full / dispatch paths."
    )


__all__ = ["ATW_CODEC_V2_CONTRACT", "main"]
