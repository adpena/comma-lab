# SPDX-License-Identifier: MIT
"""ATW Codec V1 SubstrateContract registration."""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)

ATW_CODEC_V1_CONTRACT = SubstrateContract(
    id="atw_codec_v1",
    lane_id="lane_atw_codec_design_v1_20260515",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=".omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md",
    archive_grammar="atw1_monolithic_wyner_ziv_side_info_head",
    parser_section_manifest={
        "header": "ATW1 fixed header",
        "encoder_blob": "brotli(fp16 encoder state)",
        "decoder_blob": "brotli(fp16 decoder state)",
        "latent_residual_blob": "residual latent payload",
        "wz_side_info_head_blob": "decoder-side prediction table/head",
        "meta_blob": "sorted json provenance",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=450,
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
    recipe_canary_dependency="lane_z4_cooperative_receiver_loss_step2_20260514",
    cost_band_epochs=200,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=5.00,
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,
    hook_probe_disambiguator=None,
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_220_operational_mechanism_declared",
    ),
    hook_not_applicable_rationale={
        "hook_continual_learning_anchor_kind": (
            "No empirical paired-axis anchor exists; posterior update waits for "
            "the entropy probe and paired CPU+CUDA exact eval."
        ),
        "hook_probe_disambiguator": (
            "The four ATW knob-zero regimes are documented in the design memo; "
            "a runnable probe script is a follow-up before dispatch ranking."
        ),
    },
)


@register_substrate(ATW_CODEC_V1_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "ATW Codec V1 is research-only. Use "
        "experiments/train_substrate_atw_codec_v1.py after the entropy probe "
        "and Phase 2 council approval."
    )


__all__ = ["ATW_CODEC_V1_CONTRACT", "main"]
