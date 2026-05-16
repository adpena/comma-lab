# SPDX-License-Identifier: MIT
"""NSCS03 end-to-end Ballé joint codec SubstrateContract registration."""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)

NSCS03_END_TO_END_BALLE_CONTRACT = SubstrateContract(
    id="nscs03_end_to_end_balle_joint_codec",
    lane_id="lane_nscs03_end_to_end_balle_joint_codec_20260515",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=".omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json",
    archive_grammar="ns03_monolithic_balle_main_and_hyper_latents",
    parser_section_manifest={
        "header": "NS03 fixed header",
        "g_a": "brotli(fp16 analysis state)",
        "g_s": "brotli(fp16 synthesis state)",
        "h_a": "brotli(fp16 hyper-analysis state)",
        "h_s": "brotli(fp16 hyper-synthesis state)",
        "entropy": "brotli(fp16 entropy-model state)",
        "main_latents": "int16 main latents",
        "hyper_latents": "int16 hyper latents",
        "meta": "sorted json config/provenance",
    },
    inflate_runtime_loc_budget=220,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="custom",
    score_aware_loss="custom",
    bolt_on_loc_budget=1700,
    no_op_detector_planned=True,
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    cost_band_epochs=100,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=25.00,
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="ibps_kkt",
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
            "rate-proxy/packed-byte/inflate closure and paired exact eval."
        ),
        "hook_probe_disambiguator": (
            "The closure test is the required probe; no runnable artifact exists "
            "yet, so the contract stays research-only."
        ),
    },
)


@register_substrate(NSCS03_END_TO_END_BALLE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError(
        "NSCS03 is research-only until rate-proxy/packed-byte/inflate closure "
        "and paired CPU+CUDA exact-eval custody land."
    )


__all__ = ["NSCS03_END_TO_END_BALLE_CONTRACT", "main"]
