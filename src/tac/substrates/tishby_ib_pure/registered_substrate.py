# SPDX-License-Identifier: MIT
"""Tishby IB-pure SubstrateContract registration (META layer Catalog #241/#242)."""

from __future__ import annotations

from tac.substrate_registry import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
    register_substrate,
)

TISHBY_IB_PURE_CONTRACT = SubstrateContract(
    id="tishby_ib_pure",
    lane_id="lane_tishby_ib_pure_l1_scaffold_substrate_build_plus_probes_20260516",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=(
        ".omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md"
    ),
    archive_grammar=(
        "tibp1_monolithic_variational_ib_encoder_optional_plus_variational_decoder_plus_"
        "wyner_ziv_side_info_plus_scorer_conditional_cdf"
    ),
    parser_section_manifest={
        "header": (
            "TIBP1 header (magic TIBP + version + section_count + 8 u32 lengths + sha256 digest)"
        ),
        "encoder_blob": "brotli(fp16 variational encoder state); OPTIONAL inflate-only build can be empty",
        "decoder_blob": "brotli(fp16 variational decoder state)",
        "statistic_net_blob": "brotli(fp16 MINE statistic network state); Path-MINE only; empty for Path-VIB",
        "latent_t_blob": "int8 per-pair latent table (num_pairs * latent_dim symbols)",
        "scorer_class_prior_blob": "uint8 per-pair SegNet class side-info table",
        "cdf_table_blob": "fp16 scorer-conditional CDF table (num_segnet_classes * cdf_num_symbols)",
        "meta_blob": "sorted-keys JSON utf-8 provenance metadata",
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
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    cost_band_epochs=200,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=15.00,
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token="tishby_ib_pure_variational_information_bottleneck",
    hook_continual_learning_anchor_kind=NOT_APPLICABLE_WITH_RATIONALE,
    hook_probe_disambiguator=(
        "tools/check_variational_ib_tractability.py"
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
        "catalog_270_dispatch_optimization_protocol_complete",
    ),
    hook_not_applicable_rationale={
        "hook_continual_learning_anchor_kind": (
            "No empirical paired-axis anchor exists at L1 scaffold landing; the "
            "posterior update fires from the trainer's _full_main only when "
            "the D4 probe verdict + Phase 2 council approval lift the "
            "research_only flag and a paired CPU+CUDA exact eval lands. The D4 "
            "probe at L1 returned INDEPENDENT (MI ≈ 0.006 bits/symbol per "
            ".omx/state/h_latent_given_scorer_class_tishby_ib_pure.json) which "
            "triggers DEFER-pending-research per design memo §19 + CLAUDE.md "
            "'Forbidden premature KILL'."
        ),
    },
)


@register_substrate(TISHBY_IB_PURE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Forwarding entrypoint for substrate registry consumers.

    The canonical executable entry is
    ``experiments/train_substrate_tishby_ib_pure.py``; this function exists
    only to satisfy the @register_substrate decorator contract.
    """
    raise NotImplementedError(
        "Tishby IB-pure trainer entry point lives at "
        "experiments/train_substrate_tishby_ib_pure.py. The registered_substrate "
        "module exists only for Catalog #241/#242 META layer registration; "
        "invoke the trainer module directly for smoke / full / dispatch paths."
    )


__all__ = ["TISHBY_IB_PURE_CONTRACT", "main"]
