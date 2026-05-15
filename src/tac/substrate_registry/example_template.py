"""Reference implementation: how a substrate declares its META contract.

NOT a real substrate. NOT registered as a lane. NOT dispatched. This file
exists so future substrate authors (and the migration subagent) have a
30-second-reviewable canonical example of ``@register_substrate`` usage.

Per CLAUDE.md "Beauty, simplicity, and developer experience — non-negotiable":
the example demonstrates all 36 contract fields in a single visually-coherent
block.
"""

from __future__ import annotations

from tac.substrate_registry import (
    SubstrateContract,
    register_substrate,
)

EXAMPLE_TEMPLATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="example_template",
    lane_id="lane_example_template_20260515",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=".omx/research/substrate_meta_layer_design_20260515.md",
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="example_grammar_for_documentation_only",
    parser_section_manifest={
        "header": "magic_header",
        "weights": "decoder_weight_stream",
    },
    inflate_runtime_loc_budget=80,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=200,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=10,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.10,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="not_applicable_with_rationale",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,  # within-class baseline
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_205_select_inflate_device_used",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": "documentation example; never dispatched",
        "hook_pareto_constraint": "documentation example; never dispatched",
        "hook_bit_allocator_class": "documentation example; never dispatched",
        "hook_continual_learning_anchor_kind": "documentation example; never dispatched",
        "hook_probe_disambiguator": "documentation example; never dispatched",
    },
)


@register_substrate(EXAMPLE_TEMPLATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Reference main(). A real substrate trainer would put its training loop here.

    NOTE: this function is intentionally a no-op so importing the example
    template registers the contract WITHOUT triggering any training side-effect.
    """
    return 0
