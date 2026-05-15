# SPDX-License-Identifier: MIT
"""NSCS01 SubstrateContract registration.

Per CLAUDE.md "META-LAYER-SUBSTRATE-CONTRACT-AUTO-WIRE" + Catalog #241/#242:
the canonical contract IS the single source-of-truth for the substrate.
Importing this module triggers ``@register_substrate`` validation.

Per the design memo + UNIQUE-AND-COMPLETE-PER-METHOD standing directive:
NSCS01 forks the score-aware loss layer from the canonical
``score_pair_components_dispatch`` because the SegNet-nullspace exploit
requires per-frame gradient routing the canonical helper does not expose.

This module is import-side-effect-free except for the contract registration
itself; callers should NEVER call its ``main()`` directly (the trainer at
``experiments/train_substrate_nscs01_nullspace_split_renderer.py`` is the
canonical entry point).
"""
from __future__ import annotations

from tac.substrate_registry import SubstrateContract, register_substrate

NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle (5)
    id="nscs01_nullspace_split_renderer",
    lane_id="lane_nscs01_nullspace_split_renderer_20260515",
    # research_substrate at L1 SCAFFOLD per Catalog #240 sister-protection:
    # paired CPU/CUDA Tier C custody must land before this flips to
    # contest_one_video_replay. The trainer's _full_main IS council-gated
    # via NotImplementedError per CLAUDE.md substrate-scaffolds-non-negotiable.
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=".omx/research/nscs01_nullspace_split_renderer_design_20260515.md",
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="nsp1_monolithic_single_file_per_head_quantized_split_renderer",
    parser_section_manifest={
        "header": "NSP1_HEADER_FMT (32B fixed)",
        "head0_blob": "brotli(frame_0_head_weights packed at HEAD0_BITS=4 default)",
        "head1_blob": "brotli(frame_1_head_weights packed at HEAD1_BITS=8 default)",
        "latent_blob": "brotli(per_pair_latents packed at LATENT_BITS=12 default)",
        "meta_blob": "sorted_keys_json (per-tensor shapes + scales + lo + manifests)",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli"),
    export_format="custom",  # nsp1 split-head quantized — canonical wire grammar
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=350,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    # L1 SCAFFOLD: smoke writes a real NSP1 archive; runtime renders both
    # frames; per Catalog #220 the OPERATIONAL flag requires runtime overlay
    # consumption — verified by byte-mutation smoke (HEAD0 mutation → frame_0
    # changes; HEAD1 mutation → frame_1 changes; LATENT mutation → both).
    archive_bytes_added=None,  # full renderer archive; not a sidecar overlay
    score_improvement_mechanism_status="OPERATIONAL",
    runtime_overlay_consumed=True,
    # 2.4 Recipe schema (8)
    # research_only mirrors the target_modes research_substrate classification
    # for the L1 SCAFFOLD landing per Catalog #240 sister-protection. Recipe
    # is smoke-only by construction (cost_band_epochs <= 100 invariant).
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,  # smoke band for the L1 dispatch
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=15.00,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="custom",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token="cooperative-receiver",
    hook_continual_learning_anchor_kind="paired_axis",
    hook_probe_disambiguator="tools/probe_nscs01_head0_arch_disambiguator.py",
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={},
)


@register_substrate(NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Registered substrate entry point.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
    HNeRV parity L2: this entry point is reserved for canonical contract
    registration; the actual trainer lives at
    ``experiments/train_substrate_nscs01_nullspace_split_renderer.py``.
    """
    raise NotImplementedError(
        "NSCS01 main() is reserved for SubstrateContract registration only. "
        "Use experiments/train_substrate_nscs01_nullspace_split_renderer.py "
        "for the trainer entry point. Smoke must pass byte-mutation gate "
        "(test_nscs01_substrate.py byte-mutation tests) before any Modal "
        "dispatch per Catalog #139 + #167 + the design memo."
    )


__all__ = ["NSCS01_NULLSPACE_SPLIT_RENDERER_CONTRACT", "main"]
