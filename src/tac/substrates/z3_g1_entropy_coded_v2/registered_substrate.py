# SPDX-License-Identifier: MIT
"""Z3-G1 entropy-coded v2 SubstrateContract registration.

Per CLAUDE.md "META-LAYER-SUBSTRATE-CONTRACT-AUTO-WIRE" + Catalog #241/#242:
the canonical contract IS the single source-of-truth for the substrate.
Importing this module triggers ``@register_substrate`` validation.

Fields are tuned for the v2 entropy-coded grammar per
`.omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md` § 4.

This module is import-side-effect-free except for the contract registration
itself; callers should NEVER call its ``main()`` directly (the trainer at
``experiments/train_substrate_z3_g1_entropy_coded_v2.py`` is the canonical
entry point).
"""
from __future__ import annotations

from tac.substrate_registry import SubstrateContract, register_substrate

Z3_G1_ENTROPY_CODED_V2_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle (5)
    id="z3_g1_entropy_coded_v2",
    lane_id="lane_z3_g1_entropy_coded_v2_20260515",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=".omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md",
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="z3g2_entropy_coded_sigma_table_plus_class_index_replaces_a1_latent_blob",
    parser_section_manifest={
        "header": "Z3G2_HEADER_STRUCT (~27B fixed)",
        "sigma_table_blob": "brotli(sigma_table_int8) (~300B variable)",
        "class_prior_cdf_blob": "5*uint16 = 10B fixed (class frequency counts)",
        "class_index_blob": "constriction-Huffman(class_indices, prior_cdf) (~200B variable)",
        "residual_blob": "brotli(residual_int8) (~1200B variable)",
        "per_dim_affine": "2 * 28 * float32 = 224B fixed (offset + scale)",
    },
    inflate_runtime_loc_budget=100,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "constriction>=0.4,<0.5"),
    export_format="custom",  # z3g2_entropy_coded_packet — canonical Z3G2 wire grammar
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=350,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=14,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="substrate_a1",
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=7.50,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token="cooperative-receiver",
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={
        "hook_probe_disambiguator": (
            "single defensible interpretation for the wire grammar at parser/"
            "intermediate scope. The lane is research-only because full-frame "
            "inflate.sh mutation proof, the full trainer/export/auth-eval path, "
            "and paired CPU+CUDA custody are still missing."
        ),
        "hook_continual_learning_anchor_kind": (
            "no empirical score anchor exists yet; posterior update waits for "
            "paired CPU+CUDA exact eval after the full train/export/auth-eval "
            "path lands."
        ),
    },
)


@register_substrate(Z3_G1_ENTROPY_CODED_V2_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    """Registered substrate entry point.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
    HNeRV parity L2: this entry point is reserved for the canonical
    contract registration; the actual trainer lives at
    ``experiments/train_substrate_z3_g1_entropy_coded_v2.py``.
    """
    raise NotImplementedError(
        "Z3-G1 entropy-coded v2 main() is reserved for SubstrateContract "
        "registration only. Use experiments/train_substrate_z3_g1_entropy_coded_v2.py "
        "for the trainer entry point. Smoke must pass byte-mutation gate "
        "(tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py) before any "
        "Modal dispatch per Catalog #139 + #167 + the design memo §6 + §7."
    )


__all__ = ["Z3_G1_ENTROPY_CODED_V2_CONTRACT", "main"]
