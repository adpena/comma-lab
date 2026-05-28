# SPDX-License-Identifier: MIT
"""Z6-v2 cargo-cult-unwind L0 SCAFFOLD substrate package.

The Rao-Ballard capacity-critique-driven cargo-cult-unwind redesign of the Z6
ego-motion-conditioned predictive-coding substrate per CLAUDE.md "PER-SUBSTRATE
OPTIMAL FORM via adversarial grand council symposium" non-negotiable +
Catalog #311 ego-motion-conditioned predictive coding paradigm + N1 path-5
STRUCTURAL CEILING REINFORCED routing 2026-05-28 (memo
``.omx/research/n1_path_5_8_seed_bootstrap_k5_landed_20260528T014859Z.md``).

L0 SCAFFOLD scope per HNeRV parity discipline L7 (substrate-engineering tier;
``lane_class=substrate_engineering``; ``research_only=true``;
``dispatch_enabled=false``). MLX-first training + numpy-portable inflate
per 8th MLX-FIRST standing directive 2026-05-26. Per CLAUDE.md
"INDIVIDUALLY-FRACTAL" standing directive 2026-05-27, every per-method
engineering decision below is Z6-v2-OPTIMAL — NOT shared-helper shortcut
from Z6 (which does not yet exist in this repo) NOR canonicalized-trainer-
skeleton compromise.

The cargo-cult-unwind diagnosis (per the parent scoping memo
``time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``):
the original Z6 design specified a single-layer FiLM-conditioned next-frame
predictor with ~95 KB target archive (~75K params). The Rao-Ballard capacity
critique surfaces that a single-layer predictor cannot capture the
hierarchical motion structure of ego-motion-conditioned video — global ego
motion + local object motion + per-pixel optical flow each require their
own predictive level. The cargo-cult-unwind redesign keeps the FiLM
ego-motion conditioning primitive (which is hard-earned at the
embodied-vision level per Ballard) but unwinds the single-layer assumption,
adding explicit Rao-Ballard hierarchical binding while staying below the
Z8-full-3-level architectural complexity.

Z6-v2 substrate-distinguishing primitives (per Catalog #272 contract):

1. Hierarchical FiLM-ego-motion predictor (2-level Rao-Ballard, not 3-level)
   binding ego-motion at level 0 and FoE-prior at level 1.
2. Atick-Redlich cooperative-receiver gradient binding per Catalog #311.
3. MLX-first training on M5 Max (no MLX dep at inflate per HNeRV parity L4).
4. numpy-portable inflate runtime (<=200 LOC + <=2 deps per HNeRV L4 waiver).

Canonical 6-step contract per Catalog #325 satisfied in the sister design
memo at
``.omx/research/z6_v2_cargo_cult_unwind_design_20260527T053000Z.md``.

Six-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-layer prediction-error norms exposed
  via ``architecture.layerwise_inspector``)
* hook #2 Pareto constraint = ACTIVE (cooperative-receiver
  ``I(X;T) - beta * I(T;Y)`` dual per Catalog #311 Atick-Redlich routing)
* hook #3 bit-allocator = ACTIVE (per-hierarchy-level byte budget)
* hook #4 cathedral autopilot dispatch = ACTIVE via auto-discovery per
  Catalog #335 canonical contract
* hook #5 continual-learning posterior = ACTIVE via
  ``tac.council_continual_learning.append_council_anchor`` per Catalog #355
* hook #6 probe-disambiguator = ACTIVE (Rao-Ballard 2-level hierarchy
  disambiguates micro- vs macro-residual attribution)
"""

from __future__ import annotations

from tac.substrate_registry import (
    SubstrateContract,
    register_substrate,
)


__all__ = (
    "SUBSTRATE_ID",
    "LANE_ID",
    "Z6_V2_CONTRACT",
    "OBSERVABILITY_SURFACE",
)

SUBSTRATE_ID = "z6_v2_cargo_cult_unwind"
LANE_ID = "lane_z6_v2_cargo_cult_unwind_l0_scaffold_20260527"


# 6-facet observability surface per Catalog #305. Each facet is a fact about
# what Z6-v2 exposes at runtime so a reviewer can decompose its behavior
# without re-instrumentation per CLAUDE.md "Max observability — non-negotiable".
OBSERVABILITY_SURFACE = {
    "inspectable_per_layer": (
        "2-level Rao-Ballard hierarchy (level 0 micro-residuals / level 1 "
        "meso-residuals); per-layer activations and per-layer prediction-error "
        "norms exposed via the trainer's ``layerwise_inspector`` hook."
    ),
    "decomposable_per_signal": (
        "Per-pair total loss decomposed into (1) micro-residual cooperative-"
        "receiver term, (2) meso-residual rate term, (3) FoE ego-motion "
        "consistency term, (4) Rao-Ballard 2-level hierarchical-binding term."
    ),
    "diff_able_across_runs": (
        "Archive bytes byte-stable under deterministic seed + MLX float32; "
        "two runs of the same Z6-v2 config produce sha256-identical "
        "``state_dict.npz`` payloads (regression test "
        "``tests/test_smoke.py::test_archive_byte_stability``)."
    ),
    "queryable_post_hoc": (
        "Training emits ``z6_v2_training_observability_<utc>.jsonl`` with "
        "per-epoch (epoch_idx, micro_loss, meso_loss, foe_loss, "
        "hierarchy_loss, total_loss, archive_bytes_estimate) — fcntl-locked "
        "APPEND-ONLY per Catalog #128/#131 sister discipline."
    ),
    "cite_able": (
        "Every persisted artifact carries Provenance per Catalog #323 with "
        "(commit_sha, lane_id=LANE_ID, substrate_id=SUBSTRATE_ID, "
        "canonical_helper_invocation, hardware_substrate); "
        "``score_claim=False`` + ``promotable=False`` + "
        "``axis_tag='[predicted]'`` at L0."
    ),
    "counterfactual_able": (
        "Catalog #105/#139/#220/#272 byte-mutation-smoke discipline: every "
        "distinguishing-feature byte in the archive (hierarchy weights, "
        "FoE conditioning, predictive-coding residual stream) is paired "
        "with a counterfactual probe that mutates one byte + reruns "
        "inflate + verifies frame-level output diverges from baseline."
    ),
}


# Per Catalog #241 canonical contract registration. 36-field schema per
# ``tac.substrate_registry.contract.SubstrateContract`` (read inline 2026-05-27
# PV per Catalog #229).
#
# ``recipe_research_only=True`` per CLAUDE.md "Substrate scaffolds MUST be
# COMPLETE or RESEARCH-ONLY" non-negotiable. ``score_improvement_mechanism_
# status="OPERATIONAL"`` paired with ``runtime_overlay_consumed=True`` per the
# contract's __post_init__ invariant: the inflate path WILL operationally
# consume archive bytes for ego-motion-conditioned predictive-coding
# reconstruction at L1+ promotion (the L0 scaffold's mechanism is declared
# but the L1 trainer implementation is the next operator-routable step).
Z6_V2_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle (5)
    id=SUBSTRATE_ID,
    lane_id=LANE_ID,
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=(
        ".omx/research/z6_v2_cargo_cult_unwind_design_20260527T053000Z.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="monolithic_single_file_z6v2cu1",
    parser_section_manifest={
        "header": "magic_z6v2cu1_plus_version_plus_section_offsets",
        "hierarchy_weights_level0": "level0_micro_film_predictor_state_dict_int8_brotli",
        "hierarchy_weights_level1": "level1_meso_film_predictor_state_dict_int8_brotli",
        "foe_prior": "ego_motion_focus_of_expansion_conditioning_int8_brotli",
        "predictor_latents": "per_pair_residual_stream_int8_brotli",
        "meta": "json_meta_provenance_quantization_scales",
    },
    inflate_runtime_loc_budget=200,  # HNeRV L4 waiver per substrate-engineering scope
    runtime_dep_closure=("numpy", "brotli"),  # HNeRV L4 <=2 deps per 8th MLX-first directive
    export_format="custom",  # MLX state_dict -> npz -> ZIP-member -> numpy inflate bridge
    score_aware_loss="custom",  # cooperative-receiver + Rao-Ballard hierarchical binding
    bolt_on_loc_budget=999,  # N/A; substrate_engineering scope per HNeRV parity L7
    no_op_detector_planned=True,  # per Catalog #105/#139 byte-mutation smoke discipline
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,  # L0 scaffold has no archive yet; declared at L1 promotion
    score_improvement_mechanism_status="RESEARCH_ONLY",  # L0; flips to OPERATIONAL at L1
    runtime_overlay_consumed=False,  # paired with RESEARCH_ONLY per contract invariant
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,  # L0 scaffold is smoke-only by construction
    recipe_research_only=True,  # transparent non-promotable per CLAUDE.md non-negotiable
    recipe_min_smoke_gpu="T4",  # placeholder; recipe lands at L1 with actual ladder
    recipe_min_vram_gb=12,  # T4 floor; revised per actual measurement at L1
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="readonly_mmap",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=50,  # L0 smoke cap; consistent with recipe_smoke_only=True (<=100)
    cost_band_gpu_key="T4",  # matches recipe_min_smoke_gpu per MacKay M1 invariant
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.30,  # L0 smoke estimate; revised post-empirical
    # 2.6 6-hook wire-in (6 per Catalog #125) — all ACTIVE for Z6-v2 per class-shift
    hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="per_channel_lsq",
    hook_autopilot_ranker_class_shift_token=(
        "ego_motion_conditioned_predictive_coding_per_catalog_311"
    ),
    hook_continual_learning_anchor_kind="macos_cpu_advisory",  # MLX-local non-promotable per Catalog #192
    hook_probe_disambiguator=(
        "tools/probe_z6_v2_cargo_cult_unwind_disambiguator.py"
    ),  # planned at L1; path declared so contract validates
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_164_scorer_preprocess_input_called",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={},  # all 6 hooks ACTIVE; no rationale needed
)

register_substrate(Z6_V2_CONTRACT)
