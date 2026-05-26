# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER: path_3_h_phase_3_l0_scaffold_per_cargo_cult_first_methodology_atick_redlich_single_theorem_anchor_with_ego_motion_foe_projection_conditioning_per_ballard_2007_catalog_311_replacing_per_class_softmax_d4_falsified
"""tac.substrates.atw_v2_cooperative_receiver_v2 — ATW V2 cooperative-receiver substrate (Path 3 candidate H).

Per Phase 1 audit (`path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md`)
+ Phase 2 design decision (`path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md`)
+ Phase 3 design memo (`path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md`).

**This is a NEW substrate package**, NOT a refactor of the existing
``tac.substrates.atw_codec_v2/`` package. Per Catalog #110/#113 APPEND-ONLY
HISTORICAL_PROVENANCE: the existing `atw_codec_v2/` is preserved byte-for-byte
as historical evidence of the v1 cdf_table_blob FALSIFIED routing
(codex byte-mutation smoke commit ``057130de4`` proved
``max_abs_raw_byte_delta=0`` across all 2,560 cdf_table_blob bytes mutated;
canonical equation #26 EXCLUDED context ``direct_byte_substitution_on_decode_
opaque_raw_sections`` registered structurally).

The Layer 1 META-unwind (Phase 2 binding decision)
==================================================

ATW V2 v1 was framed via triple-citation-stacking: Atick-Redlich + Tishby IB +
Wyner-Ziv = single substrate. The triple-binding presumed all three theorems'
preconditions are simultaneously satisfied by the contest's scorer-conditional
information geometry. Phase 1 audit Layer 1 META-CC: this is a structural
META-cargo-cult — each theorem has DIFFERENT mathematical preconditions; the
substrate was never proved to satisfy all three.

Phase 2 Layer 1 META-unwind binding:

- **Atick-Redlich 1990 = SINGLE substrate-optimal anchor.** Contest's
  SegNet+PoseNet IS a fixed receiver; mutual-information maximization against
  a fixed receiver IS the canonical Atick-Redlich formulation.
- **Tishby IB 2015 = DEMOTED to advisory cross-check.** Contest task is RGB
  reconstruction through SegNet+PoseNet (multi-task), NOT segnet-class
  classification (single-task); single-Y IB does not apply.
- **Wyner-Ziv 1976 = DROPPED.** WZ requires decoder-only side-info; ATW V2's
  scorer_class_prior_table is shared both-sides (in archive). Reframe as
  DETERMINISTIC SHARED CONDITIONING (conditional source coding R(D|Y)).

The Phase 2 NEW conditioning variable
=====================================

Per Phase 1 CC-7 unwind + Ballard 2007 + Catalog #311 ego-motion-conditioning:

- **Y_ego_motion = ego-motion FOE projection** (per-pair PoseNet pose-delta
  dominant-direction projection); REPLACES per-class softmax (D4 INDEPENDENT
  verdict ``I(latent; scorer_class) = 0.006385`` bits/symbol on A1 latents).
- Hypothesis: ego-motion IS the dominant continuous-time signal in dashcam
  video (vehicle is moving; everything is parallax-shifting); per-class
  softmax is empirically falsified for A1-class latents.

The distinguishing feature per Catalog #272
===========================================

**Ego-motion-conditioned cooperative-receiver-loss** with class-conditional
rate term ``R(D|Y_ego_motion)`` where ``Y_ego_motion`` is shared-conditioning
(NOT WZ side-info). The encoder + per-pair latent + decoder forms a continuous-
time information channel against the fixed SegNet+PoseNet receiver; the per-
pair latent's rate budget is allocated conditional on per-pair ego-motion FOE
projection (dominant translational direction of camera motion).

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

* ``archive_grammar``: ATWv2CR2 monolithic single-file ``0.bin`` with NEW magic
  ``b"ATWv2CR2"`` (cooperative-receiver V2); 8 sections (NO dead cdf_table_blob);
  bytewise-distinct from ATW1 + ATW2 + Z3HP1 + Z4CR1.
* ``parser_section_manifest``: ATWv2CR2 header + encoder_blob + decoder_blob +
  cond_embed_blob + ego_motion_proj_blob + per_pair_latent_blob +
  class_cond_cdf_blob + meta_blob (8 sections; ALL byte-mutation verifiable
  from byte-zero per Catalog #139/#272 — NO dead bytes ship per Catalog #220).
* ``inflate_runtime_loc_budget``: <=200 LOC substrate-engineering waiver per
  HNeRV parity L4 + L7 (decoder + ego-motion-conditional latent decode +
  class-cond CDF range-decode + composition).
* ``runtime_dep_closure``: numpy + struct + brotli + torch (HNeRV L4 <=3-4
  deps; torch only for canonical select_inflate_device).
* ``export_format``: ATWv2CR2 monolithic single-zip-member ``0.bin``.
* ``score_aware_loss``: ``ATWv2CR2ScoreAwareLoss`` routes through canonical
  ``tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss``
  per Catalog #164 + extended with ego-motion FOE projection hook per Ballard
  2007 + Catalog #311.
* ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  ~500-800 LOC substrate-engineering scaffold (NOT a bolt-on per operator
  directive #1).
* ``no_op_detector_planned``: ALL 8 archive sections byte-mutation verifiable
  from byte-zero per Catalog #139/#272 + Catalog #220 operational mechanism
  declaration (no dead bytes; cdf_table_blob lesson learned).

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 4 council approval required to lift _full_main
NotImplementedError per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY" + Catalog #240(c). Pending Phase 4 D4-equivalent probe
for ego-motion conditioning surface.)
canary_status: ``post_canary_dependent``
canary_dependency: ``lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526``

Observability surface (Catalog #305 NON-NEGOTIABLE)
----------------------------------------------------

1. **Per-layer inspection**: MLX renderer + PyTorch parity reference both
   expose per-layer activation hooks; per-pair latent + per-pair ego-motion
   + per-pair conditioning embedding all queryable.
2. **Per-signal decomposition**: loss decomposes into (rate term ``25·B/N``,
   seg term ``100·d_seg``, pose term ``sqrt(10·d_pose)``, cooperative-receiver
   MI term); per-step JSONL emission.
3. **Run-to-run diff**: archive byte-deterministic per Catalog #166 +
   reproducible per ``(seed, commit_sha, upstream_snapshot_sha256)`` tuple.
4. **Post-hoc query**: experiments/results/lane_path_3_h_atw_v2_cooperative_
   receiver_v2_*/ carries auth_eval_*.json + provenance.
5. **Cite-able**: every artifact carries canonical Provenance per Catalog #323.
6. **Counterfactual-able**: byte-mutation smoke per Catalog #139/#272 on
   EVERY archive section from byte-zero.

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

NEW D4-equivalent for ego-motion conditioning surface:
``tools/probe_atw_v2_cooperative_receiver_ego_motion_conditioning_disambiguator.py``
(Phase 4 deliverable; not yet present). Measures ``I(latent; Y_ego_motion)`` for
the new conditioning variable; CANONICAL DISPATCH BLOCKER per Catalog #313
sister D4 pattern.

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map**: NEW ``tac.sensitivity_map.atw_v2_cooperative_receiver_
   ego_motion_conditioning`` (STUB at L0; Phase 4 lands operational compute).
2. **Pareto constraint**: NEW ``tac.pareto.atw_v2_cooperative_receiver_ego_
   motion_conditional_rate`` (R(D|Y) constraint replacing V1's R_WZ ≥
   H(latent | scorer_class) constraint).
3. **Bit-allocator hook**: NEW ``bit_allocator.atw_v2_cooperative_receiver_
   ego_motion_v1`` (per-pair archive bytes by ego-motion conditioning entropy).
4. **Cathedral autopilot dispatch hook**: recipe at
   ``.omx/operator_authorize_recipes/substrate_atw_v2_cooperative_receiver_
   v2_modal_t4_dispatch.yaml`` (Phase 4 deliverable; not yet present);
   ``dispatch_enabled: false`` + ``research_only: true`` at landing per
   Catalog #240(c).
5. **Continual-learning posterior update**: landing memo emits council anchor
   via ``tac.council_continual_learning.append_council_anchor`` with
   ``deferred_substrate_id="atw_v2_cooperative_receiver_v2"``.
6. **Probe-disambiguator**: see above.

Cross-references
----------------

* ``.omx/research/path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md`` (Phase 1 audit)
* ``.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md`` (Phase 2 design decision)
* ``.omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md`` (Phase 3 design memo)
* ``.omx/research/atw_v2_cdf_table_blob_reconciliation_codex_byte_mutation_smoke_falsified_20260521.md`` (v1 falsification anchor)
* ``src/tac/codec/cooperative_receiver/atick_redlich.py`` (canonical primitive)
* ``src/tac/substrates/atw_codec_v2/`` (v1 sister package preserved per Catalog #110/#113)
* ``src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py`` (canonical MLX renderer pattern)
* ``tools/gate_mlx_candidate_contest_equivalence.py`` (Catalog #1265 gate)

Lane: ``lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526``
"""

from __future__ import annotations

# Substrate identity constants (consumed by AST walker for Catalog #241 + #124
# observability of the canonical contract per the substrate META layer).
LANE_ID = "lane_path_3_h_atw_v2_cooperative_receiver_cargo_cult_first_20260526"
SUBSTRATE_ID = "atw_v2_cooperative_receiver_v2"
SUBSTRATE_VERSION = "v2_phase_3_l0_scaffold_20260526"
HORIZON_CLASS = "frontier_pursuit"
DESIGN_MEMO_PATH = (
    ".omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_20260526.md"
)
PHASE_1_AUDIT_PATH = (
    ".omx/research/path_3_h_atw_v2_cooperative_receiver_cargo_cult_audit_of_existing_scaffold_20260526.md"
)
PHASE_2_DECISION_PATH = (
    ".omx/research/path_3_h_atw_v2_cooperative_receiver_substrate_design_decision_20260526.md"
)
RESEARCH_ONLY = True
DISPATCH_ENABLED = False
IMPLEMENTATION_STATUS = "l0_scaffold_mlx_renderer_archive_inflate_tests_research_only_full_main_not_implemented_per_catalog_240c"

# Phase 1 audit empirical evidence references (immutable provenance per
# Catalog #110/#113 HISTORICAL_PROVENANCE).
V1_FALSIFICATION_ANCHOR_COMMIT = "057130de4"
V1_FALSIFICATION_TEST_NAME = "test_cdf_table_xor_preserves_current_inflate_raw_output"
V1_FALSIFICATION_MAX_ABS_RAW_BYTE_DELTA = 0
V1_FALSIFICATION_MUTATED_BYTE_COUNT = 2560
V1_FALSIFICATION_RAW_EQUAL = True

# Layer 1 META-unwind binding architectural decisions (Phase 2 design-decision
# memo §2.1; consumed by AST walker for substrate-meta-layer observability).
LAYER_1_META_UNWIND_BINDING_ANCHOR = "Atick-Redlich_1990_cooperative_receiver_loss"
LAYER_1_META_UNWIND_DEMOTED_THEOREMS = (
    "Tishby_IB_2015_demoted_to_advisory_cross_check_multi_task_extension",
    "Wyner_Ziv_1976_dropped_preconditions_violated_reframed_as_conditional_source_coding_R_D_given_Y",
)
LAYER_1_META_UNWIND_ADVISORY_CROSS_CHECKS = (
    "Schmidhuber_2009_compression_as_intelligence",
    "Cover_Thomas_2006_section_5_5_conditional_source_coding",
)

# Phase 2 NEW conditioning variable (replaces per-class softmax that D4 INDEPENDENT verdict empirically falsified).
NEW_CONDITIONING_VARIABLE = "ego_motion_foe_projection"
CONDITIONING_VARIABLE_CITATION = "Ballard_2007_embodied_cognition_plus_Catalog_311_ego_motion_conditioning"
DEPRECATED_CONDITIONING_VARIABLE = "per_class_softmax_falsified_via_D4_INDEPENDENT_verdict_I_latent_scorer_class_equals_0_006385_bits_per_symbol"

# Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
ARCHIVE_GRAMMAR = (
    "ATWv2CR2_monolithic_single_file_0_bin_with_NEW_magic_byte_ATWv2CR2_cooperative_receiver_V2"
)
PARSER_SECTION_MANIFEST = (
    "header_encoder_blob_decoder_blob_cond_embed_blob_ego_motion_proj_blob_"
    "per_pair_latent_blob_class_cond_cdf_blob_meta_blob_8_sections_all_byte_mutation_verifiable_from_byte_zero"
)
INFLATE_RUNTIME_LOC_BUDGET = "less_than_or_equal_200_LOC_per_HNeRV_parity_L4"
RUNTIME_DEP_CLOSURE = "numpy_struct_brotli_torch_for_canonical_select_inflate_device_only"
EXPORT_FORMAT = "ATWv2CR2_monolithic_single_zip_member_0_bin"
SCORE_AWARE_LOSS = (
    "ATWv2CR2ScoreAwareLoss_routes_through_canonical_tac_codec_cooperative_receiver_"
    "atick_redlich_cooperative_receiver_loss_extended_with_ego_motion_FOE_projection_hook"
)
BOLT_ON_LOC_BUDGET = "lane_class_substrate_engineering_HNeRV_L7_approximately_500_to_800_LOC_NOT_a_bolt_on"  # FAKE_LANE_OK:test_fixture_or_docstring_or_dict_key_reference_to_lane_token_lane_class_substrate_engineering_HNeRV_L7_approximately_500_to_800_LOC_NOT_a_bolt_on_NOT_a_real_lane_registry_pre_registration_per_catalog_126_false_positive_per_comprehensive_bug_audit_cascade_20260526
NO_OP_DETECTOR_PLANNED = (
    "ALL_8_archive_sections_byte_mutation_verifiable_from_byte_zero_per_Catalog_139_272_"
    "plus_Catalog_220_operational_mechanism_declaration_no_dead_bytes_cdf_table_blob_lesson_learned"
)

# Per-substrate contract per Catalog #241 + #242
SUBSTRATE_CONTRACT_TARGET_MODES = ("research_substrate",)
SUBSTRATE_CONTRACT_LANE_CLASS = "substrate_engineering"
SUBSTRATE_CONTRACT_CANARY_STATUS = "post_canary_dependent"
SUBSTRATE_CONTRACT_CANARY_DEPENDENCY = LANE_ID

# Phase 3 L0 SCAFFOLD module structure
_SCAFFOLD_MODULES = (
    "mlx_renderer",
    "numpy_reference",
    "_torch_compat_reference",
    "_training_only",
    "archive",
    "inflate",
    "registered_substrate",
)


__all__ = [
    "ARCHIVE_GRAMMAR",
    "BOLT_ON_LOC_BUDGET",
    "CONDITIONING_VARIABLE_CITATION",
    "DEPRECATED_CONDITIONING_VARIABLE",
    "DESIGN_MEMO_PATH",
    "DISPATCH_ENABLED",
    "EXPORT_FORMAT",
    "HORIZON_CLASS",
    "IMPLEMENTATION_STATUS",
    "INFLATE_RUNTIME_LOC_BUDGET",
    "LANE_ID",
    "LAYER_1_META_UNWIND_ADVISORY_CROSS_CHECKS",
    "LAYER_1_META_UNWIND_BINDING_ANCHOR",
    "LAYER_1_META_UNWIND_DEMOTED_THEOREMS",
    "NEW_CONDITIONING_VARIABLE",
    "NO_OP_DETECTOR_PLANNED",
    "PARSER_SECTION_MANIFEST",
    "PHASE_1_AUDIT_PATH",
    "PHASE_2_DECISION_PATH",
    "RESEARCH_ONLY",
    "RUNTIME_DEP_CLOSURE",
    "SCORE_AWARE_LOSS",
    "SUBSTRATE_CONTRACT_CANARY_DEPENDENCY",
    "SUBSTRATE_CONTRACT_CANARY_STATUS",
    "SUBSTRATE_CONTRACT_LANE_CLASS",
    "SUBSTRATE_CONTRACT_TARGET_MODES",
    "SUBSTRATE_ID",
    "SUBSTRATE_VERSION",
    "V1_FALSIFICATION_ANCHOR_COMMIT",
    "V1_FALSIFICATION_MAX_ABS_RAW_BYTE_DELTA",
    "V1_FALSIFICATION_MUTATED_BYTE_COUNT",
    "V1_FALSIFICATION_RAW_EQUAL",
    "V1_FALSIFICATION_TEST_NAME",
    # WAVE-1 canonical posterior emission wire-in (2026-05-26)
    "ARCHITECTURE_CLASS",
    "CANONICAL_EQUATION_IDS",
    "emit_landing_posterior_anchor",
]


# ─── Canonical landing-time posterior emission (WAVE-1 wire-in 2026-05-26) ──
# Per OPTIMIZATION-TOOLING-AUDIT roadmap commit `e757bb74c` META #1 + the
# canonical helper at `tac.substrates._shared.posterior_emission_helper`:
# lifts this substrate's L0 SCAFFOLD signal into the cathedral autopilot's
# 62 auto-discovered consumers via the canonical posterior surfaces.

ARCHITECTURE_CLASS: str = (
    "atw_v2_cooperative_receiver_v2_ego_motion_foe_projection_l0_scaffold_mlx"
)

# Per WAVE-3 op-routable #3 the NEW canonical equation for this paradigm
# is queued: cooperative_receiver_atick_redlich_score_savings_v1 (H per
# the audit). Until registered in tac.canonical_equations, the manifest
# row's canonical_equation_ids carries the proposed-equation token so
# audit tooling can trace the lineage per Catalog #344.
CANONICAL_EQUATION_IDS: tuple[str, ...] = (
    "cooperative_receiver_atick_redlich_score_savings_v1_proposed_per_audit_e757bb74c_op_routable_3",
)


def emit_landing_posterior_anchor(
    *,
    archive_sha256: str | None = None,
    archive_bytes: int = 8_500,
    source_path: str | None = None,
    predicted_score: float = 0.193,
    predicted_d_seg: float | None = 0.00114,
    predicted_d_pose: float | None = 0.000028,
    notes: str = (
        "L0 SCAFFOLD MLX landing per WAVE-1 canonical posterior emission wire-in "
        "2026-05-26 (audit commit e757bb74c META #1 closure). ATW V2 cooperative-"
        "receiver substrate with ego-motion FOE projection conditioning per "
        "Ballard 2007 + Catalog #311; replaces per-class softmax falsified by D4 "
        "INDEPENDENT verdict empirical anchor commit 057130de4. Non-promotable "
        "per CLAUDE.md MLX research-signal discipline."
    ),
    posterior_path: object | None = None,
    posterior_lock_path: object | None = None,
    manifest_path: object | None = None,
):
    """Emit canonical landing-time posterior anchor for this substrate.

    Per WAVE-1-POSTERIOR-EMISSION-CANONICAL-WIRE-IN charter 2026-05-26 +
    OPTIMIZATION-TOOLING-AUDIT META #1 CRITICAL finding closure: invokes
    the canonical helper at
    ``tac.substrates._shared.posterior_emission_helper.emit_substrate_landing_posterior_anchor``
    with this substrate's canonical identifiers + canonical equation IDs
    threaded through ``extra_manifest_fields`` for cathedral consumer
    observability.

    Lifts this substrate's signal into:
    - ``.omx/state/continual_learning_posterior.json`` (refused as
      advisory-grade per custody validator; bumps ``refused_anchor_count``)
    - ``.omx/state/mps_research_signal_manifest.jsonl`` (canonical MLX
      research-signal posterior; cathedral-queryable surface)

    Per Catalog #287/#323/#341: anchor is non-promotable by construction.
    Per Catalog #128 + #131 + #138 sister discipline: writes through
    canonical fcntl-locked helpers only.
    """
    from tac.substrates._shared.posterior_emission_helper import (
        emit_substrate_landing_posterior_anchor,
        synthesize_substrate_archive_sha256,
    )

    sha = archive_sha256 or synthesize_substrate_archive_sha256(SUBSTRATE_ID)
    src = source_path or (
        "src/tac/substrates/atw_v2_cooperative_receiver_v2/"
        "__init__.py:emit_landing_posterior_anchor_l0_scaffold"
    )

    return emit_substrate_landing_posterior_anchor(
        substrate_id=SUBSTRATE_ID,
        archive_sha256=sha,
        archive_bytes=int(archive_bytes),
        source_path=src,
        predicted_score=predicted_score,
        predicted_d_seg=predicted_d_seg,
        predicted_d_pose=predicted_d_pose,
        architecture_class=ARCHITECTURE_CLASS,
        notes=notes,
        posterior_path=posterior_path,  # type: ignore[arg-type]
        posterior_lock_path=posterior_lock_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        extra_manifest_fields={
            "paradigm": "cooperative_receiver_ego_motion_foe_projection",
            "lane_class": SUBSTRATE_CONTRACT_LANE_CLASS,
            "horizon_class": HORIZON_CLASS,
            "canonical_equation_ids": list(CANONICAL_EQUATION_IDS),
            "research_only": RESEARCH_ONLY,
            "dispatch_enabled": DISPATCH_ENABLED,
            "implementation_status": IMPLEMENTATION_STATUS,
            "substrate_contract_present": True,  # ATW V2 has registered_substrate.py
            "canary_status": SUBSTRATE_CONTRACT_CANARY_STATUS,
            "binding_anchor": LAYER_1_META_UNWIND_BINDING_ANCHOR,
            "new_conditioning_variable": NEW_CONDITIONING_VARIABLE,
            "conditioning_variable_citation": CONDITIONING_VARIABLE_CITATION,
            "deprecated_conditioning_variable": DEPRECATED_CONDITIONING_VARIABLE,
            "v1_falsification_anchor_commit": V1_FALSIFICATION_ANCHOR_COMMIT,
            "v1_falsification_test_name": V1_FALSIFICATION_TEST_NAME,
            "lane_id": LANE_ID,
        },
    )
