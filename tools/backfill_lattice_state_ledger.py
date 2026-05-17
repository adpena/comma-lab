#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-shot backfill of canonical lattice-state ledger with all known substrates.

Per Path 2 LATTICE OF CLASS-SHIFTS canonical roadmap + operator binding
constraint 2026-05-16 *"Remember we need outside nerv-family too"*.

Run::

    .venv/bin/python tools/backfill_lattice_state_ledger.py --apply

Without --apply this prints the planned backfill without writing.

Each substrate is registered to its canonical lattice coordinate per:
- The 7 design memos landed today (2026-05-16) per Path 2 V5 + V6 verdicts
- The 31-substrate resurrection-audit corpus
- The 5-HIGH-RISK cargo-cult unwind audit verdicts
- The probe-outcomes ledger (Catalog #313) blocking verdicts
- The lane registry's current L1+ substrate lanes
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from tac.lattice_state_ledger import (
    CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
    CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
    CLASSIFICATION_PARADIGM_INTACT,
    CLASSIFICATION_TBD,
    HORIZON_ASYMPTOTIC_PURSUIT,
    HORIZON_FRONTIER_PURSUIT,
    HORIZON_NA,
    HORIZON_PLATEAU_ADJACENT,
    HORIZON_WON,
    RULE_1,
    RULE_2,
    RULE_3,
    RULE_4,
    RULE_5,
    STATUS_DEFERRED_AUDIT,
    STATUS_DEFERRED_OPERATOR,
    STATUS_DEFERRED_PROBE,
    STATUS_DISPATCHED_EVIDENCE,
    STATUS_LIFTED_DISPATCH_READY,
    STATUS_LIFTED_PENDING_COUNCIL,
    STATUS_NOT_YET_LIFTED,
    STATUS_SCAFFOLD_L0,
    register_lattice_node,
)


# Canonical backfill table: (lattice_node_id, kwargs).
# Sources cited inline.
BACKFILL: list[tuple[str, dict[str, Any]]] = [
    # ────────────────────────────────────────────────────────────
    # RULE 1 — Chroma-preserving + neural-optional substrate <60 [diagnostic-CPU]
    # ────────────────────────────────────────────────────────────
    # NSCS06 v1-v8 family — DEFERRED per operator decision 2026-05-16
    # (`.omx/research/nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516.md`)
    (
        "nscs06_v7",
        dict(
            substrate="nscs06_carmack_hotz_strip_everything",
            recipe_path=".omx/operator_authorize_recipes/substrate_nscs06_carmack_hotz_strip_everything_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py",
            lattice_rule=RULE_1,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="chroma_preserving_no_neural",
            status=STATUS_DEFERRED_OPERATOR,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
            evidence_score=58.89,
            evidence_score_axis="diagnostic-CPU; non-promotable",
            evidence_artifact_path=".omx/research/nscs06_strip_everything_family_DEFERRED_pending_breakthrough_20260516.md",
            notes="v7 cargo-cult-unwind: 105.15 -> 58.89 (44% reduction). DEFERRED per operator 2026-05-16 (307x worse than 0.192 frontier). Paradigm INTACT; implementation cargo-cult unwound; family dormant pending breakthrough.",
        ),
    ),
    (
        "nscs06_v8_path_b",
        dict(
            substrate="nscs06_v8_path_b_wavelet",
            recipe_path=".omx/operator_authorize_recipes/substrate_nscs06_v8_path_b_wavelet_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_nscs06_v8_path_b_wavelet.py",
            lattice_rule=RULE_1,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="chroma_preserving_no_neural",
            status=STATUS_DEFERRED_OPERATOR,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
            evidence_score=104.98,
            evidence_score_axis="diagnostic-CPU",
            evidence_artifact_path=".omx/research/nscs06_v8_path_b_wavelet_residual_full_stack_design_20260516.md",
            notes="v8 wavelet residual: 104.98 [diagnostic-CPU]; FALSIFIED in same operator decision as v7. SUPERSEDED-PENDING-REACTIVATION.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # RULE 2 — Nullspace-split renderer + PR95-paradigm <0.190 [contest-CPU]
    # ────────────────────────────────────────────────────────────
    (
        "nscs01",
        dict(
            substrate="nscs01_nullspace_split_renderer",
            recipe_path=".omx/operator_authorize_recipes/substrate_nscs01_nullspace_split_renderer_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_nscs01_nullspace_split_renderer.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="pr95_paradigm_nullspace_split",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_nscs01_nullspace_split_renderer_20260515",
            notes="_full_main LIFTED 2026-05-15 (~420 LOC). PR95-paradigm fully bound. Recipe research_only=true pending Phase 2 council. CANONICAL RULE 2 SUBSTRATE.",
        ),
    ),
    (
        "nscs02",
        dict(
            substrate="nscs02_downsampled_renderer",
            recipe_path=".omx/operator_authorize_recipes/substrate_nscs02_downsampled_renderer_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_nscs02_downsampled_renderer.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="pr95_paradigm_downsample_renderer",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
            evidence_artifact_path=".omx/research/nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md",
            notes="Cargo-cult unwound per HIGH-RISK 5 audit. PR95-paradigm sibling of NSCS01. Phase 2 pending.",
        ),
    ),
    (
        "nscs03",
        dict(
            substrate="nscs03_end_to_end_balle_joint_codec",
            recipe_path=".omx/operator_authorize_recipes/substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="balle_2018_end_to_end_joint_codec",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_nscs03_end_to_end_balle_joint_codec_20260515",
            notes="_full_main LIFTED 2026-05-15 (+548 LOC Ballé 2018 joint codec). 76 tests pass. Recipe research_only=true pending Phase 2 council λ_R sweep. OUTSIDE-NeRV.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # RULE 3 — Dykstra-feasibility-validated stack composition <0.180 [contest-CPU]
    # ────────────────────────────────────────────────────────────
    (
        "a_stack",
        dict(
            substrate="a_stack_nscs01_02_03_composition",
            recipe_path=".omx/operator_authorize_recipes/substrate_stack_of_stacks_modal_a100_dispatch.yaml",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="stack_composition",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            evidence_artifact_path=".omx/research/a_stack_nscs01_02_03_composition_full_stack_design_20260516.md",
            notes="3-substrate composition NSCS01 x NSCS02 x NSCS03 per T4 SYMPOSIUM V2/V6 + Dykstra-feasibility check. Predicted band [0.155, 0.175] [contest-CPU; Dykstra-validated convex-hull lower envelope]. OUTSIDE-NeRV.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # RULE 4 — Daubechies-wavelet multi-scale lattice (asymptotic floor)
    # ────────────────────────────────────────────────────────────
    (
        "z6",
        dict(
            substrate="time_traveler_l5_z6",
            recipe_path=".omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_time_traveler_l5_z6.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="predictive_coding_hierarchical",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            evidence_artifact_path=".omx/research/sextet_council_z6_phase_2_consensus_20260516.md",
            notes="L1 SCAFFOLD landed 2026-05-16. Rao-Ballard hierarchy + Hafner DreamerV3 + Wyner-Ziv side-info quadruple per Catalog #312. Sextet council PROCEED. OUTSIDE-NeRV. Predicted band [0.130, 0.160] [contest-CPU]. Phase 2 paid smoke pending council green-up.",
        ),
    ),
    (
        "rudin_floor",
        dict(
            substrate="rudin_floor_interpretable_ml",
            recipe_path=".omx/operator_authorize_recipes/substrate_rudin_floor_interpretable_ml_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_rudin_floor_interpretable_ml.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="interpretable_ml_compositional",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="L1 SCAFFOLD landed 2026-05-16; Phase 1c BLOCKED on Modal billing cap (see `.omx/research/wave_3_phase_1c_rudin_dispatch_blocked_modal_billing_cap_20260516.md`). OUTSIDE-NeRV. Predicted band [0.10, 0.13] [contest-CPU; Rudin mid-term].",
        ),
    ),
    (
        "tishby_ib_pure",
        dict(
            substrate="tishby_ib_pure",
            recipe_path=".omx/operator_authorize_recipes/substrate_tishby_ib_pure_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_tishby_ib_pure.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="info_bottleneck_pure",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="L1 SCAFFOLD with Wyner-Ziv + Atick-Redlich cooperative-receiver as PRIMARY. VIB-tractability SNR=6.75 empirical anchor. OUTSIDE-NeRV. Predicted band [0.08, 0.12] [contest-CPU; Tishby IB asymptotic].",
        ),
    ),
    # NeRV-family substrates (Wave 3 frontier-pursuit per K=13 schedule)
    (
        "sane_hnerv",
        dict(
            substrate="sane_hnerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_sane_hnerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_sane_hnerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="sane_hnerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_sane_hnerv_20260512",
            notes="NeRV-family canonical sub-class. Multiple Wave 1-2 dispatch attempts. Catalog #187 HNeRV parity guard.",
        ),
    ),
    (
        "hi_nerv",
        dict(
            substrate="hi_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_hi_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_hi_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_hi_nerv_20260512",
            notes="NeRV-family canonical. Wave 3 Phase 2 candidate per K=13 schedule.",
        ),
    ),
    (
        "ds_nerv",
        dict(
            substrate="ds_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_ds_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_ds_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_ds_nerv_20260512",
            notes="NeRV-family DSNeRV. Wave 3 Phase 2 candidate.",
        ),
    ),
    (
        "tc_nerv",
        dict(
            substrate="tc_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_tc_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_tc_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_tc_nerv_20260512",
            notes="NeRV-family TCNeRV. Wave 3 Phase 2 candidate.",
        ),
    ),
    (
        "block_nerv",
        dict(
            substrate="block_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_block_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_block_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_block_nerv_20260512",
            notes="NeRV-family BlockNeRV.",
        ),
    ),
    (
        "ff_nerv",
        dict(
            substrate="ff_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_ff_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_ff_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_DISPATCH_READY,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_ff_nerv_20260512",
            notes="NeRV-family FFNeRV.",
        ),
    ),
    (
        "e_nerv",
        dict(
            substrate="e_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_e_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_e_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_e_nerv_20260512",
            notes="NeRV-family ENeRV. research_only=true at recipe.",
        ),
    ),
    (
        "ego_nerv",
        dict(
            substrate="ego_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_ego_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_ego_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="ego_motion_focused_renderer",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_ego_nerv_20260512",
            notes="NeRV-family + ego-motion specialization. STILL counts as NeRV-family per operator binding constraint.",
        ),
    ),
    (
        "cnerv",
        dict(
            substrate="cnerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_cnerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_cnerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_cnerv_20260512",
            notes="NeRV-family CNeRV. research_only=true.",
        ),
    ),
    (
        "nervdc",
        dict(
            substrate="nervdc",
            recipe_path=".omx/operator_authorize_recipes/substrate_nervdc_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_nervdc.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_substrate_nervdc_20260512",
            notes="NeRV-family NeRV-DC.",
        ),
    ),
    (
        "lane_12_v2_nerv",
        dict(
            substrate="lane_12_v2_nerv",
            recipe_path=".omx/operator_authorize_recipes/substrate_lane_12_v2_nerv_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_lane_12_v2_nerv.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="nerv_family",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_12_v2_nerv_as_renderer",
            notes="NeRV-family Lane 12 v2 as renderer.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # Wunderkind G1 — wire-grammar class-shift (DEFERRED-per-probe per Catalog #313)
    # ────────────────────────────────────────────────────────────
    (
        "wunderkind_g1_v1",
        dict(
            substrate="wunderkind_g1_entropy_coded",
            lattice_rule=RULE_3,  # composes via wire grammar
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="wire_grammar_class_shift",
            status=STATUS_DEFERRED_PROBE,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
            evidence_artifact_path=".omx/research/wunderkind_g1_v2_wire_grammar_class_shift_full_stack_design_20260516.md",
            notes="DEFER per Catalog #313 probe outcome (wunderkind_g1_v2_per_pair_dominant_segnet_argmax_reducer_20260516) Q1 SPLIT-VERDICT. Alternative reducers (per-pair HISTOGRAM / per-region / per-segment-class / per-temporal-window) unprobed per Catalog #308.",
        ),
    ),
    (
        "z3_g1_v2",
        dict(
            substrate="z3_g1_entropy_coded_v2",
            recipe_path=".omx/operator_authorize_recipes/substrate_z3_g1_entropy_coded_v2_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="g1_entropy_coded_class_shift",
            status=STATUS_DEFERRED_AUDIT,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
            evidence_score=0.19869,
            evidence_score_axis="diagnostic-CPU; DIRECT_RESIDUAL_Z3HV2_REPRODUCTION",
            notes="Phantom-replication codex review bkrbqet3p F1 empirically confirmed: archive ships empty hyperprior/class slots; reproduces Z3 v2 baseline. research_only=true. Catalog #266 self-protect added.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # ATW codec v1/v2 — cooperative-receiver (DEFERRED-per-probe per Catalog #313)
    # ────────────────────────────────────────────────────────────
    (
        "atw_v1",
        dict(
            substrate="atw_codec_v1",
            recipe_path=".omx/operator_authorize_recipes/substrate_atw_codec_v1_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_atw_codec_v1.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="cooperative_receiver_codec",
            status=STATUS_DEFERRED_AUDIT,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
            evidence_artifact_path=".omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md",
            notes="Cargo-cult unwound per HIGH-RISK 5 audit. Wire grammar refinements pending.",
        ),
    ),
    (
        "atw_v2",
        dict(
            substrate="atw_codec_v2",
            recipe_path=".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_atw_codec_v2.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="cooperative_receiver_codec",
            status=STATUS_DEFERRED_PROBE,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_FALSIFIED,
            evidence_score=0.006385,
            evidence_score_axis="mutual_information_bits_per_symbol (D4 probe); INDEPENDENT verdict",
            evidence_artifact_path=".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
            notes="DEFER per Catalog #313 probe (atw_v2_d4_h_latent_given_scorer_class_20260516): I(latent;scorer_class)=0.006385 vs 0.5 MEANINGFUL_CONDITIONING threshold (2 orders magnitude below). Reactivation: alternative class signature OR trained ATW v2 residuals.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # Resurrection-audit Tier 1 reactivations (batched per design memo)
    # ────────────────────────────────────────────────────────────
    (
        "lane_17_imp",
        dict(
            substrate="lane_17_imp_iterative_magnitude_pruning",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="imp_iterative_magnitude_pruning",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_17_imp_10cycle",
            evidence_artifact_path=".omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md",
            notes="Resurrection-audit Tier 1 candidate. K=13 schedule plateau-bucket. Predicted band [0.193, 0.197].",
        ),
    ),
    (
        "stc_clean_source",
        dict(
            substrate="stc_clean_source",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="stc_steganography",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_stc_clean_source",
            evidence_artifact_path=".omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md",
            notes="Resurrection-audit Tier 1 candidate. Paired with apogee_int4 + IMP.",
        ),
    ),
    (
        "stc_v2",
        dict(
            substrate="stc_v2",
            recipe_path=".omx/operator_authorize_recipes/substrate_stc_v2_modal_t4_dispatch.yaml",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="stc_steganography",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="K=13 frontier-pursuit per `.omx/research/k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md` §4.2.3. Predicted band [0.180, 0.195].",
        ),
    ),
    (
        "apogee_int4",
        dict(
            substrate="apogee_int4",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="apogee_qat",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_apogee_int4",
            evidence_artifact_path=".omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md",
            notes="Resurrection-audit Tier 1. K=13 plateau-adjacent (boundary case if composed with A-STACK). Predicted band [0.188, 0.196] standalone.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # PR-family reformulated (K=13 frontier-pursuit)
    # ────────────────────────────────────────────────────────────
    (
        "pr101_reformulated",
        dict(
            substrate="pr101_lc_v2_clone_enhanced_curriculum",
            recipe_path=".omx/operator_authorize_recipes/substrate_pr101_lc_v2_clone_enhanced_curriculum_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_pr101_lc_v2_clone_enhanced_curriculum.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="frame_exploit_selector",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="K=13 frontier-pursuit. PR101 lc_v2 grammar with v8-class chroma-preservation overlay. Predicted band [0.180, 0.193].",
        ),
    ),
    (
        "pr106_r2_baseline",
        dict(
            substrate="pr106_latent_sidecar_r2_pr101_grammar_contest_cpu",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="latent_sidecar_compositional",
            status=STATUS_DISPATCHED_EVIDENCE,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_pr106_latent_sidecar_r2_pr101_grammar_contest_cpu",
            evidence_score=0.195,
            evidence_score_axis="contest-CPU; baseline re-anchor; medal-band",
            notes="K=13 plateau re-anchor; medal-band baseline for PR106 reformulated work.",
        ),
    ),
    (
        "a1_baseline",
        dict(
            substrate="a1_baseline_archive_87ec7ca5",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="pr95_paradigm_nullspace_split",
            status=STATUS_DISPATCHED_EVIDENCE,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            evidence_score=0.19285,
            evidence_score_axis="contest-CPU GHA Linux x86_64",
            notes="Current submission-grade frontier. CUDA gap +0.0335. Per A1 PR Council Round 1 + Catalog #205.",
        ),
    ),
    # ────────────────────────────────────────────────────────────
    # Other substrate trainers (architectural-class diversity)
    # ────────────────────────────────────────────────────────────
    (
        "d1_segnet_margin_polytope",
        dict(
            substrate="d1_segnet_margin_polytope",
            recipe_path=".omx/operator_authorize_recipes/substrate_d1_segnet_margin_polytope_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_d1_segnet_margin_polytope.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="polytope_margin_overlay",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            evidence_score=0.222,
            evidence_score_axis="contest-CUDA T4 (RECOVERY-1 L2 overlay landed)",
            notes="Phase 2 L2 OVERLAY OPERATIONAL per Catalog #220. Apogee anchor 2026-05-14.",
        ),
    ),
    (
        "d4_wyner_ziv_frame_0",
        dict(
            substrate="d4_wyner_ziv_frame_0",
            recipe_path=".omx/operator_authorize_recipes/substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_d4_wyner_ziv_frame_0.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="wyner_ziv_frame_zero",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Composes with NSCS01 nullspace-split. OUTSIDE-NeRV. Catalog #218 mini-batch reconstruct.",
        ),
    ),
    (
        "c6_e4_mdl_ibps",
        dict(
            substrate="c6_e4_mdl_ibps",
            recipe_path=".omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_c6_e4_mdl_ibps.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="mdl_information_bottleneck",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
            evidence_artifact_path=".omx/research/c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md",
            notes="HIGH-RISK 5 cargo-cult unwound. OUTSIDE-NeRV.",
        ),
    ),
    (
        "c1_world_model_foveation",
        dict(
            substrate="c1_world_model_foveation",
            recipe_path=".omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="world_model_foveation",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Z6/Z7/Z8 sister; Ha-Schmidhuber world model lineage. OUTSIDE-NeRV.",
        ),
    ),
    (
        "time_traveler_l5_autonomy",
        dict(
            substrate="time_traveler_l5_autonomy",
            recipe_path=".omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_time_traveler_l5_autonomy.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="predictive_coding_hierarchical",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_IMPLEMENTATION_CARGO_CULT,
            evidence_artifact_path=".omx/research/time_traveler_l5_cargo_cult_unwind_design_20260516.md",
            notes="HIGH-RISK 5 cargo-cult unwind landed. Phase 1b LIFTED. OUTSIDE-NeRV.",
        ),
    ),
    (
        "balle_renderer",
        dict(
            substrate="balle_renderer",
            recipe_path=".omx/operator_authorize_recipes/substrate_balle_renderer_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_balle_renderer.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="balle_renderer_hyperprior",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Sister of NSCS03; T1 Balle 23h timeout anchor. OUTSIDE-NeRV.",
        ),
    ),
    (
        "siren",
        dict(
            substrate="siren",
            recipe_path=".omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_siren.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="siren_implicit_neural",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="SIREN implicit neural representation. OUTSIDE-NeRV.",
        ),
    ),
    (
        "wavelet",
        dict(
            substrate="wavelet",
            recipe_path=".omx/operator_authorize_recipes/substrate_wavelet_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_wavelet.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="wavelet_codec",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Daubechies wavelet codec. OUTSIDE-NeRV.",
        ),
    ),
    (
        "vq_vae",
        dict(
            substrate="vq_vae",
            recipe_path=".omx/operator_authorize_recipes/substrate_vq_vae_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_vq_vae.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="vq_vae_codec",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="van den Oord VQ-VAE. OUTSIDE-NeRV.",
        ),
    ),
    (
        "cool_chic",
        dict(
            substrate="cool_chic",
            recipe_path=".omx/operator_authorize_recipes/substrate_cool_chic_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_cool_chic.py",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="cool_chic_neural",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Cool-Chic / C3 family. OUTSIDE-NeRV.",
        ),
    ),
    (
        "grayscale_lut",
        dict(
            substrate="grayscale_lut",
            recipe_path=".omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_grayscale_lut.py",
            lattice_rule=RULE_1,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="grayscale_lut_renderer",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Selfcomp PR#56 grayscale-LUT paradigm. OUTSIDE-NeRV.",
        ),
    ),
    (
        "dp1_pretrained_driving_prior",
        dict(
            substrate="pretrained_driving_prior",
            recipe_path=".omx/operator_authorize_recipes/substrate_pretrained_driving_prior_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_pretrained_driving_prior.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="pretrained_driving_prior",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            lane_id="lane_pretrained_driving_prior_phase_2_20260514",
            notes="DP1 Comma2k19 codebook. Catalog #209/#210/#211/#213 sister gates. OUTSIDE-NeRV.",
        ),
    ),
    (
        "self_compress_nn",
        dict(
            substrate="self_compress_nn",
            recipe_path=".omx/operator_authorize_recipes/substrate_self_compress_nn_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_self_compress_nn.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="self_compress",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Selfcomp PR#56 lineage. OUTSIDE-NeRV.",
        ),
    ),
    (
        "a1_plus_lapose",
        dict(
            substrate="a1_plus_lapose",
            recipe_path=".omx/operator_authorize_recipes/substrate_a1_plus_lapose_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_a1_plus_lapose.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="lapose_residual_bolton",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="A1 + LAPose residual bolt-on. OUTSIDE-NeRV.",
        ),
    ),
    (
        "a1_plus_wavelet_residual",
        dict(
            substrate="a1_plus_wavelet_residual",
            recipe_path=".omx/operator_authorize_recipes/substrate_a1_plus_wavelet_residual_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_a1_plus_wavelet_residual.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="wavelet_residual_bolton",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="A1 + wavelet residual bolt-on. OUTSIDE-NeRV.",
        ),
    ),
    (
        "sabor_boundary_only",
        dict(
            substrate="sabor_boundary_only_renderer",
            recipe_path=".omx/operator_authorize_recipes/substrate_sabor_boundary_only_renderer_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_sabor_boundary_only_renderer.py",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="boundary_only_renderer",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="SABOR boundary-only renderer (Yousfi-aware). OUTSIDE-NeRV.",
        ),
    ),
    (
        "s2sbs_byte_stuffing",
        dict(
            substrate="s2sbs_byte_stuffing",
            recipe_path=".omx/operator_authorize_recipes/substrate_s2sbs_byte_stuffing_modal_t4_dispatch.yaml",
            trainer_path="experiments/train_substrate_s2sbs_byte_stuffing.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="byte_stuffing_steganography",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="S2SBS byte stuffing. OUTSIDE-NeRV.",
        ),
    ),
    (
        "wyner_ziv_cooperative_receiver",
        dict(
            substrate="wyner_ziv_cooperative_receiver",
            recipe_path=".omx/operator_authorize_recipes/substrate_wyner_ziv_cooperative_receiver_modal_a100_dispatch.yaml",
            lattice_rule=RULE_4,
            horizon_class=HORIZON_ASYMPTOTIC_PURSUIT,
            architectural_class="cooperative_receiver_codec",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Wyner-Ziv cooperative-receiver primary substrate. OUTSIDE-NeRV.",
        ),
    ),
    (
        "hybrid_renderer_residual",
        dict(
            substrate="hybrid_renderer_residual",
            recipe_path=".omx/operator_authorize_recipes/substrate_hybrid_renderer_residual_modal_a100_dispatch.yaml",
            trainer_path="experiments/train_substrate_hybrid_renderer_residual.py",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="hybrid_renderer_residual",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Hybrid renderer residual composition. OUTSIDE-NeRV.",
        ),
    ),
    (
        "diffusion_renderer",
        dict(
            substrate="diffusion_renderer",
            recipe_path=".omx/operator_authorize_recipes/substrate_diffusion_renderer_modal_a100_dispatch.yaml",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_FRONTIER_PURSUIT,
            architectural_class="diffusion_renderer",
            status=STATUS_SCAFFOLD_L0,
            paradigm_vs_implementation_classification=CLASSIFICATION_TBD,
            notes="Diffusion renderer scaffold. OUTSIDE-NeRV.",
        ),
    ),
    (
        "z3_balle_hyperprior_bolton",
        dict(
            substrate="z3_balle_hyperprior_bolton",
            recipe_path=".omx/operator_authorize_recipes/substrate_z3_balle_hyperprior_bolton_modal_t4_dispatch.yaml",
            lattice_rule=RULE_3,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="z3_balle_hyperprior_bolton",
            status=STATUS_DISPATCHED_EVIDENCE,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            evidence_score=0.19778969,
            evidence_score_axis="contest-CPU",
            notes="Z3 v2 contest-CPU 0.19778969 (CUDA 0.23171). Bolt-on; current frontier-adjacent. OUTSIDE-NeRV.",
        ),
    ),
    (
        "quantizr_faithful",
        dict(
            substrate="quantizr_faithful",
            recipe_path=".omx/operator_authorize_recipes/substrate_quantizr_faithful_modal_a100_dispatch.yaml",
            lattice_rule=RULE_2,
            horizon_class=HORIZON_PLATEAU_ADJACENT,
            architectural_class="self_compress",
            status=STATUS_LIFTED_PENDING_COUNCIL,
            paradigm_vs_implementation_classification=CLASSIFICATION_PARADIGM_INTACT,
            notes="Quantizr 0.33 reproduction. OUTSIDE-NeRV.",
        ),
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually write to the ledger.")
    args = parser.parse_args(argv)

    print(f"Lattice-state backfill: {len(BACKFILL)} substrates")
    if not args.apply:
        print("[DRY-RUN] Use --apply to write to .omx/state/lattice_state.jsonl")
        for nid, kw in BACKFILL:
            rule = kw["lattice_rule"]
            arch = kw["architectural_class"]
            status = kw["status"]
            print(f"  {nid}: rule={rule} | arch={arch} | status={status}")
        return 0

    print("[APPLY] writing to .omx/state/lattice_state.jsonl")
    for nid, kw in BACKFILL:
        register_lattice_node(
            lattice_node_id=nid,
            agent="claude",
            subagent_id="coherence-audit-lattice",
            **kw,
        )
        print(f"  registered {nid}")
    print(f"Backfilled {len(BACKFILL)} substrates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
