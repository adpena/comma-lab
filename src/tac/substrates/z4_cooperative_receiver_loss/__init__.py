# SPDX-License-Identifier: MIT
"""tac.substrates.z4_cooperative_receiver_loss — Time-Traveler L5 staircase Step 2.

Per the grand council MAXIMIZE-VALUE decision (Time-Traveler PEER seat) at
`feedback_grand_council_maximize_value_landed_20260514.md` (10/11 STAIRCASE
wins; Hotz dissents): Step 2 of the across-class staircase is the
cooperative-receiver loss — a loss-only intervention layered on top of Z3
Balle hyperprior + A1 substrate (Step 1) that trains the substrate against
the FULL scorer-conditional Lagrangian per CLAUDE.md "Score-domain
Lagrangian" non-negotiable.

Core insight (Atick & Redlich 1990 / Tishby-Zaslavsky cooperative receivers)
---------------------------------------------------------------------------

The classical compression objective is to minimize ``H(X)`` (source entropy).
The cooperative-receiver insight is that when the decoder D and the receiver
R cooperate, the bit budget shrinks from ``H(X)`` to ``H(X | f_R(X))`` where
``f_R`` is everything R can compute. For the contest receiver R = SegNet +
PoseNet, the entropy that matters is the scorer-conditional residual:

    R_relevant = H(X | W_seg, A_seg, P_seg, W_pose, A_pose, P_pose)

where W, A, P are scorer weights, activations, and predictions (in the
limit, the scorer-derived statistics the contest evaluates against).

**This is a LOSS-ONLY change, not an architecture change.** Z4 inherits
the Z3 substrate's encoder + decoder + per-pair latent grammar verbatim
(Z3HP1 archive bytes). The intervention is the training Lagrangian:

    L_Z3       = α · B(θ)/N + λ_pixel · MSE(decoded, GT) + R_hyperprior
    L_Z4       = α · B(θ)/N + β_seg · d_seg(θ)
                            + γ_pose · sqrt(10 · d_pose(θ))
                            + R_hyperprior

where ``d_seg`` and ``d_pose`` are the canonical scorer distortions through
``score_pair_components`` (Catalog #164). The hypothesis (Time-Traveler /
Atick-Redlich) is that training against the contest's actual scorer-derived
distortions (not pixel-MSE) reaches Step 2 band **[0.180, 0.188]**.

**Across-class differentiation from Step 1:**

- Z3 (Step 1) trains against pixel-MSE proxy with hyperprior rate term;
  saves bytes via entropy modeling but does not align the optimization
  axis with the contest objective.
- Z4 (Step 2) trains against the contest's actual scorer objective via the
  differentiable scorer roundtrip; aligns the optimization axis with
  what is measured.
- Z3 + Z4 STACK: the bytes are still hyperprior-coded (Z3), AND the
  decoder is now scorer-aware (Z4). Predicted total Δ from Z3 baseline:
  -0.005 to -0.010 ([mathematical-derivation; pose-marginal-dominated]).

**Byte invariance**: Z4CR1 archive grammar is BYTEWISE-IDENTICAL to Z3HP1.
The inflate.py is bytewise-identical to Z3's. The only difference is the
*meta* field carrying the cooperative-receiver provenance tag for forensic
distinction at audit time.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` extends Z3HP1 (Step 1
  Balle hyperprior); same bytes, different training objective
- ``parser_section_manifest``: Z3HP1 header + encoder_blob + decoder_blob +
  latent_blob (int8 per-pair z) + meta JSON (with cooperative-receiver tag)
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver
  (reuses Z3 inflate path — encoder/decoder + latent dequant + bilinear)
- ``runtime_dep_closure``: torch + brotli + constriction only (HNeRV L4 ≤ 2 deps
  bumped to 3 to inherit Z3's hyperprior closure)
- ``export_format``: Z4CR1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``CooperativeReceiverScoreAwareLoss`` routes through
  ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory;
  Atick-Redlich H(X|W+A+P) form
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  loss-only intervention with byte-identical archive is substrate-engineering
- ``no_op_detector_planned``: training-only change; archive bytes WILL
  differ from Z3 (because trained weights differ), AND the parser_section
  manifest is unchanged. Empirical no-op detector: encode under Z4 loss,
  verify decoded RGB differs from Z3's at the same bit count.

target_modes: ``contest_one_video_replay``, ``contest_generalized``,
``research_substrate``
lane_class: ``substrate_engineering``
research_only: false (loss-only on top of byte-closed Z3 substrate;
Phase 2 council approval required to lift _full_main NotImplementedError)
canary_status: ``post_canary_dependent``
canary_dependency: ``lane_z3_balle_hyperprior_bolton_campaign_20260514``

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS":

1. **Atick-Redlich hypothesis** — the cooperative-receiver loss converges
   to ``H(X | W+A+P)`` and beats pixel-MSE because pixel-MSE wastes bits
   on perceptually-relevant but scorer-irrelevant texture.
2. **Marginal-gradient-alignment hypothesis** — pixel-MSE and scorer-MSE
   are equivalent in expectation; the gain comes purely from training
   on the same loss the eval measures (gradient alignment).

The probe sweeps ``λ_pixel`` ∈ [0.0, 0.5, 1.0] (full scorer-only,
half-mix, full-pixel) at fixed ``β_seg + γ_pose``. Compares decoded RGB
distance to GT (the pixel-MSE-favored ablation) vs scorer-component
distance (the scorer-favored ablation). Memo at
``tools/probe_z4_cooperative_receiver_vs_marginal_disambiguator.py`` (planned
post Phase 2 dispatch approval; see lane evidence).

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — Cooperative-receiver gradient norm IS the
   per-tensor importance signal (∂L_scorer/∂θ for each θ); register
   ``sensitivity_map.cooperative_receiver_grad_v1`` (planned post-Stage-1).
2. **Pareto constraint** — adds
   ``cooperative_receiver_loss ≤ ε_scorer`` to the convex feasibility
   region intersected with the Z3 rate/distortion polytope; register
   ``tac.pareto.cooperative_receiver_v1``.
3. **Bit-allocator hook** — per-tensor importance derived from
   cooperative-receiver gradient norms; ``β_seg + γ_pose`` weights become
   the bit-allocator's Lagrangian; register
   ``bit_allocator.cooperative_receiver_v1``.
4. **Cathedral autopilot dispatch hook** — recipe registered; gated by
   Catalog #167 smoke-before-full; ranker v2 (Catalog #219) receives
   ``literature_anchor=Atick-Redlich1990`` (-0.01 to -0.02 class-shift
   reward).
5. **Continual-learning posterior** — every Z4 empirical anchor seeds the
   posterior with paired ``(L_pixel, L_scorer)`` measurement; expected
   substantial MDL-density change from Z3-MDL-density anchor confirms or
   refutes the cooperative-receiver hypothesis.
6. **Probe-disambiguator** — ``λ_pixel`` ablation IS the probe; see above.

Cross-references
----------------

- Time-Traveler peer-seat council:
  ``feedback_grand_council_maximize_value_landed_20260514.md`` (STAIRCASE
  10/11; Step 2 cooperative-receiver loss conditional absorb-into-D4-decoder)
- Zen-floor field-medal council:
  ``feedback_zen_floor_field_medal_grade_council_landed_20260514.md`` (Step 2
  predicted band [0.180, 0.188])
- Long-burn campaign roadmap:
  ``feedback_long_term_multi_year_campaigns_landed_20260514.md`` (C5
  cooperative-receiver Total $30-50)
- Z1 ablation (the across-class-shift evidence):
  ``feedback_z1_mdl_ablation_landed_20260514.md`` (within-class trap)
- Step 1 sister:
  ``lane_z3_balle_hyperprior_bolton_campaign_20260514`` (Balle hyperprior;
  byte basis for Z4)
- Step 3 sister: ``lane_z5_predictive_coding_world_model_step3_20260514``
- C5 mature cooperative-receiver lane:
  ``lane_wyner_ziv_cooperative_receiver_substrate_20260513`` (DISCUS-class;
  structurally orthogonal to Z4 which is loss-only on A1+Z3)
- Atick & Redlich (1990) "Towards a theory of early visual processing"
  Neural Computation 2(3):308-320; cooperative-receiver foundational paper
- Tishby & Zaslavsky (2017) "Opening the black box of deep neural networks
  via information" (Information Bottleneck cooperative formulation)

Lane: ``lane_z4_cooperative_receiver_loss_step2_20260514``
"""

from tac.substrates.z4_cooperative_receiver_loss.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    CooperativeReceiverConfig,
    CooperativeReceiverSubstrate,
)
from tac.substrates.z4_cooperative_receiver_loss.archive import (
    Z4CR1_HEADER_FMT,
    Z4CR1_HEADER_SIZE,
    Z4CR1_MAGIC,
    Z4CR1_SCHEMA_VERSION,
    Z4CR1_SECTION_ROLES,
    CooperativeReceiverArchive,
    pack_archive,
    parse_archive,
    parse_z4cr1_archive_bytes,
)
from tac.substrates.z4_cooperative_receiver_loss.score_aware_loss import (
    CooperativeReceiverLossWeights,
    CooperativeReceiverScoreAwareLoss,
)

__all__ = [
    "CooperativeReceiverArchive",
    "CooperativeReceiverConfig",
    "CooperativeReceiverLossWeights",
    "CooperativeReceiverScoreAwareLoss",
    "CooperativeReceiverSubstrate",
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "Z4CR1_HEADER_FMT",
    "Z4CR1_HEADER_SIZE",
    "Z4CR1_MAGIC",
    "Z4CR1_SCHEMA_VERSION",
    "Z4CR1_SECTION_ROLES",
    "pack_archive",
    "parse_archive",
    "parse_z4cr1_archive_bytes",
]
