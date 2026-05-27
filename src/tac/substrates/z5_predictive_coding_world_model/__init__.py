# SPDX-License-Identifier: MIT
"""tac.substrates.z5_predictive_coding_world_model — Time-Traveler L5 staircase Step 3.

Per the grand council MAXIMIZE-VALUE decision (Time-Traveler PEER seat) at
`feedback_grand_council_maximize_value_landed_20260514.md` (10/11 STAIRCASE
wins): Step 3 of the across-class staircase is the differentiable predictive-
coding world-model — a hierarchical recurrent network that learns to
PREDICT the next pair from the prior pair plus ego-motion, then encodes
only the RESIDUAL.

Core insight (Rao & Ballard 1999 predictive-coding hierarchy)
-------------------------------------------------------------

For stationary-ergodic driving video, the asymptotic entropy is dominated
by frame-to-frame surprise. If a 2-3-layer hierarchical predictor learns
to predict ``f_{t+1}`` from prior frames + ego-motion proxy (e.g. PoseNet
output projected to a lower dim), the residual ``f_{t+1} - predicted_f_{t+1}``
has substantially lower entropy than the marginal source.

This is the third move on the staircase, building on:
- Step 1 (Z3 Balle hyperprior): saves bytes via entropy modeling
- Step 2 (Z4 cooperative-receiver loss): aligns training with the contest
  objective via scorer distortions
- Step 3 (Z5 predictive-coding world-model): replaces independent per-pair
  encoding with predictor-residual encoding

**Hypothesis:** the differentiable predictive-coding world-model may reduce
residual bytes vs Z4 enough to test the Step 3 band **[0.155, 0.180]** per
Time-Traveler's asymptote estimate (zen-floor council). That magnitude is a
contest planning prior, not something Rao-Ballard or Friston establishes. NOT
a Markov-1 first-order predictor (insufficient ego-motion context); a 2-3 layer
hierarchical predictor with foveation-hint integration (handoff to C1 sister
subagent's foveation lane when that lands).

**Across-class differentiation from Step 2:**

- Z4 encodes (encoder_state + decoder_state + per-pair z_t) independently
  per pair t.
- Z5 encodes (encoder_state + decoder_state + predictor_state +
  predictor_init_state + per-pair RESIDUAL r_t) where the predictor
  computes ``z_t = predictor(z_{t-1}, r_{t-1}, ego_motion)`` recursively.
- Total bytes: Z4_arc + ~50KB predictor (FP4-quantized) - any empirically
  measured savings from smaller per-pair residual.

**Predicted ΔS:** -0.025 to -0.038 vs Z4 → planning score band
**[0.155, 0.180]** `[planning-prior; Time-Traveler-asymptote]`. The numeric
band cannot influence promotion or rank reward until paired exact anchors land.

**Stacking with Z3+Z4 is mandatory**: Z5 cannot be evaluated in isolation
because the inflate-time predictor needs the same byte-format hyperprior
context that Z3 establishes and the same scorer-aware training signal
that Z4 provides.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` extends Z4CR1
  (Step 2 cooperative-receiver), adds predictor_state section
- ``parser_section_manifest``: Z5PCWM1 header + encoder_blob + decoder_blob +
  predictor_blob + latent_init_blob + residuals_blob + meta JSON (with
  predictive_coding_world_model_meta tag)
- ``inflate_runtime_loc_budget``: ≤200 LOC substrate-engineering waiver
  (encoder + decoder + predictor forward + latent_init + residual decode)
- ``runtime_dep_closure``: torch + brotli only (HNeRV L4 ≤ 2 deps)
- ``export_format``: Z5PCWM1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``PredictiveCodingWorldModelScoreAwareLoss`` routes
  through ``score_pair_components`` per Catalog #164; eval-roundtrip
  mandatory; cooperative-receiver Lagrangian + predictive-coding
  residual term (lambda_residual_entropy)
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  predictor + auto-decoder + hierarchical recurrence is substrate engineering
- ``no_op_detector_planned``: predictor weight section MUST be consumed by
  the inflate runtime — empirical detector: mutate predictor bytes; verify
  decoded frames change.

target_modes: ``contest_one_video_replay``, ``contest_generalized``,
``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (Phase 2 council approval required to lift _full_main
NotImplementedError before score-bearing dispatch authority)
canary_status: ``post_canary_dependent``
canary_dependency: ``lane_z4_cooperative_receiver_loss_step2_20260514``

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two defensible interpretations of "what dominates ΔS":

1. **Predictor-class hypothesis** — the hierarchical predictor's ability
   to forecast next-pair latent dominates ΔS (matches Rao-Ballard 1999;
   substantiates the world-model claim).
2. **Capacity hypothesis** — the additional 50KB of predictor parameters
   merely adds capacity to overfit to the contest video; the
   "predictive-coding" framing is post-hoc rationalization.

The probe trains TWO variants: (a) full hierarchical predictor, (b)
identity predictor (predictor_t(z_{t-1}) := z_{t-1}, no learning) with
the SAME parameter budget repurposed into the encoder/decoder. Compares
auth-eval score; if (a) beats (b) by ΔS > 0.005, predictive-coding wins.
Memo at ``tools/probe_z5_predictive_coding_vs_no_prediction_disambiguator.py``
(planned post Phase 2 dispatch approval).

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** — Predictor gradient norm IS the per-tensor
   importance signal (∂L_full/∂θ_predictor); register
   ``sensitivity_map.predictive_coding_v1``.
2. **Pareto constraint** — adds
   ``predictor_residual_entropy ≤ ε_residual`` to the convex feasibility
   region intersected with Z3+Z4's polytope; register
   ``tac.pareto.predictive_coding_v1``.
3. **Bit-allocator hook** — per-pair-residual bit allocation derives
   directly from predictor forecast uncertainty; register
   ``bit_allocator.predictive_coding_residual_v1``.
4. **Cathedral autopilot dispatch hook** — recipe registered; gated by
   Catalog #167 smoke-before-full; ranker v2 receives
   ``literature_anchor=Rao-Ballard1999`` as source-basis metadata only. No
   class-shift reward is valid until a paired exact anchor exists.
5. **Continual-learning posterior** — every Z5 empirical anchor seeds the
   posterior with the FIRST predictive-coding-class anchor; expected
   substantial MDL-density shift confirms or refutes the predictor-class
   hypothesis.
6. **Probe-disambiguator** — identity-predictor ablation IS the probe;
   see above.

Cross-references
----------------

- Time-Traveler peer-seat council:
  ``feedback_grand_council_maximize_value_landed_20260514.md`` (STAIRCASE
  10/11; Step 3 predictive-coding world-model unanimously favored)
- Zen-floor field-medal council:
  ``feedback_zen_floor_field_medal_grade_council_landed_20260514.md`` (Step 3
  asymptote band; staircase end-state)
- Long-burn campaign roadmap:
  ``feedback_long_term_multi_year_campaigns_landed_20260514.md`` (C5 mature
  cooperative-receiver $30-50; C6 MDL-IBPS in flight; this is
  staircase Step 3 ahead of C6)
- Step 1 sister:
  ``lane_z3_balle_hyperprior_bolton_campaign_20260514``
- Step 2 sister:
  ``lane_z4_cooperative_receiver_loss_step2_20260514``
- C1 foveation sister: world-model integration handoff (foveation hint
  feeds the predictor's ego-motion proxy)
- Rao & Ballard (1999) "Predictive coding in the visual cortex: a
  functional interpretation of some extra-classical receptive-field
  effects" Nature Neuroscience 2(1):79-87; predictive-coding foundational
- Friston (2010) "The free-energy principle: a unified brain theory?"
  Nature Reviews Neuroscience 11:127-138; world-model free-energy
- Hinton (2022) "Forward-Forward Algorithm" (predictive-coding modern revival)

Lane: ``lane_z5_predictive_coding_world_model_step3_20260514``
"""

from tac.substrates.z5_predictive_coding_world_model.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    HierarchicalPredictor,
    PredictiveCodingConfig,
    PredictiveCodingSubstrate,
)
from tac.substrates.z5_predictive_coding_world_model.archive import (
    Z5PCWM1_HEADER_FMT,
    Z5PCWM1_HEADER_SIZE,
    Z5PCWM1_MAGIC,
    Z5PCWM1_SCHEMA_VERSION,
    Z5PCWM1_SECTION_ROLES,
    PredictiveCodingArchive,
    PredictiveCodingArchiveNumpy,
    pack_archive,
    parse_archive,
    parse_archive_numpy,
    parse_z5pcwm1_archive_bytes,
)
from tac.substrates.z5_predictive_coding_world_model.score_aware_loss import (
    PredictiveCodingLossWeights,
    PredictiveCodingScoreAwareLoss,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "Z5PCWM1_HEADER_FMT",
    "Z5PCWM1_HEADER_SIZE",
    "Z5PCWM1_MAGIC",
    "Z5PCWM1_SCHEMA_VERSION",
    "Z5PCWM1_SECTION_ROLES",
    "HierarchicalPredictor",
    "PredictiveCodingArchive",
    "PredictiveCodingArchiveNumpy",
    "PredictiveCodingConfig",
    "PredictiveCodingLossWeights",
    "PredictiveCodingScoreAwareLoss",
    "PredictiveCodingSubstrate",
    "pack_archive",
    "parse_archive",
    "parse_archive_numpy",
    "parse_z5pcwm1_archive_bytes",
]
