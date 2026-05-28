# SPDX-License-Identifier: MIT
"""tac.substrates.time_traveler_l5_z5 — Time-Traveler L5 staircase Z5 (Rao-Ballard hierarchical predictive coding).

Per the T4 SYMPOSIUM Wave N+13 verdict ``f5d3c6835`` op-routable #1 (Z5-first
among Z4/Z5/Z6/Z7/Z8 class-shift queue) + operator NON-NEGOTIABLE 2026-05-28
+ CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
+ Catalog #311 ego-motion-conditioned predictive coding canonical + Catalog
#312 hierarchical predictive coding canonical quadruple.

This is the time-traveler-staircase-aligned NEW Z5 substrate package — sister of
``tac.substrates.time_traveler_l5_z6`` (single-level Rao-Ballard FiLM) and
``tac.substrates.time_traveler_l5_z7_mamba2`` (state-space variant). The
distinct staircase canonical naming preserves the L5 staircase progression
(``z5 -> z6 -> z7 -> z8``) per the long-burn campaign roadmap.

Z5 substrate-distinguishing primitive (per Catalog #272)
----------------------------------------------------------

Rao-Ballard 1999 HIERARCHICAL predictive coding with EXPLICIT higher-level
latent prediction:

1. **Level-0 (low-level)**: per-pair latent ``z_low_t`` reconstructs the pair
   directly via the canonical Z6-style FiLM-PixelShuffle decoder.
2. **Level-1 (high-level)**: meta-latent ``z_high_t`` PREDICTS the level-0
   latent ``z_low_t`` from prior ``z_low_{t-1}`` + ego-motion (FoE prior).
3. **Hierarchical Bayesian inference** (Rao-Ballard's canonical bidirectional
   architecture): top-down prediction ``z_low_t_pred = predict(z_high_t, ego_t)``
   competes with bottom-up encoding; the RESIDUAL ``r_t = z_low_t - z_low_t_pred``
   is what gets stored per pair.

Distinct from sister Z6/Z7:

- Z6 (``time_traveler_l5_z6``): single-level FiLM-ego-motion conditioning;
  no explicit level-1 predictor.
- Z7-Mamba-2 (``time_traveler_l5_z7_mamba2``): Mamba-2 selective state-space
  recurrence; no explicit hierarchical level boundary.
- Z5 (THIS package): EXPLICIT 2-level Rao-Ballard hierarchy with separate
  ``z_low`` + ``z_high`` per-pair latents + predictor that maps
  ``(z_high_t, ego_motion_t) -> z_low_t_pred``.

The canonical Rao-Ballard distinguishing feature is the bidirectional
inference: the high-level predictor's output IS what the low-level encoder's
output is COMPARED AGAINST, and only the residual flows into the archive bytes.
Per Catalog #311 cooperative-receiver: the scorer-aware loss binds the
prediction error to the contest's per-pair score gradient.

Catalog #124 archive-grammar 8 fields (declared inline so AST walker observes them)
-----------------------------------------------------------------------------------

- ``archive_grammar``: monolithic single-file ``0.bin`` Z5RB1 header +
  decoder_blob + low_latents_blob + high_latents_blob + ego_vecs_blob +
  predictor_blob + meta_json
- ``parser_section_manifest``: Z5RB1 magic + schema_version + 6 section roles
  (decoder, low_latents, high_latents, ego_vecs, predictor, meta_json)
- ``inflate_runtime_loc_budget``: <=200 LOC substrate-engineering waiver per
  HNeRV parity L4 (encoder NOT in inflate; only decoder + predictor + latents)
- ``runtime_dep_closure``: torch + brotli + numpy (HNeRV L4 <=2 deps; numpy is
  stdlib-adjacent per substrates._shared.numpy_portable_inflate canonical)
- ``export_format``: Z5RB1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``Z5RaoBallardScoreAwareLoss`` routes through
  ``score_pair_components`` per Catalog #164; eval-roundtrip mandatory;
  Atick-Redlich cooperative-receiver gradient binding per Catalog #311
  + Catalog #312 hierarchical-predictive-coding canonical quadruple
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV L7);
  2-level Rao-Ballard hierarchical predictor + auto-decoder is substrate
  engineering per UNIQUE-AND-COMPLETE-PER-METHOD operating mode (#290)
- ``no_op_detector_planned``: predictor + high_latents sections MUST be
  consumed by inflate; empirical detector mutates bytes + verifies frames
  change per Catalog #139/#272

target_modes: ``research_substrate``
lane_class: ``substrate_engineering``
research_only: true (per Catalog #220+#240+#325; per-substrate symposium per
Catalog #325 required before paid dispatch authority)
canary_status: ``independent_substrate`` (sister of Z6/Z7; no shared canary)

6-hook wire-in (Catalog #125 NON-NEGOTIABLE)
-------------------------------------------

1. **Sensitivity-map** = ACTIVE — predictor gradient norm IS the per-tensor
   importance signal; ``sensitivity_map.z5_rao_ballard_v1`` registered.
2. **Pareto constraint** = ACTIVE — adds ``predictor_residual_entropy <=
   epsilon_residual`` to the convex feasibility region per Catalog #312
   hierarchical quadruple.
3. **Bit-allocator hook** = ACTIVE — per-pair residual bit allocation derives
   from predictor forecast uncertainty.
4. **Cathedral autopilot dispatch hook** = ACTIVE — auto-discovered via
   canonical contract (Catalog #335); ranker receives
   ``literature_anchor=Rao-Ballard1999`` as source-basis metadata only.
5. **Continual-learning posterior** = ACTIVE — every Z5 empirical anchor seeds
   the canonical equation
   ``z5_rao_ballard_hierarchical_predictive_coding_pose_axis_savings_v1`` per
   Catalog #344.
6. **Probe-disambiguator** = ACTIVE — identity-predictor ablation IS the probe;
   if full-predictor variant does NOT beat identity by ΔS > 0.005, the
   hierarchical-predictive-coding hypothesis is refuted per Catalog #308
   alternative-probe-methodology discipline.

Cross-references
----------------

- T4 SYMPOSIUM Wave N+13 verdict ``f5d3c6835`` op-routable #1 (Z5-first)
- Time-Traveler L5 staircase Step 3 design memo at
  ``.omx/research/time_traveler_l5_z6_z7_z8_predictive_coding_world_models_asymptotic_pursuit_scoping_design_20260516.md``
- Z6-v2 commit ``c26647891`` 600-pair MLX-LOCAL pattern (sister architecture)
- Z7-Mamba-2 commit ``2224eff58`` MLX-FIRST canonical pattern (sister architecture)
- Catalog #311 ego-motion-conditioned predictive coding canonical (Atick-Redlich
  1990 + Tishby IB + Rao-Ballard 1999 + DreamerV3 Hafner)
- Catalog #312 hierarchical predictive coding canonical quadruple
- Rao & Ballard (1999) "Predictive coding in the visual cortex" Nature
  Neuroscience 2(1):79-87 — the canonical bidirectional Bayesian inference
  hierarchical architecture
- Friston (2010) "The free-energy principle: a unified brain theory?" Nature
  Reviews Neuroscience 11:127-138 — world-model free-energy
- Hafner et al. DreamerV3 (2023) "Mastering Diverse Domains through World
  Models" — modern world-model latent-dynamics revival
- Atick & Redlich (1990) "Towards a Theory of Early Visual Processing" —
  cooperative-receiver paradigm; Catalog #311 sister

Lane: ``lane_time_traveler_l5_z5_rao_ballard_mlx_local_scaffold_20260528``
"""

from tac.substrates.time_traveler_l5_z5.architecture import (
    EVAL_HW,
    NUM_PAIRS,
    Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    Z5RaoBallardConfig,
    Z5RaoBallardSubstrate,
)
from tac.substrates.time_traveler_l5_z5.archive import (
    Z5RB1_HEADER_FMT,
    Z5RB1_HEADER_SIZE,
    Z5RB1_MAGIC,
    Z5RB1_SCHEMA_VERSION,
    Z5RB1_SECTION_ROLES,
    pack_archive,
    parse_archive,
)

__all__ = [
    "EVAL_HW",
    "NUM_PAIRS",
    "Z5RB1_HEADER_FMT",
    "Z5RB1_HEADER_SIZE",
    "Z5RB1_MAGIC",
    "Z5RB1_SCHEMA_VERSION",
    "Z5RB1_SECTION_ROLES",
    "Z5RaoBallardConfig",
    "Z5RaoBallardSubstrate",
    "Z5_TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "Z5_TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "pack_archive",
    "parse_archive",
]
