# SPDX-License-Identifier: MIT
"""C1 world-model + foveation substrate (long-term campaign L1 scaffold).

Per ``~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_long_term_multi_year_campaigns_landed_20260514.md``
campaign C1: ACROSS-CLASS substrate-shift over A1/PR101/HNeRV-family
class-saturated baselines (Z1 MDL density 97-99% per
``feedback_z1_mdl_ablation_landed_20260514.md``). Predicted ΔS -0.04 to -0.06
vs A1 0.1928 ``[mathematical-derivation; first-principles-bound]`` -- predicted
band ``[0.13, 0.16]`` (staircase Step 4-5 territory).

Two orthogonal innovations vs HNeRV-class
-----------------------------------------

A. **World-model (recurrent latent dynamics)** - Ha-Schmidhuber 2018
   "World Models" (arXiv:1803.10122) + Hafner DreamerV3 2023
   (arXiv:2301.04104). Instead of HNeRV's per-frame independent decode
   ``f_t = D(z_t)``, C1 learns a hierarchical world-model that PROPAGATES
   latent state across the temporal axis:

       z_t = WorldModel(z_{t-1}, action_t)        # recurrent dynamics
       f_t = D(z_t)                                # per-frame decoder
       z_0 = z_init                                # initial latent ~16 KB

   For stationary-ergodic driving video, latent dynamics is highly
   compressible because consecutive frames share ~95% of latent content
   (Time-Traveler predictive-coding intuition). Per-frame residual surprise
   asymptotes to <100 bytes/frame for predictable ego-motion.

B. **Foveation (camera-geometry-aware bit allocation)** - Atick & Redlich
   1990 "Towards a theory of early visual processing" Neural Computation 2.

   The contest scorer resizes 1164x874 camera input to 384x512; the inverse
   resize map is rank-deficient (Carmack's shower thought per
   ``feedback_zen_floor_field_medal_grade_council_landed_20260514.md``).
   Only NEAR-FOV pixels (center 96x128 region after foveation) carry
   full-detail bits; periphery is heavily quantized. Foveation matched to
   ego-motion trajectory (predicted from PoseNet) gives 2-10x bit savings
   on periphery for stationary-ergodic forward driving.

Composition contract
--------------------

C1 is a STANDALONE substrate that emits an RGB frame stack as the contest
inflate output. The world-model + foveated decoder + residual-surprise codec
produce the full ``(1200, 874, 1164, 3)`` raw uint8 output. C1 does NOT
require a base substrate (unlike D4 which is a sidecar); it is its own
end-to-end renderer.

Catalog #124 archive-grammar 8 fields (declared inline so the AST walker observes them)
---------------------------------------------------------------------------------------

- ``archive_grammar``: C1WMFV1 monolithic single-file ``0.bin``
  (substrate-engineering scope)
- ``parser_section_manifest``: C1WMFV1 header + (a) world_model_blob
  (~10 KB FP4 quantized GRU/LSTM weights) + (b) decoder_blob
  (~50 KB FP4 per-frame decoder) + (c) z_init_blob (~16 KB initial latent)
  + (d) foveation_meta (~1 KB foveation map params + ego-motion seed) +
  (e) residual_surprise_blob (~60 KB per-frame residual brotli-compressed)
  + (f) scorer_meta_blob (sorted-keys JSON)
- ``inflate_runtime_loc_budget``: <= 200 LOC substrate-engineering waiver
  (HNeRV parity discipline L4 with rationale: full world-model unroll +
  foveated decode + residual surprise add-back requires >100 LOC)
- ``runtime_dep_closure``: torch + brotli only (HNeRV parity L4 <= 2 deps)
- ``export_format``: C1WMFV1 monolithic single-zip-member ``0.bin``
- ``score_aware_loss``: ``WorldModelFoveationScoreAwareLoss`` routes through
  the canonical ``score_pair_components`` per Catalog #164; trains
  world-model + decoder + foveation map jointly via cooperative-receiver
  loss + predictive-coding residual + foveation L1 regularizer
- ``bolt_on_loc_budget``: ``lane_class=substrate_engineering`` (HNeRV parity
  L7) — the world-model + foveated decoder + residual codec composition
  is substrate engineering, target <= 1500 LOC across all 5 modules
- ``no_op_detector_planned``: emit/parse roundtrip preserves bytes
  byte-for-byte; every archive section is structurally consumed by inflate
  (world_model -> unroll; decoder -> decode; z_init -> seed; foveation_meta
  -> bit-allocator; residual_surprise -> add-back)

target_modes: ``contest_one_video_replay``, ``contest_generalized``,
  ``research_substrate``
lane_class: ``substrate_engineering``
research_only: false (export-first, all 8 fields declared)
canary_status: ``post_canary_dependent`` (canary dependency: substrate_a1
  -- C1 is across-class FROM A1 but needs A1's contest-CUDA anchor in the
  registry as a baseline for the autopilot's ΔS calculation)

Predicted score band (NOT a claim):
- contest-CUDA: 0.130-0.160 ``[mathematical-derivation; first-principles-bound]``
  (A1 baseline 0.193 minus predicted Δ -0.04 to -0.06)

Probe-disambiguator (Catalog #125 hook #6)
------------------------------------------

Two pairs of design tensions ship as alternative modes via callable interface
+ probe-disambiguator tools per the design-tension memo:

1. **world-model recurrence** -- GRU vs LSTM vs Transformer
   (probe: ``tools/probe_c1_world_model_vs_independent_frames_disambiguator.py``)
2. **foveation strategy** -- uniform vs ego-motion-gated radial vs learned
   per-pixel attention (probe:
   ``tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py``)

Cross-references
----------------

- Master campaign ledger: ``.omx/research/campaign_c1_world_model_foveation_20260514.md``
- Council deliberation: ``feedback_grand_council_maximize_value_landed_20260514.md``
  (Carmack's resize-rank shower thought + Time-Traveler predictive-receiver
  staircase framing + Mallat wavelets-diagonalize)
- Zen-floor band: ``feedback_zen_floor_field_medal_grade_council_landed_20260514.md``
  (across-class predictions; C1 in [0.13, 0.16] band)
- Z1 MDL ablation (anchor for "within-class trap"):
  ``feedback_z1_mdl_ablation_landed_20260514.md``
- Sister long-term campaign:
  ``feedback_long_term_multi_year_campaigns_landed_20260514.md`` (C1-C7 ledgers)
- Time-Traveler predictive-receiver (sister cross-class lane):
  ``src/tac/substrates/time_traveler_l5_autonomy/``
- D4 Wyner-Ziv sister (frame-0 across-class lane):
  ``src/tac/substrates/d4_wyner_ziv_frame_0/``
- Canonical scorer contract: ``tac.substrates.score_aware_common.score_pair_components``
- Canonical inflate runtime helpers: ``tac.substrates._shared.inflate_runtime``
- Canonical trainer skeleton: ``tac.substrates._shared.trainer_skeleton``

Lane: ``lane_c1_world_model_foveation_campaign_l1_scaffold_20260514``
"""

from tac.substrates.c1_world_model_foveation.architecture import (
    DECODER_LATENT_DIM,
    EVAL_HW,
    FOVEATION_BIT_ATTENUATION_DEFAULT,
    N_FRAMES,
    NUM_PAIRS,
    PER_FRAME_RESIDUAL_BYTES_TARGET,
    TOTAL_ARCHIVE_TARGET_BYTES_MAX,
    TOTAL_ARCHIVE_TARGET_BYTES_MIN,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelFoveationSubstrate,
    WorldModelModule,
    WorldModelRecurrenceMode,
    FoveationStrategy,
    Z5RoutedWorldModel,
)
from tac.substrates.c1_world_model_foveation.archive import (
    C1WMFV1_HEADER_FMT,
    C1WMFV1_HEADER_SIZE,
    C1WMFV1_MAGIC,
    C1WMFV1_SCHEMA_VERSION,
    C1WMFV1_SECTION_ROLES,
    WorldModelFoveationArchive,
    pack_archive,
    parse_archive,
    parse_c1wmfv1_archive_bytes,
)
from tac.substrates.c1_world_model_foveation.score_aware_loss import (
    WorldModelFoveationLossWeights,
    WorldModelFoveationScoreAwareLoss,
)

__all__ = [
    "C1WMFV1_HEADER_FMT",
    "C1WMFV1_HEADER_SIZE",
    "C1WMFV1_MAGIC",
    "C1WMFV1_SCHEMA_VERSION",
    "C1WMFV1_SECTION_ROLES",
    "DECODER_LATENT_DIM",
    "EVAL_HW",
    "FOVEATION_BIT_ATTENUATION_DEFAULT",
    "FoveationStrategy",
    "N_FRAMES",
    "NUM_PAIRS",
    "PER_FRAME_RESIDUAL_BYTES_TARGET",
    "TOTAL_ARCHIVE_TARGET_BYTES_MAX",
    "TOTAL_ARCHIVE_TARGET_BYTES_MIN",
    "WorldModelConfig",
    "WorldModelFoveationArchive",
    "WorldModelFoveationConfig",
    "WorldModelFoveationLossWeights",
    "WorldModelFoveationScoreAwareLoss",
    "WorldModelFoveationSubstrate",
    "WorldModelModule",
    "WorldModelRecurrenceMode",
    "Z5RoutedWorldModel",
    "pack_archive",
    "parse_archive",
    "parse_c1wmfv1_archive_bytes",
]
