# SPDX-License-Identifier: MIT
"""time_traveler_l5_z4 — Atick-Redlich cooperative-receiver substrate (L1 MLX SCAFFOLD).

Wave N+13 Track A class-shift PRIMARY landing 2026-05-28 per T4 SYMPOSIUM
+ operator directive *"respawn and recover continue with all, ensure no signal
loss"* + the 8th MLX-FIRST standing directive 2026-05-28 + the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27.

Z4 distinguishing primitive vs sister substrates (per Catalog #272):

1. **Atick-Redlich 1990 spatial cooperative-receiver decorrelation filter** —
   the canonical retinal mutual-information whitening primitive applied at the
   latent surface. Per Atick & Redlich 1990 "Towards a Theory of Early Visual
   Processing" + "Convergent algorithm for sensory receptive field development":
   the optimal receptive field decorrelates input statistics to maximize
   ``I(stimulus; neural-response)`` under a fixed coding budget. Implemented as
   a single learned 1x1 spatial decorrelation operator on the per-pair latent
   gather before the decoder.
2. **Cooperative-receiver score-aware Lagrangian** — per Atick-Redlich +
   Tishby-Zaslavsky 2015 IB framework: ``L = alpha * B/N + beta_seg * d_seg
   + gamma_pose * sqrt(d_pose) + lambda_pixel * MSE_pixel`` where the receiver
   ``R = SegNet + PoseNet`` is the contest scorer. This is the canonical
   cooperative-receiver form per Catalog #311 amendment (spatial Atick-Redlich
   retinal MI form is admissible without temporal ego-motion conditioning;
   ego-motion-conditioned predictive coding is Z6's distinguishing primitive,
   NOT Z4's).
3. **MLX-first training on M5 Max** (no MLX dep at inflate per HNeRV parity
   L4; 8th MLX-FIRST standing directive 2026-05-28).
4. **numpy-portable inflate runtime** (<=200 LOC + <=2 deps per HNeRV L4
   waiver; 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27).
5. **Smaller parameter footprint** vs Z6-v2 (~50K vs ~300K params) — direct
   reflection of the Atick-Redlich 1990 canonical claim that decorrelation
   alone (without depth-3 hierarchical FiLM-ego-motion as in Z6-v2) is the
   minimum-sufficient cooperative-receiver primitive at the retinal layer.

# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:Atick-Redlich_1990_canonical_form_is_SPATIAL_retinal_mutual_information_NOT_temporal_ego_motion_predictive_coding_per_Catalog_311_amendment_spatial_form_admissible_via_waiver_when_substrate_is_spatial_cooperative_receiver_per_Atick_Redlich_1990_retinal_MI_canonical_NOT_Z6_pose_axis_temporal_predictive_coding

Canonical archive grammar (per Catalog #124 8 fields):

* ``archive_grammar = Z4ATR1`` (monolithic single-file 0.bin; 28-byte header
  + decoder blob + latent blob + decorrelator blob + meta blob)
* ``parser_section_manifest = decoder_state_dict + per_pair_latents +
  decorrelation_filter + meta``
* ``inflate_runtime_loc_budget = 200``
* ``runtime_dep_closure = torch + numpy + brotli + PIL``
* ``export_format = Z4ATR1_v1_monolithic_bin``
* ``score_aware_loss = CooperativeReceiverScoreAwareLoss + Catalog #164
  canonical scorer-bound gradient routing``
* ``bolt_on_loc_budget = 600``
* ``no_op_detector_planned = byte-mutation_smoke_via_tools_verify_distinguishing_feature_byte_mutation``

Six-hook wire-in declaration per Catalog #125:

* hook #1 sensitivity-map = ACTIVE (per-pair latent decorrelation residuals
  exposed via decorrelation_filter post-forward inspection)
* hook #2 Pareto constraint = ACTIVE (cooperative-receiver Lagrangian
  ``alpha * B/N + beta_seg * d_seg + gamma_pose * sqrt(d_pose)`` IS the
  Pareto-polytope-feasibility intersection of rate / seg / pose constraints
  per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable)
* hook #3 bit-allocator = ACTIVE (decorrelation filter coefficients
  prioritized for high-bit allocation per Atick-Redlich
  whitening-importance principle)
* hook #4 cathedral autopilot dispatch = ACTIVE via auto-discovery per
  Catalog #335 canonical contract through canonical equation
  ``z4_atick_redlich_cooperative_receiver_savings_v1`` registered per
  Catalog #344
* hook #5 continual-learning posterior = ACTIVE via
  ``tac.canonical_equations`` + Catalog #355 meta-Lagrangian wire-in
* hook #6 probe-disambiguator = ACTIVE (decorrelation residual norm vs
  reconstruction loss IS the canonical disambiguator between
  cooperative-receiver-class shift vs pixel-MSE within-class refinement)

Lane: ``lane_z4_atick_redlich_cooperative_receiver_substrate_scaffold_first_600pair_mlx_local_anchor_20260528``
Status: L1 SCAFFOLD ``research_only=true`` + ``dispatch_enabled=false`` per
CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" until
sextet pact symposium per Catalog #325 6-step contract lands before any
paid dispatch.

[verified-against: src/tac/substrates/z4_cooperative_receiver_loss/score_aware_loss.py Atick-Redlich Lagrangian canonical]
[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/ canonical MLX-LOCAL scaffold pattern]
[verified-against: src/tac/substrates/_shared/mlx_score_aware/bundle.py canonical harness contract]
[verified-against: CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable]
[verified-against: Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" + "Convergent algorithm for sensory receptive field development"]
[verified-against: Catalog #311 ego-motion-conditioned predictive coding canonical + spatial-Atick-Redlich-form waiver]
"""

from __future__ import annotations

from tac.substrates.time_traveler_l5_z4.architecture import (
    Z4AtickRedlichConfig,
    Z4AtickRedlichSubstrate,
)
from tac.substrates.time_traveler_l5_z4.archive import (
    BROTLI_QUALITY_V1,
    Z4ATR_HEADER_FMT,
    Z4ATR_HEADER_SIZE,
    Z4ATR_MAGIC,
    Z4ATR_SCHEMA_VERSION,
    Z4ATRArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.time_traveler_l5_z4.archive_candidate import (
    DECODER_EXCLUDED_KEYS,
    build_archive_bytes,
    build_meta,
    extract_decoder_state_dict,
)
from tac.substrates.time_traveler_l5_z4.inflate import (
    CONTEST_NUM_FRAMES,
    CONTEST_OUT_H,
    CONTEST_OUT_W,
    CONTEST_RAW_BYTES,
    inflate_one_video,
    main_cli,
    select_inflate_device,
)
from tac.substrates.time_traveler_l5_z4.score_aware_loss import (
    Z4AtickRedlichScoreAwareLoss,
    Z4AtickRedlichScoreAwareLossWeights,
)

__all__ = [
    "BROTLI_QUALITY_V1",
    "CONTEST_NUM_FRAMES",
    "CONTEST_OUT_H",
    "CONTEST_OUT_W",
    "CONTEST_RAW_BYTES",
    "DECODER_EXCLUDED_KEYS",
    "Z4ATR_HEADER_FMT",
    "Z4ATR_HEADER_SIZE",
    "Z4ATR_MAGIC",
    "Z4ATR_SCHEMA_VERSION",
    "Z4ATRArchive",
    "Z4AtickRedlichConfig",
    "Z4AtickRedlichScoreAwareLoss",
    "Z4AtickRedlichScoreAwareLossWeights",
    "Z4AtickRedlichSubstrate",
    "build_archive_bytes",
    "build_meta",
    "extract_decoder_state_dict",
    "inflate_one_video",
    "main_cli",
    "pack_archive",
    "parse_archive",
    "select_inflate_device",
]
