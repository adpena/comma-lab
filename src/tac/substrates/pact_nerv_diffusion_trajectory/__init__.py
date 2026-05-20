# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_diffusion_trajectory — Pact-NeRV-DiffusionTrajectory (L0 SKETCH).

Group 2 variant #10 of PACT-NERV-ULTIMATE. Paper-worthy bleeding-edge.

Literature anchors:
- Rombach et al. 2022 Latent Diffusion Models (arXiv:2112.10752)
- Blattmann et al. 2023 Video Latent Diffusion (arXiv:2304.08818)

Canonical OSS: `CompVis/latent-diffusion`.

Distinguishing primitive: per-pair latent diffusion trajectory. Instead of
storing the per-pair latent z directly, ship a small Gaussian-noise SEED
+ a learned LIGHTWEIGHT 5-step diffusion-trajectory predictor (per-step
delta MLP). At inflate time the predictor refines noise -> latent in
5 steps per pair.

Per the PACT-NERV-ULTIMATE memo: this variant is CARGO-CULTED-MAY-BE-PROMISING
+ RISK-LOC-EXCESS, predicted regression on contest rate term (full latent
diffusion is heavy). At the L0 task-cap (~400 LOC) we ship a LIGHTWEIGHT
diffusion-trajectory predictor (5 timesteps, depth-2 MLP per step) rather
than full UNet latent diffusion.

Status: **L0 SKETCH** (research_only=true; PAPER target only at this LOC).

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PDT)
    parser_section_manifest:   parse_archive() -> 5 sections (header +
                               decoder_blob + diffusion_predictor_blob +
                               seed_blob + meta_blob)
    inflate_runtime_loc_budget: ≤ 200 LOC (uses higher budget within HNeRV L4)
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli weights; int16 seeds
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~400 LOC (substrate_engineering tag; task-cap)
    no_op_detector_planned:    Catalog #139 + byte-mutation smoke

Cargo-cult audit per Catalog #303:
- 5 diffusion timesteps = CARGO-CULTED at L0 (alternatives: 2 / 10 / 20;
  trade-off between rate cost and refinement quality; Stage 1 sweep).
- Per-step depth-2 MLP = CARGO-CULTED at L0 (alternative: shared MLP across
  timesteps; alternative: full UNet; Stage 1 design ablation).
- Gaussian noise seed (per-pair stored int16) = HARD-EARNED (canonical
  diffusion init per Rombach §3.1).
- Linear noise schedule = CARGO-CULTED at L0 (alternative: cosine schedule
  per Nichol-Dhariwal 2102.09672).
"""

from .architecture import (
    DiffusionTrajectoryPredictor,
    PactNervDiffusionTrajectoryConfig,
    PactNervDiffusionTrajectorySubstrate,
)
from .archive import (
    PDT_HEADER_SIZE,
    PDT_MAGIC,
    PDT_SCHEMA_VERSION,
    PactNervDiffusionTrajectoryArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervDiffusionTrajectoryScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "DiffusionTrajectoryPredictor",
    "PDT_HEADER_SIZE",
    "PDT_MAGIC",
    "PDT_SCHEMA_VERSION",
    "PactNervDiffusionTrajectoryArchive",
    "PactNervDiffusionTrajectoryConfig",
    "PactNervDiffusionTrajectoryScoreAwareLoss",
    "PactNervDiffusionTrajectorySubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
