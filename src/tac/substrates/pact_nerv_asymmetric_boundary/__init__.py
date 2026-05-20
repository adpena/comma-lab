# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_asymmetric_boundary - Pact-NeRV-AsymmetricBoundary (L0 SKETCH).

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE
(commit ``e3ad4243a``) variant #15 + task-spec G3 family (asymmetric
per-class boundary selector; sister of NSCS06 v7 44% improvement per-class
chroma anchors).

Literature anchor: per_frame_difficulty_atlas_v1 canonical equation +
asymmetric warp boundary detection. The NSCS06 v6→v7 cargo-cult-unwind
achieved 44% improvement (105.15 → 58.89 contest-CUDA) via per-class
chroma anchors; the same pattern extends to per-pair-per-class asymmetric
boundary detection at frame boundaries (when scene-class transitions occur).

Hypothesis (per PACT-NERV-ULTIMATE Variant #15 + NSCS06 v7 empirical
anchor): asymmetric per-class boundary detection at scene-transition
frames captures dispersion in the contest's per-class chroma response
that per-pair-uniform conditioning misses; ~+250 bytes / predicted ΔS
[-0.004, +0.001].

Architecture (L0 SCAFFOLD):

    Per-pair latent z + per-frame boundary signal b in R^5 (per-SegNet-class)
       |
       v
    HNeRV-class base decoder
       |
       v
    Per-frame asymmetric boundary FiLM (γ, β) conditioned on per-class
    boundary signal (~100 LOC + per-class boundary detector ~150 LOC)
       |
       v
    rgb_0 / rgb_1: 1x1 Conv heads

Status: **L0 SKETCH** (research_only=true).

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PAB1 magic)
    parser_section_manifest:   header + base_decoder + boundary_film_projs
                               + latents + boundary_signals + meta
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder + FiLM projs
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~250 LOC asymmetric boundary detector + FiLM
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303:
- Asymmetric per-class boundary detection = HARD-EARNED-NSCS06-V7-44pct-improvement
- 5-class boundary signal (matches SegNet 5-class) = HARD-EARNED-EMPIRICAL
- FiLM γ+β (vs IA3 γ-only) at boundary = CARGO-CULTED at L0 (alt: IA3 γ-only)
- Static boundary detector at L0 = CARGO-CULTED (alt: learned + L1)
- HNeRV base decoder = HARD-EARNED
"""

from .architecture import (
    AsymmetricBoundaryFilm,
    PactNervAsymmetricBoundaryConfig,
    PactNervAsymmetricBoundarySubstrate,
)
from .archive import (
    PactNervAsymmetricBoundaryArchive,
    PAB1_HEADER_SIZE,
    PAB1_MAGIC,
    PAB1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervAsymmetricBoundaryScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "AsymmetricBoundaryFilm",
    "PAB1_HEADER_SIZE",
    "PAB1_MAGIC",
    "PAB1_SCHEMA_VERSION",
    "PactNervAsymmetricBoundaryArchive",
    "PactNervAsymmetricBoundaryConfig",
    "PactNervAsymmetricBoundaryScoreAwareLoss",
    "PactNervAsymmetricBoundarySubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
