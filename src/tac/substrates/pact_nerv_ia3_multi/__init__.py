# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_ia3_multi - Pact-NeRV-IA3-Multi (substrate L0 SKETCH).

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE
(commit ``e3ad4243a``) variant #14 + task-spec G3 family (multi-layer
IA3 γ-only modulation extension of sister Pact-NeRV-IA3 commit
``9cf9bdb16``; HARD-EARNED-EMPIRICALLY-SUPERIOR per FILM-FAMILY-RESEARCH
Section 8.6: multi-layer > single-layer for video-temporal conditioning
per HNeRV + TeNeRV ablation).

Literature anchor: Liu et al. 2022 IA3 [arXiv:2205.05638] + multi-layer
extension per HNeRV (Chen 2023) + TeNeRV ablation. The IA3 γ-only
modulation applied at EVERY upsample block (not just one) gains
expressiveness for video-temporal conditioning without the β term.

Hypothesis (per PACT-NERV-ULTIMATE Variant #14 + FILM-FAMILY-RESEARCH
top finding): multi-block IA3 with per-pair difficulty conditioning
captures temporal structure that single-block FiLM misses; ~+150 bytes
saved vs FiLM γ+β (predicted ΔS [-0.003, +0.001]).

Architecture (L0 SCAFFOLD; extends pact_nerv_ia3 sister):

    Per-pair latent z in R^24 + Per-pair difficulty d in R^1
       |
       v
    HNeRV-class base decoder (DepthSep + SIREN + PixelShuffle)
       |
       v
    For each upsample block:
        h_b = upsample_block(h_{b-1})
        γ_b = 1.0 + γ_proj_b(pose + difficulty)  # multi-input conditioning
        h_b = h_b * γ_b
       |
       v
    rgb_0 / rgb_1: 1x1 Conv heads

Sister of pact_nerv_ia3 (commit ``9cf9bdb16``): the distinguishing primitive
vs single-layer Pact-NeRV-IA3 is the per-pair difficulty conditioning
fused into the multi-block γ projection. Per FILM-FAMILY-RESEARCH §8.6
multi-layer is HARD-EARNED-EMPIRICALLY-SUPERIOR for video-temporal.

Status: **L0 SKETCH** (research_only=true).

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PIM1 magic)
    parser_section_manifest:   header + base_decoder + ia3_multi_gamma_projs
                               + latents + ego_poses + difficulty + meta
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder + multi-block γ_proj
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~200 LOC multi-block IA3 + difficulty MLP
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303:
- Multi-layer IA3 = HARD-EARNED-EMPIRICALLY-SUPERIOR (FILM-FAMILY §8.6)
- Per-pair difficulty conditioning = HARD-EARNED-PER-PAIR-DIFFICULTY-CANONICAL-EQUATION
- Sister-conditioning fusion (pose + difficulty) = CARGO-CULTED at L0 (L1: ablate)
- HNeRV base decoder = HARD-EARNED (PR101 GOLD baseline)
"""

from .architecture import (
    IA3MultiGammaOnlyModulation,
    PactNervIa3MultiConfig,
    PactNervIa3MultiSubstrate,
)
from .archive import (
    PactNervIa3MultiArchive,
    PIM1_HEADER_SIZE,
    PIM1_MAGIC,
    PIM1_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervIa3MultiScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "IA3MultiGammaOnlyModulation",
    "PIM1_HEADER_SIZE",
    "PIM1_MAGIC",
    "PIM1_SCHEMA_VERSION",
    "PactNervIa3MultiArchive",
    "PactNervIa3MultiConfig",
    "PactNervIa3MultiScoreAwareLoss",
    "PactNervIa3MultiSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
