# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_cross_codec_a - Pact-NeRV-CROSS-CODEC-A (substrate L0 SKETCH).

Group 4 CROSS-CODEC composition variant per PACT-NERV-ULTIMATE (commit
``e3ad4243a``) variant #16 — fec6 base codec + Pact-NeRV-A1 bolt-on side
info (CROSS-CANDIDATE finding #3 empirical SUPER_ADDITIVE signature).

Literature anchor: Atick-Redlich 1990 cooperative-receiver framing
applied to cross-codec orthogonal stacking + CROSS-CANDIDATE finding #3
empirical anchor (PR101/A1/fec6 ↔ PR106 format0d per-axis Pearson
[-0.094, -0.078] = canonical SUPER_ADDITIVE signature per Catalog #322).

Hypothesis (per PACT-NERV-ULTIMATE Variant #16 + cross_codec_orthogonality_
predictor_consumer commit ``80484241f``): fec6 frontier (Huffman k=16
selector) carries score-relevant overhead on the FEC6-selector axis;
Pact-NeRV side-info carries score-relevant overhead on the per-pair latent
axis; their composition is SUPER_ADDITIVE because the codecs operate on
DIFFERENT receptive fields of the contest scorer (top-K Jaccard < 0.05).

Architecture (L0 SCAFFOLD):

    fec6 base codec (Huffman k=16 selector + frame-exploit selector)
       |  (bytes A_fec6 in archive section base_codec_blob)
       v
    Pact-NeRV side-info bolt-on (per-pair latent + HNeRV-class decoder)
       |  (bytes B_pact_nerv in archive section pact_nerv_side_info_blob)
       v
    Composition: rgb_0 = fec6_render(pair) + pact_nerv_residual(pair)
                 rgb_1 = fec6_render(pair) + pact_nerv_residual(pair)

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (CC_A magic)
    parser_section_manifest:   header + fec6_base_blob + pact_nerv_decoder_blob
                               + pact_nerv_latent_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder; FEC6 Huffman bytes
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~150 LOC cross-codec composition primitive
    no_op_detector_planned:    Catalog #139 byte-mutation smoke
                               (both base_codec_blob AND pact_nerv_decoder_blob)

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Cross-codec orthogonal composition for SUPER_ADDITIVE pairs = HARD-EARNED-EMPIRICALLY
  (CROSS-CANDIDATE finding #3 empirical anchor)
- Static residual additive composition at L0 = CARGO-CULTED (L1: learned
  composition gate per Atick-Redlich cooperative-receiver)
- fec6 base codec inherited as-is = HARD-EARNED-EMPIRICAL (CROSS-CANDIDATE
  finding #1 frontier-saturation anchor for backbone)
- HNeRV-class Pact-NeRV side-info backbone = HARD-EARNED-LITERATURE
  (sister PACT-NERV-IA3 commit 9cf9bdb16 architecture)
"""

from .architecture import (
    PactNervCrossCodecAConfig,
    PactNervCrossCodecASubstrate,
    Fec6BaseCodecPlaceholder,
)
from .archive import (
    CC_A_HEADER_SIZE,
    CC_A_MAGIC,
    CC_A_SCHEMA_VERSION,
    PactNervCrossCodecAArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervCrossCodecAScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "CC_A_HEADER_SIZE",
    "CC_A_MAGIC",
    "CC_A_SCHEMA_VERSION",
    "Fec6BaseCodecPlaceholder",
    "PactNervCrossCodecAArchive",
    "PactNervCrossCodecAConfig",
    "PactNervCrossCodecAScoreAwareLoss",
    "PactNervCrossCodecASubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
