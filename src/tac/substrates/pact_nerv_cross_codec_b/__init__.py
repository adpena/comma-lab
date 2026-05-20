# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_cross_codec_b - Pact-NeRV-CROSS-CODEC-B (substrate L0 SKETCH).

Group 4 CROSS-CODEC composition variant per PACT-NERV-ULTIMATE (commit
``e3ad4243a``) variant #17 — PR106 latent-score-table base + Pact-NeRV-IA3
side-info bolt-on (CROSS-CANDIDATE finding #3 empirical SUPER_ADDITIVE
signature for PR106 ↔ HNeRV-class pairs + sister PR107 ↔ PR106 INDETERMINATE
+ PR107 ↔ fec6 SUPER_ADDITIVE per CROSS-CANDIDATE matrix Section 2).

Literature anchor: Atick-Redlich 1990 cooperative-receiver + CROSS-CANDIDATE
finding #3 (PR101/A1/fec6 ↔ PR106 per-axis Pearson [-0.094, -0.078]) +
Liu et al. 2022 arXiv:2205.05638 (IA3 γ-only ego-pose-conditioned per-block
modulation; sister of pact_nerv_ia3 commit 9cf9bdb16).

Hypothesis (per PACT-NERV-ULTIMATE Variant #17 + cross_codec_orthogonality_
predictor_consumer commit ``80484241f``): PR106 (format0d codec family)
carries score-relevant overhead on a different receptive field than the
HNeRV-class Pact-NeRV-IA3 backbone; their composition is SUPER_ADDITIVE
(top-K Jaccard < 0.05 across codec families).

Architecture (L0 SCAFFOLD):

    PR106 base codec (latent-score-table + format0d)
       |  (bytes A_pr106 in archive section base_codec_blob)
       v
    Pact-NeRV-IA3 side-info bolt-on (HNeRV decoder + γ-only ego-pose modulation)
       |  (bytes B_ia3 in archive section ia3_side_info_blob)
       v
    Composition: rgb_0 = pr106_render(pair) + alpha * ia3_residual(pair)
                 rgb_1 = pr106_render(pair) + alpha * ia3_residual(pair)

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (CC_B magic)
    parser_section_manifest:   header + pr106_base_blob + ia3_decoder_blob
                               + latent_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~350 LOC cross-codec composition + IA3 γ modulation
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- PR106 ↔ HNeRV-IA3 cross-codec composition = HARD-EARNED-EMPIRICAL
  (CROSS-CANDIDATE finding #3 empirical anchor)
- IA3 γ-only ego-pose modulation = HARD-EARNED-LITERATURE
  (Liu 2022 + sister pact_nerv_ia3 commit 9cf9bdb16)
- Static residual additive composition at L0 = CARGO-CULTED
  (L1: learned composition gate per Atick-Redlich 1990)
- PR106 base codec inherited as-is = HARD-EARNED-EMPIRICAL
- HNeRV-class Pact-NeRV-IA3 backbone = HARD-EARNED-LITERATURE
"""

from .architecture import (
    Pr106BaseCodecPlaceholder,
    PactNervCrossCodecBConfig,
    PactNervCrossCodecBSubstrate,
)
from .archive import (
    CC_B_HEADER_SIZE,
    CC_B_MAGIC,
    CC_B_SCHEMA_VERSION,
    PactNervCrossCodecBArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervCrossCodecBScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "CC_B_HEADER_SIZE",
    "CC_B_MAGIC",
    "CC_B_SCHEMA_VERSION",
    "PactNervCrossCodecBArchive",
    "PactNervCrossCodecBConfig",
    "PactNervCrossCodecBScoreAwareLoss",
    "PactNervCrossCodecBSubstrate",
    "Pr106BaseCodecPlaceholder",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
