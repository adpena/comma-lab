# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_neural_codec_e2e_cross - Pact-NeRV-NEURAL-CODEC-E2E-CROSS (L0 SKETCH).

Group 4 CROSS-CODEC composition variant per PACT-NERV-ULTIMATE (commit
``e3ad4243a``) variant #18 — END-TO-END neural codec compositing with
Pact-NeRV backbone where a Ballé-style hyperprior network gates the
cross-codec composition itself (NOT a fixed-bytes base codec + Pact-NeRV
residual; the BOTH codecs are jointly trained neural networks whose
composition is learned per-pair via a learned gate).

Literature anchors:
- Ballé et al. 2018 "Variational image compression with a scale hyperprior"
  (arXiv:1802.01436) — canonical hyperprior gating pattern.
- Atick-Redlich 1990 cooperative-receiver framing — the hyperprior IS the
  cooperative-receiver side information that picks which codec dominates
  per per-pair region.
- CROSS-CANDIDATE finding #3 empirical anchor (PR101/A1/fec6 <-> PR106
  per-axis Pearson [-0.094, -0.078] = SUPER_ADDITIVE signature per
  Catalog #322).
- Sister pact_nerv_cross_codec_a (fec6 base + Pact-NeRV residual) +
  pact_nerv_cross_codec_b (PR106 base + IA3 modulation).

Hypothesis (per PACT-NERV-ULTIMATE Variant #18): a learned hyperprior gate
that conditions on per-pair latent statistics can route bits between
two codec branches more efficiently than static residual additive
composition (CC_A/CC_B at L0); SUPER_ADDITIVE per Catalog #322 because
the codecs operate on DIFFERENT receptive fields AND the gate is
end-to-end trained to allocate bits per per-pair Pareto-optimal mix.

Architecture (L0 SCAFFOLD):

    Codec branch A (Pact-NeRV-class HNeRV decoder)
       |  (latents_a, decoder_state_a)
       v                                        Hyperprior gate g(z) ∈ [0, 1]
    Codec branch B (Pact-NeRV-class HNeRV decoder)  |
       |  (latents_b, decoder_state_b)               v
       v                                           Composition:
    Gate hyperprior network h(z_a, z_b)             rgb = g * branch_a + (1-g) * branch_b

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (NCEC magic)
    parser_section_manifest:   header + decoder_a_blob + decoder_b_blob +
                               hyperprior_blob + latents_a_blob + latents_b_blob
                               + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoders + hyperprior
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + lambda_gate * gate_entropy_bonus
    bolt_on_loc_budget:        ~400 LOC neural-codec composition + hyperprior
    no_op_detector_planned:    Catalog #139 byte-mutation smoke
                               (all 3 blobs: decoder_a, decoder_b, hyperprior)

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- End-to-end neural-codec composition = HARD-EARNED-LITERATURE
  (Ballé 2018 hyperprior + Atick-Redlich 1990 cooperative-receiver)
- Hyperprior gate g(z) ∈ [0,1] via sigmoid = CARGO-CULTED (L1: learned
  per-pair Pareto-optimal mixture per Boyd convex feasibility)
- Two identical HNeRV branches A,B = CARGO-CULTED (L1: heterogeneous
  branches per CROSS-CANDIDATE finding #3 SUPER_ADDITIVE signature)
- Gate entropy bonus in loss = CARGO-CULTED (L1: information-theoretic
  rate-distortion-gate Lagrangian per Shannon 1948 + R(D) bound)
"""

from .architecture import (
    HyperpriorGate,
    PactNervNeuralCodecE2ECrossConfig,
    PactNervNeuralCodecE2ECrossSubstrate,
)
from .archive import (
    NCEC_HEADER_SIZE,
    NCEC_MAGIC,
    NCEC_SCHEMA_VERSION,
    PactNervNeuralCodecE2ECrossArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervNeuralCodecE2ECrossScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "HyperpriorGate",
    "NCEC_HEADER_SIZE",
    "NCEC_MAGIC",
    "NCEC_SCHEMA_VERSION",
    "PactNervNeuralCodecE2ECrossArchive",
    "PactNervNeuralCodecE2ECrossConfig",
    "PactNervNeuralCodecE2ECrossScoreAwareLoss",
    "PactNervNeuralCodecE2ECrossSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
