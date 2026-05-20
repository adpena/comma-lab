# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_neural_codec_e2e - Pact-NeRV-NEURAL-CODEC-E2E (L0 SKETCH).

Group 1 BLEEDING-EDGE variant per PACT-NERV-ULTIMATE (commit ``e3ad4243a``)
Variant #4 (end-to-end neural codec: Ballé 2018 hyperprior + HNeRV decoder
FUSION). The ULTIMATE-PAPER + ULTIMATE-INTERPRETABILITY substrate per the
design memo.

Literature anchors:
- Ballé-Minnen-Singh-Hwang-Johnston 2018 *"Variational Image Compression
  with a Scale Hyperprior"* (arXiv:1802.01436). Canonical OSS:
  InterDigitalInc/CompressAI (Ballé hyperprior reference). The 2018 paper
  introduced (a) the entropy bottleneck for fully factorized priors and
  (b) the scale hyperprior for conditional Gaussian priors.
- Cheng-Sun-Takeuchi-Katto 2020 *"Learned Image Compression with Discretized
  Gaussian Mixture Likelihoods and Attention Modules"* (arXiv:2001.01568).

Hypothesis (per PACT-NERV-ULTIMATE Variant #4 + Ballé inner-council seat):
Replacing the HNeRV decoder's brotli/fp4 weight-export pipeline with a
LEARNED end-to-end neural codec (entropy bottleneck + hyperprior over latent
distributions) jointly optimizes the rate-distortion tradeoff at training
time. The 2018 Ballé scale-hyperprior is the canonical anchor; the entropy
bottleneck is differentiable, so the codec is co-optimized with the renderer
under the score-aware Lagrangian. Sister of Z3 Ballé hyperprior bolt-on but
PRIMARY architectural class (not bolt-on) per HNeRV parity L7.

Architecture (L0 SCAFFOLD):

    Per-pair latent z_i in R^latent_dim
       |
       v
    Hyperprior h_i = encoder_hyper(z_i)  (auxiliary latent for scale prior)
       |
       v
    Entropy bottleneck (Ballé 2018 §3): rate_loss = -log2(p(z | sigma(h)))
       |
       v
    HNeRV decoder (conv + SinAct + PixelShuffle) from z_i
       |
       v
    rgb_0 / rgb_1: per-pair RGB frame pair

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PNNC magic)
    parser_section_manifest:   header + decoder_blob + hyperprior_blob +
                               latents_blob + scales_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli, av
    export_format:             FP4+Brotli for decoder; constriction or
                               brotli-wrapped int16 for entropy-coded latents
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + delta*rate_bottleneck (Ballé differentiable)
    bolt_on_loc_budget:        substrate_engineering (NOT bolt-on per HNeRV L7)
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Ballé 2018 scale hyperprior = HARD-EARNED-LITERATURE-Balle-CompressAI-anchor
- Entropy bottleneck differentiable rate loss = HARD-EARNED-LITERATURE-Balle-2018
- Hyperprior dim = 8 at L0 = CARGO-CULTED-FOR-L0 (L1: hyperprior-dim sweep
  over {4, 8, 16, 32})
- Conditional Gaussian (not GMM per Cheng 2020) = CARGO-CULTED-MAY-BE
  (Cheng 2020 GMM is SOTA on Kodak; L1 ablation per Cheng vs Ballé hyperprior)
- HNeRV decoder backbone retained = HARD-EARNED-STRUCTURAL (HNeRV PR101 won
  contest at 0.193; codec replaces ONLY the weight-export pipeline)
"""

from .architecture import (
    HyperpriorEncoder,
    PactNervNeuralCodecE2eConfig,
    PactNervNeuralCodecE2eSubstrate,
)
from .archive import (
    PNNC_HEADER_SIZE,
    PNNC_MAGIC,
    PNNC_SCHEMA_VERSION,
    PactNervNeuralCodecE2eArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervNeuralCodecE2eScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "HyperpriorEncoder",
    "PNNC_HEADER_SIZE",
    "PNNC_MAGIC",
    "PNNC_SCHEMA_VERSION",
    "PactNervNeuralCodecE2eArchive",
    "PactNervNeuralCodecE2eConfig",
    "PactNervNeuralCodecE2eScoreAwareLoss",
    "PactNervNeuralCodecE2eSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
