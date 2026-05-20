# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_bayesian — Pact-NeRV-Bayesian (L0 SKETCH).

Group 2 variant #8 of PACT-NERV-ULTIMATE.

Literature anchors:
- MacKay 1992 *"A Practical Bayesian Framework for Backpropagation Networks"*
- Kingma & Welling 2014 VAE (arXiv:1312.6114)
- Blundell et al. 2015 *"Weight Uncertainty in Neural Networks"* (Bayes by
  Backprop), arXiv:1505.05424

MacKay inner council seat per CLAUDE.md (memorial seat). Canonical OSS:
`JavierAntoran/Bayesian-Neural-Networks`.

Distinguishing primitive: each weight in the latent embedding layer is a
learnable Gaussian (mean, log_sigma) instead of a deterministic point estimate.
At training time the substrate samples weights per-batch via the reparameterization
trick + KL divergence regularization keeps the posterior close to a unit-Gaussian
prior. The per-pair difficulty signal IS the per-pair posterior variance —
uncertainty IS difficulty per the MacKay framing.

Status: **L0 SKETCH** (research_only=true).

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PBN)
    parser_section_manifest:   parse_archive() -> 5 sections (header +
                               decoder_blob + posterior_blob + latents_blob +
                               meta_blob)
    inflate_runtime_loc_budget: ≤ 150 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli weights; posterior shipped as
                               (mean, sigma) pairs; mean used at inflate time
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + kl_weight * KL(posterior || N(0,1))
    bolt_on_loc_budget:        ~350 LOC (substrate_engineering tag; task-cap)
    no_op_detector_planned:    Catalog #139 + byte-mutation smoke

Cargo-cult audit per Catalog #303:
- Bayes-by-Backprop reparameterization trick = HARD-EARNED (Blundell 1505.05424
  + Kingma 1312.6114 + MacKay inner council seat).
- Unit-Gaussian prior N(0, 1) = HARD-EARNED (Blundell §3.1 canonical).
- KL weight 1.0 = CARGO-CULTED at L0 (alternatives: KL annealing per epoch;
  Stage 1 sweep).
- Bayesian ONLY on latent embedding layer = CARGO-CULTED at L0 (alternative:
  Bayesian decoder + Bayesian latents; Stage 1 expansion).
- Mean (not sample) at inflate time = HARD-EARNED (Blundell §4 canonical).
"""

from .architecture import (
    BayesianLinearLayer,
    PactNervBayesianConfig,
    PactNervBayesianSubstrate,
)
from .archive import (
    PBN_HEADER_SIZE,
    PBN_MAGIC,
    PBN_SCHEMA_VERSION,
    PactNervBayesianArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervBayesianScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "BayesianLinearLayer",
    "PBN_HEADER_SIZE",
    "PBN_MAGIC",
    "PBN_SCHEMA_VERSION",
    "PactNervBayesianArchive",
    "PactNervBayesianConfig",
    "PactNervBayesianScoreAwareLoss",
    "PactNervBayesianSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
