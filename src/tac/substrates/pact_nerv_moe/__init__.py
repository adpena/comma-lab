# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_moe - Pact-NeRV-MOE (substrate L0 SKETCH).

Group 1 BLEEDING-EDGE variant per PACT-NERV-ULTIMATE (commit ``e3ad4243a``)
Variant #2 (Mixture-of-Experts decoder with pose-embedding-conditioned
routing per Atick-Redlich 1990 cooperative-receiver gate). Sister of the
distilled_scorer + Z4 cooperative-receiver substrates.

Literature anchor: Shazeer-Mirhoseini-Maziarz-Davis-Le-Hinton-Dean 2017
*"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
Layer"* (arXiv:1701.06538). Switch-Transformer / Mixtral-of-Experts
canonical OSS references: state-of-the-art top-k=2 expert routing with
load-balancing auxiliary loss.

Hypothesis (per PACT-NERV-ULTIMATE Variant #2 + Z4 sister):
Different decoder experts can specialize for different scene categories
(highway / urban / parking / static-vs-dynamic) that the SegNet+PoseNet
scorer rewards differently. Per-pair pose-embedding-conditioned top-k=2
routing IS the canonical cooperative-receiver gate (Atick-Redlich 1990):
the receiver (scorer) drives the per-input dispatching of compute to the
expert that minimizes its loss.

Architecture (L0 SCAFFOLD):

    Per-pair pose embedding e_i in R^pose_dim (small Linear from latent)
       |
       v
    Router (Linear -> softmax + top-k=2 selection + load-balancing aux)
       |
       v
    K experts: each is an HNeRV-class decoder block (sin act + depthsep conv)
       |
       v
    Weighted-sum expert outputs (route_probs[k] * expert_k(z))
       |
       v
    rgb_0 / rgb_1: per-pair RGB frame pair

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PNMO magic)
    parser_section_manifest:   header + router_blob + experts_blob +
                               latents_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli, av
    export_format:             FP4+Brotli for experts; raw fp16 router
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + delta*load_balance_aux
    bolt_on_loc_budget:        ~350 LOC MoE router + K experts at L1
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Shazeer 1701.06538 sparse MoE = HARD-EARNED-LITERATURE-SHAZEER
- top-k=2 routing = HARD-EARNED-MIXTRAL-CANONICAL
- pose-embedding-conditioned routing = CARGO-CULTED-PROMISING (sister to
  Z4 cooperative-receiver; L1 sweep validates vs uniform routing)
- K=4 experts at L0 = CARGO-CULTED-FOR-L0 (L1: K-sweep over {2,4,8})
- Load-balancing aux loss weight delta=0.01 = CARGO-CULTED-LITERATURE
  default (Shazeer §4.2 reports delta in [1e-3, 1e-1]; L1 sweep)
"""

from .architecture import (
    PactNervMoeConfig,
    PactNervMoeSubstrate,
    PoseConditionedRouter,
)
from .archive import (
    PNMO_HEADER_SIZE,
    PNMO_MAGIC,
    PNMO_SCHEMA_VERSION,
    PactNervMoeArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervMoeScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "PNMO_HEADER_SIZE",
    "PNMO_MAGIC",
    "PNMO_SCHEMA_VERSION",
    "PactNervMoeArchive",
    "PactNervMoeConfig",
    "PactNervMoeScoreAwareLoss",
    "PactNervMoeSubstrate",
    "PoseConditionedRouter",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
