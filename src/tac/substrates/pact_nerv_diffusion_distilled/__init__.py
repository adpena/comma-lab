# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_diffusion_distilled - Pact-NeRV-DIFFUSION-DISTILLED (L0 SKETCH).

Group 1 BLEEDING-EDGE variant per PACT-NERV-ULTIMATE (commit ``e3ad4243a``)
Variant #3 (diffusion teacher -> 1-step student distillation). PAPER-WORTHY
canonical pattern per the design memo.

Literature anchors:
- Song-Dhariwal-Chen-Sutskever 2023 *"Consistency Models"* (arXiv:2303.01469)
  + Salimans-Ho 2022 *"Progressive Distillation for Fast Sampling of Diffusion
  Models"* (arXiv:2202.00512). Canonical OSS: openai/consistency_models.
- Yin-Gharbi-Zhang-Shechtman-Durand-Freeman-Park 2024 *"One-step Diffusion
  with Distribution Matching Distillation"* (arXiv:2311.18828; DMD).

Hypothesis (per PACT-NERV-ULTIMATE Variant #3):
A T-step diffusion teacher trained on the contest video learns a powerful
score-aware generative prior. The 1-step student distilled from the teacher
inherits the teacher's quality at inference latency comparable to the HNeRV
class. The student IS the substrate's renderer; the teacher is compress-time
only (NOT in archive.zip per CLAUDE.md "Strict scorer rule" + HNeRV parity
L4 ≤200 LOC inflate budget).

Architecture (L0 SCAFFOLD):

    Per-pair latent z_i in R^latent_dim
       |
       v
    DiffusionStudentRenderer (1-step distilled from T-step teacher)
       |  (L0 SCAFFOLD: simple latent->RGB head as stand-in for distilled
       |   student; Stage 1 dispatch lands real consistency-model distillation)
       v
    rgb_0 / rgb_1: per-pair RGB frame pair

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PNDD magic)
    parser_section_manifest:   header + student_blob + latents_blob +
                               meta_blob (teacher NOT shipped per
                               distillation contract)
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli, av
    export_format:             FP4+Brotli for student decoder
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
                               + delta*L_distill (student matches teacher logits)
    bolt_on_loc_budget:        ~400 LOC student decoder + distillation loss
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Consistency-model distillation (Song 2303.01469) = HARD-EARNED-LITERATURE
- 1-step student inference = HARD-EARNED-LITERATURE (Song §3 + Yin 2311.18828)
- Teacher T=4 timesteps at compress time = CARGO-CULTED-FOR-L0 (L1: T-sweep
  over {1, 2, 4, 8, 16})
- Student matches DECODER output (not pixel space) = CARGO-CULTED-MAY-BE
  (alternative: pixel-space distillation per DMD; L1 ablation)
"""

from .architecture import (
    DiffusionStudentDecoder,
    PactNervDiffusionDistilledConfig,
    PactNervDiffusionDistilledSubstrate,
)
from .archive import (
    PNDD_HEADER_SIZE,
    PNDD_MAGIC,
    PNDD_SCHEMA_VERSION,
    PactNervDiffusionDistilledArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervDiffusionDistilledScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "DiffusionStudentDecoder",
    "PNDD_HEADER_SIZE",
    "PNDD_MAGIC",
    "PNDD_SCHEMA_VERSION",
    "PactNervDiffusionDistilledArchive",
    "PactNervDiffusionDistilledConfig",
    "PactNervDiffusionDistilledScoreAwareLoss",
    "PactNervDiffusionDistilledSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
