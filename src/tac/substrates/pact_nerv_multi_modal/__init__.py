# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_multi_modal — Pact-NeRV-MultiModal (L0 SKETCH).

Group 2 variant #9 of PACT-NERV-ULTIMATE.

Literature anchors:
- Baltrušaitis 2019 *"Multimodal Machine Learning: A Survey and Taxonomy"*
- Radford et al. 2021 CLIP (arXiv:2103.00020)

Canonical OSS: `openai/CLIP`.

Distinguishing primitive: 3-tower conditioning fusion — ego-pose tower +
SegNet-class-prior tower + odometry/IMU tower — fused via concatenated
cross-attention-light projection. Per Catalog #311 sister discipline:
ego-motion-conditioned next-frame prediction is the canonical Rao-Ballard +
Atick-Redlich framing; this variant extends it with class-aware + odometry
side information.

Status: **L0 SKETCH** (research_only=true).

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (PMM)
    parser_section_manifest:   parse_archive() -> 6 sections (header +
                               decoder_blob + ego_pose_blob + class_prior_blob +
                               odometry_blob + meta_blob)
    inflate_runtime_loc_budget: ≤ 150 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli weights; int16 conditioning vectors
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~250 LOC (substrate_engineering tag; task-cap)
    no_op_detector_planned:    Catalog #139 + byte-mutation smoke

Cargo-cult audit per Catalog #303:
- 3-tower concat fusion = CARGO-CULTED at L0 (alternative: cross-attention
  per Baltrušaitis 2019 taxonomy; alternative: gated fusion per Highway Networks;
  Stage 1 sweep).
- Tower weights shared across (frame_0, frame_1) = CARGO-CULTED at L0
  (alternative: per-frame tower projections).
- 5-class SegNet prior dimensionality = HARD-EARNED (matches upstream SegNet
  classes; aligns with Catalog #311 ego-motion-conditioning).
- Odometry as 4-D vector = CARGO-CULTED at L0 (alternative: full IMU 9-DoF
  per Comma2k19 schema; Stage 1 ablation).
"""

from .architecture import (
    MultiModalConditioningFusion,
    PactNervMultiModalConfig,
    PactNervMultiModalSubstrate,
)
from .archive import (
    PMM_HEADER_SIZE,
    PMM_MAGIC,
    PMM_SCHEMA_VERSION,
    PactNervMultiModalArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervMultiModalScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "MultiModalConditioningFusion",
    "PMM_HEADER_SIZE",
    "PMM_MAGIC",
    "PMM_SCHEMA_VERSION",
    "PactNervMultiModalArchive",
    "PactNervMultiModalConfig",
    "PactNervMultiModalScoreAwareLoss",
    "PactNervMultiModalSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
