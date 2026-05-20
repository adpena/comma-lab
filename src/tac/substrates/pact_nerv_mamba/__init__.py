# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_mamba - Pact-NeRV-MAMBA (substrate L0 SKETCH).

Group 1 BLEEDING-EDGE variant per PACT-NERV-ULTIMATE (commit ``e3ad4243a``)
Variant #1 (Mamba state-space backbone replacement of HNeRV ConvNeXt
decoder). Sister of Z7-Mamba2 substrate per FILM-FAMILY-RESEARCH Section
7.2.

Literature anchor: Mamba (Gu & Dao 2023) [arXiv:2312.00752] + Mamba-2
(Dao & Gu 2024) [arXiv:2405.21060]. Canonical OSS repo: state-spaces/mamba
(~12K stars). Mamba's selective state-space scan (S6) achieves linear-time
sequence modeling with content-aware gating — competitive with Transformers
on language + state-of-the-art on long-range dependencies. The Mamba-2
extension (SSD = state-space duality) unifies Mamba with structured
attention via the SSM-attention equivalence theorem.

Hypothesis (per PACT-NERV-ULTIMATE Variant #1 + Z7-Mamba2 sister symposium):
Mamba-2's selective-scan SSM is a SUPERIOR temporal-recurrence prior over
the 600-pair contest video vs HNeRV's conv-based positional encoding. The
backbone replacement risks higher LOC + uncertainty on the scorer-distortion
axis (Mamba's training dynamics may differ from HNeRV's well-tuned conv
backbone).

Architecture (L0 SCAFFOLD):

    Per-pair latent z_i in R^latent_dim (sequence dimension = 600 pairs)
       |
       v
    Mamba-2 block (selective SSM scan; ~1800 LOC budget at L1)
       |  (L0 SCAFFOLD: stand-in linear-recurrence proxy ~50 LOC)
       v
    NeRV decoder (conv + SinAct + PixelShuffle upsample)
       |
       v
    rgb_0 / rgb_1: per-pair RGB frame pair

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PNMB magic)
    parser_section_manifest:   header + decoder_blob + latents_blob +
                               ssm_state_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli, av
    export_format:             FP4+Brotli for decoder; raw fp16 SSM-state
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~1800 LOC Mamba-2 backbone (L1 budget;
                               L0 stand-in ~50 LOC linear-recurrence proxy)
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Mamba-2 selective-scan SSM = HARD-EARNED-LITERATURE-Gu-Dao
- Linear-recurrence L0 stand-in = CARGO-CULTED-FOR-L0 (L1: replace with real
  mamba_ssm.modules.mamba2 via state-spaces/mamba install)
- Backbone replacement of HNeRV ConvNeXt = CARGO-CULTED-MAY-BE (untested on
  contest scorer; Mamba's video applications validated only on UCF101)
- Sequence-dimension = 600 pairs treated as time series = HARD-EARNED-
  STRUCTURAL (matches contest's temporal pair structure)
"""

from .architecture import (
    PactNervMambaConfig,
    PactNervMambaSubstrate,
    StateSpaceRecurrenceBlock,
)
from .archive import (
    PNMB_HEADER_SIZE,
    PNMB_MAGIC,
    PNMB_SCHEMA_VERSION,
    PactNervMambaArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervMambaScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "PNMB_HEADER_SIZE",
    "PNMB_MAGIC",
    "PNMB_SCHEMA_VERSION",
    "PactNervMambaArchive",
    "PactNervMambaConfig",
    "PactNervMambaScoreAwareLoss",
    "PactNervMambaSubstrate",
    "ScoreAwareLossWeights",
    "StateSpaceRecurrenceBlock",
    "pack_archive",
    "parse_archive",
]
