# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_selector_v3 - Pact-NeRV-SELECTOR-V3 (substrate L0 SKETCH).

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE
(commit ``e3ad4243a``) variant #12 + task-spec G3 family (Rice-Golomb
encoding extension of the FEC6 fixed-Huffman k=16 frame-exploit selector).

Literature anchor: Golomb 1966 *"Run-Length Encodings"* + Rice-Plaunt 1971
*"Adaptive Variable-Length Coding for Efficient Compression of Spacecraft
Television Data"*. Rice-Golomb codes are a canonical family of prefix
codes optimal for geometric distributions; for a 16-mode selector palette
where most pairs select a small-index mode (geometric-decay pattern) the
Rice-Golomb code is ~5-15 bytes more compact than fixed-Huffman per the
600-pair stream.

Hypothesis (per PACT-NERV-ULTIMATE Variant #12 + CROSS-CANDIDATE finding
#1): the FEC6 frame-exploit selector indices follow a geometric-decay
distribution (mode 0 = "none" dominates; later modes rare). Rice-Golomb
with k=2 or k=3 should achieve near-optimal code-length for this pattern.

Architecture (L0 SCAFFOLD):

    HNeRV-class base decoder (mirrors pact_nerv_ia3 sister)
       |
       v
    Per-pair selector index in [0, 16) per FEC6 k=16 palette
       |
       v
    Rice-Golomb encoder (parameter k; ~50 LOC primitive per Golomb 1966 +
    Rice 1971)
       |
       v
    rgb_0 / rgb_1: HNeRV decoder + selector-conditioned frame-0 transforms

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PSV3 magic)
    parser_section_manifest:   header + base_decoder_blob + rice_golomb_blob
                               + latents_blob + meta_blob
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder; Rice-Golomb-coded u8 selectors
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~150 LOC Rice-Golomb primitive
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- Rice-Golomb encoding for geometric-decay distributions = HARD-EARNED-LITERATURE
- Fixed k=2 parameter at L0 = CARGO-CULTED (L1 sweep: adaptive-k per stream)
- Direct symbol-by-symbol encoding = CARGO-CULTED (L1: run-length pre-processing)
- FEC6 k=16 palette inherited = HARD-EARNED-EMPIRICAL (CROSS-CANDIDATE finding #1)
"""

from .architecture import (
    PactNervSelectorV3Config,
    PactNervSelectorV3Substrate,
    RiceGolombSelectorCoder,
)
from .archive import (
    PactNervSelectorV3Archive,
    PSV3_HEADER_SIZE,
    PSV3_MAGIC,
    PSV3_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    PactNervSelectorV3ScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "PSV3_HEADER_SIZE",
    "PSV3_MAGIC",
    "PSV3_SCHEMA_VERSION",
    "PactNervSelectorV3Archive",
    "PactNervSelectorV3Config",
    "PactNervSelectorV3ScoreAwareLoss",
    "PactNervSelectorV3Substrate",
    "RiceGolombSelectorCoder",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
