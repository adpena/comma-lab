# SPDX-License-Identifier: MIT
"""tac.substrates.pact_nerv_selector_v4 - Pact-NeRV-SELECTOR-V4 (substrate L0 SKETCH).

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE
(commit ``e3ad4243a``) variant #13 + task-spec G3 family (run-length-
encoded selector extension of the FEC6 fixed-Huffman k=16 selector).

Literature anchor: Robinson-Cherry 1967 *"Results of a Prototype Television
Bandwidth Compression Scheme"* + Capon 1959 (canonical run-length encoding
references). RLE is optimal for streams with long consecutive-symbol runs;
for the FEC6 600-pair selector, if the "none" mode (selector=0) dominates
contiguous runs across pairs (e.g. static scenes), RLE can save 30-100
bytes vs per-pair fixed-Huffman.

Hypothesis (per PACT-NERV-ULTIMATE Variant #13 + CROSS-CANDIDATE finding
#1): the FEC6 frame-exploit selector exhibits temporally-coherent runs
(consecutive pairs select the same mode during static scenes). RLE
exploits this temporal structure for additional rate savings.

Architecture (L0 SCAFFOLD):

    HNeRV-class base decoder (mirrors pact_nerv_ia3 sister)
       |
       v
    Per-pair selector index in [0, 16) per FEC6 k=16 palette
       |
       v
    Run-length encoder: (value, run_length) pairs, varint-encoded run_length
    (~50 LOC primitive)
       |
       v
    rgb_0 / rgb_1: HNeRV decoder + selector-conditioned frame-0 transforms

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin (PSV4 magic)
    parser_section_manifest:   header + base_decoder + rle_selector + latents + meta
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             FP4+Brotli for decoder; RLE-coded (value, run_length) pairs
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~150 LOC RLE primitive
    no_op_detector_planned:    Catalog #139 byte-mutation smoke

Cargo-cult audit per Catalog #303:
- RLE for temporally-coherent streams = HARD-EARNED-LITERATURE
- (value, varint) representation = HARD-EARNED-LITERATURE (standard)
- No back-reference to value distribution = CARGO-CULTED at L0 (alt: RLE+Huffman hybrid at L1)
- FEC6 k=16 palette inherited = HARD-EARNED-EMPIRICAL
"""

from .architecture import (
    PactNervSelectorV4Config,
    PactNervSelectorV4Substrate,
    RunLengthSelectorCoder,
)
from .archive import (
    PactNervSelectorV4Archive,
    PSV4_HEADER_SIZE,
    PSV4_MAGIC,
    PSV4_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)
from .archive_candidate import (
    export_pact_nerv_selector_v4_mlx_archive,
    pack_archive_from_exported_state_dict,
    selector_v4_meta_from_config,
)
from .score_aware_loss import (
    PactNervSelectorV4ScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "PSV4_HEADER_SIZE",
    "PSV4_MAGIC",
    "PSV4_SCHEMA_VERSION",
    "PactNervSelectorV4Archive",
    "PactNervSelectorV4Config",
    "PactNervSelectorV4ScoreAwareLoss",
    "PactNervSelectorV4Substrate",
    "RunLengthSelectorCoder",
    "ScoreAwareLossWeights",
    "export_pact_nerv_selector_v4_mlx_archive",
    "pack_archive",
    "pack_archive_from_exported_state_dict",
    "parse_archive",
    "selector_v4_meta_from_config",
]
