# SPDX-License-Identifier: MIT
"""tac.substrates.ds_nerv — Depth-separable NeRV (substrate L0 SKETCH).

Per-frame implicit renderer using depth-separable convolutions throughout
(Quantizr-paradigm influence: depthwise-separable CNN as PR101's 88K-param
FiLM-conditioned design). The substrate trades a fraction of expressive
capacity for ~40% fewer params at the same depth, making the rate term
naturally smaller.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 22-byte header) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~90 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~510 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (DSV1)
    parser_section_manifest:   parse_archive() -> 5 sections (header + 4 blobs)
                               + "depthwise_separable_params" section
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed dsconv decoder state_dict
                               + int16 latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~510 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke
"""

from .architecture import DsnervConfig, DsnervSubstrate
from .archive import (
    DsnervArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import DsnervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "DsnervArchive",
    "DsnervConfig",
    "DsnervScoreAwareLoss",
    "DsnervSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
