# SPDX-License-Identifier: MIT
"""tac.substrates.sane_hnerv — Score-Aware NeRV Extended (substrate α).

The Fields-medal grand council 2026-05-12 PRIMARY-shipped substrate per the
B1 dual-base falsification (PR106 r2 +5204B / A1 +5216B within 12 bytes on
the magic_codec rate-axis). Codec-stacking on currently-available bases is
structurally extincted. ``sane_hnerv`` is the score-aware substrate
replacement.

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (PR95/100/101 substrate-engineering pattern) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~75 LOC mirroring PR101 = 71 LOC) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~530 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke planned) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing candidates DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets
    parser_section_manifest:   parse_archive() returns (decoder_sd, latents, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed state_dict + int16 latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~530 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke
"""

from .archive import (
    SaneHnervArchive,
    pack_archive,
    parse_archive,
)
from .architecture import SaneHnervSubstrate, SaneHnervConfig
from .score_aware_loss import SaneHnervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "SaneHnervArchive",
    "SaneHnervSubstrate",
    "SaneHnervConfig",
    "SaneHnervScoreAwareLoss",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
