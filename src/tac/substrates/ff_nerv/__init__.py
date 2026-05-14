# SPDX-License-Identifier: MIT
"""tac.substrates.ff_nerv — Frequency-domain NeRV (substrate L0 SKETCH).

Per-frame implicit renderer operating in 2D DCT frequency space; the
decoder predicts a band-limited DCT coefficient grid per pair, and the
inflate-time path reconstructs RGB frames via inverse 2D DCT (IDCT2). The
frequency-band structure is the substrate's principled prior: low-frequency
coefficients (DC + early AC) dominate perceptual content, so the rate term
is naturally banded.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.
Catalog #146 trainer runtime contract is NOT yet emitted — the trainer
``experiments/train_substrate_ff_nerv_full_main.py`` is intentionally
absent at L0; that lands at L1 SCAFFOLD per the Phase 2 lifecycle.

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 22-byte header) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~90 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (IDCT2 -> RGB; NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~520 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired via tac.differentiable_eval_roundtrip at L1) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (FFV1)
    parser_section_manifest:   parse_archive() -> 5 sections (header + 4 blobs)
                               + "frequency_band_coefficients" section
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli (numpy transitive via torch)
    export_format:             brotli-compressed dct_coeff_decoder state_dict
                               + int16 per-pair DCT coeff grid + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~520 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke
"""

from .architecture import FfnervConfig, FfnervSubstrate
from .archive import (
    FfnervArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import FfnervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "FfnervArchive",
    "FfnervConfig",
    "FfnervScoreAwareLoss",
    "FfnervSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
