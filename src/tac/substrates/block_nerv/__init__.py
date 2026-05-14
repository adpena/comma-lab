# SPDX-License-Identifier: MIT
"""tac.substrates.block_nerv — Per-Pair Block-Decoder NeRV (substrate L0 SKETCH).

Sibling of ``sane_hnerv`` (primary substrate α) at L0 SKETCH per task #522 KK
operator-approved 2026-05-12. REACTIVATES to L1 SCAFFOLD only after α's
empirical anchor at <= 0.21 reveals whether HNeRV-family is the right class
for the pose-marginal-dominated PR106 r2 operating point.

Distinguishing feature vs ``sane_hnerv``:
    Per-pair **low-rank decoder deltas** layered on top of a shared base
    decoder. The architecture is a shared SIREN/PixelShuffle base plus
    ``num_pairs`` independent low-rank residuals applied to the latent
    embedding. Total params:

        shared base: ~150K (same SIREN+PixelShuffle as sane_hnerv)
        per-pair LoRA: 600 * (latent_dim * rank * 2) ~ 70K (rank=2, latent=28)
        ── total ~ 220K (council target)

    The hypothesis: per-pair low-rank deltas let the renderer over-specialize
    to each frame-pair's idiosyncrasies (lighting, fast-motion artifacts)
    without paying the cost of fully independent per-pair decoders. Storage
    is dominated by the per-pair LoRA tables; the BNV1 archive grammar
    stores those as a separate quantized section.

Lane registration (L0 SKETCH, research_only=true per CLAUDE.md HNeRV parity
discipline opt-out):
    python tools/lane_maturity.py add-lane lane_substrate_block_nerv_20260512 \
        --name "Block-Decoder NeRV substrate (L0 SKETCH)" --phase 2 \
        --notes "research_only=true ..."

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware loss inherits sane_hnerv shape) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar BNV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~85 LOC) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~520 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin BNV1 fixed offsets
    parser_section_manifest:   parse_archive() returns (base_decoder_sd, latents, lora_table, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed state_dict + int16 latents + int8 lora-table + utf8-json meta
    score_aware_loss:          alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~520 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha's (sane_hnerv) empirical anchor must land at score <= 0.21
    AND the post-anchor diagnostic must report `seg_avg > 1e-3` OR
    "per-pair variance not consumed by shared decoder" (confirming the
    per-pair-specialization hypothesis is the right vector to attack). On
    those conditions, this substrate gets impl_complete /
    real_archive_empirical / contest_cuda gates wired.
"""

from .archive import (
    BlockNervArchive,
    pack_archive,
    parse_archive,
)
from .architecture import BlockNervSubstrate, BlockNervConfig
from .score_aware_loss import BlockNervScoreAwareLoss, BlockScoreAwareLossWeights

__all__ = [
    "BlockNervArchive",
    "BlockNervSubstrate",
    "BlockNervConfig",
    "BlockNervScoreAwareLoss",
    "BlockScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
