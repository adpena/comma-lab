# SPDX-License-Identifier: MIT
"""tac.substrates.tc_nerv — Temporal-Consistency NeRV (substrate L0 SKETCH).

Sibling of ``sane_hnerv`` (primary substrate α) at L0 SKETCH per task #522 KK
operator-approved 2026-05-12. REACTIVATES to L1 SCAFFOLD only after α's
empirical anchor at <= 0.21 reveals whether HNeRV-family is the right class
for the pose-marginal-dominated PR106 r2 operating point.

Distinguishing feature vs ``sane_hnerv``:
    A first-class **temporal-consistency regularizer** in the score-aware
    Lagrangian: ``lambda_tc * ||rgb_pred[t+1] - rgb_pred[t]||^2``. The
    hypothesis is that PoseNet's 12-channel YUV6 input (two adjacent frames)
    rewards temporal smoothness at the pose-distortion level, so a renderer
    that explicitly trains to keep adjacent frames close should reduce
    ``pose_avg`` faster than the un-regularized HNeRV variant.

The architecture is otherwise SIREN + PixelShuffle + per-pair latents,
identical in spirit to ``sane_hnerv`` but with smaller decoder channels to
hit the ~200K parameter target.

Lane registration (L0 SKETCH, research_only=true per CLAUDE.md HNeRV parity
discipline opt-out):
    python tools/lane_maturity.py add-lane lane_substrate_tc_nerv_20260512 \
        --name "Temporal-Consistency NeRV substrate (L0 SKETCH)" --phase 2 \
        --notes "research_only=true ..."

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware loss + tc regularizer) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar TCV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~80 LOC) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + lambda_tc*||delta_t||^2) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~500 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin TCV1 fixed offsets
    parser_section_manifest:   parse_archive() returns (decoder_sd, latents, tc_table, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed state_dict + int16 latents + int8 tc-table + utf8-json meta
    score_aware_loss:          alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + lambda_tc*||delta_t||^2
    bolt_on_loc_budget:        ~500 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha's (sane_hnerv) empirical anchor must land at score <= 0.21
    AND the post-anchor diagnostic must report `pose_avg < 1e-4` (confirming
    HNeRV-family is the right class for the pose-marginal-dominated PR106 r2
    regime). On both conditions, this substrate gets impl_complete /
    real_archive_empirical / contest_cuda gates wired and the score-aware
    trainer (mirroring train_substrate_sane_hnerv.py) gets dispatched.
"""

from .archive import (
    TCNervArchive,
    pack_archive,
    parse_archive,
)
from .architecture import TCNervSubstrate, TCNervConfig
from .score_aware_loss import TCNervScoreAwareLoss, TCScoreAwareLossWeights

__all__ = [
    "TCNervArchive",
    "TCNervSubstrate",
    "TCNervConfig",
    "TCNervScoreAwareLoss",
    "TCScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
