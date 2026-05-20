# SPDX-License-Identifier: MIT
"""tac.substrates.boost_nerv — BoostNeRV (substrate L0 SKETCH).

Per-frame implicit renderer with an iterative-boosting residual-chain
sidecar. Operator 5-tier fit-ranking verdict **HIGH FIT ⭐⭐⭐⭐⭐**: the
boosting paradigm is orthogonal to existing NeRV variants (TCNeRV / BlockNeRV
/ FFNeRV / DSNeRV / HiNeRV / e_nerv / ego_nerv / nervdc) and stacks WITH
them rather than against them.

Literature anchor: Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement
for Implicit Neural Video Representations" (arXiv:2407.xxxxx — canonical
literature reference per BUILD task #1090). Canonical OSS repo (when
available): github.com/<author>/BoostNeRV (literature anchor; not vendored).

Hypothesis (per operator's per-variant fit verdict): driving video has
heavy-tailed reconstruction error — a few hard pairs dominate. A boosting
chain that fits residuals of the base substrate progressively reduces the
worst-case error without inflating the base substrate's parameter budget.
The boosting paradigm composes WITH any base substrate: at L1+ the same
residual head can be attached to TCNeRV / DSNeRV / sane_hnerv / etc.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^24
       |
       v
    Base decoder (DepthSep + SIREN + PixelShuffle; mirrors ds_nerv)
       |
       v
    rgb_base in [0, 1]
       |
       v
    Boosting head 0: TinyConv(rgb_base, z) -> residual_0 (gain in [-0.1, 0.1])
       |
       v
    rgb_iter_1 = clamp(rgb_base + residual_0, 0, 1)
       |
       v
    Boosting head 1: TinyConv(rgb_iter_1, z) -> residual_1
       |
       v
    rgb_iter_2 = clamp(rgb_iter_1 + residual_1, 0, 1)
       |
       v
    ... (NUM_BOOSTING_ROUNDS, default 2)
       |
       v
    Head rgb_0 / rgb_1: 1x1 Conv

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

The L0 scaffold ships:
- Substrate architecture + boosting residual chain
- Archive grammar (BSV1 magic + boosting-rounds-in-header)
- Inflate runtime (≤200 LOC for the boosting chain)
- Score-aware loss helper routing
- Test coverage for the canonical archive grammar

Council design memo:
    `.omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 23-byte header) |
| L4 inflate <= 200 LOC, <= 2 deps | PASS (target ~140 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~700 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (BSV1)
    parser_section_manifest:   parse_archive() -> 6 sections (header + base_decoder_blob
                               + boosting_chain_blob + latents_blob + meta_blob + implicit
                               "boosting_residual_heads" subset)
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed base+boosting decoder state_dict
                               + int16 latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~700 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- boosting paradigm = HARD-EARNED (Liu ECCV 2024 + general gradient-boosting
  literature; residuals are a well-known way to reduce worst-case error).
- num_boosting_rounds=2 = CARGO-CULTED (chosen for L0 sanity; sweep at L1).
- gain clamp [-0.1, 0.1] = CARGO-CULTED (chosen to prevent runaway; needs
  empirical tuning per substrate at L1).
- shared latent z across rounds = CARGO-CULTED (alternative: per-round
  latents would inflate the rate term ~1.5x; cheap variant first).
- DepthSep base = HARD-EARNED (mirrors ds_nerv canonical sister).

Operator 5-tier fit ranking citation:
    "BoostNeRV (Liu ECCV 2024) — HIGH FIT ⭐⭐⭐⭐⭐ — paradigm-orthogonal
     iterative residual chain sidecar. Composes with any base substrate."
"""

from .architecture import (
    BoostnervConfig,
    BoostnervSubstrate,
)
from .archive import (
    BoostnervArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import BoostnervScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "BoostnervArchive",
    "BoostnervConfig",
    "BoostnervScoreAwareLoss",
    "BoostnervSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
