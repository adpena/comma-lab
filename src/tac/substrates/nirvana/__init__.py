# SPDX-License-Identifier: MIT
"""tac.substrates.nirvana — NIRVANA (substrate L0 SKETCH).

Per-patch implicit renderer with adaptive per-patch epoch scheduling.
Operator 5-tier fit-ranking verdict **MODERATE-HIGH FIT ⭐⭐⭐⭐**: patch-wise
specialization stacks orthogonally with global NeRV substrates; the adaptive
scheduler allocates more training steps to high-error patches.

Literature anchor: Maiya et al. CVPR 2024 "NIRVANA: Neural Implicit
Representations of Videos with Adaptive Networks and Autoregressive
patch-wise Modeling" (arXiv:2212.14593 — canonical literature reference
per BUILD task #1090). Canonical OSS repo (when available):
github.com/<author>/NIRVANA (literature anchor; not vendored).

Hypothesis (per operator's per-variant fit verdict): driving video has
non-uniform spatial complexity (road texture is high-frequency; sky is
low-frequency). A per-patch decoder + adaptive scheduler that gives more
training compute to the hardest patches yields better PSNR per parameter
than a global decoder forced to fit everything uniformly.

Architecture (council-approved SKETCH 2026-05-20):

    Per-pair latent z in R^16
       |
       +-- Patch grid (PATCH_GRID_H x PATCH_GRID_W, default 4x4 = 16 patches)
       |
       v
    Per-patch: PatchDecoder(z, patch_index_embedding)
         |          (small DepthSep decoder per-patch slot;
         v           output (H/PATCH_GRID_H, W/PATCH_GRID_W))
    rgb_patches in [0, 1]^(NUM_PATCHES, 3, H_patch, W_patch)
       |
       v
    Spatial assembly: stitch patches into (3, H, W)
       |
       v
    Output: rgb_0 / rgb_1

The adaptive scheduler (training-time only; archive is static):
    Maintain per-patch loss EMA. At each batch, sample patches with
    probability proportional to their loss EMA. High-loss patches get
    more gradient updates per epoch. This is the NIRVANA distinctive
    mechanism; the inflate-time runtime is unaware of it.

Status: **L0 SKETCH** (research_only=true). NO full main, NO dispatch.

The L0 scaffold ships:
- Substrate architecture + patch grid + per-patch decoder
- Archive grammar (NRV1 magic + patch grid dimensions in header)
- Inflate runtime (≤200 LOC for the spatial assembly path)
- Score-aware loss helper routing
- Test coverage for the canonical archive grammar

Council design memo:
    `.omx/research/nirvana_l0_scaffold_design_20260520T184500Z.md`

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PLANNED (score_aware_loss.py wired; trainer at L1) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar, 25-byte header) |
| L4 inflate <= 200 LOC, <= 2 deps | PASS (target ~160 LOC for patch assembly) |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B(theta)/N + d_seg + sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~750 total) |
| L8 eval-roundtrip + diff yuv6 | PLANNED (wired at L1 SCAFFOLD) |
| L9 runtime closure | PASS (torch + brotli; declared) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke in tests) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (losing variants DEFERRED) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic single-file 0.bin fixed offsets (NRV1)
    parser_section_manifest:   parse_archive() -> 6 sections (header + per_patch_decoder_blob
                               + patch_index_embedding_blob + latents_blob + meta_blob + implicit
                               "per_patch_decoder_weights" subset)
    inflate_runtime_loc_budget: <= 200 LOC
    runtime_dep_closure:       torch, brotli
    export_format:             brotli-compressed shared per-patch decoder state_dict
                               + int16 latents + utf8-json meta
    score_aware_loss:          L = alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:        ~750 LOC (substrate_engineering tag)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + byte-mutation smoke

Cargo-cult audit per Catalog #303 (HARD-EARNED vs CARGO-CULTED):
- patch-wise modeling = HARD-EARNED (Maiya CVPR 2024 + general patch-CNN
  literature; per-patch specialization is well-validated for HDR + video).
- adaptive scheduling = HARD-EARNED at training-time but L0 SCAFFOLD does
  NOT ship the adaptive scheduler yet (smoke uses uniform sampling).
- PATCH_GRID_H=PATCH_GRID_W=4 (16 patches) = CARGO-CULTED (chosen for L0
  sanity; 8x8 / 2x2 alternatives need empirical sweep).
- shared per-patch decoder weights = CARGO-CULTED (cheap variant; per-patch
  decoder weights would inflate rate ~16x).
- bilinear spatial assembly = HARD-EARNED (standard for patch-based codecs).
- DepthSep per-patch decoder = HARD-EARNED (mirrors ds_nerv canonical sister).

Operator 5-tier fit ranking citation:
    "NIRVANA (Maiya CVPR 2024) — MODERATE-HIGH FIT ⭐⭐⭐⭐ — patch-wise +
     adaptive scheduling. Stacks orthogonally with global NeRV substrates."
"""

from .architecture import (
    NirvanaConfig,
    NirvanaSubstrate,
)
from .archive import (
    NirvanaArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import NirvanaScoreAwareLoss, ScoreAwareLossWeights

__all__ = [
    "NirvanaArchive",
    "NirvanaConfig",
    "NirvanaScoreAwareLoss",
    "NirvanaSubstrate",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
