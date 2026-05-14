# SPDX-License-Identifier: MIT
"""tac.substrates.wavelet — 2D DWT + score-aware subband coding substrate (L0 SKETCH).

Per operator approval 2026-05-12 ("land 2 non-NeRV substrate L0 SKETCHes"). The
wavelet family (Mallat 1989; Daubechies 1992) takes per-frame RGB latents,
performs a 2D discrete wavelet transform (DWT) using Daubechies-4 filters, and
codes coefficients by-subband (Mallat hierarchy). The substrate is score-aware
via a small learned synthesis MLP that consumes IDWT-reconstructed coefficients
plus a frame-conditional FiLM modulation, before producing RGB output.

L0 SKETCH lane registration (research_only=true per CLAUDE.md HNeRV parity
discipline opt-out — substrate engineering, not contest-ready yet):

    python tools/lane_maturity.py add-lane lane_substrate_wavelet_20260512 \
        --name "DWT + score-aware subband substrate (L0 SKETCH)" --phase 2 \
        --notes "research_only=true; substrate_engineering exception per HNeRV L7"

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar WLV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~90 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; synthesis MLP outputs RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~560 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned + scaffolded) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:            monolithic single-file 0.bin WLV1 fixed offsets
    parser_section_manifest:    parse_archive() -> (synthesis_sd, film_sd, LL, LH, HL, HH, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:        torch, brotli
    export_format:              brotli(state_dicts) + int16(LL/LH/HL/HH subbands) + utf8-json meta
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         ~560 LOC (substrate_engineering tag)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha (sane_hnerv) empirical anchor at <= 0.21 AND post-anchor diagnostic
    flags pose-axis underperformance OR rate-axis headroom (>= 5%) — then
    Wavelet becomes a candidate for the multiresolution-rate-axis experiment.

Distinguishing feature vs sane_hnerv:
    Wavelet substrate spends parameters on **4 subband coefficient tensors**
    (LL = approximation, LH = horizontal detail, HL = vertical detail, HH =
    diagonal detail) plus a tiny synthesis MLP + FiLM conditioner. The 4
    subbands form the Mallat hierarchy at depth-1 of a Daubechies-4 DWT:

        LL: (num_pairs, C, H/2, W/2)  — low-pass, dominates rate
        LH, HL, HH: (num_pairs, C, H/2, W/2)  — detail subbands

    Coding by-subband with separate quantization scales (per Mallat) lets the
    score-aware loss allocate bits where they matter for PoseNet/SegNet, e.g.,
    high-frequency detail bands get coarser quantization than LL.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module
- No /tmp paths
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
"""

from .archive import (
    WaveletArchive,
    pack_archive,
    parse_archive,
)
from .architecture import (
    WaveletConfig,
    WaveletSubstrate,
)
from .score_aware_loss import (
    ScoreAwareLossWeights,
    WaveletScoreAwareLoss,
)

__all__ = [
    "WaveletArchive",
    "WaveletSubstrate",
    "WaveletConfig",
    "WaveletScoreAwareLoss",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
