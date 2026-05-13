"""tac.substrates.vq_vae — VQ-VAE persistent-codebook substrate (L0 SKETCH).

Per operator approval 2026-05-12 ("3 non-NeRV substrate L0 SKETCHes — vq_vae,
siren, grayscale_lut"). The Vector-Quantized Variational AutoEncoder family
(van den Oord, Vinyals, Kavukcuoglu, 2017; "Neural Discrete Representation
Learning") replaces continuous latents with a **discrete codebook** of K
embeddings. The encoder produces a per-frame spatial grid; each grid cell is
quantized to its nearest codebook entry; the decoder consumes the index grid.
At L0 SKETCH the codebook is updated via the straight-through estimator
(Bengio 2013) only; van den Oord's persistent N_c/m_c EMA buffer form
(decay=0.99 per CLAUDE.md "EMA — non-negotiable" codebook exception) is
DEFERRED-pending-substrate-engineering per R4 finding Z-8.1 (2026-05-13).
The ``codebook_ema_decay`` config field is reserved for the future buffer
implementation; the current architecture does NOT register ``ema_cluster_size``
/ ``ema_w`` buffers.

The substrate is score-aware: gradients flow through the straight-through
estimator (Bengio 2013) of the codebook quantizer, then through the decoder,
then through the differentiable eval-roundtrip + patched yuv6, then into the
contest scorers (PoseNet/SegNet). Rate is paid via per-cell codebook indices
(log2(K) bits/cell) + the encoder/decoder/codebook state-dict bytes.

L0 SKETCH lane registration (research_only=true per CLAUDE.md HNeRV parity
discipline opt-out — substrate engineering, not contest-ready yet):

    python tools/lane_maturity.py add-lane lane_substrate_vq_vae_20260512 \\
        --name "VQ-VAE persistent-codebook substrate (L0 SKETCH)" --phase 2 \\
        --notes "research_only=true; substrate_engineering exception per HNeRV L7"

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar VQV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~85 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; decoder outputs RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~540 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned + scaffolded) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:            monolithic single-file 0.bin VQV1 fixed offsets
    parser_section_manifest:    parse_archive() -> (codebook, decoder_sd, indices, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:        torch, brotli
    export_format:              brotli(state_dict + codebook) + packed-int16 indices + utf8-json meta
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         ~540 LOC (substrate_engineering tag)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha (sane_hnerv) empirical anchor at <= 0.21 AND post-anchor diagnostic
    flags rate-axis headroom (>= 5%) AND discrete-token representation is a
    candidate (vs continuous latents) — then VQ-VAE becomes a target for the
    discrete-rate-axis experiment.

Distinguishing feature vs sane_hnerv:
    VQ-VAE allocates parameters to a SHARED CODEBOOK of K discrete embeddings
    (e.g., K=512, D=8 -> 4K params) + a tiny encoder + a tiny decoder. Per
    frame the encoder produces an index grid (H/8 x W/8); rate is paid in
    log2(K) bits/cell rather than 16 bits/float-latent. This is the discrete
    counterpart to cool_chic's continuous-AR latents and represents van den
    Oord's seat in the grand council.

van den Oord grand-council seat:
    "VQ-VAE, WaveNet; practical neural compression + generative modeling;
    conceptual sibling of SegMap (discrete tokens for images)" per
    CLAUDE.md grand-council roster.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module
- No /tmp paths
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
"""

from .architecture import (
    VqVaeConfig,
    VqVaeSubstrate,
)
from .archive import (
    VqVaeArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    ScoreAwareLossWeights,
    VqVaeScoreAwareLoss,
)

__all__ = [
    "ScoreAwareLossWeights",
    "VqVaeArchive",
    "VqVaeConfig",
    "VqVaeScoreAwareLoss",
    "VqVaeSubstrate",
    "pack_archive",
    "parse_archive",
]
