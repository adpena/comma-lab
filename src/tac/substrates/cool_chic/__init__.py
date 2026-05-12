"""tac.substrates.cool_chic — Cool-Chic per-frame latent + AR prior substrate (L0 SKETCH).

Per operator approval 2026-05-12 ("land 2 non-NeRV substrate L0 SKETCHes"). The
Cool-Chic family (Ladune et al., 2023; "Cool-chic: Coordinate-based Low Complexity
Hierarchical Image Codec") replaces per-frame implicit-renderer parameters with
per-frame **learnable latents** that an **autoregressive (AR) prior** conditions
on the previous-frame latent. Rate is paid via the AR density estimate; pixels
are produced by a tiny shared synthesis MLP.

L0 SKETCH lane registration (research_only=true per CLAUDE.md HNeRV parity
discipline opt-out — substrate engineering, not contest-ready yet):

    python tools/lane_maturity.py add-lane lane_substrate_cool_chic_20260512 \
        --name "Cool-Chic per-frame AR latent substrate (L0 SKETCH)" --phase 2 \
        --notes "research_only=true; substrate_engineering exception per HNeRV L7"

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar CCV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~85 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; synthesis MLP outputs RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~530 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned + scaffolded) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:            monolithic single-file 0.bin CCV1 fixed offsets
    parser_section_manifest:    parse_archive() -> (synthesis_sd, ar_prior_sd, latents, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:        torch, brotli
    export_format:              brotli-compressed state_dicts + int16 latents + utf8-json meta
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         ~530 LOC (substrate_engineering tag)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha (sane_hnerv) empirical anchor at <= 0.21 AND post-anchor diagnostic
    flags pose-axis underperformance OR rate-axis headroom (>= 5%) — then
    Cool-Chic becomes a candidate for the per-pair-rate experiment.

Distinguishing feature vs sane_hnerv:
    Cool-Chic spends almost ALL its parameter budget on a per-frame **latent
    grid** (e.g., 2 small spatial pyramids at H/8 × W/8 + H/16 × W/16). The
    synthesis network is a TINY shared MLP (~10K params). The AR prior is a
    similarly tiny conditional density model. Total params ~200K; the bytes
    flow into the LATENT_BLOB, not the decoder.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module
- No /tmp paths
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
"""

from .archive import (
    CoolChicArchive,
    pack_archive,
    parse_archive,
)
from .architecture import (
    CoolChicConfig,
    CoolChicSubstrate,
)
from .score_aware_loss import (
    CoolChicScoreAwareLoss,
    ScoreAwareLossWeights,
)

__all__ = [
    "CoolChicArchive",
    "CoolChicSubstrate",
    "CoolChicConfig",
    "CoolChicScoreAwareLoss",
    "ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
