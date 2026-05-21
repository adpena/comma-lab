# SPDX-License-Identifier: MIT
"""tac.substrates.grayscale_lut — Selfcomp analog grayscale-LUT substrate (L0 SKETCH).

Per operator approval 2026-05-12 ("3 non-NeRV substrate L0 SKETCHes — vq_vae,
siren, grayscale_lut"). The grayscale-LUT family (Selfcomp / szabolcs-cs's
PR #56 paradigm) views an AV1 grayscale + Gaussian-LUT representation as a
true ANALOG-signal codec: a per-pair grayscale (single luminance channel) is
codec-compressed (AV1, JPEG, or simply quantized), and a learned colorization
LUT (or a tiny FiLM-conditioned RGB decoder) maps grayscale -> RGB at inflate
time. The substrate is score-aware: the LUT/decoder is trained end-to-end
through the contest scorers.

This represents Selfcomp's seat in the grand council: "architect of the
grayscale-LUT analog mask paradigm + 1.017-bpw block-FP weight
self-compression + 94K-param SegMap; PR #56's lead implementer".

The substrate is score-aware: gradients flow from contest scorers through the
LUT/colorization decoder to weights, via the differentiable eval-roundtrip +
patched yuv6. Rate is paid via (a) the grayscale-stream bytes (quantized via
int8 + brotli), (b) the LUT/decoder state-dict bytes.

L0 SKETCH lane registration (research_only=true per CLAUDE.md HNeRV parity
discipline opt-out — substrate engineering, not contest-ready yet):

    python tools/lane_maturity.py add-lane lane_substrate_grayscale_lut_20260512 \\
        --name "Grayscale-LUT analog substrate (L0 SKETCH; Selfcomp paradigm)" --phase 2 \\
        --notes "research_only=true; substrate_engineering exception per HNeRV L7"

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar GLV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~95 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; LUT produces RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~540 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned + scaffolded) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:            monolithic single-file 0.bin GLV1 fixed offsets
    parser_section_manifest:    parse_archive() -> (decoder_sd, film_sd, grayscale_int8, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:        torch, brotli
    export_format:              brotli(state_dicts) + int8(grayscale) + utf8-json meta
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         ~540 LOC (substrate_engineering tag)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha (sane_hnerv) empirical anchor at <= 0.21 AND post-anchor diagnostic
    flags either (a) rate-axis headroom (>= 5%) — analog grayscale streams are
    highly compressible and may be the cheapest-per-byte path, OR (b)
    pose-axis underperformance — the LUT can be FiLM-conditioned to spend
    pose-relevant bits.

Distinguishing feature vs sane_hnerv:
    Grayscale-LUT factors the per-pair information into:
      - A 1-channel grayscale field (~600 frames * H/4 * W/4 * 1B int8 + brotli),
        which is the dominant rate term and exploits AV1-like analog compression.
      - A tiny FiLM-conditioned RGB decoder (~94K params per Selfcomp's anchor)
        that maps grayscale + per-pair embedding -> RGB.
    The grayscale field is bilinear-upsampled at inflate time. This is the
    pure-analog counterpart to vq_vae's discrete codebook and represents
    Selfcomp's PR #56 paradigm.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module
- No /tmp paths
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
"""

from .architecture import (
    GrayscaleLutConfig,
    GrayscaleLutSubstrate,
)
from .archive import (
    GrayscaleLutArchive,
    compose_procedural_archive,
    pack_archive,
    parse_archive,
)
from .distillation_procedural_variant import (
    CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT,
    PROCEDURAL_LUT_BYTES_DEFAULT,
    PROCEDURAL_LUT_DTYPE_DEFAULT,
    PROCEDURAL_LUT_SENTINEL,
    PROCEDURAL_SEED_SIZE_BYTES,
    ProceduralVariantConfig,
    ProceduralVariantError,
    compose_with_procedural_lut,
    derive_procedural_lut_replacement,
    predicted_archive_bytes_saved,
    predicted_delta_s,
    verify_procedural_lut_in_domain,
    verify_seed_mutation_changes_lut_bytes,
)
from .score_aware_loss import (
    GrayscaleLutScoreAwareLoss,
    ScoreAwareLossWeights,
)

PROCEDURAL_VARIANT_AVAILABLE: bool = True
"""Flag set True at scaffold landing (sister of DP1 + VQ-VAE
``PROCEDURAL_VARIANT_AVAILABLE``). Trainers + cathedral consumers may key
off this flag to detect that the grayscale_lut procedural chroma-LUT
replacement variant is importable + structurally complete.

Per WAVE-3-GRAYSCALE-LUT-PROCEDURAL-TRAINER-BUILD 2026-05-20 + PR101/PR106
BUILD DESIGN landing commit ``086d3ac1d`` Top-3 #1 PIVOT.
"""


__all__ = [
    "CANONICAL_EQUATION_26_IN_DOMAIN_CONTEXT",
    "GrayscaleLutArchive",
    "GrayscaleLutConfig",
    "GrayscaleLutScoreAwareLoss",
    "GrayscaleLutSubstrate",
    "PROCEDURAL_LUT_BYTES_DEFAULT",
    "PROCEDURAL_LUT_DTYPE_DEFAULT",
    "PROCEDURAL_LUT_SENTINEL",
    "PROCEDURAL_SEED_SIZE_BYTES",
    "PROCEDURAL_VARIANT_AVAILABLE",
    "ProceduralVariantConfig",
    "ProceduralVariantError",
    "ScoreAwareLossWeights",
    "compose_procedural_archive",
    "compose_with_procedural_lut",
    "derive_procedural_lut_replacement",
    "pack_archive",
    "parse_archive",
    "predicted_archive_bytes_saved",
    "predicted_delta_s",
    "verify_procedural_lut_in_domain",
    "verify_seed_mutation_changes_lut_bytes",
]
