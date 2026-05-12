"""tac.substrates.siren — Sinusoidal-init MLP coordinate-based substrate (L0 SKETCH).

Per operator approval 2026-05-12 ("3 non-NeRV substrate L0 SKETCHes — vq_vae,
siren, grayscale_lut"). SIREN (Sitzmann, Martel, Bergman, Lindell, Wetzstein,
NeurIPS 2020; "Implicit Neural Representations with Periodic Activation
Functions") is a coordinate-based MLP with **sin activations** and a special
initialization scheme (Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
with first-layer omega=30). It maps a continuous spatial-temporal coordinate
``(x, y, pair_idx)`` directly to RGB output, with NO per-frame latent state.

This is the **purely-coordinate-based** counterpart to NeRV/HNeRV (which have
implicit per-frame latents) and Cool-Chic (which has explicit per-frame
latents). All variation across frames is encoded in the network weights
themselves; the pair index is a frequency-encoded scalar input. This makes
SIREN a true ANALYTICAL representation per Hotz's "champion analytical
shortcuts over learned complexity" charter.

The substrate is score-aware: gradients flow from contest scorers through
sin-activation MLP to weights, via the differentiable eval-roundtrip + patched
yuv6. Rate is paid via the network state-dict bytes (no separate latents).

L0 SKETCH lane registration (research_only=true per CLAUDE.md HNeRV parity
discipline opt-out — substrate engineering, not contest-ready yet):

    python tools/lane_maturity.py add-lane lane_substrate_siren_20260512 \\
        --name "SIREN coordinate-based substrate (L0 SKETCH)" --phase 2 \\
        --notes "research_only=true; substrate_engineering exception per HNeRV L7"

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (score-aware Lagrangian wired) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar SRV1) |
| L4 inflate <= 100 LOC, <= 2 deps | PASS (target ~80 LOC; torch + brotli) |
| L5 full RGB renderer | PASS (NOT a mask codec; MLP outputs RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~520 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; numpy is torch transitive) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (Catalog #139 byte-mutation smoke planned + scaffolded) |
| L12 single-LOC review discipline | PASS (each file reviewable in 30s) |
| L13 KILL last resort | PASS (DEFERRED-pending-alpha-anchor reactivation path) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:            monolithic single-file 0.bin SRV1 fixed offsets
    parser_section_manifest:    parse_archive() -> (mlp_sd, meta)
    inflate_runtime_loc_budget: <= 100 LOC
    runtime_dep_closure:        torch, brotli
    export_format:              brotli(state_dict, fp16 cpu) + utf8-json meta
    score_aware_loss:           alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)
    bolt_on_loc_budget:         ~520 LOC (substrate_engineering tag)
    no_op_detector_planned:     Catalog #139 _build_no_op_proof + byte-mutation smoke

Reactivation criteria (L0 -> L1):
    Alpha (sane_hnerv) empirical anchor at <= 0.21 AND post-anchor diagnostic
    flags rate-axis headroom (>= 5%) AND analytical coordinate representation
    is a candidate (vs latent-based) — then SIREN becomes a candidate for the
    no-latent rate-axis experiment.

Distinguishing feature vs sane_hnerv:
    SIREN puts ALL ~150K params into the MLP weights; ZERO bytes go to
    latents. The rate term is entirely dominated by the network bytes (which
    are highly compressible via FP4/8-bit quantization + brotli). Per-frame
    variation comes from a frequency-encoded pair-index input. The first-layer
    omega=30 is the Sitzmann signature; downstream omega=1.0 is the standard
    SIREN choice.

CLAUDE.md compliance:
- No silent device defaults (caller passes device explicitly)
- No scorer loading inside this module
- No /tmp paths
- No KILL verdicts in the scaffold (DEFER-pending-anchor only)
"""

from .architecture import (
    SirenConfig,
    SirenSubstrate,
)
from .archive import (
    SirenArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    ScoreAwareLossWeights,
    SirenScoreAwareLoss,
)

__all__ = [
    "ScoreAwareLossWeights",
    "SirenArchive",
    "SirenConfig",
    "SirenScoreAwareLoss",
    "SirenSubstrate",
    "pack_archive",
    "parse_archive",
]
