# SPDX-License-Identifier: MIT
"""tac.substrates.balle_renderer — Ballé-hyperprior-as-renderer substrate (β).

The Fields-medal grand council 2026-05-12 PARALLEL DEFERRED scaffold per the
"ship both candidates" verdict (10/10 ships α primary; 6/10 ships β in
parallel). The α substrate ``sane_hnerv`` landed at commit ``12c9293a``; this
β substrate reactivates after α produces its first empirical anchor.

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

Council β position (from §4.2 and §2.10):

    Ballé LEAD: "Replace the fixed factorized prior on the latent stream y
    with a learned scale hyperprior σ = h_s(z) where z is an auxiliary
    latent with its own prior p_z(z). Total rate
    R = E[-log p_z(z)] + E[-log p_y(y|σ(z))]. For our 186KB substrate,
    even a conservative 5% saving on a 162KB subset = ~8KB rate-axis
    recovery. Structurally orthogonal to the magic_codec floor B1 hit."

The substrate IS a renderer (full RGB; HNeRV parity lesson 5). The
hyperprior is the rate-axis re-shaper. The combination is the
canonical neural-compression substrate (Ballé 2018 ICLR).

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (Lagrangian wires SegNet/PoseNet) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar) |
| L4 inflate <= 100 LOC, <= 2 deps | WAIVED <= 200 LOC (GDN forward at inflate)
|                                  | per council §4.2 NEEDS-WORK; explicit tag |
| L5 full RGB renderer | PASS (NOT a mask codec) |
| L6 score-domain Lagrangian | PASS (B/N + d_seg + sqrt(d_pose) + hyperprior R) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~680 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; minimal GDN re-impl, no CompressAI dep) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke planned) |
| L12 single-LOC review discipline | PASS (architecture ≤ 350 LOC; inflate ≤ 200 LOC) |
| L13 KILL last resort | PASS (DEFERRED-pending-research per CLAUDE.md) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic_0.bin_with_hyperprior_section
                               (MAGIC|VER|ENC|DEC|HP|LATENTS|SCALES|META)
    parser_section_manifest:   parse_archive() returns BalleRendererArchive
                               with (encoder_sd, decoder_sd, hyperprior_sd,
                               latents, scales, meta)
    inflate_runtime_loc_budget: <= 200 LOC (waiver — GDN forward at inflate);
                               target ~150 LOC
    runtime_dep_closure:       torch, brotli (minimal in-file GDN, NO
                               CompressAI runtime dep)
    export_format:             brotli-compressed state_dicts + raw int16
                               main/hyper latents + sidecar JSON meta.
                               Arithmetic/range coding remains a readiness
                               blocker before exact replacement dispatch.
    score_aware_loss:          L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
                               + λ_hp·R_hyperprior(θ)
                               where R_hyperprior = E[-log p_z(z)] +
                               E[-log p_y(y|σ(z))] per Ballé 2018
    bolt_on_loc_budget:        substrate ~280 + archive ~150 + inflate ~150
                               + loss ~100 = ~680 LOC
                               (lane_class=substrate_engineering)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof +
                               executable byte-mutation smoke in
                               tests/test_balle_renderer_roundtrip.py

Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT"):
    - α (sane_hnerv) produces first empirical anchor
    - operator approves β `_full_main` follow-up subagent
    - Lightning T4 free-tier 16-config Karpathy sweep on α has completed
      and Pareto-feasibility analysis confirms β's hyperprior axis is
      complementary to whatever α explored

This module is the SCAFFOLD/L1 substrate surface. The trainer ``_full_main``
entry point is wired in ``experiments/train_substrate_balle_renderer.py``;
exact replacement dispatch is still blocked on byte-floor coding, a clean
smoke-before-full run, and same-axis exact-eval custody.
"""

from .architecture import (
    BalleRendererConfig,
    BalleRendererSubstrate,
)
from .archive import (
    BalleRendererArchive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    BalleRendererScoreAwareLoss,
    BalleScoreAwareLossWeights,
)

__all__ = [
    "BalleRendererArchive",
    "BalleRendererConfig",
    "BalleRendererScoreAwareLoss",
    "BalleRendererSubstrate",
    "BalleScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
