"""tac.substrates.hybrid_renderer_residual — γ Hybrid renderer + residual basis (L0 SKETCH).

The Fields-medal grand council 2026-05-12 DEFERRED-pending-criterion scaffold per
the "Multiple contenders → multiple paths" non-negotiable. The α substrate
``sane_hnerv`` shipped PRIMARY (10/10); the β substrate ``balle_renderer``
shipped PARALLEL (6/10); γ + δ ship as L0 SKETCH scaffolds with explicit
reactivation criteria (NO KILL per CLAUDE.md "KILL is LAST RESORT").

Council design memo:
    .omx/research/grand_council_fields_medal_substrate_design_20260512.md

Council γ design position (from §2.2 + §4.1 + §4.2 + §8):

    Dykstra CO-LEAD: "Combined: build a substrate that has hyperprior side-info
    AND score-aware pose-residual stream AND a re-architected renderer with
    lower R*. Single-axis stacking is dominated."

    γ-candidate: "Hybrid renderer-with-residual-basis (HNeRV-class renderer +
    score-aware sparse residual coefficient stream)". The renderer produces
    RGB; the residual stream corrects per-frame for score-affecting deltas
    (especially pose-axis at the 2.71× PR106 r2 operating point).

The substrate IS a renderer (full RGB; HNeRV parity lesson L5). The residual
basis is a per-pair sparse coefficient stream over a learned dictionary; the
inflate path adds (renderer_out + residual_decoder(coeffs)) → frame.

13 HNeRV parity-discipline lessons compliance — design-time declaration
(council §4.2 declared γ has L4 NEEDS-WORK and L7 NEEDS-WORK at design time):

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (Lagrangian wires SegNet/PoseNet on RGB sum) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar; HRRV1 magic) |
| L4 inflate <= 100 LOC, <= 2 deps | WAIVED <= 200 LOC (council §4.2 NEEDS-WORK note:
|                                  | residual decode adds ~30 LOC over α's 80 LOC) |
| L5 full RGB renderer | PASS (NOT a mask codec; renderer + residual sum to RGB) |
| L6 score-domain Lagrangian | PASS (alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + lambda_res*||c||_1) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~620 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (wired via tac.differentiable_eval_roundtrip) |
| L9 runtime closure | PASS (torch + brotli; minimal residual decoder, no extra deps) |
| L10 mask/pose coupling | N/A (renderer replaces full slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke planned + tested) |
| L12 single-LOC review discipline | PASS (architecture ≤ 300 LOC; inflate ≤ 200 LOC) |
| L13 KILL last resort | PASS (DEFERRED-pending-research per CLAUDE.md) |

Catalog #124 archive-grammar 8 fields (declared at design-time):
    archive_grammar:           monolithic_single_file_0.bin_with_residual_section
                               (MAGIC|VER|LDIM|RBASIS|NPAIRS|DECODER|RESDEC|
                                LATENTS|RESCOEFFS|META — 27-byte header + 7 sections)
    parser_section_manifest:   parse_archive() returns HybridRendererResidualArchive
                               with (renderer_state_dict, residual_decoder_state_dict,
                                     latents, residual_basis_coefficients, meta)
    inflate_runtime_loc_budget: <= 200 LOC (waiver — residual decode adds ~30 LOC);
                               target ~140 LOC
    runtime_dep_closure:       torch, brotli (residual decoder is in-file; NO
                               extra deps beyond α's α-set)
    export_format:             brotli-compressed state_dicts + int16 latents
                               + int16 residual coefficients + utf-8 JSON meta
    score_aware_loss:          L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
                                 + λ_res·||c||_1   (sparse residual coeff prior)
                               where d_seg + d_pose come from
                               tac.differentiable_eval_roundtrip on
                               (renderer_out + residual_decoder(coeffs))
    bolt_on_loc_budget:        substrate ~260 + archive ~180 + inflate ~150
                               + loss ~110 = ~700 LOC
                               (lane_class=substrate_engineering per L7)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof +
                               executable byte-mutation smoke in
                               tests/test_hybrid_renderer_residual_roundtrip.py
                               (mutates renderer latents AND residual coeffs)

Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT"; council §8):
    Reactivate IFF α first-anchor (Wave 3) score ≤ 0.21 [contest-CUDA]
    AND pose-marginal > seg-marginal still holds at that operating point
    (i.e., the residual-basis bolt-on has highest empirical EV given α's
    measured weakness). Until both conditions are met this substrate
    stays L0 SKETCH with research_only=true.

This module IS an L0 SKETCH landing. The training ``_full_main`` entry-point
is NOT wired here; it ships in a separate post-α-anchor follow-up subagent.
"""

from .archive import (
    HybridRendererResidualArchive,
    pack_archive,
    parse_archive,
)
from .architecture import (
    HybridRendererResidualConfig,
    HybridRendererResidualSubstrate,
)
from .score_aware_loss import (
    HybridRendererResidualScoreAwareLoss,
    HybridResidualScoreAwareLossWeights,
)

__all__ = [
    "HybridRendererResidualArchive",
    "HybridRendererResidualSubstrate",
    "HybridRendererResidualConfig",
    "HybridRendererResidualScoreAwareLoss",
    "HybridResidualScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
