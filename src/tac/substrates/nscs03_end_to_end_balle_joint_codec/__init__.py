# SPDX-License-Identifier: MIT
"""tac.substrates.nscs03_end_to_end_balle_joint_codec — NSCS03 end-to-end Ballé
joint codec (per assumptions-challenge-audit NSCS03).

**Distinguishing feature** vs. sister `balle_renderer` (β) and Z3 v2 (Ballé bolt-on):

* **balle_renderer (β)**: per-pair flat learned latents `nn.Parameter(num_pairs,
  latent_dim)`; hyperprior is an MLP (not conv); decoder is MLP+upsample to
  RGB. The latent IS the trained substrate (renderer paradigm).
* **Z3 v2**: Ballé hyperprior as a BOLT-ON over A1's existing latent stream
  (within-class refinement; rate-axis only).
* **NSCS03 (end-to-end joint codec)**: TRUE Ballé 2018 paradigm — convolutional
  ANALYSIS transform `g_a` on the per-pair pixel input, factorized-prior
  ENTROPY BOTTLENECK on the hyper-latent z, conditional-Gaussian density on
  the main latent y given σ=h_s(z), convolutional SYNTHESIS transform `g_s`
  back to pixels. Score-aware loss backpropagates THROUGH the bottleneck +
  THROUGH yuv6 + THROUGH SegNet/PoseNet ALL THE WAY to the analysis transform.
  The encoder weights, decoder weights, hyper-analysis weights, hyper-synthesis
  weights, AND entropy-bottleneck parameters are ALL trained jointly (NOT a
  frozen renderer + tunable rate proxy as in balle_renderer).

This is the canonical Ballé et al. 2018 ICLR architecture, applied as the
contest substrate. The differentiable rate term `R = -log2(p_y(y_hat)) +
-log2(p_z(z_hat))` is the intended pressure tying training to packed bytes.
The current scaffold has not yet proven proxy-rate/packed-byte/inflate/scorer
closure, so it must not claim the train-to-archive roundtrip drift is closed.

**Score movement**: unranked until a closure test compares differentiable rate
proxy, actual packed bytes, inflate output, and scorer roundtrip on the same
candidate archive.

13 HNeRV parity-discipline lessons compliance — design-time declaration:

| Lesson | Status |
|---|---|
| L1 substrate must be score-aware | PASS (joint loss wires SegNet/PoseNet through bottleneck) |
| L2 export-first archive grammar | PASS (archive.py declared BEFORE training) |
| L3 monolithic 0.bin | PASS (single-file fixed-offset grammar) |
| L4 inflate <= 100 LOC, <= 2 deps | WAIVED <= 200 LOC (per council §4.2 NEEDS-WORK; explicit tag) |
| L5 full RGB renderer | PASS (joint codec = encoder+decoder; pure RGB IO) |
| L6 score-domain Lagrangian | PASS (B/N + d_seg + sqrt(d_pose) + λ_R·R(y,z)) |
| L7 bolt-on <= 350 LOC | substrate_engineering exception (~1200 total) |
| L8 eval-roundtrip + diff yuv6 | PASS (apply_eval_roundtrip + patch_upstream_yuv6_globally) |
| L9 runtime closure | PASS (torch + brotli; reuses tac.entropy_bottleneck for math correctness) |
| L10 mask/pose coupling | N/A (full RGB renderer slot) |
| L11 no-op detector | PASS (executable byte-mutation smoke planned in tests) |
| L12 single-LOC review discipline | PASS (architecture < 600 LOC; inflate <= 200 LOC) |
| L13 KILL last resort | PASS (DEFERRED-pending-research per CLAUDE.md) |

Catalog #124 archive-grammar 8 fields:
    archive_grammar:           monolithic_0.bin_with_main_and_hyper_latents
                               (MAGIC|VER|ARCH|MAIN_LATENT|HYPER_LATENT|META)
    parser_section_manifest:   parse_archive() returns NSCS03Archive with
                               (encoder_sd, decoder_sd, hyper_analysis_sd,
                                hyper_synthesis_sd, entropy_state_sd,
                                main_latents, hyper_latents, meta)
    inflate_runtime_loc_budget: <= 200 LOC (waiver — convolutional decoder + GDN at inflate);
                               target ~180 LOC
    runtime_dep_closure:       torch, brotli (reuses tac.entropy_bottleneck for math);
                               NO CompressAI runtime dep
    export_format:             brotli-compressed state_dicts (encoder/decoder/h_a/h_s/entropy)
                               + raw int16 main+hyper latents + sidecar JSON meta
    score_aware_loss:          L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ))
                               + λ_R·(R_main(y, σ) + R_hyper(z))
                               where the rate term is END-TO-END differentiable
                               via STE through quantization
    bolt_on_loc_budget:        substrate ~600 + archive ~250 + inflate ~200
                               + loss ~200 = ~1250 LOC
                               (lane_class=substrate_engineering)
    no_op_detector_planned:    Catalog #139 _build_no_op_proof + executable
                               byte-mutation smoke in tests/test_nscs03_roundtrip.py

Dispatch status:
    research_only=true until a non-smoke trainer, operator recipe, remote
    driver, byte-closed archive export, and paired CPU+CUDA auth-eval custody
    all exist. This package is a design/runtime substrate scaffold, not a
    contest-ready dispatch target.

Reactivation criteria (per CLAUDE.md "KILL is LAST RESORT"):
    - operator approves NSCS03 _full_main follow-up subagent
    - sister NSCS01/NSCS02 produce first empirical anchor (parity reference)
    - smoke-before-full Modal A100 dispatch passes ($0.30 → $40-120 full)
    - if smoke regresses past +0.005 score: defer pending Ballé hyperprior-on-PR106-substrate
      empirical anchor (per assumptions audit NSCS03 reactivation_criteria_if_smoke_regresses)
"""

from .architecture import (
    NSCS03Config,
    NSCS03JointCodecSubstrate,
)
from .archive import (
    NSCS03Archive,
    pack_archive,
    parse_archive,
)
from .score_aware_loss import (
    NSCS03JointScoreAwareLoss,
    NSCS03ScoreAwareLossWeights,
)

__all__ = [
    "NSCS03Archive",
    "NSCS03Config",
    "NSCS03JointCodecSubstrate",
    "NSCS03JointScoreAwareLoss",
    "NSCS03ScoreAwareLossWeights",
    "pack_archive",
    "parse_archive",
]
