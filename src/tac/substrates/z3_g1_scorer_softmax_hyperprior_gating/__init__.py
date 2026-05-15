# SPDX-License-Identifier: MIT
"""Z3-G1 scorer-softmax-hyperprior-gating bolt-on substrate package.

Per Wunderkind G1 SUBSTITUTION-1:1 spec
(`feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md`):
G1 REPLACES Z3 v2's per-pair Ballé hyperprior MLP (~50KB before brotli; ~1080B
in the production direct-residual mode) with a SegNet-class-conditional sigma
TABLE (~560B fp32; ~140B int8 brotli'd) + a per-pair class index (600B at
int8). Decoder reads the table + indices, reconstructs sigma per pair without
running SegNet at inflate time (Catalog #6 strict-scorer-rule preserved).

The G1 substitution is contest-legal at the compress-side only:
- COMPRESS-SIDE: SegNet runs on GT frame to compute per-pair dominant class
  (mode of per-pixel argmax). FREE per CLAUDE.md "Strict scorer rule"
  (rule #2: scorer use during compression is FREE). Class index per pair
  is shipped in the Z3HG1 sidecar.
- INFLATE-SIDE: NO scorer. Decoder reads the per-pair class index + the
  tiny sigma table (≤1KB total) and reconstructs the conditional Gaussian
  prior per pair. Then runs the same arithmetic coder as Z3 v2 to decode
  the residual.

Architecture (1KB-budget hyperprior gating):

    For each pair p in range(600):
        class_p = mode of SegNet(GT_frame_p).argmax(1)  per pixel  (in [0, 4])
        sigma_p = sigma_table[class_p, :]   (28-dim per-dim scale)
        AC-encode residual_p under N(0, sigma_p^2)

    Archive payload:
        - per-class sigma table (5 * 28 * float32 = 560B; quantize to int8
          ~140B + brotli ~ 80B)
        - per-pair class index (600 * uint8 = 600B; brotli ~ 200-400B)
        - AC-coded residual (600 * 28 int8 ~ 16800B; brotli ~ 800-1200B)

Predicted ΔS = -0.005 to -0.015 vs A1 0.1928 [contest-CPU 1to1]
``[prediction; first-principles-bound]``.

The implementation REUSES Z3 v2's archive_v2 wire grammar (Z3HV2 magic, header,
length-prefixed brotli blobs, per-dim affine) so the inflate runtime can be
shared without code duplication. The Z3HV2 "weights" slot is repurposed for
the int8 sigma TABLE (small enough to fit in uint16 length prefix); the
"w_hat" slot is repurposed for the per-pair class index.

NO score claim. NO promotion. NO exact-eval dispatch from this module.
Tagged ``research_only=true`` until empirical smoke + full-run anchors land.

Bolt-on LOC budget: <= 350 LOC per HNeRV parity discipline L7.
"""
from __future__ import annotations

from tac.substrates.z3_g1_scorer_softmax_hyperprior_gating.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G1Config,
    Z3G1ScorerClassGatingHead,
    g1_per_pair_dominant_class_from_segnet_argmax,
    g1_total_rate_bits,
)
from tac.substrates.z3_g1_scorer_softmax_hyperprior_gating.score_aware_loss import (
    estimate_z3g1_section_overhead_bytes,
    g1_residual_rate_bits_per_sample,
    z3_g1_lagrangian,
)

__all__ = [
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "G1_NUM_SCORER_CLASSES",
    "Z3G1Config",
    "Z3G1ScorerClassGatingHead",
    "estimate_z3g1_section_overhead_bytes",
    "g1_per_pair_dominant_class_from_segnet_argmax",
    "g1_residual_rate_bits_per_sample",
    "g1_total_rate_bits",
    "z3_g1_lagrangian",
]
