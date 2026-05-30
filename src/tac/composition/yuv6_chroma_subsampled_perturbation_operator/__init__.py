# SPDX-License-Identifier: MIT
"""YUV6 chroma-subsampled perturbation operator (Yousfi-blind-spot exploit).

Canonical Fridrich-Yousfi inversion of PoseNet's FastViT-T12 attention
input: PoseNet is trained on the canonical YUV6 transform
``rgb_to_yuv6 -> (2 frames x 6 = 12 channels)`` where the 2 chroma
channels (``U_sub``, ``V_sub``) are 4:2:0 SUBSAMPLED (half-resolution
per ``tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6``).
The chroma-subsampling step averages each 2x2 spatial block into a
single ``(H//2, W//2)`` chroma sample — the canonical Fridrich
"blind-spot" because:

  1. PoseNet's FastViT-T12 attention head averages over its larger
     receptive field for the half-resolution chroma channels (the
     attention's effective receptive field on (U_sub, V_sub) is
     structurally larger than on the 4 luma channels at full
     resolution).
  2. Each (U_sub, V_sub) sample is computed as ``mean(2x2 chroma block)``
     so coordinated perturbations within a 2x2 block of the original RGB
     can produce LARGER changes in the YUV6 luma channels than in the
     subsampled chroma — the per-byte cost-discrimination signal is
     ASYMMETRIC across YUV6 axes.

This operator implements 4 canonical perturbation STRATEGIES that all
target the chroma-subsampled axes specifically (preserving the 4 luma
channels at full resolution while perturbing the 2 subsampled chroma):

  * ``LOCAL_VARIANCE_WEIGHTED`` — perturb chroma cells with high local
    variance preferentially. Yousfi-UNIWARD analog at the chroma-subsampled
    layer (untextured-region perturbations are MORE detectable; high
    chroma-variance regions absorb perturbation better).
  * ``SEGNET_GRADIENT_WEIGHTED`` — weight perturbation by a per-pixel
    SegNet gradient prior (when supplied as a (H//2, W//2) array
    aligned to the chroma grid).
  * ``POSENET_GRADIENT_WEIGHTED_VIA_MAE_V`` — weight perturbation by a
    per-pixel PoseNet gradient prior derived from the PoseNet MAE-V
    surrogate at ``tac.scorer_surrogate.posenet_mae_v`` (Deliverable B
    sister; the surrogate provides the cheap-to-inspect PoseNet response
    surface).
  * ``JOINT_ATICK_REDLICH_LINEAR_COMBINATION`` — linear combination of
    the 3 above strategies per the Atick-Redlich cooperative-receiver
    weighting (sister of canonical equation ``atick_redlich_cooperative_receiver_*``
    if registered; we surface a 3-coefficient blend with the canonical
    default ``(0.33, 0.33, 0.34)`` and let callers tune).

**Predicted ΔS band** (frontier-pursuit horizon class per Catalog #309):
[-0.001, -0.003] DIRECT. This is a foundation operator — sister
combinatorial bolt-ons (PR110-OPT-7-style packet builders consuming the
perturbed YUV6 tensor) extend the predicted ΔS band per substrate-class
composition rather than this operator alone.

**Sister-extension architecture** per CLAUDE.md "PR-or-greater parity
synergy + binding + integration discipline" 2026-05-30: this operator
provides the canonical chroma-subsampled perturbation primitive that
sister bolt-ons (e.g. PR110-OPT-7 inverse-scorer-basis at
``src/tac/composition/pr110_opt_7_fridrich_uniward_inverse_scorer_basis``)
can compose with WITHOUT each rediscovering the canonical 4:2:0
subsampling math.

**Catalog #341 Tier A canonical-routing markers** are emitted in every
returned :class:`ChromaSubsampledPerturbationResult` per CLAUDE.md
"MPS auth eval is NOISE" + Catalog #192: ``predicted_delta_adjustment=0.0``
+ ``promotable=False`` + ``axis_tag="[predicted]"`` until paired-CUDA
RATIFICATION lands per the canonical PR-submission compliance pipeline
per Catalog #370.

**Catalog #356 per-axis AxisDecomposition** support: the operator emits
a structured `predicted_axis_decomposition` field carrying canonical
Provenance per Catalog #323 so Tier-B per-axis ranking consumers can
absorb the prior.

**Slot EEE NO FAKE IMPLEMENTATIONS gate** (per 2026-05-29 honesty
discipline + CLAUDE.md HIGHEST-EMPHASIS non-negotiable 2026-05-30):
this operator genuinely applies the 4:2:0 chroma subsampling math +
genuinely materializes the per-strategy weight maps + genuinely
preserves the 4 luma channels under chroma-only perturbation. Tests
verify (a) the 4 strategies produce DISTINCT perturbation masks (Jaccard
< 1.0 across pairs), (b) the luma-preservation invariant holds, (c) the
chroma-only perturbation actually modifies chroma when materialized
back to RGB.

**Public API**:

* :class:`ChromaPerturbationStrategy` — 4-value enum.
* :class:`ChromaSubsampledPerturbationConfig` — frozen invariants.
* :class:`ChromaSubsampledPerturbationResult` — typed return.
* :func:`apply_chroma_subsampled_perturbation` — canonical helper.
* :func:`compute_chroma_perturbation_weight_map` — per-strategy mask.
* :func:`assert_luma_preservation_invariant` — runtime sanity gate.
"""
from __future__ import annotations

from tac.composition.yuv6_chroma_subsampled_perturbation_operator.operator import (
    ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS,
    BT601_KYR,
    BT601_KYG,
    BT601_KYB,
    BT601_U_DIVISOR,
    BT601_V_DIVISOR,
    DEFAULT_PERTURBATION_MAGNITUDE,
    ChromaPerturbationStrategy,
    ChromaSubsampledPerturbationConfig,
    ChromaSubsampledPerturbationConfigInvalidError,
    ChromaSubsampledPerturbationResult,
    apply_chroma_subsampled_perturbation,
    assert_luma_preservation_invariant,
    compute_chroma_perturbation_weight_map,
    rgb_to_yuv6_numpy,
    yuv6_to_rgb_numpy,
)


__all__ = [
    "ATICK_REDLICH_DEFAULT_BLEND_COEFFICIENTS",
    "BT601_KYR",
    "BT601_KYG",
    "BT601_KYB",
    "BT601_U_DIVISOR",
    "BT601_V_DIVISOR",
    "DEFAULT_PERTURBATION_MAGNITUDE",
    "ChromaPerturbationStrategy",
    "ChromaSubsampledPerturbationConfig",
    "ChromaSubsampledPerturbationConfigInvalidError",
    "ChromaSubsampledPerturbationResult",
    "apply_chroma_subsampled_perturbation",
    "assert_luma_preservation_invariant",
    "compute_chroma_perturbation_weight_map",
    "rgb_to_yuv6_numpy",
    "yuv6_to_rgb_numpy",
]
