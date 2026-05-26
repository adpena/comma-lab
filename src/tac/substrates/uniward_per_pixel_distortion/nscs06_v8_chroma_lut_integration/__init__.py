# SPDX-License-Identifier: MIT
"""UNIWARD 7th-order integration INTO NSCS06 v8 chroma LUT entropy-coded sidecar.

Per Catalog #307 paradigm-vs-implementation classification: the 5th-order
N+1 free-RGB test (PARADIGM-NULL-NO-EFFECT) and 6th-order BoostNeRV-PR110-
residual capacity-constrained test (PARADIGM-NULL-NO-EFFECT) BOTH were
IMPLEMENTATION-LEVEL falsifications at L2 raw-RGB-residual loss paths. Per
the 6th-order landing memo Carmack-dissent verdict + CLAUDE.md "Forbidden
premature KILL without research exhaustion": the diagnosed deeper mechanism
suggested UNIWARD's natural application domain (Fridrich 2014) is
**ENTROPY-CODED SIDECAR SURFACES** where per-symbol routing has direct
control over byte allocation — NOT raw-RGB pixel domain.

THIS 7th-order test integrates UNIWARD INTO the canonical UNIWARD-natural
application surface for our contest:

- **NSCS06 v8 chroma LUT** = a ``(GRAYSCALE_LEVELS, NUM_SEGNET_CLASSES, 3) =
  (16, 5, 3) = 240-effective-entry`` codebook indexed by ``(luma_quant_level,
  segnet_class)`` -> RGB triplet
- The LUT is the entropy-coded sidecar; per-pixel chroma routing happens at
  inflate via ``lookup_rgb_via_chroma_lut(gray_u8, cls_u8, chroma_lut)``
- UNIWARD per-LUT-index weighting: aggregate per-pixel UNIWARD weights into
  the ``(level, class)`` bins, then derive the LUT entry via UNIWARD-weighted
  STATISTIC (weighted median or weighted mean) instead of the canonical
  unweighted median

The (level, class) bins partition pixels; UNIWARD weights tell us which
pixels-within-a-bin matter most to scorer-conditional sensitivity. The
weighted statistic concentrates the LUT entry on the high-sensitivity
pixels within each bin — the LUT entry preserves precision exactly where
the scorer is sensitive.

This IS the canonical Fridrich UNIWARD application domain ported to our
contest: the JPEG-DCT-coefficient analog is the chroma-LUT-index domain;
both are quantized + entropy-coded + per-symbol routable.

Per CLAUDE.md "MLX portable-local-substrate authority": training-time
weight aggregation; output tagged `[macOS-MLX research-signal]` per
Catalog #192/#317/#341. Per Catalog #230 sister-disjoint discipline:
NSCS06 v8 substrate is READ-ONLY consumer-imported; this module does NOT
modify the v8 substrate's training/test paths.

Per Catalog #344 canonical equation anchor (proposed):
``uniward_per_lut_index_distortion_weight_savings_v1`` (FORMALIZATION_PENDING
until paired-CUDA empirical anchor lands per CLAUDE.md "Submission auth
eval - BOTH CPU AND CUDA").
"""

from __future__ import annotations

__all__ = [
    "INTEGRATION_NAME",
    "INTEGRATION_VERSION",
    "CONSUMER_NAME",
    "CONSUMER_VERSION",
    "CONSUMER_HOOK_NUMBERS",
]

INTEGRATION_NAME = "uniward_per_lut_index_into_nscs06_v8_chroma_lut"
INTEGRATION_VERSION = "v1_2026-05-26_7th_order"

# Canonical cathedral consumer contract per Catalog #335
CONSUMER_NAME = "uniward_per_lut_index_into_nscs06_v8_chroma_lut_integration"
CONSUMER_VERSION = "v1_2026-05-26_7th_order"
CONSUMER_HOOK_NUMBERS = (1, 5)  # sensitivity-map + continual-learning per Catalog #125
