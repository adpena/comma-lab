# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l1_promotion_landed_20260530_via_yousfi_t1_enablement_per_deferred_items_feeder_top_1_op_routable_meta_layer_register_substrate_decorator_pending_phase_2_council_symposium_per_catalog_325_after_paired_cuda_ratification_lands
"""PR110-OPT-7 Fridrich UNIWARD inverse-scorer basis via Yousfi-Tier-1 enablement (L1 IMPL_COMPLETE).

Sister of :mod:`tac.composition.pr110_opt_7_fridrich_uniward_inverse_scorer_basis`
(Slot CCC L0 SCAFFOLD landed 2026-05-30 commit ``3fd28b5b2``) at the
**SUBSTRATE-ENGINEERING** surface. Per CLAUDE.md "HNeRV / leaderboard-implementation
parity discipline" lesson 7 (substrate-engineering binds ALL ingredients
simultaneously), this L1 PROMOTION composes 5 LANDED canonical primitives
into ONE coherent substrate so the canonical 8-axis Fridrich-Yousfi cascade
becomes empirically dispatch-ready.

Per deferred-items feeder audit ``46aa6ad86`` Phase E TOP-1 op-routable:
*"PR110-OPT-7 L1 promotion via Yousfi-T1 enablement — canonical compounding
sister of Yousfi-Tier-1 pose-axis foundation"*.

Per CLAUDE.md "PR-or-greater parity (not HNeRV-specific)" META-class lesson
2026-05-30: the parity discipline is BINDING DEPTH across canonical L1-L32
ingredients + Tier-1 engineering + canonical equations registry; THIS
substrate binds 5 ingredients deeply rather than shipping 5 disjoint
bolt-ons.

Today's 5 LANDED canonical primitives composed
==============================================

1. **alaska canonical inverse-steganalysis patterns** (commit ``61a91a48e``).
   We consume canonical pattern #1 (Color separation: 11-slice YUV6 channel
   layout) via :func:`tac.composition.alaska_inverse_steganalysis_patterns.\
   branch_to_yuv6_channel_slice` so the substrate's chroma-aware Fridrich
   UNIWARD basis routes through the canonical ALASKA color decomposition
   (``Y0_UV`` branch by default; SegNet stride-2 stem 256x192 blind spot
   matches Y0+U+V at 192x256 yuv6-grid).

2. **Yousfi-Tier-1 Deliverable A: per-pair pose-vulnerability map** (commit
   ``3d027ecf9``). We consume :func:`tac.master_gradient_pose_vulnerability.\
   build_default_pose_vulnerability_map_from_canonical_anchor` to derive
   the canonical PAIR-SELECTION PRIOR (vulnerability_ratio empirical anchor
   ~363x on the canonical 600-pair fp64 tensor). The substrate preferentially
   targets POSE-VULNERABLE pairs (Yousfi-`alaska` analog at the pair surface).

3. **Yousfi-Tier-1 Deliverable B: PoseNet MAE-V surrogate** (commit
   ``3d027ecf9``). We consume :class:`tac.scorer_surrogate.posenet_mae_v.\
   PoseNetMaeVSurrogate` as the numpy-portable predicted PoseNet surrogate
   so the substrate can compute per-pixel PoseNet-grounded UNIWARD weights
   WITHOUT requiring real PoseNet (which would violate strict-scorer-rule
   at inflate time per CLAUDE.md "scorer-load-at-inflate" non-negotiable).

4. **Yousfi-Tier-1 Deliverable C: YUV6 chroma-subsampled perturbation
   operator** (commit ``3d027ecf9``). We consume
   :func:`tac.composition.yuv6_chroma_subsampled_perturbation_operator.\
   apply_chroma_subsampled_perturbation` as the canonical Y-luma-preserving
   chroma-only perturbation primitive. The Fridrich UNIWARD basis weighting
   parameterizes the chroma-subsampled operator's weight map; luma
   preservation invariant holds in YUV6 space EXACTLY (drift = 0.0).

5. **PR110-OPT-7 inverse-scorer basis L0 SCAFFOLD** (commit ``3fd28b5b2``).
   We consume :func:`tac.composition.pr110_opt_7_fridrich_uniward_inverse_\
   scorer_basis.apply_uniward_inverse_scorer_basis_to_pr110_archive` as
   the canonical Fridrich-Holub-Denemark 2014 UNIWARD weighting source.
   The 4 basis strategies (LOCAL_VARIANCE / SEGNET_GRAD / POSENET_GRAD /
   JOINT) feed the substrate's UNIWARD weight expansion.

Architecture (substrate-engineering per HNeRV parity L7)
=========================================================

::

    PR110-base archive  --[Stage 0]-->  Reconstruct 600 base pairs (frozen reference)
                                                  |
                                                  v
    Yousfi-T1 Deliverable A vulnerability map --[Stage 1]--> Pair-selection prior
                                                  |
                                                  v
    alaska canonical color-separation YUV6 slice --[Stage 2]--> Y0_UV chroma branch
                                                  |
                                                  v
    Yousfi-T1 Deliverable C chroma perturbation --[Stage 3]--> Perturbed YUV6 pairs
                                                  |
                                                  v
    Yousfi-T1 Deliverable B PoseNet MAE-V        --[Stage 4]--> Pose-grad weights
                                                  |
                                                  v
    PR110-OPT-7 inverse-scorer basis L0          --[Stage 5]--> UNIWARD per-pixel cost
                                                  |
                                                  v
    Joint scorer-Jacobian linear combination     --[Stage 6]--> Per-pair selector palette
                                                  |
                                                  v
    Archive emission per OPT7VYT1 grammar         --[Stage 7]--> archive.zip composed

Archive grammar (OPT7VYT1)
==========================

Frozen archive grammar per Catalog #146 contest 3-arg signature
+ Catalog #220 operational mechanism + Catalog #272 distinguishing-feature
contract. The substrate INVOKES the canonical Yousfi-T1 + alaska + PR110-OPT-7
helpers at archive-build time; inflate.py per Catalog #146 + Catalog #205
canonical select_inflate_device + Catalog #295 PYTHONPATH self-containment
+ ≤200 LOC HNeRV parity L4 budget (substrate-engineering exception per L7
for the L1 PROMOTION; bolt-on follow-up reverts to ≤200 LOC after Phase 2
council symposium per Catalog #325).

L1 IMPL_COMPLETE role
=====================

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this L1 IMPL_COMPLETE substrate declares ``research_only=true``
in lane registry notes per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK
pattern until paired-CUDA empirical anchor lands per Catalog #246 1:1
contest-compliant hardware.

The canonical entry point :func:`apply_substrate_to_pr110_canonical` returns
a Tier A canonical-routing-markers contribution per Catalog #341
+ AxisDecomposition per Catalog #356 + canonical Provenance per Catalog
#323. The verdict field is ``DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR``
until Phase D operator-attended dispatch lands paired anchors.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable + Slot EEE
fake-implementation audit anchor 2026-05-29: the substrate ACTUALLY composes
all 5 canonical helpers in its forward pass; the tests verify behavioral
invariants (mock.patch verification that each helper IS invoked + distinct-
output invariant verification that changing the input changes the output).
NOT a returns-canonical-markers-without-doing-work scaffold.

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356.
- Tier A canonical-routing markers per Catalog #341 + #357.
- Canonical :class:`Provenance` per Catalog #323.
- Cathedral consumer auto-discovery per Catalog #335 (companion consumer
  package at `src/tac/cathedral_consumers/pr110_opt7_substrate_via_yousfi_t1_consumer/`
  Phase 2; this substrate package is the canonical PRODUCER surface).
- NO FAKE IMPLEMENTATIONS per CLAUDE.md non-negotiable 2026-05-30 +
  Slot EEE audit anchor 2026-05-29.
- PR-or-greater parity (not HNeRV-specific) per META-class lesson 2026-05-30.

Cross-references
================

- Design memo: ``.omx/research/pr110_opt7_fridrich_uniward_inverse_scorer_basis_design_20260530.md``
  (L0 SCAFFOLD design; THIS L1 PROMOTION extends).
- Sister L0 SCAFFOLD landing: ``feedback_pr110_opt7_fridrich_uniward_inverse_scorer_basis_l0_scaffold_landed_20260530.md``.
- Yousfi-T1 canonical landing: ``feedback_yousfi_cascade_tier_1_pose_axis_canonical_helpers_a_plus_b_plus_c_landed_20260530.md``.
- alaska canonical landing: ``feedback_alaska_yousfi_canonical_pattern_extraction_landed_20260530.md``.
- Deferred-items feeder audit: ``feedback_deferred_items_feeder_audit_post_alaska_m9v3_yousfi_tier_1_wave_landed_20260530.md``.
- THIS L1 promotion landing memo: ``feedback_pr110_opt7_l1_promotion_via_yousfi_t1_landed_20260530.md``.
"""

from __future__ import annotations

# Public API surface. Narrow + explicit per Catalog #335 + Catalog #265
# canonical contract pattern.
__all__ = [
    # Canonical config + result dataclasses
    "PR110OPT7ViaYousfiT1Config",
    "PR110OPT7ViaYousfiT1Result",
    # Canonical entry point
    "apply_substrate_to_pr110_canonical",
    # Canonical helpers re-exported for caller convenience
    "build_substrate_default_config",
    "verify_canonical_helper_invocation",
    # Archive grammar constants (re-exported from archive_grammar)
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "OPT7VYT1_HEADER_FMT",
    "OPT7VYT1_HEADER_LEN",
    # Canonical defaults
    "DEFAULT_PR110_BASE_PAIRS",
    "DEFAULT_VULNERABLE_PAIR_BUDGET",
    "DEFAULT_CHROMA_PERTURBATION_MAGNITUDE",
    "DEFAULT_ALASKA_COLOR_BRANCH",
]


# Archive grammar constants. Re-exported from archive_grammar module for
# canonical Catalog #335 contract discovery + Catalog #146 frozen-offset
# discipline.
from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.archive_grammar import (
    ARCHIVE_MAGIC,
    ARCHIVE_VERSION,
    OPT7VYT1_HEADER_FMT,
    OPT7VYT1_HEADER_LEN,
)
from tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.substrate import (
    DEFAULT_ALASKA_COLOR_BRANCH,
    DEFAULT_CHROMA_PERTURBATION_MAGNITUDE,
    DEFAULT_PR110_BASE_PAIRS,
    DEFAULT_VULNERABLE_PAIR_BUDGET,
    PR110OPT7ViaYousfiT1Config,
    PR110OPT7ViaYousfiT1Result,
    apply_substrate_to_pr110_canonical,
    build_substrate_default_config,
    verify_canonical_helper_invocation,
)
