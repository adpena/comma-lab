# SPDX-License-Identifier: MIT
# LEGACY_SUBSTRATE_PRE_META_LAYER:l0_scaffold_landed_20260530_pre_meta_layer_register_substrate_decorator_pending_phase_2_council_per_catalog_325
"""PR110-OPT-11 Multi-mode-per-pair composition (L0 SCAFFOLD).

Sister of :mod:`tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1`
(L1 PROMOTION landed 2026-05-30 commit ``1230b3b9c``) at the **per-pair-composition**
surface. Per CLAUDE.md "Canonical leaderboard binding-depth discipline" lesson 7
(substrate-engineering binds ALL ingredients simultaneously), this L0 SCAFFOLD
declares the canonical archive grammar + builder + inflate runtime + trainer
scaffold for **stacking 2+ frame-0 perturbations within a pair** (the operator
NON-NEGOTIABLE per task #1323 PENDING + Wave N+34 Zone 2 PR110-OPT cluster
continuation).

The empirical analytical foundation
====================================

Per Wave N+34 macOS-CPU advisory analytical investigator
(``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``):

- Source: 600-pair x 22-mode component-row index at
  ``experiments/results/frame_exploit_segnet_posenet_20260514_pr101_mps600_codex/pair_component_rows.jsonl``
- Mode menu: 22 modes across 4 orthogonal families
  (``frame0_luma_bias`` / ``frame0_blue_chroma`` / ``frame0_rgb_bias`` /
  ``frame0_roll``) + 1 identity (``none``).
- Analytical compound ratio upper bound = 1.548x single-mode aggregate ΔS.
- Multi-mode additive upper bound ΔS = -0.00181 vs single-mode -0.00117.
- Wire cost increase 428B vs FEC6 baseline 249B = +0.000119 rate-axis ΔS.
- Net score delta upper bound = -0.00052 (DIRECTIONAL).
- Verdict: ``PROCEED_CANDIDATE_FOR_EMPIRICAL_VALIDATION`` per the analytical
  pre-screen; the L0 SCAFFOLD lands the canonical apparatus that empirical
  composed-forward-pass paired-CUDA RATIFICATION will validate at L1.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this L0 SCAFFOLD declares ``research_only=true`` in the lane
registry per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK pattern until paired-
CUDA RATIFICATION lands per Catalog #246 1:1 contest-compliant hardware. The
analytical upper bound is an UPPER BOUND per Wave N+34 + canonical anti-pattern
``cross_paradigm_stacking_additive_compounding_without_dykstra_feasibility``
(true Dykstra-projected polytope intersection per Catalog #373 yields SUB-
ADDITIVE results — actual savings will be LESS than the upper bound).

Architecture (substrate-engineering per binding-depth discipline)
=================================================================

::

    PR110-base archive  --[Stage 0]-->  Reconstruct 600 base pairs (frozen reference)
                                                  |
                                                  v
    Wave N+34 per-pair (mode_a, mode_b)  --[Stage 1]--> Per-pair multi-mode selector pair
       canonical orthogonal-family pair
       enumeration (6 family pairs)
                                                  |
                                                  v
    Frame-0 mode-A application                --[Stage 2]--> Pair P after mode A
                                                  |
                                                  v
    Frame-0 mode-B application (composes      --[Stage 3]--> Pair P after mode A+B
       within-pair on top of mode A)
                                                  |
                                                  v
    Per-pair (selector_a, selector_b) =       --[Stage 4]--> Per-pair multi-mode wire bytes
       2 x 4-bit indices (8 bits per pair)
                                                  |
                                                  v
    Archive emission per OPT11MMP grammar      --[Stage 5]--> archive.zip composed

The 4 canonical orthogonal family pairs (Wave N+34 verified)
=============================================================

- ``(luma_bias, blue_chroma)``  — Y vs U/V (4×3=12 cross-mode combinations)
- ``(luma_bias, rgb_bias)``     — luma vs RGB-channel-bias (6×8=48 combos)
- ``(luma_bias, roll)``         — color vs spatial (6×3=18 combos)
- ``(blue_chroma, rgb_bias)``   — chroma-amp vs RGB-channel-bias (3×8=24 combos)
- ``(blue_chroma, roll)``       — chroma-amp vs spatial (3×3=9 combos)
- ``(rgb_bias, roll)``          — RGB-channel-bias vs spatial (8×3=24 combos)

Total: 6 family pairs × ~135 combinations = ~135 distinct multi-mode pairs per
pair (cap at K_outer × K_inner combinations the per-pair selector encodes).

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable + Slot EEE
fake-implementation audit anchor 2026-05-29: this L0 SCAFFOLD declares the
canonical apparatus; the per-pair multi-mode application IS the canonical
distinguishing feature per Catalog #272 + the byte-mutation smoke per Catalog
#139 validates it. The trainer's ``_smoke_main`` actually applies 2 modes per
pair (not just emits canonical markers).

Canonical contracts honored
===========================

- :class:`AxisDecomposition` per Catalog #356.
- Tier A canonical-routing markers per Catalog #341 + #357.
- Canonical :class:`Provenance` per Catalog #323.
- Cathedral consumer auto-discovery per Catalog #335 (companion consumer
  package deferred to Phase 2 per Catalog #325 per-substrate symposium
  evidence requirement).
- Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK + ``research_only=true``.
- Catalog #272 distinguishing-feature integration contract.
- Catalog #295 PYTHONPATH self-containment per Slot EEE NO FAKE.
- Catalog #325 per-substrate symposium memo at landing.
- Catalog #344 canonical equation FORMALIZATION_PENDING until L1 empirical
  anchor lands.
- Catalog #309 horizon_class = ``frontier_pursuit``.
- Catalog #313 PROCEED 14-day probe outcome.
- Catalog #348 retroactive sweep memo.
- NO FAKE IMPLEMENTATIONS per CLAUDE.md non-negotiable.

Cross-references
================

- Wave N+34 analytical foundation: ``.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json``.
- Sister PR110-OPT-7 L1 PROMOTION:
  :mod:`tac.substrates.pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1`.
- Per-substrate symposium memo: ``.omx/research/per_substrate_symposium_pr110_opt11_multi_mode_per_pair_composition_20260530.md``.
- THIS L0 SCAFFOLD landing memo: ``feedback_pr110_opt11_multi_mode_per_pair_composition_l0_scaffold_landed_20260530.md``.
- Retroactive sweep memo: ``.omx/research/retroactive_sweep_for_pr110_opt11_l0_scaffold_20260530.md``.
- Canonical equation: ``pr110_opt11_multi_mode_per_pair_composition_savings_v1`` (FORMALIZATION_PENDING).
"""

from __future__ import annotations

# Public API surface per Catalog #335 + Catalog #265 canonical contract pattern.
__all__ = [
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "CANONICAL_ORTHOGONAL_FAMILY_PAIRS",
    "DEFAULT_MODES_PER_PAIR",
    "DEFAULT_PR110_BASE_PAIRS",
    "DEFAULT_SELECTOR_BITS_PER_MODE",
    "OPT11MMP_HEADER_FMT",
    "OPT11MMP_HEADER_LEN",
    "PR110OPT11Config",
    "PR110OPT11Result",
    "apply_substrate_to_pr110_canonical",
    "build_substrate_default_config",
    "verify_canonical_multi_mode_composition",
]

# Archive grammar constants re-exported for canonical Catalog #335 contract
# discovery + Catalog #146 frozen-offset discipline.
from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.archive_grammar import (
    ARCHIVE_MAGIC,
    ARCHIVE_VERSION,
    OPT11MMP_HEADER_FMT,
    OPT11MMP_HEADER_LEN,
)
from tac.substrates.pr110_opt11_multi_mode_per_pair_composition.substrate import (
    CANONICAL_ORTHOGONAL_FAMILY_PAIRS,
    DEFAULT_MODES_PER_PAIR,
    DEFAULT_PR110_BASE_PAIRS,
    DEFAULT_SELECTOR_BITS_PER_MODE,
    PR110OPT11Config,
    PR110OPT11Result,
    apply_substrate_to_pr110_canonical,
    build_substrate_default_config,
    verify_canonical_multi_mode_composition,
)
