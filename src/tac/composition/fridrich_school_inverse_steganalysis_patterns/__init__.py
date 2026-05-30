# SPDX-License-Identifier: MIT
"""Canonical Fridrich-school inverse-steganalysis patterns.

Sister landing of ``tac.composition.alaska_inverse_steganalysis_patterns``
at the BROADER-Fridrich-school surface. Where alaska extracts the canonical
2019 ALASKA-paper patterns, THIS package extracts patterns from Yousfi's
POST-alaska recent repos + Fridrich's other students.

Operator binding
----------------
Operator 2026-05-30 verbatim: *"yousfi may have more recent repos that are
even more useful or may have contributed to or been involved with or had
activity on others that may be telling same with fridrich and fridrich's
other students"* + *"Approved all when bandwidth available"*.

Canonical sources (Phase A research)
-------------------------------------
* ``github.com/YassineYousfi/autostego`` (Mar 11, 2026) -- canonical
  Alice-vs-Eve adversarial loop with HILL/WOW/S-UNIWARD + SRNet/SRM/LCLSMR.
* ``github.com/DDELab/deepsteganalysis`` (May 1, 2025) -- canonical
  PyTorch Lightning steganalysis trainer (DDE Lab Binghamton).
* ``github.com/YassineYousfi/OneHotConv`` (Jun 17, 2021) -- canonical
  Yousfi-Fridrich 2020 IEEE SPL paper "Intriguing Struggle" implementation.
* ``github.com/YassineYousfi/comma10k-baseline`` (Jul 6, 2023) -- direct
  ancestry of contest SegNet (``smp.Unet + EfficientNet`` at 874x1164).
* Filler-Judas-Fridrich 2011 IEEE TIFS -- canonical syndrome-trellis
  coding paper; multiple Python ports exist (e.g. ``daniellerch/pySTC``).

Why this matters
----------------
Per CLAUDE.md "Yousfi (challenge creator) was Fridrich's PhD student at
Binghamton DDE Lab" + "Yousfi is currently at comma.ai": Yousfi's recent
GitHub activity (autostego in March 2026; comma_video_compression_challenge
itself in March 2026) is the canonical operator-facing window into both
(a) the inverse-steganalysis attack vectors the contest scorers are designed
to detect, AND (b) the canonical engineering choices Yousfi makes when he
designs deep-learning systems (PyTorch Lightning + ``smp.Unet`` framework +
EfficientNet backbone discipline).

Public surface (7 canonical patterns)
--------------------------------------
* :class:`AliceEveScoringRule` -- canonical autostego Alice-vs-Eve scoring.
* :class:`LCLSMRSolverStrategy` -- canonical LSMR solver for linear
  steganalysis classifier.
* :class:`EfficientNetStemSurgeryStrategy` -- canonical DDELab surgery
  taxonomy + contest SegNet blind-spot constants.
* :class:`OneHotEncodingStrategy` -- canonical Yousfi-Fridrich 2020 OneHot
  CNN encoding strategies.
* :class:`Comma10kBaselineStage` -- canonical 2-stage Yousfi training
  curriculum + canonical contest backbone constants.
* :class:`STCSubMatrixHeight` -- canonical Filler 2011 STC trellis heights.
* :class:`FusionStrategy` -- canonical autostego score-level fusion +
  canonical contest fusion weights.
* :func:`build_fridrich_school_canonical_patterns_inventory` --
  operator-facing introspection.

Sister landings (existing)
--------------------------
* ``tac.composition.alaska_inverse_steganalysis_patterns`` -- ALASKA 2019
  canonical patterns (color separation + pair-constraint batch + multi-scheme
  prior + detector-aware iterative training + CMD discrimination +
  warm-start).
* ``tac.composition.pr110_opt_7_uniward_inverse_scorer_basis_expansion``
  -- Slot FF UNIWARD canonical pattern.
* ``tac.composition.hill_canonical_inverse_steganalysis_li_wang_li_huang_2014``
  -- Slot YY HILL canonical pattern.
* ``tac.composition.mipod_canonical_inverse_steganalysis_sedighi_cogranne_fridrich_2016``
  -- Slot AAA MiPOD canonical pattern.
* ``tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010``
  -- Slot CCC HUGO canonical pattern.

Cross-references
----------------
* CLAUDE.md "Yousfi's repos" (cataloged 2026-04-21).
* CLAUDE.md "Exact scorer architectures" (canonical SegNet + PoseNet
  derived from Yousfi's comma10k-baseline).
* CLAUDE.md "Fridrich inverse steganalysis" (4 canonical principles).
* Catalog #109 (vendored intake clones pristine discipline; we do NOT
  edit upstream Yousfi repos -- our patterns are ports, not modifications).
* Catalog #344 canonical equation
  ``fridrich_school_inverse_steganalysis_patterns_v1`` registered with 7
  producers + 7 consumers.
"""

from __future__ import annotations

from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_alice_vs_eve_adversarial_loop import (
    AliceEveLoopError,
    AliceEveRoundConfig,
    AliceEveRoundResult,
    AliceEveScoringRule,
    canonical_alice_seed_algorithms,
    canonical_eve_seed_detectors,
    compute_alice_score_minimax,
    compute_eve_score_minimax,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_lclsmr_linear_steganalysis_detector import (
    LCLSMRConfig,
    LCLSMRDetectorError,
    LCLSMRSolverStrategy,
    fit_lclsmr_linear_classifier,
    score_lclsmr_linear_classifier,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_efficientnet_steganalysis_surgery import (
    CONTEST_SEGNET_STRIDE_2_BLIND_SPOT,
    EfficientNetStemSurgeryStrategy,
    EfficientNetSurgeryConfig,
    EfficientNetSurgeryError,
    compute_blind_spot_resolution_from_stride,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_onehot_jpeg_steganalysis import (
    DCT_COEFFICIENT_CANONICAL_RANGE,
    OneHotEncodingConfig,
    OneHotEncodingError,
    OneHotEncodingStrategy,
    compute_one_hot_input_channels,
    encode_value_one_hot,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_comma10k_baseline_lineage import (
    COMMA10K_BASELINE_BACKBONE,
    CONTEST_NATIVE_RESOLUTION,
    CONTEST_SEGNET_BACKBONE,
    Comma10kBaselineStage,
    Comma10kCurriculumConfig,
    Comma10kLineageError,
    Comma10kTrainingStrategy,
    compute_resolution_for_stage,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_syndrome_trellis_coding_filler import (
    FILLER_2011_DISTORTION_VS_BOUND_PERCENT,
    FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE,
    STCAdaptiveEmbeddingStrategy,
    STCEmbeddingConfig,
    STCEmbeddingError,
    STCSubMatrixHeight,
    compute_stc_distortion_bound_from_sub_matrix_height,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_fusion_detector_ensemble import (
    CONTEST_FUSION_WEIGHTS_CANONICAL,
    FusionConfig,
    FusionDetectorError,
    FusionStrategy,
    compute_canonical_contest_fusion_weights,
    compute_fusion_score,
)
from tac.composition.fridrich_school_inverse_steganalysis_patterns.canonical_pattern_inventory import (
    FRIDRICH_GROUP_ATTRIBUTION,
    FRIDRICH_SCHOOL_ATTRIBUTION,
    FridrichSchoolCanonicalPatternRow,
    YOUSFI_GITHUB_HOMEPAGE,
    build_fridrich_school_canonical_patterns_inventory,
)

__all__ = (
    # Alice-vs-Eve
    "AliceEveLoopError",
    "AliceEveRoundConfig",
    "AliceEveRoundResult",
    "AliceEveScoringRule",
    "canonical_alice_seed_algorithms",
    "canonical_eve_seed_detectors",
    "compute_alice_score_minimax",
    "compute_eve_score_minimax",
    # LCLSMR
    "LCLSMRConfig",
    "LCLSMRDetectorError",
    "LCLSMRSolverStrategy",
    "fit_lclsmr_linear_classifier",
    "score_lclsmr_linear_classifier",
    # EfficientNet surgery
    "CONTEST_SEGNET_STRIDE_2_BLIND_SPOT",
    "EfficientNetStemSurgeryStrategy",
    "EfficientNetSurgeryConfig",
    "EfficientNetSurgeryError",
    "compute_blind_spot_resolution_from_stride",
    # OneHot
    "DCT_COEFFICIENT_CANONICAL_RANGE",
    "OneHotEncodingConfig",
    "OneHotEncodingError",
    "OneHotEncodingStrategy",
    "compute_one_hot_input_channels",
    "encode_value_one_hot",
    # comma10k-baseline lineage
    "COMMA10K_BASELINE_BACKBONE",
    "CONTEST_NATIVE_RESOLUTION",
    "CONTEST_SEGNET_BACKBONE",
    "Comma10kBaselineStage",
    "Comma10kCurriculumConfig",
    "Comma10kLineageError",
    "Comma10kTrainingStrategy",
    "compute_resolution_for_stage",
    # STC Filler
    "FILLER_2011_DISTORTION_VS_BOUND_PERCENT",
    "FILLER_CANONICAL_SUB_MATRIX_HEIGHT_RANGE",
    "STCAdaptiveEmbeddingStrategy",
    "STCEmbeddingConfig",
    "STCEmbeddingError",
    "STCSubMatrixHeight",
    "compute_stc_distortion_bound_from_sub_matrix_height",
    # Fusion
    "CONTEST_FUSION_WEIGHTS_CANONICAL",
    "FusionConfig",
    "FusionDetectorError",
    "FusionStrategy",
    "compute_canonical_contest_fusion_weights",
    "compute_fusion_score",
    # Inventory
    "FRIDRICH_GROUP_ATTRIBUTION",
    "FRIDRICH_SCHOOL_ATTRIBUTION",
    "FridrichSchoolCanonicalPatternRow",
    "YOUSFI_GITHUB_HOMEPAGE",
    "build_fridrich_school_canonical_patterns_inventory",
)
