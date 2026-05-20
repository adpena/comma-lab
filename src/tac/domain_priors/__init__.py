# SPDX-License-Identifier: MIT
"""tac.domain_priors — canonical domain-aware priors namespace.

Per operator NON-NEGOTIABLE 2026-05-20 + CATHEDRAL-SMARTER-DESIGN-MEMO
``.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md``
Dimension 4: this namespace exposes 4 canonical wrappers over already-landed
domain helpers so the cathedral autopilot ranker + bit-allocator + master-
gradient consumers can ingest priors WITHOUT manual ranker-cascade edits.

The wrappers SISTER-EXTEND existing helpers per CLAUDE.md
"UNIQUE-AND-COMPLETE-PER-METHOD operating mode" canonical-vs-unique decision
per layer (Catalog #290): bolt-ons share infrastructure value > unmeasured
customization value. The 4 helpers are:

  * ``per_frame_difficulty_atlas`` (wraps ``tac.master_gradient_consumers.
    per_pair_difficulty_atlas`` + per-pair → per-frame aggregator)
  * ``ego_motion_concentration_atlas`` (wraps ``tac.ego_flow`` +
    canonical Atick-Redlich / Rao-Ballard ego-motion prior)
  * ``per_class_statistical_priors`` (wraps ``tac.categorical_substrate`` +
    ``tac.categorical_label_atoms``)
  * ``comma2k19_priors`` (wraps ``tac.substrates.pretrained_driving_prior.
    local_chunk_cache.Comma2k19LocalCache`` per Catalog #213)

The 3 canonical equations registered via :mod:`tac.canonical_equations` per
Catalog #344 are:

  * ``per_frame_difficulty_atlas_v1`` (per-pair → per-frame aggregation
    formula; canonical_consumers includes cathedral autopilot ranker)
  * ``ego_motion_concentration_prior_v1`` (Atick-Redlich cooperative-
    receiver framing applied to comma2k19 distribution)
  * ``per_segnet_class_chroma_priors_v1`` (SegNet 5-class per-class
    pixel-count + chroma-variance prior; sister of Catalog #354 exploit #5
    `per_segnet_class_chroma_consumer`)

6-hook wire-in per Catalog #125:

  1. Sensitivity-map: ACTIVE — domain priors ARE the canonical per-frame /
     per-pair / per-class sensitivity surface.
  2. Pareto constraint: N/A (priors inform weights; don't constrain
     feasibility directly).
  3. Bit-allocator: ACTIVE (future) — per-frame difficulty drives per-frame
     bit allocation; per-class chroma drives per-class allocation.
  4. Cathedral autopilot dispatch: ACTIVE — ``domain_prior_consumer``
     cathedral package (Step 4.3 — separate next-wave subagent) integrates
     all 4 priors into ranking.
  5. Continual-learning posterior: ACTIVE — every domain-prior anchor
     updates the canonical equation via
     ``tac.canonical_equations.update_equation_with_empirical_anchor``.
  6. Probe-disambiguator: ACTIVE — domain priors disambiguate between
     candidates that score similarly on average but differ on hard-frame /
     easy-frame / hard-class distribution.

Cross-references:
  * CATHEDRAL-SMARTER-DESIGN-MEMO Dim 4 (Steps 4.1-4.2 implemented here)
  * Catalog #209 ``check_no_contest_video_leakage_in_distillation_callers``
    (Comma2k19 = OOD; contest video is ``upstream/videos/0.mkv``)
  * Catalog #210 ``check_dp1_codebook_provenance_metadata_present``
    (DP1 codebook provenance pattern this namespace inherits)
  * Catalog #213 ``check_comma2k19_downloads_route_through_canonical_cache``
    (`Comma2k19LocalCache` canonical helper)
  * Catalog #287 docstring-overstatement-without-evidence-tag (every prior
    surface returns `[predicted]` axis tag until paired CUDA+CPU anchor)
  * Catalog #323 canonical Provenance umbrella (every prior dataclass
    carries Provenance)
  * Catalog #335 cathedral consumer canonical contract (Step 4.3 wire-in)
  * Catalog #343 canonical frontier pointer (no hardcoded score literals)
  * Catalog #344 canonical equations registry (3 equations registered)
  * Catalog #354 master-gradient exploit consumer bundle (sister consumers
    that consume domain-prior signals)

Quick start::

    from tac.domain_priors import (
        PerFrameDifficultyAtlas,
        EgoMotionConcentrationAtlas,
        PerClassStatisticalPriors,
        Comma2k19DashcamPriors,
        build_per_frame_difficulty_from_per_pair_atlas,
        build_ego_motion_concentration_from_pose_anchors,
        build_per_class_statistical_priors_from_scorer_output,
        build_comma2k19_dashcam_priors_from_cache,
        register_domain_prior_canonical_equations,
    )
"""
from __future__ import annotations

from tac.domain_priors.per_frame_difficulty import (
    PerFrameDifficultyAtlas,
    PerFrameDifficultyEntry,
    PER_FRAME_DIFFICULTY_AGGREGATOR_VALID,
    build_per_frame_difficulty_from_per_pair_atlas,
)
from tac.domain_priors.ego_motion_concentration import (
    EgoMotionConcentrationAtlas,
    EgoMotionConcentrationEntry,
    build_ego_motion_concentration_from_pose_anchors,
    build_ego_motion_concentration_from_affine_flow,
)
from tac.domain_priors.per_class_statistical import (
    PerClassStatisticalPriors,
    PerClassPrior,
    SEGNET_CLASS_COUNT,
    SEGNET_CLASS_NAMES,
    build_per_class_statistical_priors_from_scorer_output,
)
from tac.domain_priors.comma2k19_priors import (
    Comma2k19DashcamPriors,
    build_comma2k19_dashcam_priors_from_cache,
)
from tac.domain_priors.equations import (
    DOMAIN_PRIORS_EQUATION_IDS,
    build_per_frame_difficulty_atlas_v1,
    build_ego_motion_concentration_prior_v1,
    build_per_segnet_class_chroma_priors_v1,
    build_all_domain_prior_equations,
    register_domain_prior_canonical_equations,
)


__all__ = [
    # Per-frame difficulty
    "PerFrameDifficultyAtlas",
    "PerFrameDifficultyEntry",
    "PER_FRAME_DIFFICULTY_AGGREGATOR_VALID",
    "build_per_frame_difficulty_from_per_pair_atlas",
    # Ego-motion concentration
    "EgoMotionConcentrationAtlas",
    "EgoMotionConcentrationEntry",
    "build_ego_motion_concentration_from_pose_anchors",
    "build_ego_motion_concentration_from_affine_flow",
    # Per-class statistical priors
    "PerClassStatisticalPriors",
    "PerClassPrior",
    "SEGNET_CLASS_COUNT",
    "SEGNET_CLASS_NAMES",
    "build_per_class_statistical_priors_from_scorer_output",
    # Comma2k19 dashcam priors
    "Comma2k19DashcamPriors",
    "build_comma2k19_dashcam_priors_from_cache",
    # Canonical equations
    "DOMAIN_PRIORS_EQUATION_IDS",
    "build_per_frame_difficulty_atlas_v1",
    "build_ego_motion_concentration_prior_v1",
    "build_per_segnet_class_chroma_priors_v1",
    "build_all_domain_prior_equations",
    "register_domain_prior_canonical_equations",
]
