# SPDX-License-Identifier: MIT
"""Canonical ALASKA pattern inventory + operator-facing cross-reference matrix.

Operator-facing introspection helper returning the registered patterns +
their HARD-EARNED-vs-CARGO-CULTED classification per CLAUDE.md Catalog
#303 sister discipline + Yousfi-Fridrich cascade cross-reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = (
    "AlaskaCanonicalPatternRow",
    "build_alaska_canonical_patterns_inventory",
    "ALASKA_REPO_ATTRIBUTION",
    "ALASKA_ORIGIN_PAPER_CITATION",
)


ALASKA_REPO_ATTRIBUTION: str = (
    "Copyright (c) 2019 DDE Lab, Binghamton University, NY. All Rights Reserved.\n"
    "Source: https://github.com/YassineYousfi/alaska (read-only clone at "
    "external/alaska_yousfi/). License: educational/research/non-profit use "
    "without fee per LICENSE.md."
)


ALASKA_ORIGIN_PAPER_CITATION: str = (
    "Yousfi, Y., Butora, J., Fridrich, J., & Giboulot, Q. (2019). "
    "Breaking ALASKA: Color Separation for Steganalysis in JPEG Domain. "
    "Proceedings of the ACM Workshop on Information Hiding and Multimedia "
    "Security (IH&MMSec'19), pp. 138-149. doi:10.1145/3335203.3335727"
)


@dataclass(frozen=True)
class AlaskaCanonicalPatternRow:
    """One canonical pattern row.

    Attributes
    ----------
    pattern_id
        Canonical name (e.g. ``"color_separation_5_branch"``).
    upstream_source
        Specific upstream file:line range (e.g. ``"jpeg_utils.py:50-62"``).
    tac_module
        Where the pattern lives in tac (e.g. ``"tac.composition.alaska_inverse_steganalysis_patterns.canonical_color_separation"``).
    five_axis_classification
        Mapping ``{"contest"/"problem_space"/"math"/"data"/"video"}`` ->
        HARD-EARNED-1:1 / DOCUMENTED-ADAPTATION / N/A per Catalog #303.
    cross_reference_yousfi_fridrich_axis
        Which of the 7-8 Yousfi-Fridrich-cascade axes this pattern slots
        into per CLAUDE.md "Fridrich inverse steganalysis" lineage.
    pr_score_lowering_ev_estimate
        Operator-facing predicted ΔS band ``"[low, high]"`` per Catalog
        #296 Dykstra-feasibility discipline, or ``"FORMALIZATION_PENDING"``
        per Catalog #344 sister discipline if not yet bounded.
    """

    pattern_id: str
    upstream_source: str
    tac_module: str
    five_axis_classification: Mapping[str, str]
    cross_reference_yousfi_fridrich_axis: str
    pr_score_lowering_ev_estimate: str


def build_alaska_canonical_patterns_inventory() -> tuple[AlaskaCanonicalPatternRow, ...]:
    """Return the canonical 6-pattern inventory + cross-reference matrix.

    Returns
    -------
    tuple[AlaskaCanonicalPatternRow, ...]
        6 canonical patterns extracted from
        ``github.com/YassineYousfi/alaska`` and ported as conceptual
        patterns to the comma-video contest substrate surface.
    """
    return (
        AlaskaCanonicalPatternRow(
            pattern_id="color_separation_5_branch",
            upstream_source="external/alaska_yousfi/src/tools/jpeg_utils.py:50-62 + models.py:32-40",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_color_separation",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:JPEG-YCrCb-to-contest-YUV6",
                "problem_space": "DOCUMENTED-ADAPTATION:detection-to-generation",
                "math": "HARD-EARNED-1:1:channel-wise-slice-tensor",
                "data": "DOCUMENTED-ADAPTATION:256x256-JPEG-to-1164x874-contest",
                "video": "DOCUMENTED-ADAPTATION:single-image-to-per-pair-shared-latent",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 5 (YUV6 PoseNet 12-channel decomposition; "
                "this pattern provides the canonical channel-subset taxonomy "
                "for per-branch sensitivity analysis)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.005, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "the EV depends on which branch carries the dominant pose-axis "
                "sensitivity signal which requires per-substrate empirical anchor"
            ),
        ),
        AlaskaCanonicalPatternRow(
            pattern_id="pair_constraint_batch",
            upstream_source="external/alaska_yousfi/src/tools/tf_utils.py:55-95",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_pair_constraint_batch",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:cover-stego-pair-to-frame0-frame1-pair",
                "problem_space": "DOCUMENTED-ADAPTATION:detector-training-to-generator-scorer-joint",
                "math": "HARD-EARNED-1:1:pair-structure-preserved",
                "data": "DOCUMENTED-ADAPTATION:BOSSBase-to-Comma2k19-plus-contest-video",
                "video": "HARD-EARNED-1:1:temporal-pair-frame0-frame1-is-canonical-contest-contract",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 0 (foundational; every cost-function variant Slot FF / "
                "YY / AAA / CCC inherits the canonical pair-constraint batching)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.003, -0.001] FORMALIZATION_PENDING per Catalog #344; "
                "Yousfi empirically reports ~2-3pp detector-accuracy gain "
                "from this pattern alone; adaptation to generator gradient "
                "is a sister hypothesis requiring per-substrate anchor"
            ),
        ),
        AlaskaCanonicalPatternRow(
            pattern_id="multi_scheme_dirichlet_prior",
            upstream_source="external/alaska_yousfi/src/notebooks/tf_fine_tune_branch.ipynb cell 3",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_multi_scheme_prior",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:JPEG-scheme-to-comma-cost-function",
                "problem_space": "DOCUMENTED-ADAPTATION:detector-training-to-generator-gradient-weighting",
                "math": "HARD-EARNED-1:1:categorical-Dirichlet",
                "data": "DOCUMENTED-ADAPTATION:BOSSBase-4-scheme-to-comma-4-cost-function",
                "video": "DOCUMENTED-ADAPTATION:single-image-to-per-pair-latent",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 8 (NEW; cross-cuts Slot FF UNIWARD + Slot YY HILL + Slot "
                "AAA MiPOD + Slot CCC HUGO via the dispatch-prior surface; "
                "tells the cathedral autopilot how to weight the 4 cost-function "
                "axes when allocating paid GPU spend)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.001, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "the EV is in dispatch-budget efficiency not direct score lowering"
            ),
        ),
        AlaskaCanonicalPatternRow(
            pattern_id="detector_aware_iterative_training",
            upstream_source="external/alaska_yousfi/src/tools/train_estimator.py:155-194",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_detector_aware_iterative_training",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:detector-training-to-generator-scorer-joint",
                "problem_space": "DOCUMENTED-ADAPTATION:5-class-softmax-to-per-pair-score-components",
                "math": "HARD-EARNED-1:1:canonical-3-stage-piecewise-constant-LR-plus-Adamax",
                "data": "DOCUMENTED-ADAPTATION:200k-iter-on-QF95-256x256-to-contest-resolution-budget",
                "video": "DOCUMENTED-ADAPTATION:single-image-batches-to-per-pair-latent-shared",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 0 (foundational training protocol every Slot FF / YY / "
                "AAA / CCC trainer inherits; supplies the canonical 3-stage "
                "LR schedule + Adamax + warm-start discipline)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.005, -0.001] FORMALIZATION_PENDING per Catalog #344; "
                "the canonical 3-stage curriculum empirically beats single-LR "
                "training; warm-start from single-branch to multi-branch is "
                "an additional canonical pattern that compounds"
            ),
        ),
        AlaskaCanonicalPatternRow(
            pattern_id="cmd_per_image_4_stat_discrimination",
            upstream_source="external/alaska_yousfi/src/tools/models.py:139-140",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_cmd_per_image_discrimination",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:JPEG-embedding-signal-to-contest-YUV6-perturbation",
                "problem_space": "DOCUMENTED-ADAPTATION:5-class-softmax-to-per-pair-discrimination",
                "math": "HARD-EARNED-1:1:avg-var-min-max-4-stat-tuple",
                "data": "DOCUMENTED-ADAPTATION:pre-extracted-feature-maps-to-live-per-pair-latent",
                "video": "DOCUMENTED-ADAPTATION:per-image-to-per-pair-shared-latent",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 9 (NEW; canonical discrimination-statistic helper that "
                "every Slot RR pose-axis null-projection + Slot AAA MiPOD + "
                "Slot CCC HUGO probe can adopt for per-pair embedding-signal "
                "characterization)"
            ),
            pr_score_lowering_ev_estimate=(
                "[-0.002, +0.000] FORMALIZATION_PENDING per Catalog #344; "
                "the EV is in probe-disambiguator quality (per Catalog #125 "
                "hook #6) not direct score lowering; sister of canonical "
                "frontier_primitives.compute_4_stat_pool when that lands"
            ),
        ),
        AlaskaCanonicalPatternRow(
            pattern_id="warm_start_single_to_multi_branch",
            upstream_source="external/alaska_yousfi/src/tools/train_estimator.py:118-124",
            tac_module="tac.composition.alaska_inverse_steganalysis_patterns.canonical_detector_aware_iterative_training (config field warm_start_checkpoint_branch)",
            five_axis_classification={
                "contest": "DOCUMENTED-ADAPTATION:5-branch-warm-start-to-COMMA-11-slice-warm-start",
                "problem_space": "HARD-EARNED-1:1:transfer-learning-checkpoint-rebind",
                "math": "HARD-EARNED-1:1:scope-prefix-substitution-Layer-rebind",
                "data": "DOCUMENTED-ADAPTATION:single-branch-checkpoint-to-multi-branch-warm-start",
                "video": "DOCUMENTED-ADAPTATION:per-image-checkpoint-to-per-pair-checkpoint",
            },
            cross_reference_yousfi_fridrich_axis=(
                "Axis 0 (foundational; every multi-branch substrate trainer "
                "Slot FF / YY / AAA / CCC inherits the canonical warm-start "
                "from a single-branch checkpoint pattern; this avoids cold-start "
                "expense when training the second + third + ... branches)"
            ),
            pr_score_lowering_ev_estimate=(
                "[+0.000, +0.000] DISPATCH-BUDGET-EV-NOT-SCORE-EV per Catalog "
                "#344; the EV is in 2-3x training time reduction not direct "
                "score lowering; sister of CLAUDE.md 'Race-mode rigor inversion' "
                "+ canonical dispatch optimization protocol per Catalog #270"
            ),
        ),
    )
