"""tac.blahut_arimoto — canonical Blahut-Arimoto R(D) iteration helpers.

Per WAVE-3-PATH-A.2 closure of PATH-A's documented deeper gap (canonical
equation ``categorical_posterior_capacity_vs_continuous_gaussian_v1``
registers the Shannon **necessary-condition** lower bound; this package
implements the canonical Blahut-Arimoto iteration that produces the
**actual achievable** R(D) curve for an arbitrary discrete source +
distortion-matrix triple).

Quick start::

    from tac.blahut_arimoto import (
        RateDistortionCurve,
        iterate_rate_distortion,
        iterate_categorical_rd,
        iterate_contest_scorer_rd,
    )

    # Canonical R(D) sweep (Cover & Thomas §10.8)
    curve = iterate_rate_distortion(distortion_matrix, source_distribution)

    # DreamerV3 RSSM specialization
    rssm = iterate_categorical_rd(G=32, K=32)

    # Contest-scorer specialization (operator-routable closure of PATH-A gap)
    contest = iterate_contest_scorer_rd(
        distortion_oracle,
        n_source_symbols=256,
    )

Per Catalog #344 canonical equations registry the sub-equation
``categorical_blahut_arimoto_rate_distortion_v1`` is registered alongside
the existing PATH-A equation; the sub-equation REFINES PATH-A's
necessary-condition lower bound into the achievable curve.

Cross-references
================

* PATH-A landing memo: ``feedback_dreamerv3_rssm_categorical_rd_canonical_equation_landed_20260520.md``
* PATH-A derivation: ``.omx/research/dreamerv3_rssm_categorical_rd_canonical_equation_derivation_20260520T131815Z.md``
* Cover & Thomas 2nd ed §10.8 — canonical Blahut-Arimoto formulation.
* Blahut 1972 IEEE TIT IT-18 — original Blahut derivation.
* Arimoto 1972 IEEE TIT IT-18 — original Arimoto derivation.
* Sister foundational helper:
  ``tac.symposium_impls.blahut_arimoto_theoretical_floor.blahut_arimoto_rate_distortion``
  (single-point + slope-bisection driver this package composes into a sweep).
* CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.
* Catalog #344 canonical equations registry; Catalog #323 canonical Provenance.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.blahut_arimoto.canonical import (
    DEFAULT_LAMBDA_SWEEP_POINTS,
    DEFAULT_MAX_ITER,
    DEFAULT_TOLERANCE,
    RateDistortionCurve,
    build_default_lambda_sweep,
    iterate_rate_distortion,
)
from tac.blahut_arimoto.categorical import (
    DEFAULT_C6_PATH_B2_G,
    DEFAULT_C6_PATH_B2_K,
    DEFAULT_DREAMERV3_HAFNER_G,
    DEFAULT_DREAMERV3_HAFNER_K,
    default_categorical_msl_distortion,
    iterate_categorical_rd,
)
from tac.blahut_arimoto.contest_scorer import (
    ContestScorerDistortionOracle,
    build_distortion_matrix_from_oracle,
    iterate_contest_scorer_rd,
)

__all__ = [
    # Canonical (Cover & Thomas §10.8)
    "DEFAULT_LAMBDA_SWEEP_POINTS",
    "DEFAULT_MAX_ITER",
    "DEFAULT_TOLERANCE",
    "RateDistortionCurve",
    "build_default_lambda_sweep",
    "iterate_rate_distortion",
    # Categorical specialization (DreamerV3 RSSM + C6 Path B2)
    "DEFAULT_C6_PATH_B2_G",
    "DEFAULT_C6_PATH_B2_K",
    "DEFAULT_DREAMERV3_HAFNER_G",
    "DEFAULT_DREAMERV3_HAFNER_K",
    "default_categorical_msl_distortion",
    "iterate_categorical_rd",
    # Contest-scorer specialization (operator-routable PATH-A closure)
    "ContestScorerDistortionOracle",
    "build_distortion_matrix_from_oracle",
    "iterate_contest_scorer_rd",
]
