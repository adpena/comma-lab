# SPDX-License-Identifier: MIT
"""tac.canonical_equations — typed registry of formalized empirical findings.

Per operator NON-NEGOTIABLE 2026-05-19 verbatim: *"we need to formalize all
of this and canonicalize and operationalize because I am afraid we are
learning but if we don't have systems of equations and models and such we
are just gaining tribal knowledge"*.

The package codifies the session's accumulated empirical findings into a
typed + auditable + operator-callable + auto-recalibrating system of
equations + predictive models. Without this framework, every new finding
becomes orphan knowledge invisible to future agents, cathedral autopilot,
and research subagents.

Quick start:

    from tac.canonical_equations import (
        CanonicalEquation,
        EmpiricalAnchor,
        register_canonical_equation,
        update_equation_with_empirical_anchor,
        query_equations,
        query_equations_by_consumer,
        get_equation_by_id,
        populate_initial_equations,
    )

    # Inspect the registry (returns list of latest-payload-per-equation_id)
    for eq in query_equations():
        print(eq.equation_id, eq.is_well_calibrated, eq.predicted_vs_empirical_residual)

    # Add an empirical anchor (e.g., after a Modal smoke lands)
    update_equation_with_empirical_anchor(
        "mps_drift_architecture_class_dependent_v1",
        anchor_for_segnet_class_validation,
    )

The 6 initial equations are documented in ``builtins.py`` + the landing
memo at ``feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md``.

Cross-references:
  * CLAUDE.md "Canonical equations + models registry — non-negotiable"
  * CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable (typed atom discipline)
  * Catalog #344 STRICT preflight gate (refuses new empirical-finding memos
    without equation reference)
  * Catalog #323 canonical Provenance umbrella (every equation + anchor
    carries Provenance)
  * Catalog #125 6-hook wire-in non-negotiable (hook #5 continual-learning
    posterior auto-recalibrates)
  * Catalog #245/#313 canonical 4-layer pattern this registry follows
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CANONICAL_EQUATION_SCHEMA_VERSION,
    RECALIBRATE_ON_NEW_ANCHORS,
    RECALIBRATE_ON_RESIDUAL_DRIFT,
    RECALIBRATE_ON_PARAMETER_REFIT,
    RECALIBRATE_NEVER_AUTO,
    VALID_RECALIBRATION_TRIGGERS,
    CanonicalEquation,
    DomainOfValidityViolation,
    EmpiricalAnchor,
    InvalidEquationError,
)
from tac.canonical_equations.registry import (
    CANONICAL_EQUATIONS_REGISTRY_PATH,
    CANONICAL_EQUATIONS_REGISTRY_LOCK,
    EVENT_ANCHOR_APPENDED,
    EVENT_DEPRECATED,
    EVENT_DOMAIN_REFINED,
    EVENT_RECALIBRATED,
    EVENT_REGISTERED,
    VALID_EVENT_TYPES,
    CanonicalEquationsRegistryCorruptError,
    RecalibrationReport,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
    update_equation_with_domain_refinement,
    query_equations,
    query_equations_by_domain,
    query_equations_by_consumer,
    query_equations_by_producer,
    get_equation_by_id,
    load_equation_registry_strict,
    load_registry_events_lenient,
    auto_recalibrate_from_continual_learning_posterior,
)
from tac.canonical_equations.builtins import (
    build_all_initial_equations,
    populate_initial_equations,
)
from tac.canonical_equations.bayesian_posterior_update import (
    BayesianPosterior,
    DEFAULT_NIG_PRIOR,
    NormalInverseGammaHyperparameters,
    PosteriorUpdateError,
    append_empirical_anchor_to_equation_with_posterior_update,
    bootstrap_posterior_from_anchor_residuals,
    compute_predicted_band_from_posterior,
    update_equation_with_anchor_via_conjugate_prior,
)

__all__ = [
    # Contract
    "CANONICAL_EQUATION_SCHEMA_VERSION",
    "RECALIBRATE_ON_NEW_ANCHORS",
    "RECALIBRATE_ON_RESIDUAL_DRIFT",
    "RECALIBRATE_ON_PARAMETER_REFIT",
    "RECALIBRATE_NEVER_AUTO",
    "VALID_RECALIBRATION_TRIGGERS",
    "CanonicalEquation",
    "DomainOfValidityViolation",
    "EmpiricalAnchor",
    "InvalidEquationError",
    # Registry
    "CANONICAL_EQUATIONS_REGISTRY_PATH",
    "CANONICAL_EQUATIONS_REGISTRY_LOCK",
    "EVENT_REGISTERED",
    "EVENT_ANCHOR_APPENDED",
    "EVENT_RECALIBRATED",
    "EVENT_DEPRECATED",
    "EVENT_DOMAIN_REFINED",
    "VALID_EVENT_TYPES",
    "CanonicalEquationsRegistryCorruptError",
    "RecalibrationReport",
    "register_canonical_equation",
    "update_equation_with_empirical_anchor",
    "update_equation_with_domain_refinement",
    "query_equations",
    "query_equations_by_domain",
    "query_equations_by_consumer",
    "query_equations_by_producer",
    "get_equation_by_id",
    "load_equation_registry_strict",
    "load_registry_events_lenient",
    "auto_recalibrate_from_continual_learning_posterior",
    # Initial population
    "build_all_initial_equations",
    "populate_initial_equations",
    # Bayesian posterior update (SLOT MG-2, Catalog #350)
    "BayesianPosterior",
    "DEFAULT_NIG_PRIOR",
    "NormalInverseGammaHyperparameters",
    "PosteriorUpdateError",
    "append_empirical_anchor_to_equation_with_posterior_update",
    "bootstrap_posterior_from_anchor_residuals",
    "compute_predicted_band_from_posterior",
    "update_equation_with_anchor_via_conjugate_prior",
]
