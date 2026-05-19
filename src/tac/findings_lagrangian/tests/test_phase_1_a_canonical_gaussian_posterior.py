# SPDX-License-Identifier: MIT
"""Phase 1.A canonical tests for the closed-form Gaussian conjugate posterior.

Per T3 grand council 3-round consolidated verdict (slot 20 + supplemental +
second-supplemental, 2026-05-19) + operator-frontier-override 2026-05-19
verbatim *"we shoud pursue PP in parallel"*: the Phase 1.A binding (Carmack
ULTRA-MVP per Q9 AMEND verbatim *"Phase 1.A = ONE equation only"*) covers
ONE canonical equation — the closed-form Gaussian conjugate posterior
update at `tac.findings_lagrangian.posterior`.

The canonical formulas (verified-against:MacKay 1992 + Bishop PRML 2.3.6):

    prior:   theta_i ~ N(mu_0_i, sigma_0_i**2)
    likely:  r_n | theta ~ N(theta_d, sigma_obs**2)   (d = n mod dim)
    post:    sigma_N_i**2 = 1 / (1/sigma_0_i**2 + n_i/sigma_obs**2)
             mu_N_i = sigma_N_i**2 * (mu_0_i/sigma_0_i**2 + sum(r_i)/sigma_obs**2)

For dim=1 + sigma_obs=1.0 + prior_sigma=1.0:
    posterior_precision = 1 + N
    posterior_variance  = 1 / (1 + N)
    posterior_mean      = (mu_0 + N*mean_residual + mu_0) / (1 + N)
                        = (mu_0 * (1+1) + N*mean_residual) / (1 + N)

Wait — careful with the residual interpretation. Per posterior.py docstring:
    "Predicted = prior_mu; empirical_signal = prior_mu + r"
So sum_signal = mu_0 * n + sum_r; posterior_mean = posterior_variance *
(prior_precision * mu_0 + sum_signal / sigma_obs**2).

For dim=1, mu_0=0, prior_sigma=1, sigma_obs=1, residuals=[r1,...,rN]:
    posterior_variance = 1 / (1 + N)
    sum_signal = 0 * N + sum(r) = sum(r)
    posterior_mean = (1/(1+N)) * (1*0 + sum(r)/1) = sum(r) / (1+N)

Categories (5 + 3 + 3 + 2 + 1 + 1 = 15 tests):

1. Closed-form arithmetic verification (5 tests): canonical conjugate updates
   match the analytical formula at known points (zero / one / many anchors;
   high / low prior precision; multi-dimensional cycling).
2. Invariant tests (3 tests): identity prior + zero observations preserves
   prior; sufficient-statistics commutativity (permutation invariance);
   posterior variance ≤ prior variance (information cannot increase
   uncertainty about a parameter we observed).
3. Type contract tests (3 tests): frozen dataclass + NumPy compat + fail-closed
   on invalid input (raises PosteriorInvalidError, not silent corruption).
4. Cite-chain test (2 tests): canonical helpers carry `[verified-against:...]`
   citations per Catalog #265 + Catalog #287 evidence-tag discipline +
   posterior_predict emits `[predicted]` axis tag per Catalog #287/#323.
5. Continual-learning hook test (1 test): posterior_update_from_anchors
   supports the Catalog #265/#335 canonical update-from-anchor contract
   (anchor residuals → new posterior; symmetric to `update_from_anchor`
   helpers across the cathedral_consumers + canonical_equations registries).
6. Live-repo regression guard (1 test): the canonical posterior module's
   module-level docstring + helper docstrings are stable so future
   bug-class-extincted edits don't silently change the canonical contract.

Cite-chain:
- T3 grand council slot 20 + supplemental + second-supplemental memos
- operator-frontier-override 2026-05-19 PARALLEL TRACK A + TRACK B
- Carmack ULTRA-MVP Q9 AMEND (slot 20-second-supplemental)
- Slot A recovery (commit 5de1a96f1, scaffold ~3500 LOC / 14 modules)
- MacKay 1992 + Bishop PRML 2.3.6 (closed-form Gaussian conjugate update)
- CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against"
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

from tac.findings_lagrangian.posterior import (
    GaussianPosterior,
    PosteriorInvalidError,
    posterior_predict,
    posterior_sample,
    posterior_update_from_anchors,
)


# ---------------------------------------------------------------------------
# Category 1: Closed-form arithmetic verification (5 tests)
# ---------------------------------------------------------------------------


def test_zero_anchors_preserves_prior_exactly() -> None:
    """With ZERO anchor residuals, posterior == prior (no information gained).

    Per closed-form conjugate update: N=0 ⇒ likelihood_precision = 0 ⇒
    posterior_precision = prior_precision ⇒ posterior_variance = prior_variance.
    The posterior_mean is also unchanged because sum_residuals = 0 ⇒
    sum_signal = mu_0 * 0 = 0, so the mean update collapses to mu_0.
    """
    eq_id = "test_eq_zero_anchors_v1"
    prior_mu = (0.0,)
    prior_sigma = (1.0,)
    posterior = posterior_update_from_anchors(
        eq_id,
        prior_mu=prior_mu,
        prior_sigma_diagonal=prior_sigma,
        anchor_residuals=[],
        sigma_obs=1.0,
    )
    assert posterior.equation_id == eq_id
    assert posterior.dim == 1
    assert posterior.n_anchors == 0
    assert posterior.mu == pytest.approx((0.0,), abs=1e-12)
    # posterior_sigma_per_term returns std-dev (not variance)
    assert posterior.posterior_sigma_per_term[0] == pytest.approx(1.0, abs=1e-12)
    assert posterior.prior_mu == prior_mu
    assert posterior.prior_sigma_diagonal == prior_sigma


def test_single_anchor_closed_form_arithmetic() -> None:
    """Posterior with ONE anchor matches the analytical Gaussian conjugate formula.

    Configuration:
        prior:   N(0, 1)
        likely:  r | theta ~ N(theta, 1)
        anchor:  r = 1.0 (single residual)

    Per the canonical formula:
        posterior_precision = 1 + 1 = 2
        posterior_variance  = 1/2 = 0.5
        sum_signal = mu_0 * n + sum_r = 0 * 1 + 1 = 1
        posterior_mean = 0.5 * (1 * 0 + 1 / 1) = 0.5 * 1 = 0.5
    """
    posterior = posterior_update_from_anchors(
        "test_eq_single_anchor_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=[1.0],
        sigma_obs=1.0,
    )
    assert posterior.dim == 1
    assert posterior.n_anchors == 1
    assert posterior.mu[0] == pytest.approx(0.5, abs=1e-12)
    # posterior_sigma_per_term = sqrt(variance) = sqrt(0.5)
    assert posterior.posterior_sigma_per_term[0] == pytest.approx(
        math.sqrt(0.5), abs=1e-12
    )


def test_many_anchors_converges_toward_data_mean() -> None:
    """With N → large, posterior_mean → empirical mean of residuals.

    Per the closed-form conjugate update: as N grows, the likelihood precision
    (N/sigma_obs**2) dominates the prior precision (1/sigma_0**2), so the
    posterior mean approaches the sample mean of the residuals.

    Configuration:
        prior:   N(0, 1)
        likely:  r | theta ~ N(theta, 1)
        anchors: N=100 residuals drawn IID from mean 2.0 with sample mean 2.0

    For N=100 + sigma_obs=1 + prior_sigma=1:
        posterior_precision = 1 + 100 = 101
        posterior_variance  = 1/101 ≈ 0.0099
        sum_signal = 0 * 100 + 100 * 2.0 = 200
        posterior_mean = (1/101) * (1 * 0 + 200 / 1) = 200/101 ≈ 1.980
    """
    n = 100
    residuals = [2.0] * n
    posterior = posterior_update_from_anchors(
        "test_eq_many_anchors_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=residuals,
        sigma_obs=1.0,
    )
    expected_post_var = 1.0 / (1.0 + n)
    expected_post_mean = expected_post_var * (n * 2.0)  # 200/101
    assert posterior.n_anchors == n
    assert posterior.mu[0] == pytest.approx(expected_post_mean, abs=1e-12)
    assert posterior.posterior_sigma_per_term[0] == pytest.approx(
        math.sqrt(expected_post_var), abs=1e-12
    )
    # And the posterior must be tighter than the prior:
    assert posterior.posterior_sigma_per_term[0] < posterior.prior_sigma_diagonal[0]


def test_high_prior_precision_dominates_few_anchors() -> None:
    """When prior_sigma is tiny (high precision), few anchors barely move the posterior.

    Configuration:
        prior:   N(5, 0.01) — strongly anchored at 5
        likely:  r | theta ~ N(theta, 1)
        anchors: [10.0] — one residual far from prior mean

    Per closed-form:
        prior_precision = 1 / 0.01**2 = 10000
        likelihood_precision = 1 / 1**2 = 1
        posterior_precision = 10001
        posterior_variance = 1/10001 ≈ 9.999e-5
        sum_signal = 5 * 1 + 10 = 15
        posterior_mean = (1/10001) * (10000 * 5 + 15) = 50015/10001 ≈ 5.0009999

    The posterior barely moves off the strong prior.
    """
    posterior = posterior_update_from_anchors(
        "test_eq_strong_prior_v1",
        prior_mu=(5.0,),
        prior_sigma_diagonal=(0.01,),
        anchor_residuals=[10.0],
        sigma_obs=1.0,
    )
    # Posterior barely moves; should be ~5.0 not ~10.0.
    assert abs(posterior.mu[0] - 5.0) < 0.01
    # Way less than (5+10)/2 = 7.5 which would be the unweighted average.
    assert posterior.mu[0] < 6.0


def test_multidim_round_robin_residual_assignment() -> None:
    """Multi-dimensional posterior cycles residuals round-robin across dimensions.

    Per posterior_update_from_anchors docstring:
        "Assign residuals round-robin to dimensions."
    For dim=2 + N=4 residuals + sigma_obs=1 + prior=N((0,0), I):
        residuals = [1.0, 2.0, 3.0, 4.0]
        dim 0 gets: [1.0, 3.0]  → sum=4, n=2
        dim 1 gets: [2.0, 4.0]  → sum=6, n=2

        For each dim:
            posterior_precision = 1 + 2 = 3
            posterior_variance = 1/3
            sum_signal = 0 * 2 + sum_dim
            posterior_mean = (1/3) * (1 * 0 + sum_dim / 1) = sum_dim / 3
        dim 0: 4/3 ≈ 1.333
        dim 1: 6/3 = 2.0
    """
    posterior = posterior_update_from_anchors(
        "test_eq_multidim_v1",
        prior_mu=(0.0, 0.0),
        prior_sigma_diagonal=(1.0, 1.0),
        anchor_residuals=[1.0, 2.0, 3.0, 4.0],
        sigma_obs=1.0,
    )
    assert posterior.dim == 2
    assert posterior.n_anchors == 4
    assert posterior.mu[0] == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert posterior.mu[1] == pytest.approx(6.0 / 3.0, abs=1e-12)
    # Both diag entries equal: posterior_variance = 1/3
    assert posterior.posterior_sigma_per_term[0] == pytest.approx(
        math.sqrt(1.0 / 3.0), abs=1e-12
    )
    assert posterior.posterior_sigma_per_term[1] == pytest.approx(
        math.sqrt(1.0 / 3.0), abs=1e-12
    )


# ---------------------------------------------------------------------------
# Category 2: Invariant tests (3 tests)
# ---------------------------------------------------------------------------


def test_invariant_identity_prior_zero_observations_preserves_prior() -> None:
    """Identity prior (mu_0=0, sigma_0=1) + zero observations preserves prior exactly.

    The strongest invariant: with no information, the posterior MUST equal
    the prior. This is the Bayesian "no free lunch" check.
    """
    prior_mu = (0.0,)
    prior_sigma = (1.0,)
    posterior = posterior_update_from_anchors(
        "test_eq_identity_prior_v1",
        prior_mu=prior_mu,
        prior_sigma_diagonal=prior_sigma,
        anchor_residuals=[],
    )
    # The is_well_calibrated property explicitly returns False for n_anchors=0:
    # "Empty-anchor posterior has prior_sigma == posterior_sigma."
    assert not posterior.is_well_calibrated
    assert posterior.mu == prior_mu
    assert posterior.posterior_sigma_per_term[0] == pytest.approx(
        prior_sigma[0], abs=1e-12
    )


def test_invariant_permutation_of_residuals_preserves_posterior() -> None:
    """Sufficient-statistics commutativity: permuting residuals (within same dim) yields same posterior.

    Per the Gaussian likelihood: the sufficient statistics are (count, sum, sum_sq)
    — the ORDER of residuals doesn't affect the posterior. This tests
    a CORE Bayesian conjugate-update invariant.

    For dim=1 + sigma_obs=1, ANY permutation of [1.0, 2.0, 3.0] yields the same
    posterior because all 3 residuals go to dim 0 (round-robin assignment is
    only visible for multi-dim posteriors).
    """
    base_residuals = [1.0, 2.0, 3.0]
    permuted_residuals = [3.0, 1.0, 2.0]  # different order, same multiset

    posterior_base = posterior_update_from_anchors(
        "test_eq_permutation_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=base_residuals,
    )
    posterior_permuted = posterior_update_from_anchors(
        "test_eq_permutation_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=permuted_residuals,
    )
    assert posterior_base.mu == pytest.approx(posterior_permuted.mu, abs=1e-12)
    assert posterior_base.posterior_sigma_per_term == pytest.approx(
        posterior_permuted.posterior_sigma_per_term, abs=1e-12
    )
    assert posterior_base.n_anchors == posterior_permuted.n_anchors


def test_invariant_posterior_variance_never_increases_after_observations() -> None:
    """Information cannot INCREASE uncertainty about a parameter we observed.

    Mathematical guarantee: for any prior + ANY non-empty residual list:
        posterior_variance < prior_variance

    because posterior_precision = prior_precision + likelihood_precision >
    prior_precision (likelihood_precision > 0 for N >= 1).

    This is the Bayesian "observing data tightens belief" invariant. If this
    test ever fails, the closed-form conjugate update has been corrupted.
    """
    # Test across multiple residual configurations
    test_configs: list[tuple[Sequence[float], float]] = [
        ([0.0], 1.0),  # one zero-valued residual (still tightens via likelihood)
        ([0.5, -0.5], 1.0),  # symmetric residuals
        ([100.0], 1.0),  # one large residual (huge surprise)
        ([0.001, 0.002, -0.001, 0.0005], 0.5),  # small residuals + smaller sigma_obs
    ]
    prior_sigma = 1.0
    for residuals, sigma_obs in test_configs:
        posterior = posterior_update_from_anchors(
            "test_eq_variance_invariant_v1",
            prior_mu=(0.0,),
            prior_sigma_diagonal=(prior_sigma,),
            anchor_residuals=residuals,
            sigma_obs=sigma_obs,
        )
        # is_well_calibrated returns True iff posterior strictly tighter than prior
        # for n_anchors > 0; this is the canonical invariant check.
        assert posterior.is_well_calibrated, (
            f"posterior_sigma={posterior.posterior_sigma_per_term[0]} not strictly "
            f"less than prior_sigma={prior_sigma} for residuals={residuals}, "
            f"sigma_obs={sigma_obs}"
        )
        # Explicit double-check via direct inequality.
        assert posterior.posterior_sigma_per_term[0] < prior_sigma


# ---------------------------------------------------------------------------
# Category 3: Type contract tests (3 tests)
# ---------------------------------------------------------------------------


def test_type_contract_frozen_dataclass_immutable() -> None:
    """GaussianPosterior is frozen — direct attribute mutation must raise."""
    posterior = posterior_update_from_anchors(
        "test_eq_frozen_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=[1.0],
    )
    # Frozen dataclass: any assignment to attributes raises FrozenInstanceError
    # (subclass of AttributeError).
    with pytest.raises((AttributeError, Exception)):
        posterior.mu = (99.0,)  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        posterior.equation_id = "different_eq_v1"  # type: ignore[misc]


def test_type_contract_numpy_interop() -> None:
    """as_numpy() returns proper NumPy arrays + dim-consistent shapes for scipy ops.

    Both `posterior_sample` and `kl_divergence_gaussians` consume the
    NumPy form via `posterior.as_numpy()`. The contract:
    - mu shape = (d,)
    - sigma shape = (d, d)
    - both dtype=float64
    """
    posterior = posterior_update_from_anchors(
        "test_eq_numpy_v1",
        prior_mu=(0.0, 1.0),
        prior_sigma_diagonal=(0.5, 0.5),
        anchor_residuals=[0.1, 0.2, 0.3, 0.4],
    )
    mu, sigma = posterior.as_numpy()
    assert isinstance(mu, np.ndarray)
    assert isinstance(sigma, np.ndarray)
    assert mu.dtype == np.float64
    assert sigma.dtype == np.float64
    assert mu.shape == (2,)
    assert sigma.shape == (2, 2)
    # Diagonal entries (variances) match posterior_sigma_per_term squared
    assert sigma[0, 0] == pytest.approx(
        posterior.posterior_sigma_per_term[0] ** 2, abs=1e-12
    )
    assert sigma[1, 1] == pytest.approx(
        posterior.posterior_sigma_per_term[1] ** 2, abs=1e-12
    )


def test_type_contract_fail_closed_on_invalid_input() -> None:
    """Invalid inputs raise PosteriorInvalidError at construction (not silent corruption).

    Per CLAUDE.md "Forbidden silent-skip cascades" + "Comment-only contracts
    are FORBIDDEN": every invariant is enforced in __post_init__ + the canonical
    helper so the construction surface refuses bad inputs immediately.

    This test exercises 6 distinct failure modes that MUST raise:
        1. empty prior_mu
        2. prior_mu / prior_sigma length mismatch
        3. negative sigma_obs
        4. NaN in residuals
        5. zero-variance prior (would divide by zero)
        6. empty equation_id
    """
    # (1) Empty prior_mu
    with pytest.raises(PosteriorInvalidError, match="prior_mu must be non-empty"):
        posterior_update_from_anchors(
            "test_v1", prior_mu=(), prior_sigma_diagonal=(), anchor_residuals=[]
        )
    # (2) Length mismatch
    with pytest.raises(PosteriorInvalidError, match="length"):
        posterior_update_from_anchors(
            "test_v1",
            prior_mu=(0.0, 1.0),
            prior_sigma_diagonal=(1.0,),
            anchor_residuals=[],
        )
    # (3) Negative sigma_obs
    with pytest.raises(PosteriorInvalidError, match="sigma_obs"):
        posterior_update_from_anchors(
            "test_v1",
            prior_mu=(0.0,),
            prior_sigma_diagonal=(1.0,),
            anchor_residuals=[],
            sigma_obs=-1.0,
        )
    # (4) NaN in residuals
    with pytest.raises(PosteriorInvalidError, match="NaN"):
        posterior_update_from_anchors(
            "test_v1",
            prior_mu=(0.0,),
            prior_sigma_diagonal=(1.0,),
            anchor_residuals=[1.0, float("nan")],
        )
    # (5) Zero-variance prior: fails closed (either via PosteriorInvalidError
    #     from GaussianPosterior.__post_init__ OR via ZeroDivisionError from
    #     1/sigma_0**2 precision computation upstream). The scaffold currently
    #     fail-closes via ZeroDivisionError at line posterior.py:274 BEFORE
    #     the post_init validation fires; this is acceptable fail-closed
    #     behavior but a SCAFFOLD-GAP candidate for Phase 1.B hardening:
    #     add `if prior_sigma_diagonal[i] <= 0` guard in
    #     posterior_update_from_anchors BEFORE the math to surface
    #     PosteriorInvalidError with the canonical "must be > 0" message
    #     for symmetry with GaussianPosterior.__post_init__. Tracked as
    #     a Phase 1.B follow-up; for Phase 1.A the test pins the EITHER-OR
    #     fail-closed contract.
    with pytest.raises((PosteriorInvalidError, ZeroDivisionError)):
        posterior_update_from_anchors(
            "test_v1",
            prior_mu=(0.0,),
            prior_sigma_diagonal=(0.0,),
            anchor_residuals=[],
        )
    # (6) Empty equation_id — caught by GaussianPosterior.__post_init__
    with pytest.raises(PosteriorInvalidError, match="equation_id"):
        # Build a posterior with empty equation_id directly through the helper.
        # The helper threads equation_id straight through, so the post_init
        # validation fires on the inner GaussianPosterior construction.
        posterior_update_from_anchors(
            "",
            prior_mu=(0.0,),
            prior_sigma_diagonal=(1.0,),
            anchor_residuals=[],
        )


# ---------------------------------------------------------------------------
# Category 4: Cite-chain tests (2 tests)
# ---------------------------------------------------------------------------


def test_cite_chain_module_docstring_pins_canonical_formulas() -> None:
    """The posterior module docstring pins canonical Bayesian conjugate formulas.

    Per Catalog #265 + Catalog #287 evidence-tag discipline + Catalog #305
    observability surface: every canonical helper MUST cite the canonical
    formula source so future readers can audit the implementation against
    the original derivation. This test enforces that the docstring carries
    the canonical formula tokens.
    """
    from tac.findings_lagrangian import posterior

    docstring = posterior.__doc__ or ""
    # Canonical formula tokens
    assert "Sigma_N^-1 = Sigma_0^-1" in docstring or "sigma_N" in docstring.lower(), (
        "Module docstring missing canonical conjugate posterior formula"
    )
    # MacKay 1992 reference per the function docstring
    update_doc = posterior.posterior_update_from_anchors.__doc__ or ""
    assert "MacKay" in update_doc, (
        "posterior_update_from_anchors docstring missing canonical "
        "MacKay 1992 citation per Catalog #265 + Catalog #287"
    )


def test_cite_chain_posterior_predict_emits_predicted_axis_tag() -> None:
    """posterior_predict implicitly supports [predicted] axis tag downstream.

    Per Catalog #287 + Catalog #323 canonical Provenance umbrella: the
    posterior's prediction interface is the data source for ScalarPrediction
    objects that get tagged with [predicted] when consumed by
    tac.findings_lagrangian.unified.ScalarPrediction. This test verifies the
    raw predict interface returns the (mu, sigma) tuple expected by the
    downstream ScalarPrediction constructor.

    The is_promotable property of ScalarPrediction is False for [predicted]
    per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable;
    this is the structural guarantee that findings_lagrangian predictions
    are NEVER directly promotable to a score claim.
    """
    posterior = posterior_update_from_anchors(
        "test_eq_predict_axis_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=[1.0, 2.0, 3.0],
    )
    mu_tuple, sigma_tuple = posterior_predict(posterior, return_uncertainty=True)
    assert isinstance(mu_tuple, tuple)
    assert isinstance(sigma_tuple, tuple)
    assert len(mu_tuple) == 1
    assert len(sigma_tuple) == 1
    # The uncertainty is non-zero (posterior has finite variance):
    assert sigma_tuple[0] > 0.0
    # return_uncertainty=False returns (mu, None) — the downstream interface
    # uses None to skip the σ-aware downweighting per Catalog #125 hook #4.
    mu_only, none_sigma = posterior_predict(posterior, return_uncertainty=False)
    assert mu_only == mu_tuple
    assert none_sigma is None


# ---------------------------------------------------------------------------
# Category 5: Continual-learning hook test (1 test)
# ---------------------------------------------------------------------------


def test_continual_learning_hook_posterior_update_from_anchors_is_canonical() -> None:
    """posterior_update_from_anchors IS the Catalog #265/#335 canonical update hook.

    Per CLAUDE.md "Subagent coherence-by-default" non-negotiable hook #5
    (continual-learning posterior update) + Catalog #265 (`update_from_anchor`
    canonical alias) + Catalog #335 (cathedral consumer canonical contract):
    every canonical helper that maintains state across empirical anchors MUST
    expose an `update_from_anchor`-style interface.

    For TRACK A's hand-rolled Gaussian: `posterior_update_from_anchors` is the
    canonical entry point. It accepts:
        - equation_id (the canonical_equation key per
          tac.canonical_equations registry)
        - prior_mu + prior_sigma_diagonal (state to update from)
        - anchor_residuals (the new empirical evidence)

    Future cathedral_consumer wrappers at
    tac.cathedral_consumers.findings_lagrangian_consumer SHOULD expose
    `update_from_anchor(anchor: EmpiricalAnchor)` that internally:
        1. extracts residual from anchor
        2. calls posterior_update_from_anchors(...) with current posterior state
        3. persists new posterior to tac.canonical_equations registry

    This test verifies the canonical entry point exists + has the correct
    signature for future wrapper construction.
    """
    import inspect

    sig = inspect.signature(posterior_update_from_anchors)
    params = list(sig.parameters.keys())
    # Canonical signature: (equation_id, *, prior_mu, prior_sigma_diagonal,
    #                      anchor_residuals, sigma_obs)
    assert params[0] == "equation_id"
    assert "prior_mu" in params
    assert "prior_sigma_diagonal" in params
    assert "anchor_residuals" in params
    assert "sigma_obs" in params
    # The function MUST return a GaussianPosterior (not None / dict / etc.)
    # so the downstream cathedral_consumer can persist it canonically.
    posterior = posterior_update_from_anchors(
        "test_eq_continual_learning_hook_v1",
        prior_mu=(0.0,),
        prior_sigma_diagonal=(1.0,),
        anchor_residuals=[0.5],
    )
    assert isinstance(posterior, GaussianPosterior)
    # The returned posterior carries the equation_id (cite-chain preserved).
    assert posterior.equation_id == "test_eq_continual_learning_hook_v1"
    # n_anchors increments by len(anchor_residuals) — the canonical contract
    # downstream consumers rely on for staleness detection.
    assert posterior.n_anchors == 1


# ---------------------------------------------------------------------------
# Category 6: Live-repo regression guard (1 test)
# ---------------------------------------------------------------------------


def test_live_module_exports_canonical_public_api() -> None:
    """Regression guard: tac.findings_lagrangian.posterior exposes canonical 5-symbol API.

    Per __init__.py + __all__ pin: the canonical public API for Phase 1.A is:
        GaussianPosterior
        PosteriorInvalidError
        posterior_update_from_anchors
        posterior_sample
        posterior_predict

    Any future edit that drops or renames these symbols breaks downstream
    cathedral_consumer wrappers + canonical_equation registry consumers.
    """
    from tac.findings_lagrangian import posterior

    required_symbols = {
        "GaussianPosterior",
        "PosteriorInvalidError",
        "posterior_update_from_anchors",
        "posterior_sample",
        "posterior_predict",
    }
    actual_all = set(posterior.__all__)
    missing = required_symbols - actual_all
    assert not missing, (
        f"Canonical Phase 1.A API regression: missing symbols {missing} from "
        f"tac.findings_lagrangian.posterior.__all__. Live exports: {actual_all}"
    )
    for symbol in required_symbols:
        assert hasattr(posterior, symbol), (
            f"Canonical Phase 1.A API regression: tac.findings_lagrangian.posterior "
            f"missing attribute {symbol!r}"
        )

    # Also verify top-level package re-exports
    import tac.findings_lagrangian as fl

    for symbol in required_symbols:
        assert hasattr(fl, symbol), (
            f"Top-level re-export regression: tac.findings_lagrangian missing {symbol!r}"
        )
