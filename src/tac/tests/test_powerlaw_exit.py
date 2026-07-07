"""Tests for tac.witness_control.powerlaw_exit — the weak-KAM power-law plateau/exit detector
(solver pack 2026-07-07; equation weak_kam_powerlaw_tail_exit_v1).

Behavior tests (NOT constant tests): synthetic trajectories with KNOWN generative parameters must
be recovered; the model comparison must pick the true model; the exit rule's fail-safe direction
(insufficient data => NOT exhausted) must hold — each would FAIL on a broken/fabricating fit."""
from __future__ import annotations

import math

import pytest

from tac.witness_control.powerlaw_exit import (
    fit_exponential_tail,
    fit_power_law_tail,
    fit_tail_models,
    powerlaw_meat_exit,
    remaining_meat,
)


def _powerlaw_series(a=0.002, b=0.02, alpha=0.5, n=20, step=50):
    eps = [1 + i * step for i in range(n)]
    return eps, [a + b * t ** (-alpha) for t in eps]


def _exp_series(a=0.004, b=0.01, tau=80.0, n=20, step=25):
    eps = [1 + i * step for i in range(n)]
    return eps, [a + b * math.exp(-t / tau) for t in eps]


def test_power_law_alpha_recovered_on_noiseless_synthetic():
    eps, vals = _powerlaw_series(alpha=0.5)
    fit = fit_power_law_tail(eps, vals)
    assert abs(fit.p - 0.5) < 0.02, f"alpha {fit.p} != 0.5"
    assert abs(fit.a - 0.002) < 2e-4
    assert fit.rss < 1e-10


def test_exponential_tau_recovered_on_noiseless_synthetic():
    eps, vals = _exp_series(tau=80.0)
    fit = fit_exponential_tail(eps, vals)
    assert abs(fit.p - 80.0) / 80.0 < 0.05, f"tau {fit.p} != 80"


def test_model_comparison_picks_the_true_model_both_ways():
    eps, vals = _powerlaw_series()
    assert fit_tail_models(eps, vals, n_boot=20)["preferred_model"] == "power_law"
    eps, vals = _exp_series()
    assert fit_tail_models(eps, vals, n_boot=20)["preferred_model"] == "exponential"


def test_remaining_meat_matches_closed_form():
    eps, vals = _powerlaw_series(a=0.002, b=0.02, alpha=0.5)
    fit = fit_power_law_tail(eps, vals)
    t_now, horizon = eps[-1], eps[-1] + 300
    expected = (0.002 + 0.02 * (t_now - eps[0] + 1) ** -0.5) - \
               (0.002 + 0.02 * (horizon - eps[0] + 1) ** -0.5)
    assert abs(remaining_meat(fit, t_now, horizon) - expected) < 1e-6


def test_exit_rule_fail_safe_on_insufficient_points():
    """Confound-L3 discipline: never declare exhaustion on a bad measurement."""
    out = powerlaw_meat_exit({"lane": [(1, 0.01), (2, 0.009)]}, min_points=8)
    assert out["exhausted"] is False
    assert "insufficient" in out["per_class"]["lane"]["reason"]


def test_exit_rule_per_class_binding_class_is_the_meatiest():
    e1, v1 = _powerlaw_series(a=0.001, b=0.05, alpha=0.3)   # slow tail: lots of meat
    e2, v2 = _exp_series(a=0.004, b=0.001, tau=20.0)        # saturated: no meat
    out = powerlaw_meat_exit({"lane": list(zip(e1, v1)), "road": list(zip(e2, v2))},
                             horizon_epochs=300, meat_floor=1e-4, n_boot=20)
    assert out["binding_class"] == "lane"
    assert out["exhausted"] is False        # lane still pays
    assert out["remaining_meat_estimate"] > 1e-4
    assert set(out) >= {"exhausted", "remaining_meat_estimate", "alpha", "ci", "binding_class"}


def test_exit_rule_exhausted_when_all_classes_below_floor():
    e2, v2 = _exp_series(a=0.004, b=0.0001, tau=10.0)  # fully saturated
    out = powerlaw_meat_exit({"road": list(zip(e2, v2))},
                             horizon_epochs=300, meat_floor=1e-4, n_boot=20)
    assert out["exhausted"] is True


def test_bare_list_treated_as_total():
    eps, vals = _powerlaw_series()
    out = powerlaw_meat_exit(list(zip(eps, vals)), n_boot=20)
    assert out["binding_class"] == "total"


def test_intercept_clamped_nonnegative():
    """d_seg >= 0: a rising-then-flat series must not produce a negative asymptote."""
    eps = list(range(1, 40, 2))
    vals = [0.001 + 0.0001 * (t ** -0.2) for t in eps]
    fit = fit_power_law_tail(eps, vals)
    assert fit.a >= 0.0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        fit_tail_models([1, 2, 3], [0.1, 0.2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
