# SPDX-License-Identifier: MIT
"""Tests for the canonical NIG posterior uncertainty helper (SLOT MG-1)."""
from __future__ import annotations

import math

import pytest

from tac.cathedral_consumers.risk_adjusted_ranking_consumer.uncertainty import (
    UncertaintyEstimate,
    predicted_delta_uncertainty_from_empirical_anchors,
)


# Minimal anchor-shaped fixture (duck-typed; no need to import EmpiricalAnchor).
class _FakeAnchor:
    def __init__(self, anchor_id: str, residual: float) -> None:
        self.anchor_id = anchor_id
        self.residual = residual


# ---------------------------------------------------------------------------
# UncertaintyEstimate dataclass invariants
# ---------------------------------------------------------------------------


def test_uncertainty_estimate_happy_path():
    est = UncertaintyEstimate(
        posterior_std=0.005,
        n_anchors_consumed=2,
        method="normal_inverse_gamma_posterior",
        equation_id="eq_test_v1",
        anchor_ids=("a1", "a2"),
    )
    assert est.posterior_std == 0.005
    assert est.n_anchors_consumed == 2
    assert est.method == "normal_inverse_gamma_posterior"
    assert est.equation_id == "eq_test_v1"
    assert est.anchor_ids == ("a1", "a2")


def test_uncertainty_estimate_refuses_negative_std():
    with pytest.raises(ValueError, match="strictly positive"):
        UncertaintyEstimate(
            posterior_std=-0.001,
            n_anchors_consumed=1,
            method="normal_inverse_gamma_posterior",
            equation_id="eq_test_v1",
            anchor_ids=("a1",),
        )


def test_uncertainty_estimate_refuses_zero_std():
    with pytest.raises(ValueError, match="strictly positive"):
        UncertaintyEstimate(
            posterior_std=0.0,
            n_anchors_consumed=1,
            method="normal_inverse_gamma_posterior",
            equation_id="eq_test_v1",
            anchor_ids=("a1",),
        )


def test_uncertainty_estimate_refuses_nan_std():
    with pytest.raises(ValueError, match="must not be NaN"):
        UncertaintyEstimate(
            posterior_std=float("nan"),
            n_anchors_consumed=0,
            method="normal_inverse_gamma_posterior",
            equation_id="eq_test_v1",
            anchor_ids=(),
        )


def test_uncertainty_estimate_refuses_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        UncertaintyEstimate(
            posterior_std=0.01,
            n_anchors_consumed=1,
            method="my_custom_method",
            equation_id="eq_test_v1",
            anchor_ids=("a1",),
        )


def test_uncertainty_estimate_refuses_mismatched_anchor_count():
    with pytest.raises(ValueError, match="must match"):
        UncertaintyEstimate(
            posterior_std=0.01,
            n_anchors_consumed=3,
            method="normal_inverse_gamma_posterior",
            equation_id="eq_test_v1",
            anchor_ids=("a1", "a2"),  # only 2 ids; declared 3
        )


def test_uncertainty_estimate_refuses_empty_equation_id():
    with pytest.raises(ValueError, match="equation_id"):
        UncertaintyEstimate(
            posterior_std=0.01,
            n_anchors_consumed=0,
            method="normal_inverse_gamma_posterior",
            equation_id="",
            anchor_ids=(),
        )


# ---------------------------------------------------------------------------
# predicted_delta_uncertainty_from_empirical_anchors — happy paths
# ---------------------------------------------------------------------------


def test_zero_anchors_returns_pure_prior():
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=[]
    )
    assert est.n_anchors_consumed == 0
    assert est.anchor_ids == ()
    assert est.method == "normal_inverse_gamma_posterior"
    # Pure prior is large by construction (maximum uncertainty).
    assert est.posterior_std > 1.0


def test_single_anchor_returns_finite_posterior():
    anchors = [_FakeAnchor("a1", 0.005)]
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=anchors
    )
    assert est.n_anchors_consumed == 1
    assert est.anchor_ids == ("a1",)
    # Single anchor: NIG posterior is well-defined and finite.
    assert math.isfinite(est.posterior_std)
    assert est.posterior_std > 0.0
    # Uncertainty should be MUCH smaller than the pure-prior case.
    pure_prior = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=[]
    )
    assert est.posterior_std < pure_prior.posterior_std


def test_many_anchors_low_residual_yields_tight_std():
    anchors = [_FakeAnchor(f"a{i}", 0.001) for i in range(10)]
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=anchors
    )
    assert est.n_anchors_consumed == 10
    # 10 tight anchors should yield small posterior std.
    assert est.posterior_std < 0.05


def test_many_anchors_high_residual_yields_wide_std():
    anchors = [_FakeAnchor(f"a{i}", 0.5) for i in range(10)]
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=anchors
    )
    assert est.n_anchors_consumed == 10
    # 10 wide anchors should yield substantial posterior std.
    assert est.posterior_std > 0.05


def test_empirical_bootstrap_path():
    anchors = [_FakeAnchor(f"a{i}", 0.01 * (i + 1)) for i in range(5)]
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=anchors, empirical_bootstrap=True
    )
    assert est.method == "empirical_bootstrap"
    assert est.n_anchors_consumed == 5
    assert est.posterior_std > 0.0


def test_empirical_bootstrap_falls_back_to_nig_at_small_n():
    anchors = [_FakeAnchor("a1", 0.005), _FakeAnchor("a2", 0.006)]
    # Only 2 anchors; bootstrap requires >= 3.
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "eq_test_v1", anchors=anchors, empirical_bootstrap=True
    )
    assert est.method == "normal_inverse_gamma_posterior"


# ---------------------------------------------------------------------------
# predicted_delta_uncertainty_from_empirical_anchors — edge cases / errors
# ---------------------------------------------------------------------------


def test_refuses_empty_equation_id():
    with pytest.raises(ValueError, match="equation_id"):
        predicted_delta_uncertainty_from_empirical_anchors("", anchors=[])


def test_refuses_whitespace_equation_id():
    with pytest.raises(ValueError, match="equation_id"):
        predicted_delta_uncertainty_from_empirical_anchors("   ", anchors=[])


def test_refuses_anchor_missing_residual():
    class _Bad:
        anchor_id = "a1"

    with pytest.raises(ValueError, match="missing .residual"):
        predicted_delta_uncertainty_from_empirical_anchors(
            "eq_test_v1", anchors=[_Bad()]
        )


def test_refuses_anchor_missing_id():
    class _Bad:
        residual = 0.01

    with pytest.raises(ValueError, match="missing .anchor_id"):
        predicted_delta_uncertainty_from_empirical_anchors(
            "eq_test_v1", anchors=[_Bad()]
        )


def test_refuses_nan_residual():
    anchors = [_FakeAnchor("a1", float("nan"))]
    with pytest.raises(ValueError, match="must not be NaN"):
        predicted_delta_uncertainty_from_empirical_anchors(
            "eq_test_v1", anchors=anchors
        )


def test_refuses_non_numeric_residual():
    anchors = [_FakeAnchor("a1", "not a number")]
    with pytest.raises(ValueError, match="must be numeric"):
        predicted_delta_uncertainty_from_empirical_anchors(
            "eq_test_v1", anchors=anchors
        )


def test_canonical_provenance_cite_chain():
    """The helper output carries the equation_id + anchor_ids cite chain."""
    anchors = [
        _FakeAnchor("anchor_modal_a100_2026_05_15", 0.002),
        _FakeAnchor("anchor_lightning_t4_2026_05_16", 0.003),
    ]
    est = predicted_delta_uncertainty_from_empirical_anchors(
        "brotli_cascade_bounded_per_stream_v1", anchors=anchors
    )
    assert est.equation_id == "brotli_cascade_bounded_per_stream_v1"
    assert "anchor_modal_a100_2026_05_15" in est.anchor_ids
    assert "anchor_lightning_t4_2026_05_16" in est.anchor_ids
