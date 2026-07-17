# SPDX-License-Identifier: MIT
"""Tests for ema_decay_run_geometry_v1 (ARM-C p0_ema_calibration; SPEC_v10 §13.3)."""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ema_decay_run_geometry_20260717 import (
    EQUATION_ID,
    build_ema_decay_run_geometry_v1,
)
from tac.canonical_equations.evaluators import (
    EvaluatorError,
    eval_ema_decay_run_geometry,
    populate_lawref_evaluators,
    resolve_equation_value,
)


def _ev(**kw):
    return eval_ema_decay_run_geometry(kw)


# ---------------------------------------------------------------- the closed forms
def test_decay_from_seed_fraction_exact_inverse():
    d = _ev(mode="decay_from_seed_fraction", updates_per_run=749, target_seed_fraction=0.01)
    assert d == pytest.approx(0.01 ** (1 / 749))
    # round trip: seed fraction of the derived decay is the pinned target.
    eps = _ev(mode="seed_fraction_from_decay", updates_per_run=749, ema_decay=d)
    assert eps == pytest.approx(0.01, rel=1e-9)


def test_decay_from_warmup_fraction_exact_inverse():
    d = _ev(mode="decay_from_warmup_fraction", updates_per_run=1400, warmup_fraction=0.5)
    assert d == pytest.approx(1.0 - 2.0 / (0.5 * 1400))
    phi = _ev(mode="warmup_fraction_from_decay", updates_per_run=1400, ema_decay=d)
    assert phi == pytest.approx(0.5, rel=1e-9)


def test_incumbent_0997_measured_basis_reproduced():
    # SPEC_v10 §13.3 numbers: warmup fraction of the c2 warm window; ~64% seed @ep800.
    phi = _ev(mode="warmup_fraction_from_decay", updates_per_run=749, ema_decay=0.997)
    assert phi == pytest.approx(2.0 / (0.003 * 749), rel=1e-9)
    assert phi > 0.85  # the shadow spends nearly the whole warm window inside warmup
    seed_800 = _ev(mode="seed_fraction_from_decay", updates_per_run=149, ema_decay=0.997)
    assert seed_800 == pytest.approx(0.639, abs=0.002)   # the recorded ~64%
    # warmup updates identity vs the registered helper: 2/(1-d) == 666.67 -> ceil 667.
    from tac.confound_observability import ema_warmup_updates
    assert ema_warmup_updates(0.997) == 667
    assert math.ceil(2.0 / (1.0 - 0.997)) == 667


def test_mode_numeric_codes_match_names():
    for code, name, extra in (
        (1, "decay_from_seed_fraction", {"target_seed_fraction": 0.05}),
        (2, "decay_from_warmup_fraction", {"warmup_fraction": 0.4}),
        (3, "warmup_fraction_from_decay", {"ema_decay": 0.99}),
        (4, "seed_fraction_from_decay", {"ema_decay": 0.99}),
    ):
        a = _ev(mode=code, updates_per_run=500, **extra)
        b = _ev(mode=name, updates_per_run=500, **extra)
        assert a == b


# ---------------------------------------------------------------- fail-closed domain guards
def test_rejects_nonpositive_updates():
    with pytest.raises(EvaluatorError):
        _ev(mode="seed_fraction_from_decay", updates_per_run=0, ema_decay=0.99)


def test_rejects_out_of_range_seed_fraction():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(EvaluatorError):
            _ev(mode="decay_from_seed_fraction", updates_per_run=100, target_seed_fraction=bad)


def test_rejects_infeasible_warmup_fraction():
    # phi*U <= 2 would give d <= 0 (no valid EMA decay).
    with pytest.raises(EvaluatorError):
        _ev(mode="decay_from_warmup_fraction", updates_per_run=10, warmup_fraction=0.2)


def test_rejects_unknown_mode_and_bad_decay():
    with pytest.raises(EvaluatorError):
        _ev(mode="nope", updates_per_run=10)
    with pytest.raises(EvaluatorError):
        _ev(mode="seed_fraction_from_decay", updates_per_run=10, ema_decay=1.0)


# ---------------------------------------------------------------- registration + builder
def test_evaluator_registered_for_lawref():
    populate_lawref_evaluators()
    v = resolve_equation_value(
        EQUATION_ID,
        {"mode": 1, "updates_per_run": 749, "target_seed_fraction": 0.01})
    assert v == pytest.approx(0.01 ** (1 / 749))


def test_builder_constructs_canonical_equation():
    eq = build_ema_decay_run_geometry_v1()
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 2
    assert eq.domain_of_validity["score_claim"] is False
    assert eq.domain_of_validity["research_only"] is True
    # anchors carry the measured basis, not guesses
    a = {x.anchor_id: x for x in eq.empirical_anchors}
    assert a["ema_decay_run_geometry_warmup_667_20260717"].empirical_output[
        "warmup_updates_registered"] == 667
    assert a["ema_decay_run_geometry_seed_fraction_20260717"].empirical_output[
        "seed_fraction_ep800"] == pytest.approx(0.6391, abs=1e-4)


def test_builder_residual_small_vs_spec_recorded_64pct():
    eq = build_ema_decay_run_geometry_v1()
    a = {x.anchor_id: x for x in eq.empirical_anchors}
    assert a["ema_decay_run_geometry_seed_fraction_20260717"].residual < 0.01
