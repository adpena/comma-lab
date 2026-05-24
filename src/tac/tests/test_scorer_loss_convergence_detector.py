# SPDX-License-Identifier: MIT
"""Tests for tac.scorer_loss_convergence_detector.

Verifies the OLS-slope plateau detector on synthetic loss curves and
that the safety guards (NaN, monotone-epoch, force_trigger) behave as
designed. Also verifies the canonical re-export from ``tac.training``
resolves to the same class (Lane S V2 spec).
"""
from __future__ import annotations

import math

import pytest

from tac.scorer_loss_convergence_detector import (
    ScorerLossConvergenceDetector,
)

# Lane S V2: the spec requires `tac.training.ScorerLossConvergenceDetector`
# to resolve. Verified here so a future maintainer who deletes the
# re-export gets a loud test failure instead of a silent breakage.
from tac.training import ScorerLossConvergenceDetector as _ReExported


def test_canonical_reexport_from_tac_training_is_same_class():
    assert _ReExported is ScorerLossConvergenceDetector


# ────────────────────────────────────────────────────────────────────────
# Plateau detection
# ────────────────────────────────────────────────────────────────────────


def test_detects_synthetic_plateau_after_warmup():
    """A loss that starts at 1.0, drops linearly for 100 epochs to 0.05,
    then plateaus at 0.05 must be detected within the 50-epoch sliding
    window after the plateau begins."""
    det = ScorerLossConvergenceDetector(
        window=50, slope_tolerance=1e-3, min_warmup_epochs=50,
    )
    # Phase 1 — falling: epoch 0..99 → loss = 1.0 - 0.0095*e
    for e in range(100):
        loss = 1.0 - 0.0095 * e
        assert not det.observe(e, loss), f"premature trigger at e={e}"
    # Phase 2 — flat at 0.05 for 100 more epochs
    triggered = False
    fire_epoch = None
    for e in range(100, 200):
        if det.observe(e, 0.05):
            triggered = True
            fire_epoch = e
            break
    assert triggered, "detector failed to fire on synthetic plateau"
    # Should fire within ~50 epochs (the sliding window length)
    assert fire_epoch is not None and 100 <= fire_epoch <= 160, (
        f"fire_epoch {fire_epoch} outside expected post-plateau range"
    )
    assert det.converged_at == fire_epoch
    assert det.last_slope is not None
    assert abs(det.last_slope) < 1e-3


def test_does_not_fire_during_steep_descent():
    """A loss that keeps falling at 1e-2/epoch should NOT trigger when
    slope_tolerance=1e-4 — tolerance is below the descent rate."""
    det = ScorerLossConvergenceDetector(
        window=50, slope_tolerance=1e-4, min_warmup_epochs=50,
    )
    for e in range(300):
        loss = 5.0 - 1e-2 * e  # steady descent
        assert not det.observe(e, loss), (
            f"detector fired during steep descent at e={e} "
            f"(slope ≈ -1e-2 ≫ tolerance 1e-4)"
        )


def test_min_warmup_epochs_floor_prevents_early_fire():
    """Even if the loss is constant from epoch 0, the detector must NOT
    fire before min_warmup_epochs."""
    det = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=200,
    )
    # Constant loss — slope = 0, which is < tolerance.
    for e in range(150):
        assert not det.observe(e, 0.5), (
            f"detector fired at e={e} but min_warmup_epochs=200"
        )
    # Once we cross the floor, the detector should fire on the next obs.
    for e in range(150, 250):
        triggered = det.observe(e, 0.5)
        if e >= 200 and triggered:
            assert det.converged_at >= 200
            return
    pytest.fail("detector never fired even after min_warmup_epochs floor")


def test_require_decreasing_blocks_rising_curve():
    """A loss that has low slope tolerance but is RISING (e.g. mid-
    divergence) must not trigger when require_decreasing=True."""
    det = ScorerLossConvergenceDetector(
        window=50, slope_tolerance=1e-2, min_warmup_epochs=50,
        require_decreasing=True,
    )
    # Rising loss at +5e-3/epoch (within tolerance 1e-2 but positive)
    for e in range(200):
        loss = 0.1 + 5e-3 * e
        assert not det.observe(e, loss), (
            f"detector fired on rising loss at e={e} "
            f"(slope ≈ +5e-3, sign should block)"
        )


def test_require_decreasing_false_allows_rising_within_tolerance():
    det = ScorerLossConvergenceDetector(
        window=50, slope_tolerance=1e-2, min_warmup_epochs=50,
        require_decreasing=False,
    )
    triggered = False
    for e in range(200):
        loss = 0.1 + 5e-3 * e
        if det.observe(e, loss):
            triggered = True
            break
    assert triggered, "detector with require_decreasing=False should fire"


# ────────────────────────────────────────────────────────────────────────
# Edge cases / safety guards
# ────────────────────────────────────────────────────────────────────────


def test_monotone_increasing_epoch_required():
    det = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=0,
    )
    det.observe(5, 0.5)
    with pytest.raises(ValueError, match="strictly greater"):
        det.observe(5, 0.5)
    with pytest.raises(ValueError, match="strictly greater"):
        det.observe(3, 0.5)


def test_nan_observation_skipped():
    det = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=0,
    )
    det.observe(0, 0.5)
    assert not det.observe(1, float("nan"))
    # The window should NOT include the NaN.
    assert len(det._losses) == 1


def test_inf_observation_skipped():
    det = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=0,
    )
    det.observe(0, 0.5)
    assert not det.observe(1, float("inf"))
    assert len(det._losses) == 1


def test_force_trigger_overrides_natural_detection():
    det = ScorerLossConvergenceDetector(
        window=50, slope_tolerance=1e-4, min_warmup_epochs=50,
    )
    # Steep descent — would never naturally fire.
    for e in range(100):
        det.observe(e, 5.0 - 0.05 * e)
    assert det.converged_at is None
    det.force_trigger(epoch=99)
    assert det.converged_at == 99
    # Subsequent observe() returns True (trigger sticky).
    assert det.observe(100, 0.0) is True


def test_state_dict_round_trip_preserves_history():
    det = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=0,
    )
    for e in range(15):
        det.observe(e, 0.5 - 0.001 * e)
    state = det.state_dict()
    recovered = ScorerLossConvergenceDetector(
        window=10, slope_tolerance=1e-3, min_warmup_epochs=0,
    )
    recovered.load_state_dict(state)
    # Adding the same next observation should produce the same slope on both.
    a = det.observe(15, 0.485)
    b = recovered.observe(15, 0.485)
    assert a == b
    assert math.isclose(det.last_slope, recovered.last_slope, rel_tol=1e-12)


def test_invalid_construction_raises():
    with pytest.raises(ValueError, match="window"):
        ScorerLossConvergenceDetector(window=2)
    with pytest.raises(ValueError, match="slope_tolerance"):
        ScorerLossConvergenceDetector(slope_tolerance=0)
    with pytest.raises(ValueError, match="min_warmup_epochs"):
        ScorerLossConvergenceDetector(min_warmup_epochs=-1)
