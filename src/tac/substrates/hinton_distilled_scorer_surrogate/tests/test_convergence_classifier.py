# SPDX-License-Identifier: MIT
"""Unit tests for the quartile-based convergence classifier.

The classifier lives in ``tools/run_hinton_mlx_long_training_smoke.py``
because it is the canonical telemetry consumer for the canonical smoke
CLI. These tests pin its behavior across the 4 canonical verdicts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_hinton_mlx_long_training_smoke import (  # noqa: E402
    CONVERGENCE_VERDICT_CONVERGES,
    CONVERGENCE_VERDICT_DIVERGES,
    CONVERGENCE_VERDICT_OSCILLATES,
    CONVERGENCE_VERDICT_SUB_PARADIGM,
    VALID_CONVERGENCE_VERDICTS,
    ConvergenceVerdict,
    classify_convergence,
)


def test_valid_verdicts_set_contains_four_canonical_verdicts() -> None:
    assert set(VALID_CONVERGENCE_VERDICTS) == {
        CONVERGENCE_VERDICT_CONVERGES,
        CONVERGENCE_VERDICT_DIVERGES,
        CONVERGENCE_VERDICT_OSCILLATES,
        CONVERGENCE_VERDICT_SUB_PARADIGM,
    }


def test_classifier_converges_on_monotonic_curve() -> None:
    """Monotonically decreasing curve → CONVERGES_CONSISTENTLY."""

    curve = [1.0 - 0.01 * i for i in range(100)]  # 1.0 → 0.01
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_CONVERGES
    assert verdict.loss_reduction_percent > 95.0


def test_classifier_converges_with_sgd_noise_on_tail() -> None:
    """Canonical training pattern: fast early descent + noisy converged tail
    SHOULD be classified CONVERGES (not OSCILLATES)."""

    # Fast descent in Q1, then noisy tail that stays near final value.
    curve = []
    for i in range(25):  # Q1: fast decrease
        curve.append(0.2 - 0.0075 * i)
    for i in range(75):  # Q2-Q4: noisy tail around 0.005
        # 50% chance up, 50% chance down, mean stays at 0.005
        noise = 0.002 * (1 if i % 2 == 0 else -1)
        curve.append(0.005 + noise)
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_CONVERGES, verdict.rationale


def test_classifier_diverges_on_increasing_curve() -> None:
    """Monotonically increasing curve → DIVERGES."""

    curve = [0.01 + 0.01 * i for i in range(100)]
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_DIVERGES


def test_classifier_diverges_on_nan() -> None:
    curve = [1.0, 0.5, float("nan"), 0.2, 0.1]
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_DIVERGES
    assert verdict.nan_at_epoch == 3
    assert math.isnan(verdict.final_loss)


def test_classifier_diverges_on_inf() -> None:
    curve = [1.0, 0.5, float("inf"), 0.2]
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_DIVERGES


def test_classifier_sub_paradigm_when_almost_flat() -> None:
    """Curve that barely decreases over training → SUB_PARADIGM."""

    curve = [1.0 - 0.0003 * i for i in range(100)]  # ends ~0.97
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_SUB_PARADIGM


def test_classifier_oscillates_on_macro_bouncing() -> None:
    """Curve bouncing across quartiles with no net progress → OSCILLATES.

    Quartile means: 1.0, 0.5, 1.0, 0.5 — final/initial = 0.5/1.0 = 0.5
    (does NOT exceed sub_paradigm_threshold of 0.95) so the ratio cascade
    does not auto-classify as DIVERGES/SUB_PARADIGM. Macro increases
    Q1→Q2 (1.0→0.5: decrease), Q2→Q3 (0.5→1.0: INCREASE), Q3→Q4
    (1.0→0.5: decrease) → 1/3 oscillation_score = 0.333 ≤ 0.40, so it
    hits the CONVERGES branch via the ratio cascade (final/initial=0.5).
    But the macro-instability signal must be captured: use 3 increases.
    """

    # Build curve with quartile means: 0.3, 0.6, 0.3, 0.55 (zig-zag with
    # final reduction of ~17%; ratio = 0.55/0.3 = 1.83 → DIVERGES via
    # ratio. That's also fine — the verdict captures macro-instability.
    # For a TRUE OSCILLATES verdict the ratio must be in (converges,
    # sub_paradigm) AND oscillation > 0.40. Construct accordingly:
    # Quartile means: 1.0, 0.6, 1.0, 0.7 → ratio = 0.7 (between 0.5
    # converges and 0.95 sub_paradigm) AND 2 macro increases out of 3
    # (Q1→Q2 down, Q2→Q3 UP, Q3→Q4 down) — oscillation = 1/3 ≈ 0.33,
    # falls below 0.40 threshold so this also doesn't OSCILLATE.
    # We need ALL 3 transitions to be increases or 2/3 to exceed 0.40.
    # Construct: quartile means [1.0, 1.2, 0.8, 0.7] →
    # Q1→Q2 UP, Q2→Q3 DOWN, Q3→Q4 DOWN, oscillation 1/3 ≈ 0.33, no fire
    # Construct: quartile means [0.5, 0.8, 0.55, 0.75] →
    # ratio = 0.75/0.5 = 1.5 > 1.05 → DIVERGES (catches it)
    # The classifier's OSCILLATES branch is genuinely narrow by design:
    # it fires only when ratio is in the converges/sub-paradigm middle
    # band AND oscillation > 0.40. The most direct way to hit it is to
    # construct a curve where ratio = 0.75 AND 2/3 quartile transitions
    # are increases. Means [1.0, 1.4, 0.6, 0.75] → ratio 0.75; macro
    # Q1→Q2 UP, Q2→Q3 DOWN, Q3→Q4 UP → 2/3 = 0.667 > 0.40 → OSCILLATES.
    curve: list[float] = []
    for q in [1.0, 1.4, 0.6, 0.75]:
        curve.extend([q] * 25)
    verdict = classify_convergence(curve)
    assert verdict.verdict == CONVERGENCE_VERDICT_OSCILLATES, verdict.rationale


def test_classifier_rejects_empty_curve() -> None:
    with pytest.raises(ValueError, match="loss_curve must be non-empty"):
        classify_convergence([])


def test_classifier_handles_zero_initial_loss() -> None:
    """Degenerate initial=0 case should not crash (division by zero guard)."""

    curve = [0.0] * 50
    verdict = classify_convergence(curve)
    # All zeros → ratio defined as 0.0 → CONVERGES.
    assert verdict.verdict == CONVERGENCE_VERDICT_CONVERGES


def test_convergence_verdict_rejects_invalid_verdict() -> None:
    with pytest.raises(ValueError, match="verdict must be one of"):
        ConvergenceVerdict(
            verdict="INVALID",
            initial_loss=1.0,
            final_loss=0.1,
            min_loss=0.1,
            max_loss=1.0,
            loss_reduction_percent=90.0,
            nan_at_epoch=None,
            oscillation_score=0.0,
            smoke_epochs=10,
            diverges_threshold_ratio=1.05,
            sub_paradigm_threshold_ratio=0.95,
            converges_threshold_ratio=0.5,
            rationale="test",
        )


def test_convergence_verdict_as_dict() -> None:
    verdict = ConvergenceVerdict(
        verdict=CONVERGENCE_VERDICT_CONVERGES,
        initial_loss=1.0,
        final_loss=0.1,
        min_loss=0.1,
        max_loss=1.0,
        loss_reduction_percent=90.0,
        nan_at_epoch=None,
        oscillation_score=0.0,
        smoke_epochs=10,
        diverges_threshold_ratio=1.05,
        sub_paradigm_threshold_ratio=0.95,
        converges_threshold_ratio=0.5,
        rationale="test",
    )
    d = verdict.as_dict()
    assert d["verdict"] == CONVERGENCE_VERDICT_CONVERGES
    assert d["loss_reduction_percent"] == 90.0
