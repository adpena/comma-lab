from __future__ import annotations

import math

import numpy as np
import pytest

from tac.witness_control.costate_society_diagnostics import (
    _ulp_variants,
    errors_to_behaviour,
    hierarchic_social_entropy,
)


def test_hse_identical_society_is_zero() -> None:
    behaviour = np.ones((3, 4), dtype=np.float64)
    hse, normalized, _ = hierarchic_social_entropy(behaviour)
    assert hse == pytest.approx(0.0)
    assert normalized == pytest.approx(0.0)


def test_hse_orthogonal_society_is_maximal() -> None:
    behaviour = np.eye(3, dtype=np.float64)
    hse, normalized, _ = hierarchic_social_entropy(behaviour)
    assert hse == pytest.approx(math.log2(3))
    assert normalized == pytest.approx(1.0)


def test_error_behaviour_persistence_anchor_is_exp_minus_one() -> None:
    persistence = np.asarray([1.0, 2.0, 4.0])
    errors = np.stack([persistence, persistence * 0.5])
    behaviour = errors_to_behaviour(errors, persistence)
    np.testing.assert_allclose(behaviour[0], np.exp(-1.0))
    assert np.all(behaviour[1] > behaviour[0])


def test_ulp_variants_straddle_exact_tie() -> None:
    variants = _ulp_variants(1.0, 1.0)
    signed = [recent - median for _, recent, median in variants]
    assert len(variants) == 6
    assert min(signed) < 0.0 < max(signed)
