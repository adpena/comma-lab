# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.vrghal_theorem_body_20260713 import (
    EQUATION_ID,
    build_vrghal_high_probability_fixed_operator_law_v2,
    vrghal_epoch_residual_upper_bound,
    vrghal_theorem_admission,
)


def test_epoch_residual_envelope_transcribes_theorem() -> None:
    actual = vrghal_epoch_residual_upper_bound(
        epoch=2,
        beta=0.5,
        a0=1.0,
        a1=2.0,
        a2=3.0,
        a_five_halves=4.0,
    )
    expected = 0.25 * (1.0 + 2.0 * 2.0 + 3.0 * 16.0 + 4.0 * 32.0)
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.parametrize(
    "override",
    [
        {"fixed_operator": False},
        {"lipschitz_upper_bound": 1.00001},
        {"unbiased_oracle": False},
        {"bounded_native_second_moment": False},
        {"quadratically_smoothable_space": False},
    ],
)
def test_theorem_admission_fails_closed(override: dict[str, object]) -> None:
    kwargs: dict[str, object] = {
        "fixed_operator": True,
        "lipschitz_upper_bound": 1.0,
        "unbiased_oracle": True,
        "bounded_native_second_moment": True,
        "quadratically_smoothable_space": True,
    }
    kwargs.update(override)
    assert not vrghal_theorem_admission(**kwargs)  # type: ignore[arg-type]


def test_theorem_admission_accepts_base_premises() -> None:
    assert vrghal_theorem_admission(
        fixed_operator=True,
        lipschitz_upper_bound=1.0,
        unbiased_oracle=True,
        bounded_native_second_moment=True,
        quadratically_smoothable_space=True,
    )


def test_canonical_equation_preserves_scope_and_supersession() -> None:
    equation = build_vrghal_high_probability_fixed_operator_law_v2()
    assert equation.equation_id == EQUATION_ID
    assert equation.domain_of_validity["current_pre_se_verdict"] == (
        "NO-GO_DOMINATED_BY_CERTIFIED_DIRECT_SOLVE"
    )
    assert equation.domain_of_validity["current_non_dominated_theorem_admitted_locus"] == "NONE"
    assert "vrghal_95kill_fixedpoint_equations_20260713.md" in str(
        equation.domain_of_validity["supersedes"]
    )

