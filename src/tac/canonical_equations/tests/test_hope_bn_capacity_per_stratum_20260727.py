# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.canonical_equations.hope_bn_capacity_per_stratum_20260727 import (
    EQUATION_ID,
    hope_stratum_capacity,
)


def test_capacity_is_dw_times_sqrt_k() -> None:
    dw = [float(i + 1) for i in range(16)]
    k = [4.0] * 16
    out = hope_stratum_capacity(dw, k)
    np.testing.assert_allclose(out["capacity_per_channel"], [2.0 * (i + 1) for i in range(16)])
    np.testing.assert_allclose(sum(out["capacity_share"]), 1.0)
    assert out["dead_channels_k_lt_1e-12"] == 0


def test_dead_channel_census_and_zero_total() -> None:
    out = hope_stratum_capacity([1.0] * 16, [0.0] * 16)
    assert out["total_capacity"] == 0.0
    assert out["dead_channels_k_lt_1e-12"] == 16
    np.testing.assert_allclose(out["capacity_share"], [0.0] * 16)


def test_gauge_invariance_of_the_capacity_product() -> None:
    """lambda on sqrt(K) with 1/lambda on ||dw|| leaves capacity invariant."""

    rng = np.random.default_rng(0)
    dw = rng.uniform(0.1, 2.0, 16)
    k = rng.uniform(0.1, 2.0, 16)
    lam = 5.0
    a = hope_stratum_capacity(list(dw), list(k))
    b = hope_stratum_capacity(list(dw / lam), list(k * lam**2))
    np.testing.assert_allclose(a["capacity_per_channel"], b["capacity_per_channel"], rtol=1e-12)


def test_refuses_wrong_length_negative_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="exactly 16"):
        hope_stratum_capacity([1.0] * 15, [1.0] * 16)
    with pytest.raises(ValueError, match="nonnegative finite"):
        hope_stratum_capacity([-1.0] + [1.0] * 15, [1.0] * 16)
    with pytest.raises(ValueError, match="nonnegative finite"):
        hope_stratum_capacity([1.0] * 16, [float("nan")] + [1.0] * 15)


def test_registered_evaluator_contract() -> None:
    from tac.canonical_equations.evaluators import resolve_equation_value

    out = resolve_equation_value(
        EQUATION_ID,
        {"delta_w_head_norm": [1.0] * 16, "k_diag_bucket": [1.0] * 16},
    )
    np.testing.assert_allclose(out["total_capacity"], 16.0)
    with pytest.raises(ValueError, match="canonical callable contract"):
        resolve_equation_value(EQUATION_ID, {"delta_w_head_norm": [1.0] * 16})


def test_build_equation_object() -> None:
    from tac.canonical_equations.hope_bn_capacity_per_stratum_20260727 import (
        RECEIPT,
        build_hope_bn_capacity_per_stratum_codebook_v1,
    )

    if not RECEIPT.exists():
        pytest.skip("agreement receipt not present in this checkout")
    eq = build_hope_bn_capacity_per_stratum_codebook_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.domain_of_validity["research_only"] is True
    assert eq.domain_of_validity["score_claim"] is False
