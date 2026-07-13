# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.scorer_surrogate.pre_se_multi_source_reopen_20260713 import (
    MULTI_SOURCE_FEATURE_COUNT,
    PreSEMultiSourceError,
    cheap_global_cost_accounting,
    compose_protected_feature_rows,
)
from tac.witness_dsl.pre_se_multi_source_reopen_policy_20260713 import (
    PreSEMultiSourceReopenPolicy,
)


def _aligned_rows(count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(count * 42, dtype=np.float32).reshape(count, 42)
    sensitivity = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + 5000
    block2 = np.arange(count * 144, dtype=np.float32).reshape(count, 144) + 1000
    block3 = np.arange(count * 288, dtype=np.float32).reshape(count, 288) + 2000
    return (
        np.concatenate((base, block2, sensitivity), axis=1),
        np.concatenate((base, block3, sensitivity), axis=1),
    )


def test_protected_rows_compose_shared_columns_once() -> None:
    block2, block3 = _aligned_rows()
    rows = compose_protected_feature_rows(block2, block3)
    assert rows.shape == (3, MULTI_SOURCE_FEATURE_COUNT)
    assert np.array_equal(rows[:, :42], block2[:, :42])
    assert np.array_equal(rows[:, 42:186], block2[:, 42:-2])
    assert np.array_equal(rows[:, 186:474], block3[:, 42:-2])
    assert np.array_equal(rows[:, -2:], block2[:, -2:])


def test_protected_rows_fail_closed_on_shared_column_drift() -> None:
    block2, block3 = _aligned_rows()
    block3[0, 3] += 1
    with pytest.raises(PreSEMultiSourceError, match="base columns"):
        compose_protected_feature_rows(block2, block3)


def test_policy_preserves_apples_to_apples_bars() -> None:
    contract = PreSEMultiSourceReopenPolicy().compile_measurement_contract()
    assert contract["feature_count"] == 476
    assert contract["retained_mass_bar"] == pytest.approx(0.47)
    assert contract["realized_area_fraction"] == pytest.approx(2311 / 49152)
    assert contract["round4_oracle_retained_mass"] == pytest.approx(0.5278150212253758)
    assert contract["live_trainer_argv"] == []


def test_cheap_global_accounting_deduplicates_shared_ancestors() -> None:
    modules = ("a.se", "b.se")
    model = {
        "per_se_global_reduction": [
            {"module": "a.se", "global_pool_forward_flops": 100, "gate_scalars": 8},
            {"module": "b.se", "global_pool_forward_flops": 200, "gate_scalars": 16},
        ],
        "per_conv_forward_macs": [
            {"module": "a.se.conv_reduce", "forward_macs": 4},
            {"module": "a.se.conv_expand", "forward_macs": 4},
            {"module": "b.se.conv_reduce", "forward_macs": 8},
            {"module": "b.se.conv_expand", "forward_macs": 8},
            {"module": "local.conv", "forward_macs": 999},
        ],
    }
    result = cheap_global_cost_accounting(
        model,
        upstream_se_modules=modules,
        tile_count=4,
        tiled_local_conv_macs=1000,
    )
    assert result["unique_upstream_se_reduction_count"] == 2
    assert result["global_gate_scalars_once"] == 24
    assert result["SE_MLP_forward_macs_once"] == 24
    assert result["global_forward_plus_vjp_flops_once"] == 696
    assert result[
        "true_average_per_tile_forward_plus_vjp_flops_including_amortized_globals"
    ] == pytest.approx(1174)


def test_cheap_global_accounting_rejects_duplicate_modules() -> None:
    with pytest.raises(PreSEMultiSourceError, match="duplicates"):
        cheap_global_cost_accounting(
            {"per_se_global_reduction": [], "per_conv_forward_macs": []},
            upstream_se_modules=("a", "a"),
            tile_count=4,
            tiled_local_conv_macs=1,
        )
