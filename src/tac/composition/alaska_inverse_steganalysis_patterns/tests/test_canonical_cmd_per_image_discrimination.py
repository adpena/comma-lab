# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA CMD per-image discrimination."""

from __future__ import annotations

import numpy as np
import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    CMDDiscriminationConfig,
    CMDDiscriminationError,
    CMDDiscriminationStrategy,
    compute_cmd_per_image_score,
)
from tac.composition.alaska_inverse_steganalysis_patterns.canonical_cmd_per_image_discrimination import (
    CMD_4_STAT_NAMES,
)


def test_cmd_4_stat_names_verbatim_yousfi() -> None:
    """Yousfi upstream ``tf.squeeze(tf.stack([avgp, varp, minp, maxp]))``."""
    assert CMD_4_STAT_NAMES == ("avg", "var", "min", "max")


def test_strategy_enum_4_values() -> None:
    assert {s.value for s in CMDDiscriminationStrategy} == {
        "moment_4_stat",
        "moment_2_stat",
        "extrema_2_stat",
        "raw_conv_mean",
    }


def test_config_rejects_bad_strategy_type() -> None:
    with pytest.raises(CMDDiscriminationError, match="must be CMDDiscriminationStrategy"):
        CMDDiscriminationConfig(strategy="moment_4_stat")  # type: ignore[arg-type]


def test_config_rejects_empty_spatial_axes() -> None:
    with pytest.raises(CMDDiscriminationError, match="spatial_axes empty"):
        CMDDiscriminationConfig(spatial_axes=())


def test_compute_4_stat_canonical() -> None:
    """4-stat strategy MUST return all of (avg, var, min, max)."""
    tensor = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    cfg = CMDDiscriminationConfig(strategy=CMDDiscriminationStrategy.MOMENT_4_STAT)
    out = compute_cmd_per_image_score(tensor, cfg)
    assert set(out.keys()) == {"avg", "var", "min", "max"}


def test_compute_4_stat_shape_reduction() -> None:
    """Reduces over (-2, -1) = (H, W) preserving (B, C)."""
    tensor = np.random.uniform(0, 1, size=(4, 8, 16, 16)).astype(np.float32)
    cfg = CMDDiscriminationConfig(strategy=CMDDiscriminationStrategy.MOMENT_4_STAT)
    out = compute_cmd_per_image_score(tensor, cfg)
    for name in ("avg", "var", "min", "max"):
        assert out[name].shape == (4, 8)


def test_compute_4_stat_avg_var_correctness() -> None:
    """``avg`` MUST equal numpy mean; ``var`` MUST equal numpy var.

    Verifies the helper is REAL (not a stub returning constants).
    """
    tensor = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        dtype=np.float32,
    )  # shape (2, 2, 2)
    cfg = CMDDiscriminationConfig(
        strategy=CMDDiscriminationStrategy.MOMENT_4_STAT, spatial_axes=(-2, -1)
    )
    out = compute_cmd_per_image_score(tensor, cfg)
    np.testing.assert_allclose(out["avg"], [2.5, 6.5])
    np.testing.assert_allclose(out["var"], [1.25, 1.25])
    np.testing.assert_allclose(out["min"], [1.0, 5.0])
    np.testing.assert_allclose(out["max"], [4.0, 8.0])


def test_compute_2_stat_moment_only() -> None:
    """Moment-2 returns only (avg, var)."""
    tensor = np.random.uniform(0, 1, size=(2, 4, 8, 8)).astype(np.float32)
    cfg = CMDDiscriminationConfig(strategy=CMDDiscriminationStrategy.MOMENT_2_STAT)
    out = compute_cmd_per_image_score(tensor, cfg)
    assert set(out.keys()) == {"avg", "var"}


def test_compute_2_stat_extrema_only() -> None:
    """Extrema-2 returns only (min, max)."""
    tensor = np.random.uniform(0, 1, size=(2, 4, 8, 8)).astype(np.float32)
    cfg = CMDDiscriminationConfig(strategy=CMDDiscriminationStrategy.EXTREMA_2_STAT)
    out = compute_cmd_per_image_score(tensor, cfg)
    assert set(out.keys()) == {"min", "max"}


def test_compute_raw_conv_mean_baseline() -> None:
    """Raw-mean baseline returns only ``raw_mean`` (canonical CC ablation)."""
    tensor = np.random.uniform(0, 1, size=(2, 4, 8, 8)).astype(np.float32)
    cfg = CMDDiscriminationConfig(strategy=CMDDiscriminationStrategy.RAW_CONV_MEAN)
    out = compute_cmd_per_image_score(tensor, cfg)
    assert set(out.keys()) == {"raw_mean"}


def test_compute_rejects_non_ndarray() -> None:
    cfg = CMDDiscriminationConfig()
    with pytest.raises(CMDDiscriminationError, match="must be np.ndarray"):
        compute_cmd_per_image_score([1, 2, 3], cfg)  # type: ignore[arg-type]


def test_compute_rejects_empty_tensor() -> None:
    cfg = CMDDiscriminationConfig()
    with pytest.raises(CMDDiscriminationError, match="empty"):
        compute_cmd_per_image_score(np.array([]), cfg)


def test_compute_rejects_out_of_range_axis() -> None:
    tensor = np.zeros((2, 4), dtype=np.float32)  # ndim=2
    cfg = CMDDiscriminationConfig(spatial_axes=(5,))
    with pytest.raises(CMDDiscriminationError, match="out of range"):
        compute_cmd_per_image_score(tensor, cfg)


def test_compute_substantively_distinct_per_strategy() -> None:
    """Slot EEE substantive-distinctness: each strategy MUST return a
    structurally distinct output dict (not identical content across enum
    values).

    Canonical anti-pattern this guards against: enum-padding where
    multiple enum values dispatch to the same implementation (Catalog
    #308 alternative-probe-methodology bug class).
    """
    tensor = np.random.uniform(0, 1, size=(2, 4, 8, 8)).astype(np.float32)
    keys_per_strategy: dict[str, frozenset[str]] = {}
    for strategy in CMDDiscriminationStrategy:
        cfg = CMDDiscriminationConfig(strategy=strategy)
        out = compute_cmd_per_image_score(tensor, cfg)
        keys_per_strategy[strategy.value] = frozenset(out.keys())
    # 4 strategies must produce >= 3 distinct key-sets
    # (moment_2 and moment_4 share avg+var; extrema_2 has min+max alone;
    # raw_conv_mean has raw_mean alone; moment_4 has all 4)
    n_distinct = len(set(keys_per_strategy.values()))
    assert n_distinct == 4, (
        f"strategies must produce 4 distinct key-sets; got {n_distinct}: "
        f"{keys_per_strategy}"
    )
