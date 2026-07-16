# SPDX-License-Identifier: MIT
"""Tests for the #515 B0 per-tensor ||W|| telemetry producer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.witness_control.weight_norm_telemetry import (
    WEIGHT_NORM_SCHEMA,
    WeightNormBaseline,
    baseline_from_row,
    per_tensor_norms,
    weight_norm_row,
)


def _params(scale: float = 1.0) -> dict[str, np.ndarray]:
    return {
        "film.w": scale * np.ones((3, 4), np.float32),
        "hidden.w": scale * np.full((2, 2), 2.0, np.float32),
        "__cfg_render_aa": np.asarray("none"),  # excluded bookkeeping key
    }


def test_per_tensor_norms_values_and_exclusion():
    norms = per_tensor_norms(_params())
    assert set(norms) == {"film.w", "hidden.w"}  # __cfg_* excluded
    assert norms["film.w"] == pytest.approx(math.sqrt(12.0))
    assert norms["hidden.w"] == pytest.approx(4.0)


def test_per_tensor_norms_sorted_deterministic():
    norms = per_tensor_norms(_params())
    assert list(norms) == sorted(norms)


def test_per_tensor_norms_nonfinite_fails_loud():
    bad = {"w": np.array([np.nan, 1.0])}
    with pytest.raises(ValueError, match="non-finite"):
        per_tensor_norms(bad)


def test_weight_norm_row_shape_and_globals():
    row = weight_norm_row(25, _params(), _params())
    assert row["stage"] == "weight_norm"
    assert row["schema"] == WEIGHT_NORM_SCHEMA
    assert row["ep"] == 25
    assert row["per_tensor"]["film.w"]["norm"] == pytest.approx(math.sqrt(12.0))
    assert row["per_tensor"]["film.w"]["ema_norm"] == pytest.approx(math.sqrt(12.0))
    g = row["global"]
    assert g["n_tensors"] == 2
    assert g["total_norm"] == pytest.approx(math.sqrt(12.0 + 16.0))


def test_rel_from_t0_drift_series():
    baseline = WeightNormBaseline(epoch=0, norms=per_tensor_norms(_params(1.0)))
    row = weight_norm_row(50, _params(0.5), baseline=baseline)
    # 0.5x scale => rel_from_t0 = -0.5 on every tensor
    assert row["per_tensor"]["film.w"]["rel_from_t0"] == pytest.approx(-0.5)
    assert row["global"]["rel_from_t0_min"] == pytest.approx(-0.5)
    assert row["global"]["rel_from_t0_max"] == pytest.approx(-0.5)


def test_eta_rel_read_stream():
    row = weight_norm_row(
        1, _params(), update_norms={"film.w": math.sqrt(12.0) * 0.1},
    )
    assert row["per_tensor"]["film.w"]["eta_rel"] == pytest.approx(0.1)
    assert "eta_rel" not in row["per_tensor"]["hidden.w"]  # only supplied tensors


def test_eta_rel_bad_update_norm_fails_loud():
    with pytest.raises(ValueError, match="bad update norm"):
        weight_norm_row(1, _params(), update_norms={"film.w": -1.0})


def test_baseline_from_row_resume_roundtrip():
    baseline = WeightNormBaseline(epoch=0, norms=per_tensor_norms(_params()))
    row = weight_norm_row(0, _params(), baseline=baseline)
    restored = baseline_from_row(row)
    assert restored.epoch == 0
    assert restored.norms["hidden.w"] == pytest.approx(4.0)
    # drift measured against the RESTORED baseline matches the original
    row2 = weight_norm_row(10, _params(2.0), baseline=restored)
    assert row2["per_tensor"]["hidden.w"]["rel_from_t0"] == pytest.approx(1.0)


def test_baseline_from_row_rejects_wrong_schema():
    with pytest.raises(ValueError, match="schema"):
        baseline_from_row({"schema": "other.v1", "ep": 0, "per_tensor": {"w": {"norm": 1.0}}})


def test_baseline_validation():
    with pytest.raises(ValueError):
        WeightNormBaseline(epoch=-1, norms={})
    with pytest.raises(ValueError):
        WeightNormBaseline(epoch=0, norms={"w": float("inf")})
