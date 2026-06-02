# SPDX-License-Identifier: MIT
"""Tests for MLX-to-NumPy bridge drift reporting."""

from __future__ import annotations

import numpy as np

from tac.substrates._shared.mlx_score_aware.bridge_drift import (
    build_mlx_numpy_bridge_drift_bundle,
    mlx_numpy_bridge_drift_report,
)


def test_bridge_drift_reports_exact_numpy_handoff() -> None:
    ref = np.arange(12, dtype=np.float32).reshape(3, 4)
    got = ref.copy()

    row = mlx_numpy_bridge_drift_report(
        label="unit",
        mlx_array=ref,
        numpy_array=got,
        atol=0.0,
        rtol=0.0,
    )

    assert row["schema"] == "mlx_numpy_bridge_drift.v1"
    assert row["allclose"] is True
    assert row["max_abs"] == 0.0
    assert row["blockers"] == []
    assert row["score_claim"] is False


def test_bridge_drift_bundle_fails_closed_on_mismatch() -> None:
    ref = np.zeros((2, 2), dtype=np.float32)
    got = np.ones((2, 2), dtype=np.float32)

    row = mlx_numpy_bridge_drift_report(
        label="unit",
        mlx_array=ref,
        numpy_array=got,
        atol=0.0,
        rtol=0.0,
    )
    bundle = build_mlx_numpy_bridge_drift_bundle([row], bundle_id="unit_bundle")

    assert row["allclose"] is False
    assert "mlx_numpy_bridge_drift_exceeds_tolerance" in row["blockers"]
    assert bundle["allclose"] is False
    assert "mlx_numpy_bridge_bundle_not_allclose" in bundle["blockers"]
    assert bundle["ready_for_exact_eval_dispatch"] is False
