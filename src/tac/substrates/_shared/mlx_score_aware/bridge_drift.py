# SPDX-License-Identifier: MIT
"""MLX-to-NumPy bridge drift reports for portable substrate exports.

MLX-local training is useful only if the handoff into the NumPy/archive
receiver surface is measured.  This module is intentionally tiny and
substrate-agnostic: callers materialize an MLX array, convert it to the NumPy
array that will feed the archive exporter, then record the numerical drift in a
false-authority JSON-friendly schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

MLX_NUMPY_BRIDGE_DRIFT_SCHEMA = "mlx_numpy_bridge_drift.v1"
MLX_NUMPY_BRIDGE_DRIFT_BUNDLE_SCHEMA = "mlx_numpy_bridge_drift_bundle.v1"

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def mlx_numpy_bridge_drift_report(
    *,
    label: str,
    mlx_array: Any,
    numpy_array: Any,
    atol: float = 0.0,
    rtol: float = 0.0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare an MLX array with the NumPy bridge array that will be exported.

    The helper deliberately accepts ``Any`` for the MLX input so importing this
    module never imports MLX.  ``np.asarray`` is the bridge under test.
    """

    ref = np.asarray(mlx_array)
    got = np.asarray(numpy_array)
    blockers: list[str] = []
    if ref.shape != got.shape:
        blockers.append("mlx_numpy_bridge_shape_mismatch")
    if ref.dtype != got.dtype:
        blockers.append("mlx_numpy_bridge_dtype_mismatch")
    if ref.shape == got.shape:
        finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
        if not finite:
            blockers.append("mlx_numpy_bridge_nonfinite_values")
        diff = got.astype(np.float64) - ref.astype(np.float64)
        abs_diff = np.abs(diff)
        max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
        mse = float(np.mean(diff * diff)) if diff.size else 0.0
        allclose = bool(
            finite and np.allclose(got, ref, atol=float(atol), rtol=float(rtol))
        )
        if not allclose:
            blockers.append("mlx_numpy_bridge_drift_exceeds_tolerance")
    else:
        finite = False
        max_abs = None
        mean_abs = None
        mse = None
        allclose = False

    return {
        "schema": MLX_NUMPY_BRIDGE_DRIFT_SCHEMA,
        "label": str(label),
        "shape": list(got.shape),
        "reference_shape": list(ref.shape),
        "dtype": str(got.dtype),
        "reference_dtype": str(ref.dtype),
        "atol": float(atol),
        "rtol": float(rtol),
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "mse": mse,
        "allclose": allclose,
        "extra": dict(extra or {}),
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


def build_mlx_numpy_bridge_drift_bundle(
    reports: Sequence[Mapping[str, Any]],
    *,
    bundle_id: str,
) -> dict[str, Any]:
    """Aggregate per-array bridge reports into one fail-closed payload."""

    rows = [dict(row) for row in reports]
    blockers: list[str] = []
    for row in rows:
        blockers.extend(str(v) for v in row.get("blockers") or [])
    allclose = bool(rows) and all(bool(row.get("allclose")) for row in rows)
    if not rows:
        blockers.append("mlx_numpy_bridge_no_reports")
    if not allclose:
        blockers.append("mlx_numpy_bridge_bundle_not_allclose")
    max_abs_values = [
        float(row["max_abs"])
        for row in rows
        if row.get("max_abs") is not None and np.isfinite(float(row["max_abs"]))
    ]
    return {
        "schema": MLX_NUMPY_BRIDGE_DRIFT_BUNDLE_SCHEMA,
        "bundle_id": str(bundle_id),
        "report_count": len(rows),
        "allclose": allclose,
        "max_abs": max(max_abs_values) if max_abs_values else None,
        "reports": rows,
        "blockers": list(dict.fromkeys(blockers)),
        **FALSE_AUTHORITY,
    }


__all__ = [
    "FALSE_AUTHORITY",
    "MLX_NUMPY_BRIDGE_DRIFT_BUNDLE_SCHEMA",
    "MLX_NUMPY_BRIDGE_DRIFT_SCHEMA",
    "build_mlx_numpy_bridge_drift_bundle",
    "mlx_numpy_bridge_drift_report",
]
