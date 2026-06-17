# SPDX-License-Identifier: MIT
"""Unit tests for the ACCELERATOR-PROBE-1 power-law exponent fit — the load-bearing
math that produces the verdict (a bug here would silently corrupt the d_seg(50k)
projection the whole probe turns on)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# The probe lives in experiments/ (a script, not a package); load _fit_power_law directly.
# Register the module in sys.modules under its synthetic name BEFORE exec so the probe's
# @dataclass definitions can resolve their __module__ during collection.
_PROBE_PATH = Path(__file__).resolve().parents[3] / "experiments" / "probe_accel1_margin_hinge_exponent.py"
_spec = importlib.util.spec_from_file_location("_accel1_probe", _PROBE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_accel1_probe"] = _mod
_spec.loader.exec_module(_mod)
_fit_power_law = _mod._fit_power_law


def test_recovers_known_exponent_exactly():
    # d_seg = 0.5 * step^(-0.35); sample on a log-spaced grid.
    A, p = 0.5, 0.35
    steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    d_seg = [A * s ** (-p) for s in steps]
    fit = _fit_power_law(steps, d_seg)
    assert fit["p"] == pytest.approx(p, abs=1e-6)
    assert math.exp(fit["log_A"]) == pytest.approx(A, abs=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    # d_seg(50k) projection.
    assert fit["d_seg_50k"] == pytest.approx(A * 50_000 ** (-p), rel=1e-6)


def test_steeper_exponent_projects_lower_d_seg_50k():
    steps = [1, 10, 100, 1000]
    shallow = _fit_power_law(steps, [0.04 * s ** (-0.20) for s in steps])
    steep = _fit_power_law(steps, [0.04 * s ** (-0.90) for s in steps])
    assert steep["p"] > shallow["p"]
    assert steep["d_seg_50k"] < shallow["d_seg_50k"]  # bending the exponent lowers the projection


def test_skip_drops_warmup_points():
    # First point is an outlier; skip=1 should drop it and recover the clean fit.
    steps = [1, 2, 5, 10, 20, 50]
    clean = [0.5 * s ** (-0.4) for s in steps]
    noisy = [10.0, *clean[1:]]  # corrupt the first point
    fit_no_skip = _fit_power_law(steps, noisy, skip=0)
    fit_skip = _fit_power_law(steps, noisy, skip=1)
    assert abs(fit_skip["p"] - 0.4) < abs(fit_no_skip["p"] - 0.4)


def test_too_few_points_returns_nan():
    fit = _fit_power_law([1, 2], [0.5, 0.4])
    assert math.isnan(fit["p"])
    assert fit["n_points"] == 2


def test_ignores_nonpositive_and_nonfinite():
    steps = [1, 2, 3, 4, 5]
    d_seg = [0.5, 0.0, -0.1, float("nan"), 0.3]  # only steps 1 and 5 survive -> <3 -> nan
    fit = _fit_power_law(steps, d_seg)
    assert math.isnan(fit["p"])  # only 2 valid points

    steps2 = [1, 2, 3, 4, 5, 6]
    d_seg2 = [0.5, 0.0, 0.35, 0.30, 0.27, 0.25]  # 5 valid (drop the 0.0)
    fit2 = _fit_power_law(steps2, d_seg2)
    assert fit2["n_points"] == 5
    assert math.isfinite(fit2["p"])
