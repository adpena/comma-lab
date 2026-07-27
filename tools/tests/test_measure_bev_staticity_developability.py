# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "measure_bev_staticity_developability.py"
_SPEC = importlib.util.spec_from_file_location("measure_bev_staticity_developability", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_oriented_shallow_boundary_uses_margin_ratio_and_excludes_deep_side() -> None:
    labels = np.zeros((384, 512), dtype=np.uint8)
    labels[:, 256:] = 1
    margins = np.ones((384, 512), dtype=np.float64)
    margins[:, 255] = 1.0
    margins[:, 256] = 3.0
    class_index = {"Road": 0, "Lane": 1, "Undrivable": 2, "Movable": 3, "MyCar": 4}
    points, all_points, counts = _MODULE.oriented_shallow_boundary_points(labels, margins, class_index)
    assert points["Road"].shape == (384, 2)
    assert np.allclose(points["Road"][:, 0], 255.25)
    assert points["Lane"].shape == (0, 2)
    assert all_points["Road"].shape == (384, 2)
    assert all_points["Lane"].shape == (384, 2)
    assert counts["Road"]["shallow"] == 384
    assert counts["Lane"]["deep"] == 384


def test_static_segments_exclude_event_frames() -> None:
    assert _MODULE.static_segments(8, [2, 5]) == [(0, 2), (3, 5), (6, 8)]


def test_custody_constants_pin_solved_inputs_and_ssd_waterfall() -> None:
    assert _MODULE.GT_CACHE_SHA256 == "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
    assert _MODULE.SOLVED_SEED_SHA256 == "a21dde38128bed7ff62860ef005b994b74202e0bd00a37d1df8824ee325e856b"
    assert all(str(root).startswith("/Volumes/") for root in _MODULE.SSD_ROOTS)


def test_ruling_summary_distinguishes_static_from_moving() -> None:
    base = np.linspace(-1.0, 1.0, _MODULE.GRID_BINS)
    static = np.stack([np.stack((base, base, base)) for _ in range(12)])
    forward = np.full_like(static, 20.0)
    centers = np.linspace(0.0, 10.0, _MODULE.GRID_BINS)
    static_summary = _MODULE.summarize_stratum(static, forward, centers, ground=False, fx=400.0)
    moving = static + np.arange(12, dtype=np.float64)[:, None, None] * 2.0
    moving_summary = _MODULE.summarize_stratum(moving, forward, centers, ground=False, fx=400.0)
    assert static_summary["near_static"] is True
    assert static_summary["static_fraction_at_1px_floor"] == 1.0
    assert moving_summary["residual_dynamics_fraction"] > 0.0
