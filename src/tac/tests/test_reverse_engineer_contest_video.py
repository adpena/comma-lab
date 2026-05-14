# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "experiments" / "reverse_engineer_contest_video.py"
SPEC = importlib.util.spec_from_file_location("reverse_engineer_contest_video", MODULE_PATH)
assert SPEC is not None
rev = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rev)


def test_native_foe_scales_from_scorer_geometry() -> None:
    foe = rev._native_foe()
    assert foe["scorer_x"] == 256.0
    assert foe["scorer_y"] == 174.0
    assert foe["native_x"] == 582.0
    assert 395.0 < foe["native_y"] < 397.0


def test_ring_masks_cover_image_and_horizon_band() -> None:
    masks = rev._ring_masks(874, 1164, center_x=582.0, center_y=396.0)
    ring_keys = [key for key in masks if key.startswith("r_")]
    coverage = np.zeros((874, 1164), dtype=np.int16)
    for key in ring_keys:
        coverage += masks[key].astype(np.int16)
    assert coverage.min() == 1
    assert coverage.max() == 1
    assert masks["horizon_band"].sum() > 0


def test_summary_is_json_safe() -> None:
    summary = rev._summary([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
