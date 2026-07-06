"""Tests for tac.clip_profile — the canonical MEASURED per-clip config.

The load-bearing property (operator directive 2026-07-06): on the contest clip the
auto-measured profile REPRODUCES the current hardcoded constants (no regression), and on
a different resolution it self-calibrates (drop-in). Uses the real n600 SegNet-argmax
cache + the real 0.mkv when present; skips gracefully otherwise.
"""
from pathlib import Path

import numpy as np
import pytest

from tac.camera import COMMA_INTRINSICS
from tac.clip_profile import (
    CANONICAL_CLASS_ORDER,
    OPENPILOT_DEVICE_HEIGHT_M,
    ClipProfile,
    camera_for_resolution,
    detect_class_order,
    measure,
    measure_v_horizon,
)

_REPO = Path(__file__).resolve().parents[3]
_CACHE = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
_VIDEO = _REPO / "upstream/videos/0.mkv"


def _lstars():
    if not _CACHE.is_file():
        pytest.skip("gt_n600 cache absent")
    return np.asarray(np.load(_CACHE)["lstars"])


def test_camera_match_neo_config_reproduces_scorer_intrinsics():
    cam = camera_for_resolution(1164, 874)
    assert cam.matched_openpilot_camera == "_neo_config"
    assert cam.fx_native == 910.0 and cam.cx_native == 582.0
    # scorer intrinsics reproduce the canonical hardcode (no regression)
    assert abs(cam.fx_scorer - COMMA_INTRINSICS.fx) < 1e-6
    assert abs(cam.cx_scorer - COMMA_INTRINSICS.cx) < 1e-6
    assert abs(cam.cy_scorer - COMMA_INTRINSICS.cy) < 1e-6


def test_camera_unknown_resolution_self_calibrates():
    cam = camera_for_resolution(1920, 1080)
    assert cam.matched_openpilot_camera == "unknown"
    # scorer principal point stays centered-ish under the rescale
    assert 200 < cam.cx_scorer < 320 and 150 < cam.cy_scorer < 240


def test_v_horizon_measures_174ish_on_contest_clip():
    L = _lstars()
    vh, p10, p90 = measure_v_horizon(L)
    # n600 sweep #327 found 174 OPTIMAL; the road/sky boundary lands within ~2 rows.
    assert 172 <= vh <= 178, f"measured horizon {vh} not near the hardcoded 174"
    assert p10 < vh < p90


def test_class_order_self_detects_canonical():
    L = _lstars()
    order, idx = detect_class_order(L)
    assert order == CANONICAL_CLASS_ORDER, f"self-detected {order}"
    # the spatial signatures put the right names on the right indices
    assert idx["Lane"] == 1 and idx["MyCar"] == 4 and idx["Road"] == 0


def test_full_profile_reproduces_contest_hardcodes_and_identifies_source():
    if not _VIDEO.is_file():
        pytest.skip("0.mkv absent")
    L = _lstars()
    prof = measure(_VIDEO, L, git_sha="testsha", measured_at="2026-07-06T00:00:00Z")
    assert 172 <= prof.v_horizon <= 178                       # reproduces _V_HORIZON=174
    assert prof.device_height_m == OPENPILOT_DEVICE_HEIGHT_M  # 1.22 (confirmed-correct)
    assert prof.class_order_is_canonical
    assert prof.camera.matched_openpilot_camera == "_neo_config"
    # 0.mkv provenance = the comma2k19 RAV4 segment (37,545,489 bytes)
    assert prof.video_bytes == 37_545_489
    assert "comma2k19" in prof.provenance["source_identified"]


def test_to_from_dict_roundtrip():
    L = _lstars()
    prof = measure("nonexistent.mkv", L)   # sha None (file absent) is fine for roundtrip
    d = prof.to_dict()
    back = ClipProfile.from_dict(d)
    assert back.v_horizon == prof.v_horizon
    assert back.class_order == prof.class_order
    assert back.camera.matched_openpilot_camera == prof.camera.matched_openpilot_camera
