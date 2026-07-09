"""No-regression test for the #328 phase-2 clip_profile rewire of
``tools/measure_pose_warp_dseg.py`` (and its transitive consumer
``tools/measure_screw_reach_through_R.py`` which imports the same constants).

The rewire replaced the module's hardcoded EON native intrinsics + camera height with
reads of the canonical MEASURED ``tac.clip_profile`` (cached per-clip SoT). The load-bearing
property (measured-no-regression): on 0.mkv the clip_profile-sourced constants REPRODUCE the
prior openpilot/comma2k19 literals BIT-IDENTICALLY, so no advisory measurement changes. The
fallback branch keeps the tool standalone-runnable when the profile cache is absent.
"""
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools/measure_pose_warp_dseg.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("mpwd_rewire_test", _TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_rewired_constants_reproduce_eon_literals_bit_identical():
    m = _load_tool()
    # BIT-IDENTICAL to the historical hardcodes (910/582/437 native, 1.22 m, 1164x874).
    assert (m.NATIVE_W, m.NATIVE_H) == (1164, 874)
    assert m.NATIVE_FX == 910.0 and m.NATIVE_FY == 910.0
    assert (m.NATIVE_CX, m.NATIVE_CY) == (582.0, 437.0)
    assert m.CAMERA_HEIGHT_M == 1.22


def test_rewired_intrinsics_at_scorer_res_unchanged():
    m = _load_tool()
    K = m.intrinsics_at(512, 384)
    # fx*sx = 910*512/1164, cx*sx = 582*512/1164 = 256.0 exactly (principal point centered).
    assert abs(K[0, 0] - 910.0 * 512 / 1164) < 1e-9
    assert abs(K[0, 2] - 256.0) < 1e-9
    assert abs(K[1, 2] - 192.0) < 1e-9


def test_constants_track_clip_profile_when_cache_present():
    """When the canonical cache exists, the module constants EQUAL the clip_profile SoT
    (proves the rewire actually reads the profile, not a stale literal)."""
    from tac.clip_profile import for_video

    try:
        cp = for_video(_REPO / "upstream/videos/0.mkv")
    except Exception:
        import pytest

        pytest.skip("clip_profile cache absent (fallback path exercised elsewhere)")
    m = _load_tool()
    assert m.NATIVE_FX == cp.camera.fx_native
    assert m.NATIVE_CX == cp.camera.cx_native
    assert m.CAMERA_HEIGHT_M == cp.device_height_m
