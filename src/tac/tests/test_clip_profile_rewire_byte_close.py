"""No-regression tests for the #328 Phase-2 clip_profile rewire of the score / trainer
byte path: ``tools/levelset_byte_close_and_eval.py`` (byte-close constants) and
``src/tac/boundary_math/lane_sdf_component.py`` (scorer intrinsics imported into the
levelset trainer byte path).

The rewire routes each per-clip constant through the canonical MEASURED ``tac.clip_profile``
SoT (cached per-clip). The load-bearing property (measured-no-regression): on 0.mkv the
profile-sourced constants REPRODUCE the prior hardcoded literals BIT-IDENTICALLY, so the
byte-close output + trainer byte path are unchanged. The two DISAGREEING constants
(``_V_HORIZON`` = 174 swept-optimal #327 vs profile median 175; ``_CAM_H`` = 1.2 lane-IPM vs
profile 1.22) are DELIBERATELY left hardcoded per the FEED-clipprofile2 discrepancy findings —
this test asserts they were NOT silently switched.
"""
import hashlib
import importlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _clip_profile_or_skip():
    from tac.clip_profile import for_video

    try:
        return for_video(_REPO / "upstream/videos/0.mkv")
    except Exception:
        import pytest

        pytest.skip("clip_profile cache absent (fallback literals exercised by construction)")


# ---------------------------------------------------------------------------
# byte-close: tools/levelset_byte_close_and_eval.py
# ---------------------------------------------------------------------------
def _load_byte_close():
    # plain import: the module-level constants (resolved before the tool's heavy CLI body)
    # are the rewired surface under test.
    return importlib.import_module("tools.levelset_byte_close_and_eval")


def test_byte_close_camera_rate_constants_bit_identical():
    bc = _load_byte_close()
    # native camera resolution + rate denominator (video_bytes) — byte-identical literals.
    assert (bc.CAMERA_H, bc.CAMERA_W) == (874, 1164)
    assert bc.RATE_DENOM == 37_545_489.0
    # the xi-homography intrinsics the _XI_* line consumes (fx==fy==910 native, cx/cy, height).
    assert (bc._CP_XI_FX, bc._CP_XI_CX, bc._CP_XI_CY, bc._CP_XI_D) == (910.0, 582.0, 437.0, 1.22)
    contract = bc.xi_receiver_camera_contract()
    assert contract["values"] == {
        "cx": 582.0,
        "cy": 437.0,
        "device_height_m": 1.22,
        "fx_native": 910.0,
    }
    assert hashlib.sha256(contract["canonical_json"].encode("ascii")).hexdigest() == contract[
        "canonical_json_sha256"
    ]


def test_byte_close_constants_track_clip_profile_when_cache_present():
    cp = _clip_profile_or_skip()
    bc = _load_byte_close()
    # proves the rewire reads the profile (not a stale literal) AND agrees bit-exactly.
    assert bc.CAMERA_H == int(cp.camera.native_h)
    assert bc.CAMERA_W == int(cp.camera.native_w)
    assert bc.RATE_DENOM == float(cp.video_bytes)
    assert bc._CP_XI_FX == float(cp.camera.fx_native)
    assert bc._CP_XI_CX == float(cp.camera.cx_native)
    assert bc._CP_XI_CY == float(cp.camera.cy_native)
    assert bc._CP_XI_D == float(cp.device_height_m)


# ---------------------------------------------------------------------------
# trainer byte path: src/tac/boundary_math/lane_sdf_component.py
# ---------------------------------------------------------------------------
def _load_lane_sdf():
    return importlib.import_module("tac.boundary_math.lane_sdf_component")


def test_lane_sdf_scorer_intrinsics_bit_identical():
    m = _load_lane_sdf()
    # scorer-resolution intrinsics (profile-sourced) — byte-identical to the historical literals.
    assert m._FX == 400.3
    assert m._FY == 399.5
    assert m._CX == 256.0


def test_lane_sdf_discrepant_constants_not_silently_switched():
    m = _load_lane_sdf()
    # the two routed-to-reconciliation constants MUST remain the hardcoded (non-profile) values.
    assert m._CAM_H == 1.2, "lane-IPM camera height must stay 1.2 (1.2-vs-1.22 reconciliation pending)"
    assert m._V_HORIZON == 174.0, "v_horizon must stay the n600-swept optimum 174 (#327), not profile median 175"


def test_lane_sdf_scorer_intrinsics_track_clip_profile_when_cache_present():
    cp = _clip_profile_or_skip()
    m = _load_lane_sdf()
    assert m._FX == float(cp.camera.fx_scorer)
    assert m._FY == float(cp.camera.fy_scorer)
    assert m._CX == float(cp.camera.cx_scorer)
    # and the profile DISAGREES on the two left-alone constants (the gold discrepancy findings).
    assert cp.v_horizon != m._V_HORIZON, "profile median 175 != swept 174 (documented discrepancy)"
    assert cp.device_height_m != m._CAM_H, "profile 1.22 != lane-IPM 1.2 (documented discrepancy)"
