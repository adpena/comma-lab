# SPDX-License-Identifier: MIT
"""Tests for the #205 WARP-REAL-LUMA FRAME0 pose carrier in tools/levelset_byte_close_and_eval.py.

Covers the byte-close INFRASTRUCTURE (NOT a d_pose claim): the tool-side warp bit-matches the module
authority ``warp_frame0_uint8_numpy``; serialize/parse round-trips; the LVLS1 grammar stays
byte-identical when the carrier is OFF (default-off guarantee) and reads back correctly when ON; the
capped slice is faithful; and the SHIPPED inflate.py string's inlined ``_pcar_*`` functions produce
the SAME frame0 as the tool-side oracle (the verbatim-copy faithfulness the bit-exact gate depends on).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "levelset_byte_close_and_eval", _REPO / "tools" / "levelset_byte_close_and_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


L = _load_tool()


def _rng_frame(h, w, seed):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3)).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 1. tool-side warp-from-H bit-matches the module authority warp_frame0_uint8_numpy
# --------------------------------------------------------------------------- #
def test_warp_from_H_bit_matches_module_authority():
    from tac.boundary_math import warp_real_luma_frame0 as W

    geom = W.GroundHomographyGeom.eon(pitch=0.02)
    src = _rng_frame(L.CAMERA_H, L.CAMERA_W, 7).astype(np.float64)
    xi = W.xi_from_pose_calibration(np.array([34.0, 0.1, -0.1, 0.0, -0.01, 0.0]), 0.16, 1.0, 0.02)
    H = W.homography_from_xi_numpy(xi, geom)  # fp64
    auth = W.warp_frame0_uint8_numpy(src, xi, geom)
    got = L._pcar_warp_frame0_from_H(src, H, (L.CAMERA_H, L.CAMERA_W))
    assert np.array_equal(auth, got), "tool-side warp-from-H must equal the module authority bit-for-bit"


# --------------------------------------------------------------------------- #
# 2. serialize -> parse round-trips (H exact fp64; keyframes exact; kf_of_pair)
# --------------------------------------------------------------------------- #
def test_serialize_parse_round_trip():
    P = 5
    H = np.random.default_rng(1).standard_normal((P, 3, 3)) + np.eye(3)[None]
    xi = np.random.default_rng(2).standard_normal((P, 6)).astype(np.float32)
    kfs = [_rng_frame(40, 50, s) for s in range(3)]
    kf_of_pair = [0, 0, 1, 2, 2]
    hdr_extra = {"s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 2,
                 "kf_store_h": 40, "kf_store_w": 50, "keyframe_lossless_native": False}
    blob = L.serialize_pose_carrier(H, xi, kfs, kf_of_pair, hdr_extra)
    pc = L.parse_pose_carrier(blob)
    assert np.array_equal(pc["H"], H)  # fp64 exact
    assert pc["kf_of_pair"] == kf_of_pair
    assert len(pc["keyframes"]) == 3
    for a, b in zip(pc["keyframes"], kfs, strict=True):
        assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# 3. LVLS1 grammar: default-off byte-identity + carrier read-back (manifest-flag-driven)
# --------------------------------------------------------------------------- #
def test_io_pack_default_off_byte_identical_and_read_back():
    manifest_off = b'{"a":1}'
    base, code, pose = b"BASE", b"CODE", b""
    # OFF: no lane, no pose_carrier -> identical to the 4-block grammar (default-off guarantee).
    packed_off = L._io_pack(manifest_off, base, code, pose or None, None, None)
    # a manifest WITHOUT the pose_carrier flag must read back pose_carrier=None even if trailing bytes
    import json
    m = {"pose_carrier": {"n_pairs": 2}}
    mj = json.dumps(m, separators=(",", ":")).encode()
    pcar = b"PCARBLOB"
    packed_on = L._io_pack(mj, base, code, None, None, pcar)
    man, b2, c2, p2, lane2, pc2 = L._read_blob_bytes(packed_on)
    assert lane2 is None and pc2 == pcar and b2 == base and c2 == code
    # OFF path: 4 blocks only -> reading a no-flag manifest yields both trailing None
    m_off = {"x": 1}
    packed_noflag = L._io_pack(json.dumps(m_off).encode(), base, code, None, None, None)
    _man, _b, _c, _p, lane3, pc3 = L._read_blob_bytes(packed_noflag)
    assert lane3 is None and pc3 is None
    assert packed_off.startswith(L._MAGIC)


# --------------------------------------------------------------------------- #
# 4. _cap_pose_carrier slices to eval_pairs + prunes keyframes
# --------------------------------------------------------------------------- #
def test_cap_pose_carrier():
    P = 6
    H = (np.random.default_rng(3).standard_normal((P, 3, 3)) + np.eye(3)[None])
    xi = np.zeros((P, 6), np.float32)
    kfs = [_rng_frame(20, 24, s) for s in range(6)]  # stride 1 -> 6 keyframes
    kf_of_pair = list(range(6))
    hdr_extra = {"s_t": 1.0, "s_r": 1.0, "pitch": 0.0, "stride": 1,
                 "kf_store_h": 20, "kf_store_w": 24, "keyframe_lossless_native": False}
    blob = L.serialize_pose_carrier(H, xi, kfs, kf_of_pair, hdr_extra)
    capped = L._cap_pose_carrier(blob, 2)
    pc = L.parse_pose_carrier(capped)
    assert int(pc["hdr"]["n_pairs"]) == 2
    assert len(pc["keyframes"]) == 2  # only the 2 referenced keyframes kept
    assert np.array_equal(pc["H"], H[:2])
    for i in range(2):
        assert np.array_equal(pc["keyframes"][pc["kf_of_pair"][i]], kfs[i])


# --------------------------------------------------------------------------- #
# 5. the SHIPPED inflate.py string's inlined _pcar_* == the tool-side oracle (copy faithfulness)
# --------------------------------------------------------------------------- #
def test_inflate_string_pcar_matches_tool_oracle():
    """The bit-exact gate relies on the inflate.py string being a VERBATIM copy of the tool-side warp.
    Exec the inflate string in a namespace and assert its ``_pcar_warp_f0`` / ``_pcar_frame0`` produce
    the SAME uint8 frame0 as the tool-side ``_pcar_warp_frame0_from_H`` / ``pose_carrier_frame0``."""
    ns: dict = {}
    exec(compile(L._INFLATE_PY, "<inflate.py>", "exec"), ns)  # noqa: S102 (test-only, trusted source)
    # native (no upsample): warp-from-H must be bit-identical
    from tac.boundary_math import warp_real_luma_frame0 as W

    geom = W.GroundHomographyGeom.eon(pitch=0.02)
    src = _rng_frame(L.CAMERA_H, L.CAMERA_W, 11).astype(np.float64)
    xi = W.xi_from_pose_calibration(np.array([30.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.16, 1.0, 0.02)
    H = W.homography_from_xi_numpy(xi, geom)
    tool = L._pcar_warp_frame0_from_H(src, H, (L.CAMERA_H, L.CAMERA_W))
    ship = ns["_pcar_warp_f0"](src, H, L.CAMERA_H, L.CAMERA_W)
    assert np.array_equal(tool, ship), "inflate.py _pcar_warp_f0 must equal the tool oracle bit-for-bit"

    # full _pcar_frame0 via a serialized+parsed section (downscaled keyframe -> exercises upsample too)
    P = 2
    Hs = np.stack([H, H])
    kfs = [L._downscale_keyframe(_rng_frame(L.CAMERA_H, L.CAMERA_W, 12), (L.CAMERA_H // 3, L.CAMERA_W // 3))]
    hdr_extra = {"s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 3,
                 "kf_store_h": L.CAMERA_H // 3, "kf_store_w": L.CAMERA_W // 3, "keyframe_lossless_native": False}
    blob = L.serialize_pose_carrier(Hs, np.zeros((P, 6), np.float32), kfs, [0, 0], hdr_extra)
    pc_tool = L.parse_pose_carrier(blob)
    pc_ship = ns["_pcar_parse"](blob)
    for pi in range(P):
        a = L.pose_carrier_frame0(pc_tool, pi)
        b = ns["_pcar_frame0"](pc_ship, pi, L.CAMERA_H, L.CAMERA_W)
        assert np.array_equal(a, b), f"pair {pi}: inflate _pcar_frame0 != tool oracle"


# --------------------------------------------------------------------------- #
# 6. STORE-NOTHING (Track B): serialize with EMPTY keyframes round-trips; mode readable
# --------------------------------------------------------------------------- #
def test_store_nothing_serialize_parse_empty_keyframes():
    P = 4
    H = np.random.default_rng(21).standard_normal((P, 3, 3)) + np.eye(3)[None]
    xi = np.zeros((P, 6), np.float32)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "generator": "witness_render_frame0",
                 "s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 1,
                 "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W, "keyframe_lossless_native": True}
    blob = L.serialize_pose_carrier(H, xi, [], [0] * P, hdr_extra)  # NO keyframes
    pc = L.parse_pose_carrier(blob)
    assert L.pose_carrier_mode(pc) == "store_nothing"
    assert len(pc["keyframes"]) == 0  # store-nothing stores ZERO keyframe luma
    assert np.array_equal(pc["H"], H)
    # the section is tiny (H+xi + hdr only) -- far below a single native keyframe.
    assert len(blob) < 2000, f"store-nothing section should be ~H+xi bytes, got {len(blob)}"


# --------------------------------------------------------------------------- #
# 7. STORE-NOTHING _cap_pose_carrier survives empty keyframes + preserves mode
# --------------------------------------------------------------------------- #
def test_store_nothing_cap_pose_carrier_empty_keyframes():
    P = 6
    H = np.random.default_rng(22).standard_normal((P, 3, 3)) + np.eye(3)[None]
    xi = np.zeros((P, 6), np.float32)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "generator": "witness_render_frame0",
                 "s_t": 1.0, "s_r": 1.0, "pitch": 0.0, "stride": 1,
                 "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W, "keyframe_lossless_native": True}
    blob = L.serialize_pose_carrier(H, xi, [], [0] * P, hdr_extra)
    capped = L._cap_pose_carrier(blob, 2)  # must NOT IndexError on the empty keyframe list
    pc = L.parse_pose_carrier(capped)
    assert int(pc["hdr"]["n_pairs"]) == 2
    assert len(pc["keyframes"]) == 0
    assert L.pose_carrier_mode(pc) == "store_nothing"
    assert np.array_equal(pc["H"], H[:2])


# --------------------------------------------------------------------------- #
# 8. STORE-NOTHING frame0-from-source warp == the module authority warp on the SAME source
#    (proves the generated-frame warp is the identical op as warp_real_luma, just a different src)
# --------------------------------------------------------------------------- #
def test_store_nothing_frame0_from_source_bit_matches_authority():
    from tac.boundary_math import warp_real_luma_frame0 as W

    geom = W.GroundHomographyGeom.eon(pitch=0.02)
    # a GENERATED (non-keyframe) camera-native source -- stands in for the witness's own render.
    src = _rng_frame(L.CAMERA_H, L.CAMERA_W, 33).astype(np.float64)
    xi = W.xi_from_pose_calibration(np.array([28.0, 0.0, 0.05, 0.0, 0.0, 0.0]), 0.16, 1.0, 0.02)
    H = W.homography_from_xi_numpy(xi, geom)
    pc = {"hdr": {"native_h": L.CAMERA_H, "native_w": L.CAMERA_W,
                  "pose_carrier_mode": "store_nothing"}, "H": np.stack([H])}
    got = L.pose_carrier_frame0_from_source(pc, 0, src)
    auth = W.warp_frame0_uint8_numpy(src, xi, geom)  # same warp, same src
    assert np.array_equal(got, auth), "store-nothing warp-from-source must equal the module authority"


# --------------------------------------------------------------------------- #
# 9. STORE-NOTHING build_pose_carrier_section on REAL gt (n6): 0 keyframes, tiny section, mode set
# --------------------------------------------------------------------------- #
def test_store_nothing_build_section_zero_keyframes_real_gt():
    gt = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"
    if not gt.exists():
        pytest.skip("gt_n6 cache not present")
    pc_cfg = {"s_t": 0.16, "s_r": 1.0, "pitch": 0.02, "stride": 1, "downscale": 1, "mode": "store_nothing"}
    blob, manifest, report = L.build_pose_carrier_section(str(gt), 6, pc_cfg)
    assert manifest["mode"] == "store_nothing"
    assert report["n_keyframes"] == 0
    assert report["keyframe_blob_bytes_total"] == 0  # ZERO keyframe payload (the store-nothing win)
    assert report["generator"] == "witness_render_frame0"
    # the whole section is H (72 B/pair) + xi (12 B/pair) + hdr -> tiny vs a stored keyframe.
    assert report["pose_carrier_section_bytes"] < 4000
    pc = L.parse_pose_carrier(blob)
    assert L.pose_carrier_mode(pc) == "store_nothing" and len(pc["keyframes"]) == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
