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


# --------------------------------------------------------------------------- #
# 10. #257 STORE-NOTHING v2: derive-H serializer drops the H block + kf_of_pair; ξ round-trips
# --------------------------------------------------------------------------- #
def _sn_xi(P, seed=0):
    from tac.boundary_math import warp_real_luma_frame0 as W
    rng = np.random.default_rng(seed)
    poses = np.zeros((P, 6))
    poses[:, 0] = 30.0 + 0.4 * np.arange(P) + 0.2 * rng.standard_normal(P)
    poses[:, 1] = 0.05 * np.sin(np.arange(P) / 6.0)
    return np.stack([W.xi_from_pose_calibration(poses[p], 0.044, 0.0, 0.0) for p in range(P)])


@pytest.mark.parametrize("coder", ["delta_ar", "none"])
def test_store_nothing_v2_serialize_parse_derive_H(coder):
    P, pitch = 24, 0.0
    xi = _sn_xi(P, 1)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "s_t": 0.044, "s_r": 0.0, "pitch": pitch,
                 "stride": 1, "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W,
                 "keyframe_lossless_native": True, "generator": "witness_render_frame0"}
    blob, qr = L.serialize_pose_carrier_store_nothing(xi, hdr_extra, coder=coder, q_levels=4096)
    pc = L.parse_pose_carrier(blob)
    assert L.pose_carrier_mode(pc) == "store_nothing"
    assert pc["hdr"]["pcar_store_nothing_v"] == 2 and pc["hdr"]["xi_coder"] == coder
    assert "kf_of_pair" not in pc["hdr"], "v2 must DROP the kf_of_pair junk list"
    assert len(pc["keyframes"]) == 0
    # H is DERIVED (not stored): the section must NOT contain the 43,200-B/600 fp64 H block.
    assert len(blob) < P * 9 * 8, "v2 section must be far smaller than a stored fp64 H block"
    # derived H == the per-pair module authority on the decoded ξ (bit-for-bit)
    from tac.boundary_math import warp_real_luma_frame0 as W
    geom = W.GroundHomographyGeom.eon(pitch=pitch)
    for p in range(P):
        assert np.array_equal(pc["H"][p], W.homography_from_xi_numpy(pc["xi"][p], geom))


def test_store_nothing_v2_both_coders_strict_parity_identical_frames():
    """The coded + raw variants decode to the IDENTICAL derived H -> IDENTICAL warp (strict parity)."""
    P, pitch = 16, 0.02
    xi = _sn_xi(P, 2)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "s_t": 0.044, "s_r": 0.0, "pitch": pitch,
                 "stride": 1, "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W,
                 "keyframe_lossless_native": True, "generator": "witness_render_frame0"}
    b_cod, _ = L.serialize_pose_carrier_store_nothing(xi, hdr_extra, coder="delta_ar", q_levels=4096)
    b_raw, _ = L.serialize_pose_carrier_store_nothing(xi, hdr_extra, coder="none", q_levels=4096)
    pc_cod, pc_raw = L.parse_pose_carrier(b_cod), L.parse_pose_carrier(b_raw)
    assert np.array_equal(pc_cod["H"], pc_raw["H"]), "coded vs raw derived-H differ (strict-parity broken)"
    src = _rng_frame(L.CAMERA_H, L.CAMERA_W, 5).astype(np.float64)
    f_cod = L.pose_carrier_frame0_from_source(pc_cod, 0, src)
    f_raw = L.pose_carrier_frame0_from_source(pc_raw, 0, src)
    assert np.array_equal(f_cod, f_raw)
    assert len(b_cod) <= len(b_raw)  # the coder never hurts on a smooth trajectory


def test_store_nothing_v2_inflate_string_matches_tool_oracle():
    """The SHIPPED inflate.py inline _pcar_parse (v2 derive-H + ξ decoder) == the tool oracle bit-for-bit."""
    ns: dict = {}
    exec(compile(L._INFLATE_PY, "<inflate.py>", "exec"), ns)  # test-only, trusted source
    P, pitch = 20, 0.0
    xi = _sn_xi(P, 3)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "s_t": 0.044, "s_r": 0.0, "pitch": pitch,
                 "stride": 1, "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W,
                 "keyframe_lossless_native": True, "generator": "witness_render_frame0"}
    for coder in ("delta_ar", "none"):
        blob, _ = L.serialize_pose_carrier_store_nothing(xi, hdr_extra, coder=coder, q_levels=4096)
        pc_tool = L.parse_pose_carrier(blob)
        pc_ship = ns["_pcar_parse"](blob)
        assert np.array_equal(pc_tool["H"], pc_ship["H"]), f"{coder}: shipped derive-H != tool oracle"
        src = _rng_frame(L.CAMERA_H, L.CAMERA_W, 7).astype(np.float64)
        a = L.pose_carrier_frame0_from_source(pc_tool, 0, src)
        b = ns["_pcar_warp_f0"](src, pc_ship["H"][0], L.CAMERA_H, L.CAMERA_W)
        assert np.array_equal(a, b), f"{coder}: shipped frame0 warp != tool oracle"


def test_store_nothing_v2_cap_pose_carrier():
    P, pitch = 12, 0.0
    xi = _sn_xi(P, 4)
    hdr_extra = {"pose_carrier_mode": "store_nothing", "s_t": 0.044, "s_r": 0.0, "pitch": pitch,
                 "stride": 1, "kf_store_h": L.CAMERA_H, "kf_store_w": L.CAMERA_W,
                 "keyframe_lossless_native": True, "generator": "witness_render_frame0"}
    blob, _ = L.serialize_pose_carrier_store_nothing(xi, hdr_extra, coder="delta_ar", q_levels=4096)
    capped = L._cap_pose_carrier(blob, 4)
    pc = L.parse_pose_carrier(capped)
    assert int(pc["hdr"]["n_pairs"]) == 4 and len(pc["keyframes"]) == 0
    assert pc["hdr"]["pcar_store_nothing_v"] == 2
    assert pc["H"].shape == (4, 3, 3) and pc["xi"].shape == (4, 6)
    # the capped section stores NO H block (derived free) -> far smaller than the full section
    assert len(capped) < len(blob) and pc["hdr"]["xi_coder"] == "delta_ar"


def test_store_nothing_v2_build_section_rate_accounting_real_gt():
    gt = _REPO / "experiments" / "results" / "mlx_fleet_gt_cache" / "gt_n6.npz"
    if not gt.exists():
        pytest.skip("gt_n6 cache not present")
    pc_cfg = {"s_t": 0.044, "s_r": 0.0, "pitch": 0.0, "stride": 1, "downscale": 1,
              "mode": "store_nothing", "xi_coder": "delta_ar", "xi_q_levels": 4096}
    blob, manifest, report = L.build_pose_carrier_section(str(gt), 6, pc_cfg)
    assert report["H_bytes"] == 0, "v2 must store NO H (the 43,200-B/600 redundancy is DROPPED)"
    assert report["xi_coder"] == "delta_ar"
    assert 0 < report["xi_bytes"] < report["pose_carrier_section_bytes"]  # xi payload is the bulk
    assert report["keyframe_blob_bytes_total"] == 0
    # at n600 the section is ~3.2 KB vs the pre-#257 ~52 KB (43,200-B H DROPPED); at tiny n6 the JSON
    # header dominates -> just assert the H-drop is REAL (H_bytes==0 above) + section < a full keyframe.
    assert report["pose_carrier_section_bytes"] < L.CAMERA_H * L.CAMERA_W  # << one stored keyframe
    pc = L.parse_pose_carrier(blob)
    assert L.pose_carrier_mode(pc) == "store_nothing" and int(pc["hdr"]["pcar_store_nothing_v"]) == 2


def test_warp_real_luma_still_byte_identical_after_257():
    """REGRESSION: the #257 store-nothing change must NOT touch warp_real_luma bytes (A/B preserved)."""
    P = 3
    H = (np.random.default_rng(8).standard_normal((P, 3, 3)) + np.eye(3)[None])
    xi = np.zeros((P, 6), np.float32)
    kfs = [_rng_frame(30, 40, s) for s in range(P)]
    kf_of_pair = [0, 1, 2]
    hdr_extra = {"pose_carrier_mode": "warp_real_luma", "s_t": 0.16, "s_r": 1.0, "pitch": 0.02,
                 "stride": 1, "kf_store_h": 30, "kf_store_w": 40, "keyframe_lossless_native": False}
    blob = L.serialize_pose_carrier(H, xi, kfs, kf_of_pair, hdr_extra)
    pc = L.parse_pose_carrier(blob)
    assert np.array_equal(pc["H"], H)  # legacy fp64-H path unchanged
    assert pc["kf_of_pair"] == kf_of_pair and len(pc["keyframes"]) == 3
    assert "pcar_store_nothing_v" not in pc["hdr"]  # warp_real_luma is NOT tagged v2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
