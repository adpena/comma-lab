# SPDX-License-Identifier: MIT
"""Trainer STORE-NOTHING render-path (#205 Track B, --pose-carrier-source generated).

The trainer's store-nothing f0 render (in experiments/train_levelset_witness_realized_through_R_mlx.py)
composes: the witness's OWN plain frame0 render -> up-to-camera-native (apply_contest_faithful_roundtrip
_nhwc, output_hw=CAMERA_HW; the R "up" step, == the byte-close store_nothing warp source _R) ->
carrier.render_f0 (SE(3) warp by the twist + R-down to SEG). These tests validate that composition on
the REAL carrier + R primitives (NOT the full trainer loop): shape contract + the warp is ACTIVE (a
non-zero twist moves the frame) + the store-nothing source is camera-native.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tac.boundary_math import warp_real_luma_frame0 as W  # noqa: E402
from tac.local_acceleration.pr95_hnerv_mlx_training import (  # noqa: E402
    CAMERA_HW,
    apply_contest_faithful_roundtrip_nhwc,
)


def _fake_witness_render(rh: int, rw: int, seed: int):
    """A stand-in for the witness's plain frame0 render (render_res RGB in [0,255], NHWC)."""
    rgb = np.random.default_rng(seed).uniform(0, 255, (1, rh, rw, 3)).astype(np.float32)
    return mx.array(rgb)


def test_up_to_camera_source_is_native_camera_res():
    # the store-nothing warp SOURCE = the witness render bicubic-up to camera-native uint8.
    rgb = _fake_witness_render(96, 128, 1)
    src = apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=CAMERA_HW, ste_round=True)
    assert tuple(src.shape) == (1, CAMERA_HW[0], CAMERA_HW[1], 3)  # (1, 874, 1164, 3)
    src0 = np.asarray(src[0])
    assert src0.shape == (CAMERA_HW[0], CAMERA_HW[1], 3)
    assert float(src0.min()) >= 0.0 and float(src0.max()) <= 255.0


def test_store_nothing_f0_shape_and_warp_active():
    # build a real (tiny) table-mode carrier at camera-native geom; warp the witness's up-to-camera
    # render -> the SEG-res f0 contract, and prove the warp is ACTIVE (non-zero twist moves the frame).
    P = 2
    geom = W.GroundHomographyGeom.eon(native_hw=CAMERA_HW, pitch=0.0)
    xi_stored = np.stack([W.xi_from_pose_calibration(
        np.array([30.0, 0.0, 0.05 * (p + 1), 0.0, 0.0, 0.0]), 0.16, 0.0, 0.0) for p in range(P)]).astype(np.float32)
    carrier = W.WarpRealLumaFrame0Carrier.build(xi_stored, geom, residual_mode="table", residual_scale=1.0)
    mx.eval(carrier.parameters())
    impl = carrier.impl

    rgb = _fake_witness_render(96, 128, 7)
    src_native = apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=CAMERA_HW, ste_round=True)[0]
    f0 = impl.render_f0(src_native, 0, None, ste_round=True)
    mx.eval(f0)
    assert tuple(f0.shape) == (1, W.SEG_H, W.SEG_W, 3)  # SEG-res contract (matches the f1 witness render)
    f0_np = np.asarray(f0)
    assert float(f0_np.min()) >= 0.0 and float(f0_np.max()) <= 255.0

    # the warp is ACTIVE: the store-nothing f0 (warped witness render) differs from the witness render
    # simply R-roundtripped to SEG WITHOUT the warp (a non-zero twist moves pixels).
    seg_no_warp = apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=(W.SEG_H, W.SEG_W), ste_round=True)
    mx.eval(seg_no_warp)
    diff = float(np.abs(f0_np.astype(np.int32) - np.asarray(seg_no_warp).astype(np.int32)).mean())
    assert diff > 0.5, f"store-nothing warp appears inactive (mean |diff|={diff}); the twist should move f0"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
