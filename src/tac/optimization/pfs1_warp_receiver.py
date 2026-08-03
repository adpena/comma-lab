# SPDX-License-Identifier: MIT
"""pfs1 warp receiver (vendored, generic, deterministic; rule 118).

Byte-faithful copy of the warp engine in tools/measure_pose_warp_dseg.py +
tools/measure_screw_warp_through_R.py used by p3v2 warp_base_work, with the EON
road-camera intrinsics + camera height hardcoded to the documented literals
(comma2k19 utils/camera.py: focal 910, FULL_FRAME 1164x874, pp=(582,437);
openpilot HEIGHT_INIT 1.22).  clip_profile reproduces these bit-identically on
0.mkv, so this receiver needs NO tac dependency."""
from __future__ import annotations

import numpy as np

NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22
CAMERA_H, CAMERA_W = 874, 1164
ST_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]


def intrinsics_native() -> np.ndarray:
    return np.array([[NATIVE_FX, 0.0, NATIVE_CX],
                     [0.0, NATIVE_FY, NATIVE_CY],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _target_grid(hh: int, ww: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(ww), np.arange(hh))
    return np.stack([us.ravel(), vs.ravel(), np.ones(hh * ww)], 0).astype(np.float64)


def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def pose_to_homography(pose6, K, Kinv, s_t, s_r, pitch) -> np.ndarray:
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    M = R - np.outer(t, n) / CAMERA_HEIGHT_M
    return K @ M @ Kinv


def regime_homography_ground(pose6, K, Kinv, s_t) -> np.ndarray:
    return pose_to_homography(pose6, K, Kinv, s_t, 0.0, 0.0)


def warp_rgb(src_hwc: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray) -> np.ndarray:
    Hh, Ww, C = src_hwc.shape
    srcf = src_hwc.astype(np.float64)
    flat = srcf.reshape(-1, C)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (
        np.isfinite(su) & np.isfinite(sv) & (z > 0)
        & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1)
    )
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat)
    return out.reshape(Hh, Ww, C)


def _to_uint8(frame_f: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame_f), 0.0, 255.0).astype(np.uint8)


def warp_base_f0(f1_u8: np.ndarray, target6: np.ndarray, s_t: float) -> np.ndarray:
    """Deterministic camera-res uint8 frame_0 = ground-homography warp of f1 by
    the carried pose target at translation-scale s_t.  Byte-identical to p3v2
    warp_base_work's best_cam for the stored s_t index."""
    K = intrinsics_native()
    Kinv = np.linalg.inv(K)
    grid = _target_grid(CAMERA_H, CAMERA_W)
    H = regime_homography_ground(np.asarray(target6, np.float64), K, Kinv, float(s_t))
    return _to_uint8(warp_rgb(np.asarray(f1_u8, np.float64), H, grid))
