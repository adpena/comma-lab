# SPDX-License-Identifier: MIT
"""GROUND-FRAME CHART — per-frame coordinate pre-composition for the trained witness (#194 / §17.1).

THE OBJECT (council draft §17.1, DAG FEED-ll stratification): define the witness field ONCE in a
canonical GROUND frame (the reference pair's image chart); evaluate frame ``t`` by PRE-COMPOSING the
witness's input coordinates with the per-frame ξ-homography (a CHART CHANGE on the INPUT coords, NOT
a pixel warp of the OUTPUT). The field is still TRAINED through R + the frozen scorer, so this does
NOT inherit the FEED-ll / #190 DETERMINISTIC-render d_seg floor (~0.0185 bulk) — that floor belongs
to the warp-a-stored-partition MATERIALIZER, a different (measured-dead) object.

One ξ — already stored/derived for the pose axis (``xi_pose_coder`` #257, dual-use, ZERO new archive
bytes) — buys three measured residuals:
  1. temporal flicker (44 % lane spikes): one canonical field cannot disagree with itself;
  2. the dash-comb corrector's natural home (dashes STATIC in the ground frame; phase transport =
     this chart change);
  3. §15 pinning/zero-mobility dissolves by construction (per-frame-chart pathology).

MATH (REUSED from the MEASURED FEED-ll reach gate ``tools/measure_screw_reach_through_R.py`` — the
same per-step plane-motion matrices and the same cumulative left-multiply composition; a bit-parity
test pins this module to the tool so the math cannot drift):

  per-step plane motion (regime-stratified, FEED-iz):
    ground  : M = R − t nᵀ / d      (plane-induced homography; Road/Lane/Movable)
    rotonly : M = R                 (depth→∞ sky; Undrivable)
    identity: M = I                 (ego hood; MyCar)
  cumulative source→target (fixed-plane model):
    M_cum(a→a+k) = M_{a+k} @ … @ M_{a+1};   H_fwd = K · M_cum · K⁻¹     (pixel coords)
  the CHART (this module's new object): the witness input chart for frame t is the INVERSE map
    x_ref = H_fwd(ref→t)⁻¹ · x_t
  expressed in the trainer's NORMALIZED [-1,1]² render coordinates via the affine grid map A:
    H_chart_norm[t] = A⁻¹ · H_fwd(ref→t)⁻¹ · A
  ``chart[ref] == identity`` EXACTLY by construction (M_cum over an empty product).

TRAINER WIRE-IN v0 = GROUND-plane single chart (NOT per-class stratified routing). Rationale: the
chart acts on INPUT COORDS — per-class routing needs the class, which the field itself predicts;
K-chart per-class blending is ``screw_blend``'s future consumer (designed, not v0). Ground covers
Road/Lane/Movable (~82 % of the measured flip mass); for the static bulk (Undrivable/MyCar) a ground
chart merely re-introduces the mild apparent motion every class has under today's per-frame chart —
the status quo, not a regression. The module still implements all three regimes (the math is shared)
so the stratified arm needs no new geometry.

AUTHORITY / DETERMINISM: numpy fp64 builds the homographies (deterministic, host-portable); the
coordinate pre-composition itself is fp32 (matching the trainer's fp32 coord grid) with a NumPy
reference + an MLX twin implemented op-for-op (bit-identical on the CPU stream; MLX-GPU is NOT
bit-identical cross-process — the standing discipline — so bit-exact proofs are CPU-locked, GPU is
throughput only). Every scored number downstream is advisory until an n600 byte-closed
``upstream/evaluate.py`` row; the pointer 0.19110 moves only through that.

rule-118: the chart is a FREE deterministic function of the already-COUNTED ξ table + fixed camera
calibration (``clip_profile`` intrinsics — NEVER hardcoded clip constants) → 0 new archive bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ChartCalibration",
    "GroundFrameChart",
    "GroundFrameChartError",
    "plane_motion_step",
    "cumulative_plane_motion",
    "chart_homography_pixel",
    "intrinsics_for_grid",
    "normalization_affine",
    "precompose_coords_numpy",
    "precompose_coords_mlx",
    "REGIMES",
]

REGIMES = ("ground", "rotonly", "identity")
_EPS_Z = 1e-8  # sign-preserving projective z guard (fp32-safe)


class GroundFrameChartError(ValueError):
    """A ground-frame-chart contract violation (shape / regime / config)."""


@dataclass(frozen=True)
class ChartCalibration:
    """The 3 global calibration scalars of the pose→plane-motion map (FEED-ll fit protocol).

    Defaults are the MEASURED FEED-ll reach-gate calibration (n96, fit on the real
    lstars[p]→lstars[p+1] step; ``experiments/results/screw_reach/reach_n96.json``):
    s_t = -0.003224707899359239, s_r = 0.0, pitch = -0.01. Callers with a better
    (e.g. d_pose-optimal) calibration pass their own — nothing here is clip-hardcoded
    beyond a measured-default that the flag surface overrides.
    """

    s_t: float = -0.003224707899359239
    s_r: float = 0.0
    pitch: float = -0.01


def _camera_height_m() -> float:
    from tac.clip_profile import OPENPILOT_DEVICE_HEIGHT_M

    return float(OPENPILOT_DEVICE_HEIGHT_M)


def intrinsics_for_grid(w: int, h: int, camera: Any | None = None) -> np.ndarray:
    """EON/openpilot K scaled to a (w, h) working grid via ``tac.clip_profile`` (no hardcodes).

    ``camera`` is an optional ``clip_profile.ClipCamera``; default = the canonical road camera
    (``camera_for_resolution(CAMERA_W, CAMERA_H)``). Matches the FEED-ll tool's ``intrinsics_at``
    exactly for the canonical camera (parity-tested)."""
    if camera is None:
        from tac.camera import CAMERA_H, CAMERA_W
        from tac.clip_profile import camera_for_resolution

        camera = camera_for_resolution(CAMERA_W, CAMERA_H)
    sx = float(w) / float(camera.native_w)
    sy = float(h) / float(camera.native_h)
    return np.array(
        [[camera.fx_native * sx, 0.0, camera.cx_native * sx],
         [0.0, camera.fy_native * sy, camera.cy_native * sy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# per-step + cumulative plane motion (bit-parity with the FEED-ll reach tool)
# --------------------------------------------------------------------------- #
def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    """Rodrigues axis-angle → rotation (op-for-op the reach tool's ``_expmap_so3``)."""
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def plane_motion_step(pose6: np.ndarray, calib: ChartCalibration, regime: str) -> np.ndarray:
    """One step's 3×3 plane-motion matrix M (regime-stratified; == reach tool ``_m_step``).

    ground: M = R − t nᵀ/d · identity-height; rotonly: M = R; identity: M = I.
    Pose column convention (INFERRED, the reach tool's): raw PoseNet 6-vec
    ``[fwd, lat, vert, r0, r1, r2]``; t = s_t·(x=col2, y=col1, z=col0)."""
    if regime not in REGIMES:
        raise GroundFrameChartError(f"regime {regime!r} not in {REGIMES}")
    if regime == "identity":
        return np.eye(3, dtype=np.float64)
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)
    R = _expmap_so3(calib.s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    if regime == "rotonly":
        return R
    t = calib.s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    n = np.array([0.0, -np.cos(calib.pitch), -np.sin(calib.pitch)], dtype=np.float64)
    return R - np.outer(t, n) / _camera_height_m()


def cumulative_plane_motion(
    poses: np.ndarray, src: int, dst: int, calib: ChartCalibration, regime: str
) -> np.ndarray:
    """Cumulative plane-motion M_cum for src→dst (either direction; fixed-plane model).

    dst ≥ src: ``M = M_dst @ … @ M_{src+1}`` (the reach tool's left-multiply loop, bit-parity).
    dst < src: the exact inverse of the forward product (chart for pairs BEFORE the reference).
    src == dst → identity."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 6:
        raise GroundFrameChartError(f"poses must be (P, 6); got {poses.shape}")
    P = poses.shape[0]
    lo, hi = (src, dst) if dst >= src else (dst, src)
    if not (0 <= lo and hi < P):
        raise GroundFrameChartError(f"src/dst ({src},{dst}) out of range [0,{P - 1}]")
    if regime == "identity" or src == dst:
        return np.eye(3, dtype=np.float64)
    M = np.eye(3, dtype=np.float64)
    for i in range(lo + 1, hi + 1):
        M = plane_motion_step(poses[i], calib, regime) @ M  # later steps left-multiply
    return M if dst >= src else np.linalg.inv(M)


def chart_homography_pixel(
    poses: np.ndarray, src: int, dst: int, calib: ChartCalibration, regime: str, K: np.ndarray
) -> np.ndarray:
    """Forward pixel homography H(src→dst) = K · M_cum(src→dst) · K⁻¹.

    For src=a, dst=a+k, regime='ground'/'rotonly' this is BIT-PARITY with the FEED-ll reach
    tool's ``cumulative_homography(poses, a, k, K, Kinv, params, regime)`` (pinned by test)."""
    K = np.asarray(K, dtype=np.float64)
    if src == dst or regime == "identity":
        # exact identity — mirrors the reach tool's `k == 0 or regime == "identity"` fast path
        # (and ONLY that case: the tool does NOT fast-path a numerically-identity M at k > 0,
        # so neither do we — bit-parity over convenience).
        return np.eye(3, dtype=np.float64)
    M = cumulative_plane_motion(poses, src, dst, calib, regime)
    return K @ M @ np.linalg.inv(K)


def normalization_affine(h: int, w: int) -> np.ndarray:
    """Affine A mapping the trainer's normalized [-1,1]² coords → pixel coords [0,w-1]×[0,h-1].

    Matches ``_build_render_coords`` (endpoint-inclusive linspace): x_pix = (x+1)/2·(w−1)."""
    return np.array(
        [[(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
         [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# coordinate pre-composition (fp32; NumPy reference + MLX twin, op-for-op)
# --------------------------------------------------------------------------- #
def precompose_coords_numpy(coords: np.ndarray, H_norm: np.ndarray) -> np.ndarray:
    """Apply a normalized-coords homography to an (N, 2) fp32 coord grid (NumPy reference).

    Identity fast-path returns the INPUT ARRAY UNCHANGED (bit-exact byte-identity when the
    chart is trivial). Projective divide uses a sign-preserving z guard (no inf/nan)."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise GroundFrameChartError(f"coords must be (N, 2); got {coords.shape}")
    H64 = np.asarray(H_norm, dtype=np.float64)
    if H64.shape != (3, 3):
        raise GroundFrameChartError(f"H_norm must be (3, 3); got {H64.shape}")
    if np.array_equal(H64, np.eye(3)):
        return coords
    H = H64.astype(np.float32)
    x, y = coords[:, 0], coords[:, 1]
    xw = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    yw = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    zw = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    z = np.where(np.abs(zw) < _EPS_Z, np.where(zw >= 0.0, _EPS_Z, -_EPS_Z).astype(np.float32), zw)
    return np.stack([xw / z, yw / z], axis=-1).astype(np.float32)


def precompose_coords_mlx(coords_mx: Any, H_norm: np.ndarray) -> Any:
    """MLX twin of :func:`precompose_coords_numpy` (op-for-op fp32; CPU-stream bit-parity)."""
    import mlx.core as mx

    H64 = np.asarray(H_norm, dtype=np.float64)
    if H64.shape != (3, 3):
        raise GroundFrameChartError(f"H_norm must be (3, 3); got {H64.shape}")
    if np.array_equal(H64, np.eye(3)):
        return coords_mx
    H = H64.astype(np.float32)
    x, y = coords_mx[:, 0], coords_mx[:, 1]
    xw = float(H[0, 0]) * x + float(H[0, 1]) * y + float(H[0, 2])
    yw = float(H[1, 0]) * x + float(H[1, 1]) * y + float(H[1, 2])
    zw = float(H[2, 0]) * x + float(H[2, 1]) * y + float(H[2, 2])
    eps = mx.array(_EPS_Z, dtype=mx.float32)
    z = mx.where(mx.abs(zw) < eps, mx.where(zw >= 0.0, eps, -eps), zw)
    return mx.stack([xw / z, yw / z], axis=-1)


# --------------------------------------------------------------------------- #
# the chart object (per-run static: ξ table is fixed → build ONCE)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundFrameChart:
    """Per-pair chart-change homographies for a witness render grid, built from the ξ/pose table.

    ``H_chart_norm[t]`` maps frame-t NORMALIZED render coords → reference-frame normalized
    coords (the witness input pre-composition). ``H_fwd_pix[t]`` is the forward ref→t pixel
    homography (the reach-tool-parity surface). ``chart[ref] == identity`` exactly."""

    ref_pair: int
    regime: str
    grid_hw: tuple[int, int]
    calib: ChartCalibration
    H_fwd_pix: np.ndarray     # (P, 3, 3) fp64: ref→t forward pixel homography
    H_chart_norm: np.ndarray  # (P, 3, 3) fp64: frame-t normalized coords → ref normalized coords
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        return int(self.H_chart_norm.shape[0])

    @staticmethod
    def build(
        poses: np.ndarray,
        *,
        ref_pair: int = 0,
        calib: ChartCalibration | None = None,
        grid_hw: tuple[int, int],
        regime: str = "ground",
        K: np.ndarray | None = None,
    ) -> "GroundFrameChart":
        """Build the per-pair chart from the pose/ξ table (fp64, deterministic, ~ms).

        ``poses`` (P, 6) = the stored per-pair pose table (dual-use with the pose sidecar —
        zero new archive bytes). ``grid_hw`` = (render_h, render_w). ``K`` defaults to the
        canonical clip intrinsics scaled to the grid (``intrinsics_for_grid``)."""
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 6:
            raise GroundFrameChartError(f"poses must be (P, 6); got {poses.shape}")
        if regime not in REGIMES:
            raise GroundFrameChartError(f"regime {regime!r} not in {REGIMES}")
        P = int(poses.shape[0])
        ref = int(ref_pair)
        if not (0 <= ref < P):
            raise GroundFrameChartError(f"ref_pair {ref} out of range [0,{P - 1}]")
        cal = calib if calib is not None else ChartCalibration()
        h, w = int(grid_hw[0]), int(grid_hw[1])
        Kp = np.asarray(K, dtype=np.float64) if K is not None else intrinsics_for_grid(w, h)
        A = normalization_affine(h, w)
        Ainv = np.linalg.inv(A)
        H_fwd = np.empty((P, 3, 3), dtype=np.float64)
        H_chart = np.empty((P, 3, 3), dtype=np.float64)
        # incremental composition ref→t (O(P) total, not O(P²)); exact identity at ref.
        Kinv = np.linalg.inv(Kp)
        M_up = np.eye(3, dtype=np.float64)   # M_cum(ref→t) for t ≥ ref
        for t in range(ref, P):
            if t > ref:
                M_up = plane_motion_step(poses[t], cal, regime) @ M_up
            H_fwd[t] = np.eye(3) if t == ref else Kp @ M_up @ Kinv
        M_dn = np.eye(3, dtype=np.float64)   # M_cum(t→ref) for t < ref (forward product t→ref)
        for t in range(ref - 1, -1, -1):
            M_dn = M_dn @ plane_motion_step(poses[t + 1], cal, regime)
            # M_cum(ref→t) = inv(M_cum(t→ref))
            H_fwd[t] = Kp @ np.linalg.inv(M_dn) @ Kinv
        for t in range(P):
            if t == ref:
                H_chart[t] = np.eye(3, dtype=np.float64)
            else:
                H_chart[t] = Ainv @ np.linalg.inv(H_fwd[t]) @ A
        return GroundFrameChart(
            ref_pair=ref, regime=regime, grid_hw=(h, w), calib=cal,
            H_fwd_pix=H_fwd, H_chart_norm=H_chart,
            provenance={
                "builder": "tac.boundary_math.ground_frame_chart.GroundFrameChart.build",
                "n_pairs": P, "ref_pair": ref, "regime": regime, "grid_hw": [h, w],
                "calib": {"s_t": cal.s_t, "s_r": cal.s_r, "pitch": cal.pitch},
                "rule_118": "chart derived FREE from the counted ξ/pose table + fixed calibration",
            },
        )

    def coords_for_pair_numpy(self, coords: np.ndarray, pair_idx: int) -> np.ndarray:
        """Frame-``pair_idx`` normalized render coords → ground(ref)-frame coords (fp32 ref)."""
        return precompose_coords_numpy(coords, self.H_chart_norm[int(pair_idx)])

    def coords_for_pair_mlx(self, coords_mx: Any, pair_idx: int) -> Any:
        """MLX twin of :meth:`coords_for_pair_numpy` (op-for-op fp32)."""
        return precompose_coords_mlx(coords_mx, self.H_chart_norm[int(pair_idx)])
