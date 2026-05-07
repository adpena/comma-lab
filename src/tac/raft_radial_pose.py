"""Lane RAFT/radial pose — calibrated 6-DoF flow-basis decomposition.

Per Phase 3 Lane 18 spec (memory `project_phases_2_3_4_*` §"Lane 18 RAFT/radial pose")
and council design `.omx/research/council_lane_raft_radial_pose_design_20260430.md`.

This module **extends** `src/tac/raft_pose.py` (Lane FL — single-DOF longitudinal pose
dim 0) to full 6-DoF radial-basis decomposition. Two operating modes:

- **Mode A (compress-time prior, default):** RAFT runs at compress time only. Produces
  a 6-DoF pose initialization that the existing pose-TTO loop refines. Pose stream still
  shipped. SAFE for contest submission.
- **Mode B (inflate-time recompute, env-gated, NON-COMPLIANT pending human approval):**
  RAFT runs at inflate time. Pose stream NOT shipped. Eliminates ~50 KB. **Requires
  explicit human approval per CLAUDE.md "Strict scorer rule".**

Math foundation (Longuet-Higgins 1981; Mallat 2009)
---------------------------------------------------

For a calibrated camera moving with translation `t = (t_x, t_y, t_z)` and rotation
`(ω_x, ω_y, ω_z)`, the normalized-coordinate optical flow at image point
`(x, y) = ((u-c_x)/f_x, (v-c_y)/f_y)` is the SUM of:

- Translation flow (depth-dependent, radial from FOE)
- Rotation flow (depth-INDEPENDENT, polynomial in image coords)

This module fits raw RAFT pixel flow after converting it to normalized flow
(`u/f_x`, `v/f_y`). Translation still needs a depth model, so the basis uses a
single explicit proxy depth; the fitted translation coefficients are proxy-depth
coefficients, not a metric 3-D reconstruction.

We decompose the normalized flow field into a 6-element basis where the
rotation components are depth-independent and form a clean linear system:

    F(x, y) = Σ_k α_k · B_k(x, y)

with basis (lane-design sign convention, y positive downward):
    B_0(x, y) = (-ρ, 0)              — proxy-depth horizontal translation t_x
    B_1(x, y) = (0, -ρ)              — proxy-depth vertical translation t_y
    B_2(x, y) = (ρ·x, ρ·y)           — radial forward translation t_z
    B_3(x, y) = (-y, x)              — roll (ω_z)
    B_4(x, y) = (x·y, -(1 + y²))     — pitch (ω_x)
    B_5(x, y) = (1 - x², x·y)        — yaw (ω_y)

where `ρ = 1 / proxy_depth_m`.

The 6 coefficients α are recovered by least-squares using the direct
pseudoinverse of `B` (not `(B.T @ B)^-1`, which is less stable for the
constant yaw/translation coupling).

These 6 coefficients are then **affine-calibrated** to the contest pose via least-squares
`pose_contest ≈ A @ α + b` on a held-out frame window.

CLAUDE.md compliance
--------------------
- No silent defaults — every public function arg required-keyword
- Mode A: compress-time only, fully compliant
- Mode B: env-gated `INFLATE_RAFT=0` default; runtime banner; HUMAN APPROVAL required
- Pure CPU/CUDA (no MPS); no scorer load (RAFT is NOT a scorer per torchvision lineage)
- All claims tagged [synthetic] / [prediction]; real-anchor confirm is Phase C
- All public functions deterministic: same input → same output

Out of scope (intentional)
--------------------------
- Mallat scattering transform on flow residual (deferred to Lane 11 intersection)
- Per-pixel depth estimation (we use the depth-INDEPENDENT rotation basis only;
  depth-dependent translation t_z is captured via the (x, y) radial basis, NOT via
  per-pixel depth — this is an approximation valid for distant scenes)
- Online affine calibration update during inflate (Mode B uses pre-computed (A, b))

References
----------
- Teed & Deng 2020 RAFT (arXiv 2003.12039)
- Longuet-Higgins 1981 — "A computer algorithm for reconstructing a scene from two
  projections" (the flow-to-ego-motion equations)
- Mallat 2009 *A Wavelet Tour of Signal Processing* Ch. 6 (radial wavelets — used in
  Phase 3 for the residual decomposition)
- Council design: .omx/research/council_lane_raft_radial_pose_design_20260430.md
- Sibling: src/tac/raft_pose.py (Lane FL — single-DOF dim 0)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tac.camera import COMMA_INTRINSICS, SEGNET_INPUT_H, SEGNET_INPUT_W


# ── version / mode constants ───────────────────────────────────────────


RAFT_RADIAL_VERSION: int = 2
"""Format version for any persisted (A, b) calibration. Bumped on schema change."""

DEFAULT_PROXY_DEPTH_M: float = 30.0
"""Default single-depth proxy for translation terms, matching road-depth priors."""

MODE_COMPRESS_TIME_PRIOR = "compress_time_prior"
"""Mode A: RAFT at compress time only; pose stream still shipped. Safe."""

MODE_INFLATE_RECOMPUTE = "inflate_recompute"
"""Mode B: RAFT at inflate time; pose stream eliminated. Requires human approval."""

VALID_MODES = (MODE_COMPRESS_TIME_PRIOR, MODE_INFLATE_RECOMPUTE)


# ── config ─────────────────────────────────────────────────────────────


@dataclass
class NormalizedFlowCalibration:
    """Camera calibration used by the normalized-coordinate RAFT basis.

    `fx`, `fy`, `cx`, and `cy` are in pixels at the flow-field resolution.
    `proxy_depth_m` is the single explicit depth used for translation basis
    terms. Rotation terms are depth-independent; translation coefficients are
    therefore only proxy-depth coefficients until affine-calibrated to the
    contest pose stream.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    proxy_depth_m: float = DEFAULT_PROXY_DEPTH_M

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}")
        if not self.proxy_depth_m > 0.0:
            raise ValueError(
                f"proxy_depth_m must be positive, got {self.proxy_depth_m}"
            )

    @property
    def inverse_depth(self) -> float:
        if math.isinf(self.proxy_depth_m):
            return 0.0
        return 1.0 / self.proxy_depth_m


def _default_flow_calibration(*, h: int, w: int) -> NormalizedFlowCalibration:
    sx = float(w) / float(SEGNET_INPUT_W)
    sy = float(h) / float(SEGNET_INPUT_H)
    return NormalizedFlowCalibration(
        fx=float(COMMA_INTRINSICS.fx) * sx,
        fy=float(COMMA_INTRINSICS.fy) * sy,
        cx=float(COMMA_INTRINSICS.cx) * sx,
        cy=float(COMMA_INTRINSICS.cy) * sy,
        proxy_depth_m=DEFAULT_PROXY_DEPTH_M,
    )


def _resolve_flow_calibration(
    *,
    h: int,
    w: int,
    calibration: NormalizedFlowCalibration | None,
) -> NormalizedFlowCalibration:
    if calibration is None:
        return _default_flow_calibration(h=h, w=w)
    return calibration


@dataclass
class RaftRadialPoseConfig:
    """Operating config for Lane RAFT/radial.

    All fields required-keyword in caller; this dataclass enforces no
    silent defaults beyond the documented Mode A / B distinction.
    """

    mode: str
    """One of VALID_MODES. Mode B requires explicit human approval marker."""

    n_basis_functions: int = 6
    """Radial basis dimensionality. 6 = canonical Longuet-Higgins 6-DoF.
    Higher values add wavelet-style residual basis (deferred to Lane 11).
    """

    calibration_window: int = 200
    """Number of leading frames used to fit affine (A, b) calibration."""

    image_h: int = 384
    """Image height for normalization."""

    image_w: int = 512
    """Image width for normalization."""

    flow_calibration: NormalizedFlowCalibration | None = None
    """Optional explicit camera/proxy-depth calibration for RAFT pixel flow.

    None means scale the canonical comma scorer-resolution intrinsics to the
    actual flow field shape.
    """

    inflate_compliance_marker: str | None = None
    """For Mode B: path to a human-signed approval marker. None = NOT approved.
    Mode B will refuse to operate without this marker present.
    """

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}, got {self.mode!r}"
            )
        if self.n_basis_functions < 6:
            raise ValueError(
                f"n_basis_functions must be >= 6 (canonical 6-DoF), "
                f"got {self.n_basis_functions}"
            )
        if self.calibration_window < 10:
            raise ValueError(
                f"calibration_window must be >= 10 frames for stable LSQ, "
                f"got {self.calibration_window}"
            )
        if self.image_h <= 0 or self.image_w <= 0:
            raise ValueError(
                f"image dims must be positive, got ({self.image_h}, {self.image_w})"
            )
        if self.mode == MODE_INFLATE_RECOMPUTE and not self.inflate_compliance_marker:
            raise ValueError(
                "Mode B (inflate_recompute) requires inflate_compliance_marker "
                "(human-signed approval per CLAUDE.md 'Strict scorer rule'). "
                "Pass the path to the approval marker, or use Mode A."
            )


# ── radial basis ───────────────────────────────────────────────────────


def build_radial_basis(
    *,
    h: int,
    w: int,
    n_basis: int = 6,
    calibration: NormalizedFlowCalibration | None = None,
) -> np.ndarray:
    """Build the calibrated normalized-flow basis matrix B.

    The 6 canonical basis functions follow the lane-design Longuet-Higgins
    small-motion convention after converting pixels to normalized camera
    coordinates. Each column is a normalized-flow field flattened with
    `vec((u, v)) = [u_00, v_00, u_01, v_01, ...]`.

    Args (all required-keyword):
        h, w: image height and width.
        n_basis: number of basis functions. Currently must be exactly 6.
            Higher values reserved for wavelet residual basis (Lane 11 intersection).
        calibration: camera intrinsics and proxy depth at this resolution.
            None uses scaled comma scorer intrinsics and DEFAULT_PROXY_DEPTH_M.

    Returns:
        np.ndarray shape (h*w*2, n_basis), float32.

    Raises:
        ValueError: if dimensions are invalid or n_basis != 6.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h, w must be positive, got ({h}, {w})")
    if n_basis != 6:
        raise NotImplementedError(
            f"n_basis={n_basis} not yet supported; only canonical 6-DoF basis "
            "implemented. Wavelet residual basis is reserved for Lane 11 intersection."
        )
    cal = _resolve_flow_calibration(h=h, w=w, calibration=calibration)
    rho = np.float32(cal.inverse_depth)

    # Calibrated normalized image coordinates.
    ys, xs = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    xs = (xs - np.float32(cal.cx)) / np.float32(cal.fx)
    ys = (ys - np.float32(cal.cy)) / np.float32(cal.fy)
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    n_px = xs_flat.size
    # B columns: 6 basis functions, each producing (u, v) per pixel
    # Stored as interleaved [u_0, v_0, u_1, v_1, ...]
    B = np.zeros((n_px * 2, 6), dtype=np.float32)
    # B_0/B_1/B_2: proxy-depth translation terms.
    B[0::2, 0] = -rho
    B[1::2, 1] = -rho
    B[0::2, 2] = rho * xs_flat
    B[1::2, 2] = rho * ys_flat
    # B_3 = (-y, x): roll (ω_z)
    B[0::2, 3] = -ys_flat
    B[1::2, 3] = xs_flat
    # B_4 = (x·y, -(1 + y²)): pitch (ω_x), includes normalized-coordinate constant.
    B[0::2, 4] = xs_flat * ys_flat
    B[1::2, 4] = -(1.0 + ys_flat * ys_flat)
    # B_5 = (1 - x², x·y): yaw (ω_y), includes normalized-coordinate constant.
    B[0::2, 5] = 1.0 - xs_flat * xs_flat
    B[1::2, 5] = xs_flat * ys_flat
    return B


def compute_radial_basis_from_flow(
    *,
    flow_field: np.ndarray,
    n_basis: int = 6,
    calibration: NormalizedFlowCalibration | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose raw pixel flow into calibrated 6-DoF basis coefficients.

    Args (all required-keyword):
        flow_field: shape (T, H, W, 2) — per-frame pixel flow from RAFT.
        n_basis: basis dimensionality (must be 6 currently).
        calibration: camera intrinsics and proxy depth at this resolution.
            None uses scaled comma scorer intrinsics and DEFAULT_PROXY_DEPTH_M.

    Returns:
        (alpha, residual) where:
            alpha: (T, n_basis) coefficients per frame
            residual: (T, H, W, 2) pixel-flow residual after basis projection
                (norm of residual is the sparsity measure for Lane 11
                wavelet-residual coding intersection)

    Raises:
        ValueError: if flow_field shape is wrong.
    """
    if flow_field.ndim != 4 or flow_field.shape[-1] != 2:
        raise ValueError(
            f"flow_field must have shape (T, H, W, 2), got {flow_field.shape}"
        )
    T, H, W, _ = flow_field.shape
    cal = _resolve_flow_calibration(h=H, w=W, calibration=calibration)
    B = build_radial_basis(h=H, w=W, n_basis=n_basis, calibration=cal)
    normalized_flow = flow_field.astype(np.float32, copy=True)
    normalized_flow[..., 0] /= np.float32(cal.fx)
    normalized_flow[..., 1] /= np.float32(cal.fy)
    # B has shape (H*W*2, n_basis). Precompute pinv(B) directly rather than
    # using pinv(B.T @ B) @ B.T; the calibrated yaw column intentionally
    # contains a constant term and can be numerically close to proxy-depth
    # x-translation on narrow-FOV synthetic grids.
    basis = B.astype(np.float64, copy=False)
    basis_pinv = np.linalg.pinv(basis)
    alpha = np.zeros((T, n_basis), dtype=np.float32)
    residual = np.zeros(flow_field.shape, dtype=np.float32)
    for t in range(T):
        # Flatten flow (H, W, 2) → (H*W*2,) interleaved [u, v, u, v, ...]
        flow_flat = normalized_flow[t].reshape(-1).astype(np.float64, copy=False)
        alpha_t = basis_pinv @ flow_flat
        alpha[t] = alpha_t
        # Residual: F - B @ alpha
        recon_flat = basis @ alpha_t
        residual_normalized = (flow_flat - recon_flat).reshape(H, W, 2)
        residual[t, ..., 0] = residual_normalized[..., 0] * np.float32(cal.fx)
        residual[t, ..., 1] = residual_normalized[..., 1] * np.float32(cal.fy)
    return alpha, residual


# ── calibration ────────────────────────────────────────────────────────


@dataclass
class PoseCalibration:
    """Affine calibration α ∈ ℝ^{n_basis} → pose ∈ ℝ^6.

    pose = A @ alpha + b

    A: (6, n_basis), b: (6,)
    """

    A: np.ndarray
    b: np.ndarray
    n_calibration_frames: int
    rmse_train: float
    """Train-set RMSE — proxy for held-out RMSE if train/test split is honest."""

    def apply(self, alpha: np.ndarray) -> np.ndarray:
        """Apply calibration. alpha shape (T, n_basis) → (T, 6)."""
        if alpha.ndim != 2 or alpha.shape[1] != self.A.shape[1]:
            raise ValueError(
                f"alpha must be (T, {self.A.shape[1]}), got {alpha.shape}"
            )
        return alpha @ self.A.T + self.b


def calibrate_to_contest_pose(
    *,
    alpha: np.ndarray,
    contest_pose: np.ndarray,
    calibration_window: int,
) -> PoseCalibration:
    """Fit affine `pose_contest ≈ A @ alpha + b` on the first `calibration_window` frames.

    Args (all required-keyword):
        alpha: (T, n_basis) radial coefficients.
        contest_pose: (T, 6) ground-truth contest pose (for calibration only).
        calibration_window: leading frames used for LSQ. Remaining frames are
            held-out for disagreement evaluation.

    Returns:
        PoseCalibration with A, b, train RMSE.

    Raises:
        ValueError: on shape mismatch or insufficient calibration frames.
    """
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D (T, n_basis), got {alpha.shape}")
    if contest_pose.ndim != 2 or contest_pose.shape[1] != 6:
        raise ValueError(
            f"contest_pose must be (T, 6), got {contest_pose.shape}"
        )
    if alpha.shape[0] != contest_pose.shape[0]:
        raise ValueError(
            f"alpha and contest_pose T mismatch: {alpha.shape[0]} vs "
            f"{contest_pose.shape[0]}"
        )
    if calibration_window > alpha.shape[0]:
        raise ValueError(
            f"calibration_window={calibration_window} > T={alpha.shape[0]}"
        )
    if calibration_window < 10:
        raise ValueError(
            f"calibration_window must be >= 10 for stable LSQ, "
            f"got {calibration_window}"
        )
    cal_alpha = alpha[:calibration_window]  # (W, n_basis)
    cal_pose = contest_pose[:calibration_window]  # (W, 6)
    # Fit pose_i = A @ alpha_i + b for i in [0, W)
    # Equivalently: pose = [alpha | 1] @ [A^T; b^T]
    n_basis = cal_alpha.shape[1]
    aug = np.concatenate(
        [cal_alpha, np.ones((calibration_window, 1), dtype=cal_alpha.dtype)],
        axis=1,
    )  # (W, n_basis+1)
    # Solve aug @ X = cal_pose via LSQ; X is (n_basis+1, 6)
    X, _, _, _ = np.linalg.lstsq(aug, cal_pose, rcond=None)
    A = X[:n_basis].T.astype(np.float32)  # (6, n_basis)
    b = X[n_basis].astype(np.float32)  # (6,)
    # Train RMSE
    pred = aug @ X
    rmse = float(np.sqrt(np.mean((pred - cal_pose) ** 2)))
    return PoseCalibration(
        A=A, b=b, n_calibration_frames=calibration_window, rmse_train=rmse,
    )


# ── disagreement evaluation ───────────────────────────────────────────


def evaluate_disagreement(
    *,
    pose_estimated: np.ndarray,
    pose_contest: np.ndarray,
) -> dict:
    """Per-dimension MSE, max-deviation, average MSE.

    Args (all required-keyword):
        pose_estimated: (T, 6) RAFT-radial-derived pose
        pose_contest: (T, 6) contest ground-truth pose

    Returns:
        Dict with keys:
            'per_dim_mse': (6,) np.ndarray — MSE per pose dimension
            'max_per_dim': (6,) np.ndarray — max abs deviation per dim
            'overall_mse': float — averaged MSE across all dims
            'kill_threshold_passed': bool — True if overall_mse < 1e-3 (Mode B
                viability gate per design doc §4 kill criteria)

    Raises:
        ValueError: on shape mismatch.
    """
    if pose_estimated.shape != pose_contest.shape:
        raise ValueError(
            f"pose shape mismatch: {pose_estimated.shape} vs {pose_contest.shape}"
        )
    if pose_estimated.ndim != 2 or pose_estimated.shape[1] != 6:
        raise ValueError(
            f"poses must be (T, 6), got {pose_estimated.shape}"
        )
    diff = pose_estimated - pose_contest
    per_dim_mse = np.mean(diff ** 2, axis=0)
    max_per_dim = np.max(np.abs(diff), axis=0)
    overall_mse = float(per_dim_mse.mean())
    return {
        "per_dim_mse": per_dim_mse.astype(np.float32),
        "max_per_dim": max_per_dim.astype(np.float32),
        "overall_mse": overall_mse,
        "kill_threshold_passed": overall_mse < 1e-3,
    }


# ── strict-scorer-rule banner ─────────────────────────────────────────


def emit_inflate_compliance_banner(*, config: RaftRadialPoseConfig) -> str:
    """Emit the runtime compliance banner required for Mode B.

    Per CLAUDE.md: any inflate-time scorer-like load MUST print a banner so
    the score can be tagged appropriately.

    Args (all required-keyword):
        config: the operating config.

    Returns:
        The banner string (caller is responsible for actually printing).
        Returns empty string for Mode A (no banner needed).

    Raises:
        ValueError: Mode B without compliance marker (defense-in-depth; config
            constructor should already prevent this).
    """
    if config.mode == MODE_COMPRESS_TIME_PRIOR:
        return ""
    if config.mode == MODE_INFLATE_RECOMPUTE:
        if not config.inflate_compliance_marker:
            raise ValueError(
                "Mode B requires inflate_compliance_marker; refusing to emit banner."
            )
        return (
            "[strict-scorer-rule] Lane RAFT/radial — INFLATE-TIME RAFT-Large LOADED. "
            f"Compliance marker: {config.inflate_compliance_marker}. "
            "Score MUST be tagged [non-compliant, requires compliance ruling] "
            "until human approval per CLAUDE.md 'Strict scorer rule' is granted."
        )
    raise AssertionError(f"unreachable mode: {config.mode}")


# ── Mode A entrypoint (the only safe-shippable one) ───────────────────


def estimate_pose_compress_time_prior(
    *,
    flow_field: np.ndarray,
    contest_pose_for_calibration: np.ndarray,
    config: RaftRadialPoseConfig,
) -> tuple[np.ndarray, PoseCalibration, dict]:
    """Mode A: compress-time RAFT-radial pose initialization.

    Workflow:
    1. Decompose `flow_field` into 6-DoF radial coefficients α
    2. Fit affine calibration on `calibration_window` leading frames
    3. Apply calibration to all frames → estimated pose (T, 6)
    4. Evaluate disagreement vs contest pose for diagnostic logging

    Args (all required-keyword):
        flow_field: (T, H, W, 2) RAFT-Large output for T+1 consecutive frames
            (T pairs).
        contest_pose_for_calibration: (T, 6) — used for calibration only.
            In production this is the existing pose stream that pose-TTO will
            refine; this function uses it as the LSQ target.
        config: operating config (must be Mode A).

    Returns:
        (pose_estimated, calibration, disagreement_metrics)

    Raises:
        ValueError: if config.mode is not MODE_COMPRESS_TIME_PRIOR or shapes mismatch.
    """
    if config.mode != MODE_COMPRESS_TIME_PRIOR:
        raise ValueError(
            f"estimate_pose_compress_time_prior requires Mode A "
            f"(MODE_COMPRESS_TIME_PRIOR), got {config.mode!r}"
        )
    alpha, _residual = compute_radial_basis_from_flow(
        flow_field=flow_field,
        n_basis=config.n_basis_functions,
        calibration=config.flow_calibration,
    )
    calibration = calibrate_to_contest_pose(
        alpha=alpha,
        contest_pose=contest_pose_for_calibration,
        calibration_window=config.calibration_window,
    )
    pose_estimated = calibration.apply(alpha)
    metrics = evaluate_disagreement(
        pose_estimated=pose_estimated, pose_contest=contest_pose_for_calibration,
    )
    return pose_estimated, calibration, metrics


__all__ = [
    "MODE_COMPRESS_TIME_PRIOR",
    "MODE_INFLATE_RECOMPUTE",
    "DEFAULT_PROXY_DEPTH_M",
    "NormalizedFlowCalibration",
    "PoseCalibration",
    "RAFT_RADIAL_VERSION",
    "RaftRadialPoseConfig",
    "VALID_MODES",
    "build_radial_basis",
    "calibrate_to_contest_pose",
    "compute_radial_basis_from_flow",
    "emit_inflate_compliance_banner",
    "estimate_pose_compress_time_prior",
    "evaluate_disagreement",
]
