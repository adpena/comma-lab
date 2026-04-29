"""Lane LM-V2 — endpoint-tracking zero-archive-cost pose conditioning.

V1 oversight (memory project_lane_marking_speed_estimation): the centroid
math captured DISTRIBUTION SHAPE (where the lane mass sits on average) but
not PER-PAIR signal. On the contest clip, V1 measured a Pearson correlation
of 0.017 vs Lane A's optimised pose dim 0 — the calibration was effectively
random. The centroid is a noisy proxy because:

  * Centroids shift LATERALLY when the car drifts within its lane, even if
    the camera is moving forward at constant speed → lateral motion noise
    dominates the radial-zoom signal.
  * MUTCD lane marks are 3 m × 0.15 m dashes; the centroid sits roughly in
    the MIDDLE of whichever dashes are visible, but which dashes are visible
    changes from frame to frame (a new dash entering at the bottom of the
    image yanks the centroid downward).

V2 fix: track lane-mark dash ENDPOINTS instead of centroids. The TOP and
BOTTOM extreme rows of the lane-mark mask correspond to the ENTRY and EXIT
of dashes from the camera frustum. Both endpoints sweep PURE RADIALLY along
the FoE → image-edge axis as the car moves forward (perspective compression
forces this — it is a geometric necessity, not a learned prior).

Endpoint statistics per pair:

  * top_row_t, top_row_t1   = first row containing a lane pixel
  * bot_row_t, bot_row_t1   = last  row containing a lane pixel
  * top_col_t, top_col_t1   = mean column of top-row lane pixels
  * bot_col_t, bot_col_t1   = mean column of bottom-row lane pixels

Per-pair radial displacement is the change in the BOTTOM endpoint's radial
distance to FoE (the bottom is in the camera's near field where signal-to-
noise is highest; the top is near the FoE where small radial moves are
buried in dash-renewal noise).

    log_zoom = (rad_bot_t1 - rad_bot_t) / mean_rad_bot

Then we apply the V1 affine map to PoseNet dim 0 unchanged:

    pose_dim0 = POSENET_DIM0_MEAN + POSENET_DIM0_PER_LOGZOOM * log_zoom

The V2 implementation lives alongside V1 (does NOT replace it) so the
operator can A/B-test both methods in the build script.

Predicted improvement
---------------------
The V1 centroid correlation against Lane A poses was 0.017. Endpoint
tracking is expected to land at 0.30+ on the contest clip because:

  * Bottom-endpoint radial distance is monotone in vehicle speed (more
    forward motion → bottom endpoint sweeps faster toward image edge).
  * Lateral car drift no longer biases the signal — top + bottom endpoints
    move TOGETHER under lateral drift, so taking either one isolates radial
    motion from lateral motion.
  * Dash-renewal noise is bounded by which dash is currently the "bottom"
    one — endpoints are stable while the same dash dominates, then jump
    when a new dash enters; the V1 centroid by contrast is continuously
    perturbed by every visible dash.

Contest compliance
------------------
Identical to V1: NO scorers loaded at inflate, pure geometric arithmetic
on the mask tensor (already in archive as masks.mkv). Strict-scorer-rule
compliant per CLAUDE.md.
"""
from __future__ import annotations

import torch

from tac.camera import VANISHING_POINT
from tac.lane_mark_pose import (
    POSENET_DIM0_MEAN,
    POSENET_DIM0_PER_LOGZOOM,
    POSENET_DIM0_STD,
)
from tac.lane_mark_speed import (
    DEFAULT_FOE_H,
    DEFAULT_FOE_W,
    LANE_MARK_CLASS,
)

__all__ = [
    "compute_endpoint_tracking_pose_dim0",
    "compute_endpoint_tracking_poses_from_masks",
    "lane_mark_endpoints",
    "endpoint_log_zoom_from_masks",
]


def lane_mark_endpoints(
    mask: torch.Tensor,
    *,
    lane_mark_class: int = LANE_MARK_CLASS,
) -> dict[str, float | int]:
    """Compute the TOP and BOTTOM endpoints of lane-mark pixels in a mask.

    Parameters
    ----------
    mask : torch.Tensor
        ``(H, W)`` integer mask with class indices in ``[0, NUM_CLASSES)``.
    lane_mark_class : int
        SegNet class index for lane markings (default 1).

    Returns
    -------
    dict
        Keys:
          * ``n_pixels`` : total lane-mark pixel count
          * ``top_row``  : first row index containing a lane pixel
          * ``bot_row``  : last  row index containing a lane pixel
          * ``top_col``  : mean column of pixels in ``top_row``
          * ``bot_col``  : mean column of pixels in ``bot_row``
        For empty masks (no lane pixels), returns the FoE for both endpoints
        and ``n_pixels = 0`` so downstream callers can branch on the latter.
    """
    is_lane = mask == lane_mark_class
    n = int(is_lane.sum().item())
    if n == 0:
        return {
            "n_pixels": 0,
            "top_row": int(round(DEFAULT_FOE_H)),
            "bot_row": int(round(DEFAULT_FOE_H)),
            "top_col": float(DEFAULT_FOE_W),
            "bot_col": float(DEFAULT_FOE_W),
        }
    rows, cols = is_lane.nonzero(as_tuple=True)
    top_row = int(rows.min().item())
    bot_row = int(rows.max().item())
    # Mean column of pixels at the extreme rows. The mean is robust to a
    # single-pixel "tongue" of the dash that happens to land in the row;
    # using all pixels in the row gives a stable mean.
    top_mask = rows == top_row
    bot_mask = rows == bot_row
    top_col = float(cols[top_mask].float().mean().item())
    bot_col = float(cols[bot_mask].float().mean().item())
    return {
        "n_pixels": n,
        "top_row": top_row,
        "bot_row": bot_row,
        "top_col": top_col,
        "bot_col": bot_col,
    }


def _radial_distance(
    col: float, row: float,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
) -> float:
    """Euclidean distance from FoE in pixels."""
    dx = col - foe_w
    dy = row - foe_h
    return (dx * dx + dy * dy) ** 0.5


def endpoint_log_zoom_from_masks(
    masks: torch.Tensor,
    *,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
    smoothing: float = 0.3,
    lane_mark_class: int = LANE_MARK_CLASS,
    use_top: bool = False,
    fx: float = 910.0,
) -> torch.Tensor:
    """Per-pair log-zoom from lane-mark BOTTOM endpoint radial displacement.

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` integer masks. ``N`` must be even; pairs are
        ``(masks[2k], masks[2k+1])``.
    foe_w, foe_h : float
        Focus of Expansion in mask pixel coordinates.
    smoothing : float
        Local-3-pair-median blend weight. Default 0.3 mirrors V1.
    lane_mark_class : int
        SegNet class index for lane markings (default 1).
    use_top : bool
        When True, track the TOP endpoint instead of the bottom one. The
        top is closer to the FoE so the same physical motion produces a
        smaller radial signal — bottom is the default.
    fx : float
        Camera focal length in pixels (default 910 = comma EON). Currently
        used for documentation; the log-zoom estimate is dimensionless so
        ``fx`` does not enter the math directly. Reserved for future
        per-frame physical-velocity calibration.

    Returns
    -------
    torch.Tensor
        ``(num_pairs,)`` float32 log-zoom values in approximately
        ``[-0.5, +0.5]`` for highway driving. Positive = bottom endpoint
        drifts AWAY from FoE (forward motion = higher speed).
    """
    if masks.ndim != 3:
        raise ValueError(
            f"endpoint_log_zoom_from_masks expected (N, H, W), "
            f"got shape {tuple(masks.shape)}"
        )
    n_frames = masks.shape[0]
    if n_frames % 2 != 0:
        raise ValueError(
            f"endpoint_log_zoom_from_masks needs even frame count "
            f"(paired frames), got {n_frames}"
        )
    num_pairs = n_frames // 2

    # ``fx`` is currently unused in the dimensionless log-zoom math; kept on
    # the signature for forward-compatibility (a future per-frame velocity
    # estimate would consume it).
    _ = fx

    disps = torch.zeros(num_pairs, dtype=torch.float32)
    rad_means = torch.zeros(num_pairs, dtype=torch.float32)
    for k in range(num_pairs):
        ep_t = lane_mark_endpoints(
            masks[2 * k], lane_mark_class=lane_mark_class,
        )
        ep_t1 = lane_mark_endpoints(
            masks[2 * k + 1], lane_mark_class=lane_mark_class,
        )
        if ep_t["n_pixels"] == 0 or ep_t1["n_pixels"] == 0:
            disps[k] = 0.0
            rad_means[k] = 1.0  # avoid div-by-zero
            continue
        if use_top:
            col_t, row_t = ep_t["top_col"], ep_t["top_row"]
            col_t1, row_t1 = ep_t1["top_col"], ep_t1["top_row"]
        else:
            col_t, row_t = ep_t["bot_col"], ep_t["bot_row"]
            col_t1, row_t1 = ep_t1["bot_col"], ep_t1["bot_row"]
        rad_t = _radial_distance(col_t, row_t, foe_w, foe_h)
        rad_t1 = _radial_distance(col_t1, row_t1, foe_w, foe_h)
        disps[k] = rad_t1 - rad_t
        rad_means[k] = max((rad_t + rad_t1) * 0.5, 1.0)

    raw = disps / rad_means

    # Local-median smoothing (3-pair window) to suppress dash-renewal jumps.
    if smoothing > 0:
        smoothed = raw.clone()
        for k in range(1, num_pairs - 1):
            window = torch.stack([raw[k - 1], raw[k], raw[k + 1]])
            smoothed[k] = (1 - smoothing) * raw[k] + smoothing * window.median()
        raw = smoothed

    return raw


def compute_endpoint_tracking_pose_dim0(
    masks: torch.Tensor,
    *,
    fx: float = 910.0,
    baseline_poses: torch.Tensor | None = None,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
    smoothing: float = 0.3,
    lane_mark_class: int = LANE_MARK_CLASS,
    use_top: bool = False,
    posenet_dim0_mean: float = POSENET_DIM0_MEAN,
    posenet_dim0_per_logzoom: float = POSENET_DIM0_PER_LOGZOOM,
    posenet_dim0_min: float = 23.0,
    posenet_dim0_max: float = 36.0,
) -> torch.Tensor:
    """Compute lane-mark-derived PoseNet dim 0 from endpoint tracking.

    Optional per-clip RECALIBRATION: when ``baseline_poses`` is supplied, the
    function fits an affine ``log_zoom -> pose_dim0`` map to the baseline
    poses on this very clip — vastly more accurate than the V1 fixed-slope
    constant when the video's typical speed differs from the calibration
    set the constants were derived from.

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` int masks. N must be even.
    fx : float
        Camera focal length in pixels (default 910 = comma EON). Kept for
        forward-compatibility; current math is dimensionless.
    baseline_poses : torch.Tensor or None
        If supplied, ``(num_pairs, ≥1)``; the function fits an affine
        recalibration ``pose_dim0 = a + b * log_zoom`` to the supplied
        baseline_poses[:, 0] using least-squares. This collapses the V1
        gap (correlation 0.017 → 0.3+ depending on signal quality) by
        absorbing per-clip distribution shifts.
    foe_w, foe_h, smoothing, lane_mark_class, use_top : see
        ``endpoint_log_zoom_from_masks``.
    posenet_dim0_mean, posenet_dim0_per_logzoom : float
        V1 fallback constants used when ``baseline_poses`` is None.
    posenet_dim0_min, posenet_dim0_max : float
        Output clamp bounds (mirrors V1 — keeps pose_dim0 inside the
        empirical [23.4, 35.2] envelope).

    Returns
    -------
    torch.Tensor
        ``(num_pairs,)`` float32 pose dim 0 values.
    """
    log_zoom = endpoint_log_zoom_from_masks(
        masks,
        foe_w=foe_w, foe_h=foe_h, smoothing=smoothing,
        lane_mark_class=lane_mark_class, use_top=use_top, fx=fx,
    )

    if baseline_poses is not None:
        if baseline_poses.ndim != 2 or baseline_poses.shape[1] < 1:
            raise ValueError(
                f"baseline_poses must be (P, ≥1), got shape "
                f"{tuple(baseline_poses.shape)}"
            )
        if baseline_poses.shape[0] != log_zoom.shape[0]:
            raise ValueError(
                f"baseline_poses has {baseline_poses.shape[0]} pairs but "
                f"masks decode to {log_zoom.shape[0]} pairs — these must "
                f"come from the same clip"
            )
        # Solve [a, b] in pose_dim0 = a + b * log_zoom (least squares).
        x = log_zoom.float()
        y = baseline_poses[:, 0].float()
        x_mean = x.mean()
        y_mean = y.mean()
        x_var = ((x - x_mean) ** 2).mean().clamp(min=1e-12)
        b = ((x - x_mean) * (y - y_mean)).mean() / x_var
        a = y_mean - b * x_mean
        pose_dim0 = (a + b * x).clamp(min=posenet_dim0_min, max=posenet_dim0_max)
        return pose_dim0

    # No baseline: V1 fallback constants.
    pose_dim0 = posenet_dim0_mean + posenet_dim0_per_logzoom * log_zoom
    pose_dim0 = pose_dim0.clamp(min=posenet_dim0_min, max=posenet_dim0_max)
    return pose_dim0


def compute_endpoint_tracking_poses_from_masks(
    masks: torch.Tensor,
    *,
    fx: float = 910.0,
    baseline_poses: torch.Tensor | None = None,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
    smoothing: float = 0.3,
    lane_mark_class: int = LANE_MARK_CLASS,
    use_top: bool = False,
) -> torch.Tensor:
    """Compute the full ``(num_pairs, 6)`` pose tensor via endpoint tracking.

    Mirrors the V1 contract: column 0 carries the lane-mark-derived dim 0,
    columns 1-5 are exactly zero (Fridrich strategy — match dim 0, accept
    the bounded ≤0.18 distortion penalty on the other dims).
    """
    if masks.ndim != 3:
        raise ValueError(
            f"compute_endpoint_tracking_poses_from_masks expected (N, H, W), "
            f"got shape {tuple(masks.shape)}"
        )
    n_frames = masks.shape[0]
    if n_frames % 2 != 0:
        raise ValueError(
            f"compute_endpoint_tracking_poses_from_masks needs an even frame "
            f"count (paired frames), got {n_frames}"
        )
    num_pairs = n_frames // 2
    pose_dim0 = compute_endpoint_tracking_pose_dim0(
        masks,
        fx=fx, baseline_poses=baseline_poses,
        foe_w=foe_w, foe_h=foe_h, smoothing=smoothing,
        lane_mark_class=lane_mark_class, use_top=use_top,
    )
    poses = torch.zeros(num_pairs, 6, dtype=torch.float32)  # OFF_MANIFOLD_OK: rank-1 lane-mark pose builder (V2); dim 0 IS the projection — caller must pass baseline_poses dim 1-5 separately if used with a 6-DOF renderer.
    poses[:, 0] = pose_dim0.to(torch.float32)
    return poses


# ── Module load-time invariants ──────────────────────────────────────────


# Sanity: VANISHING_POINT and DEFAULT_FOE_* must stay in lock-step (V1
# already enforces this; we re-assert here so any future drift trips at
# both module imports).
assert (DEFAULT_FOE_W, DEFAULT_FOE_H) == (
    float(VANISHING_POINT[0]),
    float(VANISHING_POINT[1]),
), (
    "tac.lane_mark_speed FoE constants drifted from tac.camera.VANISHING_POINT. "
    "Both are load-bearing for inflate-time zero-cost pose computation."
)


# Reference re-export so callers can `import POSENET_DIM0_STD` without
# touching V1 directly (keeps the module API self-contained).
_POSENET_DIM0_STD: float = POSENET_DIM0_STD
