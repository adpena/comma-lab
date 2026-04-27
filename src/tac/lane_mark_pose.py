"""Zero-archive-cost 6-DOF pose conditioning derived from lane-mark masks.

Lane M+: convert SegNet lane-marking displacement (already in masks.mkv) to a
``(num_pairs, 6)`` PoseNet-convention pose tensor at INFLATE TIME, eliminating
the need to ship ``optimized_poses.pt`` (~7-15 KB) in the archive.

Council provenance
------------------
* memory ``project_lane_marking_speed_estimation``: lane-mark centroids encode
  ego-motion radial displacement; usable at inflate from masks alone.
* memory ``project_openpilot_lane_forcing``: openpilot lane models can warm-
  start at compress time; the inflate-time path uses pure mask-derived
  centroids (no openpilot dependency at inflate).
* memory ``project_yousfi_geometric_analysis``: PoseNet output dim 0 = speed
  on a learned scale (mean=31.26, std=1.26, range [23.5, 35.1]); dims 1-5 are
  small near-zero values (std~0.05) that contribute <=0.18 score even if
  predicted as the per-clip mean.
* memory ``project_posenet_rank1_discovery``: PoseNet Jacobian effective rank
  is 1.008 — only one DOF (dim 0 = scalar radial zoom from FoE) carries signal.

The physical math
-----------------
PoseNet dim 0 is on a learned scale (NOT m/s). Empirically, dim 0 statistics
on the 1200-frame contest clip are::

    mean=31.295  std=1.265  range=[23.41, 35.17]

The lane-mark log-zoom signal :math:`s_k` (from
:func:`tac.lane_mark_speed.zoom_from_masks`) is a per-pair float in
approximately :math:`[-0.5, +0.5]` for highway driving, with positive values
meaning lanes drift AWAY from the FoE (= forward motion = higher speed).

The inverse map ``log_zoom -> pose_dim0`` is calibrated ONCE against a
reference distribution and saved as the package constants below. At inflate
time we apply the affine map::

    pose_dim0[k] = POSENET_DIM0_MEAN + POSENET_DIM0_PER_LOGZOOM * log_zoom[k]

so the per-clip mean lands at the empirical mean (31.295) and the per-pair
deviation tracks the lane-mark signal. Dims 1-5 are set to zero — per
Yousfi-Fridrich, predicting their per-clip mean (which is very small, std~0.05)
costs at most 0.18 distortion points.

This is the "Fridrich strategy" in code: match dim 0 exactly, leave the other
five at the conservative zero-prior. The renderer's FiLM conditioning only
needs to discriminate forward-motion regimes, not pose noise.

Usage at inflate
----------------
    from tac.lane_mark_pose import compute_zero_cost_poses_from_masks

    poses = compute_zero_cost_poses_from_masks(masks)  # (num_pairs, 6)
    # Pass to renderer just like a loaded optimized_poses.pt tensor
    pairs = renderer(masks_t, masks_t1, pose=poses)

Sentinel
--------
The companion build script :mod:`experiments.build_baseline_archive` writes a
sentinel file ``zero_cost_poses_v1`` into the archive when
``--use-zero-cost-poses`` is set. The inflate side detects this marker AND a
missing ``optimized_poses.pt``/``poses.pt`` to switch to the analytical path.
The marker is a 0-byte sentinel so it costs ~30 bytes in the zip header
(net win is still >7 KB).
"""
from __future__ import annotations

import torch

from tac.camera import VANISHING_POINT
from tac.lane_mark_speed import (
    DEFAULT_FOE_H,
    DEFAULT_FOE_W,
    LANE_MARK_CLASS,
    zoom_from_masks,
)

__all__ = [
    "POSENET_DIM0_MEAN",
    "POSENET_DIM0_STD",
    "POSENET_DIM0_PER_LOGZOOM",
    "ZERO_COST_POSES_SENTINEL",
    "compute_zero_cost_poses_from_masks",
]

# Sentinel filename written into the archive when poses are computed at
# inflate from masks. The presence of this 0-byte file is the inflate-side
# signal to call compute_zero_cost_poses_from_masks() instead of failing
# when no optimized_poses.pt is found.
ZERO_COST_POSES_SENTINEL: str = "zero_cost_poses_v1"

# PoseNet dim 0 (speed) statistics empirically measured against the shipped
# baseline_dilated_h64_0_90/optimized_poses.pt — the reference distribution
# the renderer was conditioned on during training. See module docstring.
POSENET_DIM0_MEAN: float = 31.295
POSENET_DIM0_STD: float = 1.265

# Calibrated coupling between the lane-mark log-zoom signal and PoseNet dim 0.
# Empirically the lane-mark log-zoom std on the contest clip is ~0.144 (real
# masks, smoothing=0.3); we want the resulting pose_dim0 std to land roughly
# at POSENET_DIM0_STD = 1.265, so the coupling is 1.265 / 0.144 ≈ 8.78.
# We use the conservative value 8.0 (slightly under-amplifying) so that an
# anomalous spike in the lane-mark signal cannot push pose_dim0 outside the
# empirical [23.4, 35.2] range — that would put the renderer's FiLM
# conditioning into a regime it never saw during training.
POSENET_DIM0_PER_LOGZOOM: float = 8.0


def compute_zero_cost_poses_from_masks(
    masks: torch.Tensor,
    *,
    foe_w: float = DEFAULT_FOE_W,
    foe_h: float = DEFAULT_FOE_H,
    smoothing: float = 0.3,
    posenet_dim0_mean: float = POSENET_DIM0_MEAN,
    posenet_dim0_per_logzoom: float = POSENET_DIM0_PER_LOGZOOM,
    posenet_dim0_min: float = 23.0,
    posenet_dim0_max: float = 36.0,
    lane_mark_class: int = LANE_MARK_CLASS,
) -> torch.Tensor:
    """Compute per-pair 6-DOF pose tensor from lane-mark displacement.

    The output is a ``(num_pairs, 6)`` float tensor matching the convention
    of PoseNet's first 6 output dims (the scored ones). Per the rank-1
    discovery (memory ``project_posenet_rank1_discovery``), dim 0 carries
    99.8% of variance and corresponds geometrically to scalar radial zoom
    from the Focus of Expansion. The remaining 5 dims are set to zero — the
    Fridrich strategy of matching dim 0 exactly and accepting the <=0.18
    distortion ceiling on the other dims.

    The mapping from lane-mark signal to pose dim 0 is::

        log_zoom[k] = zoom_from_masks(masks)[k]   # ~ [-0.5, +0.5]
        pose_dim0[k] = clamp(
            POSENET_DIM0_MEAN + POSENET_DIM0_PER_LOGZOOM * log_zoom[k],
            posenet_dim0_min,
            posenet_dim0_max,
        )

    The clamp prevents an anomalous lane-mark spike (e.g. lane-change burst
    or mask-decode artifact) from pushing the FiLM conditioning into a
    regime the renderer never saw during training.

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` integer mask tensor with class indices in
        ``[0, NUM_CLASSES)``. Lane markings are ``lane_mark_class`` (= 1
        per ``DEPTH_PRIORS_METERS`` and ``CLASS_MEAN_COLORS``).
        ``N`` MUST be even (paired frames); ``num_pairs = N // 2``.
    foe_w, foe_h : float
        Focus-of-Expansion coordinates IN MASK PIXELS. Defaults are the
        scorer-resolution (256, 174) values; if ``masks`` is at a different
        resolution the centroid distances are still meaningful (the affine
        log-zoom estimate is dimensionless), but accuracy is best at the
        native scorer resolution.
    smoothing : float
        Local-median blend weight (0..1) passed to
        :func:`tac.lane_mark_speed.zoom_from_masks` to suppress lane-change
        spikes.
    posenet_dim0_mean : float
        The per-clip mean of PoseNet dim 0 (default ``POSENET_DIM0_MEAN``,
        empirically 31.295 from the baseline ``optimized_poses.pt``).
    posenet_dim0_per_logzoom : float
        Slope of the affine map log_zoom -> pose_dim0 deviation. Default
        ``POSENET_DIM0_PER_LOGZOOM`` is calibrated to land the resulting
        pose_dim0 std near the empirical 1.265.
    posenet_dim0_min, posenet_dim0_max : float
        Output clamp bounds for pose dim 0. Defaults [23.0, 36.0] cover the
        empirical [23.41, 35.17] range with a small margin.
    lane_mark_class : int
        SegNet class index for lane markings. Default 1, verified against
        ``tac.camera.DEPTH_PRIORS_METERS`` (class 1 = "lane markings — same
        plane as road") and ``CLASS_MEAN_COLORS[1]`` ("light gray").

    Returns
    -------
    torch.Tensor
        ``(num_pairs, 6)`` float32 tensor. Column 0 carries the lane-mark-
        derived pose-dim-0 estimate in PoseNet units; columns 1-5 are zero.

    Raises
    ------
    ValueError
        If ``masks.ndim != 3`` or the frame count is odd.

    Notes
    -----
    * Returns zeros (with dim 0 = posenet_dim0_mean) for pairs where neither
      frame contains any lane-marking pixel — the only safe default when
      there is no signal. ``zoom_from_masks`` already does this internally.
    * Honors a non-default ``lane_mark_class`` by passing it through to a
      copy-mode call — the underlying ``zoom_from_masks`` reads ``LANE_MARK_
      CLASS`` directly, so we forward via a temporary mask remap if needed.
    """
    if masks.ndim != 3:
        raise ValueError(
            f"compute_zero_cost_poses_from_masks expected (N, H, W) masks, "
            f"got shape {tuple(masks.shape)}"
        )
    n_frames = masks.shape[0]
    if n_frames % 2 != 0:
        raise ValueError(
            f"compute_zero_cost_poses_from_masks needs an even frame count "
            f"(paired frames), got {n_frames}"
        )
    num_pairs = n_frames // 2

    # Forward the lane-mark class through a temporary remap when it differs
    # from the underlying module's hard-coded LANE_MARK_CLASS. The remap is
    # cheap (an int comparison + where) and avoids importing internals.
    if lane_mark_class == LANE_MARK_CLASS:
        masks_for_zoom = masks
    else:
        # Swap the operator-supplied class index into LANE_MARK_CLASS for
        # the duration of zoom_from_masks. Other class indices are
        # preserved so non-lane regions are still distinguishable in case
        # zoom_from_masks ever depends on them (it does not today).
        is_target = masks == lane_mark_class
        is_existing_lane = masks == LANE_MARK_CLASS
        masks_for_zoom = masks.clone()
        masks_for_zoom[is_existing_lane] = lane_mark_class
        masks_for_zoom[is_target] = LANE_MARK_CLASS

    log_zoom = zoom_from_masks(
        masks_for_zoom,
        foe_w=foe_w,
        foe_h=foe_h,
        smoothing=smoothing,
    )  # (num_pairs,) float32 in approximately [-0.5, +0.5]

    # Affine map to PoseNet dim 0 scale, then clamp to empirical envelope
    # so an outlier cannot put the FiLM conditioning out-of-distribution.
    pose_dim0 = posenet_dim0_mean + posenet_dim0_per_logzoom * log_zoom
    pose_dim0 = pose_dim0.clamp(min=posenet_dim0_min, max=posenet_dim0_max)

    poses = torch.zeros(num_pairs, 6, dtype=torch.float32)
    poses[:, 0] = pose_dim0.to(torch.float32)
    return poses


def _summarize_pose_tensor(poses: torch.Tensor) -> dict[str, float]:
    """Diagnostic summary used by tests + provenance logs.

    Parameters
    ----------
    poses : torch.Tensor
        ``(P, 6)`` pose tensor.

    Returns
    -------
    dict
        Per-dim mean/std/min/max for cheap regression assertions.
    """
    out: dict[str, float] = {}
    for d in range(poses.shape[1]):
        col = poses[:, d]
        out[f"dim{d}_mean"] = float(col.mean().item())
        out[f"dim{d}_std"] = float(col.std().item())
        out[f"dim{d}_min"] = float(col.min().item())
        out[f"dim{d}_max"] = float(col.max().item())
    return out


# Sanity binding: VANISHING_POINT and (DEFAULT_FOE_W, DEFAULT_FOE_H) must
# stay in lock-step. If they ever drift, the inflate-time pose math diverges
# from training-time zoom flow. This is a load-bearing invariant; we encode
# it as a module-import-time assertion so any future edit to camera.py that
# moves VANISHING_POINT trips immediately rather than producing silently-
# wrong inflate poses.
assert (DEFAULT_FOE_W, DEFAULT_FOE_H) == (
    float(VANISHING_POINT[0]),
    float(VANISHING_POINT[1]),
), (
    "tac.lane_mark_speed FoE constants drifted from tac.camera.VANISHING_POINT. "
    "Both are load-bearing for inflate-time zero-cost pose computation."
)
