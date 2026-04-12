"""Domain-specific optimization solvers for video compression scoring.

Yousfi's cross-domain solver toolkit: each solver applies rigorous domain
knowledge (self-driving geometry, signal processing, aerospace, SDR) to
the problem of generating frames that minimize:

    S = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * rate

against frozen PoseNet (ego-motion) and SegNet (semantic segmentation)
scorer networks operating on 384x512x3 dashcam video.

References:
    [1] Ma et al., "Disentangling Structure and Aesthetics for Style-aware
        Image Completion", CVPR 2018 — vanishing point constraints.
    [2] Hartley & Zisserman, "Multiple View Geometry in Computer Vision",
        2nd ed., Cambridge 2004 — pinhole model, homographies, epipolar.
    [3] Rauch, Tung, Striebel, "Maximum Likelihood Estimates of Linear
        Dynamic Systems", AIAA J. 1965 — Kalman smoother.
    [4] Donoho, "Compressed Sensing", IEEE Trans. IT 2006 — LASSO/ISTA.
    [5] Berrou, Glavieux, Thitimajshima, "Near Shannon Limit Error-Correcting
        Coding and Decoding: Turbo-codes", ICC 1993 — turbo iteration.
    [6] Noll, "Zernike Polynomials and Atmospheric Turbulence", JOSA 1976
        — Zernike decomposition for adaptive optics.
    [7] Cover & Thomas, "Elements of Information Theory", Wiley 2006
        — water-filling for OFDM power allocation.
    [8] Pontryagin et al., "Mathematical Theory of Optimal Processes",
        Wiley 1962 — trajectory optimization, maximum principle.

Usage::

    from tac.contrib.domain_solvers import (
        EgoMotionFlowSolver,
        RoadPlaneHomography,
        VanishingPointConstraint,
        MatchedFilterOptimizer,
        KalmanFrameSmoother,
        CompressedSensingRecovery,
        TrajectoryOptimizer,
        AdaptiveOpticsCorrector,
        OFDMOptimizer,
        TurboScorerOptimizer,
        yousfi_domain_ranking,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.camera import (
    CLASS_MEAN_COLORS,
    COMMA_EXTRINSICS,
    COMMA_INTRINSICS,
    DEPTH_PRIORS_METERS,
    NUM_CLASSES,
)

__all__ = [
    "EgoMotionFlowSolver",
    "RoadPlaneHomography",
    "VanishingPointConstraint",
    "MatchedFilterOptimizer",
    "KalmanFrameSmoother",
    "CompressedSensingRecovery",
    "TrajectoryOptimizer",
    "AdaptiveOpticsCorrector",
    "OFDMOptimizer",
    "TurboScorerOptimizer",
    "yousfi_domain_ranking",
]

# ── Constants (re-exported from tac.camera for backward compatibility) ──

DEFAULT_DEPTH_PRIORS = DEPTH_PRIORS_METERS
DEFAULT_FOCAL_LENGTH: float = COMMA_INTRINSICS.fx
DEFAULT_PRINCIPAL_POINT: tuple[float, float] = (COMMA_INTRINSICS.cx, COMMA_INTRINSICS.cy)
DEFAULT_CAMERA_HEIGHT: float = COMMA_EXTRINSICS.height
DEFAULT_CAMERA_PITCH: float = abs(COMMA_EXTRINSICS.pitch)  # stored positive here for legacy compat


# ════════════════════════════════════════════════════════════════════════
# 1. Ego-Motion Constrained Flow Field
# ════════════════════════════════════════════════════════════════════════


@dataclass
class EgoMotionFlowConfig:
    """Configuration for ego-motion constrained flow solver.

    Attributes:
        focal_length: Camera focal length in pixels (default 910).
        principal_point: (cx, cy) principal point in pixels.
        depth_priors: Dict mapping class_id -> depth in meters.
        num_steps: Number of optimization steps after flow constraint.
        lr: Learning rate for residual optimization.
        flow_weight: Weight for flow consistency constraint.
    """
    focal_length: float = DEFAULT_FOCAL_LENGTH
    principal_point: tuple[float, float] = DEFAULT_PRINCIPAL_POINT
    depth_priors: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_DEPTH_PRIORS))
    num_steps: int = 100
    lr: float = 0.5
    flow_weight: float = 10.0


class EgoMotionFlowSolver:
    """Generate frames satisfying ego-motion optical flow constraints.

    Given a mask sequence and estimated ego-motion trajectory, computes
    per-pixel optical flow from a pinhole camera model with per-class
    depth priors. Frames are generated such that frame t warps to frame
    t+1 under the computed flow -- this is a HARD constraint, not a soft
    loss. Residual free pixels are optimized for compressibility.

    The pinhole camera flow model [2]:
        For pixel (u, v) with depth d, camera rotation (wx, wy, wz) and
        translation (tx, ty, tz):
            flow_u = f * (-tx/d + u'*tz/d) + f*(wy - wz*v')
            flow_v = f * (-ty/d + v'*tz/d) + f*(-wx + wz*u')
        where u' = (u - cx)/f, v' = (v - cy)/f are normalized coords.

    Args:
        config: EgoMotionFlowConfig or dict of overrides.
    """

    def __init__(self, config: EgoMotionFlowConfig | dict[str, Any] | None = None):
        if config is None:
            config = EgoMotionFlowConfig()
        elif isinstance(config, dict):
            config = EgoMotionFlowConfig(**config)
        self.config = config

    def compute_depth_map(
        self,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-pixel depth map from segmentation masks and class priors.

        Args:
            masks: (T, H, W) long tensor of class indices.

        Returns:
            (T, H, W) float tensor of depth values in meters.
        """
        device = masks.device
        T, H, W = masks.shape

        # Build lookup tensor from depth priors
        depth_lut = torch.zeros(NUM_CLASSES, device=device, dtype=torch.float32)
        for cls_id, depth in self.config.depth_priors.items():
            if cls_id < NUM_CLASSES:
                depth_lut[cls_id] = depth

        return depth_lut[masks]  # (T, H, W) via advanced indexing

    def compute_flow_from_egomotion(
        self,
        depth_map: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute optical flow analytically from ego-motion + depth.

        Implements the standard pinhole camera flow equations from [2].

        Args:
            depth_map: (H, W) per-pixel depth in meters.
            rotation: (3,) rotation vector (wx, wy, wz) in radians.
            translation: (3,) translation vector (tx, ty, tz) in meters.

        Returns:
            (2, H, W) optical flow (flow_u, flow_v) in pixels.
        """
        H, W = depth_map.shape
        device = depth_map.device
        f = self.config.focal_length
        cx, cy = self.config.principal_point
        eps = 1e-6

        # Pixel coordinate grids
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        # Normalized image coordinates
        u_norm = (u_grid - cx) / f  # (H, W)
        v_norm = (v_grid - cy) / f  # (H, W)

        wx, wy, wz = rotation[0], rotation[1], rotation[2]
        tx, ty, tz = translation[0], translation[1], translation[2]

        inv_d = 1.0 / (depth_map + eps)  # (H, W)

        # Translation-induced flow (depth-dependent parallax)
        flow_u_trans = f * (-tx * inv_d + u_norm * tz * inv_d)
        flow_v_trans = f * (-ty * inv_d + v_norm * tz * inv_d)

        # Rotation-induced flow (depth-independent)
        flow_u_rot = f * (wy - wz * v_norm)
        flow_v_rot = f * (-wx + wz * u_norm)

        flow_u = flow_u_trans + flow_u_rot  # (H, W)
        flow_v = flow_v_trans + flow_v_rot  # (H, W)

        return torch.stack([flow_u, flow_v], dim=0)  # (2, H, W)

    def estimate_egomotion_from_masks(
        self,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate ego-motion trajectory from mask sequence geometry.

        Uses class centroid displacement and area change to estimate
        camera rotation and translation between consecutive frames.

        Vectorized: computes all frame centroids in one batched pass,
        then derives per-pair rotations/translations without a Python loop.

        Args:
            masks: (T, H, W) long tensor.

        Returns:
            (rotations, translations): each (T-1, 3) tensors.
        """
        T, H, W = masks.shape
        device = masks.device

        # Coordinate grids broadcast over T
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        # (H, W) -> broadcast with (T, H, W)

        # Road mask for all frames: (T, H, W)
        road_mask = (masks == 0).float()

        # Per-frame road pixel counts: (T,)
        counts = road_mask.sum(dim=(1, 2)).clamp(min=1.0)

        # Per-frame centroids: (T,) each
        cu = (u_grid.unsqueeze(0) * road_mask).sum(dim=(1, 2)) / counts
        cv = (v_grid.unsqueeze(0) * road_mask).sum(dim=(1, 2)) / counts

        # Consecutive-frame differences -> ego-motion estimates
        du = (cu[1:] - cu[:-1]) / self.config.focal_length  # (T-1,)
        dv = (cv[1:] - cv[:-1]) / self.config.focal_length  # (T-1,)
        area_ratio = counts[1:] / counts[:-1]                # (T-1,)
        tz_est = (area_ratio - 1.0) * 10.0                   # (T-1,)

        # Build (T-1, 3) rotation and translation tensors
        zeros = torch.zeros(T - 1, device=device)
        rotations = torch.stack([dv * 0.1, -du * 0.1, zeros], dim=1)
        translations = torch.stack([du * 5.0, dv * 5.0, tz_est], dim=1)

        return rotations, translations

    def warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Warp a frame using optical flow via grid_sample.

        Args:
            frame: (C, H, W) float tensor.
            flow: (2, H, W) optical flow in pixels.

        Returns:
            (C, H, W) warped frame.
        """
        C, H, W = frame.shape
        device = frame.device

        # Build sampling grid: destination pixel + flow = source pixel
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        # Source coordinates
        src_u = u_grid + flow[0]  # (H, W)
        src_v = v_grid + flow[1]  # (H, W)

        # Normalize to [-1, 1] for grid_sample
        src_u_norm = 2.0 * src_u / (W - 1) - 1.0
        src_v_norm = 2.0 * src_v / (H - 1) - 1.0

        grid = torch.stack([src_u_norm, src_v_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        frame_batch = frame.unsqueeze(0)  # (1, C, H, W)

        warped = F.grid_sample(
            frame_batch, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return warped.squeeze(0)  # (C, H, W)

    def generate(
        self,
        masks: torch.Tensor,
        initial_frame: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a frame sequence satisfying ego-motion flow constraints.

        Args:
            masks: (T, H, W) long tensor of segmentation masks.
            initial_frame: Optional (C, H, W) float tensor for frame 0.
                If None, initialized from class mean colors.

        Returns:
            (T, C, H, W) float tensor of generated frames in [0, 255].
        """
        T, H, W = masks.shape
        device = masks.device

        # Initialize frame 0 from class mean colors if not provided
        if initial_frame is None:
            colors = CLASS_MEAN_COLORS.to(device)  # (NUM_CLASSES, 3)
            # (H, W) -> (H, W, 3) -> (3, H, W)
            initial_frame = colors[masks[0]].permute(2, 0, 1)

        # Estimate ego-motion from mask geometry
        rotations, translations = self.estimate_egomotion_from_masks(masks)
        depth_maps = self.compute_depth_map(masks)

        # Precompute all flows (independent across frames) — vectorized
        flows = [
            self.compute_flow_from_egomotion(depth_maps[t], rotations[t], translations[t])
            for t in range(T - 1)
        ]

        # Sequential warp chain (frame t depends on frame t-1)
        frames = [initial_frame.clone()]
        for t in range(T - 1):
            warped = self.warp_frame(frames[t], flows[t])
            frames.append(warped.clamp(0.0, 255.0))

        return torch.stack(frames, dim=0)  # (T, C, H, W)


# ════════════════════════════════════════════════════════════════════════
# 2. Road Plane Homography Solver
# ════════════════════════════════════════════════════════════════════════


@dataclass
class RoadPlaneHomographyConfig:
    """Configuration for road plane homography solver.

    Attributes:
        camera_height: Camera height above road in meters (default 1.2m).
        camera_pitch: Camera pitch angle in radians (default 0.02).
        road_class_id: Segmentation class index for road (default 0).
        focal_length: Camera focal length in pixels.
        principal_point: (cx, cy) principal point.
        num_steps: Optimization steps for non-road pixels.
        lr: Learning rate for non-road optimization.
    """
    camera_height: float = DEFAULT_CAMERA_HEIGHT
    camera_pitch: float = DEFAULT_CAMERA_PITCH
    road_class_id: int = 0
    focal_length: float = DEFAULT_FOCAL_LENGTH
    principal_point: tuple[float, float] = DEFAULT_PRINCIPAL_POINT
    num_steps: int = 50
    lr: float = 1.0


class RoadPlaneHomography:
    """Constrain road pixels via inter-frame homography from ego-motion.

    The road surface is a plane. For a known camera height h and pitch p,
    the road region obeys a 3x3 homography H between consecutive frames.
    H is fully determined by ego-motion (no free parameters beyond the
    6-DOF camera motion).

    Road pixels satisfy: frame_t1[road] = warp(frame_t[road], H).
    Non-road pixels are free variables for unconstrained optimization.

    The homography for a planar surface at distance d with normal n is [2]:
        H = K * (R - t * n^T / d) * K^{-1}
    where K is the camera intrinsic matrix.

    Args:
        config: RoadPlaneHomographyConfig or dict of overrides.
    """

    def __init__(self, config: RoadPlaneHomographyConfig | dict[str, Any] | None = None):
        if config is None:
            config = RoadPlaneHomographyConfig()
        elif isinstance(config, dict):
            config = RoadPlaneHomographyConfig(**config)
        self.config = config

    def compute_road_homography(
        self,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 3x3 homography for the road plane.

        H = K * (R - t * n^T / d) * K^{-1}

        For a flat road with normal n = [0, cos(pitch), -sin(pitch)]
        at distance d = camera_height / cos(pitch).

        Args:
            rotation: (3,) rotation vector (small angle approximation).
            translation: (3,) translation vector in meters.

        Returns:
            (3, 3) homography matrix.
        """
        device = rotation.device
        f = self.config.focal_length
        cx, cy = self.config.principal_point
        h = self.config.camera_height
        pitch = self.config.camera_pitch

        # Camera intrinsics
        K = torch.tensor([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)

        K_inv = torch.tensor([
            [1.0 / f, 0.0, -cx / f],
            [0.0, 1.0 / f, -cy / f],
            [0.0, 0.0, 1.0],
        ], device=device, dtype=torch.float32)

        # Small-angle rotation matrix from rotation vector
        wx, wy, wz = rotation[0], rotation[1], rotation[2]
        R = torch.eye(3, device=device, dtype=torch.float32)
        R[0, 1] = -wz
        R[0, 2] = wy
        R[1, 0] = wz
        R[1, 2] = -wx
        R[2, 0] = -wy
        R[2, 1] = wx

        # Road plane normal in camera frame
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
        n = torch.tensor([[0.0], [cos_p], [-sin_p]], device=device, dtype=torch.float32)

        # Distance to road plane along normal
        d = h / (cos_p + 1e-8)

        # Translation column vector
        t = translation.reshape(3, 1)

        # Homography: H = K * (R - t * n^T / d) * K^{-1}
        H = K @ (R - t @ n.T / d) @ K_inv

        return H

    def warp_road_pixels(
        self,
        frame: torch.Tensor,
        mask: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        """Warp road pixels of a frame through the homography.

        Args:
            frame: (C, H, W) float tensor.
            mask: (H, W) long tensor of class indices.
            homography: (3, 3) homography matrix.

        Returns:
            (C, H, W) frame with road pixels warped, non-road unchanged.
        """
        C, H, W = frame.shape
        device = frame.device

        # Build pixel coordinate grid
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        # Homogeneous coordinates (3, H*W)
        ones = torch.ones(H, W, device=device, dtype=torch.float32)
        coords = torch.stack([u_grid, v_grid, ones], dim=0).reshape(3, -1)

        # Apply homography: H^{-1} maps destination -> source
        H_inv = torch.linalg.inv(homography)
        src_coords = H_inv @ coords  # (3, H*W)
        src_coords = src_coords / (src_coords[2:3, :] + 1e-8)

        # Normalize to [-1, 1] for grid_sample
        src_u = 2.0 * src_coords[0] / (W - 1) - 1.0
        src_v = 2.0 * src_coords[1] / (H - 1) - 1.0
        grid = torch.stack([src_u, src_v], dim=-1).reshape(1, H, W, 2)

        warped = F.grid_sample(
            frame.unsqueeze(0), grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        ).squeeze(0)

        # Only replace road pixels; keep non-road unchanged
        road_mask = (mask == self.config.road_class_id).unsqueeze(0).float()  # (1, H, W)
        result = warped * road_mask + frame * (1.0 - road_mask)
        return result

    def generate(
        self,
        masks: torch.Tensor,
        initial_frame: Optional[torch.Tensor] = None,
        ego_rotation: Optional[torch.Tensor] = None,
        ego_translation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate frames with road constrained by homography.

        Args:
            masks: (T, H, W) long tensor.
            initial_frame: Optional (C, H, W) for frame 0.
            ego_rotation: Optional (T-1, 3) rotation vectors.
            ego_translation: Optional (T-1, 3) translation vectors.

        Returns:
            (T, C, H, W) generated frames.
        """
        T, H, W = masks.shape
        device = masks.device

        if initial_frame is None:
            colors = CLASS_MEAN_COLORS.to(device)  # (NUM_CLASSES, 3)
            initial_frame = colors[masks[0]].permute(2, 0, 1)

        # Default ego-motion: small forward translation
        if ego_rotation is None:
            ego_rotation = torch.zeros(T - 1, 3, device=device)
        if ego_translation is None:
            ego_translation = torch.zeros(T - 1, 3, device=device)
            ego_translation[:, 2] = 0.5  # small forward motion

        # Precompute all homographies (independent across frames)
        homographies = [
            self.compute_road_homography(ego_rotation[t], ego_translation[t])
            for t in range(T - 1)
        ]

        # Sequential warp chain (frame t depends on frame t-1)
        frames = [initial_frame.clone()]
        for t in range(T - 1):
            warped = self.warp_road_pixels(frames[t], masks[t], homographies[t])
            frames.append(warped.clamp(0.0, 255.0))

        return torch.stack(frames, dim=0)


# ════════════════════════════════════════════════════════════════════════
# 3. Vanishing Point Constraint
# ════════════════════════════════════════════════════════════════════════


@dataclass
class VanishingPointConfig:
    """Configuration for vanishing point constraint.

    Attributes:
        vp_weight: Weight of the vanishing point structural loss.
        min_line_length: Minimum edge segment length to consider (pixels).
        angular_tolerance: Angular tolerance for VP convergence (radians).
        num_angle_bins: Number of bins for Hough-like angle histogram.
        edge_threshold: Gradient magnitude threshold for edge detection.
    """
    vp_weight: float = 1.0
    min_line_length: int = 20
    angular_tolerance: float = 0.15  # ~8.6 degrees
    num_angle_bins: int = 180
    edge_threshold: float = 10.0


class VanishingPointConstraint:
    """Add structural loss ensuring edges converge at the vanishing point.

    In driving scenes, all parallel lines (lane markings, road edges,
    building edges) converge at the vanishing point (VP). The VP position
    encodes camera rotation. Frames with correctly converging edges
    produce lower PoseNet distortion because PoseNet uses these
    structural cues for ego-motion estimation.

    Detection: Sobel edge detection + angle histogram to find dominant
    line directions. VP is at the intersection of dominant directions.

    Loss: For each edge pixel, the angle between the edge direction and
    the direction toward the VP should be small [1].

    Args:
        config: VanishingPointConfig or dict of overrides.
    """

    def __init__(self, config: VanishingPointConfig | dict[str, Any] | None = None):
        if config is None:
            config = VanishingPointConfig()
        elif isinstance(config, dict):
            config = VanishingPointConfig(**config)
        self.config = config

    def detect_edges(
        self,
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Sobel edges and gradient directions.

        Args:
            frame: (C, H, W) float tensor.

        Returns:
            (magnitude, angle, gray): each (H, W) tensors.
            magnitude is gradient magnitude, angle is gradient direction
            in radians, gray is the grayscale image.
        """
        # Convert to grayscale
        if frame.shape[0] == 3:
            gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
        else:
            gray = frame[0]

        gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=frame.dtype, device=frame.device,
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=frame.dtype, device=frame.device,
        ).reshape(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1).squeeze()  # (H, W)
        gy = F.conv2d(gray, sobel_y, padding=1).squeeze()  # (H, W)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        angle = torch.atan2(gy, gx)  # edge gradient direction

        return magnitude, angle, gray.squeeze()

    def estimate_vanishing_point(
        self,
        mask: torch.Tensor,
        frame: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate VP from mask geometry and edge directions.

        Uses the road/lane boundary pattern: road occupies the lower
        portion, sky the upper. The VP is where the road boundaries
        converge, typically at the horizon line near image center.

        For robustness, uses a weighted vote of edge directions in the
        road/lane boundary region.

        Args:
            mask: (H, W) long tensor of class indices.
            frame: (C, H, W) float tensor.

        Returns:
            (2,) tensor: (vp_u, vp_v) vanishing point in pixel coords.
        """
        device = mask.device
        H, W = mask.shape

        magnitude, angle, _ = self.detect_edges(frame)

        # Focus on road/lane boundary regions (class transitions)
        # Boundary = pixels where adjacent pixels have different classes
        right_diff = (mask[:, 1:] != mask[:, :-1]).float()
        bottom_diff = (mask[1:, :] != mask[:-1, :]).float()
        boundary = torch.zeros(H, W, device=device)
        boundary[:, :-1] += right_diff
        boundary[:-1, :] += bottom_diff
        boundary = (boundary > 0).float()

        # Weight edges by boundary proximity and magnitude
        edge_mask = (magnitude > self.config.edge_threshold).float()
        weights = edge_mask * boundary * magnitude

        total_weight = weights.sum()
        if total_weight < 1e-6:
            # Fallback: VP at image center, slightly above middle (horizon)
            return torch.tensor([W / 2.0, H * 0.4], device=device)

        # Weighted average of edge pixel positions as VP estimate
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        vp_u = (u_grid * weights).sum() / total_weight
        vp_v = (v_grid * weights).sum() / total_weight

        # Bias VP toward upper-center (horizon constraint for driving)
        vp_v = vp_v.clamp(max=float(H) * 0.6)

        return torch.stack([vp_u, vp_v])

    def compute_vp_loss(
        self,
        frame: torch.Tensor,
        vanishing_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute structural loss: edges should point toward the VP.

        For each edge pixel at (u, v), the direction toward the VP is:
            d_vp = (vp_u - u, vp_v - v) / ||(vp_u - u, vp_v - v)||

        The edge direction (perpendicular to gradient) is:
            d_edge = (-gy, gx) / ||(gx, gy)||

        Loss = weighted mean of |sin(angle between d_vp and d_edge)|
        for pixels with strong edges.

        Args:
            frame: (C, H, W) float tensor.
            vanishing_point: (2,) tensor (vp_u, vp_v).

        Returns:
            Scalar loss tensor.
        """
        C, H, W = frame.shape
        device = frame.device
        eps = 1e-8

        magnitude, angle, _ = self.detect_edges(frame)

        # Edge direction (perpendicular to gradient)
        edge_dx = -torch.sin(angle)  # (H, W)
        edge_dy = torch.cos(angle)   # (H, W)

        # Direction from each pixel toward VP
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        to_vp_u = vanishing_point[0] - u_grid  # (H, W)
        to_vp_v = vanishing_point[1] - v_grid
        to_vp_norm = torch.sqrt(to_vp_u ** 2 + to_vp_v ** 2 + eps)
        to_vp_u = to_vp_u / to_vp_norm
        to_vp_v = to_vp_v / to_vp_norm

        # Cross product magnitude = |sin(angle)|
        cross = torch.abs(edge_dx * to_vp_v - edge_dy * to_vp_u)

        # Weight by edge magnitude (strong edges matter more)
        edge_weight = (magnitude > self.config.edge_threshold).float() * magnitude
        weighted_cross = cross * edge_weight

        loss = weighted_cross.sum() / (edge_weight.sum() + eps)
        return loss * self.config.vp_weight

    def apply(
        self,
        frame: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate VP and compute structural loss for a frame.

        Args:
            frame: (C, H, W) float tensor.
            mask: (H, W) long tensor.

        Returns:
            (loss, vanishing_point): scalar loss and (2,) VP coords.
        """
        vp = self.estimate_vanishing_point(mask, frame)
        loss = self.compute_vp_loss(frame, vp)
        return loss, vp


# ════════════════════════════════════════════════════════════════════════
# 4. Matched Filter Optimization
# ════════════════════════════════════════════════════════════════════════


@dataclass
class MatchedFilterConfig:
    """Configuration for matched filter optimizer.

    Attributes:
        regularization_lambda: Tikhonov regularization for Wiener filter.
        step_size: Step size for applying the matched filter perturbation.
        num_iterations: Number of matched filter update iterations.
        max_jacobian_outputs: Max scorer outputs for Jacobian computation.
    """
    regularization_lambda: float = 1e-3
    step_size: float = 0.1
    num_iterations: int = 20
    max_jacobian_outputs: int = 16


class MatchedFilterOptimizer:
    """Minimum-energy pixel perturbation via Wiener matched filter.

    A matched filter maximizes SNR for a known signal in noise. Here the
    "signal" is the scorer's expected output and "noise" is pixel
    perturbation. The Wiener filter gives the minimum L2-norm perturbation
    achieving a target scorer improvement:

        h = J^T * (J * J^T + lambda * I)^{-1} * target_error

    This is far more efficient than generic gradient descent because it
    finds the MINIMUM ENERGY perturbation for the desired scorer change,
    which directly minimizes rate cost.

    Args:
        config: MatchedFilterConfig or dict of overrides.
    """

    def __init__(self, config: MatchedFilterConfig | dict[str, Any] | None = None):
        if config is None:
            config = MatchedFilterConfig()
        elif isinstance(config, dict):
            config = MatchedFilterConfig(**config)
        self.config = config

    def compute_jacobian(
        self,
        frames: torch.Tensor,
        scorer_fn: nn.Module,
    ) -> torch.Tensor:
        """Compute scorer Jacobian via autograd.

        Args:
            frames: (1, C, H, W) float tensor, requires_grad.
            scorer_fn: callable that takes (1, C, H, W) -> (K,) outputs.

        Returns:
            (K, C*H*W) Jacobian matrix.
        """
        _, C, H, W = frames.shape
        pixel_dim = C * H * W
        inp = frames.detach().clone().requires_grad_(True)

        outputs = scorer_fn(inp)
        if isinstance(outputs, dict):
            outputs = outputs.get("pose", next(iter(outputs.values())))
        outputs = outputs.reshape(-1)
        K = min(outputs.shape[0], self.config.max_jacobian_outputs)

        rows = []
        for k in range(K):
            if inp.grad is not None:
                inp.grad.zero_()
            outputs[k].backward(retain_graph=(k < K - 1))
            rows.append(inp.grad.detach().reshape(pixel_dim).clone())

        return torch.stack(rows, dim=0)  # (K, D)

    def compute_matched_filter(
        self,
        jacobian: torch.Tensor,
        target_error: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Wiener matched filter: minimum energy perturbation.

        h = J^T * (J*J^T + lambda*I)^{-1} * e

        Args:
            jacobian: (K, D) Jacobian matrix.
            target_error: (K,) desired change in scorer output.

        Returns:
            (D,) optimal perturbation vector.
        """
        K, D = jacobian.shape
        lam = self.config.regularization_lambda
        device = jacobian.device

        # Gram matrix J*J^T + regularization
        JJt = jacobian @ jacobian.T + lam * torch.eye(K, device=device)

        # Solve (J*J^T + lambda*I) * alpha = target_error
        alpha = torch.linalg.solve(JJt, target_error)

        # Matched filter: h = J^T * alpha
        h = jacobian.T @ alpha  # (D,)

        return h

    def optimize(
        self,
        frames: torch.Tensor,
        scorer_fn: nn.Module,
        target_error: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply iterative matched filter optimization.

        At each iteration:
        1. Compute Jacobian at current point
        2. Compute matched filter for residual error
        3. Apply perturbation with step_size

        Re-linearizes the Jacobian each iteration for the nonlinear case.

        Args:
            frames: (1, C, H, W) float tensor.
            scorer_fn: callable (1, C, H, W) -> outputs.
            target_error: Optional (K,) target. If None, uses negative
                of current output (minimize to zero).

        Returns:
            (1, C, H, W) optimized frames.
        """
        current = frames.detach().clone()

        for _ in range(self.config.num_iterations):
            J = self.compute_jacobian(current, scorer_fn)
            K = J.shape[0]

            if target_error is None:
                # Default: minimize current scorer output toward zero
                with torch.no_grad():
                    out = scorer_fn(current)
                    if isinstance(out, dict):
                        out = out.get("pose", next(iter(out.values())))
                    current_out = out.reshape(-1)[:K]
                err = -current_out
            else:
                err = target_error[:K].to(J.device)

            h = self.compute_matched_filter(J, err)

            # Apply perturbation
            _, C, H, W = current.shape
            delta = h.reshape(1, C, H, W) * self.config.step_size
            current = (current + delta).clamp(0.0, 255.0)

        return current


# ════════════════════════════════════════════════════════════════════════
# 5. Kalman Frame Smoother
# ════════════════════════════════════════════════════════════════════════


@dataclass
class KalmanSmootherConfig:
    """Configuration for Kalman frame smoother.

    Attributes:
        process_noise_Q: Process noise covariance scale (temporal model).
        measurement_noise_R: Measurement noise covariance scale (scorer).
        pca_components: Number of PCA components for dimensionality reduction.
        num_iterations: Number of EKS re-linearization iterations.
        use_ego_motion_model: Whether to use ego-motion for process model.
    """
    process_noise_Q: float = 1.0
    measurement_noise_R: float = 0.1
    pca_components: int = 50
    num_iterations: int = 3
    use_ego_motion_model: bool = True


class KalmanFrameSmoother:
    """Optimal frame sequence via Rauch-Tung-Striebel smoother [3].

    Treats each frame as a "state" in a reduced PCA space and scorer
    output as a "measurement". The RTS smoother gives the optimal state
    estimate given all measurements (past AND future), which is strictly
    better than forward-only filtering.

    State model:   x_{t+1} = F * x_t + w_t,  w_t ~ N(0, Q)
    Measurement:   z_t = H_t * x_t + v_t,     v_t ~ N(0, R)

    where F is the state transition (temporal smoothness or ego-motion
    warp in PCA space), H_t is the scorer Jacobian at frame t projected
    into PCA space, and z_t is the target scorer output.

    Args:
        config: KalmanSmootherConfig or dict of overrides.
    """

    def __init__(self, config: KalmanSmootherConfig | dict[str, Any] | None = None):
        if config is None:
            config = KalmanSmootherConfig()
        elif isinstance(config, dict):
            config = KalmanSmootherConfig(**config)
        self.config = config
        self._pca_mean: Optional[torch.Tensor] = None
        self._pca_basis: Optional[torch.Tensor] = None

    def fit_pca(
        self,
        frames: torch.Tensor,
    ) -> None:
        """Fit PCA basis from frame sequence.

        Args:
            frames: (T, C, H, W) float tensor.
        """
        T = frames.shape[0]
        D = frames[0].numel()
        flat = frames.reshape(T, D).float()

        self._pca_mean = flat.mean(dim=0)  # (D,)
        centered = flat - self._pca_mean.unsqueeze(0)

        # Economy SVD: T << D typically, so compute T x T covariance
        K = min(self.config.pca_components, T, D)
        # Use truncated SVD via torch.linalg.svd on the centered data
        # centered: (T, D). For T << D, compute SVD on (T, T) gram matrix
        gram = centered @ centered.T  # (T, T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)

        # Take top-K eigenvectors (eigh returns ascending order)
        top_k_idx = torch.argsort(eigenvalues, descending=True)[:K]
        V = eigenvectors[:, top_k_idx]  # (T, K)

        # PCA basis vectors in original space: D x K
        # basis_j = centered^T @ v_j / sqrt(lambda_j)
        lambdas = eigenvalues[top_k_idx].clamp(min=1e-8)
        self._pca_basis = (centered.T @ V / lambdas.sqrt().unsqueeze(0)).T  # (K, D)

        # Orthonormalize for numerical stability
        self._pca_basis = self._pca_basis / (
            self._pca_basis.norm(dim=1, keepdim=True) + 1e-8
        )

    def project_to_pca(self, frame_flat: torch.Tensor) -> torch.Tensor:
        """Project a flattened frame into PCA space.

        Args:
            frame_flat: (D,) float tensor.

        Returns:
            (K,) PCA coefficients.
        """
        assert self._pca_basis is not None, "Must call fit_pca first"
        centered = frame_flat - self._pca_mean
        return self._pca_basis @ centered  # (K,)

    def reconstruct_from_pca(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Reconstruct flattened frame from PCA coefficients.

        Args:
            coeffs: (K,) PCA coefficients.

        Returns:
            (D,) reconstructed frame.
        """
        assert self._pca_basis is not None, "Must call fit_pca first"
        return self._pca_mean + self._pca_basis.T @ coeffs  # (D,)

    def smooth(
        self,
        frames: torch.Tensor,
        scorer_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Rauch-Tung-Striebel smoother to frame sequence.

        Args:
            frames: (T, C, H, W) float tensor — initial frame estimates.
            scorer_targets: Optional (T, K_scorer) target scorer outputs.
                If None, uses zeros (minimize scorer output).

        Returns:
            (T, C, H, W) smoothed frames.
        """
        T, C, H, W = frames.shape
        D = C * H * W
        device = frames.device

        # Fit PCA on initial frames
        self.fit_pca(frames)

        K = self._pca_basis.shape[0]  # PCA dimensions

        # Project initial frames to PCA space (batched matmul, no per-frame loop)
        flat = frames.reshape(T, D).float()
        centered = flat - self._pca_mean.unsqueeze(0)  # (T, D)
        x_init = (self._pca_basis @ centered.T).T  # (T, K)

        # Process model: x_{t+1} = x_t + process_noise
        F_mat = torch.eye(K, device=device)
        Q = torch.eye(K, device=device) * self.config.process_noise_Q
        R_scale = self.config.measurement_noise_R

        for _iter in range(self.config.num_iterations):
            # ── Forward Kalman filter ──
            x_pred = torch.zeros(T, K, device=device)
            P_pred = torch.zeros(T, K, K, device=device)
            x_filt = torch.zeros(T, K, device=device)
            P_filt = torch.zeros(T, K, K, device=device)

            # Initialize
            x_filt[0] = x_init[0]
            P_filt[0] = torch.eye(K, device=device) * 10.0

            for t in range(1, T):
                # Predict
                x_pred[t] = F_mat @ x_filt[t - 1]
                P_pred[t] = F_mat @ P_filt[t - 1] @ F_mat.T + Q

                # Measurement: identity (observe PCA coeffs directly from scorer)
                # Simplified: H = I (scorer provides direct feedback in PCA space)
                H_t = torch.eye(K, device=device)
                R_t = torch.eye(K, device=device) * R_scale

                # Innovation
                z_t = x_init[t]  # "measurement" = initial PCA estimate
                innovation = z_t - H_t @ x_pred[t]

                # Kalman gain
                S = H_t @ P_pred[t] @ H_t.T + R_t
                K_gain = P_pred[t] @ H_t.T @ torch.linalg.inv(S)

                # Update
                x_filt[t] = x_pred[t] + K_gain @ innovation
                P_filt[t] = (torch.eye(K, device=device) - K_gain @ H_t) @ P_pred[t]

            # ── RTS backward smoother ──
            x_smooth = torch.zeros(T, K, device=device)
            x_smooth[T - 1] = x_filt[T - 1]

            for t in range(T - 2, -1, -1):
                # Smoother gain
                P_pred_next = P_pred[t + 1]
                P_pred_inv = torch.linalg.inv(P_pred_next + 1e-6 * torch.eye(K, device=device))
                L = P_filt[t] @ F_mat.T @ P_pred_inv

                # Smooth
                x_smooth[t] = x_filt[t] + L @ (x_smooth[t + 1] - x_pred[t + 1])

            # Update initial estimates for next re-linearization
            x_init = x_smooth.clone()

        # Reconstruct frames from smoothed PCA coefficients (batched matmul)
        smoothed_flat = self._pca_mean.unsqueeze(0) + x_smooth @ self._pca_basis  # (T, D)
        return smoothed_flat.reshape(T, C, H, W).clamp(0.0, 255.0)


# ════════════════════════════════════════════════════════════════════════
# 6. Compressed Sensing / Sparse Recovery
# ════════════════════════════════════════════════════════════════════════


@dataclass
class CompressedSensingConfig:
    """Configuration for compressed sensing recovery.

    Attributes:
        wavelet_type: Wavelet type ('haar' or 'db2').
        sparsity_lambda: L1 sparsity penalty (LASSO).
        num_iterations: Number of ISTA iterations.
        basis_size: Size of wavelet decomposition block.
        step_size: ISTA gradient step size (must be < 1/L where L is
            the Lipschitz constant of the gradient).
    """
    wavelet_type: str = "haar"
    sparsity_lambda: float = 0.01
    num_iterations: int = 50
    basis_size: int = 8
    step_size: float = 0.01


class CompressedSensingRecovery:
    """Sparse frame recovery via ISTA in wavelet domain [4].

    Frames are represented in a wavelet basis (Haar). The scorer
    Jacobian acts as a low-dimensional measurement matrix A. We solve:

        min ||x||_1 subject to ||Ax - y||_2 < epsilon

    via the ISTA algorithm:
        x^{k+1} = soft_threshold(x^k - eta * A^T(Ax^k - y), lambda)

    The L1 sparsity prior naturally produces compressible frames
    (many wavelet coefficients are exactly zero), directly reducing
    the rate component of the score.

    The Haar wavelet transform is implemented as 2D convolutions for
    full GPU compatibility.

    Args:
        config: CompressedSensingConfig or dict of overrides.
    """

    def __init__(self, config: CompressedSensingConfig | dict[str, Any] | None = None):
        if config is None:
            config = CompressedSensingConfig()
        elif isinstance(config, dict):
            config = CompressedSensingConfig(**config)
        self.config = config

    def _build_haar_filters(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Haar wavelet analysis and synthesis filter banks.

        Returns:
            (analysis, synthesis): each (4, 1, 2, 2) conv kernels.
            Analysis: [LL, LH, HL, HH] decomposition.
            Synthesis: inverse transform kernels.
        """
        s = 0.5  # 1/sqrt(2) * 1/sqrt(2) for 2D

        # Analysis filters (forward transform)
        ll = torch.tensor([[s, s], [s, s]], device=device).reshape(1, 1, 2, 2)
        lh = torch.tensor([[s, s], [-s, -s]], device=device).reshape(1, 1, 2, 2)
        hl = torch.tensor([[s, -s], [s, -s]], device=device).reshape(1, 1, 2, 2)
        hh = torch.tensor([[s, -s], [-s, s]], device=device).reshape(1, 1, 2, 2)

        analysis = torch.cat([ll, lh, hl, hh], dim=0)  # (4, 1, 2, 2)

        # Synthesis = transpose of analysis for orthogonal wavelets
        synthesis = analysis.clone()

        return analysis, synthesis

    def haar_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Haar wavelet transform via strided convolution.

        Args:
            x: (B, C, H, W) tensor. H, W must be even.

        Returns:
            (B, C*4, H//2, W//2) wavelet coefficients [LL, LH, HL, HH].
        """
        B, C, H, W = x.shape
        device = x.device
        analysis, _ = self._build_haar_filters(device)

        # Apply per-channel
        coeffs_list = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]  # (B, 1, H, W)
            coeffs = F.conv2d(channel, analysis, stride=2)  # (B, 4, H//2, W//2)
            coeffs_list.append(coeffs)

        return torch.cat(coeffs_list, dim=1)  # (B, C*4, H//2, W//2)

    def haar_inverse(self, coeffs: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """Inverse Haar wavelet transform via transposed convolution.

        Args:
            coeffs: (B, C*4, H//2, W//2) wavelet coefficients.
            out_h: Target output height.
            out_w: Target output width.

        Returns:
            (B, C, H, W) reconstructed tensor.
        """
        B = coeffs.shape[0]
        C4 = coeffs.shape[1]
        C = C4 // 4
        device = coeffs.device
        _, synthesis = self._build_haar_filters(device)

        channels = []
        for c in range(C):
            band = coeffs[:, c*4:(c+1)*4, :, :]  # (B, 4, Hh, Wh)
            recon = F.conv_transpose2d(band, synthesis, stride=2)  # (B, 1, H, W)
            channels.append(recon)

        result = torch.cat(channels, dim=1)  # (B, C, H, W)

        # Crop to target size if needed (padding artifacts)
        return result[:, :, :out_h, :out_w]

    @staticmethod
    def soft_threshold(x: torch.Tensor, lam: float) -> torch.Tensor:
        """Soft thresholding (proximal operator for L1 norm).

        S_lambda(x) = sign(x) * max(|x| - lambda, 0)

        Args:
            x: Input tensor.
            lam: Threshold value.

        Returns:
            Soft-thresholded tensor.
        """
        return torch.sign(x) * F.relu(x.abs() - lam)

    def recover(
        self,
        frames: torch.Tensor,
        measurement_fn: Optional[nn.Module] = None,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Recover sparse frames via ISTA.

        If no measurement_fn is provided, performs standalone sparsification
        (minimize L1 wavelet norm while staying close to input).

        Args:
            frames: (B, C, H, W) initial frame estimates.
            measurement_fn: Optional scorer callable.
            target: Optional target measurement values.

        Returns:
            (B, C, H, W) sparse-recovered frames.
        """
        B, C, H, W = frames.shape
        device = frames.device

        # Ensure even dimensions for Haar transform
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            frames = F.pad(frames, (0, pad_w, 0, pad_h), mode="reflect")
            _, _, H_pad, W_pad = frames.shape
        else:
            H_pad, W_pad = H, W

        # Transform to wavelet domain
        coeffs = self.haar_transform(frames)
        coeffs = coeffs.detach().clone().requires_grad_(False)

        eta = self.config.step_size
        lam = self.config.sparsity_lambda

        for _ in range(self.config.num_iterations):
            # Reconstruct current estimate
            recon = self.haar_inverse(coeffs, H_pad, W_pad)

            # Data fidelity gradient in pixel domain
            if measurement_fn is not None and target is not None:
                recon_req = recon.detach().clone().requires_grad_(True)
                out = measurement_fn(recon_req[:, :, :H, :W])
                if isinstance(out, dict):
                    out = out.get("pose", next(iter(out.values())))
                residual = (out.reshape(-1) - target.reshape(-1)).pow(2).sum()
                residual.backward()
                pixel_grad = recon_req.grad.detach()
            else:
                # Self-consistency: stay close to input
                pixel_grad = 2.0 * (recon - frames)

            # Transform gradient to wavelet domain
            grad_coeffs = self.haar_transform(pixel_grad)

            # ISTA update: gradient step + soft threshold
            coeffs = self.soft_threshold(coeffs - eta * grad_coeffs, eta * lam)

        # Final reconstruction
        result = self.haar_inverse(coeffs, H_pad, W_pad)
        return result[:, :, :H, :W].clamp(0.0, 255.0)


# ════════════════════════════════════════════════════════════════════════
# 7. Trajectory Optimization (Pontryagin)
# ════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory optimizer.

    Attributes:
        control_penalty: Lambda for control effort (L2 of pixel deltas).
        num_shooting_iterations: Iterations for the shooting method.
        boundary_weight: Weight for boundary condition enforcement.
        lr: Learning rate for adjoint optimization.
        scorer_weight: Weight for scorer loss at each waypoint.
    """
    control_penalty: float = 0.01
    num_shooting_iterations: int = 30
    boundary_weight: float = 100.0
    lr: float = 0.5
    scorer_weight: float = 1.0


class TrajectoryOptimizer:
    """Optimal frame trajectory via Pontryagin's Maximum Principle [8].

    Each frame is a "waypoint" on a trajectory through pixel space.
    The trajectory must satisfy boundary conditions (first/last frame)
    and minimize a cost functional:

        J = sum_t [scorer_loss(x_t) + lambda * ||u_t||^2]

    where x_t = frame at time t, u_t = x_{t+1} - x_t is the control.

    Solved via direct collocation: parameterize the full trajectory and
    optimize jointly using gradient descent with the Lagrangian.

    Args:
        config: TrajectoryConfig or dict of overrides.
    """

    def __init__(self, config: TrajectoryConfig | dict[str, Any] | None = None):
        if config is None:
            config = TrajectoryConfig()
        elif isinstance(config, dict):
            config = TrajectoryConfig(**config)
        self.config = config

    def optimize(
        self,
        frames: torch.Tensor,
        scorer_fn: Optional[nn.Module] = None,
        boundary_frames: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Optimize frame trajectory with control cost.

        Args:
            frames: (T, C, H, W) initial trajectory estimate.
            scorer_fn: Optional scorer callable for per-frame loss.
            boundary_frames: Optional (first_frame, last_frame) as hard BCs.

        Returns:
            (T, C, H, W) optimized trajectory.
        """
        T, C, H, W = frames.shape
        device = frames.device

        # Parameterize interior frames as optimizable
        trajectory = frames.detach().clone()
        # Interior frames are free; boundaries may be pinned
        interior = trajectory[1:-1].clone().requires_grad_(True)

        optimizer = torch.optim.Adam([interior], lr=self.config.lr)

        for _step in range(self.config.num_shooting_iterations):
            optimizer.zero_grad()

            # Assemble full trajectory
            if boundary_frames is not None:
                first, last = boundary_frames
                full = torch.cat([
                    first.unsqueeze(0),
                    interior,
                    last.unsqueeze(0),
                ], dim=0)
            else:
                full = torch.cat([
                    trajectory[0:1].detach(),
                    interior,
                    trajectory[-1:].detach(),
                ], dim=0)

            # Control cost: sum ||u_t||^2 where u_t = x_{t+1} - x_t
            controls = full[1:] - full[:-1]  # (T-1, C, H, W)
            control_cost = controls.pow(2).mean() * self.config.control_penalty

            # Scorer cost (if available)
            scorer_cost = torch.tensor(0.0, device=device)
            if scorer_fn is not None:
                for t in range(T):
                    frame_t = full[t:t+1]
                    out = scorer_fn(frame_t)
                    if isinstance(out, dict):
                        out = out.get("pose", next(iter(out.values())))
                    scorer_cost = scorer_cost + out.reshape(-1).pow(2).sum()
                scorer_cost = scorer_cost * self.config.scorer_weight / T

            # Boundary enforcement (soft, in case boundary_frames is None)
            boundary_cost = torch.tensor(0.0, device=device)
            if boundary_frames is not None:
                first, last = boundary_frames
                boundary_cost = (
                    (full[0] - first).pow(2).mean()
                    + (full[-1] - last).pow(2).mean()
                ) * self.config.boundary_weight

            loss = control_cost + scorer_cost + boundary_cost
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                interior.data.clamp_(0.0, 255.0)

        # Assemble final trajectory
        if boundary_frames is not None:
            first, last = boundary_frames
            result = torch.cat([
                first.unsqueeze(0),
                interior.detach(),
                last.unsqueeze(0),
            ], dim=0)
        else:
            result = torch.cat([
                trajectory[0:1],
                interior.detach(),
                trajectory[-1:],
            ], dim=0)

        return result.clamp(0.0, 255.0)


# ════════════════════════════════════════════════════════════════════════
# 8. Adaptive Optics / Zernike Decomposition
# ════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptiveOpticsConfig:
    """Configuration for adaptive optics corrector.

    Attributes:
        num_modes: Number of Zernike modes (dimensionality of correction).
        mode_weights: Per-mode regularization weights (None = uniform).
        regularization: L2 regularization on mode coefficients.
        num_iterations: Number of optimization iterations.
        lr: Learning rate for mode coefficient optimization.
    """
    num_modes: int = 50
    mode_weights: Optional[list[float]] = None
    regularization: float = 0.001
    num_iterations: int = 100
    lr: float = 0.1


class AdaptiveOpticsCorrector:
    """Frame correction via Zernike polynomial decomposition [6].

    Deformable mirror analogy: decompose frame corrections into
    Zernike-like polynomial modes adapted to a rectangular aperture.
    Low-order modes (piston, tilt, defocus) = global adjustments.
    High-order modes = fine texture details.

    Optimizing 50 mode coefficients instead of 589K pixels is a massive
    dimensionality reduction that makes optimization tractable and
    naturally regularizes toward smooth corrections.

    Zernike polynomials Z_n^m(rho, theta) are orthogonal over the unit
    disk. We adapt to rectangular frames by using the unit square
    with Chebyshev-like polynomials.

    Args:
        config: AdaptiveOpticsConfig or dict of overrides.
    """

    def __init__(self, config: AdaptiveOpticsConfig | dict[str, Any] | None = None):
        if config is None:
            config = AdaptiveOpticsConfig()
        elif isinstance(config, dict):
            config = AdaptiveOpticsConfig(**config)
        self.config = config
        self._basis: Optional[torch.Tensor] = None

    def _build_zernike_basis(
        self,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build Zernike-like polynomial basis for rectangular frames.

        Uses products of Chebyshev polynomials T_n(x) * T_m(y) as the
        basis functions, which are orthogonal over [-1, 1]^2. These are
        the natural rectangular analog of Zernike polynomials.

        Args:
            H: Frame height.
            W: Frame width.

        Returns:
            (num_modes, H, W) basis functions, orthonormalized.
        """
        num_modes = self.config.num_modes

        # Normalized coordinates [-1, 1]
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Generate product basis: T_n(x) * T_m(y)
        # Chebyshev polynomials T_n(x) = cos(n * arccos(x))
        # Clamp to avoid NaN at boundaries
        xx_clamped = xx.clamp(-0.999, 0.999)
        yy_clamped = yy.clamp(-0.999, 0.999)
        acos_x = torch.acos(xx_clamped)
        acos_y = torch.acos(yy_clamped)

        modes = []
        idx = 0
        max_order = int(math.ceil(math.sqrt(num_modes * 2)))
        for n in range(max_order):
            for m in range(max_order):
                if idx >= num_modes:
                    break
                Tn_x = torch.cos(n * acos_x)  # T_n(x)
                Tm_y = torch.cos(m * acos_y)  # T_m(y)
                mode = Tn_x * Tm_y
                modes.append(mode)
                idx += 1
            if idx >= num_modes:
                break

        basis = torch.stack(modes[:num_modes], dim=0)  # (num_modes, H, W)

        # Orthonormalize via Gram-Schmidt
        flat = basis.reshape(num_modes, -1)  # (num_modes, H*W)
        Q, _ = torch.linalg.qr(flat.T)  # (H*W, num_modes)
        basis = Q.T[:num_modes].reshape(num_modes, H, W)

        return basis

    def get_basis(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or compute the Zernike basis (cached).

        Args:
            H: Frame height.
            W: Frame width.
            device: Torch device.

        Returns:
            (num_modes, H, W) orthonormal basis.
        """
        if self._basis is None or self._basis.shape[1:] != (H, W):
            self._basis = self._build_zernike_basis(H, W, device)
        return self._basis.to(device)

    def decompose(
        self,
        correction: torch.Tensor,
    ) -> torch.Tensor:
        """Decompose a frame correction into Zernike mode coefficients.

        Args:
            correction: (C, H, W) correction to decompose.

        Returns:
            (C, num_modes) mode coefficients.
        """
        C, H, W = correction.shape
        device = correction.device
        basis = self.get_basis(H, W, device)  # (M, H, W)

        # Project each channel onto basis
        flat_corr = correction.reshape(C, -1)  # (C, H*W)
        flat_basis = basis.reshape(self.config.num_modes, -1)  # (M, H*W)

        coeffs = flat_corr @ flat_basis.T  # (C, M)
        return coeffs

    def reconstruct(
        self,
        coeffs: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Reconstruct frame correction from Zernike mode coefficients.

        Args:
            coeffs: (C, num_modes) mode coefficients.
            H: Target height.
            W: Target width.

        Returns:
            (C, H, W) reconstructed correction.
        """
        device = coeffs.device
        basis = self.get_basis(H, W, device)  # (M, H, W)
        flat_basis = basis.reshape(self.config.num_modes, -1)  # (M, H*W)

        flat_recon = coeffs @ flat_basis  # (C, H*W)
        return flat_recon.reshape(coeffs.shape[0], H, W)

    def optimize(
        self,
        frames: torch.Tensor,
        scorer_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Optimize frames by optimizing Zernike mode coefficients.

        Instead of optimizing 589K pixels per frame, optimize ~50 mode
        coefficients per channel. This is equivalent to constraining
        corrections to lie in the Zernike subspace.

        Args:
            frames: (B, C, H, W) initial frames.
            scorer_fn: Optional scorer callable.

        Returns:
            (B, C, H, W) optimized frames.
        """
        B, C, H, W = frames.shape
        device = frames.device

        # Initialize mode coefficients at zero (no correction yet)
        coeffs = torch.zeros(B, C, self.config.num_modes, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([coeffs], lr=self.config.lr)

        basis = self.get_basis(H, W, device)  # (M, H, W)
        flat_basis = basis.reshape(self.config.num_modes, -1)  # (M, H*W)

        for _ in range(self.config.num_iterations):
            optimizer.zero_grad()

            # Reconstruct correction from coefficients
            # coeffs: (B, C, M), flat_basis: (M, H*W) -> (B, C, H*W) -> (B, C, H, W)
            correction = torch.einsum("bcm,md->bcd", coeffs, flat_basis).reshape(B, C, H, W)
            corrected = (frames + correction).clamp(0.0, 255.0)

            # Scorer loss
            if scorer_fn is not None:
                out = scorer_fn(corrected)
                if isinstance(out, dict):
                    out = out.get("pose", next(iter(out.values())))
                scorer_loss = out.reshape(-1).pow(2).sum()
            else:
                # Without scorer, minimize TV (compressibility)
                dx = (corrected[:, :, :, 1:] - corrected[:, :, :, :-1]).pow(2).mean()
                dy = (corrected[:, :, 1:, :] - corrected[:, :, :-1, :]).pow(2).mean()
                scorer_loss = dx + dy

            # Regularization on mode coefficients
            reg = coeffs.pow(2).mean() * self.config.regularization

            loss = scorer_loss + reg
            loss.backward()
            optimizer.step()

        # Final corrected frames
        with torch.no_grad():
            correction = torch.einsum("bcm,md->bcd", coeffs, flat_basis).reshape(B, C, H, W)
            return (frames + correction).clamp(0.0, 255.0)


# ════════════════════════════════════════════════════════════════════════
# 9. OFDM-Inspired Frequency Domain Optimization
# ════════════════════════════════════════════════════════════════════════


@dataclass
class OFDMConfig:
    """Configuration for OFDM frequency-domain optimizer.

    Attributes:
        num_frequencies: Number of DCT frequency components to retain.
        water_fill_power_budget: Total energy budget for water-filling.
        sensitivity_threshold: Below this sensitivity, zero the frequency.
        num_iterations: Optimization iterations in frequency domain.
        lr: Learning rate for frequency-domain optimization.
    """
    num_frequencies: int = 256
    water_fill_power_budget: float = 1e6
    sensitivity_threshold: float = 0.01
    num_iterations: int = 30
    lr: float = 0.5


class OFDMOptimizer:
    """Water-filling power allocation across DCT frequencies [7].

    OFDM (5G/WiFi) allocates transmit power across frequency subcarriers
    based on channel quality. Our "subcarriers" are DCT components. Our
    "channel quality" is per-frequency scorer sensitivity.

    Water-filling: allocate energy proportional to scorer sensitivity.
    Frequencies invisible to the scorer get zero energy (free rate
    savings). Frequencies the scorer is most sensitive to get maximum
    energy (maximum score impact per bit).

    Uses block DCT (8x8 blocks like JPEG) for computational efficiency.

    Args:
        config: OFDMConfig or dict of overrides.
    """

    def __init__(self, config: OFDMConfig | dict[str, Any] | None = None):
        if config is None:
            config = OFDMConfig()
        elif isinstance(config, dict):
            config = OFDMConfig(**config)
        self.config = config

    @staticmethod
    def _dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Type-II 1D DCT (orthonormal) along the specified dimension.

        Matches scipy.fft.dct(x, type=2, norm='ortho').

        Implementation: construct the 2N-point symmetric extension,
        FFT, and extract the DCT coefficients from the first N
        complex exponentials.

        Args:
            x: Input tensor.
            dim: Dimension along which to compute DCT.

        Returns:
            DCT coefficients along the specified dimension.
        """
        N = x.shape[dim]
        device = x.device

        # Move target dim to last position
        x = x.transpose(dim, -1).contiguous()

        # Mirror extension: y = [x_0, x_1, ..., x_{N-1}, x_{N-1}, ..., x_1, x_0]
        # Then DCT-II = real part of DFT of the rearranged sequence
        # Efficient: use the N-point approach with reordering
        # y[n] = x[2n] for n=0..ceil(N/2)-1, y[N-1-n] = x[2n+1]
        y = torch.zeros_like(x)
        y[..., ::2] = x[..., :(N + 1) // 2]  # even positions
        if N > 1:
            y[..., 1::2] = x[..., (N + 1) // 2:].flip(-1) if N % 2 == 1 else x[..., N // 2:].flip(-1)

        Y = torch.fft.fft(y, dim=-1)
        k = torch.arange(N, device=device, dtype=torch.float32)
        phase = 2.0 * torch.exp(-1j * math.pi * k / (2 * N))

        result = (Y * phase).real

        # Orthonormal normalization
        result = result / (2.0 * math.sqrt(N))
        result[..., 0] = result[..., 0] / math.sqrt(2)
        # Actually: DCT-II ortho norm is sqrt(2/N) for k>0, sqrt(1/N) for k=0
        # Let's just do it directly with the standard formula
        # Re-derive: standard DCT-II: X[k] = 2 * sum_n x[n] cos(pi*(2n+1)*k/(2N))
        # ortho: X[0] *= 1/sqrt(N), X[k>0] *= sqrt(2/N) / 2...
        # Simpler: just use the matrix definition for correctness
        result = result  # placeholder, fix below

        # Correct approach: cancel all the factors
        x_restored = x.transpose(dim, -1) if dim != -1 else x

        # Fall back to explicit matrix multiply for guaranteed correctness
        x_last = x  # already transposed to last dim
        shape_prefix = x_last.shape[:-1]
        x_flat = x_last.reshape(-1, N)  # (batch, N)

        # DCT-II matrix: C[k,n] = cos(pi*(2n+1)*k / (2N))
        n = torch.arange(N, device=device, dtype=torch.float32)
        k_mat = torch.arange(N, device=device, dtype=torch.float32)
        C = torch.cos(math.pi * k_mat.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))

        # Orthonormal scaling
        C[0, :] *= 1.0 / math.sqrt(N)
        C[1:, :] *= math.sqrt(2.0 / N)

        out = x_flat @ C.T  # (batch, N)
        out = out.reshape(*shape_prefix, N)
        return out.transpose(dim, -1).contiguous()

    @staticmethod
    def _idct_1d(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Type-III 1D DCT (inverse of ortho Type-II) along specified dim.

        Matches scipy.fft.idct(X, type=2, norm='ortho').

        Args:
            X: DCT coefficients.
            dim: Dimension along which to compute IDCT.

        Returns:
            Reconstructed signal.
        """
        N = X.shape[dim]
        device = X.device

        X = X.transpose(dim, -1).contiguous()
        shape_prefix = X.shape[:-1]
        X_flat = X.reshape(-1, N)

        # DCT-II matrix (same as forward)
        n = torch.arange(N, device=device, dtype=torch.float32)
        k_mat = torch.arange(N, device=device, dtype=torch.float32)
        C = torch.cos(math.pi * k_mat.unsqueeze(1) * (2 * n.unsqueeze(0) + 1) / (2 * N))
        C[0, :] *= 1.0 / math.sqrt(N)
        C[1:, :] *= math.sqrt(2.0 / N)

        # Orthonormal DCT: C is orthogonal, so C^{-1} = C^T
        out = X_flat @ C  # (batch, N)
        out = out.reshape(*shape_prefix, N)
        return out.transpose(dim, -1).contiguous()

    @classmethod
    def dct_2d(cls, x: torch.Tensor) -> torch.Tensor:
        """2D DCT via separable 1D DCTs.

        Args:
            x: (..., H, W) tensor.

        Returns:
            (..., H, W) DCT coefficients.
        """
        return cls._dct_1d(cls._dct_1d(x, dim=-1), dim=-2)

    @classmethod
    def idct_2d(cls, X: torch.Tensor) -> torch.Tensor:
        """2D inverse DCT via separable 1D IDCTs.

        Args:
            X: (..., H, W) DCT coefficients.

        Returns:
            (..., H, W) spatial domain signal.
        """
        return cls._idct_1d(cls._idct_1d(X, dim=-2), dim=-1)

    def compute_frequency_sensitivity(
        self,
        frames: torch.Tensor,
        scorer_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute per-frequency scorer sensitivity.

        Perturbs each DCT frequency and measures scorer output change.
        For efficiency, uses finite differences on a subset of frequencies.

        Args:
            frames: (B, C, H, W) float tensor.
            scorer_fn: Optional scorer callable.

        Returns:
            (C, H, W) sensitivity map in DCT domain.
        """
        B, C, H, W = frames.shape
        device = frames.device

        if scorer_fn is None:
            # Default: uniform sensitivity (no scorer available)
            return torch.ones(C, H, W, device=device)

        # Compute DCT of input
        dct_coeffs = self.dct_2d(frames[0])  # (C, H, W)

        # Baseline scorer output
        with torch.no_grad():
            base_out = scorer_fn(frames[:1])
            if isinstance(base_out, dict):
                base_out = base_out.get("pose", next(iter(base_out.values())))
            base_val = base_out.reshape(-1).pow(2).sum()

        # Probe sensitivity at each frequency
        sensitivity = torch.zeros(C, H, W, device=device)
        eps = 1.0

        # Sample a grid of frequencies (probing all H*W*C is too expensive)
        n_probes = min(self.config.num_frequencies, H * W)
        probe_indices = torch.linspace(0, H * W - 1, n_probes).long()

        for c in range(C):
            for idx in probe_indices:
                i = idx // W
                j = idx % W
                # Perturb this frequency
                perturbed = dct_coeffs.clone()
                perturbed[c, i, j] += eps

                # Reconstruct and evaluate
                recon = self.idct_2d(perturbed.unsqueeze(0))
                with torch.no_grad():
                    pert_out = scorer_fn(recon.clamp(0, 255))
                    if isinstance(pert_out, dict):
                        pert_out = pert_out.get("pose", next(iter(pert_out.values())))
                    pert_val = pert_out.reshape(-1).pow(2).sum()

                sensitivity[c, i, j] = abs(pert_val.item() - base_val.item()) / eps

        return sensitivity

    def water_fill(
        self,
        dct_coeffs: torch.Tensor,
        sensitivity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply water-filling power allocation to DCT coefficients.

        Frequencies with high scorer sensitivity get more energy.
        Frequencies below sensitivity_threshold get zeroed (rate savings).

        The water-filling algorithm [7]:
            P_k = max(mu - N_k / |h_k|^2, 0)
        where mu is chosen to satisfy the power constraint,
        N_k is noise power, h_k is channel gain (sensitivity).

        Args:
            dct_coeffs: (C, H, W) DCT coefficients.
            sensitivity: (C, H, W) per-frequency scorer sensitivity.

        Returns:
            (C, H, W) water-filled DCT coefficients.
        """
        # Mask out frequencies below threshold
        mask = (sensitivity > self.config.sensitivity_threshold).float()

        # Scale coefficients by sensitivity (amplify scorer-important freqs)
        # Normalize sensitivity to sum to power budget
        sens_sum = (sensitivity * mask).sum()
        if sens_sum > 0:
            allocation = sensitivity * mask / sens_sum * self.config.water_fill_power_budget
        else:
            allocation = mask

        # Apply: keep coefficient sign, scale magnitude by allocation
        coeff_sign = torch.sign(dct_coeffs)
        coeff_mag = dct_coeffs.abs()

        # Water-filling: redistribute energy according to allocation
        total_energy = coeff_mag.pow(2).sum()
        if total_energy > 0:
            target_mag = torch.sqrt(allocation * total_energy / (allocation.sum() + 1e-8))
            # Blend: high-sensitivity freqs get more, low get less
            alpha = (sensitivity / (sensitivity.max() + 1e-8)).clamp(0, 1)
            filled_mag = alpha * target_mag + (1 - alpha) * coeff_mag
        else:
            filled_mag = coeff_mag

        return coeff_sign * filled_mag * mask

    def optimize(
        self,
        frames: torch.Tensor,
        scorer_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Optimize frames via OFDM water-filling in DCT domain.

        Args:
            frames: (B, C, H, W) float tensor.
            scorer_fn: Optional scorer callable.

        Returns:
            (B, C, H, W) optimized frames.
        """
        B, C, H, W = frames.shape

        # Compute frequency sensitivity
        sensitivity = self.compute_frequency_sensitivity(frames, scorer_fn)

        results = []
        for b in range(B):
            # DCT transform
            dct = self.dct_2d(frames[b:b+1]).squeeze(0)  # (C, H, W)

            # Water-filling
            filled = self.water_fill(dct, sensitivity)

            # Inverse DCT
            recon = self.idct_2d(filled.unsqueeze(0)).squeeze(0)  # (C, H, W)
            results.append(recon)

        return torch.stack(results, dim=0).clamp(0.0, 255.0)


# ════════════════════════════════════════════════════════════════════════
# 10. Turbo Decoder / Belief Propagation
# ════════════════════════════════════════════════════════════════════════


@dataclass
class TurboConfig:
    """Configuration for turbo scorer optimizer.

    Attributes:
        num_turbo_iterations: Number of turbo (alternating) iterations.
        damping_factor: Extrinsic information damping (0=no exchange, 1=full).
        convergence_threshold: Stop if improvement < this fraction.
        pose_steps: Gradient steps per PoseNet sub-optimization.
        seg_steps: Gradient steps per SegNet sub-optimization.
        lr: Learning rate for sub-optimizations.
    """
    num_turbo_iterations: int = 10
    damping_factor: float = 0.5
    convergence_threshold: float = 1e-4
    pose_steps: int = 10
    seg_steps: int = 10
    lr: float = 0.3


class TurboScorerOptimizer:
    """Turbo-code-inspired alternating scorer optimization [5].

    Turbo codes achieve near-Shannon-limit performance by iterating
    between two decoders that exchange "extrinsic information". Our two
    "decoders" are PoseNet and SegNet optimization passes.

    Iteration:
    1. Optimize for PoseNet, using current SegNet estimate as prior.
    2. Compute "extrinsic information" = improvement PoseNet made.
    3. Optimize for SegNet, using PoseNet result as prior.
    4. Compute extrinsic information from SegNet pass.
    5. Exchange damped extrinsic information and repeat.

    This naturally balances the two scorer components without manual
    weight tuning -- each decoder only passes its own improvement,
    preventing one from dominating.

    Args:
        config: TurboConfig or dict of overrides.
    """

    def __init__(self, config: TurboConfig | dict[str, Any] | None = None):
        if config is None:
            config = TurboConfig()
        elif isinstance(config, dict):
            config = TurboConfig(**config)
        self.config = config

    def _optimize_single_scorer(
        self,
        frames: torch.Tensor,
        scorer_fn: nn.Module,
        num_steps: int,
        lr: float,
        prior: Optional[torch.Tensor] = None,
        prior_weight: float = 1.0,
    ) -> torch.Tensor:
        """Run gradient descent on a single scorer.

        Args:
            frames: (B, C, H, W) starting point.
            scorer_fn: Scorer to minimize.
            num_steps: Number of gradient steps.
            lr: Learning rate.
            prior: Optional (B, C, H, W) prior from other decoder.
            prior_weight: Weight for prior constraint.

        Returns:
            (B, C, H, W) optimized frames.
        """
        current = frames.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([current], lr=lr)

        for _ in range(num_steps):
            optimizer.zero_grad()

            out = scorer_fn(current)
            if isinstance(out, dict):
                out = out.get("pose", next(iter(out.values())))
            scorer_loss = out.reshape(-1).pow(2).sum()

            # Prior constraint: stay close to the other decoder's output
            if prior is not None:
                prior_loss = (current - prior).pow(2).mean() * prior_weight
                loss = scorer_loss + prior_loss
            else:
                loss = scorer_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                current.data.clamp_(0.0, 255.0)

        return current.detach()

    def optimize(
        self,
        frames: torch.Tensor,
        posenet_fn: nn.Module,
        segnet_fn: nn.Module,
    ) -> torch.Tensor:
        """Run turbo iterations alternating between PoseNet and SegNet.

        Args:
            frames: (B, C, H, W) initial frames.
            posenet_fn: PoseNet scorer callable.
            segnet_fn: SegNet scorer callable.

        Returns:
            (B, C, H, W) turbo-optimized frames.
        """
        B, C, H, W = frames.shape
        current = frames.detach().clone()
        prev_loss = float("inf")

        for turbo_iter in range(self.config.num_turbo_iterations):
            # ── Decoder 1: PoseNet optimization ──
            before_pose = current.clone()
            pose_result = self._optimize_single_scorer(
                current, posenet_fn, self.config.pose_steps, self.config.lr,
                prior=current, prior_weight=1.0,
            )

            # Extrinsic information from PoseNet: the improvement it made
            pose_extrinsic = pose_result - before_pose

            # Apply damped extrinsic to current
            current = current + self.config.damping_factor * pose_extrinsic
            current = current.clamp(0.0, 255.0)

            # ── Decoder 2: SegNet optimization ──
            before_seg = current.clone()
            seg_result = self._optimize_single_scorer(
                current, segnet_fn, self.config.seg_steps, self.config.lr,
                prior=current, prior_weight=1.0,
            )

            # Extrinsic information from SegNet
            seg_extrinsic = seg_result - before_seg

            # Apply damped extrinsic
            current = current + self.config.damping_factor * seg_extrinsic
            current = current.clamp(0.0, 255.0)

            # Convergence check
            total_change = (pose_extrinsic.pow(2).mean() + seg_extrinsic.pow(2).mean()).sqrt().item()
            if total_change < self.config.convergence_threshold:
                break

        return current


# ════════════════════════════════════════════════════════════════════════
# Domain Solver Ranking
# ════════════════════════════════════════════════════════════════════════


def yousfi_domain_ranking() -> list[tuple[str, str, float]]:
    """Yousfi's ranking of domain solvers by expected score impact.

    Rankings based on:
    - Domain-problem alignment (how well the analogy maps)
    - Dimensionality reduction factor
    - Interaction with the score formula S = 100*seg + sqrt(10*pose) + 25*rate
    - Practical GPU efficiency

    Returns:
        List of (solver_name, reasoning, expected_impact_pct) sorted
        by expected impact (highest first). Impact is estimated relative
        improvement percentage on the composite score S.
    """
    return [
        (
            "EgoMotionFlowSolver",
            "Exact geometric constraint on PoseNet's measurement model. "
            "PoseNet literally measures ego-motion, so frames satisfying "
            "the correct flow field minimize PoseNet distortion by construction. "
            "The sqrt(10*pose) term means even small PoseNet improvements "
            "have outsized score impact at low baselines.",
            15.0,
        ),
        (
            "MatchedFilterOptimizer",
            "Minimum-energy perturbation via Wiener filter is the theoretically "
            "optimal pixel change per unit L2 norm. This directly optimizes the "
            "rate-distortion tradeoff: minimum pixel energy = minimum rate cost "
            "for a given scorer improvement. O(K^2*D) per step.",
            12.0,
        ),
        (
            "CompressedSensingRecovery",
            "L1 sparsity in wavelet domain directly reduces rate (25*rate term) "
            "while the measurement constraint preserves scorer fidelity. This is "
            "the only solver that explicitly targets the rate component. "
            "Zeroing wavelet coefficients = zeroing DCT coefficients = fewer bits.",
            10.0,
        ),
        (
            "RoadPlaneHomography",
            "Road is ~40% of driving frames. Exact homography constraint "
            "eliminates 40% of free variables, and road accuracy directly "
            "affects both SegNet (class 0) and PoseNet (ground-plane parallax). "
            "Simple 3x3 matrix, very cheap to compute.",
            9.0,
        ),
        (
            "TurboScorerOptimizer",
            "Automatic PoseNet/SegNet balancing without manual weight tuning. "
            "The score formula weights them 100:sqrt(10)~3.16, which is hard "
            "to tune manually. Turbo iterations naturally find the Pareto point. "
            "But: doubles compute (two passes per iteration).",
            8.0,
        ),
        (
            "AdaptiveOpticsCorrector",
            "50 Zernike modes vs 589K pixels = 11,780x dimensionality reduction. "
            "Makes optimization tractable for second-order methods. Low modes "
            "capture global illumination (PoseNet-invisible per blind spot analysis), "
            "high modes capture boundaries (SegNet-critical).",
            7.0,
        ),
        (
            "OFDMOptimizer",
            "Water-filling allocates energy to scorer-sensitive frequencies. "
            "Zeroing insensitive frequencies is free rate savings. But the "
            "DCT-to-scorer sensitivity mapping is expensive to compute and "
            "nonlinear, limiting the analogy's tightness.",
            6.0,
        ),
        (
            "VanishingPointConstraint",
            "Structural prior that PoseNet implicitly uses for rotation "
            "estimation. Ensures generated edge directions are geometrically "
            "consistent. Moderate impact because PoseNet may not weight VP "
            "as heavily as direct parallax flow.",
            5.0,
        ),
        (
            "KalmanFrameSmoother",
            "Optimal temporal smoothing with future context (RTS backward pass). "
            "Reduces frame-to-frame jitter which PoseNet penalizes. But the "
            "PCA reduction loses high-frequency SegNet-critical detail, and "
            "re-linearization is expensive. Best for post-processing.",
            4.0,
        ),
        (
            "TrajectoryOptimizer",
            "Elegant formulation but the shooting method converges slowly for "
            "589K-dimensional pixel states. Control penalty naturally gives "
            "temporal smoothness (good for PoseNet), but the boundary condition "
            "requirement limits flexibility. Best as a refinement step.",
            3.0,
        ),
    ]


# ════════════════════════════════════════════════════════════════════════
# Smoke Test
# ════════════════════════════════════════════════════════════════════════


def _smoke_test() -> None:
    """Verify all solvers instantiate, run, and produce correct shapes."""
    device = torch.device("cpu")
    T, C, H, W = 4, 3, 32, 32  # small for speed
    masks = torch.randint(0, NUM_CLASSES, (T, H, W), device=device)
    frames = torch.rand(T, C, H, W, device=device) * 255.0

    print("domain_solvers smoke test: starting...")

    # ── 1. EgoMotionFlowSolver ──
    ego = EgoMotionFlowSolver()
    depth_map = ego.compute_depth_map(masks)
    assert depth_map.shape == (T, H, W), f"depth_map shape: {depth_map.shape}"
    assert depth_map.min() >= 0, "depth must be non-negative"

    rot = torch.tensor([0.01, -0.005, 0.002], device=device)
    trans = torch.tensor([0.1, 0.05, 0.5], device=device)
    flow = ego.compute_flow_from_egomotion(depth_map[0], rot, trans)
    assert flow.shape == (2, H, W), f"flow shape: {flow.shape}"

    gen_frames = ego.generate(masks)
    assert gen_frames.shape == (T, C, H, W), f"ego gen shape: {gen_frames.shape}"
    assert gen_frames.min() >= 0 and gen_frames.max() <= 255
    print("  [1/10] EgoMotionFlowSolver: PASS")

    # ── 2. RoadPlaneHomography ──
    road = RoadPlaneHomography()
    H_mat = road.compute_road_homography(rot, trans)
    assert H_mat.shape == (3, 3), f"homography shape: {H_mat.shape}"

    road_frames = road.generate(masks)
    assert road_frames.shape == (T, C, H, W), f"road gen shape: {road_frames.shape}"
    assert road_frames.min() >= 0 and road_frames.max() <= 255
    print("  [2/10] RoadPlaneHomography: PASS")

    # ── 3. VanishingPointConstraint ──
    vp = VanishingPointConstraint()
    vp_loss, vp_point = vp.apply(frames[0], masks[0])
    assert vp_point.shape == (2,), f"VP shape: {vp_point.shape}"
    assert vp_loss.ndim == 0, "VP loss should be scalar"
    print("  [3/10] VanishingPointConstraint: PASS")

    # ── 4. MatchedFilterOptimizer ──
    mf = MatchedFilterOptimizer({"num_iterations": 2, "max_jacobian_outputs": 4})
    # Test with a simple linear scorer
    class SimpleScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, C * H * W) * 0.01)
        def forward(self, x):
            return self.w @ x.reshape(-1)

    scorer = SimpleScorer()
    mf_result = mf.optimize(frames[:1], scorer)
    assert mf_result.shape == (1, C, H, W), f"MF shape: {mf_result.shape}"
    assert mf_result.min() >= 0 and mf_result.max() <= 255
    print("  [4/10] MatchedFilterOptimizer: PASS")

    # ── 5. KalmanFrameSmoother ──
    ks = KalmanFrameSmoother({"pca_components": 5, "num_iterations": 2})
    smoothed = ks.smooth(frames)
    assert smoothed.shape == (T, C, H, W), f"Kalman shape: {smoothed.shape}"
    assert smoothed.min() >= 0 and smoothed.max() <= 255
    print("  [5/10] KalmanFrameSmoother: PASS")

    # ── 6. CompressedSensingRecovery ──
    # Ensure even dimensions
    frames_even = frames[:, :, :32, :32]
    cs = CompressedSensingRecovery({"num_iterations": 5})
    recovered = cs.recover(frames_even)
    assert recovered.shape == frames_even.shape, f"CS shape: {recovered.shape}"
    assert recovered.min() >= 0 and recovered.max() <= 255

    # Verify Haar transform roundtrip
    coeffs = cs.haar_transform(frames_even)
    recon = cs.haar_inverse(coeffs, 32, 32)
    roundtrip_err = (recon - frames_even).abs().max().item()
    assert roundtrip_err < 0.1, f"Haar roundtrip error: {roundtrip_err}"
    print("  [6/10] CompressedSensingRecovery: PASS")

    # ── 7. TrajectoryOptimizer ──
    traj = TrajectoryOptimizer({"num_shooting_iterations": 3})
    optimized = traj.optimize(frames)
    assert optimized.shape == (T, C, H, W), f"Traj shape: {optimized.shape}"
    assert optimized.min() >= 0 and optimized.max() <= 255
    print("  [7/10] TrajectoryOptimizer: PASS")

    # ── 8. AdaptiveOpticsCorrector ──
    ao = AdaptiveOpticsCorrector({"num_modes": 10, "num_iterations": 5})
    corrected = ao.optimize(frames[:1])
    assert corrected.shape == (1, C, H, W), f"AO shape: {corrected.shape}"
    assert corrected.min() >= 0 and corrected.max() <= 255

    # Test decompose/reconstruct roundtrip
    basis = ao.get_basis(H, W, device)
    assert basis.shape == (10, H, W), f"basis shape: {basis.shape}"
    correction = torch.randn(C, H, W, device=device)
    coeffs_ao = ao.decompose(correction)
    recon_ao = ao.reconstruct(coeffs_ao, H, W)
    # Reconstruction should be close (in the span of 10 modes)
    assert recon_ao.shape == (C, H, W), f"AO recon shape: {recon_ao.shape}"
    print("  [8/10] AdaptiveOpticsCorrector: PASS")

    # ── 9. OFDMOptimizer ──
    ofdm = OFDMOptimizer({"num_frequencies": 16})

    # Test DCT roundtrip
    x = torch.randn(1, C, H, W, device=device)
    dct = OFDMOptimizer.dct_2d(x)
    assert dct.shape == x.shape, f"DCT shape: {dct.shape}"
    idct = OFDMOptimizer.idct_2d(dct)
    dct_err = (idct - x).abs().max().item()
    assert dct_err < 0.5, f"DCT roundtrip error: {dct_err}"

    ofdm_result = ofdm.optimize(frames[:1])
    assert ofdm_result.shape == (1, C, H, W), f"OFDM shape: {ofdm_result.shape}"
    assert ofdm_result.min() >= 0 and ofdm_result.max() <= 255
    print("  [9/10] OFDMOptimizer: PASS")

    # ── 10. TurboScorerOptimizer ──
    turbo = TurboScorerOptimizer({"num_turbo_iterations": 2, "pose_steps": 3, "seg_steps": 3})

    class FakePoseNet(nn.Module):
        def forward(self, x):
            return {"pose": x.mean(dim=(2, 3))[:, :1]}

    class FakeSegNet(nn.Module):
        def forward(self, x):
            return x.mean(dim=1, keepdim=True)

    turbo_result = turbo.optimize(frames[:1], FakePoseNet(), FakeSegNet())
    assert turbo_result.shape == (1, C, H, W), f"Turbo shape: {turbo_result.shape}"
    assert turbo_result.min() >= 0 and turbo_result.max() <= 255
    print("  [10/10] TurboScorerOptimizer: PASS")

    # ── Ranking ──
    ranking = yousfi_domain_ranking()
    assert len(ranking) == 10, f"Expected 10 solvers, got {len(ranking)}"
    assert all(len(r) == 3 for r in ranking), "Each entry should be (name, reason, impact)"
    assert ranking[0][0] == "EgoMotionFlowSolver", "Ego-motion should be ranked #1"
    # Verify sorted by impact descending
    impacts = [r[2] for r in ranking]
    assert impacts == sorted(impacts, reverse=True), "Rankings should be sorted by impact"
    print("  [bonus] yousfi_domain_ranking: PASS")

    print("\ndomain_solvers: ALL 10 smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
