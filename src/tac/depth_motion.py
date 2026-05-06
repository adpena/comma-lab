"""Depth-aware motion predictor: geometric parallax flow from per-class depth + 6-DOF camera.

Computes optical flow analytically from learned per-class depth priors and
a small camera motion estimator.  This is geometrically principled:
objects closer to the camera have larger flow (parallax), and the flow
direction depends on the 6-DOF camera motion (3 rotation + 3 translation).

All units follow tac.camera conventions:
    - Depth: meters (z-forward from camera)
    - Focal length: pixels
    - Principal point: pixels from top-left
    - Flow: pixels per frame

~120 learnable parameters total:
    - 5 per-class depth values (nn.Parameter, in meters)
    - 15 features -> 6 camera params (linear layer: 15*6 + 6 = 96)
    - learnable focal length (2 params, in pixels)
    - learnable principal point (2 params, in pixels)
    - learnable camera height (1 param, in meters)

Output: (B, 2, H, W) flow field, same interface as MotionPredictor.

Usage::

    depth_motion = DepthAwareMotionPredictor(num_classes=5)
    flow = depth_motion(mask_t, mask_t1)  # (B, 2, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tac.camera import COMMA_INTRINSICS, COMMA_EXTRINSICS, DEPTH_PRIORS_METERS


class DepthAwareMotionPredictor(nn.Module):
    """Geometric motion from per-class depth + 6-DOF camera motion.

    ~120 parameters. Produces geometrically correct parallax flow.

    The model learns:
        1. Per-class depth (5 values, initialized from driving priors).
        2. Camera motion estimator: extracts class centroid displacement and
           area change features from mask pairs -> linear -> 6-DOF params.
        3. Focal length (learnable, initialized at 1.0).

    Flow computation (pinhole camera model):
        For each pixel (x, y) with depth d, camera rotation R (3 angles)
        and translation t (3 components):
            flow_x = fx * (tz * d - tx) / (d + eps) + fx * (ry - rz * y_norm)
            flow_y = fy * (tz * d - ty) / (d + eps) + fy * (-rx + rz * x_norm)

    Args:
        num_classes: number of segmentation classes (5 for comma SegNet).
    """

    def __init__(
        self,
        num_classes: int = 5,
        depth_priors: dict[int, float] | None = None,
        focal_length: tuple[float, float] | None = None,
        principal_point: tuple[float, float] | None = None,
        camera_height: float | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Per-class depth priors in METERS. Class IDs follow
        # tac.semantic_label_contract; values are legacy geometry seeds and
        # must not be interpreted as label provenance.
        priors = depth_priors or DEPTH_PRIORS_METERS
        if num_classes == 5:
            depth_init = torch.tensor([priors.get(i, 20.0) for i in range(5)])
        else:
            depth_init = torch.full((num_classes,), 20.0)
        self.class_depth = nn.Parameter(depth_init)

        # Camera motion estimator: per-class features -> 6-DOF
        # Features: 3 per class (centroid_dx, centroid_dy, area_change) = 3 * num_classes
        cam_input_dim = 3 * num_classes
        self.cam_linear = nn.Linear(cam_input_dim, 6, bias=True)
        # Zero-init: identity camera motion at start
        nn.init.zeros_(self.cam_linear.weight)
        nn.init.zeros_(self.cam_linear.bias)

        # Learnable focal length in PIXELS (initialized from comma intrinsics)
        fl = focal_length or (COMMA_INTRINSICS.fx, COMMA_INTRINSICS.fy)
        self.focal = nn.Parameter(torch.tensor([fl[0], fl[1]]))  # (fx, fy) pixels

        # Learnable principal point in PIXELS (initialized from comma intrinsics)
        pp = principal_point or (COMMA_INTRINSICS.cx, COMMA_INTRINSICS.cy)
        self.principal_point = nn.Parameter(torch.tensor([pp[0], pp[1]]))  # (cx, cy) pixels

        # Camera height in METERS (configurable, not learned here since depth
        # priors are already per-class learnable parameters)
        self.camera_height: float = camera_height if camera_height is not None else COMMA_EXTRINSICS.height

    def _extract_class_features(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-class centroid displacement and area change.

        Args:
            mask_t: (B, H, W) long tensor at time t.
            mask_t1: (B, H, W) long tensor at time t+1.

        Returns:
            (B, 3*num_classes) feature vector: [dx_0, dy_0, da_0, dx_1, ...].
        """
        B, H, W = mask_t.shape
        device = mask_t.device

        # Coordinate grids normalized to [-1, 1]
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        # Vectorized centroid + area computation via one-hot
        total_pixels = float(H * W)

        features = []
        for mask in (mask_t, mask_t1):
            one_hot = F.one_hot(mask, self.num_classes).float()  # (B, H, W, C)
            count = one_hot.sum(dim=(1, 2)).clamp(min=1.0)  # (B, C)
            cx = (grid_x.unsqueeze(0).unsqueeze(-1) * one_hot).sum(dim=(1, 2)) / count
            cy = (grid_y.unsqueeze(0).unsqueeze(-1) * one_hot).sum(dim=(1, 2)) / count
            area = count / total_pixels  # (B, C), normalized area
            features.append((cx, cy, area))

        cx_t, cy_t, area_t = features[0]
        cx_t1, cy_t1, area_t1 = features[1]

        # Per-class features: centroid displacement + area change
        dx = cx_t1 - cx_t  # (B, C)
        dy = cy_t1 - cy_t  # (B, C)
        da = area_t1 - area_t  # (B, C)

        # Interleave: [dx_0, dy_0, da_0, dx_1, dy_1, da_1, ...]
        out = torch.stack([dx, dy, da], dim=-1)  # (B, C, 3)
        return out.reshape(B, -1)  # (B, 3*C)

    def forward(
        self,
        mask_t: torch.Tensor,
        mask_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geometric flow from mask pair via depth + camera model.

        Uses the standard pinhole camera flow equations with:
        - Depth in meters (from learnable class_depth)
        - Focal length in pixels (from learnable focal)
        - Principal point in pixels (from learnable principal_point)
        - Flow output in pixels/frame

        Args:
            mask_t: (B, H, W) long -- mask at time t.
            mask_t1: (B, H, W) long -- mask at time t+1.

        Returns:
            (B, 2, H, W) flow in pixels/frame.
        """
        B, H, W = mask_t.shape
        device = mask_t.device
        eps = 1e-6

        # 1. Extract class features and estimate camera motion
        class_feats = self._extract_class_features(mask_t, mask_t1)  # (B, 3*C)
        cam_params = self.cam_linear(class_feats)  # (B, 6)
        # Split into rotation (rx, ry, rz) radians and translation (tx, ty, tz) meters
        rx, ry, rz = cam_params[:, 0], cam_params[:, 1], cam_params[:, 2]
        tx, ty, tz = cam_params[:, 3], cam_params[:, 4], cam_params[:, 5]

        # 2. Build per-pixel depth map (meters) from class assignments
        # class_depth: (C,) meters -> depth_map: (B, H, W) meters
        depth_safe = self.class_depth.abs().clamp(min=eps)  # ensure positive depth
        depth_map = depth_safe[mask_t]  # (B, H, W) via advanced indexing

        # 3. Build pixel coordinate grids, normalized by focal length
        # u' = (u - cx) / fx, v' = (v - cy) / fy  (dimensionless)
        fx, fy = self.focal[0], self.focal[1]
        cx_pp, cy_pp = self.principal_point[0], self.principal_point[1]

        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

        u_norm = (u_grid - cx_pp) / (fx + eps)  # (H, W), dimensionless
        v_norm = (v_grid - cy_pp) / (fy + eps)  # (H, W), dimensionless

        # Expand to (B, H, W)
        u_norm = u_norm.unsqueeze(0).expand(B, -1, -1)
        v_norm = v_norm.unsqueeze(0).expand(B, -1, -1)

        # 4. Compute flow analytically (pinhole camera parallax model)
        # All flow values in pixels/frame
        inv_d = 1.0 / (depth_map + eps)  # 1/meters
        # Reshape camera params for broadcasting: (B,) -> (B, 1, 1)
        tx_b = tx.reshape(B, 1, 1)
        ty_b = ty.reshape(B, 1, 1)
        tz_b = tz.reshape(B, 1, 1)
        rx_b = rx.reshape(B, 1, 1)
        ry_b = ry.reshape(B, 1, 1)
        rz_b = rz.reshape(B, 1, 1)

        # Translation-induced flow (depth-dependent parallax) [pixels/frame]
        # flow_u = fx * (-tx/d + u' * tz/d)
        # flow_v = fy * (-ty/d + v' * tz/d)
        flow_x_trans = fx * (-tx_b + u_norm * tz_b) * inv_d
        flow_y_trans = fy * (-ty_b + v_norm * tz_b) * inv_d

        # Rotation-induced flow (depth-independent) [pixels/frame]
        # flow_u = fx * (wy - wz * v')
        # flow_v = fy * (-wx + wz * u')
        flow_x_rot = fx * (ry_b - rz_b * v_norm)
        flow_y_rot = fy * (-rx_b + rz_b * u_norm)

        flow_x = flow_x_trans + flow_x_rot  # (B, H, W), pixels/frame
        flow_y = flow_y_trans + flow_y_rot  # (B, H, W), pixels/frame

        # Stack to (B, 2, H, W) — flow is already in pixels/frame
        flow = torch.stack([flow_x, flow_y], dim=1)

        return flow

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Smoke test ────────────────────────────────────────────────────────


def _smoke_test() -> None:
    """Verify shapes, zero-init, and gradient flow."""
    B, H, W = 2, 32, 32
    num_classes = 5

    model = DepthAwareMotionPredictor(num_classes=num_classes)

    mask_t = torch.randint(0, num_classes, (B, H, W))
    mask_t1 = torch.randint(0, num_classes, (B, H, W))

    flow = model(mask_t, mask_t1)
    assert flow.shape == (B, 2, H, W), f"Expected (B, 2, H, W), got {flow.shape}"

    # At init, cam_linear is zero -> camera motion is zero -> flow is ~zero
    assert flow.abs().max() < 1.0, f"At init, flow should be near-zero, got max {flow.abs().max():.6f}"

    # Gradient flows through
    loss = flow.sum()
    loss.backward()
    assert model.class_depth.grad is not None, "class_depth should have gradient"
    assert model.cam_linear.weight.grad is not None, "cam_linear should have gradient"
    assert model.focal.grad is not None, "focal should have gradient"
    assert model.principal_point.grad is not None, "principal_point should have gradient"

    n_params = model.param_count()
    assert n_params < 500, f"Expected ~125 params, got {n_params}"

    print(f"depth_motion: all smoke tests passed ({n_params} params)")


if __name__ == "__main__":
    _smoke_test()
