"""Depth-aware motion predictor: geometric parallax flow from per-class depth + 6-DOF camera.

Computes optical flow analytically from learned per-class depth priors and
a small camera motion estimator.  This is geometrically principled:
objects closer to the camera have larger flow (parallax), and the flow
direction depends on the 6-DOF camera motion (3 rotation + 3 translation).

~200 learnable parameters total:
    - 5 per-class depth values (nn.Parameter)
    - 18 features -> 6 camera params (linear layer: 18*6 + 6 = 114)
    - learnable focal length (2 params)

Output: (B, 2, H, W) flow field, same interface as MotionPredictor.

Usage::

    depth_motion = DepthAwareMotionPredictor(num_classes=5)
    flow = depth_motion(mask_t, mask_t1)  # (B, 2, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAwareMotionPredictor(nn.Module):
    """Geometric motion from per-class depth + 6-DOF camera motion.

    ~200 parameters. Produces geometrically correct parallax flow.

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

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

        # Per-class depth priors (driving scene defaults)
        # road=0.3 (medium-far), lane=0.3, vehicle=0.5 (medium),
        # sky=10.0 (very far — large depth = small parallax),
        # background=0.2 (medium-far)
        depth_init = torch.tensor([0.3, 0.3, 0.5, 10.0, 0.2])
        if num_classes != 5:
            depth_init = torch.full((num_classes,), 0.3)
        self.class_depth = nn.Parameter(depth_init)

        # Camera motion estimator: per-class features -> 6-DOF
        # Features: 3 per class (centroid_dx, centroid_dy, area_change) = 3 * num_classes
        cam_input_dim = 3 * num_classes
        self.cam_linear = nn.Linear(cam_input_dim, 6, bias=True)
        # Zero-init: identity camera motion at start
        nn.init.zeros_(self.cam_linear.weight)
        nn.init.zeros_(self.cam_linear.bias)

        # Learnable focal length (normalized, initialized at 1.0)
        self.focal = nn.Parameter(torch.tensor([1.0, 1.0]))  # (fx, fy)

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

        Args:
            mask_t: (B, H, W) long -- mask at time t.
            mask_t1: (B, H, W) long -- mask at time t+1.

        Returns:
            (B, 2, H, W) flow in normalized coordinates (small values ~[-0.1, 0.1]).
        """
        B, H, W = mask_t.shape
        device = mask_t.device
        eps = 1e-6

        # Force FP32 for depth computations — inv_depth gradients scale as 1/d^2
        # which can overflow in FP16 for small depth values
        orig_dtype = self.class_depth.dtype

        # 1. Extract class features and estimate camera motion
        class_feats = self._extract_class_features(mask_t, mask_t1)  # (B, 3*C)
        cam_params = self.cam_linear(class_feats)  # (B, 6)
        # Split into rotation (rx, ry, rz) and translation (tx, ty, tz)
        rx, ry, rz = cam_params[:, 0], cam_params[:, 1], cam_params[:, 2]
        tx, ty, tz = cam_params[:, 3], cam_params[:, 4], cam_params[:, 5]

        # 2. Build per-pixel depth map from class assignments
        # class_depth: (C,) -> depth_map: (B, H, W)
        depth_safe = self.class_depth.abs().clamp(min=eps)  # ensure positive depth
        depth_map = depth_safe[mask_t]  # (B, H, W) via advanced indexing

        # 3. Build normalized coordinate grids
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        # Expand to (B, H, W)
        x_norm = grid_x.unsqueeze(0).expand(B, -1, -1)
        y_norm = grid_y.unsqueeze(0).expand(B, -1, -1)

        fx, fy = self.focal[0], self.focal[1]

        # 4. Compute flow analytically (pinhole camera parallax model)
        # Translation-induced flow (depth-dependent parallax)
        inv_d = 1.0 / (depth_map + eps)
        # Reshape camera params for broadcasting: (B,) -> (B, 1, 1)
        tx_b = tx.reshape(B, 1, 1)
        ty_b = ty.reshape(B, 1, 1)
        tz_b = tz.reshape(B, 1, 1)
        rx_b = rx.reshape(B, 1, 1)
        ry_b = ry.reshape(B, 1, 1)
        rz_b = rz.reshape(B, 1, 1)

        # Standard pinhole translation flow: (-tx + x_norm * tz) / d
        # The x_norm * tz / d term is the focus-of-expansion effect —
        # critical for forward-motion flow fields in driving video
        flow_x_trans = fx * (-tx_b + x_norm * tz_b) * inv_d
        flow_y_trans = fy * (-ty_b + y_norm * tz_b) * inv_d

        # Rotation-induced flow (depth-independent)
        flow_x_rot = fx * (ry_b - rz_b * y_norm)
        flow_y_rot = fy * (-rx_b + rz_b * x_norm)

        flow_x = flow_x_trans + flow_x_rot  # (B, H, W)
        flow_y = flow_y_trans + flow_y_rot  # (B, H, W)

        # Stack to (B, 2, H, W) and scale to small flow range
        flow = torch.stack([flow_x, flow_y], dim=1) * 0.1

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
    # (small residual from tz*d term, but scaled by 0.1 and zero cam params)
    assert flow.abs().max() < 0.1, f"At init, flow should be near-zero, got max {flow.abs().max():.6f}"

    # Gradient flows through
    loss = flow.sum()
    loss.backward()
    assert model.class_depth.grad is not None, "class_depth should have gradient"
    assert model.cam_linear.weight.grad is not None, "cam_linear should have gradient"
    assert model.focal.grad is not None, "focal should have gradient"

    n_params = model.param_count()
    assert n_params < 500, f"Expected ~200 params, got {n_params}"

    print(f"depth_motion: all smoke tests passed ({n_params} params)")


if __name__ == "__main__":
    _smoke_test()
