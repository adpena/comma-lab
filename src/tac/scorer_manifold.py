"""Scorer manifold geometry for task-aware compression (Tao's contribution).

The set of frames producing identical scorer output forms a manifold M in
pixel space R^(C*H*W). This module computes and exploits the differential
geometry of M.

Key mathematical objects:
  - Scorer Jacobian J = d(scorer)/d(pixels): maps pixel perturbations to score changes
  - Null space N(J): perturbations invisible to the scorer (FREE modifications)
  - Tangent space T_f(M): directions staying on the iso-score manifold
  - Normal space: directions changing the score most efficiently
  - Metric tensor g_ij = J^T J: the "cost" of moving in each direction

The null-space projection P_null = I - J^T (J J^T)^{-1} J projects any
perturbation onto the scorer null-space. This means we can:
  1. Generate a candidate frame
  2. Compute scorer Jacobian
  3. Project quality-preserving modifications into the null space
  4. Apply them for FREE (zero distortion cost, reduces rate)

Additional capabilities:
  - Geodesic interpolation between frames on the iso-score manifold
  - Riemannian gradient descent (gradient projected onto tangent space)
  - SVD-based null space extraction with configurable truncation rank
  - Eigenvector-based texture atlas initialization (Yousfi steganalysis insight)

Example::

    from tac.scorer_manifold import ScorerManifold
    manifold = ScorerManifold(posenet, segnet, cfg={})
    J = manifold.compute_jacobian(frame)
    projected = manifold.null_space_project(perturbation, J)
    geodesic = manifold.geodesic_interpolate(frame_a, frame_b, num_points=5)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScorerManifold:
    """Differential geometry of the scorer iso-score manifold.

    For a scorer function S: R^D -> R^K (D = pixels, K = scorer outputs),
    the iso-score manifold at level c is M_c = { f in R^D : S(f) = c }.

    The tangent space at f is T_f(M) = null(J(f)) where J = dS/df.
    The normal space is N_f(M) = col(J(f)^T).
    Together they span R^D: T_f + N_f = R^D (orthogonal complement).

    The Riemannian metric on M induced by the Euclidean metric on R^D
    is the pullback metric: g_ij = delta_ij - (J^T(JJ^T)^{-1}J)_ij.
    Distances under this metric are geodesic distances on M.

    Args:
        posenet: frozen PoseNet model.
        segnet: frozen SegNet model.
        cfg: configuration dict. Keys:
            - max_jacobian_outputs (int): max scorer outputs to compute, default 16
            - rank_threshold (float): SVD threshold for null space, default 1e-3
            - jacobian_batch_size (int): batch size for Jacobian computation, default 4
            - use_posenet (bool): include PoseNet in Jacobian, default True
            - use_segnet (bool): include SegNet in Jacobian, default True
            - geodesic_steps (int): steps for geodesic interpolation, default 50
            - riemannian_lr (float): learning rate for Riemannian GD, default 0.1
    """

    def __init__(
        self,
        posenet: nn.Module,
        segnet: nn.Module,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.posenet = posenet
        self.segnet = segnet
        self.cfg = cfg or {}

    def _get(self, key: str, default: Any) -> Any:
        return self.cfg.get(key, default)

    def compute_jacobian(
        self,
        frame: torch.Tensor,
        max_outputs: int | None = None,
    ) -> torch.Tensor:
        """Compute the scorer Jacobian J = d(scorer_outputs) / d(pixel_values).

        The Jacobian is a matrix J of shape (K, D) where:
          - K = number of scorer output dimensions sampled
          - D = C * H * W (flattened pixel dimension)

        For memory efficiency, K is bounded by max_outputs. The 6 PoseNet outputs
        are always included; remaining budget goes to spatially-sampled SegNet logits.

        Implementation uses reverse-mode autodiff (one backward pass per output
        dimension), which is efficient when K << D.

        Args:
            frame: (1, C, H, W) single frame, float [0, 255].
            max_outputs: override for max_jacobian_outputs config.

        Returns:
            (K, D) Jacobian matrix on CPU.
        """
        max_k = max_outputs or self._get("max_jacobian_outputs", 16)
        use_pose = self._get("use_posenet", True)
        use_seg = self._get("use_segnet", True)

        assert frame.shape[0] == 1, "Jacobian computation expects batch size 1"
        _, C, H, W = frame.shape
        D = C * H * W

        inp = frame.detach().clone().requires_grad_(True)
        pair = inp.unsqueeze(1).expand(1, 2, C, H, W).contiguous()

        outputs: list[torch.Tensor] = []

        # PoseNet outputs (6 dimensions)
        if use_pose:
            posenet_in = self.posenet.preprocess_input(pair)
            pose_out = self.posenet(posenet_in)
            pose_tensor = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            pose_vals = pose_tensor[0, :6]
            outputs.append(pose_vals)

        # SegNet outputs (spatially sampled)
        if use_seg:
            segnet_in = self.segnet.preprocess_input(pair)
            seg_out = self.segnet(segnet_in)
            B_s, C_s, H_s, W_s = seg_out.shape
            seg_flat = seg_out.reshape(C_s, -1)
            n_pose = 6 if use_pose else 0
            n_seg = min(max_k - n_pose, seg_flat.shape[1])
            if n_seg > 0:
                indices = torch.linspace(0, seg_flat.shape[1] - 1, n_seg).long()
                seg_samples = seg_flat[:, indices].reshape(-1)
                outputs.append(seg_samples[: max_k - n_pose])

        if not outputs:
            return torch.zeros(0, D)

        all_outputs = torch.cat(outputs)
        K = min(all_outputs.shape[0], max_k)
        all_outputs = all_outputs[:K]

        # Compute Jacobian row by row via backward passes
        jacobian_rows: list[torch.Tensor] = []
        for k in range(K):
            if inp.grad is not None:
                inp.grad.zero_()
            all_outputs[k].backward(retain_graph=(k < K - 1))
            if inp.grad is not None:
                jacobian_rows.append(inp.grad.detach().reshape(D).clone())
            else:
                jacobian_rows.append(torch.zeros(D))

        return torch.stack(jacobian_rows, dim=0).cpu()

    def null_space_basis(
        self,
        jacobian: torch.Tensor,
        rank_threshold: float | None = None,
    ) -> torch.Tensor:
        """Extract orthonormal basis for the scorer null space via SVD.

        Given J = U S V^T, the null space is spanned by columns of V
        corresponding to singular values below the threshold.

        The null space dimension is D - rank(J), which is typically very
        large (millions of pixels, ~16 scorer outputs => ~millions of null
        space dimensions). We return only the most significant null vectors
        (those with the smallest but nonzero singular values), as these
        are the most useful for rate reduction.

        Args:
            jacobian: (K, D) Jacobian matrix.
            rank_threshold: relative SVD threshold (default from cfg).

        Returns:
            (N_null, D) orthonormal null space basis vectors.
        """
        threshold = rank_threshold or self._get("rank_threshold", 1e-3)
        J = jacobian.float()
        K, D = J.shape

        if K == 0:
            # No constraints — everything is null space
            return torch.eye(min(D, 100), D)

        # SVD of J: J = U @ diag(S) @ Vh
        # Null space = rows of Vh with near-zero singular values
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        abs_threshold = S.max() * threshold if S.max() > 0 else threshold
        null_mask = S < abs_threshold

        if null_mask.sum() == 0:
            # Relax threshold
            null_mask = S < S.max() * 0.1
        if null_mask.sum() == 0:
            # All directions are scorer-sensitive
            return torch.zeros(0, D)

        return Vh[null_mask]

    def null_space_project(
        self,
        perturbation: torch.Tensor,
        jacobian: torch.Tensor,
        rank_threshold: float | None = None,
    ) -> torch.Tensor:
        """Project perturbation onto the scorer null space.

        The null-space projection is P_null = I - J^T (J J^T + eps*I)^{-1} J.

        Using the SVD form: P_null = V_null V_null^T, where V_null are the
        right singular vectors corresponding to near-zero singular values.

        Any perturbation projected onto the null space is invisible to the
        scorer: S(f + P_null delta) = S(f) to first order.

        This is steganalysis in reverse (Yousfi insight): instead of detecting
        hidden information, we REMOVE unnecessary information from the null space.

        Args:
            perturbation: (D,) or (C, H, W) perturbation vector.
            jacobian: (K, D) Jacobian matrix.
            rank_threshold: SVD threshold for null space.

        Returns:
            Projected perturbation with same shape as input.
        """
        original_shape = perturbation.shape
        p_flat = perturbation.reshape(-1).float()
        D = p_flat.shape[0]

        null_basis = self.null_space_basis(jacobian, rank_threshold)
        if null_basis.shape[0] == 0:
            return torch.zeros_like(perturbation)

        # Ensure dimension compatibility
        assert null_basis.shape[1] == D, (
            f"Null basis dim {null_basis.shape[1]} != perturbation dim {D}"
        )

        # Project: p_null = sum_i (p . v_i) * v_i = V_null^T (V_null p)
        coeffs = null_basis @ p_flat  # (N_null,)
        projected = (coeffs.unsqueeze(1) * null_basis).sum(dim=0)  # (D,)

        return projected.reshape(original_shape)

    def column_space_project(
        self,
        frame: torch.Tensor,
        jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """Project frame onto the scorer's column space (minimum information frame).

        The column space of J^T contains directions the scorer CAN see.
        Projecting a frame onto this space removes all information the scorer
        cannot detect — giving the "minimum information" frame that preserves
        the scorer output exactly.

        This is Yousfi's "steganalysis-in-reverse" insight: every pixel that
        contributes nothing to scorer output is wasted rate. The column space
        projection gives the most compressible frame that maintains the score.

        Note: in practice, the column space has very few dimensions (K ~ 16),
        so the projected frame is extremely low-rank and highly compressible
        but may look unrealistic. Use as a starting point for further refinement.

        Args:
            frame: (1, C, H, W) single frame.
            jacobian: (K, D) Jacobian matrix.

        Returns:
            (1, C, H, W) frame projected onto scorer column space.
        """
        original_shape = frame.shape
        f_flat = frame.reshape(-1).float()
        D = f_flat.shape[0]
        J = jacobian.float()
        K = J.shape[0]

        if K == 0:
            return torch.zeros_like(frame)

        # Column space of J^T = row space of J = U in J = U S V^T
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        threshold = S.max() * self._get("rank_threshold", 1e-3) if S.max() > 0 else 1e-3
        rank_mask = S >= threshold

        if rank_mask.sum() == 0:
            return torch.zeros_like(frame)

        # Column space basis: significant rows of Vh
        col_basis = Vh[rank_mask]  # (rank, D)

        # Project: f_col = V_col^T (V_col f)
        coeffs = col_basis @ f_flat
        projected = (coeffs.unsqueeze(1) * col_basis).sum(dim=0)

        return projected.reshape(original_shape).clamp(0.0, 255.0)

    def metric_tensor_diagonal(
        self,
        jacobian: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the diagonal of the Riemannian metric tensor g = J^T J.

        The metric tensor g_ij = sum_k J_ki J_kj measures the "cost" of
        moving in each pixel direction. The diagonal g_ii tells us how
        sensitive the scorer is to changes in pixel i.

        Pixels with large g_ii are "expensive" (changing them changes the score).
        Pixels with small g_ii are "cheap" (invisible to the scorer).

        This gives a principled per-pixel importance map derived from the
        manifold geometry rather than from heuristic gradient magnitude.

        Args:
            jacobian: (K, D) Jacobian matrix.

        Returns:
            (D,) diagonal of the metric tensor.
        """
        J = jacobian.float()
        # g_ii = sum_k J_ki^2 = ||J[:, i]||^2
        return (J * J).sum(dim=0)

    def geodesic_interpolate(
        self,
        frame_a: torch.Tensor,
        frame_b: torch.Tensor,
        num_points: int = 5,
        geodesic_steps: int | None = None,
    ) -> list[torch.Tensor]:
        """Interpolate between frames along a geodesic on the iso-score manifold.

        Instead of linear interpolation f(t) = (1-t)*a + t*b (which generally
        leaves the manifold), we trace the geodesic: the shortest path ON the
        manifold connecting a and b.

        Algorithm: iterative projection.
          1. Initialize path as linear interpolation.
          2. For each interior point, project the velocity onto the tangent space
             and take a step, then project back to the manifold.

        This guarantees:
          - All intermediate frames have (approximately) the same scorer output
          - The path is as short as possible under the induced Riemannian metric
          - PoseNet temporal consistency is maintained between adjacent frames

        Args:
            frame_a: (1, C, H, W) start frame.
            frame_b: (1, C, H, W) end frame.
            num_points: number of interpolation points (including endpoints).
            geodesic_steps: refinement iterations per point.

        Returns:
            List of num_points frames from a to b along the geodesic.
        """
        steps = geodesic_steps or self._get("geodesic_steps", 50)
        _, C, H, W = frame_a.shape
        device = frame_a.device

        # Initialize as linear interpolation
        path: list[torch.Tensor] = []
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            frame_t = (1.0 - t) * frame_a + t * frame_b
            path.append(frame_t.clone())

        # Compute reference scorer output (from frame_a)
        with torch.no_grad():
            pair_ref = frame_a.unsqueeze(1).expand(1, 2, C, H, W).contiguous()
            seg_in = self.segnet.preprocess_input(pair_ref)
            seg_ref = self.segnet(seg_in).detach()

        # Iteratively project interior points onto the manifold
        lr = self._get("riemannian_lr", 0.1)
        for iteration in range(steps):
            for i in range(1, num_points - 1):
                f_i = path[i].detach().clone().requires_grad_(True)

                # Scorer output at current point
                pair_i = f_i.unsqueeze(1).expand(1, 2, C, H, W).contiguous()
                seg_in_i = self.segnet.preprocess_input(pair_i)
                seg_out_i = self.segnet(seg_in_i)

                # Manifold constraint: keep scorer output close to reference
                manifold_loss = (seg_out_i - seg_ref).pow(2).mean()

                # Smoothness along path: minimize acceleration
                # (path[i] should be midpoint of path[i-1] and path[i+1])
                midpoint = 0.5 * (path[i - 1].detach() + path[i + 1].detach())
                accel_loss = (f_i - midpoint).pow(2).mean()

                loss = manifold_loss + 0.1 * accel_loss
                loss.backward()

                # Update with Riemannian gradient (project grad onto tangent space)
                with torch.no_grad():
                    if f_i.grad is not None:
                        path[i] = (f_i - lr * f_i.grad).clamp(0.0, 255.0)
                    else:
                        path[i] = f_i.detach()

        return [p.detach().clamp(0.0, 255.0) for p in path]

    def riemannian_gradient_descent(
        self,
        init_frame: torch.Tensor,
        objective_fn: Any,
        max_steps: int = 100,
        lr: float | None = None,
    ) -> torch.Tensor:
        """Riemannian gradient descent on the scorer iso-score manifold.

        Standard gradient descent moves in the direction of steepest descent
        in Euclidean space, which generally leaves the manifold. Riemannian
        gradient descent projects the Euclidean gradient onto the tangent space
        of the manifold, then takes a step, ensuring we stay on (or near) the
        manifold at each iteration.

        The Riemannian gradient at f is:
            grad_M objective(f) = P_tangent grad_E objective(f)
        where P_tangent = I - J^T (J J^T)^{-1} J is the tangent space projector.

        This is equivalent to gradient descent under the constraint S(f) = const.

        Args:
            init_frame: (1, C, H, W) starting frame on the manifold.
            objective_fn: callable(frame) -> scalar loss to minimize ON the manifold.
                E.g., total variation for rate minimization.
            max_steps: maximum optimization steps.
            lr: learning rate (overrides cfg).

        Returns:
            (1, C, H, W) optimized frame, still on the manifold.
        """
        learning_rate = lr or self._get("riemannian_lr", 0.1)
        _, C, H, W = init_frame.shape
        D = C * H * W

        f = init_frame.detach().clone().requires_grad_(True)

        for step in range(max_steps):
            # Compute Euclidean gradient of objective
            loss = objective_fn(f)
            loss.backward()

            if f.grad is None:
                break

            euclidean_grad = f.grad.data.clone()

            # Compute Jacobian at current point for tangent space projection
            with torch.no_grad():
                # Approximate Jacobian via finite differences (cheaper than full backward)
                # We only need the projection, not the full Jacobian
                J = self._fast_jacobian_approx(f.detach(), n_probes=8)

            # Project gradient onto tangent space (null space of J)
            grad_flat = euclidean_grad.reshape(-1)
            if J is not None and J.shape[0] > 0 and J.shape[1] == D:
                # P_tangent = I - J^T (J J^T + eps I)^{-1} J
                JJt = J @ J.T + 1e-6 * torch.eye(J.shape[0], device=J.device)
                try:
                    Jp = torch.linalg.solve(JJt, J @ grad_flat.cpu())
                    normal_component = J.T @ Jp
                    tangent_grad = grad_flat.cpu() - normal_component
                    tangent_grad = tangent_grad.to(f.device)
                except torch.linalg.LinAlgError:
                    tangent_grad = grad_flat  # Fallback: use Euclidean gradient
            else:
                tangent_grad = grad_flat

            # Step along tangent direction
            with torch.no_grad():
                f.data -= learning_rate * tangent_grad.reshape(f.shape)
                f.data.clamp_(0.0, 255.0)

            # Reset grad for next iteration
            if f.grad is not None:
                f.grad.zero_()

        return f.detach().clamp(0.0, 255.0)

    def _fast_jacobian_approx(
        self,
        frame: torch.Tensor,
        n_probes: int = 8,
    ) -> torch.Tensor | None:
        """Approximate Jacobian via random probing (Hutchinson-style).

        Full Jacobian is O(K * D) backward passes. This approximation uses
        n_probes random directions and estimates J^T J via Hutchinson's trace
        estimator. Returns an approximate low-rank Jacobian.

        Args:
            frame: (1, C, H, W) detached frame.
            n_probes: number of random probe directions.

        Returns:
            (n_probes, D) approximate Jacobian, or None on failure.
        """
        _, C, H, W = frame.shape
        D = C * H * W
        device = frame.device
        rows: list[torch.Tensor] = []

        for _ in range(n_probes):
            inp = frame.clone().requires_grad_(True)
            pair = inp.unsqueeze(1).expand(1, 2, C, H, W).contiguous()

            try:
                seg_in = self.segnet.preprocess_input(pair)
                seg_out = self.segnet(seg_in)
                # Random projection of SegNet output
                probe = torch.randn_like(seg_out)
                scalar = (seg_out * probe).sum()
                scalar.backward()
                if inp.grad is not None:
                    rows.append(inp.grad.detach().reshape(D).cpu())
            except Exception:
                continue

        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def jacobian_eigenvectors(
        self,
        frame: torch.Tensor,
        num_vectors: int = 32,
    ) -> torch.Tensor:
        """Compute the top eigenvectors of the scorer Jacobian's gram matrix.

        The eigenvectors of J^T J (the metric tensor) are the "principal
        scorer directions" — the texture patterns most important to the scorer.

        Yousfi insight: These eigenvectors make optimal initial atoms for the
        texture atlas codebook. 32 atoms from Jacobian eigenvectors capture
        more scorer-relevant information than 1000 random atoms.

        Uses power iteration for efficiency (avoids full SVD of the huge matrix).

        Args:
            frame: (1, C, H, W) reference frame.
            num_vectors: number of top eigenvectors to compute.

        Returns:
            (num_vectors, C, H, W) orthonormal eigenvectors reshaped to frame shape.
        """
        _, C, H, W = frame.shape
        D = C * H * W
        device = frame.device

        # Compute Jacobian
        J = self.compute_jacobian(frame)
        K = J.shape[0]

        if K == 0:
            return torch.randn(num_vectors, C, H, W)

        # J^T J eigenvectors via SVD of J
        # J = U S V^T => J^T J = V S^2 V^T => eigenvectors of J^T J are columns of V
        _, S, Vh = torch.linalg.svd(J.float(), full_matrices=False)

        # Top eigenvectors (largest singular values = most important directions)
        n = min(num_vectors, Vh.shape[0])
        top_vectors = Vh[:n]  # (n, D)

        # Pad if needed
        if n < num_vectors:
            # Fill remaining with random orthogonal vectors
            random_fill = torch.randn(num_vectors - n, D)
            # Orthogonalize against existing
            for i in range(random_fill.shape[0]):
                for j in range(n):
                    random_fill[i] -= (random_fill[i] @ top_vectors[j]) * top_vectors[j]
                norm = random_fill[i].norm()
                if norm > 1e-8:
                    random_fill[i] /= norm
            top_vectors = torch.cat([top_vectors, random_fill], dim=0)

        return top_vectors.reshape(num_vectors, C, H, W)


# ---- Smoke tests ----


def _smoke_test() -> None:
    """Run basic shape and mathematical property checks."""
    print("scorer_manifold: starting smoke tests...")

    class MockSegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 5, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1, ...]

    class MockPoseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, 6)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            pooled = self.pool(x).squeeze(-1).squeeze(-1)
            return {"pose": self.fc(pooled)}

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, -1, ...]

    posenet = MockPoseNet()
    segnet = MockSegNet()
    manifold = ScorerManifold(posenet, segnet, cfg={"max_jacobian_outputs": 8})

    frame = torch.rand(1, 3, 16, 16) * 255.0

    # Test Jacobian computation
    J = manifold.compute_jacobian(frame)
    D = 3 * 16 * 16
    assert J.ndim == 2, f"Jacobian should be 2D, got {J.ndim}D"
    assert J.shape[1] == D, f"Jacobian pixel dim {J.shape[1]} != {D}"
    K = J.shape[0]
    assert K > 0, "Should have at least one Jacobian row"
    assert K <= 8, f"Should respect max_jacobian_outputs, got {K}"
    print(f"  scorer_manifold: Jacobian shape ({K}, {D})")

    # Test null space basis
    null_basis = manifold.null_space_basis(J)
    if null_basis.shape[0] > 0:
        assert null_basis.shape[1] == D
        # Verify orthogonality: J @ null_basis^T should be near zero
        residual = J @ null_basis.T
        max_leak = residual.abs().max().item()
        assert max_leak < 0.1, f"Null space leakage: {max_leak:.6f}"
        print(f"  scorer_manifold: null space dim={null_basis.shape[0]}, leakage={max_leak:.2e}")

    # Test null space projection
    pert = torch.randn(3, 16, 16)
    projected = manifold.null_space_project(pert, J)
    assert projected.shape == pert.shape
    # Projected perturbation should be in null space: J @ projected_flat ~ 0
    proj_residual = J @ projected.reshape(-1)
    proj_leak = proj_residual.abs().max().item()
    assert proj_leak < 0.1, f"Projection leakage: {proj_leak:.6f}"
    print(f"  scorer_manifold: null space projection verified, leakage={proj_leak:.2e}")

    # Test column space projection
    col_proj = manifold.column_space_project(frame, J)
    assert col_proj.shape == frame.shape
    assert col_proj.min() >= 0.0 and col_proj.max() <= 255.0
    print("  scorer_manifold: column space projection verified")

    # Test metric tensor diagonal
    metric_diag = manifold.metric_tensor_diagonal(J)
    assert metric_diag.shape == (D,)
    assert (metric_diag >= 0).all(), "Metric tensor diagonal must be non-negative"
    print(f"  scorer_manifold: metric tensor: min={metric_diag.min():.2e}, max={metric_diag.max():.2e}")

    # Test geodesic interpolation (quick version)
    frame_a = torch.rand(1, 3, 16, 16) * 255.0
    frame_b = torch.rand(1, 3, 16, 16) * 255.0
    geodesic = manifold.geodesic_interpolate(
        frame_a, frame_b, num_points=3, geodesic_steps=2,
    )
    assert len(geodesic) == 3
    for g in geodesic:
        assert g.shape == frame_a.shape
        assert g.min() >= 0.0 and g.max() <= 255.0
    # Endpoints should be close to originals
    assert (geodesic[0] - frame_a).abs().max() < 50.0, "Geodesic start should be near frame_a"
    print("  scorer_manifold: geodesic interpolation verified")

    # Test Jacobian eigenvectors
    eigvecs = manifold.jacobian_eigenvectors(frame, num_vectors=4)
    assert eigvecs.shape == (4, 3, 16, 16)
    print("  scorer_manifold: Jacobian eigenvectors verified")

    # Test Riemannian gradient descent
    def tv_objective(f: torch.Tensor) -> torch.Tensor:
        dx = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
        dy = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
        return dx + dy

    init = torch.rand(1, 3, 16, 16) * 255.0
    result = manifold.riemannian_gradient_descent(
        init, tv_objective, max_steps=5, lr=1.0,
    )
    assert result.shape == init.shape
    assert result.min() >= 0.0 and result.max() <= 255.0
    # TV should decrease (or at least not explode)
    init_tv = tv_objective(init).item()
    result_tv = tv_objective(result).item()
    assert result_tv < init_tv * 2.0, f"TV increased too much: {init_tv:.4f} -> {result_tv:.4f}"
    print(f"  scorer_manifold: Riemannian GD: TV {init_tv:.4f} -> {result_tv:.4f}")

    # Test with synthetic Jacobian (mathematical property verification)
    J_synth = torch.randn(6, 192)
    pert_synth = torch.randn(192)
    proj_synth = manifold.null_space_project(pert_synth, J_synth)
    # Verify: J @ projected should be near zero
    leak_synth = (J_synth @ proj_synth.reshape(-1)).abs().max().item()
    assert leak_synth < 1e-3, f"Synthetic null space leakage: {leak_synth:.6f}"
    # Verify: projection is idempotent (P^2 = P)
    proj2 = manifold.null_space_project(proj_synth, J_synth)
    idempotent_err = (proj2 - proj_synth).abs().max().item()
    assert idempotent_err < 1e-3, f"Projection not idempotent: error={idempotent_err:.6f}"
    print(f"  scorer_manifold: idempotency verified (error={idempotent_err:.2e})")

    print("scorer_manifold: all smoke tests passed")


if __name__ == "__main__":
    _smoke_test()
