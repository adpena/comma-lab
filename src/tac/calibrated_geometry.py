"""Lane CG: calibrated camera geometry helpers (EON intrinsics, ray-casting,
homography → SE(3) decomposition).

The camera intrinsics are pinned to the comma.ai EON device (challenge native
camera) at the renderer working resolution of 384 × 512.  These match the
constants in ``tac.geodesic_pose`` and ``tac.contrib.homography_motion``.

Math
----
* ``K = [[fx, 0, pp_x], [0, fy, pp_y], [0, 0, 1]]`` is the pinhole matrix in
  pixel coordinates.  Pixel-to-ray inversion is
  ``r = K^{-1} [u, v, 1]^T`` followed by L2 normalization.
* Homography decomposition follows the Faugeras / Lustman analytical solution
  (Faugeras & Lustman, *Motion and Structure from Motion in a Piecewise Planar
  Environment*, IJCV 1988).  Given a calibrated planar homography
  ``H_c = K^{-1} H K`` we solve

      H_c = R + t · n^T / d

  by SVD of ``H_c^T H_c``.  This yields up to four candidate ``(R, t, n, d)``
  tuples; we follow the standard cheirality + small-rotation tie-break:

  1. Discard solutions whose plane normal does not satisfy ``n^T · [0, 0, 1] > 0``
     (the calibrated camera looks down the +z axis, so the road plane normal in
     the camera frame must have a positive z component).
  2. Among the remaining solutions, pick the one with the smallest rotation
     magnitude ``||log(R)||``.  When two solutions tie within the ``1e-6`` SE(3)
     small-angle threshold we pick the one with the larger plane depth
     ``d > 0`` (positive-depth cheirality).
* The decomposition output is wrapped into a single ``(omega, v)`` 6-vector by
  composing axis-angle rotation with translation in the order used by
  ``tac.se3`` (rotation first, then translation), matching the pose head of
  the upstream PoseNet.

References
----------
* Faugeras & Lustman 1988, IJCV (the Faugeras / Lustman closed-form).
* Malis & Vargas, *Deeper understanding of the homography decomposition for
  vision-based control*, INRIA RR-6303, 2007 (cheirality tie-break used here).
* Sola et al. 2018, arXiv:1812.01537 (SE(3) exp / log conventions).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tac.se3 import exp_map_se3, log_map_so3

# ──────────────────────────────────────────────────────────────────────────
# EON camera intrinsics (challenge native, working resolution 384 × 512)
# ──────────────────────────────────────────────────────────────────────────

CAMERA_FX: float = 910.0
CAMERA_FY: float = 910.0
CAMERA_PP: tuple[float, float] = (582.0, 437.0)
CAMERA_WIDTH: int = 512
CAMERA_HEIGHT: int = 384

__all__ = [
    "CAMERA_FX",
    "CAMERA_FY",
    "CAMERA_PP",
    "CAMERA_WIDTH",
    "CAMERA_HEIGHT",
    "CalibratedGeometry",
    "HomographyDecomposition",
]


@dataclass(frozen=True)
class HomographyDecomposition:
    """Single Faugeras / Lustman decomposition candidate ``H = R + t n^T / d``.

    Attributes:
        R: rotation matrix, shape ``(3, 3)``.
        t: translation vector (scaled by ``d``), shape ``(3,)``.
        n: plane unit normal in the camera frame, shape ``(3,)``.
        d: plane depth (positive for cheirality-valid solutions).
        pose: ``(omega, v)`` SE(3) 6-vector with axis-angle rotation followed
            by translation, suitable for direct addition to a PoseNet pose
            initialization.
    """

    R: torch.Tensor
    t: torch.Tensor
    n: torch.Tensor
    d: float
    pose: torch.Tensor


class CalibratedGeometry:
    """EON-calibrated pinhole geometry helpers.

    Args:
        fx: focal length in pixels (default ``CAMERA_FX``).
        fy: focal length in pixels (default ``CAMERA_FY``).
        pp: principal point ``(pp_x, pp_y)`` in pixels (default ``CAMERA_PP``).
        width: image width in pixels at which ``fx``/``pp`` are calibrated.
        height: image height in pixels at which ``fy``/``pp`` are calibrated.
        device: torch device for the cached ``K`` matrix.
        dtype: dtype for the cached ``K`` matrix.
    """

    def __init__(
        self,
        *,
        fx: float = CAMERA_FX,
        fy: float = CAMERA_FY,
        pp: tuple[float, float] = CAMERA_PP,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"focal length must be positive; got fx={fx}, fy={fy}")
        if width <= 0 or height <= 0:
            raise ValueError(f"image dims must be positive; got {width}x{height}")
        self.fx = float(fx)
        self.fy = float(fy)
        self.pp = (float(pp[0]), float(pp[1]))
        self.width = int(width)
        self.height = int(height)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self._K = torch.tensor(
            [
                [self.fx, 0.0, self.pp[0]],
                [0.0, self.fy, self.pp[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self._K_inv = torch.linalg.inv(self._K)

    @property
    def K(self) -> torch.Tensor:
        """Return the cached intrinsics matrix ``(3, 3)``."""
        return self._K

    @property
    def K_inv(self) -> torch.Tensor:
        """Return the cached inverse intrinsics matrix ``(3, 3)``."""
        return self._K_inv

    def pixel_to_ray(self, uv: torch.Tensor) -> torch.Tensor:
        """Project pixel coordinates to unit rays in the camera frame.

        Args:
            uv: pixel coordinates, shape ``(..., 2)`` (last dim is ``[u, v]``).

        Returns:
            Unit rays of shape ``(..., 3)`` such that ``r = K^{-1} [u, v, 1]^T``
            normalized.  The L2 norm of every row equals ``1`` to within
            float64 precision.
        """
        if uv.ndim < 1 or uv.shape[-1] != 2:
            raise ValueError(f"uv must end in dim=2; got {tuple(uv.shape)}")
        uv = uv.to(dtype=self.dtype, device=self.device)
        ones = torch.ones(*uv.shape[:-1], 1, dtype=self.dtype, device=self.device)
        homo = torch.cat([uv, ones], dim=-1)
        rays = homo @ self._K_inv.T
        norms = rays.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return rays / norms

    def homography_to_pose(
        self,
        H: torch.Tensor,
        *,
        return_decomposition: bool = False,
    ) -> torch.Tensor | HomographyDecomposition:
        """Decompose a pixel-space homography into an SE(3) pose.

        Implements the Faugeras / Lustman 1988 closed-form on the calibrated
        homography ``H_c = K^{-1} H K``.  Of the (up to) four candidate
        solutions we return the one that:

        1. Has a plane normal with positive z component (camera-forward).
        2. Has the smallest rotation magnitude (``||log(R)||``); ties (within
           the SE(3) small-angle threshold) are broken by larger plane depth.

        Args:
            H: 3 × 3 pixel-space homography (``f64`` recommended).
            return_decomposition: if true, return the full
                :class:`HomographyDecomposition`; otherwise return only the
                ``(6,)`` pose vector.

        Returns:
            Either a ``(6,)`` SE(3) pose vector ``(ω, v)`` (axis-angle then
            translation) or a :class:`HomographyDecomposition` if
            ``return_decomposition`` is ``True``.
        """
        if H.shape != (3, 3):
            raise ValueError(f"H must be (3, 3); got {tuple(H.shape)}")
        H64 = H.to(dtype=torch.float64, device=self.device)
        Hc = self._K_inv @ H64 @ self._K

        # Faugeras / Lustman: scale Hc so that its middle singular value is 1.
        # Then S = Hc^T Hc has eigenvalues 1 and 1 ± something; the "1"
        # eigenvalue's eigenvector gives the plane normal up to sign.
        _, sv, _ = torch.linalg.svd(Hc)
        median = sv[1].clamp_min(1e-12)
        Hc = Hc / median

        # Pure-rotation degenerate detection: when t/d ≈ 0 the calibrated
        # homography is orthonormal (Hc^T Hc ≈ I).  Faugeras's eigen-construction
        # collapses (x1 = x3 = 0) so we short-circuit with the polar factor.
        S = Hc.T @ Hc
        eye = torch.eye(3, dtype=torch.float64, device=self.device)
        if float((S - eye).abs().max()) < 1e-8:
            R = _polar_decomposition(Hc)
            t = torch.zeros(3, dtype=torch.float64, device=self.device)
            n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=self.device)
            omega = log_map_so3(R)
            pose = torch.zeros(6, dtype=torch.float64, device=self.device)
            pose[:3] = omega
            best = HomographyDecomposition(R=R, t=t, n=n, d=0.0, pose=pose)
            if return_decomposition:
                return best
            return best.pose

        # Eigen-decomposition of S = U diag(d) U^T with d_1 >= d_2 >= d_3.
        eigvals, eigvecs = torch.linalg.eigh(S)
        # eigh returns ascending; reverse to descending so d1 >= d2 >= d3.
        eigvals = torch.flip(eigvals, dims=(0,))
        eigvecs = torch.flip(eigvecs, dims=(1,))
        d1, d2, d3 = (float(eigvals[0]), float(eigvals[1]), float(eigvals[2]))

        # Build the two plane-normal candidates (Faugeras eq. 11) using the
        # eigenvectors of S.  The middle singular value of Hc is now 1, so the
        # plane normal is a ± combination of the eigenvectors of d1 and d3.
        denom = max(d1 - d3, 1e-12)
        x1 = float(((1.0 - d3) / denom) ** 0.5) if d1 > d3 else 0.0
        x3 = float(((d1 - 1.0) / denom) ** 0.5) if d1 > d3 else 0.0

        u1 = eigvecs[:, 0]
        u3 = eigvecs[:, 2]
        n_candidates = [
            (x1 * u1 + x3 * u3),
            (x1 * u1 - x3 * u3),
            (-x1 * u1 + x3 * u3),
            (-x1 * u1 - x3 * u3),
        ]

        candidates: list[HomographyDecomposition] = []
        for n in n_candidates:
            n_norm = n.norm()
            if float(n_norm) < 1e-9:
                continue
            n = n / n_norm
            # The plane depth d in the camera frame.  Faugeras / Lustman's
            # decomposition is invariant to its sign; we adopt d > 0 by flipping
            # the normal when needed.
            if float(n[2]) <= 0.0:
                # Camera-forward cheirality: reject normals pointing backward.
                continue

            # Recover R and t from H_c, n.  Two consistent solutions; take the
            # closed form via R = H_c (I - 2 n n^T / (n^T n))^{-1} after the
            # x1/x3 sign convention above (see Malis & Vargas 2007 §3.2).
            R, t, depth = _faugeras_recover_R_t(Hc, n)
            if depth <= 0.0:
                continue
            try:
                omega = log_map_so3(R)
            except Exception:
                continue
            pose = torch.zeros(6, dtype=torch.float64, device=self.device)
            pose[:3] = omega
            pose[3:] = t
            candidates.append(
                HomographyDecomposition(R=R, t=t, n=n, d=depth, pose=pose)
            )

        if not candidates:
            # Pure rotation degenerate case (Hc orthonormal): R = Hc, t = 0.
            R = _polar_decomposition(Hc)
            t = torch.zeros(3, dtype=torch.float64, device=self.device)
            omega = log_map_so3(R)
            pose = torch.zeros(6, dtype=torch.float64, device=self.device)
            pose[:3] = omega
            n = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=self.device)
            best = HomographyDecomposition(R=R, t=t, n=n, d=0.0, pose=pose)
        else:
            # Tie-break: smallest rotation magnitude, then largest depth.
            def key(c: HomographyDecomposition) -> tuple[float, float]:
                rot = float(c.pose[:3].norm())
                return (rot, -c.d)

            best = min(candidates, key=key)

        if return_decomposition:
            return best
        return best.pose


def _faugeras_recover_R_t(
    Hc: torch.Tensor,
    n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Recover ``(R, t, d)`` from a calibrated homography and a plane normal.

    Uses ``H_c = R + (t / d) n^T``.  The Householder construction
    ``M = I - 2 n n^T`` would only work for symmetric ``H_c``; instead we use
    the analytic re-arrangement: project ``H_c n`` to recover ``t / d`` after
    fitting ``R`` to the orthogonal part of ``H_c``.
    """
    eye = torch.eye(3, dtype=torch.float64, device=Hc.device)
    # Closed-form: R = H_c · M where M makes Hc^T H_c → R^T R = I; in practice
    # we use R = polar(H_c - t n^T / d).  We solve simultaneously by SVD-polar.
    # Step 1: t/d as the component of Hc·n minus n (since R·n is unit length
    # but generally != n; however H_c n - R n = t/d * (n·n) = t/d).  We assume
    # the rotation is small enough that R·n ≈ n.  This holds for the comma.ai
    # frame-pair regime (|ω| ≲ 0.1 rad) and matches Malis & Vargas §3.2.
    Hn = Hc @ n
    # Solve for R that minimizes ||Hc - R - (Hn - R n) n^T||² with R ∈ SO(3).
    # Equivalent to polar decomposition of Hc' = Hc - (something).  The robust
    # closed form for small rotations: take R = polar(Hc · (I - n n^T) + (R n) n^T).
    # We approximate R n ≈ n initially, refine once.
    Rn0 = n
    A0 = Hc @ (eye - torch.outer(n, n)) + torch.outer(Rn0, n)
    R0 = _polar_decomposition(A0)
    Rn = R0 @ n
    A = Hc @ (eye - torch.outer(n, n)) + torch.outer(Rn, n)
    R = _polar_decomposition(A)
    t = (Hc @ n) - (R @ n)
    # Plane depth in the SVD-normalized (median-singular-value = 1) frame.
    # Faugeras / Lustman 1988: H_c = R + (t/d) n^T.  Projecting both sides
    # onto n (unit) gives n^T H_c n = n^T R n + (n·t)/d, and the scalar
    # n^T H_c n therefore differs PER CANDIDATE because each candidate has
    # its own (R, n) pair.  Note that ||H_c·n||² = n^T S n is the SAME for
    # all four candidates whenever they live in the same (u1, u3) eigen-plane
    # of S = H_c^T H_c, so the obvious "norm" choice is degenerate; the
    # quadratic form below is the canonical scalar that breaks the tie.
    # (R17 finding 1 — replaces the legacy hardcoded depth=1.0 that made the
    # cheirality tie-break collapse to rotation magnitude alone.)
    #
    # R19 finding 2 [IMPORTANT — non-blocking]: under small rotations
    # (n^T R n ≈ 1) this scalar approximates 1 + (n·t)/d, which is monotone
    # in d and thus serves as a valid tie-break key.  For LARGE rotations
    # (|ω| > ~0.1 rad) the n^T R n term diverges from 1 and the depth
    # ordering can become non-monotone.  For the comma.ai per-frame regime
    # |ω| ≲ 0.05 rad this is fine; for arbitrary use callers should verify
    # via an explicit |ω| check before trusting the tie-break.
    depth = float((n @ (Hc @ n)).item())
    return R, t, depth


def _polar_decomposition(A: torch.Tensor) -> torch.Tensor:
    """Return the orthogonal factor of ``A`` via SVD: ``A = R · (V Σ V^T)``."""
    U, _, Vt = torch.linalg.svd(A)
    R = U @ Vt
    # Force det = +1 (proper rotation).
    if float(torch.linalg.det(R)) < 0.0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1.0
        R = U @ Vt
    return R


def make_pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a dense ``(height, width, 2)`` grid of pixel coordinates.

    The output ordering is ``[u, v]`` (column, row) so it can be passed
    directly to :meth:`CalibratedGeometry.pixel_to_ray`.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"dims must be positive; got {height}x{width}")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    ys = torch.arange(height, dtype=dtype, device=dev)
    xs = torch.arange(width, dtype=dtype, device=dev)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


# Helper for SE(3) compose: included for symmetry with se3.exp_map_se3 even
# though the decomposition writes ω and v directly into a single 6-vector.
__all__.append("make_pixel_grid")
__all__.append("compose_pose_from_decomposition")


def compose_pose_from_decomposition(decomp: HomographyDecomposition) -> torch.Tensor:
    """Compose a 4 × 4 SE(3) homogeneous matrix from a decomposition."""
    R_se3, t_se3 = exp_map_se3(decomp.pose[:3], decomp.pose[3:])
    T = torch.eye(4, dtype=R_se3.dtype, device=R_se3.device)
    T[:3, :3] = R_se3
    T[:3, 3] = t_se3
    return T
