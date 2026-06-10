# SPDX-License-Identifier: MIT
"""PoseNet pose-sensitive SUBSPACE spectrum — the effective-dimension + pose-null measurement (task #80).

THE TASK-80 INVENTION (vs the #61 Jacobian-NORM saliency): the #61 module computes the per-pixel
sensitivity NORM ``w_pix = ||d pose6 / d pixel||`` — a scalar field telling WHICH pixels the 6 scored
dims read. This module measures the SUBSPACE STRUCTURE: the full signed 6xN Jacobian ``J`` (6 scored
pose dims w.r.t. the N camera-res frame pixels), its SVD spectrum, the EFFECTIVE DIMENSION (participation
ratio of the singular values), and the POSE-NULL energy fraction of a given frame perturbation.

WHY THIS IS THE POSE CRUX (the operator's hypothesis, made measurable):
  * d_pose = MSE on the FIRST 6 of 12 PoseNet pose dims (upstream/modules.py:82-84). So the pose-output
    space is exactly 6-dimensional. The map (frame pixels -> 6 pose dims) is a smooth function; its local
    linearization at the GT operating point is the 6xN Jacobian J.
  * The ROW SPACE of J (span of the 6 gradient vectors in pixel space) is the POSE-SENSITIVE SUBSPACE:
    a frame perturbation delta moves the 6 pose dims by ~ J @ delta. Any delta ORTHOGONAL to all 6 rows
    moves the pose dims by ~0 to first order -> the POSE-NULL (analog of the resize-null for SegNet).
  * The pose-sensitive subspace has dimension AT MOST 6 (it is the row space of a 6xN matrix). The
    EFFECTIVE dimension <= 6 is the participation ratio (sum sigma_i)^2 / sum(sigma_i^2): how many of the
    <=6 directions actually carry pose sensitivity (a near-rank-1 J has eff-dim ~1; a balanced J ~6).
  * THE ESCAPE: if a frame perturbation's energy lands mostly in the pose-null (the (N-6)-dim complement),
    it costs ~0 d_pose REGARDLESS of its pixel-RMSE. So a representation that confines its error to the
    pose-null can be wrong by a large pixel-RMSE and STILL hold the tube -- the pixel-RMSE<3 requirement
    (#74) would then be an artifact of ISOTROPIC error (uniform noise spreads equally into the 6-dim
    sensitive subspace), not a fundamental pose-fidelity floor.

THE SCORER FACTS (upstream/modules.py + frame_utils.py, verified):
  * PoseNet reads BOTH frames: x is (b, 2, 3, H, W) -> preprocess_input resizes to 384x512 (bilinear) ->
    rgb_to_yuv6 (HALF-res luma y00/y10/y01/y11 + 2x2-box chroma U_sub/V_sub) -> 12ch (2 frames x 6) ->
    fastvit_t12 vision -> summarizer -> Hydra pose head (12 out); d_pose = MSE on first 6.
  * The Jacobian here is of the 6 SCORED dims w.r.t. ONE frame's camera-res RGB pixels (the other frame
    held at GT). Computed by backprop through the DIFFERENTIABLE yuv6 path (``tac.differentiable_eval_
    roundtrip``) because upstream ``rgb_to_yuv6`` is ``@torch.no_grad()``/in-place and severs the gradient.
  * The Jacobian is measured at the GT operating point (pose(GT)==GT pose; the GRADIENT of d_pose vanishes
    there, but the JACOBIAN of the 6 raw pose dims does NOT -- it is the local sensitivity we need).

COMPUTE-SUBSTRATE LAW (CLAUDE.md): exact frozen CPU-torch PoseNet (NEVER mps). GT via
``frame_utils.yuv420_to_rgb`` ONLY. The full 6xN Jacobian at camera res (3*874*1164 ~ 3M) is built ROW
BY ROW (6 backward passes); the SVD is taken on the 6xN matrix (a 6x6 Gram eigendecomposition, exact and
cheap). Evidence ``[local CPU-torch advisory]``. Non-promotable.

FAIL-CLOSED (the severed-gradient signature): if the 6xN Jacobian is identically zero, the differentiable
path is broken (gradient severed) and a non-reachable gradient must NEVER masquerade as "all pose-null".
:func:`measure_pose_subspace_spectrum` raises in that case.

NO-FAKE (class 8): the spectrum is the EXACT SVD of the EXACT backprop Jacobian of the frozen PoseNet
(not a fabricated constant). The pose-null projection is the exact orthogonal complement of the measured
row space. :func:`project_onto_pose_null` is a pure deterministic linear-algebra transform of the measured
basis; an ISOTROPIC random perturbation has a known expected null fraction ((N-r)/N -> ~1) so a test can
prove the subspace is load-bearing (a perturbation BUILT inside the row space has null fraction ~0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512  # PoseNet resize target (segnet_model_input_size = (W,H) = (512,384))
N_SCORED_POSE_DIMS = 6  # d_pose = MSE on first 6 of 12 pose dims (upstream/modules.py:82-84)


class PoseSubspaceError(ValueError):
    """Raised when the measured pose Jacobian is degenerate (severed gradient)."""


@dataclass(frozen=True)
class PoseSubspaceSpectrum:
    """The measured pose-sensitive subspace structure for one frame slot of one pair.

    ``singular_values`` (r,) float >= 0 (descending): the singular values of the 6xN Jacobian (r <= 6).
    ``row_basis`` (r, N) float32: ORTHONORMAL basis of the pose-sensitive subspace (the right singular
    vectors V^T, i.e. the directions in flattened pixel space the 6 pose dims read). ``N`` is the flattened
    frame dimension (3*H*W). ``effective_dim`` is the participation ratio of the singular values.
    """

    singular_values: np.ndarray  # (r,) float64, descending
    row_basis: np.ndarray  # (r, N) float64 orthonormal rows
    n_pixels: int  # N = 3*H*W
    h: int
    w: int
    frame_slot: int  # 0 or 1
    compute_path: str  # "cpu_torch" — NEVER mps
    effective_dim: float  # participation ratio (sum s)^2 / sum s^2, in [1, r]
    rank: int  # number of singular values above the relative tolerance

    def to_summary(self) -> dict[str, float | int]:
        s = self.singular_values
        tot = float((s**2).sum())
        return {
            "rank": int(self.rank),
            "effective_dim": float(self.effective_dim),
            "n_pose_output_dims": int(N_SCORED_POSE_DIMS),
            "n_pixels": int(self.n_pixels),
            "sigma_max": float(s[0]) if s.size else 0.0,
            "sigma_min_nonzero": float(s[s > 0].min()) if (s > 0).any() else 0.0,
            "sigma_ratio_max_over_2nd": (float(s[0] / s[1]) if s.size >= 2 and s[1] > 0 else float("inf")),
            "energy_frac_top1": (float(s[0] ** 2 / tot) if tot > 0 else 0.0),
            "energy_frac_top3": (float((s[:3] ** 2).sum() / tot) if tot > 0 else 0.0),
        }


def participation_ratio(singular_values: np.ndarray) -> float:
    """Effective dimension = (sum s_i)^2 / sum(s_i^2) for nonneg s. In [1, r]; 1 for rank-1.

    This is the standard participation-ratio / effective-rank-of-energy measure. For a flat spectrum of r
    equal values it returns r; for a single dominant value it returns ~1. NO-FAKE: pure deterministic.
    """

    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    num = float(s.sum()) ** 2
    den = float((s**2).sum())
    return num / den if den > 0 else 0.0


def _orthonormal_row_basis(jac: np.ndarray, rel_tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (singular_values descending, orthonormal row basis (r,N), rank) of the (k, N) Jacobian.

    Two-stage, numerically stable: (1) the singular values come from the (k x k) Gram eigendecomposition
    ``J J^T = U S^2 U^T`` (k=6, exact + cheap; gives the spectrum without the N x N right-singular solve).
    (2) the ORTHONORMAL row-space basis comes from a QR of ``J^T`` (the row space of J == column space of
    J^T), which is numerically stable: it does NOT divide by the tiny singular values (the V^T_i = U_i^T J
    / s_i formula blows up float noise when s_i ~ 1e-6, corrupting the pose-null projection). The QR Q
    columns are an exact orthonormal basis of the row space regardless of the singular-value magnitudes.

    NO-FAKE: exact linear algebra; the basis rows are orthonormal to float64 tolerance (a test asserts
    V V^T == I_r), and being QR-derived it does NOT manufacture spurious directions from float noise.
    The ``rank`` is read from the singular spectrum (the meaningful pose-sensitive directions); the basis
    is truncated to the same rank so the pose-null is the complement of the SIGNIFICANT directions only.
    """

    k, n = jac.shape
    gram = jac @ jac.T  # (k, k)
    evals_asc, evecs_asc = np.linalg.eigh(gram)  # ascending eigenvalues, columns = eigenvectors
    order = np.argsort(evals_asc)[::-1]  # descending
    evals = np.clip(evals_asc[order], 0.0, None)
    u_left = evecs_asc[:, order]  # (k, k) left singular vectors (columns), descending
    sv = np.sqrt(evals)  # singular values (descending)
    smax = float(sv[0]) if sv.size else 0.0
    if smax <= 0.0:
        return sv, np.zeros((0, n), dtype=np.float64), 0
    rank = int(np.sum(sv > rel_tol * smax))
    rank = max(rank, 1)
    # The leading `rank` row-space directions are span{ U_i^T J : i < rank }, ORDERED by singular value.
    # Build them as W = U[:, :rank]^T @ J (rank, N), then QR-orthonormalize W^T (numerically stable: no
    # division by tiny s_i). The QR of an already-spanning ordered set returns an orthonormal basis of the
    # SAME ordered space, so the basis is the significant pose-sensitive directions (not float-noise ones).
    w = u_left[:, :rank].T @ jac  # (rank, N), rows in singular-value order
    q, _r = np.linalg.qr(w.T)  # (N, rank) orthonormal columns spanning the same space
    vt = q[:, :rank].T.astype(np.float64)  # (rank, N) orthonormal rows
    return sv, vt, rank


def measure_pose_subspace_spectrum(
    posenet,
    gt_frame0_chw,
    gt_frame1_chw,
    *,
    frame_slot: int = 0,
    work_h: int = SEG_H,
    work_w: int = SEG_W,
    rel_tol: float = 1e-4,
):
    """Measure the pose-sensitive subspace spectrum of one frame slot of a pair on the frozen PoseNet.

    ``posenet``: frozen ``upstream.modules.PoseNet`` on CPU with the differentiable yuv6 patch ACTIVE.
    ``gt_frameN_chw``: (3, H, W) float [0,255] camera-res GT frames.
    ``frame_slot``: which frame (0 or 1) the Jacobian is w.r.t. (the other held at GT).
    ``work_h/work_w``: the resolution at which the Jacobian is taken. PoseNet ALWAYS resizes the input to
        (384, 512) internally; to make the N-dim SVD tractable AND to measure the subspace at the
        resolution PoseNet actually reads, we take the Jacobian w.r.t. a (3, work_h, work_w) frame that is
        bilinear-upsampled to camera res inside the graph (so the measured directions live in the
        work-resolution pixel space, which is what a carrier would encode). Default = the 384x512 PoseNet
        working resolution (N = 3*384*512 = 589,824).

    Method (the exact 6xN row-space SVD): build the 6 backward passes of the 6 scored pose dims to the
    (3, work_h, work_w) frame; stack into a (6, N) Jacobian; SVD via the 6x6 Gram trick; participation
    ratio = effective dimension. FAIL-CLOSED if the Jacobian is identically zero (severed gradient).

    Returns a :class:`PoseSubspaceSpectrum`.
    """

    import torch
    import torch.nn.functional as F

    if frame_slot not in (0, 1):
        raise ValueError(f"frame_slot must be 0 or 1, got {frame_slot}")

    dev = torch.device("cpu")
    f0_cam = torch.as_tensor(gt_frame0_chw, dtype=torch.float32, device=dev).clone()
    f1_cam = torch.as_tensor(gt_frame1_chw, dtype=torch.float32, device=dev).clone()

    # The free variable is a (3, work_h, work_w) frame, upsampled to camera res inside the graph.
    cam_target = f0_cam if frame_slot == 0 else f1_cam
    work = F.interpolate(
        cam_target.unsqueeze(0), size=(work_h, work_w), mode="bilinear", align_corners=False
    )[0].clone()
    work.requires_grad_(True)

    work_cam = F.interpolate(
        work.unsqueeze(0), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )[0]
    if frame_slot == 0:
        f0_use, f1_use = work_cam, f1_cam
    else:
        f0_use, f1_use = f0_cam, work_cam

    x = torch.stack([f0_use, f1_use]).unsqueeze(0)  # (1,2,3,H,W)
    pin = posenet.preprocess_input(x)
    pose6 = posenet(pin)["pose"][..., :N_SCORED_POSE_DIMS].reshape(-1)  # (6,)

    n = work.numel()
    jac = np.zeros((N_SCORED_POSE_DIMS, n), dtype=np.float64)
    for kdim in range(N_SCORED_POSE_DIMS):
        g = torch.autograd.grad(
            pose6[kdim], work, retain_graph=(kdim < N_SCORED_POSE_DIMS - 1), create_graph=False
        )[0]
        jac[kdim] = g.detach().reshape(-1).cpu().numpy().astype(np.float64)

    if not np.any(np.abs(jac) > 0):
        raise PoseSubspaceError(
            "PoseNet 6xN Jacobian is identically zero — the differentiable yuv6 path is not active "
            "(gradient severed). Patch rgb_to_yuv6 before measuring; refusing to fake pose-null."
        )

    sv, row_basis, rank = _orthonormal_row_basis(jac, rel_tol=rel_tol)
    eff = participation_ratio(sv[:rank])
    return PoseSubspaceSpectrum(
        singular_values=sv,
        row_basis=row_basis,
        n_pixels=int(n),
        h=int(work_h),
        w=int(work_w),
        frame_slot=int(frame_slot),
        compute_path="cpu_torch",
        effective_dim=float(eff),
        rank=int(rank),
    )


def project_onto_pose_null(spectrum: PoseSubspaceSpectrum, delta_flat: np.ndarray) -> dict[str, np.ndarray | float]:
    """Decompose a flat frame perturbation into its pose-SENSITIVE and pose-NULL components.

    ``delta_flat`` (N,) the frame perturbation (flattened, in the same work-resolution pixel space as the
    measured ``row_basis``). Returns:
      sensitive (N,)  : the projection onto the pose-sensitive subspace (row space of J)
      null (N,)       : the orthogonal complement (the pose-null component)
      null_energy_frac: ||null||^2 / ||delta||^2  (the fraction of perturbation energy invisible to pose)
      sensitive_energy_frac: 1 - null_energy_frac

    NO-FAKE: pure deterministic projection onto the MEASURED orthonormal basis. A delta built INSIDE the
    row space has null_energy_frac ~0; an isotropic random delta has null_energy_frac ~ (N-r)/N ~ 1.
    """

    d = np.asarray(delta_flat, dtype=np.float64).reshape(-1)
    basis = np.asarray(spectrum.row_basis, dtype=np.float64)  # (r, N)
    if basis.shape[1] != d.shape[0]:
        raise ValueError(
            f"delta_flat length {d.shape[0]} != row_basis N {basis.shape[1]}; "
            "the perturbation must live in the same work-resolution pixel space as the spectrum."
        )
    coeffs = basis @ d  # (r,)
    sensitive = coeffs @ basis  # (N,) float64
    null = d - sensitive
    tot = float(d @ d)
    null_e = float(null @ null)
    sens_e = float(sensitive @ sensitive)
    # cast to float32 ONLY for the returned arrays; the energy fractions are computed in float64 above so
    # large-magnitude perturbations do not overflow the float32 square.
    null_frac = (null_e / tot) if tot > 0 else 0.0
    sens_frac = (sens_e / tot) if tot > 0 else 0.0
    return {
        "sensitive": sensitive.astype(np.float32),
        "null": null.astype(np.float32),
        "null_energy_frac": float(null_frac),
        "sensitive_energy_frac": float(sens_frac),
    }


def expected_isotropic_null_fraction(n_pixels: int, rank: int) -> float:
    """The expected pose-null energy fraction of an ISOTROPIC random perturbation: (N - r) / N.

    This is the BASELINE: isotropic (uniform) error spreads its energy equally across all N pixel
    directions, so only r/N of it lands in the pose-sensitive subspace. The pixel-RMSE<3 finding (#74) was
    measured with isotropic noise -> this is exactly why isotropic noise breaks the tube at small RMSE
    while a null-confined error of the SAME RMSE does not. NO-FAKE: pure arithmetic.
    """

    if n_pixels <= 0:
        return 0.0
    return max(0.0, float(n_pixels - rank) / float(n_pixels))


__all__ = [
    "N_SCORED_POSE_DIMS",
    "PoseSubspaceError",
    "PoseSubspaceSpectrum",
    "expected_isotropic_null_fraction",
    "measure_pose_subspace_spectrum",
    "participation_ratio",
    "project_onto_pose_null",
]
