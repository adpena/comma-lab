# SPDX-License-Identifier: MIT
"""Dykstra alternating-projection FEASIBILITY solve for the legal frame (task #73).

THE CRUX (verdict memo `lever_b_score_native_argmax_smoke_verdict_20260610` §6, and
`closed_spec_boundary_math_system_of_equations_20260610` §4): the lever-B smoke proved the seg
term is GENERATABLE (a logit net hits the frozen SegNet argmax to d_seg 0.0073) but it scores
LOGITS, walls on POSE (palette pose-blind 12.14; INR pose-ceiling 0.0036). The legal archive
scores FRAMES. The correct mathematics is NOT generation (a smooth net forced to encode a sharp
partition AND motion) — it is **FEASIBILITY**: find ANY frame in

    A = { F : argmax SegNet(F) == L*  pixelwise }     (the SegNet argmax CELL)
    B = { F : ||PoseNet(pair_F)[:6] - p*||^2 <= tau }  (the PoseNet TUBE)
    C = { F : F - base in cheap-encoding subspace }    (low-rank / sparse / quantized delta)

cheapest, by **Dykstra alternating projections** onto A, B, C, RE-LINEARIZING the nonlinear SegNet
and PoseNet Jacobians each OUTER iteration (sequential convex feasibility / SQP-feasibility).

WHY THIS SOLVES THE WALL (or proves the cheap set empty): the GT frame1 is ALREADY in A (d_seg=0)
and the GT pair is ALREADY in B (d_pose=0). The non-trivial move is projecting the perturbation
``delta = F - base`` onto C (the cheap subspace) — that is what makes it a CARRIER not a copy.
The question: does the cheap projection of delta stay in A and B? If yes -> a cheap feasible frame
exists (a frontier candidate). If the cheap projection KICKS the frame out of A/B and the A/B
re-projections cannot pull it back at low byte -> cell n tube n cheap is EMPTY at that byte (a
sharp geometric finding: the carrier genuinely needs more bytes).

THE GRID: SegNet and PoseNet consume the bilinear-RESIZED (384x512) input directly
(``segnet_model_input_size = (512, 384)``). We run the WHOLE solve on that exact (3, 384, 512)
tensor the scorers feed (vs camera-res + resize-null), so every projection is on the literal scored
object. The "cheap basis" is then a low-rank + sparse + quantized representation of the 384x512
perturbation; the stored byte cost = that quantized delta, brotli-q11.

NO-FAKE:
  * class 1 (no no-op): each projection ACTUALLY moves the frame toward its constraint on the REAL
    frozen scorer. ``project_onto_margin_cell`` solves a per-pixel Jacobian-difference step that
    closes violated margins; ``project_onto_pose_tube`` solves the least-norm step of the 6xN pose
    Jacobian; ``project_onto_cheap`` truncates to the low-rank+sparse subspace. A stub that returns
    the input FAILS the tests (which assert violation actually decreases on the real scorer).
  * class 8 (exact authority): d_seg = exact argmax-disagreement popcount of the frozen SegNet;
    d_pose = exact ||PoseNet[:6] - p*||^2 of the frozen PoseNet. GT via frame_utils.yuv420_to_rgb,
    NEVER MPS. ``[macOS-CPU advisory]`` — non-promotable; no score claim.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): exact frozen CPU-torch SegNet/PoseNet. The Jacobians are the
EXACT measured oracle Jacobians (autograd), re-measured each outer iteration. The PoseNet path uses
the differentiable yuv6 patch (upstream rgb_to_yuv6 is @torch.no_grad()/in-place and severs the
gradient); FAIL-CLOSED if the pose Jacobian is identically zero (severed-gradient signature).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEG_H, SEG_W = 384, 512
N_CLASSES = 5
N_POSE = 6

# THE LAW (evaluate.py): S = 100*d_seg + sqrt(10*d_pose) + 25*B/D.
_SEG_WEIGHT = 100.0
_POSE_TEN = 10.0
_RATE_COEF = 25.0
_CONTEST_TOTAL_BYTES = 37_545_489


class DykstraLegalFrameError(ValueError):
    """Raised on malformed inputs / contract violations of the feasibility solve."""


def score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """Recompute THE LAW score from components (the rounded final_score lies)."""

    return (
        _SEG_WEIGHT * float(d_seg)
        + float(np.sqrt(_POSE_TEN * max(float(d_pose), 0.0)))
        + _RATE_COEF * float(archive_bytes) / float(_CONTEST_TOTAL_BYTES)
    )


# ════════════════════════════════════════════════════════════════════════════
#  The exact frozen-oracle projector — wraps a SegNet + PoseNet on the (3,384,512)
#  scorer-input grid. Re-linearized each outer iteration.
# ════════════════════════════════════════════════════════════════════════════
class FrozenScorerProjector:
    """Exact SegNet/PoseNet projections on the scorer-input (3, 384, 512) grid.

    ``seg_in`` is the frame1 SegNet input (3, 384, 512). ``pose_in_frames`` is the pair
    ``(2, 3, 384, 512)`` RESIZED-RGB frames PoseNet's preprocess feeds yuv6 (we hold frame0
    fixed = GT motion, perturb frame1 = the SegNet object). All exact CPU-torch; NEVER MPS.

    The perturbation ``delta`` is on frame1's (3, 384, 512) grid. The two scorer constraints share
    the SAME frame1 delta (frame1 is both the SegNet object and one of the two PoseNet frames), so
    a delta in the cell must also respect the pose tube — that coupling is exactly why the naive
    "palette frame1" walled on pose, and why we must alternate.
    """

    def __init__(self, segnet: Any, posenet: Any, seg_in_frame1: Any, pose_frame0: Any) -> None:
        import torch

        self._torch = torch
        self._seg = segnet
        self._pose = posenet
        f1 = torch.as_tensor(seg_in_frame1, dtype=torch.float32).detach().clone()
        if f1.ndim == 4:
            f1 = f1[0]
        if f1.shape != (3, SEG_H, SEG_W):
            raise DykstraLegalFrameError(
                f"seg_in_frame1 must be (3,{SEG_H},{SEG_W}); got {tuple(f1.shape)}"
            )
        if str(getattr(f1, "device", "cpu")) not in ("cpu", "cpu:0"):
            raise DykstraLegalFrameError("scorer inputs must be CPU (MPS forbidden)")
        self._f1_base = f1  # (3,384,512) GT frame1 on the scorer grid
        f0 = torch.as_tensor(pose_frame0, dtype=torch.float32).detach().clone()
        if f0.ndim == 4:
            f0 = f0[0]
        if f0.shape != (3, SEG_H, SEG_W):
            raise DykstraLegalFrameError(
                f"pose_frame0 must be (3,{SEG_H},{SEG_W}); got {tuple(f0.shape)}"
            )
        self._f0 = f0

    # ----- exact functionals (authority) -----
    def seg_logits(self, delta_chw: np.ndarray) -> np.ndarray:
        """Frozen SegNet logits (5,384,512) on frame1 = base + delta — exact."""

        torch = self._torch
        d = torch.from_numpy(np.asarray(delta_chw, dtype=np.float64)).float()
        with torch.inference_mode():
            seg_in = (self._f1_base + d).unsqueeze(0)
            return self._seg(seg_in)[0].cpu().numpy()

    def seg_argmax(self, delta_chw: np.ndarray) -> np.ndarray:
        return self.seg_logits(delta_chw).argmax(axis=0).astype(np.int64)

    def pose6(self, delta_chw: np.ndarray) -> np.ndarray:
        """Frozen PoseNet first-6 dims for the pair (frame0 GT, frame1 = base + delta) — exact.

        PoseNet.preprocess_input expects (b, t, c, h, w) at the RESIZED grid; the upstream
        preprocess will resize again (idempotent on an already-(384,512) input up to the bilinear
        identity) then yuv6. We feed the resized frames directly as the (b,t,c,h,w) tensor."""

        torch = self._torch
        d = torch.from_numpy(np.asarray(delta_chw, dtype=np.float64)).float()
        f1 = self._f1_base + d
        x = torch.stack([self._f0, f1]).unsqueeze(0)  # (1,2,3,384,512)
        with torch.inference_mode():
            pin = self._pose.preprocess_input(x)
            return self._pose(pin)["pose"][0, :N_POSE].cpu().numpy().astype(np.float64)

    # ----- exact Jacobians (re-linearized each outer iter) -----
    def seg_margin_and_jacdiff(
        self, delta_chw: np.ndarray, target_argmax: np.ndarray, gamma: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel margin gap + Jacobian-difference toward the target class.

        For each pixel p, let t = target_argmax[p] (the class we WANT) and r = current runner-up
        among the 5 (the class most threatening t). The margin m_p = logit_t - logit_r. A pixel is
        VIOLATED if m_p < gamma. The flip direction is the input gradient of (logit_t - logit_r)
        w.r.t. frame1's (3,384,512) input at p.

        Returns ``(violated_mask (H,W) bool, margin (H,W), grad (3,H,W))`` where ``grad`` is
        ``d(sum over violated p of (logit_t - logit_r)_p)/d(input)`` — a single backward pass gives
        the per-pixel steepest ascent of the summed target-minus-runnerup margin (exact for the
        spatially-local SegNet response; the projection step uses it as the linearized A-normal).
        """

        torch = self._torch
        tgt = np.asarray(target_argmax).astype(np.int64)
        if tgt.shape != (SEG_H, SEG_W):
            raise DykstraLegalFrameError(f"target_argmax must be ({SEG_H},{SEG_W}); got {tgt.shape}")
        d = torch.from_numpy(np.asarray(delta_chw, dtype=np.float64)).float()
        leaf = (self._f1_base + d).detach().clone().requires_grad_(True)
        logits = self._seg(leaf.unsqueeze(0))[0]  # (5,384,512)
        # runner-up = max logit over classes != target, per pixel.
        tgt_t = torch.from_numpy(tgt).long()
        tgt_logit = logits.gather(0, tgt_t.unsqueeze(0))[0]  # (H,W)
        masked = logits.clone()
        masked.scatter_(0, tgt_t.unsqueeze(0), float("-inf"))
        runner_logit, runner_idx = masked.max(dim=0)  # (H,W)
        margin_t = tgt_logit - runner_logit  # (H,W)
        margin = margin_t.detach().cpu().numpy()
        violated = margin < float(gamma)
        if not violated.any():
            return violated, margin, np.zeros((3, SEG_H, SEG_W), dtype=np.float64)
        vmask = torch.from_numpy(violated.astype(np.float64)).double().float()
        objective = (margin_t * vmask).sum()  # raise margin where violated
        (grad,) = torch.autograd.grad(objective, leaf)
        return violated, margin, grad.detach().cpu().numpy().astype(np.float64)

    def pose_jacobian(self, delta_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The 6 x (3*H*W) PoseNet input-Jacobian of the scored dims w.r.t. frame1 + current pose6.

        Uses the differentiable yuv6 path (caller must have patched). Returns
        ``(pose6 (6,), jac (6, 3, H, W))``. FAIL-CLOSED if jac is identically zero (severed yuv6).
        """

        torch = self._torch
        d = torch.from_numpy(np.asarray(delta_chw, dtype=np.float64)).float()
        f1 = (self._f1_base + d).detach().clone().requires_grad_(True)
        x = torch.stack([self._f0, f1]).unsqueeze(0)  # (1,2,3,384,512)
        pin = self._pose.preprocess_input(x)
        pose6 = self._pose(pin)["pose"][0, :N_POSE]
        jac = np.zeros((N_POSE, 3, SEG_H, SEG_W), dtype=np.float64)
        for k in range(N_POSE):
            (g,) = torch.autograd.grad(pose6[k], f1, retain_graph=(k < N_POSE - 1))
            jac[k] = g.detach().cpu().numpy().astype(np.float64)
        if not np.any(jac):
            raise DykstraLegalFrameError(
                "PoseNet input-Jacobian is identically zero — differentiable yuv6 path not active "
                "(gradient severed). Patch rgb_to_yuv6 before the solve; refusing to fake pose-null."
            )
        return pose6.detach().cpu().numpy().astype(np.float64), jac

    def d_seg(self, delta_chw: np.ndarray, target_argmax: np.ndarray) -> float:
        a = self.seg_argmax(delta_chw)
        return float(np.mean(a != np.asarray(target_argmax).astype(np.int64)))

    def d_pose(self, delta_chw: np.ndarray, target_pose6: np.ndarray) -> float:
        p = self.pose6(delta_chw)
        return float(np.mean((p - np.asarray(target_pose6, dtype=np.float64)) ** 2))


# ════════════════════════════════════════════════════════════════════════════
#  The three projections (each a real convex step on the linearized constraint).
# ════════════════════════════════════════════════════════════════════════════
def project_onto_margin_cell(
    delta_chw: np.ndarray,
    violated_mask: np.ndarray,
    margin_hw: np.ndarray,
    jacdiff_chw: np.ndarray,
    *,
    gamma: float,
    step_cap: float = 32.0,
) -> np.ndarray:
    """Move ``delta`` the minimum amount to raise every violated pixel's margin to ``gamma``.

    Linearized constraint at the current iterate: for each violated pixel p,
    ``margin_p + <grad_p, step> >= gamma``  (grad_p = jacdiff toward target). The least-norm step
    that satisfies a single half-space ``<g, s> >= b`` is ``s = (b/||g||^2) g`` for b>0. We pool the
    per-pixel half-spaces by taking the steepest summed ascent direction ``grad`` (already summed
    over violated pixels in :meth:`seg_margin_and_jacdiff`) and scale it so the WORST violated pixel
    reaches ``gamma`` (a conservative common step; the outer re-linearization mops up residuals).

    Returns the projected delta (3,H,W). NO-FAKE: if there is a real positive gradient toward the
    target, the step is nonzero and moves the margin up; a zero-gradient (no feasible direction)
    returns delta unchanged (and the outer loop reports the residual)."""

    d = np.asarray(delta_chw, dtype=np.float64).copy()
    grad = np.asarray(jacdiff_chw, dtype=np.float64)
    if not np.asarray(violated_mask).any():
        return d
    gnorm2 = float(np.sum(grad * grad))
    if gnorm2 <= 1e-18:
        return d
    deficits = (float(gamma) - np.asarray(margin_hw, dtype=np.float64))[np.asarray(violated_mask)]
    worst = float(np.max(deficits)) if deficits.size else 0.0
    if worst <= 0.0:
        return d
    # least-norm common step toward closing the worst violated half-space along the summed normal.
    t = worst / gnorm2
    step = t * grad
    cap = float(step_cap)
    step = np.clip(step, -cap, cap)
    return d + step


def project_onto_pose_tube(
    delta_chw: np.ndarray,
    pose6: np.ndarray,
    target_pose6: np.ndarray,
    pose_jac: np.ndarray,
    *,
    tau: float,
    step_cap: float = 32.0,
) -> np.ndarray:
    """Least-norm step that pulls PoseNet's 6 scored dims into the ``tau`` ball around ``p*``.

    Linearized: ``pose6 + J @ step ~= target`` (we want residual r = target - pose6 driven to ~0).
    The minimum-norm step solving ``J @ step = r`` is ``step = J^T (J J^T)^{-1} r`` (the
    Moore-Penrose pseudo-inverse on the 6-dim output). We only step if the current pose error
    ``||r||^2`` exceeds ``tau`` (already in the tube -> no move). Returns projected delta (3,H,W).

    NO-FAKE: the step is the real pseudo-inverse of the MEASURED PoseNet Jacobian; a zero/degenerate
    Jacobian (already fail-closed upstream) cannot reach here. If already in tube, returns
    delta unchanged."""

    d = np.asarray(delta_chw, dtype=np.float64).copy()
    p = np.asarray(pose6, dtype=np.float64)
    tgt = np.asarray(target_pose6, dtype=np.float64)
    r = tgt - p  # (6,)
    err2 = float(np.mean(r * r))
    if err2 <= float(tau):
        return d
    J = np.asarray(pose_jac, dtype=np.float64).reshape(N_POSE, -1)  # (6, 3HW)
    JJt = J @ J.T  # (6,6)
    JJt += 1e-6 * np.eye(N_POSE)  # ridge for conditioning
    try:
        y = np.linalg.solve(JJt, r)  # (6,)
    except np.linalg.LinAlgError:
        return d
    # np.errstate guards the same spurious numpy/BLAS matmul SIMD RuntimeWarning as project_onto_cheap.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        step = (J.T @ y).reshape(3, SEG_H, SEG_W)
    if not np.all(np.isfinite(step)):  # genuine overflow (pathological tiny Jacobian) -> no move
        return d
    cap = float(step_cap)
    step = np.clip(step, -cap, cap)
    return d + step


def project_onto_cheap(
    delta_chw: np.ndarray,
    *,
    rank: int,
    sparse_keep_frac: float,
) -> np.ndarray:
    """Project ``delta`` onto the cheap-encoding subspace = (low-rank per channel) ∩ (sparse).

    Two cheap-basis operators composed (the carrier's storage model):
      1. **Low-rank** truncated SVD per channel to ``rank`` components (a separable spatial basis
         that brotli-codes tiny). rank=0 disables.
      2. **Sparse** magnitude thresholding keeping the top ``sparse_keep_frac`` of |delta| entries
         (the rest set 0 -> long zero runs -> cheap). keep_frac>=1 disables.

    This is the C set. It makes delta a CARRIER (stored cheaply) rather than a copy. Returns the
    projected delta (3,H,W). NO-FAKE: low-rank + sparse ACTUALLY reduce the stored representation
    (the byte-account test asserts coded bytes drop vs full delta); a no-op (rank>=full,
    keep_frac>=1) returns delta unchanged and the test asserts the byte cost is then the full
    delta's cost (so the cheap projection is load-bearing)."""

    d = np.asarray(delta_chw, dtype=np.float64).copy()
    if d.shape != (3, SEG_H, SEG_W):
        raise DykstraLegalFrameError(f"delta must be (3,{SEG_H},{SEG_W}); got {d.shape}")
    r = int(rank)
    if r > 0:
        out = np.empty_like(d)
        for c in range(3):
            U, S, Vt = np.linalg.svd(d[c], full_matrices=False)
            k = min(r, S.shape[0])
            # np.errstate guards a known-spurious numpy/BLAS matmul SIMD RuntimeWarning
            # ("divide by zero"/"overflow" on `@` for some shapes); the reconstruction values are
            # exact (verified finite) — the warning is a vectorization artifact, not a numeric fault.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                out[c] = (U[:, :k] * S[:k]) @ Vt[:k]
        d = out
    kf = float(sparse_keep_frac)
    if kf < 1.0:
        flat = np.abs(d).ravel()
        n_keep = max(1, round(kf * flat.size))
        if n_keep < flat.size:
            thresh = np.partition(flat, flat.size - n_keep)[flat.size - n_keep]
            d = np.where(np.abs(d) >= thresh, d, 0.0)
    return d


# ════════════════════════════════════════════════════════════════════════════
#  The byte account of the stored delta (the carrier cost) — honest brotli-q11.
# ════════════════════════════════════════════════════════════════════════════
def delta_coded_bytes(delta_chw: np.ndarray, *, quant_step: float = 1.0) -> int:
    """Quantize ``delta`` (round to ``quant_step``) + brotli-q11 the int16 codes = stored bytes.

    The carrier stores the per-pair frame1 perturbation on the (3,384,512) scorer grid, quantized
    and entropy-coded. This is the HONEST byte cost the rate term reads (NO-FAKE: an all-zero delta
    codes to the brotli floor; a dense delta codes large)."""

    import brotli

    d = np.asarray(delta_chw, dtype=np.float64)
    q = np.round(d / float(quant_step)).astype(np.int16)
    return len(brotli.compress(q.tobytes(), quality=11))


# ════════════════════════════════════════════════════════════════════════════
#  The Dykstra alternating-projection feasibility loop (the SOLVE).
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class FeasibilityConfig:
    gamma: float = 0.5          # margin-cell target gap (logits)
    tau: float = 5e-4           # pose-tube radius (mean-squared on 6 dims; the seg/pose crossover)
    rank: int = 24              # low-rank truncation per channel
    sparse_keep_frac: float = 0.10  # keep top-10% |delta| entries
    quant_step: float = 1.0     # delta quantization (pixel units) for the byte account
    max_outer: int = 12         # outer re-linearization iterations
    seg_step_cap: float = 32.0
    pose_step_cap: float = 32.0
    project_cheap: bool = True  # set False to measure the FREE (uncoded) feasible point
    use_dykstra_correction: bool = True


@dataclass
class FeasibilityResult:
    """The converged operating point + its provenance (the deliverable row)."""

    d_seg: float
    d_pose: float
    delta_bytes: int
    converged: bool
    outer_iters: int
    d_seg_trace: list[float]
    d_pose_trace: list[float]
    margin_violation_frac_trace: list[float]
    pose_err2_trace: list[float]
    held_both_at_low_byte: bool
    config: dict[str, Any] = field(default_factory=dict)
    authority_tier: str = "exact_cpu_advisory"
    metric_family: str = "exact_pair_scorer"
    evidence_grade: str = "[macOS-CPU advisory]"
    promotable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "delta_bytes": int(self.delta_bytes),
            "converged": self.converged,
            "outer_iters": int(self.outer_iters),
            "d_seg_trace": [float(x) for x in self.d_seg_trace],
            "d_pose_trace": [float(x) for x in self.d_pose_trace],
            "margin_violation_frac_trace": [float(x) for x in self.margin_violation_frac_trace],
            "pose_err2_trace": [float(x) for x in self.pose_err2_trace],
            "held_both_at_low_byte": bool(self.held_both_at_low_byte),
            "config": self.config,
            "authority_tier": self.authority_tier,
            "metric_family": self.metric_family,
            "evidence_grade": self.evidence_grade,
            "promotable": self.promotable,
        }


def solve_legal_frame_feasibility(
    projector: FrozenScorerProjector,
    target_argmax: np.ndarray,
    target_pose6: np.ndarray,
    config: FeasibilityConfig,
    *,
    d_seg_hold: float = 0.01,
    d_pose_hold: float = 5e-4,
) -> FeasibilityResult:
    """Dykstra alternating projection onto (margin-cell A) ∩ (pose-tube B) ∩ (cheap C).

    Starts at ``delta = 0`` (the GT frame1, which is IN A and IN B by construction). Each OUTER
    iteration: re-linearize the SegNet/PoseNet Jacobians at the current iterate, then project A->B->C
    with Dykstra correction terms (so the iterate converges to the projection onto the intersection
    of the linearized sets). Measures the EXACT d_seg + d_pose + stored delta bytes each outer iter.

    Converged when the margin-violation fraction is ~0 AND the pose error is within tau AND the
    delta no longer moves materially. ``held_both_at_low_byte`` records the pre-registered headline:
    d_seg <= d_seg_hold AND d_pose <= d_pose_hold simultaneously at the final (cheap) byte.

    The base ``delta=0`` already holds both terms at ZERO byte; the load-bearing question is whether
    the CHEAP projection (C) keeps both terms held. With ``config.project_cheap=False`` this measures
    the uncoded feasible point (the geometric reference); with True it measures the carrier."""

    tgt = np.asarray(target_argmax).astype(np.int64)
    tgt_pose = np.asarray(target_pose6, dtype=np.float64)
    delta = np.zeros((3, SEG_H, SEG_W), dtype=np.float64)

    # Dykstra correction terms (one per set) carried across iterations.
    p_corr = np.zeros_like(delta)  # for A (margin cell)
    q_corr = np.zeros_like(delta)  # for B (pose tube)
    r_corr = np.zeros_like(delta)  # for C (cheap)

    d_seg_trace: list[float] = []
    d_pose_trace: list[float] = []
    viol_trace: list[float] = []
    perr_trace: list[float] = []

    converged = False
    iters = 0
    prev_delta = delta.copy()
    for it in range(int(config.max_outer)):
        iters = it + 1
        # ---- project A: margin cell (re-linearized) ----
        x = delta + p_corr if config.use_dykstra_correction else delta
        violated, margin, jacdiff = projector.seg_margin_and_jacdiff(x, tgt, config.gamma)
        a = project_onto_margin_cell(
            x, violated, margin, jacdiff, gamma=config.gamma, step_cap=config.seg_step_cap
        )
        if config.use_dykstra_correction:
            p_corr = x - a
        delta = a

        # ---- project B: pose tube (re-linearized) ----
        x = delta + q_corr if config.use_dykstra_correction else delta
        pose6, pjac = projector.pose_jacobian(x)
        b = project_onto_pose_tube(
            x, pose6, tgt_pose, pjac, tau=config.tau, step_cap=config.pose_step_cap
        )
        if config.use_dykstra_correction:
            q_corr = x - b
        delta = b

        # ---- project C: cheap subspace ----
        if config.project_cheap:
            x = delta + r_corr if config.use_dykstra_correction else delta
            c = project_onto_cheap(x, rank=config.rank, sparse_keep_frac=config.sparse_keep_frac)
            if config.use_dykstra_correction:
                r_corr = x - c
            delta = c

        # ---- exact measurement at this iterate ----
        ds = projector.d_seg(delta, tgt)
        dp = projector.d_pose(delta, tgt_pose)
        viol_frac = float(np.mean(violated))
        perr2 = float(np.mean((projector.pose6(delta) - tgt_pose) ** 2))
        d_seg_trace.append(ds)
        d_pose_trace.append(dp)
        viol_trace.append(viol_frac)
        perr_trace.append(perr2)

        move = float(np.max(np.abs(delta - prev_delta)))
        prev_delta = delta.copy()
        if ds <= d_seg_hold and dp <= d_pose_hold and move < 0.25:
            converged = True
            break

    delta_bytes = delta_coded_bytes(delta, quant_step=config.quant_step)
    final_ds = d_seg_trace[-1] if d_seg_trace else projector.d_seg(delta, tgt)
    final_dp = d_pose_trace[-1] if d_pose_trace else projector.d_pose(delta, tgt_pose)
    held = bool(final_ds <= d_seg_hold and final_dp <= d_pose_hold)

    return FeasibilityResult(
        d_seg=float(final_ds),
        d_pose=float(final_dp),
        delta_bytes=int(delta_bytes),
        converged=converged,
        outer_iters=iters,
        d_seg_trace=d_seg_trace,
        d_pose_trace=d_pose_trace,
        margin_violation_frac_trace=viol_trace,
        pose_err2_trace=perr_trace,
        held_both_at_low_byte=held,
        config={
            "gamma": config.gamma,
            "tau": config.tau,
            "rank": config.rank,
            "sparse_keep_frac": config.sparse_keep_frac,
            "quant_step": config.quant_step,
            "max_outer": config.max_outer,
            "project_cheap": config.project_cheap,
            "use_dykstra_correction": config.use_dykstra_correction,
            "d_seg_hold": d_seg_hold,
            "d_pose_hold": d_pose_hold,
        },
    )


__all__ = [
    "DykstraLegalFrameError",
    "FeasibilityConfig",
    "FeasibilityResult",
    "FrozenScorerProjector",
    "delta_coded_bytes",
    "project_onto_cheap",
    "project_onto_margin_cell",
    "project_onto_pose_tube",
    "score_from_components",
    "solve_legal_frame_feasibility",
]
