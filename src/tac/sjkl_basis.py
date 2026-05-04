"""Score-Jacobian Karhunen-Loève (SJ-KL) basis primitive.

Wave-Ω-1, Council #2 (FIELDS-MEDAL session 2026-05-01).
Memo: ~/.claude/projects/-Users-adpena-Projects-pact/memory/project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md
Council vote 22/22 GO.

Theoretical claim
-----------------
For a fixed downstream task scorer S(x) = 100 * seg_dist + sqrt(10 * pose_dist),
the optimal k-dim residual subspace for encoding (GT - renderer_output) is
spanned by the top-k eigenvectors of the empirical Fisher matrix

    F(x*) = 100 * J_seg(x*)^T * J_seg(x*) + 10 * J_pose(x*)^T * J_pose(x*)

where J_seg = d_logits/d_pixel for SegNet and J_pose = d_pose6/d_pixel for PoseNet.
This is the SJ-KL basis. It is *strictly better* than DCT (PR #67's choice) for
this particular contest because DCT is optimal for *pixel MSE* not score loss.

Practical insight (the SIMPLER parameterization Council Section 5.3 hinted at)
-----------------------------------------------------------------------------
The full Fisher F is (N x N) where N = 3*384*512 = 589,824. Materializing it
is infeasible (1.4 TB). BUT:

1. F is exactly rank `5*H_seg*W_seg + 6` because J_seg has 5 logit channels at
   SegNet's internal resolution and J_pose has only 6 pose dims. So F is
   ALWAYS LOW-RANK regardless of N.

2. We can apply F as a *linear operator* via two AD passes:
       Fv = 100 * J_seg^T (J_seg v)  +  10 * J_pose^T (J_pose v)
   Each "Fv" costs 1 JVP + 1 VJP per scorer = ~4 forward-pass-equivalents.

3. Lanczos / randomized SVD recovers the top-k eigenvectors using O(k+5)
   applications of Fv. For k=8, that's ~50 forward-pass-equivalents per
   compute call — cheap enough for compress-time.

Side-info budget for the archive
--------------------------------
- k=8 global eigenvectors at 384*512*3 = 589,824 entries each
- FP16: 8 * 589,824 * 2 = 9.4 MB raw — TOO BIG
- Stored as low-rank V[j] = u[j] outer v[j] with u in R^3, v in R^(384*512):
  each eigenvector is 3 + 384*512 ~= 196,611 entries; 8 * 196,611 * 2 (FP16) = 3.1 MB
  still too big.
- BEST: store eigenvectors as a small 2D field at a downsampled resolution
  (e.g., 96 x 128 spatial × 3 ch × k=8 = 294,912 FP16 entries = 590 KB raw,
  brotli-compressed to ~150-300 KB). Decoded by bilinear upsample to 384x512.
  THIS is too big still — eats the rate budget.
- PRACTICAL: Council Section 5.3's "10.4 KB total" target requires either
  1) the eigenvectors are themselves coarse (smooth, low-frequency, < 50 DCT
     coefficients each) so they brotli-compress to ~few hundred bytes,
  2) OR we ship the basis as a small CONV operating on neighborhood pixels
     (k * 3*5*5 = 75k weights for k=8, ~150 bytes FP4),
  3) OR we use a FIXED low-resolution mesh of basis vectors (e.g., 8 x 8 grid
     per channel, encoded as bilinear-interp 5*8*8*3 = 960 floats per
     eigenvector × 8 = 7,680 floats = 15 KB FP16 raw).

This module ships option (3) — coarse-grid eigenvectors stored at 32x24 spatial
(yielding 32*24*3 = 2,304 entries × k=8 = 18,432 FP16 entries = 36.9 KB raw,
brotli-compressed to ~10-15 KB). At decode, bilinear-upsample to 384x512.

Per-pair coefficients alpha[i,j] for i in [0,600), j in [0,k) at 6-bit qint:
600 * 8 * 6 = 28,800 bits = 3.6 KB.

Total side-info: ~13-18 KB. Acceptable per Council Section 5.3.

Public API
----------
- ``compute_sjkl_basis(segnet, posenet, frames, k=8, basis_grid_h=32,
   basis_grid_w=24)`` — Lanczos on the Fisher operator; returns coarse-grid
   eigenvectors plus their scale.
- ``pack_sjkl_basis(basis_coarse, scale)`` — FP16 + brotli.
- ``unpack_sjkl_basis(payload)`` — inverse of pack.
- ``encode_residual(r, basis_coarse, scale, alpha_bits=6)`` — quantized
  projections of (GT - renderer_output) onto the basis.
- ``decode_residual(alpha_q, alpha_min, alpha_step, basis_coarse, scale,
   target_h, target_w)`` — reconstruction.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SJKL_MAGIC = b"SJKL"  # 4-byte magic prefix for pack/unpack
SJKL_BASIS_QUANT_MAGIC = b"SJQ1"


# ---------------------------------------------------------------------------
# Fisher operator (the math core)
# ---------------------------------------------------------------------------

def _rgb_to_yuv6_diff(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Differentiable copy of upstream/frame_utils.py:50-78 rgb_to_yuv6.

    Upstream uses ``@torch.no_grad`` which severs the autograd graph and
    breaks the Fisher matvec. We re-implement the same math without the
    decorator. Verified math-equivalent (test in test_sjkl_basis.py).
    """
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, : 2 * H2, : 2 * W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    # NOTE: upstream uses .clamp_() (in-place) which is grad-OK but produces
    # a CompositeImplicitAutograd warning under torch>=2.4. We use clamp().
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (
        U[..., 0::2, 0::2] + U[..., 1::2, 0::2]
        + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2] + V[..., 1::2, 0::2]
        + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
    ) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def _posenet_preprocess_diff(x: torch.Tensor) -> torch.Tensor:
    """Differentiable copy of PoseNet.preprocess_input (upstream modules.py:70-74).

    The upstream method uses rgb_to_yuv6 which is decorated with @torch.no_grad,
    breaking autograd. This is the grad-friendly variant.
    """
    import einops as _einops

    batch_size, seq_len = x.shape[:2]
    x_flat = _einops.rearrange(x, "b t c h w -> (b t) c h w", b=batch_size, t=seq_len, c=3)
    x_rs = torch.nn.functional.interpolate(
        x_flat, size=(384, 512), mode="bilinear", align_corners=False
    )
    yuv = _rgb_to_yuv6_diff(x_rs)
    return _einops.rearrange(yuv, "(b t) c h w -> b (t c) h w", b=batch_size, t=seq_len, c=6)


def _segnet_logits(segnet: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run SegNet returning logits BEFORE argmax.

    x has shape (B, T, C, H, W) per upstream/modules.py:107-109 convention.
    Returns logits at SegNet's internal resolution.

    Both upstream SegNet.preprocess_input AND TinySegNetLike.preprocess_input
    are pure ``F.interpolate`` calls (no @torch.no_grad/@inference_mode), so
    the graph survives both call paths.
    """
    seg_in = segnet.preprocess_input(x)
    return segnet(seg_in)


def _posenet_pose6(posenet: torch.nn.Module, x_pair: torch.Tensor) -> torch.Tensor:
    """Run PoseNet returning the first 6 dims of the pose head.

    x_pair has shape (B, 2, C, H, W) — the two-frame pair.
    Returns (B, 6) per upstream/modules.py:82-84.

    Bypasses the @torch.no_grad in upstream rgb_to_yuv6 by using
    ``_posenet_preprocess_diff``.
    """
    if hasattr(posenet, "head") and not hasattr(posenet, "vision"):
        # TinyPoseNetLike fixture in tests — uses its own preprocess_input
        # which doesn't sever the graph
        out = posenet(posenet.preprocess_input(x_pair))
    else:
        # Real upstream PoseNet — use the grad-friendly preprocess
        pose_in = _posenet_preprocess_diff(x_pair)
        out = posenet(pose_in)
    return out["pose"][..., :6]


def fisher_matvec(
    segnet: Optional[torch.nn.Module],
    posenet: Optional[torch.nn.Module],
    x_anchor: torch.Tensor,
    v: torch.Tensor,
    *,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
) -> torch.Tensor:
    """Apply the score-weighted Fisher F(x_anchor) to vector v.

    Fv = seg_weight * J_seg^T (J_seg v) + pose_weight * J_pose^T (J_pose v).

    J = forward-mode (JVP), J^T = reverse-mode (VJP).

    x_anchor has shape (B=1, T=2, C=3, H, W) — the linearization point. The
    Fisher is computed wrt the LAST frame (matching SegNet) for the
    seg term, and wrt both frames for the pose term BUT we only project the
    output sensitivity onto the second frame (matching the actuator's
    use case, which only perturbs frame1/frame2 of the renderer output).

    v is the perturbation, shape (3, H, W) in pixel space, applied to the
    second frame slot (frame1 in renderer convention; the actuator's target).
    """
    assert x_anchor.dim() == 5 and x_anchor.shape[0] == 1, "x_anchor must be (1,T,C,H,W)"
    H, W = x_anchor.shape[-2], x_anchor.shape[-1]
    assert v.shape == (3, H, W), f"v shape {v.shape} != (3,{H},{W})"
    device = x_anchor.device
    dtype = x_anchor.dtype

    # Build a perturbation that lives in the second-frame slot (the actuator's
    # target slot in pr67_inflate.py:884). Zero on frame index 0.
    perturb = torch.zeros_like(x_anchor)
    perturb[0, 1] = v
    out = torch.zeros_like(v)

    if segnet is not None:
        # SegNet operates on x[:, -1] (last frame). Use JVP+VJP.
        def seg_fn(inp: torch.Tensor) -> torch.Tensor:
            return _segnet_logits(segnet, inp)

        # Forward mode: J_seg @ v  (perturbation pushed through model)
        with torch.enable_grad():
            x_in = x_anchor.detach().requires_grad_(True)
            logits = seg_fn(x_in)
            # VJP closure: u -> J_seg^T u (we'll compose J^T J as VJP(JVP(.)))
            jvp_out = torch.autograd.grad(
                outputs=logits,
                inputs=x_in,
                grad_outputs=torch.ones_like(logits),
                create_graph=True,
                retain_graph=True,
            )[0]  # placeholder; real Fv below uses functional API

        # Use the functional API which is cleaner: J^T J v via double backward.
        # NOTE: torch.func.jvp + .vjp combo is slow for large models. Instead
        # use the identity:  J^T J v = grad_x (||J(x) v||^2 / 2)
        # i.e., let g(x) = sum_i (J(x) v)_i * a_i; if a is detached,
        # grad_x g = J^T a; choose a = J(x) v computed via a separate forward.
        # This requires only 1 forward (with grad) + 1 backward.
        x_in = x_anchor.detach().requires_grad_(True)
        # Compute logits with grad
        logits = seg_fn(x_in)
        # Compute J v via a standalone JVP (no grad needed for `a`)
        with torch.no_grad():
            # We need J v; use torch.autograd.functional.jvp
            from torch.autograd.functional import jvp
            _, jv = jvp(seg_fn, x_anchor.detach(), perturb, create_graph=False, strict=False)
            jv_detached = jv.detach()
        # Now grad_x (sum(logits * jv_detached)) gives J^T jv = J^T J v
        scalar = (logits * jv_detached).sum()
        jtj_v = torch.autograd.grad(scalar, x_in, retain_graph=False, create_graph=False)[0]
        # Project J^T J v back onto the second-frame slot.
        out = out + seg_weight * jtj_v[0, 1]

    if posenet is not None:
        def pose_fn(inp: torch.Tensor) -> torch.Tensor:
            return _posenet_pose6(posenet, inp)

        x_in = x_anchor.detach().requires_grad_(True)
        pose = pose_fn(x_in)
        with torch.no_grad():
            from torch.autograd.functional import jvp
            _, jv = jvp(pose_fn, x_anchor.detach(), perturb, create_graph=False, strict=False)
            jv_detached = jv.detach()
        scalar = (pose * jv_detached).sum()
        jtj_v = torch.autograd.grad(scalar, x_in, retain_graph=False, create_graph=False)[0]
        out = out + pose_weight * jtj_v[0, 1]

    return out.detach()


# ---------------------------------------------------------------------------
# Lanczos / randomized power iteration for top-k eigenvectors
# ---------------------------------------------------------------------------

def lanczos_topk(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    k: int,
    *,
    n_iters: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    shape_hint: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lanczos algorithm for top-k eigenvalues/eigenvectors of a symmetric op.

    matvec: a callable taking a flat or shape-hinted vector and returning
            the same-shape result. The operator must be symmetric PSD.
    dim:    total dimension (flattened).
    k:      number of top eigenvectors to recover.
    n_iters: total Lanczos iterations (default 2*k+5 for safety).
    shape_hint: if provided, vectors are passed/returned in this shape;
                otherwise flat (dim,) tensors.

    Returns: (eigenvalues_topk, eigenvectors_topk) where eigvecs has shape
             (k, *shape_hint) or (k, dim) if no shape_hint.
    """
    if n_iters is None:
        n_iters = 2 * k + 5
    n_iters = min(n_iters, dim)
    if device is None:
        device = torch.device("cpu")

    g = torch.Generator(device="cpu").manual_seed(seed)
    if shape_hint is not None:
        v0_flat = torch.randn(dim, generator=g, dtype=dtype)
        v0 = v0_flat.reshape(shape_hint).to(device)
    else:
        v0 = torch.randn(dim, generator=g, dtype=dtype, device=device)

    # Normalize starting vector
    v0 = v0 / (v0.norm() + 1e-12)

    # Storage for Lanczos vectors and tridiagonal coefficients
    Q = []  # list of vectors, each in shape_hint or (dim,)
    alphas = []  # diagonal entries
    betas = []  # off-diagonal entries

    q_prev = torch.zeros_like(v0)
    q = v0
    beta = 0.0

    for j in range(n_iters):
        Q.append(q.clone())
        w = matvec(q)
        if shape_hint is not None:
            w_flat = w.reshape(-1)
            q_flat = q.reshape(-1)
            q_prev_flat = q_prev.reshape(-1)
        else:
            w_flat, q_flat, q_prev_flat = w, q, q_prev

        alpha = float(torch.dot(w_flat, q_flat))
        alphas.append(alpha)
        # w = w - alpha * q - beta * q_prev
        w_flat = w_flat - alpha * q_flat - beta * q_prev_flat
        # Re-orthogonalize against all previous (Gram-Schmidt — needed
        # numerically for high-precision top-k recovery).
        for q_old in Q:
            q_old_flat = q_old.reshape(-1)
            coef = float(torch.dot(w_flat, q_old_flat))
            w_flat = w_flat - coef * q_old_flat
        beta = float(w_flat.norm())
        if beta < 1e-10:
            break
        betas.append(beta)
        q_prev = q
        q = (w_flat / beta)
        if shape_hint is not None:
            q = q.reshape(shape_hint)

    # Build tridiagonal matrix and diagonalize
    m = len(alphas)
    T = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        T[i, i] = alphas[i]
    # Off-diagonals: only place beta_i on T[i, i+1] when i+1 < m. If
    # n_iters == dim, the last computed beta is "leftover" (no q_{m+1}
    # vector to multiply against) and is dropped.
    for i in range(min(len(betas), m - 1)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    eigvals, eigvecs = np.linalg.eigh(T)
    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Reconstruct top-k Ritz vectors: y_j = Q @ eigvec_j. If Lanczos
    # produced fewer than k eigenpairs (early termination on β=0 OR
    # n_iters < k), cap k_eff to the actual count so reshape doesn't blow up.
    if shape_hint is not None:
        Q_stack = torch.stack([q.reshape(-1) for q in Q], dim=0)  # (m, dim)
    else:
        Q_stack = torch.stack(Q, dim=0)
    k_eff = min(k, m)
    eigvecs_t = torch.from_numpy(eigvecs[:, :k_eff].copy()).to(dtype=Q_stack.dtype, device=Q_stack.device)
    # ritz[k_idx, dim] = sum over m of eigvecs[m, k_idx] * Q[m]
    ritz = eigvecs_t.t() @ Q_stack  # (k_eff, dim)
    if shape_hint is not None:
        ritz = ritz.reshape((k_eff,) + shape_hint)
    eigvals_topk = torch.from_numpy(eigvals[:k_eff].copy()).to(dtype=Q_stack.dtype)
    return eigvals_topk, ritz


# ---------------------------------------------------------------------------
# Coarse-grid eigenvector encoding (the practical archive primitive)
# ---------------------------------------------------------------------------

@dataclass
class SJKLBasis:
    """Coarse-grid representation of the SJ-KL basis.

    basis_coarse: (k, 3, grid_h, grid_w) FP32 tensor — eigenvectors at low
                  spatial resolution; bilinear-upsampled to (target_h, target_w)
                  at decode time.
    scale: (k,) FP32 — per-eigenvector amplitude (positive).
    target_h, target_w: int — reconstruction resolution (e.g., 384, 512).
    """

    basis_coarse: torch.Tensor
    scale: torch.Tensor
    target_h: int
    target_w: int

    def upsample(self) -> torch.Tensor:
        """Return (k, 3, target_h, target_w) at full resolution."""
        return F.interpolate(
            self.basis_coarse,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )

    def renormalize(self) -> "SJKLBasis":
        """Renormalize each upsampled eigenvector to unit L2 norm by scaling
        the coarse basis only. The ``scale`` field is LEFT UNCHANGED — it
        retains its eigenvalue-derived ordering (sqrt(λ) prior). The
        normalized basis is what ``project_residual`` and ``decode_residual``
        consume; the scale is a pure "amplitude prior" the encoder uses
        to set per-eigenvector coefficients, not part of the basis itself.
        """
        full = self.upsample()  # (k, 3, H, W)
        norms = full.flatten(1).norm(dim=1).clamp_min(1e-12)  # (k,)
        new_basis = self.basis_coarse / norms.view(-1, 1, 1, 1)
        # Note: scale is intentionally NOT modified — it's the eigenvalue
        # prior, which depends only on the Lanczos result, not the coarse-grid
        # storage choice.
        return SJKLBasis(
            basis_coarse=new_basis,
            scale=self.scale.clone(),
            target_h=self.target_h,
            target_w=self.target_w,
        )


def compute_sjkl_basis(
    segnet: Optional[torch.nn.Module],
    posenet: Optional[torch.nn.Module],
    frames: torch.Tensor,
    *,
    k: int = 8,
    basis_grid_h: int = 32,
    basis_grid_w: int = 24,
    n_lanczos: Optional[int] = None,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    seed: int = 0,
) -> SJKLBasis:
    """Compute the global SJ-KL basis over a representative frame set.

    frames: (N, 2, C=3, H, W) — N representative frame pairs. The Fisher is
            averaged across the N pairs, then top-k eigenvectors recovered.

    Returns SJKLBasis with eigenvectors stored at (basis_grid_h, basis_grid_w)
    coarse resolution. Bilinear-upsample at decode time.

    Per CLAUDE.md "MPS auth eval is NOISE": for production use this MUST run
    on CUDA. MPS is acceptable only for development.
    """
    assert frames.dim() == 5, "frames must be (N,2,C,H,W)"
    N, T, C, H, W = frames.shape
    assert T == 2 and C == 3, f"frames must be (N,2,3,H,W), got (N,{T},{C},H,W)"

    # The Fisher matvec averaged over N anchor frames:
    def avg_matvec(v: torch.Tensor) -> torch.Tensor:
        accum = torch.zeros_like(v)
        for i in range(N):
            x_anchor = frames[i : i + 1]  # (1, 2, 3, H, W)
            accum = accum + fisher_matvec(
                segnet, posenet, x_anchor, v,
                seg_weight=seg_weight, pose_weight=pose_weight,
            )
        return accum / N

    dim = 3 * H * W
    eigvals, eigvecs_full = lanczos_topk(
        matvec=avg_matvec,
        dim=dim,
        k=k,
        n_iters=n_lanczos,
        device=frames.device,
        dtype=torch.float32,
        seed=seed,
        shape_hint=(3, H, W),
    )

    # Downsample eigenvectors to coarse grid (bilinear avg-pool style).
    eigvecs_coarse = F.interpolate(
        eigvecs_full,  # (k, 3, H, W)
        size=(basis_grid_h, basis_grid_w),
        mode="bilinear",
        align_corners=False,
    )
    # Use sqrt(eigval) as a per-eigenvector amplitude prior (Fisher
    # eigenvalues approximate inverse rate-distortion-curvature). Clamp non-
    # positive (numerical noise) to 0.
    scale = eigvals.clamp_min(0.0).sqrt()

    basis = SJKLBasis(
        basis_coarse=eigvecs_coarse,
        scale=scale,
        target_h=H,
        target_w=W,
    )
    return basis.renormalize()


# ---------------------------------------------------------------------------
# Pack / unpack of basis into bytes
# ---------------------------------------------------------------------------

def _pack_bits(values: np.ndarray, *, bits: int) -> bytes:
    """Pack unsigned integer values in little-endian bit order."""
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    if flat.size == 0:
        return b""
    max_value = (1 << bits) - 1
    if int(flat.max()) > max_value:
        raise ValueError(f"value exceeds {bits}-bit range")
    out = bytearray((int(flat.size) * bits + 7) // 8)
    bit_pos = 0
    for value in flat.tolist():
        value = int(value)
        byte_idx = bit_pos // 8
        offset = bit_pos % 8
        out[byte_idx] |= (value << offset) & 0xFF
        remaining = bits - (8 - offset)
        spill = value >> (8 - offset)
        cursor = byte_idx + 1
        while remaining > 0:
            out[cursor] |= spill & 0xFF
            spill >>= 8
            remaining -= 8
            cursor += 1
        bit_pos += bits
    return bytes(out)


def _unpack_bits(payload: bytes, *, count: int, bits: int) -> np.ndarray:
    """Inverse of _pack_bits."""
    expected = (count * bits + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"bitpacked basis length mismatch: expected {expected}, got {len(payload)}")
    out = np.zeros(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    bit_pos = 0
    for idx in range(count):
        byte_idx = bit_pos // 8
        offset = bit_pos % 8
        window = 0
        for rel in range(4):
            if byte_idx + rel < len(payload):
                window |= int(payload[byte_idx + rel]) << (8 * rel)
        out[idx] = (window >> offset) & mask
        bit_pos += bits
    return out


def pack_sjkl_basis(
    basis: SJKLBasis,
    *,
    brotli_quality: int = 11,
    basis_quant_bits: Optional[int] = 6,
) -> bytes:
    """Serialize SJKLBasis to a brotli-compressed byte payload.

    Legacy lossless layout (``basis_quant_bits=None``):
      MAGIC(4) | k(uint16) | grid_h(uint16) | grid_w(uint16) | tgt_h(uint16) | tgt_w(uint16)
        | scale_fp16[k] | basis_fp16[k * 3 * grid_h * grid_w]

    Default compact layout:
      MAGIC(4) | brotli(SJQ1 | quant_bits | dims | scale_fp16[k] |
        basis_absmax_fp16[k] | bitpacked signed quantized basis)

    Then brotli-compress the whole blob (excluding MAGIC).

    Output:
      MAGIC(4) | brotli_payload
    """
    import brotli

    k = basis.basis_coarse.shape[0]
    grid_h = basis.basis_coarse.shape[2]
    grid_w = basis.basis_coarse.shape[3]
    scale_fp16 = basis.scale.detach().to(torch.float16).cpu().numpy().tobytes()
    if basis_quant_bits is None:
        header = struct.pack(
            "<HHHHH",
            k, grid_h, grid_w, basis.target_h, basis.target_w,
        )
        basis_fp16 = basis.basis_coarse.detach().to(torch.float16).cpu().numpy().tobytes()
        raw = header + scale_fp16 + basis_fp16
    else:
        bits = int(basis_quant_bits)
        if bits < 2 or bits > 8:
            raise ValueError(f"basis_quant_bits must be in [2, 8] or None, got {basis_quant_bits}")
        basis_cpu = basis.basis_coarse.detach().to(torch.float32).cpu()
        flat = basis_cpu.flatten(1)
        basis_absmax = flat.abs().max(dim=1).values.clamp_min(1e-12)
        levels = (1 << (bits - 1)) - 1
        offset = 1 << (bits - 1)
        q = torch.clamp(
            torch.round(basis_cpu / basis_absmax.view(-1, 1, 1, 1) * levels),
            -levels,
            levels,
        ).to(torch.int16)
        q_unsigned = (q.reshape(-1).cpu().numpy().astype(np.int16) + offset).astype(np.uint8)
        header = SJKL_BASIS_QUANT_MAGIC + struct.pack(
            "<BBHHHHH",
            bits,
            0,
            k,
            grid_h,
            grid_w,
            basis.target_h,
            basis.target_w,
        )
        raw = (
            header
            + scale_fp16
            + basis_absmax.to(torch.float16).numpy().tobytes()
            + _pack_bits(q_unsigned, bits=bits)
        )
    compressed = brotli.compress(raw, quality=brotli_quality)
    return SJKL_MAGIC + compressed


def unpack_sjkl_basis(payload: bytes) -> SJKLBasis:
    """Inverse of pack_sjkl_basis."""
    import brotli

    if payload[:4] != SJKL_MAGIC:
        raise ValueError(
            f"bad magic: expected {SJKL_MAGIC!r}, got {payload[:4]!r}"
        )
    raw = brotli.decompress(payload[4:])
    if raw[:4] == SJKL_BASIS_QUANT_MAGIC:
        bits, _reserved, k, grid_h, grid_w, target_h, target_w = struct.unpack(
            "<BBHHHHH",
            raw[4:16],
        )
        if bits < 2 or bits > 8:
            raise ValueError(f"unsupported quantized SJ-KL basis bits: {bits}")
        cursor = 16
        scale_bytes = 2 * k
        scale = torch.from_numpy(
            np.frombuffer(raw[cursor : cursor + scale_bytes], dtype=np.float16).copy()
        ).to(torch.float32)
        cursor += scale_bytes
        absmax = torch.from_numpy(
            np.frombuffer(raw[cursor : cursor + scale_bytes], dtype=np.float16).copy()
        ).to(torch.float32)
        cursor += scale_bytes
        basis_n = k * 3 * grid_h * grid_w
        q_unsigned = _unpack_bits(raw[cursor:], count=basis_n, bits=bits).astype(np.int16)
        levels = (1 << (bits - 1)) - 1
        offset = 1 << (bits - 1)
        q = torch.from_numpy(q_unsigned - offset).to(torch.float32).reshape(k, 3, grid_h, grid_w)
        basis_coarse = q / float(levels) * absmax.view(-1, 1, 1, 1)
        return SJKLBasis(
            basis_coarse=basis_coarse,
            scale=scale,
            target_h=int(target_h),
            target_w=int(target_w),
        )

    k, grid_h, grid_w, target_h, target_w = struct.unpack("<HHHHH", raw[:10])
    cursor = 10
    scale_bytes = 2 * k
    scale = torch.from_numpy(
        np.frombuffer(raw[cursor : cursor + scale_bytes], dtype=np.float16).copy()
    ).to(torch.float32)
    cursor += scale_bytes
    basis_n = k * 3 * grid_h * grid_w
    basis_bytes = 2 * basis_n
    basis_coarse = torch.from_numpy(
        np.frombuffer(raw[cursor : cursor + basis_bytes], dtype=np.float16).copy()
    ).to(torch.float32).reshape(k, 3, grid_h, grid_w)
    return SJKLBasis(
        basis_coarse=basis_coarse,
        scale=scale,
        target_h=int(target_h),
        target_w=int(target_w),
    )


# ---------------------------------------------------------------------------
# Per-pair residual encoding (alpha[i, j])
# ---------------------------------------------------------------------------

def project_residual(r: torch.Tensor, basis: SJKLBasis) -> torch.Tensor:
    """Project residual r (shape (3, H, W)) onto the basis.

    Returns alpha (shape (k,)) such that r ~= sum_j alpha[j] * scale[j] *
    full_basis[j].
    """
    full = basis.upsample()  # (k, 3, H, W)
    # alpha[j] = <r, scale[j] * full[j]> / <scale[j] * full[j], scale[j] * full[j]>
    # Since renormalize() makes ||full[j]||=1, the denominator is scale[j]^2.
    # We DEFINE alpha as the coefficient such that r ~= sum alpha[j] * (scale[j] * full[j]),
    # so alpha[j] = <r, scale[j] * full[j]> / scale[j]^2 = <r, full[j]> / scale[j].
    rj = (r.unsqueeze(0) * full).flatten(1).sum(dim=1)  # (k,)
    safe_scale = basis.scale.clamp_min(1e-12)
    return rj / safe_scale


def encode_residual(
    r: torch.Tensor,
    basis: SJKLBasis,
    *,
    alpha_bits: int = 6,
) -> Tuple[np.ndarray, float, float]:
    """Project, then quantize alpha to alpha_bits per coefficient.

    Returns (alpha_q (uint), alpha_min (float), alpha_step (float)).
    Reconstruction: alpha_dequant = alpha_min + alpha_step * alpha_q
    """
    alpha = project_residual(r, basis).detach().cpu().numpy().astype(np.float32)
    levels = (1 << alpha_bits) - 1
    a_min = float(alpha.min())
    a_max = float(alpha.max())
    if a_max == a_min:
        a_step = 1.0
        a_q = np.zeros_like(alpha, dtype=_uint_dtype_for_bits(alpha_bits))
        return a_q, a_min, a_step
    a_step = (a_max - a_min) / levels
    a_q = np.clip(np.round((alpha - a_min) / a_step), 0, levels).astype(
        _uint_dtype_for_bits(alpha_bits)
    )
    return a_q, a_min, a_step


def _uint_dtype_for_bits(bits: int):
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    raise ValueError(f"alpha_bits {bits} > 16 not supported")


def decode_residual(
    alpha_q: np.ndarray,
    alpha_min: float,
    alpha_step: float,
    basis: SJKLBasis,
) -> torch.Tensor:
    """Reconstruct residual from quantized alpha.

    Returns r_hat shape (3, target_h, target_w).
    """
    alpha = torch.from_numpy(
        alpha_min + alpha_step * alpha_q.astype(np.float32)
    ).to(torch.float32)
    full = basis.upsample()  # (k, 3, H, W)
    # r_hat = sum_j alpha[j] * scale[j] * full[j]
    weights = (alpha * basis.scale).view(-1, 1, 1, 1)
    return (weights * full).sum(dim=0)


# ---------------------------------------------------------------------------
# Effective-rank helper (for the Fisher-rank empirical claim)
# ---------------------------------------------------------------------------

def effective_rank(eigenvalues: torch.Tensor, threshold: float = 1e-6) -> int:
    """Number of eigenvalues > threshold * max(eigenvalues).

    Used to verify the Fisher-low-rank claim from Council Section 5.3 (`assert
    effective_rank < 100` on a sample frame).
    """
    if eigenvalues.numel() == 0:
        return 0
    max_ev = float(eigenvalues.abs().max())
    if max_ev == 0:
        return 0
    return int((eigenvalues > threshold * max_ev).sum())
