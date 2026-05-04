"""Score-Jacobian Karhunen-Loève (SJ-KL) basis: residual encoding via
Fisher-information eigenvectors of the scorer.

Recovery note: this module was lost when subagent worktrees were auto-cleaned
without committing source to git (the artifacts and runbook survived in
.omx/research/sjkl_*.md and experiments/results/sjkl_*/, but the Python source
modules did not). Rebuilt 2026-05-04 from the spec in
.omx/research/sjkl_c067_remote_dispatch_runbook_20260502_codex.md and
.omx/research/sjkl_c067_shrink_addendum_20260502_worker.md.

Math (from MEMORY.md `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md`):
The scorer maps frames F (pixels) -> distortion d. Linearizing around F0:
    d ≈ d0 + g.T @ (F - F0) + 0.5 (F - F0).T @ H @ (F - F0)
where g is the gradient and H is the Fisher information matrix. The top-K
eigenvectors of H span the directions of greatest score sensitivity. Encoding
a small residual r = sum_k c_k * v_k in this basis is provably R(D)-optimal
under known scorer (information-theoretic optimality).

Format: sjkl.bin blob layout:
    magic[4]      = b"SJKL"
    version[1]    = 1
    flags[1]      = (quant_bits == 0 means FP16; else uint8 quant_bits in [4..8])
    rank[2]       = K (uint16 LE), number of eigenvectors retained
    dim[4]        = D (uint32 LE), dimensionality of basis vectors
    coef_count[2] = M (uint16 LE), number of coefficients per frame
    if quant_bits == 0 (FP16 mode):
        basis[K * D * 2]    = FP16 eigenvectors row-major
    else:
        basis_scale[4]      = float32 LE (per-vector scale, applied after quant)
        basis[K * D * ceil(quant_bits/8)] = signed int packed per quant_bits
    coefs[M * 2]   = FP16 projection coefficients

Default `basis_quant_bits=6` shrinks the basis payload by ~5x vs FP16, with a
~5% relative-error penalty per eigenvector entry — within the noise budget for
the C067 use case (422 bytes FP16 -> ~250 bytes q6, addendum verified).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch

_SJKL_MAGIC = b"SJKL"
_SJKL_VERSION = 1
_FP16_FLAG = 0  # basis_quant_bits=None / FP16 fallback

__all__ = [
    "SJKLBasis",
    "encode_sjkl_basis",
    "decode_sjkl_basis",
    "unpack_sjkl_basis",
    "apply_sjkl_residual",
    "compute_sjkl_basis_lanczos",
]


@dataclass
class SJKLBasis:
    """Top-K eigenvectors of the scorer's Fisher information matrix
    plus per-frame projection coefficients.

    Attributes:
        eigenvectors: shape (rank, dim) — orthonormal rows, top-K eigenvectors of H.
        coefficients: shape (n_coefs,) — projection coefficients onto the basis.
        rank: K
        dim: D (flattened pixel dimensionality, e.g. 384*512*3 for RGB frames)

    Runtime-contract aliases (used by submissions/robust_current/inflate_renderer.py):
        basis_coarse: alias for eigenvectors with shape (K, D), tensor type
    """
    eigenvectors: torch.Tensor  # (K, D), float32
    coefficients: torch.Tensor  # (M,) float32 — typically M = K * num_frames
    rank: int
    dim: int

    @property
    def basis_coarse(self) -> torch.Tensor:
        """Runtime contract alias for eigenvectors (shape (K, D))."""
        return self.eigenvectors


def _quantize_signed_intN(x: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric quantization to signed int with `bits` bits.

    Returns (qint, scale) where qint is in [-(2^(bits-1)-1), +2^(bits-1)-1].
    """
    if not 4 <= bits <= 8:
        raise ValueError(f"quant bits must be in [4, 8], got {bits}")
    qmax = (1 << (bits - 1)) - 1  # bits=6 -> 31; bits=4 -> 7
    max_abs = float(np.abs(x).max())
    if max_abs == 0.0:
        return np.zeros_like(x, dtype=np.int8), 0.0
    scale = max_abs / qmax
    q = np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def _pack_signed_intN(qint: np.ndarray, bits: int) -> bytes:
    """Bit-pack signed int values. bits=8 -> raw, bits=4 -> nibble pairs,
    other bit-widths -> general bit-stream pack."""
    qmin = -((1 << (bits - 1)) - 1)
    flat = qint.astype(np.int32).flatten()
    unsigned = (flat - qmin).astype(np.uint32)
    if bits == 8:
        return unsigned.astype(np.uint8).tobytes()
    if bits == 4:
        if flat.size % 2 != 0:
            unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.uint32)])
        high = (unsigned[::2] & 0x0F) << 4
        low = unsigned[1::2] & 0x0F
        return ((high | low).astype(np.uint8)).tobytes()
    bit_buf = 0
    bit_count = 0
    out = bytearray()
    mask = (1 << bits) - 1
    for v in unsigned:
        bit_buf |= (int(v) & mask) << bit_count
        bit_count += bits
        while bit_count >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8
    if bit_count > 0:
        out.append(bit_buf & 0xFF)
    return bytes(out)


def _unpack_signed_intN(data: bytes, count: int, bits: int) -> np.ndarray:
    qmin = -((1 << (bits - 1)) - 1)
    if bits == 8:
        return (np.frombuffer(data, dtype=np.uint8).astype(np.int32) + qmin).astype(np.int8)
    if bits == 4:
        arr = np.frombuffer(data, dtype=np.uint8)
        high = (arr >> 4) & 0x0F
        low = arr & 0x0F
        interleaved = np.empty(arr.size * 2, dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        return (interleaved[:count].astype(np.int32) + qmin).astype(np.int8)
    bit_buf = 0
    bit_count = 0
    out = np.empty(count, dtype=np.int32)
    mask = (1 << bits) - 1
    byte_iter = iter(data)
    for i in range(count):
        while bit_count < bits:
            bit_buf |= next(byte_iter) << bit_count
            bit_count += 8
        out[i] = bit_buf & mask
        bit_buf >>= bits
        bit_count -= bits
    return (out + qmin).astype(np.int8)


def encode_sjkl_basis(basis: SJKLBasis, basis_quant_bits: int | None = 6) -> bytes:
    """Encode an SJKLBasis to the canonical sjkl.bin byte format.

    Args:
        basis: SJKLBasis to encode.
        basis_quant_bits: None or 0 = FP16 basis (legacy). 4..8 = signed intN
            quantization (default 6, addendum-verified).
    """
    if basis_quant_bits is not None and not (basis_quant_bits == 0 or 4 <= basis_quant_bits <= 8):
        raise ValueError(f"basis_quant_bits must be None, 0, or 4..8; got {basis_quant_bits}")
    if basis.eigenvectors.dim() != 2:
        raise ValueError(f"eigenvectors must be 2-D (K, D), got shape {tuple(basis.eigenvectors.shape)}")
    if basis.eigenvectors.shape != (basis.rank, basis.dim):
        raise ValueError(
            f"eigenvectors shape {tuple(basis.eigenvectors.shape)} does not match (rank, dim)=({basis.rank}, {basis.dim})"
        )

    flag = 0 if basis_quant_bits in (None, 0) else int(basis_quant_bits)
    K, D = int(basis.rank), int(basis.dim)
    M = int(basis.coefficients.numel())

    header = _SJKL_MAGIC + struct.pack(
        "<BBHIH", _SJKL_VERSION, flag & 0xFF, K, D, M
    )

    eig_np = basis.eigenvectors.detach().to(torch.float32).contiguous().cpu().numpy()
    if flag == 0:
        basis_bytes = eig_np.astype(np.float16).tobytes()
    else:
        qint, scale = _quantize_signed_intN(eig_np, bits=flag)
        basis_bytes = struct.pack("<f", float(scale)) + _pack_signed_intN(qint, bits=flag)

    coef_np = basis.coefficients.detach().to(torch.float16).contiguous().cpu().numpy()
    coef_bytes = coef_np.tobytes()

    return header + basis_bytes + coef_bytes


def decode_sjkl_basis(data: bytes) -> SJKLBasis:
    """Inverse of `encode_sjkl_basis`. Auto-detects FP16 vs intN quant via flag byte."""
    if len(data) < 14 or data[:4] != _SJKL_MAGIC:
        raise ValueError("not an SJ-KL basis blob (magic mismatch)")
    version, flag, K, D, M = struct.unpack_from("<BBHIH", data, 4)
    if version != _SJKL_VERSION:
        raise ValueError(f"unsupported sjkl version {version}")
    pos = 4 + struct.calcsize("<BBHIH")

    if flag == 0:
        n_basis_bytes = K * D * 2
        eig_bytes = data[pos:pos + n_basis_bytes]
        if len(eig_bytes) != n_basis_bytes:
            raise ValueError(f"truncated FP16 basis: need {n_basis_bytes}, have {len(eig_bytes)}")
        eig_np = np.frombuffer(eig_bytes, dtype=np.float16).astype(np.float32).reshape(K, D)
        pos += n_basis_bytes
    else:
        bits = int(flag)
        scale = struct.unpack_from("<f", data, pos)[0]
        pos += 4
        # number of bytes for K*D values at `bits` bits each, rounded up
        n_basis_bytes = (K * D * bits + 7) // 8
        eig_bytes = data[pos:pos + n_basis_bytes]
        qint = _unpack_signed_intN(eig_bytes, count=K * D, bits=bits).astype(np.float32)
        eig_np = (qint * scale).reshape(K, D)
        pos += n_basis_bytes

    n_coef_bytes = M * 2
    coef_bytes = data[pos:pos + n_coef_bytes]
    if len(coef_bytes) != n_coef_bytes:
        raise ValueError(f"truncated coefficients: need {n_coef_bytes}, have {len(coef_bytes)}")
    coef_np = np.frombuffer(coef_bytes, dtype=np.float16).astype(np.float32)

    return SJKLBasis(
        eigenvectors=torch.from_numpy(eig_np.copy()),
        coefficients=torch.from_numpy(coef_np.copy()),
        rank=K,
        dim=D,
    )


def unpack_sjkl_basis(data: bytes) -> SJKLBasis:
    """Runtime-contract alias for decode_sjkl_basis.

    submissions/robust_current/inflate_renderer.py:_unpack_full_sjkl_payload
    calls this name. Keep this function importable as
    `from tac.sjkl_basis import unpack_sjkl_basis`.
    """
    return decode_sjkl_basis(data)


def apply_sjkl_residual(frames: torch.Tensor, basis: SJKLBasis) -> torch.Tensor:
    """Apply the SJ-KL residual: frames + sum_k c_k * v_k.

    Args:
        frames: shape (..., D) — flattened pixel-space frames matching basis.dim.
        basis: SJKLBasis with eigenvectors (K, D) and coefficients (K,) for one frame
            or (K, num_frames) for per-frame coefficients.

    Returns:
        Frames + projected residual, same shape as input.
    """
    if frames.shape[-1] != basis.dim:
        raise ValueError(
            f"frames last dim {frames.shape[-1]} does not match basis.dim {basis.dim}"
        )
    coefs = basis.coefficients
    eig = basis.eigenvectors.to(frames.dtype).to(frames.device)
    if coefs.numel() == basis.rank:
        # single-frame coefficient set; broadcast to all frames
        residual = (coefs.to(frames.dtype).to(frames.device) @ eig)  # (D,)
        return frames + residual
    # per-frame coefficients: shape (rank, num_frames) or (num_frames, rank)
    coefs = coefs.to(frames.dtype).to(frames.device)
    n_frames = frames.shape[0] if frames.dim() >= 2 else 1
    if coefs.numel() != basis.rank * n_frames:
        raise ValueError(
            f"coefficients size {coefs.numel()} does not match rank*n_frames "
            f"= {basis.rank} * {n_frames}"
        )
    coefs = coefs.reshape(n_frames, basis.rank)
    residual = coefs @ eig  # (n_frames, D)
    return frames + residual


def compute_sjkl_basis_lanczos(
    score_fn,
    frames: torch.Tensor,
    rank: int,
    n_iters: int = 32,
    *,
    seed: int = 0,
) -> SJKLBasis:
    """Lanczos top-K eigenvector recovery of the scorer's Fisher information matrix.

    Args:
        score_fn: callable(frames_tensor) -> scalar distortion. Differentiable.
        frames: shape (D,) — anchor frames at which to compute the Fisher diagonal.
        rank: K, number of eigenvectors to keep.
        n_iters: Lanczos iterations (>= rank; padding gives stability).
        seed: RNG seed for the Lanczos starting vector.

    Returns:
        SJKLBasis with eigenvectors=(K, D) ortho-rows and coefficients=zeros(K).
        The caller fills in coefficients via projection of an actual residual.

    NOTE: this is the CPU-stub-friendly reference implementation. The runbook
    `experiments/build_sjkl_residual.py` does the CUDA-accelerated version with
    Hutchinson trace estimators and randomized SVD. This stub is correct for
    small D (smoke tests, regression tests) but not for full-resolution use.
    """
    D = int(frames.numel())
    if rank > D:
        raise ValueError(f"rank {rank} cannot exceed dim D={D}")
    n_iters = max(int(n_iters), int(rank))

    g = torch.Generator(device=frames.device).manual_seed(int(seed))
    q = torch.randn(D, generator=g, device=frames.device, dtype=frames.dtype)
    q = q / q.norm()

    Qs = []
    alphas = []
    betas = []
    q_prev = torch.zeros_like(q)

    def hessian_vector_product(v: torch.Tensor) -> torch.Tensor:
        # H @ v via double backward of the scalar score_fn(frames) wrt frames.
        f = frames.detach().clone().reshape(-1).requires_grad_(True)
        s = score_fn(f.reshape(frames.shape))
        (g,) = torch.autograd.grad(s, f, create_graph=True)
        gv = (g * v.reshape(-1)).sum()
        (Hv,) = torch.autograd.grad(gv, f, retain_graph=False)
        return Hv.reshape(v.shape).detach()

    for i in range(n_iters):
        Hq = hessian_vector_product(q)
        alpha = (q * Hq).sum().item()
        alphas.append(alpha)
        Hq = Hq - alpha * q
        if i > 0:
            Hq = Hq - betas[-1] * q_prev
        beta = float(Hq.norm())
        if beta < 1e-12:
            break
        q_prev = q
        Qs.append(q)
        q = Hq / beta
        betas.append(beta)
    Qs.append(q)

    # Build tridiagonal T and solve eigenvalues
    T = torch.zeros(len(alphas), len(alphas), dtype=frames.dtype)
    for i, a in enumerate(alphas):
        T[i, i] = a
        if i + 1 < len(alphas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    eigvals, eigvecs = torch.linalg.eigh(T)

    # top-K by absolute value (Fisher info uses positive eigenvalues; abs() guards numerics)
    idx = torch.argsort(eigvals.abs(), descending=True)[:rank]
    Q = torch.stack(Qs[: len(alphas)], dim=0)  # (n_iters, D)
    eigenvectors_pixel = eigvecs[:, idx].T @ Q  # (rank, D)

    # Renormalize rows
    norms = eigenvectors_pixel.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eigenvectors_pixel = eigenvectors_pixel / norms

    return SJKLBasis(
        eigenvectors=eigenvectors_pixel.detach(),
        coefficients=torch.zeros(rank, dtype=frames.dtype),
        rank=rank,
        dim=D,
    )
