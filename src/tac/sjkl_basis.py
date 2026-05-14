# SPDX-License-Identifier: MIT
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

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

_SJKL_MAGIC = b"SJKL"
_SJKL_BLOCK_MAGIC = b"SJKB"      # legacy V1 dense alpha block (matches inflate_renderer.py)
_SJKL_BLOCK_V2_MAGIC = b"SJK2"   # V2 sparse bit-packed alpha block with pair indices
_SJKL_META_MAGIC = b"SJBM"       # optional basis metadata trailer: shape + per-vector scale
_SJKL_VERSION = 1
_FP16_FLAG = 0  # basis_quant_bits=None / FP16 fallback

__all__ = [
    "SJKLBasis",
    "apply_sjkl_residual",
    "compute_sjkl_basis_lanczos",
    "decode_full_sjkl_payload",
    "decode_sjkl_alpha_block",
    "decode_sjkl_basis",
    "effective_rank",
    "encode_full_sjkl_payload",
    "encode_sjkl_alpha_block_v1_dense",
    "encode_sjkl_alpha_block_v2_sparse",
    "encode_sjkl_basis",
    "fisher_matvec",
    "lanczos_topk",
    "pack_sjkl_basis",
    "unpack_sjkl_basis",
]


@dataclass(init=False)
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
    scale: torch.Tensor         # (K,) float32, applied by inflate to decoded alpha coefficients
    target_h: int
    target_w: int
    _basis_coarse_shape: tuple[int, int, int, int] | None

    def __init__(
        self,
        eigenvectors: torch.Tensor | None = None,
        coefficients: torch.Tensor | None = None,
        rank: int | None = None,
        dim: int | None = None,
        *,
        basis_coarse: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        target_h: int | None = None,
        target_w: int | None = None,
        basis_coarse_shape: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Create either a flat basis or an inflate-runtime shaped basis.

        The original codec stores flat eigenvectors `(K, D)`. The robust_current
        inflate path additionally expects `basis_coarse`, `scale`, target shape,
        and `upsample()`. Supporting both here keeps the on-disk codec reusable
        while restoring the runtime contract for charged `sjkl.bin` payloads.
        """
        if basis_coarse is not None:
            if eigenvectors is not None:
                raise ValueError("pass either eigenvectors or basis_coarse, not both")
            coarse = basis_coarse.detach().to(torch.float32).contiguous()
            if coarse.dim() != 4:
                raise ValueError(f"basis_coarse must be 4-D (K, C, H, W), got {tuple(coarse.shape)}")
            coarse_k, coarse_c, coarse_h, coarse_w = (int(x) for x in coarse.shape)
            if coarse_c != 3:
                raise ValueError(f"basis_coarse channel count must be 3, got {coarse_c}")
            inferred_dim = coarse_c * coarse_h * coarse_w
            if rank is not None and int(rank) != coarse_k:
                raise ValueError(f"rank {rank} does not match basis_coarse K={coarse_k}")
            if dim is not None and int(dim) != inferred_dim:
                raise ValueError(f"dim {dim} does not match basis_coarse C*H*W={inferred_dim}")
            eigenvectors = coarse.reshape(coarse_k, inferred_dim)
            rank = coarse_k
            dim = inferred_dim
            basis_coarse_shape = (coarse_k, coarse_c, coarse_h, coarse_w)
            target_h = coarse_h if target_h is None else int(target_h)
            target_w = coarse_w if target_w is None else int(target_w)
        elif eigenvectors is None:
            raise ValueError("SJKLBasis requires eigenvectors or basis_coarse")

        eig = eigenvectors.detach().to(torch.float32).contiguous()
        if eig.dim() != 2:
            raise ValueError(f"eigenvectors must be 2-D (K, D), got shape {tuple(eig.shape)}")
        inferred_rank, inferred_dim = (int(x) for x in eig.shape)
        rank = inferred_rank if rank is None else int(rank)
        dim = inferred_dim if dim is None else int(dim)
        if (rank, dim) != (inferred_rank, inferred_dim):
            raise ValueError(
                f"eigenvectors shape {tuple(eig.shape)} does not match "
                f"(rank, dim)=({rank}, {dim})"
            )

        if coefficients is None:
            coefs = torch.zeros(rank, dtype=torch.float32)
        else:
            coefs = coefficients.detach().to(torch.float32).contiguous().reshape(-1)

        if scale is None:
            scale_t = torch.ones(rank, dtype=torch.float32)
        else:
            scale_t = scale.detach().to(torch.float32).contiguous().reshape(-1)
        if int(scale_t.numel()) != rank:
            raise ValueError(f"scale must contain rank={rank} values, got {int(scale_t.numel())}")

        if basis_coarse_shape is not None:
            shape_rank, shape_c, shape_h, shape_w = (int(x) for x in basis_coarse_shape)
            if shape_rank != rank or shape_c * shape_h * shape_w != dim:
                raise ValueError(
                    "basis_coarse_shape must match (rank, dim): "
                    f"{basis_coarse_shape} vs ({rank}, {dim})"
                )
            if shape_c != 3:
                raise ValueError(f"basis_coarse_shape channel count must be 3, got {shape_c}")
            target_h = shape_h if target_h is None else int(target_h)
            target_w = shape_w if target_w is None else int(target_w)

        if target_h is None or target_w is None:
            inferred_shape = _infer_chw_shape(dim)
            if inferred_shape is None:
                target_h, target_w = 1, max(1, dim)
            else:
                _, target_h, target_w = inferred_shape

        self.eigenvectors = eig
        self.coefficients = coefs
        self.rank = rank
        self.dim = dim
        self.scale = scale_t
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self._basis_coarse_shape = basis_coarse_shape

    @property
    def basis_coarse(self) -> torch.Tensor:
        """Runtime contract alias for the coarse `(K, C, H, W)` basis when known.

        For legacy flat-only payloads this returns `(K, D)`, preserving the old
        alias used by earlier codec tests.
        """
        if self._basis_coarse_shape is None:
            return self.eigenvectors
        return self.eigenvectors.reshape(self._basis_coarse_shape)

    def upsample(self) -> torch.Tensor:
        """Return basis vectors as `(K, 3, target_h, target_w)` for inflate."""
        coarse = self.basis_coarse
        if coarse.dim() != 4:
            shape = _infer_chw_shape(self.dim, target_h=self.target_h, target_w=self.target_w)
            if shape is None:
                raise ValueError(
                    f"cannot reshape flat SJ-KL basis dim={self.dim} into RGB target "
                    f"{self.target_h}x{self.target_w}"
                )
            self._basis_coarse_shape = (self.rank, *shape)
            coarse = self.basis_coarse
        if int(coarse.shape[-2]) == self.target_h and int(coarse.shape[-1]) == self.target_w:
            return coarse
        return torch.nn.functional.interpolate(
            coarse,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )

    def renormalize(self) -> SJKLBasis:
        """Normalize basis vectors while preserving `scale * basis` product."""
        flat = self.eigenvectors.reshape(self.rank, -1)
        norms = flat.norm(dim=1).clamp_min(1e-12)
        self.eigenvectors = (flat / norms[:, None]).reshape_as(self.eigenvectors)
        self.scale = self.scale * norms.to(self.scale.dtype)
        return self


def _infer_chw_shape(
    dim: int,
    *,
    target_h: int | None = None,
    target_w: int | None = None,
) -> tuple[int, int, int] | None:
    """Infer `(C, H, W)` for RGB frame bases when the shape is unambiguous."""
    dim = int(dim)
    if target_h is not None and target_w is not None and dim == 3 * int(target_h) * int(target_w):
        return (3, int(target_h), int(target_w))
    common_shapes = ((3, 384, 512), (3, 512, 384), (3, 192, 256), (3, 128, 128))
    for shape in common_shapes:
        c, h, w = shape
        if dim == c * h * w:
            return shape
    if dim % 3 != 0:
        return None
    pixels = dim // 3
    side = math.isqrt(pixels)
    if side * side == pixels:
        return (3, side, side)
    return None


def _encode_sjkl_basis_metadata(basis: SJKLBasis) -> bytes:
    coarse_shape = basis._basis_coarse_shape
    scale = basis.scale.detach().to(torch.float32).cpu()
    has_scale = not torch.allclose(scale, torch.ones_like(scale), atol=0.0, rtol=0.0)
    if coarse_shape is None and not has_scale:
        return b""
    if coarse_shape is None:
        inferred = _infer_chw_shape(basis.dim, target_h=basis.target_h, target_w=basis.target_w)
        if inferred is None:
            coarse_h = 0
            coarse_w = 0
        else:
            _, coarse_h, coarse_w = inferred
    else:
        _, _, coarse_h, coarse_w = coarse_shape
    if max(int(basis.target_h), int(basis.target_w), int(coarse_h), int(coarse_w)) > 0xFFFF:
        raise ValueError("SJ-KL basis metadata dimensions must fit uint16")
    return (
        _SJKL_META_MAGIC
        + struct.pack(
            "<HHHHH",
            int(basis.target_h),
            int(basis.target_w),
            int(coarse_h),
            int(coarse_w),
            int(basis.rank),
        )
        + scale.numpy().astype(np.float32).tobytes()
    )


def _decode_sjkl_basis_metadata(
    data: bytes,
    *,
    rank: int,
    dim: int,
) -> tuple[torch.Tensor, int | None, int | None, tuple[int, int, int, int] | None]:
    scale = torch.ones(rank, dtype=torch.float32)
    target_h: int | None = None
    target_w: int | None = None
    basis_coarse_shape: tuple[int, int, int, int] | None = None
    if not data:
        inferred = _infer_chw_shape(dim)
        if inferred is not None:
            _, target_h, target_w = inferred
            basis_coarse_shape = (rank, *inferred)
        return scale, target_h, target_w, basis_coarse_shape
    if not data.startswith(_SJKL_META_MAGIC):
        raise ValueError(f"unexpected trailing SJ-KL basis bytes: {len(data)}")
    meta_header_len = 4 + struct.calcsize("<HHHHH")
    if len(data) < meta_header_len:
        raise ValueError("truncated SJ-KL basis metadata")
    target_h, target_w, coarse_h, coarse_w, scale_count = struct.unpack_from("<HHHHH", data, 4)
    if int(scale_count) != int(rank):
        raise ValueError(f"SJ-KL metadata scale_count {scale_count} does not match rank {rank}")
    expected_len = meta_header_len + 4 * rank
    if len(data) != expected_len:
        raise ValueError(f"SJ-KL metadata length mismatch: expected {expected_len}, got {len(data)}")
    scale_np = np.frombuffer(data[meta_header_len:], dtype=np.float32).copy()
    scale = torch.from_numpy(scale_np)
    if coarse_h and coarse_w:
        if dim != 3 * int(coarse_h) * int(coarse_w):
            raise ValueError(
                f"SJ-KL metadata coarse shape 3x{coarse_h}x{coarse_w} "
                f"does not match dim {dim}"
            )
        basis_coarse_shape = (rank, 3, int(coarse_h), int(coarse_w))
    return scale, int(target_h), int(target_w), basis_coarse_shape


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

    return header + basis_bytes + coef_bytes + _encode_sjkl_basis_metadata(basis)


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
    pos += n_coef_bytes
    scale, target_h, target_w, basis_coarse_shape = _decode_sjkl_basis_metadata(
        data[pos:],
        rank=K,
        dim=D,
    )

    return SJKLBasis(
        eigenvectors=torch.from_numpy(eig_np.copy()),
        coefficients=torch.from_numpy(coef_np.copy()),
        rank=K,
        dim=D,
        scale=scale,
        target_h=target_h,
        target_w=target_w,
        basis_coarse_shape=basis_coarse_shape,
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


# ============================================================
# Alpha-block codec (per-frame-pair sparse coefficients)
# ============================================================
# The alpha block is the SECOND section of the full sjkl.bin payload.
# It encodes per-pair quantized coefficient vectors for projecting frames
# onto the basis. Two on-disk formats are supported:
#   - SJKB (legacy V1 dense): all pairs in fixed order, qs as raw uint8/uint16
#   - SJK2 (V2 sparse bit-packed): explicit pair_indices list, qs bit-packed
#     to alpha_bits per value. PREFERRED for sparse selections.
#
# Byte layout (after brotli decompression — alpha block is brotli-compressed
# in the full sjkl.bin payload):
#
# SJK2 sparse format:
#   magic[4]              = b"SJK2"
#   n_pairs[2 LE uint16]
#   k[2 LE uint16]        (basis width, must match basis_coarse.shape[0])
#   alpha_bits[1 byte]    in [1, 16]
#   pair_indices[2*n_pairs] = uint16 LE per pair (no duplicates allowed)
#   mins[2*n_pairs]       = float16 LE per pair
#   steps[2*n_pairs]      = float16 LE per pair
#   packed_qs[ceil(n_pairs * k * alpha_bits / 8)]
#                         = bit-packed qs values, alpha_bits each, sequential,
#                           starting at bit 0 of byte 0
#
# SJKB legacy V1 dense format (no pair_indices; all pairs in order):
#   magic[4]              = b"SJKB"
#   n_pairs[2 LE uint16]
#   k[2 LE uint16]
#   alpha_bits[1 byte]
#   mins[2*n_pairs]       = float16 LE
#   steps[2*n_pairs]      = float16 LE
#   qs[per_alpha * n_pairs * k]
#                         = uint8 if alpha_bits<=8 else uint16 (NOT bit-packed)
#
# qs values are in [0, 2^alpha_bits - 1] and decode as
# alpha[i, j] = mins[i] + qs[i, j] * steps[i].

def _validate_alpha_inputs(qs: np.ndarray, mins: np.ndarray, steps: np.ndarray, alpha_bits: int) -> None:
    if qs.ndim != 2:
        raise ValueError(f"qs must be 2-D (n_pairs, k), got shape {qs.shape}")
    n_pairs, k = qs.shape
    if not 1 <= alpha_bits <= 16:
        raise ValueError(f"alpha_bits must be in [1, 16], got {alpha_bits}")
    if not 1 <= n_pairs <= 10_000:
        raise ValueError(f"n_pairs must be in [1, 10000], got {n_pairs}")
    if not 1 <= k <= 256:
        raise ValueError(f"k must be in [1, 256], got {k}")
    if mins.shape != (n_pairs,) or steps.shape != (n_pairs,):
        raise ValueError(
            f"mins/steps must be (n_pairs={n_pairs},), got {mins.shape} / {steps.shape}"
        )
    qmax = (1 << alpha_bits) - 1
    if int(qs.max()) > qmax or int(qs.min()) < 0:
        raise ValueError(
            f"qs values must be in [0, {qmax}] for alpha_bits={alpha_bits}; "
            f"got [{int(qs.min())}, {int(qs.max())}]"
        )


def encode_sjkl_alpha_block_v2_sparse(
    qs: np.ndarray,
    mins: np.ndarray,
    steps: np.ndarray,
    alpha_bits: int,
    pair_indices: np.ndarray,
) -> bytes:
    """Encode an SJK2 sparse bit-packed alpha block (no brotli applied here).

    Args:
        qs: (n_pairs, k) array of quantized coefficient values in [0, 2^alpha_bits - 1].
        mins: (n_pairs,) float per-pair offsets.
        steps: (n_pairs,) float per-pair scales.
        alpha_bits: bits per quantized value, in [1, 16].
        pair_indices: (n_pairs,) uint-like array of source pair indices (no duplicates).

    Returns:
        Raw bytes of the SJK2 alpha block (caller is responsible for brotli compression).
    """
    _validate_alpha_inputs(qs, mins, steps, alpha_bits)
    n_pairs, k = qs.shape
    if pair_indices.shape != (n_pairs,):
        raise ValueError(f"pair_indices must be (n_pairs={n_pairs},), got {pair_indices.shape}")
    if int(pair_indices.min()) < 0 or int(pair_indices.max()) > 0xFFFF:
        raise ValueError("pair_indices must fit in uint16 [0, 65535]")
    if len({int(x) for x in pair_indices.tolist()}) != n_pairs:
        raise ValueError("pair_indices must not contain duplicates")

    header = _SJKL_BLOCK_V2_MAGIC + struct.pack("<HHB", n_pairs, k, alpha_bits)
    indices_bytes = np.asarray(pair_indices, dtype=np.uint16).tobytes()
    mins_bytes = np.asarray(mins, dtype=np.float16).tobytes()
    steps_bytes = np.asarray(steps, dtype=np.float16).tobytes()

    flat = qs.astype(np.uint32).flatten()
    packed_len = math.ceil(n_pairs * k * alpha_bits / 8)
    packed = bytearray(packed_len)
    bit_pos = 0
    mask = (1 << alpha_bits) - 1
    for v in flat:
        v_masked = int(v) & mask
        byte_idx = bit_pos // 8
        offset = bit_pos % 8
        # write up to 4 bytes covering this value
        window = v_masked << offset
        for b in range(4):
            if byte_idx + b < packed_len:
                packed[byte_idx + b] |= (window >> (8 * b)) & 0xFF
        bit_pos += alpha_bits

    return header + indices_bytes + mins_bytes + steps_bytes + bytes(packed)


def encode_sjkl_alpha_block_v1_dense(
    qs: np.ndarray,
    mins: np.ndarray,
    steps: np.ndarray,
    alpha_bits: int,
) -> bytes:
    """Encode an SJKB legacy V1 dense alpha block (no brotli applied here).

    All n_pairs are emitted in order with no pair_indices side-channel. qs is
    raw uint8 or uint16 (not bit-packed). Use the V2 sparse format instead for
    sparse pair selections — it's smaller in nearly every case.
    """
    _validate_alpha_inputs(qs, mins, steps, alpha_bits)
    n_pairs, k = qs.shape
    header = _SJKL_BLOCK_MAGIC + struct.pack("<HHB", n_pairs, k, alpha_bits)
    mins_bytes = np.asarray(mins, dtype=np.float16).tobytes()
    steps_bytes = np.asarray(steps, dtype=np.float16).tobytes()
    qs_bytes = qs.astype(np.uint8).tobytes() if alpha_bits <= 8 else qs.astype(np.uint16).tobytes()
    return header + mins_bytes + steps_bytes + qs_bytes


def decode_sjkl_alpha_block(raw: bytes) -> dict:
    """Decode an alpha block (post-brotli-decompression bytes) — auto-detects SJK2 vs SJKB.

    Mirrors the runtime inverse in submissions/robust_current/inflate_renderer.py
    (_unpack_sjkl_alpha_block) so encode→decode roundtrip is byte-exact.
    """
    if len(raw) < 9:
        raise ValueError("SJ-KL alpha block is too short")
    if raw[:4] == _SJKL_BLOCK_V2_MAGIC:
        n_pairs, k, alpha_bits = struct.unpack("<HHB", raw[4:9])
        cursor = 9
        indices_end = cursor + 2 * n_pairs
        mins_end = indices_end + 2 * n_pairs
        steps_end = mins_end + 2 * n_pairs
        packed_len = math.ceil(n_pairs * k * alpha_bits / 8)
        qs_end = steps_end + packed_len
        if qs_end != len(raw):
            raise ValueError(
                f"SJ-KL sparse alpha block length mismatch: expected {qs_end}, got {len(raw)}"
            )
        pair_indices = np.frombuffer(raw[cursor:indices_end], dtype=np.uint16).astype(np.int64).copy()
        if len({int(x) for x in pair_indices.tolist()}) != int(pair_indices.shape[0]):
            raise ValueError("SJ-KL sparse alpha block contains duplicate pair indices")
        mins = np.frombuffer(raw[indices_end:mins_end], dtype=np.float16).astype(np.float32).copy()
        steps = np.frombuffer(raw[mins_end:steps_end], dtype=np.float16).astype(np.float32).copy()
        packed = raw[steps_end:qs_end]
        dtype = np.uint8 if alpha_bits <= 8 else np.uint16
        qs_flat = np.zeros(n_pairs * k, dtype=dtype)
        bit_pos = 0
        mask = (1 << alpha_bits) - 1
        for idx in range(int(qs_flat.shape[0])):
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            window = 0
            for b in range(4):
                if byte_idx + b < len(packed):
                    window |= packed[byte_idx + b] << (8 * b)
            qs_flat[idx] = (window >> offset) & mask
            bit_pos += alpha_bits
        return {
            "mins": mins,
            "steps": steps,
            "qs": qs_flat.reshape(n_pairs, k),
            "alpha_bits": int(alpha_bits),
            "pair_indices": pair_indices,
            "alpha_block_format": "sparse_bitpacked_v2",
        }
    if raw[:4] != _SJKL_BLOCK_MAGIC:
        raise ValueError(f"bad SJ-KL alpha block magic: {raw[:4]!r}")
    n_pairs, k, alpha_bits = struct.unpack("<HHB", raw[4:9])
    if alpha_bits <= 8:
        a_dtype = np.uint8
        per_alpha = 1
    elif alpha_bits <= 16:
        a_dtype = np.uint16
        per_alpha = 2
    else:
        raise ValueError(f"unsupported SJ-KL alpha_bits={alpha_bits}")
    cursor = 9
    mins_end = cursor + 2 * n_pairs
    steps_end = mins_end + 2 * n_pairs
    qs_end = steps_end + per_alpha * n_pairs * k
    if qs_end != len(raw):
        raise ValueError(f"SJ-KL alpha block length mismatch: expected {qs_end}, got {len(raw)}")
    mins = np.frombuffer(raw[cursor:mins_end], dtype=np.float16).astype(np.float32).copy()
    steps = np.frombuffer(raw[mins_end:steps_end], dtype=np.float16).astype(np.float32).copy()
    qs = np.frombuffer(raw[steps_end:qs_end], dtype=a_dtype).copy().reshape(n_pairs, k)
    return {
        "mins": mins,
        "steps": steps,
        "qs": qs,
        "alpha_bits": int(alpha_bits),
        "pair_indices": None,
        "alpha_block_format": "legacy_v1",
    }


# ============================================================
# Full sjkl.bin payload codec (basis section + alpha block section)
# ============================================================
# Wire format expected by submissions/robust_current/inflate_renderer.py
# (_unpack_full_sjkl_payload):
#
#   SJKL[4] + basis_len[4 LE uint32] + block_len[4 LE uint32]
#   + basis_bytes[basis_len]   (output of encode_sjkl_basis WITHOUT leading SJKL magic)
#   + alpha_block_bytes[block_len]  (brotli-compressed SJK2 or SJKB block)
#
# The runtime prepends SJKL_MAGIC back to basis_bytes before parsing the basis.


def encode_full_sjkl_payload(
    basis: SJKLBasis,
    alpha_block_bytes: bytes,
    *,
    basis_quant_bits: int | None = 6,
) -> bytes:
    """Encode a full sjkl.bin payload (basis + alpha block) per runtime contract.

    Args:
        basis: SJKLBasis (eigenvectors + optional coefficients).
        alpha_block_bytes: pre-encoded alpha block (output of one of the
            encode_sjkl_alpha_block_* functions, then brotli-compressed by caller).
        basis_quant_bits: quantization bits for basis (default 6, addendum-verified).

    Returns:
        Full sjkl.bin payload bytes ready to be written as an archive member.
    """
    basis_full = encode_sjkl_basis(basis, basis_quant_bits=basis_quant_bits)
    if not basis_full.startswith(_SJKL_MAGIC):
        raise RuntimeError("encode_sjkl_basis must produce SJKL magic prefix")
    # strip the SJKL magic from the basis section (runtime prepends it back)
    basis_section = basis_full[len(_SJKL_MAGIC):]
    if not isinstance(alpha_block_bytes, (bytes, bytearray)):
        raise TypeError("alpha_block_bytes must be bytes")
    header = _SJKL_MAGIC + struct.pack("<II", len(basis_section), len(alpha_block_bytes))
    return header + basis_section + bytes(alpha_block_bytes)


def decode_full_sjkl_payload(payload: bytes) -> tuple[SJKLBasis, dict]:
    """Decode a full sjkl.bin payload into (basis, alpha_block_dict).

    Note: the alpha block portion is returned as raw post-brotli bytes —
    callers that wrote a brotli-compressed alpha block must brotli-decompress
    those bytes BEFORE passing to decode_sjkl_alpha_block. This function does
    NOT brotli-decompress automatically because some callers emit raw
    (uncompressed) alpha blocks for testing/debugging.
    """
    if len(payload) < 12:
        raise ValueError("SJ-KL payload is too short")
    if payload[:4] != _SJKL_MAGIC:
        raise ValueError(f"bad SJ-KL payload magic: {payload[:4]!r}")
    basis_len, block_len = struct.unpack("<II", payload[4:12])
    cursor = 12
    basis_end = cursor + basis_len
    block_end = basis_end + block_len
    if basis_len <= 0 or block_len <= 0 or block_end != len(payload):
        raise ValueError(
            f"SJ-KL payload TOC mismatch: basis_len={basis_len} block_len={block_len} "
            f"total={len(payload)}"
        )
    basis = decode_sjkl_basis(_SJKL_MAGIC + payload[cursor:basis_end])
    alpha_raw = payload[basis_end:block_end]
    return (basis, {"alpha_block_raw_bytes": alpha_raw, "block_len": block_len})


# ─────────────────────────────────────────────────────────────────────────
# Recovered Fisher/Lanczos rank helpers.
#
# These are compression-time analysis primitives only. They do not run at
# inflate time and they do not produce score evidence by themselves.
# ─────────────────────────────────────────────────────────────────────────


def _scorer_outputs(model: object | None, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if model is None:
        return ()
    x = frames
    preprocess = getattr(model, "preprocess_input", None)
    if callable(preprocess):
        x = preprocess(x)
    out = model(x)  # type: ignore[misc]
    if torch.is_tensor(out):
        return (out.reshape(-1),)
    if isinstance(out, dict):
        tensors: list[torch.Tensor] = []
        for key in sorted(out):
            value = out[key]
            if not torch.is_tensor(value):
                continue
            if key == "pose" and value.shape[-1] >= 2:
                value = value[..., : value.shape[-1] // 2]
            tensors.append(value.reshape(-1))
        return tuple(tensors)
    if isinstance(out, (tuple, list)):
        return tuple(x.reshape(-1) for x in out if torch.is_tensor(x))
    raise TypeError(f"unsupported scorer output type: {type(out).__name__}")


def _embed_probe_vector(frames: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    v = v.to(device=frames.device, dtype=frames.dtype)
    if int(v.numel()) == int(frames.numel()):
        return v.reshape_as(frames), None
    if frames.dim() == 5:
        per_frame = int(frames.shape[2] * frames.shape[3] * frames.shape[4])
        if int(v.numel()) == per_frame:
            full = torch.zeros_like(frames)
            full[:, 0] = v.reshape(frames.shape[2], frames.shape[3], frames.shape[4])
            return full, tuple(v.shape)
    if frames.dim() == 4 and int(v.numel()) == int(frames.shape[1] * frames.shape[2] * frames.shape[3]):
        return v.reshape_as(frames), tuple(v.shape)
    raise ValueError(
        f"probe vector with {int(v.numel())} values is incompatible with frames shape {tuple(frames.shape)}"
    )


def _extract_probe_result(hv: torch.Tensor, original_shape: tuple[int, ...] | None, v: torch.Tensor) -> torch.Tensor:
    if original_shape is None:
        return hv.reshape_as(v)
    if hv.dim() == 5:
        selected = hv[0, 0] if int(hv.shape[0]) == 1 else hv[:, 0].sum(dim=0)
        return selected.reshape(original_shape)
    return hv.reshape(original_shape)


def fisher_matvec(seg: object, pose: object, frames: object, v: object) -> torch.Tensor:
    """Return `F @ v` for the local scorer Fisher proxy at `frames`.

    `F` is the Gauss-Newton/Fisher matrix of frozen scorer outputs at the
    anchor: `100 J_seg.T J_seg + 10 J_pose.T J_pose`. If `frames` is a
    two-frame pair and `v` is a single RGB frame vector, the probe is applied
    to frame 0 to match the current SJ-KL inflate application site.
    """
    frames_t = torch.as_tensor(frames)
    if not frames_t.is_floating_point():
        frames_t = frames_t.to(torch.float32)
    anchor = frames_t.detach().clone().requires_grad_(True)
    v_t = torch.as_tensor(v, device=anchor.device)
    v_full, original_shape = _embed_probe_vector(anchor, v_t)

    with torch.no_grad():
        seg_targets = tuple(x.detach() for x in _scorer_outputs(seg, anchor.detach()))
        pose_targets = tuple(x.detach() for x in _scorer_outputs(pose, anchor.detach()))
    if not seg_targets and not pose_targets:
        raise ValueError("fisher_matvec requires at least one scorer output")

    loss = anchor.new_zeros(())
    for out, target in zip(_scorer_outputs(seg, anchor), seg_targets, strict=True):
        loss = loss + 50.0 * (out - target).pow(2).sum()
    for out, target in zip(_scorer_outputs(pose, anchor), pose_targets, strict=True):
        loss = loss + 5.0 * (out - target).pow(2).sum()

    (grad,) = torch.autograd.grad(loss, anchor, create_graph=True)
    gv = (grad * v_full).sum()
    (hv,) = torch.autograd.grad(gv, anchor, retain_graph=False)
    return _extract_probe_result(hv.detach(), original_shape, v_t)


def lanczos_topk(
    matvec: object,
    n: int | None = None,
    k: int | None = None,
    *args: object,
    dim: int | None = None,
    n_iters: int | None = None,
    seed: int = 0,
    shape_hint: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    tol: float = 1e-10,
    reorthogonalize: bool = True,
    **_kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic symmetric Lanczos top-k eigensolver for a matvec operator."""
    del args
    if dim is None:
        if n is None:
            raise ValueError("lanczos_topk requires dim=... or positional n")
        dim = int(n)
    if k is None:
        raise ValueError("lanczos_topk requires k")
    dim = int(dim)
    k = int(k)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if not 1 <= k <= dim:
        raise ValueError(f"k must be in [1, dim={dim}], got {k}")
    n_iters = max(k, int(n_iters if n_iters is not None else max(k * 2, 8)))
    probe_shape = tuple(shape_hint) if shape_hint is not None else (dim,)
    if math.prod(probe_shape) != dim:
        raise ValueError(f"shape_hint product {math.prod(probe_shape)} does not match dim {dim}")

    device_t = torch.device(device)
    generator = torch.Generator().manual_seed(int(seed))
    q = torch.randn(probe_shape, generator=generator, dtype=dtype).to(device_t)
    q = q / q.norm().clamp_min(tol)
    q_prev = torch.zeros_like(q)
    beta_prev = q.new_zeros(())
    basis_vectors: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    betas: list[torch.Tensor] = []

    if not callable(matvec):
        raise TypeError("matvec must be callable")

    for _ in range(n_iters):
        basis_vectors.append(q)
        w = matvec(q)  # type: ignore[misc]
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, device=device_t, dtype=dtype)
        w = w.to(device=device_t, dtype=dtype).reshape(probe_shape)
        w = w - beta_prev * q_prev
        alpha = (q.reshape(-1) * w.reshape(-1)).sum()
        w = w - alpha * q
        if reorthogonalize:
            for old_q in basis_vectors:
                w = w - (old_q.reshape(-1) * w.reshape(-1)).sum() * old_q
        beta = w.norm()
        alphas.append(alpha.detach())
        if float(beta) <= tol:
            break
        betas.append(beta.detach())
        q_prev = q
        q = w / beta.clamp_min(tol)
        beta_prev = beta

    m = len(alphas)
    tri = torch.zeros((m, m), dtype=dtype, device=device_t)
    for i, alpha in enumerate(alphas):
        tri[i, i] = alpha
        if i + 1 < m:
            tri[i, i + 1] = betas[i]
            tri[i + 1, i] = betas[i]
    eigvals, eigvecs = torch.linalg.eigh(tri)
    order = torch.argsort(eigvals, descending=True)[:k]
    q_flat = torch.stack([q_i.reshape(-1) for q_i in basis_vectors[:m]], dim=0)
    ritz_flat = eigvecs[:, order].T @ q_flat
    ritz_flat = ritz_flat / ritz_flat.norm(dim=1, keepdim=True).clamp_min(tol)
    ritz = ritz_flat if shape_hint is None else ritz_flat.reshape(k, *probe_shape)
    return eigvals[order].detach(), ritz.detach()


def effective_rank(eigvals: object, *, threshold: float = 1e-4) -> int:
    """Count finite eigenvalues above `threshold * max(abs(eigvals))`."""
    vals = torch.as_tensor(eigvals, dtype=torch.float64).reshape(-1)
    vals = vals[torch.isfinite(vals)].abs()
    if vals.numel() == 0:
        return 0
    max_val = vals.max()
    if float(max_val) <= 0.0:
        return 0
    cutoff = max_val * float(threshold) if threshold < 1.0 else vals.new_tensor(float(threshold))
    return int((vals > cutoff).sum().item())


# ─────────────────────────────────────────────────────────────────────────
# Legacy compatibility symbol.
# Tests and older tools still call pack_sjkl_basis; keep it as a thin wrapper
# over the canonical encoder instead of a dispatch-blocking stub.
# ─────────────────────────────────────────────────────────────────────────


def pack_sjkl_basis(*args: object, **kwargs: object) -> bytes:
    """Backward-compatible wrapper for older SJ-KL runtime tests/tools."""
    if not args:
        raise TypeError("pack_sjkl_basis requires an SJKLBasis argument")
    basis = args[0]
    if not isinstance(basis, SJKLBasis):
        raise TypeError(f"pack_sjkl_basis expected SJKLBasis, got {type(basis).__name__}")
    basis_quant_bits = kwargs.pop("basis_quant_bits", 6)
    if kwargs:
        raise TypeError(f"unexpected pack_sjkl_basis kwargs: {sorted(kwargs)}")
    return encode_sjkl_basis(basis, basis_quant_bits=basis_quant_bits)
