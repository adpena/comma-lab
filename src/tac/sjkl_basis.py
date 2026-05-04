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
_SJKL_BLOCK_MAGIC = b"SJKB"      # legacy V1 dense alpha block (matches inflate_renderer.py)
_SJKL_BLOCK_V2_MAGIC = b"SJK2"   # V2 sparse bit-packed alpha block with pair indices
_SJKL_VERSION = 1
_FP16_FLAG = 0  # basis_quant_bits=None / FP16 fallback

__all__ = [
    "SJKLBasis",
    "encode_sjkl_basis",
    "decode_sjkl_basis",
    "unpack_sjkl_basis",
    "apply_sjkl_residual",
    "compute_sjkl_basis_lanczos",
    "encode_sjkl_alpha_block_v2_sparse",
    "encode_sjkl_alpha_block_v1_dense",
    "decode_sjkl_alpha_block",
    "encode_full_sjkl_payload",
    "decode_full_sjkl_payload",
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

import math


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
        raise ValueError(f"pair_indices must fit in uint16 [0, 65535]")
    if len(set(int(x) for x in pair_indices.tolist())) != n_pairs:
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
    if alpha_bits <= 8:
        qs_bytes = qs.astype(np.uint8).tobytes()
    else:
        qs_bytes = qs.astype(np.uint16).tobytes()
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
        if len(set(int(x) for x in pair_indices.tolist())) != int(pair_indices.shape[0]):
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
