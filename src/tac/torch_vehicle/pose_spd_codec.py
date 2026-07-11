# SPDX-License-Identifier: MIT
"""SPD-cone / Hilbert-projective-metric variant of the #140 low-rank pose-section codec.

WHAT THIS IS (and is NOT).
==========================
This is a RATE codec on the STORED POSE SECTION — the small ``(n_pairs, pose_dim)``
matrix of PoseNet targets / ξ that the stored-target sidecar appends to the archive.
It is NOT a pose-descent change and NOT a PoseNet/SegNet forward: the "distortion" is
the reconstruction MSE of the pose-target matrix (which maps to contest ``d_pose`` only
through the byte-closed decode — surrogate, NOT authority; CLAUDE.md NO-FAKE class 8).

THE GEOMETRY (Nielsen 2307.10644 / 2403.10089; the cluster's SPD thread).
=========================================================================
The centered pose data ``X_c`` (n×d) has a d×d covariance ``C = X_cᵀX_c/(n-1)`` that is
**SPD**. The baseline #140 codec is an SVD (KLT) low-rank code: it stores the top-``rank``
eigenvectors of ``C`` and quantizes each retained principal mode with the SAME ``levels``
(uniform per-mode bit depth) plus a HARD rank truncation of the rest.

The SPD-cone refinement is the classic **reverse water-filling** on the covariance
spectrum — the rate-distortion-OPTIMAL scalar-quantizer bit allocation for a Gaussian
source under MSE (which is exactly this codec's distortion). Given a water level ``θ``:
  * per-mode target distortion  ``D_i = min(λ_i, θ)`` (equalize distortion across active
    modes — the water-filling condition),
  * a mode with ``λ_i ≤ θ`` is DROPPED (its samples cost < 1 level — soft rank truncation),
  * an active mode gets ``levels_i ≈ range_i / sqrt(12·θ)`` (uniform-scalar MSE ≈
    ``(range/L)²/12`` set equal to ``θ``).
So the SPD codec spends bits on modes in proportion to their variance instead of giving
every mode the same depth. The **Hilbert projective distance** ``d_H = log(λ_max/λ_min)``
(the paper's cheap min/max-eigenvalue cone metric) is the log-condition-number that
measures exactly how much anisotropy there is to exploit — it is reported as a diagnostic:
when ``d_H`` is large the spectrum is anisotropic and water-filling has room to help; when
``d_H`` is small (or one mode dominates) it does not.

THE HONEST PREDICTION (deep-math, stated BEFORE measuring — CLAUDE.md NO-FAKE).
==============================================================================
On the REAL contest pose the spectrum is near-rank-1 (dim-0 std ≈175× dims 1-5; ~99.8%
energy in one SVD mode). Two competing effects:
  (a) water-filling SHOULD help: the weak modes are worth ~0 bits, drop them.
  (b) but the baseline ALREADY delta-zigzags + brotli-q11 the quantized modes — a
      low-variance mode quantized to ``levels`` produces a near-constant integer stream
      whose deltas are ~0 → zigzag → mostly-zero bytes → brotli crushes it to almost
      nothing. **The entropy coder already achieves the water-filling allocation for
      free.** So explicit per-mode allocation likely buys ~nothing on the near-rank-1
      pose, and pays a small fixed basis overhead → break-even-to-slightly-negative.
The POSITIVE CONTROL where SPD water-filling genuinely wins: a matrix with SEVERAL
COMPARABLE-BUT-DISTINCT eigenvalues (moderate anisotropy, no near-zero mode brotli can
crush) — there uniform-levels over-spends on the lower-variance modes and water-filling
gives fewer levels to them at matched MSE.

Both codecs share the SAME SVD basis, so this is an apples-to-apples bit-ALLOCATION A/B:
uniform-depth + hard-truncation (baseline) vs variance-adaptive water-filled depth (SPD).

Layout (APPENDED after the vendored sections; ``PSPD`` magic, decoded pure-numpy):
  [MAGIC:4='PSPD'][n:u32][pose_dim:u32][rank:u32][blob_len:u32][brotli payload]
where the payload (before compression) is:
  [mu:  pose_dim   f32]              # per-dim mean (centering)
  [Vt:  rank*pose_dim f32]           # the kept (active) right-singular basis rows
  [levels: rank  u32]               # PER-MODE quantization levels (water-filled)
  [mins:   rank  f32] [scales: rank f32]   # per-mode quant range of the principal series
  [delta_zigzag_lo: n*rank u8]
  [delta_zigzag_hi: n*rank u8]      # 1st-order temporal delta of the quantized series
"""
from __future__ import annotations

import io
import math
import struct

import brotli
import numpy as np
import torch

_POSE_SECTION_MAGIC_SPD = b"PSPD"

# uint16 zigzag of a 1st-order delta must fit: 2*levels <= 0xFFFF → levels <= 32767.
_SPD_MAX_LEVELS = 0xFFFF // 2


def hilbert_projective_distance(cov_eigs: np.ndarray, *, eps: float = 1e-12) -> float:
    """Birkhoff/Hilbert projective-cone distance of an SPD spectrum (paper #11).

    ``d_H = log(λ_max / λ_min)`` — the log-condition-number, computable from ONLY the
    extreme eigenvalues. A cheap anisotropy diagnostic: large ``d_H`` ⇒ anisotropic
    spectrum ⇒ water-filling has room to reallocate bits; ``d_H ≈ 0`` ⇒ isotropic ⇒
    uniform allocation is already optimal."""
    e = np.asarray(cov_eigs, dtype=np.float64)
    e = e[e > eps]
    if e.size == 0:
        return 0.0
    return float(math.log(float(e.max()) / float(e.min())))


def _waterfill_levels(ranges: np.ndarray, water_level: float) -> np.ndarray:
    """Per-mode uniform-quantizer levels under reverse water-filling at ``water_level``.

    ``L_i = range_i / sqrt(12·θ)`` sets each active mode's uniform-scalar MSE ≈ ``θ``
    (equal distortion — the water-filling condition). Rounded, clamped to
    ``[0, _SPD_MAX_LEVELS]``; a mode with ``L_i < 2`` is dropped (worth < 1 usable level)."""
    if water_level <= 0:
        raise ValueError(f"water_level must be > 0, got {water_level}")
    lf = np.asarray(ranges, dtype=np.float64) / math.sqrt(12.0 * water_level)
    levels = np.rint(lf).astype(np.int64)
    levels = np.clip(levels, 0, _SPD_MAX_LEVELS)
    levels[levels < 2] = 0  # drop modes not worth >=2 levels
    return levels


def encode_pose_section_spd(
    stored_pose: torch.Tensor,
    *,
    water_level: float,
) -> bytes:
    """Encode ``(n_pairs, pose_dim)`` pose with the SPD-cone water-filled codec.

    ``water_level`` (θ) is the reverse-water-filling level: SMALLER θ ⇒ more levels /
    fewer modes dropped ⇒ higher fidelity + more bytes; LARGER θ ⇒ coarser + smaller.
    Every retained mode targets uniform-quant MSE ≈ θ, so θ IS (approximately) the
    per-mode reconstruction MSE floor. Modes with covariance below θ are dropped
    (soft rank truncation). Returns the FULL ``PSPD`` section, ready to append."""
    t = stored_pose.detach().cpu().float()
    n, dpose = int(t.shape[0]), int(t.shape[1])

    mu = t.mean(dim=0)  # (dpose,)
    x_centered = t - mu.unsqueeze(0)  # (n, dpose)
    # SVD → the SPD covariance eigenbasis (right-singular vectors of X_c are the
    # eigenvectors of C = X_cᵀX_c). Same basis the baseline uses; only the per-mode
    # bit ALLOCATION differs.
    _u, s, vt = torch.linalg.svd(x_centered, full_matrices=False)
    series_full = x_centered @ vt.T  # (n, m) principal time-series, m=min(n,dpose)

    snp = series_full.numpy()
    ranges = snp.max(axis=0) - snp.min(axis=0)  # (m,) per-mode dynamic range
    levels_all = _waterfill_levels(ranges, water_level)  # (m,)

    active = np.nonzero(levels_all >= 2)[0]
    rank = int(active.size)

    out = io.BytesIO()
    out.write(_POSE_SECTION_MAGIC_SPD)
    out.write(struct.pack("<III", n, dpose, rank))

    if rank == 0:
        # Degenerate: everything dropped → reconstruct is just mu. Store mu, empty basis.
        payload = io.BytesIO()
        payload.write(mu.to(torch.float32).numpy().tobytes())
        blob = brotli.compress(payload.getvalue(), quality=11)
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
        return out.getvalue()

    vt_r = vt.numpy()[active]  # (rank, dpose)
    series_r = snp[:, active]  # (n, rank)
    levels_r = levels_all[active].astype(np.int64)  # (rank,)
    mins = series_r.min(axis=0)  # (rank,)
    maxs = series_r.max(axis=0)
    scales = np.clip((maxs - mins) / levels_r.astype(np.float64), 1e-12, None)

    q = np.rint((series_r - mins[None, :]) / scales[None, :])
    q = np.clip(q, 0, levels_r[None, :]).astype(np.int64)  # (n, rank)
    delta = np.empty_like(q)
    delta[0] = q[0]
    delta[1:] = q[1:] - q[:-1]
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()

    payload = io.BytesIO()
    payload.write(mu.to(torch.float32).numpy().tobytes())
    payload.write(vt_r.astype(np.float32).tobytes())
    payload.write(levels_r.astype(np.uint32).tobytes())
    payload.write(mins.astype(np.float32).tobytes())
    payload.write(scales.astype(np.float32).tobytes())
    payload.write(lo)
    payload.write(hi)
    blob = brotli.compress(payload.getvalue(), quality=11)

    out.write(struct.pack("<I", len(blob)))
    out.write(blob)
    return out.getvalue()


def decode_pose_section_spd(section_bytes: bytes) -> torch.Tensor:
    """Inverse of :func:`encode_pose_section_spd` → ``(n_pairs, pose_dim)`` pose (numpy)."""
    buf = io.BytesIO(section_bytes)
    magic = buf.read(4)
    if magic != _POSE_SECTION_MAGIC_SPD:
        raise ValueError(
            f"bad SPD pose-section magic {magic!r} (expected {_POSE_SECTION_MAGIC_SPD!r})"
        )
    n, dpose, rank = struct.unpack("<III", buf.read(12))
    blob_len = struct.unpack("<I", buf.read(4))[0]
    raw = brotli.decompress(buf.read(blob_len))
    rb = io.BytesIO(raw)
    mu = np.frombuffer(rb.read(dpose * 4), dtype=np.float32).astype(np.float32)

    if rank == 0:
        pose = np.broadcast_to(mu[None, :], (n, dpose)).copy()
        return torch.from_numpy(pose.astype(np.float32))

    vt_r = np.frombuffer(rb.read(rank * dpose * 4), dtype=np.float32).reshape(rank, dpose)
    rb.read(rank * 4)  # per-mode levels: offset-advance only (reconstruction uses mins/scales)
    mins = np.frombuffer(rb.read(rank * 4), dtype=np.float32).astype(np.float64)
    scales = np.frombuffer(rb.read(rank * 4), dtype=np.float32).astype(np.float64)
    total = n * rank
    lo = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, rank)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int64) // 2,
        -(delta_zz.astype(np.int64) // 2) - 1,
    )
    q = np.empty_like(delta)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    series = q.astype(np.float64) * scales[None, :] + mins[None, :]  # (n, rank)
    # np.matmul on some numpy/BLAS builds emits SPURIOUS divide/overflow/invalid
    # RuntimeWarnings on this float64 SIMD path even when the result is finite and
    # exact (verified: no nan/inf in the output). Silence the false positive so the
    # shipped codec does not spew warnings; the result is asserted correct by the
    # round-trip tests, not by this guard.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pose = series @ vt_r.astype(np.float64) + mu[None, :].astype(np.float64)
    return torch.from_numpy(pose.astype(np.float32))


def spd_pose_section_fidelity(
    stored_pose: torch.Tensor,
    *,
    water_level: float,
) -> tuple[int, float]:
    """Return the REAL ``(bytes, mse)`` of the SPD codec on ``stored_pose``.

    Encodes then round-trip-decodes the ACTUAL pose (Catalog #304: the encode IS the
    empirical bit-spend, the round-trip IS the empirical fidelity — nothing asserted)."""
    section = encode_pose_section_spd(stored_pose, water_level=water_level)
    rec = decode_pose_section_spd(section)
    mse = float(((rec - stored_pose.detach().cpu().float()) ** 2).mean().item())
    return len(section), mse


def spd_fit_to_mse(
    stored_pose: torch.Tensor,
    target_mse: float,
    *,
    n_steps: int = 48,
    theta_lo: float | None = None,
    theta_hi: float | None = None,
) -> tuple[float, int, float]:
    """Bisection-search a ``water_level`` whose round-trip MSE is ≤ ``target_mse``.

    Returns ``(water_level, bytes, mse)`` at the SMALLEST bytes that still meets the MSE
    budget (largest θ with mse ≤ target). Bytes and MSE are MEASURED, never asserted —
    the fair 'matched-MSE' A/B point against the baseline codec."""
    t = stored_pose.detach().cpu().float()
    xc = (t - t.mean(dim=0, keepdim=True)).numpy()
    ranges = xc @ np.linalg.svd(xc, full_matrices=False)[2].T
    span = float((ranges.max(axis=0) - ranges.min(axis=0)).max())
    # θ maps to per-mode MSE ~ θ; bracket generously around the data scale.
    hi = theta_hi if theta_hi is not None else max(span * span, 1e-6)
    lo = theta_lo if theta_lo is not None else max(1e-14, target_mse * 1e-6)
    best = None
    for _ in range(int(n_steps)):
        mid = math.sqrt(lo * hi)
        nbytes, mse = spd_pose_section_fidelity(t, water_level=mid)
        if mse <= target_mse:
            best = (mid, nbytes, mse)
            lo = mid  # try coarser (larger θ, fewer bytes) while still meeting budget
        else:
            hi = mid  # too coarse, need finer
    if best is None:
        # Never met the budget in-bracket; return the finest we tried.
        nbytes, mse = spd_pose_section_fidelity(t, water_level=lo)
        return lo, nbytes, mse
    return best


def spd_fit_to_bytes(
    stored_pose: torch.Tensor,
    target_bytes: int,
    *,
    n_steps: int = 48,
    theta_lo: float | None = None,
    theta_hi: float | None = None,
) -> tuple[float, int, float]:
    """Bisection-search a ``water_level`` whose section size is ≤ ``target_bytes``.

    Returns ``(water_level, bytes, mse)`` at the LOWEST MSE that still fits the byte
    budget (smallest θ with bytes ≤ target). The fair 'matched-bytes' A/B point."""
    t = stored_pose.detach().cpu().float()
    xc = (t - t.mean(dim=0, keepdim=True)).numpy()
    ranges = xc @ np.linalg.svd(xc, full_matrices=False)[2].T
    span = float((ranges.max(axis=0) - ranges.min(axis=0)).max())
    hi = theta_hi if theta_hi is not None else max(span * span, 1e-6)
    lo = theta_lo if theta_lo is not None else 1e-14
    best = None
    for _ in range(int(n_steps)):
        mid = math.sqrt(lo * hi)
        nbytes, mse = spd_pose_section_fidelity(t, water_level=mid)
        if nbytes <= target_bytes:
            best = (mid, nbytes, mse)
            hi = mid  # try finer (smaller θ, lower MSE) while still fitting bytes
        else:
            lo = mid  # too big, coarsen
    if best is None:
        nbytes, mse = spd_pose_section_fidelity(t, water_level=hi)
        return hi, nbytes, mse
    return best


__all__ = [
    "decode_pose_section_spd",
    "encode_pose_section_spd",
    "hilbert_projective_distance",
    "spd_fit_to_bytes",
    "spd_fit_to_mse",
    "spd_pose_section_fidelity",
]
