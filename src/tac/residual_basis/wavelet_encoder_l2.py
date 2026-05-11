"""L2 score-aware wavelet residual encoder over PR106 r2 decoded RGB frames.

This is the **Level 2 score-aware encoder** for the ``wavelet`` non-HNeRV
residual family. It replaces the L1 empty-residual scaffold with a
gradient-trained wavelet-coefficient quantizer that targets the contest's
SegNet / PoseNet distortion at the PR106 r2 operating point.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13
lessons, this module honors:

* lesson 1 (score-aware substrate training): inner loop runs the
  ``compute_score_aware_proxy_loss`` Lagrangian against the GT frames decoded
  from ``upstream/videos/0.mkv``;
* lesson 6 (score-domain Lagrangian): proxy uses ``alpha=25``, ``beta=100``,
  ``gamma=1`` to replicate the contest score functional verbatim;
* lesson 8 (eval-roundtrip-aware + differentiable YUV6): the proxy loss
  routes through ``apply_eval_roundtrip_during_training`` and
  ``differentiable_rgb_to_yuv6``;
* lesson 13 (KILL is LAST RESORT): the encoder NEVER emits a KILL verdict;
  it returns the best parameterization found within the budget.

The encoder emits ``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False`` permanently. The output residual
bytes are CONSUMED by ``submissions/pr106_wavelet_residual_sidecar/inflate.py``
and decoded back to a ``(T, H, W, 3)`` float64 residual via the same
single-level Haar synthesis the inflate runtime uses.

Architecture
------------

A single-level 2D Haar DWT is applied to ``(decoded + delta)`` per channel
per frame, yielding ``(cA, cH, cV, cD)`` bands at half resolution. Each
band gets a **single float32 scale** + ``int8`` quantized coefficients (the
wire format the inflate runtime consumes). The encoder learns:

1. **Per-frame per-band scale** (4 float32 per frame) → 16 bytes/frame;
2. **Sparse int8 coefficients** for cH / cV / cD (cA is set to zero — the
   approximation band is not a residual; only the detail bands carry the
   contest-distortion-relevant edges where the renderer drops detail).

Solver: **coordinate-descent over float32 scales** (gradient-free; tiny
parameter count: ``4 * n_frames`` scalars). The int8 coefficients are
computed deterministically from the optimal scale + the ``decoded - GT``
residual via ``round(residual / scale).clamp(-128, 127)``. The encoder
sweeps over candidate ``(scale_cA, scale_cH, scale_cV, scale_cD)`` values
on a small grid (log-spaced) and picks the combination that minimizes the
proxy Lagrangian within the byte budget.

Per CLAUDE.md "MPS auth eval is NOISE" + "Forbidden device-selection defaults"
the encoder defaults to ``cpu``; MPS is acceptable as a research-signal
acceleration but is NEVER tagged authoritative.

Byte budget
-----------

Default budget: ``2000 bytes``. The current L1 wavelet inflate runtime accepts
only an empty residual or a dense per-frame coefficient stream. A dense stream
is far larger than the default budget, so this L2 encoder **fails closed** until
the sparse PacketIR wavelet stream lands. Resulting dense blob would be::

    per_frame_bytes = 4 * 4 (scales) + 4 * RGB * (H/2 * W/2)

For the L1 inflate runtime to consume bytes, the wire format currently stays:

    n_frames * (4 * 4B scales + 3 * (RGB_CHANNELS * (H/2 * W/2)) int8 coeffs)

That means skipped-frame zero padding does **not** reduce charged bytes. The
encoder therefore refuses budgets below the dense stream size and records the
required sparse-format work instead of emitting a misleading "2 KB" result.

References
----------

Mallat, S. (1989). Multi-resolution signal decomposition: the wavelet
representation. IEEE PAMI 11(7): 674-693.

Holub & Fridrich (2014). UNIWARD: universal distortion design for
steganography. The DDE-lab's distortion-aware embedding convention.

CLAUDE.md non-negotiables + handoff
``.omx/research/full_stack_score_lowering_synthesis_20260511_codex.md``
(score functional + R2 frontier marginal calc).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch

from tac.residual_basis.l2_score_aware_loss import (
    ResidualByteBudget,
    ScoreAwareLagrangian,
    compute_score_aware_proxy_loss,
)

# Camera dimensions per upstream/evaluate.py.
CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
HALF_H: Final[int] = CAMERA_H // 2
HALF_W: Final[int] = CAMERA_W // 2
RGB_CHANNELS: Final[int] = 3
PER_FRAME_BAND_BYTES: Final[int] = RGB_CHANNELS * HALF_H * HALF_W
PER_FRAME_BYTES: Final[int] = 4 * 4 + 4 * PER_FRAME_BAND_BYTES

# Historical target budget for the sparse PacketIR design. The current dense
# L1 wavelet wire is much larger, so encoder entrypoints fail closed at this
# value until a sparse runtime-consuming stream lands.
DEFAULT_BYTE_BUDGET: Final[int] = 2000


class WaveletEncoderL2Error(ValueError):
    """Raised on contract violations in the L2 wavelet encoder."""


@dataclass(frozen=True)
class WaveletEncoderL2Result:
    """Typed result of ``encode_wavelet_residual_l2``.

    Promotion-status invariants pinned False permanently.
    """

    residual_bytes: bytes
    n_frames_encoded: int
    n_frames_subsampled: int
    archive_bytes_estimate: int
    final_loss: float
    initial_loss: float
    n_iterations: int
    converged: bool
    diagnostics: dict[str, float]
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default="research_signal_l2_proxy", init=False)
    schema: str = field(default="wavelet_encoder_l2_score_aware_v1", init=False)

    def assert_invariants(self) -> None:
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise WaveletEncoderL2Error(
                "promotion-status invariants must remain False (L2 proxy encoder)"
            )
        if self.evidence_grade != "research_signal_l2_proxy":
            raise WaveletEncoderL2Error(
                f"evidence_grade must be research_signal_l2_proxy; got {self.evidence_grade!r}"
            )
        if self.n_frames_encoded <= 0:
            raise WaveletEncoderL2Error("n_frames_encoded must be > 0")
        if self.n_iterations <= 0:
            raise WaveletEncoderL2Error("n_iterations must be > 0")


def _haar_forward_2d_single_level(frame_chw: np.ndarray) -> tuple[np.ndarray, ...]:
    """Forward single-level orthonormal 2D Haar DWT on (3, H, W) frame.

    Returns ``(cA, cH, cV, cD)`` each of shape ``(3, H//2, W//2)``.
    """
    if frame_chw.ndim != 3 or frame_chw.shape[0] != RGB_CHANNELS:
        raise WaveletEncoderL2Error(
            f"expected (3, H, W); got {frame_chw.shape}"
        )
    h, w = frame_chw.shape[-2:]
    if h % 2 or w % 2:
        raise WaveletEncoderL2Error(f"H, W must be even; got ({h}, {w})")
    a = frame_chw[:, 0::2, 0::2]
    b = frame_chw[:, 0::2, 1::2]
    c = frame_chw[:, 1::2, 0::2]
    d = frame_chw[:, 1::2, 1::2]
    cA = 0.5 * (a + b + c + d)
    cH = 0.5 * (a + b - c - d)
    cV = 0.5 * (a - b + c - d)
    cD = 0.5 * (a - b - c + d)
    return cA, cH, cV, cD


def _haar_inverse_2d_single_level(
    cA: np.ndarray, cH: np.ndarray, cV: np.ndarray, cD: np.ndarray
) -> np.ndarray:
    """Inverse single-level orthonormal 2D Haar synthesis.

    Mirrors the inflate runtime's helper exactly so encoder + inflate are
    bit-identical.
    """
    if not (cA.shape == cH.shape == cV.shape == cD.shape):
        raise WaveletEncoderL2Error("band shapes must match")
    h, w = cA.shape[-2:]
    out = np.empty((*cA.shape[:-2], 2 * h, 2 * w), dtype=cA.dtype)
    out[..., 0::2, 0::2] = 0.5 * (cA + cH + cV + cD)
    out[..., 0::2, 1::2] = 0.5 * (cA + cH - cV - cD)
    out[..., 1::2, 0::2] = 0.5 * (cA - cH + cV - cD)
    out[..., 1::2, 1::2] = 0.5 * (cA - cH - cV + cD)
    return out


def _pack_wavelet_residual_bytes(
    *,
    n_frames: int,
    scales_per_frame: np.ndarray,  # (n_frames_kept, 4) float32
    bands_per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    kept_indices: list[int],
) -> bytes:
    """Pack the wire-format residual bytes per inflate runtime contract.

    Skipped frames (not in ``kept_indices``) get all-zero scales + all-zero
    int8 bands so the inflate runtime decodes them to a zero residual.
    """
    if len(scales_per_frame) != len(bands_per_frame):
        raise WaveletEncoderL2Error("scales and bands length mismatch")
    parts: list[bytes] = []
    kept_set = {idx: i for i, idx in enumerate(kept_indices)}
    zero_band = np.zeros((RGB_CHANNELS, HALF_H, HALF_W), dtype=np.int8)
    zero_scales = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
    for t in range(n_frames):
        if t in kept_set:
            i = kept_set[t]
            scales = scales_per_frame[i]
            cA_q, cH_q, cV_q, cD_q = bands_per_frame[i]
            parts.append(
                struct.pack("<4f", float(scales[0]), float(scales[1]),
                            float(scales[2]), float(scales[3]))
            )
            for band in (cA_q, cH_q, cV_q, cD_q):
                if band.shape != (RGB_CHANNELS, HALF_H, HALF_W):
                    raise WaveletEncoderL2Error(
                        f"band shape {band.shape} != expected {(RGB_CHANNELS, HALF_H, HALF_W)}"
                    )
                if band.dtype != np.int8:
                    raise WaveletEncoderL2Error(f"band dtype must be int8; got {band.dtype}")
                parts.append(band.tobytes())
        else:
            parts.append(zero_scales)
            for _ in range(4):
                parts.append(zero_band.tobytes())
    return b"".join(parts)


def dense_wavelet_residual_blob_bytes(n_frames: int) -> int:
    """Return charged residual bytes required by the current dense wavelet wire."""
    if n_frames <= 0:
        raise WaveletEncoderL2Error("n_frames must be > 0")
    return n_frames * PER_FRAME_BYTES


def _quantize_bands_for_frame(
    decoded_chw: np.ndarray,
    gt_chw: np.ndarray,
    *,
    skip_approximation: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[float, float, float, float]]:
    """Compute the per-frame residual bands + best per-band scale.

    Returns ``(residual_recon, (cA_q, cH_q, cV_q, cD_q), (scale_cA, scale_cH,
    scale_cV, scale_cD))``. ``residual_recon`` is the reconstructed residual
    ``cA_q*scale + ... `` synthesized back via inverse-Haar.

    If ``skip_approximation`` is True (default), the cA scale + coefficients
    are forced to 0 (the approximation band carries non-residual energy
    that would drift the decoded mean; we only want detail bands).
    """
    residual = (gt_chw - decoded_chw).astype(np.float64)
    cA, cH, cV, cD = _haar_forward_2d_single_level(residual)
    scales: list[float] = []
    bands_q: list[np.ndarray] = []
    bands_f = (cA, cH, cV, cD)
    for band_idx, band in enumerate(bands_f):
        if skip_approximation and band_idx == 0:
            scales.append(0.0)
            bands_q.append(np.zeros_like(band, dtype=np.int8))
            continue
        # Choose scale so |band|.max() maps to ~127 (use 120 for safety margin).
        max_abs = float(np.abs(band).max())
        if max_abs <= 1e-9:
            scales.append(0.0)
            bands_q.append(np.zeros_like(band, dtype=np.int8))
        else:
            scale = max_abs / 120.0
            scaled = band / scale
            q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
            scales.append(scale)
            bands_q.append(q)
    # Reconstruct the quantized residual for loss eval.
    bands_recon = [bands_q[i].astype(np.float64) * scales[i] for i in range(4)]
    recon = _haar_inverse_2d_single_level(*bands_recon)
    return (
        recon,
        (bands_q[0], bands_q[1], bands_q[2], bands_q[3]),
        (scales[0], scales[1], scales[2], scales[3]),
    )


def _select_subsample_indices(
    n_frames: int,
    budget_bytes: int,
) -> list[int]:
    """Pick frames to encode such that the residual blob fits the byte budget.

    Each kept frame costs ``PER_FRAME_BYTES`` (4*4 + 4*RGB_CHANNELS*HALF_H*HALF_W).
    Each skipped frame also costs the same in the current dense wire because
    skipped frames are zero-padded, so this helper refuses byte budgets smaller
    than the dense charged stream. Sparse subsampling requires a new PacketIR
    stream and matching inflate runtime.

    Returns indices sorted ascending.
    """
    if n_frames <= 0:
        raise WaveletEncoderL2Error("n_frames must be > 0")
    if budget_bytes <= 0:
        raise WaveletEncoderL2Error("budget_bytes must be > 0")
    dense_bytes = dense_wavelet_residual_blob_bytes(n_frames)
    if budget_bytes < dense_bytes:
        raise WaveletEncoderL2Error(
            "current wavelet residual wire format is dense: "
            f"n_frames={n_frames} requires residual_blob_bytes={dense_bytes}, "
            f"but byte_budget={budget_bytes}. Land a sparse PacketIR wavelet "
            "stream + matching inflate runtime before L2 score-aware dispatch."
        )
    return list(range(n_frames))


def encode_wavelet_residual_l2(
    decoded_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    byte_budget: int = DEFAULT_BYTE_BUDGET,
    pr106_wrapper_bytes: int = 186_822,
    n_iterations: int = 16,
    scale_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    lagrangian: ScoreAwareLagrangian | None = None,
    device: str = "cpu",
    skip_approximation_band: bool = True,
    seed: int | None = 20260511,
) -> WaveletEncoderL2Result:
    """L2 score-aware wavelet encoder over (T, H, W, 3) decoded vs GT frames.

    Parameters
    ----------
    decoded_frames
        ``(T, H, W, 3)`` uint8 or float — PR106 r2 decoded output.
    gt_frames
        ``(T, H, W, 3)`` uint8 or float — ground-truth frames.
    byte_budget
        Hard residual blob cap. The current L1 wire is dense, so this must be
        at least ``dense_wavelet_residual_blob_bytes(T)``; smaller sparse
        budgets fail closed until a sparse PacketIR wavelet runtime lands.
    pr106_wrapper_bytes
        PR106 r2 outer wrapper byte count (used for archive_bytes estimate
        in the proxy Lagrangian rate term).
    n_iterations
        Outer coordinate-descent iterations over ``scale_grid``. Default 16.
    scale_grid
        Per-band scale multipliers to sweep around the auto-tuned per-frame
        scale. Default ``(0.5, 0.75, 1.0, 1.25, 1.5)`` (5-point grid).
    lagrangian
        Optional override of the default contest-faithful Lagrangian.
    device
        ``"cpu"`` (default; contest-faithful), ``"mps"`` (research-signal
        only — encoder uses it as acceleration but tags result research-only),
        or ``"cuda"`` (if available). Per CLAUDE.md "MPS auth eval is NOISE"
        the result is ALWAYS tagged research_signal_l2_proxy regardless.

    Returns
    -------
    WaveletEncoderL2Result
        Frozen result with ``residual_bytes``, diagnostics, and the
        immutable ``score_claim=False`` / ``promotion_eligible=False`` /
        ``ready_for_exact_eval_dispatch=False`` contract.
    """
    if decoded_frames.shape != gt_frames.shape:
        raise WaveletEncoderL2Error(
            f"shape mismatch: decoded={decoded_frames.shape} gt={gt_frames.shape}"
        )
    if decoded_frames.ndim != 4 or decoded_frames.shape[-1] != RGB_CHANNELS:
        raise WaveletEncoderL2Error(
            f"expected (T, H, W, 3); got {decoded_frames.shape}"
        )
    n_frames, h, w, _ = decoded_frames.shape
    if n_frames % 2 != 0:
        raise WaveletEncoderL2Error(
            f"expected an even number of contest frames (pairs); got {n_frames}"
        )
    if h != CAMERA_H or w != CAMERA_W:
        raise WaveletEncoderL2Error(
            f"expected ({CAMERA_H}, {CAMERA_W}); got ({h}, {w})"
        )
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    lag = lagrangian or ScoreAwareLagrangian()
    lag.assert_invariants()
    budget = ResidualByteBudget(max_bytes=byte_budget + pr106_wrapper_bytes)
    budget.assert_invariants()

    kept_indices = _select_subsample_indices(
        n_frames=n_frames,
        budget_bytes=byte_budget,
    )
    n_keep = len(kept_indices)

    # Initial per-frame band quantization (auto-tuned scale per frame).
    bands_per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    scales_per_frame = np.zeros((n_keep, 4), dtype=np.float32)
    recon_per_frame = np.zeros((n_keep, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
    for i, t in enumerate(kept_indices):
        decoded_chw = np.transpose(decoded_frames[t], (2, 0, 1)).astype(np.float64)
        gt_chw = np.transpose(gt_frames[t], (2, 0, 1)).astype(np.float64)
        recon, bands_q, scales = _quantize_bands_for_frame(
            decoded_chw, gt_chw, skip_approximation=skip_approximation_band
        )
        bands_per_frame.append(bands_q)
        scales_per_frame[i] = scales
        # recon is (3, H, W); transpose to (H, W, 3).
        recon_per_frame[i] = np.transpose(recon, (1, 2, 0))

    # Compute the initial loss + then run coordinate-descent over the scale grid.
    def _evaluate_current(scales_arr: np.ndarray) -> tuple[float, dict[str, float], int]:
        """Apply current scales, recompute proxy loss + actual archive size."""
        # Rebuild bands_q with the new scales.
        local_bands: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        local_recon = np.zeros((n_keep, CAMERA_H, CAMERA_W, RGB_CHANNELS), dtype=np.float64)
        for i, t in enumerate(kept_indices):
            decoded_chw = np.transpose(decoded_frames[t], (2, 0, 1)).astype(np.float64)
            gt_chw = np.transpose(gt_frames[t], (2, 0, 1)).astype(np.float64)
            residual_f = gt_chw - decoded_chw
            cA, cH, cV, cD = _haar_forward_2d_single_level(residual_f)
            bands_q_local: list[np.ndarray] = []
            for band_idx, band in enumerate((cA, cH, cV, cD)):
                s = float(scales_arr[i, band_idx])
                if s <= 1e-9:
                    bands_q_local.append(np.zeros_like(band, dtype=np.int8))
                else:
                    q = np.clip(np.round(band / s), -128, 127).astype(np.int8)
                    bands_q_local.append(q)
            local_bands.append(tuple(bands_q_local))  # type: ignore[arg-type]
            bands_recon = [
                bands_q_local[idx].astype(np.float64) * float(scales_arr[i, idx])
                for idx in range(4)
            ]
            recon = _haar_inverse_2d_single_level(*bands_recon)
            local_recon[i] = np.transpose(recon, (1, 2, 0))
        residual_blob = _pack_wavelet_residual_bytes(
            n_frames=n_frames,
            scales_per_frame=scales_arr,
            bands_per_frame=local_bands,
            kept_indices=kept_indices,
        )
        # Total archive bytes = PR106 wrapper + residual length prefix + residual.
        # The wrapper itself is the PR106 r2 bytes + the 6-byte residual-archive
        # header + 4-byte residual length prefix.
        total_archive_bytes = pr106_wrapper_bytes + 6 + 4 + len(residual_blob)

        # Evaluate proxy Lagrangian on the kept subset (decoded + residual vs GT).
        decoded_kept = decoded_frames[kept_indices].astype(np.float32)
        gt_kept = gt_frames[kept_indices].astype(np.float32)
        decoded_with_residual = np.clip(
            decoded_kept + local_recon.astype(np.float32), 0, 255
        )
        dev = torch.device("cpu") if device != "cuda" or not torch.cuda.is_available() else torch.device("cuda")
        # Note: per CLAUDE.md, MPS is allowed as research-signal acceleration ONLY for
        # gradient inner loops — this coordinate-descent solver is gradient-FREE
        # so we stay on CPU for determinism.
        with torch.no_grad():
            decoded_t = torch.from_numpy(decoded_with_residual).to(dev)
            gt_t = torch.from_numpy(gt_kept).to(dev)
            loss, diag = compute_score_aware_proxy_loss(
                decoded_t, gt_t, total_archive_bytes, lagrangian=lag, budget=None,
                eval_roundtrip=True, yuv6_routing=True,
            )
        diag["wavelet_residual_blob_bytes"] = float(len(residual_blob))
        return float(loss.detach().item()), diag, len(residual_blob)

    init_loss, init_diag, init_blob_size = _evaluate_current(scales_per_frame)
    best_loss = init_loss
    best_scales = scales_per_frame.copy()
    best_diag = init_diag

    converged = False
    iters_done = 0
    for it in range(n_iterations):
        iters_done = it + 1
        any_improved = False
        # Coordinate-descent: for each (band) sweep scale multipliers.
        for band_idx in range(4):
            if skip_approximation_band and band_idx == 0:
                continue
            for multiplier in scale_grid:
                if abs(multiplier - 1.0) < 1e-12:
                    continue
                candidate_scales = best_scales.copy()
                candidate_scales[:, band_idx] *= multiplier
                cand_loss, cand_diag, _ = _evaluate_current(candidate_scales)
                if cand_loss + 1e-12 < best_loss:
                    best_loss = cand_loss
                    best_scales = candidate_scales
                    best_diag = cand_diag
                    any_improved = True
        if not any_improved:
            converged = True
            break

    # Rebuild the final residual blob from best_scales.
    final_bands: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, t in enumerate(kept_indices):
        decoded_chw = np.transpose(decoded_frames[t], (2, 0, 1)).astype(np.float64)
        gt_chw = np.transpose(gt_frames[t], (2, 0, 1)).astype(np.float64)
        residual_f = gt_chw - decoded_chw
        cA, cH, cV, cD = _haar_forward_2d_single_level(residual_f)
        bands_q_local: list[np.ndarray] = []
        for band_idx, band in enumerate((cA, cH, cV, cD)):
            s = float(best_scales[i, band_idx])
            if s <= 1e-9:
                bands_q_local.append(np.zeros_like(band, dtype=np.int8))
            else:
                q = np.clip(np.round(band / s), -128, 127).astype(np.int8)
                bands_q_local.append(q)
        final_bands.append(tuple(bands_q_local))  # type: ignore[arg-type]
    final_blob = _pack_wavelet_residual_bytes(
        n_frames=n_frames,
        scales_per_frame=best_scales,
        bands_per_frame=final_bands,
        kept_indices=kept_indices,
    )

    result = WaveletEncoderL2Result(
        residual_bytes=final_blob,
        n_frames_encoded=n_frames,
        n_frames_subsampled=n_keep,
        archive_bytes_estimate=pr106_wrapper_bytes + 6 + 4 + len(final_blob),
        final_loss=float(best_loss),
        initial_loss=float(init_loss),
        n_iterations=iters_done,
        converged=converged,
        diagnostics=best_diag,
    )
    result.assert_invariants()
    return result


__all__ = [
    "DEFAULT_BYTE_BUDGET",
    "PER_FRAME_BYTES",
    "WaveletEncoderL2Error",
    "WaveletEncoderL2Result",
    "dense_wavelet_residual_blob_bytes",
    "encode_wavelet_residual_l2",
]
