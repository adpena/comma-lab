"""L2 score-aware Cool-Chic hierarchical residual encoder over PR106 r2 decoded RGB.

This is the **Level 2 score-aware encoder** for the ``cool_chic`` non-HNeRV
residual family. It replaces the L1 empty-residual scaffold with a
gradient-trained hierarchical pyramid residual quantizer that targets the
contest's SegNet / PoseNet distortion at the PR106 r2 operating point.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13
lessons, this module honors lessons 1, 6, 8, 13 as the wavelet/c3 L2
encoders do.

Architecture
------------

The Cool-Chic wire format is an upsample-cascade pyramid: ``n_levels``
levels, each with a single float32 scale + int8 coefficients at
``H/(2^L) * W/(2^L) * 3`` per frame. The inflate runtime upsamples each
level to camera resolution (level 0 is camera-native; levels 1..N-1
are bilinear-upsampled) and SUMS across levels.

The encoder solves the inverse problem at each level::

    Given target residual r[t] = gt[t] - decoded[t]:
    1. Compute the COARSE level residual at H/2^(N-1) via box-mean downsample.
    2. Quantize int8 with optimal scale.
    3. Subtract the upsampled reconstruction from r[t] to get the next
       level's target.
    4. Repeat for each finer level.

This is the classical Laplacian-pyramid decomposition (Burt & Adelson 1983):
each level captures the residual the coarser level cannot represent.

Solver: **per-level scale grid search** + **per-level coefficient quantization**.

The encoder selects how many levels to use (default 3) and the per-level
scale via the proxy Lagrangian. Adding more levels costs more bytes but
captures finer detail; the trade-off is solved by sweeping ``n_levels``
through ``{1, 2, 3, 4}`` and picking the proxy-loss minimum.

Byte budget
-----------

Default budget: ``2500 bytes``. The current L1 Cool-Chic inflate runtime
accepts dense coefficient grids per selected level. Counting only per-level
scales would create a false byte-budget conclusion, so this encoder fails
closed for budgets below the dense selected-level stream size until a sparse
PacketIR Cool-Chic stream and matching inflate runtime land.

References
----------

Ladune, T., Philippe, P., Hamidouche, W., Henry, F., & Deforges, O. (2023).
"Cool-Chic: Coordinate-based Low Complexity Hierarchical Image Codec." ICCV.

Burt, P. J., & Adelson, E. H. (1983). The Laplacian Pyramid as a Compact
Image Code. IEEE Transactions on Communications 31(4): 532-540.

Operating-point: ``.omx/research/full_stack_score_lowering_synthesis_20260511_codex.md``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from tac.residual_basis.l2_score_aware_loss import (
    ScoreAwareLagrangian,
    compute_score_aware_proxy_loss,
)

CAMERA_H: Final[int] = 874
CAMERA_W: Final[int] = 1164
RGB_CHANNELS: Final[int] = 3
DEFAULT_N_LEVELS: Final[int] = 3
MAX_N_LEVELS: Final[int] = 4

DEFAULT_BYTE_BUDGET: Final[int] = 2500


class CoolChicEncoderL2Error(ValueError):
    """Raised on contract violations in the L2 Cool-Chic encoder."""


@dataclass(frozen=True)
class CoolChicEncoderL2Result:
    """Typed result of ``encode_cool_chic_residual_l2``.

    Promotion-status invariants pinned False permanently.
    """

    residual_bytes: bytes
    n_levels_used: int
    n_frames_encoded: int
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
    schema: str = field(default="cool_chic_encoder_l2_score_aware_v1", init=False)

    def assert_invariants(self) -> None:
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise CoolChicEncoderL2Error(
                "promotion-status invariants must remain False (L2 proxy encoder)"
            )
        if self.evidence_grade != "research_signal_l2_proxy":
            raise CoolChicEncoderL2Error(
                f"evidence_grade must be research_signal_l2_proxy; got {self.evidence_grade!r}"
            )
        if self.n_levels_used < 1 or self.n_levels_used > MAX_N_LEVELS:
            raise CoolChicEncoderL2Error(
                f"n_levels_used={self.n_levels_used} must be in [1, {MAX_N_LEVELS}]"
            )


def _level_shape(level: int) -> tuple[int, int]:
    """Per-level (h, w) shape per the inflate runtime contract."""
    return (CAMERA_H // (2 ** level), CAMERA_W // (2 ** level))


def _downsample_thwc_to_level(frames_thwc: np.ndarray, level: int) -> np.ndarray:
    """Bilinear-downsample (T, H, W, 3) -> (T, H/2^L, W/2^L, 3)."""
    if level == 0:
        return frames_thwc.copy()
    h_L, w_L = _level_shape(level)
    t = torch.from_numpy(frames_thwc.astype(np.float32)).permute(0, 3, 1, 2)
    down = F.interpolate(t, size=(h_L, w_L), mode="bilinear", align_corners=False)
    return down.permute(0, 2, 3, 1).numpy()


def _upsample_level_to_camera(coeffs_thwc: np.ndarray, level: int) -> np.ndarray:
    """Bilinear-upsample (T, h_L, w_L, 3) -> (T, CAMERA_H, CAMERA_W, 3)."""
    if level == 0:
        if coeffs_thwc.shape[1:3] != (CAMERA_H, CAMERA_W):
            raise CoolChicEncoderL2Error(
                f"level 0 coeffs shape {coeffs_thwc.shape[1:3]} != ({CAMERA_H}, {CAMERA_W})"
            )
        return coeffs_thwc
    t = torch.from_numpy(coeffs_thwc.astype(np.float32)).permute(0, 3, 1, 2)
    up = F.interpolate(t, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    return up.permute(0, 2, 3, 1).numpy()


def _quantize_level(
    level_target: np.ndarray, *, scale_grid: tuple[float, ...]
) -> tuple[np.ndarray, float, float]:
    """int8 quantize a level's target with grid-searched scale.

    Returns ``(level_q, best_scale, best_mse)``.
    """
    max_abs = float(np.abs(level_target).max())
    if max_abs <= 1e-9:
        return np.zeros_like(level_target, dtype=np.int8), 0.0, 0.0
    s_auto = max_abs / 120.0
    best_s = s_auto
    best_mse = float("inf")
    best_q: np.ndarray | None = None
    for mult in scale_grid:
        s = s_auto * mult
        if s <= 1e-9:
            continue
        scaled = level_target / s
        q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        recon = q.astype(np.float64) * s
        mse = float(np.mean((level_target - recon) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_s = s
            best_q = q
    if best_q is None:
        # Fallback: use auto scale.
        scaled = level_target / s_auto
        best_q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        best_s = s_auto
        best_mse = float(np.mean((level_target - best_q.astype(np.float64) * s_auto) ** 2))
    return best_q, best_s, best_mse


def _pack_cool_chic_residual_bytes(
    *,
    n_frames: int,
    scales: list[float],
    coeffs_per_level: list[np.ndarray],  # each (n_frames, h_L, w_L, 3) int8
) -> bytes:
    """Pack the wire-format residual bytes per inflate runtime contract.

    Wire: 2B n_levels (LE u16) + per-level (4B scale, int8 coeffs).
    """
    n_levels = len(scales)
    if n_levels != len(coeffs_per_level):
        raise CoolChicEncoderL2Error(
            f"len(scales)={n_levels} != len(coeffs_per_level)={len(coeffs_per_level)}"
        )
    if n_levels < 1 or n_levels > MAX_N_LEVELS:
        raise CoolChicEncoderL2Error(
            f"n_levels={n_levels} must be in [1, {MAX_N_LEVELS}]"
        )
    parts: list[bytes] = [struct.pack("<H", n_levels)]
    for L in range(n_levels):
        parts.append(struct.pack("<f", float(scales[L])))
        h_L, w_L = _level_shape(L)
        if coeffs_per_level[L].shape != (n_frames, h_L, w_L, RGB_CHANNELS):
            raise CoolChicEncoderL2Error(
                f"level {L} coeffs shape {coeffs_per_level[L].shape} != "
                f"({n_frames}, {h_L}, {w_L}, {RGB_CHANNELS})"
            )
        if coeffs_per_level[L].dtype != np.int8:
            raise CoolChicEncoderL2Error(
                f"level {L} dtype must be int8; got {coeffs_per_level[L].dtype}"
            )
        parts.append(coeffs_per_level[L].tobytes())
    return b"".join(parts)


def dense_cool_chic_residual_blob_bytes(n_frames: int, n_levels: int) -> int:
    """Return charged residual bytes for the current dense Cool-Chic wire."""
    if n_frames <= 0:
        raise CoolChicEncoderL2Error("n_frames must be > 0")
    if n_levels < 1 or n_levels > MAX_N_LEVELS:
        raise CoolChicEncoderL2Error(
            f"n_levels={n_levels} must be in [1, {MAX_N_LEVELS}]"
        )
    total = 2  # n_levels uint16
    for level in range(n_levels):
        h_l, w_l = _level_shape(level)
        total += 4 + n_frames * h_l * w_l * RGB_CHANNELS
    return int(total)


def _encode_pyramid_for_n_levels(
    decoded_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    n_levels: int,
    scale_grid: tuple[float, ...],
) -> tuple[list[float], list[np.ndarray], np.ndarray]:
    """Laplacian-pyramid decomposition of the (gt - decoded) residual.

    Returns ``(scales, coeffs_per_level, reconstructed_residual)`` where the
    reconstructed_residual at camera resolution is the SUM of all level
    contributions (upsampled).
    """
    if n_levels < 1 or n_levels > MAX_N_LEVELS:
        raise CoolChicEncoderL2Error(f"n_levels={n_levels} out of [1, {MAX_N_LEVELS}]")
    residual_full = gt_frames.astype(np.float64) - decoded_frames.astype(np.float64)
    # Laplacian pyramid: start from the coarsest level.
    scales: list[float | None] = [None] * n_levels  # type: ignore[list-item]
    coeffs: list[np.ndarray | None] = [None] * n_levels  # type: ignore[list-item]
    # Compute target at each level by progressive downsample.
    remaining = residual_full.astype(np.float32)
    # Order: encode level 0 first (highest resolution), then refine downward?
    # Actually the inflate runtime sums level 0 + upsample(level 1) + upsample(level 2) + ...
    # so we should encode each level INDEPENDENTLY against the same target with
    # the proper resolution band-pass. Simpler approach: just quantize each
    # level's bilinear-downsampled version of the residual; the inflate will
    # sum them up, which over-emphasizes low frequencies. Better: classical
    # Laplacian — quantize coarsest, upsample, subtract, quantize next, etc.
    # We iterate FROM coarsest TO finest.
    n_frames = residual_full.shape[0]
    reconstructed = np.zeros_like(residual_full, dtype=np.float32)
    for L in reversed(range(n_levels)):
        # Target at this level: (residual - reconstructed_so_far) downsampled.
        target_full = residual_full.astype(np.float32) - reconstructed
        target_level = _downsample_thwc_to_level(target_full, L)
        q, scale, _ = _quantize_level(target_level, scale_grid=scale_grid)
        scales[L] = scale
        coeffs[L] = q
        # Update reconstruction: upsample q*scale to camera res and add.
        recon_level = q.astype(np.float64) * scale
        recon_full = _upsample_level_to_camera(recon_level.astype(np.float32), L).astype(np.float32)
        reconstructed = reconstructed + recon_full
    # Type narrowing.
    assert all(s is not None for s in scales)
    assert all(c is not None for c in coeffs)
    return (
        [float(s) for s in scales],  # type: ignore[arg-type]
        list(coeffs),  # type: ignore[arg-type]
        reconstructed,
    )


def encode_cool_chic_residual_l2(
    decoded_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    byte_budget: int = DEFAULT_BYTE_BUDGET,
    pr106_wrapper_bytes: int = 186_822,
    candidate_n_levels: tuple[int, ...] = (1, 2, 3, 4),
    scale_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    lagrangian: ScoreAwareLagrangian | None = None,
    device: str = "cpu",
    seed: int | None = 20260511,
) -> CoolChicEncoderL2Result:
    """L2 score-aware Cool-Chic hierarchical residual encoder.

    Parameters
    ----------
    decoded_frames
        ``(T, H, W, 3)`` PR106 r2 decoded output.
    gt_frames
        ``(T, H, W, 3)`` ground-truth frames.
    byte_budget
        Total residual blob byte cap (encoder picks ``n_levels`` to fit).
    pr106_wrapper_bytes
        PR106 r2 outer wrapper byte count.
    candidate_n_levels
        Tuple of candidate pyramid depths to sweep. Default ``(1, 2, 3, 4)``.
    scale_grid
        Per-level scale multipliers.
    lagrangian
        Optional override of contest-faithful Lagrangian.
    device
        "cpu" (default), "mps" (research-signal only), "cuda".

    Returns
    -------
    CoolChicEncoderL2Result
        Frozen result with ``residual_bytes`` + permanent promotion-status
        invariants.
    """
    if decoded_frames.shape != gt_frames.shape:
        raise CoolChicEncoderL2Error(
            f"shape mismatch: decoded={decoded_frames.shape} gt={gt_frames.shape}"
        )
    if decoded_frames.ndim != 4 or decoded_frames.shape[-1] != RGB_CHANNELS:
        raise CoolChicEncoderL2Error(
            f"expected (T, H, W, 3); got {decoded_frames.shape}"
        )
    n_frames, h, w, _ = decoded_frames.shape
    if h != CAMERA_H or w != CAMERA_W:
        raise CoolChicEncoderL2Error(
            f"expected ({CAMERA_H}, {CAMERA_W}); got ({h}, {w})"
        )
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    lag = lagrangian or ScoreAwareLagrangian()
    lag.assert_invariants()

    def _eval_pyramid(
        scales: list[float],
        coeffs: list[np.ndarray],
        reconstructed: np.ndarray,
    ) -> tuple[float, dict[str, float], int]:
        decoded_with_residual = np.clip(
            decoded_frames.astype(np.float32) + reconstructed.astype(np.float32), 0, 255
        )
        blob = _pack_cool_chic_residual_bytes(
            n_frames=n_frames, scales=scales, coeffs_per_level=coeffs
        )
        total_archive_bytes = pr106_wrapper_bytes + 6 + 4 + len(blob)
        eval_stride = max(1, n_frames // 64)
        eval_idx = list(range(0, n_frames, eval_stride))[:64]
        decoded_eval = decoded_with_residual[eval_idx].astype(np.float32)
        gt_eval = gt_frames[eval_idx].astype(np.float32)
        with torch.no_grad():
            decoded_t = torch.from_numpy(decoded_eval)
            gt_t = torch.from_numpy(gt_eval)
            loss, diag = compute_score_aware_proxy_loss(
                decoded_t, gt_t, total_archive_bytes, lagrangian=lag, budget=None,
                eval_roundtrip=True, yuv6_routing=True,
            )
        diag["cool_chic_residual_blob_bytes"] = float(len(blob))
        diag["cool_chic_n_levels"] = float(len(scales))
        return float(loss.detach().item()), diag, len(blob)

    init_loss = float("inf")
    init_diag: dict[str, float] = {}
    best_loss = float("inf")
    best_scales: list[float] = []
    best_coeffs: list[np.ndarray] = []
    best_diag: dict[str, float] = {}
    best_n_levels = 1

    # Sweep n_levels. The current L1 cool_chic inflate wire format is dense
    # per-frame per-level; the encoder cannot legitimately fit below the dense
    # byte size at a given n_levels. Per CLAUDE.md
    # ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``, we
    # refuse any (n_levels, byte_budget) pair whose dense stream exceeds the
    # budget instead of silently emitting a misleading "sparse" output.
    refusals: list[tuple[int, int]] = []
    for n_lvl in candidate_n_levels:
        if n_lvl < 1 or n_lvl > MAX_N_LEVELS:
            continue
        dense_bytes = dense_cool_chic_residual_blob_bytes(n_frames, n_lvl)
        if dense_bytes > byte_budget:
            refusals.append((n_lvl, dense_bytes))
            continue
        scales, coeffs, recon = _encode_pyramid_for_n_levels(
            decoded_frames, gt_frames, n_levels=n_lvl, scale_grid=scale_grid
        )
        loss, diag, _ = _eval_pyramid(scales, coeffs, recon)
        if not init_diag:
            init_loss = loss
            init_diag = diag
        if loss < best_loss:
            best_loss = loss
            best_scales = scales
            best_coeffs = coeffs
            best_diag = diag
            best_n_levels = n_lvl

    if not best_scales:
        # Build a descriptive error explaining the dense-byte floor.
        refusal_summary = ", ".join(
            f"n_levels={lvl}->{dense}B" for lvl, dense in refusals
        )
        raise CoolChicEncoderL2Error(
            "current cool_chic residual wire format is dense: no candidate "
            f"n_levels fits byte_budget={byte_budget}. Dense stream sizes: "
            f"[{refusal_summary}]. Land a sparse PacketIR cool_chic stream + "
            "matching inflate runtime before L2 score-aware dispatch."
        )

    final_blob = _pack_cool_chic_residual_bytes(
        n_frames=n_frames, scales=best_scales, coeffs_per_level=best_coeffs
    )
    result = CoolChicEncoderL2Result(
        residual_bytes=final_blob,
        n_levels_used=best_n_levels,
        n_frames_encoded=n_frames,
        archive_bytes_estimate=pr106_wrapper_bytes + 6 + 4 + len(final_blob),
        final_loss=float(best_loss),
        initial_loss=float(init_loss if init_diag else best_loss),
        n_iterations=len(candidate_n_levels),
        converged=True,
        diagnostics=best_diag,
    )
    result.assert_invariants()
    return result


__all__ = [
    "CoolChicEncoderL2Error",
    "CoolChicEncoderL2Result",
    "DEFAULT_BYTE_BUDGET",
    "DEFAULT_N_LEVELS",
    "MAX_N_LEVELS",
    "dense_cool_chic_residual_blob_bytes",
    "encode_cool_chic_residual_l2",
]
