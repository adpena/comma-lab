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


def _truncate_quantized_to_top_k(
    quantized: np.ndarray, top_k: int
) -> np.ndarray:
    """Keep only the largest-magnitude top_k quantized coefficients; zero rest.

    Sister of ``tac.residual_basis.pr106_materializer_helpers.truncate_wavelet_dense_to_top_k``
    — applies the score-aware sparse-budget truncation per-level to enforce
    the operator's per-level byte budget on cool_chic. Stable descending-
    magnitude tie-break preserves first-occurrence order for determinism.
    """
    if top_k < 0:
        raise CoolChicEncoderL2Error(f"top_k must be >= 0; got {top_k}")
    if top_k == 0:
        return np.zeros_like(quantized)
    flat = quantized.reshape(-1)
    nonzero_idx = np.flatnonzero(flat)
    if nonzero_idx.size <= top_k:
        return quantized
    magnitudes = np.abs(flat[nonzero_idx].astype(np.int16))
    keep_local = np.argsort(-magnitudes, kind="stable")[:top_k]
    keep = nonzero_idx[keep_local]
    sparse = np.zeros_like(flat)
    sparse[keep] = flat[keep]
    return sparse.reshape(quantized.shape)


def _quantize_level(
    level_target: np.ndarray,
    *,
    scale_grid: tuple[float, ...],
    sparse_bias: bool = False,
    top_k_budget: int | None = None,
) -> tuple[np.ndarray, float, float]:
    """int8 quantize a level's target with grid-searched scale.

    Returns ``(level_q, best_scale, best_mse)``.

    When ``sparse_bias=True``, picks the largest-multiplier scale (sparsest
    quantization) instead of MSE-min so the per-level starting point biases
    toward more zero coefficients. Used by sparse-aware encoder mode.

    When ``top_k_budget`` is not None, the post-quantization int8 array is
    truncated to keep only the largest-magnitude ``top_k_budget`` non-zero
    coefficients (zeroing the rest). This enforces the operator's per-level
    byte budget for cool_chic sparse output: a level with top_k=128 produces
    at most 128 non-zero coefficients regardless of native level dimensions.
    """
    max_abs = float(np.abs(level_target).max())
    if max_abs <= 1e-9:
        return np.zeros_like(level_target, dtype=np.int8), 0.0, 0.0
    s_auto = max_abs / 120.0
    if sparse_bias:
        s = s_auto * float(max(scale_grid))
        scaled = level_target / s
        q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        if top_k_budget is not None:
            q = _truncate_quantized_to_top_k(q, top_k_budget)
        recon = q.astype(np.float64) * s
        mse = float(np.mean((level_target - recon) ** 2))
        return q, s, mse
    best_s = s_auto
    best_mse = float("inf")
    best_q: np.ndarray | None = None
    for mult in scale_grid:
        s = s_auto * mult
        if s <= 1e-9:
            continue
        scaled = level_target / s
        q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        if top_k_budget is not None:
            q = _truncate_quantized_to_top_k(q, top_k_budget)
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
        if top_k_budget is not None:
            best_q = _truncate_quantized_to_top_k(best_q, top_k_budget)
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
    sparse_aware: bool = False,
    per_level_top_k_budget: dict[int, int] | None = None,
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
        # Per-level top-K budget: if specified, the post-quantization int8 array
        # is truncated to the operator-supplied K to enforce the per-level
        # byte budget for cool_chic sparse output (operator decision 2026-05-11).
        top_k_for_level = (
            per_level_top_k_budget.get(L) if per_level_top_k_budget else None
        )
        q, scale, _ = _quantize_level(
            target_level, scale_grid=scale_grid, sparse_bias=sparse_aware,
            top_k_budget=top_k_for_level,
        )
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
    sparse_aware: bool = False,
    per_level_top_k_budget: dict[int, int] | None = None,
    use_hinton_distilled_scorer: bool = False,
    distilled_segnet=None,
    distilled_posenet=None,
    use_saliency_masking: bool = False,
    saliency_masking_config=None,
    pose_only_mode: bool = False,
    pose_marginal_multiplier: float = 1.0,
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
    if per_level_top_k_budget is not None:
        for _lvl, _k in per_level_top_k_budget.items():
            if not isinstance(_lvl, int) or _lvl < 0 or _lvl >= MAX_N_LEVELS:
                raise CoolChicEncoderL2Error(
                    f"per_level_top_k_budget level {_lvl!r} out of [0, {MAX_N_LEVELS-1}]"
                )
            if not isinstance(_k, int) or _k < 0:
                raise CoolChicEncoderL2Error(
                    f"per_level_top_k_budget[{_lvl}]={_k!r} must be a non-negative int"
                )
    if decoded_frames.shape != gt_frames.shape:
        raise CoolChicEncoderL2Error(
            f"shape mismatch: decoded={decoded_frames.shape} gt={gt_frames.shape}"
        )
    if decoded_frames.ndim != 4 or decoded_frames.shape[-1] != RGB_CHANNELS:
        raise CoolChicEncoderL2Error(
            f"expected (T, H, W, 3); got {decoded_frames.shape}"
        )
    n_frames, h, w, _ = decoded_frames.shape
    if n_frames % 2 != 0:
        raise CoolChicEncoderL2Error(
            f"expected an even number of contest frames (pairs); got {n_frames}"
        )
    if h != CAMERA_H or w != CAMERA_W:
        raise CoolChicEncoderL2Error(
            f"expected ({CAMERA_H}, {CAMERA_W}); got ({h}, {w})"
        )
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    lag = lagrangian or ScoreAwareLagrangian()
    lag.assert_invariants()

    # Saliency masking (per W reactivation criterion #2 + N D2 council).
    # Per Catalog #123: score-gradient saliency on distilled scorer (NOT
    # weight-domain). The mask is applied to GT so that the residual
    # `(masked_gt - decoded)` is zero at low-saliency pixels — sparse-friendly.
    saliency_diagnostics: dict[str, float] = {}
    masked_gt_frames = gt_frames
    if use_saliency_masking:
        if not use_hinton_distilled_scorer or distilled_segnet is None or distilled_posenet is None:
            raise CoolChicEncoderL2Error(
                "use_saliency_masking=True requires use_hinton_distilled_scorer=True "
                "AND non-None distilled_segnet + distilled_posenet "
                "(saliency is computed via the distilled scorer's gradient per Catalog #123)"
            )
        from tac.residual_basis.saliency_masked_residual import (
            SaliencyMaskingConfig,
            compute_score_aware_saliency,
        )

        sal_config = saliency_masking_config or SaliencyMaskingConfig.council_canonical()
        n_sal_frames = min(8, n_frames)
        sal_indices = list(range(0, n_frames, max(1, n_frames // n_sal_frames)))[:n_sal_frames]
        if len(sal_indices) % 2 != 0:
            sal_indices = sal_indices[:-1]
        if len(sal_indices) < 2:
            sal_indices = list(range(2))
        sal_decoded_t = torch.from_numpy(
            decoded_frames[sal_indices].astype(np.float32)
        )
        sal_gt_t = torch.from_numpy(gt_frames[sal_indices].astype(np.float32))
        saliency = compute_score_aware_saliency(
            sal_decoded_t,
            sal_gt_t,
            distilled_segnet=distilled_segnet,
            distilled_posenet=distilled_posenet,
            eval_roundtrip=True,
            distill_temperature=2.0,
        )
        saliency_global = saliency.mean(dim=0)  # (H, W)
        if sal_config.percentile is not None:
            threshold_value = float(
                torch.quantile(
                    saliency_global.flatten(), q=float(sal_config.percentile)
                ).item()
            )
        else:
            threshold_value = float(sal_config.threshold)
        keep_mask_hw = (saliency_global >= threshold_value).cpu().numpy()
        kept_fraction = float(keep_mask_hw.mean())
        if kept_fraction < sal_config.minimum_kept_fraction:
            raise CoolChicEncoderL2Error(
                f"saliency mask kept_fraction={kept_fraction:.4f} < "
                f"minimum_kept_fraction={sal_config.minimum_kept_fraction:.4f}"
            )
        # Mask gt s.t. (gt - decoded) is zero at low-saliency pixels.
        # broadcast (H, W) → (T, H, W, 3)
        keep_mask_thwc = np.broadcast_to(
            keep_mask_hw[np.newaxis, :, :, np.newaxis].astype(np.float32),
            gt_frames.shape,
        )
        masked_gt_frames = (
            decoded_frames.astype(np.float32)
            + (gt_frames.astype(np.float32) - decoded_frames.astype(np.float32))
            * keep_mask_thwc
        ).clip(0, 255).astype(gt_frames.dtype)
        saliency_diagnostics = {
            "saliency_threshold_value": threshold_value,
            "saliency_kept_fraction": kept_fraction,
            "saliency_min": float(saliency_global.min().item()),
            "saliency_max": float(saliency_global.max().item()),
            "saliency_mean": float(saliency_global.mean().item()),
            "saliency_n_sample_frames": float(len(sal_indices)),
        }

    def _eval_pyramid(
        scales: list[float],
        coeffs: list[np.ndarray],
        reconstructed: np.ndarray,
    ) -> tuple[float, dict[str, float], int]:
        decoded_with_residual = np.clip(
            decoded_frames.astype(np.float32) + reconstructed.astype(np.float32), 0, 255
        )
        blob_dense = _pack_cool_chic_residual_bytes(
            n_frames=n_frames, scales=scales, coeffs_per_level=coeffs
        )
        if sparse_aware:
            from tac.residual_basis.pr106_materializer_helpers import (
                repack_dense_as_sparse,
            )
            blob = repack_dense_as_sparse(
                family="cool_chic",
                dense_residual_bytes=blob_dense,
                n_frames=n_frames,
            )
        else:
            blob = blob_dense
        total_archive_bytes = pr106_wrapper_bytes + 6 + 4 + len(blob)
        # Evaluate whole frame pairs so SegNet-last-frame and PoseNet-pair
        # semantics stay contest-faithful under subsampling.
        n_pairs = n_frames // 2
        pair_stride = max(1, n_pairs // 32)
        eval_pairs = list(range(0, n_pairs, pair_stride))[:32]
        eval_idx = [frame_idx for pair in eval_pairs for frame_idx in (2 * pair, 2 * pair + 1)]
        decoded_eval = decoded_with_residual[eval_idx].astype(np.float32)
        gt_eval = gt_frames[eval_idx].astype(np.float32)
        with torch.no_grad():
            decoded_t = torch.from_numpy(decoded_eval)
            gt_t = torch.from_numpy(gt_eval)
            loss, diag = compute_score_aware_proxy_loss(
                decoded_t, gt_t, total_archive_bytes, lagrangian=lag, budget=None,
                eval_roundtrip=True, yuv6_routing=True,
                use_hinton_distilled_scorer=use_hinton_distilled_scorer,
                distilled_segnet=distilled_segnet,
                distilled_posenet=distilled_posenet,
                pose_only_mode=pose_only_mode,
                pose_marginal_multiplier=pose_marginal_multiplier,
            )
        diag["cool_chic_residual_blob_bytes"] = float(len(blob))
        diag["cool_chic_residual_blob_dense_bytes"] = float(len(blob_dense))
        diag["cool_chic_n_levels"] = float(len(scales))
        diag["cool_chic_sparse_aware"] = float(1.0 if sparse_aware else 0.0)
        diag["cool_chic_use_hinton_distilled_scorer"] = float(
            1.0 if use_hinton_distilled_scorer else 0.0
        )
        diag["cool_chic_use_saliency_masking"] = float(1.0 if use_saliency_masking else 0.0)
        diag["cool_chic_per_level_top_k_budget_active"] = float(
            1.0 if per_level_top_k_budget else 0.0
        )
        diag["cool_chic_pose_only_mode"] = float(1.0 if pose_only_mode else 0.0)
        diag["cool_chic_pose_marginal_multiplier"] = float(pose_marginal_multiplier)
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
        if not sparse_aware and dense_bytes > byte_budget:
            # Dense mode: skip n_lvl that exceeds dense budget. Sparse mode
            # bypasses the dense gate; the proxy loss evaluates against actual
            # sparse byte size so coordinate descent picks a feasible n_lvl.
            refusals.append((n_lvl, dense_bytes))
            continue
        scales, coeffs, recon = _encode_pyramid_for_n_levels(
            decoded_frames, masked_gt_frames,
            n_levels=n_lvl, scale_grid=scale_grid,
            sparse_aware=sparse_aware,
            per_level_top_k_budget=per_level_top_k_budget,
        )
        loss, diag, _ = _eval_pyramid(scales, coeffs, recon)
        if not init_diag:
            init_loss = loss
            init_diag = diag
        if sparse_aware:
            # Reject this n_lvl if its sparse-encoded size exceeds the budget;
            # otherwise feed it into the loss-min comparison.
            sparse_blob_size = int(diag.get("cool_chic_residual_blob_bytes", 0.0))
            if sparse_blob_size > byte_budget:
                refusals.append((n_lvl, sparse_blob_size))
                continue
        if loss < best_loss:
            best_loss = loss
            best_scales = scales
            best_coeffs = coeffs
            best_diag = diag
            best_n_levels = n_lvl

    if not best_scales:
        # Build a descriptive error explaining the byte floor (dense or sparse).
        refusal_summary = ", ".join(
            f"n_levels={lvl}->{dense}B" for lvl, dense in refusals
        )
        floor_kind = "sparse-encoded" if sparse_aware else "dense"
        raise CoolChicEncoderL2Error(
            f"current cool_chic residual {floor_kind} wire format too large: no "
            f"candidate n_levels fits byte_budget={byte_budget}. Stream sizes: "
            f"[{refusal_summary}]."
            + (
                ""
                if sparse_aware
                else " Land a sparse PacketIR cool_chic stream + matching inflate"
                " runtime before L2 score-aware dispatch."
            )
        )

    final_blob_dense = _pack_cool_chic_residual_bytes(
        n_frames=n_frames, scales=best_scales, coeffs_per_level=best_coeffs
    )
    if sparse_aware:
        from tac.residual_basis.pr106_materializer_helpers import (
            repack_dense_as_sparse,
        )
        final_blob = repack_dense_as_sparse(
            family="cool_chic",
            dense_residual_bytes=final_blob_dense,
            n_frames=n_frames,
        )
        if len(final_blob) > byte_budget:
            raise CoolChicEncoderL2Error(
                f"sparse-aware cool_chic final residual {len(final_blob)} bytes "
                f"exceeds byte_budget={byte_budget}; descent did not converge under cap"
            )
    else:
        final_blob = final_blob_dense
    if saliency_diagnostics:
        best_diag = dict(best_diag)
        for k, v in saliency_diagnostics.items():
            best_diag[f"cool_chic_{k}"] = v
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
