"""L2 score-aware C3 (conditional residual) encoder over PR106 r2 decoded RGB.

This is the **Level 2 score-aware encoder** for the ``c3`` non-HNeRV residual
family. It replaces the L1 empty-residual scaffold with a gradient-trained
per-frame-delta conditional residual quantizer that targets the contest's
SegNet / PoseNet distortion at the PR106 r2 operating point.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13
lessons, this module honors:

* lesson 1 (score-aware substrate training): inner loop runs the
  ``compute_score_aware_proxy_loss`` Lagrangian against the GT frames decoded
  from ``upstream/videos/0.mkv``;
* lesson 6 (score-domain Lagrangian): proxy uses contest-faithful weights;
* lesson 8 (eval-roundtrip-aware + differentiable YUV6): the proxy loss
  routes through ``apply_eval_roundtrip_during_training`` and
  ``differentiable_rgb_to_yuv6``;
* lesson 13 (KILL is LAST RESORT): the encoder NEVER emits a KILL verdict.

The encoder emits ``score_claim=False``, ``promotion_eligible=False``,
``ready_for_exact_eval_dispatch=False`` permanently. The output residual
bytes are CONSUMED by ``submissions/pr106_c3_residual_sidecar/inflate.py``
which decodes them via cumulative-sum + bilinear-upsample exactly mirroring
the encoder's forward pass.

Architecture
------------

The C3 wire format is per-frame ``(4B float32 scale, int8 coeffs at
H/4 * W/4 * 3)``. The inflate runtime cumulative-sums across the time axis
(so frame t's residual is the sum of all deltas[0:t+1] * scale[0:t+1]) and
bilinear-upsamples to camera resolution.

This integrates-then-upsamples conditioning is C3-style: each frame's
residual depends on all PRIOR frame deltas. The encoder must therefore
solve the INVERSE problem: given the target per-frame residual
``r[t] = gt[t] - decoded[t]`` (downsampled to quarter res), find deltas
``d[t]`` such that ``cumsum_t(d) ≈ r``. The optimal delta is the
**first-difference** of ``r``::

    d[0] = r[0]
    d[t] = r[t] - r[t-1]  for t >= 1

After int8 quantization with a per-frame scale, the deltas are entropy-
sparser than the absolute residuals (motion-prediction style).

Solver: **per-frame scale grid search + first-difference encoding**.

* Compute per-frame target residual at quarter resolution
  (``r_q[t] = downsample(gt[t] - decoded[t], scale=4)``).
* Compute first-difference ``d[t] = r_q[t] - r_q[t-1]`` (for t >= 1) and
  ``d[0] = r_q[0]``.
* For each frame, pick the scale that minimizes ``|round(d[t]/s) * s -
  d[t]|`` subject to ``|round(d[t]/s)| <= 127``.
* Run the proxy Lagrangian over the reconstructed cumsum-and-upsample
  residual to ensure the picked scales improve seg+pose without exceeding
  the rate budget.

Byte budget
-----------

Default budget: ``1500 bytes``. The current L1 C3 inflate runtime accepts a
dense per-frame delta stream only. Skipped-frame zero padding does not reduce
charged bytes, so the L2 encoder fails closed below the dense stream size until
a sparse PacketIR C3 stream and matching inflate runtime land.

References
----------

Kim, H., Bauer, M., Theis, L., Schwarz, J. R., & Dupont, E. (2024).
"C3: High-performance and low-complexity neural compression from a single
image." CVPR.

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
QUARTER_H: Final[int] = CAMERA_H // 4
QUARTER_W: Final[int] = CAMERA_W // 4
RGB_CHANNELS: Final[int] = 3
PER_FRAME_DELTA_BYTES: Final[int] = QUARTER_H * QUARTER_W * RGB_CHANNELS
PER_FRAME_BYTES: Final[int] = 4 + PER_FRAME_DELTA_BYTES

# Historical target budget for the sparse PacketIR design. The current dense
# L1 C3 wire is much larger, so encoder entrypoints fail closed at this value
# until a sparse runtime-consuming stream lands.
DEFAULT_BYTE_BUDGET: Final[int] = 1500


class C3EncoderL2Error(ValueError):
    """Raised on contract violations in the L2 C3 encoder."""


@dataclass(frozen=True)
class C3EncoderL2Result:
    """Typed result of ``encode_c3_residual_l2``.

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
    schema: str = field(default="c3_encoder_l2_score_aware_v1", init=False)

    def assert_invariants(self) -> None:
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise C3EncoderL2Error(
                "promotion-status invariants must remain False (L2 proxy encoder)"
            )
        if self.evidence_grade != "research_signal_l2_proxy":
            raise C3EncoderL2Error(
                f"evidence_grade must be research_signal_l2_proxy; got {self.evidence_grade!r}"
            )


def _downsample_to_quarter(frame_hwc: np.ndarray) -> np.ndarray:
    """Bilinear-downsample (H, W, 3) -> (QUARTER_H, QUARTER_W, 3) via torch.

    Uses torch's interpolate for exact parity with the inflate runtime's
    bilinear-upsample inverse.
    """
    if frame_hwc.ndim != 3 or frame_hwc.shape[-1] != RGB_CHANNELS:
        raise C3EncoderL2Error(f"expected (H, W, 3); got {frame_hwc.shape}")
    t = torch.from_numpy(frame_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    down = F.interpolate(t, size=(QUARTER_H, QUARTER_W), mode="bilinear", align_corners=False)
    return down.squeeze(0).permute(1, 2, 0).numpy()


def _pack_c3_residual_bytes(
    *,
    n_frames: int,
    scales_per_frame: np.ndarray,  # (n_frames,) float32
    deltas_q: np.ndarray,  # (n_frames, QUARTER_H, QUARTER_W, 3) int8
) -> bytes:
    """Pack the wire-format residual bytes per inflate runtime contract."""
    if scales_per_frame.shape != (n_frames,):
        raise C3EncoderL2Error(
            f"scales shape {scales_per_frame.shape} != ({n_frames},)"
        )
    if deltas_q.shape != (n_frames, QUARTER_H, QUARTER_W, RGB_CHANNELS):
        raise C3EncoderL2Error(
            f"deltas shape {deltas_q.shape} != "
            f"({n_frames}, {QUARTER_H}, {QUARTER_W}, {RGB_CHANNELS})"
        )
    if deltas_q.dtype != np.int8:
        raise C3EncoderL2Error(f"deltas dtype must be int8; got {deltas_q.dtype}")
    parts: list[bytes] = []
    for t in range(n_frames):
        parts.append(struct.pack("<f", float(scales_per_frame[t])))
        parts.append(deltas_q[t].tobytes())
    return b"".join(parts)


def dense_c3_residual_blob_bytes(n_frames: int) -> int:
    """Dense per-frame stream byte count under the current inflate wire format.

    Per inflate runtime contract: ``n_frames * PER_FRAME_BYTES``. Skipped frames
    still cost the same number of bytes (zero-padded) — there is no sparse
    encoding in the L1 wire format. This helper is the budget honesty floor.
    """
    if n_frames <= 0:
        raise C3EncoderL2Error("n_frames must be > 0")
    return int(n_frames * PER_FRAME_BYTES)


def _select_subsample_indices(n_frames: int, budget_bytes: int) -> list[int]:
    """Pick frames to encode such that meaningful coefficients fit budget.

    The current L1 c3 inflate wire format is dense per-frame; skipped frames
    are zero-padded but still charged. This helper therefore refuses budgets
    below the dense byte count rather than silently emitting a misleading
    "sparse" output (per CLAUDE.md
    ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``).
    Sparse subsampling requires a new PacketIR c3 stream + matching inflate.
    """
    if n_frames <= 0:
        raise C3EncoderL2Error("n_frames must be > 0")
    if budget_bytes <= 0:
        raise C3EncoderL2Error("budget_bytes must be > 0")
    dense_bytes = dense_c3_residual_blob_bytes(n_frames)
    if budget_bytes < dense_bytes:
        raise C3EncoderL2Error(
            "current c3 residual wire format is dense: "
            f"n_frames={n_frames} requires residual_blob_bytes={dense_bytes}, "
            f"but byte_budget={budget_bytes}. Land a sparse PacketIR c3 stream "
            "+ matching inflate runtime before L2 score-aware dispatch."
        )
    return list(range(n_frames))


def encode_c3_residual_l2(
    decoded_frames: np.ndarray,
    gt_frames: np.ndarray,
    *,
    byte_budget: int = DEFAULT_BYTE_BUDGET,
    pr106_wrapper_bytes: int = 186_822,
    n_iterations: int = 8,
    scale_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    lagrangian: ScoreAwareLagrangian | None = None,
    device: str = "cpu",
    seed: int | None = 20260511,
    sparse_aware: bool = False,
    use_hinton_distilled_scorer: bool = False,
    distilled_segnet=None,
    distilled_posenet=None,
    use_saliency_masking: bool = False,
    saliency_masking_config=None,
    pose_only_mode: bool = False,
    pose_marginal_multiplier: float = 1.0,
) -> C3EncoderL2Result:
    """L2 score-aware C3 conditional residual encoder.

    Parameters
    ----------
    decoded_frames
        ``(T, H, W, 3)`` PR106 r2 decoded output.
    gt_frames
        ``(T, H, W, 3)`` ground-truth frames.
    byte_budget
        Hard residual blob cap. When ``sparse_aware=False`` (default), the L1
        wire is dense and this must be ``>= dense_c3_residual_blob_bytes(T)``.
        When ``sparse_aware=True``, this is the cap on the sparse-encoded
        ``residual_bytes`` actually emitted; the encoder evaluates the proxy
        Lagrangian against the sparse byte size and returns sparse-repacked
        bytes ready for the ``c3_sparse`` (0x22) family.
    pr106_wrapper_bytes
        PR106 r2 outer wrapper byte count.
    n_iterations
        Outer coordinate-descent iterations.
    scale_grid
        Per-frame scale multipliers to sweep.
    lagrangian
        Optional override of contest-faithful Lagrangian.
    device
        "cpu" (default), "mps" (research-signal only), "cuda".
    sparse_aware
        If True, optimize for byte-cost AFTER sparse PacketIR encoding (RLE-of-
        zeros over per-frame int8 deltas, wrapped in temporal-subsampled). The
        encoder returns ``residual_bytes`` that are the SPARSE-REPACKED bytes
        ready for the 0x22 family inflate runtime; the rate term in the proxy
        Lagrangian uses ``len(sparse_residual_bytes)`` not the dense size. Per
        HNeRV parity discipline lesson 6 (score-domain Lagrangian) this makes
        the inner loop directly optimize the contest-relevant byte cost.

    Returns
    -------
    C3EncoderL2Result
        Frozen result with ``residual_bytes`` + permanent promotion-status
        invariants. When ``sparse_aware=True``, ``residual_bytes`` are
        sparse-repacked (family ``c3_sparse``, format_id 0x22) and
        ``archive_bytes_estimate`` reflects the sparse-encoded size.
    """
    if decoded_frames.shape != gt_frames.shape:
        raise C3EncoderL2Error(
            f"shape mismatch: decoded={decoded_frames.shape} gt={gt_frames.shape}"
        )
    if decoded_frames.ndim != 4 or decoded_frames.shape[-1] != RGB_CHANNELS:
        raise C3EncoderL2Error(
            f"expected (T, H, W, 3); got {decoded_frames.shape}"
        )
    n_frames, h, w, _ = decoded_frames.shape
    if n_frames % 2 != 0:
        raise C3EncoderL2Error(
            f"expected an even number of contest frames (pairs); got {n_frames}"
        )
    if h != CAMERA_H or w != CAMERA_W:
        raise C3EncoderL2Error(
            f"expected ({CAMERA_H}, {CAMERA_W}); got ({h}, {w})"
        )
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    lag = lagrangian or ScoreAwareLagrangian()
    lag.assert_invariants()
    if sparse_aware:
        # Sparse mode: bypass the dense-budget gate. The byte budget refers to
        # the sparse-encoded ``residual_bytes`` cap; the encoder evaluates the
        # rate term against the actual sparse byte size each iteration.
        if byte_budget <= 0:
            raise C3EncoderL2Error("byte_budget must be > 0")
        kept_indices = list(range(n_frames))
        # Sparsity coefficient threshold sweep: per HNeRV parity discipline
        # lesson 6 we need the proxy Lagrangian to find a coefficient-magnitude
        # cutoff that maximizes sparsity for a given seg+pose loss. The
        # coordinate descent picks the threshold that minimizes the proxy loss.
        sparse_threshold_grid: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
    else:
        kept_indices = _select_subsample_indices(n_frames, byte_budget)
        sparse_threshold_grid = (0,)
    n_keep = len(kept_indices)
    is_kept = np.zeros(n_frames, dtype=bool)
    for t in kept_indices:
        is_kept[t] = True

    # Step 0: saliency masking (per W reactivation criterion #2 + N D2 council).
    # Zero out residual at low-saliency pixels BEFORE quantization so the
    # zeroed regions become RLE-of-zeros runs in the sparse PacketIR encoding.
    # Per Catalog #123: saliency is computed via score gradient on the
    # distilled scorer (NOT a weight-domain proxy).
    saliency_diagnostics: dict[str, float] = {}
    if use_saliency_masking:
        if not use_hinton_distilled_scorer or distilled_segnet is None or distilled_posenet is None:
            raise C3EncoderL2Error(
                "use_saliency_masking=True requires use_hinton_distilled_scorer=True "
                "AND non-None distilled_segnet + distilled_posenet "
                "(saliency is computed via the distilled scorer's gradient per Catalog #123)"
            )
        from tac.residual_basis.saliency_masked_residual import (
            SaliencyMaskingConfig,
            compute_score_aware_saliency,
        )

        sal_config = saliency_masking_config or SaliencyMaskingConfig.council_canonical()
        # Compute saliency on a small subset of frames to avoid OOM.
        # The saliency map is per-pixel at camera resolution; we use it as
        # a per-pixel weight on the GT residual.
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
        # Average saliency across the sampled frames as a global per-pixel mask.
        saliency_global = saliency.mean(dim=0)  # (H, W)
        sal_h, sal_w = saliency_global.shape
        # Compute per-pixel keep mask via percentile or threshold.
        if sal_config.percentile is not None:
            threshold_value = float(
                torch.quantile(saliency_global.flatten(), q=float(sal_config.percentile)).item()
            )
        else:
            threshold_value = float(sal_config.threshold)
        keep_mask_hw = (saliency_global >= threshold_value).cpu().numpy()
        kept_fraction = float(keep_mask_hw.mean())
        if kept_fraction < sal_config.minimum_kept_fraction:
            raise C3EncoderL2Error(
                f"saliency mask kept_fraction={kept_fraction:.4f} < "
                f"minimum_kept_fraction={sal_config.minimum_kept_fraction:.4f} "
                "(saliency mask too aggressive)"
            )
        saliency_diagnostics = {
            "saliency_threshold_value": threshold_value,
            "saliency_kept_fraction": kept_fraction,
            "saliency_min": float(saliency_global.min().item()),
            "saliency_max": float(saliency_global.max().item()),
            "saliency_mean": float(saliency_global.mean().item()),
            "saliency_n_sample_frames": float(len(sal_indices)),
        }
    else:
        keep_mask_hw = None

    # Step 1: compute per-frame quarter-res target residual r_q[t].
    r_q = np.zeros((n_frames, QUARTER_H, QUARTER_W, RGB_CHANNELS), dtype=np.float64)
    for t in range(n_frames):
        residual = gt_frames[t].astype(np.float64) - decoded_frames[t].astype(np.float64)
        if keep_mask_hw is not None:
            # Apply per-pixel saliency mask BEFORE quarter-res downsample.
            # Broadcast (H, W) mask across the 3 RGB channels.
            residual = residual * keep_mask_hw[..., np.newaxis].astype(np.float64)
        r_q[t] = _downsample_to_quarter(residual.astype(np.float32)).astype(np.float64)

    # Step 2: compute per-frame deltas (first-difference). For non-kept frames,
    # delta is zero (so cumsum is preserved across the kept subset).
    kept_set = set(kept_indices)
    deltas = np.zeros_like(r_q)
    prev_target = np.zeros_like(r_q[0])
    for t in range(n_frames):
        if t in kept_set:
            deltas[t] = r_q[t] - prev_target
            prev_target = r_q[t]
        # else: delta stays at zero; prev_target unchanged.

    # Step 3: per-frame scale search.
    def _quantize_delta_with_scale(
        delta_f: np.ndarray, scale: float
    ) -> tuple[np.ndarray, float]:
        """int8 quantize ``delta_f`` with given scale. Returns (delta_q, reconstruction_mse)."""
        if scale <= 1e-9:
            return np.zeros_like(delta_f, dtype=np.int8), float(np.mean(delta_f * delta_f))
        scaled = delta_f / scale
        q = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        recon = q.astype(np.float64) * scale
        return q, float(np.mean((delta_f - recon) ** 2))

    def _auto_scale_for(delta_f: np.ndarray) -> float:
        max_abs = float(np.abs(delta_f).max())
        if max_abs <= 1e-9:
            return 0.0
        return max_abs / 120.0

    deltas_q = np.zeros_like(r_q, dtype=np.int8)
    scales = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        if t not in kept_set:
            continue
        s_auto = _auto_scale_for(deltas[t])
        if sparse_aware:
            # Per HNeRV parity discipline lesson 6 (score-domain Lagrangian):
            # in sparse-aware mode the rate term penalizes nonzero coefficients
            # directly. Bias the per-frame initial scale toward the LARGEST
            # multiplier in the grid (yielding the sparsest int8 quantization);
            # coordinate descent can refine downward if the proxy loss
            # demands more fidelity than the sparse-friendly start provides.
            best_s = s_auto * max(scale_grid)
        else:
            best_s = s_auto
            best_mse = float("inf")
            for mult in scale_grid:
                s = s_auto * mult
                _, mse = _quantize_delta_with_scale(deltas[t], s)
                if mse < best_mse:
                    best_mse = mse
                    best_s = s
        q, _ = _quantize_delta_with_scale(deltas[t], best_s)
        deltas_q[t] = q
        scales[t] = best_s

    # Step 4: compute the proxy Lagrangian on the reconstructed residual.
    def _apply_sparse_threshold(deltas_arr: np.ndarray, thr: int) -> np.ndarray:
        """Zero out int8 coefficients with magnitude below thr (sparsity sweep)."""
        if thr <= 0:
            return deltas_arr
        out = deltas_arr.copy()
        out[np.abs(out) < thr] = 0
        return out

    def _evaluate(scales_arr: np.ndarray, deltas_arr: np.ndarray) -> tuple[float, dict[str, float], int]:
        """Reconstruct via cumsum + upsample; eval proxy loss.

        When ``sparse_aware=True`` the archive_bytes term reflects the
        SPARSE-REPACKED byte cost of the dense (scales, deltas) pair, so the
        proxy Lagrangian directly optimizes for the contest-relevant byte
        size after sparse encoding.
        """
        # Integrate cumulatively.
        integrated = np.zeros_like(r_q)
        running = np.zeros_like(r_q[0])
        for t in range(n_frames):
            running = running + deltas_arr[t].astype(np.float64) * float(scales_arr[t])
            integrated[t] = running
        # Bilinear upsample to camera resolution.
        t_chw = torch.from_numpy(integrated).permute(0, 3, 1, 2).float()
        up = F.interpolate(t_chw, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
        residual_full = up.permute(0, 2, 3, 1).numpy()
        decoded_with_residual = np.clip(decoded_frames.astype(np.float32) + residual_full, 0, 255)
        blob_dense = _pack_c3_residual_bytes(
            n_frames=n_frames, scales_per_frame=scales_arr, deltas_q=deltas_arr
        )
        if sparse_aware:
            # Per HNeRV parity discipline lesson 6: the rate term must reflect
            # the BYTES THAT THE INFLATE RUNTIME ACTUALLY CONSUMES. For the
            # 0x22 c3_sparse runtime that's the sparse-repacked size.
            from tac.residual_basis.pr106_materializer_helpers import (
                repack_dense_as_sparse,
            )
            blob = repack_dense_as_sparse(
                family="c3", dense_residual_bytes=blob_dense, n_frames=n_frames,
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
        diag["c3_residual_blob_bytes"] = float(len(blob))
        diag["c3_residual_blob_dense_bytes"] = float(len(blob_dense))
        diag["c3_sparse_aware"] = float(1.0 if sparse_aware else 0.0)
        diag["c3_use_hinton_distilled_scorer"] = float(
            1.0 if use_hinton_distilled_scorer else 0.0
        )
        diag["c3_use_saliency_masking"] = float(1.0 if use_saliency_masking else 0.0)
        diag["c3_pose_only_mode"] = float(1.0 if pose_only_mode else 0.0)
        diag["c3_pose_marginal_multiplier"] = float(pose_marginal_multiplier)
        return float(loss.detach().item()), diag, len(blob)

    # Initial threshold sweep (sparse_aware only; thr=0 in dense mode).
    best_loss = float("inf")
    best_scales = scales.copy()
    best_deltas = deltas_q.copy()
    best_diag: dict[str, float] = {}
    init_loss = float("inf")
    init_diag: dict[str, float] = {}
    for thr in sparse_threshold_grid:
        cand_deltas_init = _apply_sparse_threshold(deltas_q, thr)
        cand_loss_init, cand_diag_init, _ = _evaluate(scales, cand_deltas_init)
        cand_diag_init["sparse_threshold"] = float(thr)
        if not init_diag:
            init_loss = cand_loss_init
            init_diag = cand_diag_init
        if cand_loss_init < best_loss:
            best_loss = cand_loss_init
            best_scales = scales.copy()
            best_deltas = cand_deltas_init
            best_diag = cand_diag_init

    # Coordinate-descent across global scale multipliers + threshold grid.
    converged = False
    iters_done = 0
    for it in range(n_iterations):
        iters_done = it + 1
        any_improved = False
        for mult in scale_grid:
            if abs(mult - 1.0) < 1e-12:
                continue
            cand_scales = best_scales.copy() * float(mult)
            # Re-quantize deltas with the new scales.
            base_cand_deltas = np.zeros_like(deltas_q)
            for t in range(n_frames):
                if t not in kept_set:
                    continue
                cand_q, _ = _quantize_delta_with_scale(deltas[t], float(cand_scales[t]))
                base_cand_deltas[t] = cand_q
            for thr in sparse_threshold_grid:
                cand_deltas = _apply_sparse_threshold(base_cand_deltas, thr)
                cand_loss, cand_diag, _ = _evaluate(cand_scales, cand_deltas)
                cand_diag["sparse_threshold"] = float(thr)
                if cand_loss + 1e-12 < best_loss:
                    best_loss = cand_loss
                    best_scales = cand_scales
                    best_deltas = cand_deltas
                    best_diag = cand_diag
                    any_improved = True
        if not any_improved:
            converged = True
            break

    final_blob_dense = _pack_c3_residual_bytes(
        n_frames=n_frames, scales_per_frame=best_scales, deltas_q=best_deltas
    )
    if sparse_aware:
        from tac.residual_basis.pr106_materializer_helpers import (
            repack_dense_as_sparse,
        )
        final_blob = repack_dense_as_sparse(
            family="c3", dense_residual_bytes=final_blob_dense, n_frames=n_frames,
        )
        if len(final_blob) > byte_budget:
            raise C3EncoderL2Error(
                f"sparse-aware c3 final residual {len(final_blob)} bytes exceeds "
                f"byte_budget={byte_budget}; coordinate descent did not converge under cap"
            )
    else:
        final_blob = final_blob_dense
    # Merge saliency diagnostics into the result diagnostics dict.
    if saliency_diagnostics:
        best_diag = dict(best_diag)
        for k, v in saliency_diagnostics.items():
            best_diag[f"c3_{k}"] = v
    result = C3EncoderL2Result(
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
    "C3EncoderL2Error",
    "C3EncoderL2Result",
    "DEFAULT_BYTE_BUDGET",
    "PER_FRAME_BYTES",
    "dense_c3_residual_blob_bytes",
    "encode_c3_residual_l2",
]
