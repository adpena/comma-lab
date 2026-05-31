# SPDX-License-Identifier: MIT
"""Z8 joint P18/P19 dead-zone RATE ATTACK on the Mallat wavelet detail bands.

Z8's binding axis is RATE (~97% of the 104.94 [macOS-CPU advisory] contest
score = ``25 * 4.05``); distortion is near-lossless (seg+pose ~ 3.6). The
99.5%-of-bytes ``wavelet_blob`` stores the **raw float32 Mallat detail-band
DWT coefficients of the real video frames** (per
``canonical_quadruple_binding._build_pair_pyramid_from_real_frames``). Zeroing
the small / score-irrelevant detail coefficients is classic wavelet
compression; the joint P18/P19 surface decides WHICH coefficients are safe to
zero (seg-flat AND pose-null = DEADZONE) versus which to protect (seg-boundary
OR pose-sensitive = PROTECT).

The joint per-atom weight (``tac.optimization.joint_p18_p19_waterfill``):

    w_i = 100 * |dL_seg/dx_i| + (5 / sqrt(10 * d_pose)) * ||J_pose,i||_{Sigma^-1}

Here the "atoms" are the per-pair Mallat detail coefficients. Because the
orthonormal Daubechies synthesis (``recompose_from_next_level``) is LINEAR,
its adjoint equals the analysis (``decompose_to_next_level``); a per-pixel
scorer saliency therefore pushes EXACTLY into the coefficient domain via the
forward DWT. The two real, tractable, $0-CPU pixel saliencies are:

- **P18 SegNet boundary saliency** — the contest ``d_seg`` is the SegNet
  argmax-flip rate; a recon perturbation flips a pixel's class most easily
  where the GT SegNet top-2 logit margin is small. One real SegNet forward on
  the GT frame yields ``seg_sal[p] = exp(-margin[p] / tau)`` (boundary-weighted;
  high = flip-prone = PROTECT).
- **P19 PoseNet pixel sensitivity** — the contest ``d_pose`` is the MSE on the
  first ``out//2`` pose dims; one real backward of the differentiable PoseNet
  yields ``pose_sal[p] = ||d(pose head)/d(pixel_p)||`` (the per-pixel pose
  Jacobian norm; near-zero = pose-null = safe to coarsen).

Both pixel saliencies are pushed to the coefficient domain via the analysis
DWT chain, combined into the joint ``w_i``, and the low-``w_i`` detail
coefficients (seg-flat AND pose-null below thresholds) are ACTUALLY zeroed.
The wavelet blob is re-packed and the byte-closed archive rate + real
DistortionNet d_seg / d_pose are RE-MEASURED — the full-video replay (never the
stale tangent plane) is the authority per the joint-surface fresh-surface
discipline.

Catalog #125 hook #3 (bit-allocator): the joint surface IS the dead-zone /
coarsening bit-allocator prior for the rate-binding wavelet codec.

ALL numbers are ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]``,
non-promotable, with canonical Provenance, per Catalog #192/#341/#127/#323.
Local macOS-CPU is NOT 1:1 contest hardware -> explicitly NOT a [contest-CPU]
claim. $0 local, NO cloud, NO paid GPU.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.joint_p18_p19_waterfill import (
    JointP18P19WaterfillConfig,
    build_joint_p18_p19_waterfill_surface,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HPC1_HEADER_FMT,
    Z8HPC1_HEADER_SIZE,
    parse_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    pack_pair_pyramids_to_wavelet_blob,
    parse_pair_blobs_from_wavelet_blob,
)
from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
    WaveletDetail2D,
)

# Default pose null threshold: a coefficient is "pose-null" when its pushed
# PoseNet pixel-gradient energy is below this fraction of the per-pair max.
DEFAULT_POSE_NULL_FRACTION: float = 0.05
# Default SegNet boundary temperature for the exp(-margin/tau) weighting.
DEFAULT_SEG_BOUNDARY_TAU: float = 1.0


@dataclass(frozen=True)
class DeadzoneRateAttackResult:
    """Result of one dead-zone application over a parsed wavelet blob."""

    modified_wavelet_blob: bytes
    coefficients_total: int
    coefficients_zeroed: int
    deadzone_fraction: float
    distortion_protect_fraction: float
    joint_weight_threshold: float
    coefficients_already_zero: int

    @property
    def coefficients_newly_zeroed(self) -> int:
        return self.coefficients_zeroed - self.coefficients_already_zero


# ---------------------------------------------------------------------------
# Per-pixel real scorer saliencies (P18 + P19)
# ---------------------------------------------------------------------------


def segnet_boundary_pixel_saliency(
    posenet: Any,
    segnet: Any,
    gt_pair_btchw_unit: Any,
    *,
    tau: float = DEFAULT_SEG_BOUNDARY_TAU,
) -> np.ndarray:
    """Return ``(H, W)`` SegNet boundary saliency for the GT frame.

    The contest ``d_seg`` is the argmax-flip rate; a pixel flips most easily
    where the GT SegNet top-2 logit margin is small, so
    ``seg_sal[p] = exp(-margin[p] / tau)`` is the real boundary-weighted
    flip-proneness (high = PROTECT). One real SegNet forward; no backward.

    ``gt_pair_btchw_unit`` is the GT pair as a torch tensor of shape
    ``(1, 2, H, W, C)`` in [0, 1] (DistortionNet input convention). The SegNet
    boundary map is returned on the SegNet model grid (384x512), the grid the
    argmax-flip distortion is actually computed on.
    """

    import torch

    del posenet  # SegNet term only needs SegNet
    x = gt_pair_btchw_unit
    # DistortionNet.preprocess: b t h w c -> b t c h w
    xp = x.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)  # uses last frame -> (1, 3, 384, 512)
        seg_logits = segnet(seg_in)  # (1, 5, 384, 512)
        top2 = torch.topk(seg_logits, k=2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (1, 384, 512)
        sal = torch.exp(-margin / float(tau))  # high = small margin = flip-prone
    return sal[0].detach().cpu().numpy().astype(np.float64)


def posenet_pixel_jacobian_norm(
    posenet: Any,
    segnet: Any,
    gt_pair_btchw_unit: Any,
) -> np.ndarray:
    """Return ``(2, H, W)`` per-pixel PoseNet Jacobian energy for the GT pair.

    The contest ``d_pose`` is the MSE on the first ``out//2`` pose dims; the
    per-pixel pose sensitivity is ``||d(pose head[:out//2]) / d(pixel)||`` via a
    single real backward of the differentiable PoseNet. Near-zero = pose-null =
    safe to coarsen. Returns the per-frame (2 frames) per-pixel L2 gradient norm
    summed over the 3 RGB channels, on the native ``(H, W)`` grid (the grid the
    wavelet coefficients live on).
    """

    import torch

    del segnet  # PoseNet term only needs PoseNet
    x = gt_pair_btchw_unit.clone().detach().requires_grad_(True)  # (1, 2, H, W, C)
    xp = x.permute(0, 1, 4, 2, 3).contiguous().float()
    pose_in = posenet.preprocess_input(xp)
    pose_out = posenet(pose_in)
    pose_head = pose_out["pose"]  # (1, out)
    half = pose_head.shape[-1] // 2
    pose_used = pose_head[..., :half]  # only the distortion dims
    scalar = pose_used.pow(2).sum()
    grad = torch.autograd.grad(scalar, x, retain_graph=False)[0]  # (1, 2, H, W, C)
    # Per-pixel L2 over channels -> (2, H, W)
    norm = grad[0].pow(2).sum(dim=-1).sqrt()
    return norm.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Push pixel saliency into the coefficient domain via the analysis DWT (adjoint)
# ---------------------------------------------------------------------------


def _resize_pixel_map_to_grid(pixel_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest/area resize a ``(H, W)`` pixel saliency to ``(target_h, target_w)``.

    Used to map the SegNet 384x512 boundary saliency back onto the native
    coefficient grid. Uses torch bilinear interpolate to match the recon resize
    convention in the scorer tool.
    """

    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(pixel_map.astype(np.float32))[None, None, ...]
    up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy().astype(np.float64)


def push_pixel_saliency_to_detail_coeffs(
    binding: Any,
    pixel_saliency_hw: np.ndarray,
    *,
    num_levels: int,
) -> list[dict[str, np.ndarray]]:
    """Push a ``(H, W)`` per-pixel saliency into per-level detail-coeff saliency.

    For an orthonormal Daubechies basis the synthesis adjoint equals the
    analysis. The per-pixel saliency is run through the SAME analysis chain
    (``decompose_to_next_level``) the coefficients were produced by, so the
    resulting detail-subband magnitudes are the exact adjoint image of the
    pixel saliency onto each detail coefficient.

    Returns a list (deepest-first, matching the stored ``frame_*_details``
    order) of dicts ``{"lh": ..., "hl": ..., "hh": ...}`` with the absolute
    per-coefficient saliency for each subband (shape ``(H/2^l, W/2^l, C)``).
    """

    # Saliency is single-channel (H, W); broadcast to C channels so the DWT
    # operates per channel exactly like the coefficient transform.
    hw = np.abs(np.asarray(pixel_saliency_hw, dtype=np.float64))
    # NHWC with C=3 (the coefficient grid is RGB).
    current = np.repeat(hw[None, ..., None], 3, axis=-1)  # (1, H, W, 3)
    out: list[dict[str, np.ndarray]] = []
    for level_idx in range(num_levels - 1):
        ll, detail = binding.m5_per_level[level_idx].decompose_to_next_level(current)
        out.append(
            {
                "lh": np.abs(np.asarray(detail.lh, dtype=np.float64))[0],
                "hl": np.abs(np.asarray(detail.hl, dtype=np.float64))[0],
                "hh": np.abs(np.asarray(detail.hh, dtype=np.float64))[0],
            }
        )
        current = np.asarray(ll, dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Joint surface over per-pair detail coefficients
# ---------------------------------------------------------------------------


def build_joint_surface_for_pair(
    *,
    seg_detail_saliency: list[dict[str, np.ndarray]],
    pose_detail_saliency: list[dict[str, np.ndarray]],
    d_pose: float,
    pose_null_fraction: float = DEFAULT_POSE_NULL_FRACTION,
) -> dict[str, Any]:
    """Build the joint P18/P19 waterfill surface over a pair's detail coeffs.

    Flattens all per-level (lh, hl, hh) saliencies into a single atom grid and
    routes them through the canonical
    ``build_joint_p18_p19_waterfill_surface``. The SegNet boundary saliency is
    the P18 ``segnet_argmax_gradient`` term; the PoseNet pixel-Jacobian saliency
    is the P19 single-dim pose Jacobian (Sigma^-1 = 1, the pose energy is
    already the Mahalanobis-style L2 norm). The pose-null threshold is set to a
    fraction of the per-pair max pose saliency.
    """

    seg_flat = _flatten_detail_saliency(seg_detail_saliency)
    pose_flat = _flatten_detail_saliency(pose_detail_saliency)
    if seg_flat.shape != pose_flat.shape:
        raise ValueError("seg/pose detail saliency shape mismatch")
    pose_max = float(pose_flat.max()) if pose_flat.size else 0.0
    pose_null_threshold = float(pose_null_fraction) * pose_max
    cfg = JointP18P19WaterfillConfig(
        d_pose=float(max(d_pose, 1e-12)),
        pose_inverse_variance=(1.0,),
        pose_null_threshold=pose_null_threshold,
    )
    surface = build_joint_p18_p19_waterfill_surface(
        segnet_argmax_gradient=seg_flat,
        pose_jacobian=pose_flat[..., None],  # single pose dim
        config=cfg,
    )
    return surface


def _flatten_detail_saliency(detail_saliency: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Flatten per-level (lh, hl, hh) saliency arrays into a single 1-D vector."""

    parts: list[np.ndarray] = []
    for level in detail_saliency:
        for key in ("lh", "hl", "hh"):
            parts.append(np.asarray(level[key], dtype=np.float64).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Apply dead-zone: ACTUALLY zero the dead-zone detail coefficients
# ---------------------------------------------------------------------------


def _detail_coeff_offsets(details: list[WaveletDetail2D]) -> list[tuple[str, int, int, tuple[int, ...]]]:
    """Return ``[(subband_key, level_idx, flat_offset, shape), ...]`` for a frame's
    details, matching the flattening order in ``_flatten_detail_saliency``."""

    offsets: list[tuple[str, int, int, tuple[int, ...]]] = []
    pos = 0
    for level_idx, detail in enumerate(details):
        for key in ("lh", "hl", "hh"):
            arr = np.asarray(getattr(detail, key), dtype=np.float64)
            offsets.append((key, level_idx, pos, arr.shape))
            pos += int(arr.size)
    return offsets


def apply_deadzone_to_pair_details(
    details: list[WaveletDetail2D],
    deadzone_mask_flat: np.ndarray,
) -> tuple[list[WaveletDetail2D], int, int]:
    """Zero the dead-zone detail coefficients of one frame's pyramid.

    ``deadzone_mask_flat`` is a 1-D boolean over the flattened (lh, hl, hh)
    coefficients in the SAME order as ``_flatten_detail_saliency``. Returns
    ``(new_details, n_total, n_already_zero_in_deadzone)``. The coefficients are
    ACTUALLY set to 0.0 (this is the real rate-attack mutation, not a marker).
    """

    offsets = _detail_coeff_offsets(details)
    n_total = sum(int(np.prod(shape)) for _key, _lvl, _off, shape in offsets)
    if deadzone_mask_flat.shape[0] != n_total:
        raise ValueError(
            f"deadzone mask len {deadzone_mask_flat.shape[0]} != coeff total {n_total}"
        )
    new_by_level: dict[int, dict[str, np.ndarray]] = {}
    already_zero = 0
    for key, level_idx, off, shape in offsets:
        size = int(np.prod(shape))
        sub = np.asarray(getattr(details[level_idx], key), dtype=np.float32).copy()
        flat_sub = sub.reshape(-1)
        m = deadzone_mask_flat[off : off + size]
        already_zero += int(np.count_nonzero((flat_sub == 0.0) & m))
        flat_sub[m] = 0.0
        new_by_level.setdefault(level_idx, {})[key] = flat_sub.reshape(shape).astype(np.float32)
    new_details: list[WaveletDetail2D] = []
    for level_idx in range(len(details)):
        lv = new_by_level[level_idx]
        new_details.append(WaveletDetail2D(lh=lv["lh"], hl=lv["hl"], hh=lv["hh"]))
    return new_details, n_total, already_zero


# ---------------------------------------------------------------------------
# Magnitude-deadzone fallback (no scorer) — used for the SegNet-only baseline
# and for cheap smokes; the joint path overrides w_i with real saliencies.
# ---------------------------------------------------------------------------


def magnitude_deadzone_mask_for_pair(
    details: list[WaveletDetail2D],
    *,
    keep_fraction: float,
) -> np.ndarray:
    """Return a 1-D boolean dead-zone mask keeping the top ``keep_fraction`` by
    magnitude (zero the rest). Pure magnitude — the rate-only baseline."""

    flat = _flatten_coeffs(details)
    return _threshold_keep_top_fraction(np.abs(flat), keep_fraction)


def _flatten_coeffs(details: list[WaveletDetail2D]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for detail in details:
        for key in ("lh", "hl", "hh"):
            parts.append(np.asarray(getattr(detail, key), dtype=np.float64).reshape(-1))
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _threshold_keep_top_fraction(weights: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Return a dead-zone (True = zero) mask that keeps the top ``keep_fraction``
    of atoms by ``weights`` and zeros the rest."""

    n = weights.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    keep_fraction = float(min(max(keep_fraction, 0.0), 1.0))
    n_keep = round(keep_fraction * n)
    if n_keep >= n:
        return np.zeros((n,), dtype=bool)
    if n_keep <= 0:
        return np.ones((n,), dtype=bool)
    # Keep the n_keep largest-weight atoms; zero (dead-zone) everything else.
    thresh = np.partition(weights, n - n_keep)[n - n_keep]
    deadzone = weights < thresh
    # Tie-break so we keep EXACTLY n_keep: among == thresh atoms, keep enough.
    keep = ~deadzone
    over = int(keep.sum()) - n_keep
    if over > 0:
        tie_idx = np.flatnonzero((weights == thresh) & keep)
        deadzone[tie_idx[:over]] = True
    return deadzone


def joint_keep_priority_for_pair(
    coeff_magnitude_flat: np.ndarray,
    seg_term_flat: np.ndarray,
    pose_term_flat: np.ndarray,
    pose_null_mask_flat: np.ndarray,
    *,
    seg_protect_gain: float = 1.0,
    pose_protect_gain: float = 1.0,
) -> np.ndarray:
    """Scorer-modulated rate-distortion KEEP priority over detail coefficients.

    Classic wavelet RD-optimal compression keeps the largest-magnitude
    coefficients (they carry the most L2 reconstruction energy). The joint
    P18/P19 surface does NOT replace that ranking — it REFINES it: among
    coefficients of comparable magnitude, prefer to zero the seg-flat AND
    pose-null ones (won't move the score) and protect the seg-boundary OR
    pose-sensitive ones (will move the score). The keep priority is therefore
    the product of the reconstruction energy and a per-atom scorer protection
    factor:

        keep_priority_i = |coeff_i| * (1 + seg_protect_gain * seg_hat_i)
                                    * (1 + pose_protect_gain * pose_hat_i)

    where ``seg_hat_i`` / ``pose_hat_i`` are the per-atom seg/pose saliencies
    normalized into [0, 1] (so the magnitude-only baseline is the
    ``seg_protect_gain = pose_protect_gain = 0`` special case). Pose-null atoms
    (``pose_null_mask_flat == True``) contribute zero pose-protection, so they
    fall to the bottom of the keep priority first — exactly the P19
    "release pose-null to the dead-zone, protect pose-sensitive" behavior.
    """

    mag = np.abs(np.asarray(coeff_magnitude_flat, dtype=np.float64))
    seg = np.asarray(seg_term_flat, dtype=np.float64)
    pose = np.asarray(pose_term_flat, dtype=np.float64)
    pose_null = np.asarray(pose_null_mask_flat, dtype=bool)
    seg_hat = seg / (seg.max() + 1e-12) if seg.size else seg
    pose_hat = pose / (pose.max() + 1e-12) if pose.size else pose
    # Pose-null atoms get NO pose-protection (released to the dead-zone first).
    pose_hat = np.where(pose_null, 0.0, pose_hat)
    return (
        mag
        * (1.0 + float(seg_protect_gain) * seg_hat)
        * (1.0 + float(pose_protect_gain) * pose_hat)
    )


def joint_deadzone_mask_for_pair(
    keep_priority_flat: np.ndarray,
    *,
    keep_fraction: float,
) -> np.ndarray:
    """Dead-zone mask keeping the top ``keep_fraction`` of atoms by the
    scorer-modulated RD keep priority (zero the rest).

    ``keep_priority_flat`` is the output of :func:`joint_keep_priority_for_pair`
    (reconstruction energy modulated by seg/pose protection). Returns a 1-D
    boolean dead-zone mask (True = zero this coefficient).
    """

    return _threshold_keep_top_fraction(
        np.asarray(keep_priority_flat, dtype=np.float64), keep_fraction
    )


# ---------------------------------------------------------------------------
# Splice a modified wavelet_blob back into a parsed Z8HPC1 archive
# ---------------------------------------------------------------------------


def splice_wavelet_blob_into_archive(archive_bytes: bytes, new_wavelet_blob: bytes) -> bytes:
    """Rebuild Z8HPC1 archive bytes with ``new_wavelet_blob`` replacing the
    wavelet section (header WAVELET_BLOB_LEN updated; all other sections kept
    byte-identical). This is the byte-closed re-pack after a dead-zone mutation.
    """

    sections = parse_z8hpc1_archive_bytes(archive_bytes)
    header_fields = list(struct.unpack(Z8HPC1_HEADER_FMT, archive_bytes[:Z8HPC1_HEADER_SIZE]))
    # Field order per Z8HPC1_HEADER_FMT:
    # 0 magic, 1 version, 2 num_levels, 3 g_bytes, 4 k0, 5 k1, 6 k2,
    # 7 num_pairs, 8 decoder_latent, 9 base_channels, 10 wavelet_basis_id,
    # 11 decoder_blob_len, 12 indices_blob_len, 13 wavelet_blob_len,
    # 14 wz_blob_len, 15 dreamer_state_blob_len, 16 meta_blob_len, 17 reserved
    header_fields[13] = len(new_wavelet_blob)  # WAVELET_BLOB_LEN
    new_header = struct.pack(Z8HPC1_HEADER_FMT, *header_fields)

    def _sec(name: str) -> bytes:
        start, length = sections[name]
        return archive_bytes[start : start + length]

    return (
        new_header
        + _sec("decoder_blob")
        + _sec("indices_blob")
        + new_wavelet_blob
        + _sec("wyner_ziv_blob")
        + _sec("dreamer_state_blob")
        + _sec("meta_blob")
    )


__all__ = [
    "DEFAULT_POSE_NULL_FRACTION",
    "DEFAULT_SEG_BOUNDARY_TAU",
    "DeadzoneRateAttackResult",
    "_resize_pixel_map_to_grid",
    "apply_deadzone_to_pair_details",
    "build_joint_surface_for_pair",
    "joint_deadzone_mask_for_pair",
    "joint_keep_priority_for_pair",
    "magnitude_deadzone_mask_for_pair",
    "pack_pair_pyramids_to_wavelet_blob",
    "parse_pair_blobs_from_wavelet_blob",
    "posenet_pixel_jacobian_norm",
    "push_pixel_saliency_to_detail_coeffs",
    "segnet_boundary_pixel_saliency",
    "splice_wavelet_blob_into_archive",
]
