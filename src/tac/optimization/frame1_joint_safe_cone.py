# SPDX-License-Identifier: MIT
"""Frame-1 JOINT SAFE CONE — the per-pixel/per-region perturbation budget for the
frame BOTH contest scorers read.

The two frames of a contest pair are NOT symmetric under the evaluator:

- **SegNet** reads ONLY ``x[:, -1, ...]`` (the *last* frame = frame1) per
  ``upstream/modules.py:108``. ``d_seg`` is the per-pixel argmax-flip RATE
  (``modules.py:112``). So the SegNet constraint is a **frame1-only** constraint.
- **PoseNet** reads BOTH frames via YUV6 — 12 channels = 2 frames * 6
  (``modules.py:74``, ``IN_CHANS = 6 * 2``). ``d_pose`` is the MSE on the first
  ``out // 2 == 6`` pose dims (``modules.py:84``). So the PoseNet constraint
  touches frame1 through its frame1 input channels.

frame0 already has a methodology: it is SegNet-blind *by construction* (SegNet
never sees it), so PR110 family exploits perturb frame0 freely up to the PoseNet
budget. **frame1 had no methodology** — every byte that touches frame1
(LF/HF carrier coefficients, source-state quantization, residual sidecars, the
masks-derived render) is constrained by BOTH scorers simultaneously and the
joint budget was never computed.

This module computes the **JOINT SAFE CONE**:

    safe(delta) = { SegNet source-class argmax margin survives delta
                    AND  J_pose . delta is small }

i.e. for every frame1 pixel ``p`` a perturbation budget (cone radius) such that
applying ``|delta_p| <= radius_p`` to frame1 leaves ``d_seg`` unchanged (the
argmax of pixel ``p`` does not flip) AND keeps the linearized pose response
within tolerance. Where the cone radius is zero the pixel is a *fragile boundary*
pixel (small SegNet margin OR high pose Jacobian) — the binding-constraint set
no frame1-touching byte may move.

REUSE (no-duplicative-code; orphan inventory 2026-06-09): the two half-cones
share their measured math lineage with the existing z8 half-cones (dedup audit
2026-06-10: the math is the SAME scorer forward; the OUTPUT differs because the
cone needs frame1-specialized quantities the z8 surface does not emit, so this
module derives them directly rather than wrapping the z8 functions):

- **Seg-safe half** — sister of
  :func:`tac.substrates.z8_hierarchical_predictive_coding.joint_p18_p19_deadzone_rate_attack.segnet_boundary_pixel_saliency`
  (same real SegNet forward + top-2 logit margin, verified non-constant
  2026-06-09). z8 returns ``exp(-margin/tau)`` *saliency* on the wavelet-coeff
  domain; the cone needs the **raw margin AND the argmax class id** per frame1
  pixel (the distance-to-flip budget source + the behavioral-check reference), so
  :func:`measure_segnet_frame1_margin` returns ``(margin, argmax_cls)`` directly.
- **Pose-null half** — sister of z8's
  :func:`...posenet_pixel_jacobian_norm` (same real differentiable PoseNet
  pixel-Jacobian backward). z8 returns the ``(2, H, W)`` norm for BOTH frames on
  the coeff grid; the cone needs the **frame1-channel-only** ``(H, W)`` norm
  (``grad[0, 1]``) plus a fail-closed zero-energy guard, so
  :func:`measure_posenet_frame1_jacobian` derives that variant.
  **CRITICAL**: this is only gradient-reachable when ``rgb_to_yuv6`` is
  differentiable; the upstream helper is ``@torch.no_grad()`` / in-place and
  SEVERS the pose gradient (the CLAUDE.md "eval_roundtrip / differentiable YUV6"
  non-negotiable). This module patches it via
  :func:`tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` and
  **fails closed** if the resulting Jacobian is identically zero (the
  not-reachable signature) so a non-reachable gradient can never silently
  produce an all-permissive cone.
- **Coupled weight** — the joint sensitivity coupling follows the canonical
  P18/P19 formula ``w_i = 100*|dL_seg| + (5/sqrt(10*d_pose))*||J_pose,i||``
  (``tac.optimization.joint_p18_p19_waterfill``). The cone radius is the
  INVERSE of this joint sensitivity (high sensitivity -> small budget).

NO-FAKE discipline (MEMORY.md Slot RR): a prior ``apply_pose_axis_null_projection``
returned canonical markers while applying ZERO perturbation. This module never
returns a marker in place of work: every cone field is derived from a measured
scorer response, and the companion CLI/test validates the cone *behaviorally*
(perturb inside -> d_seg stable; perturb outside -> d_seg moves). A cone that
does not discriminate is a FAKE cone.

ALL numbers produced from this module are ``[macOS-CPU advisory]`` /
mechanism-only, non-promotable: local macOS-CPU is NOT 1:1 contest hardware.
$0 local, NO cloud, NO paid GPU. The cone is a *budget surface* the downstream
RD waterfiller / repair / PR110++ frame1-modes / SNeRV LF quantizer consume; it
proposes radii, the full-video exact DistortionNet replay ratifies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRAME1_JOINT_SAFE_CONE_SCHEMA = "frame1_joint_safe_cone.v1"

# The contest pose term is sqrt(10 * d_pose); the AIL marginal gain that couples
# the pose Jacobian into the joint weight is 5 / sqrt(10 * d_pose) (the
# derivative of the pose term). Mirrors joint_p18_p19_waterfill.pose_ail_gain.
_POSE_AIL_NUMERATOR: float = 5.0
_POSE_TEN: float = 10.0
# SegNet weight in the contest score: d_seg contributes 100 * d_seg.
_SEG_SCORE_WEIGHT: float = 100.0
# Default boundary temperature for the exp(-margin/tau) flip-proneness map.
DEFAULT_SEG_BOUNDARY_TAU: float = 1.0
# A pixel whose pose-Jacobian energy is below this fraction of the per-pair max
# is "pose-null" (safe to coarsen on the pose axis).
DEFAULT_POSE_NULL_FRACTION: float = 0.05
# An all-zero PoseNet Jacobian is the "gradient not reachable" signature
# (upstream rgb_to_yuv6 severed the graph). Fail closed below this.
_POSE_JACOBIAN_REACHABLE_FLOOR: float = 1e-30


class Frame1JointSafeConeError(ValueError):
    """Raised when frame1 joint-safe-cone inputs/contracts are malformed or when
    a measured scorer gradient is not reachable (fail-closed, never permissive)."""


@dataclass(frozen=True)
class Frame1ConeConfig:
    """Configuration for the frame1 joint safe cone.

    ``seg_tau`` controls the boundary flip-proneness weighting. ``d_pose`` is the
    operating-point pose distortion that sets the pose AIL marginal gain
    ``5/sqrt(10*d_pose)`` — at the PR106 frontier (d_pose ~ 3.4e-5) the pose
    marginal is large, so the pose half of the cone binds harder. ``seg_margin_tol``
    is the fraction of the per-pixel SegNet margin a perturbation is allowed to
    consume before the argmax is considered at flip risk (the seg budget = a
    fraction of the distance-to-flip). ``pose_response_tol`` is the linearized
    pose response (in pose-output units) a single pixel's perturbation is allowed
    to inject before it is considered pose-unsafe.
    """

    seg_tau: float = DEFAULT_SEG_BOUNDARY_TAU
    d_pose: float = 3.4e-5
    seg_margin_tol: float = 0.5
    pose_response_tol: float = 1e-3
    pose_null_fraction: float = DEFAULT_POSE_NULL_FRACTION
    d_pose_floor: float = 1e-12
    # Numerical floors so a near-zero sensitivity does not produce an infinite
    # radius (the cone is reported clamped to this maximum pixel perturbation,
    # in the scorer input units [0, 255]).
    max_radius: float = 255.0
    min_radius_floor: float = 0.0
    # A pixel whose safe radius is below this is a FRAGILE BOUNDARY pixel: it has
    # less than half a code-level (uint8 quantization step) of joint budget, so
    # no frame1-touching byte may move it. The "empty cone" / binding-constraint
    # set per the directive. 0.5 = half a uint8 step (the contest evaluator
    # round/uint8-quantizes recon frames before scoring).
    fragile_radius_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.seg_tau <= 0.0:
            raise Frame1JointSafeConeError("seg_tau must be > 0")
        if self.d_pose < 0.0:
            raise Frame1JointSafeConeError("d_pose must be >= 0")
        if self.d_pose_floor <= 0.0:
            raise Frame1JointSafeConeError("d_pose_floor must be > 0")
        if not 0.0 < self.seg_margin_tol <= 1.0:
            raise Frame1JointSafeConeError("seg_margin_tol must be in (0, 1]")
        if self.pose_response_tol <= 0.0:
            raise Frame1JointSafeConeError("pose_response_tol must be > 0")
        if not 0.0 <= self.pose_null_fraction <= 1.0:
            raise Frame1JointSafeConeError("pose_null_fraction must be in [0, 1]")
        if self.max_radius <= 0.0:
            raise Frame1JointSafeConeError("max_radius must be > 0")
        if self.min_radius_floor < 0.0:
            raise Frame1JointSafeConeError("min_radius_floor must be >= 0")
        if self.fragile_radius_threshold < 0.0:
            raise Frame1JointSafeConeError("fragile_radius_threshold must be >= 0")

    @property
    def effective_d_pose(self) -> float:
        return float(max(self.d_pose, self.d_pose_floor))

    @property
    def pose_ail_gain(self) -> float:
        """The pose marginal coupling gain ``5 / sqrt(10 * d_pose)``."""
        return _POSE_AIL_NUMERATOR / float(np.sqrt(_POSE_TEN * self.effective_d_pose))


@dataclass(frozen=True)
class Frame1JointSafeCone:
    """Per-pixel + per-region frame1 joint safe-cone budget surface.

    All per-pixel maps are on the SegNet model grid (384x512) — the grid the
    contest ``d_seg`` argmax-flip is actually computed on, and the grid the
    SegNet boundary margin / PoseNet frame1-channel Jacobian are measured on.
    """

    schema: str
    # (H, W) SegNet top1-top2 logit margin on frame1 (distance to argmax flip).
    seg_margin: np.ndarray
    # (H, W) per-pixel seg perturbation budget = seg_margin_tol * margin / boundary_slope.
    seg_margin_budget: np.ndarray
    # (H, W) PoseNet frame1-channel pixel-Jacobian L2 norm (pose response per unit pixel).
    pose_jacobian_norm: np.ndarray
    # (H, W) per-pixel pose perturbation budget = pose_response_tol / J_pose.
    pose_budget: np.ndarray
    # (H, W) joint coupled sensitivity w = 100*seg_sensitivity + ail_gain*J_pose.
    joint_sensitivity: np.ndarray
    # (H, W) joint cone radius = min(seg_margin_budget, pose_budget), clamped.
    joint_cone_radius: np.ndarray
    # (H, W) bool: True where the cone is empty (radius == min_radius_floor;
    # the strict zero-margin already-at-flip set).
    empty_cone_mask: np.ndarray
    # (H, W) bool: True where the cone is FRAGILE (radius < fragile_radius_threshold;
    # less than half a uint8 code-level of joint budget). The binding-constraint /
    # "no frame1 byte may move this" set the directive calls the empty cone.
    fragile_cone_mask: np.ndarray
    # (H, W) int SegNet argmax class id on frame1 (the source class per pixel).
    seg_argmax_class: np.ndarray
    # per-class region aggregates keyed by class id (mean/min/usable-fraction radius).
    per_region: dict[int, dict[str, float]]
    # scalar summary stats.
    summary: dict[str, Any]
    config: Frame1ConeConfig
    # provenance / false-authority markers (Catalog #341 / #323 Tier A).
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Real scorer measurement (reuses the verified z8 pixel saliencies)
# ---------------------------------------------------------------------------


def _ensure_pair_btchwc(pair: Any) -> Any:
    """Validate/return a frame1 pair as a torch tensor ``(1, 2, H, W, C)`` float
    in the scorer input range [0, 255] (DistortionNet input convention)."""

    import torch

    if not isinstance(pair, torch.Tensor):
        pair = torch.as_tensor(np.asarray(pair))
    pair = pair.float()
    if pair.ndim != 5 or pair.shape[0] != 1 or pair.shape[1] != 2 or pair.shape[-1] != 3:
        raise Frame1JointSafeConeError(
            "frame1 pair must be a (1, 2, H, W, 3) tensor in scorer units [0, 255]; "
            f"got shape {tuple(pair.shape)}"
        )
    return pair


def measure_segnet_frame1_margin(
    segnet: Any,
    pair_btchwc_unit255: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(seg_margin_HW, seg_argmax_class_HW)`` on the SegNet grid.

    SegNet reads ONLY the last frame (frame1) per ``modules.py:108``. The
    per-pixel top1-top2 logit margin is the *distance to argmax flip* — the
    seg-safe budget source. Returns the margin (>= 0) and the argmax source
    class id per pixel. One real SegNet forward; no backward.
    """

    import torch

    pair = _ensure_pair_btchwc(pair_btchwc_unit255)
    # DistortionNet.preprocess: b t h w c -> b t c h w
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)  # last frame -> (1, 3, 384, 512)
        seg_logits = segnet(seg_in)  # (1, 5, 384, 512)
        top2 = torch.topk(seg_logits, k=2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)  # (1, H, W)
        argmax_cls = seg_logits.argmax(dim=1)  # (1, H, W)
    return (
        margin[0].detach().cpu().numpy().astype(np.float64),
        argmax_cls[0].detach().cpu().numpy().astype(np.int64),
    )


def measure_segnet_frame1_boundary_slope(
    segnet: Any,
    pair_btchwc_unit255: Any,
) -> np.ndarray:
    """Return ``(H, W)`` per-pixel SegNet boundary slope ``||d(margin)/d(frame1 pixel)||``.

    The seg budget is ``distance_to_flip / sensitivity``: the margin is the
    distance, this Jacobian norm is the local sensitivity of the margin to a
    frame1 pixel perturbation. A real backward of the SegNet top1-top2 margin
    w.r.t. the frame1 pixels. Returned on the SegNet grid (384x512).
    """

    import torch

    pair = _ensure_pair_btchwc(pair_btchwc_unit255).clone().detach()
    pair.requires_grad_(True)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    seg_in = segnet.preprocess_input(xp)  # (1, 3, 384, 512), depends on frame1
    seg_logits = segnet(seg_in)  # (1, 5, 384, 512)
    top2 = torch.topk(seg_logits, k=2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)  # (1, H, W)
    scalar = margin.sum()
    grad = torch.autograd.grad(scalar, pair, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        raise Frame1JointSafeConeError(
            "SegNet boundary slope gradient is not reachable (None) — the SegNet "
            "preprocess severed the graph from frame1; cannot compute seg budget."
        )
    # frame1 is index 1 along the t-axis. Per-pixel L2 over channels -> (H, W).
    g1 = grad[0, 1]  # (H, W, C) on native grid
    slope = g1.pow(2).sum(dim=-1).sqrt()  # (H_native, W_native)
    slope_np = slope.detach().cpu().numpy().astype(np.float64)
    return slope_np


def measure_posenet_frame1_jacobian(
    posenet: Any,
    pair_btchwc_unit255: Any,
) -> np.ndarray:
    """Return ``(H, W)`` per-pixel PoseNet *frame1-channel* Jacobian L2 norm.

    PoseNet reads BOTH frames (12 = 2*6 channels). ``d_pose`` is the MSE on the
    first ``out//2`` pose dims. The frame1 budget cares about the frame1 input
    channels only, so this returns the L2 norm of
    ``d(pose[:out//2]) / d(frame1 pixel)`` — the contribution of a frame1 pixel
    perturbation to the pose output. A single real backward of the differentiable
    PoseNet.

    **GRADIENT-REACHABILITY (fail-closed)**: the upstream ``rgb_to_yuv6`` is
    ``@torch.no_grad()`` / in-place and would sever this gradient (returning an
    all-zero Jacobian = the not-reachable signature). The caller MUST patch it
    via :func:`tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally`
    BEFORE calling; this function asserts the resulting Jacobian is non-trivial
    and RAISES if it is identically zero, so a non-reachable gradient can never
    silently masquerade as an all-permissive (pose-null everywhere) cone.
    """

    import torch

    pair = _ensure_pair_btchwc(pair_btchwc_unit255).clone().detach()
    pair.requires_grad_(True)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    pose_in = posenet.preprocess_input(xp)
    pose_out = posenet(pose_in)
    pose_head = pose_out["pose"]  # (1, out)
    half = pose_head.shape[-1] // 2
    pose_used = pose_head[..., :half]  # only the distortion dims
    scalar = pose_used.pow(2).sum()
    grad = torch.autograd.grad(scalar, pair, retain_graph=False, allow_unused=True)[0]
    if grad is None:
        raise Frame1JointSafeConeError(
            "PoseNet frame1 Jacobian is not reachable (grad is None) — patch "
            "upstream rgb_to_yuv6 via patch_upstream_yuv6_globally() before "
            "measuring the pose cone (differentiable-YUV6 non-negotiable)."
        )
    g1 = grad[0, 1]  # frame1 channel: (H, W, C)
    norm = g1.pow(2).sum(dim=-1).sqrt()  # (H, W)
    norm_np = norm.detach().cpu().numpy().astype(np.float64)
    total_energy = float(np.abs(norm_np).sum())
    if total_energy <= _POSE_JACOBIAN_REACHABLE_FLOOR:
        raise Frame1JointSafeConeError(
            "PoseNet frame1 Jacobian is identically zero (total energy "
            f"{total_energy:.3e}) — the YUV6 gradient was NOT reachable. "
            "Refusing to emit an all-permissive pose-null cone. Patch "
            "rgb_to_yuv6 with patch_upstream_yuv6_globally() first."
        )
    return norm_np


# ---------------------------------------------------------------------------
# Grid alignment helpers
# ---------------------------------------------------------------------------


def _resize_to_grid(pixel_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear-resize a ``(H, W)`` map to ``(target_h, target_w)``."""

    import torch
    import torch.nn.functional as F

    src = np.asarray(pixel_map, dtype=np.float32)
    if src.shape == (target_h, target_w):
        return src.astype(np.float64)
    t = torch.from_numpy(src)[None, None, ...]
    up = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Cone assembly: intersect the seg-safe and pose-null half-cones
# ---------------------------------------------------------------------------


def assemble_joint_cone(
    *,
    seg_margin: np.ndarray,
    seg_boundary_slope: np.ndarray,
    seg_argmax_class: np.ndarray,
    pose_jacobian_norm: np.ndarray,
    config: Frame1ConeConfig,
) -> Frame1JointSafeCone:
    """Intersect the two half-cones into the joint frame1 safe cone.

    Seg-safe budget (distance-to-flip / sensitivity):
        seg_margin_budget_p = seg_margin_tol * margin_p / (slope_p + eps)

    Pose-safe budget (linearized response tolerance):
        pose_budget_p = pose_response_tol / (J_pose_p + eps)

    Joint cone radius is the INTERSECTION (the more binding of the two):
        radius_p = clamp(min(seg_margin_budget_p, pose_budget_p), floor, max)

    Joint coupled sensitivity (canonical P18/P19 coupling; reported for the
    bit-allocator hook — high sensitivity = bind, low = free byte):
        w_p = 100 * slope_p + pose_ail_gain * J_pose_p
    """

    seg_margin = np.asarray(seg_margin, dtype=np.float64)
    h, w = seg_margin.shape
    slope = _resize_to_grid(seg_boundary_slope, h, w)
    jpose = _resize_to_grid(pose_jacobian_norm, h, w)
    argmax_cls = np.asarray(seg_argmax_class, dtype=np.int64)
    if argmax_cls.shape != seg_margin.shape:
        raise Frame1JointSafeConeError("seg_argmax_class shape must match seg_margin")

    eps = 1e-12
    seg_budget = float(config.seg_margin_tol) * seg_margin / (slope + eps)
    pose_budget = float(config.pose_response_tol) / (jpose + eps)

    joint_sensitivity = _SEG_SCORE_WEIGHT * slope + config.pose_ail_gain * jpose

    radius = np.minimum(seg_budget, pose_budget)
    radius = np.clip(radius, config.min_radius_floor, config.max_radius)
    # Where the SegNet margin is exactly zero the pixel is already at the flip
    # boundary -> empty cone regardless of slope (no budget to move the argmax).
    radius = np.where(seg_margin <= 0.0, config.min_radius_floor, radius)

    empty_mask = radius <= config.min_radius_floor
    fragile_mask = radius < float(config.fragile_radius_threshold)

    per_region = _per_class_region_aggregates(radius, argmax_cls, empty_mask, fragile_mask)

    # "Usable budget" = a pixel with at least a fragile-threshold of safe room.
    usable = ~fragile_mask
    n = int(radius.size)
    summary = {
        "n_pixels": n,
        "grid_h": int(h),
        "grid_w": int(w),
        "fragile_radius_threshold": float(config.fragile_radius_threshold),
        # usable = not fragile (>= half a uint8 step of joint budget).
        "usable_budget_fraction": float(usable.mean()),
        # empty cone = the fragile / binding-constraint set (the directive's
        # "where the cone is empty = the fragile boundary set").
        "empty_cone_fraction": float(fragile_mask.mean()),
        "strict_zero_margin_fraction": float(empty_mask.mean()),
        "mean_radius_usable": float(radius[usable].mean()) if usable.any() else 0.0,
        "median_radius_all": float(np.median(radius)),
        "p10_radius_all": float(np.percentile(radius, 10)),
        "p90_radius_all": float(np.percentile(radius, 90)),
        "seg_binds_fraction": float((seg_budget <= pose_budget).mean()),
        "pose_binds_fraction": float((pose_budget < seg_budget).mean()),
        "mean_seg_margin": float(seg_margin.mean()),
        "mean_seg_boundary_slope": float(slope.mean()),
        "mean_pose_jacobian_norm": float(jpose.mean()),
        "pose_null_fraction": float((jpose < float(config.pose_null_fraction) * (jpose.max() + eps)).mean()),
        "pose_ail_gain": config.pose_ail_gain,
    }

    provenance = {
        "evidence_grade": "macOS-CPU advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_tag": "[macOS-CPU advisory]",
        "hardware_substrate": "local_macos_cpu",
        "promotable": False,
    }

    return Frame1JointSafeCone(
        schema=FRAME1_JOINT_SAFE_CONE_SCHEMA,
        seg_margin=seg_margin,
        seg_margin_budget=seg_budget,
        pose_jacobian_norm=jpose,
        pose_budget=pose_budget,
        joint_sensitivity=joint_sensitivity,
        joint_cone_radius=radius,
        empty_cone_mask=empty_mask,
        fragile_cone_mask=fragile_mask,
        seg_argmax_class=argmax_cls,
        per_region=per_region,
        summary=summary,
        config=config,
        provenance=provenance,
    )


def _per_class_region_aggregates(
    radius: np.ndarray,
    argmax_cls: np.ndarray,
    empty_mask: np.ndarray,
    fragile_mask: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Aggregate the cone radius per SegNet source-class region."""

    out: dict[int, dict[str, float]] = {}
    for cls in np.unique(argmax_cls):
        m = argmax_cls == cls
        r = radius[m]
        frag = fragile_mask[m]
        empt = empty_mask[m]
        usable = ~frag
        out[int(cls)] = {
            "n_pixels": int(m.sum()),
            "usable_budget_fraction": float(usable.mean()) if r.size else 0.0,
            "empty_cone_fraction": float(frag.mean()) if r.size else 0.0,
            "strict_zero_margin_fraction": float(empt.mean()) if r.size else 0.0,
            "mean_radius": float(r.mean()) if r.size else 0.0,
            "mean_radius_usable": float(r[usable].mean()) if usable.any() else 0.0,
            "min_radius": float(r.min()) if r.size else 0.0,
            "max_radius": float(r.max()) if r.size else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Top-level: compute the cone for one frame1 pair (real scorers)
# ---------------------------------------------------------------------------


def compute_frame1_joint_safe_cone(
    *,
    segnet: Any,
    posenet: Any,
    pair_btchwc_unit255: Any,
    config: Frame1ConeConfig | None = None,
) -> Frame1JointSafeCone:
    """Compute the frame1 joint safe cone for one pair using the REAL scorers.

    ``segnet`` / ``posenet`` are the loaded upstream models (CPU). ``posenet``
    MUST have a gradient-reachable ``rgb_to_yuv6`` (patch via
    :func:`tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally`
    before calling) — :func:`measure_posenet_frame1_jacobian` fails closed
    otherwise. ``pair_btchwc_unit255`` is the GT pair as ``(1, 2, H, W, 3)`` in
    scorer units [0, 255].
    """

    cfg = config or Frame1ConeConfig()
    seg_margin, seg_argmax = measure_segnet_frame1_margin(segnet, pair_btchwc_unit255)
    seg_slope = measure_segnet_frame1_boundary_slope(segnet, pair_btchwc_unit255)
    pose_jac = measure_posenet_frame1_jacobian(posenet, pair_btchwc_unit255)
    # seg margin/argmax are on the SegNet grid; align slope/pose to that grid.
    return assemble_joint_cone(
        seg_margin=seg_margin,
        seg_boundary_slope=seg_slope,
        seg_argmax_class=seg_argmax,
        pose_jacobian_norm=pose_jac,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Behavioral validation: the cone's falsifiable NO-FAKE proof
# ---------------------------------------------------------------------------


def measure_pair_distortion(dn: Any, gt_pair: Any, cand_pair: Any) -> tuple[float, float]:
    """Return ``(d_seg, d_pose)`` of one candidate frame1 pair vs the GT pair
    via the real upstream DistortionNet. Inputs are ``(1, 2, H, W, 3)`` [0, 255].

    CANONICAL exact CPU-torch pair-distortion measurement (NEVER MPS). This is
    the single home for the ``dn.compute_distortion`` wrapper that the #35 cone
    behavioral validation AND the Class-2 / Class-3 frame1 atom screeners
    (:mod:`tac.optimization.frame1_seg_safe_pose_atoms`,
    :mod:`tac.optimization.frame1_seg_repair_atoms`) all share — they import this
    rather than re-declaring an identical helper (dedup audit 2026-06-10)."""

    import torch

    with torch.inference_mode():
        d_pose, d_seg = dn.compute_distortion(gt_pair.float(), cand_pair.float())
    return float(d_seg.mean()), float(d_pose.mean())


# Internal alias preserved for the cone's own callsites.
_measure_distortion = measure_pair_distortion


def validate_cone_behaviorally(
    *,
    distortion_net: Any,
    gt_pair_btchwc_unit255: Any,
    cone: Frame1JointSafeCone,
    inside_scale: float = 0.5,
    outside_scale: float = 2.0,
    seg_stable_tol: float = 5e-4,
    min_discrimination_ratio: float = 3.0,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Behaviorally falsify the cone: perturb frame1 INSIDE the cone vs OUTSIDE.

    Frame1-only perturbations are applied (frame0 left untouched) so the result
    isolates the frame1 budget:

    - **INSIDE**: every frame1 pixel is perturbed by ``inside_scale * radius_p``
      with a random sign. By construction the argmax must not flip and the
      linearized pose response must stay within tolerance -> ``d_seg`` ~ 0 and
      ``d_pose`` small. A cone whose inside perturbation moves ``d_seg`` is too
      permissive (FAKE-permissive).
    - **OUTSIDE (fragile)**: the FRAGILE pixels are perturbed by
      ``outside_scale * fragile_radius_threshold`` (a perturbation deliberately
      exceeding their tiny budget) -> ``d_seg`` / ``d_pose`` MUST move measurably.
      A cone whose outside perturbation does NOT move the score does not
      discriminate (FAKE-undiscriminating).

    Returns a dict with the measured deltas and PASS/FAIL verdicts. A cone that
    does not discriminate (inside moves seg OR outside does not move seg) is a
    FAKE cone per the NO-FAKE discipline.
    """

    import numpy as np
    import torch

    pair = _ensure_pair_btchwc(gt_pair_btchwc_unit255)
    h, w = cone.joint_cone_radius.shape
    # frame1 native grid may differ from the SegNet cone grid; map the cone
    # radius onto the frame1 pixel grid the perturbation is applied on.
    f1_native = pair[0, 1]  # (Hn, Wn, C)
    hn, wn = int(f1_native.shape[0]), int(f1_native.shape[1])
    radius_native = _resize_to_grid(cone.joint_cone_radius, hn, wn)
    fragile_native = _resize_to_grid(cone.fragile_cone_mask.astype(np.float64), hn, wn) > 0.5

    base_seg, base_pose = _measure_distortion(distortion_net, pair, pair)

    rng = np.random.default_rng(rng_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(hn, wn)).astype(np.float64)

    # INSIDE: perturb every pixel by inside_scale * its own safe radius.
    delta_in = (signs * radius_native * float(inside_scale))[..., None]  # (Hn, Wn, 1)
    cand_in = pair.clone()
    cand_in[0, 1] = (f1_native + torch.from_numpy(delta_in).float()).clamp(0.0, 255.0)
    in_seg, in_pose = _measure_distortion(distortion_net, pair, cand_in)

    # OUTSIDE: perturb ONLY the fragile pixels by a perturbation that exceeds
    # their (sub-threshold) budget -> must move the score.
    out_delta = np.zeros((hn, wn), dtype=np.float64)
    bump = float(outside_scale) * float(cone.config.fragile_radius_threshold)
    # ensure a meaningful absolute bump even if the threshold is tiny.
    bump = max(bump, 4.0)
    out_delta[fragile_native] = signs[fragile_native] * bump
    cand_out = pair.clone()
    cand_out[0, 1] = (f1_native + torch.from_numpy(out_delta[..., None]).float()).clamp(0.0, 255.0)
    out_seg, out_pose = _measure_distortion(distortion_net, pair, cand_out)

    inside_seg_delta = in_seg - base_seg
    inside_pose_delta = in_pose - base_pose
    outside_seg_delta = out_seg - base_seg
    outside_pose_delta = out_pose - base_pose

    inside_seg_stable = abs(inside_seg_delta) <= float(seg_stable_tol)
    outside_moves = (abs(outside_seg_delta) > float(seg_stable_tol)) or (
        abs(outside_pose_delta) > abs(inside_pose_delta) + 1e-9
    )
    # Relative-discrimination ratios: the cone is a LINEARIZED budget, so the
    # robust falsifiable proof is that perturbing OUTSIDE the cone moves the
    # score substantially more than perturbing INSIDE it (not a brittle absolute
    # zero). Inf-safe ratio when the inside delta is ~0 (the strongest case).
    seg_ratio = abs(outside_seg_delta) / (abs(inside_seg_delta) + 1e-12)
    pose_ratio = abs(outside_pose_delta) / (abs(inside_pose_delta) + 1e-12)
    discriminates = bool(
        inside_seg_stable
        and outside_moves
        and seg_ratio >= float(min_discrimination_ratio)
        and pose_ratio >= float(min_discrimination_ratio)
    )

    return {
        "schema": "frame1_joint_safe_cone_behavioral_validation.v1",
        "base_d_seg": base_seg,
        "base_d_pose": base_pose,
        "inside_d_seg": in_seg,
        "inside_d_pose": in_pose,
        "outside_d_seg": out_seg,
        "outside_d_pose": out_pose,
        "inside_seg_delta": inside_seg_delta,
        "inside_pose_delta": inside_pose_delta,
        "outside_seg_delta": outside_seg_delta,
        "outside_pose_delta": outside_pose_delta,
        "n_fragile_pixels_perturbed": int(fragile_native.sum()),
        "inside_scale": float(inside_scale),
        "outside_scale": float(outside_scale),
        "outside_bump_abs": bump,
        "seg_stable_tol": float(seg_stable_tol),
        "min_discrimination_ratio": float(min_discrimination_ratio),
        "seg_discrimination_ratio": float(seg_ratio),
        "pose_discrimination_ratio": float(pose_ratio),
        "inside_seg_stable": bool(inside_seg_stable),
        "outside_moves_score": bool(outside_moves),
        "cone_discriminates": discriminates,
        "evidence_grade": "macOS-CPU advisory",
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "DEFAULT_POSE_NULL_FRACTION",
    "DEFAULT_SEG_BOUNDARY_TAU",
    "FRAME1_JOINT_SAFE_CONE_SCHEMA",
    "Frame1ConeConfig",
    "Frame1JointSafeCone",
    "Frame1JointSafeConeError",
    "assemble_joint_cone",
    "compute_frame1_joint_safe_cone",
    "measure_pair_distortion",
    "measure_posenet_frame1_jacobian",
    "measure_segnet_frame1_boundary_slope",
    "measure_segnet_frame1_margin",
    "validate_cone_behaviorally",
]
