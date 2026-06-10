# SPDX-License-Identifier: MIT
"""Class-3 — frame-1 Seg-POSITIVE correction atoms (repair, not tricks).

The contest objective is the evaluator quotient
``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``.  ``d_seg`` is the
per-pixel argmax-flip RATE between a vehicle's RENDERED frame-1 and the GT
frame-1 (``upstream/modules.py:112``: SegNet reads only ``x[:, -1, ...]``).  Where
a vehicle's reconstruction has driven a frame-1 pixel's SegNet argmax to the
WRONG class — typically at class boundaries, thin classes, or the fragile
boundary tube where the SOURCE-class margin is small — that pixel costs the
score 1/N of the seg term.  This module generates the SECOND frame-1 atom class:
**Seg-POSITIVE repair atoms** — targeted color/luma corrections that push the
rendered frame-1's SegNet argmax BACK toward the GT class at exactly those
small-but-recoverable margin pixels.

This is NOT a trick (Class-2 is the seg-SAFE pose lever; Class-3 is the direct
seg-REPAIR lever).  It is ALSO SNeRV's nonrate lever: SNeRV's seg term (0.2468)
is precisely the accumulated frame-1 SegNet disagreement these atoms repair, at
the bytes cost of coding the correction support.

THE METHODOLOGY (design `pr110pp_frame1_joint_methodology_v1`, Class 3)
======================================================================
1. **GENERATED FROM the margin field + fragile masks + thin-class supports**
   (NEVER global 8x8 tiles).  The repair targets pixels where the rendered
   frame's SOURCE-class margin is *small-but-recoverable*: the rendered SegNet
   argmax disagrees with the GT argmax (a flip we can fix) AND the GT-class logit
   is close behind the rendered top class (the margin to recover is small, so a
   small correction can flip it back).  The candidate support is the union of:
     a. **boundary-tube** pixels (small rendered top1-top2 margin = near a wall);
     b. **thin-class support** pixels (the GT pixel belongs to a class with few
        pixels in the frame — these are the high-leverage repairs);
     c. **fragile-cone** pixels (the #35 fragile mask = the binding-constraint set
        where the joint budget is empty and small errors flip the argmax).
2. **Margin-normal corrections**.  Each supported pixel is corrected by a signed
   step along the direction that increases the GT-class SegNet logit relative to
   the rendered top class — operationally a luma/color shift toward the GT
   frame-1's appearance at that pixel (the margin-normal direction), scaled by the
   recoverable margin so the step is the smallest that flips the argmax back.
3. **Admission by THE LAW** (design verbatim):
   ``100*Δd_seg + Δsqrt(10*d_pose) + 25*Δbytes/N < 0``.  An atom is admitted only
   if its measured exact ΔS (seg term improvement minus the pose term cost minus
   the rate cost of coding the correction support) is net-negative.  ``Δbytes`` is
   the atom's coded cost (support map + correction values + selector bits est).

VEHICLE-AGNOSTIC INTERFACE (design verbatim)
============================================
*"Design the atom interface vehicle-agnostic (operates on rendered frames +
cone/margin maps, not on a specific carrier)."*  A Class-3 atom is generated from
``(rendered_frame1, gt_frame1, margin_field, fragile_mask)`` and applied to ANY
vehicle's rendered frame-1 — SNeRV render, frontier compose, PR110++ frames,
HiNeRV, PACT-VQ.  It carries NO carrier-specific state.  The repair target (the
GT-class appearance) and the margin field come from the cone artifacts; the
correction is a delta on the rendered frame the vehicle emits.

THE FALSIFIABLE CHECK (NO-FAKE, exact CPU-torch screening)
==========================================================
Every atom is screened on the exact local CPU-torch DistortionNet (real frozen
scorers, NEVER MPS): ``d_seg`` of (repaired render vs GT) must be STRICTLY LESS
than ``d_seg`` of (original render vs GT) — the repair must actually reduce the
argmax-flip rate.  An atom that does not reduce d_seg is REJECTED.  The pose
term cost is measured (a frame-1 change touches PoseNet) and folded into THE LAW.

COMPUTE-SUBSTRATE LAW (operator correction 2026-06-10)
======================================================
- GENERATION + search: MLX (the recoverable-margin ranking saturates the M5 Max
  128GB unified memory); numpy reference is the portability oracle (Catalog #383).
- ADVISORY screening: local CPU-torch exact frozen scorers.  ``[macOS-CPU advisory]``.
- RANKING + admission: contest host ONLY (the R1 lesson).  This module emits
  advisory rows + a host-ranking packet; it NEVER claims an on-host accept.
- MPS: NEVER — any artifact with an MPS ancestor is contamination requiring
  rebuild.

Evidence grade: ``[macOS-CPU advisory]`` / ``[macOS-MLX research-signal]`` —
non-promotable per Catalog #192/#341/#127/#323.  $0 local, NO cloud, NO paid GPU,
NO MPS.  The atoms PROPOSE; the contest-host exact replay ratifies.

Cross-references
----------------
- ``tac.optimization.frame1_joint_safe_cone`` (#35; the margin field + fragile mask)
- ``tac.optimization.evaluator_response_atlas`` (#36; fragile clusters to protect/repair)
- ``tac.optimization.frame1_seg_safe_pose_atoms`` (Class-2 sister; the seg-SAFE lever)
- ``tac.optimization.resize_null_preimage`` (#49; universal postprocessor)
- ``tac.optimization.lf_payload_rate_distortion`` (#46; THE LAW rate term)
- ``tac.logit_margin_sensitivity_weighted`` (the margin-weighted loss family)
- ``.omx/research/pr110pp_frame1_joint_methodology_v1_20260610.md`` (the design)
- ``.omx/research/snerv_rate_attack_round2_directive_20260610.md`` (SNeRV nonrate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.optimization.frame1_seg_safe_pose_atoms import _resize_map

FRAME1_SEG_REPAIR_ATOM_SCHEMA = "frame1_seg_repair_atom.v1"

# Canonical Tier A non-promotable false-authority markers (Catalog #341/#323).
SEG_REPAIR_ATOM_PROVENANCE: dict[str, Any] = {
    "evidence_grade": "macOS-CPU advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "authority_host": "macos_cpu_advisory",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "hardware_substrate": "local_macos_cpu",
}

# THE LAW constants (mirror lf_payload_rate_distortion / resize_null_preimage).
_RATE_COEF: float = 25.0
_CONTEST_TOTAL_BYTES: int = 37_545_489
_POSE_TEN: float = 10.0
_SEG_SCORE_WEIGHT: float = 100.0


class Frame1SegRepairAtomError(ValueError):
    """Raised on malformed Class-3 repair-atom inputs / contract violations."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegRepairAtomConfig:
    """Configuration for Class-3 Seg-POSITIVE repair atom generation.

    ``boundary_margin_percentile`` selects boundary-tube pixels: those whose
    rendered top1-top2 margin is in the BOTTOM ``boundary_margin_percentile``
    (small margin = near a class wall = recoverable).  ``thin_class_max_fraction``
    flags a SegNet class as "thin" when it covers <= this fraction of the GT
    frame's pixels (thin classes are high-leverage repairs).  ``include_fragile``
    unions the #35 fragile-cone mask into the candidate support (the empty-cone
    binding-constraint set).  ``support_top_fraction`` is the fraction of the
    candidate flip-pixels (ranked by recoverable margin) the atom repairs.
    ``correction_fraction`` is the fraction of the per-pixel GT-vs-render
    appearance gap the correction moves (1.0 = snap fully to GT appearance).
    ``min_support_pixels`` keeps the atom from degenerating to a no-op.
    """

    boundary_margin_percentile: float = 0.5
    thin_class_max_fraction: float = 0.10
    include_fragile: bool = True
    support_top_fraction: float = 0.25
    correction_fraction: float = 1.0
    min_support_pixels: int = 16
    # bytes-per-support-pixel estimate for the rate cost of coding the correction
    # (a coarse advisory: a coded sparse correction stores ~ position + value).
    bytes_per_support_pixel: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 < self.boundary_margin_percentile <= 1.0:
            raise Frame1SegRepairAtomError("boundary_margin_percentile must be in (0, 1]")
        if not 0.0 < self.thin_class_max_fraction <= 1.0:
            raise Frame1SegRepairAtomError("thin_class_max_fraction must be in (0, 1]")
        if not 0.0 < self.support_top_fraction <= 1.0:
            raise Frame1SegRepairAtomError("support_top_fraction must be in (0, 1]")
        if not 0.0 < self.correction_fraction <= 1.0:
            raise Frame1SegRepairAtomError("correction_fraction must be in (0, 1]")
        if self.min_support_pixels < 1:
            raise Frame1SegRepairAtomError("min_support_pixels must be >= 1")
        if self.bytes_per_support_pixel < 0.0:
            raise Frame1SegRepairAtomError("bytes_per_support_pixel must be >= 0")


# ---------------------------------------------------------------------------
# The repair atom: a frame-1 correction restricted to recoverable flip pixels
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame1SegRepairAtom:
    """A frame-1 Seg-POSITIVE repair atom (a correction field + provenance).

    The atom is a ``(H, W, C)`` per-pixel signed correction on the frame-1 scorer
    grid (384x512), restricted to the recoverable flip pixels (boundary-tube ∪
    thin-class ∪ fragile).  ``apply`` returns the repaired rendered frame-1
    (vehicle-agnostic: the correction is added to ANY rendered frame-1).
    """

    schema: str
    pair_index: int
    target_frame: int  # always 1 for Class-3
    support_or_cone_id: str
    # (H, W, C) float64 signed per-pixel correction on the frame-1 scorer grid.
    correction: np.ndarray
    # (H, W) bool support mask (where correction is non-zero).
    support_mask: np.ndarray
    n_support_pixels: int
    correction_fraction: float
    n_boundary: int
    n_thin_class: int
    n_fragile: int
    config: SegRepairAtomConfig
    provenance: dict[str, Any] = field(default_factory=lambda: dict(SEG_REPAIR_ATOM_PROVENANCE))

    def apply(self, rendered_frame1_hwc_unit255: Any) -> Any:
        """Return the repaired rendered frame-1 (vehicle-agnostic).

        ``rendered_frame1`` is ``(H, W, 3)`` in scorer units [0,255] — the
        vehicle's reconstruction.  The correction is added and clamped to
        [0, 255].  Returns a numpy ``(H, W, 3)`` float64 frame.
        """

        r = np.asarray(rendered_frame1_hwc_unit255, dtype=np.float64)
        if r.ndim != 3 or r.shape[-1] != 3:
            raise Frame1SegRepairAtomError(
                f"rendered_frame1 must be (H, W, 3); got {r.shape}"
            )
        h, w = r.shape[0], r.shape[1]
        corr = self.correction
        if corr.shape[:2] != (h, w):
            corr = np.stack(
                [_resize_map(corr[:, :, ch], h, w) for ch in range(corr.shape[2])],
                axis=-1,
            )
        return np.clip(r + corr, 0.0, 255.0)

    def apply_to_pair(self, pair_btchwc_unit255: Any) -> Any:
        """Return the candidate pair with the repair applied to FRAME-1 only.

        ``pair`` is ``(1, 2, H, W, 3)`` — the vehicle's rendered pair; frame-0 is
        left untouched (Class-3 repairs frame-1).  Returns a torch tensor.
        """

        import torch

        if not isinstance(pair_btchwc_unit255, torch.Tensor):
            pair = torch.as_tensor(np.asarray(pair_btchwc_unit255)).float()
        else:
            pair = pair_btchwc_unit255.float().clone()
        if pair.ndim != 5 or pair.shape[1] != 2 or pair.shape[-1] != 3:
            raise Frame1SegRepairAtomError(
                f"pair must be (1, 2, H, W, 3); got {tuple(pair.shape)}"
            )
        rendered_f1 = pair[0, 1].detach().cpu().numpy()
        repaired = self.apply(rendered_f1)
        cand = pair.clone()
        cand[0, 1] = torch.from_numpy(repaired).float()
        return cand


@dataclass(frozen=True)
class SegRepairAtomRow:
    """The advisory row a Class-3 atom emits after exact-scorer screening.

    Mirrors the design row schema (Class 3 fields).  ``d_seg_delta`` must be < 0
    for an accepted atom (the repair reduced the argmax-flip rate).
    ``score_delta_advisory`` is THE LAW ΔS; ``accepted`` requires net-negative ΔS.
    """

    schema: str
    pair_index: int
    target_frame: int
    support_or_cone_id: str
    d_seg_delta: float
    d_pose_delta: float
    score_delta_advisory: float
    selector_bits_est: float
    value_per_byte: float
    n_support_pixels: int
    preimage_tier1_applied: bool
    preimage_max_abs_residual: float
    preimage_bytes_freed: int
    authority_host: str
    accepted: bool
    rejected_reason: str
    provenance: dict[str, Any] = field(default_factory=lambda: dict(SEG_REPAIR_ATOM_PROVENANCE))

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pair_index": self.pair_index,
            "target_frame": self.target_frame,
            "support_or_cone_id": self.support_or_cone_id,
            "d_seg_delta": self.d_seg_delta,
            "d_pose_delta": self.d_pose_delta,
            "score_delta_advisory": self.score_delta_advisory,
            "selector_bits_est": self.selector_bits_est,
            "value_per_byte": self.value_per_byte,
            "n_support_pixels": self.n_support_pixels,
            "preimage_tier1_applied": self.preimage_tier1_applied,
            "preimage_max_abs_residual": self.preimage_max_abs_residual,
            "preimage_bytes_freed": self.preimage_bytes_freed,
            "authority_host": self.authority_host,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "authority_tier": "advisory",
            "metric_family": "joint_seg_pose",
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Repair-target measurement: where the rendered frame's SegNet argmax is WRONG
# ---------------------------------------------------------------------------
def measure_segnet_argmax(segnet: Any, frame1_hwc_unit255: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(argmax_HW, margin_HW)`` of a single frame-1 under the real SegNet.

    Builds a degenerate pair ``(1, 2, H, W, 3)`` (frame-0 == frame-1 since SegNet
    reads only the last frame) and runs one real forward.  ``argmax`` is the
    per-pixel class id; ``margin`` is the top1-top2 logit margin (distance to
    flip).  Exact CPU-torch (NEVER MPS).
    """

    import torch

    r = np.asarray(frame1_hwc_unit255, dtype=np.float64)
    if r.ndim != 3 or r.shape[-1] != 3:
        raise Frame1SegRepairAtomError(f"frame1 must be (H, W, 3); got {r.shape}")
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1, 2, H, W, 3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(xp)  # last frame -> (1, 3, 384, 512)
        logits = segnet(seg_in)  # (1, 5, 384, 512)
        top2 = torch.topk(logits, k=2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)[0]
        argmax = logits.argmax(dim=1)[0]
    return (
        argmax.detach().cpu().numpy().astype(np.int64),
        margin.detach().cpu().numpy().astype(np.float64),
    )


@dataclass(frozen=True)
class RepairTargets:
    """The repair-target fields a Class-3 atom needs (from real scorers + GT)."""

    rendered_argmax: np.ndarray  # (H, W) rendered frame-1 SegNet argmax
    gt_argmax: np.ndarray  # (H, W) GT frame-1 SegNet argmax (the target)
    rendered_margin: np.ndarray  # (H, W) rendered top1-top2 margin
    fragile_mask: np.ndarray  # (H, W) bool #35 fragile-cone mask
    # the per-pixel appearance gap GT - rendered (the correction direction).
    appearance_gap: np.ndarray  # (H, W, C)

    @classmethod
    def measure(
        cls,
        *,
        segnet: Any,
        rendered_frame1_hwc_unit255: Any,
        gt_frame1_hwc_unit255: Any,
        fragile_mask: np.ndarray,
    ) -> RepairTargets:
        """Measure the repair targets with the real SegNet (exact CPU-torch)."""

        r = np.asarray(rendered_frame1_hwc_unit255, dtype=np.float64)
        g = np.asarray(gt_frame1_hwc_unit255, dtype=np.float64)
        if r.shape != g.shape or r.ndim != 3 or r.shape[-1] != 3:
            raise Frame1SegRepairAtomError(
                f"rendered/gt frame1 must be matching (H, W, 3); got {r.shape} vs {g.shape}"
            )
        rendered_argmax, rendered_margin = measure_segnet_argmax(segnet, r)
        gt_argmax, _ = measure_segnet_argmax(segnet, g)
        h, w = r.shape[0], r.shape[1]
        frag = _resize_map(np.asarray(fragile_mask, dtype=np.float64), h, w) > 0.5
        gap = g - r  # GT - rendered = the correction direction toward GT appearance
        return cls(
            rendered_argmax=rendered_argmax,
            gt_argmax=gt_argmax,
            rendered_margin=rendered_margin,
            fragile_mask=frag,
            appearance_gap=gap,
        )


# ---------------------------------------------------------------------------
# Recoverable-margin ranking: MLX (unified mem) + numpy reference (Catalog #383)
# ---------------------------------------------------------------------------
SEG_REPAIR_LEVERAGE_FP32_ATOL = 1e-3


def _boundary_margin_cut(
    margin: np.ndarray, flip: np.ndarray, percentile: float
) -> float:
    """The boundary-tube margin cut: the ``percentile`` quantile of the rendered
    margin over flip pixels (small margin = near a class wall = recoverable).
    Falls back to the frame max when there are no flip pixels (no tube)."""

    if margin[flip].size:
        return float(np.quantile(margin[flip], percentile))
    return float(margin.max()) if margin.size else 1.0


def _thin_class_mask(gt_argmax: np.ndarray, max_fraction: float) -> np.ndarray:
    """Boolean ``(H, W)`` mask: True where the GT pixel belongs to a thin class
    (a class covering <= ``max_fraction`` of the frame's pixels)."""

    h, w = gt_argmax.shape
    total = float(h * w)
    out = np.zeros((h, w), dtype=bool)
    for cls in np.unique(gt_argmax):
        frac = float((gt_argmax == cls).sum()) / total
        if frac <= float(max_fraction):
            out |= gt_argmax == cls
    return out


def repair_leverage_numpy(targets: RepairTargets, config: SegRepairAtomConfig) -> np.ndarray:
    """Canonical numpy reference: per-pixel repair leverage over flip pixels.

    A pixel is a candidate repair iff the rendered SegNet argmax DISAGREES with
    the GT argmax (a flip we can fix).  Among flip pixels the leverage is high
    when the rendered margin is SMALL (recoverable: a small correction can flip
    the argmax back) AND the pixel is in a high-leverage region (boundary-tube ∪
    thin-class ∪ fragile).  Non-flip pixels score 0 (nothing to repair).
    """

    flip = targets.rendered_argmax != targets.gt_argmax
    # recoverable: small rendered margin (cheap to flip back). Invert + normalise.
    margin = targets.rendered_margin
    max_m = float(margin.max()) if margin.size else 1.0
    recover = (max_m - margin) / (max_m + 1e-12)  # 1 = smallest margin = recoverable

    # region weight: boundary-tube ∪ thin-class ∪ fragile (shared with the MLX path).
    region = _region_mask_numpy(targets, config)

    lev = np.where(flip & region, recover, 0.0)
    return lev.astype(np.float64)


def repair_leverage_mlx(targets: RepairTargets, config: SegRepairAtomConfig) -> np.ndarray:
    """MLX (Apple unified memory) repair leverage; matches the numpy reference to
    fp32 tolerance (Catalog #383).  The flip/region MASKS are computed in numpy
    (integer argmax compares + per-class fractions are not the parallel hot path);
    the recoverable-margin normalisation + elementwise product is the MLX kernel.
    Raises ``ImportError`` if MLX is unavailable (caller falls back to numpy)."""

    import mlx.core as mx

    flip = targets.rendered_argmax != targets.gt_argmax
    boundary_thin_fragile = _region_mask_numpy(targets, config)
    cand = flip & boundary_thin_fragile

    margin = mx.array(targets.rendered_margin.astype(np.float32))
    max_m = mx.max(margin)
    mx.eval(max_m)
    recover = (max_m - margin) / (max_m + 1e-12)
    cand_mx = mx.array(cand.astype(np.float32))
    lev = recover * cand_mx
    mx.eval(lev)
    return np.asarray(lev).astype(np.float64)


def _region_mask_numpy(targets: RepairTargets, config: SegRepairAtomConfig) -> np.ndarray:
    """The boundary-tube ∪ thin-class ∪ fragile region mask (numpy; shared by the
    MLX path which only accelerates the recoverable-margin product)."""

    flip = targets.rendered_argmax != targets.gt_argmax
    margin = targets.rendered_margin
    bcut = _boundary_margin_cut(margin, flip, config.boundary_margin_percentile)
    boundary = margin <= bcut
    thin = _thin_class_mask(targets.gt_argmax, config.thin_class_max_fraction)
    region = boundary | thin
    if config.include_fragile:
        region = region | targets.fragile_mask
    return region


def repair_leverage(
    targets: RepairTargets, config: SegRepairAtomConfig, *, prefer_mlx: bool = True
) -> np.ndarray:
    """Repair leverage with MLX-first + numpy-portable fallback."""

    if prefer_mlx:
        try:
            return repair_leverage_mlx(targets, config)
        except ImportError:
            pass
    return repair_leverage_numpy(targets, config)


# ---------------------------------------------------------------------------
# Atom generation: margin-normal correction over recoverable flip pixels
# ---------------------------------------------------------------------------
def generate_seg_repair_atom(
    *,
    pair_index: int,
    targets: RepairTargets,
    config: SegRepairAtomConfig | None = None,
    prefer_mlx: bool = True,
) -> Frame1SegRepairAtom:
    """Generate ONE Class-3 Seg-POSITIVE repair atom from the repair targets.

    1. candidate = flip pixels (rendered argmax != GT argmax) in the high-leverage
       region (boundary-tube ∪ thin-class ∪ fragile).
    2. support = top ``support_top_fraction`` of candidates by recoverable-margin
       leverage (small rendered margin = cheapest to flip back).
    3. correction_p = correction_fraction * appearance_gap_p (move the pixel
       toward the GT appearance — the margin-normal direction that increases the
       GT-class logit).

    The returned atom carries a non-empty support (or raises) — a degenerate
    empty-support atom is a FAKE atom per the NO-FAKE discipline.
    """

    cfg = config or SegRepairAtomConfig()
    flip = targets.rendered_argmax != targets.gt_argmax
    region = _region_mask_numpy(targets, cfg)
    candidate = flip & region
    h, w = flip.shape

    n_candidate = int(candidate.sum())
    if n_candidate == 0:
        raise Frame1SegRepairAtomError(
            f"pair {pair_index}: no recoverable flip pixels (rendered argmax already "
            f"matches GT on the boundary/thin/fragile region — nothing to repair)"
        )

    leverage = repair_leverage(targets, cfg, prefer_mlx=prefer_mlx)
    lev_masked = np.where(candidate, leverage, -np.inf)

    n_support = max(
        int(cfg.min_support_pixels),
        round(float(cfg.support_top_fraction) * n_candidate),
    )
    n_support = min(n_support, n_candidate)

    flat = lev_masked.reshape(-1)
    if n_support >= flat.size:
        top_idx = np.where(np.isfinite(flat))[0]
    else:
        part = np.argpartition(-flat, n_support - 1)[:n_support]
        part = part[np.isfinite(flat[part])]
        top_idx = part
    support_mask = np.zeros(h * w, dtype=bool)
    support_mask[top_idx] = True
    support_mask = support_mask.reshape(h, w)

    n_actual = int(support_mask.sum())
    if n_actual == 0:
        raise Frame1SegRepairAtomError(
            f"pair {pair_index}: repair atom support is empty after ranking (FAKE atom)"
        )

    # correction = correction_fraction * (GT - rendered) on the support pixels.
    correction = np.zeros((h, w, targets.appearance_gap.shape[2]), dtype=np.float64)
    gap = targets.appearance_gap
    for ch in range(gap.shape[2]):
        plane = np.zeros((h, w), dtype=np.float64)
        plane[support_mask] = (float(cfg.correction_fraction) * gap[:, :, ch])[support_mask]
        correction[:, :, ch] = plane

    # provenance counts: how many of each region type are in the support.
    margin = targets.rendered_margin
    bcut = _boundary_margin_cut(margin, flip, cfg.boundary_margin_percentile)
    boundary = (margin <= bcut) & support_mask
    thin = _thin_class_mask(targets.gt_argmax, cfg.thin_class_max_fraction) & support_mask
    frag = targets.fragile_mask & support_mask

    support_id = f"seg_repair_pair{pair_index}_n{n_actual}_cf{cfg.correction_fraction:g}"
    return Frame1SegRepairAtom(
        schema=FRAME1_SEG_REPAIR_ATOM_SCHEMA,
        pair_index=int(pair_index),
        target_frame=1,
        support_or_cone_id=support_id,
        correction=correction,
        support_mask=support_mask,
        n_support_pixels=n_actual,
        correction_fraction=float(cfg.correction_fraction),
        n_boundary=int(boundary.sum()),
        n_thin_class=int(thin.sum()),
        n_fragile=int(frag.sum()),
        config=cfg,
        provenance=dict(SEG_REPAIR_ATOM_PROVENANCE),
    )


# ---------------------------------------------------------------------------
# Exact CPU-torch screening: THE LAW admission (100*Δd_seg + Δpose + rate < 0)
# ---------------------------------------------------------------------------
def _measure_exact_distortion(dn: Any, gt_pair: Any, cand_pair: Any) -> tuple[float, float]:
    """Return ``(d_seg, d_pose)`` of a candidate pair vs the GT pair via the real
    upstream DistortionNet (exact CPU-torch, NEVER MPS)."""

    import torch

    with torch.inference_mode():
        d_pose, d_seg = dn.compute_distortion(gt_pair.float(), cand_pair.float())
    return float(d_seg.mean()), float(d_pose.mean())


def screen_repair_atom_exact(
    *,
    atom: Frame1SegRepairAtom,
    distortion_net: Any,
    rendered_pair_btchwc_unit255: Any,
    gt_pair_btchwc_unit255: Any,
    selector_bits_est: float | None = None,
    preimage_tier1_applied: bool = False,
    preimage_max_abs_residual: float = 0.0,
    preimage_bytes_freed: int = 0,
) -> SegRepairAtomRow:
    """Screen ONE repair atom on the exact CPU-torch scorers — THE LAW admission.

    ``rendered_pair`` is the vehicle's RENDERED pair (the baseline); ``gt_pair`` is
    the GT (the score is measured render-vs-GT).  The atom is admitted iff:
      - the repaired ``d_seg`` (repaired-render vs GT) is STRICTLY LESS than the
        original ``d_seg`` (original-render vs GT) — the repair actually works; AND
      - the measured ΔS = ``100*Δd_seg + Δsqrt(10*d_pose) + 25*Δbytes/N`` < 0 (THE
        LAW: net score improvement after paying the pose cost + correction bytes).
    """

    cfg = atom.config
    # baseline: original render vs GT.
    base_seg, base_pose = _measure_exact_distortion(
        distortion_net, gt_pair_btchwc_unit255, rendered_pair_btchwc_unit255
    )
    # repaired: repaired render vs GT.
    repaired_pair = atom.apply_to_pair(rendered_pair_btchwc_unit255)
    rep_seg, rep_pose = _measure_exact_distortion(
        distortion_net, gt_pair_btchwc_unit255, repaired_pair
    )

    d_seg_delta = rep_seg - base_seg  # negative = repair reduced seg disagreement
    d_pose_delta = rep_pose - base_pose

    # selector bits: default = coded cost of the correction support (advisory est).
    if selector_bits_est is None:
        selector_bits = 8.0 * float(cfg.bytes_per_support_pixel) * atom.n_support_pixels
    else:
        selector_bits = float(selector_bits_est)
    selector_bytes = selector_bits / 8.0
    net_bytes = selector_bytes - float(preimage_bytes_freed)

    seg_term_delta = _SEG_SCORE_WEIGHT * d_seg_delta
    pose_term_delta = float(
        np.sqrt(_POSE_TEN * max(rep_pose, 0.0)) - np.sqrt(_POSE_TEN * max(base_pose, 0.0))
    )
    rate_delta = _RATE_COEF * net_bytes / float(_CONTEST_TOTAL_BYTES)
    score_delta = seg_term_delta + pose_term_delta + rate_delta

    seg_reduced = d_seg_delta < 0.0
    law_passes = score_delta < 0.0

    if not seg_reduced:
        accepted = False
        reason = "seg_not_reduced"
    elif not law_passes:
        accepted = False
        reason = "law_net_nonnegative"
    else:
        accepted = True
        reason = ""

    byte_cost = max(selector_bytes, 1e-9)
    value_per_byte = float(-score_delta / byte_cost) if byte_cost > 0 else 0.0

    return SegRepairAtomRow(
        schema=FRAME1_SEG_REPAIR_ATOM_SCHEMA,
        pair_index=atom.pair_index,
        target_frame=atom.target_frame,
        support_or_cone_id=atom.support_or_cone_id,
        d_seg_delta=d_seg_delta,
        d_pose_delta=d_pose_delta,
        score_delta_advisory=score_delta,
        selector_bits_est=selector_bits,
        value_per_byte=value_per_byte,
        n_support_pixels=atom.n_support_pixels,
        preimage_tier1_applied=bool(preimage_tier1_applied),
        preimage_max_abs_residual=float(preimage_max_abs_residual),
        preimage_bytes_freed=int(preimage_bytes_freed),
        authority_host="macos_cpu_advisory",
        accepted=accepted,
        rejected_reason=reason,
        provenance=dict(SEG_REPAIR_ATOM_PROVENANCE),
    )


__all__ = [
    "FRAME1_SEG_REPAIR_ATOM_SCHEMA",
    "SEG_REPAIR_ATOM_PROVENANCE",
    "SEG_REPAIR_LEVERAGE_FP32_ATOL",
    "Frame1SegRepairAtom",
    "Frame1SegRepairAtomError",
    "RepairTargets",
    "SegRepairAtomConfig",
    "SegRepairAtomRow",
    "generate_seg_repair_atom",
    "measure_segnet_argmax",
    "repair_leverage",
    "repair_leverage_mlx",
    "repair_leverage_numpy",
    "screen_repair_atom_exact",
]
