# SPDX-License-Identifier: MIT
"""Class-2 — frame-1 Seg-SAFE pose action atoms (the lab's first frame-1 machinery).

The contest objective is the evaluator quotient
``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``.  Frame-0 was easy:
SegNet reads only ``x[:, -1, ...]`` (frame1) per ``upstream/modules.py:108``, so a
frame-0 perturbation has ``Δd_seg ≡ 0`` and the action class reduces to the pose
term vs selector bytes (the matured PR110 lane).  **Frame-1 is the constrained
JOINT problem**: a frame-1 perturbation touches BOTH scorers (SegNet directly,
PoseNet through its frame1 input channels), so ``Δd_seg ≠ 0`` AND ``Δd_pose ≠ 0``
in general.  This module generates the FIRST of the two frame-1 atom classes the
design names: **Seg-SAFE pose atoms** — perturbations that help PoseNet while
staying inside the source SegNet chamber (``Δd_seg`` exactly 0 = argmax-identical).

THE METHODOLOGY (design `pr110pp_frame1_joint_methodology_v1`, Class 2)
======================================================================
1. **Support restricted to the OPEN cone**.  The #35 joint safe cone gives, per
   frame-1 pixel, a perturbation budget ``joint_cone_radius`` such that
   ``|delta_p| <= radius_p`` leaves the SegNet argmax unflipped AND the linearized
   pose response within tolerance.  An atom's support is restricted BY
   CONSTRUCTION to pixels with ``joint_cone_radius >= open_cone_threshold`` — the
   fragile 51.4% (``fragile_cone_mask`` / ``empty_cone``) is excluded.  This is the
   structural seg-safety: a perturbation that never touches a fragile pixel and
   stays within each open pixel's certified radius cannot flip the argmax.
2. **Direction = the measured pose-Jacobian, seg-flat**.  Within the open cone we
   want the directions that move PoseNet (high ``pose_jacobian_norm``) while the
   SegNet margin is FAR from flipping (high ``seg_margin``, low boundary slope).
   The atom support is the intersection ``open_cone ∧ pose_sensitive ∧ seg_flat``,
   ranked by the **seg-safe pose leverage** ``pose_jacobian_norm * seg_margin``
   (pose response per unit, weighted by distance-to-flip headroom).
3. **Amplitude within the cone radius**.  Each supported pixel is perturbed by a
   signed fraction of its OWN ``joint_cone_radius`` (``amplitude_fraction * radius_p``),
   so the per-pixel perturbation is certified seg-safe by construction.  The sign
   is chosen to push pose in the direction the pose-Jacobian indicates (the
   gradient sign), i.e. a descent step on ``d_pose`` against the GT pair.

THE FALSIFIABLE PER-ATOM CHECK (NO-FAKE, design verbatim)
=========================================================
*"Exact d_seg must come back UNCHANGED (the falsifiable per-action check) — any
seg movement disqualifies the action class instance."*  Every generated atom is
screened on the **exact local CPU-torch** SegNet/PoseNet (the real frozen
scorers, NEVER MPS): the candidate frame-1 must produce an argmax-IDENTICAL
SegNet output vs the GT (``d_seg`` == 0 exactly).  An atom whose ``d_seg`` moves
even by one pixel is REJECTED (``accepted=False``, reason ``seg_argmax_moved``).
The ``d_pose`` advisory delta is recorded for accepted atoms (negative = pose
improved).  This is real work measured on real scorers — never a marker.

COMPUTE-SUBSTRATE LAW (operator correction 2026-06-10)
======================================================
- **GENERATION + search**: MLX (the unified-memory leverage ranking / direction
  search saturates the M5 Max 128GB).  A numpy reference is the canonical
  portability oracle (Catalog #383 Backend pattern); the two agree to fp32 tol.
- **ADVISORY screening**: local CPU-torch exact frozen scorers (the per-atom
  d_seg/d_pose check above).  Tag ``[macOS-CPU advisory]``.
- **RANKING + admission**: the contest host ONLY (the R1 lesson; off-host
  ordering does not transfer at the 1e-5 pose scale).  This module emits advisory
  rows + a host-ranking packet; it NEVER claims an on-host accept.
- **MPS: NEVER** — not for ranking, not for generation, not for anything.  Any
  artifact with an MPS ancestor is contamination requiring rebuild.

POSTPROCESS (Class 5, #49, universal)
=====================================
Every generated atom's frame-1 is postprocessed through the #49 resize-null
preimage tier-1 (``apply_tier1_zero_weight_fill``): the certified zero-weight
camera pixels are filled at the entropy-optimal value at PROVEN zero scorer
change (``max|R x̃ - R x| == 0``).  Atoms are generated in the scorer-visible
384x512 projection, so the preimage proof is carried as a reference
(``preimage_tier1_applied`` + ``preimage_max_abs_residual``) — bytes shrink at
certified zero change.

Evidence grade: every quantity is ``[macOS-CPU advisory]`` (generation) /
``[macOS-MLX research-signal]`` (search) — local macOS is NOT 1:1 contest
hardware.  Non-promotable per Catalog #192/#341/#127/#323.  $0 local, NO cloud,
NO paid GPU, NO MPS.  The atoms PROPOSE; the contest-host exact replay with the
noise floor ratifies.

Cross-references
----------------
- ``tac.optimization.frame1_joint_safe_cone`` (#35; the OPEN cone this consumes)
- ``tac.optimization.evaluator_response_atlas`` (#36; the per-pair budget index)
- ``tac.optimization.resize_null_preimage`` (#49; the universal postprocessor)
- ``tac.optimization.lf_payload_rate_distortion`` (#46; THE LAW rate term)
- ``.omx/research/pr110pp_frame1_joint_methodology_v1_20260610.md`` (the design)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FRAME1_SEG_SAFE_POSE_ATOM_SCHEMA = "frame1_seg_safe_pose_atom.v1"

# Canonical Tier A non-promotable false-authority markers (Catalog #341/#323).
SEG_SAFE_ATOM_PROVENANCE: dict[str, Any] = {
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

# THE LAW constants (mirrors lf_payload_rate_distortion / resize_null_preimage).
_RATE_COEF: float = 25.0
_CONTEST_TOTAL_BYTES: int = 37_545_489
# Pose term is sqrt(10 * d_pose); the score is 100*d_seg + sqrt(10*d_pose) + rate.
_POSE_TEN: float = 10.0
_SEG_SCORE_WEIGHT: float = 100.0


class Frame1SegSafePoseAtomError(ValueError):
    """Raised on malformed Class-2 atom inputs / contract violations (fail-closed)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegSafePoseAtomConfig:
    """Configuration for Class-2 Seg-SAFE pose atom generation.

    ``open_cone_threshold`` is the minimum ``joint_cone_radius`` (in scorer input
    units [0,255]) for a pixel to be eligible support — fragile pixels (radius
    below the cone's ``fragile_radius_threshold``, default 0.5) are excluded BY
    CONSTRUCTION.  ``amplitude_fraction`` is the fraction of each pixel's own cone
    radius the atom perturbs (< 1 keeps it strictly inside the certified budget).
    ``support_top_fraction`` is the fraction of eligible pixels (ranked by seg-safe
    pose leverage) the atom's support covers.  ``min_support_pixels`` keeps the
    atom from degenerating to a no-op.  ``seg_flat_percentile`` selects pixels
    whose SegNet margin is in the top ``(1 - seg_flat_percentile)`` (far from
    flipping) so the atom stays seg-flat.
    """

    open_cone_threshold: float = 0.5
    amplitude_fraction: float = 0.5
    support_top_fraction: float = 0.05
    min_support_pixels: int = 64
    seg_flat_percentile: float = 0.5
    # Per-atom seg-stable tolerance: exact d_seg must be 0 (argmax-identical).
    # A nonzero floor exists only to absorb float-compare noise on the EXACT
    # argmax-equality measure (which is integer 0/1 per pixel -> mean == 0.0).
    seg_exact_tol: float = 0.0

    def __post_init__(self) -> None:
        if self.open_cone_threshold < 0.0:
            raise Frame1SegSafePoseAtomError("open_cone_threshold must be >= 0")
        if not 0.0 < self.amplitude_fraction <= 1.0:
            raise Frame1SegSafePoseAtomError("amplitude_fraction must be in (0, 1]")
        if not 0.0 < self.support_top_fraction <= 1.0:
            raise Frame1SegSafePoseAtomError("support_top_fraction must be in (0, 1]")
        if self.min_support_pixels < 1:
            raise Frame1SegSafePoseAtomError("min_support_pixels must be >= 1")
        if not 0.0 <= self.seg_flat_percentile < 1.0:
            raise Frame1SegSafePoseAtomError("seg_flat_percentile must be in [0, 1)")
        if self.seg_exact_tol < 0.0:
            raise Frame1SegSafePoseAtomError("seg_exact_tol must be >= 0")


# ---------------------------------------------------------------------------
# The atom: a frame-1 perturbation restricted to the open cone
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame1SegSafePoseAtom:
    """A frame-1 Seg-SAFE pose action atom (a perturbation field + provenance).

    The atom is a ``(H, W)`` per-pixel signed perturbation ``delta`` on the
    frame-1 scorer grid (384x512), with support restricted to the open cone.
    ``apply`` returns the perturbed pair (frame-0 untouched).  The atom carries
    its support id, the cone it came from, and the leverage ranking used.
    """

    schema: str
    pair_index: int
    target_frame: int  # always 1 for Class-2 (frame-1)
    support_or_cone_id: str
    # (H, W) float64 signed per-pixel perturbation on the frame-1 scorer grid.
    delta: np.ndarray
    # (H, W) bool support mask (where delta is non-zero).
    support_mask: np.ndarray
    n_support_pixels: int
    amplitude_fraction: float
    mean_abs_amplitude: float
    config: SegSafePoseAtomConfig
    provenance: dict[str, Any] = field(default_factory=lambda: dict(SEG_SAFE_ATOM_PROVENANCE))

    def apply(self, pair_btchwc_unit255: Any) -> Any:
        """Return the candidate pair with the atom applied to FRAME-1 only.

        ``pair`` is ``(1, 2, H, W, 3)`` in scorer units [0,255]; frame-0 is left
        untouched (Class-2 isolates frame-1).  The perturbation is broadcast over
        the 3 RGB channels (a luma-equivalent shift) and clamped to [0, 255].
        """

        import torch

        if not isinstance(pair_btchwc_unit255, torch.Tensor):
            pair = torch.as_tensor(np.asarray(pair_btchwc_unit255)).float()
        else:
            pair = pair_btchwc_unit255.float().clone()
        if pair.ndim != 5 or pair.shape[1] != 2 or pair.shape[-1] != 3:
            raise Frame1SegSafePoseAtomError(
                f"pair must be (1, 2, H, W, 3); got {tuple(pair.shape)}"
            )
        h, w = int(pair.shape[2]), int(pair.shape[3])
        delta_grid = _resize_map(self.delta, h, w)
        cand = pair.clone()
        d = torch.from_numpy(delta_grid[..., None]).float()  # (H, W, 1)
        cand[0, 1] = (pair[0, 1] + d).clamp(0.0, 255.0)
        return cand


@dataclass(frozen=True)
class SegSafePoseAtomRow:
    """The advisory row a Class-2 atom emits after exact-scorer screening.

    Mirrors the design row schema (Class 2 fields).  ``d_seg_delta`` must be 0
    for an accepted atom (argmax-identical); ``d_pose_delta`` is the advisory pose
    improvement (negative = better).  ``score_delta_advisory`` is the advisory ΔS
    from THE LAW (since accepted atoms have Δd_seg=0, ΔS = pose term + rate term).
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
    provenance: dict[str, Any] = field(default_factory=lambda: dict(SEG_SAFE_ATOM_PROVENANCE))

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
# Cone-map loading (consumes the #35 cone .npz / atlas index)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConeFields:
    """The per-pair cone fields a Class-2 atom needs (from a #35 cone .npz)."""

    joint_cone_radius: np.ndarray  # (H, W)
    seg_margin: np.ndarray  # (H, W)
    pose_jacobian_norm: np.ndarray  # (H, W)
    fragile_cone_mask: np.ndarray  # (H, W) bool
    seg_argmax_class: np.ndarray  # (H, W) int

    @classmethod
    def from_npz(cls, path: str) -> ConeFields:
        """Load the cone fields from a #35 cone-map ``.npz`` (fail-closed)."""

        d = np.load(path)
        required = (
            "joint_cone_radius",
            "seg_margin",
            "pose_jacobian_norm",
            "fragile_cone_mask",
            "seg_argmax_class",
        )
        for k in required:
            if k not in d:
                raise Frame1SegSafePoseAtomError(
                    f"cone .npz {path} missing key {k!r}; has {list(d.keys())}"
                )
        return cls(
            joint_cone_radius=np.asarray(d["joint_cone_radius"], dtype=np.float64),
            seg_margin=np.asarray(d["seg_margin"], dtype=np.float64),
            pose_jacobian_norm=np.asarray(d["pose_jacobian_norm"], dtype=np.float64),
            fragile_cone_mask=np.asarray(d["fragile_cone_mask"], dtype=bool),
            seg_argmax_class=np.asarray(d["seg_argmax_class"], dtype=np.int64),
        )

    @classmethod
    def from_cone(cls, cone: Any) -> ConeFields:
        """Build from an in-memory :class:`Frame1JointSafeCone`."""

        return cls(
            joint_cone_radius=np.asarray(cone.joint_cone_radius, dtype=np.float64),
            seg_margin=np.asarray(cone.seg_margin, dtype=np.float64),
            pose_jacobian_norm=np.asarray(cone.pose_jacobian_norm, dtype=np.float64),
            fragile_cone_mask=np.asarray(cone.fragile_cone_mask, dtype=bool),
            seg_argmax_class=np.asarray(cone.seg_argmax_class, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# Seg-safe pose leverage: MLX (unified mem) + numpy reference (Catalog #383)
# ---------------------------------------------------------------------------
SEG_SAFE_LEVERAGE_FP32_ATOL = 1e-3


def seg_safe_pose_leverage_numpy(fields: ConeFields) -> np.ndarray:
    """Canonical numpy reference: per-pixel seg-safe pose leverage.

    The leverage is ``pose_jacobian_norm * seg_margin`` over the OPEN cone, with
    fragile pixels zeroed: a pixel scores high when PoseNet is sensitive to it
    (high ``pose_jacobian_norm`` = the action moves pose) AND the SegNet margin is
    large (high ``seg_margin`` = far from flipping = seg-flat headroom).  Fragile
    pixels (``fragile_cone_mask``) and below-threshold radius pixels score 0 BY
    CONSTRUCTION — they are never atom support.
    """

    pj = fields.pose_jacobian_norm
    sm = fields.seg_margin
    lev = pj * sm
    lev = np.where(fields.fragile_cone_mask, 0.0, lev)
    return lev.astype(np.float64)


def seg_safe_pose_leverage_mlx(fields: ConeFields) -> np.ndarray:
    """MLX (Apple unified memory) seg-safe pose leverage; matches the numpy
    reference to fp32 tolerance (Catalog #383 Backend contract).  Raises
    ``ImportError`` if MLX is unavailable (caller falls back to numpy)."""

    import mlx.core as mx

    pj = mx.array(fields.pose_jacobian_norm.astype(np.float32))
    sm = mx.array(fields.seg_margin.astype(np.float32))
    frag = mx.array(fields.fragile_cone_mask.astype(np.float32))
    lev = pj * sm * (1.0 - frag)
    mx.eval(lev)
    return np.asarray(lev).astype(np.float64)


def seg_safe_pose_leverage(fields: ConeFields, *, prefer_mlx: bool = True) -> np.ndarray:
    """Seg-safe pose leverage with MLX-first + numpy-portable fallback."""

    if prefer_mlx:
        try:
            return seg_safe_pose_leverage_mlx(fields)
        except ImportError:
            pass
    return seg_safe_pose_leverage_numpy(fields)


def _resize_map(pixel_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear-resize a ``(H, W)`` map to ``(target_h, target_w)`` (numpy-portable)."""

    src = np.asarray(pixel_map, dtype=np.float64)
    if src.shape == (target_h, target_w):
        return src
    sh, sw = src.shape
    # numpy-portable separable bilinear via index interpolation (align_corners=False).
    ys = (np.arange(target_h) + 0.5) * sh / target_h - 0.5
    xs = (np.arange(target_w) + 0.5) * sw / target_w - 0.5
    ys = np.clip(ys, 0, sh - 1)
    xs = np.clip(xs, 0, sw - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


# ---------------------------------------------------------------------------
# Atom generation: restrict support to the open cone, direction from pose-Jac
# ---------------------------------------------------------------------------
def generate_seg_safe_pose_atom(
    *,
    pair_index: int,
    fields: ConeFields,
    pose_grad_sign: np.ndarray | None = None,
    config: SegSafePoseAtomConfig | None = None,
    prefer_mlx: bool = True,
) -> Frame1SegSafePoseAtom:
    """Generate ONE Class-2 Seg-SAFE pose atom from a pair's cone fields.

    1. eligible = (joint_cone_radius >= open_cone_threshold) AND NOT fragile.
       (Fragile 51.4% excluded BY CONSTRUCTION.)
    2. seg-flat = seg_margin >= percentile(seg_margin[eligible], seg_flat_percentile).
    3. support = top ``support_top_fraction`` of (eligible ∧ seg-flat) by leverage
       ``pose_jacobian_norm * seg_margin``.
    4. amplitude_p = amplitude_fraction * joint_cone_radius_p (strictly inside the
       certified per-pixel budget).
    5. sign_p = pose_grad_sign_p (push pose toward improvement); default = +1 (a
       caller with the measured pose gradient passes its sign so the atom is a
       pose-descent step; without it the atom is still seg-safe, the screening
       picks the better of +/- by trying both — see ``generate_signed_atoms``).

    The returned atom carries a non-empty support (or raises) — a degenerate
    empty-support atom is a FAKE atom per the NO-FAKE discipline.
    """

    cfg = config or SegSafePoseAtomConfig()
    radius = fields.joint_cone_radius
    margin = fields.seg_margin
    fragile = fields.fragile_cone_mask
    h, w = radius.shape

    eligible = (radius >= float(cfg.open_cone_threshold)) & (~fragile)
    if not eligible.any():
        raise Frame1SegSafePoseAtomError(
            f"pair {pair_index}: no eligible open-cone pixels at threshold "
            f"{cfg.open_cone_threshold} (the whole frame is fragile?)"
        )

    # seg-flat gate: keep only pixels whose margin is far from flipping.
    elig_margins = margin[eligible]
    flat_cut = float(np.quantile(elig_margins, cfg.seg_flat_percentile))
    seg_flat = margin >= flat_cut
    candidate = eligible & seg_flat
    if not candidate.any():
        candidate = eligible  # degenerate fallback: all eligible are seg-flat enough

    leverage = seg_safe_pose_leverage(fields, prefer_mlx=prefer_mlx)
    lev_masked = np.where(candidate, leverage, -np.inf)

    n_candidate = int(candidate.sum())
    n_support = max(
        int(cfg.min_support_pixels),
        round(float(cfg.support_top_fraction) * n_candidate),
    )
    n_support = min(n_support, n_candidate)

    flat = lev_masked.reshape(-1)
    if n_support >= flat.size:
        top_idx = np.where(np.isfinite(flat))[0]
    else:
        # top-n by leverage (argpartition for O(n)); ties resolved deterministically.
        part = np.argpartition(-flat, n_support - 1)[:n_support]
        # keep only finite (candidate) picks.
        part = part[np.isfinite(flat[part])]
        top_idx = part
    support_mask = np.zeros(h * w, dtype=bool)
    support_mask[top_idx] = True
    support_mask = support_mask.reshape(h, w)

    n_actual = int(support_mask.sum())
    if n_actual == 0:
        raise Frame1SegSafePoseAtomError(
            f"pair {pair_index}: atom support is empty after ranking (FAKE atom)"
        )

    # amplitude = fraction of each pixel's own cone radius (certified seg-safe).
    amp = float(cfg.amplitude_fraction) * radius
    if pose_grad_sign is None:
        sign = np.ones((h, w), dtype=np.float64)
    else:
        sg = _resize_map(np.asarray(pose_grad_sign, dtype=np.float64), h, w)
        sign = np.sign(sg)
        sign[sign == 0.0] = 1.0

    delta = np.zeros((h, w), dtype=np.float64)
    delta[support_mask] = (sign * amp)[support_mask]

    mean_abs = float(np.abs(delta[support_mask]).mean()) if n_actual else 0.0
    support_id = f"open_cone_pair{pair_index}_n{n_actual}_amp{cfg.amplitude_fraction:g}"

    return Frame1SegSafePoseAtom(
        schema=FRAME1_SEG_SAFE_POSE_ATOM_SCHEMA,
        pair_index=int(pair_index),
        target_frame=1,
        support_or_cone_id=support_id,
        delta=delta,
        support_mask=support_mask,
        n_support_pixels=n_actual,
        amplitude_fraction=float(cfg.amplitude_fraction),
        mean_abs_amplitude=mean_abs,
        config=cfg,
        provenance=dict(SEG_SAFE_ATOM_PROVENANCE),
    )


# ---------------------------------------------------------------------------
# Exact CPU-torch screening: the falsifiable per-atom seg-unchanged check
# ---------------------------------------------------------------------------
def _measure_exact_distortion(dn: Any, gt_pair: Any, cand_pair: Any) -> tuple[float, float]:
    """Return ``(d_seg, d_pose)`` of a candidate pair vs the GT pair via the real
    upstream DistortionNet (exact CPU-torch, NEVER MPS)."""

    import torch

    with torch.inference_mode():
        d_pose, d_seg = dn.compute_distortion(gt_pair.float(), cand_pair.float())
    return float(d_seg.mean()), float(d_pose.mean())


def screen_atom_exact(
    *,
    atom: Frame1SegSafePoseAtom,
    distortion_net: Any,
    gt_pair_btchwc_unit255: Any,
    config: SegSafePoseAtomConfig | None = None,
    selector_bits_est: float = 0.0,
    preimage_tier1_applied: bool = False,
    preimage_max_abs_residual: float = 0.0,
    preimage_bytes_freed: int = 0,
) -> SegSafePoseAtomRow:
    """Screen ONE atom on the exact CPU-torch scorers — the falsifiable check.

    THE LAW per design: an atom is accepted iff its exact ``d_seg`` is UNCHANGED
    (argmax-identical => ``d_seg`` == 0).  Any seg movement disqualifies the atom
    (``accepted=False``, reason ``seg_argmax_moved``).  The ``d_pose`` advisory
    delta is recorded; ``score_delta_advisory`` is the ΔS (since accepted atoms
    have Δd_seg=0, ΔS = sqrt(10*(base+Δ)) - sqrt(10*base) + rate term).
    """

    cfg = config or atom.config
    base_seg, base_pose = _measure_exact_distortion(
        distortion_net, gt_pair_btchwc_unit255, gt_pair_btchwc_unit255
    )
    cand = atom.apply(gt_pair_btchwc_unit255)
    cand_seg, cand_pose = _measure_exact_distortion(
        distortion_net, gt_pair_btchwc_unit255, cand
    )

    d_seg_delta = cand_seg - base_seg
    d_pose_delta = cand_pose - base_pose

    seg_unchanged = abs(d_seg_delta) <= float(cfg.seg_exact_tol)

    # ΔS advisory: 100*Δd_seg + (sqrt(10*pose_cand) - sqrt(10*pose_base)) + rate.
    pose_term_delta = float(
        np.sqrt(_POSE_TEN * max(cand_pose, 0.0)) - np.sqrt(_POSE_TEN * max(base_pose, 0.0))
    )
    seg_term_delta = _SEG_SCORE_WEIGHT * d_seg_delta
    # selector bytes are a rate cost; preimage freed bytes are a rate gain.
    selector_bytes = float(selector_bits_est) / 8.0
    net_bytes = selector_bytes - float(preimage_bytes_freed)
    rate_delta = _RATE_COEF * net_bytes / float(_CONTEST_TOTAL_BYTES)
    score_delta = seg_term_delta + pose_term_delta + rate_delta

    if not seg_unchanged:
        accepted = False
        reason = "seg_argmax_moved"
    elif d_pose_delta >= 0.0:
        # seg stayed safe but pose did not improve -> not a useful Class-2 atom.
        accepted = False
        reason = "pose_not_improved"
    else:
        accepted = True
        reason = ""

    # value per byte: advisory ΔS improvement per encoded byte (for ranking the
    # menu; never a pair-count). Negative score_delta = improvement.
    byte_cost = max(selector_bytes, 1e-9)
    value_per_byte = float(-score_delta / byte_cost) if byte_cost > 0 else 0.0

    return SegSafePoseAtomRow(
        schema=FRAME1_SEG_SAFE_POSE_ATOM_SCHEMA,
        pair_index=atom.pair_index,
        target_frame=atom.target_frame,
        support_or_cone_id=atom.support_or_cone_id,
        d_seg_delta=d_seg_delta,
        d_pose_delta=d_pose_delta,
        score_delta_advisory=score_delta,
        selector_bits_est=float(selector_bits_est),
        value_per_byte=value_per_byte,
        n_support_pixels=atom.n_support_pixels,
        preimage_tier1_applied=bool(preimage_tier1_applied),
        preimage_max_abs_residual=float(preimage_max_abs_residual),
        preimage_bytes_freed=int(preimage_bytes_freed),
        authority_host="macos_cpu_advisory",
        accepted=accepted,
        rejected_reason=reason,
        provenance=dict(SEG_SAFE_ATOM_PROVENANCE),
    )


def generate_signed_atoms(
    *,
    pair_index: int,
    fields: ConeFields,
    config: SegSafePoseAtomConfig | None = None,
    prefer_mlx: bool = True,
) -> tuple[Frame1SegSafePoseAtom, Frame1SegSafePoseAtom]:
    """Generate the +sign and -sign variants of the same support atom.

    Without the measured pose gradient sign per pixel, the seg-safe direction is
    known (the support) but the pose-improving SIGN is not.  The screening tries
    BOTH signs on the exact scorer and keeps whichever improves pose (both stay
    seg-safe by construction since they share the certified-radius amplitude).
    """

    cfg = config or SegSafePoseAtomConfig()
    base = generate_seg_safe_pose_atom(
        pair_index=pair_index, fields=fields, pose_grad_sign=None,
        config=cfg, prefer_mlx=prefer_mlx,
    )
    neg = Frame1SegSafePoseAtom(
        schema=base.schema,
        pair_index=base.pair_index,
        target_frame=base.target_frame,
        support_or_cone_id=base.support_or_cone_id + "_neg",
        delta=-base.delta,
        support_mask=base.support_mask,
        n_support_pixels=base.n_support_pixels,
        amplitude_fraction=base.amplitude_fraction,
        mean_abs_amplitude=base.mean_abs_amplitude,
        config=base.config,
        provenance=dict(base.provenance),
    )
    return base, neg


__all__ = [
    "FRAME1_SEG_SAFE_POSE_ATOM_SCHEMA",
    "SEG_SAFE_ATOM_PROVENANCE",
    "SEG_SAFE_LEVERAGE_FP32_ATOL",
    "ConeFields",
    "Frame1SegSafePoseAtom",
    "Frame1SegSafePoseAtomError",
    "SegSafePoseAtomConfig",
    "SegSafePoseAtomRow",
    "generate_seg_safe_pose_atom",
    "generate_signed_atoms",
    "screen_atom_exact",
    "seg_safe_pose_leverage",
    "seg_safe_pose_leverage_mlx",
    "seg_safe_pose_leverage_numpy",
]
