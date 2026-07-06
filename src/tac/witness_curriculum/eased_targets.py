"""Difficulty-homotopy eased-target operators (task #323).

See the package docstring for the geometry + the adversarial findings. These are pure
numpy/scipy, deterministic, per-frame, n600-ready. Each returns an eased integer label
map in which a rare class has been made more winnable; the caller anneals the easing
parameter → 0 (which returns the true target) under the per-class-λ costate gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# CE-birth thresholds (memory: rare-class won't birth below ~1.5% area).
_BIRTH_AREA = 0.015
_SEG_MIN_PX = 50          # a "coherent segment" is a connected component ≥ this
_SEG_COH_FRAC = 0.60      # birthable-by-coherence: ≥60% of class mass in coherent segments


@dataclass(frozen=True)
class Birthability:
    """Whether a class mask is birthable, and by which criterion.

    A class births under CE when it has enough AREA and its mass sits in COHERENT
    segments (either one big blob, or several coherent curve-segments — both are cheap
    for a coord-INR). We report both so blob-classes and curve-classes use the right test.
    """
    area_frac: float
    largest_cc_frac: float        # blob criterion: mass in the single largest component
    coherent_seg_frac: float      # curve criterion: mass in components ≥ _SEG_MIN_PX
    n_components: int
    birthable_blob: bool
    birthable_curve: bool

    @property
    def birthable(self) -> bool:
        return self.birthable_blob or self.birthable_curve


def birthability(mask: np.ndarray) -> Birthability:
    """Classify a boolean class mask (blob vs curve birthability)."""
    tot = int(mask.sum())
    if tot == 0:
        return Birthability(0.0, 0.0, 0.0, 0, False, False)
    lab, n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)
    largest = float(sizes.max()) / tot if n else 0.0
    coherent = float(sizes[sizes >= _SEG_MIN_PX].sum()) / tot if n else 0.0
    area = float(mask.mean())
    return Birthability(
        area_frac=area, largest_cc_frac=largest, coherent_seg_frac=coherent,
        n_components=int(n),
        birthable_blob=(area >= _BIRTH_AREA and largest >= 0.35),
        birthable_curve=(area >= _BIRTH_AREA and coherent >= _SEG_COH_FRAC),
    )


def sdf_dilation_eased(labels: np.ndarray, cls: int, radius: int) -> np.ndarray:
    """BLOB homotopy: class ``cls`` claims any pixel within ``radius`` of a true-cls pixel.

    T_r = {x : dist(x, class) ≤ r}. Continuity+transfer proven (SDF 1-Lipschitz ⇒
    Hausdorff-continuous nested filtration ⇒ bounded step-debt; = forward-Euler of
    ∂ₜu=+|∇u|). radius=0 returns the true target. MEASURED GO for movable (class 3).
    """
    if radius <= 0:
        return labels
    m = ndimage.binary_dilation(labels == cls, iterations=int(radius))
    out = labels.copy()
    out[m] = cls
    return out


def oriented_width_eased(labels: np.ndarray, cls: int, width: int,
                         min_component: int = 4, *,
                         tangent_mode: str = "vp",
                         vanishing_point: tuple[float, float] | None = None) -> np.ndarray:
    """CURVE homotopy (SECONDARY birth-gradient widener; manifold-preserving).

    Widens each coherent curve-segment of ``cls`` ALONG its lane tangent so every eased
    target stays a (thin) curve — never a blob — keeping it on the class's ~8-dim curve
    manifold (isotropic dilation would leave it; measured NO-GO). This does NOT bridge
    inter-dash gaps (adversarially confirmed: the tangent is too local); its validated
    role is a low-debt, on-manifold widening of the already-coherent dash segments to
    give CE more birth-gradient. width=0 returns the true target. The PRIMARY lane lever
    is the per-class-λ loss re-weight, not this.

    ``tangent_mode``:
      "vp" (default, openpilot-aligned) — the direction is the analytic camera geometry:
        a lane marking lies along the forward road direction, whose image tangent points
        at the VANISHING POINT, so ``tangent = normalize(VP − centroid)`` (VP=(256,174)
        in scorer 512×384 coords = openpilot ``_neo_config`` road cam rescaled; task #325
        openpilot mine). Weights-free, rule-118-clean (a generic deterministic prior, NOT
        a learned artifact — supercombo is never imported). MEASURED n600: agrees with
        the legacy shape-PCA within 15° for 83.4% of components (median 3.5°) and FIXES
        the noisy 7.5% tail (>30°) where a short-dash shape-PCA picks a spurious axis.
        Degenerate fallback to PCA only when the centroid sits on the VP (dir undefined).
      "pca" (legacy) — the component's own principal axis (kept for the clean A/B).
    """
    if width <= 0:
        return labels
    if vanishing_point is None:
        from tac.camera import VANISHING_POINT
        vanishing_point = (float(VANISHING_POINT[0]), float(VANISHING_POINT[1]))
    vpx, vpy = float(vanishing_point[0]), float(vanishing_point[1])
    mask = labels == cls
    lab, n = ndimage.label(mask)
    out = labels.copy()
    W = int(width)
    for k in range(1, n + 1):
        comp = lab == k
        ys, xs = np.where(comp)
        if len(xs) < min_component:
            grown = ndimage.binary_dilation(comp, iterations=1)   # too small to orient
        else:
            cx, cy = float(xs.mean()), float(ys.mean())
            ax = None
            if tangent_mode == "vp":
                v = np.array([vpx - cx, vpy - cy], dtype=np.float64)
                nrm = float(np.hypot(v[0], v[1]))
                if nrm > 1.0:                         # well-defined away from the VP
                    ax = v / nrm
            if ax is None:                             # "pca" mode, or VP-degenerate
                pts = np.c_[xs, ys].astype(np.float64)
                pts -= pts.mean(axis=0)
                _, vecs = np.linalg.eigh(pts.T @ pts)
                ax = vecs[:, -1]                       # principal (x, y) dir
            t = np.arange(-W, W + 1)
            lx = np.round(t * ax[0]).astype(int)
            ly = np.round(t * ax[1]).astype(int)
            se = np.zeros((2 * W + 1, 2 * W + 1), dtype=bool)
            se[ly + W, lx + W] = True
            grown = ndimage.binary_dilation(comp, structure=se)
        out[grown] = cls
    return out
