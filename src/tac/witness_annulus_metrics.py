# SPDX-License-Identifier: MIT
"""Pure, testable metric math for the codim-1 boundary ANNULUS convergence telemetry.

This module holds ONLY numpy/CPU metric math (no rendering, no torch, no MLX). The
CLI ``tools/witness_annulus_convergence.py`` is a thin wrapper that renders each
preserved checkpoint through the REAL render-through-R + REAL CPU-torch SegNet path
(reused from ``tools/witness_per_stage_annulus_attribution.py`` -- NOT reimplemented)
and then feeds the resulting maps here.

WHAT THE ANNULUS IS
-------------------
The annulus = the boundary band where the GT SegNet margin (top1-top2 logit gap) is
SMALL. d_seg flips concentrate here (MEASURED: the bottom ~2.26% margin pixels hold
~89% of d_seg). "Annulus convergence" = does this boundary band's flip fraction tighten
as training progresses, WITHOUT leaving structural (interior) flips behind.

Two annulus definitions are supported (both keep the SAME fixed GT-margin field across
all checkpoints so the series is apples-to-apples):
  * fixed-threshold  : annulus = {px : |gt_margin| < band}
  * bottom-k fraction: annulus = the smallest-``k``-fraction of gt_margin pixels

KEY OBSERVABILITY WIN
---------------------
``interior_flip_frac`` = d_seg restricted to NON-annulus pixels. A converging run drives
this -> 0; a nonzero interior flip fraction is a STRUCTURAL miss (a whole region on the
wrong side of the argmax), NOT boundary jitter. Separating the two is the point.

AUTHORITY: numpy-fp32 is the verdict authority. Every consumer row is tagged
``[macOS-numpy advisory . NON-PROMOTABLE]``. The frontier pointer (0.19110) is UNMOVED;
this is advisory telemetry (SENSE layer for the #247 costate controller, task #333).

CANONICAL comma10k class order (CLAUDE.md NON-NEGOTIABLE -- never luma-sort):
  0=Road  1=Lane  2=Undrivable(incl sky)  3=Movable(cars)  4=MyCar(ego hood)
"""
from __future__ import annotations

import numpy as np

# CANONICAL comma10k class order (CLAUDE.md NON-NEGOTIABLE).
CLASS_NAMES: dict[int, str] = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
N_CLASSES = 5
ADVISORY = "[macOS-numpy advisory . NON-PROMOTABLE]"


# ---------------------------------------------------------------------------
# annulus masks (defined on the FIXED GT margin field -> constant across checkpoints).
# ---------------------------------------------------------------------------
def annulus_mask_threshold(gt_margin: np.ndarray, band: float) -> np.ndarray:
    """Annulus = pixels whose |GT margin| < ``band``.

    ``gt_margin`` is the GT SegNet top1-top2 gap (>=0), so this selects the low-margin
    boundary band. Returns a bool array with the same shape as ``gt_margin``.
    """
    gm = np.abs(np.asarray(gt_margin, np.float32))
    return gm < float(band)


def annulus_mask_bottom_k(gt_margin: np.ndarray, k: float) -> np.ndarray:
    """Annulus = the smallest-``k``-fraction of |GT margin| pixels (0 < k <= 1).

    The threshold is the ``k`` quantile of |gt_margin| over ALL pixels; pixels with
    |gt_margin| <= that quantile are the annulus. Returns a bool array.
    """
    if not (0.0 < float(k) <= 1.0):
        raise ValueError(f"bottom-k must be in (0, 1]; got {k!r}")
    gm = np.abs(np.asarray(gt_margin, np.float32))
    thr = float(np.quantile(gm, float(k)))
    return gm <= thr


# ---------------------------------------------------------------------------
# flip maps + restricted flip fractions.
# ---------------------------------------------------------------------------
def flip_map(witness_argmax: np.ndarray, gt_argmax: np.ndarray) -> np.ndarray:
    """Per-pixel d_seg flip: witness realized argmax != GT argmax. Bool array."""
    return np.asarray(witness_argmax).astype(np.int64) != np.asarray(gt_argmax).astype(np.int64)


def restricted_flip_frac(flip: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of pixels IN ``mask`` that are flips (flip rate within the region).

    Returns 0.0 when the mask is empty (no pixels to average over).
    """
    m = np.asarray(mask, bool)
    n = int(m.sum())
    if n == 0:
        return 0.0
    return float((np.asarray(flip, bool) & m).sum()) / n


def flip_mass_share(flip: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of ALL flips that fall inside ``mask`` (how much of d_seg lives here).

    Returns 0.0 when there are no flips at all.
    """
    fl = np.asarray(flip, bool)
    total = int(fl.sum())
    if total == 0:
        return 0.0
    return float((fl & np.asarray(mask, bool)).sum()) / total


def per_class_annulus_flip_frac(
    flip: np.ndarray, annulus: np.ndarray, gt_argmax: np.ndarray, n_classes: int = N_CLASSES
) -> dict[int, float]:
    """Flip rate within the annulus, split by GT class.

    For each class c: (# flips in {annulus AND gt==c}) / (# pixels in {annulus AND gt==c}).
    Classes with no annulus pixels report 0.0. This answers "which class boundary is
    converging vs stuck".
    """
    fl = np.asarray(flip, bool)
    an = np.asarray(annulus, bool)
    g = np.asarray(gt_argmax).astype(np.int64)
    out: dict[int, float] = {}
    for c in range(n_classes):
        region = an & (g == c)
        n = int(region.sum())
        out[c] = (float((fl & region).sum()) / n) if n else 0.0
    return out


# ---------------------------------------------------------------------------
# realized (witness) margin percentiles within the annulus.
# ---------------------------------------------------------------------------
def margin_percentiles(
    witness_margin: np.ndarray, mask: np.ndarray, ps: tuple[float, ...] = (10.0, 50.0)
) -> dict[str, float]:
    """Percentiles of the realized WITNESS margin-toward-GT within ``mask``.

    ``witness_margin`` = logit[GT] - max_{k!=GT} logit[k] of the witness readout: >0 means
    the witness argmax already picks GT (correct); rising p10/p50 across checkpoints means
    the witness is pushing its OWN margin AWAY from the flip threshold (real convergence,
    not cosmetic argmax luck). Empty mask -> NaN for every requested percentile.
    """
    m = np.asarray(mask, bool)
    vals = np.asarray(witness_margin, np.float32)[m]
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(vals, p)) if vals.size else float("nan")
    return out


# ---------------------------------------------------------------------------
# Gibbs / ringing proxy: high-frequency energy of the witness margin in the boundary
# ring vs the interior. Ratio > 1 <=> the boundary oscillates more than the interior
# (a ringing/overshoot signature). DOCUMENTED PROXY -- not a calibrated Gibbs amplitude.
# ---------------------------------------------------------------------------
def _laplacian_2d(field: np.ndarray) -> np.ndarray:
    """4-neighbour discrete Laplacian on a single (h,w) grid (edges under-count, fine for a proxy)."""
    f = np.asarray(field, np.float32)
    lap = -4.0 * f
    lap[1:, :] += f[:-1, :]
    lap[:-1, :] += f[1:, :]
    lap[:, 1:] += f[:, :-1]
    lap[:, :-1] += f[:, 1:]
    return lap


def gibbs_ring_proxy(witness_margin: np.ndarray, annulus: np.ndarray) -> float:
    """Ratio of high-freq margin energy in the annulus ring vs the interior.

    Per pair: energy = mean(Laplacian(witness_margin)^2). We average the squared Laplacian
    over annulus pixels and over interior pixels (pooled across all pairs), then return
    ``ring_energy / (interior_energy + eps)``. >1 => more high-frequency oscillation on the
    boundary than the interior (ringing proxy). Returns 0.0 if either region is empty.

    ``witness_margin`` / ``annulus`` are (V,h,w) (batched) or (h,w) (single).
    """
    wm = np.asarray(witness_margin, np.float32)
    an = np.asarray(annulus, bool)
    if wm.ndim == 2:
        wm = wm[None]
        an = an[None]
    ring_sq: list[np.ndarray] = []
    int_sq: list[np.ndarray] = []
    for t in range(wm.shape[0]):
        lap = _laplacian_2d(wm[t])
        sq = lap * lap
        ring_sq.append(sq[an[t]])
        int_sq.append(sq[~an[t]])
    ring = np.concatenate(ring_sq) if ring_sq else np.empty(0, np.float32)
    inter = np.concatenate(int_sq) if int_sq else np.empty(0, np.float32)
    if ring.size == 0 or inter.size == 0:
        return 0.0
    return float(ring.mean()) / (float(inter.mean()) + 1e-12)


# ---------------------------------------------------------------------------
# one-checkpoint metric bundle.
# ---------------------------------------------------------------------------
def checkpoint_metrics(
    witness_argmax: np.ndarray,
    witness_margin: np.ndarray,
    gt_argmax: np.ndarray,
    gt_margin: np.ndarray,
    band: float = 2.0,
    bottom_k: float = 0.05,
    n_classes: int = N_CLASSES,
) -> dict:
    """All per-checkpoint annulus metrics for one rendered checkpoint.

    Arrays are (V,h,w) (or (h,w)); ``gt_margin`` is the FIXED GT top1-top2 field (same for
    every checkpoint), ``witness_margin`` is the per-checkpoint realized margin-toward-GT.
    Returns a machine-readable dict (JSON-safe floats/ints).
    """
    flip = flip_map(witness_argmax, gt_argmax)
    total_px = int(flip.size)
    overall_d_seg = float(flip.mean()) if total_px else 0.0

    an_thr = annulus_mask_threshold(gt_margin, band)
    an_bk = annulus_mask_bottom_k(gt_margin, bottom_k)

    def _block(annulus: np.ndarray) -> dict:
        interior = ~annulus
        return {
            "annulus_area_frac": (float(annulus.mean()) if annulus.size else 0.0),
            "annulus_flip_frac": restricted_flip_frac(flip, annulus),
            "interior_flip_frac": restricted_flip_frac(flip, interior),
            "annulus_flip_mass_share": flip_mass_share(flip, annulus),
            "annulus_margin": margin_percentiles(witness_margin, annulus, ps=(10.0, 50.0)),
            "per_class_annulus_flip_frac": per_class_annulus_flip_frac(flip, annulus, gt_argmax, n_classes),
            "gibbs_ring_proxy": gibbs_ring_proxy(witness_margin, annulus),
        }

    return {
        "overall_d_seg": overall_d_seg,
        "total_px": total_px,
        "n_flips": int(flip.sum()),
        "band": float(band),
        "bottom_k": float(bottom_k),
        "threshold": _block(an_thr),
        "bottom_k_def": _block(an_bk),
    }


# ---------------------------------------------------------------------------
# cross-checkpoint convergence rate.
# ---------------------------------------------------------------------------
def convergence_rate(epochs: list[float], values: list[float]) -> float:
    """d(value)/d(epoch): least-squares slope over the (epoch,value) series.

    Sign convention is natural: for a flip-fraction series a NEGATIVE slope = converging
    (flips falling per epoch). NaN values are dropped. With <2 finite points -> 0.0; if all
    epochs are identical -> 0.0 (undefined slope).
    """
    e = np.asarray(epochs, np.float64)
    v = np.asarray(values, np.float64)
    good = np.isfinite(e) & np.isfinite(v)
    e, v = e[good], v[good]
    if e.size < 2 or float(np.ptp(e)) == 0.0:
        return 0.0
    # slope of the OLS line v ~ a + b*e.
    b = float(np.polyfit(e, v, 1)[0])
    return b


def _series_pick(series: list[dict], defn: str, key: str) -> tuple[list[float], list[float]]:
    epochs = [float(s["epoch"]) for s in series]
    vals = [float(s["metrics"][defn][key]) for s in series]
    return epochs, vals


def convergence_series(
    series: list[dict], defn: str = "threshold", n_classes: int = N_CLASSES
) -> dict:
    """Cross-checkpoint convergence rates from a list of per-checkpoint records.

    Each record: ``{"name": str, "epoch": float, "metrics": <checkpoint_metrics output>}``.
    The list is sorted by epoch internally. ``defn`` selects the annulus definition
    ("threshold" or "bottom_k_def"). Returns overall + per-class dV/dEpoch for the
    annulus flip-frac and interior flip-frac, plus the raw sorted (epoch,value) curves for
    plotting. NEGATIVE annulus/interior slope = converging.
    """
    if defn not in ("threshold", "bottom_k_def"):
        raise ValueError(f"defn must be 'threshold' or 'bottom_k_def'; got {defn!r}")
    s = sorted(series, key=lambda r: float(r["epoch"]))
    epochs = [float(r["epoch"]) for r in s]

    ann_e, ann_v = _series_pick(s, defn, "annulus_flip_frac")
    int_e, int_v = _series_pick(s, defn, "interior_flip_frac")
    d_seg_v = [float(r["metrics"]["overall_d_seg"]) for r in s]

    per_class_rate: dict[int, float] = {}
    per_class_curve: dict[int, list[float]] = {}
    for c in range(n_classes):
        cv = [float(r["metrics"][defn]["per_class_annulus_flip_frac"][c]
                    if c in r["metrics"][defn]["per_class_annulus_flip_frac"]
                    else r["metrics"][defn]["per_class_annulus_flip_frac"][str(c)]) for r in s]
        per_class_rate[c] = convergence_rate(epochs, cv)
        per_class_curve[c] = cv

    return {
        "defn": defn,
        "epochs": epochs,
        "overall_d_seg_curve": d_seg_v,
        "overall_d_seg_rate": convergence_rate(epochs, d_seg_v),
        "annulus_flip_frac_curve": ann_v,
        "annulus_flip_frac_rate": convergence_rate(ann_e, ann_v),
        "interior_flip_frac_curve": int_v,
        "interior_flip_frac_rate": convergence_rate(int_e, int_v),
        "per_class_annulus_flip_frac_rate": per_class_rate,
        "per_class_annulus_flip_frac_curve": per_class_curve,
        "margin_p10_curve": [float(r["metrics"][defn]["annulus_margin"]["p10"]) for r in s],
        "margin_p50_curve": [float(r["metrics"][defn]["annulus_margin"]["p50"]) for r in s],
        "annulus_area_frac_curve": [float(r["metrics"][defn]["annulus_area_frac"]) for r in s],
    }
