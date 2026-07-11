# SPDX-License-Identifier: MIT
"""Scorer-geometry SENSOR — the trajectory-INDEPENDENT λ source (task #426, operator
extension 2026-07-11: "It can also perhaps be trained on seg and pose or to understand
it or on the data and regimes used to train the official upstream seg and pose").

λ = ∂S/∂x and S is ENTIRELY a function of the two FROZEN scorers. The organ's biggest
measured limit is n=1 trajectory; the scorer geometry is a second instrument that can
be probed without limit — HERE from CACHED artifacts only ($0, no scorer forward, the
live run slot untouched): ``gt_n96.npz`` carries the SegNet argmax labels (``lstars``)
AND the margin field (``margins`` — the measured Fisher surrogate, ρ 0.978), i.e. the
scorer's own confidence geometry on the contest video.

What this module computes (all DERIVED from cached measured tensors, deterministic):
  * per-class FLIP SUSCEPTIBILITY — the fraction of each class's pixel mass sitting at
    low margin (argmax-flippable under small perturbation). This is the scorer-side
    prior on WHERE d_seg mass is movable (the codim-1 annulus, per-class).
  * a recalibrated lever-feature map — lever class-weights re-weighted by
    susceptibility (a lever nominally targeting class c only matters where the scorer
    is actually flippable on c).

HARD RAILS (binding): the frozen scorers are PINNED AUTHORITY — nothing here touches,
retrains, or monkeypatches them; nothing here ships (the organ never enters
archive.zip; no scorer weights/derived tables leak to the witness); every number is
[macOS advisory] NON-PROMOTABLE, never a score. The integration arm (G_ridge_scorerprior
in the tournament) is admitted by BACKTEST only (Gödel gate), never asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from tac.witness_control.lambda_net import N_CLASSES, lever_features

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_GT_CACHE = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"


@dataclass(frozen=True)
class ScorerGeometryPrior:
    """The cached-scorer-geometry prior: per-class flip susceptibility (canonical order
    Road/Lane/Undrivable/Movable/MyCar). All DERIVED from the cached margin field."""

    susceptibility: tuple[float, ...]      # per-class mean exp(-margin/τ) over class pixels
    low_margin_share: tuple[float, ...]    # share of the low-margin (flippable) mass per class
    margin_tau: float                      # the scale (median margin) used
    n_frames: int
    source: str
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (cached-tensor derived)"

    def class_reweight(self) -> np.ndarray:
        """Susceptibility as a normalized reweighting vector (sums to N_CLASSES so a
        uniform prior maps to itself)."""
        s = np.asarray(self.susceptibility, dtype=np.float64)
        tot = float(s.sum())
        return s * (N_CLASSES / tot) if tot > 0 else np.ones(N_CLASSES)


@lru_cache(maxsize=4)
def scorer_flip_susceptibility(cache_path: str | None = None, *,
                               frame_stride: int = 4) -> ScorerGeometryPrior:
    """Compute the per-class flip susceptibility from the CACHED margin field ($0).

    susceptibility_c = E[exp(-margin/τ) | argmax=c] with τ = the median margin (scale-
    free); low_margin_share_c = P(argmax=c | margin < τ/4) — where the flippable mass
    lives. Frame stride bounds compute; deterministic (no RNG)."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    lstars = z["lstars"][::max(int(frame_stride), 1)]        # (F, H, W) int labels
    margins = z["margins"][::max(int(frame_stride), 1)].astype(np.float64)
    tau = float(np.median(margins))
    if tau <= 0:
        tau = float(np.mean(margins)) or 1.0
    flip_kernel = np.exp(-margins / tau)
    low = margins < (tau / 4.0)
    susc, share = [], []
    low_total = max(float(low.sum()), 1.0)
    for c in range(N_CLASSES):
        mask = lstars == c
        n_c = float(mask.sum())
        susc.append(float(flip_kernel[mask].mean()) if n_c > 0 else 0.0)
        share.append(float((low & mask).sum()) / low_total)
    return ScorerGeometryPrior(
        susceptibility=tuple(susc), low_margin_share=tuple(share), margin_tau=tau,
        n_frames=int(lstars.shape[0]), source=str(p))


def scorer_recalibrated_phi(name: str, prior: ScorerGeometryPrior) -> np.ndarray:
    """Lever features with the class-weight block re-weighted by flip susceptibility.

    A lever targeting class c matters only where the scorer is flippable on c — the
    scorer-side calibration of the design surface (2607.08046 calibration finding
    transposed: read the instrument's internal confidence, not its stated output)."""
    phi = lever_features(name).copy()
    cw = phi[:N_CLASSES] * prior.class_reweight()
    tot = float(cw.sum())
    orig = float(phi[:N_CLASSES].sum())
    if tot > 0 and orig > 0:
        cw = cw * (orig / tot)                 # preserve the lever's total class mass
    phi[:N_CLASSES] = cw
    return phi


def _nearest_resize_gray(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Deterministic nearest-neighbor resample of a 2D array to (h, w)."""
    ri = (np.arange(h) * img.shape[0]) // h
    ci = (np.arange(w) * img.shape[1]) // w
    return img[ri][:, ci]


def _local_variance(gray: np.ndarray, k: int = 5) -> np.ndarray:
    """Local variance via integral images (uniform k×k window; numpy-only)."""
    pad = k // 2
    g = np.pad(gray.astype(np.float64), pad, mode="edge")
    g2 = g * g
    cs = np.cumsum(np.cumsum(g, axis=0), axis=1)
    cs2 = np.cumsum(np.cumsum(g2, axis=0), axis=1)

    def _box(c):
        return (c[k - 1:, k - 1:] - np.pad(c[:-k + 1 or None, k - 1:], ((1, 0), (0, 0)))[:-1]
                - np.pad(c[k - 1:, :-k + 1 or None], ((0, 0), (1, 0)))[:, :-1]
                + np.pad(c[:-k + 1 or None, :-k + 1 or None], ((1, 0), (1, 0)))[:-1, :-1])

    n = float(k * k)
    s = _box(cs)[:gray.shape[0], :gray.shape[1]]
    s2 = _box(cs2)[:gray.shape[0], :gray.shape[1]]
    return np.clip(s2 / n - (s / n) ** 2, 0.0, None)


@lru_cache(maxsize=4)
def uniward_cost_susceptibility(cache_path: str | None = None, *,
                                frame_stride: int = 8) -> ScorerGeometryPrior:
    """UNIWARD-weighted flip susceptibility (Holub-Fridrich cost geometry: flips in
    TEXTURED/high-variance regions are steganalytically cheap, smooth regions dear).

    susceptibility_c = E[exp(-margin/τ) · σ_norm | argmax=c] over cached frames — the
    detectability-COST-weighted boundary mass per class (inverse-steganalysis framing:
    λ = boundary normal × embedding cost). All from cached tensors; $0; deterministic.
    NOTE the trainer-lever msal_uni INERT verdict (L76) does NOT bind here — that was
    the msal texture PROXY inside the witness loss; this is the ORGAN's cost model."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    frames = z["gt_f1"][::step]                       # (F, 874, 1164, 3) uint8
    tau = float(np.median(margins)) or 1.0
    h, w = lstars.shape[1], lstars.shape[2]
    susc = np.zeros(N_CLASSES)
    counts = np.zeros(N_CLASSES)
    for f in range(lstars.shape[0]):
        gray = frames[f].astype(np.float64).mean(axis=2)
        var = _local_variance(_nearest_resize_gray(gray, h, w))
        sig = np.sqrt(var)
        sig_norm = sig / (float(np.median(sig)) + 1e-9)   # texture cheapness (scale-free)
        kern = np.exp(-margins[f] / tau) * np.clip(sig_norm, 0.0, 4.0)
        for c in range(N_CLASSES):
            m = lstars[f] == c
            susc[c] += float(kern[m].sum())
            counts[c] += float(m.sum())
    susc = np.where(counts > 0, susc / np.maximum(counts, 1.0), 0.0)
    return ScorerGeometryPrior(
        susceptibility=tuple(float(v) for v in susc),
        low_margin_share=tuple(0.0 for _ in range(N_CLASSES)),
        margin_tau=tau, n_frames=int(lstars.shape[0]),
        source=f"{p} (UNIWARD-weighted; Holub-Fridrich cost geometry)")


class ScorerPriorRidgeAdjoint:
    """Candidate G — the ridge SOLVE on the scorer-recalibrated design surface.

    Identical closed-form solve to A_ridge_solve, but every lever feature φ is passed
    through ``scorer_recalibrated_phi`` — injecting the trajectory-independent scorer
    geometry into the response operator. Admitted only if it BACKTESTS better."""

    name = "G_ridge_scorerprior"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 prior: ScorerGeometryPrior | None = None):
        from tac.witness_control.lambda_net import RidgeSolveAdjoint
        self._inner = RidgeSolveAdjoint(ridge=ridge)
        self.prior = prior if prior is not None else scorer_flip_susceptibility(cache_path)

    def _re(self, phi: np.ndarray) -> np.ndarray:
        out = phi.copy()
        cw = out[:N_CLASSES] * self.prior.class_reweight()
        tot, orig = float(cw.sum()), float(out[:N_CLASSES].sum())
        if tot > 0 and orig > 0:
            cw = cw * (orig / tot)
        out[:N_CLASSES] = cw
        return out

    def fit(self, intervals, phis: np.ndarray, seed: int = 0) -> None:
        re_phis = np.stack([self._re(p) for p in phis])
        self._inner.fit(intervals, re_phis, seed=seed)

    def response(self, x, ctx, phi, path=None) -> np.ndarray:
        return self._inner.response(x, ctx, self._re(phi), path)

    def base(self, x, ctx, path=None) -> np.ndarray:
        return self._inner.base(x, ctx)
