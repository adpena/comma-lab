# SPDX-License-Identifier: MIT
"""SCORER-MODEL ARMS of the #426 costate organ — the P0 build (operator GO 2026-07-11).

The organ's biggest measured limit is n=1 trajectory (envelope §4). These arms give the
adjoint λ = ∂S/∂x SCORER-side instruments that do not depend on the campaign trajectory,
each following the #428 distillation-survey order and each admitted ONLY via the
tournament backtest (the Gödel gate) — never asserted:

  * ARM H — SMOOTHED-ARGMAX METRIC RELAXATION (survey #1, Berthet 2002.08676 /
    Vlastelica 1912.02175 / Gumbel 1611.01144). d_seg's non-differentiability lives in
    the METRIC (per-pixel argmax disagreement), not the teacher. Replace the argmax flip
    indicator with its logistic smoothing P(flip) = σ((δ−m)/ε) on the CACHED margin
    field m (computed through the REAL frozen SegNet — gt_n96.npz, the measured Fisher
    surrogate ρ 0.978): the per-class gradient of the smoothed metric is EXACT
    (∂d_seg^ε/∂δ_c = mean_{argmax=c} σ'(m/ε)/ε), zero model error, no surrogate-vs-
    teacher gap to audit. ε defaults to the median margin — mathematically the SAME knob
    as the measured Maslov temperature τ=ε=ħ (L75). Top-2 approximation stated honestly:
    the smoothing covers flips to the runner-up class (the pairwise tropical/Laguerre
    boundary structure of #284 — the dominant flip mode by construction of the margin).

  * ARM J — ADVERSARIAL-BOUNDARY GEOMETRY (survey #4, Heo 1805.05532 boundary-supporting
    samples; DeepFool-style minimal perturbation read in the IMAGE PLANE). The minimal
    flip perturbation for the measured dominant flip mode — GT-side sub-pixel ADVECTION
    (L85) — is the pixel-plane distance to the argmax boundary. Per-class susceptibility
    = class mass within radius r of an inter-class boundary, PAIR-WEIGHTED by the
    measured Young/Herring surface tension σ_cc′ (#382 ``length_sigma``: LOW tension =
    fragile interface, e.g. Road–Lane 0.377). Composes the margin-polytope free-budget
    picture (``boundary_math.margin_polytope``, the #47 first-order flip system) with
    the v8 pair geometry. Faithfulness is AUDITED, not asserted: the ball-agreement
    acceptance (faithful-KD 2306.04431 transposed) checks the margin-surrogate's
    predicted flip set against the ACTUAL label-change set under the advection ball.

  * ARM I — comma10k REGIME MODEL (the trajectory-INDEPENDENT λ source; the highest-
    value arm per the operator directive). SegNet's REAL training distribution is
    comma10k (0 contest frames — L80), so per-class training support is a scorer-side
    prior that exists before ANY trajectory: classes RARE in training are under-trained
    → flippable → λ-hot (the DERIVED hypothesis; it matches the measured lane-long-tail
    crux — Lane is 0.55% of comma10k mass and the flip-dominant class). The prior is
    built ONCE from a bounded fetch of real comma10k masks
    (``tools/build_comma10k_regime_prior.py``) into a durable JSON artifact; this
    module only READS the artifact ($0, offline, deterministic).

  * ARM K — PER-CLASS λ-HEADS with the v8 CARRIER-JOIN / GEOMETRY-RECONCILE. The
    diagonal lever-feature class block misses measured cross-class coupling: a
    Lane-targeting lever moves Road too, through the SHARED Road–Lane boundary. The
    coupling matrix is REUSED v8 math, not rebuilt: boundary-pair adjacency MEASURED
    from the cached argmax partition (the power-diagram/tropical pair structure, #284)
    weighted by the fitted Young σ_cc′ (#382; low tension ⇒ stronger coupling). Feeds
    the #430 coherent-schedule composer with per-class λ.

HARD RAILS (binding, inherited from ``scorer_geometry``): cached artifacts ONLY — no
scorer forward, no network, the live #205 run slot untouched; the frozen scorers are
PINNED AUTHORITY (never touched/retrained/shipped); nothing here enters archive.zip;
every number [macOS advisory] NON-PROMOTABLE, never a score. No actuation surface
(containment source-scan). Each arm is a tournament candidate admitted by BACKTEST only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.boundary_math.length_sigma import resolve_length_sigma_matrix
from tac.witness_control.lambda_net import N_CLASSES, RidgeSolveAdjoint, lever_features
from tac.witness_control.scorer_geometry import DEFAULT_GT_CACHE

_REPO = Path(__file__).resolve().parents[3]
#: durable comma10k prior artifact (built once by tools/build_comma10k_regime_prior.py)
DEFAULT_COMMA10K_PRIOR = (
    _REPO / "experiments/results/comma10k_regime_prior/comma10k_class_prior.json")

#: canonical comma10k mask palette (repo README), in the CANONICAL class order
#: Road0 / Lane1 / Undrivable2 / Movable3 / MyCar4 — the orders coincide.
COMMA10K_PALETTE: tuple[tuple[int, int, int], ...] = (
    (64, 32, 32),      # 1 road      #402020
    (255, 0, 0),       # 2 lane      #ff0000
    (128, 128, 96),    # 3 undrivable #808060
    (0, 255, 102),     # 4 movable   #00ff66
    (204, 0, 255),     # 5 my car    #cc00ff
)


# ─────────────────────────────────────────────────────────────────────────────
# shared ridge-with-recalibrated-φ pattern (the G-arm pattern, factored)
# ─────────────────────────────────────────────────────────────────────────────
class _ReweightedRidgeArm:
    """Ridge SOLVE with the lever-feature class block passed through a recalibration.

    Subclasses set ``self._weight`` (a (5,) vector) or ``self._coupling`` (a (5,5)
    matrix). Total class mass of each lever is preserved (only the DISTRIBUTION over
    classes changes) — identical contract to ``ScorerPriorRidgeAdjoint``."""

    name = "_reweighted_ridge"

    def __init__(self, ridge: float = 1e-2):
        self._inner = RidgeSolveAdjoint(ridge=ridge)
        self._weight: np.ndarray | None = None      # (5,) multiplicative reweight
        self._coupling: np.ndarray | None = None    # (5,5) linear coupling (rows: out)

    def _re(self, phi: np.ndarray) -> np.ndarray:
        out = np.asarray(phi, dtype=np.float64).copy()
        cw = out[:N_CLASSES]
        orig = float(cw.sum())
        if self._coupling is not None:
            cw = self._coupling @ cw
        if self._weight is not None:
            cw = cw * self._weight
        tot = float(cw.sum())
        if tot > 0 and orig > 0:
            cw = cw * (orig / tot)
        out[:N_CLASSES] = cw
        return out

    def fit(self, intervals, phis: np.ndarray, seed: int = 0) -> None:
        self._inner.fit(intervals, np.stack([self._re(p) for p in phis]), seed=seed)

    def response(self, x, ctx, phi, path=None) -> np.ndarray:
        return self._inner.response(x, ctx, self._re(phi), path)

    def base(self, x, ctx, path=None) -> np.ndarray:
        return self._inner.base(x, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# ARM H — smoothed-argmax metric relaxation (survey #1; ZERO model error)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SmoothedArgmaxField:
    """The ε-smoothed d_seg metric + its EXACT per-class gradient, from the cached
    margin field of the REAL frozen SegNet (no surrogate, no model error).

    ``grad_per_class[c]`` = ∂d_seg^ε/∂δ_c: the increase of the smoothed flip metric per
    unit uniform adversarial logit-shift toward flipping on class-c pixels. Positive by
    construction; the λ DIRECTION over classes (advisory, never a score)."""

    epsilon: float                      # the smoothing temperature (τ = ε = ħ, L75)
    smoothed_dseg: float                # mean σ(−m/ε) over all pixels (the relaxed metric)
    grad_per_class: tuple[float, ...]   # exact ∂(smoothed d_seg)/∂δ_c, canonical order
    n_frames: int
    source: str
    approximation: str = ("top-2 pairwise (flip to the runner-up class — the tropical/"
                          "Laguerre boundary structure; exact for the dominant flip mode)")
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (cached-tensor derived)"

    def class_reweight(self) -> np.ndarray:
        g = np.asarray(self.grad_per_class, dtype=np.float64)
        tot = float(g.sum())
        return g * (N_CLASSES / tot) if tot > 0 else np.ones(N_CLASSES)


def smoothed_dseg_and_grad(margins: np.ndarray, lstars: np.ndarray,
                           epsilon: float) -> tuple[float, np.ndarray]:
    """The ε-smoothed metric and its EXACT per-class gradient (pure function, testable
    against finite differences). margins/lstars: (F,H,W); epsilon > 0."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    m = margins.astype(np.float64)
    s = 1.0 / (1.0 + np.exp(m / epsilon))            # σ(−m/ε): the smoothed flip prob
    sprime = s * (1.0 - s) / epsilon                 # ∂σ((δ−m)/ε)/∂δ at δ=0
    n_pix = float(m.size)
    grad = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        grad[c] = float(sprime[lstars == c].sum()) / n_pix
    return float(s.mean()), grad


def smoothed_argmax_field(cache_path: str | Path | None = None, *,
                          epsilon: float | None = None,
                          frame_stride: int = 4) -> SmoothedArgmaxField:
    """Compute the smoothed-argmax field from the CACHED margin field ($0, offline).

    ε defaults to the median margin (the same scale ``scorer_geometry`` uses — the
    measured Maslov temperature knob τ=ε=ħ per L75)."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    eps = float(epsilon) if epsilon is not None else float(np.median(margins))
    if eps <= 0:
        eps = float(np.mean(margins)) or 1.0
    sd, grad = smoothed_dseg_and_grad(margins, lstars, eps)
    return SmoothedArgmaxField(
        epsilon=eps, smoothed_dseg=sd,
        grad_per_class=tuple(float(g) for g in grad),
        n_frames=int(lstars.shape[0]), source=str(p))


class SmoothedArgmaxRidgeAdjoint(_ReweightedRidgeArm):
    """ARM H — ridge SOLVE on the smoothed-argmax-recalibrated design surface.

    The class block of every lever feature is reweighted by the EXACT gradient of the
    ε-smoothed d_seg metric (survey #1: metric relaxation through the real frozen
    teacher — zero model error). Admitted by BACKTEST only."""

    name = "H_smoothed_argmax"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 field: SmoothedArgmaxField | None = None):
        super().__init__(ridge=ridge)
        f = field if field is not None else smoothed_argmax_field(cache_path)
        self.field = f
        self._weight = f.class_reweight()


# ─────────────────────────────────────────────────────────────────────────────
# ARM J — adversarial-boundary geometry (+ the ball-agreement faithfulness audit)
# ─────────────────────────────────────────────────────────────────────────────
def _shift2d(a: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    """Non-wrapping 2D shift (np.roll wraps — wrong geometry at frame edges)."""
    out = np.full_like(a, fill)
    h, w = a.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    yt = slice(max(-dy, 0), h + min(-dy, 0))
    xt = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = a[yt, xt]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Chebyshev (8-neighbor square) binary dilation via non-wrapping shifts."""
    acc = mask.copy()
    for _ in range(max(int(radius), 0)):
        nxt = acc.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    nxt |= _shift2d(acc, dy, dx, False)
        acc = nxt
    return acc


def boundary_pair_shares(lstars: np.ndarray) -> np.ndarray:
    """MEASURED (5,5) symmetric inter-class boundary-pair shares from cached argmax
    labels (the power-diagram pair-adjacency of the partition, #284). Normalized to
    sum 1 over off-diagonal pairs; diagonal 0. lstars: (F,H,W) or (H,W) int."""
    ls = lstars if lstars.ndim == 3 else lstars[None]
    counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for f in range(ls.shape[0]):
        lab = ls[f]
        for dy, dx in ((0, 1), (1, 0)):
            nb = _shift2d(lab, dy, dx, -1)
            diff = (nb != lab) & (nb >= 0)
            a = lab[diff]
            b = nb[diff]
            np.add.at(counts, (a, b), 1.0)
    counts = counts + counts.T
    np.fill_diagonal(counts, 0.0)
    tot = float(counts.sum())
    return counts / tot if tot > 0 else counts


@dataclass(frozen=True)
class AdversarialBoundaryPrior:
    """Per-class adversarial-boundary susceptibility: class mass within the advection
    ball of an inter-class boundary, pair-weighted by 1/σ_cc′ (low Young tension =
    fragile interface). All from cached tensors; deterministic; advisory."""

    susceptibility: tuple[float, ...]     # per-class, canonical order
    pair_share: tuple[tuple[float, ...], ...]   # the measured (5,5) boundary adjacency
    radius_px: int
    sigma_preset: str
    n_frames: int
    source: str
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (cached-tensor derived)"

    def class_reweight(self) -> np.ndarray:
        s = np.asarray(self.susceptibility, dtype=np.float64)
        tot = float(s.sum())
        return s * (N_CLASSES / tot) if tot > 0 else np.ones(N_CLASSES)


def adversarial_boundary_susceptibility(
        cache_path: str | Path | None = None, *, radius_px: int = 2,
        sigma_preset: str = "fitted-20260707",
        frame_stride: int = 8) -> AdversarialBoundaryPrior:
    """ARM-J sensor: minimal-flip (advection-ball) susceptibility per class ($0).

    For each unordered class pair (c,c′): the pair-boundary band (dilated radius r)
    intersected with class-c mass, weighted 1/σ_cc′ (#382 fitted Young matrix; low
    tension ⇒ fragile ⇒ up-weighted). Sum over partners, normalize by class mass."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    sigma = resolve_length_sigma_matrix(sigma_preset)
    if sigma is None:                                  # "all-ones" → uniform tension
        sigma = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    mass = np.zeros(N_CLASSES, dtype=np.float64)
    cls_count = np.zeros(N_CLASSES, dtype=np.float64)
    for f in range(lstars.shape[0]):
        lab = lstars[f]
        for c in range(N_CLASSES):
            cls_count[c] += float((lab == c).sum())
        for a in range(N_CLASSES):
            for b in range(a + 1, N_CLASSES):
                pair_boundary = np.zeros(lab.shape, dtype=bool)
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nb = _shift2d(lab, dy, dx, -1)
                    pair_boundary |= ((lab == a) & (nb == b)) | ((lab == b) & (nb == a))
                if not pair_boundary.any():
                    continue
                band = _dilate(pair_boundary, radius_px)
                w = 1.0 / float(sigma[a, b])
                mass[a] += w * float((band & (lab == a)).sum())
                mass[b] += w * float((band & (lab == b)).sum())
    susc = np.where(cls_count > 0, mass / np.maximum(cls_count, 1.0), 0.0)
    return AdversarialBoundaryPrior(
        susceptibility=tuple(float(v) for v in susc),
        pair_share=tuple(tuple(float(v) for v in row)
                         for row in boundary_pair_shares(lstars)),
        radius_px=int(radius_px), sigma_preset=str(sigma_preset),
        n_frames=int(lstars.shape[0]), source=str(p))


def ball_agreement_audit(cache_path: str | Path | None = None, *,
                         radius_px: int = 1, frame_stride: int = 8,
                         iou_floor: float = 0.5) -> dict:
    """The faithfulness ACCEPTANCE for the margin-surrogate boundary model
    (faithful-KD 2306.04431 ball-agreement, transposed to the advection ball).

    PREDICTED flip set: pixels whose cached SegNet MARGIN is below a rank-matched
    threshold (predicted mass = actual mass, so the audit tests pure GEOMETRY, not
    calibration). ACTUAL flip set: pixels whose cached argmax LABEL changes under some
    shift in the Chebyshev r-ball (the measured dominant flip mode — GT-side sub-pixel
    advection, L85). Two independent cached sources (logit confidence vs partition
    geometry) → a real audit, $0. Returns precision/recall/IoU + the faithful flag."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    r = max(int(radius_px), 1)
    ious, precs, recs = [], [], []
    for f in range(lstars.shape[0]):
        lab = lstars[f]
        actual = np.zeros(lab.shape, dtype=bool)
        valid = np.ones(lab.shape, dtype=bool)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                nb = _shift2d(lab, dy, dx, -1)
                actual |= (nb != lab) & (nb >= 0)
                valid &= _shift2d(np.ones(lab.shape, dtype=bool), dy, dx, False)
        actual &= valid
        n_act = int(actual.sum())
        if n_act == 0:
            continue
        m = margins[f][valid]
        thresh = float(np.partition(m, n_act - 1)[n_act - 1])   # rank-matched mass
        predicted = (margins[f] <= thresh) & valid
        inter = float((predicted & actual).sum())
        union = float((predicted | actual).sum())
        ious.append(inter / union if union > 0 else 1.0)
        precs.append(inter / max(float(predicted.sum()), 1.0))
        recs.append(inter / float(n_act))
    iou = float(np.mean(ious)) if ious else float("nan")
    out = {
        "radius_px": r, "n_frames": int(lstars.shape[0]),
        "iou": iou,
        "precision": float(np.mean(precs)) if precs else float("nan"),
        "recall": float(np.mean(recs)) if recs else float("nan"),
        "iou_floor": float(iou_floor),
        "faithful": bool(ious and iou >= iou_floor),
        "protocol": "rank-matched margin threshold vs advection-ball label-change set "
                    "(2306.04431 ball-agreement; two independent cached sources)",
        "axis_tag": "[macOS advisory] NON-PROMOTABLE", "score_claim": False,
    }
    return out


class AdversarialBoundaryRidgeAdjoint(_ReweightedRidgeArm):
    """ARM J — ridge SOLVE on the adversarial-boundary-recalibrated design surface.
    Admitted by BACKTEST only; its faithfulness audit is ``ball_agreement_audit``."""

    name = "J_adv_boundary"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 prior: AdversarialBoundaryPrior | None = None):
        super().__init__(ridge=ridge)
        pr = prior if prior is not None else adversarial_boundary_susceptibility(cache_path)
        self.prior = pr
        self._weight = pr.class_reweight()


# ─────────────────────────────────────────────────────────────────────────────
# ARM I — comma10k regime model (trajectory-INDEPENDENT; reads the durable artifact)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Comma10kRegimePrior:
    """Per-class statistics of SegNet's REAL training distribution (comma10k masks).

    ``class_pixel_share``: fraction of labeled mask pixels per canonical class.
    ``class_boundary_share``: fraction of inter-class boundary pixels touching the class.
    ``pair_share``: (5,5) inter-class boundary adjacency over the comma10k corpus.
    ``unmatched_frac``: mask pixels not matching the 5-color palette (honesty bound)."""

    class_pixel_share: tuple[float, ...]
    class_boundary_share: tuple[float, ...]
    pair_share: tuple[tuple[float, ...], ...]
    n_masks: int
    unmatched_frac: float
    source: str
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (comma10k-derived; 0 contest frames)"

    def rarity_reweight(self) -> np.ndarray:
        """The DERIVED direction: rare-in-training ⇒ under-trained ⇒ flippable ⇒ λ-hot
        (matches the measured lane-long-tail crux). Normalized to sum N_CLASSES."""
        s = np.asarray(self.class_pixel_share, dtype=np.float64)
        inv = 1.0 / np.maximum(s, 1e-6)
        return inv * (N_CLASSES / float(inv.sum()))

    def to_jsonable(self) -> dict:
        return {
            "class_pixel_share": list(self.class_pixel_share),
            "class_boundary_share": list(self.class_boundary_share),
            "pair_share": [list(r) for r in self.pair_share],
            "n_masks": self.n_masks, "unmatched_frac": self.unmatched_frac,
            "source": self.source, "axis_tag": self.axis_tag,
            "class_order": "Road/Lane/Undrivable/Movable/MyCar (canonical)",
        }


def comma10k_labels_from_rgb(rgb: np.ndarray, *, tol: int = 8) -> np.ndarray:
    """Map a comma10k mask RGB array (H,W,3) to canonical class labels; -1 = unmatched.
    Exact-palette masks match at tol=0; a small tolerance absorbs resave artifacts."""
    h, w = rgb.shape[:2]
    lab = np.full((h, w), -1, dtype=np.int64)
    r = rgb.astype(np.int64)
    for c, (pr, pg, pb) in enumerate(COMMA10K_PALETTE):
        m = ((np.abs(r[..., 0] - pr) <= tol) & (np.abs(r[..., 1] - pg) <= tol)
             & (np.abs(r[..., 2] - pb) <= tol))
        lab[m] = c
    return lab


def build_comma10k_prior_from_labels(labels: list[np.ndarray], *,
                                     source: str) -> Comma10kRegimePrior:
    """Reduce decoded comma10k label maps to the regime prior (pure, testable)."""
    if not labels:
        raise ValueError("no comma10k label maps supplied")
    pix = np.zeros(N_CLASSES, dtype=np.float64)
    unmatched = 0.0
    total = 0.0
    bnd_touch = np.zeros(N_CLASSES, dtype=np.float64)
    pair = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for lab in labels:
        total += float(lab.size)
        unmatched += float((lab < 0).sum())
        for c in range(N_CLASSES):
            pix[c] += float((lab == c).sum())
        for dy, dx in ((0, 1), (1, 0)):
            nb = _shift2d(lab, dy, dx, -1)
            diff = (nb != lab) & (nb >= 0) & (lab >= 0)
            a, b = lab[diff], nb[diff]
            np.add.at(pair, (a, b), 1.0)
            np.add.at(bnd_touch, a, 1.0)
            np.add.at(bnd_touch, b, 1.0)
    pair = pair + pair.T
    np.fill_diagonal(pair, 0.0)
    pair_tot = float(pair.sum()) or 1.0
    pix_tot = float(pix.sum()) or 1.0
    bnd_tot = float(bnd_touch.sum()) or 1.0
    return Comma10kRegimePrior(
        class_pixel_share=tuple(float(v) for v in pix / pix_tot),
        class_boundary_share=tuple(float(v) for v in bnd_touch / bnd_tot),
        pair_share=tuple(tuple(float(v) for v in row) for row in pair / pair_tot),
        n_masks=len(labels), unmatched_frac=float(unmatched / max(total, 1.0)),
        source=source)


def load_comma10k_prior(path: str | Path | None = None) -> Comma10kRegimePrior:
    """Load the durable artifact (built by ``tools/build_comma10k_regime_prior.py``).
    Fail-closed: absent artifact raises (the arm then reports honestly in the
    tournament as FAILED, never a silent fallback)."""
    p = Path(path) if path else DEFAULT_COMMA10K_PRIOR
    if not p.exists():
        raise FileNotFoundError(
            f"comma10k prior artifact absent: {p} — build it once with "
            "tools/build_comma10k_regime_prior.py (bounded fetch, ~1-2 MB)")
    obj = json.loads(p.read_text())
    return Comma10kRegimePrior(
        class_pixel_share=tuple(float(v) for v in obj["class_pixel_share"]),
        class_boundary_share=tuple(float(v) for v in obj["class_boundary_share"]),
        pair_share=tuple(tuple(float(v) for v in r) for r in obj["pair_share"]),
        n_masks=int(obj["n_masks"]), unmatched_frac=float(obj["unmatched_frac"]),
        source=str(obj.get("source", str(p))))


class Comma10kRegimeRidgeAdjoint(_ReweightedRidgeArm):
    """ARM I — ridge SOLVE on the comma10k-rarity-recalibrated design surface (the
    trajectory-independent regime model). Admitted by BACKTEST only."""

    name = "I_comma10k_regime"

    def __init__(self, ridge: float = 1e-2, prior_path: str | None = None,
                 prior: Comma10kRegimePrior | None = None):
        super().__init__(ridge=ridge)
        pr = prior if prior is not None else load_comma10k_prior(prior_path)
        self.prior = pr
        self._weight = pr.rarity_reweight()


# ─────────────────────────────────────────────────────────────────────────────
# ARM K — per-class λ-heads with the v8 σ_cc′ carrier-join / geometry-reconcile
# ─────────────────────────────────────────────────────────────────────────────
def perclass_coupling_matrix(cache_path: str | Path | None = None, *,
                             sigma_preset: str = "fitted-20260707",
                             alpha: float = 0.5,
                             frame_stride: int = 8) -> np.ndarray:
    """The measured cross-class coupling C = α·I + (1−α)·rownorm(A ⊙ 1/σ) where A is
    the boundary-pair adjacency MEASURED from the cached argmax partition (#284 pair
    structure) and σ the fitted Young tension (#382; low tension ⇒ stronger coupling).
    α=0.5 is a DERIVED-AT-CONFIG blend (half own-class, half boundary-mediated);
    the backtest arbitrates the arm, never this constant alone."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    lstars = z["lstars"][::max(int(frame_stride), 1)]
    A = boundary_pair_shares(lstars)
    sigma = resolve_length_sigma_matrix(sigma_preset)
    if sigma is None:
        sigma = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    W = A / sigma
    np.fill_diagonal(W, 0.0)
    rows = W.sum(axis=1, keepdims=True)
    W = np.divide(W, rows, out=np.zeros_like(W), where=rows > 0)
    return float(alpha) * np.eye(N_CLASSES) + (1.0 - float(alpha)) * W


class PerClassCoupledRidgeAdjoint(_ReweightedRidgeArm):
    """ARM K — per-class λ-heads reconciled through the v8 pair geometry: the class
    block of every lever feature is propagated through the measured boundary-coupling
    matrix (a Lane lever moves Road through the shared Road–Lane boundary). Feeds the
    #430 coherent-schedule composer with per-class λ. Admitted by BACKTEST only."""

    name = "K_perclass_v8"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 sigma_preset: str = "fitted-20260707", alpha: float = 0.5,
                 coupling: np.ndarray | None = None):
        super().__init__(ridge=ridge)
        C = coupling if coupling is not None else perclass_coupling_matrix(
            cache_path, sigma_preset=sigma_preset, alpha=alpha)
        if C.shape != (N_CLASSES, N_CLASSES):
            raise ValueError(f"coupling must be (5,5), got {C.shape}")
        self.coupling = C
        self._coupling = C

    def perclass_lambda(self, model_response: np.ndarray,
                        grad_s: np.ndarray) -> np.ndarray:
        """Per-class marginal-ΔS decomposition of one lever's response (the per-class
        λ-heads readout the #430 composer consumes): λ_c = g_c · r_c (elementwise,
        before the scalar sum) — advisory ranking vector."""
        r = np.asarray(model_response, dtype=np.float64)[:N_CLASSES]
        g = np.asarray(grad_s, dtype=np.float64)[:N_CLASSES]
        return g * r


# ─────────────────────────────────────────────────────────────────────────────
# SHRINK-TO-PRIOR ridge (the MEASURED cure for the φ-recalibration formulation)
# ─────────────────────────────────────────────────────────────────────────────
# MEASURED NEGATIVE (2026-07-11, #205 trajectory, verdict_scope: FORMULATION): passing
# the scorer prior as a φ-class-block RESCALE into an unconstrained ridge re-fit is
# INERT on the forecast gate (WF 0.003535 ≡ plain ridge to 4 decimals; early-fold MAE
# identical) — the solve simply refits M against the rescaled features and the prior
# cancels. The ADJACENT CURE (the neg↔cure discipline): make the prior the ridge
# REGULARIZATION TARGET — shrink the coefficient toward a PRIOR-MEAN response instead
# of toward zero. At small n the null space is huge and the ridge term OWNS it, so the
# solve returns the prior-structured response where data is silent (the Bayesian
# small-n limit) and lets data override it as folds accrue. This is the formulation in
# which a trajectory-INDEPENDENT prior can actually cure the n=1 fragility.
class PriorMeanRidgeAdjoint:
    """Ridge SOLVE shrunk toward a scorer-prior-structured response (not toward 0).

    Stage 1 (empirical-Bayes, tiny): fit per-channel base b and ONE scalar κ so that
    dx/dt ≈ b − κ·(s ⊙ occ_class) on the train intervals — s is the per-class scorer
    susceptibility (which classes are movable), occ_class the share-weighted class
    occupancy of the controls. 7 parameters — fittable at n=2.
    Stage 2: the full ridge solve with target coef0 built from (b, κ, s):
    coef = argmin ‖Φ·coef − Y‖² + r·scale·‖coef − coef0‖². Same query API as ridge."""

    name = "priormean_base"

    def __init__(self, prior_class_weight: np.ndarray, ridge: float = 1e-2):
        w = np.asarray(prior_class_weight, dtype=np.float64)
        if w.shape != (N_CLASSES,):
            raise ValueError(f"prior_class_weight must be (5,), got {w.shape}")
        tot = float(w.sum())
        self.s = w * (N_CLASSES / tot) if tot > 0 else np.ones(N_CLASSES)
        self.ridge = float(ridge)
        self.coef: np.ndarray | None = None
        self._state_dim = N_CLASSES + 1

    def _rows(self, intervals, phis: np.ndarray):
        rows, ys, occs = [], [], []
        for iv in intervals:
            occ = phis.T @ iv.u_mean
            rows.append(np.concatenate([[1.0], iv.x0, occ]))
            ys.append(iv.dxdt())
            occs.append(occ[:N_CLASSES])
        return np.stack(rows), np.stack(ys), np.stack(occs)

    def fit(self, intervals, phis: np.ndarray, seed: int = 0) -> None:
        S = self._state_dim
        Phi, Y, occ_cls = self._rows(intervals, phis)
        # stage 1: b_c + κ·(−s_c·occ_c) per class channel; bytes channel base-only
        drive = -(occ_cls * self.s[None, :])                     # (N, 5)
        n = len(intervals)
        A = np.zeros((n * N_CLASSES, N_CLASSES + 1))
        y1 = np.zeros(n * N_CLASSES)
        for i in range(n):
            for c in range(N_CLASSES):
                A[i * N_CLASSES + c, c] = 1.0
                A[i * N_CLASSES + c, N_CLASSES] = drive[i, c]
                y1[i * N_CLASSES + c] = Y[i, c]
        sol, *_ = np.linalg.lstsq(A, y1, rcond=None)
        b, kappa = sol[:N_CLASSES], float(max(sol[N_CLASSES], 0.0))  # κ ≥ 0 (prior sign)
        # coef0: intercept=b (+ measured mean bytes drift), C=0, M0 = −κ·diag(s) on the
        # class block of φ (a class-targeting lever reduces its class's d_seg where the
        # scorer says the class is movable)
        p = Phi.shape[1]
        coef0 = np.zeros((p, S))
        coef0[0, :N_CLASSES] = b
        coef0[0, N_CLASSES] = float(Y[:, N_CLASSES].mean())
        for c in range(N_CLASSES):
            coef0[1 + S + c, c] = -kappa * self.s[c]
        gram = Phi.T @ Phi
        scale = float(np.mean(np.diag(gram))) or 1.0
        r = self.ridge * scale
        self.coef = np.linalg.solve(gram + r * np.eye(p), Phi.T @ Y + r * coef0)
        self.kappa, self.b = kappa, b

    def response(self, x, ctx, phi, path=None) -> np.ndarray:
        assert self.coef is not None, "fit first"
        return self.coef[1 + self._state_dim:].T @ np.asarray(phi, dtype=np.float64)

    def base(self, x, ctx, path=None) -> np.ndarray:
        assert self.coef is not None, "fit first"
        return self.coef[0] + self.coef[1:1 + self._state_dim].T @ np.asarray(
            x, dtype=np.float64)


class Comma10kPriorMeanAdjoint(PriorMeanRidgeAdjoint):
    """ARM L — shrink-to-prior ridge with the comma10k rarity prior (the cure
    formulation of the trajectory-independent regime model)."""

    name = "L_priormean_comma10k"

    def __init__(self, ridge: float = 1e-2, prior_path: str | None = None):
        prior = load_comma10k_prior(prior_path)
        super().__init__(prior.rarity_reweight(), ridge=ridge)
        self.prior = prior


class AdvBoundaryPriorMeanAdjoint(PriorMeanRidgeAdjoint):
    """ARM M — shrink-to-prior ridge with the adversarial-boundary susceptibility."""

    name = "M_priormean_advb"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None):
        prior = adversarial_boundary_susceptibility(cache_path)
        super().__init__(prior.class_reweight(), ridge=ridge)
        self.prior = prior


# ─────────────────────────────────────────────────────────────────────────────
# early-fold walk-forward (the n=1-fragility instrument): per-fold errors so the
# trajectory-independent arms can be scored WHERE trajectory data is scarcest.
# ─────────────────────────────────────────────────────────────────────────────
def walkforward_per_fold(traj, architecture: str, seed: int = 0) -> dict:
    """Per-fold walk-forward errors (mirrors ``lambda_net.backtest``'s walk-forward
    loop but returns the fold-resolved errors instead of the mean — the instrument
    for the out-of-distribution / small-train-set question the comma10k arm answers)."""
    from tac.witness_control.lambda_net import (
        _predict_interval, build_intervals, fit_score_composition, make_model)
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        raise ValueError(f"need ≥3 intervals; have {len(intervals)}")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    wcls = comp.class_weights
    folds = []
    for hold in range(2, len(intervals)):
        model = make_model(architecture)
        model.fit(intervals[:hold], phis, seed=seed)
        iv = intervals[hold]
        pred = _predict_interval(model, architecture, iv, traj.lever_names)
        meas = iv.dxdt()
        heur = intervals[hold - 1].dxdt()
        folds.append({
            "fold": hold, "n_train": hold, "ep0": iv.ep0, "ep1": iv.ep1,
            "err_model": abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep,
            "err_heuristic": abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep,
            "perclass_err_model": float(
                np.mean(np.abs(pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep,
        })
    early = [f for f in folds if f["n_train"] <= 3]
    return {
        "architecture": architecture,
        "folds": folds,
        "early_mae_model": (float(np.mean([f["err_model"] for f in early]))
                            if early else float("nan")),
        "early_mae_heuristic": (float(np.mean([f["err_heuristic"] for f in early]))
                                if early else float("nan")),
        "mae_model": float(np.mean([f["err_model"] for f in folds])),
        "mae_heuristic": float(np.mean([f["err_heuristic"] for f in folds])),
        "axis_tag": "[macOS advisory] NON-PROMOTABLE", "score_claim": False,
    }


__all__ = [
    "COMMA10K_PALETTE",
    "DEFAULT_COMMA10K_PRIOR",
    "AdversarialBoundaryPrior",
    "AdversarialBoundaryRidgeAdjoint",
    "Comma10kRegimePrior",
    "Comma10kRegimeRidgeAdjoint",
    "PerClassCoupledRidgeAdjoint",
    "SmoothedArgmaxField",
    "SmoothedArgmaxRidgeAdjoint",
    "adversarial_boundary_susceptibility",
    "ball_agreement_audit",
    "boundary_pair_shares",
    "build_comma10k_prior_from_labels",
    "comma10k_labels_from_rgb",
    "load_comma10k_prior",
    "perclass_coupling_matrix",
    "smoothed_argmax_field",
    "smoothed_dseg_and_grad",
    "walkforward_per_fold",
]
