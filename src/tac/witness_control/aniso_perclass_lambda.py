# SPDX-License-Identifier: MIT
"""ANISOTROPIC-COUPLED PER-CLASS λ — the P0 physics-model directive (operator 2026-07-11,
task #433): the organ's per-class-λ must CONSUME the V9·CGauge carrier equations, never
treat classes as independent scalar knobs.

Operator verbatim: "Bulk × sensitivity × impact on d_seg are ONE interacting per-class
ANISOTROPIC dynamical system; each class has a very different profile and they interact;
also remember anisotropic. Our V9·CGauge carrier work models that beautifully and we have
all the equations."

The physics model, each ingredient a REGISTERED law consumed (never re-derived):

  * BULK vs SENSITIVITY = the two Fisher-metric regimes (``cgauge_master_action_v1`` A2:
    flat interior + anisotropic boundary; margin field = the measured Fisher surrogate,
    Pearson 0.978). Per class, the ε-smoothed flip-sensitivity field σ'(m/ε)/ε (ARM H's
    EXACT metric-relaxation gradient, ε = τ = ħ per L75) is SPLIT into the interior
    (bulk) part and the boundary-annulus part, both MEASURED from the cached margin
    field. The split is the per-class "profile" — Road's flip mass spreads interior +
    annulus, Lane's concentrates on the annulus (measured below, reported per class).
  * PER-CLASS-PAIR anisotropic surface tension σ_cc′ (#382, Γ-limit/Young's law —
    ``junction_young_angle_sigma_fit_v1`` via ``resolve_length_sigma_matrix``): a
    DIFFERENT tension per class pair, never one scalar per class.
  * CLASS INTERACTION through the Laguerre power-diagram GENERATORS (#284,
    ``argmax_of_sdf_is_additively_weighted_power_diagram_v1`` /
    ``textured_power_diagram_sufficient_statistic_v1``): boundary-pair adjacency is
    MEASURED from the cached argmax partition, so a Road–Lane edge flip couples the two
    classes BY CONSTRUCTION.
  * ANISOTROPIC = along-tangent ≠ across-normal
    (``anisotropic_basis_along_tangent_frequency_deficit_v1`` +
    ``cgauge_curvelet_parabolic_bank_v1``): per pair band, the margin field's gradient
    energy is decomposed along the boundary tangent vs across the normal (structure
    read from the pair-interface orientation) — λ is DIRECTIONAL. Pairs with heavy
    along-tangent margin structure (the dash comb) sit in the measured under-supplied
    direction (need 25 vs supplied 8 cyc/unit → deficit 3.125×) and are up-weighted.

The coupling matrix (rows = responding class c, columns = lever-target class c′)::

    C_phys[c,:] = bulk_frac_c · e_c
                  + bnd_frac_c · ( ½·e_c + ½·P_c )        (interface symmetry: EXACT)
    P_c[c′] ∝ pair_susc[c,c′] · (1/σ_cc′) · aniso_cc′     (measured, per pair)

The ½ own/partner split of the boundary part is DERIVED_EXACT from the margin
definition m = φ_top1 − φ_top2: the two classes at an interface enter m with
coefficients ±1 — exactly symmetric roles.

FORMULATION DISCIPLINE (measured 2026-07-11, scorer_model_arms wave): a φ-class-block
RESCALE/COUPLING into an unconstrained ridge re-fit is largely ABSORBED by the solve
(the G/H/I/J/K neutrality, understood structurally). So the PRIMARY physics-consuming
formulation here is the PRIOR-MEAN (shrink-to-prior) one, where the ridge null space is
OWNED by the physics-structured target and data overrides it as folds accrue — with the
SCORE-LAW-PINNED scale cure for the measured empirical-Bayes-κ failure (arm L): the
prior's DIRECTION + RELATIVE magnitudes are fully pinned by the physics
(C_phys ∘ exact smoothed-argmax gradient), leaving ONE global gain κ fit by a 1-dof
robust projection (vs the 7-parameter lstsq that injected variance at n=2). The
coupling-reweight formulation ships too (N) but is expected-near-neutral and labeled so.

ARMS (each admitted ONLY via the walk-forward tri-gate backtest; letters continue the
lambda_net tournament):

  * N_aniso_coupled       — coupling-reweight ridge with C_phys (expected-near-neutral).
  * P_priormean_aniso     — THE P0 ARM: shrink-to-prior ridge, prior mean
                            M0[c,c′] = −κ·C_phys[c,c′]·g^ε_c′ (anisotropic-coupled,
                            score-law-pinned relative scale, 1-dof κ).
  * Q_priormean_iso       — the ISOTROPIC-INDEPENDENT ablation of P: M0_unit = I
                            (same formulation, no physics) — the honest A/B.
  * R_priormean_c10k_scorelaw — the OPEN comma10k-family member (thread 1): comma10k
                            rarity DIRECTION × Fisher sensitivity × σ_cc′-anisotropic
                            coupling, score-law-pinned scale (NOT empirical-Bayes κ).
  * O_openpilot_geom      — openpilot ISOLATED arm, reweight formulation (thread 2).
  * S_priormean_openpilot — openpilot ISOLATED arm, prior-mean formulation (the
                            non-absorbable test; ego ground-plane/hood geometry only,
                            Movable excluded BY MODEL SCOPE — ego motion does not
                            explain independently-moving objects, per the L83 carrier
                            law movable→event/contour).

HARD RAILS (inherited): cached artifacts ONLY (gt_n96.npz margins/lstars + the durable
comma10k prior + σ_cc′ preset) — no scorer forward, no network, the live #205 run slot
untouched; numpy/CPU only; nothing here enters archive.zip; every number
[macOS advisory] NON-PROMOTABLE, never a score. No actuation surface. The organ is
MEANS; pointer 0.19108282 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.boundary_math.length_sigma import resolve_length_sigma_matrix
from tac.canonical_equations.cgauge_parametrization_optima_20260711 import (
    along_tangent_deficit_ratio,
)
from tac.witness_control.lambda_net import N_CLASSES, RidgeSolveAdjoint, lever_features
from tac.witness_control.scorer_geometry import DEFAULT_GT_CACHE
from tac.witness_control.scorer_model_arms import (
    _dilate,
    _ReweightedRidgeArm,
    _shift2d,
    load_comma10k_prior,
    smoothed_dseg_and_grad,
)

#: canonical class order — Road0 / Lane1 / Undrivable2 / Movable3 / MyCar4 (NEVER luma-sort)
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")

#: openpilot ego-geometry model scope: the lane-poly/homography/ego-screw prior explains
#: STATIC-scene advection (ground plane + horizon crack + store-once hood). Movable is
#: outside the model BY DEFINITION (object motion is not ego-explained; L83 carrier law).
OPENPILOT_STATIC_CLASSES = (0, 1, 2, 4)


# ─────────────────────────────────────────────────────────────────────────────
# small pure helpers (numpy-only; no scipy)
# ─────────────────────────────────────────────────────────────────────────────
def _box_smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """(2k+1)² box smoothing via padded cumsum (pure numpy, edge-replicated)."""
    if k <= 0:
        return a.astype(np.float64)
    p = np.pad(a.astype(np.float64), k, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    n = 2 * k + 1
    return (c[n:n + h, n:n + w] - c[:h, n:n + w]
            - c[n:n + h, :w] + c[:h, :w]) / float(n * n)


def _all_class_boundary(lab: np.ndarray) -> np.ndarray:
    """Union of all inter-class edges (the binding residual is the UNION, never Lane-only)."""
    b = np.zeros(lab.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        nb = _shift2d(lab, dy, dx, -1)
        b |= (nb != lab) & (nb >= 0)
    return b


def _pair_boundary(lab: np.ndarray, a: int, b: int) -> np.ndarray:
    pb = np.zeros(lab.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = _shift2d(lab, dy, dx, -1)
        pb |= ((lab == a) & (nb == b)) | ((lab == b) & (nb == a))
    return pb


# deterministic per-process memo caches (the tournament re-constructs each arm per
# fold; the cached-tensor physics is fold-INDEPENDENT — identical inputs, identical
# outputs — so memoizing is purely a wall-clock saving, not a data leak: nothing in
# these profiles touches trajectory data)
_PROFILE_CACHE: dict[tuple, AnisoClassProfiles] = {}
_GRAD_CACHE: dict[tuple, np.ndarray] = {}
_OP_CACHE: dict[tuple, OpenpilotGeometryPrior] = {}


def measure_flip_temperature(cache_path: str | Path | None = None, *,
                             radius_px: int = 1, frame_stride: int = 8) -> float:
    """The MEASURED flip-relevant temperature ε_flip ($0, two independent cached sources).

    Per frame: the actual advection-ball label-change mass (the measured dominant flip
    mode — GT-side sub-pixel advection, L85) rank-matches a margin threshold (exactly
    the ``ball_agreement_audit`` construction); ε_flip = the median threshold over
    frames. WHY THIS AND NOT THE GLOBAL MEDIAN MARGIN: at ε = global median (arm H's
    registered τ knob) the sensitivity field σ'(m/ε) is nearly FLAT — the per-class
    sensitivity shares degenerate to the AREA shares (MEASURED 2026-07-11: shares
    [.233,.007,.494,.016,.250] ≈ areas [.229,.006,.493,.016,.256]) — i.e. classes as
    area knobs, the exact failure mode the P0 directive forbids. The flip-relevant
    regime lives at the margin scale where flips ACTUALLY occur; both scales are
    reported, the backtest arbitrates the arm."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    r = max(int(radius_px), 1)
    thresholds = []
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
        thresholds.append(float(np.partition(m, n_act - 1)[n_act - 1]))
    if not thresholds:
        raise ValueError("no flip mass found in the cache (degenerate labels)")
    return float(np.median(thresholds))


# ─────────────────────────────────────────────────────────────────────────────
# THE MEASURED PHYSICS PROFILES (all from the cached margin field + argmax partition)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AnisoClassProfiles:
    """The per-class ANISOTROPIC dynamical-system profile, measured from cache.

    All susceptibility masses are ε-smoothed flip sensitivity σ'(m/ε)/ε (the EXACT
    gradient field of the relaxed d_seg metric — ARM H's zero-model-error object),
    normalized per-pixel. Advisory, never a score."""

    epsilon: float                              # τ = ε (default: MEASURED flip temperature)
    bulk_susc: tuple[float, ...]                # per-class interior sensitivity mass
    boundary_susc: tuple[float, ...]            # per-class annulus sensitivity mass
    bulk_frac: tuple[float, ...]                # bulk/(bulk+boundary) per class
    total_susc_share: tuple[float, ...]         # class share of TOTAL flip sensitivity
    annulus_area_frac: float                    # annulus area / all pixels (context)
    annulus_susc_frac: float                    # annulus sensitivity / total (context)
    pair_susc: tuple[tuple[float, ...], ...]    # (5,5): class-c sens. mass in (c,c′) band ⊙ 1/σ
    aniso_ratio: tuple[tuple[float, ...], ...]  # (5,5): along-tangent / across-normal energy
    sigma_preset: str
    radius_px: int
    n_frames: int
    source: str
    registered_deficit_ratio: float             # 25/8 = 3.125 (consumed, not re-derived)
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (cached-tensor derived)"

    def coupling_matrix(self) -> np.ndarray:
        """C_phys (rows: responding class, cols: lever-target class). Row-stochastic.

        C_phys[c,:] = bulk_frac_c·e_c + bnd_frac_c·(½·e_c + ½·P_c) with
        P_c ∝ pair_susc[c,:] ⊙ aniso — the ½ split is DERIVED_EXACT from the margin
        symmetry (m = φ_top1 − φ_top2; ±1 coefficients)."""
        ps = np.asarray(self.pair_susc, dtype=np.float64)
        an = np.asarray(self.aniso_ratio, dtype=np.float64)
        # anisotropy gauge: geometric-mean-1 over OBSERVED pairs (mirrors the σ gauge)
        obs = (ps > 0) & (an > 0)
        if obs.any():
            g = float(np.exp(np.mean(np.log(an[obs]))))
            an_n = np.where(obs, an / g, 1.0)
        else:
            an_n = np.ones_like(an)
        W = ps * an_n
        np.fill_diagonal(W, 0.0)
        rows = W.sum(axis=1, keepdims=True)
        P = np.divide(W, rows, out=np.zeros_like(W), where=rows > 0)
        # a class with NO measured pair partners keeps its boundary half on itself
        # (row-stochasticity by construction, never silently lost mass)
        for c in range(N_CLASSES):
            if rows[c, 0] <= 0:
                P[c, c] = 1.0
        bf = np.asarray(self.bulk_frac, dtype=np.float64)
        C = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
        for c in range(N_CLASSES):
            C[c, c] += bf[c] + (1.0 - bf[c]) * 0.5
            C[c, :] += (1.0 - bf[c]) * 0.5 * P[c, :]
        return C

    def to_jsonable(self) -> dict:
        return {
            "class_order": list(CLASS_NAMES),
            "epsilon_tau_hbar": self.epsilon,
            "bulk_susc": list(self.bulk_susc),
            "boundary_susc": list(self.boundary_susc),
            "bulk_frac": list(self.bulk_frac),
            "total_susc_share": list(self.total_susc_share),
            "annulus_area_frac": self.annulus_area_frac,
            "annulus_susc_frac": self.annulus_susc_frac,
            "pair_susc": [list(r) for r in self.pair_susc],
            "aniso_ratio_along_over_across": [list(r) for r in self.aniso_ratio],
            "coupling_matrix": [list(r) for r in self.coupling_matrix()],
            "sigma_preset": self.sigma_preset,
            "radius_px": self.radius_px,
            "n_frames": self.n_frames,
            "registered_along_tangent_deficit": self.registered_deficit_ratio,
            "axis_tag": self.axis_tag,
            "score_claim": False,
        }


def measure_aniso_class_profiles(
        cache_path: str | Path | None = None, *, radius_px: int = 2,
        sigma_preset: str = "fitted-20260707", frame_stride: int = 8,
        epsilon: float | None = None, smooth_k: int = 3) -> AnisoClassProfiles:
    """Measure the per-class anisotropic profiles from the cached scorer geometry ($0).

    Per frame: sensitivity field s' = σ'(m/ε)/ε on the cached margin m; annulus =
    Chebyshev r-dilation of the all-class boundary UNION; per class the bulk/boundary
    split of s'; per pair the band s' mass ⊙ 1/σ_cc′ and the along-tangent vs
    across-normal margin-gradient energy (interface orientation from the smoothed pair
    indicator's gradient)."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    key = (str(p), radius_px, sigma_preset, frame_stride, epsilon, smooth_k)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    sigma = resolve_length_sigma_matrix(sigma_preset)
    if sigma is None:                                  # "all-ones" → uniform tension
        sigma = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    eps = (float(epsilon) if epsilon is not None
           else measure_flip_temperature(p, frame_stride=frame_stride))
    if eps <= 0:
        eps = float(np.median(margins)) or 1.0
    r = max(int(radius_px), 1)

    bulk = np.zeros(N_CLASSES)
    bnd = np.zeros(N_CLASSES)
    pair_s = np.zeros((N_CLASSES, N_CLASSES))
    along_e = np.zeros((N_CLASSES, N_CLASSES))
    across_e = np.zeros((N_CLASSES, N_CLASSES))
    ann_area = 0.0
    tot_area = 0.0
    for f in range(lstars.shape[0]):
        lab = lstars[f]
        m = margins[f]
        s = 1.0 / (1.0 + np.exp(m / eps))
        sprime = s * (1.0 - s) / eps                   # exact ∂σ((δ−m)/ε)/∂δ at δ=0
        annulus = _dilate(_all_class_boundary(lab), r)
        ann_area += float(annulus.sum())
        tot_area += float(lab.size)
        gm_y, gm_x = np.gradient(m)
        for c in range(N_CLASSES):
            cm = lab == c
            bulk[c] += float(sprime[cm & ~annulus].sum())
            bnd[c] += float(sprime[cm & annulus].sum())
        for a in range(N_CLASSES):
            for b in range(a + 1, N_CLASSES):
                pb = _pair_boundary(lab, a, b)
                if not pb.any():
                    continue
                band = _dilate(pb, r)
                w = 1.0 / float(sigma[a, b])
                pair_s[a, b] += w * float(sprime[band & (lab == a)].sum())
                pair_s[b, a] += w * float(sprime[band & (lab == b)].sum())
                # interface orientation: gradient of the smoothed ±1 pair indicator
                d = np.where(lab == a, 1.0, np.where(lab == b, -1.0, 0.0))
                ds = _box_smooth(d, smooth_k)
                ny, nx = np.gradient(ds)
                nn = np.sqrt(ny * ny + nx * nx)
                ok = band & (nn > 1e-9)
                if not ok.any():
                    continue
                nyu, nxu = ny[ok] / nn[ok], nx[ok] / nn[ok]
                # tangent = normal rotated 90°
                al = (gm_y[ok] * (-nxu) + gm_x[ok] * nyu) ** 2
                ac = (gm_y[ok] * nyu + gm_x[ok] * nxu) ** 2
                along_e[a, b] += float(al.sum())
                along_e[b, a] += float(al.sum())
                across_e[a, b] += float(ac.sum())
                across_e[b, a] += float(ac.sum())

    tot = float(bulk.sum() + bnd.sum()) or 1.0
    denom = np.maximum(bulk + bnd, 1e-30)
    aniso = np.divide(along_e, across_e,
                      out=np.zeros_like(along_e), where=across_e > 0)
    out = AnisoClassProfiles(
        epsilon=eps,
        bulk_susc=tuple(float(v) for v in bulk),
        boundary_susc=tuple(float(v) for v in bnd),
        bulk_frac=tuple(float(v) for v in bulk / denom),
        total_susc_share=tuple(float(v) for v in (bulk + bnd) / tot),
        annulus_area_frac=float(ann_area / max(tot_area, 1.0)),
        annulus_susc_frac=float(bnd.sum() / tot),
        pair_susc=tuple(tuple(float(v) for v in row) for row in pair_s),
        aniso_ratio=tuple(tuple(float(v) for v in row) for row in aniso),
        sigma_preset=str(sigma_preset), radius_px=r,
        n_frames=int(lstars.shape[0]), source=str(p),
        registered_deficit_ratio=float(along_tangent_deficit_ratio()))
    _PROFILE_CACHE[key] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OPENPILOT ISOLATED GEOMETRY PRIOR (thread 2 — never bundled with comma10k/regime)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class OpenpilotGeometryPrior:
    """Ego-geometry addressability per class, measured from the cached partition.

    The openpilot physical prior (lane-poly + homography ground plane + ego-screw ξ +
    store-once hood) explains STATIC-scene advection below the horizon plus the
    horizon crack itself (v_horizon(ξ), L83). Per class: the fraction of its flip
    sensitivity that lies in the ego-addressable region. Movable = 0 BY MODEL SCOPE
    (object motion is not ego-explained), stated not hidden."""

    horizon_row: int
    hood_top_row: int
    horizon_band_px: int
    addressable_frac: tuple[float, ...]        # per class ∈ [0,1]
    addressable_susc: tuple[float, ...]        # per class sensitivity mass in-region
    n_frames: int
    source: str
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE (cached-tensor derived)"

    def class_reweight(self) -> np.ndarray:
        s = np.asarray(self.addressable_susc, dtype=np.float64)
        tot = float(s.sum())
        return s * (N_CLASSES / tot) if tot > 0 else np.ones(N_CLASSES)

    def to_jsonable(self) -> dict:
        return {
            "class_order": list(CLASS_NAMES),
            "horizon_row": self.horizon_row, "hood_top_row": self.hood_top_row,
            "horizon_band_px": self.horizon_band_px,
            "addressable_frac": list(self.addressable_frac),
            "addressable_susc": list(self.addressable_susc),
            "movable_scope_note": "Movable excluded BY MODEL SCOPE (ego motion does "
                                  "not explain object motion; L83 movable→event)",
            "n_frames": self.n_frames, "axis_tag": self.axis_tag,
            "score_claim": False,
        }


def measure_openpilot_geometry_prior(
        cache_path: str | Path | None = None, *, frame_stride: int = 8,
        horizon_band_px: int = 4, epsilon: float | None = None,
) -> OpenpilotGeometryPrior:
    """Measure the openpilot ego-addressable susceptibility per class ($0, cached).

    Horizon row per frame = deepest row whose row-mean Undrivable occupancy ≥ 0.5
    (the top sky/offroad block's lower edge); hood-top row = shallowest row whose
    row-mean MyCar occupancy ≥ 0.5 (the static hood block's upper edge, IoU 0.994
    static per the measured class signatures). Ego-addressable region = rows below
    (horizon − band) — the ground plane, the horizon crack band, and the hood."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    key = (str(p), frame_stride, horizon_band_px, epsilon)
    if key in _OP_CACHE:
        return _OP_CACHE[key]
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    eps = (float(epsilon) if epsilon is not None
           else measure_flip_temperature(p, frame_stride=frame_stride))
    if eps <= 0:
        eps = float(np.median(margins)) or 1.0
    h = lstars.shape[1]
    horizons, hoods = [], []
    addr_mass = np.zeros(N_CLASSES)
    total_mass = np.zeros(N_CLASSES)
    for f in range(lstars.shape[0]):
        lab = lstars[f]
        m = margins[f]
        s = 1.0 / (1.0 + np.exp(m / eps))
        sprime = s * (1.0 - s) / eps
        undriv = (lab == 2).mean(axis=1)
        rows2 = np.where(undriv >= 0.5)[0]
        horizon = int(rows2.max()) if rows2.size else int(0.35 * h)
        mycar = (lab == 4).mean(axis=1)
        rows4 = np.where(mycar >= 0.5)[0]
        hood_top = int(rows4.min()) if rows4.size else h
        horizons.append(horizon)
        hoods.append(hood_top)
        region = np.zeros(lab.shape, dtype=bool)
        region[max(horizon - int(horizon_band_px), 0):, :] = True
        for c in range(N_CLASSES):
            cm = lab == c
            total_mass[c] += float(sprime[cm].sum())
            if c in OPENPILOT_STATIC_CLASSES:
                addr_mass[c] += float(sprime[cm & region].sum())
    frac = np.divide(addr_mass, np.maximum(total_mass, 1e-30))
    out = OpenpilotGeometryPrior(
        horizon_row=int(np.median(horizons)), hood_top_row=int(np.median(hoods)),
        horizon_band_px=int(horizon_band_px),
        addressable_frac=tuple(float(v) for v in frac),
        addressable_susc=tuple(float(v) for v in addr_mass),
        n_frames=int(lstars.shape[0]), source=str(p))
    _OP_CACHE[key] = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SCORE-LAW-PINNED PRIOR-MEAN RIDGE (the non-absorbable physics formulation)
# ─────────────────────────────────────────────────────────────────────────────
class PhysicsPriorMeanAdjoint:
    """Ridge SOLVE shrunk toward a PHYSICS-STRUCTURED prior-mean response.

    The measured cure chain: (i) φ-rescale/coupling into an unconstrained re-fit is
    ABSORBED (G/H/I/J/K neutrality) → the prior must be the ridge TARGET; (ii) the
    7-parameter empirical-Bayes (b, κ) lstsq at n=2 INJECTS variance (arm L failure) →
    here the prior's DIRECTION + RELATIVE per-class magnitudes are fully pinned by the
    physics matrix ``m0_unit`` (score-law/Fisher/σ_cc′ content), the intercept prior is
    the per-channel MEDIAN drift (robust order statistic), and exactly ONE global gain
    κ ≥ 0 is fit by a 1-dof projection. coef = argmin ‖Φc − Y‖² + r·scale·‖c − c0‖²."""

    name = "physics_priormean"

    def __init__(self, m0_unit: np.ndarray, ridge: float = 1e-2):
        M = np.asarray(m0_unit, dtype=np.float64)
        if M.shape != (N_CLASSES, N_CLASSES):
            raise ValueError(f"m0_unit must be (5,5), got {M.shape}")
        if not np.all(np.isfinite(M)) or np.any(M < 0):
            raise ValueError("m0_unit must be finite and non-negative")
        tot = float(M.sum())
        if tot <= 0:
            raise ValueError("m0_unit must have positive mass")
        # gauge: total mass N_CLASSES (matches the reweight convention) so κ is comparable
        self.m0_unit = M * (N_CLASSES / tot)
        self.ridge = float(ridge)
        self.coef: np.ndarray | None = None
        self.kappa: float | None = None
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
        b0 = np.median(Y, axis=0)                            # robust per-channel drift
        # 1-dof κ: residual (class channels) projected on the physics drive direction
        drive = -(occ_cls @ self.m0_unit.T)                  # (N, 5): −M0u @ occ per row
        resid = Y[:, :N_CLASSES] - b0[None, :N_CLASSES]
        num = float(np.sum(drive * resid))
        den = float(np.sum(drive * drive))
        kappa = max(num / den, 0.0) if den > 0 else 0.0
        p = Phi.shape[1]
        coef0 = np.zeros((p, S))
        coef0[0, :] = b0
        for c in range(N_CLASSES):          # response channel
            for cp in range(N_CLASSES):     # lever-target class (occ feature)
                coef0[1 + S + cp, c] = -kappa * self.m0_unit[c, cp]
        gram = Phi.T @ Phi
        scale = float(np.mean(np.diag(gram))) or 1.0
        r = self.ridge * scale
        self.coef = np.linalg.solve(gram + r * np.eye(p), Phi.T @ Y + r * coef0)
        self.kappa = kappa

    def response(self, x, ctx, phi, path=None) -> np.ndarray:
        assert self.coef is not None, "fit first"
        return self.coef[1 + self._state_dim:].T @ np.asarray(phi, dtype=np.float64)

    def base(self, x, ctx, path=None) -> np.ndarray:
        assert self.coef is not None, "fit first"
        return self.coef[0] + self.coef[1:1 + self._state_dim].T @ np.asarray(
            x, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# the physics ingredients composed into M0_unit matrices (per arm)
# ─────────────────────────────────────────────────────────────────────────────
def smoothed_grad_per_class(cache_path: str | Path | None = None, *,
                            epsilon: float | None = None,
                            frame_stride: int = 4) -> np.ndarray:
    """ARM H's EXACT ∂(smoothed d_seg)/∂δ_c (the Fisher/impact half), as a (5,) vector."""
    p = Path(cache_path) if cache_path else DEFAULT_GT_CACHE
    if not p.exists():
        raise FileNotFoundError(f"scorer-geometry cache absent: {p}")
    key = (str(p), epsilon, frame_stride)
    if key in _GRAD_CACHE:
        return _GRAD_CACHE[key].copy()
    z = np.load(p)
    step = max(int(frame_stride), 1)
    lstars = z["lstars"][::step]
    margins = z["margins"][::step].astype(np.float64)
    eps = (float(epsilon) if epsilon is not None
           else measure_flip_temperature(p, frame_stride=frame_stride))
    if eps <= 0:
        eps = float(np.median(margins)) or 1.0
    _, grad = smoothed_dseg_and_grad(margins, lstars, eps)
    _GRAD_CACHE[key] = grad
    return grad.copy()


def aniso_coupled_m0(profiles: AnisoClassProfiles,
                     grad_per_class: np.ndarray) -> np.ndarray:
    """M0_unit[c,c′] = C_phys[c,c′] · g^ε_c′ — a lever on class c′ shifts c′ logits
    (impact g^ε_c′, the EXACT smoothed-metric gradient), and the flip response
    propagates to class c through the measured bulk/boundary anisotropic coupling."""
    g = np.asarray(grad_per_class, dtype=np.float64)
    if g.shape != (N_CLASSES,):
        raise ValueError(f"grad_per_class must be (5,), got {g.shape}")
    return profiles.coupling_matrix() * g[None, :]


def c10k_scorelaw_m0(profiles: AnisoClassProfiles, grad_per_class: np.ndarray,
                     rarity: np.ndarray) -> np.ndarray:
    """The OPEN comma10k-family member (thread 1): rarity DIRECTION (which classes are
    under-trained) × Fisher impact × σ_cc′-anisotropic coupling; the SCALE is pinned by
    the shared 1-dof κ machinery — NOT the failed empirical-Bayes fit."""
    r = np.asarray(rarity, dtype=np.float64)
    if r.shape != (N_CLASSES,):
        raise ValueError(f"rarity must be (5,), got {r.shape}")
    g = np.asarray(grad_per_class, dtype=np.float64)
    return profiles.coupling_matrix() * (r * g)[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# the tournament arms
# ─────────────────────────────────────────────────────────────────────────────
class AnisoCoupledRidgeAdjoint(_ReweightedRidgeArm):
    """ARM N — coupling-reweight formulation of the anisotropic per-class physics
    (EXPECTED-NEAR-NEUTRAL: the re-fit absorbs φ recalibrations — measured; shipped for
    the honest comparison row + the per-class λ readout, admitted by backtest only)."""

    name = "N_aniso_coupled"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 profiles: AnisoClassProfiles | None = None):
        super().__init__(ridge=ridge)
        pr = profiles if profiles is not None else measure_aniso_class_profiles(cache_path)
        self.profiles = pr
        C = pr.coupling_matrix()
        g = smoothed_grad_per_class(cache_path)
        gt = float(g.sum())
        self._coupling = C
        self._weight = g * (N_CLASSES / gt) if gt > 0 else np.ones(N_CLASSES)

    def perclass_lambda(self, model_response: np.ndarray,
                        grad_s: np.ndarray) -> np.ndarray:
        """Per-class marginal-ΔS readout (the #430 composer contract, K-compatible):
        λ_c = g_c · r_c after propagating the response through C_phys."""
        r = self._coupling @ np.asarray(model_response, dtype=np.float64)[:N_CLASSES]
        g = np.asarray(grad_s, dtype=np.float64)[:N_CLASSES]
        return g * r


class AnisoPriorMeanAdjoint(PhysicsPriorMeanAdjoint):
    """ARM P — THE P0 ARM: shrink-to-prior ridge whose prior mean is the full
    anisotropic-coupled score-law-pinned physics (C_phys ∘ exact smoothed gradient)."""

    name = "P_priormean_aniso"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 profiles: AnisoClassProfiles | None = None):
        pr = profiles if profiles is not None else measure_aniso_class_profiles(cache_path)
        g = smoothed_grad_per_class(cache_path)
        super().__init__(aniso_coupled_m0(pr, g), ridge=ridge)
        self.profiles = pr
        self.grad_per_class = g


class IsoPriorMeanAdjoint(PhysicsPriorMeanAdjoint):
    """ARM Q — the ISOTROPIC-INDEPENDENT ablation of P: identical formulation and κ
    machinery, M0_unit = I (no coupling, no per-class physics). The honest A/B that
    isolates the anisotropy+coupling content."""

    name = "Q_priormean_iso"

    def __init__(self, ridge: float = 1e-2):
        super().__init__(np.eye(N_CLASSES), ridge=ridge)


class C10kScoreLawPriorMeanAdjoint(PhysicsPriorMeanAdjoint):
    """ARM R — the OPEN comma10k-family member (thread 1): rarity direction with the
    score-law-pinned scale + σ_cc′/Fisher anisotropic coupling (the named cure for the
    measured empirical-Bayes-κ failure of arm L)."""

    name = "R_priormean_c10k_scorelaw"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 prior_path: str | None = None,
                 profiles: AnisoClassProfiles | None = None):
        pr = profiles if profiles is not None else measure_aniso_class_profiles(cache_path)
        g = smoothed_grad_per_class(cache_path)
        prior = load_comma10k_prior(prior_path)
        super().__init__(c10k_scorelaw_m0(pr, g, prior.rarity_reweight()), ridge=ridge)
        self.profiles = pr
        self.c10k_prior = prior


class OpenpilotGeomRidgeAdjoint(_ReweightedRidgeArm):
    """ARM O — openpilot ISOLATED arm, reweight formulation (expected-near-neutral by
    the measured absorption mechanism; shipped so the isolated verdict covers both
    formulations)."""

    name = "O_openpilot_geom"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 prior: OpenpilotGeometryPrior | None = None):
        super().__init__(ridge=ridge)
        pr = prior if prior is not None else measure_openpilot_geometry_prior(cache_path)
        self.prior = pr
        self._weight = pr.class_reweight()


class OpenpilotPriorMeanAdjoint(PhysicsPriorMeanAdjoint):
    """ARM S — openpilot ISOLATED arm, prior-mean formulation (the non-absorbable
    test): M0_unit = diag(ego-addressable susceptibility). Isolated = no comma10k, no
    cross-class coupling — the clean openpilot-alone verdict."""

    name = "S_priormean_openpilot"

    def __init__(self, ridge: float = 1e-2, cache_path: str | None = None,
                 prior: OpenpilotGeometryPrior | None = None):
        pr = prior if prior is not None else measure_openpilot_geometry_prior(cache_path)
        super().__init__(np.diag(pr.class_reweight()), ridge=ridge)
        self.prior = pr


# ─────────────────────────────────────────────────────────────────────────────
# SAO-style single-anchor trust region on Λ between walk-forward refits (thread 3)
# ─────────────────────────────────────────────────────────────────────────────
def sao_trustregion_walkforward(traj, *, radii: tuple[float, ...] = (0.25, 0.5, 1.0),
                                ridge: float = 1e-2, seed: int = 0) -> dict:
    """SAO-style trust region: between consecutive walk-forward refits, the ridge
    coefficient update is clamped to ‖Δcoef‖_F ≤ radius·‖coef_prev‖_F (single-rollout
    anchored update — the envelope §7 growth-spine item, now MEASURED). All radii are
    pre-registered and ALL reported (no tuning-until-it-wins). Returns per-radius
    walk-forward MAE vs the unclamped ridge and the persistence heuristic."""
    from tac.witness_control.lambda_net import (
        _predict_interval,
        build_intervals,
        fit_score_composition,
    )
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 4:
        raise ValueError(f"need ≥4 intervals; have {len(intervals)}")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    wcls = comp.class_weights

    def _stack(m: RidgeSolveAdjoint) -> np.ndarray:
        return np.concatenate([m.a[None, :], m.C.T, m.M.T], axis=0)

    def _unstack(coef: np.ndarray, ridge_: float) -> RidgeSolveAdjoint:
        m = RidgeSolveAdjoint(ridge=ridge_)
        S = N_CLASSES + 1
        m.a = coef[0].copy()
        m.C = coef[1:1 + S].T.copy()
        m.M = coef[1 + S:].T.copy()
        return m

    out: dict = {"radii": {}, "axis_tag": "[macOS advisory] NON-PROMOTABLE",
                 "score_claim": False}
    errs_plain, errs_heur = [], []
    per_radius_errs: dict[float, list[float]] = {r: [] for r in radii}
    prev: dict[float, np.ndarray | None] = dict.fromkeys(radii)
    for hold in range(2, len(intervals)):
        model = RidgeSolveAdjoint(ridge=ridge)
        model.fit(intervals[:hold], phis, seed=seed)
        iv = intervals[hold]
        meas = iv.dxdt()
        heur = intervals[hold - 1].dxdt()
        pred_plain = _predict_interval(model, "A_ridge_solve", iv, traj.lever_names)
        errs_plain.append(
            abs(float(wcls @ (pred_plain[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        errs_heur.append(
            abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
        coef_new = _stack(model)
        for r in radii:
            if prev[r] is None:
                coef_used = coef_new
            else:
                delta = coef_new - prev[r]
                fn = float(np.linalg.norm(delta))
                cap = r * max(float(np.linalg.norm(prev[r])), 1e-12)
                coef_used = prev[r] + (delta * (cap / fn) if fn > cap else delta)
            prev[r] = coef_used
            m_tr = _unstack(coef_used, ridge)
            pred = _predict_interval(m_tr, "A_ridge_solve", iv, traj.lever_names)
            per_radius_errs[r].append(
                abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
    out["wf_mae_ridge_plain"] = float(np.mean(errs_plain))
    out["wf_mae_persistence"] = float(np.mean(errs_heur))
    for r in radii:
        out["radii"][str(r)] = float(np.mean(per_radius_errs[r]))
    out["n_folds"] = len(errs_plain)
    out["protocol"] = ("pre-registered radii, all reported; trust region clamps the "
                      "Frobenius norm of the ridge coefficient update between "
                      "walk-forward refits (SAO single-anchor spirit)")
    return out


__all__ = [
    "CLASS_NAMES",
    "OPENPILOT_STATIC_CLASSES",
    "AnisoClassProfiles",
    "AnisoCoupledRidgeAdjoint",
    "AnisoPriorMeanAdjoint",
    "C10kScoreLawPriorMeanAdjoint",
    "IsoPriorMeanAdjoint",
    "OpenpilotGeomRidgeAdjoint",
    "OpenpilotGeometryPrior",
    "OpenpilotPriorMeanAdjoint",
    "PhysicsPriorMeanAdjoint",
    "aniso_coupled_m0",
    "c10k_scorelaw_m0",
    "measure_aniso_class_profiles",
    "measure_openpilot_geometry_prior",
    "sao_trustregion_walkforward",
    "smoothed_grad_per_class",
]
