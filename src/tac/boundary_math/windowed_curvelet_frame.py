# SPDX-License-Identifier: MIT
"""Genuinely localized windowed-curvelet / directional-Gabor frame (task #502).

WHY THIS EXISTS (NO-FAKE, catalog #351). The witness's current oriented basis
(``tac.boundary_math.lever_b_levelset_generator.polar_directional_fourier_*``)
is GLOBAL plane waves: ``[sin(2*pi X@B), cos(2*pi X@B)]`` with paired envelope
``sin^2 + cos^2 == 1`` at every coordinate -> NOT spatially localized (audited by
``optimal_basis_20260714.audit_legacy_polar_bank``: envelope span ~1e-7). Calling
that a "curvelet" is precisely the catalog-#351 fake class. This module builds a
frame that is GENUINELY localized: each atom is a spatially windowed, oriented
oscillation with parabolic (curvelet) anisotropic scaling, so its paired envelope
is a spatial Gaussian bump (large span), it has an explicit translation index, and
its support is elongated along the boundary tangent (width ~ length^2). It PASSES a
localization certificate that a plain oriented-Fourier basis structurally FAILS
(``localization_certificate`` runs both and asserts the discrimination) -- the
swap-test the anti-fake rule demands.

MATH (second-generation curvelet, spatial-atom realization = directional Gabor
wavelet with parabolic scaling; Candes-Donoho 2004). Atom (scale j, orientation l,
center c):

    theta   = pi * l / L_j          L_j = n_orient0 * 2**(j//2)   (angular doubling)
    f_j     = f0 * base**j                                        (radial octave)
    u_n     =  (x-cx) cos(theta) + (y-cy) sin(theta)   # across the boundary (normal)
    u_t     = -(x-cx) sin(theta) + (y-cy) cos(theta)   # along the boundary (tangent)
    sigma_n = w0 * r**(-j)          sigma_t = w0 * r**(-j/2)
    env     = exp(-0.5 * ((u_n/sigma_n)**2 + (u_t/sigma_t)**2))
    real    = env * cos(2*pi f_j * u_n)     imag = env * sin(2*pi f_j * u_n)

Parabolic scaling: sigma_t = sqrt(w0 * sigma_n)  =>  sigma_n ~ sigma_t**2, i.e. the
support width (across) scales as the length (along) squared -- the curvelet law that
gives optimal N-term approximation for C2 curved edges. The oscillation runs ACROSS
the boundary (normal) at f_j while the envelope is thin across / long along -- an
atom shaped like an oriented edge segment.

The bank is fully determined by a handful of GENERIC scalars (no seed, no video
data): reconstructed bit-identically at decode -> rule-118 FREE. Only learned/
selected coefficients (the per-atom weights a trained witness would store) are
counted. This module supplies the frame ONLY; wiring it into the trainer forward
and the generated ``inflate.py`` (op-parity) + a real n600 through-R d_seg receipt
is the OWED heavier step (needs a run; operator-GO / CONTAINMENT).

Portability: the forward is written against a generic array module (``numpy`` is the
deterministic fp64 authority; ``mlx`` mirrors it elementwise). ``mlx_parity_check``
proves the two agree.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WindowedCurveletConfig:
    """Generic (seed-free, video-free) windowed-curvelet frame parameters.

    n_scales J radial octaves; n_orient0 L0 orientations at the coarsest scale with
    curvelet angular doubling L_j = L0 * 2**(j//2); f0/base the radial frequency
    octaves; w0 the coarsest envelope width; width_ratio r the per-octave envelope
    shrink (parabolic: sigma_t = sqrt(w0 * sigma_n)); n_trans translations PER AXIS
    (a J-independent tiling grid over the coord bounding box).  min_sigma clamps the
    finest envelope so atoms never collapse below the sampling grid.

    Fully determined by these scalars -> bit-identical regeneration at decode, zero
    archive bytes, no GT leak.
    """

    n_scales: int = 4
    n_orient0: int = 6
    f0: float = 2.0
    base: float = 2.0
    w0: float = 0.5
    width_ratio: float = 2.0
    n_trans: int = 2
    coord_margin: float = 0.5
    min_sigma: float = 0.02
    aniso: float = 1.0  # constant elongation boost on sigma_t; PRESERVES the parabolic
    #                     law in j (sigma_t = aniso*sqrt(w0*sigma_n) -> sigma_n ~ sigma_t^2).
    #                     aniso>1 makes atoms edge-shaped (long along tangent) at every scale.

    def __post_init__(self) -> None:
        for name in ("n_scales", "n_orient0", "n_trans"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        for name in ("f0", "base", "w0", "width_ratio", "min_sigma", "aniso"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
        if float(self.base) <= 1.0 or float(self.width_ratio) <= 1.0:
            raise ValueError("base and width_ratio must exceed 1 (radial/parabolic octaves)")
        if float(self.aniso) < 1.0:
            raise ValueError("aniso must be >= 1 (tangent-elongation boost)")
        if not math.isfinite(float(self.coord_margin)) or float(self.coord_margin) < 0.0:
            raise ValueError("coord_margin must be finite and >= 0")


@dataclass(frozen=True)
class AtomIndex:
    """One frame atom's deterministic index (scale, orientation, center)."""

    scale: int
    orient: int
    theta: float
    freq: float
    sigma_n: float
    sigma_t: float
    cx: float
    cy: float


def _sigma_pair(cfg: WindowedCurveletConfig, j: int) -> tuple[float, float]:
    r = float(cfg.width_ratio)
    sigma_n = max(float(cfg.w0) * r ** (-j), float(cfg.min_sigma))
    sigma_t = max(float(cfg.aniso) * float(cfg.w0) * r ** (-0.5 * j), float(cfg.min_sigma))
    return sigma_n, sigma_t


def curvelet_atom_index(cfg: WindowedCurveletConfig) -> tuple[AtomIndex, ...]:
    """Deterministic atom list (scale x orientation x translation grid).

    Order is stable: for j, for l, for (ty, tx) row-major over the translation grid.
    This order IS the counted-coefficient order at decode.
    """

    m = float(cfg.coord_margin)
    lo, hi = -1.0 + m, 1.0 - m
    if cfg.n_trans == 1:
        centers = [0.0]
    else:
        centers = list(np.linspace(lo, hi, cfg.n_trans))
    atoms: list[AtomIndex] = []
    for j in range(int(cfg.n_scales)):
        f_j = float(cfg.f0) * (float(cfg.base) ** j)
        l_j = int(cfg.n_orient0) * (2 ** (j // 2))
        sigma_n, sigma_t = _sigma_pair(cfg, j)
        for orient in range(l_j):
            theta = math.pi * orient / l_j
            for cy in centers:
                for cx in centers:
                    atoms.append(
                        AtomIndex(j, orient, theta, f_j, sigma_n, sigma_t, float(cx), float(cy))
                    )
    return tuple(atoms)


def windowed_curvelet_feats(coords: np.ndarray, cfg: WindowedCurveletConfig, *, xp=np) -> np.ndarray:
    """Return (P, 2*D) real frame features [real_cols | imag_cols].

    Columns 0..D-1 are ``env*cos(2*pi f u_n)``; columns D..2D-1 are
    ``env*sin(2*pi f u_n)`` for the SAME atom order (``curvelet_atom_index``). The
    paired envelope real^2 + imag^2 = env^2 is a spatial Gaussian bump (localized).

    ``xp`` is the array module (numpy = fp64 authority, or mlx). numpy computes in
    fp64 then casts to fp32 (matching the polar-Fourier reference convention).
    """

    atoms = curvelet_atom_index(cfg)
    x = xp.asarray(coords)[:, 0]
    y = xp.asarray(coords)[:, 1]
    two_pi = 2.0 * math.pi
    real_cols = []
    imag_cols = []
    for a in atoms:
        ct, st = math.cos(a.theta), math.sin(a.theta)
        dx = x - a.cx
        dy = y - a.cy
        u_n = dx * ct + dy * st
        u_t = -dx * st + dy * ct
        env = xp.exp(-0.5 * ((u_n / a.sigma_n) ** 2 + (u_t / a.sigma_t) ** 2))
        phase = two_pi * a.freq * u_n
        real_cols.append(env * xp.cos(phase))
        imag_cols.append(env * xp.sin(phase))
    real = xp.stack(real_cols, axis=-1)
    imag = xp.stack(imag_cols, axis=-1)
    out = xp.concatenate([real, imag], axis=-1)
    if xp is np:
        return np.asarray(out, dtype=np.float32)
    return out


def n_atoms(cfg: WindowedCurveletConfig) -> int:
    """Number of atoms D (half the feature columns 2D)."""

    return len(curvelet_atom_index(cfg))


# ---------------------------------------------------------------------------
# Localization certificate -- the catalog-#351 swap-test.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizationCertificate:
    """Per-atom spatial-window evidence that discriminates a real frame from Fourier.

    curvelet_envelope_span: max over atoms of ptp(env^2) across the coord grid.  A
        genuinely localized atom's paired envelope is a Gaussian bump -> span ~ O(1).
    fourier_envelope_span: the same statistic for the polar-Fourier bank (constant
        envelope 1 -> span ~ 1e-7).
    curvelet_energy_concentration: median over atoms of the fraction of an atom's
        L2 energy contained in the 10% highest-magnitude coordinates (localized -> ~1).
    tangent_over_normal_extent: median over atoms of realized sigma_t_eff / sigma_n_eff
        (2nd spatial moments of env^2) -> > 1 means elongated ALONG the tangent.
    parabolic_scaling_monotone: sigma_n shrinks strictly faster than sigma_t across
        scale (the width ~ length^2 curvelet law) -- checked on the config.
    passes: True iff curvelet is localized (span, concentration) AND a plain oriented
        Fourier basis would FAIL the same span gate (the swap-test).
    """

    curvelet_envelope_span: float
    fourier_envelope_span: float
    curvelet_energy_concentration: float
    tangent_over_normal_extent: float
    parabolic_scaling_monotone: bool
    span_gate: float
    concentration_gate: float
    passes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _envelope_energy_stats(feats: np.ndarray, coords: np.ndarray) -> tuple[float, float, float]:
    """Return (max envelope span, median energy concentration@10%, median sigma_t/sigma_n)."""

    D = feats.shape[1] // 2
    real = feats[:, :D].astype(np.float64)
    imag = feats[:, D:].astype(np.float64)
    env = real ** 2 + imag ** 2  # (P, D) paired window energy
    spans = np.ptp(env, axis=0)  # per atom
    max_span = float(spans.max(initial=0.0))

    # energy concentration: fraction of each atom's energy in its top-10% pixels.
    P = env.shape[0]
    k = max(1, int(round(0.10 * P)))
    concs = []
    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    ratios = []
    for i in range(D):
        e = env[:, i]
        tot = e.sum()
        if tot <= 1e-12:
            continue
        top = np.sort(e)[-k:].sum()
        concs.append(top / tot)
        # realized 2nd moments about the energy centroid, projected on principal axes.
        w = e / tot
        cx = float((w * x).sum())
        cy = float((w * y).sum())
        vx = float((w * (x - cx) ** 2).sum())
        vy = float((w * (y - cy) ** 2).sum())
        vxy = float((w * (x - cx) * (y - cy)).sum())
        cov = np.array([[vx, vxy], [vxy, vy]], dtype=np.float64)
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, 1e-18, None)
        ratios.append(math.sqrt(evals[1] / evals[0]))  # major / minor extent
    conc = float(np.median(concs)) if concs else 0.0
    ratio = float(np.median(ratios)) if ratios else 1.0
    return max_span, conc, ratio


def localization_certificate(
    cfg: WindowedCurveletConfig,
    *,
    height: int = 33,
    width: int = 33,
    span_gate: float = 0.10,
    concentration_gate: float = 0.30,
) -> LocalizationCertificate:
    """Prove the frame is localized AND that oriented Fourier fails the same gate.

    Builds both dictionaries on the SAME coord grid and compares. A localized
    curvelet atom concentrates energy (span/concentration high); the polar-Fourier
    bank's paired envelope is constant (span ~ 1e-7) -> it fails the span gate. If a
    future edit silently swaps in oriented Fourier here, ``passes`` flips to False.
    """

    from tac.boundary_math.lever_b_levelset_generator import (
        PolarDirectionalFourierBankConfig,
        build_coords,
        polar_directional_fourier_B,
        polar_directional_fourier_feats,
    )

    coords = build_coords(height, width)
    cfeats = windowed_curvelet_feats(coords, cfg)
    c_span, c_conc, c_ratio = _envelope_energy_stats(cfeats, coords)

    ffeats = polar_directional_fourier_feats(
        coords, polar_directional_fourier_B(PolarDirectionalFourierBankConfig())
    )
    f_span, _, _ = _envelope_energy_stats(ffeats, coords)

    # parabolic monotonicity: sigma_n must shrink strictly faster than sigma_t.
    mono = True
    prev = None
    for j in range(int(cfg.n_scales)):
        sn, st = _sigma_pair(cfg, j)
        ratio_j = sn / st  # = r**(-j/2) -> strictly decreasing in j (curvelet law)
        if prev is not None and not (ratio_j < prev - 1e-9):
            mono = False
        prev = ratio_j

    curvelet_localized = (c_span >= span_gate) and (c_conc >= concentration_gate)
    fourier_fails = f_span < span_gate
    passes = bool(curvelet_localized and fourier_fails and c_ratio > 1.0 and mono)
    return LocalizationCertificate(
        curvelet_envelope_span=c_span,
        fourier_envelope_span=f_span,
        curvelet_energy_concentration=c_conc,
        tangent_over_normal_extent=c_ratio,
        parabolic_scaling_monotone=mono,
        span_gate=span_gate,
        concentration_gate=concentration_gate,
        passes=passes,
    )


def mlx_parity_check(cfg: WindowedCurveletConfig, *, height: int = 17, width: int = 17,
                     tol: float = 1e-4) -> dict[str, Any]:
    """Confirm the mlx forward matches the numpy fp64->fp32 authority (if mlx present)."""

    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(height, width)
    ref = windowed_curvelet_feats(coords, cfg)  # numpy authority
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # pragma: no cover - mlx optional
        return {"mlx_available": False, "reason": str(exc)}
    mfeats = windowed_curvelet_feats(mx.array(coords), cfg, xp=mx)
    got = np.asarray(mfeats, dtype=np.float32)
    max_abs = float(np.max(np.abs(got - ref)))
    return {"mlx_available": True, "max_abs_diff": max_abs, "within_tol": bool(max_abs <= tol)}


__all__ = [
    "AtomIndex",
    "LocalizationCertificate",
    "WindowedCurveletConfig",
    "curvelet_atom_index",
    "localization_certificate",
    "mlx_parity_check",
    "n_atoms",
    "windowed_curvelet_feats",
]
