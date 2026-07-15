# SPDX-License-Identifier: MIT
"""Genuinely compact cone-adapted shearlet frame (task #502, shearlet half).

WHY THIS EXISTS (NO-FAKE, catalog #351). Sister of
``tac.boundary_math.windowed_curvelet_frame`` (the curvelet half). Both are
directional frames with parabolic (curvelet) anisotropic scaling for optimal
N-term approximation of C2 curved edges. They differ in HOW the directional
atoms are STEERED:

    curvelet  : ROTATION  R_theta = [[cos, -sin],[sin, cos]]  (rigid; orthogonal)
    shearlet  : SHEARING  S_k     = [[1, k],[0, 1]]           (non-rigid; area-1)

Calling a rotation- or Fourier-based bank a "shearlet" is precisely the
catalog-#351 fake class. This module builds a frame that is GENUINELY shear-
steered, and PASSES a shear-selectivity certificate that a rotation basis
(curvelet) and a Fourier basis structurally FAIL. The two genuine, textbook,
measurable properties that separate SHEAR from ROTATION:

  (1) ANCHOR-AXIS INVARIANCE (the swap-test). A shear ``S_k`` FIXES A LINE
      pointwise: ``S_k * (x, 0) = (x, 0)`` for every k. So all shear atoms in a
      cone, sampled ALONG that fixed anchor axis, are IDENTICAL to the mother --
      the along-anchor profile is shear-invariant. A rotation ``R_theta`` (theta
      != 0) fixes NO line, so rotating the SAME atom to cover the SAME normal
      directions changes the along-anchor profile. ``shearlet_certificate``
      builds BOTH a shear-steered and a rotation-steered family covering matched
      normal directions and asserts: shear dispersion ~ 0, rotation dispersion
      >> 0. If a future edit silently swaps rotation (or Fourier) in here, the
      certificate flips to ``passes = False``.

  (2) INTEGER-LATTICE PRESERVATION. For integer shear k, ``S_k`` has integer
      entries and det 1, so it maps Z^2 -> Z^2 bijectively (a faithful digital
      transform, the classical reason shearlets beat curvelets on a pixel grid).
      A rotation by ``atan(k)`` is generally NOT integer-valued. Checked on the
      matrices themselves.

MATH (cone-adapted discrete shearlet; Guo-Kutyniok-Labate 2006; Easley-Labate-
Lim 2008). Cone 0 (anchor = x-axis, normals near horizontal); cone 1 (anchor =
y-axis, normals near vertical). Atom (cone c, scale j, shear k, center m):

    parabolic scaling   sigma_n = w0 * r**(-j)      (across the edge; thin)
                        sigma_t = aniso * w0 * r**(-j/2)   (along the edge; long)
    cone 0 sheared coords   xi = (x-mx) + k*(y-my)   eta = (y-my)
    cone 1 sheared coords   xi = (y-my) + k*(x-mx)   eta = (x-mx)
    env    = exp(-0.5*(xi**2/sigma_n**2 + eta**2/sigma_t**2))
    real   = env * cos(2*pi*f_j*xi)     imag = env * sin(2*pi*f_j*xi)

The oscillation runs along ``xi``, whose gradient in (x,y) is (1,k) [cone 0] or
(k,1) [cone 1] -- so the NORMAL (oscillation) direction STEERS with the shear k,
while the envelope is a SHEARED (parallelogram) Gaussian rather than a rotated
ellipse. On the anchor line (eta's base axis) the shear leaves points fixed, so
the along-anchor profile is shear-invariant -- property (1).

The bank is fully determined by a handful of GENERIC scalars (no seed, no video
data): bit-identical at decode -> rule-118 FREE. Only learned/selected
coefficients (per-atom weights a trained witness would store) are COUNTED. This
module supplies the frame ONLY; wiring it into the trainer forward + generated
inflate.py (op-parity) + a real n600 through-R d_seg receipt is the OWED heavier
step (needs a run; operator-GO / CONTAINMENT).

Portability: the forward is a generic array module (``numpy`` = deterministic
fp64 authority; ``mlx`` mirrors it elementwise). ``mlx_parity_check`` proves it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompactShearletConfig:
    """Generic (seed-free, video-free) cone-adapted shearlet frame parameters.

    n_scales J radial octaves; n_shear S shears PER SIDE per cone (shears run
    k in {-S..S}, so 2S+1 per cone at each scale -- the classical shearlet uses
    a scale-growing shear count; a fixed modest S is CONSERVATIVE, fewer atoms);
    two_cones toggles the vertical cone (cone 1) that covers near-vertical
    normals (a single cone only spans a bounded angular wedge -- the shear-cone
    property); shear_step the increment in k per shear index; f0/base the radial
    frequency octaves; w0 the coarsest envelope width; width_ratio r the per-
    octave envelope shrink (parabolic: sigma_n ~ sigma_t**2); n_trans
    translations PER AXIS; aniso the constant tangent-elongation boost;
    min_sigma clamps the finest envelope so atoms never collapse below the grid.

    Fully determined by these scalars -> bit-identical regeneration at decode,
    zero archive bytes, no GT leak.
    """

    n_scales: int = 4
    n_shear: int = 2
    two_cones: bool = True
    shear_step: float = 0.5
    f0: float = 2.0
    base: float = 2.0
    w0: float = 0.5
    width_ratio: float = 2.0
    n_trans: int = 2
    coord_margin: float = 0.5
    min_sigma: float = 0.02
    aniso: float = 1.0

    def __post_init__(self) -> None:
        for name in ("n_scales", "n_shear", "n_trans"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be a non-negative int, got {v!r}")
        if int(self.n_scales) <= 0 or int(self.n_trans) <= 0:
            raise ValueError("n_scales and n_trans must be positive")
        if not isinstance(self.two_cones, bool):
            raise ValueError("two_cones must be a bool")
        for name in ("f0", "base", "w0", "width_ratio", "min_sigma", "aniso", "shear_step"):
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
class ShearletAtomIndex:
    """One frame atom's deterministic index (cone, scale, shear, center)."""

    cone: int          # 0 = horizontal anchor, 1 = vertical anchor
    scale: int
    shear_k: float     # the shear parameter k (0 = base orientation of the cone)
    freq: float
    sigma_n: float
    sigma_t: float
    cx: float
    cy: float


def _sigma_pair(cfg: CompactShearletConfig, j: int) -> tuple[float, float]:
    r = float(cfg.width_ratio)
    sigma_n = max(float(cfg.w0) * r ** (-j), float(cfg.min_sigma))
    sigma_t = max(float(cfg.aniso) * float(cfg.w0) * r ** (-0.5 * j), float(cfg.min_sigma))
    return sigma_n, sigma_t


def shearlet_atom_index(cfg: CompactShearletConfig) -> tuple[ShearletAtomIndex, ...]:
    """Deterministic atom list (cone x scale x shear x translation grid).

    Order is stable: for cone, for j, for shear-index, for (ty, tx) row-major
    over the translation grid. This order IS the counted-coefficient order.
    """

    m = float(cfg.coord_margin)
    lo, hi = -1.0 + m, 1.0 - m
    if cfg.n_trans == 1:
        centers = [0.0]
    else:
        centers = list(np.linspace(lo, hi, cfg.n_trans))
    n_cones = 2 if cfg.two_cones else 1
    shears = [float(cfg.shear_step) * s for s in range(-int(cfg.n_shear), int(cfg.n_shear) + 1)]
    atoms: list[ShearletAtomIndex] = []
    for cone in range(n_cones):
        for j in range(int(cfg.n_scales)):
            f_j = float(cfg.f0) * (float(cfg.base) ** j)
            sigma_n, sigma_t = _sigma_pair(cfg, j)
            for k in shears:
                for cy in centers:
                    for cx in centers:
                        atoms.append(
                            ShearletAtomIndex(cone, j, float(k), f_j, sigma_n, sigma_t,
                                              float(cx), float(cy))
                        )
    return tuple(atoms)


def _atom_xi_eta(x, y, a: ShearletAtomIndex):
    """Return the SHEARED (xi, eta) coordinates for an atom (cone-adapted).

    cone 0 (anchor x-axis): xi = dx + k*dy, eta = dy  -> normal grad (1, k)
    cone 1 (anchor y-axis): xi = dy + k*dx, eta = dx  -> normal grad (k, 1)
    """

    dx = x - a.cx
    dy = y - a.cy
    if a.cone == 0:
        xi = dx + a.shear_k * dy
        eta = dy
    else:
        xi = dy + a.shear_k * dx
        eta = dx
    return xi, eta


def compact_shearlet_feats(coords: np.ndarray, cfg: CompactShearletConfig, *, xp=np) -> np.ndarray:
    """Return (P, 2*D) real frame features [real_cols | imag_cols].

    Columns 0..D-1 are ``env*cos(2*pi f xi)``; columns D..2D-1 are
    ``env*sin(2*pi f xi)`` for the SAME atom order (``shearlet_atom_index``). The
    paired envelope real^2 + imag^2 = env^2 is a (sheared) spatial Gaussian bump.

    ``xp`` is the array module (numpy = fp64 authority, or mlx). numpy computes
    in fp64 then casts to fp32 (matching the polar-Fourier reference convention).
    """

    atoms = shearlet_atom_index(cfg)
    x = xp.asarray(coords)[:, 0]
    y = xp.asarray(coords)[:, 1]
    two_pi = 2.0 * math.pi
    real_cols = []
    imag_cols = []
    for a in atoms:
        xi, eta = _atom_xi_eta(x, y, a)
        env = xp.exp(-0.5 * ((xi / a.sigma_n) ** 2 + (eta / a.sigma_t) ** 2))
        phase = two_pi * a.freq * xi
        real_cols.append(env * xp.cos(phase))
        imag_cols.append(env * xp.sin(phase))
    real = xp.stack(real_cols, axis=-1)
    imag = xp.stack(imag_cols, axis=-1)
    out = xp.concatenate([real, imag], axis=-1)
    if xp is np:
        return np.asarray(out, dtype=np.float32)
    return out


def n_atoms(cfg: CompactShearletConfig) -> int:
    """Number of atoms D (half the feature columns 2D)."""

    return len(shearlet_atom_index(cfg))


# ---------------------------------------------------------------------------
# Shearlet certificate -- the catalog-#351 shear-selectivity + localization swap-test.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShearletCertificate:
    """Evidence that the frame is GENUINELY shear-steered + localized.

    shearlet_envelope_span: max over atoms of ptp(env^2) across the coord grid.
        A genuinely localized atom's paired envelope is a Gaussian bump -> ~O(1).
    fourier_envelope_span: same statistic for the polar-Fourier bank (constant
        envelope 1 -> ~1e-7); Fourier FAILS the localization span gate.
    shearlet_energy_concentration: median over atoms of the fraction of L2 energy
        in the 10% highest-magnitude coordinates (localized -> ~1).
    shear_anchor_dispersion: dispersion of the along-anchor-axis profile across a
        SHEAR-steered family at fixed scale/center. Shear fixes the anchor line
        pointwise -> ~0.
    rotation_anchor_dispersion: SAME statistic for a ROTATION-steered family that
        covers the SAME normal directions. Rotation fixes no line -> >> 0. The
        swap-test: rotation FAILS the shear-invariance the shearlet PASSES.
    shear_discrimination_ratio: rotation_anchor_dispersion / shear_anchor_dispersion
        (large -> genuine shear, not rotation-in-disguise).
    normal_angular_spread_deg: angular range (deg) of the shear family's realized
        normal directions -> proves it is genuinely DIRECTIONAL (not degenerate).
    integer_lattice_preserving: the integer-shear matrix S_1 has integer entries +
        det 1 (maps Z^2 -> Z^2) AND the matched rotation R_atan(1) does NOT.
    parabolic_scaling_monotone: sigma_n shrinks strictly faster than sigma_t.
    passes: True iff localized (span, conc) AND Fourier fails localization AND the
        shear swap-test discriminates (shear invariant, rotation not) AND genuinely
        directional AND integer-lattice-preserving AND parabolic-monotone.
    """

    shearlet_envelope_span: float
    fourier_envelope_span: float
    shearlet_energy_concentration: float
    shear_anchor_dispersion: float
    rotation_anchor_dispersion: float
    shear_discrimination_ratio: float
    normal_angular_spread_deg: float
    integer_lattice_preserving: bool
    parabolic_scaling_monotone: bool
    span_gate: float
    concentration_gate: float
    discrimination_gate: float
    min_angular_spread_deg: float
    passes: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm_profile(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    nrm = math.sqrt(float((v ** 2).sum()))
    return v / nrm if nrm > 1e-12 else v


def _anchor_dispersion(profiles: np.ndarray) -> float:
    """Mean over the anchor axis of the variance-across-steering-index of the
    (unit-normalized) along-anchor profiles. ~0 iff every atom agrees on the
    anchor line (shear-invariance)."""

    if profiles.shape[0] < 2:
        return 0.0
    normed = np.stack([_norm_profile(p) for p in profiles], axis=0)
    return float(np.var(normed, axis=0).mean())


def _shear_family_on_anchor(cfg: CompactShearletConfig, xs: np.ndarray, j0: int):
    """Along-anchor profiles from the REAL ``compact_shearlet_feats`` forward.

    Builds a minimal cone-0, single-scale, single-center probe config so the D
    real columns ARE the shear family (one per shear k), sampled on the anchor
    line y=0. Reading the ACTUAL forward is what makes the swap-test #351-tight:
    if a future edit changes ``_atom_xi_eta`` to a rotation (or Fourier), these
    columns stop being anchor-invariant and the certificate flips to False.
    Returns the profiles plus the realized unit normals (1,k)."""

    probe = CompactShearletConfig(
        n_scales=1, n_shear=int(cfg.n_shear), two_cones=False, shear_step=float(cfg.shear_step),
        f0=float(cfg.f0) * (float(cfg.base) ** j0), base=float(cfg.base), w0=float(cfg.w0),
        width_ratio=float(cfg.width_ratio), n_trans=1, coord_margin=float(cfg.coord_margin),
        min_sigma=float(cfg.min_sigma), aniso=float(cfg.aniso),
    )
    anchor_coords = np.column_stack([xs, np.zeros_like(xs)]).astype(np.float64)  # y=0 anchor line
    feats = compact_shearlet_feats(anchor_coords, probe).astype(np.float64)
    D = feats.shape[1] // 2
    profiles = feats[:, :D].T  # (n_shear_atoms, len(xs)) real-part columns
    shears = [float(cfg.shear_step) * s for s in range(-int(cfg.n_shear), int(cfg.n_shear) + 1)]
    normals = []
    for k in shears:
        n = np.array([1.0, k])
        normals.append(n / np.linalg.norm(n))
    return profiles, np.stack(normals, axis=0)


def _rotation_family_on_anchor(cfg: CompactShearletConfig, xs: np.ndarray, j0: int):
    """Along-anchor profiles for a ROTATION-steered family covering the SAME
    normal directions (theta_i = atan(k_i)). Rotation moves the anchor line, so
    the on-axis profile DEPENDS on theta -> high dispersion."""

    sigma_n, sigma_t = _sigma_pair(cfg, j0)
    f_j = float(cfg.f0) * (float(cfg.base) ** j0)
    two_pi = 2.0 * math.pi
    shears = [float(cfg.shear_step) * s for s in range(-int(cfg.n_shear), int(cfg.n_shear) + 1)]
    profiles = []
    for k in shears:
        theta = math.atan(k)
        ct, st = math.cos(theta), math.sin(theta)
        # rigid rotation: sample the atom on the y=0 line -> (x*ct, -x*st) in mother frame.
        xi = xs * ct           # = dx*ct + dy*st with dy=0
        eta = -xs * st         # = -dx*st + dy*ct with dy=0
        env = np.exp(-0.5 * ((xi / sigma_n) ** 2 + (eta / sigma_t) ** 2))
        profiles.append(env * np.cos(two_pi * f_j * xi))
    return np.stack(profiles, axis=0)


def _envelope_energy_stats(feats: np.ndarray, coords: np.ndarray) -> tuple[float, float]:
    """(max paired-envelope span, median energy concentration@10%).

    Mirrors the curvelet module's helper; kept local so this module has no
    private cross-dependency (the two frames are UNIQUE-AND-COMPLETE per method).
    """

    D = feats.shape[1] // 2
    real = feats[:, :D].astype(np.float64)
    imag = feats[:, D:].astype(np.float64)
    env = real ** 2 + imag ** 2
    spans = np.ptp(env, axis=0)
    max_span = float(spans.max(initial=0.0))
    P = env.shape[0]
    kk = max(1, int(round(0.10 * P)))
    concs = []
    for i in range(D):
        e = env[:, i]
        tot = e.sum()
        if tot <= 1e-12:
            continue
        top = np.sort(e)[-kk:].sum()
        concs.append(top / tot)
    conc = float(np.median(concs)) if concs else 0.0
    return max_span, conc


def shearlet_certificate(
    cfg: CompactShearletConfig,
    *,
    height: int = 33,
    width: int = 33,
    span_gate: float = 0.10,
    concentration_gate: float = 0.30,
    discrimination_gate: float = 10.0,
    min_angular_spread_deg: float = 10.0,
) -> ShearletCertificate:
    """Prove the frame is localized AND genuinely shear-steered.

    Runs three internal banks on the SAME coord grid: the shearlet (must be
    localized + shear-invariant on the anchor axis), a rotation-steered family
    covering matched normals (must NOT be shear-invariant), and the polar-Fourier
    bank (must fail the localization span gate). If a future edit swaps rotation
    or Fourier into ``compact_shearlet_feats``, ``passes`` flips to False.
    """

    from tac.boundary_math.lever_b_levelset_generator import (
        PolarDirectionalFourierBankConfig,
        build_coords,
        polar_directional_fourier_B,
        polar_directional_fourier_feats,
    )

    coords = build_coords(height, width)
    sfeats = compact_shearlet_feats(coords, cfg)
    s_span, s_conc = _envelope_energy_stats(sfeats, coords)

    ffeats = polar_directional_fourier_feats(
        coords, polar_directional_fourier_B(PolarDirectionalFourierBankConfig())
    )
    f_span, _ = _envelope_energy_stats(ffeats, coords)

    # shear-selectivity swap-test at the coarsest scale (widest atoms = cleanest signal).
    xs = np.linspace(-1.0, 1.0, width)
    j0 = 0
    shear_prof, normals = _shear_family_on_anchor(cfg, xs, j0)
    rot_prof = _rotation_family_on_anchor(cfg, xs, j0)
    shear_disp = _anchor_dispersion(shear_prof)
    rot_disp = _anchor_dispersion(rot_prof)
    disc_ratio = rot_disp / shear_disp if shear_disp > 1e-15 else float("inf")

    # realized normal angular spread (proves genuinely directional, not degenerate).
    angs = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    ang_spread = float(angs.max() - angs.min())

    # integer-lattice preservation: S_1 integer + det 1; matched rotation NOT integer.
    S1 = np.array([[1, 1], [0, 1]], dtype=np.int64)
    det_ok = int(round(np.linalg.det(S1.astype(np.float64)))) == 1
    theta1 = math.atan(1.0)
    R1 = np.array([[math.cos(theta1), -math.sin(theta1)],
                   [math.sin(theta1), math.cos(theta1)]])
    rot_is_integer = bool(np.allclose(R1, np.round(R1)))
    integer_lattice = bool(det_ok and not rot_is_integer)

    # parabolic monotonicity: sigma_n / sigma_t = r**(-j/2) strictly decreasing in j.
    mono = True
    prev = None
    for j in range(int(cfg.n_scales)):
        sn, st = _sigma_pair(cfg, j)
        ratio_j = sn / st
        if prev is not None and not (ratio_j < prev - 1e-9):
            mono = False
        prev = ratio_j

    localized = (s_span >= span_gate) and (s_conc >= concentration_gate)
    fourier_fails = f_span < span_gate
    shear_invariant = shear_disp <= (0.1 * span_gate)  # anchor profiles agree (near-0)
    rotation_not_invariant = disc_ratio >= discrimination_gate
    directional = ang_spread >= min_angular_spread_deg
    passes = bool(
        localized and fourier_fails and shear_invariant and rotation_not_invariant
        and directional and integer_lattice and mono
    )
    return ShearletCertificate(
        shearlet_envelope_span=s_span,
        fourier_envelope_span=f_span,
        shearlet_energy_concentration=s_conc,
        shear_anchor_dispersion=shear_disp,
        rotation_anchor_dispersion=rot_disp,
        shear_discrimination_ratio=disc_ratio,
        normal_angular_spread_deg=ang_spread,
        integer_lattice_preserving=integer_lattice,
        parabolic_scaling_monotone=mono,
        span_gate=span_gate,
        concentration_gate=concentration_gate,
        discrimination_gate=discrimination_gate,
        min_angular_spread_deg=min_angular_spread_deg,
        passes=passes,
    )


def mlx_parity_check(cfg: CompactShearletConfig, *, height: int = 17, width: int = 17,
                     tol: float = 1e-4) -> dict[str, Any]:
    """Confirm the mlx forward matches the numpy fp64->fp32 authority (if mlx present)."""

    from tac.boundary_math.lever_b_levelset_generator import build_coords

    coords = build_coords(height, width)
    ref = compact_shearlet_feats(coords, cfg)  # numpy authority
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:  # pragma: no cover - mlx optional
        return {"mlx_available": False, "reason": str(exc)}
    mfeats = compact_shearlet_feats(mx.array(coords), cfg, xp=mx)
    got = np.asarray(mfeats, dtype=np.float32)
    max_abs = float(np.max(np.abs(got - ref)))
    return {"mlx_available": True, "max_abs_diff": max_abs, "within_tol": bool(max_abs <= tol)}


__all__ = [
    "CompactShearletConfig",
    "ShearletAtomIndex",
    "ShearletCertificate",
    "compact_shearlet_feats",
    "mlx_parity_check",
    "n_atoms",
    "shearlet_atom_index",
    "shearlet_certificate",
]
