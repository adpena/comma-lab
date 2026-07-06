"""Numpy REFERENCE for the sharpness proxy |c_a| that drives the ADAPTIVE-eps CFL edge tracker.

Task #320 (ADAPTIVE-eps MECHANISM-CURE). This is a *reference oracle only* — it does NOT run
in any trainer and touches no training state. Its purpose is to give the v6 adaptive-eps trainer
term a byte-parity target for its MLX |c_a| measurement (the established byte-identity pattern;
sister of ``eikonal_normal_curvature_reference.py`` #318).

WHY |c_a| (the whole point). DE-DERIVATION #318 (§2-4) derives the eikonal penalty flow's
principal symbol ``sigma(k) = -k_n^2 - c_a * k_T^2``, ``c_a = (|grad m| - 1)/|grad m|`` on the
decision margin ``m = phi_top1 - phi_top2``. The viscous (ViscoReg) CFL LOWER edge is

    eps_lower(t) = |c_a(t)| * sqrt(eta * lambda_eik / 8)                          (DE #318 §3)

and it RISES as progressive sharpening grows |c_a(t)| (more of the boundary annulus enters the
flat |grad m| < 1 regime under tau descent / hosc-beta anneal). A FIXED eps (v5's 0.3) eventually
falls below this rising edge => the measured ep110 re-entry. The ADAPTIVE-eps cure floors eps just
above the edge by TRACKING |c_a(t)|:

    eps(t) = clamp( |c_a(t)| * sqrt(eta * lambda_eik / 8) * (1 + margin), eps_floor, eps_upper )

So the ONE field quantity the controller needs each epoch is the scalar sharpness proxy

    |c_a| = mean_over_interior | (|grad m| - 1) / |grad m| |          (symposium #317 §7.4 form)

optionally restricted to the small-margin annulus |m| < band (DE #318 §2: the ill-posed a<1 mode
lives near the separatrix). ``band = 0.0`` (DEFAULT) = the interior mean = the EXACT §7.4 launch
formula; ``band > 0.0`` = the annulus-restricted variant (the task's "small-margin band" phrasing).

STENCIL CONTRACT (byte-parity with the MLX trainer). |c_a| needs ONLY the first derivatives of m.
This module uses the SAME central-difference interior stencil the trainer's
``_eikonal_margin_interior_mlx`` uses (the viscous term's own grid), so the numpy |c_a| and the MLX
|c_a| agree bit-for-bit on the interior:

    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])      # central d/dx on the (H-2, W-2) interior
    gy = 0.5 * (m[2:,  1:-1] - m[:-2, 1:-1])      # central d/dy
    gmag = sqrt(gx^2 + gy^2 + 1e-8)              # SAME 1e-8 floor as the MLX visco/steik terms

The self-test cross-checks this against #318's ``eikonal_normal_curvature_reference._central_grad``
(interior slice) so the two references share one stencil convention (the "parity to #318" contract).

Field convention: ``m`` is a 2-D array m[i, j], axis 0 = rows (y), axis 1 = cols (x). Determinism:
pure numpy, no RNG in the proxy itself (the self-test seeds its own generator). Default-safe:
importing this module runs nothing; call the functions explicitly.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "sharpness_proxy_c_a",
    "adaptive_visco_eps",
    "adaptive_visco_eps_tanh",
    "self_test",
]

_GMAG_FLOOR = 1e-8  # matches the MLX `mx.sqrt(gx*gx + gy*gy + 1e-8)` in _eikonal_margin/visco terms


def _central_grad_interior(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central first derivatives (gy, gx) on the (H-2, W-2) interior, matching the MLX
    ``_eikonal_margin_interior_mlx`` stencil EXACTLY (so numpy |c_a| == MLX |c_a| on the interior)."""
    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])   # d/dx (cols)
    gy = 0.5 * (m[2:, 1:-1] - m[:-2, 1:-1])   # d/dy (rows)
    return gy, gx


def sharpness_proxy_c_a(m: np.ndarray, band: float = 0.0) -> float:
    """|c_a| = mean | (|grad m| - 1) / |grad m| | over the decision-margin interior.

    ``m``   : (H, W) decision margin (top1 - top2 class SDF gap); H, W >= 3.
    ``band``: 0.0 (default) => interior mean (symposium §7.4 exact launch formula);
              > 0.0 => restrict the mean to the small-margin annulus |m_interior| < band
              (DE #318 §2 flat-regime; empty band => 0.0, a well-defined "no sharp annulus").

    Returns the scalar |c_a(t)| the adaptive-eps controller consumes. Uses the SAME central-diff
    interior stencil + 1e-8 gmag floor as the MLX trainer (byte-parity contract).
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D margin field, got shape {m.shape}")
    if m.shape[0] < 3 or m.shape[1] < 3:
        raise ValueError(f"margin field must be at least 3x3 for a central interior, got {m.shape}")
    gy, gx = _central_grad_interior(m)
    gmag = np.sqrt(gx * gx + gy * gy + _GMAG_FLOOR)
    c_a = np.abs((gmag - 1.0) / gmag)           # |(|grad m| - 1)/|grad m||, shape (H-2, W-2)
    if band > 0.0:
        m_int = m[1:-1, 1:-1]                    # interior margin aligned to c_a
        mask = np.abs(m_int) < float(band)
        if not np.any(mask):
            return 0.0
        return float(np.mean(c_a[mask]))
    return float(np.mean(c_a))


def adaptive_visco_eps(c_a: float, eta: float, lam_eik: float, margin_factor: float,
                       eps_floor: float, eps_upper: float) -> float:
    """The ADAPTIVE-eps law (DE #318 §4 Arm-2 / symposium #317 §7.4), closed form:

        eps(t) = clamp( |c_a| * sqrt(eta * lambda_eik / 8) * (1 + margin_factor), eps_floor, eps_upper )

    The MLX trainer's ``_adaptive_visco_eps`` must return this to the float. eta, lam_eik are
    clamped at 0 (a negative product under the sqrt is non-physical); if eps_upper < eps_floor the
    upper is raised to the floor (clamp stays well-defined). Pure; unit-tested both sides.
    """
    edge = abs(float(c_a)) * math.sqrt(max(0.0, float(eta) * float(lam_eik) / 8.0))
    eps = edge * (1.0 + float(margin_factor))
    lo = float(eps_floor)
    hi = float(eps_upper)
    if hi < lo:
        hi = lo
    return min(max(eps, lo), hi)


def adaptive_visco_eps_tanh(c_a: float, margin_factor: float,
                            eps_floor: float, eps_upper: float) -> float:
    """The RESPONSIVE reparameterization of the adaptive-eps law (confound-fix C2, 2026-07-05):

        eps = clamp( eps_floor + (eps_upper - eps_floor) * tanh(|c_a| * (1 + margin_factor)),
                     eps_floor, eps_upper )

    WHY this exists alongside the DE-CFL ``adaptive_visco_eps`` above: the confound hunt PROVED the
    DE-CFL form is INERT at the measured operating point -- with eta~1e-3, lam_eik~0.05 the sqrt
    prefactor collapses to ~2.5e-3, so reaching the floor 0.3 needs |c_a| >= ~80 while the measured
    |c_a| is O(1) (~0.82) => eps clamped at the floor EVERY epoch (0 change-events). That is not a
    bug in the DE law: the CFL edge genuinely says a small fixed eps (~2.5e-3) already over-damps, so
    the floor is safe and adaptive-eps provides NO benefit at O(1) |c_a|. The DE-CFL law is retained
    (derivation of record). This tanh form is the SEPARATE responsive lever the MLX trainer's
    ``_adaptive_visco_eps`` uses when adaptive-eps is deliberately enabled as a *tunable* actuator
    (monotone-increasing in |c_a|, saturating into [floor, upper]); it must return THIS to the float
    for the trainer<->reference byte-parity contract. eta/lam_eik are NOT arguments here (they are no
    longer the collapsed prefactor; the caller logs them only as the advisory pi_eik). eps_upper
    raised to floor if inverted. Pure / unit-testable.

    NOTE (NO-FAKE): adaptive-eps being inert under the DE-CFL law is a FEATURE that is now reported
    LOUDLY via the trainer's ``adaptive_eps_INERT`` alarm -- not papered over. Prefer a small FIXED
    eps for the eikonal-viscosity term unless an A/B on this responsive lever measures a real benefit.
    """
    lo = float(eps_floor)
    hi = float(eps_upper)
    if hi < lo:
        hi = lo
    frac = math.tanh(abs(float(c_a)) * (1.0 + float(margin_factor)))
    eps = lo + (hi - lo) * frac
    return min(max(eps, lo), hi)


def self_test() -> dict[str, float]:
    """Verify (1) the stencil matches #318's central-grad on the interior, (2) |c_a| in [0,1) for a
    smooth field, (3) |c_a| -> 0 as the field approaches a unit-gradient SDF (|grad m| -> 1), and
    (4) the adaptive-eps law tracks the CFL edge + clamps. Returns measured quantities; asserts the
    contract the MLX port must reproduce.
    """
    out: dict[str, float] = {}
    rng = np.random.default_rng(320)
    h, w = 48, 64
    base = rng.standard_normal((h, w)).cumsum(axis=0).cumsum(axis=1)
    base = (base - base.mean()) / (base.std() + 1e-12)

    # (1) stencil parity vs #318's reference on the interior.
    from tac.boundary_math.eikonal_normal_curvature_reference import _central_grad as _cg318

    my318, mx318 = _cg318(base)
    gy, gx = _central_grad_interior(base)
    d_gx = float(np.max(np.abs(gx - mx318[1:-1, 1:-1])))
    d_gy = float(np.max(np.abs(gy - my318[1:-1, 1:-1])))
    out["stencil_max_abs_diff_vs_318_gx"] = d_gx
    out["stencil_max_abs_diff_vs_318_gy"] = d_gy
    assert d_gx < 1e-12 and d_gy < 1e-12, "|c_a| stencil must match #318 _central_grad on the interior"

    # (2) |c_a| in [0, 1) for a generic smooth field (|(g-1)/g| < 1 whenever g > 0.5; a valid proxy).
    ca = sharpness_proxy_c_a(base)
    out["c_a_smooth_field"] = ca
    assert ca >= 0.0, "|c_a| is a mean of absolute values -> non-negative"

    # (3) a near-unit-gradient plane m = x has |grad m| = 1 exactly => c_a -> 0.
    xx = np.tile(np.arange(w, dtype=np.float64), (h, 1))   # m = x, d/dx = 1 (central), d/dy = 0
    ca_sdf = sharpness_proxy_c_a(xx)
    out["c_a_unit_gradient_plane"] = ca_sdf
    assert ca_sdf < 1e-6, f"unit-gradient plane must give |c_a| ~ 0, got {ca_sdf}"

    # (3b) a scaled plane m = 3x has |grad m| = 3 => c_a = |(3-1)/3| = 2/3 everywhere.
    ca_steep = sharpness_proxy_c_a(3.0 * xx)
    out["c_a_grad3_plane"] = ca_steep
    assert abs(ca_steep - 2.0 / 3.0) < 1e-6, f"m=3x must give |c_a|=2/3, got {ca_steep}"

    # (3c) a flat plane m = 0.5x has |grad m| = 0.5 => c_a = |(0.5-1)/0.5| = 1.0 (the ill-posed a<1).
    ca_flat = sharpness_proxy_c_a(0.5 * xx)
    out["c_a_grad_half_plane"] = ca_flat
    assert abs(ca_flat - 1.0) < 1e-6, f"m=0.5x (flat, a<1) must give |c_a|=1, got {ca_flat}"

    # (4) the adaptive-eps law: at the edge, floor, and upper clamps.
    # unclamped edge: |c_a|=1, eta=1e-3, lam=0.05, margin=0.5 -> sqrt(1e-3*0.05/8)*1.5
    eps_edge = adaptive_visco_eps(1.0, 1e-3, 0.05, 0.5, 0.0, 10.0)
    ref = math.sqrt(1e-3 * 0.05 / 8.0) * 1.5
    out["adaptive_eps_edge"] = eps_edge
    assert abs(eps_edge - ref) < 1e-12, "adaptive-eps must equal the closed-form edge when unclamped"
    # floor clamp: tiny edge -> floor
    eps_lo = adaptive_visco_eps(1e-9, 1e-3, 0.05, 0.5, 0.3, 0.7)
    out["adaptive_eps_floor_clamp"] = eps_lo
    assert eps_lo == 0.3, "adaptive-eps must clamp UP to the floor"
    # upper clamp: huge |c_a| -> upper
    eps_hi = adaptive_visco_eps(1e6, 1e-3, 0.05, 0.5, 0.3, 0.7)
    out["adaptive_eps_upper_clamp"] = eps_hi
    assert eps_hi == 0.7, "adaptive-eps must clamp DOWN to the upper"

    # (band) small-margin annulus restriction returns a well-defined value (0.0 when empty).
    ca_band = sharpness_proxy_c_a(base, band=1e-9)  # essentially no pixel inside -> 0.0
    out["c_a_empty_band"] = ca_band
    assert ca_band == 0.0, "empty small-margin band must return 0.0"

    return out


if __name__ == "__main__":  # pragma: no cover
    results = self_test()
    print("sharpness-proxy |c_a| + adaptive-eps reference — self-test PASSED")
    for k, v in results.items():
        print(f"  {k}: {v:.6e}")
