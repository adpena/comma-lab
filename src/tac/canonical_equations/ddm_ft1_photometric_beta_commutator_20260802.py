# SPDX-License-Identifier: MIT
"""ddm_ft1 — WHERE DOES COORDINATE-FIT STALENESS ACTUALLY COUPLE?

ddm_fs1 established the DEFECT (an auto-exposure ``(a,b)`` fitted at ``beta=0``
still shipping beside a non-zero ``beta``).  This module establishes the
MECHANISM, because "there is staleness" and "the staleness costs something" are
different claims and only the second one licenses spending.

THE PRE-REGISTERED DERIVATION AND WHERE IT BREAKS (this is the finding).

  Premise (CONFIRMED, measured).  Rolling shutter acts on the image DOMAIN as a
  row-blended pair of homographies; auto-exposure acts on the image RANGE as
  ``a*I + b``.  The receiver's ``warp_rgb`` is linear in the source and preserves
  constants (its bilinear weights sum to 1 and its out-of-frame fallback is the
  identity), so as OPERATORS::

      W_beta(a*I + b) == a*W_beta(I) + b        exactly

  MEASURED 2026-08-02 through the shipped receiver: max relative residual
  ``3.4e-16`` (float round-off) over 3 pairs at the shipped ``(a,b)``.  The
  commutation premise is CONFIRMED; see ``OPERATOR_COMMUTATION_CONTROL``.

  Inference drawn from it (FALSIFIED).  "Therefore the coupling is identically
  zero except through the uint8 clamp, so coupling magnitude scales with the
  SATURATED-PIXEL FRACTION and near-zero-clip pairs can be retired."

  Why it does not follow.  Commuting OPERATORS do not imply a vanishing MIXED
  PARTIAL of the LOSS.  With ``X_beta := W_beta(I)`` and ``g`` the frozen
  PoseNet MSE, ``L(a,b,beta) = g(a*X_beta + b)`` and::

      d2L/da dbeta = <grad g, dX/dbeta> + a * <Hess_g @ dX/dbeta, X_beta>

  Both terms are generically non-zero for ANY non-zero ``dX/dbeta``, with NO
  clamp anywhere in the expression.  The coupling is O(||dX/dbeta||) -- carried
  by how much the geometry actually moved the image, times the CURVATURE of the
  frozen scorer.  Clipping is one additive channel, not the mechanism.

  MEASURED consequence (n600, scorer-free, ``tools/ddm_ft1_clip_fraction_
  falsifier.py``).  On the 244 stale pairs the clamp channel carries a mean
  ``1.5e-05`` SHARE of the total uint8 change; its median is EXACTLY ZERO and
  185 of 244 pairs have no clamp channel at all.  It does not rank: Pearson
  ``+0.04`` against ``|delta beta|``, and the raw clip fraction is ``-0.03``.
  The reductio is one pair: pair 17 ships ``beta = -7.5`` and moves 43.9% of
  its pixels (``pre_rms`` 4.00, the LARGEST drift in the population) with a
  clamp channel of EXACTLY ZERO -- the clip criterion puts the worst pair in
  the population at the top of its retirement list.  See ``FT1_N600_CENSUS``.
  The free retirement the prediction promised is NOT available.

THE THIRD CHANNEL THE DERIVATION OMITTED.  ``_to_uint8`` is ``clip(round(.))``.
ROUND is non-invertible too and it bites on EVERY pixel, not only saturated
ones.  MEASURED: sub-quantum moves account for 67.7% of all uint8 changes on
the stale set, a per-pixel rate ~3.1e+04x the clamp channel.  The derivation
named the SMALLER of the two quantization channels and missed the dominant one.

WHAT IS OWED (do not read this module as closing the question).  Every number
here is IMAGE-DOMAIN and scorer-free: it bounds the channels through which
coupling can travel, it does not measure ``d2L/d(a,b)dbeta`` itself.  One
PoseNet job -- re-solve ``(a,b)`` at the shipped geometry on the top-ranked
pairs and difference the realized ``d_pose`` -- converts this from a falsified
predictor plus a derived mechanism into a measured coupling.  That job needs
the n600 scorer slot, which was occupied when this landed.

Consumers: ``tools/ddm_ft1_clip_fraction_falsifier.py`` (the n600 measurement),
and any coordinate-descent solver deciding whether a stale coefficient is worth
re-solving -- ``staleness_priority`` for the ranking that survived,
``fiber_transport_delta_ab`` for the cheap first-order repair, and
``transport_admissible`` for the hull boundary it may not cross.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: The discrete beta magnitudes the shipped ``(a,b)`` was fitted against
#: (``ddm_v4d_resolve._beta_select`` seed menu).  Sign is not free: at fit time
#: the receiver pinned it to the pose yaw sign, so an OPPOSING sign is outside
#: the sampled set even when its magnitude is inside ``[0, 1]``.
FIT_MENU_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0)

#: MEASURED 2026-08-02 through the shipped v4d receiver primitives.  The
#: derivation's algebraic premise, executed rather than asserted.
OPERATOR_COMMUTATION_CONTROL: dict[str, Any] = {
    "claim": "W_beta(a*I + b) == a*W_beta(I) + b",
    "max_rel_residual": 3.3900463544404e-16,
    "verdict": "CONFIRMED_AT_FLOAT_ROUNDOFF",
    "axis": "[macOS-CPU scorer-free advisory]",
}

#: MEASURED 2026-08-02, n600 (all 600 pairs, no subset), scorer-free, through
#: the real v4d receiver primitives, on ``mq1_emit/final_mq1.jsonl``.  ``stale``
#: = non-identity (a,b) shipping beside a non-zero beta.  Fractions are
#: per-pixel means over the 874x1164x3 camera frame.
#: Artifact: ``<v4d run dir>/ft1/{ft1_channels.jsonl, ft1_receipt.json}``.
FT1_N600_CENSUS: dict[str, Any] = {
    "pairs": 600,
    "stale": 244,
    # --- hull escape, CORRECTED upward from ddm_fs1's 59 ---
    # fs1 counted magnitude extrapolation only.  At fit time the sign was PINNED
    # to the pose yaw (beta = g*yaw_sign, g >= 0), so an opposing sign is a
    # direction the fit never sampled either -- a hull escape fs1's min/max test
    # cannot see.  63 pairs ship one; the union with the 59 magnitude escapes is
    # 101 pairs, of which 100 are stale.
    "extrapolated_magnitude": 59,
    "opposing_sign": 63,
    "outside_fitted_set_union": 101,
    "stale_and_outside_fitted_set": 100,
    # --- the pre-registered predictor: degenerate, then anti-correlated ---
    "clamp_channel_mean": 1.265e-06,
    "clamp_channel_median": 0.0,
    "clamp_channel_max": 8.978e-05,
    "stale_with_exactly_zero_clamp_channel": 185,
    "clamp_share_of_total_change_mean": 1.478e-05,
    "pearson_clip_sym_diff_vs_abs_delta_beta": 0.042,
    "pearson_clip_frac_vs_abs_delta_beta": -0.030,
    "pearson_clip_frac_vs_pre_rms": -0.048,
    # --- what actually moved, and what does rank it ---
    "u8_diff_frac_mean": 0.1067,
    "u8_diff_frac_max": 0.5363,
    "pre_rms_mean": 0.5047,
    "pre_rms_max": 4.0030,
    "round_channel_share_of_u8_change": 0.677,
    "pearson_pre_rms_vs_abs_delta_beta": 0.632,
    # --- the reductio ---
    "clip_criterion_would_retire": 185,
    "geometry_criterion_would_retire": 1,
    "worst_drift_pair": {"pair": 17, "beta_mag": -7.5, "pre_rms": 4.003,
                         "u8_diff_frac": 0.4391, "clip_sym_diff": 0.0},
    "verdict": "PREDICTION_FALSIFIED_CLAMP_IS_NOT_THE_MECHANISM",
    # formulation-level: the clip-fraction PREDICTOR is dead on this vehicle's
    # stale population.  The clamp CHANNEL is real, just bounded to a ~1.5e-05
    # share -- and the d_pose coupling itself is UNMEASURED here (scorer-free).
    "verdict_scope": "formulation",
    "axis": "[macOS-CPU scorer-free advisory]",
    "score_claim": False,
}

#: MEASURED 2026-08-02, n600, scorer-free: the FULL fit->ship displacement.
#: The census above isolates the BETA axis.  This one compares each shipped
#: composite against the geometry the ``(a,b)`` was actually fitted at
#: (``final_refine.jsonl``), and it makes a second stale partner visible:
#: ddm_fs1 counted beta only, but ``pw1`` moved the pose on 89 rows and ``mq1``
#: moved p1/p2 on 128 more -- all without re-solving ``(a,b)`` (MEASURED: zero
#: rows changed (a,b) across either stage).  178 pairs therefore ship an (a,b)
#: fitted against a POSE that no longer exists.  This does not contradict the
#: 244; it widens the definition from "beta drifted" to "any fitted partner
#: drifted", which is what staleness actually means.
#: Artifact: ``<v4d run dir>/ft1_full/{ft1_channels.jsonl, ft1_receipt.json}``.
FT1_N600_FULL_DISPLACEMENT_CENSUS: dict[str, Any] = {
    "pairs": 600,
    "stale_any_partner": 267,
    "pose_dims_drifted": 178,
    "stale_via_beta_only": 93,
    "stale_via_both": 174,
    "ab_resolved_by_later_stages": 0,
    "clamp_channel_mean": 1.736e-06,
    "clamp_channel_median": 0.0,
    "stale_with_exactly_zero_clamp_channel": 199,
    "clamp_share_of_total_change_mean": 1.481e-05,
    "u8_diff_frac_mean": 0.1171,
    "pre_rms_mean": 0.7616,
    "pre_rms_max": 10.4504,
    "round_channel_share_of_u8_change": 0.625,
    "pairs_with_inert_partner": 0,
    "pearson_clip_frac_vs_abs_delta_beta": -0.019,
    "pearson_clip_sym_diff_vs_abs_delta_beta": 0.051,
    "clip_criterion_would_retire": 199,
    "clip_criterion_retired_max_pre_rms": 6.8034,
    "verdict": "PREDICTION_FALSIFIED_AND_STALE_PARTNER_SET_IS_WIDER",
    "verdict_scope": "formulation",
    "axis": "[macOS-CPU scorer-free advisory]",
    "score_claim": False,
}


def commutator_channels(*, y_fit: np.ndarray, y_ship: np.ndarray,
                        lo: float = 0.0, hi: float = 255.0) -> dict[str, Any]:
    """Decompose ``uint8(y_ship) != uint8(y_fit)`` into its three channels.

    ``y_fit`` and ``y_ship`` are the SAME pair's pre-quantization composites at
    the fit-time and shipped values of the partner coordinate, with the shipped
    photometric map already applied.  Both are float arrays of identical shape.

    The channels are reported separately because they carry different physics
    and different cures: the CLAMP channel is the only operator-level
    non-commutation, the ROUND channel is a universal sub-quantum dither, and
    the GEOMETRY channel is the one that couples through scorer curvature and
    that no clip-fraction argument can see.
    """
    if y_fit.shape != y_ship.shape:
        raise ValueError(f"shape mismatch {y_fit.shape} vs {y_ship.shape}")
    r_fit, r_ship = np.round(y_fit), np.round(y_ship)
    sat_fit = (r_fit < lo) | (r_fit > hi)
    sat_ship = (r_ship < lo) | (r_ship > hi)
    u8_fit = np.clip(r_fit, lo, hi)
    u8_ship = np.clip(r_ship, lo, hi)
    diff = u8_ship != u8_fit
    pre = y_ship - y_fit
    sub_quantum = np.abs(pre) < 0.5
    n = float(u8_fit.size)
    return {
        # CLAMP channel: saturation status flipped -> operator non-commutation.
        "clip_frac_fit": float(sat_fit.mean()),
        "clip_frac_ship": float(sat_ship.mean()),
        "clip_sym_diff": float((sat_fit != sat_ship).mean()),
        "clip_mass_fit": float((np.maximum(lo - y_fit, 0.0)
                                + np.maximum(y_fit - hi, 0.0)).mean()),
        "clip_mass_ship": float((np.maximum(lo - y_ship, 0.0)
                                 + np.maximum(y_ship - hi, 0.0)).mean()),
        # ROUND channel: moved < half a quantum yet changed uint8.
        "round_only_frac": float((diff & sub_quantum).sum() / n),
        # GEOMETRY channel: what the beta actually did to the image.
        "pre_rms": float(np.sqrt(np.mean(pre * pre))),
        "pre_max_abs": float(np.abs(pre).max()),
        "u8_diff_frac": float(diff.mean()),
        "u8_diff_rms": float(np.sqrt(np.mean((u8_ship - u8_fit) ** 2))),
        "beta_inert": bool(not diff.any()),
    }


def clamp_channel_is_inert(channels: Mapping[str, Any], *,
                           tol: float = 0.0) -> bool:
    """Did NO pixel change saturation status between the two geometries?

    True means the clamp contributed EXACTLY zero to the coupling for this pair
    -- the operators commuted on every pixel.  It does NOT mean the pair is
    free of staleness: the geometry and round channels are untouched by this
    predicate, which is precisely why the clip-fraction retirement failed.
    """
    return float(channels["clip_sym_diff"]) <= tol


def staleness_priority(channels: Mapping[str, Any]) -> dict[str, Any]:
    """Rank a stale coefficient by the drift it is actually applied across.

    The mixed partial that makes staleness cost anything is
    ``<grad g, dX/dbeta> + a*<Hess_g @ dX/dbeta, X>``: both terms are first
    order in ``dX/dbeta``, so the pre-quantization RMS displacement is the
    leading-order ranking key.  MEASURED to correlate ``+0.63`` with
    ``|delta beta|`` while the clip fraction correlates ``-0.03``.

    ``clip_sym_diff`` is returned only as an ADDITIVE bound on the clamp
    channel.  It is NOT the key and must never be used as one: on this
    population it would retire the largest-drift pair in the set.
    """
    pre = float(channels["pre_rms"])
    return {
        "key": pre,
        "key_name": "pre_rms",
        "u8_diff_frac": float(channels["u8_diff_frac"]),
        "clamp_channel_bound": float(channels["clip_sym_diff"]),
        "retirable": bool(channels.get("beta_inert", False)),
        "forbidden_key": "clip_frac_ship/clip_sym_diff -- FALSIFIED n600 "
                         "2026-08-02, see FT1_N600_CENSUS",
    }


def transport_admissible(*, delta_partner: float,
                         fit_menu: Sequence[float] = FIT_MENU_MAGNITUDES,
                         opposes_fit_sign: bool = False) -> dict[str, Any]:
    """May a FIRST-ORDER transport be used, or is a joint re-solve required?

    Linear transport is a Taylor step; it carries warrant only where the fit
    actually sampled.  Outside the fitted hull -- a magnitude past the menu, or
    a sign the menu could not express -- the correct action is the real joint
    ``(a,b,beta)`` Gauss-Newton, not a longer extrapolation.  Returning the
    boundary explicitly is what keeps the two from being blurred.
    """
    lo, hi = min(fit_menu), max(fit_menu)
    mag = abs(float(delta_partner))
    outside_magnitude = bool(mag > hi + 1e-12 or mag < lo - 1e-12)
    outside = bool(outside_magnitude or opposes_fit_sign)
    return {
        "admissible": not outside,
        "outside_magnitude": outside_magnitude,
        "outside_sign": bool(opposes_fit_sign),
        "fit_hull": (lo, hi),
        "required_action": "linear_fiber_transport" if not outside
                           else "joint_gauss_newton_resolve",
    }


def fiber_transport_delta_ab(*, hessian_ab: np.ndarray,
                             mixed_partial: np.ndarray,
                             delta_partner: float) -> np.ndarray:
    """First-order repair of a stale ``(a,b)`` after its partner moved.

    ``beta`` is the BASE and ``(a,b)`` a FIBER over it; the staleness is a
    holonomy.  Differentiating the Gauss-Newton stationarity condition
    ``dL/d(a,b) = 0`` along ``beta`` (implicit function theorem) gives::

        delta(a,b) = -H_ab^{-1} @ (d2L/d(a,b)dbeta) * delta_beta

    ``H_ab`` is the 2x2 Gauss-Newton Hessian the solver ALREADY forms from its
    ``GAIN_FD`` / ``BIAS_FD`` finite differences; ``mixed_partial`` costs one
    additional difference in ``beta`` per pair and is exactly the quantity
    sequential coordinate descent discards.  Both arguments must come from the
    SAME linearization point as the shipped coefficients.

    This is a Taylor step, not a solve: gate it with ``transport_admissible``.
    """
    h = np.asarray(hessian_ab, np.float64).reshape(2, 2)
    m = np.asarray(mixed_partial, np.float64).reshape(2)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(m)):
        raise ValueError("non-finite transport inputs")
    # Symmetric GN Hessians are PSD; a singular one means the fit direction is
    # unidentified, and an unidentified direction must not be transported.
    w = np.linalg.eigvalsh(0.5 * (h + h.T))
    if float(w.min()) <= 1e-12 * max(float(w.max()), 1.0):
        raise ValueError("H_ab is singular/indefinite -- direction "
                         "unidentified at the fit point; transport refused")
    return -np.linalg.solve(h, m) * float(delta_partner)
