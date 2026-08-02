"""ddm_lg1 (#808) — the CONSTRAIN-AND-PROTECT layer for burn-4.

Scorer-coordinate Lane guard. Three protection mechanisms, all expressed as
ADDITIVE per-pixel loss-weight addends folded into the tr1 trainer's EXISTING
``seg_pixel_w`` hook (train_tr1_partition_renderer_mlx.py L1517-1520):

  1. lambda_lane primal-dual constraint  — hold realized Lane error <= the ep641
     endpoint budget. Dual ascent at GATE cadence; the multiplier lambda_lane
     enters the loss as extra per-GT-Lane-pixel weight.
  2. born_lane protection mask            — up-weight the currently-WON Lane
     support (gt==Lane & realized==Lane) so bulk-class descent cannot erode it;
     weighted by the measured Lane head sensitivity.
  3. lane margin-floor emphasis           — extra weight on low-QA80-margin Lane
     pixels (the flip-prone tail), floor DERIVED from the Lane margin field p10.

DEFAULT-OFF and byte-identical when disabled: the trainer calls this ONLY under
``if cfg.lane_guard``; when off, nothing here runs, no state is touched, no RNG.
Consumes the a1 gate's EXISTING realized argmax (``realized_gate._realized_argmax``)
=> ZERO new scorer passes.

Authority: ``[macOS-CPU advisory]``; ``research_only=True``; ``score_claim=False``;
pointer ``0.1910828242 [contest-CPU]`` UNMOVED. Everything here is MEANS.

Scorer-coordinate provenance (constants-are-poison: every default carries its
value-provenance LADDER CLASS and its source.  Re-derived at source 2026-08-01 by
ddm_hl1; one entry below is class 4 / UNVERIFIED and says so — the previous blanket
claim that every default was artifact-DERIVED was itself the disguise this file's
own copied-table bug wore):

  * Budget ``LANE_BUDGET_S_UNITS = 0.12589`` (Lane per-class S at the ep641
    endpoint) — /Volumes/VertigoDataTier/pact/ddm_xp1_20260731/xp1_verdict.json
    ``base_per_class_S_units[1]``; ckpt stage_seg_trunk_tau_final.npz sha256
    40553db8be98215a67205d3670aa15d9b9edbe2322380ce169d8448af670f2db (ep641).
    d_seg units = S/100 = 0.0012589.
    LADDER CLASS 3 (measured_anchor), re-verified 2026-08-01: value and ckpt sha
    both match the artifact EXACTLY.  Caveat: the artifact is OUT OF REPO on an
    external volume, so no gate can check it and it is unverifiable on any other
    host — the citation is only as durable as that volume.
  * Lane head sensitivity ratio (derive_lane_head_sensitivity_ratio) — the four
    Lane-pair rank-4 head normals are the four LARGEST of all ten class pairs:
    Lane-Movable 4.007, Road-Lane 3.953, Lane-MyCar 3.862, Lane-Undrivable 3.748
    (mean 3.8925) vs all-ten-pair mean 3.2544 => 1.1961.  RESOLVED THROUGH the
    canonical producer ``tac.canonical_equations.segnet_head_rank4_flipdist_20260715``
    (equation ``segnet_head_rank4_linear_flipdist_v1``; head is EXACTLY rank-4
    linear, flip distance d = |margin_cc'| / ||w_c - w_c'||) — NOT copied as
    literals.  Memo: .omx/research/segnet_recursive_fractal_factorization_20260715.md §2.
  * Measured unprotected erosion ``EROSION_S_MEASURED = 0.00151`` (rung-1's
    unprotected continuation eroded the Lane pool by +0.00151 S while bulk classes
    descended) — xp1 (task #808 brief) — the scale that DERIVES eta_lambda.
    LADDER CLASS 4 (hardcoded), UNVERIFIED as of 2026-08-01: the value appears
    NOWHERE in xp1_verdict.json (all 23 keys checked); the citation is a prose
    brief, not a machine-readable artifact, and it carries no typed
    HardcodedWaiverCustody.  RE-DERIVATION TRIGGER: emit the rung-1 unprotected
    Lane erosion into a machine-readable verdict artifact and cite that, OR
    record typed waiver custody naming an owner.  Owner: lg1 successor.
    Consequence if wrong: it scales ``derive_eta_lambda`` only (dual step size),
    so an error changes how FAST the constraint engages, not the budget it holds.
  * Per-class Lane error definition matches the budget EXACTLY: ddm_qa92
    ``_per_class_flip_counts`` + P formula ``100 * flips / (n*384*512)``
    (experiments/ddm_qa92_carrier_discriminator.py L188, L353).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS

# ---- MEASURED scorer-coordinate constants (comma10k canonical order) -----------
LANE_CLASS = 1  # [Road, Lane, Undrivable, Movable, MyCar] — MEASURED (CLAUDE.md); NEVER luma-sort
N_CLASSES = 5
SEG_H, SEG_W = 384, 512  # frozen SegNet argmax plane (modules.py preprocess)

# ddm_xp1_20260731/xp1_verdict.json base_per_class_S_units[1] @ ep641 endpoint.
LANE_BUDGET_S_UNITS = 0.12589
LANE_BUDGET_DSEG = LANE_BUDGET_S_UNITS / 100.0  # 0.0012589

# Rank-4 head class-pair normals ||w_c - w_c'||, RESOLVED from the canonical producer
# (equation ``segnet_head_rank4_linear_flipdist_v1``) rather than copied as literals: a
# literal copied out of a measured artifact LOOKS wired, passes review, and cannot track
# its source — if the head is re-measured the copy goes stale in silence.  Descending
# order is fixed so the derived mean is bit-reproducible.
# The FOUR Lane pairs are the four LARGEST of all ten (the frozen net amplifies Lane).
_LANE_PAIR_NORMS = tuple(
    sorted((v for k, v in HEAD_PAIR_NORMS.items() if "Lane" in k.split("-")), reverse=True)
)
_ALL_PAIR_NORMS = tuple(sorted(HEAD_PAIR_NORMS.values(), reverse=True))

# Fail-closed shape canary: the frozen 5-class head has exactly C(5,2)=10 pairs, 4 of
# which touch Lane.  If the canonical producer ever changes shape (class renamed, class
# added), REFUSE at import rather than silently averaging the wrong set.
if len(_ALL_PAIR_NORMS) != 10 or len(_LANE_PAIR_NORMS) != 4:  # pragma: no cover - guard
    raise ImportError(
        "lane_guard: canonical HEAD_PAIR_NORMS shape changed "
        f"({len(_ALL_PAIR_NORMS)} pairs, {len(_LANE_PAIR_NORMS)} Lane pairs; expected 10/4) "
        "— re-derive the Lane head-sensitivity ratio before using the guard"
    )

# xp1-measured unprotected Lane erosion over the rung-1 continuation (S-units).
EROSION_S_MEASURED = 0.00151

# The dual's INTEGRATION TIME in gates: a sustained violation needs this many gates to
# drive lambda from 0 to lambda_target.  Single source for BOTH the dual step size
# (derive_eta_lambda) and the ratchet's averaging window (derive_ratchet_budget) so the
# two loops run at the SAME bandwidth and cannot resonate against each other.
N_GATES_TO_ENGAGE_DEFAULT = 10


def derive_lane_head_sensitivity_ratio() -> float:
    """Ratio of mean Lane-pair head-normal magnitude to the all-pair mean (MEASURED,
    fractal memo §2).  ~1.196: per unit penultimate perturbation, Lane boundaries move
    ~1.2x more than the average class pair — so Lane content is proportionally more
    scorer-sensitive and protection on it is proportionally more valuable."""
    return float(np.mean(_LANE_PAIR_NORMS) / np.mean(_ALL_PAIR_NORMS))


LANE_HEAD_SENSITIVITY_RATIO = derive_lane_head_sensitivity_ratio()  # 1.19607...


# ---- DERIVED dual hyperparameters (no bare constants) --------------------------
def derive_eta_lambda(
    lambda_target: float = 1.0,
    n_gates_to_engage: int = N_GATES_TO_ENGAGE_DEFAULT,
    erosion_s: float = EROSION_S_MEASURED,
) -> tuple[float, dict[str, Any]]:
    """Dual step size eta_lambda, DERIVED so a SUSTAINED violation of the measured
    erosion magnitude (``erosion_s`` S-units) drives lambda from 0 to ``lambda_target``
    over ``n_gates_to_engage`` gates:

        eta_lambda = lambda_target / (n_gates_to_engage * erosion_s)

    Rationale for each input (all MEASURED / principled, none tuned-to-fit):
      * lambda_target = 1.0 — one extra unit of per-Lane-pixel loss weight, the SAME
        natural scale as the sn1 ``class_weight_lane`` lever (1.0 = off; +1.0 doubles
        the marginal Lane pressure).
      * n_gates_to_engage = 10 — a DELIBERATELY slow engage so the dual reacts to a
        persistent drift, not single-gate descent noise (the a1 gate's own smooth
        channel varies gate-to-gate).
      * erosion_s = 0.00151 — the xp1-MEASURED unprotected Lane erosion; the typical
        sustained-violation magnitude the constraint must answer.

    At steady erosion (g ~ erosion_s) one gate's update is eta*g ~ lambda_step_cap
    (see derive_lambda_step_cap) => the caps-law-bounded step is self-consistent."""
    eta = float(lambda_target) / (float(n_gates_to_engage) * float(erosion_s))
    prov = {
        "formula": "lambda_target / (n_gates_to_engage * erosion_s)",
        "lambda_target": float(lambda_target),
        "n_gates_to_engage": int(n_gates_to_engage),
        "erosion_s": float(erosion_s),
        "erosion_s_source": "ddm_xp1_20260731 unprotected rung-1 Lane erosion",
        "value": eta,
    }
    return eta, prov


def derive_lambda_step_cap(
    lambda_target: float = 1.0, n_gates_to_engage: int = N_GATES_TO_ENGAGE_DEFAULT,
) -> float:
    """Per-gate dual-step ceiling (caps-law): |d lambda| <= lambda_target/n_gates so
    NO single gate can move the multiplier more than one engage-increment — one gate
    cannot thrash the primal."""
    return float(lambda_target) / float(n_gates_to_engage)


# ---- the BUDGET SCHEDULE (the #822 defect: a constant budget can never bind) ---------
#
# MEASURED DEFECT (re-derived at source 2026-08-01 by ddm_bs2 from the burn-4 primary
# telemetry, /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl,
# 64 `lane_guard` gate rows):
#
#     lambda_lane   == 0.0        on 64 / 64 gates   (100% of mass at the KKT lower bound)
#     g = realized - budget < 0   on 64 / 64 gates   (max g = -0.003452, min = -0.053665)
#     budget_s_units               ONE value, 0.12589, on 64 / 64 gates
#     realized_lane_s              0.122438 -> 0.072225   (the run won 0.050213 S of Lane)
#
# The budget was pinned at the ep641 STARTING level, so the constraint was slack at gate 1
# and got monotonically SLACKER as the primal descended.  A constraint that is satisfied by
# a growing margin has multiplier 0 forever: that is correct KKT on a mis-specified problem,
# not a broken instrument.  Concretely the constant budget licenses the primal to give back
# ALL 0.050213 S-units of won Lane before the guard can begin to respond.
#
# WHY A RATCHET IS THE DERIVED SHAPE (not a picked one).  Lane's cost is not symmetric in
# time.  Lane is 985 GT components on a 384x512 plane, 44-55% of them already erased, and a
# thin component that loses its last supporting pixels has NO gradient support left to
# recover through - erasure is effectively irreversible (the prominence/erasure result;
# per_component_min_flip_distance is the local statement of the same geometry).  The correct
# constraint for an irreversible loss is "do not give back what has been won", i.e. a
# MONOTONE NON-INCREASING budget that tracks the achieved level.  Feasibility is free: the
# target is always a level the run has ALREADY HELD for `mean_gates` gates, never an
# extrapolation.
#
# WHY THE DEADBAND IS DERIVED, NOT PICKED.  A zero-deadband ratchet locks onto a lucky low
# gate and then binds forever = a thrash generator (gc13, verbatim: "Never close a loop below
# its noise floor").  So the budget sits k*sigma above the achieved mean, with:
#   * sigma  MEASURED online from the guard's OWN realized series (first-difference
#            estimator, trend-agnostic), never a constant;
#   * k      DERIVED by requiring that NOISE ALONE cannot move the dual by more than one
#            step cap over the horizon (see derive_deadband_k).
#
# WHY THE RATCHET ALSO FIXES A SECOND, INDEPENDENT DEFECT.  The constant budget compares a
# 36-of-600-pair gate estimate against a constant that was measured on all 600 pairs - an
# apples-to-oranges comparison carrying gd1's MEASURED +3.34% Lane design error.  The gate
# subset is FIXED (`gate_ids_n == 36` and `gate_basis == "ema_shadow"` on all 64 rows), so a
# ratchet - which compares the estimator only to ITSELF - is immune to that bias by
# construction: any constant subset offset cancels in the difference.

# lambda_max: bounded safety ceiling at 5x the natural unit (a sustained SEVERE
# violation can lift Lane weight to ~5, comparable to a strong class_weight_lane;
# above that the primal would be dominated by one class => refuse further ascent).
LAMBDA_MAX_DEFAULT = 5.0


def derive_noise_floor(
    history: list[float] | tuple[float, ...] | np.ndarray,
    min_gates: int = N_GATES_TO_ENGAGE_DEFAULT,
) -> tuple[float | None, dict[str, Any]]:
    """Gate-to-gate noise floor ``sigma`` of the realized Lane series, MEASURED online
    from the guard's own history — never a constant.

    Estimator: ``sigma = sd(diff(history)) / sqrt(2)`` (the standard trend-agnostic
    first-difference / Allan-style estimator).  First-differencing annihilates ANY
    constant slope exactly, so a descending run does not inflate sigma — which an OLS
    residual sd does whenever the descent has curvature.  Cross-checked on the burn-4
    series (64 gates): first-difference 0.00142148, its outlier-robust MAD twin
    0.00146636 (agree to 3.2%, so no outlier contamination), OLS-detrend 0.00356611
    (2.5x inflated by within-window curvature — this is exactly the estimator NOT to use).

    Returns ``(None, prov)`` until ``min_gates`` samples exist: before that the caller
    MUST fall back to the static budget rather than ratchet on an unmeasured floor.
    """
    arr = np.asarray(list(history), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < max(3, int(min_gates)):
        return None, {
            "estimator": "sd(diff)/sqrt(2)", "n_samples": int(arr.size),
            "min_gates": int(min_gates), "value": None,
            "note": "insufficient history — caller must use the static budget",
        }
    d = np.diff(arr)
    sigma = float(np.std(d, ddof=1) / math.sqrt(2.0))
    mad = float(np.median(np.abs(d - np.median(d))) * 1.4826 / math.sqrt(2.0))
    return sigma, {
        "estimator": "sd(diff)/sqrt(2)  (trend-agnostic first-difference)",
        "n_samples": int(arr.size), "value": sigma,
        "mad_cross_check": mad,
        "mad_agreement_rel": (abs(sigma - mad) / sigma if sigma > 0 else 0.0),
    }


def _std_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _std_normal_sf(x: float) -> float:
    """P(Z > x) = Phi(-x), via erfc (no scipy dependency in the trainer hot path)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _partial_expectation(k: float) -> float:
    """E[(Z-k)^+] for Z~N(0,1) = phi(k) - k*Phi(-k).  Strictly decreasing in k."""
    return _std_normal_pdf(k) - k * _std_normal_sf(k)


def derive_deadband_k(
    sigma: float,
    eta_lambda: float,
    lambda_step_cap: float,
    n_gates_horizon: int,
) -> tuple[float, dict[str, Any]]:
    """Deadband multiplier ``k``, DERIVED from the requirement that NOISE ALONE cannot
    move the dual by more than ONE step cap over the horizon.

    With a one-sided deadband at ``k*sigma`` and residuals ~N(0, sigma), the expected
    dual drift accumulated purely from noise over ``W`` gates is

        E[ sum_W eta * max(0, resid - k*sigma) ] = eta * W * sigma * E[(Z-k)^+]

    Requiring that to stay at or below one ``lambda_step_cap`` gives the closed condition

        phi(k) - k*Phi(-k)  <=  lambda_step_cap / (eta * sigma * W)

    solved here by bisection (the left side is strictly decreasing in k).  Every input is
    MEASURED or already DERIVED — sigma online, eta and cap from derive_eta_lambda /
    derive_lambda_step_cap, W the gate horizon — so k is a formula output, never a knob.
    It carries the right physics in both directions: a noisier gate or a longer horizon
    widens the deadband (k grows like sqrt(2 ln W), the Bonferroni-like scaling), and a
    perfectly clean gate collapses it to zero.

    *** THIS IS A LOWER BOUND ON THE REQUIRED k, NOT THE DEADBAND TO SHIP. ***
    The derivation above prices the residual against a FIXED reference level.  The budget
    this guard actually uses is a RUNNING MINIMUM, and the minimum of W correlated trailing
    means is biased low by roughly ``(sigma/sqrt(m)) * sqrt(2 ln W)`` — selection bias that
    EATS the deadband.  MEASURED (ddm_bs2 2026-08-01, 200 null trials x 64 gates at the
    burn-4 sigma): shipping this analytic k gave a null engagement rate of 36.2% of gates,
    200/200 trials engaging — a thrash generator, exactly the failure the deadband exists
    to prevent.  ``calibrate_deadband_k`` keeps this value only as the bisection's lower
    bracket and solves the SAME condition against the true running-min statistic.
    """
    if not (sigma > 0.0) or not (eta_lambda > 0.0) or not (lambda_step_cap > 0.0):
        return 0.0, {
            "formula": "phi(k) - k*Phi(-k) = cap/(eta*sigma*W)", "value": 0.0,
            "note": "degenerate input (sigma/eta/cap non-positive) — deadband disabled",
            "sigma": float(sigma), "eta_lambda": float(eta_lambda),
            "lambda_step_cap": float(lambda_step_cap),
        }
    w = max(1, int(n_gates_horizon))
    rhs = float(lambda_step_cap) / (float(eta_lambda) * float(sigma) * w)
    if rhs >= _partial_expectation(0.0):  # noise is already too small to matter
        k = 0.0
    else:
        lo, hi = 0.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _partial_expectation(mid) > rhs:
                lo = mid
            else:
                hi = mid
        k = 0.5 * (lo + hi)
    return float(k), {
        "formula": "solve phi(k) - k*Phi(-k) = lambda_step_cap/(eta_lambda*sigma*W)",
        "sigma": float(sigma), "eta_lambda": float(eta_lambda),
        "lambda_step_cap": float(lambda_step_cap), "n_gates_horizon": int(w),
        "rhs": float(rhs), "value": float(k), "deadband_s_units": float(k) * float(sigma),
    }


def _null_expected_max_lambda(
    k: float, gain: float, lam_max_rel: float,
    horizon: int, mean_gates: int, n_trials: int, seed: int,
) -> float:
    """``E[ max_t lambda(t) ] / lambda_step_cap`` under the NULL (a stationary series plus
    N(0,sigma) noise, i.e. no erosion at all), driving the EXACT ratchet + projected-dual
    arithmetic this module ships.  Deterministic for a given ``seed``; vectorised over
    trials.

    NORMALISED, because the statistic is scale-free in a single dimensionless group.
    Write noise = sigma*z.  Then level = sigma*mean(z), budget = sigma*(min_t mean(z)+k),
    g = sigma*(z_t - mean - k), and

        step = clip(eta*g, +-cap) = cap * clip( (eta*sigma/cap) * (z - mean - k), +-1 )
        lambda = cap * clip( sum steps, 0, lambda_max/cap )

    so lambda/cap depends on (sigma, eta, cap, lambda_max) ONLY through
    ``gain = eta*sigma/cap`` and ``lam_max_rel = lambda_max/cap``.  Calibrating in these
    two numbers rather than four collapses the memo key and makes the shipped k a
    function of the run's actual loop gain — which is what it physically is.
    """
    rng = np.random.default_rng(seed)
    m = max(1, int(mean_gates))
    n, w = int(n_trials), int(horizon)
    z = rng.normal(0.0, 1.0, (n, w))
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(z, axis=1)], axis=1)
    idx = np.arange(w)
    lo = np.maximum(0, idx + 1 - m)
    level = (csum[:, idx + 1] - csum[:, lo]) / (idx + 1 - lo)[None, :]
    budget = np.minimum.accumulate(level + k, axis=1)          # the RUNNING MINIMUM
    g = z - budget
    step = np.clip(gain * g, -1.0, 1.0)
    lam = np.zeros(n)
    peak = np.zeros(n)
    for t in range(w):                                          # rectified integrator
        lam = np.clip(lam + step[:, t], 0.0, lam_max_rel)
        peak = np.maximum(peak, lam)
    return float(peak.mean())


def calibrate_deadband_k(
    sigma: float,
    eta_lambda: float,
    lambda_step_cap: float,
    lambda_max: float = LAMBDA_MAX_DEFAULT,
    horizon: int = 64,
    mean_gates: int = N_GATES_TO_ENGAGE_DEFAULT,
    n_trials: int = 128,
    seed: int = 20260801,
) -> tuple[float, dict[str, Any]]:
    """The SHIPPED deadband multiplier: the SAME condition as ``derive_deadband_k``
    ("noise alone must not move the dual by more than one ``lambda_step_cap`` over the
    horizon"), but evaluated against the TRUE running-minimum statistic instead of the
    fixed-reference analytic surrogate.

    Only the evaluation changes, never the condition.  The analytic k remains the
    bisection's lower bracket, so the shipped value is always >= the analytic one and the
    gap IS the min-selection bias, reported as ``selection_bias_k``.  Bisection is on a
    monotone statistic (larger k => looser budget => smaller null lambda), seeded, and run
    once at build time — it never touches the training hot path.

    Why not just widen k by a fudge factor: the bias depends on sigma, ``mean_gates`` AND
    the horizon jointly, and it interacts with the step cap and the lambda>=0 projection
    (the dual is a clipped, rectified integrator, not a linear filter).  Simulating the
    shipped arithmetic is the only way to price it without inventing a constant.
    """
    k_lo, prov_lo = derive_deadband_k(sigma, eta_lambda, lambda_step_cap, horizon)
    if not (sigma > 0.0) or not (eta_lambda > 0.0) or not (lambda_step_cap > 0.0):
        return k_lo, {"calibrated": False, "reason": "degenerate input",
                      "analytic": prov_lo, "value": float(k_lo)}
    gain = float(eta_lambda) * float(sigma) / float(lambda_step_cap)
    lam_max_rel = float(lambda_max) / float(lambda_step_cap)
    kw = {"gain": gain, "lam_max_rel": lam_max_rel, "horizon": int(horizon),
          "mean_gates": int(mean_gates), "n_trials": int(n_trials), "seed": int(seed)}
    lo, hi = float(k_lo), float(k_lo)
    for _ in range(40):                       # expand until the null is quiet enough
        if _null_expected_max_lambda(hi, **kw) <= 1.0:
            break
        lo, hi = hi, (hi * 2.0 if hi > 1e-9 else 1.0)
    else:                                     # pragma: no cover - unreachable in practice
        return hi, {"calibrated": False, "reason": "no k satisfied the null bound",
                    "analytic": prov_lo, "value": float(hi)}
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if _null_expected_max_lambda(mid, **kw) > 1.0:
            lo = mid
        else:
            hi = mid
    k = float(hi)
    return k, {
        "calibrated": True,
        "condition": "E[max_t lambda | NULL] <= lambda_step_cap",
        "value": k,
        "analytic_lower_bracket_k": float(k_lo),
        "selection_bias_k": k - float(k_lo),
        "null_expected_max_lambda_rel": _null_expected_max_lambda(k, **kw),
        "loop_gain_eta_sigma_over_cap": gain,
        "lambda_max_rel": lam_max_rel,
        "n_trials": int(n_trials), "seed": int(seed), "horizon": int(horizon),
        "mean_gates": int(mean_gates), "deadband_s_units": k * float(sigma),
        "analytic": prov_lo,
    }


_K_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}


def _cached_calibrated_k(
    sigma: float, eta_lambda: float, lambda_step_cap: float,
    lambda_max: float, horizon: int, mean_gates: int,
) -> tuple[float, dict[str, Any]]:
    """Memoised ``calibrate_deadband_k``, invoked once per gate, so the key quantises the
    two inputs that drift — and does so in the SCALE-FREE coordinates the statistic
    actually depends on (see ``_null_expected_max_lambda``):

      * the loop gain ``eta*sigma/cap`` to 3 significant figures.  k varies only
        logarithmically in the gain, so 0.1% gain granularity is far below any effect on
        the deadband, and a drifting sigma stops thrashing the cache.
      * ``horizon`` UP to the next power of two, CONSERVATIVE by construction (a longer
        horizon yields a larger k, a wider deadband, a quieter guard) and capping the
        recalibrations per run at log2(n_gates).

    k is scale-free but the DEADBAND is not: the returned k is always multiplied by the
    caller's live sigma, so quantising the key never quantises the shipped budget.
    """
    h = 1
    while h < max(1, int(horizon)):
        h <<= 1
    gain = float(eta_lambda) * float(sigma) / float(lambda_step_cap)
    gain_q = float(f"{gain:.3g}") if gain > 0.0 else 0.0
    key = (gain_q, h, int(mean_gates),
           round(float(lambda_max) / float(lambda_step_cap), 3))
    hit = _K_CACHE.get(key)
    if hit is None:
        # Re-express the quantised gain at the caller's cap/eta so the simulated loop is
        # the caller's loop; sigma_eff differs from sigma only by the 3-sig-fig rounding.
        sigma_eff = gain_q * float(lambda_step_cap) / float(eta_lambda)
        k, prov = calibrate_deadband_k(
            sigma=sigma_eff, eta_lambda=float(eta_lambda),
            lambda_step_cap=float(lambda_step_cap), lambda_max=float(lambda_max),
            horizon=h, mean_gates=int(mean_gates))
        prov = dict(prov, cache_key_horizon_pow2=h, cache_key_gain_3sf=gain_q,
                    live_sigma=float(sigma), deadband_s_units=k * float(sigma))
        _K_CACHE[key] = (k, prov)
        hit = (k, prov)
    return hit[0], dict(hit[1], live_sigma=float(sigma),
                        deadband_s_units=hit[0] * float(sigma))


def derive_ratchet_budget(
    history: list[float] | tuple[float, ...] | np.ndarray,
    prev_budget_s: float,
    eta_lambda: float,
    lambda_step_cap: float,
    mean_gates: int = N_GATES_TO_ENGAGE_DEFAULT,
    n_gates_horizon: int | None = None,
    lambda_max: float = LAMBDA_MAX_DEFAULT,
    calibrate: bool = True,
    k_override: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """The BUDGET SCHEDULE: a monotone non-increasing ratchet that locks in won Lane.

        L_hat(t) = mean of the last ``mean_gates`` realized values   (achieved level)
        sigma    = derive_noise_floor(history)                       (measured floor)
        k        = derive_deadband_k(sigma, eta, cap, W)             (derived deadband)
        budget(t) = min( budget(t-1),  L_hat(t) + k*sigma )          (RATCHET)

    ``mean_gates`` is the dual's own integration time (N_GATES_TO_ENGAGE_DEFAULT): the
    budget's averaging window and the dual's engagement window are the SAME bandwidth, so
    the level loop and the dual loop cannot resonate.  ``n_gates_horizon`` defaults to the
    self-deriving ``max(mean_gates, len(history))`` — the deadband widens as the run
    accumulates more chances for a spurious exceedance, which is the correct scaling.

    Fails SAFE: with too little history to measure sigma the previous budget is returned
    unchanged, so the guard degrades exactly to its pre-ratchet behaviour rather than
    ratcheting on an unmeasured noise floor.
    """
    hist = np.asarray(list(history), dtype=np.float64).ravel()
    # R1 review catch: derive_noise_floor filters non-finite values but the trailing MEAN
    # below did not, so a single non-finite realized value would poison the budget through
    # min(prev, nan) — whose result is order-dependent.  Filter ONCE, here, for both.
    hist = hist[np.isfinite(hist)]
    m = max(1, int(mean_gates))
    sigma, sig_prov = derive_noise_floor(hist, min_gates=m)
    if sigma is None:
        return float(prev_budget_s), {
            "engaged": False, "reason": "insufficient_history",
            "budget_s": float(prev_budget_s), "sigma": sig_prov,
        }
    w = int(n_gates_horizon) if n_gates_horizon else max(m, int(hist.size))
    if k_override is not None:
        k = float(k_override)
        k_prov = {"value": k, "source": "k_override (caller-supplied; A/B or test)"}
    elif calibrate:
        k, k_prov = _cached_calibrated_k(sigma, eta_lambda, lambda_step_cap, lambda_max, w, m)
    else:
        k, k_prov = derive_deadband_k(sigma, eta_lambda, lambda_step_cap, w)
    level = float(np.mean(hist[-m:]))
    target = level + k * sigma
    new_budget = min(float(prev_budget_s), target)
    return new_budget, {
        "engaged": True,
        "budget_s": float(new_budget),
        "prev_budget_s": float(prev_budget_s),
        "achieved_level_s": level,
        "mean_gates": m,
        "sigma_s": float(sigma),
        "deadband_k": float(k),
        "deadband_s": float(k * sigma),
        "ratchet_target_s": float(target),
        "tightened_by_s": float(prev_budget_s) - float(new_budget),
        "sigma_provenance": sig_prov,
        "k_provenance": k_prov,
    }


def derive_margin_floor(
    lane_margins: np.ndarray, pct: float = 10.0,
) -> tuple[float, dict[str, Any]]:
    """Margin floor DERIVED as the ``pct``-th percentile of the QA80 margin field
    RESTRICTED to GT-Lane pixels (the flip-prone low-margin tail — 100% of flips are
    in the bottom GT-margin decile; sg1 §1.3).  Data-derived per run, never a bare
    constant.  Cross-check: the fractal memo §3 measured Road-Lane FEATURE flip-dist
    p10 = 0.024 (a different metric; this floor lives in the trainer's QA80 field)."""
    vals = np.asarray(lane_margins, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, {"note": "no lane-margin samples", "pct": float(pct), "value": 0.0}
    floor = float(np.percentile(vals, pct))
    return floor, {
        "formula": f"percentile_{pct}(QA80 margin | gt==Lane)",
        "pct": float(pct),
        "n_samples": int(vals.size),
        "value": floor,
        "cross_check_fractal_memo_feature_p10": 0.024,
    }


# ---- budget-matched realized Lane error (matches xp1/qa92 exactly) --------------
def per_class_lane_flip_S(
    realized: np.ndarray, gts: np.ndarray, lane_class: int = LANE_CLASS,
) -> float:
    """Realized Lane per-class error in S-units, using the EXACT ddm_qa92 definition
    so it is directly comparable to the ``LANE_BUDGET_S_UNITS`` budget:

        S_lane = 100 * sum_pairs #(gt==Lane & realized != gt) / (n_pairs * H * W)

    ``realized`` / ``gts`` are (n, H, W) int arrays (or stacked lists thereof)."""
    realized = np.asarray(realized)
    gts = np.asarray(gts)
    if realized.shape != gts.shape:
        raise ValueError(f"realized {realized.shape} != gts {gts.shape}")
    n = realized.shape[0]
    total_px = float(n * realized.shape[1] * realized.shape[2])
    flip = realized != gts
    lane_flips = int(((gts == lane_class) & flip).sum())
    return 100.0 * lane_flips / total_px


def born_lane_support_mask(
    realized: np.ndarray, gt: np.ndarray, lane_class: int = LANE_CLASS,
) -> np.ndarray:
    """(H,W) float32 mask of the currently-WON Lane support: pixels where the GT is
    Lane AND the realized argmax already agrees (gt==Lane & realized==Lane).  These
    are exactly the pixels holding the born (surviving) Lane components alive — the
    content the unprotected continuation ERODES first.  Protecting them (extra loss
    weight) preserves won structure without blanket-freezing all Lane.

    Scipy-free (a correctly-classified Lane pixel is by construction inside a
    surviving component); the per-component / per-cell Fisher-anchor refinement is
    the DEFERRED heavier variant (see the memo deferred table)."""
    realized = np.asarray(realized)
    gt = np.asarray(gt)
    return ((gt == lane_class) & (realized == lane_class)).astype(np.float32)


def per_component_min_flip_distance(
    margin: np.ndarray,
    born_mask: np.ndarray,
    dw_norm: float = max(_LANE_PAIR_NORMS),
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form per-born-component MIN flip distance in the head-hyperplane metric
    (piece-3 helper; the rank-4-head exact form d = |m| / ||dw_cc'|| — fractal memo §3,
    canonical equation ``segnet_head_rank4_linear_flipdist_v1``).

    ``margin``: (H,W) scorer margin field (winner-vs-runner |m|, the QA80/gt-cache field).
    ``born_mask``: (H,W) born-lane support (born_lane_support_mask output).
    ``dw_norm``: the head-normal magnitude; default = max of the four MEASURED Lane-pair
    normals (4.007, Lane-Movable) => the CONSERVATIVE (smallest-d) bound, since the
    binding pair for a component is whichever gives min d.

    Returns ``(labels, min_d)``: 8-conn component labels (0 = background) over the born
    mask + the per-component min flip distance array indexed by label-1.  A per-component
    hinge on min_d is the DEFERRED loss term (see the memo deferred table for the exact
    insertion point in pair_loss); this helper is the landed closed-form surface."""
    from scipy import ndimage

    born = np.asarray(born_mask) > 0.5
    m = np.abs(np.asarray(margin, dtype=np.float64))
    labels, n = ndimage.label(born, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labels, np.zeros((0,), dtype=np.float64)
    min_m = ndimage.minimum(m, labels=labels, index=np.arange(1, n + 1))
    return labels, np.asarray(min_m, dtype=np.float64) / float(dw_norm)


# ---- config + state ------------------------------------------------------------
@dataclass(frozen=True)
class LaneGuardConfig:
    """DERIVED defaults (constants-are-poison).  ``enabled=False`` => the trainer
    never invokes the guard => byte-identical."""

    enabled: bool = False
    budget_s: float = LANE_BUDGET_S_UNITS          # ep641 Lane S (xp1) — the RATCHET's start
    eta_lambda: float = 0.0                          # 0.0 => derive at build (derive_eta_lambda)
    lambda_step_cap: float = 0.0                     # 0.0 => derive (derive_lambda_step_cap)
    lambda_max: float = LAMBDA_MAX_DEFAULT
    born_protect_weight: float = 0.0                 # 0.0 => born mask OFF; scaled by sensitivity
    margin_floor_weight: float = 0.0                 # 0.0 => margin-floor emphasis OFF
    margin_floor_pct: float = 10.0                   # percentile for derive_margin_floor
    lane_sensitivity_ratio: float = LANE_HEAD_SENSITIVITY_RATIO
    # ---- budget SCHEDULE (ddm_bs2 #871) ----
    # DEFAULT FALSE so an existing sealed ticket recompiles BIT-IDENTICAL and any landed
    # run stays reproducible.  "Off" here is NOT a forgotten default: with the ratchet off
    # the guard SELF-REPORTS its own inertness every gate (`inert_slack_gates` +
    # `inertness_alarm` in the gate_update telemetry row), so the state is tracked and
    # surfaced rather than silent.
    budget_ratchet: bool = False
    ratchet_mean_gates: int = N_GATES_TO_ENGAGE_DEFAULT   # matched to the dual's integration time
    ratchet_horizon_gates: int = 0                        # 0 => self-derive max(mean_gates, n_seen)

    def resolved(self) -> LaneGuardConfig:
        """Fill any 0.0-sentinel derived field from its derivation (idempotent).
        Fail-closed on sign-inverting values (a negative eta would make the dual
        DESCEND on violation — the constraint silently becomes a reward)."""
        if (self.eta_lambda < 0.0 or self.lambda_step_cap < 0.0
                or self.lambda_max <= 0.0):
            raise ValueError(
                f"lane_guard config invalid: eta_lambda={self.eta_lambda}, "
                f"lambda_step_cap={self.lambda_step_cap}, lambda_max={self.lambda_max} "
                "(eta/cap must be >= 0, lambda_max > 0)")
        if self.ratchet_mean_gates < 1 or self.ratchet_horizon_gates < 0:
            raise ValueError(
                f"lane_guard config invalid: ratchet_mean_gates={self.ratchet_mean_gates} "
                f"(must be >= 1), ratchet_horizon_gates={self.ratchet_horizon_gates} "
                "(must be >= 0; 0 => self-derive)")
        eta = self.eta_lambda or derive_eta_lambda()[0]
        cap = self.lambda_step_cap or derive_lambda_step_cap()
        return LaneGuardConfig(
            enabled=self.enabled, budget_s=self.budget_s, eta_lambda=eta,
            lambda_step_cap=cap, lambda_max=self.lambda_max,
            born_protect_weight=self.born_protect_weight,
            margin_floor_weight=self.margin_floor_weight,
            margin_floor_pct=self.margin_floor_pct,
            lane_sensitivity_ratio=self.lane_sensitivity_ratio,
            budget_ratchet=self.budget_ratchet,
            ratchet_mean_gates=self.ratchet_mean_gates,
            ratchet_horizon_gates=self.ratchet_horizon_gates,
        )


@dataclass
class LaneGuardState:
    """Mutable dual + protection state.  Updated ONLY at gate cadence."""

    lambda_lane: float = 0.0
    margin_floor: float | None = None
    born_masks: dict[int, np.ndarray] = field(default_factory=dict)
    n_gates: int = 0
    last_g_s: float = 0.0
    last_realized_lane_s: float = 0.0
    # ---- budget-schedule state (ddm_bs2 #871) ----
    realized_history: list[float] = field(default_factory=list)
    budget_s_current: float | None = None   # None => cfg.budget_s (pre-ratchet behaviour)
    inert_slack_gates: int = 0              # consecutive gates with lambda==0 AND g<0


def dual_ascent(
    state: LaneGuardState,
    cfg: LaneGuardConfig,
    realized_lane_s: float,
    budget_s: float | None = None,
) -> float:
    """Bounded projected dual ascent at gate cadence:

        g       = realized_lane_s - budget_s              (constraint violation, S-units)
        step    = clip(eta * g, -cap, +cap)               (caps-law: one gate can't thrash)
        lambda <- clip(lambda + step, 0, lambda_max)       (KKT: multiplier >= 0)

    CAPS-LAW RECONCILIATION: the v9 rule "loss weights change at STAGE boundaries
    only" governs the PRIMAL loss weights.  lambda_lane is NOT a loss weight — it is a
    KKT dual multiplier whose update cadence is, by primal-dual optimization theory,
    the constraint-EVALUATION cadence (here the a1 GATE, the only point the realized
    constraint g is measured).  Within a gate interval lambda is CONSTANT, so the
    primal sees a slowly-varying, per-interval-fixed weight — no per-step thrash,
    consistent with the caps-law spirit.  ``lambda_step_cap`` + ``lambda_max`` bound
    it so a single noisy gate cannot dominate the primal.

    ``budget_s`` overrides ``cfg.budget_s`` for this gate (the RATCHETED budget); None
    keeps the static config budget, i.e. the exact pre-ratchet behaviour."""
    eff_budget = float(cfg.budget_s if budget_s is None else budget_s)
    g = float(realized_lane_s) - eff_budget
    step = max(-cfg.lambda_step_cap, min(cfg.lambda_step_cap, cfg.eta_lambda * g))
    new_lambda = max(0.0, min(cfg.lambda_max, state.lambda_lane + step))
    state.lambda_lane = float(new_lambda)
    state.last_g_s = float(g)
    state.last_realized_lane_s = float(realized_lane_s)
    return state.lambda_lane


def gate_update(
    state: LaneGuardState,
    cfg: LaneGuardConfig,
    realized_argmax: np.ndarray,
    gts: np.ndarray,
    gate_ids: tuple[int, ...] | list[int],
    lane_margins_by_id: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run once per a1 gate.  Reads the EXISTING realized argmax (zero new scorer
    passes), advances the dual, refreshes the born-lane masks for the gate pairs, and
    derives the margin floor on the first gate.  Returns a telemetry row.

    NOTE on geometry (honest): at n600 the gate set is the fd2 SUBSET (~36 pairs), so
    ``realized_lane_s`` here is an UNBIASED-if-representative ESTIMATE of the full-n600
    Lane level the budget was measured on; below n600 the gate set is ALL pairs and it
    is exact.  The dual reacts to a persistent drift regardless of the subset's
    absolute-level noise."""
    state.n_gates += 1
    realized_arr = np.asarray(realized_argmax)
    gts_arr = np.asarray(gts)
    realized_lane_s = per_class_lane_flip_S(realized_arr, gts_arr, LANE_CLASS)
    state.realized_history.append(float(realized_lane_s))

    # --- BUDGET SCHEDULE: ratchet BEFORE the dual sees the violation ---------------
    if state.budget_s_current is None:
        state.budget_s_current = float(cfg.budget_s)
    if cfg.budget_ratchet:
        new_budget, ratchet_prov = derive_ratchet_budget(
            state.realized_history, state.budget_s_current,
            cfg.eta_lambda, cfg.lambda_step_cap,
            mean_gates=cfg.ratchet_mean_gates,
            n_gates_horizon=(cfg.ratchet_horizon_gates or None),
            # R1 review catch: the null calibration depends on lambda_max/cap, so a run that
            # overrides --lane-guard-lambda-max MUST calibrate against ITS ceiling, not the
            # 5.0 default.  Omitting this silently calibrated every run as if lambda_max=5.
            lambda_max=cfg.lambda_max,
        )
        state.budget_s_current = float(new_budget)
    else:
        ratchet_prov = {"engaged": False, "reason": "budget_ratchet_disabled",
                        "budget_s": float(state.budget_s_current)}

    lam = dual_ascent(state, cfg, realized_lane_s, budget_s=state.budget_s_current)

    # --- INERTNESS self-report: "off" is a TRACKED state, never a silent default ----
    # The burn-4 defect signature is exactly (lambda == 0) AND (g < 0) sustained: a guard
    # that cannot engage.  Count it and raise once the run has had a full engagement
    # window of opportunity, so an inert guard can never again be mistaken for a
    # satisfied one by a reader of this telemetry.
    if lam <= 0.0 and state.last_g_s < 0.0:
        state.inert_slack_gates += 1
    else:
        state.inert_slack_gates = 0
    inertness_alarm = state.inert_slack_gates >= max(1, int(cfg.ratchet_mean_gates))

    # born-lane support masks for THIS gate's pairs (used next epoch's pair_loss).
    if cfg.born_protect_weight > 0.0:
        for k, gid in enumerate(gate_ids):
            state.born_masks[int(gid)] = born_lane_support_mask(
                realized_arr[k], gts_arr[k], LANE_CLASS)

    # derive the margin floor once, from the first gate's Lane-restricted margins.
    if cfg.margin_floor_weight > 0.0 and state.margin_floor is None and lane_margins_by_id:
        lane_vals = []
        for k, gid in enumerate(gate_ids):
            m = lane_margins_by_id.get(int(gid))
            if m is not None:
                lane_vals.append(np.asarray(m)[gts_arr[k] == LANE_CLASS])
        if lane_vals:
            floor, _ = derive_margin_floor(np.concatenate(lane_vals), cfg.margin_floor_pct)
            state.margin_floor = floor

    return {
        "event": "lane_guard",
        "n_gates": state.n_gates,
        "realized_lane_s_units": float(realized_lane_s),
        "budget_s_units": float(state.budget_s_current),
        "budget_s_static": float(cfg.budget_s),
        "budget_ratchet": bool(cfg.budget_ratchet),
        "ratchet": ratchet_prov,
        # The guard's own inertness receipt (ddm_bs2 #871): burn-4 ran 64/64 gates in
        # exactly this state and no telemetry field said so.
        "inert_slack_gates": int(state.inert_slack_gates),
        "inertness_alarm": bool(inertness_alarm),
        "g_s_units": float(state.last_g_s),
        "lambda_lane": float(lam),
        # KKT complementarity residual lambda*g (the #549 KKTDiagnostics-aligned surface:
        # ~0 at equilibrium — either the constraint is slack or the dual has settled).
        "complementarity": float(lam * state.last_g_s),
        "lambda_max": float(cfg.lambda_max),
        "eta_lambda": float(cfg.eta_lambda),
        "lambda_step_cap": float(cfg.lambda_step_cap),
        "margin_floor": (None if state.margin_floor is None else float(state.margin_floor)),
        "born_mask_pairs": (len(state.born_masks) if cfg.born_protect_weight > 0.0 else 0),
        "lane_sensitivity_ratio": float(cfg.lane_sensitivity_ratio),
        "score_neutral": False,
        "score_claim": False,
        "evidence_axis": "[macOS-CPU advisory]",
    }


def pixel_weight_addend(
    lstar: np.ndarray,
    lane_margin: np.ndarray | None,
    state: LaneGuardState,
    cfg: LaneGuardConfig,
    idx: int,
) -> np.ndarray | None:
    """Additive per-pixel loss-weight for ONE pair, folded into the trainer's
    ``seg_pixel_w`` (final weight = 1 + class_weight_lane_term + THIS).  Returns None
    if the total addend is identically zero (so ``seg_pixel_w`` stays None and the
    byte path is preserved).

        w += lambda_lane                              on GT-Lane pixels        (constraint)
        w += born_protect_weight * sensitivity        on born-lane support     (protection)
        w += margin_floor_weight  * relu(1 - m/floor) on low-margin GT-Lane    (margin floor)
    """
    lstar = np.asarray(lstar)
    is_lane = (lstar == LANE_CLASS).astype(np.float32)
    addend = np.zeros(lstar.shape, dtype=np.float32)
    active = False

    if state.lambda_lane > 0.0:
        addend = addend + state.lambda_lane * is_lane
        active = True

    if cfg.born_protect_weight > 0.0:
        bm = state.born_masks.get(int(idx))
        if bm is not None:
            addend = addend + (cfg.born_protect_weight * cfg.lane_sensitivity_ratio) * bm
            active = True

    if (cfg.margin_floor_weight > 0.0 and state.margin_floor is not None
            and state.margin_floor > 0.0 and lane_margin is not None):
        m = np.asarray(lane_margin, dtype=np.float32)
        deficit = np.maximum(0.0, 1.0 - m / float(state.margin_floor))
        addend = addend + cfg.margin_floor_weight * deficit * is_lane
        active = True

    if not active or not np.any(addend):
        return None
    return addend


__all__ = [
    "EROSION_S_MEASURED",
    "LAMBDA_MAX_DEFAULT",
    "LANE_BUDGET_DSEG",
    "LANE_BUDGET_S_UNITS",
    "LANE_CLASS",
    "LANE_HEAD_SENSITIVITY_RATIO",
    "N_CLASSES",
    "N_GATES_TO_ENGAGE_DEFAULT",
    "SEG_H",
    "SEG_W",
    "LaneGuardConfig",
    "LaneGuardState",
    "born_lane_support_mask",
    "derive_deadband_k",
    "derive_eta_lambda",
    "derive_lambda_step_cap",
    "derive_lane_head_sensitivity_ratio",
    "derive_margin_floor",
    "derive_noise_floor",
    "derive_ratchet_budget",
    "dual_ascent",
    "gate_update",
    "per_class_lane_flip_S",
    "per_component_min_flip_distance",
    "pixel_weight_addend",
]
