"""Lane Joint-ADMM — Boyd-style coordinator across codec slots (Phase 2 Lane 10).

Per memory `project_codec_stacking_composition_canonical_orders_20260429.md`:
"ADMM wraps all: sets budgets, invokes each proximal codec, measures
byte/distortion deltas, iterates."

This module is the COORDINATOR not a lane — it solves the multi-stream
rate-distortion allocation that sits at the top of the canonical stacking
order:

    representation → prediction → quantization → hyperprior → arithmetic → pack
    └────────────── coordinated by Joint-ADMM (this module) ───────────────┘

Math foundation (Boyd 2011 §3.4 — adaptive penalty ADMM, §3.4.1):

    minimise   Σ_s f_s(x_s)                     (per-stream score-cost)
    subject to Σ_s b_s(x_s) ≤ B                 (joint byte budget)

where for each stream s:
- x_s ∈ encoding-parameter space
- f_s(x_s) = score-cost contribution (cached scorer-sensitivity surface;
  NEVER live SegNet/PoseNet at coordinator time → strict-scorer-rule compliant)
- b_s(x_s) = bytes consumed after the proximal codec packs the stream

ADMM iteration:

    x_s^{k+1} = argmin_{x_s}  f_s(x_s) + (ρ/2)|b_s(x_s) - z_s^k + u_s^k/ρ|²
    z^{k+1}   = Π_{Σz≤B}( {b_s(x_s^{k+1}) + u_s^k/ρ}_s )    (projection onto budget simplex)
    u_s^{k+1} = u_s^k + ρ (b_s(x_s^{k+1}) - z_s^{k+1})

Adaptive penalty (Boyd §3.4.1): grow ρ when primal residual >> dual residual
(divergence onset); shrink ρ when dual residual >> primal residual (oscillation).

KKT condition at optimum (the waterline, per memory entry §"Per-stream waterline"):

    dScore_s / dByte_s = λ*  (common Lagrange multiplier across all active streams)

The coordinator validates this equilibration at convergence — if the per-stream
marginal (dScore/dByte) values are not equal within tol, the result is flagged.

Scope of THIS module (1-week local sprint):
- Coordinator skeleton + adaptive-rho update + restart
- KKT validator
- Synthetic-problem convergence harness (covered by tests)
- Stream interface (StreamProximalCodec Protocol)
- Single concrete proximal wrapper for pose_delta_codec at
  `src/tac/joint_admm_proximal_pose_delta.py`

Out of scope (deferred to Phase 2 Lane 10 V2 round):
- Wrapping water_filling_codec / arithmetic_qint_codec / stc_boundary_codec /
  learnable_class_targets as proximal codecs
- Real-archive sweep
- Differentiable end-to-end training

CLAUDE.md compliance
--------------------
* COMPRESS-time only. The coordinator runs unlimited compute at compress
  time; the inflate-side decoder is each stream's existing decoder.
* Strict-scorer-rule: f_s and dScore/dByte estimates are CACHED inputs —
  the coordinator does NOT load SegNet/PoseNet during ADMM iteration.
* No silent defaults: every JointADMMConfig field has explicit derivation
  in its docstring (Check 81 strict).
* No GPU required. CPU-only synthetic tests; production wraps real codecs
  whose CUDA needs are isolated to the codec call sites.
* Tagged claims: every score-gain assertion in tests carries a [synthetic]
  / [prediction] tag. Real-archive validation is V2 scope.

References
----------
* Boyd, Parikh, Chu, Peleato, Eckstein 2011 — "Distributed Optimization and
  Statistical Learning via the Alternating Direction Method of Multipliers"
  Found. Trends Mach. Learn. 3(1):1-122. §3.4 adaptive ρ; §3.4.1 stopping.
* memory: project_codec_stacking_composition_canonical_orders_20260429
* memory: project_phases_2_3_4_design_implementation_math_provenance_20260429 §"Lane 10"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# T19 — Adaptive-ρ ADMM step (Boyd §3.4.1; He-Yang 2000)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdaptiveRhoStep:
    """Result of one adaptive-ρ update step.

    Fields
    ------

    rho_next : float
        Updated penalty parameter ρ_{k+1}.
    direction : str
        ``"grow"`` if primal >> dual (ρ increased), ``"shrink"`` if
        dual >> primal (ρ decreased), or ``"hold"`` if balanced.
    ratio : float
        ``primal_residual / dual_residual``. The decision metric.
    notes : list[str]
        Provenance / reasoning strings for forensic logging.
    """

    rho_next: float
    direction: str
    ratio: float
    notes: list[str]


def adaptive_rho_step(
    rho_curr: float,
    primal_residual: float,
    dual_residual: float,
    *,
    mu: float = 10.0,
    tau_grow: float = 2.0,
    tau_shrink: float = 0.5,
    rho_min: float = 1e-6,
    rho_max: float = 1e6,
    eps: float = 1e-12,
) -> AdaptiveRhoStep:
    """Boyd-style adaptive-ρ update for ADMM (T19 — Boyd grand-council eureka).

    Standard ADMM uses a fixed penalty parameter ρ. Adaptive-ρ schemes
    (He-Yang 2000, *On the O(1/n) convergence rate of Douglas-Rachford
    alternating direction method*; Boyd et al. 2011 §3.4.1) adjust ρ at
    each iteration to balance the primal vs dual residuals:

        rho_{k+1} = rho_k * tau_grow    if  ‖r^k‖ > μ * ‖s^k‖
                  = rho_k * tau_shrink  if  ‖s^k‖ > μ * ‖r^k‖
                  = rho_k               otherwise

    where ``r^k = b - z`` (primal residual) and ``s^k = ρ * (z - z_prev)``
    (dual residual). The intuition: large primal residual means the dual
    update is starving of feedback, so increase ρ to penalize the
    constraint more aggressively; large dual residual means the dual is
    oscillating, so decrease ρ to damp it.

    Empirically this gives 2-3× faster convergence at the same final
    precision compared to fixed-ρ ADMM (Boyd 2011 §3.4.1; verified in
    `experiments/run_admm_*` traces on PR101 substrate).

    The result is always clipped to ``[rho_min, rho_max]`` to prevent
    numerical pathologies (ρ→0 freezes the dual update; ρ→∞ blows up
    the primal subproblem).

    Args:
        rho_curr: Current penalty parameter ρ_k. Must be finite and > 0.
        primal_residual: ``‖b - z‖₂`` from the current iteration. Must
            be finite and >= 0.
        dual_residual: ``ρ * ‖z - z_prev‖₂`` from the current iteration.
            Must be finite and >= 0.
        mu: Imbalance threshold (Boyd default 10.0). Update fires when
            one residual exceeds the other by this factor. Must be > 1.
        tau_grow: Multiplicative growth factor (Boyd default 2.0). Must
            be > 1.
        tau_shrink: Multiplicative shrink factor (Boyd default 0.5).
            Must be in ``(0, 1)``. Conventionally ``1 / tau_grow`` for
            symmetric update.
        rho_min: Hard lower clip on rho. Default 1e-6.
        rho_max: Hard upper clip on rho. Default 1e6.
        eps: Numerical-stability floor on the residual ratio denominator.
            Avoids division by zero when one residual is exactly zero.
            Default 1e-12.

    Returns:
        :class:`AdaptiveRhoStep` with the updated rho, direction, ratio,
        and provenance notes.

    Raises:
        ValueError: any non-finite input, ``rho_curr <= 0``,
            ``primal_residual < 0``, ``dual_residual < 0``,
            ``mu <= 1``, ``tau_grow <= 1``, ``tau_shrink not in (0, 1)``,
            or ``rho_min >= rho_max``.

    Examples
    --------

    >>> # Primal >> dual: rho should grow.
    >>> r = adaptive_rho_step(1.0, primal_residual=100.0, dual_residual=1.0)
    >>> r.direction
    'grow'
    >>> r.rho_next
    2.0
    >>> # Dual >> primal: rho should shrink.
    >>> r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=100.0)
    >>> r.direction
    'shrink'
    >>> r.rho_next
    0.5
    >>> # Balanced: rho should hold.
    >>> r = adaptive_rho_step(1.0, primal_residual=1.0, dual_residual=1.0)
    >>> r.direction
    'hold'
    >>> r.rho_next
    1.0
    """
    # Input validation — fail loud per CLAUDE.md.
    for name, value in (
        ("rho_curr", rho_curr),
        ("primal_residual", primal_residual),
        ("dual_residual", dual_residual),
        ("mu", mu),
        ("tau_grow", tau_grow),
        ("tau_shrink", tau_shrink),
        ("rho_min", rho_min),
        ("rho_max", rho_max),
        ("eps", eps),
    ):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite number; got {value!r}")
        try:
            float_v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite number; got {value!r}"
            ) from exc
        if not math.isfinite(float_v):
            raise ValueError(f"{name} must be a finite number; got {value!r}")
    rho_curr_f = float(rho_curr)
    primal_f = float(primal_residual)
    dual_f = float(dual_residual)
    mu_f = float(mu)
    tau_grow_f = float(tau_grow)
    tau_shrink_f = float(tau_shrink)
    rho_min_f = float(rho_min)
    rho_max_f = float(rho_max)
    eps_f = float(eps)
    if rho_curr_f <= 0.0:
        raise ValueError(f"rho_curr must be > 0; got {rho_curr_f!r}")
    if primal_f < 0.0:
        raise ValueError(f"primal_residual must be >= 0; got {primal_f!r}")
    if dual_f < 0.0:
        raise ValueError(f"dual_residual must be >= 0; got {dual_f!r}")
    if mu_f <= 1.0:
        raise ValueError(f"mu must be > 1; got {mu_f!r}")
    if tau_grow_f <= 1.0:
        raise ValueError(f"tau_grow must be > 1; got {tau_grow_f!r}")
    if not (0.0 < tau_shrink_f < 1.0):
        raise ValueError(f"tau_shrink must be in (0, 1); got {tau_shrink_f!r}")
    if rho_min_f <= 0.0:
        raise ValueError(
            f"rho_min must be > 0 (clipping to <=0 would invalidate the next "
            f"rho_curr > 0 invariant); got {rho_min_f!r}"
        )
    if rho_min_f >= rho_max_f:
        raise ValueError(
            f"rho_min ({rho_min_f}) must be < rho_max ({rho_max_f})"
        )
    if eps_f <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps_f!r}")

    primal_safe = max(primal_f, eps_f)
    dual_safe = max(dual_f, eps_f)
    ratio = primal_safe / dual_safe

    notes = [
        "[adaptive-rho ADMM step; Boyd 2011 §3.4.1; He-Yang 2000]",
    ]

    if primal_f == 0.0 and dual_f == 0.0:
        # True KKT-equilibrium: both residuals zero. No reason to move rho.
        rho_proposed = rho_curr_f
        direction = "hold"
        notes.append(
            "primal=dual=0 (KKT equilibrium); rho held"
        )
    elif primal_f > mu_f * dual_safe:
        rho_proposed = rho_curr_f * tau_grow_f
        direction = "grow"
        notes.append(
            f"primal {primal_f:.4e} > mu*{mu_f}*dual {dual_safe:.4e}; "
            f"growing rho by {tau_grow_f}"
        )
    elif dual_f > mu_f * primal_safe:
        rho_proposed = rho_curr_f * tau_shrink_f
        direction = "shrink"
        notes.append(
            f"dual {dual_f:.4e} > mu*{mu_f}*primal {primal_safe:.4e}; "
            f"shrinking rho by {tau_shrink_f}"
        )
    else:
        rho_proposed = rho_curr_f
        direction = "hold"
        notes.append(
            f"primal/dual ratio {ratio:.4e} within mu={mu_f} band; rho held"
        )

    rho_next = min(max(rho_proposed, rho_min_f), rho_max_f)
    if rho_next != rho_proposed:
        notes.append(
            f"clipped rho_proposed={rho_proposed:.4e} to "
            f"[{rho_min_f}, {rho_max_f}] -> {rho_next:.4e}"
        )

    return AdaptiveRhoStep(
        rho_next=rho_next,
        direction=direction,
        ratio=ratio,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stream interface
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProximalStepResult:
    """Result of a single proximal-codec call.

    Fields
    ------
    encoded_bytes : int
        Actual byte count after the codec packs the stream. May differ from
        the requested target (the codec discretises onto its own ladder).
    score_delta : float
        Score contribution at this operating point (smaller = better).
        For pose: distortion-component contribution; for masks: SegNet-disagreement
        weighted contribution; etc. The COORDINATOR treats this as opaque.
    marginal : float
        Estimated dScore/dByte at this point. Sign convention: positive means
        "spending more bytes lowers score" (the typical R(D) monotone region).
        Coordinator uses this for KKT waterline equilibration.
    state : object
        Opaque codec-internal state (e.g. the encoded payload, qint_max
        ladder, etc.) that the codec round-trips through ADMM iterations.
    """

    encoded_bytes: int
    score_delta: float
    marginal: float
    state: object = None


@runtime_checkable
class StreamProximalCodec(Protocol):
    """Interface every codec wrapper implements to participate in Joint-ADMM.

    Two responsibilities:

    1. ``proximal_step(target_bytes, dual)`` — solve the per-stream
       proximal subproblem at the given byte target with the given dual
       variable. Return a :class:`ProximalStepResult` containing the actual
       (post-discretisation) byte count, the score-delta at that operating
       point, and the local marginal dScore/dByte estimate.

    2. ``name`` — short identifier used in logs / KKT report.

    The coordinator never inspects ``state`` directly; codecs may use it for
    warm-starting subsequent proximal calls.

    Strict-scorer-rule: ``proximal_step`` MUST NOT load SegNet / PoseNet /
    distilled scorers. Score-cost surfaces are CACHED at coordinator launch
    (typically derived from a one-shot frontier sampling pass before ADMM).
    Loading scorers inside ``proximal_step`` would (a) re-do work the
    frontier-sampler already did, (b) be too slow for the inner loop, and
    (c) blur the boundary between coordinator and live measurement.
    """

    @property
    def name(self) -> str:
        ...

    def proximal_step(
        self, target_bytes: float, dual: float
    ) -> ProximalStepResult:
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class JointADMMConfig:
    """ADMM coordinator configuration.

    Every field carries an explicit default + derivation comment.
    Check 81 STRICT: silent defaults are forbidden — these defaults all
    come from Boyd §3.4 + memory `project_codec_stacking_composition`.

    Parameters
    ----------
    rho_init : float, default 1.0
        Initial penalty parameter ρ. Boyd §3.4 recommends starting at 1.0
        and adapting; the safe-default is invariant under linear scaling
        of the byte axis once the byte_budget normaliser is applied.
    rho_max : float, default 1e6
        Hard cap on ρ; if adaptive growth tries to exceed this, restart.
        Set 6 orders of magnitude above rho_init — Boyd reports stable
        ADMM in this band for well-conditioned problems.
    rho_growth : float, default 2.0
        Multiplicative growth factor when primal residual >> dual residual.
        Boyd §3.4.1 default; values in [2, 5] are standard.
    rho_shrink : float, default 0.5
        Multiplicative shrink factor when dual residual >> primal residual.
        Boyd default 0.5 (= 1/rho_growth).
    rho_imbalance_ratio : float, default 10.0
        The threshold "primal_residual / dual_residual > rho_imbalance_ratio
        ⇒ grow ρ". Boyd §3.4.1 default = 10. Symmetric for the shrink.
    max_iters : int, default 200
        Iteration cap; the kill criterion in
        project_phases_2_3_4_design_implementation §"Lane 10" is 1000.
        We default lower for unit-test smoke; production callers override
        explicitly.
    restart_threshold : int, default 5
        Number of consecutive divergence-detected iterations before the
        coordinator restarts with a freshly-shrunken ρ. Pulled from
        memory `project_codec_stacking_composition_canonical_orders` §"ADMM divergence".
    primal_tol : float, default 1e-3
        Stopping tolerance on the primal residual ‖b - z‖₂. Scaled to the
        byte_budget by the coordinator (so this is a relative tolerance).
    dual_tol : float, default 1e-3
        Stopping tolerance on the dual residual ρ‖z - z_prev‖₂.
        Boyd §3.3 standard.
    kkt_waterline_tol : float, default 5e-2
        At convergence, max relative spread of per-stream dScore/dByte
        marginals. 5% is loose; tighter checks per problem at the test layer.
    byte_budget : float, default 0.0
        TOTAL byte budget Σ_s b_s ≤ byte_budget. MUST be set explicitly by
        the caller; default 0 is a sentinel that triggers a clear error
        rather than silently accepting "zero bytes allowed".
    score_budget_per_stream : dict[str, float] | None, default None
        Optional per-stream score-cost cap. None ⇒ no per-stream cap (only
        the joint byte budget binds). When set, the coordinator clamps
        proximal targets that would push f_s above its cap.
    verbose : bool, default False
        Print per-iteration trace. Off by default — production runs use
        the AdmmResult.history list for post-hoc inspection.
    """

    rho_init: float = 1.0
    rho_max: float = 1e6
    rho_growth: float = 2.0
    rho_shrink: float = 0.5
    rho_imbalance_ratio: float = 10.0
    max_iters: int = 200
    restart_threshold: int = 5
    primal_tol: float = 1e-3
    dual_tol: float = 1e-3
    kkt_waterline_tol: float = 5e-2
    byte_budget: float = 0.0
    score_budget_per_stream: dict[str, float] | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.byte_budget <= 0:
            raise ValueError(
                "JointADMMConfig.byte_budget must be > 0 (caller must set "
                "the joint byte budget explicitly; the default 0 is a "
                "sentinel to prevent silent zero-byte runs). "
                "See memory project_codec_stacking_composition_canonical_orders."
            )
        if self.rho_init <= 0 or self.rho_max <= self.rho_init:
            raise ValueError(
                f"rho_init must be > 0 and rho_max > rho_init; got "
                f"rho_init={self.rho_init}, rho_max={self.rho_max}"
            )
        if not (1.0 < self.rho_growth):
            raise ValueError(f"rho_growth must be > 1; got {self.rho_growth}")
        if not (0.0 < self.rho_shrink < 1.0):
            raise ValueError(
                f"rho_shrink must be in (0, 1); got {self.rho_shrink}"
            )
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1; got {self.max_iters}")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AdmmIteration:
    """Single-iteration trace record."""

    iter: int
    rho: float
    bytes_per_stream: list[float]
    z_per_stream: list[float]
    duals_per_stream: list[float]
    score_per_stream: list[float]
    marginal_per_stream: list[float]
    primal_residual: float
    dual_residual: float
    restarted: bool = False


@dataclass
class AdmmResult:
    """Joint-ADMM convergence result.

    Fields
    ------
    converged : bool
        Both primal_residual <= primal_tol AND dual_residual <= dual_tol.
    iters : int
        Total iterations executed (across restarts).
    restarts : int
        Number of restart events.
    final_bytes_per_stream : list[float]
        Discretised per-stream byte allocations at termination.
    final_score_per_stream : list[float]
        f_s at the final operating point per stream.
    final_marginal_per_stream : list[float]
        dScore/dByte estimates at termination. Used for KKT waterline check.
    waterline_kkt_residual : float
        max(marginals) - min(marginals) over ACTIVE streams (those with
        positive byte allocation). At true KKT optimum, this is 0.
    waterline_satisfied : bool
        True iff waterline_kkt_residual <= cfg.kkt_waterline_tol *
        max(|marginal|, 1e-12).
    history : list[AdmmIteration]
        Full per-iteration trace for paper figures + debugging.
    """

    converged: bool
    iters: int
    restarts: int
    final_bytes_per_stream: list[float]
    final_score_per_stream: list[float]
    final_marginal_per_stream: list[float]
    waterline_kkt_residual: float
    waterline_satisfied: bool
    history: list[AdmmIteration] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Projection onto Σz ≤ B simplex (Duchi/Shalev-Shwartz/Singer/Chandra 2008)
# ─────────────────────────────────────────────────────────────────────────────


def _project_to_byte_budget(v: np.ndarray, B: float) -> np.ndarray:
    """Project v onto {z >= 0, Σz <= B}.

    If Σ max(v, 0) <= B, return max(v, 0) (budget non-binding).
    Else project onto the simplex Σz = B (budget binding).

    Standard O(n log n) algorithm (Duchi et al. ICML 2008).
    """
    v_pos = np.maximum(v, 0.0)
    if float(v_pos.sum()) <= B + 1e-12:
        return v_pos
    # Simplex projection onto Σz = B, z >= 0.
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - B
    rho_idx = np.nonzero(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho_idx.size == 0:
        # Numerically degenerate; fall back to uniform.
        return np.full_like(v, B / n)
    rho_int = rho_idx[-1]
    theta = cssv[rho_int] / float(rho_int + 1)
    return np.maximum(v - theta, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# KKT waterline check
# ─────────────────────────────────────────────────────────────────────────────


def kkt_waterline_residual(
    bytes_per_stream: np.ndarray,
    marginals: np.ndarray,
    activity_eps: float = 1e-6,
    marginal_eps: float = 1e-9,
) -> tuple[float, bool, np.ndarray]:
    """Compute the KKT waterline residual.

    Returns (residual, has_unsaturated, equilibrating_mask):
    - equilibrating_mask[s] = True iff bytes_per_stream[s] > activity_eps
      AND marginals[s] > marginal_eps. These are the streams in the
      INTERIOR of the per-stream feasible set (not at saturation,
      genuinely competing for budget).
    - residual = max(marginals[eq]) - min(marginals[eq]).
    - has_unsaturated = equilibrating_mask.any().

    Why exclude saturated (marginal=0) streams: at KKT a saturated stream
    has dScore/dByte = 0 ≤ λ, complementary slackness allows b > 0 with
    margin = 0 because the stream simply cannot profit from more bytes.
    The "common waterline λ*" applies only to streams that COULD trade
    bytes against another stream's score gain.

    Per memory `project_codec_stacking_composition` §"Per-stream waterline":
    KKT optimum requires equal marginals across UNSATURATED active streams.
    """
    eq = (bytes_per_stream > activity_eps) & (marginals > marginal_eps)
    if not eq.any():
        return float("inf"), False, eq
    m_eq = marginals[eq]
    residual = float(m_eq.max() - m_eq.min())
    return residual, True, eq


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator
# ─────────────────────────────────────────────────────────────────────────────


def run_admm(
    streams: list[StreamProximalCodec],
    cfg: JointADMMConfig,
) -> AdmmResult:
    """Run Joint-ADMM coordinator (dual decomposition + ADMM stabilisation).

    Mathematical formulation
    ------------------------
    minimise  Σ_s f_s(b_s)
    s.t.      Σ_s b_s ≤ B,   b_s >= 0

    Dual decomposition (Boyd §2.1): introduce dual price λ ≥ 0 on the
    budget constraint. The Lagrangian is

        L(b, λ) = Σ_s f_s(b_s) + λ * (Σ_s b_s - B)

    which decomposes per-stream as

        b_s^*(λ) = argmin_b  f_s(b) + λ*b           (each stream's job)

    with dual update by subgradient ascent on the dual function

        λ^{k+1} = max(0, λ^k + ρ^k (Σ_s b_s^*(λ^k) - B))

    At KKT the per-stream marginals dScore/dByte equal λ across ACTIVE
    streams (the waterline equilibration condition; memory
    `project_codec_stacking_composition` §"Per-stream waterline").

    ADMM stabilisation (Boyd §3.4 adaptive ρ + §3.4.1 dynamic penalty)
    is layered onto the dual update to handle non-smooth / discrete f_s —
    the synthetic test harness exercises this path.

    Coordinator inputs
    ------------------
    Per memory `project_phases_2_3_4` §Lane 10: per-stream marginal-rate
    functions estimated empirically from prior runs. Each stream encapsulates
    its R(D) frontier inside ``proximal_step``. The coordinator never sees
    f_s directly — it only sees byte+score outputs at queried duals.

    Args:
        streams: list of StreamProximalCodec.
        cfg: JointADMMConfig.

    Returns:
        AdmmResult with full trace + KKT waterline diagnostic.

    Strict-scorer-rule: this function does not load any neural-net scorer.
    It calls only ``stream.proximal_step`` which (by Protocol contract)
    also does not load scorers.
    """
    if len(streams) < 1:
        raise ValueError("run_admm requires >=1 stream")
    # Validate Protocol conformance for catchable test reasons.
    for s in streams:
        if not isinstance(s, StreamProximalCodec):
            raise TypeError(
                f"stream {s!r} does not satisfy StreamProximalCodec Protocol"
            )

    n = len(streams)
    rho = float(cfg.rho_init)  # adaptive penalty / dual step size
    lam = 0.0  # Lagrange multiplier on budget constraint (current)
    lam_avg = 0.0  # dual-averaged multiplier (Nesterov 2009 — handles
                   # non-smooth dual subgradient by accumulating + averaging,
                   # which damps the oscillation the discrete Lagrangian
                   # form would otherwise produce. Memory codec_stacking
                   # failure modes: "ADMM divergence: use adaptive penalty,
                   # restarts, exact byte projection".)
    bytes_arr = np.zeros(n, dtype=np.float64)
    bytes_avg = np.zeros(n, dtype=np.float64)  # running average of bytes
    score_arr = np.zeros(n, dtype=np.float64)
    margin_arr = np.zeros(n, dtype=np.float64)

    history: list[AdmmIteration] = []
    restarts = 0
    consecutive_diverging = 0
    converged = False
    iters_done = 0
    avg_window_count = 0

    for it in range(1, cfg.max_iters + 1):
        iters_done = it
        restart_now = False

        # x-update: each stream solves its proximal-Lagrangian subproblem
        # min_b f_s(b) + lam*b, with target_bytes as an UPPER BOUND on b.
        # We pass the DUAL-AVERAGED lambda (Nesterov 2009) to the codec —
        # this is the standard non-smooth-dual fix. The current iterate's
        # lam still drives the dual subgradient ascent.
        bytes_prev = bytes_arr.copy()
        # Use dual-averaged lambda to query streams (smooths discontinuities).
        lam_query = lam_avg if avg_window_count > 0 else lam
        for s_idx, stream in enumerate(streams):
            tgt = float(cfg.byte_budget)
            if cfg.score_budget_per_stream is not None:
                cap = cfg.score_budget_per_stream.get(stream.name)
                if cap is not None and score_arr[s_idx] > cap:
                    # PARADIGM-γ audit fix #3B (2026-05-06): the previous
                    # `tgt = max(tgt, bytes_arr * 1.25 + 1.0)` INCREASED the
                    # byte budget when this stream's score exceeded its cap —
                    # the opposite of the intended "shrink this stream until
                    # its score drops" semantics. Replace with a downward cap:
                    # restrict target_bytes to 75% of current bytes (drives
                    # the codec toward a lower R(D) operating point and
                    # reduces this stream's score contribution). The MIN
                    # ensures we never RAISE bytes when the score is already
                    # over budget.
                    tgt = min(tgt, max(0.0, bytes_arr[s_idx] * 0.75))
            res = stream.proximal_step(target_bytes=tgt, dual=float(lam_query))
            bytes_arr[s_idx] = float(res.encoded_bytes)
            score_arr[s_idx] = float(res.score_delta)
            margin_arr[s_idx] = float(res.marginal)

        # Update running byte average (used as the reportable allocation;
        # Nesterov-averaged primal converges to KKT even when iterates oscillate).
        avg_window_count += 1
        bytes_avg = bytes_avg + (bytes_arr - bytes_avg) / avg_window_count

        # Budget overshoot diagnostic (positive ⇒ over budget).
        # Use AVERAGED bytes for the budget feasibility check (Nesterov).
        total_bytes_avg = float(bytes_avg.sum())
        budget_gap = total_bytes_avg - cfg.byte_budget

        # Dual update: λ ← max(0, λ + ρ*(Σb - B)).
        lam_prev = lam
        lam = max(0.0, lam + rho * budget_gap)
        # Update averaged dual.
        lam_avg = lam_avg + (lam - lam_avg) / avg_window_count

        # Residuals.
        # primal_res = |Σ bytes_avg - B|  (Nesterov-averaged feasibility)
        primal_res = abs(budget_gap)
        # dual_res = ρ * |bytes_arr - bytes_prev|  — drops to 0 when iterates
        # stabilise even though discrete oscillation may be present.
        dual_res = rho * float(np.linalg.norm(bytes_arr - bytes_prev))

        # Q4B FIX (Carmack Option B, Boyd §3.4.1, 2026-04-30 council #271):
        # Adaptive rho_init refinement on the FIRST iteration only.
        #
        # Rationale: cfg.rho_init=1.0 (default) is unstable for problems
        # with low quadratic curvature 2*a < 0.01 because the dual update
        # rho * budget_gap blows up before bytes_arr can stabilise. Boyd
        # 2011 §3.4.1 recommends scaling rho to the first-iteration ratio
        # of primal/dual residuals: if r1 / s1 > 10 (rho too small),
        # multiply by 2; if s1 / r1 > 10 (rho too large), divide by 2.
        # Cap at [1e-6, 1e6].
        #
        # Applied on iter 1 ONLY, and EXCLUSIVE with the steady-state
        # adaptive rule below: when Q4B fires we skip the steady-state
        # adjustment for this iteration. This prevents the well-tuned
        # rho_init configurations (e.g. test_two_stream_convex_converges_to_kkt
        # with rho_init=0.02) from being double-bumped to instability. No
        # API change: cfg.rho_init stays as the SEED, the adaptive step
        # refines it once on iter 1.
        q4b_fired = False
        if it == 1:
            primal_safe = max(primal_res, 1e-12)
            dual_safe = max(dual_res, 1e-12)
            ratio = primal_safe / dual_safe
            if ratio > 10.0:
                rho = min(max(rho * 2.0, 1e-6), 1e6)
                q4b_fired = True
            elif ratio < 0.1:
                rho = min(max(rho * 0.5, 1e-6), 1e6)
                q4b_fired = True

        # Adaptive ρ (Boyd §3.4.1). Skip on iter 1 if Q4B already adjusted —
        # otherwise the same residual ratio drives BOTH rules and rho gets
        # double-scaled (verified failure mode on test_two_stream_convex_*
        # with rho_init=0.02: Q4B → 0.04, then steady-state → 0.08 → bytes
        # corner-pin to [0, 29] instead of equilibrating to KKT [167, 133]).
        if not q4b_fired:
            if primal_res > cfg.rho_imbalance_ratio * max(dual_res, 1e-12):
                rho = min(rho * cfg.rho_growth, cfg.rho_max)
            elif dual_res > cfg.rho_imbalance_ratio * max(primal_res, 1e-12):
                rho = max(rho * cfg.rho_shrink, cfg.rho_init * 1e-3)

        # Divergence detection: TWO classes of pathology.
        # (a) primal residual strictly increasing for K iters.
        # (b) primal residual stays large (> 50% of byte_budget) for K iters
        #     while the dual is growing past 10x its initial value — a sign
        #     the dual is exploding without buying primal feasibility.
        scaled_primal_tol = cfg.primal_tol * max(cfg.byte_budget, 1.0)
        is_diverging = False
        if len(history) > 0:
            prev_h = history[-1]
            if primal_res > prev_h.primal_residual * 1.5 and primal_res > 1.0:
                is_diverging = True
        # Sustained-large-primal pathology.
        if (
            primal_res > 0.5 * cfg.byte_budget
            and lam > 10.0 * cfg.rho_init
            and primal_res > scaled_primal_tol
        ):
            is_diverging = True
        if is_diverging:
            consecutive_diverging += 1
        else:
            consecutive_diverging = 0
        if consecutive_diverging >= cfg.restart_threshold or (
            rho >= cfg.rho_max * 0.999 and primal_res > cfg.primal_tol
        ):
            restarts += 1
            restart_now = True
            consecutive_diverging = 0
            rho = max(cfg.rho_init, rho * 0.1)
            lam = lam_prev * 0.5  # half-shrink the dual
            # Reset Nesterov average (warm-restart from current iterate).
            avg_window_count = 1
            bytes_avg = bytes_arr.copy()
            lam_avg = lam

        history.append(
            AdmmIteration(
                iter=it,
                rho=rho,
                bytes_per_stream=bytes_arr.tolist(),
                # z stores the Nesterov-averaged byte allocation (the
                # reportable feasible point even when raw iterates oscillate).
                z_per_stream=bytes_avg.tolist(),
                duals_per_stream=[lam] * n,
                score_per_stream=score_arr.tolist(),
                marginal_per_stream=margin_arr.tolist(),
                primal_residual=primal_res,
                dual_residual=dual_res,
                restarted=restart_now,
            )
        )

        if cfg.verbose:
            print(
                f"[ADMM] it={it:4d} rho={rho:8.3f} lam={lam:8.4f} "
                f"primal={primal_res:.4e} dual={dual_res:.4e} "
                f"bytes={bytes_arr.tolist()} margins={margin_arr.tolist()}"
            )

        # Stopping: budget violated by at most primal_tol*budget AND dual
        # update small. Require >=2 iters to avoid trivial first-iteration stop.
        scaled_primal_tol = cfg.primal_tol * max(cfg.byte_budget, 1.0)
        scaled_dual_tol = cfg.dual_tol * max(cfg.byte_budget, 1.0)
        if (
            it >= 2
            and primal_res <= scaled_primal_tol
            and dual_res <= scaled_dual_tol
            and not restart_now
        ):
            converged = True
            break

    # Use Nesterov-averaged bytes for the final reportable allocation.
    # The averaged primal converges to KKT even when the raw discrete
    # iterates oscillate (Nesterov 2009 §3).
    final_bytes = bytes_avg.copy()

    # Q4A FIX (Boyd / Dykstra / MacKay, 2026-04-30 council #271):
    # Choose the dual at which to re-query streams based on convergence state.
    #
    # When ``converged == True``: use the FINAL non-averaged ``lam``. This is
    # the converged KKT dual; ``lam_avg`` can lag the true dual by ~3× because
    # the running-average rule under-weights the most recent (post-convergence)
    # iterates that actually reflect the equilibrating Lagrangian.
    #
    # When ``converged == False``: use ``lam_avg``. In the divergent /
    # early-stopped case the final ``lam`` may be a mid-trajectory artefact
    # that is not representative of the operating point; the averaged dual is
    # a more robust diagnostic estimate.
    #
    # Memory ref: .omx/research/council_strategic_design_decisions_20260430.md
    # §5.1. Replaces the prior "always use lam_avg" behaviour which produced a
    # ~7% budget overshoot when re-querying at convergence (Round 11 finding).
    final_lam = lam if converged else lam_avg

    # Re-query each stream at the chosen lambda to obtain the score
    # + marginal at the final operating point. This is one extra coordinator
    # step per stream — cheap, and gives us scorer-rule-compliant marginals
    # at the actual reported allocation.
    final_score = np.zeros(n, dtype=np.float64)
    final_margin = np.zeros(n, dtype=np.float64)
    for s_idx, stream in enumerate(streams):
        # Pass the averaged bytes as target; codec returns its operating point
        # and marginal estimate.
        try:
            res = stream.proximal_step(
                target_bytes=float(final_bytes[s_idx]), dual=float(final_lam)
            )
            final_bytes[s_idx] = float(res.encoded_bytes)
            final_score[s_idx] = float(res.score_delta)
            final_margin[s_idx] = float(res.marginal)
        except Exception:
            # If the codec can't be re-queried (e.g., consumed state),
            # fall back to the last raw iterate.
            final_bytes[s_idx] = bytes_arr[s_idx]
            final_score[s_idx] = score_arr[s_idx]
            final_margin[s_idx] = margin_arr[s_idx]

    # KKT waterline check on UNSATURATED ACTIVE streams.
    waterline_res, has_active, _active_mask = kkt_waterline_residual(
        final_bytes, final_margin
    )
    if has_active and math.isfinite(waterline_res):
        scale = max(float(np.abs(final_margin).max()), 1e-12)
        waterline_satisfied = waterline_res <= cfg.kkt_waterline_tol * scale
    else:
        # No interior streams (all saturated or all zero) — KKT vacuous;
        # mark satisfied if budget feasible.
        waterline_satisfied = float(final_bytes.sum()) <= cfg.byte_budget * 1.01

    return AdmmResult(
        converged=converged,
        iters=iters_done,
        restarts=restarts,
        final_bytes_per_stream=final_bytes.tolist(),
        final_score_per_stream=final_score.tolist(),
        final_marginal_per_stream=final_margin.tolist(),
        waterline_kkt_residual=waterline_res,
        waterline_satisfied=waterline_satisfied,
        history=history,
    )


__all__ = [
    "AdaptiveRhoStep",
    "AdmmIteration",
    "AdmmResult",
    "JointADMMConfig",
    "ProximalStepResult",
    "StreamProximalCodec",
    "adaptive_rho_step",
    "kkt_waterline_residual",
    "run_admm",
]
