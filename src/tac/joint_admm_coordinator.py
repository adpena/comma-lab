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
) -> tuple[float, bool, np.ndarray]:
    """Compute the KKT waterline residual.

    Returns (residual, has_active, active_mask):
    - active_mask[s] = True if bytes_per_stream[s] > activity_eps (interior).
    - residual = max(marginals[active]) - min(marginals[active]).
    - has_active = active_mask.any().

    Per memory `project_codec_stacking_composition` §"Per-stream waterline":
    KKT optimum requires equal marginals across active streams.
    """
    active = bytes_per_stream > activity_eps
    if not active.any():
        return float("inf"), False, active
    m_active = marginals[active]
    residual = float(m_active.max() - m_active.min())
    return residual, True, active


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator
# ─────────────────────────────────────────────────────────────────────────────


def run_admm(
    streams: list[StreamProximalCodec],
    cfg: JointADMMConfig,
) -> AdmmResult:
    """Run Joint-ADMM coordinator.

    Boyd 2011 §3.4 alternating-direction-method-of-multipliers with
    adaptive ρ (§3.4.1) + restart on divergence.

    Args:
        streams: list of StreamProximalCodec — one per archive byte stream
            (e.g. renderer.bin / masks.mkv / poses.pt / class_targets / ...).
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
    rho = float(cfg.rho_init)
    # Equal initial allocation across streams as warm-start.
    z = np.full(n, cfg.byte_budget / n, dtype=np.float64)
    u = np.zeros(n, dtype=np.float64)  # scaled dual variables
    bytes_arr = np.zeros(n, dtype=np.float64)
    score_arr = np.zeros(n, dtype=np.float64)
    margin_arr = np.zeros(n, dtype=np.float64)

    history: list[AdmmIteration] = []
    restarts = 0
    consecutive_diverging = 0
    converged = False
    iters_done = 0
    z_prev = z.copy()

    for it in range(1, cfg.max_iters + 1):
        iters_done = it
        restart_now = False

        # x-update: each stream solves its proximal subproblem
        # at target = z_s - u_s/rho (the standard scaled-dual ADMM target).
        targets = np.maximum(z - u / max(rho, 1e-12), 0.0)
        for s_idx, stream in enumerate(streams):
            tgt = float(targets[s_idx])
            # Optional per-stream score cap clamps target.
            if cfg.score_budget_per_stream is not None:
                cap = cfg.score_budget_per_stream.get(stream.name)
                if cap is not None and score_arr[s_idx] > cap:
                    # Inflate target to push score down (heuristic — proximal
                    # codec ultimately decides the operating point).
                    tgt = max(tgt, bytes_arr[s_idx] * 1.25)
            res = stream.proximal_step(target_bytes=tgt, dual=float(u[s_idx]))
            # Exact byte projection: codec returned actual encoded_bytes
            # (which may differ from tgt due to discretisation). We treat
            # encoded_bytes as ground truth for the byte stream — that is
            # the "exact byte projection after every codec call" non-negotiable
            # in CLAUDE.md FORBIDDEN section + memory codec_stacking failure modes.
            bytes_arr[s_idx] = float(res.encoded_bytes)
            score_arr[s_idx] = float(res.score_delta)
            margin_arr[s_idx] = float(res.marginal)

        # z-update: project (bytes_arr + u/rho) onto budget simplex.
        z_prev = z.copy()
        z_target = bytes_arr + u / max(rho, 1e-12)
        z = _project_to_byte_budget(z_target, cfg.byte_budget)

        # u-update: u <- u + rho (bytes_arr - z)
        u = u + rho * (bytes_arr - z)

        # Residuals (Boyd §3.3)
        primal_res = float(np.linalg.norm(bytes_arr - z))
        dual_res = float(rho * np.linalg.norm(z - z_prev))

        # Adaptive ρ (Boyd §3.4.1)
        if primal_res > cfg.rho_imbalance_ratio * dual_res:
            new_rho = min(rho * cfg.rho_growth, cfg.rho_max)
            if new_rho > rho:
                # u <- u / growth_factor when ρ grows (so the scaled dual
                # u/ρ is preserved).
                u = u * (rho / new_rho)
                rho = new_rho
        elif dual_res > cfg.rho_imbalance_ratio * primal_res:
            new_rho = max(rho * cfg.rho_shrink, cfg.rho_init * 1e-3)
            if new_rho < rho:
                u = u * (rho / new_rho)
                rho = new_rho

        # Divergence detection: residuals strictly increasing for K iters.
        if len(history) > 0:
            prev = history[-1]
            if (
                primal_res > prev.primal_residual * 1.5
                and dual_res > prev.dual_residual * 1.5
            ):
                consecutive_diverging += 1
            else:
                consecutive_diverging = 0
        # Restart on sustained divergence OR when ρ would saturate at max.
        if consecutive_diverging >= cfg.restart_threshold or (
            rho >= cfg.rho_max * 0.999 and primal_res > cfg.primal_tol
        ):
            restarts += 1
            restart_now = True
            consecutive_diverging = 0
            # Restart: shrink ρ aggressively + re-warm-start z to current
            # bytes. Boyd §3.4 + memory codec_stacking failure modes.
            rho = max(cfg.rho_init, rho * 0.1)
            z = np.maximum(bytes_arr, 0.0)
            z_total = z.sum()
            if z_total > cfg.byte_budget:
                z = z * (cfg.byte_budget / max(z_total, 1e-12))
            u = np.zeros(n, dtype=np.float64)

        history.append(
            AdmmIteration(
                iter=it,
                rho=rho,
                bytes_per_stream=bytes_arr.tolist(),
                z_per_stream=z.tolist(),
                duals_per_stream=u.tolist(),
                score_per_stream=score_arr.tolist(),
                marginal_per_stream=margin_arr.tolist(),
                primal_residual=primal_res,
                dual_residual=dual_res,
                restarted=restart_now,
            )
        )

        if cfg.verbose:
            print(
                f"[ADMM] it={it:4d} rho={rho:8.3f} primal={primal_res:.4e} "
                f"dual={dual_res:.4e} bytes={bytes_arr.tolist()} "
                f"margins={margin_arr.tolist()}"
            )

        # Stopping (Boyd §3.3): both residuals below tolerance AND no recent
        # restart (we want one clean iteration after restart).
        scaled_primal_tol = cfg.primal_tol * max(cfg.byte_budget, 1.0)
        scaled_dual_tol = cfg.dual_tol * max(cfg.byte_budget, 1.0)
        if (
            primal_res <= scaled_primal_tol
            and dual_res <= scaled_dual_tol
            and not restart_now
        ):
            converged = True
            break

    # KKT waterline check on ACTIVE streams.
    waterline_res, has_active, _active_mask = kkt_waterline_residual(
        bytes_arr, margin_arr
    )
    if has_active and math.isfinite(waterline_res):
        scale = max(float(np.abs(margin_arr).max()), 1e-12)
        waterline_satisfied = waterline_res <= cfg.kkt_waterline_tol * scale
    else:
        waterline_satisfied = False

    return AdmmResult(
        converged=converged,
        iters=iters_done,
        restarts=restarts,
        final_bytes_per_stream=bytes_arr.tolist(),
        final_score_per_stream=score_arr.tolist(),
        final_marginal_per_stream=margin_arr.tolist(),
        waterline_kkt_residual=waterline_res,
        waterline_satisfied=waterline_satisfied,
        history=history,
    )


__all__ = [
    "AdmmIteration",
    "AdmmResult",
    "JointADMMConfig",
    "ProximalStepResult",
    "StreamProximalCodec",
    "kkt_waterline_residual",
    "run_admm",
]
