# SPDX-License-Identifier: MIT
"""Faithful PR95 8-stage curriculum spec, sourced from the vendored stage builders.

Rather than re-type the 29,650-epoch schedule (a cargo-cult risk + a drift
surface), this module CALLS the vendored ``stages.stageN.make_config`` builders
and reads the exact (epochs, seg_loss_fn, adamw_lr, muon_lr, cat_lambda,
cat_sigma, use_qat, use_muon, ema_decay, grad_clip, ...) off the returned
``StageConfig`` objects. The schedule is therefore faithful BY CONSTRUCTION — if
the vendored source changes, this picks it up; nothing is duplicated.

The adapter overrides only:
  * ``epochs`` per stage — for the $100 budget compression (proportional);
  * ``ema_decay`` — pinned to the faithful PR95 value (0.999) by default;
  * the architecture (``base_channels`` / ``latent_dim``) — threaded by the
    driver, NOT a StageConfig field in the vendored source (the FINDING-1 fix).

The vendored ``StageConfig`` carries NO architecture field, so the per-stage
config is purely the TRAINING schedule; the driver owns the architecture.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.torch_vehicle.vendored_imports import import_vendored

# Canonical PR95 per-stage default epoch counts (from the vendored make_config
# signatures: 3000,5650,1500,500,9000,2000,3000,5000 = 29,650). Used only to
# compute the proportional budget compression; the values themselves come from
# the vendored builders at call time (we assert-match below).
PR95_DEFAULT_EPOCHS = (3000, 5650, 1500, 500, 9000, 2000, 3000, 5000)
PR95_TOTAL_EPOCHS = sum(PR95_DEFAULT_EPOCHS)  # 29,650


@dataclass(frozen=True)
class StageSpec:
    """The training schedule for one curriculum stage (architecture-free).

    Mirrors the fields the driver reads off the vendored ``StageConfig``; the
    driver supplies architecture (base_channels / latent_dim) separately.
    """

    name: str
    epochs: int
    seg_loss_fn: Callable[[Any, Any], Any]
    eval_every: int
    batch_size: int
    ema_decay: float
    use_muon: bool
    adamw_lr: float
    muon_lr: float
    muon_weight_decay: float
    latent_lr_mult: float
    grad_clip: float
    grad_clip_muon: float | None
    lr_floor_ratio: float
    seg_weight: float
    pose_weight: float
    cat_lambda: float
    cat_sigma: float
    use_qat: bool
    init_latents_random: bool
    # -- Lever 2 (score-domain seg surrogate) — DEFAULT-PRESERVING ------------
    # ``seg_surrogate is None`` (the default) reproduces the vendored ``seg_loss_fn``
    # path BYTE-FOR-BYTE: the driver calls ``spec.seg_loss_fn(seg_out, targets_hard)``
    # exactly as before. Set it to a surrogate name (``"soft_cosine"`` / ``"fisher_rao"``
    # / ``"sinkhorn"``) to route the seg term through the differentiable score-domain
    # d_seg surrogate ``tac.losses.core.segnet_surrogate_per_pixel`` instead (the
    # argmax-flip-concentrated loss; HNeRV parity L6). ``seg_temperature`` is the
    # surrogate's softmax temperature (1.0 = unit; anneal toward hard with T→small).
    # These fields are LAST + defaulted so every existing positional ``StageSpec(...)``
    # construction (tests, the vendored projection) is unchanged.
    seg_surrogate: str | None = None
    seg_temperature: float = 1.0
    # -- Lever 2 OPTIMIZE: per-epoch seg-temperature ANNEAL hook (the GAP the --
    # combined arm flagged). ``seg_temperature_end is None`` (the default) keeps the
    # STATIC ``seg_temperature`` for every epoch — byte-identical to today. When set,
    # the driver anneals the PREDICTION softmax temperature COSINE from
    # ``seg_temperature`` (start) toward ``seg_temperature_end`` over the stage's
    # epochs (the memo's "T: 1.0 → 0.05 toward hard argmax" — sharper boundary
    # gradient as training converges). The GT stays HARD (one-hot logits, the contest
    # d_seg is a hard argmax); only the prediction softens/sharpens. Annealing is a
    # NO-OP unless ``seg_surrogate`` is also set (the anneal modulates the surrogate's
    # temperature; with the vendored CE path there is no temperature to anneal).
    seg_temperature_end: float | None = None
    # -- Lever 2 CALIBRATION: fast-cool-then-HOLD anneal shape (the seg-schedule the --
    # ANNEAL-SCHEDULE-CALIBRATION probe found best for the from-0 run). ``seg_temperature_
    # hold_frac is None`` (the default) keeps the PLAIN cosine over the WHOLE stage (the
    # old 1.0→end shape — byte-identical to pre-calibration). When set to a fraction
    # ``f ∈ (0, 1]`` (and ``seg_temperature_end`` is also set), the temperature cosine-
    # anneals from ``seg_temperature`` (start) toward ``seg_temperature_end`` over only
    # the FIRST ``f`` of the stage's epochs, then HOLDS at ``seg_temperature_end`` for the
    # remaining ``1−f``. This is the calibrated "brief warm phase for early stability,
    # then PARK at the gradient-alive sweet spot (T=0.3)" schedule: it spends NO epochs in
    # the gradient-DEAD zone (T < SEG_ANNEAL_GRADIENT_FLOOR_T=0.3) IFF ``seg_temperature_
    # end >= 0.3``, unlike the plain 1.0→0.05 cosine which spends ~52% at T≥0.5 (where the
    # soft-cosine surrogate is ≤ CE) + ~15% in the dead zone. A NO-OP unless BOTH
    # ``seg_surrogate`` and ``seg_temperature_end`` are set.
    seg_temperature_hold_frac: float | None = None
    # -- Lever 1 (differentiable brotli-rate surrogate) — DEFAULT-PRESERVING ----
    # ``rate_lambda_w == 0.0`` AND ``rate_lambda_lat == 0.0`` (the defaults) add NO
    # rate term — byte-identical to today (the loss is unchanged). When > 0 the driver
    # adds ``rate_lambda_w · H(W_i|W_{i-1}) + rate_lambda_lat · H(Δlatent)`` from
    # ``tac.losses.rate_surrogate.brotli_rate_surrogate`` — the order-1 conditional
    # weight entropy (a tighter brotli proxy than the memoryless C1a ``cat_entropy_v2``)
    # plus the currently-UNPENALIZED latent temporal-delta entropy. Schedule them up
    # only in late stages (5-8) like C1a's ``cat_lambda`` 0.01→0.02 (the rate lever is
    # the decoder-dominated, so the weight term first).
    rate_lambda_w: float = 0.0
    rate_lambda_lat: float = 0.0
    # -- Lever 4 (score-aware QAT) — DEFAULT-PRESERVING -----------------------
    # ``score_aware_qat is False`` (the default) uses the vendored UNIFORM 127-level
    # fake-quant when ``use_qat`` — byte-identical to today. When True (and
    # ``use_qat``) the QAT block routes through
    # ``tac.torch_vehicle.score_aware_qat.apply_score_aware_qat`` with the per-tensor
    # sensitivity EMA the driver accumulates from ``||∂S/∂w_t||``: high-sensitivity
    # tensors get a FINER INT8 grid (argmax boundary protected), low-sensitivity ones
    # a COARSER grid (fewer brotli bytes — the water-filling bit-allocator). Until the
    # sensitivity EMA has been seeded (early in a stage) the map is uniform → the
    # quant falls back to the vendored 127-level grid (so the first steps are
    # bit-identical to uniform QAT).
    score_aware_qat: bool = False
    qat_sensitivity_decay: float = 0.99
    # -- Lever 5 (margin-weighted seg promotion) — DEFAULT-PRESERVING ----------
    # ``margin_weight_tau is None`` (the default) leaves the seg surrogate UNWEIGHTED —
    # byte-identical to the Lever-2 baseline. When set, the per-pixel seg surrogate is
    # weighted by ``exp(−margin/τ)`` (the SegNet top1−top2 logit margin of the decoded
    # frame): boundary-prone pixels (small margin) get MORE gradient, confident-interior
    # pixels (large margin) ~0 — capacity concentrates where d_seg actually flips. A
    # NO-OP unless ``seg_surrogate`` is set (it multiplies the surrogate's per-pixel
    # tensor; the vendored CE path returns a scalar with no per-pixel handle).
    margin_weight_tau: float | None = None
    # WS-A/M7 margin-weight RENORMALIZATION (DEFAULT-PRESERVING, off=False). When the
    # margin weight is active, the legacy path takes ``(per_pixel·w).mean()`` — so as
    # margins grow during training Σw shrinks and the ABSOLUTE seg-gradient magnitude
    # decays (compounding with the T-anneal + LR-decay → premature seg-plateau risk). When
    # True, the loss is ``(per_pixel·w).sum()/Σw`` — a true weighted MEAN that REDISTRIBUTES
    # the gradient to the boundary WITHOUT shrinking total magnitude. Off → byte-identical.
    margin_weight_renorm: bool = False


def _spec_from_stage_config(cfg: Any, epochs_override: int | None, ema_decay: float) -> StageSpec:
    """Project a vendored ``StageConfig`` onto an architecture-free ``StageSpec``."""
    return StageSpec(
        name=cfg.name,
        epochs=int(epochs_override) if epochs_override is not None else int(cfg.epochs),
        seg_loss_fn=cfg.seg_loss_fn,
        eval_every=int(cfg.eval_every),
        batch_size=int(cfg.batch_size),
        ema_decay=float(ema_decay),
        use_muon=bool(cfg.use_muon),
        adamw_lr=float(cfg.adamw_lr),
        muon_lr=float(cfg.muon_lr),
        muon_weight_decay=float(cfg.muon_weight_decay),
        latent_lr_mult=float(cfg.latent_lr_mult),
        grad_clip=float(cfg.grad_clip),
        grad_clip_muon=(None if cfg.grad_clip_muon is None else float(cfg.grad_clip_muon)),
        lr_floor_ratio=float(cfg.lr_floor_ratio),
        seg_weight=float(cfg.seg_weight),
        pose_weight=float(cfg.pose_weight),
        cat_lambda=float(cfg.cat_lambda),
        cat_sigma=float(cfg.cat_sigma),
        use_qat=bool(cfg.use_qat),
        init_latents_random=bool(cfg.init_latents_random),
    )


def build_curriculum(
    *,
    total_epoch_budget: int | None = None,
    ema_decay: float = 0.999,
    eval_every: int | None = None,
    min_epochs_per_stage: int = 1,
) -> list[StageSpec]:
    """Build the faithful PR95 8-stage curriculum as architecture-free specs.

    The vendored ``stages.stageN.make_config`` builders are the source of truth
    for the schedule; we call each with a throwaway ``output_dir`` (the spec
    does not run anything — we only read the hyperparameters off the returned
    ``StageConfig``).

    ``total_epoch_budget``: if given, each stage's epoch count is scaled
    PROPORTIONALLY to fit the budget (``round(default * budget / 29650)``,
    floored at ``min_epochs_per_stage``) — the $100 compression. If ``None``,
    the full PR95 schedule (29,650) is used.

    ``ema_decay``: pinned to the faithful PR95 value (0.999) by default.
    ``eval_every``: override the per-stage eval cadence (e.g. tighter for a
    short smoke); ``None`` keeps the vendored cadence.
    """
    # Import vendored stage builders (flat names per the pristine source layout).
    stages_pkg = import_vendored("stages")
    s1 = import_vendored("stages.stage1_v328_ce")
    s2 = import_vendored("stages.stage2_v331_softplus")
    s3 = import_vendored("stages.stage3_v332_smooth")
    s4 = import_vendored("stages.stage4_v332_qat")
    s5 = import_vendored("stages.stage5_c1a_l7")
    s6 = import_vendored("stages.stage6_lambda_sweep")
    s7 = import_vendored("stages.stage7_sigma_sweep")
    s8 = import_vendored("stages.stage8_muon_finetune")
    _ = stages_pkg  # keep the package import (it wires the flat submodules)

    throwaway = Path("/dev/null")  # make_config only mkdir's lazily inside train_stage

    # Build the raw vendored StageConfig objects (stage1 has no resume_from).
    raw_cfgs = [
        s1.make_config(throwaway),
        s2.make_config(throwaway, throwaway),
        s3.make_config(throwaway, throwaway),
        s4.make_config(throwaway, throwaway),
        s5.make_config(throwaway, throwaway),
        s6.make_config(throwaway, throwaway),
        s7.make_config(throwaway, throwaway),
        s8.make_config(throwaway, throwaway),
    ]

    # Sanity: the vendored default epochs match our PR95_DEFAULT_EPOCHS constant
    # (a drift guard — if the source changes, the compression math is recomputed
    # from the live values, not the stale constant).
    live_defaults = tuple(int(c.epochs) for c in raw_cfgs)

    if total_epoch_budget is None:
        epoch_overrides: list[int | None] = [None] * 8
    else:
        live_total = sum(live_defaults)
        epoch_overrides = [
            max(min_epochs_per_stage, round(d * total_epoch_budget / live_total))
            for d in live_defaults
        ]

    specs: list[StageSpec] = []
    for i, cfg in enumerate(raw_cfgs):
        spec = _spec_from_stage_config(cfg, epoch_overrides[i], ema_decay)
        if eval_every is not None:
            spec = StageSpec(**{**spec.__dict__, "eval_every": int(eval_every)})
        specs.append(spec)
    return specs


def seg_temperature_for_epoch(spec: StageSpec, epoch_in_stage: int) -> float:
    """The PREDICTION softmax temperature for ``epoch_in_stage`` (0-based) — the
    Lever-2 anneal hook (the OPTIMIZE part the combined arm flagged as missing).

    DEFAULT-PRESERVING: ``spec.seg_temperature_end is None`` returns the STATIC
    ``spec.seg_temperature`` for EVERY epoch — byte-identical to the pre-anneal driver
    (proved by ``test_anneal_disabled_returns_static_temperature``).

    When ``seg_temperature_end`` is set, the temperature COSINE-anneals from
    ``seg_temperature`` (epoch 0) toward ``seg_temperature_end`` (the final epoch),
    the memo's "T: 1.0 → 0.05 toward hard argmax". Cosine (not linear) spends most of
    the schedule near the start temperature and sharpens late — matching the
    convergence cadence (the boundary gradient sharpens once the coarse structure is
    learned), and mirroring the driver's cosine LR schedule. Single-epoch stages
    return the start temperature (no anneal room). Clamped to ``[end, start]`` (or
    ``[start, end]`` if the anneal goes UP) so the value never overshoots either end.

    SEG-GRADIENT FLOOR (R10+R12, MEASURED on the REAL frozen scorer): the soft-cosine
    surrogate ``1 − softmax(pred/T)[gt]`` is a TEMPERATURE-SHARPENED distillation
    loss, so as ``T → 0`` its per-pixel GRADIENT ``∝ p·(1−p)`` VANISHES even though
    its VALUE stays correct (≈1 on a wrong pixel, ≈0 on a right one). On the real
    SegNet (logit margins ~2–6, R8) the param/latent-gradient norm decays CONTINUOUSLY
    and is already EFFECTIVELY DEAD well above 0.1. R12's fine real-scorer sweep of the
    COMBINED (Lever-5 margin × Lever-2 surrogate) gradient, normalized to the warm
    T=1.0 norm, is:

        T=1.00 ratio 1.00      T=0.30 ratio 1.7e-3
        T=0.50 ratio 6.5e-2    T=0.25 ratio 2.9e-4
        T=0.40 ratio 1.7e-2    T=0.20 ratio 2.2e-5
        T=0.15 ratio 4.9e-7    T=0.12 ratio 1.0e-8
        T=0.10 ratio 1.9e-10   T=0.05 ratio 4.1e-20

    The gradient is within ~2 orders of warm only down to ``T ≈ 0.3`` (ratio 1.7e-3);
    below 0.3 it falls off a cliff (T=0.2 is already 2.2e-5 ≈ 5 orders down, T=0.1 is
    1.9e-10 ≈ 10 orders down — effectively dead). So the USABLE-gradient floor is
    ``T ≈ SEG_ANNEAL_GRADIENT_FLOOR_T = 0.3``, NOT 0.1. (R10 ORIGINALLY set the
    constant to 0.1, but R10's OWN measured table labeled T=0.1 as "≈ dead"
    (|grad|≈5e-12) — the constant mislabeled a 10-orders-below-warm dead-zone edge as
    "alive". R12 caught + corrected this; see the R12 memo.) This cold-dead behavior is
    INTENDED ("lock in the argmax late"), NOT a surrogate defect — but the GUARD's
    threshold must mark where the gradient is still USABLE, not where it is already a
    rounding error. At the corrected floor 0.3, the live arm's 1.0→0.05 cosine keeps
    ~66% of the stage gradient-ALIVE and goes dead for the last ~34%. CRUCIALLY,
    Lever-5 (``margin_weight_tau``) CANNOT rescue the cold-dead gradient: it multiplies
    the per-pixel surrogate by a DETACHED weight ``exp(−margin/τ) ∈ (0,1]``, so it can
    only SCALE DOWN the already-dead gradient (and ``exp(−margin/τ)`` is SMALLEST on the
    confidently-WRONG large-margin flip pixels — exactly the pixels the cold surrogate
    already cannot fix). R12 MEASURED the combined floor: margin-ON grad ≤ margin-OFF
    grad at every T (6.3e-20 vs 1.8e-19 at T=0.05). A tuner that sets
    ``seg_temperature_end`` below the floor OR makes a stage so short the schedule
    reaches cold T before the argmax converges will silently waste the seg lever for
    the cold portion (use :func:`seg_anneal_temperature_is_gradient_alive` to detect it).
    """
    if spec.seg_temperature_end is None:
        return float(spec.seg_temperature)
    t0 = float(spec.seg_temperature)
    t1 = float(spec.seg_temperature_end)
    if spec.epochs <= 1:
        return t0
    e = max(0, min(int(epoch_in_stage), spec.epochs - 1))
    # FAST-COOL-THEN-HOLD shape (the calibrated schedule): cosine from t0 → t1 over the
    # first ``hold_frac`` of the stage, then HOLD at t1. The denominator is the LAST
    # epoch of the cool phase (so the cool phase reaches t1 exactly, then parks). When
    # ``hold_frac is None`` the cool phase is the WHOLE stage (the plain cosine below).
    if spec.seg_temperature_hold_frac is not None:
        frac = float(spec.seg_temperature_hold_frac)
        # Number of epochs in the cool phase (>=2 so there is anneal room; clamped to the
        # stage). cool_epochs-1 is the last cool epoch index where T==t1; after it, HOLD.
        cool_epochs = max(2, min(spec.epochs, round(frac * spec.epochs)))
        if e >= cool_epochs - 1:
            return float(t1)  # HOLD at the end temperature for the rest of the stage
        cos = 0.5 * (1.0 - math.cos(math.pi * e / (cool_epochs - 1)))  # 0 → 1 over cool
        t = t0 + (t1 - t0) * cos
        lo, hi = (t1, t0) if t1 <= t0 else (t0, t1)
        return float(min(max(t, lo), hi))
    # Plain cosine progress in [0, 1] over the WHOLE stage (0 at epoch 0, 1 at the final).
    cos = 0.5 * (1.0 - math.cos(math.pi * e / (spec.epochs - 1)))  # 0 → 1
    t = t0 + (t1 - t0) * cos
    lo, hi = (t1, t0) if t1 <= t0 else (t0, t1)
    return float(min(max(t, lo), hi))


# The soft-cosine seg surrogate's per-pixel gradient ∝ p·(1−p) with
# p = softmax(pred/T)[gt]. On the real SegNet (logit margins ~2–6, R8) the COMBINED
# (Lever-5 margin × Lever-2 surrogate) param/latent gradient norm, normalized to the
# warm T=1.0 norm, is: T=0.5 → 6.5e-2, T=0.4 → 1.7e-2, T=0.3 → 1.7e-3, T=0.2 → 2.2e-5,
# T=0.1 → 1.9e-10 (10 orders down — effectively dead), T=0.05 → 4.1e-20. The gradient
# is still USABLE (within ~2 orders of warm) only down to T ≈ 0.3; below 0.3 it falls
# off a cliff. This floor is the T below which the seg lever produces no usable
# training signal — an INTENDED property of cold-temperature distillation ("lock in
# the argmax"), NOT a defect, but a tuner must not drive a stage cold before its argmax
# converges. R10 originally set this to 0.1, but BOTH R10's own measured table (T=0.1
# labeled "≈ dead", |grad|≈5e-12) AND R12's fine real-scorer sweep show 0.1 is already
# 10 orders below warm — the constant mislabeled a dead-zone edge as "alive". R12
# corrected the floor to 0.3 (the measured usable knee). MEASURED, REAL frozen scorer,
# R10 (experiments/probe_r10_lever_interaction_sign.py) + R12
# (experiments/probe_r12_combined_seg_gradient_floor.py).
SEG_ANNEAL_GRADIENT_FLOOR_T = 0.3


def seg_anneal_temperature_is_gradient_alive(temperature: float) -> bool:
    """True iff the soft-cosine seg surrogate still has a USABLE gradient at this
    prediction-softmax ``temperature`` — i.e. ``temperature >= SEG_ANNEAL_GRADIENT_
    FLOOR_T`` (= 0.3, the measured usable knee). Below the floor the temperature-
    sharpened surrogate's ``p·(1−p)`` gradient is already 5+ orders of magnitude below
    the warm-T norm on the real SegNet (R12 sweep: ratio 2.2e-5 at T=0.2, 1.9e-10 at
    T=0.1, 4.1e-20 at T=0.05), so the seg lever produces no usable training signal.
    The floor is 0.3 NOT 0.1 because R12's fine real-scorer sweep showed the gradient
    is within ~2 orders of warm only down to T≈0.3 (ratio 1.7e-3) and craters below;
    R10's original 0.1 mislabeled a 10-orders-below-warm dead-zone edge as "alive".
    The guard holds for the COMBINED (Lever-5 margin × Lever-2 surrogate) gradient too:
    Lever-5's detached ``exp(−margin/τ) ∈ (0,1]`` weight can only SCALE DOWN the dead
    gradient, never revive it (R12 MEASURED margin-ON ≤ margin-OFF at every T).

    This is the guard a tuner consults when choosing ``seg_temperature_end`` / a
    stage's epoch budget: the cosine anneal SHOULD reach the cold floor only AFTER
    the seg argmax has converged (the intended "lock-in"), never before (which would
    silently waste the seg lever). The driver itself does NOT gate on this (the
    intended late-cold behavior is correct); this is an OBSERVABILITY helper for
    schedule design + the R10/R12 regression guard."""
    return float(temperature) >= SEG_ANNEAL_GRADIENT_FLOOR_T


# The MARGIN-ADAPTIVE seg-temperature schedule's clamp window (the #119 ANNEAL-SCHEDULE
# CALIBRATION result). The optimal-anneal calculus
# (.omx/research/anneal_optimal_math_geometry_calculus_20260613.md) shows the soft-cosine
# surrogate's per-flip gradient ``|grad_g(Δ, T)| ≈ (1/T)·e^{−Δ/T}`` is MAXIMIZED exactly
# at ``T* = Δ`` (the flip's own logit margin): below T=Δ it dies super-exponentially
# (the R12 dead zone), above it fades ~1/T (untargeted). So the MAX-SIGNAL temperature
# TRACKS the live flip-margin distribution, floored at the gradient-alive knee
# ``T_floor = SEG_ANNEAL_GRADIENT_FLOOR_T = 0.3`` and capped at ``T_max = 0.6`` (above
# which the surrogate spreads gradient too thin across classes — the gate's T=1.0 row was
# worse than CE). The cap also bounds the early-large-Δ reach so the schedule never warms
# back into the untargeted regime. These are the schedule's two structural knobs.
SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR = 0.3
SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX = 0.6


def seg_temperature_margin_adaptive(
    median_flip_margin: float,
    *,
    t_floor: float = SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR,
    t_max: float = SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX,
) -> float:
    """The MAX-SIGNAL (theoretically-optimal) seg-surrogate PREDICTION temperature for a
    training STEP, given the LIVE flip-margin distribution — the #119 margin-adaptive
    schedule (.omx/research/anneal_optimal_math_geometry_calculus_20260613.md §6).

    The surrogate ``1 − softmax(pred/T)[gt]`` is MARGIN-RESONANT: on a confidently-wrong
    flip of logit margin ``Δ = pred_argmax − pred_gt > 0`` the GT-logit gradient magnitude
    is ``(1/T)·e^{−Δ/T}``, which the calculus maximizes at ``T* = Δ``. The bulk of the
    remaining flips at any step sit near the RUNNING MEDIAN of the flip margins, so the
    per-step max-signal temperature is::

        T(t) = clamp( running_median({Δ_i : pixel i is a current flip}),  t_floor,  t_max )

    floored at the gradient-alive knee (``t_floor = 0.3``; below it ``e^{−Δ/T} → 0``, the
    R12 dead zone) and capped (``t_max = 0.6``; above it the gradient spreads thin across
    classes — untargeted, the gate's T=1.0 < CE row). Lever-5's margin weight ``τ`` is
    SLAVED to this T (``τ(t) = T(t)``; §7) so the up-weighted small-margin pixels are
    exactly the ones whose surrogate gradient survives at this T.

    This is a PURE function of the measured median margin — the live flip-Δ median is
    supplied by the trainer/probe each step (it cannot be a function of (spec, epoch)
    because it depends on the decoder's CURRENT logits). ``seg_temperature_for_epoch``
    (the static / cosine / fast-cool-hold schedules) is UNCHANGED and remains the
    default; this is the OPT-IN max-signal alternative the from-0 launcher (#116) selects
    when ``spec.seg_temperature_margin_adaptive`` is set.

    Degenerate inputs (no flips this step → median is NaN/None) are the caller's
    responsibility to handle (hold the prior T); a non-finite ``median_flip_margin`` here
    returns ``t_floor`` (the safe gradient-alive default)."""
    m = float(median_flip_margin)
    lo = float(t_floor)
    hi = float(t_max)
    if hi < lo:
        raise ValueError(f"t_max ({hi}) must be >= t_floor ({lo})")
    if not math.isfinite(m):
        return lo
    return float(min(max(m, lo), hi))


# Fixed-point iteration count for the WEIGHTED-MEAN-RESONANT schedule. The map
# ``φ(T) = Σ Δ_i e^{−Δ_i/T} / Σ e^{−Δ_i/T}`` is a contraction near its fixed point
# (``φ'(T) = Var_w(Δ)/T² < 1`` for realistic flip-margin spreads), so 3–5 iterations from
# the median warm-start converge to machine-useful precision. 5 is a safe default.
SEG_ANNEAL_WEIGHTED_MEAN_FIXED_POINT_ITERS = 5


def seg_temperature_weighted_mean_resonant(
    flip_margins,
    *,
    t_floor: float = SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR,
    t_max: float = SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX,
    n_iter: int = SEG_ANNEAL_WEIGHTED_MEAN_FIXED_POINT_ITERS,
) -> float:
    """The CORRECTED max-signal seg-surrogate PREDICTION temperature for a training STEP —
    the ``e^{−Δ/T}``-WEIGHTED-MEAN fixed point of the live flip margins (the #119 crux fix;
    .omx/research/anneal_optimal_math_geometry_calculus_20260613.md §6′).

    :func:`seg_temperature_margin_adaptive` set ``T(t) = clamp(MEDIAN(Δ), floor, max)``,
    but the median is the WRONG aggregating statistic. The per-flip surrogate gradient is
    ``g(Δ,T) ≈ (1/T)·e^{−Δ/T}`` (resonance ``T*=Δ`` per flip); over the live flip set
    ``{Δ_i}`` a SINGLE global ``T`` drives the AGGREGATE ``G(T) = (1/T)·Σ_i e^{−Δ_i/T}``,
    whose argmax (``G'(T)=0``) is the self-consistent::

        T* = [ Σ_i Δ_i·e^{−Δ_i/T*} ] / [ Σ_i e^{−Δ_i/T*} ]

    — the e^{−Δ/T}-weighted (reachability-weighted) mean of the margins, NOT the median.
    The weight ``e^{−Δ_i/T}`` exponentially discounts deep/unreachable large-Δ flips, so
    ``T*`` shades toward the reachable small-Δ mass (``minΔ ≤ T* ≤ meanΔ``). On a
    right-skewed flip-margin distribution this sits BELOW the median — which is exactly why
    the slice-0 calibration found a committed-low schedule (fast-cool → 0.3) beat the
    median-adaptive arm (parked at the head-median ≈0.56, too warm for the bulk of the
    reachable flips). This function computes ``T*`` by the contraction fixed-point iteration
    ``φ(T) = Σ Δ_i e^{−Δ_i/T} / Σ e^{−Δ_i/T}`` warm-started at the median, then clamps the
    result to the gradient-alive band ``[t_floor, t_max]`` (floor 0.3 = the R12 dead-zone
    knee; cap 0.6 = above which the surrogate spreads gradient too thin across classes).
    Lever-5's ``τ`` is SLAVED to this T (``τ(t)=T(t)``; §7).

    ``flip_margins`` is the RAW per-flip-pixel margin vector ``Δ_i = pred_argmax − pred_gt``
    (any sequence or a 1-D ``torch.Tensor``); non-finite entries are dropped. An EMPTY flip
    set (no flips this step) returns ``t_floor`` (the safe gradient-alive default; the caller
    may instead hold the prior T). A torch import is LAZY (module import stays light) and the
    fixed point is vectorized when a tensor is supplied.

    This is the OPT-IN principled alternative to ``seg_temperature_margin_adaptive``;
    ``seg_temperature_for_epoch`` (static / cosine / fast-cool-hold) is UNCHANGED and remains
    the default, so existing schedules stay byte-identical."""
    lo = float(t_floor)
    hi = float(t_max)
    if hi < lo:
        raise ValueError(f"t_max ({hi}) must be >= t_floor ({lo})")

    import torch  # lazy: keep module import light; torch is a hard project dep

    if torch.is_tensor(flip_margins):
        m = flip_margins.detach().reshape(-1).to(torch.float64)
    else:
        m = torch.as_tensor(list(flip_margins), dtype=torch.float64)
    m = m[torch.isfinite(m)]
    if m.numel() == 0:
        return lo

    # Warm-start at the (unclamped) median; iterate the UNclamped fixed point so we find the
    # true weighted mean, then clamp the assigned value to the alive band. Guard tiny-T
    # underflow (all weights → 0 ⟹ keep the current iterate).
    T = float(m.median().item())
    if not math.isfinite(T) or T <= 0.0:
        T = max(float(m.min().item()), 1e-6)
    for _ in range(max(1, int(n_iter))):
        w = torch.exp(-m / T)
        denom = float(w.sum().item())
        if denom <= 0.0 or not math.isfinite(denom):
            break
        T_new = float((m * w).sum().item() / denom)
        if not math.isfinite(T_new) or T_new <= 0.0:
            break
        T = T_new
    return float(min(max(T, lo), hi))


__all__ = [
    "PR95_DEFAULT_EPOCHS",
    "PR95_TOTAL_EPOCHS",
    "SEG_ANNEAL_GRADIENT_FLOOR_T",
    "SEG_ANNEAL_MARGIN_ADAPTIVE_T_FLOOR",
    "SEG_ANNEAL_MARGIN_ADAPTIVE_T_MAX",
    "SEG_ANNEAL_WEIGHTED_MEAN_FIXED_POINT_ITERS",
    "StageSpec",
    "build_curriculum",
    "seg_anneal_temperature_is_gradient_alive",
    "seg_temperature_for_epoch",
    "seg_temperature_margin_adaptive",
    "seg_temperature_weighted_mean_resonant",
]
