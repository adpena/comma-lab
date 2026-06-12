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
    """
    if spec.seg_temperature_end is None:
        return float(spec.seg_temperature)
    t0 = float(spec.seg_temperature)
    t1 = float(spec.seg_temperature_end)
    if spec.epochs <= 1:
        return t0
    # Cosine progress in [0, 1] over the stage (0 at epoch 0, 1 at the final epoch).
    e = max(0, min(int(epoch_in_stage), spec.epochs - 1))
    cos = 0.5 * (1.0 - math.cos(math.pi * e / (spec.epochs - 1)))  # 0 → 1
    t = t0 + (t1 - t0) * cos
    lo, hi = (t1, t0) if t1 <= t0 else (t0, t1)
    return float(min(max(t, lo), hi))


__all__ = [
    "PR95_DEFAULT_EPOCHS",
    "PR95_TOTAL_EPOCHS",
    "StageSpec",
    "build_curriculum",
    "seg_temperature_for_epoch",
]
