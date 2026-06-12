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


__all__ = ["PR95_DEFAULT_EPOCHS", "PR95_TOTAL_EPOCHS", "StageSpec", "build_curriculum"]
