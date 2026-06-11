# SPDX-License-Identifier: MIT
"""Canonical EMA warmup-decay schedule — the ONE source of truth (2026-06-11).

THE BUG THIS EXTINCTS (operator: *"the source MLX port is the source of the
poison, all must be fixed ... be wary of fruit of the poisoned tree"*):

A weight-EMA shadow with a CONSTANT decay ``d`` has an exponential time constant
``tau = 1/(1-d)`` *steps* (0.997 -> 333; 0.999 -> 1000). On a SHORT run (our MLX
loops are ~6 steps/epoch, a curriculum stage ~240 steps) the shadow stays
``~init`` weights for the whole run. Anything that reads the shadow — eval
(``use_ema_for_eval``) OR export (the archive bytes the shadow) — then reflects
near-init weights even though the LIVE weights converged. This silently:

  * froze the capstone ``exact_d_seg`` at the init 0.505 ("seg-capacity wall")
    while the live d_seg descended to 0.041 (``diag_curriculum_ema_lag.py``);
  * would ship a near-init-seg archive on any short-run export.

THE FIX — warmup the decay so the shadow TRACKS the live weights from step 1:

    decay_t = min(target_decay, (1 + t) / (warmup_const + t))     # timm ModelEmaV2

At ``t=0`` this is ~0.09 (shadow == live; nothing to average yet); it ramps to
``target_decay`` as updates accumulate, recovering the intended Polyak variance
reduction once enough iterates exist to average. This is the CORRECT correction
for a WEIGHT-INIT EMA — the Adam-style ``shadow/(1 - decay^t)`` bias correction
assumes a ZERO init and is WRONG here (it would divide a non-zero-init shadow).

Every weight-EMA in the repo (``_CapstoneWeightEMA``, ``_MlxEMA`` x2,
``tac.training.EMA``, the ``_EMA`` shadows, ``PolyakEMAShadow``) routes its
per-update decay through :func:`warmup_ema_decay` so the schedule lives in ONE
place. Codebook EMAs (van den Oord VQ: ``VectorQuantizerEMA*``,
``ema_update_from_last``) are DELIBERATELY excluded — they adapt fast by design
(decay 0.99) and are not eval/export shadows (CLAUDE.md EMA non-negotiable).
"""
from __future__ import annotations

# The timm ``ModelEmaV2`` warmup constant. At update ``t`` the warmup factor is
# ``(1+t)/(WARMUP_CONST+t)``: ~0.09 at t=0, 0.5 at t=WARMUP_CONST, asymptotes 1.
DEFAULT_EMA_WARMUP_CONST = 10.0


def warmup_ema_decay(
    num_updates: int,
    target_decay: float,
    *,
    warmup_const: float = DEFAULT_EMA_WARMUP_CONST,
) -> float:
    """Return the warmup-ramped EMA decay for update ``num_updates``.

    ``min(target_decay, (1 + t) / (warmup_const + t))`` — the shadow tracks the
    live weights early (decay << target while ``t`` is small) and converges to
    ``target_decay`` as updates accumulate. Monotone non-decreasing in ``t``;
    never exceeds ``target_decay``.

    Args:
        num_updates: number of EMA updates applied SO FAR (1-based at the first
            update; pass the post-increment counter). Negative is clamped to 0.
        target_decay: the asymptotic decay cap (e.g. 0.997). Returned once the
            warmup ramp catches up.
        warmup_const: the timm warmup constant (default 10). Larger = slower ramp.

    Returns:
        the effective decay in ``[(1)/(warmup_const), target_decay]``.
    """
    t = max(0, int(num_updates))
    warmup = (1.0 + t) / (float(warmup_const) + t)
    return min(float(target_decay), warmup)
