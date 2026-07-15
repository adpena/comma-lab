# SPDX-License-Identifier: MIT
"""Canonical-equations note for the 2026-07-15 confound hardening.

Triality anchor:

* EMA live-gap window: ``U_warm = ceil(2 / (1 - beta_ema))``.
* partial-freeze alarm: ``0.02 < accepted_frac < 0.5``.
* d_seg positive control: ``d_seg[t+1] < d_seg[t]`` for every control step.

The dependency-free executable definitions live in :mod:`tac.confound_observability`
so importing the trainer does not eagerly load the full canonical-equations registry.
"""

from tac.confound_observability import (  # re-export: one executable source of truth
    PARTIAL_FREEZE_HI,
    PARTIAL_FREEZE_LO,
    VERDICT_LIVE_GAP_AUTO_WARMUP,
    VERDICT_LIVE_GAP_EXPLICIT_OFF,
    ema_warmup_updates,
    is_known_dseg_descent,
    is_partial_freeze,
    verdict_live_gap_due,
)

__all__ = (
    "PARTIAL_FREEZE_HI",
    "PARTIAL_FREEZE_LO",
    "VERDICT_LIVE_GAP_AUTO_WARMUP",
    "VERDICT_LIVE_GAP_EXPLICIT_OFF",
    "ema_warmup_updates",
    "is_known_dseg_descent",
    "is_partial_freeze",
    "verdict_live_gap_due",
)
