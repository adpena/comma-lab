# SPDX-License-Identifier: MIT
"""Pure, dependency-free predicates for confound observability.

The functions classify when telemetry is due; none actuates training.
"""

from __future__ import annotations

import math
from itertools import pairwise

VERDICT_LIVE_GAP_AUTO_WARMUP = -1
VERDICT_LIVE_GAP_EXPLICIT_OFF = 0
PARTIAL_FREEZE_LO = 0.02
PARTIAL_FREEZE_HI = 0.5


def ema_warmup_updates(ema_decay: float) -> int:
    r"""Return the DERIVED two-time-constant EMA warmup ``ceil(2/(1-beta))``."""

    beta = float(ema_decay)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay!r}")
    return math.ceil(2.0 / (1.0 - beta))


def verdict_live_gap_due(
    mode: int,
    *,
    epoch: int,
    ema_updates: int,
    ema_decay: float,
) -> bool:
    """Whether the read-only live-vs-EMA verdict observer is due.

    ``-1`` is automatic warmup mode, ``0`` an explicit opt-out, and ``K > 0``
    retains the DSL lever's all-run every-Kth-verdict semantics.
    """

    cadence = int(mode)
    if cadence < VERDICT_LIVE_GAP_AUTO_WARMUP:
        raise ValueError(f"verdict-live-gap mode must be -1, 0, or K>0, got {mode!r}")
    if cadence == VERDICT_LIVE_GAP_EXPLICIT_OFF:
        return False
    if cadence == VERDICT_LIVE_GAP_AUTO_WARMUP:
        return int(ema_updates) < ema_warmup_updates(ema_decay)
    return int(epoch) % cadence == 0


def is_partial_freeze(accepted_frac: float) -> bool:
    """The L1 open alarm band: ``0.02 < accepted_frac < 0.5``."""

    frac = float(accepted_frac)
    return PARTIAL_FREEZE_LO < frac < PARTIAL_FREEZE_HI


def is_known_dseg_descent(values: list[float] | tuple[float, ...]) -> bool:
    """Known-effect L3 positive control: at least two strictly descending d_seg values."""

    vals = tuple(float(v) for v in values)
    return len(vals) >= 2 and all(b < a for a, b in pairwise(vals))
