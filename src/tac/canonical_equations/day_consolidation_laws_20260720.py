# SPDX-License-Identifier: MIT
"""Two measured laws from FEED-DAY-CONSOLIDATION-20260720B, with real evaluators.

Registered per the triality drift-gate (equations leg for measured findings):

1. ``realization_breakeven_bytes_v1`` — the r2b interaction/realization break-even
   law: a repair stream is admissible only if its charged bytes are at or below
   ``realized_recovery_S / RATE_PRICE_S_PER_BYTE``. Anchor: the measured n600 r2b
   falsification control (realization 1,585/16,751 = 9.462%, realized recovery
   0.0012332316583976016 S -> break-even 1,852.09 B vs the 27,313 B feasible
   stream = the 14.75x reduction demand).

2. ``fixed_c1_pointer_crossing_cap_bytes_v1`` — the corrected fixed-C1 archive
   cap: at the capstone spine's measured nonrate distortion, the archive must be
   at or below ``floor((pointer_S - nonrate_S) / RATE_PRICE_S_PER_BYTE)`` bytes
   to cross the pointer. Anchor: the inverse-solve arm's corrected constant
   216,223 B (supersedes the 216,300/264,320 approximations); the residual
   between this module's evaluator (fed the r2b-replay nonrate S) and the arm's
   exact-input constant is recorded honestly, never hidden.

The remaining five §B4 ledger laws stay OWED at the #582(b) reconcile because
their anchor custody (receipts/SHAs) lives on unmerged arm branches; these two
have main-visible inputs and closed-form evaluators, so they register now.
"""
from __future__ import annotations

import math

# The contest rate price: 25 score-units per 37,545,489 archive bytes
# (upstream/evaluate.py:63,92 — the only byte term the score counts).
RATE_PRICE_S_PER_BYTE: float = 25.0 / 37_545_489.0


def breakeven_bytes(realized_recovery_s: float) -> float:
    """Greatest charged-byte budget a repair stream may spend and still pay rent.

    A stream recovering ``realized_recovery_s`` score-units breaks even at
    exactly ``realized_recovery_s / RATE_PRICE_S_PER_BYTE`` bytes; one byte
    more is a net regression. Realized (hard-oracle) recovery only — never the
    scheduled upper bound (the measured 9.462% realization gap is the reason
    this law exists).
    """
    if realized_recovery_s < 0:
        raise ValueError("realized_recovery_s must be >= 0")
    return realized_recovery_s / RATE_PRICE_S_PER_BYTE


def fixed_c1_cap_bytes(pointer_s: float, nonrate_s: float) -> int:
    """Largest archive (bytes) that crosses ``pointer_s`` at fixed distortion.

    ``nonrate_s`` is the vehicle's measured distortion-only score
    (100*d_seg + sqrt(10*d_pose)); the cap is the byte budget at which
    ``nonrate_s + rate(bytes)`` equals the pointer. Floor: the last integer
    byte count strictly at-or-below the crossing.
    """
    if nonrate_s >= pointer_s:
        raise ValueError("nonrate_s must be below pointer_s for a finite cap")
    return math.floor((pointer_s - nonrate_s) / RATE_PRICE_S_PER_BYTE)
