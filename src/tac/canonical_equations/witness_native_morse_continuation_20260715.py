# SPDX-License-Identifier: MIT
"""Witness-native Morse continuation controls for task #302.

The inherited PR95 scalars mixed three unrelated conventions: Muon used a
fraction of Adam's learning rate, while L7 used a discontinuous threshold and a
literal multiplier.  The level-set action instead gives a local trust-region
step and a continuous margin homotopy:

    eta_mu = sqrt(2 delta_KL / lambda_max(F))
    m_l7   = m_safe
    w_l7   = 0

``w_l7=0`` is structural: under the unified-tau continuation, an extra hard L7
indicator is not a term in the Euler-Lagrange action and would double-own the
same sharpening force.  ``m_l7`` remains the R-survival boundary for legacy
telemetry/resume compatibility, but cannot apply a loss when the multiplier is
zero.  Efficacy and the checkpoint-local curvature are deliberately not guessed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EQUATION_ID = "witness_native_morse_continuation_v1"


@dataclass(frozen=True)
class MorseContinuationControls:
    muon_lr: float
    l7_mult: float
    l7_threshold: float
    trust_region_kl: float
    fisher_curvature_upper: float
    m_safe: float


def derive_morse_continuation_controls(
    *,
    trust_region_kl: float,
    fisher_curvature_upper: float,
    m_safe: float,
) -> MorseContinuationControls:
    """Derive all three legacy argv scalars from action-native quantities."""

    delta = float(trust_region_kl)
    curvature = float(fisher_curvature_upper)
    margin = float(m_safe)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(
            f"trust_region_kl must be finite and > 0, got {trust_region_kl!r}"
        )
    if not math.isfinite(curvature) or curvature <= 0.0:
        raise ValueError(
            "fisher_curvature_upper must be finite and > 0, "
            f"got {fisher_curvature_upper!r}"
        )
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f"m_safe must be finite and > 0, got {m_safe!r}")
    return MorseContinuationControls(
        muon_lr=math.sqrt(2.0 * delta / curvature),
        l7_mult=0.0,
        l7_threshold=margin,
        trust_region_kl=delta,
        fisher_curvature_upper=curvature,
        m_safe=margin,
    )


__all__ = [
    "EQUATION_ID",
    "MorseContinuationControls",
    "derive_morse_continuation_controls",
]
