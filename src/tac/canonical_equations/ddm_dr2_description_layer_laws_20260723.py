# SPDX-License-Identifier: MIT
"""Canonical description-layer laws measured by ddm_dr2 (2026-07-23).

Three measured facts, one evaluator surface:

1. Mode-race dominance at the EXACT coding layer: a per-record RDO mode
   envelope (static/track/re-key) is DOMINATED by the SDWL1 typed-section
   causal-delta baseline (challenger 70,700 B vs 68,464 B, +2,236 B).
   verdict_scope: formulation-at-exact-layer; the lossy-tolerance mode
   question remains OPEN (blocked on the sensitivity quantizer).
2. U1 rung-0 description headroom: exact SDWL1 description bytes sit inside
   the strict sub-0.15 archive cap with headroom for base+corrections+pose
   (154,524 - 68,464 = 86,060 B at c1 pose held).
3. Static-fraction attribution correction: the widely quoted 99.1%
   static-in-image figure belongs to the MOVABLE BAND (0.9914933...), not
   Road; the lane corridor measures 0.9777958...; and pixel recurrence is
   NOT record constancy (recurrence upper-bounds describable constancy).

Anchors: dr2 receipt .omx/research/ddm_dr2_scc_outside_view_receipt_20260723.json
(sha256 4e3cec44f9176342642b363d67e64b3ff92b010f01c879c8f4e44f00839f8ce7),
merged main 4c507517767d53da7ae867fe1e71689e522c6145. All rows are
[macOS-CPU advisory] coder/inventory measurements: score_claim=false,
promotable=false.
"""

from __future__ import annotations

EQUATION_ID = "ddm_dr2_description_layer_laws_v1"

# Measured constants (dr2 receipt, SHA-pinned above).
SDWL1_EXACT_DESCRIPTION_BYTES = 68_464
MODE_RACE_CHALLENGER_BYTES = 70_700
STRICT_SUB015_CAP_BYTES_POSE_HELD = 154_524
G4_MOVABLE_BAND_STATIC_FRACTION = 0.9914933227057525
G4_LANE_CORRIDOR_STATIC_FRACTION = 0.9777958023930171


def mode_race_delta_bytes(challenger_bytes: int, baseline_bytes: int) -> int:
    """Dominance margin: positive => challenger DOMINATED at the exact layer."""
    for name, value in (("challenger_bytes", challenger_bytes), ("baseline_bytes", baseline_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer byte count")
    return challenger_bytes - baseline_bytes


def description_headroom_bytes(cap_bytes: int, description_bytes: int) -> int:
    """Rung-0 headroom = archive cap minus exact description stream bytes."""
    for name, value in (("cap_bytes", cap_bytes), ("description_bytes", description_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer byte count")
    return cap_bytes - description_bytes


def record_constancy_upper_bound(pixel_recurrence_fraction: float) -> float:
    """Pixel recurrence UPPER-BOUNDS record constancy; never equates it.

    The correction law: a static-in-image pixel fraction f admits record
    constancy at most f (attribution must be per-stratum and record-level).
    """
    if not isinstance(pixel_recurrence_fraction, float) or not (0.0 <= pixel_recurrence_fraction <= 1.0):
        raise ValueError("pixel_recurrence_fraction must be a float in [0, 1]")
    return pixel_recurrence_fraction
