"""Per-term MEASURED distortion floors — the P8 FLOOR-FIRST resolver for the duty-to-measure ranking.

Design philosophy P8 (``design_philosophies_eightfold_20260709``): *before optimizing any term,
derive/measure its floor; optimize only the gap-to-floor; a surface at floor is CLOSED to polish.*
This module makes the campaign's MEASURED per-term floors first-class + queryable so the costate
duty-to-measure ranking (``activation_ledger.duty_to_measure_ranked``) can consult them: a lever whose
target term (``d_seg`` / ``d_pose`` / ``rate``) is at its floor cannot buy any more ΔS on that axis
(``AT_FLOOR`` → ranks ~0), and a lever's achievable ΔS is CAPPED by the term's headroom-to-floor.

NO-FAKE (the SUPREME rule): every floor here is resolved from a NAMED canonical anchor with an explicit
value-provenance LABEL — never a guessed number. A term whose floor is regime-dependent (no single
number), LOOSE/refuted, or genuinely un-measured returns a floor with ``usable_for_capping=False``: it
is SURFACED (provenance cited) but does NOT change the ranking. Only a clean numeric ``MEASURED`` floor
caps est / fires ``AT_FLOOR``. This keeps the LIVE ranking honest (most floors are OWED — exactly what
P8 predicts) while making the machinery active the moment a clean floor + a measured current-term value
land together.

Value-provenance ladder (``FloorSpec.label``):
  * ``MEASURED``               — a clean measured numeric floor with a canonical anchor (usable for capping).
  * ``MEASURED_REGIME_DEPENDENT`` — measured but no single number (an identity/curve); surfaced, not capping.
  * ``LOOSE``                  — a bound tagged LOOSE/refuted by its own source; surfaced, never caps.
  * ``OWED_UNMEASURED``        — the floor is an owed measurement (P8/P2); surfaced as a duty-to-measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# canonical value-provenance labels (the ladder above)
FLOOR_MEASURED = "MEASURED"
FLOOR_MEASURED_REGIME_DEPENDENT = "MEASURED_REGIME_DEPENDENT"
FLOOR_LOOSE = "LOOSE"
FLOOR_OWED_UNMEASURED = "OWED_UNMEASURED"
VALID_FLOOR_LABELS = frozenset(
    {FLOOR_MEASURED, FLOOR_MEASURED_REGIME_DEPENDENT, FLOOR_LOOSE, FLOOR_OWED_UNMEASURED}
)

# floor_status tokens the ranking stamps per row (the observable consequence)
FLOOR_STATUS_AT_FLOOR = "AT_FLOOR"                    # current term ≤ floor -> lever buys ~0 on this axis
FLOOR_STATUS_CAPPED = "HEADROOM_CAPPED"               # est exceeded headroom-to-floor -> est capped down
FLOOR_STATUS_ABOVE_FLOOR = "ABOVE_FLOOR"             # measured floor + current known, est within headroom
FLOOR_STATUS_CURRENT_UNKNOWN = "FLOOR_KNOWN_CURRENT_UNKNOWN"  # numeric floor known, no current-term value
FLOOR_STATUS_UNMEASURED = "FLOOR_UNMEASURED"          # no usable numeric floor (regime-dependent/LOOSE/owed)

# S = 100·d_seg + √(10·d_pose) + 25·bytes/N.  Per-axis coefficient to map a TERM delta to an S delta.
_S_PER_DSEG = 100.0
_POSE_MULT = 10.0  # S_pose = √(10·d_pose)
_RATE_DENOM = 37_545_489.0  # bytes denominator (rate term = 25·bytes/N)


@dataclass(frozen=True)
class FloorSpec:
    """One MEASURED (or surfaced-but-unmeasured) per-term floor, with value-provenance.

    ``value`` is the floor in the term's own units (``d_seg`` fraction / ``d_pose`` MSE / ``rate`` S-units).
    ``value`` is ``None`` when the floor has no single number (regime-dependent / owed). ``label`` places
    the floor on the value-provenance ladder; only ``MEASURED`` with a numeric ``value`` is
    ``usable_for_capping``. ``provenance`` is the canonical anchor (equation_id / memo path) that measured
    it — NO-FAKE: every floor cites its source.
    """

    axis: str
    value: float | None
    label: str
    provenance: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.axis not in ("d_seg", "d_pose", "rate"):
            raise ValueError(f"floor axis must be d_seg/d_pose/rate, got {self.axis!r}")
        if self.label not in VALID_FLOOR_LABELS:
            raise ValueError(f"invalid floor label {self.label!r}; must be one of {sorted(VALID_FLOOR_LABELS)}")
        if self.value is not None:
            v = float(self.value)
            if v != v or v < 0:  # NaN or negative
                raise ValueError(f"floor value must be a non-negative number or None, got {self.value!r}")
        if not self.provenance:
            raise ValueError("floor provenance is required (NO-FAKE: every floor cites its measured source)")
        if self.label == FLOOR_MEASURED and self.value is None:
            raise ValueError("a MEASURED floor requires a numeric value (use MEASURED_REGIME_DEPENDENT for None)")

    @property
    def usable_for_capping(self) -> bool:
        """Only a clean MEASURED numeric floor caps est / fires AT_FLOOR. Everything else is surface-only."""
        return self.label == FLOOR_MEASURED and self.value is not None

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "value": self.value,
            "label": self.label,
            "provenance": self.provenance,
            "note": self.note,
            "usable_for_capping": self.usable_for_capping,
        }


# ── the canonical MEASURED-floor registry (resolved from canonical_equations/memos) ──────────────────
# Every entry cites its anchor. d_seg has a clean measured achievable floor (the anisotropic lane-offload
# regime); rate's info-theoretic floor is LOOSE by its own source; d_pose's floor is an owed measurement.
def resolve_term_floors() -> dict[str, FloorSpec]:
    """The per-axis floor registry (resolved from canonical anchors, NEVER guessed). Pure; no I/O.

    * ``d_seg`` — MEASURED 0.00087 (best measured achievable d_seg, lane-offloaded regime,
      ``anisotropic_basis_two_regime_allocation_v1``). Used as a CONSERVATIVE lower bound on achievable
      d_seg for capping (the real total floor is ≥ this, so capping never over-fires AT_FLOOR — the
      NO-FAKE-safe direction). The regime-dependent label-noise identity
      (``independent_flicker_jitter_dseg_floor_smooth_optimal_v1``, #141) is cited in the note.
    * ``rate`` — 0.11797 info-theoretic (rate-only) floor, but tagged LOOSE by its own source
      (CANONICAL_RESEARCH_INDEX: "LOOSE / REFUTED"); SURFACED, never caps.
    * ``d_pose`` — OWED: the counted-seed floor is un-measured (P8) and the single-seed deterministic
      spine leaves across-seed d_pose variance UNKNOWN (P2); SURFACED as a duty-to-measure.
    """
    return {
        "d_seg": FloorSpec(
            axis="d_seg",
            value=0.00087,
            label=FLOOR_MEASURED,
            provenance="anisotropic_basis_two_regime_allocation_v1 (MEASURED lane d_seg 0.00087, "
                       "lane-offloaded regime)",
            note="conservative achievable-d_seg lower bound; the label-noise floor is regime-dependent "
                 "(independent_flicker_jitter_dseg_floor_smooth_optimal_v1, #141 — an identity, no single number)",
        ),
        "rate": FloorSpec(
            axis="rate",
            value=0.11797,
            label=FLOOR_LOOSE,
            provenance="information_theoretic_floor S_floor=0.11797 (25·177169/37,545,489; "
                       "GOAL_standing_v3_20260610.md) — LOOSE/REFUTED per CANONICAL_RESEARCH_INDEX",
            note="surface-only: a LOOSE bound never caps est (a witness stores a tiny sufficient "
                 "statistic well below the frontier rate; the info-theoretic rate floor is not a hard limit)",
        ),
        "d_pose": FloorSpec(
            axis="d_pose",
            value=None,
            label=FLOOR_OWED_UNMEASURED,
            provenance="counted-seed floor OWED (P8) + single-seed spine -> across-seed d_pose variance "
                       "UNMEASURED (P2)",
            note="duty-to-measure: no clean d_pose floor measured; surfaced so it is not silently orphaned",
        ),
    }


def _term_delta_to_s(axis: str, term_lo: float, term_hi: float) -> float:
    """S-units gained by moving a term from ``term_hi`` (worse) down to ``term_lo`` (better). ≥ 0."""
    if term_hi <= term_lo:
        return 0.0
    if axis == "d_seg":
        return _S_PER_DSEG * (term_hi - term_lo)
    if axis == "rate":
        return term_hi - term_lo  # rate floors are already in S-units
    if axis == "d_pose":
        # nonlinear: S_pose = √(10·d_pose); headroom in S = √(10·hi) − √(10·lo)
        return math.sqrt(_POSE_MULT * term_hi) - math.sqrt(_POSE_MULT * term_lo)
    return 0.0


@dataclass(frozen=True)
class FloorApplication:
    """The result of applying a term's floor to one lever's ΔS estimate."""

    floor_status: str
    capped_est: float | None      # est after capping to headroom-to-floor (== est when not capped/unknown)
    headroom_s: float | None      # ΔS still available above the floor (None when current/floor unknown)
    floor: FloorSpec | None       # the consulted floor (surfaced even when not usable_for_capping)


def apply_term_floor(
    axis: str | None,
    est_delta_s: float | None,
    current_term_value: float | None,
    floor: FloorSpec | None,
) -> FloorApplication:
    """Consult a term's floor for one lever's ΔS estimate (pure; the ranking's floor-aware core).

    Semantics (P8 gap-to-floor):
      * No usable numeric floor (regime-dependent / LOOSE / owed / axis absent) -> ``FLOOR_UNMEASURED``,
        est passes through unchanged (surfaced only).
      * Numeric MEASURED floor but no ``current_term_value`` -> ``FLOOR_KNOWN_CURRENT_UNKNOWN``, est
        passes through (cannot judge at-floor without the current term value — an owed measurement).
      * current ≤ floor -> ``AT_FLOOR``, capped_est=0.0 (the surface is CLOSED to polish on this axis).
      * est exceeds the S-headroom to the floor -> ``HEADROOM_CAPPED``, capped_est=headroom.
      * otherwise -> ``ABOVE_FLOOR``, est unchanged.
    """
    if floor is None or not floor.usable_for_capping or floor.value is None:
        return FloorApplication(FLOOR_STATUS_UNMEASURED, est_delta_s, None, floor)
    if current_term_value is None:
        return FloorApplication(FLOOR_STATUS_CURRENT_UNKNOWN, est_delta_s, None, floor)
    headroom = _term_delta_to_s(axis or floor.axis, floor.value, float(current_term_value))
    if headroom <= 0.0:
        return FloorApplication(FLOOR_STATUS_AT_FLOOR, 0.0, 0.0, floor)
    if est_delta_s is None:
        # a registered duty-to-ESTIMATE lever above its floor: no est to cap, but headroom is measured.
        return FloorApplication(FLOOR_STATUS_ABOVE_FLOOR, None, headroom, floor)
    if float(est_delta_s) > headroom:
        return FloorApplication(FLOOR_STATUS_CAPPED, headroom, headroom, floor)
    return FloorApplication(FLOOR_STATUS_ABOVE_FLOOR, float(est_delta_s), headroom, floor)


__all__ = [
    "FLOOR_LOOSE",
    "FLOOR_MEASURED",
    "FLOOR_MEASURED_REGIME_DEPENDENT",
    "FLOOR_OWED_UNMEASURED",
    "FLOOR_STATUS_ABOVE_FLOOR",
    "FLOOR_STATUS_AT_FLOOR",
    "FLOOR_STATUS_CAPPED",
    "FLOOR_STATUS_CURRENT_UNKNOWN",
    "FLOOR_STATUS_UNMEASURED",
    "VALID_FLOOR_LABELS",
    "FloorApplication",
    "FloorSpec",
    "apply_term_floor",
    "resolve_term_floors",
]
