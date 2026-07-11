# SPDX-License-Identifier: MIT
"""SENSE: the oracle-floor detector — recognize when the LABEL-SMOOTH witness has
reached the temporal-majority (Morse-Smale finest-persistence) floor, so the
controller can switch from the label-descent regime to the APPEARANCE-PHASE regime.

This is the SENSE half of the #247/#303 costate controller's phase-reframe fold
(memo ``.omx/research/flicker_transform_geometry_term_design_20260710.md``,
DAG FEED-flicker-term, DAG FEED-phase-curriculum-costate). NOT an actuator: it
reads the advisory verdict d_seg trajectory and emits a REGIME signal; the
controller's DECIDE layer consumes it and RECOMMENDS (advisory-only; heavy/config/
stop = operator-GO, CONTAINMENT — this module never fires anything).

THE DERIVED FACT (not a hand-set threshold; §2.5 of the memo, eq candidate
``gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1``):

  A label-smooth witness (CE / margin / soft-cosine objective) CANNOT descend below
  the GT temporal-majority oracle floor d_seg = 0.005318 (the stride-2 single-frame
  SPIKE rate, n600). This floor is a MORSE-SMALE PERSISTENCE THRESHOLD, not a
  nuisance constant: the spikes are the space-time margin field's birth-death pairs
  BELOW one resolution/temporal cell (spatial persistence p50 0.56 < annulus 0.897,
  temporal persistence = 1 scored frame). The label-smooth stages of the flow
  (CE->tau->l7 = coarse->fine curvelet scale = persistence order, MEMORY L3) resolve
  every critical point ABOVE one cell; when d_seg flattens at ~0.0053 the flow has
  reached its finest label-persistence level. What remains (sub-0.15 needs 0.00077-
  0.00118, 4.5-7x BELOW the floor) is pierceable ONLY by the deterministic
  APPEARANCE-PHASE channel (existence proof: real-frame content through R reaches
  d_seg 0.00086, FEED-ma). That channel is the phase tail's forces: T1
  (phase_advection_consistency, cross-pair) + #360's four within-pair forces.

So ``label_floor_reached`` is the DERIVED transition event (the analogue of the C1
CE->tau critical-nucleus hand-off #315): the signal that label descent is EXHAUSTED
and the remaining descent is PHASE. It fires iff the run is (a) in a label-smooth
stage, (b) at-or-near the persistence floor, and (c) flat (the flow has settled at
that persistence level).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tac.witness_control.costate_estimator import (
    DEFAULT_STALL_REL_EPS,
    slope_with_stderr,
)

# ---- classification sentinels ---------------------------------------------------
NO_FLOOR = "NO_FLOOR"                       # not at the floor (still descending / above band / wrong stage)
LABEL_FLOOR_REACHED = "LABEL_FLOOR_REACHED"  # the phase-regime switch event
LABEL_FLOOR_UNIDENTIFIABLE = "LABEL_FLOOR_UNIDENTIFIABLE"  # too few same-stage rows to decide

# ---- DERIVED constants (each traces to the level-set energy / a measured anchor) --
#: The temporal-majority oracle floor = GT stride-2 single-frame spike rate, n600
#: (memo §2.4/§2.5). This is the Morse-Smale persistence threshold at one resolution
#: cell, NOT a tuned number: a smooth-label witness's converged residual 0.00496-0.0052
#: sits AT it. Anchor: flicker memo Part D + eq gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1.
ORACLE_LABEL_FLOOR_DSEG = 0.005318
#: Band around the floor within which "at the floor" is declared. Lower edge = the
#: measured witness popout floor 0.00496 (#205/L67); upper edge = a small margin above
#: the oracle so a run APPROACHING the floor (still 10-20% above, per-pair conditioning
#: not yet saturated) is caught at the transition, not one epoch late. Both edges are
#: MEASURED endpoints of the same converged-residual band, not free knobs.
FLOOR_BAND_LO = 0.00496
FLOOR_BAND_HI = 0.00700
#: "flat" = the SAME calibrated relative-slope gate the #315 binding-stall / verdict-trend
#: detectors use (partitions the space consistently across the controller). At the floor
#: the flow has settled at its finest label-persistence level ==> |rel slope| <= this.
FLOOR_FLAT_REL_EPS = DEFAULT_STALL_REL_EPS
#: minimum same-stage rows over which the flat test is meaningful (a slope needs a window).
MIN_FLOOR_WINDOW = 3
#: the label-smooth stages of the flow (coarse->fine persistence order, L3). The floor is
#: only meaningful in a label-smooth objective; a phase stage is already past it.
LABEL_SMOOTH_STAGES = ("CE", "tau", "tau_softplus", "l7", "smooth", "c1a")

#: the phase-tail forces the floor hands off TO (the energy's missing terms). These are
#: the DUTY-TO-MEASURE lever names the DECIDE queue ranks HIGH when the floor is reached
#: (the T1 + #360 build agents register the Lever factories; SENSE ranks them). Kept as a
#: PREFIX/substring set so a lever name the build agents finalize still matches (never-invent:
#: the canonical names are the memo's DSL spec + #360's Force factories).
PHASE_TAIL_LEVERS = (
    "phase_advection_consistency",   # T1 (cross-pair ξ-advected sub-pixel phase)
    "seg_phase_advect",              # T1 argv stem (--seg-phase-advect-*)
    "seg_spike_reweight",            # T2 aleatoric-downweight (#274, BUILT)
    "seg_spike_downweight",
    "seg_coherent_upweight",
    "p0_force",                      # #360's four within-pair forces (Force 1-3 + satisficing hinge)
    "force_area_lagrange",
    "force_screw_consistency",
    "force_subpix",
    "satisficing_hinge",
    "phase_carrier",                 # the terminal Morse-Smale skeleton residual store
)


def _is_phase_tail_lever(name: str) -> bool:
    # normalize the argv (hyphen) and ledger (underscore) forms to one key before matching
    n = str(name or "").lower().replace("-", "_")
    return any(stem in n for stem in PHASE_TAIL_LEVERS)


@dataclass(frozen=True)
class LabelFloorSignal:
    """Whether the LABEL-SMOOTH witness has reached the temporal-majority persistence
    floor (the phase-regime switch event). Every field is MEASURED from the verdict
    rows; NO fabricated ΔS. ``fired()`` is True ONLY for ``LABEL_FLOOR_REACHED``.
    """

    classification: str            # NO_FLOOR | LABEL_FLOOR_REACHED | LABEL_FLOOR_UNIDENTIFIABLE
    stage: str
    stage_is_label_smooth: bool
    n_stage: int
    epoch_latest: int | None
    d_seg_latest: float | None
    oracle_floor: float
    floor_band: tuple[float, float]
    within_floor_band: bool
    rel_slope_per_ep: float        # over the same-stage window (flatness gate)
    flat: bool
    flat_rel_eps: float
    reason: str
    evidence: tuple[str, ...]

    def fired(self) -> bool:
        return self.classification == LABEL_FLOOR_REACHED

    @property
    def phase_regime(self) -> str | None:
        """The regime tag the DECIDE layer keys on (None when not at the floor)."""
        return LABEL_FLOOR_REACHED if self.fired() else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        d["floor_band"] = list(self.floor_band)
        d["phase_regime"] = self.phase_regime
        return d

    def to_confound_alarm_row(self) -> dict:
        """Typed telemetry row (LOUD, matching the trainer L1 ``confound_alarm``
        convention). ADVISORY — this is a healthy regime transition, never a halt
        (CONTAINMENT). Semantically it is a REGIME signal, not a confound; it rides the
        same LOUD channel so a reader/controller cannot miss the hand-off point."""
        return {
            "stage": "regime_signal",
            "alarm": "label_floor_reached" if self.fired() else "label_floor_not_reached",
            "classification": self.classification,
            "seg_form": self.stage,
            "ep": self.epoch_latest,
            "d_seg": self.d_seg_latest,
            "oracle_floor": self.oracle_floor,
            "within_floor_band": self.within_floor_band,
            "rel_slope_per_ep": self.rel_slope_per_ep,
            "flat": self.flat,
            "reason": self.reason,
            "axis": "[macOS-CPU advisory] NON-PROMOTABLE",
        }


def _latest_stage_rows(verdicts: list[dict]) -> tuple[str, list[dict]]:
    rows = [v for v in verdicts
            if isinstance(v.get("d_seg"), (int, float))
            and isinstance(v.get("epoch"), (int, float))]
    if not rows:
        return "", []
    latest_stage = str(rows[-1].get("seg_form", ""))
    return latest_stage, [v for v in rows if str(v.get("seg_form", "")) == latest_stage]


def _rel_slope_per_ep(rows: list[dict]) -> float:
    eps = [float(r["epoch"]) for r in rows]
    ys = [float(r["d_seg"]) for r in rows]
    if len(ys) < 2:
        return 0.0
    fit = slope_with_stderr(eps, ys)
    level = sum(ys) / len(ys)
    return (fit.slope / abs(level)) if abs(level) > 0.0 else 0.0


def _stage_is_label_smooth(stage: str) -> bool:
    s = str(stage or "")
    return any(s == ls or s.startswith(ls) for ls in LABEL_SMOOTH_STAGES)


def label_floor_reached(
    verdicts: list[dict],
    *,
    oracle_floor: float = ORACLE_LABEL_FLOOR_DSEG,
    band: tuple[float, float] = (FLOOR_BAND_LO, FLOOR_BAND_HI),
    flat_rel_eps: float = FLOOR_FLAT_REL_EPS,
    min_window: int = MIN_FLOOR_WINDOW,
) -> LabelFloorSignal:
    """Detect the label-smooth persistence-floor transition from the advisory verdict
    d_seg trajectory (same math as the sibling #315/verdict-trend overlays; unforked).

    Fires ``LABEL_FLOOR_REACHED`` iff ALL hold (all MEASURED, no guess):
      (1) the latest stage is a LABEL-SMOOTH objective (CE/tau/l7/smooth/c1a) — a phase
          stage is already past the floor;
      (2) the latest d_seg sits WITHIN the persistence-floor band [0.00496, 0.00700]
          (the converged-residual band around the oracle 0.005318);
      (3) the same-stage d_seg is FLAT (|rel slope| <= the calibrated gate) — the flow
          has settled at its finest label-persistence level.
    """
    stage, rows = _latest_stage_rows(verdicts)
    if not rows:
        return LabelFloorSignal(
            classification=LABEL_FLOOR_UNIDENTIFIABLE, stage="", stage_is_label_smooth=False,
            n_stage=0, epoch_latest=None, d_seg_latest=None, oracle_floor=oracle_floor,
            floor_band=band, within_floor_band=False, rel_slope_per_ep=0.0, flat=False,
            flat_rel_eps=flat_rel_eps, reason="no same-stage d_seg verdict rows",
            evidence=("label-floor detector: empty verdict series",))

    d_seg_latest = float(rows[-1]["d_seg"])
    epoch_latest = int(rows[-1]["epoch"])
    is_label = _stage_is_label_smooth(stage)
    lo, hi = float(band[0]), float(band[1])
    within = lo <= d_seg_latest <= hi
    rel_slope = _rel_slope_per_ep(rows)
    flat = abs(rel_slope) <= float(flat_rel_eps)

    if len(rows) < int(min_window):
        return LabelFloorSignal(
            classification=LABEL_FLOOR_UNIDENTIFIABLE, stage=stage,
            stage_is_label_smooth=is_label, n_stage=len(rows), epoch_latest=epoch_latest,
            d_seg_latest=d_seg_latest, oracle_floor=oracle_floor, floor_band=band,
            within_floor_band=within, rel_slope_per_ep=rel_slope, flat=flat,
            flat_rel_eps=flat_rel_eps,
            reason=f"only {len(rows)} same-stage rows (need >= {min_window} for a flat test)",
            evidence=(f"stage={stage} n={len(rows)} d_seg={d_seg_latest:.5f}",))

    fired = bool(is_label and within and flat)
    if fired:
        cls = LABEL_FLOOR_REACHED
        reason = (
            f"LABEL-SMOOTH persistence floor reached: d_seg {d_seg_latest:.5f} in band "
            f"[{lo:.5f},{hi:.5f}] around the temporal-majority oracle {oracle_floor:.5f}, "
            f"FLAT (|rel slope| {abs(rel_slope):.2e} <= {flat_rel_eps:.0e}) in label stage "
            f"'{stage}'. Label descent is EXHAUSTED — the remaining 4.5-7x descent to the "
            f"sub-0.15 need band (0.00077-0.00118) is APPEARANCE-PHASE (deterministic spikes), "
            f"pierceable only by the phase-tail forces (T1 + #360). Switch to the phase regime."
        )
    else:
        cls = NO_FLOOR
        why = []
        if not is_label:
            why.append(f"stage '{stage}' not label-smooth (already past the label floor)")
        if not within:
            why.append(f"d_seg {d_seg_latest:.5f} outside floor band [{lo:.5f},{hi:.5f}]")
        if not flat:
            why.append(f"still descending (|rel slope| {abs(rel_slope):.2e} > {flat_rel_eps:.0e})")
        reason = "not at the label floor: " + "; ".join(why)

    return LabelFloorSignal(
        classification=cls, stage=stage, stage_is_label_smooth=is_label, n_stage=len(rows),
        epoch_latest=epoch_latest, d_seg_latest=d_seg_latest, oracle_floor=oracle_floor,
        floor_band=band, within_floor_band=within, rel_slope_per_ep=rel_slope, flat=flat,
        flat_rel_eps=flat_rel_eps, reason=reason,
        evidence=(
            f"stage={stage} label_smooth={is_label} d_seg={d_seg_latest:.5f} "
            f"band=[{lo:.5f},{hi:.5f}] within={within} rel_slope={rel_slope:.2e} flat={flat}",
            "oracle floor = GT stride-2 spike rate 0.005318 (flicker memo §2.4/§2.5; "
            "eq gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1)",
            "existence proof below floor: real-frame content through R d_seg 0.00086 (FEED-ma)",
        ),
    )
