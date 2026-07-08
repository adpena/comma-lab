"""Sensor->start EVENT WIRINGS for the level-set witness trainer (the three v7 transitions).

Operator override 2026-07-08 (verbatim, the CHARTER): *"Build the wirings dumbass that's the whole
point."* A cap-only launch re-ships epoch scripting; the operator OVERRODE the T3 council's
launch-with-caps consensus. The three v7 schedule transitions must FIRE ON SENSORS, with the numeric
``--<x>-start-epoch`` demoted to a fail-safe BACKSTOP CAP that fires ONLY if the event has not fired
by that epoch, emitting a LOUD ``cap_fired_before_event`` telemetry row (a firing cap is
falsification-relevant per S5 — it means the wired sensor never triggered, which the next run must see).

The three transitions + their wired sensors (the wiring-gap list from crucible_v7_authored is the spec):
  1. **Muon entry <- powerlaw_meat exit** of the tau-descent (``tac.witness_control.powerlaw_exit``).
     Carries S2's REV-B positive control: the meat-exit is gated on nucleation-complete (LADDER arms
     done) so an island-birth anneal transient cannot be misread as first-order tau-descent
     exhaustion -> premature Muon before the lane nucleates.
  2. **lane-band start <- lane-class critical-nucleus guard** (#315/#302 per-class nucleus reading,
     generalized to the LANE class: born part_frac>0 AND formed within_flip<=thresh; pi1=w/sigma~=5).
     Carries S3's mitigation: a ``would_fire`` row is emittable EVERY verdict epoch regardless of
     event mode so calibration data accrues even under cap operation (build the highest-value OWED
     wiring CALIBRATED in v7.1, not blind).
  3. **seg-chroma-boundary start <- annulus-plateau** (#333 annulus-convergence telemetry promoted
     from observability to a trigger: a plateau detector on annulus_frac with a dwell window; the
     detector params carry req-T LawRef-or-tagged provenance, no bare constants).

PURE + unit-testable (NO MLX / model / async / wall-clock): the trainer computes the raw sensor
ingredients it already has (the d_seg history trajectory, per-class nucleus stats, the annulus_frac
series) and calls these deciders. **Byte-identity when the ``--<x>-start-event`` flags are absent is
guaranteed by the OFF branch of** :meth:`EventBackstopGate.update` **reducing to the EXACT fixed-epoch
comparison the incumbent used** (``ep >= cap``), emitting NO telemetry.

S4 R1 (role discriminator) lives in the DECLARATION surface, not here:
``tac.witness_dsl.typed_config.ScheduleGovernance.role`` (fires|backstops) +
``tools.schedule_provenance_gate`` — an event names role=fires; a cap declares role=backstops
referencing its event, so a sensor field on a CAP can never be misread as a firing claim.

means != ends: advisory control logic. Only a byte-closed n600 exact row < 0.19110 from
``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the pointer 0.19110.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SENSOR_POWERLAW_MEAT",
    "SENSOR_LANE_NUCLEUS",
    "SENSOR_ANNULUS_PLATEAU",
    "RECOGNISED_START_EVENT_SENSORS",
    "ANNULUS_PLATEAU_REL_EPS",
    "ANNULUS_PLATEAU_DWELL_WINDOWS",
    "ANNULUS_PLATEAU_MIN_EPOCHS",
    "GateStep",
    "EventBackstopGate",
    "muon_gate_state_arrays",
    "muon_gate_restore_from_cfg",
    "ladder_arms_complete",
    "muon_meat_event",
    "lane_nucleus_event",
    "annulus_plateau_event",
    "lane_would_fire_row",
]

# ── the recognised start-event SENSOR NAMES (the value a ``--<x>-start-event`` flag takes). Each is
#    the wired code/telemetry sensor the transition fires on; the paired ``--<x>-start-epoch`` is its
#    backstop cap. These are the sensor NAMES (validated against a whitelist), distinct from the
#    ``--curriculum-*`` sensor FLAGS the schedule-provenance gate recognises in a governance block.
SENSOR_POWERLAW_MEAT = "powerlaw_meat"      # muon <- tau-descent weak-KAM exhaustion (+ REV-B pos.ctrl)
SENSOR_LANE_NUCLEUS = "lane_nucleus"        # lane-band <- lane-class critical nucleus born + formed
SENSOR_ANNULUS_PLATEAU = "annulus_plateau"  # seg-chroma <- annulus_frac plateau (formed boundary)
RECOGNISED_START_EVENT_SENSORS: frozenset[str] = frozenset({
    SENSOR_POWERLAW_MEAT, SENSOR_LANE_NUCLEUS, SENSOR_ANNULUS_PLATEAU,
})

# ── annulus-plateau detector params (req-T value-provenance: TAGGED, not bare). Anchor: #333
#    annulus-convergence telemetry (annulus = ~97% of d_seg is boundary-jitter, not region-miss);
#    the plateau of the within-annulus flip fraction marks a FORMED boundary chroma can then sharpen.
#    rel_eps mirrors the curriculum plateau eps (--curriculum-plateau-rel-eps 1e-4, T3 2026-07-05 C1
#    recalibration: separates a true plateau from a mid-descent slope); dwell/min_epochs are the
#    verdict-cadence dwell the boundary must hold before it is declared formed.
ANNULUS_PLATEAU_REL_EPS = 1e-4        # |relative LS slope| threshold (TAGGED: sister of curriculum-plateau-rel-eps #302 C1)
ANNULUS_PLATEAU_DWELL_WINDOWS = 4     # trailing verdict points in the plateau slope window (TAGGED: sister of curriculum-plateau-windows)
ANNULUS_PLATEAU_MIN_EPOCHS = 150      # min epochs elapsed in the window before a plateau may fire (TAGGED: sister of curriculum-min-stage-epochs)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# The event/backstop gate primitive — ONE per transition. Shared event-vs-fixed-cap logic; the
# per-sensor readers below produce the ``event_fired`` boolean it consumes.
# ════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GateStep:
    """The result of one :meth:`EventBackstopGate.update` call at epoch ``ep``.

    ``start_reached``  the transition's start condition is satisfied at/through this epoch (latches).
    ``just_fired``     the gate transitioned OFF->ON at THIS call (fire the treatment exactly once).
    ``fired_by``       ``"event"`` | ``"cap"`` | ``None`` — how it fired.
    ``telemetry``      a JSON-ready row to print, or ``None``. **None in the byte-identity (event
                       mode OFF) branch** so a flags-absent run emits NOTHING new; a LOUD
                       ``cap_fired_before_event`` row when a backstop cap fires under event mode.
    """

    start_reached: bool
    just_fired: bool
    fired_by: "str | None"
    telemetry: "dict | None"


@dataclass
class EventBackstopGate:
    """One sensor->start transition with a fail-safe backstop cap.

    ``sensor is None``  => EVENT MODE OFF: the gate reduces to the EXACT fixed-epoch comparison the
                           incumbent trainer used (``ep >= cap``), emitting NO telemetry -> the run
                           is byte-identical to the pre-wiring path. This is the binding contract.
    ``sensor`` set      => EVENT MODE ON: fire on the sensor (``event_fired``); the numeric ``cap``
                           is the BACKSTOP that fires ONLY if the event has not fired by ``cap``,
                           emitting a LOUD ``cap_fired_before_event`` row (falsification-relevant).
    ``cap is None`` / ``cap <= 0`` => cap DISABLED (only the event can fire; no backstop).

    Latching: once fired, ``update`` returns ``start_reached=True, just_fired=False, telemetry=None``
    forever after — the caller fires the one-time treatment on ``just_fired``.
    """

    name: str                    # transition label, e.g. "muon" / "lane_band" / "seg_chroma_boundary"
    start_epoch_flag: str        # e.g. "--muon-start-epoch"
    start_event_flag: str        # e.g. "--muon-start-event"
    sensor: "str | None"         # the wired sensor NAME (None => event mode OFF => pure fixed cap)
    cap: "int | None"            # backstop cap epoch (None / <=0 => cap disabled)
    _fired_epoch: "int | None" = field(default=None, repr=False)
    _fired_by: "str | None" = field(default=None, repr=False)

    @property
    def event_mode(self) -> bool:
        return self.sensor is not None

    @property
    def fired(self) -> bool:
        return self._fired_epoch is not None

    @property
    def fired_epoch(self) -> "int | None":
        return self._fired_epoch

    def _cap_hit(self, ep: int) -> bool:
        return self.cap is not None and int(self.cap) > 0 and int(ep) >= int(self.cap)

    def update(self, ep: int, *, event_fired: bool = False) -> GateStep:
        """Advance the gate to epoch ``ep``; return the :class:`GateStep`.

        ``event_fired`` is ignored in event-mode-OFF (the caller need not compute a sensor when the
        wiring is absent). The OFF branch is byte-identical: ``start_reached == (ep >= cap)`` with NO
        telemetry."""
        ep = int(ep)
        if self._fired_epoch is not None:
            return GateStep(True, False, self._fired_by, None)

        if not self.event_mode:
            # ── EVENT MODE OFF: the incumbent fixed-cap comparison, verbatim, no telemetry.
            if self._cap_hit(ep):
                self._fired_epoch, self._fired_by = ep, "cap"
                return GateStep(True, True, "cap", None)
            return GateStep(False, False, None, None)

        # ── EVENT MODE ON.
        if event_fired:
            self._fired_epoch, self._fired_by = ep, "event"
            return GateStep(True, True, "event", {
                "stage": "start_event_fired", "transition": self.name,
                "start_event_flag": self.start_event_flag, "sensor": self.sensor,
                "epoch": ep, "fired_by": "event", "cap": self.cap,
                "note": ("wired sensor fired the transition (the cap "
                         f"{self.start_epoch_flag} {self.cap} was the backstop, not reached)"),
            })
        if self._cap_hit(ep):
            self._fired_epoch, self._fired_by = ep, "cap"
            # LOUD: a firing backstop cap means the wired sensor never triggered by the cap epoch —
            # falsification-relevant (S5). The next run must SEE this, not silently accept the cap.
            return GateStep(True, True, "cap", {
                "stage": "cap_fired_before_event", "transition": self.name,
                "start_event_flag": self.start_event_flag, "sensor": self.sensor,
                "start_epoch_flag": self.start_epoch_flag, "epoch": ep, "cap": self.cap,
                "fired_by": "cap",
                "note": ("FAIL-SAFE BACKSTOP FIRED: the wired sensor "
                         f"{self.sensor!r} did NOT trigger the {self.name} transition by the cap "
                         f"epoch {self.cap}; the transition fired on the fixed-epoch backstop. A "
                         "firing cap is falsification-relevant (S5) — re-calibrate the sensor."),
            })
        return GateStep(False, False, None, None)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Muon-gate resume persistence — mirrors tau_advance.tau_advance_state_arrays / _restore_from_cfg.
# Written ONLY when the gate is in EVENT mode (a sensor is wired) => ZERO new sidecar keys for
# clock/cap muon (event-muon OFF) => byte-identical. MAJOR-1: in event mode the switch fires on its
# SENSOR at an epoch < the backstop cap, so the ACTUAL fire epoch — NOT the cap — is the resume-
# critical fact. A crash between the sensor fire and the cap must resume INTO the finisher at the
# fire epoch (Muon optimizer identity + frozen τ); the cap-only comparison would miss it and restore
# a fresh AdamW against a Muon checkpoint (key mismatch) + trip the frozen-τ maybe_advance assert.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def muon_gate_state_arrays(gate: "EventBackstopGate | None"):
    """Serialize the muon gate's FIRED state to resume-sidecar arrays (``__mg_*`` keys). Returns
    ``{}`` when ``gate is None`` OR the gate is in EVENT MODE OFF (``sensor is None`` — clock/cap
    muon), so a non-event-muon run writes ZERO new keys => the sidecar is byte-identical. In event
    mode persists the actual sensor-fire epoch (``-1`` sentinel when not yet fired) + the fired-by
    label, so a crash-resume reconstructs the finisher-entry decision from the FIRE epoch. Requires
    numpy (the caller has it)."""
    if gate is None or not gate.event_mode:
        return {}
    import numpy as np
    return {
        "__mg_fired_epoch": np.asarray(-1 if gate._fired_epoch is None else int(gate._fired_epoch)),
        "__mg_fired_by": np.asarray("" if gate._fired_by is None else str(gate._fired_by)),
    }


def muon_gate_restore_from_cfg(gate: "EventBackstopGate | None", cfg: dict) -> bool:
    """Restore the muon gate's FIRED state from a resume sidecar cfg (inverse of
    :func:`muon_gate_state_arrays`, parsed through ``_load_resume_state``'s ``a.item()``). Returns
    True iff a FIRED state was restored (event-muon that had already fired at the crash). Returns
    False — leaving the gate fresh/unfired — for a pre-fix sidecar (no ``__mg_*`` keys), clock/cap
    muon, or an event-muon gate that had NOT fired yet (``-1`` sentinel): in every False case the
    resume falls back BYTE-IDENTICALLY to the incumbent cap-based finisher decision."""
    if gate is None or "__mg_fired_epoch" not in cfg:
        return False
    _fe = int(cfg.get("__mg_fired_epoch", -1))
    if _fe < 0:
        return False  # persisted but not yet fired at the crash => stay fresh (re-arm the sensor)
    gate._fired_epoch = _fe
    _fb = cfg.get("__mg_fired_by", "")
    gate._fired_by = str(_fb) if _fb else "event"
    return True


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Per-sensor event readers. Each is PURE (numpy-free numeric / deterministic), so the same measured
# ingredients always give the same boolean — the deterministic-reproducibility spine on the sensors.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def ladder_arms_complete(ep: int, arm_windows: "list[int] | tuple[int, ...]") -> bool:
    """S2 REV-B positive control: are ALL LADDER arms past their scheduled (birth+hold+anneal)
    window at epoch ``ep``? (``arm_windows`` = each arm's ``birth_epochs+hold_epochs+anneal_epochs``,
    the absolute epoch its ``scheduled_radius`` reaches 0.) An empty list (LADDER off) => VACUOUSLY
    complete (no nucleation to wait on) => the positive control never blocks. This is the
    ``nucleation_complete`` input to :func:`muon_meat_event`: while any arm still anneals, an
    island-birth transient d_seg dip must not be read as first-order tau exhaustion."""
    windows = [int(w) for w in (arm_windows or [])]
    return all(int(ep) >= w for w in windows)


def muon_meat_event(trajectory, *, nucleation_complete: bool,
                    horizon_epochs: float = 300.0, meat_floor: float = 1e-4,
                    min_points: int = 8) -> dict:
    """Muon-entry event: the powerlaw_meat exit of the tau-descent, gated on the S2 REV-B positive
    control ``nucleation_complete``.

    ``trajectory`` = a d_seg trajectory (a list of ``(epoch, d_seg)`` pairs — the verdict history —
    or a per-class ``{class: [(epoch, d_seg), ...]}`` dict). Passed straight to
    :func:`tac.witness_control.powerlaw_exit.powerlaw_meat_exit`, whose ``exhausted`` verdict is
    fail-safe (too-few / unfittable points => NOT exhausted; never declare exhaustion on a bad
    measurement — the confound-L3 verdict-clearance discipline).

    ``fired`` = ``meat_exhausted AND nucleation_complete``: the tau-descent's extrapolated remaining
    meat is below ``meat_floor`` for every fittable class AND every LADDER arm has finished nucleating,
    so entering Muon cannot abandon a still-forming lane island. Returns a JSON-ready dict."""
    from tac.witness_control.powerlaw_exit import powerlaw_meat_exit

    verdict = powerlaw_meat_exit(
        trajectory, horizon_epochs=float(horizon_epochs),
        meat_floor=float(meat_floor), min_points=int(min_points))
    meat_exhausted = bool(verdict.get("exhausted", False))
    fired = bool(meat_exhausted and nucleation_complete)
    return {
        "sensor": SENSOR_POWERLAW_MEAT,
        "fired": fired,
        "meat_exhausted": meat_exhausted,
        "nucleation_complete": bool(nucleation_complete),
        "remaining_meat": verdict.get("remaining_meat_estimate"),
        "alpha": verdict.get("alpha"),
        "binding_class": verdict.get("binding_class"),
        "reason": (
            verdict.get("reason") if not meat_exhausted
            else ("meat exhausted AND nucleation complete -> Muon entry"
                  if nucleation_complete else
                  "meat exhausted but LADDER arms still nucleating (REV-B positive control HOLDS "
                  "Muon: an island-birth transient must not be read as tau exhaustion)")),
    }


def lane_nucleus_event(part_frac: float, within_flip: float, *,
                       within_flip_thresh: float, min_part_frac: float = 0.0) -> dict:
    """Lane-band start event: the LANE class's critical-nucleus reading (the #315/#302 per-class
    hand-off predicate, applied to the lane class only).

    Lane is above nucleus iff BORN (``part_frac > min_part_frac`` — nonzero predicted mass; MCF
    cannot nucleate a zero-mass class) AND FORMED (``within_flip <= within_flip_thresh`` — the lane
    partition has settled below the flip threshold; the pi1=w/sigma~=5 knee, within_flip the
    operational proxy). ``fired`` = born AND formed. Returns a JSON-ready dict."""
    born = float(part_frac) > float(min_part_frac)
    formed = float(within_flip) <= float(within_flip_thresh)
    return {
        "sensor": SENSOR_LANE_NUCLEUS,
        "fired": bool(born and formed),
        "born": bool(born), "formed": bool(formed),
        "part_frac": float(part_frac), "within_flip": float(within_flip),
        "within_flip_thresh": float(within_flip_thresh), "min_part_frac": float(min_part_frac),
    }


def lane_would_fire_row(ep: int, ev: dict, *, event_mode: bool) -> dict:
    """The S3 ``would_fire`` telemetry row for the lane-nucleus sensor — emittable EVERY verdict
    epoch regardless of whether the lane-band event mode is ON, so calibration data (the dash-birth
    timing — the HIGHEST-value OWED wiring per S3) accrues even under cap operation. Pure telemetry
    (never read back into training); wraps a :func:`lane_nucleus_event` dict."""
    return {
        "stage": "lane_band_would_fire", "epoch": int(ep),
        "event_mode": bool(event_mode),
        "would_fire": bool(ev.get("fired", False)),
        "born": bool(ev.get("born", False)), "formed": bool(ev.get("formed", False)),
        "part_frac": round(float(ev.get("part_frac", 0.0)), 6),
        "within_flip": round(float(ev.get("within_flip", 0.0)), 6),
        "within_flip_thresh": float(ev.get("within_flip_thresh", 0.0)),
        "sensor": SENSOR_LANE_NUCLEUS,
        "axis": "[macOS-numpy advisory] NON-PROMOTABLE",
        "note": "OBSERVABILITY-ONLY (never read into training); the lane-band<-nucleus wiring's "
                "calibration signal (S3 R-S3: build the OWED wiring CALIBRATED in v7.1, not blind).",
    }


def annulus_plateau_event(series, *, rel_eps: float = ANNULUS_PLATEAU_REL_EPS,
                          dwell_windows: int = ANNULUS_PLATEAU_DWELL_WINDOWS,
                          min_epochs: int = ANNULUS_PLATEAU_MIN_EPOCHS) -> dict:
    """seg-chroma-boundary start event: a PLATEAU detector on the annulus_frac trajectory (#333
    annulus-convergence telemetry promoted to a trigger).

    ``series`` = a list of ``(epoch, annulus_frac)`` verdict points (annulus_frac = the within-annulus
    flip fraction; its plateau marks a FORMED boundary chroma can then sharpen). Fires when, over the
    trailing ``dwell_windows`` points, (a) at least ``min_epochs`` have elapsed in the window (the
    DWELL — the boundary must HOLD, not just momentarily flatten) AND (b) the |relative
    least-squares slope| (slope / mean over the window) is ``<= rel_eps``. The LS-slope-over-mean
    plateau math mirrors the trainer's ``_stage_converged`` so the detector is calibrated on the same
    scale as the curriculum plateau. Fail-safe: fewer than ``dwell_windows`` points => NOT fired.
    Returns a JSON-ready dict."""
    pts = sorted((float(e), float(v)) for e, v in (series or []))
    n = len(pts)
    if n < int(dwell_windows) or int(dwell_windows) < 2:
        return {"sensor": SENSOR_ANNULUS_PLATEAU, "fired": False, "n": n,
                "reason": f"insufficient points ({n} < dwell_windows {dwell_windows})"}
    win = pts[-int(dwell_windows):]
    es = [p[0] for p in win]
    ys = [p[1] for p in win]
    span = es[-1] - es[0]
    # least-squares slope of y vs epoch over the window; rel_slope = slope / mean(|y|).
    m = len(es)
    mean_e = sum(es) / m
    mean_y = sum(ys) / m
    sxx = sum((e - mean_e) ** 2 for e in es)
    sxy = sum((e - mean_e) * (y - mean_y) for e, y in zip(es, ys))
    slope = (sxy / sxx) if sxx > 0 else 0.0
    denom = abs(mean_y) if abs(mean_y) > 1e-12 else 1e-12
    rel_slope = abs(slope) / denom
    dwell_ok = span >= float(min_epochs)
    plateau_ok = rel_slope <= float(rel_eps)
    fired = bool(dwell_ok and plateau_ok)
    return {
        "sensor": SENSOR_ANNULUS_PLATEAU,
        "fired": fired, "n": n,
        "slope": slope, "rel_slope": rel_slope, "mean_annulus_frac": mean_y,
        "span_epochs": span, "dwell_ok": bool(dwell_ok), "plateau_ok": bool(plateau_ok),
        "rel_eps": float(rel_eps), "dwell_windows": int(dwell_windows), "min_epochs": int(min_epochs),
        "reason": ("annulus_frac plateaued over the dwell window -> formed boundary" if fired else
                   f"rel_slope {rel_slope:.2e} > eps {rel_eps:.2e}" if not plateau_ok else
                   f"dwell {span:.0f} < min_epochs {min_epochs}"),
    }
