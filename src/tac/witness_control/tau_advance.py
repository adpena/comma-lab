"""SELF-PACED τ-ADVANCE: event-driven octave ladder for the level-set witness τ-anneal.

Operator directive 2026-07-08 (verbatim): *"Why is there a fixed number of epochs if our
schedule and curriculum are no longer supposed to be hardcoded like pr95"*. The τ-anneal
denominator (``--anneal-epochs`` clocking ``τ(t)``, and the LR pin's ``t/1000``) is the LAST
clock-hardcoding in the crucible_v7 schedule. This module converts the τ-descent from an
epoch-clock to an EVENT-driven octave ladder per the S6-R4 blind derivation (element 5): *"τ
advances on per-scale RELAXATION, self-triggered, one param at a time"* (Ch.6 §3 critical slowing:
the leading relaxation eigenvalue → 0 near each scale transition, so the continuation must advance
τ ONLY when the current scale's transient has relaxed — a clock cannot slow itself when a specific
scale is still relaxing).

THE LADDER (``--tau-advance-mode event``). τ holds at octave ``k`` — a rung of the GEOMETRIC ladder
``τ_k = start·(end/start)^(k/N)`` (the SAME values the geometric clock passes through at prog=k/N,
so ``event`` reuses the ``clock`` ladder VALUES; only the DWELL is event-driven). It advances to
``k+1`` when the per-band RELAXATION SENSOR fires: a ``powerlaw_meat``-style detector
(:func:`tac.witness_control.powerlaw_exit.powerlaw_meat_exit`) run on the through-R seg-loss
trajectory WITHIN THE CURRENT OCTAVE (dwell-gated: no fire before ``min_dwell`` epochs; thin-data
fail-safe: too-few points ⇒ NOT exhausted ⇒ hold). A per-octave MAX-DWELL cap (the ``FAIL_SAFE_CAP``)
advances anyway if the sensor never fires, emitting a LOUD ``cap_fired_before_event`` row (S5
falsification-relevant, exactly like :class:`tac.witness_control.event_wirings.EventBackstopGate`).

BYTE-IDENTITY (the binding contract). ``--tau-advance-mode clock`` (DEFAULT) reduces
:meth:`TauAdvanceController.tau_for_epoch` to calling the incumbent per-epoch clock function
UNCHANGED (the geometric-by-epoch ``_softmax_temp_for_epoch``) and NEVER advances / emits — so a
``clock``-mode run is byte-identical to the pre-build path. The trainer instantiates a controller
ONLY in event mode (the clock path stays textually the incumbent calls).

COUPLINGS (handled explicitly; the octave FRACTION ``k/N`` is the event-mode analog of clock prog):
  * **β co-anneal** — the hosc profile steepness β↑ rides the same continuation as τ↓ (Ch.4
    Deriv-3, one Γ-limit two coupled continuations). :meth:`hosc_beta_for_epoch` interpolates β on
    the octave fraction (the exact incumbent interp form, prog := k/N).
  * **LR pin** — the derivation-consistent choice is LR ∝ the τ-control's OWN progress (S6/DE): in
    event mode that is the octave fraction, NOT a fixed 1000-epoch clock. :meth:`lr_anneal_fraction`
    returns k/N; warmup stays REAL-epoch (initial optimizer stabilization is genuinely time-based).
  * **unify-τ loss-τ = render-τ** — follows automatically: the trainer couples the unified-L_τ loss
    temperature to ``model.softmax_temp``, which event mode sets from the ladder.
  * **Muon / TAIL ordering (NO double-driver of τ)** — the τ-advance is FROZEN at the Muon switch
    (:meth:`freeze`); post-Muon τ is driven ONLY by the finisher freeze + the TAIL τ_k cycles.
    :meth:`maybe_advance` asserts it is never called while frozen (the structural no-double-driver
    guard). This mirrors the clock freeze (``_anneal_ep = muon_start_epoch`` once ``muon_switched``).

RESUME DETERMINISM (launch-critical). The controller OWNS its current-octave d_seg history (fed via
:meth:`ingest`, decide-on-previous, single-threaded in the main loop) so it does not depend on the
trainer's non-persisted ``history`` list. Every checkpoint persists the full controller state
(rung, octave-start epoch, frozen flag, last-seen epoch, current-octave history, fire log) via
:func:`tau_advance_state_arrays` / :func:`tau_advance_restore_from_cfg`, so a crash-resume
reconstructs the IDENTICAL subsequent τ trajectory bit-faithfully.

RESUME-INVARIANT — ``--verdict-every`` (SEAL v7 R1 confound MINOR-3). The octave DWELL is measured in
LOOP epochs, but the sensor history arrives on the VERDICT cadence (async, decide-on-previous). The
τ trajectory is therefore deterministic and resume-safe ONLY at a FIXED ``--verdict-every`` — a resume
that CHANGES the verdict cadence would re-shape the within-octave point spacing and NOT reproduce the
same fire epochs. ``--verdict-every`` is thus part of the τ-advance determinism contract (alongside
seed + config): keep it constant across a resume. (:meth:`ingest` re-ingests from ``_last_seen_epoch``,
so a same-cadence resume IS bit-faithful; see :func:`test_resume_mid_octave_reproduces_identical_subsequent_tau_sequence`.)

PURE (no MLX / model / async / wall-clock): the same measured ingredients always give the same
decision — the deterministic-reproducibility spine on the sensor. means != ends: advisory control
logic; only a byte-closed n600 ``upstream/evaluate.py`` row < 0.19110 moves the pointer 0.19110.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "TAU_ADVANCE_MODES",
    "SENSOR_PER_BAND_RELAXATION",
    "SENSOR_GRADED_STATE",
    "TAU_OCTAVE_CAP_SLACK",
    "TAU_OCTAVE_MEAT_FLOOR",
    "TAU_OCTAVE_MEAT_HORIZON",
    "TAU_OCTAVE_MEAT_MIN_POINTS",
    "tau_octave_ladder",
    "derive_n_octaves",
    "derive_octave_max_dwell",
    "AdvanceStep",
    "TauAdvanceController",
    "tau_advance_state_arrays",
    "tau_advance_restore_from_cfg",
]

# ── the two modes the trainer flag takes. clock = the incumbent geometric-by-epoch anneal
#    (byte-identical DEFAULT); event = the self-paced octave ladder this module implements.
TAU_ADVANCE_MODES: tuple[str, str] = ("clock", "event")

# ── the wired sensor NAME (the schedule-provenance-gate surface): the τ-advance FIRES on the
#    per-band relaxation of the through-R seg loss within the current octave.
SENSOR_PER_BAND_RELAXATION = "per_band_relaxation"

# ── which graded STATE the relaxation sensor consumes (SEAL v7 R1 confound MINOR-2). The d_seg the
#    sensor ingests is the VERDICT d_seg, which is EMA-SHADOW-graded (the deploy authority — the same
#    state the score is claimed on). This is a DELIBERATE choice: the EMA shadow LAGS the live weights,
#    so an octave is declared relaxed slightly LATE — i.e. the sensor is biased CONSERVATIVE (fire-late,
#    never fire-early). That bias is desirable for a schedule-advance trigger (advancing τ too early is
#    the harmful direction). Stamped onto the advance telemetry (``graded_state``) so the EMA-vs-live
#    axis is AUDITABLE, not implicit. It is score-neutral observability (never read back into training).
SENSOR_GRADED_STATE = "ema_shadow"

# ── FAIL_SAFE_CAP slack (req-T value-provenance: TAGGED, not bare). The per-octave max-dwell cap =
#    the clock-equivalent per-octave dwell (anneal_epochs / N) × this slack: generous enough that a
#    genuinely slow relaxation (critical slowing at a sharp scale) is not cut short, bounded enough
#    that a dead / mis-calibrated sensor cannot stall the descent forever (the EventBackstopGate
#    philosophy: a firing cap is falsification-relevant, not a silent default). Sister of the
#    event_wirings backstop discipline.
TAU_OCTAVE_CAP_SLACK = 2.0

# ── per-octave relaxation-sensor params (TAGGED: sisters of event_wirings.muon_meat_event, which
#    reads the SAME powerlaw_meat exit of the tau-descent). meat_floor = the extrapolated remaining
#    d_seg meat below which the within-octave descent is declared relaxed; horizon/min_points are
#    the fit horizon and the thin-data fail-safe (too-few points ⇒ NOT exhausted ⇒ hold the octave).
#    UNITS NOTE (SEAL v7 R1 deepmath MINOR-5): this meat_floor is in d_seg UNITS (remaining
#    extrapolated d_seg). It shares the bare literal 1e-4 with the TAIL's ``stop_marginal_s`` only by
#    COINCIDENCE — that quantity is in S/ep (a per-epoch score-marginal, derived from ν·forfeit on the
#    TAIL surface). The two are dimensionally distinct and MUST be tuned/derived independently; do not
#    copy a re-tune of one onto the other.
TAU_OCTAVE_MEAT_FLOOR = 1e-4        # TAGGED: sister of muon_meat_event meat_floor (d_seg units)
TAU_OCTAVE_MEAT_HORIZON = 300.0     # TAGGED: sister of muon_meat_event horizon_epochs
TAU_OCTAVE_MEAT_MIN_POINTS = 8      # TAGGED: sister of muon_meat_event min_points


def tau_octave_ladder(start: float, end: float, n_octaves: int) -> tuple[float, ...]:
    """The GEOMETRIC octave ladder ``τ_k = start·(end/start)^(k/N)`` for ``k=0..N`` (``N = n_octaves``).

    These are EXACTLY the values the geometric clock anneal (``_softmax_temp_for_epoch`` with
    ``--tau-anneal-shape geometric``) passes through at ``prog = k/N`` — so event mode reuses the
    clock ladder VALUES; only the per-rung DWELL differs. Endpoints are exact (τ_0 == start,
    τ_N == end). Requires ``start > 0``, ``end > 0``, ``n_octaves >= 1``."""
    if not (start > 0.0 and end > 0.0):
        raise ValueError(f"tau_octave_ladder requires start>0 and end>0 (got start={start}, end={end})")
    n = int(n_octaves)
    if n < 1:
        raise ValueError(f"tau_octave_ladder requires n_octaves>=1 (got {n_octaves})")
    ratio = end / start
    return tuple(float(start * (ratio ** (k / n))) for k in range(n + 1))


def derive_n_octaves(anneal_epochs: int, min_stage_epochs: int) -> int:
    """Default rung count N: tie the ladder resolution to the relaxation-dwell floor so each octave
    gets ~``min_stage_epochs`` of clock-equivalent dwell (the one-continuation-param window; Ch.6 §3
    critical slowing). ``N = max(1, round(anneal_epochs / min_stage_epochs))`` — DERIVED from
    existing flags (``--anneal-epochs`` + ``--curriculum-min-stage-epochs``), no bare literal.
    Example: 1500 / 250 = 6 octaves."""
    _ms = max(1, int(min_stage_epochs))
    return max(1, int(round(int(anneal_epochs) / _ms)))


def derive_octave_max_dwell(anneal_epochs: int, n_octaves: int,
                            slack: float = TAU_OCTAVE_CAP_SLACK) -> int:
    """Default per-octave MAX-DWELL (the FAIL_SAFE_CAP): the clock-equivalent per-octave dwell
    ``ceil(anneal_epochs / N)`` × ``slack``. DERIVED (anneal_epochs, N, tagged slack); no bare
    literal. This is a PER-OCTAVE watchdog (advance-anyway if the sensor never fires), NOT a global
    "floor before Muon" constraint — the controller freezes at the Muon switch regardless of rung
    (matching the clock freeze), so a large cap cannot conflict with the Muon ordering.

    FLAT-CAP CHOICE (SEAL v7 R1 deepmath MINOR-3, deliberate). The cap is the SAME for every rung,
    while critical slowing (Ch.6 §3: the leading relaxation eigenvalue → 0 as τ → small) means the
    LATE (small-τ) octaves genuinely need LONGER relaxation, so a flat cap can in principle fire before
    the hardest scale relaxes. We keep it FLAT on purpose rather than invent a rung-scaled growth law:
    the exact critical-slowing exponent is UNMEASURED here, so any ``cap_k = f(k)`` would be a bare
    guessed constant (a value-provenance-ladder violation and a new unmeasured knob). Instead the
    honest instrument is the LOUD ``cap_fired_before_event`` telemetry row (S5 falsification-relevant):
    if a late octave hits the cap before relaxing, the next run SEES it and re-calibrates the cap (or
    promotes a MEASURED rung-scaling law) from data — never from an a-priori guess."""
    n = max(1, int(n_octaves))
    per = math.ceil(int(anneal_epochs) / n)
    return max(1, int(math.ceil(per * float(slack))))


@dataclass(frozen=True)
class AdvanceStep:
    """The result of one :meth:`TauAdvanceController.maybe_advance` call at epoch ``ep``.

    ``advanced``   the controller moved to a new octave rung at THIS call.
    ``rung``       the current rung AFTER this call (0..N).
    ``tau``        ``ladder[rung]`` (the render/loss τ the trainer sets for this epoch onward).
    ``fired_by``   ``"event"`` | ``"cap"`` | ``None`` — how the advance fired (None if no advance).
    ``telemetry``  a JSON-ready row to print, or ``None`` (None when no advance / clock mode).
    """

    advanced: bool
    rung: int
    tau: float
    fired_by: "str | None"
    telemetry: "dict | None"


@dataclass
class TauAdvanceController:
    """Self-paced τ-advance over a geometric octave ladder (event mode) or the incumbent per-epoch
    clock (clock mode). See the module docstring for the full contract.

    ``mode``            ``"clock"`` (byte-identical; τ from the injected clock fn, never advances) or
                        ``"event"`` (the octave ladder).
    ``ladder``          the geometric τ rungs ``τ_0..τ_N`` (:func:`tau_octave_ladder`).
    ``per_octave_cap``  the FAIL_SAFE_CAP max-dwell epochs per octave (:func:`derive_octave_max_dwell`).
    ``min_dwell``       min epochs in an octave before the relaxation sensor may fire (the
                        one-continuation-param dwell floor; from ``--curriculum-min-stage-epochs``).
    ``sensor``          the relaxation detector (injectable for tests); default powerlaw_meat.
    """

    mode: str
    ladder: tuple[float, ...]
    per_octave_cap: int
    min_dwell: int
    meat_floor: float = TAU_OCTAVE_MEAT_FLOOR
    horizon_epochs: float = TAU_OCTAVE_MEAT_HORIZON
    min_points: int = TAU_OCTAVE_MEAT_MIN_POINTS
    sensor: "callable | None" = None
    # ── STATE (all persisted for bit-faithful resume) ──
    _rung: int = 0
    _octave_start_epoch: "int | None" = None          # epoch the CURRENT octave began (first ingest)
    _last_seen_epoch: int = -1                          # highest verdict epoch already ingested
    _frozen: bool = False                              # set at the Muon switch: no further advance
    _octave_hist: list = field(default_factory=list)   # [(epoch, d_seg)] since the current octave began
    _fire_log: list = field(default_factory=list)      # [(rung_reached, epoch, "event"|"cap")]

    def __post_init__(self) -> None:
        if self.mode not in TAU_ADVANCE_MODES:
            raise ValueError(f"TauAdvanceController mode must be in {TAU_ADVANCE_MODES} (got {self.mode!r})")
        if self.mode == "event":
            if len(self.ladder) < 2:
                raise ValueError("event mode requires a ladder with >=2 rungs (n_octaves>=1)")
            if int(self.per_octave_cap) < 1:
                raise ValueError(f"per_octave_cap must be >= 1 (got {self.per_octave_cap})")

    # ── properties ────────────────────────────────────────────────────────────────────────────
    @property
    def event_mode(self) -> bool:
        return self.mode == "event"

    @property
    def rung(self) -> int:
        return self._rung

    @property
    def n_octaves(self) -> int:
        return len(self.ladder) - 1

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def at_floor(self) -> bool:
        """True iff the ladder has advanced to its last rung (τ == τ_end)."""
        return self._rung >= self.n_octaves

    @property
    def fire_log(self) -> tuple:
        return tuple(self._fire_log)

    def octave_fraction(self) -> float:
        """The current octave progress ``k/N`` in [0, 1] — the event-mode analog of clock prog,
        consumed by the β and LR couplings."""
        return self._rung / max(self.n_octaves, 1)

    # ── τ / coupling readouts ───────────────────────────────────────────────────────────────────
    def tau_for_epoch(self, ep: int, clock_fn, *, muon_switched: bool = False) -> float:
        """The render (and, via the trainer coupling, loss) τ for epoch ``ep``.

        clock mode: ``clock_fn(ep)`` UNCHANGED (byte-identical). event mode: the held rung value
        ``ladder[rung]`` (frozen after the Muon switch, exactly like the clock freeze which holds τ
        at the muon-start value). ``clock_fn`` is the trainer's ``lambda e: _softmax_temp_for_epoch(
        muon_start_epoch if muon_switched else e, args)`` — so the clock branch reproduces the
        incumbent freeze too."""
        if not self.event_mode:
            return float(clock_fn(int(ep)))
        return float(self.ladder[self._rung])

    def hosc_beta_for_epoch(self, hosc_beta_start: float, hosc_beta_end: "float | None", *,
                            activation: str, shape: str = "linear") -> "float | None":
        """Event-mode annealed hosc β on the octave fraction (β↑ co-anneals with τ↓; Ch.4 Deriv-3).

        Returns ``None`` under the SAME contract as the incumbent ``_hosc_beta_for_epoch`` (activation
        != hosc, OR ``hosc_beta_end`` unset, OR end == start) so the caller leaves ``model.hosc_beta``
        untouched. Otherwise interpolates β from ``hosc_beta_start`` (rung 0) to ``hosc_beta_end``
        (rung N) with the incumbent linear/cosine forms, prog := ``octave_fraction()``."""
        if (activation != "hosc" or hosc_beta_end is None or hosc_beta_end == hosc_beta_start):
            return None
        prog = self.octave_fraction()
        if str(shape) == "cosine":
            return float(hosc_beta_end + 0.5 * (hosc_beta_start - hosc_beta_end) * (1.0 + math.cos(math.pi * prog)))
        return float(hosc_beta_start + (hosc_beta_end - hosc_beta_start) * prog)

    def lr_anneal_fraction(self) -> float:
        """The LR-anneal progress in event mode: the octave fraction (LR ∝ the τ-control's OWN
        progress, S6/DE — in event mode that is the octave ladder, NOT a fixed-epoch clock). The
        trainer feeds this in place of ``(ep - warmup) / (lr_anneal_epochs - warmup)`` after warmup;
        warmup itself stays real-epoch (initial optimizer stabilization is genuinely time-based).

        BEHAVIOR CHANGE — STEP FUNCTION (SEAL v7 R1 deepmath MINOR-4, not a bug). Because
        :meth:`octave_fraction` holds ``k/N`` constant WITHIN an octave and jumps at each advance, the
        event-mode LR-anneal fraction is a STEP function, unlike the incumbent clock's smooth per-epoch
        cosine. The LR therefore JUMPS at each octave advance; a jump can transient the Muon moment
        buffers. This is intended (LR rides the τ-control's OWN event progress, S6/DE) and is why run-1
        is recommended in CLOCK mode (smooth cosine) — event-mode step-LR is a run-2+ item."""
        return self.octave_fraction()

    # ── observation + advance ───────────────────────────────────────────────────────────────────
    def ingest(self, history) -> None:
        """Feed NEW verdict points (epoch > ``_last_seen_epoch``) from the trainer's ``history`` list
        into the current octave's own d_seg history. Single-threaded (main-loop, decide-on-previous)
        and idempotent per epoch, so it is deterministic and resume-safe (the controller owns its
        octave history; it does NOT depend on the trainer's non-persisted ``history`` after resume).
        No-op in clock mode / when frozen.

        GRADED STATE (SEAL v7 R1 confound MINOR-2): the ``d_seg`` read from ``history`` is the VERDICT
        d_seg, EMA-SHADOW-graded (deploy authority). The shadow lags live weights ⇒ the sensor is
        biased CONSERVATIVE (relaxation declared slightly late, never early). Deliberate; see
        :data:`SENSOR_GRADED_STATE` (stamped onto the advance telemetry for auditability)."""
        if not self.event_mode or self._frozen:
            return
        for h in (history or []):
            if "d_seg" not in h or "epoch" not in h:
                continue
            e = int(h["epoch"])
            if e <= self._last_seen_epoch:
                continue
            self._octave_hist.append((e, float(h["d_seg"])))
            self._last_seen_epoch = e

    def _relaxation_fired(self) -> "dict | None":
        """Run the per-octave relaxation sensor on the current-octave history. Returns the sensor
        verdict dict (with ``exhausted``), or ``None`` if the sensor cannot yet run (thin-data
        fail-safe / no history)."""
        traj = list(self._octave_hist)
        if len(traj) < int(self.min_points):
            return None
        sensor = self.sensor
        if sensor is None:
            from tac.witness_control.powerlaw_exit import powerlaw_meat_exit
            sensor = powerlaw_meat_exit
        return sensor(traj, horizon_epochs=float(self.horizon_epochs),
                      meat_floor=float(self.meat_floor), min_points=int(self.min_points))

    def maybe_advance(self, ep: int) -> AdvanceStep:
        """Advance the octave ladder at epoch ``ep`` if the current octave has RELAXED (sensor fired
        after ``min_dwell`` dwell) OR the per-octave MAX-DWELL cap is hit (LOUD backstop). Only in
        event mode, only while not frozen (a frozen controller is a NO-DOUBLE-DRIVER assert failure),
        only while not at the floor. Advancing resets the octave window (new octave, cleared history)."""
        ep = int(ep)
        if not self.event_mode:
            return AdvanceStep(False, self._rung, float("nan"), None, None)
        # NO-DOUBLE-DRIVER ordering assert: the Muon finisher / TAIL own τ post-freeze; the ladder
        # must never advance once frozen (structural guard, per the ordering non-negotiable).
        assert not self._frozen, (
            "TauAdvanceController.maybe_advance called while FROZEN — the Muon finisher / TAIL are "
            "the sole τ drivers post-freeze; advancing the ladder here would double-drive τ")
        if self._octave_start_epoch is None:
            self._octave_start_epoch = ep
        if self.at_floor:
            return AdvanceStep(False, self._rung, float(self.ladder[self._rung]), None, None)
        dwell = ep - int(self._octave_start_epoch)
        # (a) the relaxation event (dwell-gated): the octave's through-R descent has exhausted.
        sensor_verdict = self._relaxation_fired() if dwell >= int(self.min_dwell) else None
        event_fired = bool(sensor_verdict and sensor_verdict.get("exhausted", False))
        # (b) the FAIL_SAFE_CAP: advance anyway if the sensor never fired by the max-dwell.
        cap_fired = (not event_fired) and (dwell >= int(self.per_octave_cap))
        if not (event_fired or cap_fired):
            return AdvanceStep(False, self._rung, float(self.ladder[self._rung]), None, None)
        fired_by = "event" if event_fired else "cap"
        prev_rung = self._rung
        self._rung += 1
        self._octave_start_epoch = ep
        self._octave_hist = []                      # new octave: its relaxation is measured afresh
        self._fire_log.append((int(self._rung), ep, fired_by))
        tau = float(self.ladder[self._rung])
        if fired_by == "event":
            telem = {
                "stage": "tau_octave_advance", "fired_by": "event", "epoch": ep,
                "from_rung": int(prev_rung), "to_rung": int(self._rung),
                "tau_from": round(float(self.ladder[prev_rung]), 6), "tau_to": round(tau, 6),
                "dwell_epochs": int(dwell), "sensor": SENSOR_PER_BAND_RELAXATION,
                # (SEAL v7 R1 confound MINOR-2) which graded state the sensor consumed — EMA shadow
                # (conservative-late bias, deliberate). Auditable, never read back into training.
                "graded_state": SENSOR_GRADED_STATE,
                # (SEAL v7 R1 MINOR-1) when the sensor fires EXACTLY at/after the max-dwell boundary
                # the event wins (correct attribution — the sensor DID fire) but the coincidence is
                # S5-suspicious; flag it so the LOUD cap row's absence at this boundary is not read as
                # a clean sensor fire. Additive telemetry (never read back into training).
                "dwell_at_cap": bool(dwell >= int(self.per_octave_cap)),
                "remaining_meat": (sensor_verdict or {}).get("remaining_meat_estimate"),
                "alpha": (sensor_verdict or {}).get("alpha"),
                "note": ("per-band relaxation fired: the within-octave through-R seg descent "
                         "exhausted (powerlaw_meat) -> advance one geometric octave"),
            }
        else:
            telem = {
                # LOUD: the sensor never fired by the max-dwell -> a firing backstop is
                # falsification-relevant (S5); the next run must SEE it, not silently accept it.
                "stage": "cap_fired_before_event", "fired_by": "cap", "epoch": ep,
                "from_rung": int(prev_rung), "to_rung": int(self._rung),
                "tau_from": round(float(self.ladder[prev_rung]), 6), "tau_to": round(tau, 6),
                "dwell_epochs": int(dwell), "per_octave_cap": int(self.per_octave_cap),
                "sensor": SENSOR_PER_BAND_RELAXATION,
                "note": ("FAIL-SAFE BACKSTOP FIRED: the per-band relaxation sensor did NOT declare "
                         f"octave {prev_rung} exhausted within the max-dwell {self.per_octave_cap}; "
                         "the ladder advanced on the cap. Falsification-relevant (S5) — re-calibrate."),
            }
        return AdvanceStep(True, self._rung, tau, fired_by, telem)

    def freeze(self, ep: int) -> dict:
        """FREEZE the ladder at the Muon switch (τ holds at the current rung forever; the finisher +
        TAIL own τ thereafter — the no-double-driver contract). Idempotent. Returns a LOUD-if-not-
        floored telemetry row (the finisher conditions boundary placement against this stationary τ,
        exactly like the clock freeze holds τ at ``_softmax_temp_for_epoch(muon_start_epoch)``)."""
        already = self._frozen
        self._frozen = True
        return {
            "stage": "tau_advance_frozen_at_muon", "epoch": int(ep),
            "rung": int(self._rung), "n_octaves": int(self.n_octaves),
            "tau_frozen": round(float(self.ladder[self._rung]), 6),
            "at_floor": bool(self.at_floor), "already_frozen": bool(already),
            "note": ("τ-advance FROZEN at the Muon switch (finisher conditions boundary placement "
                     "against a stationary τ; TAIL τ_k cycles are the sole post-Muon τ driver). "
                     + ("floored." if self.at_floor
                        else f"NOT at floor (rung {self._rung}/{self.n_octaves}) — event τ-descent "
                             "did not complete before Muon (as clock also freezes mid-descent).")),
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════
# Resume persistence — mirrors the trainer's _cl_state_arrays / _cl_restore_from_cfg pattern. Written
# ONLY when the controller exists (event mode) => ZERO new sidecar keys in clock mode => byte-identical.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def tau_advance_state_arrays(ctrl: "TauAdvanceController | None"):
    """Serialize the controller state to resume-sidecar arrays (``__ta_*`` keys). Returns ``{}`` when
    ``ctrl is None`` (clock mode) so the sidecar is byte-identical. Requires numpy (the caller has it).
    Persists EVERYTHING the forward τ trajectory depends on: rung, octave-start epoch, last-seen
    epoch, frozen flag, the current-octave (epoch, d_seg) history, and the fire log — so a resume
    reconstructs the identical subsequent τ sequence bit-faithfully (the launch-critical contract)."""
    if ctrl is None:
        return {}
    import numpy as np
    hist = list(ctrl._octave_hist)
    fires = list(ctrl._fire_log)
    return {
        "__ta_rung": np.asarray(int(ctrl._rung)),
        "__ta_octave_start": np.asarray(-1 if ctrl._octave_start_epoch is None else int(ctrl._octave_start_epoch)),
        "__ta_last_seen": np.asarray(int(ctrl._last_seen_epoch)),
        "__ta_frozen": np.asarray(1 if ctrl._frozen else 0),
        "__ta_hist_ep": np.asarray([int(e) for e, _ in hist], np.int64),
        "__ta_hist_dseg": np.asarray([float(d) for _, d in hist], np.float64),
        "__ta_fire_rung": np.asarray([int(r) for r, _, _ in fires], np.int64),
        "__ta_fire_epoch": np.asarray([int(e) for _, e, _ in fires], np.int64),
        "__ta_fire_by": np.asarray([str(b) for _, _, b in fires]),
    }


def tau_advance_restore_from_cfg(ctrl: "TauAdvanceController | None", cfg: dict) -> bool:
    """Restore the controller state from a resume sidecar cfg (inverse of :func:`tau_advance_state_arrays`,
    parsed through ``_load_resume_state``'s ``a.item() if a.size==1 else a.tolist()``). Returns True if
    state was restored, False if the sidecar predates the feature / clock mode (the controller then
    starts fresh — deterministic going forward, mirroring the closed-loop cap-fallback honesty)."""
    if ctrl is None or "__ta_rung" not in cfg:
        return False

    def _as_list(x) -> list:
        return x if isinstance(x, list) else [] if x is None else [x]

    ctrl._rung = int(cfg["__ta_rung"])
    _os = int(cfg.get("__ta_octave_start", -1))
    ctrl._octave_start_epoch = None if _os < 0 else _os
    ctrl._last_seen_epoch = int(cfg.get("__ta_last_seen", -1))
    ctrl._frozen = bool(int(cfg.get("__ta_frozen", 0)))
    _hep = _as_list(cfg.get("__ta_hist_ep", []))
    _hds = _as_list(cfg.get("__ta_hist_dseg", []))
    ctrl._octave_hist = [(int(e), float(d)) for e, d in zip(_hep, _hds)]
    _fr = _as_list(cfg.get("__ta_fire_rung", []))
    _fe = _as_list(cfg.get("__ta_fire_epoch", []))
    _fb = _as_list(cfg.get("__ta_fire_by", []))
    ctrl._fire_log = [(int(r), int(e), str(b)) for r, e, b in zip(_fr, _fe, _fb)]
    return True
