# SPDX-License-Identifier: MIT
"""LADDER full island-birth homotopy — the per-class-λ-GATED continuation (task #323).

This is the CONTROL LAW on top of the SPATIAL easers in ``eased_targets`` +
``tac.boundary_math.island_protection.eased_island_masks``. Those give a class-aware eased
target at a FIXED support radius; this module supplies the missing piece the amplify/seed
machinery only partially realized: a per-epoch, per-class support SCHEDULE that

  (1) is GATED by the per-class costate λ_c — support flows to a class ONLY while its
      marginal-ΔS-per-unit-improvement λ_c exceeds a gate (UNIFORM always-on amplification is
      the MEASURED net-negative anti-pattern — T3 islands-treatment symposium
      ``council_t3_symposium_islands_treatment_arm_20260706``: Δd_seg ∝ n_big3 − n_isl, so
      support must be withdrawn from a class the moment its residual is won, else n_big3 ≳ n_isl
      turns net-negative). The gate is FRACTIONAL (a soft λ-band, not a hard switch) so the
      support fades continuously as λ_c approaches the floor;

  (2) rides the LADDER continuation form (CT-2 §6, arXiv bifurcation-control): island birth is a
      controlled fold of the runner-up margin field; the homotopy is numerical continuation in
      the difficulty parameter, whose binding rule is the pseudo-arclength step bound
      Δr ≤ c_arc/‖dθ/dλ‖ (the 1-Lipschitz easing). We expose a continuous scheduled radius and a
      1-Lipschitz stepper so the trainer never hard-switches the eased target;

  (3) MOVABLE arm — dilation-GO: the support radius is ceiling'd by the critical-nucleus RELEASE
      law r*(t) = coeff·σ_eff(t) (LawRef ``critical_nucleus_release_v1``; MEASURED dilation knee
      native 44.6% → +1px 90.0% → +2px 98.3%). As τ anneals, σ_eff shrinks, r* shrinks → the
      dilation homotopy releases protection ON SCHEDULE (§5 Import 2, DERIVED), not by a
      hand-tuned clock;

  (4) LANE arm — curve-prior: the support is grown ALONG the openpilot VP-tangent (the eased mask
      geometry stays on the ~8-dim lane manifold; isotropic dilation of a curve is the measured
      NO-GO). Lane's birth barrier is AREA/MARGIN not nucleus-radius, so its release is the
      λ-gate + schedule (no isotropic nucleus ceiling), with a dash-gate PHASE window.

DISCIPLINE: pure numpy/float, deterministic, per-epoch. The module holds NO training state; the
trainer owns the (tiny) previous-radius state for the 1-Lipschitz stepper and feeds the MEASURED
per-class λ_c (from the #315 per-class verdict sensors / shadow observer rows). Default-unset
byte-identity is a TRAINER guarantee (the lever is default-OFF); this module only computes the
schedule once the lever is engaged.

Class order is the CANONICAL comma10k order (NEVER luma-sort): 0=Road 1=Lane 2=Undrivable
3=Movable 4=MyCar. This module names the two island arms LANE and MOVABLE explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Canonical island-class arm names (self-detected at run time; the index is NOT hardcoded per
# the comma10k-order non-negotiable — the trainer passes the detected lane_cls/movable_cls).
ARM_LANE = "lane"
ARM_MOVABLE = "movable"


def _smoothstep(x: float) -> float:
    """C¹ smoothstep on [0,1] (3x²−2x³); clamps outside. Used for continuous ease + soft gate."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


def perclass_lambda_proxy(d_seg_by_class: float, flip_share_by_class: float) -> float:
    """MEASURED per-class costate proxy λ_c = flip_share_c · d_seg_by_class_c.

    The marginal value of spending support on class c = how much of the residual it CARRIES
    (flip_share) × its class-conditional flip RATE (d_seg_by_class). Both are read straight from
    the per-class verdict (``tac.witness_control.perclass_verdict.per_class_dseg_fields``): as the
    class consolidates BOTH fall → λ_c → 0 → the gate withdraws support. This is the in-trainer
    stand-in for the shadow observer's per-class λ row (#315); the gate law below is agnostic to
    how λ_c is produced, so a richer costate estimate can replace this without touching the gate.
    """
    return max(0.0, float(flip_share_by_class)) * max(0.0, float(d_seg_by_class))


@dataclass(frozen=True)
class LadderArmSpec:
    """One island arm's homotopy schedule + gate.

    r0            : peak support (px). Movable = SDF-dilation radius; lane = along-tangent width.
    birth_epochs  : WINNABLE-HOLD window [1, birth_epochs] over which support is HELD at r0 — the
                    LADDER easy-variant learning phase (the rare class is dilated/eased so it is
                    trivially winnable, and the witness learns to produce it during early plastic CE).
                    The physical 0→r0 ease-IN at run start is supplied by the 1-Lipschitz stepper
                    (``LadderHomotopy.step_radius``), not the schedule — so the eased target is never
                    hard-switched on.
    hold_epochs   : additional epochs to HOLD r0 after ``birth_epochs`` before the transfer anneal.
    anneal_epochs : TRANSFER window over which support anneals r0 → 0 (the eased target sharpens back
                    to the TRUE argmax). Full continuation ends at birth_epochs+hold_epochs+anneal_epochs.
    lambda_gate   : the per-class costate FLOOR. Support flows while λ_c ≥ lambda_gate.
    gate_softness : soft-gate band width as a FRACTION of lambda_gate ∈ (0,1]. The gate multiplier
                    ramps 0→1 over λ_c ∈ [lambda_gate·(1−softness), lambda_gate] (smoothstep). At
                    softness→0 it is the hard step "λ_c > gate"; default 0.5 = continuous (no hard
                    switch, per the LADDER continuation form).
    """

    r0: float = 2.0
    birth_epochs: int = 60
    hold_epochs: int = 0
    anneal_epochs: int = 200
    lambda_gate: float = 0.0
    gate_softness: float = 0.5

    def __post_init__(self) -> None:
        if self.r0 < 0.0:
            raise ValueError(f"r0 must be >= 0, got {self.r0}")
        if self.birth_epochs < 0 or self.hold_epochs < 0 or self.anneal_epochs < 0:
            raise ValueError("birth/hold/anneal epochs must be >= 0")
        if not (0.0 < self.gate_softness <= 1.0):
            raise ValueError(f"gate_softness must be in (0,1], got {self.gate_softness}")
        if self.lambda_gate < 0.0:
            raise ValueError(f"lambda_gate must be >= 0, got {self.lambda_gate}")

    # -- the continuation schedule (difficulty ramp), independent of λ -------------------------
    def scheduled_radius(self, epoch: int) -> float:
        """LADDER continuation schedule: r0 →(hold [1, birth+hold])→ r0 →(anneal)→ 0. Starts EASED
        (support = r0 = the winnable variant, per LADDER "learn the easy variant first"), then anneals
        the assistance back to the true target (support → 0). Continuous, piecewise-smooth in epoch;
        the physical 0→r0 ease-in at run start is the stepper's job. Returns 0 before ep1 and after the
        transfer completes."""
        if epoch < 1 or self.r0 == 0.0:
            return 0.0
        hold_end = self.birth_epochs + self.hold_epochs
        if epoch <= hold_end:
            return self.r0
        a = self.anneal_epochs
        if a > 0 and epoch <= hold_end + a:
            return self.r0 * _smoothstep(1.0 - (epoch - hold_end) / a)
        return 0.0

    # -- the per-class-λ soft gate --------------------------------------------------------------
    def gate_multiplier(self, lambda_c: float) -> float:
        """Fractional support gate ∈ [0,1]: 1 while λ_c ≥ lambda_gate, fading to 0 across the soft
        band [lambda_gate·(1−softness), lambda_gate]. lambda_gate == 0 ⇒ always-open (=1) — the
        UNGATED arm (used by the movable dilation-GO which is sound independent of the probe)."""
        g = self.lambda_gate
        if g <= 0.0:
            return 1.0
        lo = g * (1.0 - self.gate_softness)
        span = g - lo
        if span <= 0.0:
            return 1.0 if lambda_c >= g else 0.0
        return _smoothstep((float(lambda_c) - lo) / span)


@dataclass(frozen=True)
class LadderHomotopy:
    """The full per-class LADDER island-birth homotopy (movable dilation-GO + lane curve-prior).

    ``release_coeff`` + ``sigma_eff_default`` seed the MOVABLE critical-nucleus release ceiling
    r*(t) = release_coeff·σ_eff(t) (LawRef ``critical_nucleus_release_v1``; consumed via
    ``tac.canonical_equations.evaluators``). The trainer feeds the CURRENT σ_eff(t) (interface
    width, shrinking as τ anneals); absent that, ``sigma_eff_default`` is used.

    ``lane_dash_gate`` restricts the LANE arm's active window to the birth+hold+anneal phase
    (the comb/dash phase where dashes exist) rather than an isotropic all-epoch support.

    ``max_step_px`` is the pseudo-arclength / 1-Lipschitz cap on the radius change PER CONTINUATION
    STEP (one ``step_radius`` call = one trainer refresh, every ``--ladder-refresh-every`` epochs; the
    trainer may pass a smaller adaptive step). It bites only at discontinuities (the initial 0→r0
    jump, a gate slam) — the smooth schedule anneal is well under it — turning any jump into a
    controlled continuation rather than a hard switch.
    """

    movable: LadderArmSpec = field(default_factory=LadderArmSpec)
    lane: LadderArmSpec = field(default_factory=lambda: LadderArmSpec(r0=2.0, birth_epochs=80,
                                                                      anneal_epochs=260))
    release_coeff: float = 0.95
    sigma_eff_default: float = 1.5
    lane_dash_gate: bool = True
    max_step_px: float = 1.0

    def __post_init__(self) -> None:
        if self.release_coeff < 0.0:
            raise ValueError("release_coeff must be >= 0")
        if self.sigma_eff_default <= 0.0:
            raise ValueError("sigma_eff_default must be > 0")
        if self.max_step_px <= 0.0:
            raise ValueError("max_step_px must be > 0")

    def _arm(self, arm: str) -> LadderArmSpec:
        if arm == ARM_MOVABLE:
            return self.movable
        if arm == ARM_LANE:
            return self.lane
        raise ValueError(f"unknown arm {arm!r} (expected {ARM_LANE!r} or {ARM_MOVABLE!r})")

    def release_ceiling(self, arm: str, sigma_eff: float | None = None) -> float:
        """MOVABLE: r*(t) = release_coeff·σ_eff(t) via LawRef ``critical_nucleus_release_v1``
        (the dilation-GO release law). LANE: no isotropic nucleus ceiling (barrier is
        area/margin) ⇒ +inf; its release is the schedule + λ-gate + dash-phase."""
        if arm == ARM_LANE:
            return float("inf")
        s = self.sigma_eff_default if sigma_eff is None else float(sigma_eff)
        if s <= 0.0:
            return 0.0
        # Consume the registered evaluator (single source of truth; req-T ladder DERIVED-AT-CONFIG).
        try:
            from tac.canonical_equations.evaluators import resolve_equation_value

            return float(resolve_equation_value(
                "critical_nucleus_release_v1", {"coeff": self.release_coeff, "sigma_eff": s}))
        except Exception:
            # Fail-open to the closed-form (identical arithmetic) — never crash the schedule.
            return self.release_coeff * s

    def support_radius(self, arm: str, epoch: int, lambda_c: float,
                       sigma_eff: float | None = None) -> float:
        """The GATED, ceiling'd, continuous support radius (px) for ``arm`` at ``epoch``.

        support = gate_multiplier(λ_c) · min(scheduled_radius(epoch), release_ceiling).
        LANE additionally zeroes out outside the dash-phase window when ``lane_dash_gate``.
        Returns 0.0 when the class's λ_c is below the (soft) gate — the anti-uniform-amplification
        guarantee: no support to a class whose residual is already won."""
        spec = self._arm(arm)
        sched = spec.scheduled_radius(epoch)
        if arm == ARM_LANE and self.lane_dash_gate:
            # dash-phase = the arm's own birth+hold+anneal window; outside it, no lane support.
            end = spec.birth_epochs + spec.hold_epochs + spec.anneal_epochs
            if epoch < 1 or (end > 0 and epoch > end):
                return 0.0
        ceil = self.release_ceiling(arm, sigma_eff)
        gate = spec.gate_multiplier(lambda_c)
        return gate * min(sched, ceil)

    def step_radius(self, arm: str, epoch: int, lambda_c: float, prev_radius: float,
                    sigma_eff: float | None = None, max_step: float | None = None) -> float:
        """1-Lipschitz continuation step (pseudo-arclength fail-safe): move ``prev_radius`` toward
        ``support_radius(...)`` by at most ``max_step`` px (default ``max_step_px``) PER CALL (one call
        = one trainer refresh). Guarantees the eased target never hard-switches even if λ_c jumps (the
        adiabatic tracking guard)."""
        target = self.support_radius(arm, epoch, lambda_c, sigma_eff)
        cap = self.max_step_px if max_step is None else float(max_step)
        delta = target - float(prev_radius)
        if delta > cap:
            return float(prev_radius) + cap
        if delta < -cap:
            return float(prev_radius) - cap
        return target

    def rung(self, arm: str, epoch: int, lambda_c: float, prev_radius: float | None = None,
             sigma_eff: float | None = None) -> int:
        """Integer support rung the trainer's LAZY mask rebuild consumes (px). Rounds the
        (optionally 1-Lipschitz-stepped) continuous radius; ``prev_radius`` None ⇒ ungated snap to
        ``support_radius``. Always >= 0."""
        if prev_radius is None:
            r = self.support_radius(arm, epoch, lambda_c, sigma_eff)
        else:
            r = self.step_radius(arm, epoch, lambda_c, prev_radius, sigma_eff)
        return max(0, int(round(r)))


# ── S2-REV-A: the LADDER↔Muon stagger invariant (T3 v7 council, REV-A) ───────────────────────────
# Feasibility (position_V7_S2 §composition): the three curriculum arms COMPOSE only because they are
# temporally STAGGERED — each LADDER arm's support anneals to 0 at ``birth+hold+anneal`` epochs
# (``scheduled_radius`` returns 0 after the window, regardless of λ_c), and the Muon finisher /
# TAIL fire strictly AFTER. If a future config lengthened an arm's anneal past ``muon_start`` it would
# silently create a LIVE-LADDER-during-Muon (and hence during-TAIL) overlap — the codim-2-at-the-floor
# / TAIL×LADDER-resurrection risk REV-A exists to extinct. This is the ONE structural guard the
# feasibility seat binds; it is not a bug today (max arm window 260 ≪ muon cap 726) but a guard so the
# stagger cannot rot silently. The Muon-entry EVENT wiring (S2 REV-B) additionally gates the sensor on
# nucleation-complete (all arms past their window), so checking against the fixed backstop CAP is the
# sound NECESSARY condition on BOTH the cap and event-armed epoch domains.
LADDER_LAMBDA_GATE_PROVENANCE: dict[str, dict] = {
    "--ladder-movable-lambda-gate": {
        "value": 0.0, "ladder_class": "derived_at_config",
        "rationale": ("λ-gate 0.0 = OPEN (always-on) for the movable dilation-GO arm: the "
                      "critical-nucleus RELEASE law r*(t)=coeff·σ_eff is sound INDEPENDENT of the "
                      "class's lane-share, so the arm needs no costate floor (self-documenting "
                      "falling rule: 'λ never binds; births are release-law-driven'). Sibling of "
                      "the LawRef'd release_coeff/sigma_eff (DERIVED-AT-CONFIG)."),
    },
    "--ladder-lane-lambda-gate": {
        "value": 0.0, "ladder_class": "derived_at_config",
        "rationale": ("λ-gate 0.0 = OPEN for the lane curve-prior arm at launch: the lane arm's "
                      "birth barrier is AREA/MARGIN not nucleus-radius, so its release is the "
                      "schedule + dash-phase window; the costate floor is left OPEN so the T3-split "
                      "movable-first policy is expressed by the SCHEDULE, not a hard λ cut. Carries "
                      "the DERIVED/DEFAULTED tag of its release-law siblings, not a bare literal."),
    },
}


def ladder_arm_window(birth_epochs: int, hold_epochs: int, anneal_epochs: int) -> int:
    """The absolute epoch a LADDER arm's ``scheduled_radius`` reaches 0 = birth+hold+anneal
    (single source of truth for the stagger invariant + the S2 REV-B nucleation-complete control)."""
    return int(birth_epochs) + int(hold_epochs) + int(anneal_epochs)


def ladder_muon_stagger_violation(
    *,
    ladder_on: bool,
    lane_window: int,
    movable_window: int,
    muon_start_epoch: "int | None",
) -> "str | None":
    """S2-REV-A stagger invariant: ``max(LADDER arm birth+hold+anneal windows) < muon_start``.

    Returns an error STRING naming the violating window(s) + their arithmetic, or ``None`` when the
    invariant holds OR is N/A (LADDER off, or no Muon finisher). ``muon_start_epoch`` is the fixed
    backstop CAP (``--muon-start-epoch``); on the event-armed domain the Muon event is additionally
    gated on nucleation-complete (all arms past their window, S2 REV-B), so the cap is the sound
    ceiling to check against. Pure + total ⇒ $0 unit-testable; consumed by BOTH the DSL
    ``WitnessProgram.validate`` and the trainer's ``validate_ladder_muon_stagger_config``."""
    if not ladder_on or muon_start_epoch is None:
        return None
    ms = int(muon_start_epoch)
    offenders: list[str] = []
    if int(lane_window) >= ms:
        offenders.append(f"lane arm window {int(lane_window)} (birth+hold+anneal)")
    if int(movable_window) >= ms:
        offenders.append(f"movable arm window {int(movable_window)} (birth+hold+anneal)")
    if not offenders:
        return None
    return (
        "LADDER↔Muon STAGGER VIOLATION (S2-REV-A): "
        + " AND ".join(offenders)
        + f" >= --muon-start-epoch ({ms}). A LADDER arm whose support is still LIVE when the Muon "
        "finisher (and the post-Muon TAIL) fire creates a codim-2-at-the-floor / TAIL×LADDER "
        "resurrection overlap — the three arms COMPOSE only when temporally staggered (each arm's "
        "scheduled_radius reaches 0 at birth+hold+anneal, strictly BEFORE muon_start). Shorten the "
        "arm's anneal window (or raise --muon-start-epoch) so max(arm windows) < muon_start."
    )


def homotopy_from_flags(
    *,
    movable_r0: float = 2.0, movable_birth_epochs: int = 60, movable_hold_epochs: int = 0,
    movable_anneal_epochs: int = 200, movable_lambda_gate: float = 0.0,
    lane_r0: float = 2.0, lane_birth_epochs: int = 80, lane_hold_epochs: int = 0,
    lane_anneal_epochs: int = 260, lane_lambda_gate: float = 0.0,
    gate_softness: float = 0.5, release_coeff: float = 0.95, sigma_eff: float = 1.5,
    lane_dash_gate: bool = True, max_step_px: float = 1.0,
) -> LadderHomotopy:
    """Build a ``LadderHomotopy`` from the flat trainer-flag kwargs (the one place flag names map to
    the arm specs; keeps the trainer wiring a single call). movable gate defaults OPEN (dilation-GO
    is sound independent of lane share); lane gate is the LADDER verifier gate."""
    return LadderHomotopy(
        movable=LadderArmSpec(r0=movable_r0, birth_epochs=movable_birth_epochs,
                              hold_epochs=movable_hold_epochs, anneal_epochs=movable_anneal_epochs,
                              lambda_gate=movable_lambda_gate, gate_softness=gate_softness),
        lane=LadderArmSpec(r0=lane_r0, birth_epochs=lane_birth_epochs,
                           hold_epochs=lane_hold_epochs, anneal_epochs=lane_anneal_epochs,
                           lambda_gate=lane_lambda_gate, gate_softness=gate_softness),
        release_coeff=release_coeff, sigma_eff_default=sigma_eff,
        lane_dash_gate=lane_dash_gate, max_step_px=max_step_px,
    )


__all__ = [
    "ARM_LANE",
    "ARM_MOVABLE",
    "LADDER_LAMBDA_GATE_PROVENANCE",
    "LadderArmSpec",
    "LadderHomotopy",
    "homotopy_from_flags",
    "ladder_arm_window",
    "ladder_muon_stagger_violation",
    "perclass_lambda_proxy",
]
