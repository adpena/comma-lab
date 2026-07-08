"""tac.witness_control.tail_cycles — the TAIL_k warm-restart refinement controller.

The post-Muon refinement stage of the level-set witness schedule (T5 crucible
requirement L; DRAFT_OPTIMAL_STACK_v6 §2.2e / v5 §2.2e; row-9 τ_k law): after the
Muon finisher, run K cycles that mine the power-law tail the exponential-window
finisher exit leaves on the bone. The τ_k law has TWO operating points (both
sealed-config-legal; ``next_tau`` implements the single clamped rule that realizes
both — the ladder math is IDENTICAL, only the entry τ_0 vs τ_end relation differs):

* **DESCENDING-LADDER operating point** (``τ_0 > τ_end``): each cycle re-sharpens τ
  DOWNWARD (halving / live-m_q) and re-warms LR ∝ τ_k, a genuine coarse→fine ladder
  until τ hits the ``τ_end`` / live-m_q floor. This is the general refinement behavior.
* **TURNPIKE-DWELL operating point** (``τ_0 = τ_end``, the SEALED crucible case):
  ``--softmax-temp-end 0.31`` = ``τ_end`` and the anneal holds τ = 0.31 through the
  Muon freeze, so the tail ENTERS at the floor. ``next_tau`` then clamps to ``τ_end``
  EVERY cycle (constant τ*) and LR stays constant → ``tau_halving`` and ``lr_prop_coeff``
  are DEAD KNOBS at this operating point. The tail is a constant-τ* dwell that mines
  purely via the per-cycle meat-exit + the net-ΔS PowerPlay stop + per-cycle ckpt.
  This is derivation-CONSISTENT with the witness-native turnpike ("extra budget extends
  the TAIL at τ*, not the transients"); it is NOT SGDR (τ never resets UP) and there is
  no saw-tooth as-built (SEAL-v7-r1 structure-round R-4, resolved). ``TailController``
  emits a one-time ``tail_turnpike_active`` telemetry note when this case is live
  (dead-knob visibility; the default-off-orphan discipline).

LAWS (all provenance-cited; the sealed crucible values are the DEFAULTS a caller may
override — none is invented here):

* **τ_k law (v6 §row-9):**  ``τ_k = max(τ_{k−1}·halving, τ*_k)`` clamped to ``≥ τ_end``,
  where ``τ*_k = m_q(k)/ln5`` from the LIVE margin field (SC-3 live-m_q) WHEN available,
  else the halving fallback. One rule, both operating points above (``next_tau``).
* **LR ∝ τ_k:**  ``lr_k = lr_ref · coeff · τ_k/τ_ref`` (``lr_for_tau``) — re-warms the
  step in proportion to τ_k; moments are NEVER reset (the trainer sets the optimizer LR
  without re-init). Constant when τ_k is constant (the turnpike operating point).
* **dwell floor = settle = 3/ν = 237 ep** (``settle_window_v1``): a cycle may NOT meat-exit
  before it has run ``dwell_min`` epochs (a shorter cycle measures its own transient —
  the M-S2 confound class).
* **cycle floor / fail-safe cap = settle + 150 = 387.09 ep** (``tail_cycle_floor_v1``):
  a cycle that never meat-exits is force-cut at ``cycle_floor_epochs`` (this is what makes
  ``k_max = floor(budget / cycle_floor)`` the structural bound the sealed budget uses).
* **per-cycle meat exit:**  the weak-KAM power-law tail exit
  (:func:`tac.witness_control.powerlaw_exit.powerlaw_meat_exit`) applied at PER-CYCLE scope
  on the cycle's OWN verdict rows (fail-safe: unfittable ⇒ NOT exhausted).
* **PowerPlay stop:**  stop the whole tail when a completed cycle's marginal ΔS/epoch falls
  below ``stop_marginal_s`` — the DERIVED attribution floor ``s* = ν·forfeit ≈ 6.897e-6 S/ep``
  (``forfeit_matched_exit_v1``; ``stop_marginal_s_lawref()``; SEAL-v7-r1 MAJOR-1). Was a
  HARDCODED 1e-4 (14.5× coarser — it TRUNCATED the exact power-law tail this module exists to
  mine); ``powerplay_variant_ii`` campaign-meta.
* **k_max (req-B):**  a hard net-cycle fail-safe cap so a dead/vacuous exit degrades to a
  capped schedule, never an unbounded run.

READ-ONLY + advisory like the rest of ``witness_control``: this module DECIDES τ/LR/exit
from the recorded verdict trace; the trainer APPLIES the decision. It never mutates a run,
never claims a score. Deterministic: same (config, verdict rows) → same decisions (the
reproducibility spine). The trainer consumes it default-OFF (``--tail-cycles-max 0`` ⇒ no
controller ⇒ byte-identical).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tac.witness_control.powerlaw_exit import powerlaw_meat_exit

if TYPE_CHECKING:
    from tac.witness_dsl.lawref import LawRef

__all__ = [
    "RATE_DENOM_BYTES",
    "STOP_MARGINAL_S_DERIVED",
    "TAIL_CONSTANT_PROVENANCE",
    "TailController",
    "TailCycleConfig",
    "TailStep",
    "lr_for_tau",
    "next_tau",
    "stop_marginal_s_lawref",
    "tail_constant_provenance",
    "tau_star_from_mq",
]

_LN5 = math.log(5.0)
_TURNPIKE_EPS = 1e-9  # τ_0 within this of τ_end ⇒ the constant-τ* turnpike operating point
_S_PER_DSEG = 100.0  # S = 100·d_seg + √(10·d_pose) + rate; d_seg-only S units throughout.
# S = 100·d_seg + √(10·d_pose) + 25·|archive.zip|/37_545_489 (upstream/evaluate.py:92). The TAIL
# stop's rate leg (S1-R1) reads the CODED-bytes delta through this rate term so a byte-INFLATING
# cycle cannot read as a pure win on the d_seg-marginal alone.
_S_PER_RATE = 25.0
RATE_DENOM_BYTES = 37_545_489  # contest rate denominator (upstream original-video bytes)

# ── PowerPlay attribution floor s* = ν·forfeit (SEAL-v7-r1 MAJOR-1; forfeit_matched_exit_v1) ──
# The DERIVED PowerPlay stop floor, replacing the struck HARDCODED 1e-4 (14.5× coarser). The two
# inputs are the campaign's own MEASURED anchors (DRAFT v6 §2.2f/g); the product is the derived
# floor. Named so the dataclass default, the LawRef, and the provenance row are one value by
# construction (no bare-literal drift; req-T value-provenance ladder).
_NU_TAU_SOFTPLUS = 0.012653   # ν(tau_softplus) BINDING per-stage exponential-meat fit (P-CT1)
_FORFEIT_S = 5.450779e-4      # P-CT3 restore-EMA-best forfeit S (DRAFT v6 §2.2f)
STOP_MARGINAL_S_DERIVED = _NU_TAU_SOFTPLUS * _FORFEIT_S  # s* ≈ 6.8968706687e-6 S/ep (→ 6.897e-6)


# ── TAIL constant PROVENANCE (req-T value-provenance ladder; S4-R2 + S1) ──────────────────────────
# Every sealed TAIL constant carries its ladder provenance so it "cannot rot silently" (req T / L22:
# no bare literals). cycle_floor / dwell / stop_marginal_s CITE registered laws (derived_at_config);
# tau_halving has NO closed-form derivation, so it stays HARDCODED-WITH-WAIVER with a real rationale
# (the honest req-T class-4). Machine-readable so the crucible manifest can surface the rows.
TAIL_CONSTANT_PROVENANCE: dict[str, dict] = {
    "cycle_floor_epochs": {
        "value": 387.09, "ladder_class": "derived_at_config", "equation_id": "tail_cycle_floor_v1",
        "rationale": "settle 3/ν + one dwell floor (150) = coeff/ν + tail_extra (P-CT1 settle law).",
    },
    "dwell_min": {
        "value": 237, "ladder_class": "derived_at_config", "equation_id": "settle_window_v1",
        "rationale": "3-e-fold settle window 3/ν @ ν(τ)=0.012653 (P-CT1 exponential-meat settle law).",
    },
    "tau_halving": {
        "value": 0.5, "ladder_class": "hardcoded_waiver", "equation_id": None,
        "rationale": ("Geometric DESCENDING-ladder base: the τ_k law FORM is "
                      "τ_k = max(τ_{k−1}·halving, m_q/ln5) clamped ≥ τ_end; 0.5 = one octave (halve τ) "
                      "per cycle — a DESIGN choice, not a derived law. NOTE: at the SEALED turnpike "
                      "operating point (τ_0 = τ_end = 0.31) this knob is INERT — next_tau clamps to "
                      "τ_end every cycle, so halving is a dead knob (constant-τ* dwell). It governs "
                      "only a DESCENDING entry (τ_0 > τ_end). HARDCODED-WITH-WAIVER: re-derive if a "
                      "live-m_q τ*_k schedule replaces the halving fallback."),
    },
    "stop_marginal_s": {
        "value": STOP_MARGINAL_S_DERIVED, "ladder_class": "derived_at_config",
        "equation_id": "forfeit_matched_exit_v1",
        "rationale": ("PowerPlay attribution floor s* = ν(tau_softplus)·forfeit = "
                      "0.012653·5.450779e-4 ≈ 6.897e-6 S/ep (DRAFT v6 §2.2f/g; forfeit_matched_exit_v1; "
                      "stop_marginal_s_lawref()). DERIVED-AT-CONFIG (was HARDCODED 1e-4, 14.5× coarser "
                      "— it TRUNCATED the exact power-law tail; SEAL-v7-r1 MAJOR-1). S1 rate-aware: the "
                      "d_seg leg minus the coded-bytes delta through the 25·bytes/B rate term. NOTE: "
                      "the seal's s*_tail = ν(muon_fin)·forfeit_tail (ν=0.003289, finer) is a run-2 "
                      "duty-to-measure — no forfeit_tail measured yet; the FORM covers it once it lands."),
    },
}


def tail_constant_provenance() -> dict[str, dict]:
    """The TAIL constants' req-T provenance rows (a defensive copy). Consumed by the crucible manifest
    so every sealed TAIL literal is auditable (no silent literals; S4-R2 auditability contract)."""
    return {k: dict(v) for k, v in TAIL_CONSTANT_PROVENANCE.items()}


def tau_star_from_mq(m_q: float) -> float:
    """τ*_k = m_q/ln5 — the Maslov-bounded τ target from the flip-mass GT-margin quantile
    (``tau*_end = m_q/ln5``; ``tools/witness_tau_mq_confirm.py``)."""
    return float(m_q) / _LN5


def next_tau(prev_tau: float, cfg: "TailCycleConfig", live_mq: float | None = None) -> float:
    """τ_k = max(τ_{k−1}·halving, τ*_k=m_q/ln5) clamped to ≥ τ_end (v6 §row-9).

    ONE rule, two operating points (see the module docstring): with ``τ_{k−1} > τ_end`` it
    DESCENDS (a coarse→fine ladder) until it hits the ``τ_end`` / live-m_q floor; with
    ``τ_{k−1} = τ_end`` (the sealed turnpike case) it clamps to ``τ_end`` every call → a
    constant-τ* dwell (τ never resets UP, so it is NOT SGDR). ``live_mq is None`` (the default
    / SC-3-live-unavailable path) ⇒ the halving fallback ``τ_{k−1}·halving`` clamped to ``τ_end``."""
    halved = float(prev_tau) * float(cfg.tau_halving)
    cand = halved if live_mq is None else max(halved, tau_star_from_mq(live_mq))
    return max(cand, float(cfg.tau_end))


def stop_marginal_s_lawref() -> "LawRef":
    """The DERIVED PowerPlay attribution floor as a compilable LawRef (SEAL-v7-r1 MAJOR-1).

    ``s* = ν(tau_softplus)·forfeit`` via ``forfeit_matched_exit_v1`` (ladder DERIVED-AT-CONFIG;
    both inputs are literal MEASURED anchors config-tagged to the mod32cap/tau_softplus stage per
    P-CT1). ``resolve()``-ing it yields ``STOP_MARGINAL_S_DERIVED`` ≈ 6.897e-6 S/ep — the value the
    dataclass default + the provenance row already carry (one value by construction). Lazy import
    keeps this module light + import-order-proof (mirrors how the resolver lazy-imports evaluators)."""
    from tac.witness_dsl.lawref import (
        LADDER_DERIVED_AT_CONFIG,
        InputRef,
        LawRef,
    )

    _tags = {"schedule": "mod32cap", "stage": "tau_softplus"}
    return LawRef(
        equation_id="forfeit_matched_exit_v1",
        inputs={
            "nu": InputRef.literal(
                _NU_TAU_SOFTPLUS,
                provenance="ν(tau_softplus) BINDING per-stage exponential-meat fit "
                           "(DRAFT v6 §2.2g fold-2 P-CT1)",
                config_tags=_tags,
            ),
            "forfeit": InputRef.literal(
                _FORFEIT_S,
                provenance="P-CT3 restore-EMA-best forfeit S (DRAFT v6 §2.2f, "
                           "VERIFIED full-precision from the trace)",
                config_tags=_tags,
            ),
        },
        ladder_class=LADDER_DERIVED_AT_CONFIG,
    )


def lr_for_tau(tau: float, tau_ref: float, lr_ref: float, coeff: float = 1.0) -> float:
    """LR ∝ τ_k: ``lr = lr_ref · coeff · τ/τ_ref``. Degenerate ``tau_ref ≤ 0`` ⇒ ``lr_ref·coeff``."""
    if float(tau_ref) <= 0.0:
        return float(lr_ref) * float(coeff)
    return float(lr_ref) * float(coeff) * (float(tau) / float(tau_ref))


@dataclass(frozen=True)
class TailCycleConfig:
    """The sealed TAIL laws (crucible defaults; a caller may override any). ``k_max`` is the
    only required field (the req-B fail-safe net-cycle cap = ``--tail-cycles-max``)."""

    k_max: int
    cycle_floor_epochs: float = 387.09   # tail_cycle_floor_v1 (settle 237 + 150 dwell floor)
    dwell_min: int = 237                 # settle_window_v1 (3/ν @ ν(tau)=0.012653)
    tau_halving: float = 0.5             # τ_k = max(τ_{k−1}·halving, τ*_k); HARDCODED-WITH-WAIVER
    #                                      (req-T: TAIL_CONSTANT_PROVENANCE['tau_halving'] — SGDR base)
    tau_end: float = 0.31                # knee floor (tau_end_knee_launch_v1); τ never below this
    lr_prop_coeff: float = 1.0           # LR_k = lr_ref·coeff·τ_k/τ_ref (INERT at the turnpike τ_0=τ_end)
    stop_marginal_s: float = STOP_MARGINAL_S_DERIVED  # PowerPlay stop: cycle NET-ΔS/ep below this
    #                                      (S1-R1: rate-aware when byte_rows given — d_seg marginal MINUS
    #                                      the coded-bytes rate cost). DERIVED s* = ν·forfeit ≈ 6.897e-6
    #                                      (forfeit_matched_exit_v1; was HARDCODED 1e-4; TAIL_CONSTANT_PROVENANCE).
    meat_floor: float = 1e-4             # per-cycle powerlaw_meat exit floor
    meat_horizon: float = 300.0          # per-cycle meat extrapolation horizon
    min_points: int = 8                  # min verdict rows before a meat-exit may fire

    def validate(self) -> list[str]:
        p: list[str] = []
        if self.k_max < 1:
            p.append(f"TailCycleConfig: k_max must be >= 1 (req-B fail-safe cap), got {self.k_max}")
        if self.cycle_floor_epochs <= 0:
            p.append(f"TailCycleConfig: cycle_floor_epochs must be > 0, got {self.cycle_floor_epochs}")
        if self.dwell_min < 0:
            p.append(f"TailCycleConfig: dwell_min must be >= 0, got {self.dwell_min}")
        if self.dwell_min > self.cycle_floor_epochs:
            p.append(f"TailCycleConfig: dwell_min ({self.dwell_min}) > cycle_floor_epochs "
                     f"({self.cycle_floor_epochs}) — a cycle could never satisfy dwell before the cap")
        if not (0.0 < self.tau_halving < 1.0):
            p.append(f"TailCycleConfig: tau_halving must be in (0,1), got {self.tau_halving}")
        if self.tau_end < 0.0:
            p.append(f"TailCycleConfig: tau_end must be >= 0, got {self.tau_end}")
        if self.min_points < 4:
            p.append(f"TailCycleConfig: min_points must be >= 4 (two 3-param tail models), "
                     f"got {self.min_points}")
        return p


@dataclass(frozen=True)
class TailStep:
    """One per-epoch controller decision the trainer APPLIES (τ, LR) + control events."""

    tau: float
    lr: float
    begin_cycle: bool        # this epoch starts a NEW cycle (trainer saves the per-cycle ckpt)
    cycle_k: int             # 1-based cycle index
    stage_tag: str           # e.g. "stageTail1" (the preserved per-stage ckpt tag)
    stop: bool               # PowerPlay / k_max fail-safe: the trainer breaks the loop
    reason: str
    marginal: float = float("nan")  # (S4-R2) the MEASURED net-ΔS/ep NUMERATOR at the decision (the
    #                                 value compared to stop_marginal_s), stamped for post-hoc audit —
    #                                 NaN when no completed-cycle marginal was computed this epoch.


class TailController:
    """Deterministic warm-restart TAIL controller. ``step(ep, rows, live_mq)`` per tail epoch;
    ``rows`` is the recorded verdict trace ``[(epoch, d_seg), ...]`` (decide-on-previous — the
    trainer passes the history observed through the prior verdict, no lookahead)."""

    def __init__(self, cfg: TailCycleConfig, *, tau_ref: float, lr_ref: float, tau0: float):
        problems = cfg.validate()
        if problems:
            raise ValueError("TailCycleConfig invalid: " + "; ".join(problems))
        self.cfg = cfg
        self.tau_ref = float(tau_ref)
        self.lr_ref = float(lr_ref)
        self.tau_k = float(tau0)
        self.lr_k = float(lr_ref)
        self.k = 0
        self.cycle_start_ep: int | None = None
        self._cycle_start_dseg: float | None = None
        self._cycle_start_bytes: float | None = None
        self.stopped = False
        # FIX-2 (SEAL-v7-r1 R-4): the SEALED case enters at τ_0 = τ_end ⇒ next_tau clamps to τ_end
        # every cycle ⇒ a constant-τ* turnpike dwell, so tau_halving + lr_prop_coeff are DEAD KNOBS
        # (no SGDR saw-tooth as-built). Surface it once so a reader/controller never mistakes the
        # dead knobs for a live descending ladder (the default-off-orphan / dead-knob-visibility rule).
        self.is_turnpike = abs(float(tau0) - float(cfg.tau_end)) <= _TURNPIKE_EPS
        self._turnpike_note_emitted = False

    def _tag(self) -> str:
        return f"stageTail{self.k}"

    def turnpike_note(self) -> dict | None:
        """The dead-knob telemetry note for the turnpike operating point, else ``None`` (the
        descending-ladder case has no dead knobs). Pure/read-only — the emitted content."""
        if not self.is_turnpike:
            return None
        return {
            "stage": "tail_turnpike_active",
            "tau0": round(float(self.tau_k), 6),
            "tau_end": round(float(self.cfg.tau_end), 6),
            "dead_knobs": ["tau_halving", "lr_prop_coeff"],
            "note": "τ_0 = τ_end ⇒ constant-τ* turnpike dwell; tau_halving + lr_prop_coeff are DEAD "
                    "KNOBS at this operating point; the tail mines via per-cycle meat-exit + the "
                    "net-ΔS PowerPlay stop (no SGDR saw-tooth as-built; SEAL-v7-r1 R-4)",
        }

    @staticmethod
    def _dseg_at_or_before(ep: int, rows: list[tuple[int, float]]) -> float | None:
        cand = [(e, d) for e, d in rows if e <= ep]
        return cand[-1][1] if cand else None

    def _cycle_exit(self, ep: int, rows: list[tuple[int, float]]) -> tuple[bool, str]:
        """Does the CURRENT cycle exit at ``ep``? cycle_floor cap (hard) OR (dwell satisfied AND
        per-cycle powerlaw_meat exhausted)."""
        length = ep - int(self.cycle_start_ep) + 1
        if length >= self.cfg.cycle_floor_epochs:
            return True, "cycle_floor_cap"
        if length >= self.cfg.dwell_min:
            cyc = [(e, d) for e, d in rows if e >= int(self.cycle_start_ep)]
            if len(cyc) >= self.cfg.min_points:
                v = powerlaw_meat_exit(cyc, horizon_epochs=self.cfg.meat_horizon,
                                       meat_floor=self.cfg.meat_floor, min_points=self.cfg.min_points)
                if bool(v["exhausted"]):
                    return True, "meat_exit"
        return False, ""

    def _cycle_net_marginal(
        self, ep: int, rows: list[tuple[int, float]],
        byte_rows: "list[tuple[int, float]] | None",
    ) -> float | None:
        """The completed cycle's NET-ΔS/epoch marginal (the value the PowerPlay stop compares to
        ``stop_marginal_s``). ``None`` when no start-of-cycle d_seg is known.

        d_seg leg (always): ``100·(d_seg_start − d_seg_end)/len`` (S-units/ep improvement).
        rate leg (S1-R1, only when ``byte_rows`` given): MINUS ``25·(bytes_end − bytes_start)/B/len`` —
        so a cycle that lowers d_seg while INFLATING coded bytes has its net marginal cut by the rate
        cost and cannot read as a pure win. ``byte_rows is None`` ⇒ the d_seg-only marginal (BYTE-
        IDENTICAL to the pre-S1-R1 stop; the OFF/rate-blind path is preserved for the caller that does
        not thread bytes)."""
        d_end = self._dseg_at_or_before(ep, rows)
        if self._cycle_start_dseg is None or d_end is None:
            return None
        length = max(ep - int(self.cycle_start_ep), 1)
        marginal = _S_PER_DSEG * (self._cycle_start_dseg - d_end) / length
        if byte_rows is not None and self._cycle_start_bytes is not None:
            b_end = self._dseg_at_or_before(ep, byte_rows)  # reuse the at-or-before selector for bytes
            if b_end is not None:
                rate_cost = _S_PER_RATE * (float(b_end) - float(self._cycle_start_bytes)) / RATE_DENOM_BYTES
                marginal -= rate_cost / length
        return marginal

    def step(self, ep: int, rows: list[tuple[int, float]], live_mq: float | None = None,
             byte_rows: "list[tuple[int, float]] | None" = None) -> TailStep:
        """One tail-epoch decision. ``rows`` = verdict d_seg history ``[(ep, d_seg)]``; ``byte_rows``
        (S1-R1, optional) = the coded-bytes history ``[(ep, blob_bytes)]`` for the RATE-AWARE stop.
        ``byte_rows is None`` ⇒ the d_seg-marginal-only stop (byte-identical to the pre-S1-R1 path)."""
        rows = sorted((int(e), float(d)) for e, d in rows)
        _brows = sorted((int(e), float(b)) for e, b in byte_rows) if byte_rows is not None else None
        if self.is_turnpike and not self._turnpike_note_emitted:
            self._turnpike_note_emitted = True
            print(json.dumps(self.turnpike_note()))  # one-time dead-knob telemetry (FIX-2)
        if self.stopped:
            return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True, "already stopped")
        begin, reason = False, "continue"
        if self.cycle_start_ep is None:
            begin, reason = True, "first cycle"
        else:
            exit_now, exit_reason = self._cycle_exit(ep, rows)
            if exit_now:
                marginal = self._cycle_net_marginal(ep, rows, _brows)
                if marginal is not None and marginal < self.cfg.stop_marginal_s:
                    self.stopped = True
                    _leg = "net-ΔS (rate-aware)" if _brows is not None else "d_seg-marginal"
                    return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True,
                                    f"powerplay stop ({exit_reason}): {_leg} {marginal:.3e} ΔS/ep < "
                                    f"stop_marginal_s {self.cfg.stop_marginal_s}",
                                    marginal=float(marginal))
                if self.k >= self.cfg.k_max:
                    self.stopped = True
                    return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True,
                                    f"k_max {self.cfg.k_max} reached (req-B fail-safe); {exit_reason}",
                                    marginal=(float(marginal) if marginal is not None else float("nan")))
                begin, reason = True, exit_reason
        if begin:
            self.k += 1
            self.tau_k = next_tau(self.tau_k, self.cfg, live_mq)
            self.lr_k = lr_for_tau(self.tau_k, self.tau_ref, self.lr_ref, self.cfg.lr_prop_coeff)
            self.cycle_start_ep = int(ep)
            self._cycle_start_dseg = self._dseg_at_or_before(ep, rows)
            self._cycle_start_bytes = (self._dseg_at_or_before(ep, _brows)
                                       if _brows is not None else None)
        return TailStep(self.tau_k, self.lr_k, begin, self.k, self._tag(), False, reason)
