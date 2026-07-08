"""tac.witness_control.tail_cycles — the TAIL_k warm-restart refinement controller.

The post-Muon warm-restart stage of the level-set witness schedule (T5 crucible
requirement L; DRAFT_OPTIMAL_STACK_v6 §2.2e / v5 §2.2e; row-9 τ_k law): after the
Muon finisher, run K warm-restart cycles that each re-sharpen τ and re-warm the LR,
mining the power-law tail the exponential-window finisher exit leaves on the bone.

LAWS (all provenance-cited; the sealed crucible values are the DEFAULTS a caller may
override — none is invented here):

* **τ_k law (v6 §row-9):**  ``τ_k = max(τ_{k−1}·halving, τ*_k)`` clamped to ``≥ τ_end``,
  where ``τ*_k = m_q(k)/ln5`` from the LIVE margin field (SC-3 live-m_q) WHEN available,
  else the halving fallback. ``next_tau`` implements both branches.
* **LR ∝ τ_k:**  ``lr_k = lr_ref · coeff · τ_k/τ_ref`` (``lr_for_tau``) — the warm-restart
  re-warms the step in proportion to the re-sharpened τ; moments are NEVER reset (the
  trainer sets the optimizer LR without re-init — warm restart).
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
  below ``stop_marginal_s`` (the attribution floor — the cheapest-new-unsolved cycle yields
  nothing measurable, so stop; ``powerplay_variant_ii`` campaign-meta).
* **k_max (req-B):**  a hard net-cycle fail-safe cap so a dead/vacuous exit degrades to a
  capped schedule, never an unbounded run.

READ-ONLY + advisory like the rest of ``witness_control``: this module DECIDES τ/LR/exit
from the recorded verdict trace; the trainer APPLIES the decision. It never mutates a run,
never claims a score. Deterministic: same (config, verdict rows) → same decisions (the
reproducibility spine). The trainer consumes it default-OFF (``--tail-cycles-max 0`` ⇒ no
controller ⇒ byte-identical).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from tac.witness_control.powerlaw_exit import powerlaw_meat_exit

__all__ = [
    "TailCycleConfig",
    "TailStep",
    "TailController",
    "tau_star_from_mq",
    "next_tau",
    "lr_for_tau",
]

_LN5 = math.log(5.0)
_S_PER_DSEG = 100.0  # S = 100·d_seg + √(10·d_pose) + rate; d_seg-only S units throughout.


def tau_star_from_mq(m_q: float) -> float:
    """τ*_k = m_q/ln5 — the Maslov-bounded τ target from the flip-mass GT-margin quantile
    (``tau*_end = m_q/ln5``; ``tools/witness_tau_mq_confirm.py``)."""
    return float(m_q) / _LN5


def next_tau(prev_tau: float, cfg: "TailCycleConfig", live_mq: float | None = None) -> float:
    """τ_k = max(τ_{k−1}·halving, τ*_k=m_q/ln5) clamped to ≥ τ_end (v6 §row-9).

    ``live_mq is None`` (the default / SC-3-live-unavailable path) ⇒ the halving fallback
    ``τ_{k−1}·halving`` clamped to ``τ_end``."""
    halved = float(prev_tau) * float(cfg.tau_halving)
    cand = halved if live_mq is None else max(halved, tau_star_from_mq(live_mq))
    return max(cand, float(cfg.tau_end))


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
    tau_halving: float = 0.5             # τ_k = max(τ_{k−1}·halving, τ*_k)
    tau_end: float = 0.31                # knee floor (tau_end_knee_launch_v1); τ never below this
    lr_prop_coeff: float = 1.0           # LR_k = lr_ref·coeff·τ_k/τ_ref
    stop_marginal_s: float = 1e-4        # PowerPlay stop: cycle marginal ΔS/ep below this
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
        self.stopped = False

    def _tag(self) -> str:
        return f"stageTail{self.k}"

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

    def _powerplay_below_floor(self, ep: int, rows: list[tuple[int, float]]) -> bool:
        """Cycle marginal ΔS/epoch below the attribution floor (nothing left to mine)."""
        d_end = self._dseg_at_or_before(ep, rows)
        if self._cycle_start_dseg is None or d_end is None:
            return False
        length = max(ep - int(self.cycle_start_ep), 1)
        marginal = _S_PER_DSEG * (self._cycle_start_dseg - d_end) / length
        return marginal < self.cfg.stop_marginal_s

    def step(self, ep: int, rows: list[tuple[int, float]], live_mq: float | None = None) -> TailStep:
        rows = sorted((int(e), float(d)) for e, d in rows)
        if self.stopped:
            return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True, "already stopped")
        begin, reason = False, "continue"
        if self.cycle_start_ep is None:
            begin, reason = True, "first cycle"
        else:
            exit_now, exit_reason = self._cycle_exit(ep, rows)
            if exit_now:
                if self._powerplay_below_floor(ep, rows):
                    self.stopped = True
                    return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True,
                                    f"powerplay stop ({exit_reason}): marginal ΔS/ep < "
                                    f"{self.cfg.stop_marginal_s}")
                if self.k >= self.cfg.k_max:
                    self.stopped = True
                    return TailStep(self.tau_k, self.lr_k, False, self.k, self._tag(), True,
                                    f"k_max {self.cfg.k_max} reached (req-B fail-safe); {exit_reason}")
                begin, reason = True, exit_reason
        if begin:
            self.k += 1
            self.tau_k = next_tau(self.tau_k, self.cfg, live_mq)
            self.lr_k = lr_for_tau(self.tau_k, self.tau_ref, self.lr_ref, self.cfg.lr_prop_coeff)
            self.cycle_start_ep = int(ep)
            self._cycle_start_dseg = self._dseg_at_or_before(ep, rows)
        return TailStep(self.tau_k, self.lr_k, begin, self.k, self._tag(), False, reason)
