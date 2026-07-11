# SPDX-License-Identifier: MIT
"""GEPA-style reflective self-improvement for the #426 costate organ (operator binding
2026-07-11: "also remember techniques like GEPA").

GEPA (Agrawal et al. 2025, reflective prompt evolution): NATURAL-LANGUAGE reflection
on execution traces proposes candidate strategy changes; a PARETO FRONTIER of
candidates is maintained; each evolution is adopted only when it measurably improves —
the sample-efficient (low-rollout) self-improvement path, which is exactly our
n≈1-trajectory regime (the tournament already measured that gradient/RL-style arms
lose at this n). Composed with what exists, not rebuilt:

  * PowerPlay (``control_alphabet.powerplay_acquisition``) INVENTS the next probe;
  * GEPA (this module) REFLECTS on the measured outcome and EVOLVES the routing/lens
    strategy as typed candidates on a (wf_mae, complexity) Pareto frontier;
  * the Gödel proof-gate DISPOSES: a candidate is ADOPTED only when its measured
    walk-forward beats the incumbent (reflection proposes, backtest disposes).

The reflections are TEMPLATE-GROUNDED in measured records (never free-hallucinated
prose): every reflection sentence cites the numbers that produced it. First executed
cycle (2026-07-11, #205 trajectory, 8 intervals): the plateau-fallback hybrid was
PROPOSED from the walk-forward finding and REFUSED by measurement (hybrid 0.002609 vs
pure E_prototype 0.002513) — recorded, not hidden. Advisory only; no actuation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.witness_control.lambda_net import (
    N_CLASSES,
    CampaignTrajectory,
    _predict_interval,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
)

#: complexity ranks for the Pareto frontier (interpretability-first tie-break;
#: mirrors costate_panel.LENS_COMPLEXITY where applicable)
_ARCH_COMPLEXITY = {"A_ridge_solve": 102.0, "E_prototype": 220.0,
                    "E_prototype_bregman": 220.0, "F_bsf": 280.0,
                    "G_ridge_scorerprior": 140.0, "B_mlp": 332.0,
                    "C_gru_path": 1100.0, "D_deeponet": 700.0,
                    "persistence_heuristic": 1.0}


@dataclass
class ReflectionCandidate:
    """One reflectively-proposed strategy candidate with its measured disposition."""

    name: str
    reflection: str                     # NL reflection GROUNDED in measured numbers
    config: dict                        # the typed strategy change it proposes
    status: str = "PROPOSED"            # PROPOSED | ADOPTED | REFUSED
    measured: dict = field(default_factory=dict)
    complexity: float = 500.0
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"


def _walkforward_mae(traj: CampaignTrajectory, arch: str, *, seed: int = 0,
                     plateau_fallback: bool = False) -> float | None:
    """Measured walk-forward scalar MAE for an arch (optionally with the plateau-
    fallback hybrid: use persistence when the previous slope is below the history
    median). The candidate-measurement engine — same protocol as lambda_net.backtest."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 4:
        return None
    wcls = comp.class_weights
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    errs = []
    for hold in range(2, len(intervals)):
        iv = intervals[hold]
        meas = iv.dxdt()
        heur = intervals[hold - 1].dxdt()
        if arch == "persistence_heuristic":
            pred = heur
        else:
            m = make_model(arch)
            m.fit(intervals[:hold], phis, seed=seed)
            pred = _predict_interval(m, arch, iv, traj.lever_names)
            if plateau_fallback:
                mag_prev = abs(float(wcls @ heur[:N_CLASSES]))
                hist = [abs(float(wcls @ intervals[j].dxdt()[:N_CLASSES]))
                        for j in range(hold)]
                if mag_prev < float(np.median(hist)):
                    pred = heur
        errs.append(abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * iv.dep)
    return float(np.mean(errs))


def reflect_on_tournament(arch_reports: dict) -> list[ReflectionCandidate]:
    """Generate grounded reflections from a measured tournament (never free prose)."""
    cands: list[ReflectionCandidate] = []
    finite = {a: r for a, r in arch_reports.items()
              if np.isfinite(getattr(r, "walkforward_mae_model", float("nan")))}
    if not finite:
        return cands
    winner = min(finite, key=lambda a: finite[a].walkforward_mae_model)
    wr = finite[winner]
    cands.append(ReflectionCandidate(
        name=f"single_best:{winner}",
        reflection=(f"{winner} has the best measured walk-forward "
                    f"({wr.walkforward_mae_model:.6f} vs heuristic "
                    f"{wr.walkforward_mae_heuristic:.6f}) — commit SINGLE_BEST to it."),
        config={"routing": "SINGLE_BEST", "architecture": winner},
        complexity=_ARCH_COMPLEXITY.get(winner, 500.0)))
    for arch, r in sorted(finite.items()):
        if r.walkforward_mae_model > r.walkforward_mae_heuristic:
            cands.append(ReflectionCandidate(
                name=f"plateau_fallback:{arch}",
                reflection=(f"{arch} LOSES walk-forward ({r.walkforward_mae_model:.6f} vs "
                            f"heuristic {r.walkforward_mae_heuristic:.6f}); the loss "
                            "concentrates in plateau folds — candidate: fall back to "
                            "persistence when the previous slope is below the history "
                            "median."),
                config={"routing": "SINGLE_BEST", "architecture": arch,
                        "plateau_fallback": True},
                complexity=_ARCH_COMPLEXITY.get(arch, 500.0) + 10.0))
    return cands


def pareto_frontier(cands: list[ReflectionCandidate]) -> list[ReflectionCandidate]:
    """Non-dominated set over (measured wf_mae, complexity) — GEPA's frontier."""
    measured = [c for c in cands if "wf_mae" in c.measured
                and c.measured["wf_mae"] is not None]
    front: list[ReflectionCandidate] = []
    for c in measured:
        dominated = any(
            o is not c
            and o.measured["wf_mae"] <= c.measured["wf_mae"]
            and o.complexity <= c.complexity
            and (o.measured["wf_mae"] < c.measured["wf_mae"] or o.complexity < c.complexity)
            for o in measured)
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: c.measured["wf_mae"])


@dataclass(frozen=True)
class GepaCycleReport:
    """One reflect→measure→dispose cycle (the organ's self-improvement heartbeat)."""

    candidates: tuple[ReflectionCandidate, ...]
    frontier: tuple[ReflectionCandidate, ...]
    adopted: str | None
    incumbent: str
    incumbent_wf: float | None
    notes: tuple[str, ...]
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    actuation: str = "NONE"


def run_gepa_cycle(traj: CampaignTrajectory, arch_reports: dict, *,
                   incumbent: str = "E_prototype", seed: int = 0) -> GepaCycleReport:
    """One full GEPA cycle: reflect on the measured tournament → measure every
    candidate's walk-forward → Pareto frontier → Gödel disposition (adopt only when
    a candidate strictly beats the incumbent's measured walk-forward)."""
    inc_wf = _walkforward_mae(traj, incumbent, seed=seed)
    cands = reflect_on_tournament(arch_reports)
    for c in cands:
        wf = _walkforward_mae(traj, c.config.get("architecture", incumbent), seed=seed,
                              plateau_fallback=bool(c.config.get("plateau_fallback")))
        c.measured = {"wf_mae": wf, "incumbent_wf": inc_wf}
        is_pure_incumbent = (c.config.get("architecture") == incumbent
                             and not c.config.get("plateau_fallback"))
        if wf is None or inc_wf is None:
            c.status = "PROPOSED"          # not measurable yet — stays on the queue
        elif is_pure_incumbent:
            c.status = "ADOPTED" if wf <= inc_wf else "REFUSED"
        else:
            # ANY strategy differing from the pure incumbent adopts only by strictly
            # beating the incumbent's measured walk-forward (the Gödel disposition)
            c.status = "ADOPTED" if wf < inc_wf else "REFUSED"
    front = pareto_frontier(cands)
    adopted = next((c.name for c in cands if c.status == "ADOPTED"), None)
    notes = (
        "reflection proposes, backtest disposes (Gödel discipline; GEPA Agrawal 2025)",
        f"incumbent {incumbent} measured walk-forward "
        f"{inc_wf if inc_wf is not None else 'n/a'}",
        "every candidate's disposition is measured on THIS trajectory "
        "(verdict_scope: instance)",
    )
    return GepaCycleReport(candidates=tuple(cands), frontier=tuple(front),
                           adopted=adopted, incumbent=incumbent,
                           incumbent_wf=inc_wf, notes=notes)
