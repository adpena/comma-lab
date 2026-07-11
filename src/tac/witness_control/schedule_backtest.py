# SPDX-License-Identifier: MIT
"""#430 COHERENT-SCHEDULE BACKTEST — selective intervention vs always-on vs the
hand-scheduled #205 curriculum, replayed through the organ's backtested response model.

Operator GO 2026-07-11: the organ's control output is the JOINT schedule — coordinated
lever-bundles gated on JOINT state (per-class nucleus · boundary-formed · plateau ·
per-class λ), a state-gated cascade (island-birth → boundary-form → τ-sharpen⊕repair →
finish) — NOT independent epoch/dwell floors. Literature backing folded per the
coordinator directive: "Remember When It Matters" (Wu et al., arXiv 2607.08716)
MEASURED that SELECTIVE INTERVENTION (act only when the state calls) beats always-on
injection, passive exposure, and general retrieval (+8.3pp Terminal-Bench / +6.8pp
τ²-Bench) — the hand-schedule's hardcoded epoch/dwell floors are exactly the losing
"always-on/passive" family; the organ's state-triggered bundle is the winning shape.
This module gives that claim a MEASURED backtest ON OUR trajectory instead of borrowing
the paper's number.

HONESTY TIER (stated on every artifact): this is a MODEL-BASED COUNTERFACTUAL replay —
the policies are rolled forward through the response model fitted on the measured #205
intervals (the walk-forward-gated tournament family). The model's own trustworthiness
is quantified alongside (self-replay residual: rolling the MEASURED control through the
model must approximately reproduce the MEASURED trajectory) and the verdict is
per-model (the state-DEPENDENT prototype family + the affine ridge as sensitivity).
It is NOT a live A/B — that remains the gates_owed on the OperatorGoTicket. Advisory:
[macOS advisory] NON-PROMOTABLE, score_claim=False; the organ never launches anything.

CONTAINMENT: read-only sensing + pure numpy replay; the only artifact is the returned
report + the HEAVY ticket (which structurally cannot execute).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tac.witness_control.control_alphabet import (
    OperatorGoTicket,
    compose_schedule_bundle_ticket,
)
from tac.witness_control.lambda_net import (
    N_CLASSES,
    CampaignTrajectory,
    Interval,
    ScoreComposition,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
)

#: the #430 state-gated cascade (mirrors the control_alphabet ticket template; each
#: stage names its GATE — a joint-state condition — and its coordinated BUNDLE)
CASCADE_STAGES: tuple[dict, ...] = (
    {"name": "island_birth",
     "gate": "movable_unborn",
     "bundle": ("island_amplify", "area_constraint", "persistence")},
    {"name": "boundary_form",
     "gate": "islands_born_boundary_forming",
     "bundle": ("seg", "eikonal")},
    {"name": "sharpen_repair",
     "gate": "boundary_formed_not_eroding",
     "bundle": ("chroma_boundary", "lane_edge", "thin_lane", "subpix",
                "margin_saliency")},
    {"name": "finish",
     "gate": "plateau_coupled",
     "bundle": ("weight_entropy",)},
)


@dataclass(frozen=True)
class StateGates:
    """DERIVED-AT-CONFIG gate thresholds, read off the measured trajectory (recorded —
    the value-provenance ladder: every constant carries its derivation)."""

    movable_init: float           # Movable d_seg at trajectory start
    movable_born_frac: float      # born when x[Movable] < frac × init (default 0.5)
    boundary_formed_level: float  # weighted d_seg below the measured median = formed
    plateau_slope: float          # |d(weighted)/dep| below this = plateau
    provenance: str = ""

    def flags(self, x: np.ndarray, slope_w: float, wcls: np.ndarray) -> dict[str, bool]:
        dw = float(wcls @ x[:N_CLASSES])
        movable_unborn = bool(x[3] >= self.movable_born_frac * self.movable_init)
        formed = bool(dw < self.boundary_formed_level)
        eroding = bool(slope_w > 0.0)
        plateau = bool(abs(slope_w) < self.plateau_slope)
        return {
            "movable_unborn": movable_unborn,
            "islands_born_boundary_forming": bool(not movable_unborn and not formed),
            "boundary_formed_not_eroding": bool(formed and not eroding and not plateau),
            "plateau_coupled": bool(formed and plateau),
        }


def derive_state_gates(intervals: list[Interval], comp: ScoreComposition,
                       *, born_frac: float = 0.5,
                       plateau_frac: float = 0.1) -> StateGates:
    """Read the gate thresholds off the measured trajectory (DERIVED, recorded)."""
    wcls = comp.class_weights
    dws = [float(wcls @ iv.x0[:N_CLASSES]) for iv in intervals]
    dws.append(float(wcls @ intervals[-1].x1[:N_CLASSES]))
    slopes = [abs(float(wcls @ iv.dxdt()[:N_CLASSES])) for iv in intervals]
    init_slope = slopes[0] if slopes else 1.0
    return StateGates(
        movable_init=float(intervals[0].x0[3]),
        movable_born_frac=float(born_frac),
        boundary_formed_level=float(np.median(dws)),
        plateau_slope=float(plateau_frac) * float(init_slope),
        provenance=(f"derived from the measured trajectory: movable_init={intervals[0].x0[3]:.4f}, "
                    f"formed=median weighted d_seg {float(np.median(dws)):.4f}, "
                    f"plateau={plateau_frac}×initial slope {init_slope:.6f}"))


def _shift_toward(u: np.ndarray, lam: np.ndarray, bundle_idx: list[int],
                  budget: float) -> np.ndarray:
    """One selective intervention: move ≤``budget`` share mass from the worst-λ
    non-bundle donors onto the best-λ bundle lever (the Hamiltonian vertex of
    ``control_alphabet.hamiltonian_decide``, recipient constrained to the bundle)."""
    if not bundle_idx:
        return u.copy()
    out = u.copy()
    best = min(bundle_idx, key=lambda j: lam[j])
    donors = sorted((j for j in range(len(u)) if j not in bundle_idx),
                    key=lambda j: lam[j], reverse=True)
    remaining = float(budget)
    for d in donors:
        if remaining <= 0 or lam[d] <= lam[best]:
            break
        take = min(out[d], remaining)
        out[d] -= take
        out[best] += take
        remaining -= take
    return out


@dataclass(frozen=True)
class ReplayResult:
    """One policy's replay through the response model."""

    policy: str
    weighted_dseg_series: tuple[float, ...]   # at each verdict epoch (incl. start)
    final_weighted_dseg: float
    dseg_epoch_integral: float                # trapezoid ∫ weighted d_seg d(epoch)
    interventions: tuple[str, ...]            # which stage fired at which interval

    def to_dict(self) -> dict:
        return {"policy": self.policy,
                "weighted_dseg_series": list(self.weighted_dseg_series),
                "final_weighted_dseg": self.final_weighted_dseg,
                "dseg_epoch_integral": self.dseg_epoch_integral,
                "interventions": list(self.interventions)}


def _lambda_at(model, x: np.ndarray, ctx: np.ndarray, path,
               lever_names: tuple[str, ...], grad_s: np.ndarray) -> np.ndarray:
    return np.asarray([float(grad_s @ model.response(x, ctx, lever_features(n), path))
                       for n in lever_names])


def replay_policy(model, intervals: list[Interval], comp: ScoreComposition,
                  lever_names: tuple[str, ...], policy: str,
                  gates: StateGates, *, budget: float = 0.10) -> ReplayResult:
    """Roll the policy forward through the fitted response model over the measured
    interval grid (same epochs, same Δep — only the CONTROL differs).

    Policies: ``hand`` (the measured #205 shares — the hand-schedule), ``selective``
    (hand + state-gated cascade intervention when a gate fires), ``always_on`` (hand +
    the same intervention EVERY interval, gate ignored — 2607.08716's losing baseline),
    ``uniform`` (passive uniform exposure over all levers)."""
    wcls = comp.class_weights
    grad_s = comp.grad_s_wrt_state()
    x = intervals[0].x0.copy()
    series = [float(wcls @ x[:N_CLASSES])]
    interventions: list[str] = []
    prev_slope = -abs(float(wcls @ intervals[0].dxdt()[:N_CLASSES]))  # descending start
    for k, iv in enumerate(intervals):
        u_hand = iv.u_mean.copy()
        if policy == "uniform":
            u = np.full_like(u_hand, 1.0 / max(len(u_hand), 1))
        elif policy == "hand":
            u = u_hand
        else:
            lam = _lambda_at(model, x, iv.ctx, iv.path, lever_names, grad_s)
            flags = gates.flags(x, prev_slope, wcls)
            if policy == "selective":
                stage = next((s for s in CASCADE_STAGES if flags.get(s["gate"])), None)
                bundle_idx = [j for j, n in enumerate(lever_names)
                              if stage is not None and n in stage["bundle"]]
            elif policy == "always_on":
                # inject every interval regardless of state (the losing family per
                # 2607.08716 — a REAL baseline, not a strawman: it gets the SAME
                # λ-ordering + budget + reallocation mechanics; only the gate is
                # removed — recipient = best-λ lever over the whole cascade union,
                # donors = every other lever)
                union = tuple(n for s in CASCADE_STAGES for n in s["bundle"])
                cand = [j for j, n in enumerate(lever_names) if n in union]
                bundle_idx = [min(cand, key=lambda j: lam[j])] if cand else []
                stage = {"name": "always_on"}
            else:
                raise ValueError(f"unknown policy {policy!r}")
            if stage is not None and bundle_idx:
                u = _shift_toward(u_hand, lam, bundle_idx, budget)
                interventions.append(f"iv{k}@ep{iv.ep0:.0f}:{stage['name']}")
            else:
                u = u_hand
        resp = np.stack([model.response(x, iv.ctx, lever_features(n), iv.path)
                         for n in lever_names])
        # A_ridge_solve's base() takes no path arg (mirror lambda_net._predict_interval)
        base = (model.base(x, iv.ctx)
                if getattr(model, "name", "") == "A_ridge_solve"
                else model.base(x, iv.ctx, iv.path))
        dx = base + resp.T @ u
        x_new = x + iv.dep * dx
        x_new[:N_CLASSES] = np.clip(x_new[:N_CLASSES], 0.0, None)  # d_seg ≥ 0
        prev_slope = float(wcls @ ((x_new - x)[:N_CLASSES])) / max(iv.dep, 1e-9)
        x = x_new
        series.append(float(wcls @ x[:N_CLASSES]))
    eps = [intervals[0].ep0] + [iv.ep1 for iv in intervals]
    # manual trapezoid (np.trapezoid is numpy>=2-only; version-proof + trivial)
    integral = float(sum(0.5 * (series[i] + series[i + 1]) * (eps[i + 1] - eps[i])
                         for i in range(len(series) - 1)))
    return ReplayResult(policy=policy, weighted_dseg_series=tuple(series),
                        final_weighted_dseg=float(series[-1]),
                        dseg_epoch_integral=integral,
                        interventions=tuple(interventions))


@dataclass(frozen=True)
class Schedule430Report:
    """The #430 backtest verdict for one response model. Advisory, model-based."""

    model_arch: str
    results: tuple[ReplayResult, ...]
    self_replay_mae: float          # |model(hand-replay) − measured| per verdict, mean
    self_replay_final_gap: float    # gap at the final verdict
    measured_final_weighted_dseg: float
    gates_provenance: str
    selective_beats_hand: bool
    selective_beats_always_on: bool
    tier: str = ("MODEL-BASED COUNTERFACTUAL (fit on all measured intervals; "
                 "trust bounded by the self-replay residual; live A/B owed)")
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    score_claim: bool = False
    promotable: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["results"] = [r.to_dict() for r in self.results]
        return d


def backtest_schedule_430(traj: CampaignTrajectory, *, arch: str = "E_prototype_bregman",
                          seed: int = 0, budget: float = 0.10) -> Schedule430Report:
    """Run the 4-policy replay (hand / selective / always_on / uniform) through the
    named response model + the self-replay trust check."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        raise ValueError(f"need ≥3 intervals; have {len(intervals)}")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    model = make_model(arch)
    model.fit(intervals, phis, seed=seed)
    gates = derive_state_gates(intervals, comp)

    results = tuple(replay_policy(model, intervals, comp, traj.lever_names, pol, gates,
                                  budget=budget)
                    for pol in ("hand", "selective", "always_on", "uniform"))
    by = {r.policy: r for r in results}

    # self-replay trust: the hand replay vs the MEASURED weighted d_seg series
    wcls = comp.class_weights
    measured = [float(wcls @ intervals[0].x0[:N_CLASSES])]
    measured += [float(wcls @ iv.x1[:N_CLASSES]) for iv in intervals]
    hand = by["hand"].weighted_dseg_series
    gaps = [abs(a - b) for a, b in zip(hand, measured, strict=True)]
    return Schedule430Report(
        model_arch=arch, results=results,
        self_replay_mae=float(np.mean(gaps)),
        self_replay_final_gap=float(gaps[-1]),
        measured_final_weighted_dseg=float(measured[-1]),
        gates_provenance=gates.provenance,
        selective_beats_hand=bool(
            by["selective"].dseg_epoch_integral < by["hand"].dseg_epoch_integral),
        selective_beats_always_on=bool(
            by["selective"].dseg_epoch_integral < by["always_on"].dseg_epoch_integral))


def compose_430_ticket(traj: CampaignTrajectory,
                       reports: list[Schedule430Report]) -> OperatorGoTicket:
    """Wrap the coherent schedule in the HEAVY OperatorGoTicket, with the MEASURED
    backtest rows in the justification (the ticket cannot execute — operator-GO only).
    Per-class λ comes from the K arm (per-class heads + v8 reconcile) when it fits."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    model = make_model("K_perclass_v8")
    model.fit(intervals, phis, seed=0)
    grad_s = comp.grad_s_wrt_state()
    x = intervals[-1].x1
    lam = {n: float(grad_s @ model.response(x, intervals[-1].ctx,
                                            lever_features(n), intervals[-1].path))
           for n in traj.lever_names}
    rows = "; ".join(
        f"{r.model_arch}: selective ∫dseg·dep {next(x for x in r.results if x.policy == 'selective').dseg_epoch_integral:.3f} "
        f"vs hand {next(x for x in r.results if x.policy == 'hand').dseg_epoch_integral:.3f} "
        f"vs always-on {next(x for x in r.results if x.policy == 'always_on').dseg_epoch_integral:.3f} "
        f"(self-replay MAE {r.self_replay_mae:.4f})"
        for r in reports)
    return compose_schedule_bundle_ticket(
        lam, {f"stage_done:{s['bundle'][0]}": False for s in CASCADE_STAGES},
        tier="MODEL-BASED-BACKTESTED (2607.08716 selective-intervention shape; "
             "counterfactual replay through the walk-forward-gated response model)",
        backtest_row=f"MEASURED replay rows [{rows}]",
        status="MODEL-BASED-BACKTESTED (replay A/B measured through the response "
               "model; LIVE A/B still owed — the gates_owed)")


__all__ = [
    "CASCADE_STAGES",
    "ReplayResult",
    "Schedule430Report",
    "StateGates",
    "backtest_schedule_430",
    "compose_430_ticket",
    "derive_state_gates",
    "replay_policy",
]
