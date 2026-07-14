# SPDX-License-Identifier: MIT
"""Regime-conditional SELF-DISPATCH for the #426 costate organ (task #436).

Operator 2026-07-11: "the costate organ is smart enough to know what to use and when to
use it, or it should be." This module closes the gap between "or it should be" and "is".

THE GAP IT CLOSES. The organ's arbitration layer (``continual_costate.arbitrate_architecture``
+ the DSL ``RoutingSpec`` SINGLE_BEST) picks ONE globally-best arm. But the closed-form
GP-costate result (``closed_form_gp_costate_posterior_20260711.md``) MEASURED that the
optimal tool is REGIME-CONDITIONAL: ``T_gp_costate_posterior`` is the best arm on the
TRANSIENT folds (its calibrated shrinkage beats persistence walk-forward 2-4×) yet LOSES
to persistence on the PLATEAU (every learned arm does; envelope §3). A global-single-best
arbiter structurally cannot express that. This is the per-STATE dispatcher: classify the
current regime PAST-ONLY, route to the arm whose MEASURED walk-forward skill is best for
that regime, and defer to persistence when history is insufficient (the meta-λ
self-monitor's "know when to use nothing").

THE MEASURED regime→tool RULES (structural priors — the backtest TESTS them, never fits
them on this trajectory's fold outcomes):
  * transient  → ``T_gp_costate_posterior``  (GP memo §4: first arm to beat persistence WF;
                 the mean win is CONCENTRATED in the early transient folds ep50–100)
  * plateau    → ``persistence``              (envelope §3: persistence beats every arm once
                 the trajectory settles; the meta-λ already flags this)
  * uncertain  → ``persistence``              (defer: <2 observed intervals → cannot classify)

INTERPRETABLE-BY-DESIGN (Rudin, hard req). Every dispatch emits an OBSERVATORY row: which
regime was classified, which tool was chosen, WHY (the deciding past-only signal), plus the
cited per-regime WF ranking prior. Opaque dispatch is a fail. The regime signal is the union
of (a) the transient/plateau discriminator (recent observed |d_seg slope| vs its running
median — the SAME rule ``self_monitor`` uses as its plateau component, τ=1.0, not tuned),
(b) the PRISM prototype router's named regime at the latest past state, (c) its routing
entropy (regime ambiguity). ALL computable PAST-ONLY (no fold-outcome leakage — verified in
the backtest below: the classifier sees ``intervals[:hold]`` and never ``intervals[hold]``).

HONEST SCOPE. n=1 trajectory. The regime→tool policy was DERIVED FROM this same trajectory
(the GP memo analyzed these exact folds), so confirming it here is CONSISTENT-with, not
OUT-OF-SAMPLE-of, the motivating data; the generalization test (does the transient/plateau
boundary + policy transfer) is owed at ≥2 trajectories. Every number [macOS advisory]
NON-PROMOTABLE, score_claim=false. CPU/numpy only. No actuation surface (advisory).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.witness_control.lambda_net import (
    N_CLASSES,
    CampaignTrajectory,
    Interval,
    ScoreComposition,
    _predict_interval,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
)
from tac.witness_control.router_stability import (
    RouterGateCertificate,
    calibrate_router_forecast,
    certify_fp32_gate,
)

#: the tool a persistence route resolves to (the incumbent heuristic, NOT a lambda_net arm)
PERSISTENCE = "persistence"

#: the MEASURED regime→tool policy (structural prior; the backtest tests it, does not fit it)
DISPATCH_POLICY: dict[str, str] = {
    "transient": "T_gp_costate_posterior",
    "plateau": PERSISTENCE,
    "uncertain": PERSISTENCE,
}

#: the per-regime WF ranking PRIOR — the OBSERVATORY justification (cited from the GP-costate
#: memo §4 + envelope §3, NOT recomputed with look-ahead over the fold this call decides).
PER_REGIME_WF_PRIOR: dict[str, str] = {
    "transient": "T_gp_costate_posterior ≺ persistence "
                 "(GP-costate memo §4: GP calibrated-shrinkage wins the transient 2-4×)",
    "plateau": "persistence ≺ every learned arm "
               "(envelope §3: settled trajectory favors the persistence extrapolation)",
    "uncertain": "persistence (default incumbent; <2 observed intervals ⇒ cannot classify)",
}

#: the arm pool the global-single-best baseline is the strongest member of (all numpy, $0):
#: persistence + the dispatch-reachable GP arm + the seal's WF-winning prototype family.
DEFAULT_ARM_POOL: tuple[str, ...] = (
    PERSISTENCE, "T_gp_costate_posterior", "E_prototype", "E_prototype_bregman", "F_bsf")


# ─────────────────────────────────────────────────────────────────────────────
# regime classification (PAST-ONLY)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RegimeClassification:
    """The past-only regime call + its legible deciding signals (max observability)."""

    regime: str                       # transient | plateau | uncertain
    deciding_signal: str              # the falling-rule readback of WHY
    recent_slope_mag: float           # |class-weighted d_seg slope| of the latest OBSERVED interval
    median_slope_mag: float           # running median over the observed history
    plateau: bool                     # the transient/plateau discriminator bit
    prototype_regime: str             # PRISM named regime at the latest past state
    routing_entropy_nats: float       # prototype mixture entropy (regime ambiguity)
    n_past_intervals: int
    meta_lambda_surprise: bool = False   # the meta-λ self-monitor's model-distrust guard
    surprise_ratio: float = float("nan") # model self-forecast error / persistence error
    gate_certificate: RouterGateCertificate | None = None
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"


#: the meta-λ "surprise" ratio above which the organ distrusts its own model-based λ (the
#: self_monitor component-1 threshold, verbatim: model self-forecast error > 1.5× persistence)
META_LAMBDA_SURPRISE_RATIO = 1.5


def _model_surprise(past: list[Interval], comp: ScoreComposition,
                    lever_names: tuple[str, ...], *, incumbent: str = "E_prototype",
                    seed: int = 0) -> tuple[bool, float]:
    """The meta-λ self-monitor's component-1 (self_monitor.py) computed PAST-ONLY.

    Fit the incumbent interpretable head on all-but-the-latest OBSERVED interval, predict the
    latest OBSERVED interval, compare the class-weighted d_seg-rate error to the persistence
    extrapolation's error. ratio > 1.5 ⇒ the model was SURPRISED by the freshest data (a
    regime-shift / model-is-wrong signal) ⇒ distrust the model-based λ and defer. Needs ≥3
    observed intervals (fit needs ≥2); fewer ⇒ (False, nan) — the signal is unavailable, not
    a distrust. STRICTLY past-only: the fold target is never in ``past``."""
    if len(past) < 3:
        return False, float("nan")
    wcls = comp.class_weights
    phis = np.stack([lever_features(n) for n in lever_names])
    try:
        m = make_model(incumbent)
        m.fit(past[:-1], phis, seed=seed)
        pred = _predict_interval(m, incumbent, past[-1], lever_names)
    except Exception:
        return False, float("nan")
    meas = past[-1].dxdt()
    heur = past[-2].dxdt()
    e_model = abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES])))
    e_heur = abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES])))
    ratio = e_model / max(e_heur, 1e-12)
    return bool(ratio > META_LAMBDA_SURPRISE_RATIO), float(ratio)


def classify_regime(past: list[Interval], comp: ScoreComposition,
                    lever_names: tuple[str, ...], *, seed: int = 0,
                    meta_lambda_guard: bool = True) -> RegimeClassification:
    """Classify the current regime from PAST intervals only (no fold-outcome leakage).

    Discriminator (the self_monitor plateau component, τ=1.0, not tuned): the trajectory is
    on a PLATEAU when the latest OBSERVED class-weighted |d_seg slope| is below its running
    median; else it is in a TRANSIENT (still moving fast — where the GP shrinkage forecaster
    beats persistence). <2 observed intervals ⇒ cannot classify ⇒ uncertain (defer).

    ``meta_lambda_guard`` (the self-monitor governing the defer — task #436): in a nominal
    TRANSIENT, if the meta-λ model-surprise signal fires (the interpretable head just
    mispredicted the freshest observed interval ×1.5 worse than persistence), the organ
    DISTRUSTS its own model and downgrades the regime to 'uncertain' → defer to persistence.
    This is the organ 'knowing when to use nothing'."""
    wcls = comp.class_weights
    mags = [abs(float(wcls @ iv.dxdt()[:N_CLASSES])) for iv in past]
    proto_name, ent = "unclassified", float("inf")
    # PRISM named regime + routing entropy at the latest past state (interpretive context)
    if len(past) >= 1:
        try:
            from tac.witness_control.prototype_router import PrototypeRouterLens
            phis = np.stack([lever_features(n) for n in lever_names])
            lens = PrototypeRouterLens()
            lens.fit(past, phis, seed=seed)
            att = lens.attribute(past[-1].x1, past[-1].ep1, comp.grad_s_wrt_state())
            ent = float(att.mixture_entropy)
            proto_name = att.fired[0][0].split(" [")[0] if att.fired else "unclassified"
        except Exception:
            proto_name, ent = "unclassified", float("inf")
    if len(mags) < 2:
        cert = certify_fp32_gate(
            recent_slope_mag=(mags[-1] if mags else 0.0),
            median_slope_mag=0.0,
            n_past_intervals=len(past),
            surprise_ratio=float("nan"),
            meta_lambda_guard=meta_lambda_guard,
            policy=DISPATCH_POLICY,
            surprise_threshold=META_LAMBDA_SURPRISE_RATIO,
        )
        return RegimeClassification(
            regime=cert.selected_regime,
            deciding_signal=(f"insufficient history ({len(mags)} observed interval(s) < 2) "
                             "→ cannot classify → defer to persistence"),
            recent_slope_mag=(mags[-1] if mags else float("nan")),
            median_slope_mag=float("nan"), plateau=False,
            prototype_regime=proto_name, routing_entropy_nats=ent,
            n_past_intervals=len(past), gate_certificate=cert)
    # The discrete branch itself is canonical NumPy-fp32.  The upstream response
    # estimates may be higher precision; they are explicitly rounded before selection.
    recent = float(np.float32(mags[-1]))
    med = float(np.float32(np.median(np.asarray(mags, dtype=np.float32))))
    slope_cert = certify_fp32_gate(
        recent_slope_mag=recent,
        median_slope_mag=med,
        n_past_intervals=len(past),
        surprise_ratio=float("nan"),
        meta_lambda_guard=False,
        policy=DISPATCH_POLICY,
        surprise_threshold=META_LAMBDA_SURPRISE_RATIO,
    )
    plateau = slope_cert.selected_regime == "plateau"
    regime = slope_cert.selected_regime
    rel = " (razor-thin: recent≈median)" if med > 0 and abs(recent - med) / med < 0.02 else ""
    signal = (f"recent |d_seg slope| {recent:.2e} {'<' if plateau else '≥'} running median "
              f"{med:.2e} over {len(mags)} obs → {regime}{rel}")
    # meta-λ defer governor: distrust the model-based λ when it was just surprised
    surprise_raw, ratio = (_model_surprise(past, comp, lever_names, seed=seed)
                           if (meta_lambda_guard and not plateau)
                           else (False, float("nan")))
    cert = certify_fp32_gate(
        recent_slope_mag=recent,
        median_slope_mag=med,
        n_past_intervals=len(past),
        surprise_ratio=ratio,
        meta_lambda_guard=meta_lambda_guard,
        policy=DISPATCH_POLICY,
        surprise_threshold=META_LAMBDA_SURPRISE_RATIO,
    )
    regime = cert.selected_regime
    surprise = bool(regime == "uncertain" and not plateau and surprise_raw)
    if surprise:
        signal += (f"; BUT meta-λ model-surprise ×{ratio:.1f} > "
                   f"{META_LAMBDA_SURPRISE_RATIO} (self-monitor distrusts the head — the "
                   "freshest observed interval was mispredicted) → defer to persistence")
    return RegimeClassification(
        regime=regime, deciding_signal=signal, recent_slope_mag=recent,
        median_slope_mag=med, plateau=plateau, prototype_regime=proto_name,
        routing_entropy_nats=ent, n_past_intervals=len(past),
        meta_lambda_surprise=surprise, surprise_ratio=ratio,
        gate_certificate=cert)


# ─────────────────────────────────────────────────────────────────────────────
# the dispatch decision (the interpretable observatory row)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DispatchDecision:
    """WHICH tool the organ dispatches for the current regime, WHY (Rudin contract)."""

    classification: RegimeClassification
    tool: str                         # the routed arm name, or PERSISTENCE
    rationale: str                    # the falling-rule readback
    per_regime_wf_ranking: str        # the cited prior justification (no look-ahead)
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    actuation: str = "NONE"

    def explain(self) -> str:
        """The observatory line: regime → tool, with the deciding signal + ranking prior."""
        c = self.classification
        return (f"dispatch@{c.n_past_intervals}iv: regime={c.regime} "
                f"[{c.deciding_signal}] → TOOL={self.tool} "
                f"(prototype={c.prototype_regime}, entropy={c.routing_entropy_nats:.2f} nats; "
                f"per-regime WF prior: {self.per_regime_wf_ranking})")


def dispatch_decision(past: list[Interval], comp: ScoreComposition,
                      lever_names: tuple[str, ...], *, seed: int = 0,
                      meta_lambda_guard: bool = True) -> DispatchDecision:
    """The per-state self-dispatch: classify past-only, route by the measured policy."""
    cls = classify_regime(past, comp, lever_names, seed=seed,
                          meta_lambda_guard=meta_lambda_guard)
    if cls.gate_certificate is None:
        raise RuntimeError("regime classification omitted the fp32 gate certificate")
    tool = cls.gate_certificate.selected_tool
    rationale = (f"regime '{cls.regime}' → policy → {tool}: {cls.deciding_signal}")
    ranking = PER_REGIME_WF_PRIOR[cls.regime]
    if cls.regime == "uncertain" and cls.meta_lambda_surprise:
        ranking = ("persistence (meta-λ surprise defer: the latest observed model error "
                   f"was ×{cls.surprise_ratio:.2f} persistence; no target-fold look-ahead)")
    return DispatchDecision(classification=cls, tool=tool, rationale=rationale,
                            per_regime_wf_ranking=ranking)


def dispatch_for_trajectory(traj: CampaignTrajectory, *, seed: int = 0,
                            meta_lambda_guard: bool = True) -> DispatchDecision:
    """Dispatch for the CURRENT (latest) state — the live decision the organ would make.

    Uses ALL measured intervals as 'past' (the deployment call: forecast the NEXT interval)."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if not intervals:
        raise ValueError("dispatch_for_trajectory: no measured intervals")
    return dispatch_decision(intervals, comp, traj.lever_names, seed=seed,
                             meta_lambda_guard=meta_lambda_guard)


# ─────────────────────────────────────────────────────────────────────────────
# the forecast (route → arm → dx/dep prediction)
# ─────────────────────────────────────────────────────────────────────────────
def _forecast(tool: str, past: list[Interval], target: Interval,
              lever_names: tuple[str, ...], phis: np.ndarray, seed: int) -> np.ndarray:
    """Produce the dx/dep forecast for ``target`` using ``tool`` fit on ``past`` only."""
    if tool == PERSISTENCE:
        return past[-1].dxdt()                       # the incumbent walk-forward extrapolation
    model = make_model(tool)
    model.fit(past, phis, seed=seed)
    return _predict_interval(model, tool, target, lever_names)


# ─────────────────────────────────────────────────────────────────────────────
# THE BACKTEST — the arbiter (does per-STATE dispatch beat picking ONE arm?)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DispatchBacktest:
    """Walk-forward verdict: dispatcher vs persistence vs global-single-best (past-only)."""

    n_folds: int
    dispatcher_wf_mae: float
    dispatcher_wf_mae_no_meta_guard: float   # ablation: dispatcher w/o the meta-λ defer guard
    persistence_wf_mae: float
    global_single_best_arm: str
    global_single_best_wf_mae: float
    per_arm_wf_mae: dict[str, float]
    beats_persistence: bool
    beats_global_single_best: bool
    meta_lambda_guard: bool
    gate_min_boundary_margin_ulps: float
    gate_unstable_fold_count: int
    forecast_calibration: dict
    fold_rows: tuple[dict, ...]
    verdict: str
    notes: tuple[str, ...] = field(default_factory=tuple)
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    score_claim: bool = False
    promotable: bool = False

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fold_rows"] = [dict(r) for r in self.fold_rows]
        d["notes"] = list(self.notes)
        return d


def _standalone_wf(arm: str, intervals: list[Interval], comp: ScoreComposition,
                   lever_names: tuple[str, ...], phis: np.ndarray, seed: int) -> list[float]:
    """Per-fold walk-forward errors for ONE fixed arm (the global-single-best candidates)."""
    wcls = comp.class_weights
    errs: list[float] = []
    for hold in range(2, len(intervals)):
        pred = _forecast(arm, intervals[:hold], intervals[hold], lever_names, phis, seed)
        meas = intervals[hold].dxdt()
        errs.append(abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * intervals[hold].dep)
    return errs


def backtest_dispatch(traj: CampaignTrajectory, *, seed: int = 0,
                      arm_pool: tuple[str, ...] = DEFAULT_ARM_POOL,
                      meta_lambda_guard: bool = True) -> DispatchBacktest:
    """The honesty gate for the dispatcher (walk-forward, past-only, no look-ahead).

    For each fold ``hold`` (predicting ``intervals[hold]`` from ``intervals[:hold]``):
      1. classify the regime from ``intervals[:hold]`` ONLY (verified past-only — the target
         ``intervals[hold]`` is never read by the classifier);
      2. route to the policy's tool and forecast the slope;
      3. score the class-weighted d_seg-rate error vs the MEASURED slope.
    Compare the dispatcher's mean WF MAE to (a) persistence and (b) the GLOBAL-SINGLE-BEST
    fixed arm (the strongest single member of ``arm_pool``). Beating persistence is the floor;
    beating global-single-best is the real test — if it does NOT, per-state dispatch is
    over-fitting the regime labels (an HONEST NEGATIVE; the regime→tool rules still bank)."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        raise ValueError(f"dispatch backtest needs ≥3 intervals; have {len(intervals)}")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    wcls = comp.class_weights

    # standalone WF for every candidate single arm (the global-single-best baseline)
    per_arm: dict[str, list[float]] = {
        a: _standalone_wf(a, intervals, comp, traj.lever_names, phis, seed) for a in arm_pool}
    per_arm_mae = {a: float(np.mean(v)) for a, v in per_arm.items()}
    gsb_arm = min(per_arm_mae, key=lambda a: per_arm_mae[a])
    gsb_mae = per_arm_mae[gsb_arm]
    persistence_mae = per_arm_mae.get(PERSISTENCE, float("nan"))

    # the dispatcher walk-forward (past-only classify each fold), guarded + ablation
    disp_errs: list[float] = []
    disp_errs_noguard: list[float] = []
    rows: list[dict] = []
    for hold in range(2, len(intervals)):
        past = intervals[:hold]
        target = intervals[hold]
        meas = target.dxdt()
        dec = dispatch_decision(past, comp, traj.lever_names, seed=seed,
                                meta_lambda_guard=meta_lambda_guard)
        pred = _forecast(dec.tool, past, target, traj.lever_names, phis, seed)
        err = abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES]))) * target.dep
        disp_errs.append(err)
        # ablation route (no meta-λ guard) — the guard's marginal effect, measured not asserted
        dec_ng = dispatch_decision(past, comp, traj.lever_names, seed=seed,
                                   meta_lambda_guard=False)
        pred_ng = _forecast(dec_ng.tool, past, target, traj.lever_names, phis, seed)
        disp_errs_noguard.append(
            abs(float(wcls @ (pred_ng[:N_CLASSES] - meas[:N_CLASSES]))) * target.dep)
        # the oracle (look-ahead) route for THIS fold — reported for diagnosis only
        fold_arm_err = {a: per_arm[a][hold - 2] for a in arm_pool}
        oracle_arm = min(fold_arm_err, key=lambda a: fold_arm_err[a])
        rows.append({
            "fold_hold": hold, "epoch": round(float(target.ep0), 1),
            "regime": dec.classification.regime, "tool": dec.tool,
            "deciding_signal": dec.classification.deciding_signal,
            "prototype_regime": dec.classification.prototype_regime,
            "meta_lambda_surprise": bool(dec.classification.meta_lambda_surprise),
            "gate_certificate": dec.classification.gate_certificate.to_dict(),
            "dispatcher_err": round(err, 8),
            "persistence_err": round(fold_arm_err.get(PERSISTENCE, float("nan")), 8),
            "global_single_best_err": round(fold_arm_err.get(gsb_arm, float("nan")), 8),
            "oracle_arm": oracle_arm,
            "route_matches_oracle": bool(dec.tool == oracle_arm),
            "per_arm_err": {a: float(fold_arm_err[a]) for a in sorted(fold_arm_err)},
        })

    disp_mae = float(np.mean(disp_errs))
    disp_mae_ng = float(np.mean(disp_errs_noguard))
    beats_p = disp_mae < persistence_mae
    beats_g = disp_mae < gsb_mae
    n_correct = sum(1 for r in rows if r["route_matches_oracle"])
    if beats_g and beats_p:
        verdict = (f"BEATS BOTH: dispatcher WF {disp_mae:.6f} < global-single-best "
                   f"({gsb_arm}) {gsb_mae:.6f} < persistence {persistence_mae:.6f} — "
                   f"per-state dispatch earns its regime split (routed {n_correct}/{len(rows)} "
                   f"folds to the oracle arm). PROVISIONAL: n=1 trajectory, margin over "
                   f"global-single-best {100 * (gsb_mae - disp_mae) / gsb_mae:.1f}% is within "
                   "fold-noise at this n; the policy is in-sample-derived (out-of-sample "
                   "generalization owed at ≥2 trajectories).")
    elif beats_p and not beats_g:
        verdict = (f"HONEST NEGATIVE (beats persistence, NOT global-single-best): dispatcher "
                   f"WF {disp_mae:.6f} vs global-single-best ({gsb_arm}) {gsb_mae:.6f} — "
                   "per-state dispatch is OVER-FITTING the regime labels; picking the one "
                   f"arm {gsb_arm} is better. The regime→tool rules still BANK as structural "
                   "knowledge; the dispatcher does not adopt.")
    else:
        verdict = (f"HONEST NEGATIVE: dispatcher WF {disp_mae:.6f} does not beat persistence "
                   f"{persistence_mae:.6f} — regime split not justified on this trajectory.")
    notes = (
        "walk-forward, past-only: the classifier sees intervals[:hold], never intervals[hold] "
        "(no fold-outcome leakage — the deciding signal is the observed-slope history only)",
        "global-single-best = the strongest FIXED single arm over the pool "
        f"{list(arm_pool)} (the arbiter this dispatcher must beat to earn per-state routing)",
        f"meta-λ defer guard {'ON' if meta_lambda_guard else 'OFF'}: guarded WF {disp_mae:.6f} "
        f"vs no-guard {disp_mae_ng:.6f} — the guard's marginal effect is MEASURED, not asserted "
        "(on the #205 trajectory it flips one late transient fold the head was surprised on)",
        "oracle_arm/route_matches_oracle are LOOK-AHEAD diagnostics (reported, never used to "
        "route) — they measure how often the past-only classifier matched the best-in-hindsight arm",
        f"n={len(intervals)} intervals / {len(rows)} folds — small-data regime; every margin "
        "is verdict_scope: instance (re-runs per record accrual via the organ ledger)",
        "router gate comparisons are deterministic NumPy-fp32; every fold carries a "
        "selection-margin/ULP certificate and a fixed tie rule",
    )
    boundary_ulps = []
    for row in rows:
        cert = row["gate_certificate"]
        for key in ("slope_margin_ulps", "surprise_margin_ulps"):
            if cert.get(key) is not None:
                boundary_ulps.append(float(cert[key]))
    unstable = sum(
        not bool(row["gate_certificate"]["stable_beyond_float32_roundoff"])
        for row in rows
    )
    forecast_calibration = calibrate_router_forecast(rows).to_dict()
    return DispatchBacktest(
        n_folds=len(rows), dispatcher_wf_mae=disp_mae,
        dispatcher_wf_mae_no_meta_guard=disp_mae_ng, persistence_wf_mae=persistence_mae,
        global_single_best_arm=gsb_arm, global_single_best_wf_mae=gsb_mae,
        per_arm_wf_mae=per_arm_mae, beats_persistence=beats_p,
        beats_global_single_best=beats_g, meta_lambda_guard=meta_lambda_guard,
        gate_min_boundary_margin_ulps=(min(boundary_ulps) if boundary_ulps else float("nan")),
        gate_unstable_fold_count=unstable,
        forecast_calibration=forecast_calibration,
        fold_rows=tuple(rows), verdict=verdict, notes=notes)
