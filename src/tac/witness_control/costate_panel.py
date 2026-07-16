# SPDX-License-Identifier: MIT
"""The costate PANEL — task #426: a diverse learned council of architectural lenses.

Not a pick-one benchmark: each lens is a distinct QUESTION-ANSWERER exercising a
distinct TOOL (the Rashomon/grand-council discipline realized as architecture):

  flow          (A ridge SOLVE / DMDc)   "where is the flow HEADING?"      → trajectory integrator
  pointwise     (B MLP)                  "what does THIS state imply?"     → state snapshot
  sequence      (C GRU over dense path)  "what's the long-horizon PATTERN?"→ per-epoch telemetry
  operator_field(D DeepONet branch⊗trunk)"what does the design-surface     → lever registry /
                                          GEOMETRY say (incl. never-fired)?"  duty-to-measure
  graph_precedent (retrieval)            "what does campaign HISTORY imply?"→ activation ledger /
                                                                             graph memory
  spawn_agent   (System-2, self-gated)   "novel / uncertain / open-ended"  → the whole tool
                                                                             universe (bounded)

ROUTING SPECTRUM (each a first-class mode, arbitrated by the routing benchmark):
  * SINGLE_BEST       — one a-priori lens (the anti-kitchen-sink floor).
  * QUESTION_ROUTER   — centralized: pick the lens with best inner-fold skill for the
                        regime (RouteLLM lineage; degenerates to inner-selected-best on
                        a single-regime trajectory — stated, not hidden).
  * SELF_ACTIVATION   — routing-free (arXiv 2604.00801 lineage): each lens gates ITSELF
                        by its own train-side confidence; load-balancing caps any lens.
                        No external arbiter — the ensemble of self-gates IS the routing.
  * COMPONENT_FUSION  — finest grain (FusionRoute arXiv 2601.05106 lineage): the λ/forecast
                        field is assembled PER COMPONENT (per state channel), each channel
                        weighted by the lenses' measured per-channel inner skill.

DISAGREEMENT IS A FIRST-CLASS SIGNAL: per-component dispersion across active lenses is
the panel's own uncertainty made legible; the spawn_agent lens SELF-ACTIVATES on it
(System-2 escalation — the organ knows when it doesn't know). Spawn = SpawnTicket
(``control_alphabet``), containment inherited, harness-executed only.

Every number here is [macOS advisory] NON-PROMOTABLE; the panel is MEANS (a campaign
accelerator), never a pointer-mover.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.witness_control.control_alphabet import SpawnTicket, compose_spawn_ticket
from tac.witness_control.lambda_net import (
    CampaignTrajectory,
    Interval,
    N_CLASSES,
    STATE_DIM,
    build_intervals,
    evaluate_lambda_field,
    fit_score_composition,
    lever_features,
    make_model,
    rashomon_lambda_set,
)

# ── the lens registry (question + tool coverage is part of a lens's VALUE) ──
@dataclass(frozen=True)
class LensSpec:
    name: str
    architecture: str            # lambda_net arch | "graph_precedent" | "spawn_agent"
    question: str
    tools_exercised: tuple[str, ...]
    predictive: bool             # participates in the forecast fusion?


LENSES: tuple[LensSpec, ...] = (
    LensSpec("flow", "A_ridge_solve", "where is the flow heading?",
             ("trajectory-integrator", "ncde/dmdc-solve"), True),
    LensSpec("pointwise", "B_mlp", "what does this state imply?",
             ("state-snapshot",), True),
    LensSpec("sequence", "C_gru_path", "what is the long-horizon pattern?",
             ("dense-telemetry-stream",), True),
    LensSpec("operator_field", "D_deeponet",
             "what does the design-surface geometry say (incl. never-fired levers)?",
             ("lever-registry", "duty-to-measure"), True),
    LensSpec("prototype", "E_prototype",
             "which recognizable REGIME is this, and WHY (interpretable-by-design)?",
             ("prototype-neighborhoods", "observatory-attribution"), True),
    LensSpec("block_subspace", "F_bsf",
             "which curved multidim regime CHART is this state on (BSF block-sparse "
             "featurizer geometry)?",
             ("block-subspace-charts", "observatory-attribution"), True),
    LensSpec("exact_factorized", "V_exact_factorized_residual",
             "what does the exact head x resize-kernel x composed-gain adjoint recommend?",
             ("canonical-equations", "ncde-trajectory", "morse-smale-events"), True),
    LensSpec("graph_precedent", "graph_precedent",
             "what does campaign history/structure imply?",
             ("lever-activation-ledger", "graph-memory-recall"), False),
    LensSpec("spawn_agent", "spawn_agent",
             "novel / uncertain / open-ended reasoning",
             ("web", "oss", "codex", "claude-code", "whole-tool-universe"), False),
)

#: name → LensSpec (index-independent lookup; new lenses may be inserted anywhere)
_LENS_BY_NAME = {s.name: s for s in LENSES}
_GRAPH_PRECEDENT_SPEC = _LENS_BY_NAME["graph_precedent"]
_SPAWN_AGENT_SPEC = _LENS_BY_NAME["spawn_agent"]

ROUTING_MODES = ("SINGLE_BEST", "QUESTION_ROUTER", "SELF_ACTIVATION", "COMPONENT_FUSION",
                 "EVIDENCE_SHRUNK_STACKING")

#: effective fitted-parameter counts per predictive lens (the MDL complexity prior of
#: EVIDENCE_SHRUNK_STACKING — survey sharpening #1: self-activation must be a partially-
#: pooled posterior over lens competence (Yao-Pirš-Vehtari-Gelman arXiv 2101.08954),
#: NOT a trained router and NOT raw train-confidence; the prior favors low complexity).
LENS_COMPLEXITY = {"flow": 102.0, "pointwise": 332.0, "sequence": 1100.0,
                   "operator_field": 700.0, "prototype": 220.0, "block_subspace": 280.0,
                   "exact_factorized": 5.0}
#: torch-training lenses (expensive; the spread-gate may skip them — 2607.08046
#: finding 3: route by the PRE-SET answer-distribution spread; low spread → commit
#: cheap with no accuracy loss, high spread → pay for the full panel)
_EXPENSIVE_LENSES = frozenset({"pointwise", "sequence", "operator_field"})
#: prior strength (pseudo-fold count) for the partial pooling — at n_folds ≈ α the
#: evidence and the prior share weight equally; evidence dominates as folds accrue.
STACKING_PRIOR_STRENGTH = 3.0
#: load-balancing cap for self-activation (no lens may dominate the routing-free mix)
SELF_ACTIVATION_CAP = 0.5
#: disagreement level (weighted per-component dispersion / scale) that wakes System-2
SPAWN_DISAGREEMENT_THRESHOLD = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# graph-precedent lens — retrieval over the lever activation ledger (#411 spirit:
# the campaign graph consulted, not re-derived). HONEST: this is a RETRIEVAL lens
# (a learned GNN over the DAG is owed when >1 campaign trajectory exists — it
# cannot be trained honestly on one graph instance).
# ─────────────────────────────────────────────────────────────────────────────
def graph_precedent(ledger_path: str | Path = ".omx/state/lever_activation_ledger.jsonl",
                    ) -> dict[str, dict]:
    """Per-lever precedent from the activation ledger: fired-count + last event."""
    out: dict[str, dict] = {}
    p = Path(ledger_path)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        lever = str(r.get("lever", "")).strip()
        if not lever:
            continue
        row = out.setdefault(lever, {"fired_count": 0, "events": []})
        if r.get("event") == "fired":
            row["fired_count"] += 1
        row["events"].append({"event": r.get("event"), "ts": r.get("ts"),
                              "run_ref": r.get("run_ref")})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# panel verdict containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LensReport:
    spec: LensSpec
    activation: float                       # post-load-balancing weight in the fused mix
    self_confidence: float                  # the lens's OWN gate (train-side; no arbiter)
    forecast_dxdt: tuple[float, ...] | None # (STATE_DIM,) predicted dx/dt (None = abstain)
    lambda_field: dict[str, float] | None   # per-lever marginal-ΔS (None = abstain)
    insight: str                            # the lens's distinct 1-line answer to ITS question


@dataclass(frozen=True)
class PanelVerdict:
    routing_mode: str
    reports: tuple[LensReport, ...]
    fused_dxdt: tuple[float, ...]           # the panel's forecast (per routing mode)
    consensus_lambda: dict[str, float]      # activation-weighted per-lever field
    disagreement: float                     # weighted per-component dispersion (legible uncertainty)
    disagreement_by_channel: tuple[float, ...]
    spawn_tickets: tuple[SpawnTicket, ...]  # System-2 escalations (self-activated)
    status: str                             # honesty stamp from the routing benchmark
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    actuation: str = "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# fitting helpers
# ─────────────────────────────────────────────────────────────────────────────
def _fit_lens(arch: str, intervals: list[Interval], phis: np.ndarray, seed: int):
    m = make_model(arch)
    m.fit(intervals, phis, seed=seed)
    return m


def _lens_predict(model, arch: str, iv: Interval, lever_names: tuple[str, ...]) -> np.ndarray:
    base = (model.base(iv.x0, iv.ctx) if arch == "A_ridge_solve"
            else model.base(iv.x0, iv.ctx, iv.path))
    resp = np.stack([model.response(iv.x0, iv.ctx, lever_features(n), iv.path)
                     for n in lever_names])
    return base + resp.T @ iv.u_mean


def _train_residual(model, arch: str, intervals: list[Interval],
                    lever_names: tuple[str, ...]) -> np.ndarray:
    """Per-channel mean-abs train residual — the substrate of SELF-confidence."""
    errs = [np.abs(_lens_predict(model, arch, iv, lever_names) - iv.dxdt())
            for iv in intervals]
    return np.mean(np.stack(errs), axis=0)


def _self_confidence(train_resid: np.ndarray, scale: np.ndarray) -> float:
    """Routing-free self-gate: exp(−residual/scale), averaged over channels.

    Train-side only (no external arbiter) — the honest small-data analogue of a
    lens 'determining its own activation by its own confidence' (2604.00801). The
    KNOWN failure mode (overfit lens → overconfident) is contained by the
    load-balancing cap, and the routing benchmark MEASURES whether it survives."""
    r = np.mean(train_resid / (scale + 1e-12))
    return float(math.exp(-min(r, 50.0)))


def _complexity_prior(names: list[str]) -> np.ndarray:
    """MDL prior over lenses: p ∝ 1/complexity (normalized). Low-complexity lenses
    carry the prior mass — the theory-forced small-data default (2405.15992)."""
    inv = np.asarray([1.0 / LENS_COMPLEXITY.get(n, 500.0) for n in names])
    return inv / inv.sum()


def _shrunk_weights(names: list[str], inner_skill: dict[str, np.ndarray], channel: int,
                    n_evidence_folds: int, alpha: float = STACKING_PRIOR_STRENGTH,
                    ) -> np.ndarray:
    """EVIDENCE_SHRUNK_STACKING per-channel weights: posterior ∝ prior × evidence,
    partially pooled — w = (n·evidence + α·prior)/(n + α). At small n the weights sit
    on the prior (low-complexity lenses); measured per-channel skill inflates a lens
    exactly where it is competent (the hierarchical-stacking behavior, poor-man's
    form; the full Bayesian treatment is owed at >1 campaign trajectory)."""
    prior = _complexity_prior(names)
    errs = np.asarray([float(inner_skill.get(n, np.full(STATE_DIM, np.inf))[channel])
                       for n in names])
    finite = errs[np.isfinite(errs)]
    scale = float(np.min(finite)) if finite.size else 1.0
    ev = np.exp(-errs / max(scale, 1e-12))
    ev = ev / max(float(ev.sum()), 1e-12)
    n = float(max(n_evidence_folds, 0))
    w = (n * ev + alpha * prior) / (n + alpha)
    return w / max(float(w.sum()), 1e-12)


def _load_balance(weights: dict[str, float], cap: float = SELF_ACTIVATION_CAP) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights} if weights else {}
    w = {k: v / total for k, v in weights.items()}
    for _ in range(4):  # iterative cap-and-renormalize (adaptive load balancing)
        over = {k for k, v in w.items() if v > cap}
        if not over or len(over) == len(w):
            break
        excess = sum(w[k] - cap for k in over)
        under_total = sum(v for k, v in w.items() if k not in over)
        for k in list(w):
            if k in over:
                w[k] = cap
            elif under_total > 0:
                w[k] += excess * (w[k] / under_total)
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


# ─────────────────────────────────────────────────────────────────────────────
# THE PANEL
# ─────────────────────────────────────────────────────────────────────────────
def run_panel(traj: CampaignTrajectory, *, routing_mode: str = "COMPONENT_FUSION",
              seed: int = 0, status: str = "SPECULATIVE-UNTIL-BACKTESTED",
              inner_skill: dict[str, np.ndarray] | None = None,
              ledger_path: str | Path = ".omx/state/lever_activation_ledger.jsonl",
              single_best_lens: str = "flow",
              spread_gate: bool = False,
              ) -> PanelVerdict:
    """Fit every predictive lens on ALL intervals and fuse per ``routing_mode``.

    ``inner_skill`` (lens → per-channel MAE from held-out folds) feeds QUESTION_ROUTER
    and COMPONENT_FUSION; when absent those modes fall back to self-confidence weights
    with an explicit note in the insight line (never silently).

    ``single_best_lens`` — the a-priori lens SINGLE_BEST commits to (arbitrated by the
    routing benchmark + the organ ledger, no longer hardwired to "flow").
    ``spread_gate`` — pre-decision spread routing (2607.08046 finding 3): compute the
    cheap Rashomon spread FIRST; when the near-optimal set already AGREES (no unstable
    levers) skip fitting the expensive torch lenses (they are skipped LOUDLY via the
    insight line). Default False so benchmarks stay full-panel comparable; the digest
    runs with the gate ON (compute policy, recorded — not orphaned signal)."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"routing_mode {routing_mode!r} not in {ROUTING_MODES}")
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 2:
        raise ValueError("panel needs ≥2 measured intervals")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    last = intervals[-1]
    dx_scale = np.mean(np.abs(np.stack([iv.dxdt() for iv in intervals])), axis=0)

    # pre-decision spread (cheap, numpy-only) — decides whether the expensive lenses
    # are worth their compute THIS call.
    skipped_expensive: tuple[str, ...] = ()
    rset = rashomon_lambda_set(traj, comp, intervals)
    if spread_gate and not rset.unstable_levers:
        skipped_expensive = tuple(sorted(_EXPENSIVE_LENSES))

    reports: list[LensReport] = []
    predictions: dict[str, np.ndarray] = {}
    fields: dict[str, dict[str, float]] = {}
    confidences: dict[str, float] = {}
    for spec in LENSES:
        if not spec.predictive:
            continue
        if spec.name in skipped_expensive:
            reports.append(LensReport(
                spec=spec, activation=0.0, self_confidence=0.0, forecast_dxdt=None,
                lambda_field=None,
                insight="SKIPPED by the pre-decision spread gate (Rashomon set agrees; "
                        "commit-cheap per 2607.08046 §pre-set-spread routing)"))
            continue
        model = _fit_lens(spec.architecture, intervals, phis, seed)
        pred = _lens_predict(model, spec.architecture, last, traj.lever_names)
        resid = _train_residual(model, spec.architecture, intervals, traj.lever_names)
        conf = _self_confidence(resid, dx_scale)
        fld = evaluate_lambda_field(model, traj, comp, intervals, status=status)
        predictions[spec.name] = pred
        fields[spec.name] = dict(fld.per_lever)
        confidences[spec.name] = conf
        wcls = comp.class_weights
        d_seg_rate = float(wcls @ pred[:N_CLASSES])
        reports.append(LensReport(
            spec=spec, activation=0.0, self_confidence=conf,
            forecast_dxdt=tuple(pred.tolist()), lambda_field=fields[spec.name],
            insight=f"predicts d_seg rate {d_seg_rate:+.2e}/ep at ep{last.ep1:.0f}+ "
                    f"(train-side confidence {conf:.3f})"))

    # ---- routing / fusion ----
    lens_names = list(predictions)
    P = np.stack([predictions[n] for n in lens_names])          # (K, STATE_DIM)
    if routing_mode == "SINGLE_BEST":
        pick_sb = single_best_lens if single_best_lens in lens_names else lens_names[0]
        weights = {n: (1.0 if n == pick_sb else 0.0) for n in lens_names}
        W = np.asarray([[weights[n]] * STATE_DIM for n in lens_names])
    elif routing_mode == "SELF_ACTIVATION":
        weights = _load_balance({n: confidences[n] for n in lens_names})
        W = np.asarray([[weights[n]] * STATE_DIM for n in lens_names])
    elif routing_mode == "QUESTION_ROUTER":
        if inner_skill:
            tot = {n: float(np.mean(inner_skill.get(n, np.full(STATE_DIM, np.inf))))
                   for n in lens_names}
            pick = min(tot, key=lambda n: tot[n])
        else:
            pick = max(confidences, key=lambda n: confidences[n])
        weights = {n: (1.0 if n == pick else 0.0) for n in lens_names}
        W = np.asarray([[weights[n]] * STATE_DIM for n in lens_names])
    elif routing_mode == "EVIDENCE_SHRUNK_STACKING":
        n_ev = (len(intervals) - 1) if inner_skill else 0
        W = np.zeros((len(lens_names), STATE_DIM))
        for c in range(STATE_DIM):
            W[:, c] = _shrunk_weights(lens_names, inner_skill or {}, c, n_ev)
        weights = {n: float(np.mean(W[i])) for i, n in enumerate(lens_names)}
    else:  # COMPONENT_FUSION — per-channel inverse-error softmin over inner skill
        W = np.zeros((len(lens_names), STATE_DIM))
        for c in range(STATE_DIM):
            if inner_skill:
                errs = np.asarray([float(inner_skill.get(n, np.full(STATE_DIM, np.inf))[c])
                                   for n in lens_names])
                scale = float(np.min(errs[np.isfinite(errs)])) if np.any(np.isfinite(errs)) else 1.0
                w = np.exp(-errs / max(scale, 1e-12))
            else:
                w = np.asarray([confidences[n] for n in lens_names])
            w = w / max(float(w.sum()), 1e-12)
            W[:, c] = w
        weights = {n: float(np.mean(W[i])) for i, n in enumerate(lens_names)}
    fused = np.einsum("kc,kc->c", W, P)

    # disagreement: activation-weighted per-channel dispersion / trajectory scale
    mean_pred = fused
    disp = np.sqrt(np.einsum("kc,kc->c", W, (P - mean_pred[None, :]) ** 2))
    dis_by_ch = disp / (dx_scale + 1e-12)
    disagreement = float(np.mean(dis_by_ch))

    # consensus λ-field (activation-weighted mean over lens fields)
    consensus: dict[str, float] = {}
    for lever in traj.lever_names:
        consensus[lever] = float(sum(weights[n] * fields[n].get(lever, 0.0)
                                     for n in lens_names))

    # graph-precedent lens (retrieval; abstains from forecast)
    prec = graph_precedent(ledger_path)
    reports.append(LensReport(
        spec=_GRAPH_PRECEDENT_SPEC, activation=0.0, self_confidence=1.0, forecast_dxdt=None,
        lambda_field=None,
        insight=f"ledger precedent for {len(prec)} levers "
                f"({sum(1 for v in prec.values() if v['fired_count'] > 0)} ever fired); "
                "retrieval lens — learned GNN owed at >1 campaign trajectory"))

    # System-2 self-activation — TWO triggers, both action-level (survey sharpening #2):
    # (a) Rashomon-set sign disagreement on a decision-relevant lever (near-optimal
    #     response models disagree about WHAT TO DO — no System-1 fusion is honest);
    # (b) gross cross-lens forecast dispersion (a lens's regime assumption is violated).
    tickets: list[SpawnTicket] = []
    if rset.unstable_levers:
        tickets.append(compose_spawn_ticket(
            question=(f"The Rashomon set of near-optimal response models (jackknife "
                      f"ridge ensemble, {rset.n_members} members) DISAGREES on the SIGN "
                      f"of the costate for decision-relevant lever(s) "
                      f"{list(rset.unstable_levers)} at ep{last.ep1:.0f}. Design the "
                      "cheapest $0/cached probe (or the single-lever A/B worth an "
                      "operator-GO ticket) that resolves the sign."),
            trigger=f"Rashomon sign-disagreement on {len(rset.unstable_levers)} "
                    f"decision-relevant lever(s)",
            context_lines=(f"run: {traj.run_dir}",
                           *(f"{n}: values={[round(v, 6) for v in rset.per_lever_values[n]]} "
                             f"agreement={rset.sign_agreement[n]:.2f}"
                             for n in rset.unstable_levers[:4]))))
    if disagreement > SPAWN_DISAGREEMENT_THRESHOLD:
        worst = int(np.argmax(dis_by_ch))
        tickets.append(compose_spawn_ticket(
            question=(f"The costate panel's lenses disagree {disagreement:.2f}× the "
                      f"trajectory scale (worst channel {worst}) at ep{last.ep1:.0f}. "
                      "Read the run telemetry and the harvest memo, diagnose WHICH lens's "
                      "regime assumption is violated, and propose the $0 probe that "
                      "resolves the disagreement."),
            trigger=f"panel disagreement {disagreement:.2f} > {SPAWN_DISAGREEMENT_THRESHOLD}",
            context_lines=(f"run: {traj.run_dir}", f"channels dispersion: {dis_by_ch.round(2).tolist()}")))
    reports.append(LensReport(
        spec=_SPAWN_AGENT_SPEC, activation=1.0 if tickets else 0.0,
        self_confidence=disagreement, forecast_dxdt=None, lambda_field=None,
        insight=("ESCALATED (System-2 awake): " + tickets[0].trigger) if tickets else
                f"dormant (disagreement {disagreement:.2f} ≤ {SPAWN_DISAGREEMENT_THRESHOLD})"))

    # stamp activations into the predictive reports
    stamped = []
    for r in reports:
        if r.spec.name in weights:
            stamped.append(LensReport(spec=r.spec, activation=weights[r.spec.name],
                                      self_confidence=r.self_confidence,
                                      forecast_dxdt=r.forecast_dxdt,
                                      lambda_field=r.lambda_field, insight=r.insight))
        else:
            stamped.append(r)
    return PanelVerdict(routing_mode=routing_mode, reports=tuple(stamped),
                        fused_dxdt=tuple(fused.tolist()), consensus_lambda=consensus,
                        disagreement=disagreement,
                        disagreement_by_channel=tuple(dis_by_ch.tolist()),
                        spawn_tickets=tuple(tickets), status=status)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING BENCHMARK — nested leave-one-interval-out: which routing grain earns
# its complexity on held-out forecast? (The anti-kitchen-sink arbiter.)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RoutingBenchmark:
    per_mode_mae: dict[str, float]              # scalar d_seg-rate MAE per routing mode
    per_mode_perclass_mae: dict[str, float]
    per_lens_solo_mae: dict[str, float]         # each lens alone (the ablation)
    winner: str
    n_folds: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        return d


def routing_benchmark(traj: CampaignTrajectory, *, seed: int = 0,
                      folds: tuple[int, ...] | None = None,
                      single_best_lens: str = "flow") -> RoutingBenchmark:
    """Nested LOO: outer folds score the fused forecast; inner folds supply the skill
    weights (so no mode ever sees its own held-out interval — no leakage).

    ``folds`` restricts the outer folds (chunked-resumable execution: run one fold per
    process invocation and aggregate the reports — the L47 harness-kill discipline)."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        raise ValueError("routing benchmark needs ≥3 intervals")
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    wcls = comp.class_weights
    pred_lenses = [s for s in LENSES if s.predictive]

    mode_errs: dict[str, list[float]] = {m: [] for m in ROUTING_MODES}
    mode_perr: dict[str, list[float]] = {m: [] for m in ROUTING_MODES}
    solo_errs: dict[str, list[float]] = {s.name: [] for s in pred_lenses}
    dx_scale = np.mean(np.abs(np.stack([iv.dxdt() for iv in intervals])), axis=0)

    fold_ids = tuple(folds) if folds is not None else tuple(range(len(intervals)))
    for hold in fold_ids:
        train = intervals[:hold] + intervals[hold + 1:]
        held = intervals[hold]
        meas = held.dxdt()
        # inner skill: per-lens per-channel MAE over inner LOO folds of `train`
        inner_skill: dict[str, np.ndarray] = {}
        preds: dict[str, np.ndarray] = {}
        confs: dict[str, float] = {}
        for spec in pred_lenses:
            inner_errs = []
            for j in range(len(train)):
                inner_train = train[:j] + train[j + 1:]
                if len(inner_train) < 2:
                    continue
                m = _fit_lens(spec.architecture, inner_train, phis, seed)
                inner_errs.append(np.abs(
                    _lens_predict(m, spec.architecture, train[j], traj.lever_names)
                    - train[j].dxdt()))
            inner_skill[spec.name] = (np.mean(np.stack(inner_errs), axis=0)
                                      if inner_errs else np.full(STATE_DIM, np.inf))
            model = _fit_lens(spec.architecture, train, phis, seed)
            preds[spec.name] = _lens_predict(model, spec.architecture, held, traj.lever_names)
            confs[spec.name] = _self_confidence(
                _train_residual(model, spec.architecture, train, traj.lever_names), dx_scale)
            e = preds[spec.name] - meas
            solo_errs[spec.name].append(abs(float(wcls @ e[:N_CLASSES])) * held.dep)

        names = [s.name for s in pred_lenses]
        P = np.stack([preds[n] for n in names])
        for mode in ROUTING_MODES:
            if mode == "SINGLE_BEST":
                sb = single_best_lens if single_best_lens in names else names[0]
                W = np.asarray([[1.0 if n == sb else 0.0] * STATE_DIM for n in names])
            elif mode == "SELF_ACTIVATION":
                w = _load_balance({n: confs[n] for n in names})
                W = np.asarray([[w[n]] * STATE_DIM for n in names])
            elif mode == "QUESTION_ROUTER":
                tot = {n: float(np.mean(inner_skill[n])) for n in names}
                pick = min(tot, key=lambda n: tot[n])
                W = np.asarray([[1.0 if n == pick else 0.0] * STATE_DIM for n in names])
            elif mode == "EVIDENCE_SHRUNK_STACKING":
                W = np.zeros((len(names), STATE_DIM))
                for c in range(STATE_DIM):
                    W[:, c] = _shrunk_weights(names, inner_skill, c, len(train))
            else:  # COMPONENT_FUSION
                W = np.zeros((len(names), STATE_DIM))
                for c in range(STATE_DIM):
                    errs = np.asarray([inner_skill[n][c] for n in names])
                    finite = errs[np.isfinite(errs)]
                    scale = float(np.min(finite)) if finite.size else 1.0
                    w_c = np.exp(-errs / max(scale, 1e-12))
                    W[:, c] = w_c / max(float(w_c.sum()), 1e-12)
            fused = np.einsum("kc,kc->c", W, P)
            e = fused - meas
            mode_errs[mode].append(abs(float(wcls @ e[:N_CLASSES])) * held.dep)
            mode_perr[mode].append(float(np.mean(np.abs(e[:N_CLASSES]))) * held.dep)

    per_mode = {m: float(np.mean(v)) for m, v in mode_errs.items()}
    per_mode_pc = {m: float(np.mean(v)) for m, v in mode_perr.items()}
    per_solo = {n: float(np.mean(v)) for n, v in solo_errs.items()}
    winner = min(per_mode, key=lambda m: per_mode[m])
    notes = (
        f"nested LOO over {len(intervals)} folds; inner folds supply skill weights "
        "(no leakage)",
        "QUESTION_ROUTER on a single-regime trajectory degenerates to inner-selected-"
        "best — stated, not hidden",
        "solo ablation = each lens alone (a fusion mode must beat the best solo lens "
        "to EARN its complexity; parsimony wins ties — kitchen_sink anti-pattern)",
        "graph_precedent + spawn_agent abstain from forecast (they answer different "
        "questions; their value is tool/question coverage, reported separately)",
    )
    return RoutingBenchmark(per_mode_mae=per_mode, per_mode_perclass_mae=per_mode_pc,
                            per_lens_solo_mae=per_solo, winner=winner,
                            n_folds=len(fold_ids), notes=notes)
