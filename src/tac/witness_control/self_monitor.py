# SPDX-License-Identifier: MIT
"""Self-activation awareness — the organ's META-λ head (task #426, operator binding
2026-07-11: "it having awareness [of] its own activation may be important signal for
it catching its own errors").

The forecaster-probing method (arXiv 2607.08046: internal representations calibrate
better than verbalized outputs) turned INWARD: the organ probes ITS OWN activation
state to predict when the primary λ is untrustworthy, and closes the loop with a
typed CORRECTION (advisory routing adjustment — never actuation). Unifies the four
uncertainty signals that already exist into ONE self-awareness readout:

  1. SELF-FORECAST ERROR (the innovation signal on the freshest data): fit the
     incumbent lens on all-but-the-last interval, predict the last, compare to the
     MEASURED outcome, normalized by the persistence heuristic's error. r > 1 means
     the organ is CURRENTLY losing to the incumbent extrapolation — surprise = the
     model is wrong = it will learn = do not trust the point λ.
  2. RASHOMON AGREEMENT: sign agreement across the jackknife near-optimal set on
     decision-relevant levers (disagreement = the action is not identified).
  3. ROUTING ENTROPY: the prototype mixture entropy at the latest state (regime
     ambiguity — the router itself is unsure which chart it is on).
  4. FAITHFULNESS GAP: the stated-vs-counterfactual attribution audit (an
     interpretable head whose story diverges from its computation is flagged).
  Plus the PLATEAU flag (measured regime where persistence is competitive — the
  2026-07-11 walk-forward review found the model loses exactly there).

The readout is a FALLING-RULE-style report (Rudin: the decision IS the explanation):
per-component score + the rule that fired + the typed correction. Everything is
advisory ([macOS advisory] NON-PROMOTABLE); no actuation surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tac.witness_control.lambda_net import (
    N_CLASSES,
    CampaignTrajectory,
    _predict_interval,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
    rashomon_lambda_set,
)

#: corrections the meta-λ may recommend (typed, closed alphabet — advisory only)
CORRECTIONS = ("commit", "widen_band", "prefer_persistence", "escalate_system2",
               "distrust_attribution_use_solve")


@dataclass(frozen=True)
class SelfAwarenessReport:
    """The meta-λ readout: does the organ trust its own primary λ right now?"""

    trust: float                        # [0,1]; 1 = every self-probe clean
    flags: tuple[str, ...]              # the rules that fired (falling-rule readback)
    corrections: tuple[str, ...]        # typed advisory corrections (⊆ CORRECTIONS)
    components: dict                    # every probe's raw value (max observability)
    axis_tag: str = "[macOS advisory] NON-PROMOTABLE"
    actuation: str = "NONE"

    def explain(self) -> str:
        if not self.flags:
            return f"meta-λ trust {self.trust:.2f}: all self-probes clean → commit"
        return (f"meta-λ trust {self.trust:.2f}: " + "; ".join(self.flags)
                + f" → {', '.join(self.corrections)}")


def self_activation_probe(traj: CampaignTrajectory, *, incumbent: str = "E_prototype",
                          seed: int = 0, faith_tol: float = 0.35) -> SelfAwarenessReport:
    """Probe the organ's own activation state (all $0, trajectory-history only)."""
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    if len(intervals) < 3:
        return SelfAwarenessReport(
            trust=0.0, flags=("insufficient intervals for self-probing",),
            corrections=("widen_band", "prefer_persistence"),
            components={"n_intervals": len(intervals)})
    wcls = comp.class_weights
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    flags: list[str] = []
    corrections: list[str] = []
    components: dict = {}
    scores: list[float] = []

    # 1. self-forecast error on the freshest measured interval (the innovation signal)
    m = make_model(incumbent)
    m.fit(intervals[:-1], phis, seed=seed)
    last = intervals[-1]
    pred = _predict_interval(m, incumbent, last, traj.lever_names)
    meas = last.dxdt()
    heur = intervals[-2].dxdt()
    e_model = abs(float(wcls @ (pred[:N_CLASSES] - meas[:N_CLASSES])))
    e_heur = abs(float(wcls @ (heur[:N_CLASSES] - meas[:N_CLASSES])))
    ratio = e_model / max(e_heur, 1e-12)
    components["self_forecast_error_ratio"] = ratio
    s1 = math.exp(-max(0.0, ratio - 1.0))
    scores.append(s1)
    if ratio > 1.5:
        flags.append(f"self-forecast losing to persistence ×{ratio:.1f} (surprise)")
        corrections.append("prefer_persistence")

    # 2. Rashomon agreement on decision-relevant levers
    rset = rashomon_lambda_set(traj, comp, intervals)
    contested = rset.contested()
    agree_vals = [rset.sign_agreement[n] for n in rset.sign_agreement] or [1.0]
    components["rashomon_mean_agreement"] = float(np.mean(agree_vals))
    components["rashomon_unstable"] = list(rset.unstable_levers)
    scores.append(float(np.mean(agree_vals)))
    if rset.unstable_levers:
        flags.append(f"Rashomon sign-contest on {list(rset.unstable_levers)}")
        corrections.append("escalate_system2")
    components["rashomon_contested"] = contested

    # 3. routing entropy (regime ambiguity) + 4. faithfulness gap
    try:
        from tac.witness_control.prototype_router import PrototypeRouterLens
        lens = PrototypeRouterLens()
        lens.fit(intervals, phis, seed=seed)
        x = last.x1
        att = lens.attribute(x, last.ep1, comp.grad_s_wrt_state())
        components["routing_entropy_nats"] = att.mixture_entropy
        max_ent = math.log(max(len(lens.prototypes), 2))
        s3 = 1.0 - min(att.mixture_entropy / max_ent, 1.0)
        scores.append(s3)
        if att.mixture_entropy > math.log(2.0):
            flags.append(f"regime ambiguity (routing entropy {att.mixture_entropy:.2f} nats)")
            corrections.append("widen_band")
        faith = lens.faithfulness_audit(x, last.ctx, comp.grad_s_wrt_state(),
                                        lever_features("seg"), rel_tol=faith_tol)
        components["faithfulness_max_rel_gap"] = faith["max_rel_gap"]
        s4 = math.exp(-max(0.0, faith["max_rel_gap"] - faith_tol))
        scores.append(s4)
        if not faith["faithful"]:
            flags.append(f"attribution unfaithful (gap {faith['max_rel_gap']:.2f})")
            corrections.append("distrust_attribution_use_solve")
    except Exception as exc:  # probe failure is itself a flag, never a crash
        flags.append(f"router self-probe unavailable ({type(exc).__name__})")
        scores.append(0.5)

    # 5. plateau regime detector (measured 2026-07-11: persistence is competitive there)
    mags = [abs(float(wcls @ iv.dxdt()[:N_CLASSES])) for iv in intervals]
    plateau = mags[-1] < float(np.median(mags))
    components["plateau_regime"] = plateau
    if plateau:
        flags.append("plateau regime (persistence competitive; measured 2026-07-11)")
        corrections.append("widen_band")
        scores.append(0.8)

    trust = float(np.prod(scores)) ** (1.0 / max(len(scores), 1))  # geometric mean
    if not corrections:
        corrections.append("commit")
    # dedupe, preserve order
    seen: set[str] = set()
    cor = tuple(c for c in corrections if not (c in seen or seen.add(c)))
    return SelfAwarenessReport(trust=trust, flags=tuple(flags), corrections=cor,
                               components=components)
