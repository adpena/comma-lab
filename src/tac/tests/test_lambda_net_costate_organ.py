# SPDX-License-Identifier: MIT
"""Tests for the #426 costate organ: lambda_net + costate_panel + control_alphabet +
the CostateAgent DSL. NO-FAKE: the planted-dynamics test proves the λ-net REALLY
learns a known response (not markers); the containment tests prove the act side
structurally cannot execute; the DSL test compiles and exercises the REAL wiring."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.control_alphabet import (
    HEAVY_ACTIONS,
    OperatorGoTicket,
    SpawnTicket,
    compose_spawn_ticket,
    feasibility_projection,
    hamiltonian_decide,
    rank_duty_to_measure,
)
from tac.witness_control.lambda_net import (
    CampaignTrajectory,
    STATE_DIM,
    RidgeSolveAdjoint,
    backtest,
    build_intervals,
    fit_score_composition,
    lever_features,
    rashomon_lambda_set,
    read_trajectory,
)

REPO = Path(__file__).resolve().parents[3]
LIVE_RUN = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"


# ───────────────────────── synthetic trajectory (planted ground truth) ─────────────────────────
def _synthetic_traj(n_verdicts: int = 7, seed: int = 3) -> tuple[CampaignTrajectory, np.ndarray]:
    """Plant a KNOWN linear response: class-3 (Movable) improves ∝ island share;
    class-0 (Road) improves ∝ seg share. The λ-net must recover the planted signs."""
    rng = np.random.default_rng(seed)
    lever_names = ("seg", "island_amplify", "eikonal", "chroma_boundary")
    d = np.asarray([0.10, 0.40, 0.02, 0.90, 0.01], dtype=float)
    verdicts, loss_terms = [], []
    ep = 0.0
    for k in range(n_verdicts):
        verdicts.append({
            "stage": "verdict", "epoch": ep, "seg_form": "unify_tau",
            "d_seg": float(np.dot([.23, .006, .49, .012, .28], d)),
            "d_seg_by_class": [float(x) for x in d],
            "d_pose": 5.0, "blob_bytes": 85_000.0, "ep_loss": 400.0,
        })
        seg_mag = 3.0 + 0.5 * math.sin(k)          # varying shares (identifiability)
        isl_mag = 0.2 + 0.15 * k
        for e in range(int(ep), int(ep) + 20):
            loss_terms.append({"stage": "loss_terms", "ep": float(e), "gnorm": 5.0,
                               "terms": {"seg": seg_mag, "island_amplify": isl_mag,
                                          "eikonal": 0.03, "chroma_boundary": 0.0}})
        tot = seg_mag + isl_mag + 0.03
        u_seg, u_isl = seg_mag / tot, isl_mag / tot
        # planted dynamics over the 20-epoch interval:
        d = d.copy()
        d[0] = max(d[0] - 0.10 * u_seg * 20 / 20, 1e-4)      # Road ← seg
        d[3] = max(d[3] - 0.40 * u_isl * 20 / 20, 1e-4)      # Movable ← island
        d += rng.normal(0, 1e-4, size=5)
        d = np.clip(d, 1e-5, 1.0)
        ep += 20.0
    traj = CampaignTrajectory(run_dir="synthetic", verdicts=tuple(verdicts),
                              loss_terms=tuple(loss_terms), lever_names=lever_names)
    return traj, d


def test_score_composition_recovers_weights_synthetic():
    traj, _ = _synthetic_traj()
    comp = fit_score_composition(traj.verdicts)
    assert comp.fit_residual < 1e-6
    assert np.allclose(comp.class_weights, [.23, .006, .49, .012, .28], atol=1e-3)


def test_ridge_solve_recovers_planted_response_signs():
    """NO-FAKE core: the fitted field must recover the PLANTED causal structure —
    island_amplify improves Movable (channel 3) more than eikonal does."""
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    assert len(intervals) >= 5
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = RidgeSolveAdjoint()
    m.fit(intervals, phis)
    iv = intervals[-1]
    r_isl = m.response(iv.x0, iv.ctx, lever_features("island_amplify"))
    r_eik = m.response(iv.x0, iv.ctx, lever_features("eikonal"))
    assert r_isl[3] < 0, "island response on Movable must be negative (improving)"
    assert r_isl[3] < r_eik[3], "island must improve Movable more than eikonal"


def test_backtest_on_synthetic_beats_heuristic():
    traj, _ = _synthetic_traj()
    report, field = backtest(traj, architecture="A_ridge_solve")
    assert report.n_intervals >= 5
    assert math.isfinite(report.forecast_mae_model)
    # planted linear dynamics: the solve must beat naive persistence
    assert report.forecast_mae_model <= report.forecast_mae_heuristic
    assert field.status in ("BACKTESTED-PASS", "BACKTESTED-FAIL")


def test_backtest_determinism():
    traj, _ = _synthetic_traj()
    r1, f1 = backtest(traj, architecture="A_ridge_solve")
    r2, f2 = backtest(traj, architecture="A_ridge_solve")
    assert r1.forecast_mae_model == r2.forecast_mae_model
    assert f1.per_lever == f2.per_lever


def test_rashomon_set_shapes():
    traj, _ = _synthetic_traj()
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    rset = rashomon_lambda_set(traj, comp, intervals)
    assert rset.n_members == len(intervals)
    assert set(rset.sign_agreement) == set(traj.lever_names)
    for v in rset.sign_agreement.values():
        assert 0.0 <= v <= 1.0


# ───────────────────────── live-run integration (read-only) ─────────────────────────
@pytest.mark.skipif(not LIVE_RUN.exists(), reason="live run dir absent")
def test_read_trajectory_live_run_readonly():
    traj = read_trajectory(LIVE_RUN)
    assert traj.n_verdicts >= 6
    assert len(traj.loss_terms) >= 100
    comp = fit_score_composition(traj.verdicts)
    # the fitted class weights must recover the canonical GT areas (CLAUDE.md order)
    assert comp.fit_residual < 1e-5
    assert abs(comp.class_weights[2] - 0.493) < 0.02   # Undrivable ≈ .493
    assert abs(comp.class_weights[1] - 0.0059) < 0.005  # Lane ≈ .0059


# ───────────────────────── control alphabet (containment) ─────────────────────────
def test_heavy_actions_return_tickets_never_execute():
    for action in HEAVY_ACTIONS:
        out = feasibility_projection(action, justification="t", gates_owed=("x",))
        assert isinstance(out, OperatorGoTicket)
        assert out.actuation == "NONE"
        assert not hasattr(out, "execute")


def test_unknown_action_refused():
    with pytest.raises(ValueError, match="never-invent-tools"):
        feasibility_projection("rm_rf_everything")


def test_hamiltonian_decide_moves_budget_to_best_lever():
    lam = {"a": -1.0, "b": +0.5, "c": 0.0}
    shares = {"a": 0.2, "b": 0.4, "c": 0.4}
    d = hamiltonian_decide(lam, shares, budget=0.1)
    assert d.actuation == "NONE"
    assert d.proposed_shares["a"] == pytest.approx(0.3)   # gained the budget
    assert d.proposed_shares["b"] == pytest.approx(0.3)   # worst donor paid
    assert d.hamiltonian_value < sum(lam[k] * v for k, v in shares.items())


def test_hamiltonian_decide_respects_sealed():
    lam = {"a": -1.0, "b": +0.5}
    d = hamiltonian_decide(lam, {"a": 0.5, "b": 0.5}, sealed=("a", "b"))
    assert d.proposed_shares == {"a": 0.5, "b": 0.5}      # projection froze everything


def test_spawn_ticket_inherits_containment_and_contract():
    t = compose_spawn_ticket("q?", "trigger")
    assert isinstance(t, SpawnTicket)
    assert t.requires_harness is True
    assert "INHERITED CONTAINMENT" in t.prompt
    assert "operator-GO" in t.prompt
    # the standard subagent contract is embedded (grounded-progress discipline)
    assert "grounded" in t.prompt.lower() or "GROUNDED" in t.prompt


def test_rank_duty_to_measure_orders_never_fired_by_field():
    rows = rank_duty_to_measure({"x": -0.5, "y": -0.1, "z": -0.9},
                                identified={"x": False, "y": False, "z": False},
                                ever_fired={"y": False, "x": False, "z": True})
    assert [r["lever"] for r in rows] == ["x", "y"]       # z fired → excluded
    assert "PARTIAL" in rows[0]["tier"]


# ───────────────────────── panel + routing ─────────────────────────
def test_panel_runs_on_synthetic_and_escalation_is_typed():
    from tac.witness_control.costate_panel import ROUTING_MODES, run_panel
    traj, _ = _synthetic_traj()
    # only the cheap mode in tests (torch lenses are exercised by the benchmark tool)
    v = run_panel(traj, routing_mode="SINGLE_BEST",
                  ledger_path="nonexistent_ledger.jsonl")
    assert v.routing_mode in ROUTING_MODES
    assert v.actuation == "NONE"
    assert len(v.fused_dxdt) == STATE_DIM
    assert v.status == "SPECULATIVE-UNTIL-BACKTESTED"
    for t in v.spawn_tickets:
        assert isinstance(t, SpawnTicket)
    names = [r.spec.name for r in v.reports]
    assert "flow" in names and "spawn_agent" in names and "graph_precedent" in names


def test_shrunk_weights_sit_on_prior_at_zero_evidence():
    from tac.witness_control.costate_panel import _complexity_prior, _shrunk_weights
    names = ["flow", "pointwise", "sequence", "operator_field"]
    w0 = _shrunk_weights(names, {}, channel=0, n_evidence_folds=0)
    assert np.allclose(w0, _complexity_prior(names))
    assert w0[0] == max(w0)  # the SOLVE carries the prior mass (MDL)
    # with strong evidence for `pointwise`, its weight must inflate
    skill = {"flow": np.full(STATE_DIM, 1.0), "pointwise": np.full(STATE_DIM, 1e-6),
             "sequence": np.full(STATE_DIM, 1.0), "operator_field": np.full(STATE_DIM, 1.0)}
    w = _shrunk_weights(names, skill, channel=0, n_evidence_folds=50)
    assert w[1] > w0[1]


# ───────────────────────── the CostateAgent DSL ─────────────────────────
def test_costate_agent_program_validates_and_compiles():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    prog = derive_costate_agent_v1(str(LIVE_RUN if LIVE_RUN.exists() else "."))
    assert prog.validate_program() == []
    organ = prog.compile()
    # act-time projection is the containment governor (typed, not a comment)
    out = organ.act("launch_training", justification="t")
    assert isinstance(out, OperatorGoTicket)
    t = organ.spawn("q?", "trig")
    assert "INHERITED CONTAINMENT" in t.prompt


def test_costate_agent_dsl_fail_closed_surfaces():
    from tac.witness_dsl.costate_agent_dsl import (
        ActuatorSpec, ExpertSpec, RoutingSpec, SensorSpec)
    with pytest.raises(Exception, match="never-invent"):
        ActuatorSpec(action="fabricated_tool", cost_tier="SAFE")
    with pytest.raises(Exception, match="operator_go_gated"):
        ActuatorSpec(action="launch_training", cost_tier="HEAVY")
    with pytest.raises(Exception, match="inherits_containment"):
        ActuatorSpec(action="spawn_reasoning_agent", cost_tier="SAFE")
    with pytest.raises(Exception, match="never-invent"):
        ExpertSpec(name="x", architecture="Z_fabricated", question="?",
                   tools_exercised=())
    with pytest.raises(Exception, match="not in"):
        RoutingSpec(mode="MADE_UP_MODE")
    with pytest.raises(Exception, match="reason"):
        SensorSpec(name="s", source="run_telemetry", enabled=False)


def test_dsl_act_refuses_undeclared_action():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    organ = derive_costate_agent_v1(".").compile()
    with pytest.raises(ValueError, match="closed alphabet"):
        organ.act("dispatch_remote")   # HEAVY member not declared by v1 program


# ───────────────────────── containment source-scan (this trio) ─────────────────────────
def test_new_modules_have_no_actuation_tokens():
    forbidden = ("import subprocess", "from subprocess", "os.system(", "os.exec",
                 "os.spawn", "os.popen(", "os.kill(", "import signal", "Popen(")
    for rel in ("src/tac/witness_control/lambda_net.py",
                "src/tac/witness_control/control_alphabet.py",
                "src/tac/witness_control/costate_panel.py",
                "src/tac/witness_control/prototype_router.py",
                "src/tac/witness_control/continual_costate.py",
                "src/tac/witness_control/scorer_geometry.py",
                "src/tac/witness_control/self_monitor.py",
                "src/tac/witness_control/gepa_reflection.py",
                "src/tac/witness_dsl/costate_agent_dsl.py"):
        src = (REPO / rel).read_text()
        for tok in forbidden:
            assert tok not in src, f"{tok!r} in {rel}"


# ───────────────────────── PRISM interpretable prototype router (Rudin) ─────────────────────────
def test_prototype_router_names_regimes_from_measured_signature():
    """NO-FAKE: prototype names must be DERIVED from the per-class d_seg signature, not
    asserted. The synthetic planted trajectory starts Movable-dominant → island regime."""
    from tac.witness_control.prototype_router import PrototypeRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = PrototypeRouterLens()
    m.fit(intervals, phis)
    assert m.prototypes, "must fit prototypes"
    # the earliest state (d_class[3]=Movable ~0.9) must be named an island regime
    names = {p.name for p in m.prototypes}
    assert any("island" in n for n in names), f"expected an island regime, got {names}"
    # coarse + fine scales both present (Daubechies cascade)
    scales = {p.scale for p in m.prototypes}
    assert scales == {"coarse", "fine"}


def test_prototype_router_backtests_and_is_interpretable():
    """Interpretable-by-design is a HARD requirement AND must still earn its place —
    report the honest cost vs the solve; here it must at least beat the persistence heuristic."""
    traj, _ = _synthetic_traj()
    report, field = backtest(traj, architecture="E_prototype")
    assert report.forecast_mae_model <= report.forecast_mae_heuristic  # earns its place
    # the observatory attribution is a contract: it routes + explains
    from tac.witness_control.prototype_router import PrototypeRouterLens
    comp = fit_score_composition(traj.verdicts)
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = PrototypeRouterLens(); m.fit(intervals, phis)
    x = intervals[-1].x1
    att = m.attribute(x, 999.0, comp.grad_s_wrt_state())
    assert att.fired, "the router must route through ≥1 prototype"
    assert "routed through" in att.explain()
    assert att.mixture_entropy >= 0.0


def test_prototype_suppression_is_traceable():
    """PRISM suppression: zeroing a prototype removes its regime-response, auditably."""
    from tac.witness_control.prototype_router import PrototypeRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = PrototypeRouterLens(); m.fit(intervals, phis)
    x = intervals[-1].x1
    target = m.prototypes[0].name
    r_full = m.response(x, intervals[-1].ctx, lever_features("seg"))
    r_supp = m.response(x, intervals[-1].ctx, lever_features("seg"), suppress=(target,))
    comp = fit_score_composition(traj.verdicts)
    att = m.attribute(x, 1.0, comp.grad_s_wrt_state(), suppress=(target,))
    assert att.suppressed == (target,)
    assert not np.allclose(r_full, r_supp) or target not in {p.name for p in m.prototypes[:1]}


# ───────────────────────── Schmidhuber self-improvement (PowerPlay / Gödel / curiosity) ─────────────────────────
def test_powerplay_acquisition_curiosity_ranking():
    from tac.witness_control.control_alphabet import powerplay_acquisition
    # a fired lever the model was WRONG about (high curiosity) must rank above a fired
    # lever the model predicted correctly (zero surprise), at equal blast/cost.
    lam = {"surprising": -0.10, "correct": -0.10, "unknown": -0.05}
    measured = {"surprising": +0.30, "correct": -0.10}   # surprising: big predicted-vs-real gap
    fired = {"surprising": True, "correct": True, "unknown": False}
    rows = powerplay_acquisition(lam, measured_binding=measured, ever_fired=fired)
    by = {r.lever: r for r in rows}
    assert by["surprising"].curiosity > by["correct"].curiosity  # surprise = compression progress
    assert by["surprising"].acquisition > by["correct"].acquisition
    assert by["unknown"].kind == "explore-unknown"
    assert by["correct"].kind == "exploit-surprise"


def test_godel_proof_gate_requires_improvement_proof():
    from tac.witness_control.control_alphabet import GodelProofGate
    ok = GodelProofGate.evaluate("c", backtest_passed=True, predicted_delta_s=-0.01)
    assert ok.admissible and ok.actuation == "NONE"
    no_bt = GodelProofGate.evaluate("c", backtest_passed=False, predicted_delta_s=-0.01)
    assert not no_bt.admissible and "no backtest proof" in no_bt.reason
    regress = GodelProofGate.evaluate("c", backtest_passed=True, predicted_delta_s=+0.01)
    assert not regress.admissible and "never-regress" in regress.reason
    none_ds = GodelProofGate.evaluate("c", backtest_passed=True, predicted_delta_s=None)
    assert not none_ds.admissible


def test_dsl_interpretable_head_is_hard_requirement():
    from tac.witness_dsl.costate_agent_dsl import RoutingSpec
    with pytest.raises(Exception, match="interpretable-by-design"):
        RoutingSpec(interpretable_head=False)


def test_dsl_organ_observatory_and_curiosity_wired():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    if not LIVE_RUN.exists():
        pytest.skip("live run dir absent")
    organ = derive_costate_agent_v1(str(LIVE_RUN)).compile()
    att = organ.observe()
    assert "routed through" in att.explain()
    g = organ.prove_improvement("x", backtest_passed=True, predicted_delta_s=-0.001)
    assert g.admissible
    acq = organ.acquire({"a": -0.1, "b": -0.2}, ever_fired={"a": False, "b": False})
    assert acq and acq[0].acquisition >= acq[-1].acquisition
