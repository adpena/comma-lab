# SPDX-License-Identifier: MIT
"""Tests for the 2026-07-11 adversarial-review enhancement wave of the #426 costate
organ: walk-forward gate · SpawnTicket clause typing · Gödel report-typed proofs ·
Bregman/BSF/scorer-prior arms · faithfulness audit · continual-learning compounding
(triality ledger + graduation) · self-monitoring meta-λ · GEPA reflection · DSL
describe/render + arbitration. NO-FAKE: every behavior exercised on real or planted
data, never marker-checked."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tac.witness_control.lambda_net import (
    ARCHITECTURES,
    backtest,
    build_intervals,
    fit_score_composition,
    lever_features,
    make_model,
)
from tac.tests.test_lambda_net_costate_organ import LIVE_RUN, _synthetic_traj


# ───────────────────────── walk-forward gate ─────────────────────────
def test_backtest_reports_walkforward_fields():
    traj, _ = _synthetic_traj()
    report, field = backtest(traj, architecture="A_ridge_solve")
    assert math.isfinite(report.walkforward_mae_model)
    assert math.isfinite(report.walkforward_mae_heuristic)
    assert any("WALK-FORWARD" in n for n in report.notes)
    # the walk-forward verdict matches its own numbers
    assert report.passed_walkforward == (
        report.walkforward_mae_model <= report.walkforward_mae_heuristic)


def test_backtest_pass_requires_walkforward_when_computable():
    """A model that wins LOO but loses walk-forward must NOT stamp BACKTESTED-PASS.

    MEASURED on the synthetic (2026-07-11): the planted trajectory's later folds are
    persistence-trivial (heuristic WF 0.0005) — every arch LOSES walk-forward there,
    so LOO-only passes are correctly demoted to BACKTESTED-FAIL (the look-ahead-
    flattered pass this gate extincts)."""
    traj, _ = _synthetic_traj()
    for arch in ("A_ridge_solve", "E_prototype"):
        report, field = backtest(traj, architecture=arch)
        if report.passed:
            assert report.passed_walkforward
        if not report.passed_walkforward:
            assert not report.passed
            assert field.status == "BACKTESTED-FAIL"


# ───────────────────────── containment hardening ─────────────────────────
def test_spawn_ticket_type_enforces_containment_clause():
    from tac.witness_control.control_alphabet import SpawnTicket
    with pytest.raises(ValueError, match="INHERITED_CONTAINMENT_CLAUSE"):
        SpawnTicket(question="q", trigger="t", prompt="do heavy stuff freely")


def test_spawn_ticket_requires_harness_true():
    from tac.witness_control.control_alphabet import (
        INHERITED_CONTAINMENT_CLAUSE, SpawnTicket)
    with pytest.raises(ValueError, match="requires_harness"):
        SpawnTicket(question="q", trigger="t",
                    prompt="x " + INHERITED_CONTAINMENT_CLAUSE,
                    requires_harness=False)


def test_godel_evaluate_from_report_reads_passed_and_law():
    from tac.witness_control.control_alphabet import GodelProofGate
    traj, _ = _synthetic_traj()
    report, _ = backtest(traj, architecture="A_ridge_solve")
    g = GodelProofGate.evaluate_from_report("adopt", report, -0.001)
    assert g.admissible == report.passed
    bad_law = GodelProofGate.evaluate_from_report("adopt", report, -0.001,
                                                  law_ref="not_a_registered_law_v1")
    assert not bad_law.admissible and "law_ref" in bad_law.reason


# ───────────────────────── new architecture arms ─────────────────────────
def test_new_arms_registered_and_constructible():
    assert {"E_prototype_bregman", "F_bsf", "G_ridge_scorerprior"} <= set(ARCHITECTURES)
    m = make_model("E_prototype_bregman")
    assert m.distance == "bregman_kl"
    f = make_model("F_bsf")
    assert f.name == "F_bsf"


def test_bregman_lens_fits_and_backtests():
    traj, _ = _synthetic_traj()
    report, _ = backtest(traj, architecture="E_prototype_bregman")
    assert math.isfinite(report.forecast_mae_model)


def test_bsf_lens_fits_block_charts():
    from tac.witness_control.prototype_router import BlockSubspaceRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = BlockSubspaceRouterLens()
    m.fit(intervals, phis)
    dims = [p.basis.shape[0] for p in m.prototypes if p.basis is not None]
    assert dims and max(dims) <= 2          # multidim charts, bounded block_dim
    # routing still sparse + normalized
    w = m._mixture(intervals[-1].x1)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    report, _ = backtest(traj, architecture="F_bsf")
    assert math.isfinite(report.forecast_mae_model)


def test_scorer_geometry_prior_and_arm():
    from tac.witness_control.scorer_geometry import (
        DEFAULT_GT_CACHE, scorer_flip_susceptibility, scorer_recalibrated_phi)
    if not DEFAULT_GT_CACHE.exists():
        pytest.skip("gt cache absent")
    prior = scorer_flip_susceptibility()
    assert len(prior.susceptibility) == 5
    assert all(v >= 0 for v in prior.susceptibility)
    assert prior.margin_tau > 0
    phi = scorer_recalibrated_phi("lane_edge", prior)
    base = lever_features("lane_edge")
    # total class mass preserved; distribution reweighted
    assert abs(float(phi[:5].sum()) - float(base[:5].sum())) < 1e-9


# ───────────────────────── faithfulness audit ─────────────────────────
def test_faithfulness_audit_reports_gap():
    from tac.witness_control.prototype_router import PrototypeRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    comp = fit_score_composition(traj.verdicts)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    m = PrototypeRouterLens()
    m.fit(intervals, phis)
    out = m.faithfulness_audit(intervals[-1].x1, intervals[-1].ctx,
                               comp.grad_s_wrt_state(), lever_features("seg"))
    assert out["rows"], "must audit the fired prototypes"
    assert 0.0 <= out["max_rel_gap"]
    assert isinstance(out["faithful"], bool)


# ───────────────────────── continual-learning compounding ─────────────────────────
def _fake_reports(traj, wf_a=0.003, wf_b=0.002):
    from tac.witness_control.lambda_net import BacktestReport
    mk = lambda arch, wf, ok: BacktestReport(  # noqa: E731
        architecture=arch, n_intervals=6, forecast_mae_model=wf, forecast_mae_heuristic=0.004,
        forecast_perclass_mae_model=wf, forecast_perclass_mae_heuristic=0.05,
        binding_auroc_model=1.0, binding_auroc_magnitude_heuristic=1.0,
        passed=ok, notes=("t",), walkforward_mae_model=wf,
        walkforward_mae_heuristic=0.004, walkforward_perclass_mae_model=wf,
        walkforward_perclass_mae_heuristic=0.05, passed_walkforward=ok)
    return {"A_ridge_solve": mk("A_ridge_solve", wf_a, True),
            "B_mlp": mk("B_mlp", wf_b, True)}


def test_organ_ledger_roundtrip_and_dedup(tmp_path):
    from tac.witness_control.continual_costate import (
        append_trajectory_record, compose_trajectory_record, load_organ_memory)
    from tac.witness_control.prototype_router import PrototypeRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    lens = PrototypeRouterLens(); lens.fit(intervals, phis)
    ledger = tmp_path / "ledger.md"
    rec = compose_trajectory_record(traj, _fake_reports(traj), lens.prototypes)
    append_trajectory_record(rec, ledger)
    # re-record the SAME run: latest-per-run must dedup (no double count)
    rec2 = dict(rec); rec2["generated_at"] = "29990101T000000Z"
    append_trajectory_record(rec2, ledger)
    mem = load_organ_memory(ledger)
    assert mem.n_records == 1 and mem.n_runs == 1
    assert mem.prototype_library, "regimes reconstructed"
    assert mem.records[0]["generated_at"] == "29990101T000000Z"


def test_arbitration_graduation_requires_min_records(tmp_path):
    from tac.witness_control.continual_costate import (
        GRADUATION_MIN_RECORDS, append_trajectory_record, arbitrate_architecture,
        compose_trajectory_record, load_organ_memory)
    traj, _ = _synthetic_traj()
    ledger = tmp_path / "ledger.md"
    # B_mlp wins wf on every record, but graduation needs ≥3 DISTINCT runs
    for i in range(GRADUATION_MIN_RECORDS):
        rec = compose_trajectory_record(traj, _fake_reports(traj), [])
        rec["run_ref"] = f"run_{i}"
        rec["generated_at"] = f"2026071{i}T000000Z"
        append_trajectory_record(rec, ledger)
        arb = arbitrate_architecture(load_organ_memory(ledger))
        if i + 1 < GRADUATION_MIN_RECORDS:
            assert arb.recommended == "A_ridge_solve"
            assert "not graduated" in arb.graduation["B_mlp"]
        else:
            assert arb.recommended == "B_mlp"
            assert "GRADUATED" in arb.graduation["B_mlp"]


def test_arbitration_empty_defaults_to_solve(tmp_path):
    from tac.witness_control.continual_costate import (
        arbitrate_architecture, load_organ_memory)
    arb = arbitrate_architecture(load_organ_memory(tmp_path / "absent.md"))
    assert arb.recommended == "A_ridge_solve"
    assert arb.n_records == 0


def test_graph_builder_indexes_organ_ledger(tmp_path):
    from tac.graph_memory.build import parse_costate_organ_ledger
    from tac.graph_memory.model import Graph
    from tac.witness_control.continual_costate import (
        append_trajectory_record, compose_trajectory_record)
    from tac.witness_control.prototype_router import PrototypeRouterLens
    traj, _ = _synthetic_traj()
    intervals = build_intervals(traj)
    phis = np.stack([lever_features(n) for n in traj.lever_names])
    lens = PrototypeRouterLens(); lens.fit(intervals, phis)
    ledger = tmp_path / "ledger.md"
    append_trajectory_record(
        compose_trajectory_record(traj, _fake_reports(traj), lens.prototypes), ledger)
    g = Graph()
    n = parse_costate_organ_ledger(g, ledger)
    assert n > 0
    kinds = g.counts_by_type()
    assert kinds.get("regime", 0) >= 1
    assert kinds.get("finding", 0) >= 1


# ───────────────────────── self-monitoring meta-λ ─────────────────────────
def test_self_activation_probe_produces_typed_report():
    from tac.witness_control.self_monitor import CORRECTIONS, self_activation_probe
    traj, _ = _synthetic_traj()
    rep = self_activation_probe(traj)
    assert 0.0 <= rep.trust <= 1.0
    assert rep.actuation == "NONE"
    assert all(c in CORRECTIONS for c in rep.corrections)
    assert "self_forecast_error_ratio" in rep.components
    assert "explain" not in rep.components  # explain is a method, components are raw
    assert rep.explain()


def test_self_probe_flags_untrustworthy_on_tiny_data():
    from tac.witness_control.self_monitor import self_activation_probe
    traj, _ = _synthetic_traj(n_verdicts=3)
    rep = self_activation_probe(traj)
    assert rep.trust <= 0.5 or rep.corrections != ("commit",)


# ───────────────────────── GEPA reflection ─────────────────────────
def test_gepa_cycle_reflects_measures_and_disposes():
    from tac.witness_control.gepa_reflection import run_gepa_cycle
    traj, _ = _synthetic_traj()
    reports = {a: backtest(traj, architecture=a)[0]
               for a in ("A_ridge_solve", "E_prototype")}
    cyc = run_gepa_cycle(traj, reports, incumbent="E_prototype")
    assert cyc.candidates, "reflection must propose"
    for c in cyc.candidates:
        assert c.status in ("PROPOSED", "ADOPTED", "REFUSED")
        assert c.reflection and any(ch.isdigit() for ch in c.reflection), \
            "reflections must be grounded in measured numbers"
    assert cyc.actuation == "NONE"
    # the frontier is non-dominated over (wf, complexity)
    for c in cyc.frontier:
        assert c.measured.get("wf_mae") is not None


def test_gepa_pareto_frontier_nondominated():
    from tac.witness_control.gepa_reflection import (
        ReflectionCandidate, pareto_frontier)
    a = ReflectionCandidate("a", "r 1", {}, measured={"wf_mae": 0.002}, complexity=100)
    b = ReflectionCandidate("b", "r 2", {}, measured={"wf_mae": 0.003}, complexity=50)
    c = ReflectionCandidate("c", "r 3", {}, measured={"wf_mae": 0.004}, complexity=200)
    front = pareto_frontier([a, b, c])
    names = {x.name for x in front}
    assert names == {"a", "b"}      # c dominated by both


# ───────────────────────── DSL: describe/render + arbitration ─────────────────────────
def test_dsl_describe_and_render():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    prog = derive_costate_agent_v1(str(LIVE_RUN if LIVE_RUN.exists() else "."))
    d = prog.describe()
    assert d["_derived"]["n_actuators_heavy_gated"] >= 4
    assert d["_derived"]["single_best_lens"] == prog.routing.single_best_lens
    lines = prog.render_lines()
    assert any("containment" in ln for ln in lines)
    assert any("routing" in ln for ln in lines)


def test_dsl_arbitrated_variant_maps_to_base_lens(tmp_path, monkeypatch):
    """Tournament VARIANTS (E_prototype_bregman / G_ridge_scorerprior) must route to
    their base panel lens — the silent-fallback gap found in the round-2 attack."""
    import tac.witness_control.continual_costate as cc
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_arbitrated
    traj, _ = _synthetic_traj()
    ledger = tmp_path / "ledger.md"
    reports = _fake_reports(traj)
    # make the Bregman variant the winner
    from tac.witness_control.lambda_net import BacktestReport
    d = reports["A_ridge_solve"].to_dict(); d["notes"] = tuple(d["notes"])
    d.update(architecture="E_prototype_bregman", walkforward_mae_model=0.001,
             passed=True, passed_walkforward=True)
    reports["E_prototype_bregman"] = BacktestReport(**d)
    rec = cc.compose_trajectory_record(traj, reports, [])
    cc.append_trajectory_record(rec, ledger)
    monkeypatch.setattr(cc, "ORGAN_LEDGER", ledger)
    p = derive_costate_agent_arbitrated(".")
    assert p.routing.single_best_lens == "prototype"
    assert "organ-ledger arbitration" in p.routing.provenance


def test_dsl_arbitrated_derivation_fail_open():
    from tac.witness_dsl.costate_agent_dsl import (
        derive_costate_agent_arbitrated, derive_costate_agent_v1)
    p1 = derive_costate_agent_v1(".")
    p2 = derive_costate_agent_arbitrated(".")
    assert p2.name == p1.name       # fail-open / consistent program
    assert p2.routing.mode == p1.routing.mode


def test_panel_single_best_lens_and_spread_gate():
    from tac.witness_control.costate_panel import run_panel
    traj, _ = _synthetic_traj()
    v = run_panel(traj, routing_mode="SINGLE_BEST", single_best_lens="prototype",
                  spread_gate=True, ledger_path="nonexistent.jsonl")
    acts = {r.spec.name: r.activation for r in v.reports if r.spec.predictive}
    assert acts.get("prototype", 0.0) == pytest.approx(1.0)
    # any gate-skipped lens is reported LOUDLY, never silently absent
    for r in v.reports:
        if r.spec.predictive and r.forecast_dxdt is None:
            assert "SKIPPED" in r.insight


def test_compiled_organ_self_monitor_and_reflect_wired():
    from tac.witness_dsl.costate_agent_dsl import derive_costate_agent_v1
    if not LIVE_RUN.exists():
        pytest.skip("live run dir absent")
    organ = derive_costate_agent_v1(str(LIVE_RUN)).compile()
    rep = organ.self_monitor()
    assert 0.0 <= rep.trust <= 1.0
