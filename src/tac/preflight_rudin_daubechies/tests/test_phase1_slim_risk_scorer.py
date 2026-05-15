# SPDX-License-Identifier: MIT
"""Tests for Phase 1 (SLIM risk scorer over preflight gate verdicts)."""
from __future__ import annotations

import json

import pytest

from tac.preflight_rudin_daubechies import (
    DEFAULT_PREFLIGHT_SPARSITY_TARGET,
    GateVerdictPanel,
    PreflightSLIMRiskScorer,
    PreflightSLIMTrainingError,
    explain_preflight_risk_prediction,
    predict_dispatch_risk_score_with_rationale,
)


def test_gate_verdict_panel_empty_default():
    panel = GateVerdictPanel()
    assert panel.violated_gate_numbers() == ()
    assert panel.scope_axis == "changed-file-subset"


def test_gate_verdict_panel_violated_gate_numbers_extracts():
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED", "5": "PASSED", "146": "VIOLATED"})
    assert panel.violated_gate_numbers() == (1, 146)


def test_gate_verdict_panel_violated_gate_numbers_skips_non_int():
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED", "not_a_number": "VIOLATED"})
    assert panel.violated_gate_numbers() == (1,)


def test_gate_verdict_panel_as_feature_vector():
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED", "5": "PASSED"})
    fv = panel.as_feature_vector(gate_order=[1, 5, 7])
    assert fv == [1.0, 0.0, 0.0]


def test_scorer_rejects_zero_integer_bound():
    with pytest.raises(PreflightSLIMTrainingError):
        PreflightSLIMRiskScorer(integer_bound=0)


def test_scorer_rejects_zero_sparsity():
    with pytest.raises(PreflightSLIMTrainingError):
        PreflightSLIMRiskScorer(sparsity_target=0)


def test_scorer_rejects_oversparse():
    with pytest.raises(PreflightSLIMTrainingError):
        PreflightSLIMRiskScorer(sparsity_target=10000, feature_order=[1, 2])


def test_scorer_cold_start_seeds_meta_meta_first():
    scorer = PreflightSLIMRiskScorer()
    coefs = scorer.coefficients
    assert len(coefs) == DEFAULT_PREFLIGHT_SPARSITY_TARGET
    # META-meta gates (118, 159, 176, 185, 235) should appear before Tier-1 in seeds.
    proxy_names = [int(c.proxy_name) for c in coefs]
    meta_meta_set = {118, 159, 176, 185, 235}
    # The first 5 seeds (sparsity=8, 5 META-meta) should all be META-meta.
    seeded_meta_meta = [pn for pn in proxy_names if pn in meta_meta_set]
    assert len(seeded_meta_meta) == 5
    # Their coefficients should be 50 (META-meta risk).
    for c in coefs:
        if int(c.proxy_name) in meta_meta_set:
            assert c.integer_coef == 50


def test_scorer_predict_violated_meta_meta_returns_50():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel(verdicts={"118": "VIOLATED"})
    risk = scorer.predict(panel)
    # gate 118 is META-meta with seed 50; intercept defaults to 0.
    assert risk == 50.0


def test_scorer_predict_passed_panel_returns_intercept():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel(verdicts={"1": "PASSED", "146": "PASSED"})
    risk = scorer.predict(panel)
    # No violations -> only intercept contributes.
    assert risk == float(scorer.intercept)


def test_confidence_tag_cold_start():
    scorer = PreflightSLIMRiskScorer()
    assert "cold-start" in scorer.confidence_tag()


def test_confidence_tag_after_anchor():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    scorer.update_from_anchor(25.0, panel)
    tag = scorer.confidence_tag()
    assert "n=1-anchor-posterior" in tag


def test_update_from_anchor_rejects_non_panel():
    scorer = PreflightSLIMRiskScorer()
    with pytest.raises(TypeError):
        scorer.update_from_anchor(25.0, "not_a_panel")


def test_update_from_anchor_rejects_nonfinite():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel()
    with pytest.raises(PreflightSLIMTrainingError):
        scorer.update_from_anchor(float("inf"), panel)


def test_update_from_anchor_persists_to_store(tmp_path):
    store = tmp_path / "anchors.jsonl"
    scorer = PreflightSLIMRiskScorer(store_path=store)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"}, snapshot_id="snap1")
    scorer.update_from_anchor(25.0, panel)
    raw = store.read_text(encoding="utf-8").strip()
    rec = json.loads(raw)
    assert rec["observed_dispatch_risk"] == 25.0
    assert rec["panel"]["snapshot_id"] == "snap1"


def test_explain_includes_intercept_and_violated_gates():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    expl = explain_preflight_risk_prediction(scorer, panel)
    assert "predicted_dispatch_risk" in expl
    assert "intercept(" in expl
    assert "gate_1(VIOLATED)" in expl


def test_predict_dispatch_risk_score_with_rationale_returns_REFUSE_above_threshold():
    scorer = PreflightSLIMRiskScorer()
    # Trigger 3 META-meta violations to push above 50 threshold.
    panel = GateVerdictPanel(verdicts={"118": "VIOLATED", "159": "VIOLATED", "176": "VIOLATED"})
    risk, rationale, verdict = predict_dispatch_risk_score_with_rationale(panel, scorer=scorer)
    assert risk == 150.0  # 3 * 50
    assert verdict == "REFUSE"
    assert "rule chain:" in rationale


def test_predict_dispatch_risk_score_with_rationale_returns_OK_below_25_pct():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel()  # no violations
    risk, rationale, verdict = predict_dispatch_risk_score_with_rationale(panel, scorer=scorer)
    assert risk == 0.0
    assert verdict == "OK"


def test_predict_dispatch_risk_score_with_rationale_returns_WARN_at_25():
    scorer = PreflightSLIMRiskScorer()
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    risk, rationale, verdict = predict_dispatch_risk_score_with_rationale(panel, scorer=scorer)
    assert risk == 25.0
    assert verdict == "WARN"


def test_continual_learning_loop_refits_after_anchors(tmp_path):
    store = tmp_path / "anchors.jsonl"
    scorer = PreflightSLIMRiskScorer(store_path=store)
    panel1 = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    panel2 = GateVerdictPanel(verdicts={"5": "VIOLATED"})
    # Add 5 anchors that drive the empirical posterior.
    for _ in range(5):
        scorer.update_from_anchor(25.0, panel1)
        scorer.update_from_anchor(15.0, panel2)
    assert scorer.n_anchors == 10
    # Now the scorer should produce posterior-grade predictions.
    assert scorer.confidence_tag().startswith("[preflight-risk; n=10")


def test_load_from_store_recovers_anchors(tmp_path):
    store = tmp_path / "anchors.jsonl"
    scorer1 = PreflightSLIMRiskScorer(store_path=store)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    scorer1.update_from_anchor(25.0, panel)
    # Build a second scorer pointed at the same store.
    scorer2 = PreflightSLIMRiskScorer(store_path=store)
    assert scorer2.n_anchors == 1
