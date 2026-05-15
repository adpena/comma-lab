# SPDX-License-Identifier: MIT
"""Tests for SLIM ranker over Taylor proxies (Phase 1)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from tac.autopilot_rudin_daubechies import (
    DEFAULT_INTEGER_COEFFICIENT_BOUND,
    DEFAULT_SPARSITY_TARGET,
    ProxyPanel,
    SLIMCoefficient,
    SLIMRanker,
    SLIMTrainingError,
    explain_slim_prediction,
)
from tac.autopilot_rudin_daubechies.slim_ranker import (
    CANONICAL_PROXY_NAMES,
)


# ── ProxyPanel ────────────────────────────────────────────────────────────


def test_proxy_panel_default_all_none():
    p = ProxyPanel()
    assert p.seg_p0 is None
    assert p.pose_p2 is None
    assert p.rate_p3 is None
    assert p.panel_axis == "macos_cpu_advisory"


def test_proxy_panel_as_feature_vector_handles_none():
    p = ProxyPanel(seg_p0=0.5, pose_p2=0.012)
    feats = p.as_feature_vector()
    assert len(feats) == len(CANONICAL_PROXY_NAMES)
    # seg_p0 is index 0 and pose_p2 is index 6 in the canonical order.
    assert feats[0] == 0.5
    assert feats[6] == 0.012
    # Missing values are zero.
    assert feats[1] == 0.0
    assert feats[2] == 0.0


def test_proxy_panel_as_feature_vector_filters_nan_and_inf():
    p = ProxyPanel(seg_p0=float("nan"), pose_p0=float("inf"))
    feats = p.as_feature_vector()
    # Non-finite values are coerced to zero (consistent with missing).
    assert feats[0] == 0.0
    assert feats[4] == 0.0


def test_proxy_panel_custom_feature_order():
    p = ProxyPanel(rate_p0=0.05)
    feats = p.as_feature_vector(feature_order=("rate_p0", "rate_p1"))
    assert feats == [0.05, 0.0]


def test_proxy_panel_axis_label_preserved():
    p = ProxyPanel(panel_axis="contest_cuda")
    assert p.panel_axis == "contest_cuda"


# ── SLIMCoefficient invariants ────────────────────────────────────────────


def test_slim_coefficient_rejects_non_int():
    with pytest.raises(SLIMTrainingError, match="must be an int"):
        SLIMCoefficient(proxy_name="seg_p0", integer_coef=1.5, bound=10)  # type: ignore[arg-type]


def test_slim_coefficient_rejects_bool_disguised_as_int():
    with pytest.raises(SLIMTrainingError, match="must be an int"):
        SLIMCoefficient(proxy_name="seg_p0", integer_coef=True, bound=10)


def test_slim_coefficient_rejects_zero_bound():
    with pytest.raises(SLIMTrainingError, match="bound must be int >= 1"):
        SLIMCoefficient(proxy_name="seg_p0", integer_coef=0, bound=0)


def test_slim_coefficient_rejects_out_of_bound():
    with pytest.raises(SLIMTrainingError, match="violates"):
        SLIMCoefficient(proxy_name="seg_p0", integer_coef=11, bound=10)


def test_slim_coefficient_accepts_valid():
    c = SLIMCoefficient(proxy_name="rate_p3", integer_coef=-7, bound=10)
    assert c.integer_coef == -7
    assert c.bound == 10


# ── SLIMRanker construction ───────────────────────────────────────────────


def test_slim_ranker_default_first_principles_seed():
    r = SLIMRanker()
    # Cold-start should seed seg_p0 / pose_p0 / rate_p0.
    coef_names = {c.proxy_name for c in r.coefficients}
    # sparsity_target=5 by default, so all 3 first-principles seeds fit.
    assert "seg_p0" in coef_names
    assert "pose_p0" in coef_names
    assert "rate_p0" in coef_names


def test_slim_ranker_seed_coefficients_match_scorer_formula():
    r = SLIMRanker(integer_bound=100, sparsity_target=3)
    coefs = {c.proxy_name: c.integer_coef for c in r.coefficients}
    # 100 * S_seg cap at integer_bound=100.
    assert coefs["seg_p0"] == 100
    # sqrt(10) ≈ 3.16; nearest int = 3.
    assert coefs["pose_p0"] == 3
    # 25 * R cap at integer_bound=100 -> 25.
    assert coefs["rate_p0"] == 25


def test_slim_ranker_rejects_zero_integer_bound():
    with pytest.raises(SLIMTrainingError, match="integer_bound"):
        SLIMRanker(integer_bound=0)


def test_slim_ranker_rejects_zero_sparsity():
    with pytest.raises(SLIMTrainingError, match="sparsity_target"):
        SLIMRanker(sparsity_target=0)


def test_slim_ranker_rejects_oversparse():
    with pytest.raises(SLIMTrainingError, match="exceeds feature_count"):
        SLIMRanker(sparsity_target=100)


def test_slim_ranker_rejects_empty_feature_order():
    with pytest.raises(SLIMTrainingError, match="non-empty"):
        SLIMRanker(feature_order=())


# ── SLIMRanker prediction ─────────────────────────────────────────────────


def test_slim_ranker_cold_start_predict_matches_seeds():
    r = SLIMRanker(integer_bound=100, sparsity_target=3)
    panel = ProxyPanel(seg_p0=0.001, pose_p0=0.0001, rate_p0=0.005)
    pred = r.predict(panel)
    # 0 + 100*0.001 + 3*0.0001 + 25*0.005 = 0.1 + 0.0003 + 0.125 = 0.2253
    assert pred == pytest.approx(0.2253, abs=1e-4)


def test_slim_ranker_predict_zero_panel():
    r = SLIMRanker()
    pred = r.predict(ProxyPanel())
    assert pred == 0.0


def test_slim_ranker_predict_ignores_unselected_proxy():
    r = SLIMRanker(integer_bound=10, sparsity_target=2)
    # sparsity_target=2 means only seg_p0 + pose_p0 are seeded; rate_p0 is dropped.
    panel = ProxyPanel(seg_p0=0.01, pose_p0=0.001, rate_p0=999.0)
    # Pred ignores rate_p0 (0 coef).
    pred = r.predict(panel)
    # 0 + 10*0.01 + 3*0.001 = 0.103
    assert pred == pytest.approx(0.103, abs=1e-4)


# ── SLIMRanker continual learning ─────────────────────────────────────────


def test_slim_ranker_update_from_anchor_refits():
    r = SLIMRanker(integer_bound=10, sparsity_target=3, rng_seed=42)
    panel = ProxyPanel(seg_p0=0.01, pose_p0=0.001, rate_p0=0.005)
    r.update_from_anchor(0.20, panel, axis="contest_cuda")
    assert r.n_anchors == 1
    # Confidence tag updates.
    assert "n=1-anchor-posterior" in r.confidence_tag()


def test_slim_ranker_update_from_anchor_rejects_non_panel():
    r = SLIMRanker()
    with pytest.raises(TypeError, match="must be ProxyPanel"):
        r.update_from_anchor(0.20, {"seg_p0": 0.01})  # type: ignore[arg-type]


def test_slim_ranker_update_from_anchor_rejects_inf_score():
    r = SLIMRanker()
    panel = ProxyPanel(seg_p0=0.01)
    with pytest.raises(SLIMTrainingError, match="finite"):
        r.update_from_anchor(float("inf"), panel)


def test_slim_ranker_update_improves_fit_with_more_anchors(tmp_path):
    store = tmp_path / "slim_store.jsonl"
    r = SLIMRanker(
        integer_bound=10,
        sparsity_target=3,
        rng_seed=0,
        store_path=store,
    )
    # Synthetic anchors that a 2-coef SLIM can fit perfectly:
    # observed = 5 * seg_p0 + 2 * rate_p0
    for seg, rate in [(0.01, 0.05), (0.02, 0.03), (0.005, 0.10), (0.03, 0.02)]:
        panel = ProxyPanel(seg_p0=seg, rate_p0=rate)
        score = 5 * seg + 2 * rate
        r.update_from_anchor(score, panel)
    # After refits the ranker should be near-perfect on the synthetic data.
    pred = r.predict(ProxyPanel(seg_p0=0.01, rate_p0=0.05))
    assert abs(pred - (5 * 0.01 + 2 * 0.05)) < 0.05  # tolerance for integer rounding


def test_slim_ranker_persistence_round_trip(tmp_path):
    store = tmp_path / "slim_store.jsonl"
    r1 = SLIMRanker(rng_seed=0, store_path=store)
    panel = ProxyPanel(seg_p0=0.01, pose_p0=0.001, rate_p0=0.005)
    r1.update_from_anchor(0.18, panel, axis="contest_cuda")
    r1.update_from_anchor(0.19, panel, axis="contest_cuda")
    # Re-instantiate; the new ranker should see both anchors.
    r2 = SLIMRanker(rng_seed=0, store_path=store)
    assert r2.n_anchors == 2


def test_slim_ranker_jsonl_corruption_does_not_silently_zero_pool(tmp_path):
    store = tmp_path / "slim_store.jsonl"
    r = SLIMRanker(rng_seed=0, store_path=store)
    panel = ProxyPanel(seg_p0=0.01, rate_p0=0.005)
    r.update_from_anchor(0.20, panel)
    # Simulate corruption: append a malformed line.
    with store.open("a") as fh:
        fh.write("not json {{{\n")
    # New ranker should ignore the corrupt line, still see the valid one.
    r2 = SLIMRanker(rng_seed=0, store_path=store)
    assert r2.n_anchors == 1


# ── SLIMRanker fcntl concurrency ──────────────────────────────────────────


def _concurrent_writer_worker(args):
    store_path_str, lock_path_str, observed_score, seg_value = args
    from tac.autopilot_rudin_daubechies import ProxyPanel, SLIMRanker

    r = SLIMRanker(
        rng_seed=int(observed_score * 1000),
        store_path=Path(store_path_str),
        lock_path=Path(lock_path_str),
    )
    panel = ProxyPanel(seg_p0=seg_value, candidate_id=f"cand_{seg_value}")
    r.update_from_anchor(observed_score, panel, axis="contest_cuda")
    return observed_score


def test_slim_ranker_concurrent_anchor_writes_all_survive(tmp_path):
    """4-worker stress: all 4 distinct anchors land per Catalog #128/#131."""
    store = tmp_path / "slim_store.jsonl"
    lock = tmp_path / "slim_store.jsonl.lock"
    payloads = [
        (str(store), str(lock), 0.18, 0.01),
        (str(store), str(lock), 0.19, 0.02),
        (str(store), str(lock), 0.20, 0.03),
        (str(store), str(lock), 0.21, 0.04),
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_concurrent_writer_worker, payloads)
    assert sorted(results) == [0.18, 0.19, 0.20, 0.21]
    # All 4 lines made it to disk.
    lines = [
        ln for ln in store.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) == 4
    observed = sorted(json.loads(ln)["observed_score"] for ln in lines)
    assert observed == [0.18, 0.19, 0.20, 0.21]


# ── SLIMRanker explanation surface ────────────────────────────────────────


def test_explain_slim_prediction_returns_readable_chain():
    r = SLIMRanker(integer_bound=10, sparsity_target=3)
    panel = ProxyPanel(seg_p0=0.01, pose_p0=0.001, rate_p0=0.005)
    expl = explain_slim_prediction(r, panel)
    assert "predicted_score" in expl
    assert "intercept" in expl
    assert "seg_p0" in expl
    assert "rate_p0" in expl


def test_slim_ranker_explain_method_alias():
    r = SLIMRanker()
    panel = ProxyPanel(seg_p0=0.01)
    expl = r.explain(panel)
    assert "predicted_score" in expl


def test_explain_slim_prediction_handles_missing_proxy():
    r = SLIMRanker(integer_bound=10, sparsity_target=3)
    panel = ProxyPanel()  # all None
    expl = explain_slim_prediction(r, panel)
    assert "None" in expl  # missing proxies surface explicitly


# ── SLIMRanker confidence tag ─────────────────────────────────────────────


def test_confidence_tag_cold_start():
    r = SLIMRanker()
    assert r.confidence_tag() == "[prediction; first-principles-bound]"


def test_confidence_tag_after_update():
    r = SLIMRanker()
    r.update_from_anchor(0.20, ProxyPanel(seg_p0=0.01))
    assert "n=1-anchor-posterior" in r.confidence_tag()
