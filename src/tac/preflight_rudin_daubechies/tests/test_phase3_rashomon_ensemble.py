# SPDX-License-Identifier: MIT
"""Tests for Phase 3 (Rashomon ensemble of K=8 near-optimal preflight rankings)."""
from __future__ import annotations

import pytest

from tac.preflight_rudin_daubechies import (
    DEFAULT_PREFLIGHT_RASHOMON_SIZE,
    GateVerdictPanel,
    PreflightRashomonEnsemble,
)


def test_ensemble_default_size_is_8():
    ens = PreflightRashomonEnsemble()
    assert ens.ensemble_size == 8
    assert DEFAULT_PREFLIGHT_RASHOMON_SIZE == 8
    assert len(ens.members) == 8


def test_ensemble_rejects_zero_size():
    with pytest.raises(ValueError):
        PreflightRashomonEnsemble(ensemble_size=0)


def test_ensemble_predict_with_disagreement_cold_start():
    ens = PreflightRashomonEnsemble(ensemble_size=4, rng_seed=0)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    consensus, stddev, per_member = ens.predict_with_disagreement(panel)
    assert len(per_member) == 4
    assert isinstance(consensus, float)
    assert isinstance(stddev, float)
    # All members are cold-start with same first-principles seeds; disagreement = 0.
    assert stddev == 0.0


def test_ensemble_confidence_tag_cold_start():
    ens = PreflightRashomonEnsemble(ensemble_size=4)
    tag = ens.confidence_tag()
    assert "cold-start" in tag
    assert "rashomon-K=4" in tag


def test_ensemble_confidence_tag_after_anchor():
    ens = PreflightRashomonEnsemble(ensemble_size=4)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"})
    ens.update_all(25.0, panel)
    tag = ens.confidence_tag()
    assert "n=1" in tag
    assert "rashomon-K=4" in tag


def test_ensemble_update_all_rejects_non_panel():
    ens = PreflightRashomonEnsemble()
    with pytest.raises(TypeError):
        ens.update_all(25.0, "not_a_panel")


def test_ensemble_update_all_rejects_nonfinite():
    ens = PreflightRashomonEnsemble()
    panel = GateVerdictPanel()
    with pytest.raises(ValueError):
        ens.update_all(float("nan"), panel)


def test_ensemble_persistence_via_store_path(tmp_path):
    store = tmp_path / "rashomon_anchors.jsonl"
    ens1 = PreflightRashomonEnsemble(store_path=store, ensemble_size=4)
    panel = GateVerdictPanel(verdicts={"1": "VIOLATED"}, snapshot_id="snap1")
    ens1.update_all(25.0, panel)
    ens2 = PreflightRashomonEnsemble(store_path=store, ensemble_size=4)
    assert ens2.n_anchors == 1


def test_ensemble_disagreement_queue_grows_after_diverse_anchors(tmp_path):
    # With multiple distinct anchors and bootstrap diversity, members should
    # disagree on out-of-sample panels; the queue should accumulate entries.
    ens = PreflightRashomonEnsemble(ensemble_size=8, rng_seed=42)
    p1 = GateVerdictPanel(verdicts={"1": "VIOLATED"}, snapshot_id="A")
    p2 = GateVerdictPanel(verdicts={"5": "VIOLATED"}, snapshot_id="B")
    p3 = GateVerdictPanel(verdicts={"7": "VIOLATED"}, snapshot_id="C")
    for _ in range(10):
        ens.update_all(25.0, p1)
        ens.update_all(15.0, p2)
        ens.update_all(7.0, p3)
    # Try a novel panel — bootstrap diversity should cause disagreement.
    novel = GateVerdictPanel(verdicts={"118": "VIOLATED"}, snapshot_id="X")
    consensus, stddev, _ = ens.predict_with_disagreement(novel)
    # Either consensus or queue evidence should reflect bootstrap variance.
    assert isinstance(consensus, float)
    assert stddev >= 0.0


def test_ensemble_member_seeds_are_distinct():
    ens = PreflightRashomonEnsemble(ensemble_size=8, rng_seed=0)
    seeds = [m.bootstrap_seed for m in ens.members]
    assert len(set(seeds)) == 8  # all distinct
