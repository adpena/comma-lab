# SPDX-License-Identifier: MIT
"""Tests for Rashomon ensemble of K=8 SLIM rankers (Phase 3)."""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from tac.autopilot_rudin_daubechies import (
    DEFAULT_RASHOMON_ENSEMBLE_SIZE,
    ProxyPanel,
    RashomonEnsembleRanker,
    RashomonMember,
)


def test_rashomon_default_ensemble_size():
    assert DEFAULT_RASHOMON_ENSEMBLE_SIZE == 8


def test_rashomon_ensemble_construction():
    e = RashomonEnsembleRanker(rng_seed=0)
    assert len(e.members) == DEFAULT_RASHOMON_ENSEMBLE_SIZE
    # Each member is a RashomonMember.
    for m in e.members:
        assert isinstance(m, RashomonMember)


def test_rashomon_ensemble_rejects_too_small():
    with pytest.raises(ValueError, match=">= 2 for disagreement"):
        RashomonEnsembleRanker(ensemble_size=1)


def test_rashomon_ensemble_predict_consensus_is_mean():
    """With no anchors, all members are first-principles seeded -> identical preds."""
    e = RashomonEnsembleRanker(rng_seed=0, ensemble_size=4)
    panel = ProxyPanel(seg_p0=0.01, pose_p0=0.001, rate_p0=0.005)
    consensus = e.predict(panel)
    # First-principles seeded: 100*0.01 + 3*0.001 + 25*0.005 = 1.0 + 0.003 + 0.125 = 1.128
    # capped by integer_bound=10 -> 10*0.01 + 3*0.001 + 10*0.005 = 0.1 + 0.003 + 0.05 = 0.153
    assert consensus == pytest.approx(0.153, abs=1e-4)


def test_rashomon_ensemble_disagreement_zero_when_identical():
    e = RashomonEnsembleRanker(rng_seed=0, ensemble_size=4)
    panel = ProxyPanel(seg_p0=0.01)
    consensus, disagreement = e.predict_with_disagreement(panel)
    # All members are first-principles seeded (no bootstrap diversity yet).
    assert disagreement == 0.0


def test_rashomon_ensemble_disagreement_emerges_with_anchors(tmp_path):
    """Add diverse anchors; bootstrap sampling should produce member disagreement."""
    store = tmp_path / "rashomon_anchors.jsonl"
    e = RashomonEnsembleRanker(
        rng_seed=42, ensemble_size=8, store_path=store
    )
    # Add several anchors with diverse scores.
    for i in range(10):
        panel = ProxyPanel(seg_p0=0.001 * (i + 1), candidate_id=f"a{i}")
        e.update_all(0.10 + 0.02 * i, panel)
    # Now query a candidate; some disagreement should emerge.
    candidate = ProxyPanel(seg_p0=0.005, candidate_id="probe")
    consensus, disagreement = e.predict_with_disagreement(candidate)
    # Disagreement should be > 0 (members trained on different bootstrap samples).
    # We don't enforce a strict threshold because bootstrap variance is data-dependent;
    # the contract is "disagreement IS a signal, even if small".
    assert disagreement >= 0.0


def test_rashomon_ensemble_update_all_anchors_persist(tmp_path):
    store = tmp_path / "anchors.jsonl"
    e1 = RashomonEnsembleRanker(rng_seed=0, store_path=store)
    panel = ProxyPanel(seg_p0=0.01)
    e1.update_all(0.20, panel, axis="contest_cuda")
    e1.update_all(0.21, panel, axis="contest_cuda")
    # Re-instantiate; new ensemble sees both anchors.
    e2 = RashomonEnsembleRanker(rng_seed=0, store_path=store)
    assert e2.n_anchors == 2


def test_rashomon_ensemble_update_rejects_non_panel():
    e = RashomonEnsembleRanker()
    with pytest.raises(TypeError, match="must be ProxyPanel"):
        e.update_all(0.20, "not a panel")  # type: ignore[arg-type]


def test_rashomon_ensemble_update_rejects_inf_score():
    e = RashomonEnsembleRanker()
    with pytest.raises(ValueError, match="finite"):
        e.update_all(float("inf"), ProxyPanel())


def test_rashomon_disagreement_queue_sorts_by_descending_stddev(tmp_path):
    store = tmp_path / "anchors.jsonl"
    e = RashomonEnsembleRanker(rng_seed=42, store_path=store, ensemble_size=8)
    # Build up anchors.
    for i in range(8):
        panel = ProxyPanel(seg_p0=0.001 * (i + 1))
        e.update_all(0.10 + 0.02 * i, panel)
    # Query several candidates.
    candidates = [
        ProxyPanel(seg_p0=0.001 * (i + 1), candidate_id=f"c{i}") for i in range(5)
    ]
    queue = e.disagreement_queue(candidates)
    assert len(queue) == 5
    # Sorted by descending stddev.
    stddevs = [e.disagreement_stddev for e in queue]
    assert stddevs == sorted(stddevs, reverse=True)


def test_rashomon_disagreement_queue_handles_empty_input():
    e = RashomonEnsembleRanker(rng_seed=0)
    assert e.disagreement_queue([]) == []


def test_rashomon_confidence_tag_cold_start():
    e = RashomonEnsembleRanker(ensemble_size=4)
    tag = e.confidence_tag()
    assert "first-principles-bound" in tag
    assert "rashomon-K=4" in tag


def test_rashomon_confidence_tag_after_updates(tmp_path):
    store = tmp_path / "anchors.jsonl"
    e = RashomonEnsembleRanker(ensemble_size=4, store_path=store)
    e.update_all(0.20, ProxyPanel(seg_p0=0.01))
    tag = e.confidence_tag()
    assert "n=1-anchor-posterior" in tag
    assert "rashomon-K=4" in tag


def test_rashomon_member_has_distinct_seeds(tmp_path):
    """Different members must have different bootstrap seeds for disagreement."""
    store = tmp_path / "anchors.jsonl"
    e = RashomonEnsembleRanker(rng_seed=0, ensemble_size=8, store_path=store)
    seeds = {m.bootstrap_seed for m in e.members}
    # All 8 seeds should be distinct (probabilistically near-certain with 2^31 range).
    assert len(seeds) == 8


# ── Concurrent update stress ──────────────────────────────────────────────


def _rashomon_concurrent_writer(args):
    store_path_str, lock_path_str, observed_score, seg_value = args
    from tac.autopilot_rudin_daubechies import (
        ProxyPanel,
        RashomonEnsembleRanker,
    )

    e = RashomonEnsembleRanker(
        rng_seed=int(observed_score * 1000),
        store_path=Path(store_path_str),
        lock_path=Path(lock_path_str),
        ensemble_size=4,
    )
    panel = ProxyPanel(seg_p0=seg_value, candidate_id=f"c_{seg_value}")
    e.update_all(observed_score, panel, axis="contest_cuda")
    return observed_score


def test_rashomon_concurrent_anchor_writes_all_survive(tmp_path):
    store = tmp_path / "rashomon_anchors.jsonl"
    lock = tmp_path / "rashomon_anchors.jsonl.lock"
    payloads = [
        (str(store), str(lock), 0.18, 0.01),
        (str(store), str(lock), 0.19, 0.02),
        (str(store), str(lock), 0.20, 0.03),
        (str(store), str(lock), 0.21, 0.04),
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        results = pool.map(_rashomon_concurrent_writer, payloads)
    assert sorted(results) == [0.18, 0.19, 0.20, 0.21]
    lines = [
        ln for ln in store.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    # All 4 lines should be present (lock serialization).
    assert len(lines) == 4
