# SPDX-License-Identifier: MIT
"""Pins for ``tools/ddm_wc2_shard_solve.py`` -- the ddm_jg3 wall-clock shard.

The arm was written to prove sharding is DECISION-IDENTICAL.  Its own replay
falsified that (see the tool docstring and the arm memo): fresh processes
reproduce each other exactly but do not reproduce a long-running one, because
the solver screens at the argmax decision boundary where every site is a
near-tie.  So these pins cover what actually decides admissibility:

* the partition arithmetic -- which pairs a shard owns, balanced on predicted
  cost, and visited in permutation order so no shard prefix is biased;
* the decision comparator -- which must NOT report equivalence vacuously;
* the yield comparator -- scored in net delta S, the solver's own objective,
  because repair is bought with tokens and tokens cost rate;
* the committed admissibility receipt, so the arm's warrant cannot go vacuous
  when the SSD-tier run artifacts are absent from a given host.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import ddm_wc2_shard_solve as wc2  # noqa: E402

# The live run's own log recorded this visit order.  Reproducing it is what
# licenses the orchestrator to name the REMAINING pairs instead of guessing.
LIVE_FIRST_12 = [281, 230, 34, 378, 101, 306, 395, 383, 300, 351, 224, 405]

RECEIPT = REPO / ".omx" / "research" / "ddm_wc2_equivalence_receipt_20260819.json"


def _row(pair: int, **over):
    row = {
        "pair": pair,
        "flips_before": 40,
        "flips_after": 30,
        "repaired": 10,
        "tokens_changed": 8,
        "screened_candidates": 100,
        "evaluations": 200,
        "packing_residual_max": 2,
        "rejected_for_separation": 1,
        "accept_separation_chosen": 64,
        "keep_fraction_chosen": 1.0,
        "accepted": [[1, 2, 3], [4, 5, 6]],
        "separation_sweep": [
            {
                "accept_separation": 64,
                "keep_fraction": 1.0,
                "tokens": 8,
                "flips_after": 30,
                "repaired": 10,
                "net_delta_S": -1e-6,
            }
        ],
        "seconds": 123.4,
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# The visit order
# ---------------------------------------------------------------------------


def test_visit_permutation_reproduces_the_live_run_order():
    perm = wc2.visit_permutation()
    assert perm[:12].tolist() == LIVE_FIRST_12
    assert len(perm) == 600
    assert sorted(perm.tolist()) == list(range(600))


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_plan_shards_partitions_exactly_and_excludes_done():
    perm = list(range(20))
    done = {3, 7, 11}
    lists = wc2.plan_shards(perm, done, 4)
    flat = [p for bucket in lists for p in bucket]
    assert sorted(flat) == sorted(set(perm) - done)
    assert len(flat) == len(set(flat)), "a pair must not be solved twice"
    assert not (set(flat) & done), "a completed pair must not be re-solved"


def test_plan_shards_is_never_a_contiguous_block():
    lists = wc2.plan_shards(list(range(60)), (), 3)
    for bucket in lists:
        contiguous = bucket == list(range(bucket[0], bucket[0] + len(bucket)))
        assert not contiguous, "contiguous blocks re-introduce the bp2/na2 prefix bias"


def test_plan_shards_weighted_balances_load_far_better_than_stride():
    # The measured shape is a heavy tail (55.7 s to 657.0 s per pair).  The tail
    # is placed on positions CONGRUENT to the shard count, which is precisely
    # where a stride interleave collapses: every heavy pair lands in one bucket
    # and the fleet then waits on that shard.  Weighted assignment must not care.
    perm = list(range(60))
    heavy = {0, 3, 6, 9, 12, 15}
    weights = {p: (100.0 if p in heavy else 1.0) for p in perm}
    balanced = wc2.plan_shards(perm, (), 3, weights)
    stride = wc2.plan_shards(perm, (), 3)

    def spread(buckets):
        loads = [sum(weights[p] for p in b) for b in buckets]
        return max(loads) / (sum(loads) / len(loads))

    assert spread(stride) > 2.0, "the fixture must actually defeat stride"
    assert spread(balanced) < 1.05
    assert spread(balanced) < spread(stride)


def test_plan_shards_weighted_still_visits_in_permutation_order():
    # LPT emits heaviest-first; a heaviest-first prefix is the biased-population
    # shape.  The planner must restore permutation order inside each shard.
    perm = [5, 1, 9, 3, 7, 2, 8, 4, 6, 0]
    weights = {p: float(p) for p in perm}
    rank = {p: i for i, p in enumerate(perm)}
    for bucket in wc2.plan_shards(perm, (), 3, weights):
        ranks = [rank[p] for p in bucket]
        assert ranks == sorted(ranks), "shard prefix must stay an unbiased sample"


def test_plan_shards_refuses_zero_shards():
    with pytest.raises(wc2.ShardError):
        wc2.plan_shards([1, 2, 3], (), 0)


# ---------------------------------------------------------------------------
# The decision signature
# ---------------------------------------------------------------------------


def test_decision_signature_ignores_wall_clock_only():
    assert wc2.decision_signature(_row(1)) == wc2.decision_signature(
        _row(1, seconds=999.9)
    )


def test_decision_signature_is_insensitive_to_accepted_ordering():
    a = _row(1, accepted=[[1, 2, 3], [4, 5, 6]])
    b = _row(1, accepted=[[4, 5, 6], [1, 2, 3]])
    assert wc2.decision_signature(a) == wc2.decision_signature(b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted", [[1, 2, 4]]),
        ("flips_after", 31),
        ("repaired", 9),
        ("tokens_changed", 7),
        ("accept_separation_chosen", 32),
        ("keep_fraction_chosen", 0.5),
        ("packing_residual_max", 3),
        ("evaluations", 201),
    ],
)
def test_decision_signature_changes_when_a_decision_changes(field, value):
    assert wc2.decision_signature(_row(1)) != wc2.decision_signature(
        _row(1, **{field: value})
    )


def test_decision_signature_catches_a_changed_sweep_score():
    changed = _row(1)
    changed["separation_sweep"] = [
        {**changed["separation_sweep"][0], "net_delta_S": -2e-6}
    ]
    assert wc2.decision_signature(_row(1)) != wc2.decision_signature(changed)


# ---------------------------------------------------------------------------
# The comparator
# ---------------------------------------------------------------------------


def test_compare_decisions_equivalent_when_only_timing_differs():
    ref = {1: _row(1), 2: _row(2)}
    replay = {1: _row(1, seconds=1.0), 2: _row(2, seconds=2.0)}
    report = wc2.compare_decisions(ref, replay)
    assert report["equivalent"] is True
    assert report["compared"] == 2
    assert report["mismatches"] == []


def test_compare_decisions_names_the_differing_field():
    ref = {1: _row(1)}
    replay = {1: _row(1, accepted=[[9, 9, 9]])}
    report = wc2.compare_decisions(ref, replay)
    assert report["equivalent"] is False
    assert report["mismatches"][0]["fields"] == ["accepted"]


def test_compare_decisions_is_not_vacuously_true_on_no_overlap():
    # The silent-instrument failure: nothing compared must never read as PASS.
    report = wc2.compare_decisions({1: _row(1)}, {2: _row(2)})
    assert report["compared"] == 0
    assert report["equivalent"] is False


# ---------------------------------------------------------------------------
# The yield comparator -- the metric that decides admissibility
# ---------------------------------------------------------------------------


def test_yield_comparison_reports_no_bias_when_deltas_cancel():
    ref = {p: _row(p, repaired=10, tokens_changed=8) for p in range(1, 9)}
    replay = {
        p: _row(p, repaired=10 + (1 if p % 2 else -1), tokens_changed=8)
        for p in range(1, 9)
    }  # tokens held equal so the sign is carried by repair alone
    out = wc2.yield_comparison(ref, replay)
    assert out["repaired_delta"] == 0
    assert out["replay_wins"] == 4 and out["replay_losses"] == 4
    assert out["biased_at_p05"] is False


def test_yield_comparison_scores_in_net_delta_S_not_repair_alone():
    # Fewer repairs bought with far fewer tokens is a BETTER score contribution.
    # A repair-only comparator would call this a loss; the score says it is a win.
    ref = {1: _row(1, repaired=18, tokens_changed=20)}
    replay = {1: _row(1, repaired=19, tokens_changed=16)}
    out = wc2.yield_comparison(ref, replay)
    assert out["replay_wins"] == 1
    assert out["net_delta_S_shard_advantage"] > 0


def test_yield_comparison_detects_a_systematic_penalty():
    # Every pair strictly worse: the sign test must call this biased.
    ref = {p: _row(p, repaired=100, tokens_changed=50) for p in range(1, 11)}
    replay = {p: _row(p, repaired=97, tokens_changed=50) for p in range(1, 11)}
    out = wc2.yield_comparison(ref, replay)
    assert out["repaired_delta"] == -30
    assert out["replay_losses"] == 10 and out["replay_wins"] == 0
    assert out["sign_test_p_value"] < 0.05
    assert out["biased_at_p05"] is True


def test_yield_comparison_ties_do_not_manufacture_significance():
    ref = {p: _row(p, repaired=10) for p in range(1, 21)}
    replay = {p: _row(p, repaired=10) for p in range(1, 21)}
    out = wc2.yield_comparison(ref, replay)
    assert out["ties"] == 20
    assert out["sign_test_p_value"] == 1.0
    assert out["biased_at_p05"] is False


def test_yield_comparison_reports_the_denominator():
    out = wc2.yield_comparison({}, {})
    assert out["pairs_compared"] == 0
    assert out["biased_at_p05"] is False


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


def test_merge_checkpoints_unions_disjoint_shards():
    merged, conflicts = wc2.merge_checkpoints(
        {"s0": {1: _row(1), 3: _row(3)}, "s1": {2: _row(2)}}
    )
    assert sorted(merged) == [1, 2, 3]
    assert conflicts == []


def test_merge_checkpoints_accepts_an_identical_duplicate():
    merged, conflicts = wc2.merge_checkpoints(
        {"s0": {1: _row(1)}, "s1": {1: _row(1, seconds=7.0)}}
    )
    assert conflicts == []
    assert sorted(merged) == [1]


def test_merge_checkpoints_refuses_to_pick_a_winner_on_conflict():
    merged, conflicts = wc2.merge_checkpoints(
        {"s0": {1: _row(1)}, "s1": {1: _row(1, accepted=[[9, 9, 9]])}}
    )
    assert len(conflicts) == 1
    assert conflicts[0]["pair"] == 1
    assert conflicts[0]["sources"] == ["s0", "s1"]


# ---------------------------------------------------------------------------
# The checkpoint reader
# ---------------------------------------------------------------------------


def test_read_checkpoint_drops_a_truncated_final_line(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text(
        json.dumps(_row(1)) + "\n" + json.dumps(_row(2)) + "\n" + '{"pair": 3, "fli'
    )
    rows = wc2.read_checkpoint(path)
    assert sorted(rows) == [1, 2], "a partial row is not a completed pair"


def test_read_checkpoint_missing_file_is_empty(tmp_path):
    assert wc2.read_checkpoint(tmp_path / "absent.jsonl") == {}


# ---------------------------------------------------------------------------
# The shard invocation
# ---------------------------------------------------------------------------


def test_shard_argv_carries_the_live_seed_and_pair_list():
    argv = wc2.shard_argv(Path("/store"), "n600_wc2s0", [5, 9, 1])
    # The seed feeds ``site_seed``, and ``solve_pair`` draws its site subsample
    # from ``default_rng(site_seed + pair)``.  A different seed here would be a
    # different instrument, not a shard.
    assert argv[argv.index("--seed") + 1] == str(wc2.LIVE_SEED)
    assert argv[argv.index("--pair-list") + 1] == "5,9,1"
    assert argv[argv.index("--tag") + 1] == "n600_wc2s0"
    assert "--resume" in argv


# ---------------------------------------------------------------------------
# The committed equivalence receipt from the real replay
# ---------------------------------------------------------------------------


def test_committed_admissibility_receipt_carries_its_denominator():
    assert RECEIPT.exists(), (
        "the admissibility receipt is the arm's whole warrant; it is committed "
        "so this pin can never pass vacuously"
    )
    report = json.loads(RECEIPT.read_text())
    # Decision-identity is FALSIFIED and the receipt must say so, not claim it.
    assert report["verdict"] == "NOT_DECISION_IDENTICAL_BIAS_MEASURED"
    assert report["equivalent"] is False
    yield_report = report["yield"]
    # The DENOMINATOR must be present and non-empty: a comparison over zero pairs
    # is the silent-instrument failure, where skip reads as green.
    assert yield_report["pairs_compared"] > 0

    # Whatever the sign, the receipt must carry the sign test and the bound so a
    # reader can price the trade instead of taking a verdict on trust.
    assert "sign_test_p_value" in yield_report
    assert "net_delta_S_shard_advantage" in yield_report
    assert "projected_full_run_delta_S" in report
    assert report["projection_is_extrapolation_not_measurement"] is True

    # THE HONESTY INVARIANT, and the reason this pin is not a sample-size ritual:
    # the receipt may never assert a bias the sign test does not support.  This
    # binds at any n, where a fixed threshold would only have bound at one.
    if yield_report["sign_test_p_value"] >= 0.05:
        assert yield_report["biased_at_p05"] is False, (
            "receipt claims a bias its own sign test does not establish"
        )
