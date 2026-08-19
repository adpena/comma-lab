"""Tests for ``experiments/ddm_jg3_joint_solve.py``.

These test BEHAVIOUR, not constants.  The forbidden shape here is
"assert BITS_PER_SEG_CELL == 10.18" -- that verifies a literal, and a literal that
is wrong stays wrong.  So the score constants are re-derived from the contest
scoring function independently of the module, and the packing is checked by
constructing site sets whose correct partition is known by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments"))

jg3 = pytest.importorskip("ddm_jg3_joint_solve")


# ---------------------------------------------------------------------------
# The scored arithmetic, re-derived rather than re-quoted
# ---------------------------------------------------------------------------


def test_seg_cell_and_byte_are_derived_from_the_contest_score():
    """S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489, and nothing else."""
    cells = 600 * 384 * 512
    assert pytest.approx(100.0 / cells, rel=1e-12) == jg3.S_PER_SEG_CELL
    assert pytest.approx(25.0 / 37_545_489, rel=1e-12) == jg3.S_PER_ARCHIVE_BYTE
    # One repaired cell is worth this many bytes; ddm_jg1 S0 published 1.273.
    assert pytest.approx(1.273, abs=5e-4) == jg3.BYTES_PER_SEG_CELL
    assert pytest.approx(8 * jg3.BYTES_PER_SEG_CELL, rel=1e-12) == jg3.BITS_PER_SEG_CELL


def test_break_even_yield_is_below_jg1_iterated_yield():
    """The measured fact that makes the stopping rule load-bearing.

    ``ddm_jg1`` S1e measured an 8-pass iterated yield of **0.390** cells/token.
    If that is below break-even then iterating to exhaustion makes the score WORSE,
    which is why this solver is a first pass with a Lagrangian test rather than a
    descent to convergence.
    """
    break_even = jg3.break_even_yield()
    assert break_even == pytest.approx(
        jg3.RATE_PRIOR_BITS_PER_TOKEN / jg3.BITS_PER_SEG_CELL, rel=1e-12
    )
    assert 0.40 < break_even < 0.41
    assert break_even > 0.390, "jg1's iterated yield must sit BELOW break-even"
    assert break_even < 1.55, "jg1's first-pass yield must sit ABOVE break-even"


# ---------------------------------------------------------------------------
# The packing, which is what makes an n600 solve affordable at all
# ---------------------------------------------------------------------------


def test_independent_batches_respect_separation_and_lose_nothing():
    rng = np.random.default_rng(7)
    sites = np.stack(
        [rng.integers(0, jg3.GRID_H, 200), rng.integers(0, jg3.GRID_W, 200)], axis=1
    )
    separation = 32
    batches = jg3.independent_batches(sites, separation)
    seen = np.concatenate(batches)
    # every site placed exactly once -- a partition, not a filter
    assert sorted(seen.tolist()) == list(range(len(sites)))
    for batch in batches:
        members = sites[batch]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                chebyshev = max(
                    abs(int(members[i, 0]) - int(members[j, 0])),
                    abs(int(members[i, 1]) - int(members[j, 1])),
                )
                assert chebyshev >= separation


def test_independent_batches_is_deterministic():
    """Same site set in, same partition out -- the run must be reproducible."""
    sites = np.array([[0, 0], [10, 10], [64, 64], [65, 65], [200, 300]])
    first = jg3.independent_batches(sites, 32)
    second = jg3.independent_batches(sites, 32)
    assert [b.tolist() for b in first] == [b.tolist() for b in second]


def test_independent_batches_separates_a_known_cluster():
    """Three sites 5 px apart cannot share a batch at separation 32."""
    sites = np.array([[100, 100], [100, 105], [100, 110]])
    batches = jg3.independent_batches(sites, 32)
    assert len(batches) == 3
    assert all(len(b) == 1 for b in batches)


def test_window_counts_reads_the_box_it_claims():
    plane = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=bool)
    plane[100, 100] = True
    plane[100, 108] = True  # inside a +/-15 window of (100, 100)
    plane[100, 130] = True  # outside it
    counts = jg3.window_counts(plane, np.array([[100, 100]]), 15)
    assert counts.tolist() == [2]


def test_window_counts_clips_at_the_frame_edge():
    plane = np.ones((jg3.GRID_H, jg3.GRID_W), dtype=bool)
    counts = jg3.window_counts(plane, np.array([[0, 0]]), 15)
    assert counts.tolist() == [16 * 16]


def test_packing_windows_are_disjoint_at_the_shipped_defaults():
    """The geometric precondition the packing control tests empirically.

    ``DEFAULT_SEPARATION`` must exceed ``2*DEFAULT_WINDOW`` or two packed sites'
    windows overlap and the per-site attribution double-counts.
    """
    assert jg3.DEFAULT_SEPARATION > 2 * jg3.DEFAULT_WINDOW
    # and the window must exceed ddm_jg1's MEASURED influence radius of 0-11 px
    assert jg3.DEFAULT_WINDOW > 11


# ---------------------------------------------------------------------------
# The accepted-set separation -- the defect a control caught, now guarded
# ---------------------------------------------------------------------------


def test_select_separated_enforces_the_constraint_it_names():
    """The regression guard for this arm's worst defect.

    The first run of the solver screened each move in isolation and then applied
    every winner, pooled across screening batches.  Adjacent winners merged into a
    block move -- which ``ddm_jg1`` S1c measured at -55% (r=1) -- and pair 283
    changed 66 tokens to repair 3 cells where jg1 repaired 25 from 20.
    """
    scored = {
        (100, 100): (9.0, 3, 1.0, 2),
        (100, 104): (8.0, 2, 1.0, 3),   # 4 px away -- must lose to the stronger one
        (100, 200): (7.0, 2, 1.0, 4),   # 100 px away -- must survive
    }
    kept = jg3.select_separated(scored, 64)
    assert [key for key, _ in kept] == [(100, 100), (100, 200)]


def test_select_separated_keeps_the_strongest_of_a_cluster():
    scored = {
        (10, 10): (1.0, 1, 1.0, 1),
        (10, 12): (5.0, 5, 1.0, 2),  # strongest
        (10, 14): (3.0, 3, 1.0, 3),
    }
    kept = jg3.select_separated(scored, 64)
    assert len(kept) == 1
    assert kept[0][0] == (10, 12)


def test_select_separated_is_a_pure_function_of_its_input():
    """Ties must break deterministically or a re-run can select a different set.

    Sister of the ``ddm_cw1`` container finding: configs 5 and 6 tied at 176,420 B
    and only the lower-index one reproduced the shipped bytes, so an unbroken tie
    silently flickers between equally-scored options.  Same hazard here.
    """
    scored = {(0, 0): (5.0, 1, 1.0, 1), (0, 200): (5.0, 1, 1.0, 2)}
    first = jg3.select_separated(scored, 64)
    second = jg3.select_separated(dict(reversed(list(scored.items()))), 64)
    assert [k for k, _ in first] == [k for k, _ in second]


def test_select_separated_at_zero_separation_keeps_everything():
    scored = {(0, 0): (1.0, 1, 1.0, 1), (0, 1): (2.0, 1, 1.0, 2)}
    assert len(jg3.select_separated(scored, 0)) == 2


# ---------------------------------------------------------------------------
# The proposal class
# ---------------------------------------------------------------------------


def test_candidates_exclude_no_ops_and_stay_in_bounds():
    tokens = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=np.uint8)
    moves = jg3.candidates_for_site(tokens, 100, 100)
    # 5 offsets x 5 classes, minus the 5 that would write the value already there
    assert len(moves) == 5 * (jg3.NUM_CLASSES - 1)
    for move in moves:
        assert 0 <= move.y < jg3.GRID_H
        assert 0 <= move.x < jg3.GRID_W
        assert move.value != int(tokens[move.y, move.x])


def test_candidates_are_clipped_at_a_corner():
    tokens = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=np.uint8)
    moves = jg3.candidates_for_site(tokens, 0, 0)
    # (0,0), (0,1), (1,0) are in bounds; (0,-1) and (-1,0) are not
    assert len({(m.y, m.x) for m in moves}) == 3


def test_candidate_family_is_all_class_not_gt_only():
    """``ddm_jg1`` S1e correction 2 measured **0 of 12** accepted edits chose GT.

    The winning edits are ADVERSARIAL -- they write a class that is wrong at that
    cell to steer the painted RGB so SegNet lands on GT.  A GT-only family would
    therefore miss the actual optimum, so every class must be offered.
    """
    tokens = np.full((jg3.GRID_H, jg3.GRID_W), 2, dtype=np.uint8)
    moves = jg3.candidates_for_site(tokens, 50, 50)
    offered = {m.value for m in moves}
    assert offered == {0, 1, 3, 4}


# ---------------------------------------------------------------------------
# The projection arithmetic
# ---------------------------------------------------------------------------


def test_projection_reproduces_the_jg2_headline_from_its_own_inputs():
    """jg2 S1g: 18,000 cells / 11,600 tokens -> net -0.010950, S ~ 0.145576."""
    out = jg3.project(
        repaired=90, tokens=58, pairs_solved=3, bits_per_token=4.1379, pose_ratio=1.073
    )
    assert out["repaired_projected_n600"] == pytest.approx(18000.0, rel=1e-9)
    assert out["tokens_projected_n600"] == pytest.approx(11600.0, rel=1e-9)
    assert out["seg_delta_S"] == pytest.approx(-0.015259, abs=2e-6)
    assert out["rate_delta_S"] == pytest.approx(0.003995, abs=2e-6)
    assert out["pose_delta_S"] == pytest.approx(0.000314, abs=2e-6)
    assert out["net_delta_S"] == pytest.approx(-0.010950, abs=5e-6)
    assert out["projected_S"] == pytest.approx(0.145576, abs=5e-6)
    assert out["clears_sub_015"] is True


def test_projection_flips_to_refusal_below_the_break_even_yield():
    """At jg1's iterated yield of 0.390 the move must make the score WORSE."""
    repaired, tokens = 39, 100  # yield 0.390
    out = jg3.project(repaired=repaired, tokens=tokens, pairs_solved=1)
    assert out["net_delta_S"] > 0.0
    assert out["clears_sub_015"] is False


def test_projection_prefers_a_measured_rate_leg_and_says_so():
    modelled = jg3.project(repaired=90, tokens=58, pairs_solved=3)
    measured = jg3.project(
        repaired=90, tokens=58, pairs_solved=3, measured_archive_delta_bytes=30
    )
    assert modelled["rate_source"].startswith("jg2_prior")
    assert measured["rate_source"].startswith("measured_reencoder")
    # jg2 S1f MEASURED +30 B for exactly this 3-pair edit set; scaled by 200 that
    # is 6,000 B, which must agree with the prior leg to well under a percent
    assert measured["rate_bytes_projected"] == pytest.approx(6000.0, rel=1e-9)
    assert measured["rate_bytes_projected"] == pytest.approx(
        modelled["rate_bytes_projected"], rel=0.01
    )


def test_projection_refuses_zero_pairs():
    with pytest.raises(jg3.Jg3Error):
        jg3.project(repaired=1, tokens=1, pairs_solved=0)


def test_projection_carries_its_false_authority():
    out = jg3.project(repaired=90, tokens=58, pairs_solved=3)
    assert out["score_claim"] is False
    assert out["promotable"] is False
    assert "advisory" in out["axis"]


# ---------------------------------------------------------------------------
# The rate ranker
# ---------------------------------------------------------------------------


def test_logit_price_refuses_a_wrong_sized_field(tmp_path):
    """A logits file from another generation must REFUSE, not silently truncate.

    ``np.memmap`` raises on a too-small file but happily ignores the tail of a
    too-large one, so without this check a different generation's field would be
    read as if it were ours and every candidate rank would be quietly wrong.
    """
    path = tmp_path / "logits.i16"
    np.zeros((3, jg3.PLANE, jg3.NUM_CLASSES), dtype=np.int16).tofile(path)
    with pytest.raises(jg3.Jg3Error, match="expected"):
        jg3.LogitPrice(path, pairs=2)


def test_logit_price_falls_back_to_the_flat_prior_when_absent():
    pricer = jg3.LogitPrice(Path("/nonexistent/logits.i16"))
    assert pricer.available is False
    tokens = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=np.uint8)
    moves = [jg3.Candidate(1, 1, 3), jg3.Candidate(2, 2, 4)]
    bits = pricer.bits_for(0, moves, tokens)
    assert bits.tolist() == [jg3.RATE_PRIOR_BITS_PER_TOKEN] * 2


def test_logit_price_signs_a_confident_flip_as_expensive(tmp_path):
    """A flip away from a confident class must cost POSITIVE bits.

    Built from a synthetic logit file so the sign is checked against arithmetic we
    control, not against the shipped model's opinion.
    """
    path = tmp_path / "logits.i16"
    # 2 pairs, not 600: the shipped field is 1.18 GB and a unit test has no business
    # allocating or writing it.  ``LogitPrice`` takes the pair count so a small
    # fixture is a legal field rather than a truncated one.
    block = np.zeros((2, jg3.PLANE, jg3.NUM_CLASSES), dtype=np.int16)
    # class 0 strongly favoured everywhere: logits/8 = (10, 0, 0, 0, 0)
    block[:, :, 0] = int(10 * jg3.LOGIT_SCALE)
    block.tofile(path)
    pricer = jg3.LogitPrice(path, pairs=2)
    assert pricer.available is True
    tokens = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=np.uint8)
    bits = pricer.bits_for(0, [jg3.Candidate(5, 5, 1)], tokens)
    # log2(p0/p1) with a 10-nat gap is ~14.4 bits
    assert bits[0] == pytest.approx(10.0 / np.log(2), rel=1e-3)


def test_logit_price_signs_a_flip_toward_the_model_as_a_credit(tmp_path):
    path = tmp_path / "logits.i16"
    block = np.zeros((2, jg3.PLANE, jg3.NUM_CLASSES), dtype=np.int16)
    block[:, :, 1] = int(10 * jg3.LOGIT_SCALE)
    block.tofile(path)
    pricer = jg3.LogitPrice(path, pairs=2)
    tokens = np.zeros((jg3.GRID_H, jg3.GRID_W), dtype=np.uint8)
    bits = pricer.bits_for(0, [jg3.Candidate(5, 5, 1)], tokens)
    assert bits[0] < 0.0, "a flip toward the model's own preference is a CREDIT"


# ---------------------------------------------------------------------------
# Custody
# ---------------------------------------------------------------------------


def test_pointer_constants_match_the_canonical_frontier_pointer():
    """The base this arm subtracts from must be the pointer, not a charter memory."""
    import json

    pointer = json.loads(
        (REPO / ".omx/state/canonical_frontier_pointer.json").read_text()
    )
    cuda = pointer["our_local_frontier_contest_cuda"]
    assert pytest.approx(cuda["score"], rel=1e-15) == jg3.BASE_S
    assert cuda["archive_sha256"] == jg3.POINTER_ARCHIVE_SHA
    assert cuda["extra"]["archive_bytes"] == jg3.BASE_ARCHIVE_BYTES


def test_base_S_decomposes_into_its_three_measured_legs_EXACTLY():
    """The three legs must reconstruct the pointer BIT-IDENTICALLY.

    Not "to 8 decimals" -- exactly.  The pointer's score was computed from these
    three components, so anything less than equality means a component is not the
    one that shipped.  This test is what caught this arm inheriting
    ``d_pose = 7.649246787e-06`` from ``ddm_jg1``/``ddm_jg2`` when the T4 receipt
    carries **7.65e-06**; the inherited value misses by 4.3e-07 in S.
    """
    total = (
        100.0 * jg3.BASE_D_SEG
        + (10.0 * jg3.BASE_D_POSE) ** 0.5
        + 25.0 * jg3.BASE_ARCHIVE_BYTES / jg3.SCORE_RATE_DENOMINATOR
    )
    assert total == jg3.BASE_S


def test_base_components_come_from_the_t4_receipt_not_a_memo():
    """Re-read the receipt the pointer names, and refuse any drift from it."""
    import json

    pointer = json.loads(
        (REPO / ".omx/state/canonical_frontier_pointer.json").read_text()
    )
    receipt_path = REPO / pointer["our_local_frontier_contest_cuda"]["source_path"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["avg_segnet_dist"] == jg3.BASE_D_SEG
    assert receipt["avg_posenet_dist"] == jg3.BASE_D_POSE
    assert receipt["archive_size_bytes"] == jg3.BASE_ARCHIVE_BYTES
    assert receipt["score"] == jg3.BASE_S
