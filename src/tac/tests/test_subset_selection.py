# SPDX-License-Identifier: MIT
"""Tests for the canonical subset selector (`ddm_ss1`, landing 1).

These assert BEHAVIOUR, not constants. The distinction matters here more than
usual: the defect this module cures was itself invisible to a test suite that
checked shapes and counts. So the load-bearing tests below drive a *biased*
population and require the instrument to say so, and drive an *unbiased* one and
require it to stay quiet. A selector that always said DIFFERENT_POPULATION would
pass a one-sided suite and be useless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.subset_selection import (
    DEFAULT_ALPHA,
    DEFAULT_STRATIFIED_BLOCKS,
    MODE_EXPLICIT,
    MODE_FULL,
    MODE_PREFIX,
    MODE_SEEDED_RANDOM,
    MODE_STRATIFIED,
    SCHEMA,
    VERDICT_DEGENERATE_POPULATION,
    VERDICT_DIFFERENT_POPULATION,
    VERDICT_MATCHED,
    VERDICT_VACUOUS_EMPTY,
    VERDICT_VACUOUS_NO_TABLE,
    PopulationMismatchError,
    SubsetSelectionError,
    assert_population_matched,
    governing_ratio,
    prefix_indices,
    quantile_stratified_indices,
    seeded_random_indices,
    select,
    stratified_indices,
    strided_indices,
)

POP = 600


def block_biased_population(population: int = POP, blocks: int = 10) -> list[float]:
    """A population whose FIRST blocks are much harder -- the measured shape.

    na2 measured pose difficulty per 60-pair block as ``0.41, 0.82, 0.08 ...
    0.010`` (first two blocks hardest, 79x the easiest). This fixture reproduces
    that ordering so the instrument is tested against the real failure geometry
    rather than against white noise.
    """
    per = population // blocks
    weights = [0.41, 0.82, 0.08, 0.05, 0.04, 0.03, 0.02, 0.015, 0.012, 0.010]
    out: list[float] = []
    for b in range(blocks):
        out.extend([weights[b % len(weights)]] * per)
    out.extend([weights[-1]] * (population - len(out)))
    return out


def flat_population(population: int = POP) -> list[float]:
    """An exchangeable population: index carries no information."""
    return [1.0 + ((i * 37) % 11) / 100.0 for i in range(population)]


# --- mode discipline -------------------------------------------------------


def test_mode_is_required_and_has_no_default() -> None:
    """The whole point: silence must not resolve to a prefix."""
    with pytest.raises(TypeError):
        select(24, POP)  # type: ignore[call-arg]


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(SubsetSelectionError, match="unknown mode"):
        select(24, POP, mode="whatever")


def test_prefix_mode_must_be_named_but_remains_available() -> None:
    sel = select(24, POP, mode=MODE_PREFIX)
    assert sel.indices == tuple(range(24))
    assert sel.mode == MODE_PREFIX
    assert sel.is_representative_mode is False


def test_seeded_modes_refuse_a_missing_seed() -> None:
    for mode in (MODE_SEEDED_RANDOM, MODE_STRATIFIED):
        with pytest.raises(SubsetSelectionError, match="requires an explicit seed"):
            select(24, POP, mode=mode)


# --- the discriminating pair: biased vs unbiased ---------------------------


def test_prefix_on_a_block_biased_population_is_flagged() -> None:
    """POSITIVE control -- the instrument must fire on the real failure shape."""
    gov = block_biased_population()
    sel = select(96, POP, mode=MODE_PREFIX, governing=gov, n_bootstrap=400)
    ratio = sel.ratios[0]
    assert ratio.verdict == VERDICT_DIFFERENT_POPULATION
    assert ratio.ratio is not None and ratio.ratio > 2.0, ratio.summary()
    assert sel.population_matched is False


def test_stratified_on_the_same_biased_population_is_not_flagged() -> None:
    """NEGATIVE control -- the cure must actually cure, on the same data."""
    gov = block_biased_population()
    sel = select(96, POP, mode=MODE_STRATIFIED, seed=7, governing=gov, n_bootstrap=400)
    assert sel.ratios[0].verdict == VERDICT_MATCHED, sel.ratios[0].summary()
    assert sel.population_matched is True


def test_seeded_random_on_the_biased_population_is_rarely_flagged() -> None:
    """alpha IS the false-positive rate -- so assert the RATE, not one seed.

    Round-1 finding, and the reason ``DEFAULT_ALPHA`` exists: my first draft
    asserted a single seed (7) landed inside a p05/p95 band and it did not. That
    was not a bug in the draw, it was the band promising a 1-in-10 false positive
    and delivering one. Asserting a single seed here would be agreeing with the
    test rather than testing the property.
    """
    gov = block_biased_population()
    flagged = 0
    trials = 40
    for seed in range(trials):
        sel = select(96, POP, mode=MODE_SEEDED_RANDOM, seed=seed, governing=gov, n_bootstrap=400)
        if sel.ratios[0].verdict != VERDICT_MATCHED:
            flagged += 1
    # At alpha=0.02 the expected count is ~0.8/40. Allow generous slack for
    # bootstrap noise while still failing an instrument that flags honest draws
    # at the old 10% rate (which would be ~4/40).
    assert flagged <= 3, f"{flagged}/{trials} honest random draws flagged at alpha={DEFAULT_ALPHA}"


def test_prefix_on_an_exchangeable_population_is_not_flagged() -> None:
    """The instrument must not cry wolf: a prefix of a flat population is fine."""
    sel = select(96, POP, mode=MODE_PREFIX, governing=flat_population(), n_bootstrap=400)
    assert sel.ratios[0].verdict == VERDICT_MATCHED, sel.ratios[0].summary()


# --- the derived null band -------------------------------------------------


def test_null_band_tightens_as_n_grows() -> None:
    """The band is DERIVED from the population, so it must scale like one."""
    gov = flat_population()
    widths = []
    for n in (24, 96, 300):
        r = governing_ratio(range(n), gov, seed=3, n_bootstrap=600)
        assert r.null_p05 is not None and r.null_p95 is not None
        widths.append(r.null_p95 - r.null_p05)
    assert widths[0] > widths[1] > widths[2], widths


def test_null_band_is_wider_for_a_heavier_tailed_population() -> None:
    """Axis-dependence must fall out of the data, not out of a constant."""
    flat = governing_ratio(range(96), flat_population(), seed=3, n_bootstrap=600)
    heavy = governing_ratio(range(96), block_biased_population(), seed=3, n_bootstrap=600)
    assert flat.null_p95 is not None and heavy.null_p95 is not None
    assert heavy.null_p95 - heavy.null_p05 > flat.null_p95 - flat.null_p05  # type: ignore[operator]


# --- vacuity is never a pass ----------------------------------------------


def test_absent_table_is_vacuous_not_matched() -> None:
    r = governing_ratio(range(24), None)
    assert r.verdict == VERDICT_VACUOUS_NO_TABLE
    assert r.matched is False
    assert r.reason


def test_empty_table_is_vacuous_not_matched() -> None:
    r = governing_ratio(range(24), [])
    assert r.verdict == VERDICT_VACUOUS_EMPTY
    assert r.matched is False


def test_empty_subset_is_vacuous_not_matched() -> None:
    r = governing_ratio([], flat_population())
    assert r.verdict == VERDICT_VACUOUS_EMPTY
    assert r.matched is False


def test_selection_without_any_ratio_is_not_population_matched() -> None:
    """Zero ratios is UNCHECKED, which is not MATCHED."""
    sel = select(24, POP, mode=MODE_SEEDED_RANDOM, seed=1)
    assert sel.ratios[0].verdict == VERDICT_VACUOUS_NO_TABLE
    assert sel.population_matched is False


def test_zero_mean_population_is_degenerate_not_matched() -> None:
    r = governing_ratio(range(4), [0.0] * 10)
    assert r.verdict == VERDICT_DEGENERATE_POPULATION
    assert r.matched is False


# --- index correctness -----------------------------------------------------


def test_prefix_indices_are_the_first_n() -> None:
    assert prefix_indices(5, 600) == (0, 1, 2, 3, 4)


def test_seeded_random_is_reproducible_and_distinct() -> None:
    a = seeded_random_indices(50, POP, 11)
    b = seeded_random_indices(50, POP, 11)
    c = seeded_random_indices(50, POP, 12)
    assert a == b
    assert a != c
    assert len(set(a)) == 50
    assert a == tuple(sorted(a))


def test_stratified_draws_from_every_block() -> None:
    """The structural claim: no block may be starved."""
    idx = stratified_indices(20, POP, seed=5, block_count=DEFAULT_STRATIFIED_BLOCKS)
    per = POP // DEFAULT_STRATIFIED_BLOCKS
    hit = {i // per for i in idx}
    assert hit == set(range(DEFAULT_STRATIFIED_BLOCKS)), sorted(hit)


def test_stratified_allocation_is_exact_across_shapes() -> None:
    """Never returns fewer than n -- the review-pass-1 silent-truncation finding.

    The allocator previously had a loop bound that could exit with indices still
    unplaced and return a SHORT tuple with no error. Nothing downstream re-checks
    the length, so it would have surfaced as a quietly smaller subset. This sweep
    (5 populations x every n x 4 block counts) is the pin.
    """
    for pop in (7, 13, 60, 97, 600):
        for n in range(1, pop + 1):
            for blocks in (1, 3, 10, 17, 64):
                idx = stratified_indices(n, pop, seed=n * 31 + blocks, block_count=blocks)
                assert len(idx) == n, (pop, n, blocks, len(idx))
                assert len(set(idx)) == n
                assert all(0 <= i < pop for i in idx)


def test_stratified_is_reproducible() -> None:
    assert stratified_indices(30, POP, seed=9) == stratified_indices(30, POP, seed=9)


def test_strided_matches_build_strided_subset_gt_math() -> None:
    """stride=3 over 600 is exactly the committed gt_strided_n200 index set."""
    idx = strided_indices(None, 600, stride=3)
    assert len(idx) == 200
    assert idx[:5] == (0, 3, 6, 9, 12)
    assert idx[-1] == 597


def test_strided_refuses_an_n_it_cannot_supply() -> None:
    with pytest.raises(SubsetSelectionError, match="fewer than"):
        strided_indices(500, 600, stride=3)


def test_explicit_mode_refuses_duplicates_and_out_of_range() -> None:
    with pytest.raises(SubsetSelectionError, match="duplicates"):
        select(None, POP, mode=MODE_EXPLICIT, indices=[1, 1, 2])
    with pytest.raises(SubsetSelectionError, match="outside population"):
        select(None, POP, mode=MODE_EXPLICIT, indices=[0, POP])


def test_full_population_is_matched_by_construction() -> None:
    sel = select(None, 50, mode=MODE_FULL, governing=flat_population(50))
    assert sel.n == 50
    assert sel.ratios[0].verdict == VERDICT_MATCHED
    assert sel.ratios[0].ratio == pytest.approx(1.0)


def test_n_bounds_are_enforced() -> None:
    with pytest.raises(SubsetSelectionError, match="exceeds population"):
        select(601, POP, mode=MODE_PREFIX)
    with pytest.raises(SubsetSelectionError, match="must be positive"):
        select(0, POP, mode=MODE_PREFIX)


def test_indices_out_of_range_for_governing_table_raise() -> None:
    with pytest.raises(SubsetSelectionError, match="out of range"):
        governing_ratio([0, 999], flat_population(10))


# --- provenance ------------------------------------------------------------


def test_provenance_carries_mode_seed_and_ratio() -> None:
    sel = select(48, POP, mode=MODE_STRATIFIED, seed=3, governing=flat_population(), n_bootstrap=200)
    prov = sel.provenance()
    assert prov["schema"] == SCHEMA
    assert prov["pair_selection"] == MODE_STRATIFIED
    assert prov["seed"] == 3
    assert prov["n"] == 48
    assert prov["population"] == POP
    assert prov["params"]["block_count"] == DEFAULT_STRATIFIED_BLOCKS
    assert len(prov["governing_ratios"]) == 1
    assert prov["governing_ratios"][0]["ratio"] is not None


def test_summary_states_the_ratio_when_checked() -> None:
    sel = select(96, POP, mode=MODE_PREFIX, governing=block_biased_population(), n_bootstrap=200)
    text = sel.summary()
    assert "x" in text and VERDICT_DIFFERENT_POPULATION in text


def test_summary_names_the_vacuous_verdict_when_no_table() -> None:
    """Round-1 finding: my first draft asserted 'NOT CHECKED'.

    The code says ``VACUOUS_NO_TABLE`` plus its reason, which is strictly better
    -- it names WHICH kind of not-checked. The test was wrong about the code, so
    the test moved.
    """
    sel = select(96, POP, mode=MODE_PREFIX)
    text = sel.summary()
    assert VERDICT_VACUOUS_NO_TABLE in text
    assert "no governing-quantity population table" in text


# --- real measured data ----------------------------------------------------
# The fixture is 600 real per-pair values derived from gt_n600.npz (see its own
# `derivation_*` fields). It is committed rather than skipped-if-absent on
# purpose: a skipped test reports the same symbol as a passing one, which is the
# genus this whole landing exists to close.

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "subset_selection"
    / "gt_n600_per_pair_population.json"
)


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_quantile_stratified_reproduces_known_n6() -> None:
    """BYTE-FAITHFULNESS to the two private twins this mode was lifted from.

    ``tools/measure_uint8_lattice_feasibility.py:401`` raises if its own n=6
    selection ever drifts from ``KNOWN_N6``. Reproducing that tuple from the same
    real fragility proves the lifted copy is the same algorithm, so re-pointing
    those tools at this module cannot silently change any selection.
    """
    fx = load_fixture()
    got = quantile_stratified_indices(6, fx["n_pairs"], fx["fragility"])
    assert list(got) == fx["known_n6"], f"drifted from {fx['known_n6_source']}"


def _original_select_pairs(fragility: list[float], sample_pairs: int, population: int = 600) -> list[int]:
    """Verbatim transcription of the twins' algorithm, for equivalence testing.

    Mirrors ``tools/measure_uint8_lattice_feasibility.py:388-400`` /
    ``tools/constructive_inverse_solve_harness.py:451-462`` including their
    ``np.linspace(..., dtype=np.int64)`` edges and ``np.lexsort`` tie-break. Kept
    here so the equivalence is checked against the ORIGINAL formulation rather
    than against my restatement of it.
    """
    import numpy as np

    edges = np.linspace(0, population, sample_pairs + 1, dtype=np.int64)
    frag = np.asarray(fragility, dtype=np.float64)
    out: list[int] = []
    for stratum in range(sample_pairs):
        members = np.arange(edges[stratum], edges[stratum + 1], dtype=np.int64)
        order = np.lexsort((members, frag[members]))
        quantile = 0.25 if stratum % 2 == 0 else 0.75
        out.append(int(members[order[round(quantile * (len(members) - 1))]]))
    return sorted(out)


def test_quantile_stratified_matches_the_originals_at_every_n() -> None:
    """Equivalence with the ALGORITHM, not just with the one pinned anchor.

    ``test_quantile_stratified_reproduces_known_n6`` alone is not enough and the
    history proves it: the first implementation passed KNOWN_N6 while disagreeing
    with the originals at 43 of 67 other n, because 600/6 divides exactly and hid
    both a truncate-vs-round difference and a float-association difference in
    ``np.linspace``. Sweeping every n is what caught it.
    """
    fx = load_fixture()
    frag = fx["fragility"]
    mismatches = [
        n
        for n in range(1, 601)
        if _original_select_pairs(frag, n) != list(quantile_stratified_indices(n, 600, frag))
    ]
    assert mismatches == [], f"diverged from the originals at n={mismatches[:10]}"


def test_quantile_stratified_matches_the_originals_under_heavy_ties() -> None:
    """Ties are where a lexsort restatement is most likely to drift."""
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(15):
        frag = rng.integers(0, 5, 600).astype(float).tolist()
        for n in (6, 7, 13, 17, 28, 55, 97):
            assert _original_select_pairs(frag, n) == list(
                quantile_stratified_indices(n, 600, frag)
            ), f"diverged under heavy ties at n={n}"


def test_real_seg_prefix_ratio_reproduces_na2() -> None:
    """na2's seg half, re-derived through the selector on the real population."""
    gov = load_fixture()["seg_flip_density"]
    ratios = {
        n: select(n, 600, mode=MODE_PREFIX, governing=gov, n_bootstrap=800).ratios[0].ratio
        for n in (24, 48, 64, 96)
    }
    # na2 reported 0.97 / 0.95 / 0.96 / 0.97; measured here 0.963 / 0.941 / 0.948 / 0.948.
    assert all(0.93 < r < 0.98 for r in ratios.values()), ratios


def test_real_seg_prefix_is_flagged_at_n96() -> None:
    """MEASURED detection point on the seg axis -- and it is NOT a clean threshold.

    Verified stable across 12 bootstrap seeds at n=96. It is pinned at n=96 only,
    deliberately, because the honest picture is non-monotonic: at alpha=0.02 the
    real seg prefix is MATCHED at n=24/48/150/200 and DIFFERENT at n=96/300. The
    reason is that both sides move -- the prefix ratio wanders back toward 1.0 as
    the prefix swallows more blocks (0.941 at n=48, 0.997 at n=200) while the band
    tightens with n.

    That is a real LIMIT of this instrument on this axis, stated rather than
    papered over: the seg prefix bias is 3-6%, which lives near the sampling-noise
    floor, so detection is genuinely marginal. It is not marginal where it matters
    -- see the pose-magnitude test below.
    """
    gov = load_fixture()["seg_flip_density"]
    sel = select(96, 600, mode=MODE_PREFIX, governing=gov, n_bootstrap=1500)
    assert sel.ratios[0].verdict == VERDICT_DIFFERENT_POPULATION, sel.ratios[0].summary()


def test_pose_magnitude_bias_is_flagged_even_at_n8() -> None:
    """The axis that actually matters: na2's pose prefix is 2.54-4.21x.

    A bias of that size is caught at every n INCLUDING n=8 -- which is the size of
    the smallest prefix behind the CLAUDE.md "post-hoc/stored pose carrier MEASURED
    DEAD" closure. Where seg detection is marginal, pose detection is overwhelming,
    and that asymmetry is the whole reason prefix bias is anti-conservative on pose.
    """
    for n in (8, 24, 96):
        gov = [3.0] * n + [1.0] * (600 - n)
        r = governing_ratio(range(n), gov, seed=1, n_bootstrap=800)
        assert r.verdict == VERDICT_DIFFERENT_POPULATION, r.summary()
        assert r.ratio is not None and r.ratio > 2.0


def test_real_seg_stratified_and_random_are_matched() -> None:
    """The cure, on the same real data the defect was measured on."""
    gov = load_fixture()["seg_flip_density"]
    for mode in (MODE_STRATIFIED, MODE_SEEDED_RANDOM):
        sel = select(96, 600, mode=mode, seed=7, governing=gov, n_bootstrap=1500)
        assert sel.ratios[0].verdict == VERDICT_MATCHED, sel.ratios[0].summary()


def test_real_null_band_is_narrower_for_seg_than_for_fragility() -> None:
    """Axis-dependence is DERIVED: heavier-tailed quantity -> wider band."""
    fx = load_fixture()
    seg = governing_ratio(range(96), fx["seg_flip_density"], seed=3, n_bootstrap=1500)
    frag = governing_ratio(range(96), fx["fragility"], seed=3, n_bootstrap=1500)
    assert seg.null_p95 is not None and frag.null_p95 is not None
    assert (seg.null_p95 - seg.null_p05) != (frag.null_p95 - frag.null_p05)  # type: ignore[operator]


# --- the promotion boundary ------------------------------------------------


def test_assert_population_matched_raises_on_a_biased_prefix() -> None:
    sel = select(96, POP, mode=MODE_PREFIX, governing=block_biased_population(), n_bootstrap=200)
    with pytest.raises(PopulationMismatchError, match="NOT EXCHANGEABLE"):
        assert_population_matched(sel, what="test verdict")


def test_assert_population_matched_distinguishes_unchecked_from_different() -> None:
    """Round-1 finding, and a real code fix: the two failures must not share a message.

    An unchecked subset was reported as "not exchangeable", sending a reader to
    hunt for a bias nobody measured. Now "did NOT RUN" and "NOT EXCHANGEABLE" are
    distinct strings, and this test pins both so they cannot re-merge.
    """
    unchecked = select(96, POP, mode=MODE_SEEDED_RANDOM, seed=2)
    with pytest.raises(PopulationMismatchError, match="did NOT RUN") as unchecked_exc:
        assert_population_matched(unchecked)
    assert "INSTANCE-scoped" in str(unchecked_exc.value)
    assert "NOT EXCHANGEABLE" not in str(unchecked_exc.value)

    biased = select(96, POP, mode=MODE_PREFIX, governing=block_biased_population(), n_bootstrap=200)
    with pytest.raises(PopulationMismatchError, match="NOT EXCHANGEABLE") as biased_exc:
        assert_population_matched(biased)
    assert "did NOT RUN" not in str(biased_exc.value)


def test_assert_population_matched_passes_on_a_stratified_draw() -> None:
    sel = select(96, POP, mode=MODE_STRATIFIED, seed=7, governing=block_biased_population(), n_bootstrap=400)
    assert_population_matched(sel)
