# SPDX-License-Identifier: MIT
"""Tests for the ddm_fx1 fixed-point log-odds mixer.

The load-bearing properties, in the order a reviewer should distrust them:

1. EXACTNESS -- no transcendental reaches the decision path (AST walk), and the
   radical decomposition really does compute the weighted geometric mean.
2. NESTING -- one family at weight 1.0 is BIT-IDENTICAL to the shipped law, and
   all-zero weights are BIT-IDENTICAL to uncorrected HPAC.  Without these the
   race measures plumbing, not mixing.
3. RECEIVER CLOSURE -- the coding row for a group cannot depend on that group's
   own symbols (causality), and two independent replays of the same prefix reach
   an identical adaptive state (determinism).  Together these are what makes the
   decoder able to regenerate the weights with zero transmitted bytes.

Each test is written so that it FAILS if the corresponding property is broken,
not merely if the code crashes: the mutation to imagine is "delete the feature
and see whether this still passes".
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_fx1_logistic_mixer_corrector import (
    BASE_MEMBER,
    POWER_BITS,
    WEIGHT_STORE_ONE,
    FixedPointLogisticMixer,
    dyadic_power,
    family_specs,
    round_shift,
    stretch_from_radical,
)
from experiments.ddm_rr4_free_corrector_v2 import FreeCorrector

PLANE = 384 * 512
MODULE = Path(__file__).resolve().parents[4] / "experiments" / "ddm_fx1_logistic_mixer_corrector.py"

BANNED = {"log", "log2", "log10", "exp", "exp2", "expm1", "log1p", "power", "float_power", "pow"}


# --- 1. exactness -----------------------------------------------------------


def test_module_has_no_transcendental_on_the_decision_path():
    """The rr2 desync class must be structurally impossible, not merely absent.

    ``log``/``exp`` are not correctly rounded and differ by an ULP across
    platforms; one ULP moves an RC64 frequency by 128 counts and desynchronises
    the decoder.  ``sqrt`` is exempt: IEEE-754 REQUIRES it to be correctly
    rounded, which is the whole basis of this module.
    """
    tree = ast.parse(MODULE.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            offenders.append(f"line {node.lineno}: ** operator")
        if isinstance(node, ast.Attribute) and node.attr in BANNED:
            offenders.append(f"line {node.lineno}: .{node.attr}")
        if isinstance(node, ast.Name) and node.id in BANNED:
            offenders.append(f"line {node.lineno}: {node.id}")
    assert not offenders, f"transcendental on the decision path: {offenders}"


@pytest.mark.parametrize("value", [0.0625, 0.5, 0.9, 1.0, 1.1, 2.0, 7.5, 16.0])
@pytest.mark.parametrize("weight_units", [-128, -64, -33, -1, 0, 1, 17, 64, 65, 129, 512])
def test_dyadic_power_matches_the_true_weighted_geometric_mean(value, weight_units):
    """The radical decomposition must actually equal ``value ** (W / 2**b)``."""
    got = dyadic_power(
        np.array([value]),
        [np.array([_nth_root(value, index + 1)]) for index in range(POWER_BITS)],
        np.array([weight_units], dtype=np.int64),
        POWER_BITS,
    )[0]
    want = math.pow(value, weight_units / (1 << POWER_BITS))
    assert got == pytest.approx(want, rel=1e-12), f"{value}**({weight_units}/64)"


def _nth_root(value: float, times: int) -> float:
    root = np.float64(value)
    for _ in range(times):
        root = np.sqrt(root)
    return float(root)


def test_integer_weight_returns_the_base_bit_identically():
    """Weight exactly 1.0 must not round the multiplier at all.

    This is what makes the single-family identity control meaningful: if the
    integer path introduced even one ULP, a null change would show a nonzero
    byte delta and every later verdict would be reading plumbing noise.
    """
    values = np.array([0.0625, 0.3, 1.0, 1.7, 9.0, 16.0])
    got = dyadic_power(values, None, np.full(values.shape, 1 << POWER_BITS, np.int64), POWER_BITS)
    assert np.array_equal(got, values)
    assert got.tobytes() == values.tobytes()


def test_zero_weight_returns_exactly_one():
    """All-zero weights must reproduce uncorrected HPAC exactly."""
    values = np.array([0.0625, 0.5, 1.0, 3.0, 16.0])
    got = dyadic_power(values, None, np.zeros(values.shape, np.int64), POWER_BITS)
    assert np.array_equal(got, np.ones_like(values))


def test_stretch_tracks_the_natural_logarithm():
    """The gradient surrogate must be monotone, correctly signed, and close."""
    for value in (0.0625, 0.25, 0.8, 1.0, 1.5, 4.0, 16.0):
        radical = np.array([_nth_root(value, POWER_BITS)])
        got = float(stretch_from_radical(radical, POWER_BITS)[0])
        want = math.log(value)
        assert math.copysign(1.0, got) == math.copysign(1.0, want) or want == 0.0
        assert abs(got - want) <= 0.03 * abs(want) + 1e-12


def test_round_shift_is_unbiased_where_a_bare_shift_is_not():
    """A bare ``>>`` floors, leaking a systematic -1 on every negative update."""
    values = np.array([-7, -3, -1, 0, 1, 3, 7], dtype=np.int64)
    assert np.array_equal(round_shift(values, 1), np.array([-3, -1, 0, 0, 1, 2, 4]))
    assert (values >> np.int64(1)).sum() < round_shift(values, 1).sum()
    assert np.array_equal(round_shift(values, 0), values)


# --- 2. nesting -------------------------------------------------------------


def _drive(corrector, frames, groups, rng_rows, tokens):
    """Run a corrector over a synthetic field, collecting every coding row."""
    rows = []
    for frame in range(frames):
        corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
        for flat in groups:
            probability = rng_rows[frame][flat]
            predicted = probability.argmax(axis=1).astype(np.int64)
            state = corrector.group_state(probability, predicted, flat)
            rows.append(corrector.coding_row(state))
            corrector.observe(state, tokens[frame][flat])
        corrector.end_frame(tokens[frame])
    return rows


def _synthetic(seed=0, frames=2, n_groups=6):
    rng = np.random.default_rng(seed)
    order = rng.permutation(PLANE)
    groups = [np.sort(chunk) for chunk in np.array_split(order, n_groups)]
    rows, tokens = [], []
    for _ in range(frames):
        logits = rng.normal(0.0, 2.0, size=(PLANE, 5))
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        rows.append(probability.astype(np.float32))
        tokens.append(rng.integers(0, 5, size=PLANE).astype(np.uint8))
    return groups, rows, tokens


def test_single_family_weight_one_is_bit_identical_to_the_shipped_law():
    """POSITIVE CONTROL: the mixer must COLLAPSE onto ddm_rr4 when it is told to.

    Static weights, one family, weight 1.0.  Any difference at all here means a
    later byte delta cannot be attributed to mixing.
    """
    groups, rows, tokens = _synthetic()
    mixed = _drive(
        FixedPointLogisticMixer(PLANE, families=("shipped_joint",), learn=False),
        len(rows), groups, rows, tokens,
    )
    shipped = _drive(FreeCorrector(PLANE), len(rows), groups, rows, tokens)
    for index, (left, right) in enumerate(zip(mixed, shipped, strict=True)):
        assert left.tobytes() == right.tobytes(), f"coding row {index} diverged"


def test_all_zero_weights_are_bit_identical_to_uncorrected_hpac():
    """POSITIVE CONTROL: weight 0 must emit the receiver's own bytes untouched."""
    groups, rows, tokens = _synthetic(seed=3)
    mixed = _drive(
        FixedPointLogisticMixer(
            PLANE, families=("shipped_joint",), learn=False, initial_weights=(0.0,)
        ),
        len(rows), groups, rows, tokens,
    )
    for index, flat in enumerate(groups):
        assert mixed[index].tobytes() == rows[0][flat].tobytes()


# --- 3. receiver closure ----------------------------------------------------


def test_coding_row_cannot_depend_on_the_group_it_is_about_to_code():
    """CAUSALITY: the receiver has not decoded this group yet when it needs the row.

    Two runs identical except for the SYMBOLS of the group being coded.  If any
    coding row for that group differs, the encoder is using information the
    decoder does not have and the stream would desynchronise.
    """
    groups, rows, tokens = _synthetic(seed=7, frames=1)
    other = [tokens[0].copy()]
    other[0][groups[0]] = (other[0][groups[0]] + 1) % 5

    first = _drive(_mixer(), 1, groups, rows, tokens)
    second = _drive(_mixer(), 1, groups, rows, other)
    assert first[0].tobytes() == second[0].tobytes(), "row leaked its own group's symbols"


def test_two_independent_replays_reach_an_identical_adaptive_state():
    """DETERMINISM: the decoder regenerates the weights exactly, so zero bytes ship.

    This is the rule-118 claim in executable form.  It must hold for the LEARNED
    weights, not only the count tables -- the learner is the new state.
    """
    groups, rows, tokens = _synthetic(seed=11, frames=3)
    left, right = _mixer(), _mixer()
    rows_left = _drive(left, len(rows), groups, rows, tokens)
    rows_right = _drive(right, len(rows), groups, rows, tokens)

    for index, (a, b) in enumerate(zip(rows_left, rows_right, strict=True)):
        assert a.tobytes() == b.tobytes(), f"coding row {index} diverged between replays"
    assert np.array_equal(left.weights, right.weights)
    for one, two in zip(left.families, right.families, strict=True):
        assert np.array_equal(one.counts, two.counts)
        assert np.array_equal(one.hits, two.hits)
        assert np.array_equal(one.phat_q, two.phat_q)


def test_count_bucketed_weights_stay_causal_and_deterministic():
    """The evidence-count axis reads counts the receiver already has.

    The bucket is taken from the counts standing BEFORE the group is observed, so
    a decoder computes the identical index.  Both receiver-closure properties are
    re-asserted here rather than assumed to carry over, because the count axis
    adds a new read of mutable state and that is exactly where a causality leak
    would hide.
    """
    groups, rows, tokens = _synthetic(seed=29, frames=3)
    kwargs = {"families": ("shipped_joint", "surprise_only"), "count_buckets": 8}
    left, right = _mixer(**kwargs), _mixer(**kwargs)
    rows_left = _drive(left, len(rows), groups, rows, tokens)
    rows_right = _drive(right, len(rows), groups, rows, tokens)
    for index, (a, b) in enumerate(zip(rows_left, rows_right, strict=True)):
        assert a.tobytes() == b.tobytes(), f"row {index} diverged between replays"
    assert np.array_equal(left.weights, right.weights)

    other = [token.copy() for token in tokens]
    other[0][groups[0]] = (other[0][groups[0]] + 1) % 5
    first = _drive(_mixer(**kwargs), 1, groups, rows, tokens)
    second = _drive(_mixer(**kwargs), 1, groups, rows, other)
    assert first[0].tobytes() == second[0].tobytes(), "count axis leaked the group's symbols"


def test_count_buckets_of_one_reproduces_the_unbucketed_mixer_exactly():
    """The new axis must be a strict generalisation, not a silent behaviour change."""
    groups, rows, tokens = _synthetic(seed=31, frames=2)
    names = ("shipped_joint", "surprise_only")
    plain = _drive(_mixer(families=names, count_buckets=1), len(rows), groups, rows, tokens)
    same = _drive(_mixer(families=names), len(rows), groups, rows, tokens)
    for left, right in zip(plain, same, strict=True):
        assert left.tobytes() == right.tobytes()


def test_the_learner_actually_moves_the_weights():
    """Guards against a silently inert learner passing every exactness test.

    A mixer that never updates would satisfy determinism, causality and both
    nesting controls perfectly -- and measure exactly 0 bytes forever.  The race
    would then report a false third closure.
    """
    groups, rows, tokens = _synthetic(seed=13, frames=3)
    mixer = _mixer(families=("shipped_joint", "surprise_only"))
    before = mixer.weights.copy()
    _drive(mixer, len(rows), groups, rows, tokens)
    assert not np.array_equal(before, mixer.weights), "learner is inert"
    assert np.all(np.abs(mixer.weights) <= 8 * WEIGHT_STORE_ONE), "weights escaped the clamp"


def test_learning_disabled_leaves_the_weights_exactly_where_they_started():
    groups, rows, tokens = _synthetic(seed=17, frames=2)
    mixer = _mixer(learn=False, families=("shipped_joint", "surprise_only"))
    before = mixer.weights.copy()
    _drive(mixer, len(rows), groups, rows, tokens)
    assert np.array_equal(before, mixer.weights)


# --- construction guards ----------------------------------------------------


def test_every_declared_family_builds_and_indexes_within_its_own_table():
    """An out-of-range index would silently alias two contexts into one bin.

    ``base_odds`` is exempt from the count assertion and asserted the other way:
    it is a DIRECT function of the prior's own row, so it must never accumulate a
    table.  If it did, it would be double-counting evidence another member owns.
    """
    groups, rows, tokens = _synthetic(seed=19, frames=1)
    names = tuple(family_specs())
    assert BASE_MEMBER in names
    mixer = _mixer(families=names)
    _drive(mixer, 1, groups, rows, tokens)
    for family in mixer.families:
        assert family.counts.size == family.size, family.name
        if family.name == BASE_MEMBER:
            assert family.counts.sum() == 0, "base member must hold no statistics"
        elif family.count_limit:
            # A recency member deliberately forgets, so its total is below the
            # position count; what must still hold is that it saw everything and
            # never indexed outside its own table.
            assert 0 < family.counts.sum() <= PLANE, family.name
            assert family.hits.sum() <= family.counts.sum(), family.name
        else:
            assert family.counts.sum() == PLANE, family.name


def test_recency_member_forgets_and_a_cumulative_one_does_not():
    """The recency twin must actually bound its counts, not merely declare a limit.

    Without this a ``count_limit`` that never fires would look like a new member,
    race as a duplicate of the cumulative one, and quietly waste a mixer slot.
    """
    groups, rows, tokens = _synthetic(seed=37, frames=3, n_groups=3)
    # Same context RULE on both sides, so the only difference is the forgetting.
    mixer = _mixer(families=("surprise_only", "surprise_fast256"))
    _drive(mixer, len(rows), groups, rows, tokens)
    cumulative, recency = mixer.families
    assert recency.count_limit == 256
    assert cumulative.count_limit == 0
    assert cumulative.rule is recency.rule
    assert recency.counts.max() <= 2 * recency.count_limit, "recency member never halved"
    assert cumulative.counts.max() > recency.counts.max(), "cumulative member forgot"


def test_base_member_starts_neutral_so_the_mixer_still_begins_at_the_shipped_law():
    """Enlisting the prior as a member must not perturb the starting point."""
    groups, rows, tokens = _synthetic(seed=23, frames=1)
    mixed = _drive(
        _mixer(families=("shipped_joint", BASE_MEMBER), learn=False),
        1, groups, rows, tokens,
    )
    shipped = _drive(FreeCorrector(PLANE), 1, groups, rows, tokens)
    for left, right in zip(mixed, shipped, strict=True):
        assert left.tobytes() == right.tobytes()


def test_unknown_family_and_context_are_refused():
    with pytest.raises(ValueError):
        FixedPointLogisticMixer(PLANE, families=("no_such_family",))
    with pytest.raises(ValueError):
        FixedPointLogisticMixer(PLANE, mixer_context="no_such_context")
    with pytest.raises(ValueError):
        FixedPointLogisticMixer(PLANE, families=())


def _mixer(**kwargs):
    kwargs.setdefault("families", ("shipped_joint",))
    return FixedPointLogisticMixer(PLANE, **kwargs)


# --- shipping surface -------------------------------------------------------


def test_shipped_alias_is_a_drop_in_for_the_encoder_contract():
    """The build chain constructs a corrector as ``FreeCorrector(plane)``.

    If this signature drifts, the byte-close silently falls back to whatever the
    env var default is and the measured candidate is not what ships.
    """
    from experiments.ddm_fx1_logistic_mixer_corrector import SHIPPED_CONFIG
    from experiments.ddm_fx1_logistic_mixer_corrector import FreeCorrector as Fx1

    corrector = Fx1(PLANE)
    assert isinstance(corrector, FixedPointLogisticMixer)
    assert tuple(f.name for f in corrector.families) == SHIPPED_CONFIG["families"]
    assert corrector.mixer_context_name == SHIPPED_CONFIG["mixer_context"]
    assert corrector.count_buckets == SHIPPED_CONFIG["count_buckets"]
    assert corrector.lr_shift == SHIPPED_CONFIG["lr_shift"]
    assert corrector.learn is True


def test_shipped_alias_still_starts_at_the_incumbent():
    """Even with eleven members enlisted, frame zero must emit the shipped law."""
    from experiments.ddm_fx1_logistic_mixer_corrector import FreeCorrector as Fx1

    groups, rows, tokens = _synthetic(seed=41, frames=1, n_groups=2)
    mixed = _drive(Fx1(PLANE), 1, groups, rows, tokens)
    shipped = _drive(FreeCorrector(PLANE), 1, groups, rows, tokens)
    # The very first group is coded before any symbol has been observed, so the
    # learner cannot have moved yet and the two must agree exactly.
    assert mixed[0].tobytes() == shipped[0].tobytes()
