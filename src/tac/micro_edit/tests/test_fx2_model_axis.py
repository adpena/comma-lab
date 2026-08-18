# SPDX-License-Identifier: MIT
"""Tests for the ddm_fx2 probability-model axis: causal geometry + the SSE stage.

The load-bearing properties, in the order a reviewer should distrust them:

1. EXACTNESS -- no transcendental reaches the decision path (AST walk).  The
   whole cross-platform argument of this lineage rests on it, and ``ddm_rr2``
   is the receipt for what happens when it is violated (refused at S = 27.83).
2. NESTING -- with the SSE stage off, this class must be BIT-IDENTICAL to
   ``ddm_fx1``; with the SSE stage on but COLD, it must still be bit-identical.
   Without both, a byte delta cannot be attributed to the new mechanism.
3. NOT INERT -- the widened template must actually read more neighbours than the
   narrow one, and a warm SSE stage must actually move the estimate.  A test
   suite that only checks nesting would pass on a module that does nothing.
4. RECEIVER CLOSURE -- causality (a group's coding row cannot depend on that
   group's own symbols) and determinism (two replays reach the same state).
   Together these are what make the receiver able to regenerate the model with
   zero transmitted bytes.

Each test is written so it FAILS if the property is broken, not merely if the
code crashes: the mutation to imagine is "delete the feature and see whether
this still passes".
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_fx1_logistic_mixer_corrector import FixedPointLogisticMixer
from experiments.ddm_fx2_model_axis_corrector import (
    CAUSAL_OFFSETS,
    HEIGHT,
    HOMOGENEITY_LEVELS,
    SPATIAL4_LEVELS,
    SSE_CONTEXTS,
    WIDTH,
    Fx2ModelAxisMixer,
    fx2_family_specs,
    fx2_mixer_contexts,
)
from experiments.ddm_rr4_free_corrector_v2 import MIN_COUNT, NUM_CLASSES

PLANE = HEIGHT * WIDTH
MODULE = Path(__file__).resolve().parents[3].parent / "experiments" / (
    "ddm_fx2_model_axis_corrector.py"
)
GROUP_INDEX = Path(
    "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/group_index.u8"
)

INHERITED_11 = (
    "shipped_joint",
    "temporal_spatial",
    "surprise_only",
    "spatial_surprise",
    "spatial_boundary",
    "run_surprise",
    "boundary_surprise",
    "temporal_surprise",
    "shipped_fast256",
    "shipped_fast4096",
    "surprise_fast256",
)


# --- 1. exactness -----------------------------------------------------------


def test_module_has_no_transcendental_on_the_decision_path():
    """The ddm_rr2 refusal in one assertion.

    ``log``/``exp``/``pow`` are libm routines that are NOT correctly rounded and
    differ by an ULP across platforms; one ULP at p ~ 0.5 moves an RC64 integer
    frequency by 128 counts and desynchronises the decoder for the rest of the
    stream.  ``sqrt`` is permitted because IEEE-754 REQUIRES it to be correctly
    rounded, which is the entire basis of the radical construction.
    """
    banned = {"log", "log2", "log10", "log1p", "exp", "exp2", "expm1", "power", "pow"}
    tree = ast.parse(MODULE.read_text())
    offences = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            offences.append(f"** at line {node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr in banned:
            offences.append(f"{node.attr} at line {node.lineno}")
        if isinstance(node, ast.Name) and node.id in banned:
            offences.append(f"{node.id} at line {node.lineno}")
    assert not offences, f"transcendental on the decision path: {offences}"


# --- helpers ----------------------------------------------------------------


def _synthetic(seed=0, frames=2, n_groups=6):
    rng = np.random.default_rng(seed)
    order = rng.permutation(PLANE)
    groups = [np.sort(chunk) for chunk in np.array_split(order, n_groups)]
    rows, tokens = [], []
    for _ in range(frames):
        logits = rng.normal(0.0, 2.0, size=(PLANE, NUM_CLASSES))
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        rows.append(probability.astype(np.float32))
        tokens.append(rng.integers(0, NUM_CLASSES, size=PLANE).astype(np.uint8))
    return groups, rows, tokens


def _drive(corrector, groups, rng_rows, tokens):
    """Run a corrector over a synthetic field, collecting every coding row."""
    out = []
    for frame in range(len(rng_rows)):
        corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
        for flat in groups:
            probability = rng_rows[frame][flat]
            predicted = probability.argmax(axis=1).astype(np.int64)
            state = corrector.group_state(probability, predicted, flat)
            out.append(corrector.coding_row(state))
            corrector.observe(state, tokens[frame][flat])
        corrector.end_frame(tokens[frame])
    return out


# --- 2. nesting -------------------------------------------------------------


def test_sse_off_is_bit_identical_to_the_fx1_mixer():
    """POSITIVE CONTROL: the whole fx2 delta must be the new mechanism.

    Same members, same mixer context, SSE off.  Any difference here means a
    measured byte delta cannot be attributed to the SSE stage or the widened
    template, only to plumbing.
    """
    groups, rows, tokens = _synthetic(seed=11, frames=3)
    fx2 = _drive(
        Fx2ModelAxisMixer(
            PLANE,
            families=INHERITED_11,
            mixer_context="cls_boundary_agree_ubin8",
            sse_context="off",
        ),
        groups,
        rows,
        tokens,
    )
    fx1 = _drive(
        FixedPointLogisticMixer(
            PLANE, families=INHERITED_11, mixer_context="cls_boundary_agree_ubin8"
        ),
        groups,
        rows,
        tokens,
    )
    for index, (left, right) in enumerate(zip(fx2, fx1, strict=True)):
        assert left.tobytes() == right.tobytes(), f"coding row {index} diverged from fx1"


def test_a_cold_sse_stage_costs_exactly_nothing():
    """A useless SSE stage must be free BIT-FOR-BIT, not merely cheap.

    One group in one frame, so every SSE bin stays below ``MIN_COUNT`` for the
    whole run and its multiplier is exactly 1.0.  Multiplying by exactly 1.0 is
    the identity in IEEE-754, so the rows must match byte for byte.
    """
    groups, rows, tokens = _synthetic(seed=5, frames=1, n_groups=1)
    warm = _drive(
        Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="cls_bnd_qbin"),
        groups,
        rows,
        tokens,
    )
    cold = _drive(
        Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="off"),
        groups,
        rows,
        tokens,
    )
    for index, (left, right) in enumerate(zip(warm, cold, strict=True)):
        assert left.tobytes() == right.tobytes(), f"cold SSE changed row {index}"


def test_an_unobserved_sse_table_returns_exactly_one():
    """The mechanism behind the previous test, asserted directly."""
    mixer = Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="qbin")
    assert mixer.sse is not None
    index = np.arange(SSE_CONTEXTS["qbin"][0], dtype=np.int64)
    assert np.all(mixer.sse.multiplier(index) == 1.0)
    # And it stays exactly 1.0 right up to the count floor, never approximately.
    mixer.sse.counts[:] = MIN_COUNT - 1
    mixer.sse.hits[:] = MIN_COUNT - 1
    assert np.all(mixer.sse.multiplier(index) == 1.0)


# --- 3. not inert -----------------------------------------------------------


def test_a_warm_sse_stage_actually_moves_the_estimate():
    """ANTI-INERT: a stage that never changes anything would pass every test above."""
    groups, rows, tokens = _synthetic(seed=17, frames=4)
    with_sse = _drive(
        Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="qbin"),
        groups,
        rows,
        tokens,
    )
    without = _drive(
        Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="off"),
        groups,
        rows,
        tokens,
    )
    assert any(
        left.tobytes() != right.tobytes()
        for left, right in zip(with_sse, without, strict=True)
    ), "the SSE stage never changed a single coding row"


def test_the_widened_template_reads_neighbours_the_narrow_one_cannot():
    """ANTI-INERT for R1: ``spatial4`` must differ from fx1's ``spatial``.

    A hand-built decoded state where the UP-RIGHT neighbour carries the
    predicted class and neither LEFT nor UP does.  fx1's two-neighbour level
    cannot see it; the widened one must.
    """
    mixer = Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="off")
    mixer.begin_frame(np.zeros(PLANE, dtype=np.int64))
    y, x = 10, 10
    here = y * WIDTH + x
    up_right = (y - 1) * WIDTH + (x + 1)
    mixer.known[up_right] = True
    mixer.current[up_right] = 3

    flat = np.array([here], dtype=np.int64)
    base = np.array([3], dtype=np.int64)
    classes, available = mixer._causal_neighbours(flat)
    narrow = mixer._spatial_level(flat, base)
    wide = mixer._spatial4_level(classes, available, base)

    assert narrow[0] == 0, "fx1's template should see nothing here"
    assert wide[0] == 2, "the widened template should see one agreeing neighbour"


def test_homogeneity_separates_a_cell_interior_from_a_decoded_boundary():
    """The geometric feature: distinct classes among causal neighbours."""
    mixer = Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context="off")
    mixer.begin_frame(np.zeros(PLANE, dtype=np.int64))
    y, x = 40, 40
    here = np.array([y * WIDTH + x], dtype=np.int64)

    for dx, dy in CAUSAL_OFFSETS:
        neighbour = (y + dy) * WIDTH + (x + dx)
        mixer.known[neighbour] = True
        mixer.current[neighbour] = 2
    classes, available = mixer._causal_neighbours(here)
    assert mixer._homogeneity_level(classes, available)[0] == 1, "unanimous = interior"

    boundary_neighbour = (y + CAUSAL_OFFSETS[0][1]) * WIDTH + (x + CAUSAL_OFFSETS[0][0])
    mixer.current[boundary_neighbour] = 4
    classes, available = mixer._causal_neighbours(here)
    assert mixer._homogeneity_level(classes, available)[0] == 2, "disagreement = boundary"

    # Nothing decoded yet is its own level, distinct from "unanimous".
    mixer.known[:] = False
    classes, available = mixer._causal_neighbours(here)
    assert mixer._homogeneity_level(classes, available)[0] == 0


def test_the_learner_moves_the_sse_exponent_when_asked_and_not_otherwise():
    """ANTI-INERT for the optional learned SSE weight."""
    groups, rows, tokens = _synthetic(seed=23, frames=3)
    learned = Fx2ModelAxisMixer(
        PLANE, families=("shipped_joint",), sse_context="qbin", sse_learn_weight=True
    )
    start = learned.sse_weight.copy()
    _drive(learned, groups, rows, tokens)
    assert not np.array_equal(learned.sse_weight, start), "the SSE exponent never moved"

    fixed = Fx2ModelAxisMixer(
        PLANE, families=("shipped_joint",), sse_context="qbin", sse_learn_weight=False
    )
    start = fixed.sse_weight.copy()
    _drive(fixed, groups, rows, tokens)
    assert np.array_equal(fixed.sse_weight, start)


# --- 4. receiver closure ----------------------------------------------------


def test_coding_row_cannot_depend_on_the_group_it_is_about_to_code():
    """CAUSALITY, with the SSE stage and the widened template both active.

    Two runs identical except for the SYMBOLS of the group being coded.  If any
    coding row for that group differs, the encoder is using information the
    receiver does not have and the stream desynchronises.
    """
    groups, rows, tokens = _synthetic(seed=31, frames=1)
    other = tokens[0].copy()
    other[groups[2]] = (other[groups[2]] + 1) % NUM_CLASSES

    def run(token_plane):
        mixer = Fx2ModelAxisMixer(
            PLANE,
            families=INHERITED_11,
            mixer_context="cls_boundary_agree_homog_ubin8",
            sse_context="cls_bnd_qbin",
        )
        mixer.begin_frame(np.zeros(PLANE, dtype=np.int64))
        captured = None
        for position, flat in enumerate(groups):
            probability = rows[0][flat]
            predicted = probability.argmax(axis=1).astype(np.int64)
            state = mixer.group_state(probability, predicted, flat)
            row = mixer.coding_row(state)
            if position == 2:
                captured = row
            mixer.observe(state, token_plane[flat])
        return captured

    assert run(tokens[0]).tobytes() == run(other).tobytes()


def test_two_independent_replays_reach_an_identical_adaptive_state():
    """DETERMINISM: the receiver must regenerate the encoder's model exactly."""
    groups, rows, tokens = _synthetic(seed=37, frames=3)

    def run():
        mixer = Fx2ModelAxisMixer(
            PLANE,
            families=INHERITED_11,
            mixer_context="cls_boundary_agree_homog_ubin8",
            sse_context="cls_homog_qbin",
            sse_learn_weight=True,
        )
        emitted = _drive(mixer, groups, rows, tokens)
        return mixer, emitted

    first, first_rows = run()
    second, second_rows = run()
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.sse_weight, second.sse_weight)
    assert first.sse is not None and second.sse is not None
    assert np.array_equal(first.sse.counts, second.sse.counts)
    assert np.array_equal(first.sse.hits, second.sse.hits)
    assert np.array_equal(first.sse.phat_q, second.sse.phat_q)
    for left, right in zip(first_rows, second_rows, strict=True):
        assert left.tobytes() == right.tobytes()


# --- 5. the declared surfaces all build and stay in bounds ------------------


def test_every_declared_family_builds_and_indexes_within_its_own_table():
    """Every member must size its table to its own rule and observe every position.

    A member with a ``count_limit`` HALVES saturated bins on purpose (recency), so
    its surviving total is below the plane -- asserting equality there would be
    asserting that the forgetting does not happen.
    """
    groups, rows, tokens = _synthetic(seed=41, frames=1, n_groups=3)
    for name, spec in fx2_family_specs().items():
        if name == "base_odds":
            continue  # the fx1 member with no table of its own, refused there
        mixer = Fx2ModelAxisMixer(PLANE, families=(name,), sse_context="off")
        _drive(mixer, groups, rows, tokens)
        family = mixer.families[0]
        assert family.counts.size == spec[0], name
        total = int(family.counts.sum())
        if family.count_limit:
            assert 0 < total < PLANE, f"{name}: a recency member must forget"
        else:
            assert total == PLANE, f"{name}: a cumulative member must count every position"


def test_every_declared_mixer_context_indexes_within_its_own_weight_table():
    groups, rows, tokens = _synthetic(seed=43, frames=1, n_groups=3)
    for name, (size, _) in fx2_mixer_contexts().items():
        mixer = Fx2ModelAxisMixer(
            PLANE, families=("shipped_joint",), mixer_context=name, sse_context="off"
        )
        assert mixer.weights.shape[0] == size
        _drive(mixer, groups, rows, tokens)


def test_every_declared_sse_context_indexes_within_its_own_table():
    groups, rows, tokens = _synthetic(seed=47, frames=1, n_groups=3)
    for name, (size, _) in SSE_CONTEXTS.items():
        mixer = Fx2ModelAxisMixer(PLANE, families=("shipped_joint",), sse_context=name)
        _drive(mixer, groups, rows, tokens)
        if size:
            assert mixer.sse is not None
            assert mixer.sse.counts.size == size
            assert int(mixer.sse.counts.sum()) == PLANE, name
        else:
            assert mixer.sse is None


def test_unknown_family_and_context_are_refused():
    for kwargs in (
        {"families": ("no_such_member",)},
        {"mixer_context": "no_such_context"},
        {"sse_context": "no_such_sse"},
        {"families": ()},
    ):
        with pytest.raises(ValueError):
            Fx2ModelAxisMixer(PLANE, **kwargs)


def test_shipped_alias_is_a_drop_in_for_the_encoder_contract():
    """The encoder builds ``FreeCorrector(plane)`` with no other arguments."""
    from experiments.ddm_fx2_model_axis_corrector import SHIPPED_CONFIG, FreeCorrector

    corrector = FreeCorrector(PLANE)
    assert [f.name for f in corrector.families] == list(SHIPPED_CONFIG["families"])
    assert corrector.mixer_context_name == SHIPPED_CONFIG["mixer_context"]
    assert corrector.sse_context_name == SHIPPED_CONFIG["sse_context"]


# --- 6. the R1 causality claim, against the REAL decode order ---------------


@pytest.mark.skipif(not GROUP_INDEX.exists(), reason="retained group index not mounted")
def test_the_widened_template_offsets_really_are_causal_in_the_shipped_order():
    """The docstring's arithmetic, checked against the shipped group index.

    A neighbour is decoded before this position exactly when its group index is
    strictly smaller.  Every offset in ``CAUSAL_OFFSETS`` must clear 97% of
    in-bounds positions, and UP-RIGHT specifically must match UP -- that
    equality is the finding the widened template is built on.
    """
    index = np.fromfile(GROUP_INDEX, dtype=np.uint8).astype(np.int64).reshape(HEIGHT, WIDTH)
    x = np.tile(np.arange(WIDTH), (HEIGHT, 1))
    y = np.tile(np.arange(HEIGHT).reshape(-1, 1), (1, WIDTH))

    fraction = {}
    for dx, dy in CAUSAL_OFFSETS:
        nx, ny = x + dx, y + dy
        inside = (nx >= 0) & (nx < WIDTH) & (ny >= 0) & (ny < HEIGHT)
        neighbour = index[np.clip(ny, 0, HEIGHT - 1), np.clip(nx, 0, WIDTH - 1)]
        causal = inside & (neighbour < index)
        fraction[(dx, dy)] = float(causal.sum()) / float(inside.sum())

    for offset, value in fraction.items():
        assert value > 0.97, f"offset {offset} is only {value:.4%} causal"
    assert fraction[(1, -1)] == pytest.approx(fraction[(0, -1)], abs=1e-9), (
        "up-right must be exactly as causal as up; that equality is the R1 finding"
    )


def test_level_alphabets_are_wide_enough_for_what_the_template_can_emit():
    """A too-narrow alphabet would silently fold two regimes into one cell."""
    assert len(CAUSAL_OFFSETS) + 2 <= SPATIAL4_LEVELS
    assert min(NUM_CLASSES, len(CAUSAL_OFFSETS)) + 1 <= HOMOGENEITY_LEVELS
