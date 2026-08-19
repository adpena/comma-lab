# SPDX-License-Identifier: MIT
"""Tests for ``ddm_ma1``'s within-miss relative law.

The suite is built around the properties that make the candidate admissible
rather than around its arithmetic: exactness (no libm on the decision path),
NESTING (the class collapses onto the law it extends, bit-for-bit), MASS
PRESERVATION (the hit event cannot move), and CAUSALITY (nothing is read that the
receiver has not already decoded).  Mutation checks at the end inject three real
defects and assert the suite catches each one, so a green run means something.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_fx2_model_axis_corrector import SHIPPED_CONFIG as FX2_CONFIG
from experiments.ddm_fx2_model_axis_corrector import Fx2ModelAxisMixer
from experiments.ddm_ma1_within_miss_corrector import (
    MISS_CLAMP_HIGH,
    MISS_CLAMP_LOW,
    UNKNOWN,
    FreeCorrector,
    Ma1WithinMissCorrector,
    miss_cells,
)
from tac.micro_edit.coder_replay import NUM_CLASSES, PLANE

MODULE = Path(__file__).resolve().parents[4] / "experiments" / "ddm_ma1_within_miss_corrector.py"


# --- helpers ----------------------------------------------------------------


def _synthetic(seed=0, frames=3, n_groups=6):
    """A small field with the shipped driving shape: causal groups, real rows."""
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


def _drive(corrector, groups, rng_rows, tokens, with_args=False):
    """Run a corrector over a synthetic field, collecting every coding row.

    ``with_args`` also returns each group's ``state.arg``.  That matters: the
    hit/miss split -- and therefore the column this module must leave alone -- is
    defined by the argmax of the PRE-correction row, and an odds multiplier below
    1 can move the argmax of the POST-correction row somewhere else.  A test that
    re-derives the column with ``argmax`` on the output is testing the wrong
    column whenever those two disagree.
    """
    out, args = [], []
    for frame in range(len(rng_rows)):
        corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
        for flat in groups:
            probability = rng_rows[frame][flat]
            predicted = probability.argmax(axis=1).astype(np.int64)
            state = corrector.group_state(probability, predicted, flat)
            out.append(corrector.coding_row(state))
            args.append(state.arg.copy())
            corrector.observe(state, tokens[frame][flat])
        corrector.end_frame(tokens[frame])
    return (out, args) if with_args else out


def _config(**overrides):
    cfg = dict(FX2_CONFIG)
    cfg.update(overrides)
    return cfg


# --- 1. exactness -----------------------------------------------------------


def test_module_has_no_transcendental_on_the_decision_path():
    """The ddm_rr2 refusal (S = 27.83) in one assertion.

    ``log``/``exp``/``pow`` are libm routines that are NOT correctly rounded and
    differ by an ULP across platforms; one ULP moves an RC64 integer frequency
    and desynchronises the decoder for the rest of the stream.  This module needs
    none of them -- the estimator is a ratio of smoothed counts.
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


# --- 2. nesting: the control every delta rests on ---------------------------


def test_within_miss_off_is_bit_identical_to_the_live_fx2_law():
    """POSITIVE CONTROL. With the sector off this class IS ddm_fx2's D1 build.

    Without this, a measured byte delta could be the new sector OR the extra
    subclass layer, and the two would be indistinguishable.
    """
    groups, rows, tokens = _synthetic(seed=11)
    live = _drive(Fx2ModelAxisMixer(PLANE, **FX2_CONFIG), groups, rows, tokens)
    off = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)), groups, rows, tokens
    )
    for a, b in zip(live, off, strict=True):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), "not bit-identical"


def test_within_miss_on_actually_changes_the_rows():
    """The sister of the nesting test: an inert stage would also pass it."""
    groups, rows, tokens = _synthetic(seed=12)
    off = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)), groups, rows, tokens
    )
    on = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    assert any(not np.array_equal(a, b) for a, b in zip(off, on, strict=True))


def test_a_cold_model_is_exactly_the_shipped_law():
    """A cell below ``min_count`` emits the prior's own relative law, exactly.

    Mirrors ddm_rr4's own contract that a cold context emits exactly HPAC.
    """
    groups, rows, tokens = _synthetic(seed=13)
    off = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)), groups, rows, tokens
    )
    cold = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=True, miss_min_count=10**9)),
        groups,
        rows,
        tokens,
    )
    for a, b in zip(off, cold, strict=True):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


# --- 3. the invariants that keep the hit event out of it --------------------


def test_the_argmax_column_is_never_touched():
    """d(hit event) == 0 by construction: q is whatever the inherited law said."""
    groups, rows, tokens = _synthetic(seed=14)
    off, args = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)),
        groups, rows, tokens, with_args=True,
    )
    on = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    for a, b, arg in zip(off, on, args, strict=True):
        index = np.arange(a.shape[0])
        assert np.array_equal(
            a[index, arg].view(np.uint32), b[index, arg].view(np.uint32)
        ), "the argmax column moved; the hit event is no longer separable"


def test_non_argmax_mass_is_preserved():
    """The reweight redistributes inside the sector, it does not resize it."""
    groups, rows, tokens = _synthetic(seed=15)
    off, args = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)),
        groups, rows, tokens, with_args=True,
    )
    on = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    for a, b, arg in zip(off, on, args, strict=True):
        index = np.arange(a.shape[0])
        mask = np.ones(a.shape, dtype=bool)
        mask[index, arg] = False
        np.testing.assert_allclose(
            np.where(mask, a, 0.0).sum(axis=1),
            np.where(mask, b, 0.0).sum(axis=1),
            rtol=1e-6,
            atol=1e-9,
        )


def test_rows_stay_valid_probabilities():
    groups, rows, tokens = _synthetic(seed=16)
    on = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    for row in on:
        assert np.all(np.isfinite(row))
        assert np.all(row > 0.0), "a zero column would make its symbol uncodable"
        np.testing.assert_allclose(row.astype(np.float64).sum(axis=1), 1.0, rtol=2e-5)


# --- 4. the learner is alive and clamped ------------------------------------


def test_the_learner_records_only_misses():
    """``observe`` folds a record exactly when the decoded symbol was a miss."""
    groups, rows, tokens = _synthetic(seed=17, frames=2)
    corrector = Ma1WithinMissCorrector(PLANE, **_config(within_miss=True))
    misses = 0
    for frame in range(2):
        corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
        for flat in groups:
            probability = rows[frame][flat]
            predicted = probability.argmax(axis=1).astype(np.int64)
            state = corrector.group_state(probability, predicted, flat)
            corrector.coding_row(state)
            symbols = tokens[frame][flat]
            misses += int(np.sum(symbols != state.arg))
            corrector.observe(state, symbols)
        corrector.end_frame(tokens[frame])
    assert int(corrector._miss_seen.sum()) == misses
    assert int(corrector._miss_counts.sum()) == misses


def test_the_multiplier_is_clamped():
    groups, rows, tokens = _synthetic(seed=18)
    corrector = Ma1WithinMissCorrector(PLANE, **_config(within_miss=True))
    _drive(corrector, groups, rows, tokens)
    warm = np.flatnonzero(corrector._miss_seen >= corrector._miss_min_count)
    assert warm.size, "nothing warmed up; the test is vacuous"
    m = corrector._miss_multiplier(warm)
    assert np.all(m >= MISS_CLAMP_LOW - 1e-12)
    assert np.all(m <= MISS_CLAMP_HIGH + 1e-12)


def test_expected_mass_accumulator_is_integer():
    """Fixed point, so ``np.add.at`` cannot depend on summation order."""
    corrector = Ma1WithinMissCorrector(PLANE, **_config(within_miss=True))
    assert corrector._miss_expect.dtype == np.int64
    assert corrector._miss_counts.dtype == np.int64
    assert corrector._miss_seen.dtype == np.int64


# --- 5. causality + the cell rules ------------------------------------------


def test_cells_are_sized_to_their_alphabet():
    for name, (size, rule) in miss_cells().items():
        nb = np.full((4, 64), UNKNOWN, dtype=np.int64)
        prev = np.full(64, UNKNOWN, dtype=np.int64)
        assert 0 <= int(rule(nb, prev).max()) < size, name
        rng = np.random.default_rng(3)
        nb = rng.integers(0, NUM_CLASSES + 1, size=(4, 4096))
        prev = rng.integers(0, NUM_CLASSES + 1, size=4096)
        index = rule(nb, prev)
        assert index.min() >= 0 and index.max() < size, name


def test_undecoded_neighbours_read_as_unknown_not_as_a_class():
    """A not-yet-decoded neighbour must be its own level.

    Folding it into class 0 would let "I have no information" pool with real
    evidence about the road class, which is both wrong and unfalsifiable.
    """
    corrector = Ma1WithinMissCorrector(PLANE, **_config(within_miss=True))
    corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
    flat = np.arange(64, dtype=np.int64)
    cell = corrector._miss_cell(flat)
    b = NUM_CLASSES + 1
    assert np.all(cell == (((UNKNOWN * b + UNKNOWN) * b + UNKNOWN) * b + UNKNOWN))


def test_the_cell_only_reads_already_decoded_positions():
    """CAUSALITY. Overwrite every not-yet-decoded token; the cell must not move."""
    groups, rows, tokens = _synthetic(seed=19, frames=1)
    corrector = Ma1WithinMissCorrector(PLANE, **_config(within_miss=True))
    corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
    flat = groups[0]
    probability = rows[0][flat]
    predicted = probability.argmax(axis=1).astype(np.int64)
    state = corrector.group_state(probability, predicted, flat)
    corrector.coding_row(state)
    corrector.observe(state, tokens[0][flat])

    later = groups[1]
    before = corrector._miss_cell(later)
    unknown_mask = ~corrector.known
    corrector.current[unknown_mask] = (corrector.current[unknown_mask] + 3) % NUM_CLASSES
    after = corrector._miss_cell(later)
    assert np.array_equal(before, after), "the cell read an undecoded position"


# --- 6. the shipping surface ------------------------------------------------


def test_receiver_drop_in_takes_no_arguments():
    """A decoder is constructed as ``FreeCorrector(plane)`` and nothing else."""
    corrector = FreeCorrector(PLANE)
    assert corrector._within_miss is True
    assert corrector._miss_cell_name == "nb3_prev1"


def test_shipped_config_keeps_the_inherited_half_frozen():
    """This arm adds a sector; it must not silently re-select fx2's model."""
    from experiments.ddm_ma1_within_miss_corrector import SHIPPED_CONFIG

    for key, value in FX2_CONFIG.items():
        assert SHIPPED_CONFIG[key] == value, f"inherited key {key} was changed"


def test_state_is_resumable_across_a_fresh_instance():
    """Two runs of the same field give the same rows: no hidden global state."""
    groups, rows, tokens = _synthetic(seed=20)
    a = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    b = _drive(Ma1WithinMissCorrector(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))


# --- 7. mutation checks: prove the suite is not vacuous ----------------------


def test_mutation_an_inert_stage_is_caught():
    """Defect: the reweight is computed and thrown away."""
    groups, rows, tokens = _synthetic(seed=21)

    class Inert(Ma1WithinMissCorrector):
        def coding_row(self, state):
            row = Fx2ModelAxisMixer.coding_row(self, state)
            self._miss_pending = self._miss_pending  # computed, never applied
            return row

    off = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)), groups, rows, tokens
    )
    inert = _drive(Inert(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    assert all(np.array_equal(a, b) for a, b in zip(off, inert, strict=True)), "setup"
    with pytest.raises(AssertionError):
        assert any(not np.array_equal(a, b) for a, b in zip(off, inert, strict=True))


def test_mutation_touching_the_argmax_column_is_caught():
    """Defect: the reweight is applied to all five columns, not four."""
    groups, rows, tokens = _synthetic(seed=22)

    class Leaky(Ma1WithinMissCorrector):
        def coding_row(self, state):
            row = super().coding_row(state)
            index = np.arange(state.arg.size)
            out = row.copy()
            out[index, state.arg] = (out[index, state.arg] * np.float32(0.999)).astype(np.float32)
            return out

    off, args = _drive(
        Ma1WithinMissCorrector(PLANE, **_config(within_miss=False)),
        groups, rows, tokens, with_args=True,
    )
    leaky = _drive(Leaky(PLANE, **_config(within_miss=True)), groups, rows, tokens)
    moved = False
    for a, b, arg in zip(off, leaky, args, strict=True):
        index = np.arange(a.shape[0])
        if not np.array_equal(a[index, arg].view(np.uint32), b[index, arg].view(np.uint32)):
            moved = True
            break
    assert moved, "the argmax-column invariant would not have caught this"


def test_mutation_a_non_causal_read_is_caught():
    """Defect: the cell reads the neighbour whether or not it is decoded."""
    groups, rows, tokens = _synthetic(seed=23, frames=1)

    class NonCausal(Ma1WithinMissCorrector):
        def _miss_cell(self, flat):
            from experiments.ddm_fx2_model_axis_corrector import CAUSAL_OFFSETS, HEIGHT, WIDTH

            x = flat % WIDTH
            y = flat // WIDTH
            nb = np.empty((len(CAUSAL_OFFSETS), flat.size), dtype=np.int64)
            for slot, (dx, dy) in enumerate(CAUSAL_OFFSETS):
                src = np.clip(y + dy, 0, HEIGHT - 1) * WIDTH + np.clip(x + dx, 0, WIDTH - 1)
                nb[slot] = self.current[src].astype(np.int64)  # no `known` gate
            prev1 = np.full(flat.size, UNKNOWN, dtype=np.int64)
            return self._miss_rule(nb, prev1)

    corrector = NonCausal(PLANE, **_config(within_miss=True))
    corrector.begin_frame(np.zeros(PLANE, dtype=np.int64))
    flat = groups[0]
    probability = rows[0][flat]
    state = corrector.group_state(probability, probability.argmax(axis=1).astype(np.int64), flat)
    corrector.coding_row(state)
    corrector.observe(state, tokens[0][flat])

    later = groups[1]
    before = corrector._miss_cell(later)
    unknown_mask = ~corrector.known
    corrector.current[unknown_mask] = (corrector.current[unknown_mask] + 3) % NUM_CLASSES
    after = corrector._miss_cell(later)
    assert not np.array_equal(before, after), "the causality test would not have caught this"
