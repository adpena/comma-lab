"""Tests for the ddm_rr4 platform-exact free corrector.

These are the falsifiers ddm_rr4 pre-registered.  The load-bearing ones are
``test_no_transcendental_*`` (v2 provably never evaluates a libm routine in the
decision path, which is what makes encoder and decoder agree across platforms)
and ``test_idempotent_*`` (a cold context emits exactly HPAC, bit for bit, which
ddm_rr2's v1 failed on 50.09% of real positions).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ddm_rr2_free_corrector as v1
import ddm_rr4_free_corrector_v2 as v2

MODULE_PATH = Path(v2.__file__)
BANNED_CALLS = {"log", "log2", "log10", "exp", "exp2", "expm1", "log1p", "power", "float_power"}


def probability_table(logits: np.ndarray, precision: int = 8) -> np.ndarray:
    """The receiver's own _probability_table, reproduced for the fixtures."""
    quantized = np.clip(np.rint(np.asarray(logits, dtype=np.float32) * precision), -32768, 32767).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def make_rows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.integers(-160, 160, size=(n, v2.NUM_CLASSES)).astype(np.float32) / 8.0
    return probability_table(logits)


def warm(corrector, rows, rounds: int = 6) -> None:
    """Drive enough groups that contexts pass MIN_COUNT and multipliers move."""
    rng = np.random.default_rng(7)
    positions = np.arange(rows.shape[0])
    corrector.begin_frame(rng.integers(0, v2.BOUNDARY_LEVELS, size=rows.shape[0]))
    for _ in range(rounds):
        state = corrector.group_state(rows, rows.argmax(axis=1), positions)
        corrector.coding_row(state)
        corrector.observe(state, rng.integers(0, v2.NUM_CLASSES, size=rows.shape[0]))


# --- the platform-exactness falsifiers ---------------------------------------


def test_no_transcendental_in_source() -> None:
    """AST walk: the module must not name a libm routine anywhere."""
    tree = ast.parse(MODULE_PATH.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in BANNED_CALLS:
            offenders.append(node.attr)
        if isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            offenders.append(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            offenders.append("**")
    assert offenders == [], f"transcendental reached the decision path: {sorted(set(offenders))}"


def test_no_transcendental_at_runtime(monkeypatch) -> None:
    """Runtime proof: with libm poisoned, a warm correction still completes."""
    rows = make_rows(4096)
    corrector = v2.FreeCorrector(rows.shape[0])
    warm(corrector, rows)

    def poisoned(*_args, **_kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("v2 called a transcendental")

    for name in ("log", "log2", "log10", "exp", "exp2", "power", "float_power"):
        monkeypatch.setattr(np, name, poisoned)

    positions = np.arange(rows.shape[0])
    state = corrector.group_state(rows, rows.argmax(axis=1), positions)
    multiplier = corrector.odds_multiplier(state)
    row = corrector.coding_row(state)
    assert np.any(multiplier != 1.0), "fixture did not warm any context"
    assert np.all(np.isfinite(row))


def test_idempotent_on_cold_contexts_bit_exact() -> None:
    """A cold context must emit the receiver's own bytes, unchanged."""
    rows = make_rows(200_000)
    corrector = v2.FreeCorrector(rows.shape[0])
    corrector.begin_frame(np.zeros(rows.shape[0], dtype=np.int64))
    state = corrector.group_state(rows, rows.argmax(axis=1), np.arange(rows.shape[0]))
    assert np.all(corrector.odds_multiplier(state) == 1.0)
    assert np.array_equal(corrector.coding_row(state), rows)


def test_v1_is_not_idempotent_the_bug_this_fixes() -> None:
    """The regression guard: v1's cold rows are NOT the receiver's bytes."""
    rows = make_rows(200_000)
    corrector = v1.FreeCorrector(rows.shape[0])
    corrector.begin_frame(np.zeros(rows.shape[0], dtype=np.int64))
    state = corrector.group_state(rows, rows.argmax(axis=1), np.arange(rows.shape[0]))
    assert np.all(corrector.delta(state) == 0.0)
    assert not np.array_equal(corrector.coding_row(state), rows)


# --- the estimator is unchanged, only its arithmetic is ----------------------


def test_ubin_matches_the_v1_partition() -> None:
    """v2's threshold table reproduces floor(-log2(1-p)/U_STEP)."""
    one_minus = np.concatenate(
        [
            np.geomspace(1e-12, 0.999, 200_000),
            np.ldexp(1.0, -np.arange(0, 32)),  # exact bin edges
        ]
    )
    reference = np.clip((-np.log2(one_minus) / v2.U_STEP).astype(np.int64), 0, v2.U_BINS - 1)
    below = np.searchsorted(v2._SURPRISE_ASC, one_minus, side="left")
    measured = np.clip((v2.U_BINS - 1) - below, 0, v2.U_BINS - 1)
    disagree = int((reference != measured).sum())
    assert disagree / one_minus.size < 1e-4, f"partition drifted on {disagree} of {one_minus.size}"


def test_odds_multiplier_equals_two_to_the_v1_delta() -> None:
    """v2's ratio must equal 2**delta from v1, to float64 round-off."""
    rows = make_rows(8192, seed=3)
    positions = np.arange(rows.shape[0])
    rng = np.random.default_rng(11)
    symbols = [rng.integers(0, v2.NUM_CLASSES, size=rows.shape[0]) for _ in range(8)]

    a, b = v1.FreeCorrector(rows.shape[0]), v2.FreeCorrector(rows.shape[0])
    boundary = rng.integers(0, v2.BOUNDARY_LEVELS, size=rows.shape[0])
    a.begin_frame(boundary)
    b.begin_frame(boundary)
    for drawn in symbols:
        sa = a.group_state(rows, rows.argmax(axis=1), positions)
        sb = b.group_state(rows, rows.argmax(axis=1), positions)
        a.observe(sa, drawn)
        b.observe(sb, drawn)

    sa = a.group_state(rows, rows.argmax(axis=1), positions)
    sb = b.group_state(rows, rows.argmax(axis=1), positions)
    expected = np.exp2(np.clip(a.delta(sa), -v2.DELTA_CLIP, v2.DELTA_CLIP))
    measured = b.odds_multiplier(sb)
    assert np.allclose(measured, expected, rtol=1e-9, atol=0.0)


def test_corrected_row_is_a_distribution_the_coder_accepts() -> None:
    """RC64 refuses rows outside [0.99998, 1.00002] or with a zero entry."""
    rows = make_rows(8192, seed=5)
    corrector = v2.FreeCorrector(rows.shape[0])
    warm(corrector, rows)
    state = corrector.group_state(rows, rows.argmax(axis=1), np.arange(rows.shape[0]))
    row = corrector.coding_row(state).astype(np.float64)
    assert np.all(row > 0.0)
    total = row.sum(axis=1)
    assert np.all(total > 0.99998) and np.all(total < 1.00002)


def test_argmax_mass_moves_in_the_direction_of_the_multiplier() -> None:
    rows = make_rows(8192, seed=9)
    corrector = v2.FreeCorrector(rows.shape[0])
    warm(corrector, rows)
    state = corrector.group_state(rows, rows.argmax(axis=1), np.arange(rows.shape[0]))
    multiplier = corrector.odds_multiplier(state)
    row = corrector.coding_row(state).astype(np.float64)
    got = row[np.arange(row.shape[0]), state.arg]
    up, down = multiplier > 1.0, multiplier < 1.0
    assert np.all(got[up] >= state.p_max[up] - 1e-12)
    assert np.all(got[down] <= state.p_max[down] + 1e-12)


# --- housekeeping ------------------------------------------------------------


def test_state_dict_round_trip() -> None:
    rows = make_rows(2048, seed=13)
    corrector = v2.FreeCorrector(rows.shape[0])
    warm(corrector, rows)
    restored = v2.FreeCorrector(rows.shape[0])
    restored.load_state_dict(corrector.state_dict())
    for field in ("counts", "hits", "phat_q", "prev1", "prev2", "run"):
        assert np.array_equal(getattr(restored, field), getattr(corrector, field))
    assert restored.have_prev == corrector.have_prev


def test_accumulator_is_integer_and_order_independent() -> None:
    """int64 counters make np.add.at's summation order irrelevant."""
    rows = make_rows(4096, seed=17)
    corrector = v2.FreeCorrector(rows.shape[0])
    warm(corrector, rows)
    assert corrector.counts.dtype == np.int64
    assert corrector.hits.dtype == np.int64
    assert corrector.phat_q.dtype == np.int64


def test_api_matches_v1() -> None:
    for name in ("begin_frame", "group_state", "coding_row", "observe", "end_frame", "state_dict", "load_state_dict"):
        assert hasattr(v2.FreeCorrector, name), name


def test_shape_guards() -> None:
    corrector = v2.FreeCorrector(16)
    with pytest.raises(ValueError):
        corrector.begin_frame(np.zeros(4, dtype=np.int64))
    corrector.begin_frame(np.zeros(16, dtype=np.int64))
    with pytest.raises(ValueError):
        corrector.group_state(np.zeros((16, 3), dtype=np.float32), np.zeros(16), np.arange(16))
