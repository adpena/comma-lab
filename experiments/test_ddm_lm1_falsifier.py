#!/usr/bin/env python3
"""Correctness tests for the ddm_lm1 falsifier's load-bearing arithmetic.

The whole verdict rests on `score_rung` returning the EXACT hindsight-ideal
conditional entropy and the EXACT Krichevsky-Trofimov prequential code length,
and on `bars_for` implementing the corrected `ddm_no1` bar (see that memo's
CORRECTION section, commit 4257fa1006).  Both are checked here against
independent brute-force computations rather than against themselves.

Run:  PYTHONPATH=experiments .venv/bin/python -m pytest experiments/test_ddm_lm1_falsifier.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ddm_lm1_learned_model_falsifier import (
    DEMAND_BYTES,
    HPAC_DELTA,
    HPAC_PATCH,
    LADDER,
    N_CLASS,
    N_GROUPS,
    SHIPPED_MODEL_BYTES,
    SHIPPED_STREAM_BYTES,
    SHIPPED_TOTAL_BYTES,
    bars_for,
    group_index,
    score_rung,
)


def _brute_hindsight_bits(codes: np.ndarray, tokens: np.ndarray) -> float:
    """Independent hindsight-ideal: Sum_c Sum_t N_ct * log2(N_c / N_ct)."""
    table: dict[int, list[int]] = {}
    for c, t in zip(codes.tolist(), tokens.tolist(), strict=True):
        table.setdefault(c, [0] * N_CLASS)[t] += 1
    total = 0.0
    for counts in table.values():
        n_c = sum(counts)
        for n_ct in counts:
            if n_ct:
                total += n_ct * math.log2(n_c / n_ct)
    return total


def _brute_kt_bits(codes: np.ndarray, tokens: np.ndarray) -> float:
    """Independent KT: sequential Dirichlet(1/2) prediction, coded in order.

    This is deliberately the SEQUENTIAL form while `score_rung` uses the closed
    gamma form.  They must agree -- that equivalence is the reason the closed
    form is admissible as a real, achievable, zero-stored-byte code length.
    """
    counts: dict[int, list[float]] = {}
    total = 0.0
    for c, t in zip(codes.tolist(), tokens.tolist(), strict=True):
        row = counts.setdefault(c, [0.0] * N_CLASS)
        n = sum(row)
        p = (row[t] + 0.5) / (n + 0.5 * N_CLASS)
        total += -math.log2(p)
        row[t] += 1.0
    return total


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_score_rung_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 4000
    codes = rng.integers(0, 25, size=n).astype(np.int64)
    # Tokens correlated with context so the entropies are non-degenerate.
    tokens = ((codes % N_CLASS) + rng.integers(0, 2, size=n)) % N_CLASS
    tokens = tokens.astype(np.int64)

    hind, kt, n_ctx = score_rung(codes.copy(), tokens.copy())

    assert n_ctx == len(set(codes.tolist()))
    assert hind == pytest.approx(_brute_hindsight_bits(codes, tokens), rel=1e-9)
    assert kt == pytest.approx(_brute_kt_bits(codes, tokens), rel=1e-9)


def test_kt_never_below_hindsight() -> None:
    """The learning cost is non-negative: prequential >= hindsight-optimal.

    This is the property that lets the hindsight column bound adaptive models
    too, which section 5 of the memo relies on.
    """
    rng = np.random.default_rng(11)
    for n_ctx in (2, 17, 200, 3000):
        n = 20_000
        codes = rng.integers(0, n_ctx, size=n).astype(np.int64)
        tokens = rng.integers(0, N_CLASS, size=n).astype(np.int64)
        hind, kt, _ = score_rung(codes, tokens)
        assert kt >= hind - 1e-6, f"KT {kt} < hindsight {hind} at n_ctx={n_ctx}"


def test_hindsight_is_zero_for_deterministic_context() -> None:
    codes = np.arange(500, dtype=np.int64) % 50
    tokens = (codes % N_CLASS).astype(np.int64)
    hind, kt, _ = score_rung(codes, tokens)
    assert hind == pytest.approx(0.0, abs=1e-9)
    assert kt > 0.0  # an adaptive coder still pays to LEARN a deterministic map


def test_hindsight_is_maximal_for_independent_context() -> None:
    """Context carrying no information cannot beat the marginal entropy."""
    rng = np.random.default_rng(3)
    n = 60_000
    codes = rng.integers(0, 40, size=n).astype(np.int64)
    tokens = rng.integers(0, N_CLASS, size=n).astype(np.int64)
    hind, _, _ = score_rung(codes, tokens)
    marginal = n * math.log2(N_CLASS)
    assert hind <= marginal + 1e-6
    assert hind > marginal * 0.98  # close to it, since context is uninformative


def test_bars_match_the_corrected_no1_arithmetic() -> None:
    """break-even 127,292 - W; demand 127,292 - W - 42,228 = 85,064 - W."""
    assert SHIPPED_TOTAL_BYTES == SHIPPED_STREAM_BYTES + SHIPPED_MODEL_BYTES == 127_292
    for w in (0, 5_000, 10_000, 20_000):
        b = bars_for(100_000.0, w)
        assert b["break_even_bar_stream_bytes"] == 127_292 - w
        assert b["demand_bar_stream_bytes"] == 127_292 - w - DEMAND_BYTES == 85_064 - w


def test_bars_reproduce_no1_recovery_fractions() -> None:
    """The r-table 29.6% / 34.0% / 42.8% must fall out of the same bar."""
    expected = {5_000: 0.296, 10_000: 0.340, 20_000: 0.428}
    for w, r_expected in expected.items():
        r = (DEMAND_BYTES + w - SHIPPED_MODEL_BYTES) / SHIPPED_STREAM_BYTES
        assert r == pytest.approx(r_expected, abs=5e-4)
        # and the same r must land exactly on the demand bar
        stream = (1.0 - r) * SHIPPED_STREAM_BYTES
        assert stream == pytest.approx(bars_for(stream, w)["demand_bar_stream_bytes"], abs=1.0)


def test_bars_net_and_break_even_flag_are_consistent() -> None:
    b = bars_for(120_000.0, 5_000)  # total 125,000 < 127,292
    assert b["clears_break_even"] is True
    assert b["net_vs_shipped_bytes"] == pytest.approx(2_292.0)
    assert b["clears_demand"] is False
    worse = bars_for(193_065.0, 0)
    assert worse["clears_break_even"] is False
    assert worse["shortfall_vs_break_even_bytes"] == pytest.approx(193_065.0 - 127_292)


def test_decode_group_geometry_matches_shipped_runtime() -> None:
    """g = (x % 64) + 2 * (y % 64), 190 groups -- read from cpr1 at source."""
    assert (HPAC_PATCH, HPAC_DELTA) == (64, 2)
    assert N_GROUPS == (1 + HPAC_DELTA) * HPAC_PATCH - HPAC_DELTA == 190
    groups = group_index()
    assert groups.min() == 0 and groups.max() == N_GROUPS - 1
    assert groups[0, 1] == 1 and groups[1, 0] == 2 and groups[1, 1] == 3


def test_every_current_frame_ladder_offset_is_mask_a_causal() -> None:
    """A current-frame neighbour is legal only if dx + 2*dy < 0."""
    for kind, dy, dx in LADDER:
        if kind == "cur":
            assert dx + HPAC_DELTA * dy < 0, f"({dy},{dx}) is not causal"
    # the two documented boundary cases
    assert 1 * HPAC_DELTA + -3 < 0  # (+1,-3) legal by the skew
    assert 1 * HPAC_DELTA + -2 == 0  # (+1,-2) is SAME group, illegal


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
