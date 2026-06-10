#!/usr/bin/env python
"""NO-FAKE behavioral tests for the #51 seg-repair carrier economics verdict.

These guard the load-bearing CLAIMS of the verdict memo (so a future agent cannot
silently re-introduce the falsified "seg-repair sidecar works" assumption):

1. The information-theoretic position floor for addressing scattered flips EXCEEDS
   THE LAW break-even (the carrier is provably incapable, not just empirically).
2. The honest brotli-coded carrier bytes for scattered flips EXCEED 1.27 B/flip.
3. The per-flip score value (8.48e-7) and break-even (1.27 B/flip) constants are
   exactly the contest-derived values (no fabricated economics).

Run: .venv/bin/python -m pytest experiments/results/frontier_seg_repair_pool_20260610/test_carrier_economics.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screen_repair_atoms import code_correction_bytes  # noqa: E402

N_CONTEST = 37_545_489
RATE_COEF = 25.0
SCH, SCW = 384, 512


def _break_even_bytes_per_flip() -> float:
    """Contest-derived LAW break-even: a fixed flip is worth 100/(600*H*W) score;
    a byte costs 25/N. Break-even = value / cost-per-byte."""
    per_flip_score = 100.0 * 1.0 / (600 * SCH * SCW)
    cost_per_byte = RATE_COEF / N_CONTEST
    return per_flip_score / cost_per_byte


def test_break_even_constant_is_contest_derived():
    be = _break_even_bytes_per_flip()
    # the verdict cites 1.27 B/flip
    assert 1.26 < be < 1.28, f"break-even drifted from contest derivation: {be}"


def test_per_flip_score_value_is_8e7():
    per_flip = 100.0 * 1.0 / (600 * SCH * SCW)
    assert abs(per_flip - 8.477e-7) < 1e-9, per_flip


def test_information_theoretic_position_floor_exceeds_break_even():
    """log2(C(M, K)) / K (bytes/flip, position-only floor) > 1.27 break-even,
    for the measured mean K=110 flips/pair. This is the rigorous clincher: even
    a perfect entropy coder of positions cannot clear THE LAW."""
    M = SCH * SCW
    K = 110
    logC_bits = (math.lgamma(M + 1) - math.lgamma(K + 1) - math.lgamma(M - K + 1)) / math.log(2)
    floor_bytes_per_flip = (logC_bits / 8.0) / K
    be = _break_even_bytes_per_flip()
    assert floor_bytes_per_flip > be, (
        f"position floor {floor_bytes_per_flip:.3f} should exceed break-even {be:.3f} "
        "(if this fails, the seg-repair sidecar may be viable — re-open the lane)"
    )
    # the verdict cites 1.525
    assert 1.5 < floor_bytes_per_flip < 1.55, floor_bytes_per_flip


def test_honest_carrier_bytes_exceed_break_even_for_scattered_flips():
    """A realistic scattered flip pattern (110 flips spread over rows 171-292, all
    columns) codes to > 1.27 honest brotli bytes/flip — the empirical confirmation
    of the floor. NO synthetic-uniform shortcut: positions mimic the measured
    middle-band scatter."""
    rng = np.random.default_rng(1234)
    n = 110
    rows = rng.integers(171, 293, size=n)
    cols = rng.integers(0, SCW, size=n)
    support = np.zeros((SCH, SCW), dtype=bool)
    support[rows, cols] = True
    n_actual = int(support.sum())
    # correction values: small int8 (the gradient-sign * step typical magnitude)
    corr = np.zeros((SCH, SCW, 3), dtype=np.float64)
    corr[support] = rng.integers(-16, 17, size=(n_actual, 3)).astype(np.float64)
    cbytes = code_correction_bytes(support, corr)
    bpf = cbytes / n_actual
    be = _break_even_bytes_per_flip()
    assert bpf > be, (
        f"honest carrier {bpf:.3f} B/flip should exceed break-even {be:.3f} "
        "(scattered flips are entropy-expensive to address)"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
