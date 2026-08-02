# SPDX-License-Identifier: MIT
"""ddm_pw1 — guards for the two removed pose-search bounds.

The live v4d pose solve bounded two searches with hardcoded constants that the
shipped n600 solution SATURATED (dim0: 124/600 pairs at the +-0.048 coarse
bound carrying 37.4% of the pose mass; beta: 76/600 pairs at the top of the
3-entry menu carrying 26.4%).  Both were replaced by self-terminating outward
brackets.  These tests guard the two properties that make that replacement
safe, and the backward-compatibility property that keeps an archive rebuilt
from a pre-ddm_pw1 final JSONL byte-identical.

They test BEHAVIOUR, not constants: each would fail if the bracket stopped
expanding, accepted a worse point, left the quantization lattice, or if the
derived beta table stopped reducing to the seed menu on legacy rows.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load(name: str, relpath: str):
    """Import a repo script by path (experiments/ and tools/ are not packages)."""
    spec = importlib.util.spec_from_file_location(name, _REPO / relpath)
    if spec is None or spec.loader is None:  # pragma: no cover - env guard
        pytest.skip(f"cannot load {relpath}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pw1():
    return _load("_pw1_ab", "tools/pw1_pose_menu_saturation_ab.py")


# --------------------------------------------------------------------------- #
# The outward bracket: monotone-safe, self-terminating, lattice-respecting.
# --------------------------------------------------------------------------- #
def test_bracket_never_returns_a_worse_point(pw1):
    """Adversarial evaluator: every probe is worse than the start."""
    calls = []

    def worse(x):
        calls.append(x)
        return 100.0

    x, d, n, probes = pw1.bracket_out(worse, 0.0, 1.0, 0.5, 20)
    assert (x, d) == (0.0, 1.0)
    assert n == 2, "should probe both directions once, then stop"
    assert all(p["phase"] == "probe" for p in probes)


def test_bracket_reaches_a_minimum_outside_the_first_step(pw1):
    """The whole point: the optimum sits far past the seed step."""
    # Minimum at 8.0; the first outward step is only 0.5, so reaching it
    # REQUIRES the doubling expansion.
    x, d, n, probes = pw1.bracket_out(lambda v: (v - 8.0) ** 2, 0.0, 64.0,
                                      0.5, 20)
    assert any(p["phase"] == "expand" for p in probes), "never expanded"
    assert d < 64.0 and abs(x - 8.0) < 8.0
    # monotone: the reported d is the best probe seen
    assert d == min([64.0] + [p["d"] for p in probes])


def test_bracket_terminates_on_a_monotonically_improving_evaluator(pw1):
    """Unbounded improvement must still stop at the doubling cap."""
    x, d, n, _ = pw1.bracket_out(lambda v: -v, 0.0, 0.0, 0.5, 7)
    assert n <= 2 + 7, "expansion exceeded its doubling bound"


def test_bracket_stays_on_the_quantization_lattice(pw1):
    """Every returned dim0 must be a value the receiver can reconstruct."""
    def snap(v):
        return float(np.round(v * 4.0) / 4.0)

    x, d, n, probes = pw1.bracket_out(lambda v: (v - 3.0) ** 2, 0.0, 9.0,
                                      0.5, 20, quantize=snap)
    assert x == snap(x)
    assert all(p["x"] == snap(p["x"]) for p in probes)


def test_bracket_zero_cost_claim_is_two_evaluations(pw1):
    """The 'costs nothing when the bound did not bind' claim, measured."""
    n_calls = 0

    def f(v):
        nonlocal n_calls
        n_calls += 1
        return abs(v) + 1.0

    pw1.bracket_out(f, 0.0, 1.0, 0.5, 20)
    assert n_calls == 2


# --------------------------------------------------------------------------- #
# The derived beta table: legacy rows must reduce to the seed menu exactly.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def builder():
    return _load("_pw1_build", "experiments/ddm_v4d_build_composed_archive.py")


def test_legacy_rows_reproduce_the_seed_table_and_identity_indices(builder):
    """This is the byte-identical-rebuild guard, as a unit property.

    A pre-ddm_pw1 final JSONL carries only beta_idx.  If every seed entry is
    used, the derived table MUST equal BETA_MAGS and the index map MUST be the
    identity -- otherwise the rebuilt archive's beta section changes and the
    measured sha f1f3288062... no longer reproduces.
    """
    final = {0: {"beta_idx": 0}, 1: {"beta_idx": 2},
             2: {"beta_idx": 1}, 3: {"beta_idx": 2}}
    table, idx = builder.derive_beta_table(final, 4)
    assert table == list(builder.BETA_MAGS)
    assert [int(v) for v in idx] == [0, 2, 1, 2]
    assert idx.dtype == np.uint8


def test_extended_magnitudes_sort_into_the_table_and_index_correctly(builder):
    """Negative (against yaw) and >1.0 entries both survive into the table."""
    final = {0: {"beta_mag": -3.5}, 1: {"beta_mag": 0.0},
             2: {"beta_mag": 2.5}, 3: {"beta_mag": -3.5}}
    table, idx = builder.derive_beta_table(final, 4)
    assert table == [-3.5, 0.0, 2.5]
    assert [int(v) for v in idx] == [0, 1, 2, 0]


def test_beta_mag_wins_over_a_stale_beta_idx_on_the_same_row(builder):
    """Mixed rows: the explicit magnitude is authoritative."""
    final = {0: {"beta_idx": -1, "beta_mag": 4.5}, 1: {"beta_idx": 0}}
    table, idx = builder.derive_beta_table(final, 2)
    assert table == [0.0, 4.5]
    assert [int(v) for v in idx] == [1, 0]


def test_table_larger_than_uint8_is_refused(builder):
    """beta_idx is a uint8 column; a 257-entry table must fail closed."""
    final = {i: {"beta_mag": float(i)} for i in range(257)}
    with pytest.raises(SystemExit):
        builder.derive_beta_table(final, 257)


def test_unrecoverable_beta_index_fails_closed(builder):
    """beta_idx=-1 marks an extended magnitude; it must NOT wrap to the last
    seed entry.  This is the negative-index class, not one call site."""
    with pytest.raises(SystemExit):
        builder.derive_beta_table({0: {"beta_idx": -1}}, 1)
