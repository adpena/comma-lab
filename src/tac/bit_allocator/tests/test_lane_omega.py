# SPDX-License-Identifier: MIT
"""Tests for src/tac/bit_allocator/lane_omega.py — Lane Ω water-fill bit allocator.

Originally at ``src/tac/tests/test_bit_allocator.py`` (legacy module location);
renamed by WAVE-3-FORENSIC-FIX-2 2026-05-20 to match the renamed Lane Ω
submodule ``src/tac/bit_allocator/lane_omega.py`` (the legacy module was
shadowed when the sister WAVE-2-PREREQ-BIT-ALLOCATOR wave landed the
``src/tac/bit_allocator/`` package 2026-05-19). The package's ``__init__.py``
re-exports ``allocate_bits`` / ``allocation_report`` / ``DEFAULT_*`` so the
public API surface ``from tac.bit_allocator import allocate_bits`` is
preserved without modification.

Pins these properties:

  1. Budget conservation: sum(bits) ≤ total_bits ALWAYS (no overshoot).
  2. Determinism: same input → same output (no random / nondeterministic ops).
  3. Monotonic in importance: w_a > w_b → bits[w_a] ≥ bits[w_b].
  4. Bit floor / ceiling: every assigned bit is in [min_bits, max_bits].
  5. Shape preservation: output tensor for layer L has same shape as input
     importance tensor for L.
  6. Allocation report sanity: histogram sums to total_weights, mean is
     within [min_bits, max_bits], per_layer entries cover every layer.
  7. Edge cases: empty importance raises, infeasible budget raises, all-zero
     importance falls back to flat min_bits.
"""
from __future__ import annotations

import pytest
import torch

import tac.bit_allocator.lane_omega as lane_omega
from tac.bit_allocator import (
    DEFAULT_ALPHA,
    DEFAULT_MAX_BITS,
    DEFAULT_MIN_BITS,
    allocate_bits,
    allocation_report,
)


# ── Budget conservation ───────────────────────────────────────────────────


def test_budget_not_exceeded_simple():
    imp = {
        "a.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "b.weight": torch.tensor([0.5, 0.1, 0.01]),
    }
    bits = allocate_bits(imp, total_bits=20, min_bits=1, max_bits=8)
    s = sum(int(t.sum().item()) for t in bits.values())
    assert s <= 20, f"sum(bits) = {s} exceeds budget 20"


@pytest.mark.parametrize("budget", [10, 50, 200, 1000])
def test_budget_not_exceeded_random(budget: int):
    torch.manual_seed(42)
    imp = {f"layer_{i}.weight": torch.rand(8, 8) for i in range(5)}
    n = sum(t.numel() for t in imp.values())
    if budget < n:  # would be infeasible
        with pytest.raises(ValueError, match="infeasible budget"):
            allocate_bits(imp, total_bits=budget, min_bits=1, max_bits=8)
        return
    bits = allocate_bits(imp, total_bits=budget, min_bits=1, max_bits=8)
    s = sum(int(t.sum().item()) for t in bits.values())
    assert s <= budget


# ── Determinism ───────────────────────────────────────────────────────────


def test_determinism_same_input_same_output():
    torch.manual_seed(123)
    imp = {f"l{i}.weight": torch.rand(4, 4) for i in range(3)}
    # 3 * 16 = 48 weights → budget 96 = 2 bits/weight average
    out1 = allocate_bits(imp, total_bits=96, alpha=0.5)
    out2 = allocate_bits(imp, total_bits=96, alpha=0.5)
    for k in out1:
        assert torch.equal(out1[k], out2[k]), f"non-deterministic at layer {k!r}"


# ── Monotonic in importance ───────────────────────────────────────────────


def test_monotonic_in_importance():
    """For two weights in the same layer with importance a > b, we must have
    bits[a] >= bits[b]. The water-fill is monotonic in I."""
    imp = {"l.weight": torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])}
    bits = allocate_bits(imp, total_bits=20, min_bits=1, max_bits=8)
    b = bits["l.weight"].tolist()
    for i in range(len(b) - 1):
        assert b[i] <= b[i + 1], (
            f"non-monotonic bits at index {i}: {b[i]} > {b[i + 1]} "
            f"despite imp[{i}] < imp[{i + 1}]"
        )


# ── Bit range constraints ─────────────────────────────────────────────────


def test_bits_within_range():
    imp = {"l.weight": torch.tensor([0.1, 1.0, 10.0, 100.0, 1000.0])}
    bits = allocate_bits(imp, total_bits=30, min_bits=2, max_bits=6)
    b = bits["l.weight"]
    assert b.min().item() >= 2
    assert b.max().item() <= 6


def test_bits_floor_when_budget_exceeds_all_max():
    """If budget ≥ n × max_bits, every weight gets max_bits."""
    imp = {"l.weight": torch.tensor([1.0, 2.0, 3.0, 4.0])}
    bits = allocate_bits(imp, total_bits=999_999, min_bits=1, max_bits=8)
    assert (bits["l.weight"] == 8).all(), (
        f"expected all 8s when budget overflows, got {bits['l.weight'].tolist()}"
    )


# ── Shape preservation ────────────────────────────────────────────────────


def test_output_shape_matches_input():
    imp = {
        "a": torch.zeros(3, 4, 5),
        "b": torch.ones(7, 2),
        "c": torch.full((10,), 0.5),
    }
    bits = allocate_bits(imp, total_bits=120, min_bits=1, max_bits=4)
    assert bits["a"].shape == (3, 4, 5)
    assert bits["b"].shape == (7, 2)
    assert bits["c"].shape == (10,)
    for v in bits.values():
        assert v.dtype == torch.uint8


# ── Allocation report ─────────────────────────────────────────────────────


def test_allocation_report_sanity():
    torch.manual_seed(7)
    imp = {f"l{i}.weight": torch.rand(8) for i in range(4)}
    bits = allocate_bits(imp, total_bits=80, min_bits=1, max_bits=8)
    rep = allocation_report(bits, imp)
    assert rep["total_weights"] == 32
    assert sum(rep["bits_histogram_0_8"]) == 32
    assert rep["min_bits"] >= 1
    assert rep["max_bits"] <= 8
    assert set(rep["per_layer"].keys()) == set(bits.keys())
    for layer, info in rep["per_layer"].items():
        assert info["n"] == bits[layer].numel()
        assert info["total_bits"] == int(bits[layer].sum().item())


# ── Edge cases ────────────────────────────────────────────────────────────


def test_empty_importance_raises():
    with pytest.raises(ValueError, match="empty"):
        allocate_bits({}, total_bits=10)


def test_infeasible_budget_raises():
    imp = {"l.weight": torch.tensor([1.0, 1.0, 1.0])}  # 3 weights
    # 3 weights × min_bits=4 = 12 floor, budget 5 → infeasible
    with pytest.raises(ValueError, match="infeasible"):
        allocate_bits(imp, total_bits=5, min_bits=4, max_bits=8)


def test_all_zero_importance_falls_back_to_min():
    imp = {"l.weight": torch.zeros(5)}
    bits = allocate_bits(imp, total_bits=10, min_bits=1, max_bits=8)
    assert (bits["l.weight"] == 1).all()


def test_negative_importance_rejected():
    imp = {"l.weight": torch.tensor([-1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="negative"):
        allocate_bits(imp, total_bits=20)


def test_nonfinite_importance_rejected():
    imp = {"l.weight": torch.tensor([1.0, float("nan"), 3.0])}
    with pytest.raises(ValueError, match="non-finite"):
        allocate_bits(imp, total_bits=20)


def test_invalid_bit_range():
    imp = {"l.weight": torch.tensor([1.0, 2.0])}
    with pytest.raises(ValueError, match="invalid bit range"):
        allocate_bits(imp, total_bits=10, min_bits=5, max_bits=3)


def test_invalid_alpha():
    imp = {"l.weight": torch.tensor([1.0, 2.0])}
    with pytest.raises(ValueError, match="alpha"):
        allocate_bits(imp, total_bits=10, alpha=0.0)


def test_zero_total_bits_raises():
    imp = {"l.weight": torch.tensor([1.0, 2.0])}
    with pytest.raises(ValueError, match="positive"):
        allocate_bits(imp, total_bits=0)


# ── Cross-layer interaction ───────────────────────────────────────────────


def test_high_importance_layer_gets_more_bits_than_low():
    """A layer with uniformly high importance should get more bits/weight on
    average than a layer with uniformly low importance, given the same n."""
    imp = {
        "high.weight": torch.full((20,), 100.0),
        "low.weight": torch.full((20,), 0.01),
    }
    bits = allocate_bits(imp, total_bits=80, min_bits=1, max_bits=8)
    high_mean = float(bits["high.weight"].float().mean().item())
    low_mean = float(bits["low.weight"].float().mean().item())
    assert high_mean > low_mean, (
        f"importance-based allocation should give high-imp layer more bits "
        f"on average; got high={high_mean}, low={low_mean}"
    )


# ── Default constants are sensible ────────────────────────────────────────


def test_defaults_are_safe():
    assert DEFAULT_MIN_BITS >= 1
    assert DEFAULT_MAX_BITS <= 8
    assert 0 < DEFAULT_ALPHA <= 1.0


# ── Round 10 bracket growth regression ───────────────────────────────────


def test_round10_counterexample_spends_budget_after_bracket_growth():
    imp = {"l.weight": torch.tensor([100.0, 1.0])}
    bits = allocate_bits(imp, total_bits=12, alpha=0.5, min_bits=1, max_bits=8)
    total = int(bits["l.weight"].sum().item())
    assert total >= 11
    assert total <= 12


def test_round10_budget_fully_spent_when_no_weight_at_max_bits():
    imp = {"l.weight": torch.tensor([1.0, 4.0, 9.0, 16.0])}
    bits = allocate_bits(imp, total_bits=18, alpha=0.5, min_bits=1, max_bits=8)
    b = bits["l.weight"].to(torch.int64)
    assert int(b.sum().item()) == 18
    assert int(b.max().item()) < 8


def test_codex_finding3_bracket_grows_when_initial_c_hi_underspends():
    """Codex finding 3 anchor — VALUE/SIGN regression.

    With importance=[100, 1] and total_bits=12 the previous allocator
    converged to bits=[8, 1] (sum=9) because c_hi was capped at the value
    that saturates the highest-importance weight. The bracket-growth fix
    must spend the full budget by giving the low-importance weight more
    bits while the high-importance weight stays at max_bits.

    This test pins both the SIGN of the fix (more bits to the low-imp
    weight) and the VALUE (sum lands within 1 of total_bits).
    """
    imp = {"l.weight": torch.tensor([100.0, 1.0])}
    bits = allocate_bits(imp, total_bits=12, alpha=0.5, min_bits=1, max_bits=8)
    b = bits["l.weight"].tolist()
    assert b[0] == 8, f"high-imp weight should saturate at max_bits=8, got {b[0]}"
    assert b[1] >= 4, f"low-imp weight should rise above min_bits, got {b[1]}"
    total = b[0] + b[1]
    assert total >= 11, (
        f"total bits should approach budget=12 (got {total}); "
        f"bracket-growth fix is regressed"
    )
    assert total <= 12


def test_round10_bracket_growth_safety_cap_prevents_infinite_loop(monkeypatch):
    calls = {"n": 0}

    def fake_bits_for_c(flat_imp, c, alpha, min_bits, max_bits):
        calls["n"] += 1
        return torch.full_like(flat_imp, min_bits, dtype=torch.int64)

    monkeypatch.setattr(lane_omega, "_bits_for_c", fake_bits_for_c)
    bits = lane_omega.allocate_bits(
        {"l.weight": torch.tensor([1.0, 2.0])},
        total_bits=3,
        alpha=0.5,
        min_bits=1,
        max_bits=8,
    )

    total = int(bits["l.weight"].sum().item())
    assert 2 <= total <= 3
    assert calls["n"] < 120
