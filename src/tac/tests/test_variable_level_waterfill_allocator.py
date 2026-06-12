# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the math-optimal reverse-waterfill allocator
(``tac.losses.variable_level_waterfill_allocator``).

The allocator consumes a MEASURED per-tensor RD table and solves the KKT reverse-
waterfill (spend coarsening budget where distortion is cheapest, stop where the marginal
distortion-per-byte exceeds the byte value). These tests verify the MECHANISM, not
constants:

* Lens-1 (the prompt's KKT requirement): the accepted marginals are MONOTONE
  non-decreasing and bounded by the byte value, with the rejected frontier ABOVE it —
  the marginal-equalization condition HOLDS (not a degenerate constant). A FAKE allocator
  that returned a fixed level would fail the "distinct steps walking up to the boundary"
  structure.
* the greedy lands a STRICTLY BETTER net than the crude uniform-band allocation on a
  table where the uniform band mis-spends (coarsens a pose-sensitive tensor).
* the byte-target solve hits the target at MINIMUM distortion (cheapest-first).
* default-preserving: an empty / all-127 table gives an all-127 allocation (byte-identical
  codec path), and the net is exactly 0.
* the net composition matches the contest-score arithmetic bit-for-bit.
"""
from __future__ import annotations

import math

from tac.losses.variable_level_waterfill_allocator import (
    DEFAULT_LEVEL_GRID,
    byte_saving_score_value,
    lower_convex_hull_levels,
    net_score_delta_from_components,
    solve_waterfill_allocation,
    sqrt_pose_term,
    verify_kkt_marginal_equalization,
)

_N = 37_545_489


def _rd(byte_saving: float, dist_cost: float) -> tuple[float, float]:
    return (float(byte_saving), float(dist_cost))


def _convex_curve(byte_steps, dist_steps) -> dict[int, tuple[float, float]]:
    """Build a cumulative RD curve over DEFAULT_LEVEL_GRID from per-step marginals.

    ``byte_steps``/``dist_steps`` are the marginal (saving, cost) for each coarsening hop
    127->96->64->48->32->16. The 127 entry is (0,0). Cumulative sums form the curve.
    """
    grid = sorted(set(DEFAULT_LEVEL_GRID), reverse=True)  # 127,96,64,48,32,16
    curve: dict[int, tuple[float, float]] = {grid[0]: (0.0, 0.0)}
    cb = cd = 0.0
    for i, lv in enumerate(grid[1:]):
        cb += byte_steps[i]
        cd += dist_steps[i]
        curve[lv] = (cb, cd)
    return curve


def test_all_127_table_gives_all_127_allocation_zero_net():
    """An empty coarsening (every tensor only has the 127 entry) -> all-127 allocation,
    net exactly 0 (the default-preserving / byte-identical path)."""
    rd_table = {"a": {127: (0.0, 0.0)}, "b": {127: (0.0, 0.0)}}
    alloc = solve_waterfill_allocation(rd_table)
    assert all(v == 127 for v in alloc.levels.values())
    assert alloc.net_score_delta == 0.0
    assert alloc.n_coarsened == 0
    holds, _ = verify_kkt_marginal_equalization(alloc)
    assert holds


def test_kkt_marginal_equalization_holds_not_constant():
    """Lens-1: on a multi-tensor table the accepted marginals are DISTINCT and monotone
    non-decreasing, bounded by the byte value, with the rejected frontier above it —
    the KKT marginal-equalization condition (not a constant)."""
    # tensor X: very cheap distortion-per-byte (coarsen hard); tensor Y: expensive.
    # Use enough byte saving that some steps are net-positive and some are net-negative,
    # so the greedy has a real frontier.
    bv = byte_saving_score_value(1.0)
    # cheap tensor: 2000 B/step, tiny dist cost so ratio << byte value early
    x_byte = [2000, 2000, 1500, 1000, 800]
    x_dist = [bv * 500, bv * 900, bv * 1400, bv * 1800, bv * 5000]  # rising ratio
    # expensive tensor: small byte saving, large dist cost (ratio > byte value always)
    y_byte = [300, 200, 150, 100, 80]
    y_dist = [bv * 10000, bv * 20000, bv * 30000, bv * 40000, bv * 50000]
    rd_table = {
        "x": _convex_curve(x_byte, x_dist),
        "y": _convex_curve(y_byte, y_dist),
    }
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    holds, msg = verify_kkt_marginal_equalization(alloc)
    assert holds, msg
    accepted = [s for s in alloc.trace if s.accepted]
    assert len(accepted) >= 2, "expected multiple accepted coarsening steps"
    ratios = [s.marginal_ratio for s in accepted]
    # distinct (not a constant) AND monotone non-decreasing
    assert len({round(r, 18) for r in ratios}) >= 2, f"marginals are constant: {ratios}"
    assert all(ratios[i] <= ratios[i + 1] + 1e-12 for i in range(len(ratios) - 1))
    # the expensive tensor Y should be (mostly) protected: its ratio exceeds the byte value
    # so the greedy leaves it at/near 127.
    assert alloc.levels["y"] >= alloc.levels["x"], (
        f"expensive tensor Y ({alloc.levels['y']}) coarsened more than cheap X "
        f"({alloc.levels['x']}) — waterfill spent on the wrong tensor"
    )


def test_rejected_frontier_above_byte_value():
    """The first rejected step's marginal ratio is >= the byte value, and the max accepted
    ratio is <= the byte value (the KKT boundary brackets the byte value)."""
    bv = byte_saving_score_value(1.0)
    # one tensor whose marginal ratio crosses the byte value mid-curve
    t_byte = [1000, 1000, 1000, 1000, 1000]
    t_dist = [bv * 200, bv * 600, bv * 950, bv * 2000, bv * 5000]  # crosses 1000*bv
    rd_table = {"t": _convex_curve(t_byte, t_dist)}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    assert alloc.kkt_max_accepted_ratio is not None
    assert alloc.kkt_min_rejected_ratio is not None
    assert alloc.kkt_max_accepted_ratio <= bv + 1e-18
    assert alloc.kkt_min_rejected_ratio >= bv - 1e-18
    holds, msg = verify_kkt_marginal_equalization(alloc)
    assert holds, msg


def test_waterfill_beats_crude_uniform_band_when_uniform_misspends():
    """The math-optimal waterfill achieves a STRICTLY better net than a crude uniform-band
    allocation that coarsens a pose-sensitive (expensive) tensor it shouldn't.

    Crude-uniform = coarsen BOTH tensors to 64 (the uniform min_level_ratio≈0.5 behavior).
    Waterfill = coarsen only the cheap tensor, protect the expensive one.
    """
    bv = byte_saving_score_value(1.0)
    cheap = _convex_curve([3000, 3000, 2000, 1500, 1000],
                          [bv * 100, bv * 200, bv * 400, bv * 800, bv * 1600])
    expensive = _convex_curve([500, 400, 300, 200, 100],
                             [bv * 20000, bv * 40000, bv * 60000, bv * 80000, bv * 100000])
    rd_table = {"cheap": cheap, "expensive": expensive}

    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    waterfill_net = alloc.net_score_delta

    # crude uniform: force both to 64 (2 hops each), compute the net from the table
    def _net_at(levels: dict[str, int]) -> float:
        cb = cd = 0.0
        for t, lv in levels.items():
            b, d = rd_table[t][lv]
            cb += b
            cd += d
        return cd - byte_saving_score_value(cb)

    crude_net = _net_at({"cheap": 64, "expensive": 64})
    assert waterfill_net < crude_net, (
        f"waterfill net {waterfill_net:.3e} not better than crude-uniform-64 "
        f"{crude_net:.3e} — the math-optimal allocation did not improve on the crude band"
    )
    # the waterfill protects the expensive tensor (keeps it at/near 127)
    assert alloc.levels["expensive"] > 64


def test_byte_target_solve_hits_target_at_minimum_distortion():
    """A byte-target solve reaches (at least) the target byte saving and does so
    cheapest-first (the accepted marginals are monotone => minimum distortion for that
    byte budget)."""
    bv = byte_saving_score_value(1.0)
    a = _convex_curve([1000, 1000, 1000, 1000, 1000],
                     [bv * 100, bv * 300, bv * 700, bv * 1500, bv * 3000])
    b = _convex_curve([1000, 1000, 1000, 1000, 1000],
                     [bv * 50, bv * 150, bv * 500, bv * 1200, bv * 4000])
    rd_table = {"a": a, "b": b}
    target = 4000.0
    alloc = solve_waterfill_allocation(rd_table, byte_target=target)
    assert alloc.total_byte_saving >= target - 1e-9
    accepted = [s for s in alloc.trace if s.accepted]
    ratios = [s.marginal_ratio for s in accepted]
    assert all(ratios[i] <= ratios[i + 1] + 1e-12 for i in range(len(ratios) - 1)), (
        f"byte-target solve did not take cheapest-first: {ratios}"
    )


def test_non_saving_steps_are_skipped():
    """A coarser grid level that does NOT save bytes (d_byte<=0, e.g. brotli non-monotone
    on a tiny tensor) is SKIPPED — the greedy never accepts an infinite-ratio step."""
    bv = byte_saving_score_value(1.0)
    # tensor whose 96 level saves 0 bytes (non-monotone) but 64 saves real bytes
    curve = {127: (0.0, 0.0), 96: (0.0, bv * 10), 64: (500.0, bv * 50),
             48: (700.0, bv * 100), 32: (800.0, bv * 200), 16: (850.0, bv * 400)}
    rd_table = {"t": curve}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    # it must NOT have taken the 127->96 zero-byte step (that would be infinite ratio)
    for s in alloc.trace:
        if s.accepted:
            assert s.byte_saving > 0, f"accepted a non-saving step: {s}"


def test_net_composition_matches_contest_arithmetic():
    """``net_score_delta_from_components`` equals the explicit contest-score delta."""
    d_byte = 1500.0
    d_seg_cost = 0.0008  # advisory d_seg uptick
    # pose sqrt-term delta
    d_pose_base, d_pose_var = 0.0003, 0.00035
    d_pose_sqrt = sqrt_pose_term(d_pose_var) - sqrt_pose_term(d_pose_base)
    net = net_score_delta_from_components(d_byte, d_seg_cost, d_pose_sqrt)
    expected = (100.0 * d_seg_cost + d_pose_sqrt) - 25.0 * d_byte / _N
    assert math.isclose(net, expected, rel_tol=0, abs_tol=1e-15)


def test_byte_value_per_byte_is_25_over_N():
    """The byte value (score units per byte saved) is exactly 25/N."""
    assert math.isclose(byte_saving_score_value(1.0), 25.0 / _N, rel_tol=0, abs_tol=1e-18)
    assert math.isclose(byte_saving_score_value(1000.0), 25.0 * 1000.0 / _N, abs_tol=1e-15)


def test_sqrt_pose_term_matches_probe_formula():
    """``sqrt_pose_term`` matches the contest pose contribution sqrt(10*d_pose + 1e-12)."""
    for dp in (0.0, 0.0001, 0.001, 0.01):
        assert math.isclose(sqrt_pose_term(dp), math.sqrt(10.0 * dp + 1e-12), abs_tol=1e-18)


def test_greedy_is_globally_optimal_on_convex_table():
    """On a table with strictly-convex (increasing-marginal) RD curves, the greedy
    cheapest-first IS the global optimum: exhaustively enumerate all allocations over a
    2-tensor / 3-level grid and confirm the greedy net is the minimum."""
    bv = byte_saving_score_value(1.0)
    grid = (127, 64, 16)
    # convex curves: marginal dist cost strictly increases per hop
    a = {127: (0.0, 0.0), 64: (1000.0, bv * 200), 16: (1500.0, bv * 900)}
    b = {127: (0.0, 0.0), 64: (800.0, bv * 100), 16: (1100.0, bv * 1200)}
    rd_table = {"a": a, "b": b}
    alloc = solve_waterfill_allocation(rd_table, level_grid=grid, net_stop=True)

    # brute force all 3x3 allocations
    best = math.inf
    for la in grid:
        for lb in grid:
            cb = a[la][0] + b[lb][0]
            cd = a[la][1] + b[lb][1]
            net = cd - byte_saving_score_value(cb)
            best = min(best, net)
    assert math.isclose(alloc.net_score_delta, best, rel_tol=0, abs_tol=1e-12), (
        f"greedy net {alloc.net_score_delta:.6e} != brute-force optimum {best:.6e}"
    )


def test_levels_feed_codec_builder_default_preserving():
    """The allocator's level dict is consumable by the codec builder and an all-127
    allocation is byte-identical to vendored (the integration contract)."""
    from tac.losses.variable_level_codec import build_decoder_blob_variable_or_vendored
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    import torch

    g = torch.Generator().manual_seed(7)
    sd = {
        "blocks.0.weight": torch.randn(8, 4, 3, 3, generator=g),
        "rgb.weight": torch.randn(3, 4, 1, 1, generator=g),
    }
    # all-127 allocation from a trivial RD table
    rd_table = {k: {127: (0.0, 0.0)} for k in sd}
    alloc = solve_waterfill_allocation(rd_table)
    vend = codec.encode_decoder(codec.quantize_state_dict(sd))
    blob, is_var = build_decoder_blob_variable_or_vendored(sd, alloc.levels)
    assert is_var is False
    assert blob == vend, "all-127 waterfill allocation not byte-identical to vendored"


def test_protects_expensive_when_no_step_is_net_positive():
    """If EVERY tensor's first coarsening step is net-negative (too expensive), the
    waterfill coarsens NOTHING (all-127), giving net 0 — it never makes the score worse."""
    bv = byte_saving_score_value(1.0)
    # every step has a marginal ratio way above the byte value
    expensive = _convex_curve([100, 100, 100, 100, 100],
                             [bv * 100000, bv * 200000, bv * 300000, bv * 400000, bv * 500000])
    rd_table = {"t1": expensive, "t2": expensive}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    assert all(v == 127 for v in alloc.levels.values()), (
        "waterfill coarsened despite every step being net-negative"
    )
    assert alloc.net_score_delta == 0.0
    assert alloc.n_coarsened == 0


def test_rd_table_missing_127_baseline_skips_tensor():
    """A tensor whose curve lacks the 127 baseline cannot form a marginal -> it is left
    uncoarsened (defensive: no crash, no phantom step)."""
    bv = byte_saving_score_value(1.0)
    rd_table = {
        "ok": _convex_curve([1000, 1000, 1000, 1000, 1000],
                           [bv * 100, bv * 200, bv * 400, bv * 800, bv * 1600]),
        "bad": {64: (500.0, bv * 100)},  # no 127 entry
    }
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    assert alloc.levels["bad"] == 127  # untouched (current starts at max grid)
    # the ok tensor still gets coarsened
    assert alloc.levels["ok"] < 127


def test_trace_net_after_is_monotone_decreasing_for_accepted():
    """Each ACCEPTED step's running net is strictly better (more negative) than before —
    the greedy only accepts net-improving steps under net_stop."""
    bv = byte_saving_score_value(1.0)
    curve = _convex_curve([2000, 2000, 1500, 1000, 500],
                         [bv * 100, bv * 300, bv * 700, bv * 1500, bv * 5000])
    rd_table = {"t": curve}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    accepted = [s for s in alloc.trace if s.accepted]
    nets = [s.net_after for s in accepted]
    for i in range(len(nets) - 1):
        assert nets[i + 1] <= nets[i] + 1e-18, f"net not improving at step {i}: {nets}"
    # final net equals the allocation net
    if accepted:
        assert math.isclose(nets[-1], alloc.net_score_delta, abs_tol=1e-15)


def test_verifier_rejects_fake_non_cheapest_first_allocation():
    """Lens-1 NO-FAKE: a FAKE allocation whose trace is NOT cheapest-first (accepted
    marginals out of order) is REJECTED by the verifier — proving the verifier checks the
    KKT/Lagrange ordering, not just that *some* trace exists.

    Construct a WaterfillAllocation by hand with accepted marginals that DECREASE (the
    opposite of the greedy cheapest-first order) and confirm verify returns False."""
    from tac.losses.variable_level_waterfill_allocator import (
        WaterfillAllocation,
        WaterfillStep,
        byte_saving_score_value,
    )

    bv = byte_saving_score_value(1.0)
    # a fake trace: two accepted steps whose marginal ratios DECREASE (5e-7 then 1e-7) —
    # a real greedy would have taken the 1e-7 step FIRST. The verifier must catch this.
    fake = WaterfillAllocation(
        levels={"a": 64, "b": 64},
        total_byte_saving=2000.0,
        total_dist_cost=bv * 1200.0,
        net_score_delta=bv * 1200.0 - byte_saving_score_value(2000.0),
        trace=(
            WaterfillStep("a", 127, 64, 1000.0, bv * 500.0, 5e-7, -1.0, True),
            WaterfillStep("b", 127, 64, 1000.0, bv * 100.0, 1e-7, -2.0, True),
        ),
        byte_value_per_byte=bv,
        kkt_max_accepted_ratio=5e-7,
        kkt_min_rejected_ratio=None,
        n_coarsened=2,
    )
    holds, msg = verify_kkt_marginal_equalization(fake)
    assert not holds, f"verifier wrongly ACCEPTED a non-cheapest-first fake trace: {msg}"
    assert "monotone" in msg.lower()


def test_verifier_rejects_accepted_step_above_byte_value():
    """Lens-1 NO-FAKE: an allocation that ACCEPTED a step whose marginal exceeds the byte
    value (net-WORSENING) is rejected — the KKT upper bound is enforced, not decorative."""
    from tac.losses.variable_level_waterfill_allocator import (
        WaterfillAllocation,
        WaterfillStep,
        byte_saving_score_value,
    )

    bv = byte_saving_score_value(1.0)
    # accepted a step at ratio 2*bv (above the byte value) with a rejected frontier above —
    # the max-accepted-ratio > byte value must trip the upper-bound check.
    fake = WaterfillAllocation(
        levels={"a": 64},
        total_byte_saving=1000.0,
        total_dist_cost=bv * 2000.0,
        net_score_delta=bv * 2000.0 - byte_saving_score_value(1000.0),
        trace=(
            WaterfillStep("a", 127, 64, 1000.0, bv * 2000.0, 2.0 * bv, 1.0, True),
            WaterfillStep("a", 64, 16, 500.0, bv * 3000.0, 6.0 * bv, 2.0, False),
        ),
        byte_value_per_byte=bv,
        kkt_max_accepted_ratio=2.0 * bv,
        kkt_min_rejected_ratio=6.0 * bv,
        n_coarsened=1,
    )
    holds, msg = verify_kkt_marginal_equalization(fake)
    assert not holds, f"verifier wrongly ACCEPTED an above-byte-value step: {msg}"
    assert "exceeds byte value" in msg


def test_lower_convex_hull_drops_dominated_levels():
    """``lower_convex_hull_levels`` drops a DOMINATED (above-the-hull) intermediate level and
    keeps the convex vertices — the R8 non-convexity fix."""
    bv = byte_saving_score_value(1.0)
    # 127:(0,0), 96:(1000, 5*bv), 64:(1100, 50*bv) DOMINATED, 16:(2000, 100*bv).
    # 96 lies ABOVE the line from 127 to 16? slope 127->16 = 100bv/2000 = 0.05bv/B;
    # at byte 1000 the hull line is 50bv; 96's dist 5bv is BELOW -> 96 is a hull vertex.
    # Make 64 dominated: put it ABOVE the 96->16 segment.
    curve = {127: (0.0, 0.0), 96: (1000.0, bv * 5), 64: (1100.0, bv * 90),
             16: (2000.0, bv * 50)}
    hull = lower_convex_hull_levels(curve)
    # 64 (1100, 90bv) is above the 96(1000,5bv)->16(2000,50bv) segment -> dropped.
    assert 64 not in hull, f"dominated level 64 not dropped from hull: {hull}"
    assert 127 in hull and 96 in hull and 16 in hull


def test_hull_restores_monotone_on_nonconvex_curve():
    """R8 REGRESSION: a NON-CONVEX RD curve (the run2 failure mode) is convexified by the
    hull so the greedy's accepted marginals ARE monotone and the KKT verifier PASSES — the
    bug that made run2 report kkt_holds=false is fixed."""
    bv = byte_saving_score_value(1.0)
    # two tensors with non-convex raw curves (a coarser hop cheaper-per-byte than a finer):
    # tensor a: 96 expensive-per-byte, 64 cheap-per-byte (non-convex) -> hull skips 96.
    a = {127: (0.0, 0.0), 96: (200.0, bv * 80), 64: (1200.0, bv * 120),
         48: (1600.0, bv * 300), 32: (1900.0, bv * 900), 16: (2100.0, bv * 3000)}
    b = {127: (0.0, 0.0), 96: (300.0, bv * 30), 64: (1000.0, bv * 100),
         48: (1400.0, bv * 250), 32: (1700.0, bv * 800), 16: (1900.0, bv * 2500)}
    rd_table = {"a": a, "b": b}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    holds, msg = verify_kkt_marginal_equalization(alloc)
    assert holds, f"hull did NOT restore the KKT certificate on a non-convex curve: {msg}"
    accepted = [s for s in alloc.trace if s.accepted]
    ratios = [s.marginal_ratio for s in accepted]
    assert all(ratios[i] <= ratios[i + 1] + 1e-12 for i in range(len(ratios) - 1)), (
        f"hull did not restore monotone marginals: {ratios}"
    )


def test_hull_allocation_dominates_raw_on_nonconvex_net():
    """The hull-based allocation achieves a net AT LEAST AS GOOD as any raw single-hop
    allocation on a non-convex curve (it can SKIP a dominated level to reach a cheaper one)."""
    bv = byte_saving_score_value(1.0)
    # raw greedy would stop at 96 (expensive first hop); hull skips to 64 (cheaper overall).
    curve = {127: (0.0, 0.0), 96: (100.0, bv * 90), 64: (1500.0, bv * 200),
             48: (1700.0, bv * 600), 32: (1800.0, bv * 1500), 16: (1850.0, bv * 4000)}
    rd_table = {"t": curve}
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    # the hull lets the greedy reach 64 (net-negative) even though the 127->96 hop alone
    # is net-positive (90bv/100B > bv). Confirm it coarsened past 96.
    assert alloc.levels["t"] <= 64, (
        f"hull allocation stopped at {alloc.levels['t']} — did not skip the dominated 96 "
        "level to reach the net-improving 64 level"
    )
    assert alloc.net_score_delta < 0


def test_default_level_grid_is_descending_unique():
    """The default level grid is the canonical {127,96,64,48,32,16} (descending, 127 first)."""
    assert DEFAULT_LEVEL_GRID[0] == 127
    assert len(set(DEFAULT_LEVEL_GRID)) == len(DEFAULT_LEVEL_GRID)
    assert list(DEFAULT_LEVEL_GRID) == sorted(DEFAULT_LEVEL_GRID, reverse=True)
    assert min(DEFAULT_LEVEL_GRID) >= 16  # respects the codec min_abs_levels floor
