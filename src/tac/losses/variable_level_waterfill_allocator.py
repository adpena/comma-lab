# SPDX-License-Identifier: MIT
"""Math-optimal per-tensor reverse-waterfill allocator (KKT) for the variable-level codec.

THE PROBLEM this solves. The variable-level codec
(``tac.losses.variable_level_codec``) stores each tensor at ITS OWN ``n_levels`` and
the per-tensor grid IS the deployed grid — so a coarse-grid tensor SURVIVES inflate
(unlike the vendored-127-requant Lever-4 indirect path). But which tensor gets which
level count is an ALLOCATION decision. The first allocation tried
(``levels_from_sensitivity_for_codec`` with a uniform ``min_level_ratio`` knob) is a
CRUDE rank-normalized linear band: it coarsens ~27/28 tensors to roughly the same
fraction regardless of each tensor's ACTUAL byte-vs-distortion trade. The MEASURED net
on the real basin EMA + real frozen SegNet/PoseNet was NEAR-BREAKEVEN-BUT-WORSE
(+0.001053 / +0.006030) — the uniform allocation mis-shapes the spend, coarsening
pose-sensitive tensors it should leave at 127.

THE MATH (reverse water-filling / KKT). We want to choose, per tensor ``t``, a level
count ``n_t in {127, 96, 64, 48, 32, 16}`` (coarsening spends a "byte budget" of
``byte_saving_t(n_t)`` and incurs a distortion cost ``dist_t(n_t)``) to MINIMIZE total
advisory distortion subject to a target total byte saving — equivalently to MAXIMIZE
the NET contest-score improvement
``net = byte_saving_ΔS  −  distortion_ΔS``
where ``byte_saving_ΔS = 25 · (−Δbytes) / N`` (a coarse grid LOWERS S via the rate
term) and ``distortion_ΔS = 100·Δd_seg + Δsqrt(10·d_pose)`` (coarsening RAISES S via
distortion). At the optimum the Lagrangian stationarity (KKT) condition is that the
MARGINAL distortion-per-byte ``dΔS_dist_t / dByte_t`` is EQUALIZED across all
partially-coarsened tensors and ≤ the marginal byte-VALUE for every coarsened tensor —
this is the classic reverse-waterfilling result (Cover & Thomas Ch.10): spend the
coarsening budget ONLY where distortion is cheapest, stop where the marginal distortion
cost of the next byte saved exceeds the byte's score value.

THE ALGORITHM (greedy realizes the KKT optimum on a discrete grid). Because the
per-tensor RD curves are convex-ish and the steps are discrete, a greedy that
repeatedly takes the AVAILABLE STEP with the smallest marginal ``dΔS_dist / dByte``
(over all (tensor, next-coarser-level) candidates), accepting it only while the running
NET improves, lands at an allocation where the accepted steps' marginal ratios are all
≤ the byte value and the first REJECTED step's marginal exceeds it — i.e. the
marginal-equalization (KKT) frontier. We expose BOTH:
  * ``solve_waterfill_allocation`` — the greedy solve to a chosen byte target (or the
    net-maximizing stopping point), returning the allocation + the per-step trace.
  * ``verify_kkt_marginal_equalization`` — a NO-FAKE check that the accepted-step
    marginals are bounded by the byte value and the rejected frontier exceeds it (the
    KKT condition HOLDS at the stopping point, not a constant).

CONVEXITY caveat (honest scope). The greedy single-hop walk is the GLOBAL optimum when
each per-tensor RD curve is CONVEX (marginal distortion-per-byte non-decreasing as the
grid coarsens) — verified by ``test_greedy_is_globally_optimal_on_convex_table``'s
brute-force comparison. Real brotli/scorer RD curves are convex-ISH but not guaranteed
(a tiny tensor's brotli can be non-monotone). The greedy is therefore a strong heuristic
on real curves; ``verify_kkt_marginal_equalization`` certifies the REALIZED stopping
point satisfies the marginal bound regardless, and the probe MEASURES the deployed net at
the realized allocation (the deployed bytes + real scorer are the truth, not the RD-table
prediction) — so a non-convex curve cannot produce a phantom net claim, only a
(possibly) sub-optimal allocation that is still measured honestly.

NO-FAKE / EMPIRICAL discipline (Catalog #304 — forbidden closed-form-CDF-allocator-
without-empirical-bit-spend-proof). This allocator consumes MEASURED per-tensor RD
curves ``{tensor: {n_levels: (byte_saving, dist_cost)}}`` (the probe measures them on
the REAL frozen scorer + REAL weights) — it does NOT assume a closed-form rate or
distortion model. The allocator is pure arithmetic over the measured table; the
``solve`` result is only as honest as the RD table fed in, and the probe that feeds it
is the empirical bit-spend + distortion proof.

CARRIER-AGNOSTIC ``[CORE]`` (ledger ITEM B/F): this module operates on tensor NAMES +
a measured RD table + the contest-score constants — it imports NO base_ch20 constant
and NO codec. It is reusable on Track A or Track B unchanged. The base_ch20 grammar
adapter lives in ``variable_level_codec.build_decoder_blob_variable_or_vendored``.

Score-claim discipline: ``SCORE_CLAIM = False`` — every number this produces is an
ALLOCATION over an advisory RD table; the deployed net requires the byte-closed archive
measured on the real scorer (the probe), and a promotion requires the dual CPU/CUDA
600-pair exact eval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Contest-score constants (the ONLY non-tensor inputs; identical on Track A/B).
_ARCHIVE_NORMALIZER = 37_545_489  # the contest's archive-size normalizer N
# The canonical level grid the RD curves are measured at (coarsest-last is NOT required;
# the solver sorts by level so any order is accepted). 127 = the vendored base (no coarsen).
DEFAULT_LEVEL_GRID: tuple[int, ...] = (127, 96, 64, 48, 32, 16)


def byte_saving_score_value(byte_saving: float) -> float:
    """The contest-score VALUE of saving ``byte_saving`` bytes (a POSITIVE saving lowers
    S by this much via the rate term ``25·bytes/N``). This is the marginal "price" the
    waterfill pays distortion against — a byte saved is worth ``25/N`` score units."""
    return 25.0 * float(byte_saving) / _ARCHIVE_NORMALIZER


@dataclass(frozen=True)
class WaterfillStep:
    """One accepted (or rejected-frontier) coarsening step in the greedy trace.

    ``tensor`` moved from ``from_level`` to ``to_level``; this step saved
    ``byte_saving`` bytes (marginal, vs ``from_level``) at a distortion cost
    ``dist_cost`` (marginal advisory ΔS_dist vs ``from_level``). ``marginal_ratio`` is
    ``dist_cost / byte_saving`` = the distortion-per-byte the KKT condition equalizes.
    ``net_after`` is the running NET score improvement (negative = better) AFTER this
    step. ``accepted`` records whether the greedy took it.
    """

    tensor: str
    from_level: int
    to_level: int
    byte_saving: float
    dist_cost: float
    marginal_ratio: float
    net_after: float
    accepted: bool


@dataclass(frozen=True)
class WaterfillAllocation:
    """The solved allocation + full trace + KKT-verification fields.

    ``levels`` maps tensor → chosen ``n_levels`` (tensors at 127 are uncoarsened).
    ``total_byte_saving`` / ``total_dist_cost`` are the cumulative measured deltas at
    the chosen allocation. ``net_score_delta`` = ``distortion_cost − byte_value`` in
    score units (negative = the allocation IMPROVES the advisory score). ``trace`` is
    the ordered greedy steps (accepted, then the first rejected frontier step if any).
    ``kkt_max_accepted_ratio`` / ``kkt_min_rejected_ratio`` are the marginal-equalization
    bounds: at the optimum every accepted ratio ≤ the byte value ≤ the rejected frontier.
    """

    levels: dict[str, int]
    total_byte_saving: float
    total_dist_cost: float
    net_score_delta: float
    trace: tuple[WaterfillStep, ...]
    byte_value_per_byte: float
    kkt_max_accepted_ratio: float | None
    kkt_min_rejected_ratio: float | None
    n_coarsened: int = field(default=0)


def lower_convex_hull_levels(curve: dict[int, tuple[float, float]]) -> list[int]:
    """Per-tensor LOWER CONVEX HULL of the RD points (R8 review fix for non-convex curves).

    Real brotli/scorer RD curves are convex-ISH but NOT guaranteed: a coarser grid can
    have a LOWER marginal distortion-per-byte than a finer one (brotli alignment + the
    discrete grid). A greedy single-hop walk over the RAW points can then (a) stop at a
    DOMINATED level (never optimal) and (b) produce a non-monotone marginal sequence that
    spuriously trips the KKT check. The fix is to restrict each tensor to its LOWER CONVEX
    HULL in ``(byte_saving, dist_cost)`` space: the hull vertices are exactly the levels
    that can be optimal at SOME Lagrange multiplier (byte value). Coarsening ALONG the hull
    gives strictly-increasing marginal ``dDist/dByte`` PER TENSOR (the correct convex
    structure), so the Lagrangian marginal-allocation greedy over hull segments is globally
    optimal for the separable problem (Everett 1963; Cover & Thomas Ch.10 reverse
    waterfilling on the convexified RD curve).

    Returns the SORTED-BY-BYTE-SAVING list of hull level counts (always includes the 127
    baseline at (0,0); a dominated intermediate level is dropped). Points are ordered by
    increasing byte_saving; the hull keeps only vertices where the incremental slope
    strictly increases (a lower convex boundary).
    """
    # points: (byte_saving, dist_cost, level), sorted by byte_saving ascending.
    pts = sorted(((b, d, lv) for lv, (b, d) in curve.items()), key=lambda x: (x[0], x[1]))
    if len(pts) <= 2:
        return [lv for _b, _d, lv in pts]
    hull: list[tuple[float, float, int]] = []
    for b, d, lv in pts:
        # maintain a LOWER convex hull: pop while the last turn is not a left (convex) turn.
        while len(hull) >= 2:
            (b0, d0, _l0), (b1, d1, _l1) = hull[-2], hull[-1]
            # cross product of (p1-p0) x (p_new-p0); <= 0 means p1 is above/on the lower hull
            cross = (b1 - b0) * (d - d0) - (d1 - d0) * (b - b0)
            if cross <= 1e-18:
                hull.pop()
            else:
                break
        hull.append((b, d, lv))
    return [lv for _b, _d, lv in hull]


def _candidate_steps(
    rd_table: dict[str, dict[int, tuple[float, float]]],
    current: dict[str, int],
    level_grid: tuple[int, ...],
    hulls: dict[str, list[int]] | None = None,
) -> list[tuple[str, int, int, float, float]]:
    """Enumerate the next-coarser HULL MARGINAL step for every tensor not yet at the floor.

    For tensor ``t`` currently at level ``L``, the candidate is the next coarser level on
    ``t``'s LOWER CONVEX HULL (``hulls[t]``) — NOT merely the next grid level. The marginal
    byte saving / dist cost is the DIFFERENCE between the hull RD values at the next hull
    level and ``L`` (so a hop can SKIP a dominated grid level — the convex-hull fix). Hull
    marginals are strictly increasing per tensor, so the greedy walks the convex hull
    correctly. Returns ``[(tensor, L, L', d_byte, d_dist), ...]`` with ``d_byte`` POSITIVE
    for a saving and ``d_dist`` the marginal distortion ΔS_dist.
    """
    steps: list[tuple[str, int, int, float, float]] = []
    for tensor, cur in current.items():
        curve = rd_table.get(tensor)
        if curve is None:
            continue
        hull = hulls[tensor] if hulls is not None else lower_convex_hull_levels(curve)
        # hull is sorted by byte_saving ascending == level descending (127 first). Find the
        # next hull level strictly coarser (smaller) than the current level.
        coarser = [lv for lv in hull if lv < cur]
        if not coarser or cur not in curve:
            continue
        nxt = max(coarser)  # the immediately-next hull vertex below the current level
        byte_cur, dist_cur = curve[cur]
        byte_nxt, dist_nxt = curve[nxt]
        d_byte = byte_nxt - byte_cur  # cumulative saving is larger at coarser -> positive
        d_dist = dist_nxt - dist_cur  # cumulative dist cost is larger at coarser
        steps.append((tensor, cur, nxt, d_byte, d_dist))
    return steps


def solve_waterfill_allocation(
    rd_table: dict[str, dict[int, tuple[float, float]]],
    *,
    level_grid: tuple[int, ...] = DEFAULT_LEVEL_GRID,
    byte_target: float | None = None,
    net_stop: bool = True,
) -> WaterfillAllocation:
    """Greedy reverse-waterfill: take the smallest marginal ``dΔS_dist/dByte`` step next.

    ``rd_table`` maps tensor → ``{n_levels: (cumulative_byte_saving, cumulative_dist_cost)}``
    where ``cumulative_byte_saving`` is bytes saved vs the 127 baseline (POSITIVE) and
    ``cumulative_dist_cost`` is the advisory ΔS_dist vs 127 (typically POSITIVE — coarsening
    hurts). The 127 entry MUST be present with ``(0.0, 0.0)``.

    The greedy maintains a current allocation (all 127), repeatedly evaluating every
    tensor's next-coarser MARGINAL step and taking the one with the smallest
    ``d_dist / d_byte`` ratio (the cheapest distortion-per-byte). A step is ACCEPTED while:
      * ``byte_target`` is set: until the cumulative byte saving reaches the target
        (greedy still takes cheapest-first, so it hits the target at minimum distortion); OR
      * ``net_stop`` (default): while the running NET ``= dist_cost − byte_value`` keeps
        IMPROVING (going more negative) — i.e. while the step's marginal ratio is below the
        byte value ``25/N``. This is the net-maximizing stopping point (the KKT frontier).

    Steps with ``d_byte <= 0`` (a coarser grid that does NOT save bytes — possible if
    brotli is non-monotone on a tiny tensor) are SKIPPED (infinite marginal ratio). The
    result records the KKT-equalization bounds + the full trace for the verifier.

    CONVEX-HULL FIX (R8 review). Each tensor is restricted to its LOWER CONVEX HULL
    (``lower_convex_hull_levels``) so the greedy walks the convexified RD curve: hull
    marginals are strictly increasing PER TENSOR, dominated grid levels are skipped, and
    the Lagrangian marginal-allocation greedy over hull segments is globally optimal for the
    separable problem (Everett 1963). Without the hull, real non-convex brotli curves made
    the greedy stop at dominated levels AND spuriously trip the KKT monotone check.
    """
    current: dict[str, int] = {t: max(level_grid) for t in rd_table}
    base_level = max(level_grid)
    # precompute each tensor's lower convex hull ONCE (the convex-hull fix).
    hulls: dict[str, list[int]] = {
        t: lower_convex_hull_levels(curve) for t, curve in rd_table.items()
    }
    # running cumulative (vs 127 baseline)
    cum_byte = 0.0
    cum_dist = 0.0
    trace: list[WaterfillStep] = []
    accepted_ratios: list[float] = []
    rejected_ratio: float | None = None
    byte_value = byte_saving_score_value(1.0)  # score value per single byte = 25/N

    while True:
        cands = _candidate_steps(rd_table, current, level_grid, hulls)
        # marginal ratio dist/byte; skip non-saving steps (d_byte<=0 -> never beneficial)
        scored: list[tuple[float, str, int, int, float, float]] = []
        for tensor, cur, nxt, d_byte, d_dist in cands:
            if d_byte <= 0:
                continue
            ratio = d_dist / d_byte
            scored.append((ratio, tensor, cur, nxt, d_byte, d_dist))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        ratio, tensor, cur, nxt, d_byte, d_dist = scored[0]

        # running net before vs if-we-take (the byte value is folded into both via
        # ``byte_saving_score_value`` — the marginal byte VALUE is implicit in net_if_take).
        net_before = cum_dist - byte_saving_score_value(cum_byte)
        net_if_take = (cum_dist + d_dist) - byte_saving_score_value(cum_byte + d_byte)

        take = False
        if byte_target is not None:
            # take cheapest-first until we reach the target byte saving
            take = cum_byte < byte_target
        elif net_stop:
            # take while the step IMPROVES the net (marginal dist cost < byte value)
            take = net_if_take < net_before - 1e-15
        else:
            take = True

        if take:
            current[tensor] = nxt
            cum_byte += d_byte
            cum_dist += d_dist
            accepted_ratios.append(ratio)
            trace.append(
                WaterfillStep(
                    tensor=tensor, from_level=cur, to_level=nxt,
                    byte_saving=d_byte, dist_cost=d_dist, marginal_ratio=ratio,
                    net_after=net_if_take, accepted=True,
                )
            )
        else:
            # record the FIRST rejected frontier step (the KKT boundary) and stop
            rejected_ratio = ratio
            trace.append(
                WaterfillStep(
                    tensor=tensor, from_level=cur, to_level=nxt,
                    byte_saving=d_byte, dist_cost=d_dist, marginal_ratio=ratio,
                    net_after=net_if_take, accepted=False,
                )
            )
            break

    net = cum_dist - byte_saving_score_value(cum_byte)
    n_coarse = sum(1 for v in current.values() if v < base_level)
    return WaterfillAllocation(
        levels=dict(current),
        total_byte_saving=cum_byte,
        total_dist_cost=cum_dist,
        net_score_delta=net,
        trace=tuple(trace),
        byte_value_per_byte=byte_value,
        kkt_max_accepted_ratio=(max(accepted_ratios) if accepted_ratios else None),
        kkt_min_rejected_ratio=rejected_ratio,
        n_coarsened=n_coarse,
    )


def verify_kkt_marginal_equalization(alloc: WaterfillAllocation) -> tuple[bool, str]:
    """NO-FAKE check that the greedy stopping point satisfies the reverse-waterfill KKT
    condition: every ACCEPTED marginal ``dΔS_dist/dByte`` ≤ the byte value ``25/N`` ≤ the
    first REJECTED marginal (when net_stop). The accepted marginals are also MONOTONE
    NON-DECREASING along the trace (the greedy takes cheapest-first), which is the
    marginal-equalization signature — distinct values walking UP to the frontier, NOT a
    constant. Returns ``(holds, explanation)``.

    This is the test the prompt's Lens-1 demands: a verifier that the allocation ACTUALLY
    equalizes the marginal at the boundary (the Lagrange-multiplier interpretation), not a
    constant ratio. A FAKE allocator that returned a fixed level for every tensor would
    have a DEGENERATE trace (one ratio, no frontier) and fail the "distinct marginals
    walking up to the boundary" structure.

    R8 REVIEW NOTE (the monotone certificate is now VALID). The global greedy sequence is
    monotone non-decreasing BECAUSE each tensor is restricted to its lower convex hull
    (``solve_waterfill_allocation`` precomputes ``lower_convex_hull_levels``): per-tensor
    hull marginals strictly increase, and the global-cheapest-next rule then yields a
    monotone GLOBAL sequence (after taking tensor t's hull hop, t's next hull marginal ≥ the
    one just taken AND every other tensor's available marginal was ≥ the global min just
    taken). A monotone VIOLATION therefore now signals a HULL-CONSTRUCTION bug (raw
    non-convex points leaked into the greedy), NOT merely a non-convex input — which is why
    it is a hard FAIL.
    """
    accepted = [s for s in alloc.trace if s.accepted]
    bv = alloc.byte_value_per_byte
    if not accepted:
        return True, "no coarsening accepted (allocation is all-127); KKT trivially holds"
    # 1. accepted marginals are sorted non-decreasing. With the per-tensor lower-convex-hull
    #    restriction this is GUARANTEED for the global greedy sequence (R8 fix); a violation
    #    means raw non-convex points leaked into the greedy (a hull-construction bug).
    ratios = [s.marginal_ratio for s in accepted]
    monotone = all(ratios[i] <= ratios[i + 1] + 1e-12 for i in range(len(ratios) - 1))
    if not monotone:
        return False, (
            "accepted marginal ratios are NOT monotone non-decreasing — the convex-hull "
            "restriction failed to convexify the per-tensor curves (a hull-construction "
            f"bug, since hull marginals must increase monotonically): {ratios[:6]}..."
        )
    # 2. every accepted marginal <= byte value (net_stop boundary)
    if alloc.kkt_min_rejected_ratio is not None:
        max_acc = alloc.kkt_max_accepted_ratio or 0.0
        if not (max_acc <= bv + 1e-18):
            return False, (
                f"max accepted marginal {max_acc:.3e} exceeds byte value {bv:.3e} — an "
                "accepted step was NOT net-improving (KKT upper bound violated)"
            )
        if not (alloc.kkt_min_rejected_ratio >= bv - 1e-18):
            return False, (
                f"rejected frontier marginal {alloc.kkt_min_rejected_ratio:.3e} is BELOW "
                f"the byte value {bv:.3e} — a net-improving step was wrongly rejected "
                "(KKT lower bound violated)"
            )
        boundary = (
            f"KKT marginal-equalization HOLDS: accepted marginals in "
            f"[{ratios[0]:.3e}, {max_acc:.3e}] all <= byte value {bv:.3e} <= rejected "
            f"frontier {alloc.kkt_min_rejected_ratio:.3e}; {len(accepted)} distinct steps "
            "walk monotonically UP to the boundary (not a constant)."
        )
        return True, boundary
    # net_stop exhausted the grid (no rejected frontier) OR byte_target solve — only the
    # monotone-cheapest-first structure is checkable.
    return True, (
        f"KKT cheapest-first ordering HOLDS: {len(accepted)} accepted marginals monotone "
        f"non-decreasing in [{ratios[0]:.3e}, {ratios[-1]:.3e}] (no rejected frontier: the "
        "grid was exhausted or a byte_target was set)."
    )


def net_score_delta_from_components(
    d_byte_saving: float, d_seg_cost: float, d_pose_cost_sqrt: float
) -> float:
    """Compose the NET advisory contest-score delta from measured components.

    ``net = distortion_cost − byte_value`` in score units, where ``distortion_cost =
    100·Δd_seg + Δ(sqrt(10·d_pose))`` and ``byte_value = 25·byte_saving/N``. NEGATIVE net
    = the allocation IMPROVES the advisory score. Provided so the probe and the allocator
    share ONE composition (no drift). ``d_pose_cost_sqrt`` is the already-differenced
    sqrt-term delta (the probe computes ``sqrt(10·d_pose_var) − sqrt(10·d_pose_base)``)."""
    distortion_cost = 100.0 * d_seg_cost + d_pose_cost_sqrt
    return distortion_cost - byte_saving_score_value(d_byte_saving)


def sqrt_pose_term(d_pose: float) -> float:
    """The contest pose contribution ``sqrt(10·d_pose)`` (with the 1e-12 floor the
    probes use), so the probe and allocator agree bit-for-bit on the pose term."""
    return math.sqrt(10.0 * float(d_pose) + 1e-12)


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "DEFAULT_LEVEL_GRID",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "WaterfillAllocation",
    "WaterfillStep",
    "byte_saving_score_value",
    "net_score_delta_from_components",
    "solve_waterfill_allocation",
    "sqrt_pose_term",
    "verify_kkt_marginal_equalization",
]
