# SPDX-License-Identifier: MIT
"""Amari generalized-Pythagorean cross-term diagnostic for the reverse-waterfill.

WHAT THIS MEASURES. The #157 reverse-waterfill bit allocator
(``tac.losses.variable_level_waterfill_allocator.solve_waterfill_allocation``)
composes its total distortion cost ADDITIVELY: ``total_dist_cost`` is the SUM of the
per-tensor marginal ``dist_cost`` steps. That additive sum is EXACT iff the allocation
components are *dual-orthogonal* in the sense of Amari's generalized Pythagorean theorem
(``docs/paper/information_geometric_foundations.md`` §5, Thm 6.12): in a dually-flat
space the canonical (Bregman) divergence splits ``D[P:Q] = D[P:R] + D[R:Q]`` with NO
cross-term exactly when the two legs meet at a dual-orthogonal (right) angle. Where the
components are NOT dual-orthogonal, the theorem NAMES the error — the cross-term — and
that cross-term IS the second-order interaction / Volterra term the meta-Lagrangian
already carries (per CLAUDE.md "Meta-Lagrangian/Pareto solver").

THE MATH (crisp, measurable). For two allocation components ``A`` and ``B``, measure the
score change ``ΔS`` each produces THROUGH THE REAL DISTORTION ORACLE (not the additive
RD table), then form the second-order mixed difference::

    cross(A, B) = ΔS(A ∪ B) − ΔS(A) − ΔS(B)

where ``ΔS(X) = oracle(X) − oracle(∅)`` (score change vs baseline; subtracting the
baseline makes the estimate robust to a non-zero oracle floor). Then:

  * ``cross ≈ 0``  ⟺  A, B dual-orthogonal  ⟺  the additive waterfill is EXACT for the
    pair (the Pythagorean decomposition has no cross-term).
  * ``cross > 0``  ⟺  SUPER-additive: the joint distortion EXCEEDS the sum, so the
    waterfill UNDER-estimates the coarsening cost of A and B together.
  * ``cross < 0``  ⟺  SUB-additive: the joint distortion is LESS than the sum, so the
    waterfill OVER-estimates the coarsening cost.

The sign therefore tells the apply-pass whether its additive ``total_dist_cost`` is an
optimistic or pessimistic estimate, and the magnitude bounds the correction.

ORACLE-AGNOSTIC BY DESIGN. ``measure_pairwise_crossterms`` takes ANY callable
``oracle(active: frozenset) -> float``. It works with a cheap synthetic oracle NOW (the
positive controls below) and the real through-R n600 d_seg/d_pose oracle LATER, with no
code change — only the callable differs. Oracle results are cached by ``frozenset`` so an
expensive oracle is evaluated at most once per distinct active set.

SCORE-CLAIM DISCIPLINE. This module is RATE-AXIS APPARATUS. It sharpens the additivity
TRUST of #157/#406's waterfill; it moves NO score itself. Every number it produces is a
diagnostic over whatever oracle is supplied — with a synthetic oracle the numbers are a
positive control, with the real oracle they are an advisory (never-promotable) signal
until a byte-closed dual CPU/CUDA 600-pair exact eval says otherwise.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

# An oracle maps the set of ACTIVE allocation components to a scalar score. By convention
# ``oracle(frozenset())`` is the baseline (no component applied); the diagnostic subtracts
# it so the estimate is robust to a non-zero floor. The score is "lower is better"
# (contest S), but the cross-term is a difference, so the diagnostic is sign-consistent
# for any monotone convention.
DistortionOracle = Callable[[frozenset], float]

# Orthogonality tolerance below which a cross-term is treated as "no interaction". Callers
# tune it per oracle noise floor; the default is a tight numeric-zero for exact oracles.
DEFAULT_ORTHOGONALITY_ABS_TOL = 1e-12


def _regime(cross: float, abs_tol: float) -> str:
    if abs(cross) <= abs_tol:
        return "orthogonal"
    return "superadditive" if cross > 0.0 else "subadditive"


@dataclass(frozen=True)
class CrossTermPair:
    """The measured Amari cross-term for one component pair ``(a, b)``.

    ``delta_a`` / ``delta_b`` are ``ΔS`` (vs baseline) for each component alone;
    ``delta_ab`` is ``ΔS`` for both applied jointly; ``cross = delta_ab − delta_a −
    delta_b`` is the dual-orthogonality deviation. ``regime`` classifies the pair as
    ``orthogonal`` (|cross| ≤ tol → additive is exact), ``superadditive`` (cross > 0 →
    waterfill under-estimates the joint cost) or ``subadditive`` (cross < 0 →
    over-estimates).
    """

    a: Hashable
    b: Hashable
    delta_a: float
    delta_b: float
    delta_ab: float
    cross: float
    regime: str

    @property
    def abs_cross(self) -> float:
        return abs(self.cross)

    @property
    def additive_prediction(self) -> float:
        """What the waterfill's additive sum would predict for the joint ΔS."""
        return self.delta_a + self.delta_b


@dataclass(frozen=True)
class AdditivityReport:
    """Pairwise cross-term matrix + a scalar additivity-error fraction.

    ``singles`` maps each component to its ``ΔS`` (vs baseline). ``pairs`` is every
    unordered pair's :class:`CrossTermPair`. ``total_abs_cross`` = Σ|cross|,
    ``total_abs_single`` = Σ|ΔS_single|. ``additivity_error_fraction`` =
    ``total_abs_cross / denom`` (denom per the ``denom`` arg) — the fraction of the
    additive prediction that is actually second-order interaction; ``0`` means the
    waterfill's additivity is exact, larger means the additive sum is less trustworthy.
    ``worst_pairs`` are the ``worst_k`` pairs by |cross| (the tensors whose joint
    coarsening the waterfill mis-estimates most). ``n_oracle_calls`` is the number of
    DISTINCT oracle evaluations (baseline + singles + pairs, de-duplicated by cache).
    """

    components: tuple[Hashable, ...]
    baseline: float
    singles: dict[Hashable, float]
    pairs: tuple[CrossTermPair, ...]
    total_abs_cross: float
    total_abs_single: float
    additivity_error_fraction: float
    worst_pairs: tuple[CrossTermPair, ...]
    orthogonality_abs_tol: float
    n_oracle_calls: int
    denom_kind: str = field(default="singles")

    @property
    def n_interacting_pairs(self) -> int:
        """Pairs whose |cross| exceeds the orthogonality tolerance."""
        return sum(1 for p in self.pairs if p.regime != "orthogonal")

    def pair(self, a: Hashable, b: Hashable) -> CrossTermPair | None:
        key = frozenset((a, b))
        for p in self.pairs:
            if frozenset((p.a, p.b)) == key:
                return p
        return None


def measure_pairwise_crossterms(
    components: Iterable[Hashable],
    oracle: DistortionOracle,
    *,
    subtract_baseline: bool = True,
    worst_k: int = 5,
    denom: str = "singles",
    orthogonality_abs_tol: float = DEFAULT_ORTHOGONALITY_ABS_TOL,
) -> AdditivityReport:
    """Measure the pairwise Amari cross-term matrix of an allocation's components.

    ``components`` — the allocation components (e.g. the coarsened tensor names from a
    realized :class:`WaterfillAllocation`, or any hashable atoms).
    ``oracle`` — ``oracle(active: frozenset) -> float``; the diagnostic evaluates it on
    ``∅``, each ``{c}``, and each ``{a, b}`` pair (results cached by frozenset). Supply a
    synthetic oracle for a positive control now, the real through-R oracle later.
    ``denom`` — ``"singles"`` (default): error fraction = Σ|cross| / Σ|ΔS_single|;
    ``"joint_all"``: divide by |ΔS(all components)| (one extra oracle call).

    Returns an :class:`AdditivityReport`. Pure arithmetic over the oracle; no score claim.
    """
    comps = tuple(components)
    if denom not in ("singles", "joint_all"):
        raise ValueError(f"denom must be 'singles' or 'joint_all', got {denom!r}")

    cache: dict[frozenset, float] = {}

    def _call(active: frozenset) -> float:
        if active not in cache:
            cache[active] = float(oracle(active))
        return cache[active]

    baseline = _call(frozenset()) if subtract_baseline else 0.0

    def _delta(active: frozenset) -> float:
        return _call(active) - baseline

    singles: dict[Hashable, float] = {c: _delta(frozenset((c,))) for c in comps}

    pairs: list[CrossTermPair] = []
    for a, b in combinations(comps, 2):
        d_a = singles[a]
        d_b = singles[b]
        d_ab = _delta(frozenset((a, b)))
        cross = d_ab - d_a - d_b
        pairs.append(
            CrossTermPair(
                a=a,
                b=b,
                delta_a=d_a,
                delta_b=d_b,
                delta_ab=d_ab,
                cross=cross,
                regime=_regime(cross, orthogonality_abs_tol),
            )
        )

    total_abs_cross = math.fsum(p.abs_cross for p in pairs)
    total_abs_single = math.fsum(abs(v) for v in singles.values())

    if denom == "joint_all":
        joint_all = abs(_delta(frozenset(comps))) if comps else 0.0
        denom_value = joint_all
    else:
        denom_value = total_abs_single

    error_fraction = (total_abs_cross / denom_value) if denom_value > 0.0 else 0.0

    worst = tuple(sorted(pairs, key=lambda p: p.abs_cross, reverse=True)[: max(0, worst_k)])

    return AdditivityReport(
        components=comps,
        baseline=baseline,
        singles=singles,
        pairs=tuple(pairs),
        total_abs_cross=total_abs_cross,
        total_abs_single=total_abs_single,
        additivity_error_fraction=error_fraction,
        worst_pairs=worst,
        orthogonality_abs_tol=orthogonality_abs_tol,
        n_oracle_calls=len(cache),
        denom_kind=denom,
    )


def additivity_canary(
    report: AdditivityReport,
    *,
    rel_tol: float = 0.05,
    abs_tol: float | None = None,
) -> tuple[bool, str]:
    """Positive-control canary: FIRE when a report shows non-negligible interaction.

    Returns ``(fired, message)``. ``fired`` is ``True`` when the additivity is NOT
    trustworthy — either the ``additivity_error_fraction`` exceeds ``rel_tol`` OR (if
    ``abs_tol`` is given) any single pair's |cross| exceeds ``abs_tol``. A well-behaved
    dual-orthogonal allocation keeps the canary QUIET; an interacting allocation FIRES it,
    which is exactly the positive control that proves the tool measures what it claims.
    """
    worst = report.worst_pairs[0] if report.worst_pairs else None
    frac = report.additivity_error_fraction
    rel_fired = frac > rel_tol
    abs_fired = abs_tol is not None and worst is not None and worst.abs_cross > abs_tol
    fired = bool(rel_fired or abs_fired)
    if not fired:
        return False, (
            f"QUIET: additivity trustworthy — error fraction {frac:.3e} <= rel_tol "
            f"{rel_tol:.3e}, {report.n_interacting_pairs}/{len(report.pairs)} pairs "
            "interact above the orthogonality tolerance (waterfill additive sum is "
            "dual-orthogonal / Pythagorean-exact)."
        )
    worst_desc = (
        f"worst pair ({worst.a!r},{worst.b!r}) cross={worst.cross:+.3e} [{worst.regime}]"
        if worst is not None
        else "no pairs"
    )
    return True, (
        f"FIRED: additivity NOT trustworthy — error fraction {frac:.3e} > rel_tol "
        f"{rel_tol:.3e}; {worst_desc}. The additive waterfill total_dist_cost carries a "
        "systematic cross-term (the meta-Lagrangian Volterra / interaction term)."
    )


# --------------------------------------------------------------------------------------
# Reusable oracle builders (score-neutral): a NULL from the waterfill's own RD table and a
# synthetic bilinear control. Both let anyone validate the diagnostic without a GPU.
# --------------------------------------------------------------------------------------


def rd_table_additive_oracle(
    rd_table: dict[Hashable, dict[int, tuple[float, float]]],
    levels: dict[Hashable, int],
) -> DistortionOracle:
    """Build the ADDITIVE-BY-CONSTRUCTION oracle the #157 waterfill implicitly assumes.

    ``rd_table`` is the waterfill's ``{tensor: {n_levels: (byte_saving, dist_cost)}}`` and
    ``levels`` the realized per-tensor allocation. The returned oracle sums each active
    tensor's measured ``dist_cost`` at its allocated level — i.e. it is the waterfill's own
    additive prediction. Feeding THIS oracle to :func:`measure_pairwise_crossterms` yields
    ``cross ≈ 0`` on EVERY pair by construction: it is the NULL / dual-orthogonal baseline
    the REAL through-R oracle is compared against. Any non-zero cross-term the real oracle
    reports is the deviation from THIS additive model.
    """

    def _oracle(active: frozenset) -> float:
        total = 0.0
        for tensor in active:
            curve = rd_table.get(tensor)
            if curve is None:
                continue
            lvl = levels.get(tensor)
            if lvl is None or lvl not in curve:
                continue
            total += float(curve[lvl][1])  # dist_cost at the allocated level
        return total

    return _oracle


def bilinear_synthetic_oracle(
    linear: dict[Hashable, float],
    couplings: dict[frozenset, float] | None = None,
) -> DistortionOracle:
    """A synthetic oracle with EXACT, known cross-terms — the positive control.

    ``linear[c]`` is component ``c``'s solo distortion; ``couplings[{a,b}]`` is a bilinear
    coupling that only contributes when BOTH ``a`` and ``b`` are active. The oracle is::

        oracle(S) = Σ_{c in S} linear[c]  +  Σ_{{a,b} ⊆ S} couplings[{a,b}]

    so ``cross(a, b) = couplings.get({a, b}, 0.0)`` EXACTLY, with baseline ``0``. A pair
    with no coupling is dual-orthogonal (cross = 0, canary quiet); a pair with a positive
    coupling is superadditive (cross > 0, canary fires); negative coupling is subadditive.
    This is the deterministic positive control the tests assert against and the reference
    anyone can use to re-verify the diagnostic without the GPU oracle.
    """
    couplings = couplings or {}

    def _oracle(active: frozenset) -> float:
        total = math.fsum(linear.get(c, 0.0) for c in active)
        for key, val in couplings.items():
            if key <= active:
                total += val
        return total

    return _oracle


def quadratic_support_oracle(
    vectors: dict[Hashable, Sequence[float]],
) -> DistortionOracle:
    """A geometric synthetic oracle: ``oracle(S) = || Σ_{c in S} v_c ||²``.

    Then ``ΔS(S) = ||Σ v_c||²`` (baseline 0) and ``cross(a, b) = 2 · <v_a, v_b>`` EXACTLY
    — the cross-term IS twice the inner product, tying the additivity deviation directly to
    dual-orthogonality: orthogonal support vectors → cross = 0 (Pythagorean-exact),
    aligned support → cross > 0 (shared distortion, superadditive), anti-aligned → cross <
    0. This mirrors the real mechanism: two coarsened tensors that perturb OVERLAPPING
    scorer-response directions interact; disjoint-support tensors do not.
    """

    def _oracle(active: frozenset) -> float:
        if not active:
            return 0.0
        dim = len(next(iter(vectors.values())))
        acc = [0.0] * dim
        for c in active:
            v = vectors[c]
            for i in range(dim):
                acc[i] += v[i]
        return math.fsum(x * x for x in acc)

    return _oracle


# --------------------------------------------------------------------------------------
# #157 / #406 validation surface (ADDITIVE / diagnostic only — does NOT change the solve).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WaterfillAdditivityDiagnostic:
    """Compares the #157 waterfill's additive ``total_dist_cost`` against the true joint.

    ``waterfill_additive_dist_cost`` is the solver's summed cost over the coarsened
    tensors; ``true_joint_delta`` is the real oracle's ``ΔS`` for ALL coarsened tensors
    applied together; ``residual = true_joint_delta − waterfill_additive_dist_cost`` is the
    total additivity error (the sum of all cross-terms, to second order). ``report`` holds
    the pairwise breakdown that LOCALISES that residual to the worst-interacting tensor
    pairs. ``trustworthy`` is ``True`` when the canary stays quiet at ``rel_tol``.
    """

    coarsened_components: tuple[Hashable, ...]
    waterfill_additive_dist_cost: float
    true_joint_delta: float
    residual: float
    report: AdditivityReport
    canary_fired: bool
    canary_message: str

    @property
    def trustworthy(self) -> bool:
        return not self.canary_fired


def diagnose_waterfill_additivity(
    rd_table: dict[Hashable, dict[int, tuple[float, float]]],
    levels: dict[Hashable, int],
    joint_oracle: DistortionOracle,
    *,
    base_level: int = 127,
    worst_k: int = 5,
    rel_tol: float = 0.05,
    orthogonality_abs_tol: float = DEFAULT_ORTHOGONALITY_ABS_TOL,
) -> WaterfillAdditivityDiagnostic:
    """Validate a realized #157 allocation's additivity WITHOUT changing the solve.

    ``rd_table`` / ``levels`` are the waterfill's measured RD table and realized
    allocation; ``joint_oracle(active)`` measures the REAL score change of coarsening the
    ``active`` subset of tensors to their allocated levels (this is the expensive
    through-R oracle at n600 — DEFERRED per the task's memory containment). The coarsened
    components are the tensors whose allocated level is below ``base_level``. Returns a
    :class:`WaterfillAdditivityDiagnostic` comparing the solver's additive cost to the true
    joint and localising the residual to pairs. Pure diagnostic — it reads the allocation,
    never mutates it.
    """
    coarsened = tuple(t for t, lv in levels.items() if lv < base_level)
    additive_cost = math.fsum(
        float(rd_table[t][levels[t]][1])
        for t in coarsened
        if t in rd_table and levels[t] in rd_table[t]
    )
    baseline = float(joint_oracle(frozenset()))
    true_joint = float(joint_oracle(frozenset(coarsened))) - baseline
    residual = true_joint - additive_cost
    report = measure_pairwise_crossterms(
        coarsened,
        joint_oracle,
        worst_k=worst_k,
        orthogonality_abs_tol=orthogonality_abs_tol,
    )
    fired, msg = additivity_canary(report, rel_tol=rel_tol)
    return WaterfillAdditivityDiagnostic(
        coarsened_components=coarsened,
        waterfill_additive_dist_cost=additive_cost,
        true_joint_delta=true_joint,
        residual=residual,
        report=report,
        canary_fired=fired,
        canary_message=msg,
    )


# Score-claim discipline metadata (this module is rate-axis apparatus; no score claim).
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "DEFAULT_ORTHOGONALITY_ABS_TOL",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "AdditivityReport",
    "CrossTermPair",
    "DistortionOracle",
    "WaterfillAdditivityDiagnostic",
    "additivity_canary",
    "bilinear_synthetic_oracle",
    "diagnose_waterfill_additivity",
    "measure_pairwise_crossterms",
    "quadratic_support_oracle",
    "rd_table_additive_oracle",
]
