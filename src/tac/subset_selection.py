# SPDX-License-Identifier: MIT
"""Canonical seeded subset selector with MANDATORY population-ratio provenance.

Why this module exists
----------------------
``ddm_na2`` (2026-08-03) MEASURED the enabling defect: **110 measurement/verdict
tools slice ``[:n]`` on pairs/frames and ZERO offered a stratified or random
selector.** With no alternative in the repo, a document that is silent on its
selection mode is not *unknown* -- it is **PREFIX**, and the measured silence
rate is **259 of 362 (71.5%)** same-line candidates.

That silence is not neutral, because prefix bias is **anti-conservative on the
pose axis**:

===== ============================= ====== ====== ====== ====== ==================
axis   governing quantity            n=24   n=48   n=64   n=96   direction
===== ============================= ====== ====== ====== ====== ==================
seg    flip-prone pixel density      0.97x  0.95x  0.96x  0.97x  EASIER (3-5%)
pose   ``d_pose_shipped_f16``        2.54x  2.64x  2.65x  4.21x  HARDER (154-321%)
===== ============================= ====== ====== ====== ====== ==================

Pose re-derived 2026-08-04 by ``ddm_mi1`` from the 600-row PFS1 D2 receipt
``/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl``
(sha256 ``d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5``,
field ``d_pose_shipped_f16``). Nuance preserved: under this module's current
p01/p99 random band, n=24's 2.535x prefix is inside the band, while n=48/64/96
are outside; n=8 is not bankable population evidence.

Independently re-derived here for the seg axis (2026-08-03, ``ddm_ss1``, by
streaming ``gt_n600.npz`` ``margins``, 0.38 s): at flip-margin thresholds
0.01/0.05/0.10/0.25 -- a 25x span -- the n=24/48/64/96 prefix ratios are
0.963/0.941/0.948/0.948, 0.952/0.948/0.949/0.946, 0.955/0.943/0.946/0.944 and
0.961/0.943/0.949/0.949. **The seg half reproduces and is threshold-stable.**

**Mechanism, measured:** video order is temporally correlated, so a prefix is a
contiguous SCENE BLOCK, not a sample. Pose difficulty per 60-pair block runs
``0.41, 0.82, 0.08 ... 0.010`` -- the first 120 pairs are the two hardest blocks,
**79x** the easiest. Compounding it, ``600_pair_independence_test_result_...``
measured **serial effective N = 40.22 of 600**, so an n=8 prefix is ~0.5
effective samples and n=24 is ~1.6.

The one-line rule this module makes mechanical
----------------------------------------------
``ddm_bp2`` stated the cure in a sentence: *"report the subset's mean of the
governing quantity against the population's. If the ratio is not ~1, the verdict
is about a different population."* ``ddm_sq1`` §2.9 is the model of correct
conduct: *"No population claim. All numbers are n=32 ... the subset is 0.2692x
population on d_pose and is reported as such."*

So :func:`select` does not merely pick indices. **It refuses to pick them
without recording how**, and when handed the governing quantity it reports the
subset/population ratio *and* whether that ratio is inside the band a seeded
random subset of the same size would produce.

Three design commitments, each answering a measured failure
-----------------------------------------------------------
1. **No default mode.** ``mode`` is a required keyword. ``MODE_PREFIX`` remains
   available -- prefixes are legitimate for reproducing a prior prefix-scoped
   number -- but it can only be reached by naming it, so it can never again be
   what silence resolves to. (na2 LAW C: silence resolves to the DEFAULT.)

2. **The null band is DERIVED, not a constant.** A hardcoded "ratio within
   +/-10%" tolerance would be exactly the borrowed-constant poison the campaign
   keeps paying for. Instead :func:`governing_ratio` bootstraps the ratio's own
   null distribution from the population at the *same* n and seed policy, and
   reports a derived null band. na2's measured random-sample p95 at n=96 (1.48x) is then a
   *derived output* of this function, not a magic number inside it.

3. **An absent or empty governing table is VACUOUS, never PASS.** The genus
   ``[]``-is-the-success-channel has now bitten this repo at four layers in one
   week. A missing table yields ``VERDICT_VACUOUS_NO_TABLE`` with a stated
   reason, which is not a matched population and must never be read as one.

Scope
-----
This module is PURE: no I/O, no numpy-file knowledge, no scorer. It takes a
population size and (optionally) a governing-quantity sequence, and returns
indices plus provenance. Callers that need to *derive* a governing table from
cached ground truth use :mod:`tac.subset_selection_tables`.

Sisters
-------
``tools/build_strided_subset_gt.py`` (the pre-existing strided GT builder --
:data:`MODE_STRIDED` reproduces its index math exactly, so subsets built by that
tool can be described by this selector) and
``tac.formula_extinctions.stratified_kfold_video_chunks`` (train/val chunk
stratification: a different purpose, but the same measured temporal-correlation
mechanism that makes contiguous blocks non-exchangeable).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

SCHEMA = "pact.subset_selection.v1"

# --- selection modes -------------------------------------------------------
# String values are the canonical provenance labels. `video_order_prefix` is
# spelled the way the one previously-remediated tool spelled it, so existing
# emitted provenance stays joinable.
MODE_PREFIX = "video_order_prefix"
MODE_SEEDED_RANDOM = "seeded_random"
MODE_STRATIFIED = "stratified_blocks"
MODE_QUANTILE_STRATIFIED = "quantile_stratified"
MODE_STRIDED = "strided"
MODE_EXPLICIT = "explicit_indices"
MODE_FULL = "full_population"

ALL_MODES: tuple[str, ...] = (
    MODE_PREFIX,
    MODE_SEEDED_RANDOM,
    MODE_STRATIFIED,
    MODE_QUANTILE_STRATIFIED,
    MODE_STRIDED,
    MODE_EXPLICIT,
    MODE_FULL,
)

#: Modes that draw a subset without any representativeness argument. A verdict
#: taken on one of these is INSTANCE-scoped to that subset until a governing
#: ratio says otherwise.
UNREPRESENTATIVE_MODES: frozenset[str] = frozenset({MODE_PREFIX})

# --- governing-ratio verdicts ----------------------------------------------
VERDICT_MATCHED = "MATCHED"
VERDICT_DIFFERENT_POPULATION = "DIFFERENT_POPULATION"
VERDICT_VACUOUS_NO_TABLE = "VACUOUS_NO_TABLE"
VERDICT_VACUOUS_EMPTY = "VACUOUS_EMPTY"
VERDICT_DEGENERATE_POPULATION = "DEGENERATE_POPULATION"

#: Verdicts that do NOT license a population claim. Note that VACUOUS_* are in
#: here: "we could not check" is not "we checked and it matched".
NON_MATCHED_VERDICTS: frozenset[str] = frozenset(
    {
        VERDICT_DIFFERENT_POPULATION,
        VERDICT_VACUOUS_NO_TABLE,
        VERDICT_VACUOUS_EMPTY,
        VERDICT_DEGENERATE_POPULATION,
    }
)

#: Bootstrap replicate count for the null band. 2000 is enough for a stable band
#: at n<=600 and costs ~10 ms in pure Python; it is a precision knob on an
#: advisory band, not a physical constant.
DEFAULT_BOOTSTRAP = 2000

#: Two-sided false-positive rate of the null band. This is NOT a tolerance on the
#: ratio -- it is the probability that an HONEST seeded-random draw is flagged,
#: and it is the only free parameter in the check. It is exposed as alpha rather
#: than as a "ratio within +/-x%" threshold precisely so the knob is a stated
#: error rate rather than a borrowed constant.
#:
#: 0.02 (p01/p99), not 0.10 (p05/p95), and the reason is measured: during this
#: module's own test development a p05/p95 band flagged a legitimate seeded
#: random draw at n=96 on the block-biased fixture -- exactly the 1-in-10 the
#: band promises. An instrument that fires on 10% of honest draws is one whose
#: users learn to ignore it, which is the failure mode this whole landing exists
#: to prevent. At alpha=0.02 the na2 pose prefix (4.21x against a random p95 of
#: 1.48x) is still caught by a wide margin.
DEFAULT_ALPHA = 0.02

#: Default number of contiguous temporal blocks for MODE_STRATIFIED. 10 blocks
#: over the canonical 600-pair population reproduces the 60-pair block structure
#: na2 measured the pose difficulty profile on (0.41, 0.82, 0.08 ... 0.010).
DEFAULT_STRATIFIED_BLOCKS = 10


class SubsetSelectionError(ValueError):
    """Raised when a selection request is structurally invalid."""


class PopulationMismatchError(AssertionError):
    """Raised by :func:`assert_population_matched` on a non-matched subset."""


@dataclass(frozen=True)
class GoverningRatio:
    """Subset-vs-population comparison for one governing quantity.

    ``ratio`` is ``subset_mean / population_mean``. ``null_p05``/``null_p95``
    bound what a seeded-random subset of the same size produces on this same
    population, so ``ratio`` outside that band means the subset is not a sample
    of the population -- it is a sample of a different one.
    """

    name: str
    subset_mean: float | None
    population_mean: float | None
    ratio: float | None
    n_subset: int
    n_population: int
    verdict: str
    null_p05: float | None = None
    null_p95: float | None = None
    n_bootstrap: int = 0
    reason: str = ""

    @property
    def matched(self) -> bool:
        """True only when the check ran AND the ratio sits inside the null band."""
        return self.verdict == VERDICT_MATCHED

    def summary(self) -> str:
        """One-line human summary -- the line ``ddm_bp2`` said would have caught it."""
        if self.ratio is None:
            return f"{self.name}: {self.verdict} ({self.reason})"
        band = ""
        if self.null_p05 is not None and self.null_p95 is not None:
            band = f", seeded-random band [{self.null_p05:.4g}, {self.null_p95:.4g}] (n_boot={self.n_bootstrap})"
        return (
            f"{self.name}: subset mean {self.subset_mean:.6g} vs population "
            f"{self.population_mean:.6g} = {self.ratio:.4g}x{band} => {self.verdict}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "subset_mean": self.subset_mean,
            "population_mean": self.population_mean,
            "ratio": self.ratio,
            "n_subset": self.n_subset,
            "n_population": self.n_population,
            "verdict": self.verdict,
            "null_p05": self.null_p05,
            "null_p95": self.null_p95,
            "n_bootstrap": self.n_bootstrap,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Selection:
    """Chosen indices plus the provenance that makes them interpretable."""

    indices: tuple[int, ...]
    mode: str
    population: int
    seed: int | None = None
    params: dict[str, object] = field(default_factory=dict)
    ratios: tuple[GoverningRatio, ...] = ()

    @property
    def n(self) -> int:
        return len(self.indices)

    @property
    def is_representative_mode(self) -> bool:
        return self.mode not in UNREPRESENTATIVE_MODES

    @property
    def population_matched(self) -> bool:
        """True only if at least one governing ratio ran AND every one matched.

        Zero ratios is **not** matched: an unchecked subset is unchecked. This
        is the ``[]``-is-not-success rule applied at the selector's own surface.
        """
        return bool(self.ratios) and all(r.matched for r in self.ratios)

    def unmatched_ratios(self) -> tuple[GoverningRatio, ...]:
        return tuple(r for r in self.ratios if not r.matched)

    def provenance(self) -> dict[str, object]:
        """The record every subset-scoped result should carry."""
        return {
            "schema": SCHEMA,
            "pair_selection": self.mode,
            "n": self.n,
            "population": self.population,
            "seed": self.seed,
            "params": dict(self.params),
            "indices_head": list(self.indices[:16]),
            "representative_mode": self.is_representative_mode,
            "population_matched": self.population_matched,
            "governing_ratios": [r.as_dict() for r in self.ratios],
        }

    def summary(self) -> str:
        head = (
            f"selection: mode={self.mode} n={self.n}/{self.population} "
            f"seed={self.seed}"
        )
        if not self.ratios:
            return head + " | governing ratio NOT CHECKED (subset is INSTANCE-scoped)"
        return head + " | " + " ; ".join(r.summary() for r in self.ratios)


def _validate_n(n: int, population: int) -> None:
    if population <= 0:
        raise SubsetSelectionError(f"population must be positive, got {population}")
    if n <= 0:
        raise SubsetSelectionError(f"n must be positive, got {n}")
    if n > population:
        raise SubsetSelectionError(f"n={n} exceeds population={population}")


def _require_seed(seed: int | None, mode: str) -> int:
    if seed is None:
        raise SubsetSelectionError(
            f"mode={mode!r} requires an explicit seed so the subset is reproducible; "
            "pass seed=<int>"
        )
    return int(seed)


def prefix_indices(n: int, population: int) -> tuple[int, ...]:
    """The first ``n`` indices in video order.

    Kept as a named, first-class mode precisely so it can be *chosen*. It is the
    right selection when reproducing an existing prefix-scoped number; it is the
    wrong one for any population claim.
    """
    _validate_n(n, population)
    return tuple(range(n))


def seeded_random_indices(n: int, population: int, seed: int) -> tuple[int, ...]:
    """``n`` distinct indices drawn without replacement, reproducibly.

    Uses :class:`random.Random` rather than numpy so this module stays pure and
    import-light; the draw is deterministic for a given ``(n, population, seed)``.
    """
    _validate_n(n, population)
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(range(population), n)))


def stratified_indices(
    n: int,
    population: int,
    seed: int,
    *,
    block_count: int = DEFAULT_STRATIFIED_BLOCKS,
) -> tuple[int, ...]:
    """Seeded draw spread proportionally across contiguous temporal blocks.

    This is the mode that directly attacks the measured mechanism. na2 found the
    first 120 of 600 pairs are the two hardest 60-pair blocks (79x the easiest);
    a prefix therefore samples two blocks out of ten. Stratifying over contiguous
    blocks forces representation from every block, and seeding *within* each
    block keeps it reproducible.

    Allocation uses largest-remainder so the per-block counts sum to exactly
    ``n`` without a block being silently starved by floor().
    """
    _validate_n(n, population)
    if block_count <= 0:
        raise SubsetSelectionError(f"block_count must be positive, got {block_count}")
    block_count = min(block_count, population)

    # Contiguous blocks; the first (population % block_count) blocks get one extra
    # element so every index belongs to exactly one block.
    base, extra = divmod(population, block_count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for b in range(block_count):
        size = base + (1 if b < extra else 0)
        bounds.append((start, start + size))
        start += size

    # Largest-remainder allocation of n across blocks, proportional to block size,
    # capped by block size so an over-allocated small block cannot ask for more
    # indices than it contains.
    quotas: list[float] = [(hi - lo) * n / population for lo, hi in bounds]
    counts = [min(math.floor(q), bounds[i][1] - bounds[i][0]) for i, q in enumerate(quotas)]
    remainder = n - sum(counts)
    order = sorted(
        range(block_count),
        key=lambda i: (-(quotas[i] - math.floor(quotas[i])), i),
    )
    # Review pass 1 finding: this loop previously carried a `cursor > block_count:
    # break` guard, which could exit with remainder > 0 and return FEWER than n
    # indices -- silently, since nothing downstream re-checks the length. A silent
    # wrong-length return is the same genus this module exists to close, so the
    # bound is gone: `progressed` alone guarantees termination (every pass either
    # places at least one index or raises), and total capacity is `population >= n`
    # by _validate_n, so the raise is genuinely unreachable rather than a fallback.
    while remainder > 0:
        progressed = False
        for i in order:
            if remainder == 0:
                break
            lo, hi = bounds[i]
            if counts[i] < hi - lo:
                counts[i] += 1
                remainder -= 1
                progressed = True
        if not progressed:  # pragma: no cover - unreachable while n <= population
            raise SubsetSelectionError(
                f"could not allocate n={n} across {block_count} blocks of "
                f"population={population}; this is a bug, not a config error"
            )

    rng = random.Random(seed)
    picked: list[int] = []
    for (lo, hi), take in zip(bounds, counts, strict=True):
        if take:
            picked.extend(rng.sample(range(lo, hi), take))
    return tuple(sorted(picked))


def quantile_stratified_indices(
    n: int,
    population: int,
    rank_by: Sequence[float],
    *,
    quantiles: tuple[float, float] = (0.25, 0.75),
) -> tuple[int, ...]:
    """Deterministic content-stratified draw: one index per equal temporal stratum.

    **This is not a new algorithm.** It is the selector that already existed in
    this repo, lifted here verbatim from the ``_select_pairs`` private helper
    that was copy-pasted between ``tools/measure_uint8_lattice_feasibility.py``
    (:376) and ``tools/constructive_inverse_solve_harness.py`` (:435). Those two
    twins were the only genuinely stratified pair selection in the codebase, and
    being ``_``-private and duplicated they could not be reused -- which is a
    large part of why 110 other sites reached for ``[:n]`` instead.

    The algorithm: cut the population into ``n`` equal temporal strata; within
    each, rank by ``rank_by`` (the originals pass a per-pair *fragility*,
    ``mean(cached margin < m_safe)``); take the ``0.25`` quantile in even strata
    and the ``0.75`` quantile in odd ones, so the draw is neither uniformly easy
    nor uniformly hard; break ties by pair index so it is reproducible without a
    seed.

    Verified byte-faithful against the originals' own drift guard: with the real
    ``gt_n600`` fragility at ``m_safe = 0.039180326461791926`` this returns
    ``KNOWN_N6 = (90, 175, 277, 381, 424, 573)``. See
    ``test_quantile_stratified_reproduces_known_n6``.

    Deterministic and content-aware, where :func:`stratified_indices` is seeded
    and content-blind; prefer this one when a meaningful per-index difficulty is
    available, and that one when it is not.
    """
    _validate_n(n, population)
    ranks = [float(v) for v in rank_by]
    if len(ranks) != population:
        raise SubsetSelectionError(
            f"rank_by has {len(ranks)} entries but population is {population}"
        )
    lo_q, hi_q = quantiles
    for q in (lo_q, hi_q):
        if not (0.0 <= q <= 1.0):
            raise SubsetSelectionError(f"quantiles must lie in [0,1], got {quantiles}")

    # Equal temporal strata, reproducing `np.linspace(0, population, n + 1,
    # dtype=np.int64)` from the originals EXACTLY. Two non-obvious properties, both
    # found by measurement rather than by reading:
    #
    #  1. The int64 cast TRUNCATES, it does not round (n=7 over 600 -> edge 85).
    #     My first draft used round() and still passed the KNOWN_N6 regression,
    #     because 600/6 divides exactly so the two agree. A sweep over more n then
    #     measured 43 of 67 values disagreeing. Verifying against the one anchor a
    #     tool pins is agreeing with the test; the equality that matters is with
    #     the ALGORITHM, across n.
    #  2. linspace multiplies by a PRECOMPUTED step, so the float error is
    #     `i * (population/n)`, not `(i*population)/n`. At n=28, i=21 those differ:
    #     449.99999999999994 (truncates to 449) versus exactly 450.0. Four n values
    #     out of 105 still disagreed until this line matched the association order.
    #     The final endpoint is pinned to `population` because linspace sets it.
    #
    # Tie handling needed no change and was separately measured identical:
    # np.lexsort((members, rank[members])) == sorted(members, key=(rank, index)).
    step = population / n
    edges = [int(i * step) for i in range(n)] + [population]
    picked: list[int] = []
    for stratum in range(n):
        members = list(range(edges[stratum], edges[stratum + 1]))
        if not members:  # pragma: no cover - impossible while n <= population
            raise SubsetSelectionError("empty stratum")
        # Sort by rank value, ties broken by index -- the lexsort the originals use.
        order = sorted(members, key=lambda m: (ranks[m], m))
        q = lo_q if stratum % 2 == 0 else hi_q
        # Python's round() is banker's rounding, matching numpy's round() in the
        # originals; the position formula is theirs unchanged.
        position = round(q * (len(order) - 1))
        picked.append(order[int(position)])
    return tuple(sorted(picked))


def strided_indices(
    n: int | None,
    population: int,
    *,
    stride: int,
    offset: int = 0,
) -> tuple[int, ...]:
    """Every ``stride``-th index, matching ``tools/build_strided_subset_gt.py``.

    Deterministic and unseeded, and it spans the whole drive -- which is why the
    pre-existing strided GT builder chose it ("strided = every Nth pair so it
    SPANS the whole continuous comma2k19 drive -> scene diversity"). Reproduced
    here so subsets that tool already built can be *described* by this selector
    rather than re-derived beside it.
    """
    if population <= 0:
        raise SubsetSelectionError(f"population must be positive, got {population}")
    if stride <= 0:
        raise SubsetSelectionError(f"stride must be positive, got {stride}")
    if not (0 <= offset < population):
        raise SubsetSelectionError(f"offset {offset} outside population {population}")
    idx = tuple(range(offset, population, stride))
    if n is not None:
        if n > len(idx):
            raise SubsetSelectionError(
                f"stride={stride} offset={offset} yields {len(idx)} indices, fewer than n={n}"
            )
        idx = idx[:n]
    return idx


def _mean(values: Sequence[float]) -> float:
    return math.fsum(float(v) for v in values) / len(values)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile on an already-sorted sequence."""
    if not sorted_values:  # pragma: no cover - guarded by caller
        raise SubsetSelectionError("empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def governing_ratio(
    indices: Iterable[int],
    governing: Sequence[float] | None,
    *,
    name: str = "governing",
    seed: int = 0,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> GoverningRatio:
    """Compare the subset's mean of a governing quantity to the population's.

    The null band is **derived here, not configured**: ``n_bootstrap`` seeded
    random subsets of the same size are drawn from this same population and the
    p05/p95 of their ratios reported. A ratio outside that band is a measured
    statement that the subset is not exchangeable with the population -- which is
    what na2's 4.21x pose prefix (against a measured random p95 of 1.48x) is.

    A ``None`` or empty ``governing`` returns a VACUOUS verdict with its reason,
    never a pass.
    """
    idx = tuple(int(i) for i in indices)
    if governing is None:
        return GoverningRatio(
            name=name,
            subset_mean=None,
            population_mean=None,
            ratio=None,
            n_subset=len(idx),
            n_population=0,
            verdict=VERDICT_VACUOUS_NO_TABLE,
            reason="no governing-quantity population table supplied",
        )
    pop = [float(v) for v in governing]
    if not pop:
        return GoverningRatio(
            name=name,
            subset_mean=None,
            population_mean=None,
            ratio=None,
            n_subset=len(idx),
            n_population=0,
            verdict=VERDICT_VACUOUS_EMPTY,
            reason="governing-quantity table is empty (examined 0 values)",
        )
    if not idx:
        return GoverningRatio(
            name=name,
            subset_mean=None,
            population_mean=_mean(pop),
            ratio=None,
            n_subset=0,
            n_population=len(pop),
            verdict=VERDICT_VACUOUS_EMPTY,
            reason="subset is empty (examined 0 indices)",
        )
    bad = [i for i in idx if not (0 <= i < len(pop))]
    if bad:
        raise SubsetSelectionError(
            f"indices out of range for governing table of length {len(pop)}: {bad[:8]}"
        )

    pop_mean = _mean(pop)
    sub_mean = _mean([pop[i] for i in idx])
    if pop_mean == 0.0:
        return GoverningRatio(
            name=name,
            subset_mean=sub_mean,
            population_mean=pop_mean,
            ratio=None,
            n_subset=len(idx),
            n_population=len(pop),
            verdict=VERDICT_DEGENERATE_POPULATION,
            reason="population mean is exactly 0; a ratio is undefined",
        )

    ratio = sub_mean / pop_mean
    if len(idx) >= len(pop):
        # The "subset" is the population; the ratio is 1 by construction and a
        # bootstrap band would be degenerate.
        return GoverningRatio(
            name=name,
            subset_mean=sub_mean,
            population_mean=pop_mean,
            ratio=ratio,
            n_subset=len(idx),
            n_population=len(pop),
            verdict=VERDICT_MATCHED,
            null_p05=1.0,
            null_p95=1.0,
            n_bootstrap=0,
            reason="subset is the full population",
        )

    if not (0.0 < alpha < 1.0):
        raise SubsetSelectionError(f"alpha must lie in (0,1), got {alpha}")
    rng = random.Random(seed)
    n = len(idx)
    span = range(len(pop))
    draws = [_mean([pop[i] for i in rng.sample(span, n)]) / pop_mean for _ in range(max(1, n_bootstrap))]
    draws.sort()
    p05 = _percentile(draws, alpha / 2.0)
    p95 = _percentile(draws, 1.0 - alpha / 2.0)
    verdict = VERDICT_MATCHED if p05 <= ratio <= p95 else VERDICT_DIFFERENT_POPULATION
    return GoverningRatio(
        name=name,
        subset_mean=sub_mean,
        population_mean=pop_mean,
        ratio=ratio,
        n_subset=n,
        n_population=len(pop),
        verdict=verdict,
        null_p05=p05,
        null_p95=p95,
        n_bootstrap=max(1, n_bootstrap),
    )


def select(
    n: int | None,
    population: int,
    *,
    mode: str,
    seed: int | None = None,
    block_count: int = DEFAULT_STRATIFIED_BLOCKS,
    stride: int | None = None,
    offset: int = 0,
    indices: Sequence[int] | None = None,
    rank_by: Sequence[float] | None = None,
    quantiles: tuple[float, float] = (0.25, 0.75),
    governing: Sequence[float] | None = None,
    governing_name: str = "governing",
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> Selection:
    """Select ``n`` of ``population`` indices under an explicitly named ``mode``.

    ``mode`` has **no default**. That is the point: with 110 tools slicing
    ``[:n]`` and nothing else on offer, an omitted mode used to resolve silently
    to a prefix. Here it cannot resolve to anything -- it must be said.

    Pass ``governing`` (a population-length sequence of the per-pair quantity the
    verdict turns on) to get the subset/population ratio computed and banded. It
    is optional only because not every call site has a table; a call without it
    returns ``population_matched == False`` and says so.
    """
    if mode not in ALL_MODES:
        raise SubsetSelectionError(f"unknown mode {mode!r}; expected one of {ALL_MODES}")

    params: dict[str, object] = {}
    if mode == MODE_FULL:
        chosen = tuple(range(population))
    elif mode == MODE_EXPLICIT:
        if indices is None:
            raise SubsetSelectionError("mode=explicit_indices requires indices=[...]")
        chosen = tuple(int(i) for i in indices)
        if len(set(chosen)) != len(chosen):
            raise SubsetSelectionError("explicit indices contain duplicates")
        out_of_range = [i for i in chosen if not (0 <= i < population)]
        if out_of_range:
            raise SubsetSelectionError(
                f"explicit indices outside population {population}: {out_of_range[:8]}"
            )
    elif mode == MODE_STRIDED:
        if stride is None:
            raise SubsetSelectionError("mode=strided requires stride=<int>")
        chosen = strided_indices(n, population, stride=stride, offset=offset)
        params["stride"] = stride
        params["offset"] = offset
    elif mode == MODE_PREFIX:
        if n is None:
            raise SubsetSelectionError("mode=video_order_prefix requires n")
        chosen = prefix_indices(n, population)
    elif mode == MODE_SEEDED_RANDOM:
        if n is None:
            raise SubsetSelectionError("mode=seeded_random requires n")
        chosen = seeded_random_indices(n, population, _require_seed(seed, mode))
    elif mode == MODE_STRATIFIED:
        if n is None:
            raise SubsetSelectionError("mode=stratified_blocks requires n")
        chosen = stratified_indices(
            n, population, _require_seed(seed, mode), block_count=block_count
        )
        params["block_count"] = min(block_count, population)
    elif mode == MODE_QUANTILE_STRATIFIED:
        if n is None:
            raise SubsetSelectionError("mode=quantile_stratified requires n")
        if rank_by is None:
            raise SubsetSelectionError(
                "mode=quantile_stratified requires rank_by=<population-length difficulty>"
            )
        chosen = quantile_stratified_indices(n, population, rank_by, quantiles=quantiles)
        params["quantiles"] = list(quantiles)
    else:  # pragma: no cover - ALL_MODES is exhaustive
        raise SubsetSelectionError(f"unhandled mode {mode!r}")

    ratios: tuple[GoverningRatio, ...] = ()
    if governing is not None or mode != MODE_FULL:
        ratios = (
            governing_ratio(
                chosen,
                governing,
                name=governing_name,
                seed=0 if seed is None else int(seed),
                n_bootstrap=n_bootstrap,
                alpha=alpha,
            ),
        )
    return Selection(
        indices=chosen,
        mode=mode,
        population=population,
        seed=seed,
        params=params,
        ratios=ratios,
    )


def assert_population_matched(selection: Selection, *, what: str = "verdict") -> None:
    """Raise unless every governing ratio ran and matched.

    Use at the boundary where a subset-scoped number is about to be promoted to a
    population claim. The message states the ratio, because the number is the
    argument.
    """
    if selection.population_matched:
        return
    unmatched = selection.unmatched_ratios()
    scope = f"mode={selection.mode}, n={selection.n}/{selection.population}"

    # "We could not check" and "we checked and it differs" are DIFFERENT failures
    # and must not share a message. Collapsing them is the vacuity genus wearing
    # an error string: a reader told "not exchangeable" will go hunting for a bias
    # that was never measured. (Caught by this module's own round-1 tests.)
    vacuous = [r for r in unmatched if r.verdict in {VERDICT_VACUOUS_NO_TABLE, VERDICT_VACUOUS_EMPTY}]
    if not selection.ratios or len(vacuous) == len(unmatched):
        reasons = "; ".join(r.summary() for r in vacuous) or "no governing ratio was computed"
        raise PopulationMismatchError(
            f"{what}: the population check did NOT RUN, so this subset ({scope}) is "
            f"INSTANCE-scoped -- not shown to differ, and not shown to match. {reasons}. "
            "Pass governing=<population-length sequence> to check it."
        )
    detail = " ; ".join(r.summary() for r in unmatched)
    raise PopulationMismatchError(
        f"{what}: subset ({scope}) is NOT EXCHANGEABLE with the population -- {detail}"
    )


__all__ = [
    "ALL_MODES",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_STRATIFIED_BLOCKS",
    "MODE_EXPLICIT",
    "MODE_FULL",
    "MODE_PREFIX",
    "MODE_SEEDED_RANDOM",
    "MODE_STRATIFIED",
    "MODE_STRIDED",
    "NON_MATCHED_VERDICTS",
    "SCHEMA",
    "UNREPRESENTATIVE_MODES",
    "VERDICT_DEGENERATE_POPULATION",
    "VERDICT_DIFFERENT_POPULATION",
    "VERDICT_MATCHED",
    "VERDICT_VACUOUS_EMPTY",
    "VERDICT_VACUOUS_NO_TABLE",
    "GoverningRatio",
    "PopulationMismatchError",
    "Selection",
    "SubsetSelectionError",
    "assert_population_matched",
    "governing_ratio",
    "prefix_indices",
    "seeded_random_indices",
    "select",
    "stratified_indices",
    "strided_indices",
]
