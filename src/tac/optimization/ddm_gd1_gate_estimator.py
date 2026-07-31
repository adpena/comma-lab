# SPDX-License-Identifier: MIT
"""ddm_gd1 (#817) — unbiased estimators for the A1 realized-gate instrument.

The live gate (``experiments/train_tr1_partition_renderer_mlx.py::resolve_gate_ids``)
is a NON-PROBABILITY sample: a deterministic 4-pair block plus a 32-pair simple random
sample (SRS) without replacement from the remaining 596. The trainer reduces it with an
UNWEIGHTED mean, which over-weights the block by ``(4/36)/(4/600) = 16.67x`` and is
therefore biased for the n600 quantity it stands in for -- with a bias that DRIFTS as
the block's hardness differential drifts (MEASURED, ddm_gc14: -1.10e-6 -> +1.52e-5,
overstating the window_02 descent by 7.2%).

This module is the pure-arithmetic fix and its algebra. It is scorer-free, has no MLX /
torch / heavy imports, and is safe to call from a training gate path.

Estimators
----------
``horvitz_thompson_mean`` -- unbiased for the population mean in expectation over the
SRS draw, at ZERO extra scorer cost (same 36 renders, different weights).

``anchored_mean`` -- HT plus an additive bias correction estimated at the most recent
full-n600 anchor. Removes the residual (fixed-seed) SRS error to first order; carries
the anchor's epoch so staleness is explicit (staleness-is-a-named-confound).

``neyman_stratified_gate`` -- a re-DESIGN (not a re-weighting): optimal allocation of a
fixed sample budget across sensitivity strata. Changing the gate SET breaks comparability
with a running campaign's own history, so this is a campaign-boundary move, pre-registered
rather than hot-swapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GateDesign:
    """The sampling design of a gate set.

    ``block_ids`` are included with probability 1 (a deterministic instrument block).
    ``srs_ids`` are one simple-random-sample-without-replacement draw from the remaining
    ``n_population - n_block`` units.
    """

    n_population: int
    block_ids: tuple[int, ...]
    srs_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.n_population <= 0:
            raise ValueError("n_population must be positive")
        if set(self.block_ids) & set(self.srs_ids):
            raise ValueError("block_ids and srs_ids must be disjoint")
        bad = [i for i in (*self.block_ids, *self.srs_ids) if not 0 <= i < self.n_population]
        if bad:
            raise ValueError(f"ids outside population: {bad[:5]}")
        if not self.srs_ids:
            raise ValueError("srs_ids must be non-empty")

    @property
    def n_block(self) -> int:
        return len(self.block_ids)

    @property
    def n_srs(self) -> int:
        return len(self.srs_ids)

    @property
    def n_gate(self) -> int:
        return self.n_block + self.n_srs

    @property
    def n_offblock(self) -> int:
        return self.n_population - self.n_block

    @property
    def block_overweight_coefficient(self) -> float:
        """The exact multiplier on ``(Xbar_block - Xbar_srs)`` in the unweighted bias."""
        return self.n_block / self.n_gate - self.n_block / self.n_population

    def weights(self) -> tuple[np.ndarray, np.ndarray]:
        """(block_weights, srs_weights) that sum to 1 and are unbiased for the pop mean."""
        wb = np.full(self.n_block, 1.0 / self.n_population, dtype=np.float64)
        ws = np.full(
            self.n_srs, (self.n_offblock / self.n_population) / self.n_srs, dtype=np.float64
        )
        return wb, ws


def _split(design: GateDesign, x_population: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_population, dtype=np.float64)
    if x.shape != (design.n_population,):
        raise ValueError(f"expected shape ({design.n_population},), got {x.shape}")
    block = x[list(design.block_ids)]
    srs = x[list(design.srs_ids)]
    off_mask = np.ones(design.n_population, dtype=bool)
    off_mask[list(design.block_ids)] = False
    return block, srs, x[off_mask]


def horvitz_thompson_mean(design: GateDesign, gate_values: dict[int, float]) -> float:
    """Design-unbiased estimate of the population mean from the 36 gate values only.

    ``gate_values`` maps pair id -> measured per-pair quantity (e.g. realized d_seg).
    This is the drop-in replacement for ``float(np.mean(dsegs))``; it needs no extra
    renders and no scorer calls.
    """
    missing = [i for i in (*design.block_ids, *design.srs_ids) if i not in gate_values]
    if missing:
        raise KeyError(f"gate_values missing ids: {missing[:5]}")
    wb, ws = design.weights()
    block = np.array([gate_values[i] for i in design.block_ids], dtype=np.float64)
    srs = np.array([gate_values[i] for i in design.srs_ids], dtype=np.float64)
    return float(wb @ block + ws @ srs)


def anchored_mean(
    ht_now: float, ht_at_anchor: float, n600_at_anchor: float
) -> float:
    """HT estimate corrected by the residual offset measured at the last n600 anchor.

    The correction ``b = n600_at_anchor - ht_at_anchor`` is the fixed-seed SRS error at
    the anchor. It is only as fresh as the anchor -- callers MUST carry the anchor's
    epoch alongside the value (freshness at consumption).
    """
    return float(ht_now + (n600_at_anchor - ht_at_anchor))


@dataclass(frozen=True)
class BiasDecomposition:
    pop_mean: float
    unweighted: float
    horvitz_thompson: float
    block_mean: float
    srs_mean: float
    offblock_mean: float
    total_error: float
    block_overweight_term: float
    srs_sampling_term: float

    @property
    def removable_fraction(self) -> float:
        """Share of the unweighted estimator's error removed EXACTLY by re-weighting."""
        if self.total_error == 0.0:
            return 0.0
        return abs(self.block_overweight_term) / (
            abs(self.block_overweight_term) + abs(self.srs_sampling_term)
        )


def bias_decomposition(design: GateDesign, x_population: np.ndarray) -> BiasDecomposition:
    """Exact split of (unweighted gate mean - population mean) into its two terms.

    Requires the FULL population vector, so it is an audit/design tool -- not something
    the gate itself can run (if you had the population vector you would not need a gate).
    """
    block, srs, off = _split(design, x_population)
    x = np.asarray(x_population, dtype=np.float64)
    pop_mean = float(x.mean())
    unw = float(np.concatenate([block, srs]).mean())
    wb, ws = design.weights()
    ht = float(wb @ block + ws @ srs)
    term_block = design.block_overweight_coefficient * float(block.mean() - srs.mean())
    term_srs = (design.n_offblock / design.n_population) * float(srs.mean() - off.mean())
    return BiasDecomposition(
        pop_mean=pop_mean,
        unweighted=unw,
        horvitz_thompson=ht,
        block_mean=float(block.mean()),
        srs_mean=float(srs.mean()),
        offblock_mean=float(off.mean()),
        total_error=unw - pop_mean,
        block_overweight_term=term_block,
        srs_sampling_term=term_srs,
    )


def srs_standard_error(design: GateDesign, x_population: np.ndarray) -> float:
    """Finite-population SE of the SRS component's contribution to the pop-mean estimate.

    This is the instrument's own NOISE FLOOR: a window-over-window change smaller than a
    couple of these is not resolvable by the gate at all.
    """
    _, _, off = _split(design, x_population)
    n = design.n_srs
    var = float(off.var(ddof=1))
    fpc = 1.0 - n / design.n_offblock
    se_srs_mean = float(np.sqrt(max(var, 0.0) * fpc / n))
    return (design.n_offblock / design.n_population) * se_srs_mean


def neyman_stratified_gate(
    design: GateDesign, sensitivity: np.ndarray, n_strata: int = 4
) -> dict[str, Any]:
    """Design a stratified replacement for the SRS component (same sample budget).

    Strata are equal-count quantile bins of ``sensitivity`` (a scorer-free per-pair
    proxy). Neyman allocation puts sample where the WITHIN-stratum dispersion is, which
    minimises the variance of the stratified mean at fixed budget. Returns the design
    plus the predicted variance ratio vs SRS -- a DESIGN prediction on the proxy, not a
    measurement on d_seg.
    """
    x = np.asarray(sensitivity, dtype=np.float64)
    off_mask = np.ones(design.n_population, dtype=bool)
    off_mask[list(design.block_ids)] = False
    off_ids = np.flatnonzero(off_mask)
    vals = x[off_ids]
    order = np.argsort(vals, kind="stable")
    bins = np.array_split(order, n_strata)

    sizes = np.array([len(b) for b in bins], dtype=np.float64)
    sds = np.array([float(vals[b].std(ddof=1)) if len(b) > 1 else 0.0 for b in bins])
    wts = sizes * sds
    budget = design.n_srs
    if wts.sum() <= 0:
        alloc = np.full(n_strata, budget // n_strata, dtype=int)
    else:
        raw = budget * wts / wts.sum()
        alloc = np.floor(raw).astype(int)
        alloc = np.maximum(alloc, 1)
        while alloc.sum() > budget:
            alloc[int(np.argmax(alloc))] -= 1
        while alloc.sum() < budget:
            alloc[int(np.argmax(raw - alloc))] += 1

    # Variance of the stratified mean (proportional to) vs SRS, on the proxy.
    n_off = float(len(off_ids))
    v_str = float(
        sum(
            (sizes[k] / n_off) ** 2 * (sds[k] ** 2 / max(alloc[k], 1)) * (1.0 - alloc[k] / sizes[k])
            for k in range(n_strata)
        )
    )
    v_srs = float(vals.var(ddof=1) / budget * (1.0 - budget / n_off))
    rng = np.random.default_rng(0)
    chosen: list[int] = []
    for k, b in enumerate(bins):
        pick = rng.choice(len(b), size=int(alloc[k]), replace=False)
        chosen.extend(int(off_ids[b[p]]) for p in pick)
    return {
        "n_strata": n_strata,
        "stratum_sizes": sizes.astype(int).tolist(),
        "stratum_sds": sds.tolist(),
        "neyman_allocation": alloc.tolist(),
        "predicted_var_ratio_stratified_over_srs": (v_str / v_srs) if v_srs > 0 else None,
        "predicted_se_reduction_pct": (100.0 * (1.0 - float(np.sqrt(v_str / v_srs))))
        if v_srs > 0
        else None,
        "candidate_srs_replacement_ids": sorted(chosen),
        "verdict_scope": "DESIGN prediction on a scorer-free proxy; not a d_seg measurement",
        "comparability_note": (
            "changing the gate SET breaks comparability with a running campaign's own gate "
            "series -- campaign-boundary move, pre-register; re-weighting (HT) does not"
        ),
    }
