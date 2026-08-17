# SPDX-License-Identifier: MIT
"""ddm_me1 - MIXTURE of context families over the shipped transport law.

WHY A MIXTURE, DERIVED FROM OUR OWN MEASUREMENT.

``ddm_me1_spatial_context_corrector`` added one crude causal-spatial factor to the
shipped joint context and MEASURED **+67.12 B on n600** -- a loss. The mechanism is
plain: refining a joint context MULTIPLIES the bin count (51,200 -> 256,000), and the
statistics per bin thin out faster than the new feature informs. Conditioning is not
free; every extra factor is paid for in learning cost.

The fix is architectural, not a bigger context. Run SEVERAL SMALL context families in
parallel and BLEND their opinions. Each family stays densely populated, so each is
well estimated early, and the blend can still express what no single family sees.
(This is the classic context-mixing result from lossless image/text coding; the same
structural choice is what the public opal_v1 entry describes as "55 families of
causal contexts". The mechanism class is public; this implementation is ours, derived
from our own shipped law and our own measured dilution result above.)

THE BLEND MUST BE PLATFORM-EXACT. ``ddm_rr4_free_corrector_v2`` exists because a
single ``log2``/``exp2`` round trip desynchronised the arithmetic decoder between
macOS-arm64 and Linux-x86_64 -- one float32 ULP moves an RC64 frequency by 128
counts. A geometric-mean blend would reintroduce ``pow`` and re-open exactly that
wound. So the blend here is a COUNT-WEIGHTED ARITHMETIC MEAN IN ODDS SPACE::

    m = (sum_k w_k * m_k) / (sum_k w_k)          w_k = 0 when family k is cold

which uses only +, *, / -- all IEEE-754 correctly rounded, hence bit-identical on
every conforming platform. A cold family contributes nothing rather than dragging
the mixture toward 1.0, so adding a family can never dilute the families that are
already confident. With one family this reduces EXACTLY to the shipped law.

RULE 118: generic decoder code, zero transmitted bytes, statistics estimated online
from already-decoded symbols. Distortion unchanged by construction -- only the
probability handed to the coder moves; the decoded token field is bit-identical.
"""
from __future__ import annotations

import numpy as np

from experiments.ddm_rr4_free_corrector_v2 import (
    BOUNDARY_LEVELS,
    KT_ALPHA,
    MIN_COUNT,
    NUM_CLASSES,
    ODDS_HIGH,
    ODDS_LOW,
    PHAT_SCALE,
    RUN_LEVELS,
    U_BINS,
    FreeCorrector,
    GroupState,
)

__all__ = ["ContextFamily", "MixedContextCorrector", "default_families"]

HEIGHT = 384
WIDTH = 512
SPATIAL_LEVELS = 5
QUALITY_EPS = 1e-6
"""Floor on the quality denominator; keeps the reciprocal finite without a log."""


class ContextFamily:
    """One small context model: an index rule plus its own count tables."""

    __slots__ = ("counts", "err_sum", "hits", "name", "obs", "phat_q", "rule", "size")

    def __init__(self, name: str, size: int, rule) -> None:
        self.name = name
        self.size = int(size)
        self.rule = rule
        self.counts = np.zeros(self.size, dtype=np.int64)
        self.hits = np.zeros(self.size, dtype=np.int64)
        self.phat_q = np.zeros(self.size, dtype=np.int64)
        # Running absolute prediction error, the family's measured QUALITY.
        # Accumulated with +,-,*,/ only: no log, so the blend stays platform-exact.
        self.err_sum = 0.0
        self.obs = 0.0

    def multiplier_and_weight(self, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """This family's odds multiplier and its confidence weight.

        Identical arithmetic to the shipped ``odds_multiplier``. The WEIGHT combines
        two independent things, because MEASUREMENT showed that using only one of
        them fails:

        * LOCAL sufficiency -- a count shrinkage ``n/(n+MIN_COUNT)``, saturating, so a
          bin observed 10x does not speak as loudly as one observed 10,000x.
        * GLOBAL quality -- the reciprocal of the family's running mean absolute
          prediction error.

        The first alone is what the 5-family run used, and it MEASURED +1,398 B: raw
        counts reward CRUDENESS. ``surprise_only`` has 320 bins, so its per-bin counts
        run into the millions and it drowns out the refined family that actually
        carries the signal. Count is confidence in a model's ESTIMATE, never evidence
        that the model is the right one -- so the quality term has to be there too.
        """
        count = self.counts[index].astype(np.float64)
        denominator = count + 2.0 * KT_ALPHA
        hit_numerator = self.hits[index].astype(np.float64) + KT_ALPHA
        hit_denominator = denominator - hit_numerator
        expected = self.phat_q[index].astype(np.float64) / PHAT_SCALE
        exp_numerator = expected + KT_ALPHA
        exp_denominator = denominator - exp_numerator

        multiplier = np.ones(index.shape, dtype=np.float64)
        usable = (
            (self.counts[index] >= MIN_COUNT)
            & (hit_numerator > 0.0)
            & (hit_denominator > 0.0)
            & (exp_numerator > 0.0)
            & (exp_denominator > 0.0)
        )
        multiplier[usable] = (hit_numerator[usable] * exp_denominator[usable]) / (
            hit_denominator[usable] * exp_numerator[usable]
        )
        np.clip(multiplier, ODDS_LOW, ODDS_HIGH, out=multiplier)
        local = count / (count + float(MIN_COUNT))
        quality = 1.0 / (QUALITY_EPS + (self.err_sum / self.obs if self.obs > 0.0 else 1.0))
        weight = np.where(usable, local * quality, 0.0)
        return multiplier, weight

    def observe(self, index: np.ndarray, hit: np.ndarray, p_max_q: np.ndarray) -> None:
        # Score the family BEFORE folding the observation in: its predicted hit rate
        # at this context versus what actually happened. Absolute error keeps the
        # accumulation transcendental-free.
        count = self.counts[index].astype(np.float64)
        predicted_rate = (self.hits[index].astype(np.float64) + KT_ALPHA) / (
            count + 2.0 * KT_ALPHA
        )
        self.err_sum += float(np.abs(hit.astype(np.float64) - predicted_rate).sum())
        self.obs += float(index.size)
        np.add.at(self.counts, index, 1)
        np.add.at(self.hits, index, hit)
        np.add.at(self.phat_q, index, p_max_q)


def default_families() -> list[tuple[str, int, object]]:
    """The family set: the shipped joint context plus small orthogonal views.

    Each rule maps a dict of per-position features to a flat index. Sizes are kept
    small deliberately -- the measured lesson is that density beats refinement.
    """

    def shipped(f):
        head = ((f["cls"] * U_BINS + f["ubin"]) * 2 + f["agree1"]) * 2 + f["agree2"]
        return (head * RUN_LEVELS + f["run"]) * BOUNDARY_LEVELS + f["boundary"]

    def spatial(f):
        return (f["cls"] * SPATIAL_LEVELS + f["spatial"]) * U_BINS + f["ubin"]

    def spatial_boundary(f):
        return (f["cls"] * SPATIAL_LEVELS + f["spatial"]) * BOUNDARY_LEVELS + f["boundary"]

    def surprise_only(f):
        return f["cls"] * U_BINS + f["ubin"]

    def temporal_spatial(f):
        head = (f["cls"] * 2 + f["agree1"]) * 2 + f["agree2"]
        return head * SPATIAL_LEVELS + f["spatial"]

    return [
        ("shipped_joint", NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS, shipped),
        ("spatial_surprise", NUM_CLASSES * SPATIAL_LEVELS * U_BINS, spatial),
        ("spatial_boundary", NUM_CLASSES * SPATIAL_LEVELS * BOUNDARY_LEVELS, spatial_boundary),
        ("surprise_only", NUM_CLASSES * U_BINS, surprise_only),
        ("temporal_spatial", NUM_CLASSES * 2 * 2 * SPATIAL_LEVELS, temporal_spatial),
    ]


class MixedContextCorrector(FreeCorrector):
    """The shipped transport law driven by a blend of context families."""

    def __init__(self, plane: int, families: list | None = None) -> None:
        super().__init__(plane)
        if plane != HEIGHT * WIDTH:
            raise ValueError("mixed context assumes the shipped 384x512 plane")
        specs = families if families is not None else default_families()
        self.families = [ContextFamily(n, s, r) for n, s, r in specs]
        self.current = np.zeros(plane, dtype=np.uint8)
        self.known = np.zeros(plane, dtype=bool)
        self._pending: dict | None = None

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        super().begin_frame(boundary_flat)
        self.known[:] = False
        self.current[:] = 0

    def _spatial_level(self, flat: np.ndarray, base_class: np.ndarray) -> np.ndarray:
        x = flat % WIDTH
        y = flat // WIDTH
        left = np.maximum(flat - 1, 0)
        up = np.maximum(flat - WIDTH, 0)
        has_left = (x > 0) & self.known[left]
        has_up = (y > 0) & self.known[up]
        agree_left = has_left & (self.current[left].astype(np.int64) == base_class)
        agree_up = has_up & (self.current[up].astype(np.int64) == base_class)
        available = has_left.astype(np.int64) + has_up.astype(np.int64)
        agreeing = agree_left.astype(np.int64) + agree_up.astype(np.int64)
        return np.where(available == 0, 0, agreeing + 1).astype(np.int64)

    def group_state(
        self, probability: np.ndarray, predicted: np.ndarray, positions: np.ndarray
    ) -> GroupState:
        state = super().group_state(probability, predicted, positions)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)

        # Recover the sub-features the shipped context packed, so families can
        # recombine them without recomputing the surprise bin.
        packed = state.context
        boundary = packed % BOUNDARY_LEVELS
        rest = packed // BOUNDARY_LEVELS
        run = rest % RUN_LEVELS
        rest //= RUN_LEVELS
        agree2 = rest % 2
        rest //= 2
        agree1 = rest % 2
        rest //= 2
        ubin = rest % U_BINS
        cls = rest // U_BINS

        features = {
            "cls": cls,
            "ubin": ubin,
            "agree1": agree1,
            "agree2": agree2,
            "run": run,
            "boundary": boundary,
            "spatial": self._spatial_level(flat, base_class),
        }
        indices = [fam.rule(features) for fam in self.families]
        self._pending = {"flat": flat, "indices": indices}
        return state

    def odds_multiplier(self, state: GroupState) -> np.ndarray:
        """Count-weighted arithmetic mean of the families' odds multipliers."""
        if self._pending is None:
            raise RuntimeError("odds_multiplier() called without a group_state()")
        indices = self._pending["indices"]
        numerator = np.zeros(state.context.shape, dtype=np.float64)
        denominator = np.zeros(state.context.shape, dtype=np.float64)
        for family, index in zip(self.families, indices, strict=True):
            multiplier, weight = family.multiplier_and_weight(index)
            numerator += weight * multiplier
            denominator += weight
        blended = np.ones(state.context.shape, dtype=np.float64)
        live = denominator > 0.0
        blended[live] = numerator[live] / denominator[live]
        np.clip(blended, ODDS_LOW, ODDS_HIGH, out=blended)
        return blended

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        hit = (decoded == state.arg).astype(np.int64)
        if self._pending is None:
            raise RuntimeError("observe() called without a group_state()")
        for family, index in zip(self.families, self._pending["indices"], strict=True):
            family.observe(index, hit, state.p_max_q)
        flat = self._pending["flat"]
        self.current[flat] = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        self.known[flat] = True
        self._pending = None
