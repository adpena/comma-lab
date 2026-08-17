"""ddm_rr2 - the free decode-time probability corrector, ONE implementation.

This module is the single source of truth for the ``ddm_rr1`` H1 rung.  The
encoder imports it from the repository; the receiver imports a byte-identical
copy shipped as ``runtime/free_corrector.py`` inside the candidate runtime
tree.  ``ddm_rr1`` §4 step 2 demands exactly that: "The two must be one
function used twice, not two implementations, or they will drift."  A sha256
equality check between the two copies is part of the build receipt, so the
demand is enforced rather than asserted.

WHAT IT DOES.  The shipped F26 receiver computes, per causal patch group, a
five-class probability row from the HPAC logits plus the transmitted RCF1
boundary table.  This corrector keeps every one of those rows exactly and
applies a per-context log2-odds shift to the HIT event only - the binary event
"the decoded symbol equals the row's argmax".  The remaining mass is rescaled
proportionally, so the corrected row is still a proper distribution and the
conditional refinement among the four non-argmax classes is untouched:

    P'(argmax) = q
    P'(c)      = (1 - q) * P(c) / (1 - P(argmax))     for c != argmax

WHY IT IS FREE (contest rule 118).  The shift is estimated online from symbols
that have ALREADY been decoded, by a fixed generic rule with fixed constants.
Encoder and decoder run this same code over the same already-decoded symbols
and therefore hold identical tables at every step.  Nothing is transmitted, so
no video-derived content enters ``inflate.py``.  The constants below are first
principles - a Krichevsky-Trofimov estimator, a power-of-two bin width, a cold
context floor, a symmetric odds clamp - and were never swept against the scored
clip.  ``ddm_rr1`` §5.6 states the boundary plainly: sweeping them on video 0
and keeping the argmax turns them into video-derived scalars that must then be
counted in ``archive.zip``.  They are therefore frozen here.

THE CONTEXT (``ddm_rr1`` stage 3, rung H1 - the measured optimum of this
estimator family at 1,598.30 B saved):

    (base argmax class, surprise bin, agrees with t-1, agrees with t-2,
     unchanged-run length saturated at 8, boundary bucket)          = 51,200

Statistics refresh at every one of the 190 causal group boundaries, which is
the granularity the shipped decode loop already runs at.

DISTORTION.  None, by construction.  The corrector changes only the probability
model handed to the arithmetic coder; the decoded token field is bit-identical,
so ``d_seg`` and ``d_pose`` cannot move.

This file must stay dependency-free apart from numpy: it is copied verbatim
into the receiver runtime.
"""

from __future__ import annotations

import numpy as np

# --- generic constants (first principles; never swept against the clip) ------

NUM_CLASSES = 5
U_STEP = 0.5  # width of a surprise bin, in bits of -log2(1 - p_max)
U_BINS = 64  # saturates the surprise axis at 32 bits
RUN_LEVELS = 8  # unchanged-run length, saturated
RUN_CAP = 255  # run counter saturation, so the state stays uint8-sized
BOUNDARY_LEVELS = 5  # the RCF1 boundary bucket alphabet
KT_ALPHA = 0.5  # Krichevsky-Trofimov smoothing
MIN_COUNT = 32.0  # below this a context emits delta = 0, i.e. exactly HPAC
DELTA_CLIP = 4.0  # symmetric log2-odds clamp (a 16x odds shift)
PROB_EPS = 1e-9

CONTEXT_SIZE = NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS


class GroupState:
    """Per-group quantities the corrector derives once and reuses."""

    __slots__ = ("arg", "context", "logit_p", "one_minus", "p_max", "row64")

    def __init__(
        self,
        row64: np.ndarray,
        arg: np.ndarray,
        p_max: np.ndarray,
        one_minus: np.ndarray,
        logit_p: np.ndarray,
        context: np.ndarray,
    ) -> None:
        self.row64 = row64
        self.arg = arg
        self.p_max = p_max
        self.one_minus = one_minus
        self.logit_p = logit_p
        self.context = context


class FreeCorrector:
    """Adaptive hit-event corrector shared by the encoder and the receiver.

    The caller drives it in the shipped decode order::

        corrector.begin_frame(boundary_flat)
        for each causal group:
            state = corrector.group_state(probability, predicted, positions)
            coding_row = corrector.coding_row(state)      # hand to the coder
            corrector.observe(state, symbols)             # after coding
        corrector.end_frame(tokens_flat)
    """

    __slots__ = ("boundary", "counts", "have_prev", "hits", "phat", "plane", "prev1", "prev2", "run")

    def __init__(self, plane: int) -> None:
        self.plane = int(plane)
        self.counts = np.zeros(CONTEXT_SIZE, dtype=np.float64)
        self.hits = np.zeros(CONTEXT_SIZE, dtype=np.float64)
        self.phat = np.zeros(CONTEXT_SIZE, dtype=np.float64)
        self.prev1 = np.zeros(self.plane, dtype=np.uint8)
        self.prev2 = np.zeros(self.plane, dtype=np.uint8)
        self.run = np.zeros(self.plane, dtype=np.int64)
        self.have_prev = False
        self.boundary = np.full(self.plane, BOUNDARY_LEVELS - 1, dtype=np.int64)

    # -- driving ------------------------------------------------------------

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        """Pin this frame's boundary buckets (the receiver already has them)."""
        boundary = np.asarray(boundary_flat, dtype=np.int64).reshape(-1)
        if boundary.size != self.plane:
            raise ValueError("boundary bucket plane size mismatch")
        self.boundary = boundary

    def group_state(
        self,
        probability: np.ndarray,
        predicted: np.ndarray,
        positions: np.ndarray,
    ) -> GroupState:
        """Derive the context and the base odds for one causal group.

        ``probability`` is the receiver's own float32 row, unchanged;
        ``predicted`` is the argmax of the pre-table HPAC logits, which is also
        what selects the RCF1 correction cell; ``positions`` are the group's
        flat plane indices.
        """
        row64 = np.asarray(probability, dtype=np.float32).astype(np.float64)
        if row64.ndim != 2 or row64.shape[1] != NUM_CLASSES:
            raise ValueError("probability rows must have shape [n, 5]")
        index = np.arange(row64.shape[0])
        arg = row64.argmax(axis=1)
        p_max = row64[index, arg]
        one_minus = np.maximum(1.0 - p_max, PROB_EPS)
        logit_p = np.log2(np.maximum(p_max, PROB_EPS) / one_minus)

        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        if base_class.size != row64.shape[0] or flat.size != row64.shape[0]:
            raise ValueError("group predicted/positions length mismatch")

        surprise = -np.log2(one_minus)
        ubin = np.clip((surprise / U_STEP).astype(np.int64), 0, U_BINS - 1)
        if self.have_prev:
            agree1 = (self.prev1[flat].astype(np.int64) == base_class).astype(np.int64)
            agree2 = (self.prev2[flat].astype(np.int64) == base_class).astype(np.int64)
        else:
            agree1 = np.zeros(flat.size, dtype=np.int64)
            agree2 = np.zeros(flat.size, dtype=np.int64)
        run = np.minimum(self.run[flat], RUN_LEVELS - 1)

        head = ((base_class * U_BINS + ubin) * 2 + agree1) * 2 + agree2
        context = (head * RUN_LEVELS + run) * BOUNDARY_LEVELS + self.boundary[flat]
        return GroupState(row64, arg, p_max, one_minus, logit_p, context)

    def delta(self, state: GroupState) -> np.ndarray:
        """Per-position log2-odds shift from strictly already-decoded symbols."""
        context = state.context
        count = self.counts[context]
        denominator = count + 2.0 * KT_ALPHA
        empirical = np.clip((self.hits[context] + KT_ALPHA) / denominator, PROB_EPS, 1.0 - PROB_EPS)
        expected = np.clip((self.phat[context] + KT_ALPHA) / denominator, PROB_EPS, 1.0 - PROB_EPS)
        shift = np.log2(empirical / (1.0 - empirical)) - np.log2(expected / (1.0 - expected))
        np.clip(shift, -DELTA_CLIP, DELTA_CLIP, out=shift)
        shift[count < MIN_COUNT] = 0.0
        return shift

    def coding_row(self, state: GroupState) -> np.ndarray:
        """The corrected float32 probability row to hand to the RC64 coder."""
        q = np.clip(
            1.0 / (1.0 + np.exp2(-(state.logit_p + self.delta(state)))),
            PROB_EPS,
            1.0 - PROB_EPS,
        )
        row = state.row64 * ((1.0 - q) / state.one_minus)[:, None]
        row[np.arange(row.shape[0]), state.arg] = q
        return row.astype(np.float32)

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        """Fold this group's decoded symbols into the statistics."""
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        if decoded.size != state.context.size:
            raise ValueError("decoded symbol count does not match the group")
        hit = (decoded == state.arg).astype(np.float64)
        np.add.at(self.counts, state.context, 1.0)
        np.add.at(self.hits, state.context, hit)
        np.add.at(self.phat, state.context, state.p_max)

    def end_frame(self, tokens_flat: np.ndarray) -> None:
        """Advance the per-pixel temporal memory once the frame is complete."""
        current = np.asarray(tokens_flat, dtype=np.uint8).reshape(-1)
        if current.size != self.plane:
            raise ValueError("token plane size mismatch")
        if self.have_prev:
            self.run = np.where(current == self.prev1, np.minimum(self.run + 1, RUN_CAP), 0)
            self.prev2 = self.prev1
        self.prev1 = current.copy()
        self.have_prev = True

    # -- crash resumability -------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray | bool]:
        return {
            "counts": self.counts,
            "hits": self.hits,
            "phat": self.phat,
            "prev1": self.prev1,
            "prev2": self.prev2,
            "run": self.run,
            "have_prev": np.array([self.have_prev], dtype=np.bool_),
        }

    def load_state_dict(self, state: dict) -> None:
        self.counts = np.asarray(state["counts"], dtype=np.float64).copy()
        self.hits = np.asarray(state["hits"], dtype=np.float64).copy()
        self.phat = np.asarray(state["phat"], dtype=np.float64).copy()
        self.prev1 = np.asarray(state["prev1"], dtype=np.uint8).copy()
        self.prev2 = np.asarray(state["prev2"], dtype=np.uint8).copy()
        self.run = np.asarray(state["run"], dtype=np.int64).copy()
        self.have_prev = bool(np.asarray(state["have_prev"]).reshape(-1)[0])
