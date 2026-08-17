# SPDX-License-Identifier: MIT
"""ddm_me1 - the live corrector's law, extended with CAUSAL SPATIAL context.

DERIVATION (first principles from our own shipped law, not from anyone's code).

``ddm_rr4_free_corrector_v2`` conditions its odds multiplier on

    (predicted_class, surprise_bin, agree_prev1, agree_prev2, run, boundary)

Every one of those six terms is either a property of the current prediction or a
TEMPORAL memory of the same pixel in earlier frames. There is no SPATIAL term at
all: the law never looks at the already-decoded neighbours in the frame it is
currently coding. For a segmentation-mask field -- a piecewise-constant field whose
error concentrates on class boundaries -- the causal spatial neighbourhood is the
single most informative context available, and we are not using it.

It is legally available. The decode order is a 190-group diagonal wavefront, and
MEASURED on the shipped ``group_index``: 98.63% of left neighbours and 98.69% of up
neighbours sit in a STRICTLY earlier group. This module uses a neighbour ONLY when
its group is strictly earlier, so encoder and decoder see identical state; where a
neighbour is not yet decoded the feature takes a dedicated "unavailable" level,
exactly as the shipped law does for frame 0 via ``have_prev``.

WHAT IS UNCHANGED. The transport form is untouched -- the same Krichevsky-Trofimov
smoothed odds multiplier, the same MIN_COUNT gate, the same symmetric odds clamp,
the same rank-one (argmax vs complement) split with the inherited relative law
preserved inside the complement. Only the CONTEXT INDEX changes. Distortion is
therefore unchanged by construction: the decoded token field is bit-identical.

RULE 118. Like the law it extends, this ships as generic decoder CODE in
``inflate.py`` and transmits nothing: the statistics are estimated online from
already-decoded symbols by a fixed generic rule. No table, no weights, no
video-derived constant. Counted archive bytes added: ZERO.

The added feature is deliberately the crudest possible spatial term -- the count of
causally-available immediate neighbours (left, up) whose decoded class equals the
current prediction, plus one level for "neither available". If a 5-level crude
feature does not pay, a richer one is unlikely to, and the honest negative is
cheap. Levels::

    0  neither neighbour causally available
    1  available neighbours, none agree with the prediction
    2  exactly one agrees
    3  two agree
    4  available but disagreeing in a mixed way (reserved; see _spatial_level)
"""
from __future__ import annotations

import numpy as np

from experiments.ddm_rr4_free_corrector_v2 import (
    BOUNDARY_LEVELS,
    NUM_CLASSES,
    RUN_LEVELS,
    U_BINS,
    FreeCorrector,
    GroupState,
)

__all__ = ["SPATIAL_CONTEXT_SIZE", "SPATIAL_LEVELS", "SpatialContextCorrector"]

HEIGHT = 384
WIDTH = 512

SPATIAL_LEVELS = 5
"""Number of levels the causal-spatial feature can take (see module docstring)."""

SPATIAL_CONTEXT_SIZE = (
    NUM_CLASSES * U_BINS * 2 * 2 * RUN_LEVELS * BOUNDARY_LEVELS * SPATIAL_LEVELS
)


class SpatialContextCorrector(FreeCorrector):
    """The shipped law with one extra causal-spatial context factor.

    Inherits ``odds_multiplier`` / ``coding_row`` / ``observe`` unchanged -- they read
    ``state.context`` and the count tables, both of which are simply larger here.
    """

    def __init__(self, plane: int) -> None:
        super().__init__(plane)
        if plane != HEIGHT * WIDTH:
            raise ValueError("spatial context assumes the shipped 384x512 plane")
        self.counts = np.zeros(SPATIAL_CONTEXT_SIZE, dtype=np.int64)
        self.hits = np.zeros(SPATIAL_CONTEXT_SIZE, dtype=np.int64)
        self.phat_q = np.zeros(SPATIAL_CONTEXT_SIZE, dtype=np.int64)
        self.current = np.zeros(plane, dtype=np.uint8)
        self.known = np.zeros(plane, dtype=bool)
        self._pending_flat: np.ndarray | None = None

    def begin_frame(self, boundary_flat: np.ndarray) -> None:
        super().begin_frame(boundary_flat)
        self.known[:] = False
        self.current[:] = 0

    def _spatial_level(self, flat: np.ndarray, base_class: np.ndarray) -> np.ndarray:
        """Causal neighbour agreement level for each position in the group."""
        x = flat % WIDTH
        y = flat // WIDTH

        left = flat - 1
        has_left = (x > 0) & self.known[np.maximum(left, 0)]
        up = flat - WIDTH
        has_up = (y > 0) & self.known[np.maximum(up, 0)]

        left_class = self.current[np.maximum(left, 0)].astype(np.int64)
        up_class = self.current[np.maximum(up, 0)].astype(np.int64)

        agree_left = has_left & (left_class == base_class)
        agree_up = has_up & (up_class == base_class)

        available = has_left.astype(np.int64) + has_up.astype(np.int64)
        agreeing = agree_left.astype(np.int64) + agree_up.astype(np.int64)

        level = np.where(available == 0, 0, agreeing + 1)
        return level.astype(np.int64)

    def group_state(
        self,
        probability: np.ndarray,
        predicted: np.ndarray,
        positions: np.ndarray,
    ) -> GroupState:
        state = super().group_state(probability, predicted, positions)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)
        level = self._spatial_level(flat, base_class)
        state.context = state.context * SPATIAL_LEVELS + level
        self._pending_flat = flat
        return state

    def observe(self, state: GroupState, symbols: np.ndarray) -> None:
        """Fold the group in, then PUBLISH it so later groups may use it as context.

        Publishing here (rather than in a separate call) keeps the driving API
        byte-identical to the shipped corrector, so the encoder and the receiver
        need no new call and cannot desynchronise by forgetting one.
        """
        super().observe(state, symbols)
        flat = self._pending_flat
        if flat is None:
            raise RuntimeError("observe() called without a matching group_state()")
        self.current[flat] = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        self.known[flat] = True
        self._pending_flat = None
