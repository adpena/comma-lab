# SPDX-License-Identifier: MIT
"""Tie-aware factor-2 uint8 preimage selection and its optimality certificate.

The disjoint factor-2 lattice (``uint8_lattice_feasibility.DisjointResizeOperator``)
maps one exact ``uint8`` scorer plane ``Y`` to *many* in-box integer camera
preimages: every 2x2 block ``sum(c_ij x_ij) = y*D`` has, for interior ``y``,
tens to hundreds of exact solutions (measured 35-931 per block).  Because the
supports are disjoint, a per-block choice assembles a valid whole-frame preimage
whose bilinear resize equals, block by block, the chosen candidate's resize.

A *tie-aware* policy would, among those exact ties, pick the camera preimage
whose native resize ``A_fp32(X)`` best reproduces the intended plane ``Y`` (or,
more ambitiously, the unrounded scorer reference).  This module implements that
policy AND the certificate that makes it a no-op on the contest factor-2 spine.

Measured fact (see ``tools/measure_tie_aware_preimage_ab.py`` and the equation
``factor2_canonical_preimage_fp32_exact_v1``): the **canonical support-fill**
preimage (every camera tap in a block set to that block's target byte ``y``)
already reproduces ``Y`` with **zero** native-fp32 resize error, because a
locally constant camera region resizes to that constant exactly (bilinear
weights sum to 1.0 in fp32 for the exact integer half-pixel geometry).  Zero is
the global minimum of ``|A_fp32(X) - Y|``, so no tie member can beat canonical.

Consequence for the vehicle: on the exact-plane spine the whole receiver-side
preimage contributes 0 distortion; the officially-scored 0.047 S is
plane-quantization (``Y = round(exact_resize(gt))`` vs the unrounded reference),
which no 0-byte preimage policy can recover.  The selector below therefore
returns canonical with an ``optimal_certificate`` and never searches on that
geometry.  The search path exists for generality (a hypothetical operator where
canonical leaves residual) and is exercised by the tests.

Nothing here changes any archive byte: the stored payload is the plane ``Y``;
only the scorer-free camera preimage the receiver reconstructs is at issue, and
every returned frame is verified to carry the exact ``A_num(X) = D*Y`` numerators
(byte-identity of the scorer plane preserved by construction).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import gcd

import numpy as np

from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    Uint8LatticeError,
    realize_factor2_uint8_scorer_plane,
)

# A resize oracle maps a uint8 camera frame (H,W,C) to its scorer plane (h,w,C)
# as real values.  The authority oracle is the exact frozen path
# (torch fp32 ``F.interpolate(..., bilinear)``); callers inject it.  The module
# never assumes a particular float model — it only compares oracle output to Y.
ResizeOracle = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CanonicalPreimageResidual:
    """Native-resize reproduction error of the canonical support-fill preimage."""

    max_abs: float
    mean_abs: float
    nonzero_values: int
    total_values: int

    @property
    def is_fp32_exact(self) -> bool:
        return self.nonzero_values == 0 and self.max_abs == 0.0


@dataclass(frozen=True)
class TieAwarePreimageResult:
    """Outcome of tie-aware preimage selection for one exact uint8 plane."""

    frame: np.ndarray
    optimal_certificate: bool
    canonical_residual_max: float
    blocks_searched: int
    blocks_improved: int
    numerator_exact: bool

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(np.asarray(self.frame))
        object.__setattr__(
            self,
            "frame",
            np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape),
        )


def canonical_preimage_fp32_residual(
    operator: DisjointResizeOperator,
    target: np.ndarray,
    resize_oracle: ResizeOracle,
) -> CanonicalPreimageResidual:
    """Measure ``|resize_oracle(canonical_preimage(Y)) - Y|`` block by block.

    ``target`` is the exact uint8 scorer plane ``Y``.  The canonical support-fill
    preimage is realized, passed through the injected resize oracle, and compared
    to ``Y``.  On the contest factor-2 geometry with a native fp32 bilinear oracle
    this is identically zero (the module's central measured fact).
    """

    if not isinstance(operator, DisjointResizeOperator):
        raise Uint8LatticeError("residual diagnostic requires DisjointResizeOperator")
    y = _target_uint8(target, operator)
    frame = realize_factor2_uint8_scorer_plane(operator, y)
    resized = np.asarray(resize_oracle(frame), dtype=np.float64)
    yf = y.astype(np.float64)
    if resized.shape != yf.shape:
        raise Uint8LatticeError(
            "resize oracle output shape does not match the scorer plane"
        )
    diff = np.abs(resized - yf)
    return CanonicalPreimageResidual(
        max_abs=float(diff.max(initial=0.0)),
        mean_abs=float(diff.mean()) if diff.size else 0.0,
        nonzero_values=int(np.count_nonzero(diff)),
        total_values=int(diff.size),
    )


def enumerate_block_preimages(
    row_numerators: tuple[int, int],
    col_numerators: tuple[int, int],
    y: int,
    *,
    max_candidates: int | None = None,
) -> np.ndarray:
    """Return in-box exact preimages of one 2x2 factor-2 block.

    Solves ``c00 x00 + c01 x01 + c10 x10 + c11 x11 = y*D`` over
    ``x in [0,255]^4`` with ``c = outer(row_numerators, col_numerators)`` and
    ``D = sum(row_numerators)*sum(col_numerators)``.  Returns an ``(K,4)`` int
    array of ``[x00,x01,x10,x11]`` rows.  The canonical all-``y`` member is always
    emitted first so a bounded ``max_candidates`` truncation never drops it.
    """

    a0, a1 = (int(row_numerators[0]), int(row_numerators[1]))
    b0, b1 = (int(col_numerators[0]), int(col_numerators[1]))
    if min(a0, a1, b0, b1) < 1:
        raise Uint8LatticeError("factor-2 block numerators must be positive")
    if not 0 <= int(y) <= 255:
        raise Uint8LatticeError("block target byte must be in [0,255]")
    y = int(y)
    c00, c01, c10, c11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
    d = (a0 + a1) * (b0 + b1)
    total = y * d
    if max_candidates is not None and int(max_candidates) < 1:
        raise Uint8LatticeError("max_candidates must be a positive integer")
    box = np.arange(256, dtype=np.int64)
    # Canonical all-y member first so a bounded truncation never drops it.
    solutions: list[tuple[int, int, int, int]] = [(y, y, y, y)]
    # Enumerate (x00, x01); the residual fixes a 1-constraint 2-var tail solved
    # in closed form over x10 with x11 determined and range/divisibility checked.
    g_tail = gcd(c10, c11)
    for x00 in range(256):
        base = total - c00 * x00
        rem = base - c01 * box  # shape (256,) residual for c10 x10 + c11 x11
        feasible = (rem >= 0) & (rem <= 255 * (c10 + c11)) & (rem % g_tail == 0)
        for x01 in np.nonzero(feasible)[0]:
            r = int(rem[x01])
            # c10 x10 + c11 x11 = r ; iterate x10, x11 determined.
            num = r - c10 * box
            ok = (num >= 0) & (num % c11 == 0)
            x11 = num // c11
            ok &= x11 <= 255
            for x10 in np.nonzero(ok)[0]:
                cand = (x00, int(x01), int(x10), int(x11[x10]))
                if cand == (y, y, y, y):
                    continue  # already emitted first
                solutions.append(cand)
                if max_candidates is not None and len(solutions) >= int(max_candidates):
                    return np.asarray(solutions, dtype=np.int64)
    return np.asarray(solutions, dtype=np.int64)


def select_tie_aware_factor2_uint8(
    operator: DisjointResizeOperator,
    target: np.ndarray,
    resize_oracle: ResizeOracle,
    *,
    search_when_residual_positive: bool = True,
    max_candidates_per_block: int = 512,
) -> TieAwarePreimageResult:
    """Select the fp32-closest exact camera preimage for an exact uint8 plane.

    Fast path (the contest factor-2 spine): if the canonical support-fill already
    resizes to ``Y`` with zero error, it is the global minimizer of
    ``|A_fp32(X) - Y|`` and is returned with ``optimal_certificate=True`` — no
    tie search is needed or possible-to-beat.

    Search path (general operators where canonical leaves residual): for each
    block with nonzero canonical residual, enumerate the exact tie members, resize
    a per-block candidate frame family through the oracle, and keep the member
    that minimizes ``|A_fp32 - Y|`` at that block.  Disjoint supports make the
    per-block assembly a valid whole-frame preimage.

    The returned frame always carries the exact ``A_num(X) = D*Y`` numerators
    (verified), so the scorer plane — and therefore every archive byte — is
    preserved.
    """

    if not isinstance(operator, DisjointResizeOperator):
        raise Uint8LatticeError("tie-aware selection requires DisjointResizeOperator")
    y = _target_uint8(target, operator)
    canonical = realize_factor2_uint8_scorer_plane(operator, y)
    residual = canonical_preimage_fp32_residual(operator, y, resize_oracle)

    if residual.is_fp32_exact or not search_when_residual_positive:
        verification = operator.verify_factor2_uint8(canonical, y)
        return TieAwarePreimageResult(
            frame=canonical,
            optimal_certificate=bool(residual.is_fp32_exact),
            canonical_residual_max=residual.max_abs,
            blocks_searched=0,
            blocks_improved=0,
            numerator_exact=bool(verification.numerator_exact),
        )

    frame, searched, improved = _search_ties(
        operator, y, canonical, resize_oracle, int(max_candidates_per_block)
    )
    verification = operator.verify_factor2_uint8(frame, y)
    if not verification.numerator_exact:
        raise Uint8LatticeError(
            "tie-aware selection produced a non-exact preimage; refusing"
        )
    return TieAwarePreimageResult(
        frame=frame,
        optimal_certificate=False,
        canonical_residual_max=residual.max_abs,
        blocks_searched=searched,
        blocks_improved=improved,
        numerator_exact=True,
    )


def _search_ties(
    operator: DisjointResizeOperator,
    y: np.ndarray,
    canonical: np.ndarray,
    resize_oracle: ResizeOracle,
    max_candidates_per_block: int,
) -> tuple[np.ndarray, int, int]:
    """Per-block tie search over blocks where canonical leaves fp32 residual."""

    row_num = [tuple(int(v) for v in s.numerators) for s in operator.row_supports]
    col_num = [tuple(int(v) for v in s.numerators) for s in operator.col_supports]
    row_idx = [tuple(int(v) for v in s.indices) for s in operator.row_supports]
    col_idx = [tuple(int(v) for v in s.indices) for s in operator.col_supports]
    if any(len(r) != 2 for r in row_num) or any(len(c) != 2 for c in col_num):
        raise Uint8LatticeError("tie search requires a 2x2 factor-2 operator")

    base_resized = np.asarray(resize_oracle(canonical), dtype=np.float64)
    yf = y.astype(np.float64)
    block_res = np.abs(base_resized - yf)  # (h,w,C)
    out = np.array(canonical, dtype=np.uint8, copy=True)
    channels = y.shape[2]
    searched = 0
    improved = 0
    for oi in range(y.shape[0]):
        for oj in range(y.shape[1]):
            for ch in range(channels):
                if block_res[oi, oj, ch] <= 0.0:
                    continue
                searched += 1
                best = _best_tie_for_block(
                    operator,
                    canonical,
                    row_idx[oi],
                    col_idx[oj],
                    row_num[oi],
                    col_num[oj],
                    int(y[oi, oj, ch]),
                    ch,
                    float(base_resized[oi, oj, ch]),
                    resize_oracle,
                    max_candidates_per_block,
                )
                if best is not None:
                    (x00, x01, x10, x11) = best
                    out[row_idx[oi][0], col_idx[oj][0], ch] = x00
                    out[row_idx[oi][0], col_idx[oj][1], ch] = x01
                    out[row_idx[oi][1], col_idx[oj][0], ch] = x10
                    out[row_idx[oi][1], col_idx[oj][1], ch] = x11
                    improved += 1
    return out, searched, improved


def _best_tie_for_block(
    operator: DisjointResizeOperator,
    canonical: np.ndarray,
    row_idx: Sequence[int],
    col_idx: Sequence[int],
    row_num: tuple[int, int],
    col_num: tuple[int, int],
    y: int,
    ch: int,
    canonical_error: float,
    resize_oracle: ResizeOracle,
    max_candidates: int,
) -> tuple[int, int, int, int] | None:
    """Return the block tap assignment minimizing fp32 error, or None if none beats canonical."""

    ties = enumerate_block_preimages(row_num, col_num, y, max_candidates=max_candidates)
    best_error = canonical_error
    best: tuple[int, int, int, int] | None = None
    probe = np.array(canonical, dtype=np.uint8, copy=True)
    for row in ties:
        x00, x01, x10, x11 = (int(v) for v in row)
        probe[row_idx[0], col_idx[0], ch] = x00
        probe[row_idx[0], col_idx[1], ch] = x01
        probe[row_idx[1], col_idx[0], ch] = x10
        probe[row_idx[1], col_idx[1], ch] = x11
        resized = np.asarray(resize_oracle(probe), dtype=np.float64)
        err = abs(float(resized[_block_out_index(operator, row_idx, col_idx)][ch]) - float(y))
        if err < best_error - 1e-15:
            best_error = err
            best = (x00, x01, x10, x11)
    # restore probe for cleanliness is unnecessary (local copy).
    return best


def _block_out_index(
    operator: DisjointResizeOperator,
    row_idx: Sequence[int],
    col_idx: Sequence[int],
) -> tuple[int, int]:
    """Map a block's camera row/col taps to its scorer-plane output index."""

    oi = next(
        i for i, s in enumerate(operator.row_supports) if tuple(s.indices) == tuple(row_idx)
    )
    oj = next(
        j for j, s in enumerate(operator.col_supports) if tuple(s.indices) == tuple(col_idx)
    )
    return oi, oj


def _target_uint8(target: np.ndarray, operator: DisjointResizeOperator) -> np.ndarray:
    raw = np.asarray(target)
    if raw.dtype != np.uint8:
        raise Uint8LatticeError("tie-aware target must be an exact uint8 scorer plane")
    if raw.ndim == 2:
        raw = raw[:, :, None]
    if raw.ndim != 3 or raw.shape[:2] != (operator.scorer_h, operator.scorer_w):
        raise Uint8LatticeError("tie-aware target must match the operator scorer geometry")
    return np.ascontiguousarray(raw)
