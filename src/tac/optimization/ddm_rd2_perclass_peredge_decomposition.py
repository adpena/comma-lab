# SPDX-License-Identifier: MIT
"""ddm_rd2 -- per-CLASS **and per-EDGE** decomposition of the SegNet argmax residual.

Why this module exists
----------------------
``ddm_fl1`` decomposed the GT-flicker statistic per CLASS, and ``ddm_xp1`` decomposed the
ep641 residual per CLASS. Both stored **marginals only**: xp1's cached chunks carry
``cls_gt (P,5)`` / ``cls_base (P,5)`` and no 5x5 joint, so the per-EDGE view cannot be
recovered from either cache -- it has never been computed at ANY endpoint. This module
computes it, and computes the per-class view in the SAME pass with fl1's exact convention
so the two are guaranteed to use one partition.

Operator directive 2026-08-02, verbatim: *"there are no floors. It's all a matter of proper
deep math and engineering."* Accordingly this module NEVER reports a ratio as
distance-to-a-bound. ``ExhaustionIndicator`` names what a ratio >= 1 actually licenses:
**the current representation of that class/edge has run out**, not that the residual is
irreducible. The reference statistic is FORMULATION-scoped (fl1 records it pierced by
phase-faithful PR130 at 2.966e-4 and by ep641 in aggregate), and the field names say so.

Three views, one pass
---------------------
1. ``per_class``   -- residual charged by GT label ``lstars[t]`` (xp1 / fl1 convention).
2. ``per_edge``    -- the full 5x5 JOINT (GT class -> predicted class). Row-sum minus the
   diagonal reproduces (1) exactly; that identity is asserted, not assumed.
3. ``boundary``    -- the GEOMETRIC edge view: mass on/off the spatial band around each
   unordered class-pair boundary in the GT partition. Answers "which boundaries carry it".

Authority
---------
``[macOS-CPU advisory]`` derived-from-cached-argmax. ``score_claim=false``. No scorer
forward is performed anywhere in this module: it consumes argmax label fields that some
other, slot-holding arm produced. The flicker path needs only cached GT and is $0.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CLASS_ORDER",
    "N_CLASSES",
    "EdgeDecomposition",
    "ExhaustionIndicator",
    "boundary_band_masses",
    "confusion_from_labels",
    "edge_rows",
    "exhaustion_table",
    "flicker_confusion",
    "iter_chunks",
    "per_class_from_confusion",
    "residual_confusion",
]

# comma10k CANONICAL order. NEVER luma-derived -- CLAUDE.md records the luma sort as a
# measured 3x-repeated error. Verified live in fl1 against cached lstars (range [0,4]).
CLASS_ORDER: tuple[str, ...] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_CLASSES = len(CLASS_ORDER)

_S_COEFF = 100.0


def _check_labels(a: np.ndarray, name: str) -> np.ndarray:
    if a.ndim != 3:
        raise ValueError(f"{name} must be (P, H, W), got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} is empty; an empty scope is VACUOUS, never a clean pass")
    lo, hi = int(a.min()), int(a.max())
    if lo < 0 or hi >= N_CLASSES:
        raise ValueError(f"{name} labels out of range [0,{N_CLASSES - 1}]: got [{lo},{hi}]")
    return a


def iter_chunks(n: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Half-open [start, stop) spans covering ``range(n)``. Bounds peak RSS."""
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


def confusion_from_labels(
    gt: np.ndarray, pred: np.ndarray, *, chunk: int = 64
) -> np.ndarray:
    """5x5 JOINT count matrix ``C[c][c'] = #{gt == c & pred == c'}``.

    This is the object xp1 discarded. ``C.sum(1)`` is xp1's ``cls_gt``; ``C.sum(0)`` is
    ``cls_base``; the two marginals do not determine ``C``, which is exactly why the
    per-edge view could not be back-derived from the cache.
    """
    gt = _check_labels(gt, "gt")
    pred = _check_labels(pred, "pred")
    if gt.shape != pred.shape:
        raise ValueError(f"gt {gt.shape} and pred {pred.shape} must match")
    out = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for lo, hi in iter_chunks(gt.shape[0], chunk):
        g = gt[lo:hi].astype(np.int64, copy=False).ravel()
        p = pred[lo:hi].astype(np.int64, copy=False).ravel()
        out += np.bincount(
            g * N_CLASSES + p, minlength=N_CLASSES * N_CLASSES
        ).reshape(N_CLASSES, N_CLASSES)
    return out


def per_class_from_confusion(conf: np.ndarray, denom_px: int) -> np.ndarray:
    """Per-class residual in S-units, charged by GT class (row-sum minus diagonal)."""
    if conf.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"conf must be {(N_CLASSES, N_CLASSES)}, got {conf.shape}")
    if denom_px <= 0:
        raise ValueError("denom_px must be > 0")
    off = conf.sum(axis=1) - np.diag(conf)
    return _S_COEFF * off.astype(np.float64) / float(denom_px)


@dataclass(frozen=True)
class EdgeDecomposition:
    """One measured decomposition: the joint, both marginal views, and the denominator."""

    confusion: np.ndarray
    denom_px: int
    n_pairs: int
    convention: str
    axis_tag: str = "[macOS-CPU advisory]"
    score_claim: bool = False

    def __post_init__(self) -> None:
        if self.confusion.shape != (N_CLASSES, N_CLASSES):
            raise ValueError("confusion must be 5x5")
        if self.denom_px <= 0 or self.n_pairs <= 0:
            raise ValueError("denom_px and n_pairs must be > 0")

    @property
    def per_class_S(self) -> np.ndarray:
        return per_class_from_confusion(self.confusion, self.denom_px)

    @property
    def total_S(self) -> float:
        return float(self.per_class_S.sum())

    @property
    def per_edge_S(self) -> np.ndarray:
        """S-units per ORDERED edge (gt -> pred). Diagonal forced to 0 (agreement)."""
        m = _S_COEFF * self.confusion.astype(np.float64) / float(self.denom_px)
        np.fill_diagonal(m, 0.0)
        return m

    def identity_holds(self, atol: float = 1e-12) -> bool:
        """per-class == per-edge row sums. Asserted, never assumed."""
        return bool(np.allclose(self.per_edge_S.sum(axis=1), self.per_class_S, atol=atol))


def edge_rows(dec: EdgeDecomposition, *, top: int = 0) -> list[dict]:
    """Flatten the ordered-edge matrix to rows, largest first."""
    m = dec.per_edge_S
    rows = [
        {
            "gt_class": CLASS_ORDER[i],
            "pred_class": CLASS_ORDER[j],
            "edge": f"{CLASS_ORDER[i]}->{CLASS_ORDER[j]}",
            "S": float(m[i, j]),
            "px": int(dec.confusion[i, j]),
            "share_of_total": (float(m[i, j]) / dec.total_S) if dec.total_S else 0.0,
        }
        for i in range(N_CLASSES)
        for j in range(N_CLASSES)
        if i != j
    ]
    rows.sort(key=lambda r: -r["S"])
    return rows[:top] if top else rows


def flicker_confusion(
    lstars: np.ndarray,
    *,
    chunk: int = 64,
    neighbour: str = "prev",
    denom: str = "interior",
) -> EdgeDecomposition:
    """Per-EDGE GT-flicker, reproducing fl1's per-CLASS numbers as row sums.

    fl1 definition (verbatim): a spike is a pixel whose scored-frame GT argmax
    ``lstars[t]`` differs from BOTH stride-2 neighbours. fl1 charged it to ``lstars[t]``
    and separately tallied the prev/next marginals; it never took the JOINT
    ``(lstars[t], neighbour_label)`` -- which is the class the label flickers TO, i.e. the
    EDGE the instability lives on.

    ``denom`` -- MEASURED 2026-08-02 by ddm_rd2: fl1's REGISTERED aggregate 0.005318 and
    its published per-class vector are the **/598 interior** convention, not the /600 one
    its prose calls "primary". Reproduced to <5e-5 per class under ``denom="interior"``
    and off by exactly 600/598 under ``denom="all"``. This matters because a residual
    measured over all 600 scored pairs joined against a /598 reference mixes
    denominators; ``"all"`` is the convention that is commensurable with such a residual.
    Both are exposed so the choice is explicit at every call site rather than inherited.

    $0: consumes cached GT only. No scorer forward.
    """
    lstars = _check_labels(lstars, "lstars")
    if neighbour not in ("prev", "next"):
        raise ValueError("neighbour must be 'prev' or 'next'")
    if denom not in ("interior", "all"):
        raise ValueError("denom must be 'interior' (fl1 registered, /598) or 'all' (/600)")
    n, h, w = lstars.shape
    if n < 3:
        raise ValueError("need >= 3 pairs for an interior stride-2 neighbourhood")
    conf = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    # interior scored frames are indices 1..n-2
    for lo, hi in iter_chunks(n - 2, chunk):
        prev = lstars[lo : hi].astype(np.int64, copy=False)
        cur = lstars[lo + 1 : hi + 1].astype(np.int64, copy=False)
        nxt = lstars[lo + 2 : hi + 2].astype(np.int64, copy=False)
        spike = (cur != prev) & (cur != nxt)
        other = prev if neighbour == "prev" else nxt
        c = cur[spike].ravel()
        o = other[spike].ravel()
        if c.size:
            conf += np.bincount(
                c * N_CLASSES + o, minlength=N_CLASSES * N_CLASSES
            ).reshape(N_CLASSES, N_CLASSES)
    n_denom = (n - 2) if denom == "interior" else n
    return EdgeDecomposition(
        confusion=conf,
        denom_px=n_denom * h * w,
        n_pairs=n,
        convention=(
            f"fl1 spike (differs from BOTH stride-2 neighbours), charged by lstars[t]; "
            f"edge target = {neighbour}-neighbour label; denom={denom} (/{n_denom} frames)"
        ),
    )


def residual_confusion(
    lstars: np.ndarray, pstars: np.ndarray, *, chunk: int = 64
) -> EdgeDecomposition:
    """Per-EDGE residual of OUR argmax against GT -- the live-base re-join.

    ``pstars`` is our rendered frame's frozen-CPU-torch SegNet argmax, same shape as
    ``lstars``. This module does NOT produce it (that is a scorer pass); it consumes it.
    """
    conf = confusion_from_labels(lstars, pstars, chunk=chunk)
    n, h, w = lstars.shape
    return EdgeDecomposition(
        confusion=conf,
        denom_px=n * h * w,
        n_pairs=n,
        convention="residual charged by GT class lstars[t] (xp1 _per_class_flip_counts)",
    )


def boundary_band_masses(
    lstars: np.ndarray,
    flip_mask: np.ndarray,
    *,
    chunk: int = 32,
) -> dict:
    """GEOMETRIC per-edge view: flip mass on the spatial band of each class-pair boundary.

    For every pixel we look at its 4-neighbourhood in the GT partition. A pixel of class c
    with a class-c' neighbour lies on the unordered boundary {c, c'}. We report, per
    boundary, how much flip mass sits on it -- and how much sits in class INTERIORS (no
    differing neighbour), which is the part no boundary term can reach.

    This is the view the operator's *"interactions and edges and boundaries"* names, and it
    is orthogonal to the label-space confusion: a Road->Lane confusion can occur on a
    Road|Lane boundary OR in the Road interior, and those demand different representations.
    """
    lstars = _check_labels(lstars, "lstars")
    if flip_mask.shape != lstars.shape:
        raise ValueError(f"flip_mask {flip_mask.shape} must match lstars {lstars.shape}")
    n = lstars.shape[0]
    pair_mass = np.zeros((N_CLASSES, N_CLASSES), np.int64)  # unordered, upper triangle used
    interior_mass = np.zeros(N_CLASSES, np.int64)
    pair_area = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    interior_area = np.zeros(N_CLASSES, np.int64)

    for lo, hi in iter_chunks(n, chunk):
        lab = lstars[lo:hi].astype(np.int8, copy=False)
        fl = flip_mask[lo:hi].astype(bool, copy=False)
        on_any = np.zeros(lab.shape, bool)
        # 4-neighbourhood: one vertical and one horizontal incidence per pixel pair.
        # Slices are written out rather than built by arithmetic -- the earlier
        # tuple-concat-with-ternary form was correct but unreadable, which is how a
        # silent axis bug ships.
        _S = slice(None)
        for a_idx, b_idx in (
            ((_S, slice(None, -1), _S), (_S, slice(1, None), _S)),   # vertical
            ((_S, _S, slice(None, -1)), (_S, _S, slice(1, None))),   # horizontal
        ):
            a, b = lab[a_idx], lab[b_idx]
            diff = a != b
            if not diff.any():
                continue
            lo_c = np.minimum(a[diff], b[diff]).astype(np.int64)
            hi_c = np.maximum(a[diff], b[diff]).astype(np.int64)
            np.add.at(pair_area, (lo_c, hi_c), 1)
            # a pixel is "on" boundary {c,c'} if either side of the incidence flipped
            fa, fb = fl[a_idx], fl[b_idx]
            m = diff & (fa | fb)
            if m.any():
                lo_m = np.minimum(a[m], b[m]).astype(np.int64)
                hi_m = np.maximum(a[m], b[m]).astype(np.int64)
                np.add.at(pair_mass, (lo_m, hi_m), 1)
            on_any[a_idx] |= diff
            on_any[b_idx] |= diff
        inter = ~on_any
        for c in range(N_CLASSES):
            sel = inter & (lab == c)
            interior_area[c] += int(sel.sum())
            interior_mass[c] += int((sel & fl).sum())

    return {
        "boundary_pair_flip_incidences": pair_mass.tolist(),
        "boundary_pair_area_incidences": pair_area.tolist(),
        "interior_flip_px": interior_mass.tolist(),
        "interior_area_px": interior_area.tolist(),
        "class_order": list(CLASS_ORDER),
        "note": (
            "pair_* are 4-neighbour INCIDENCE counts (upper triangle, unordered) and are "
            "NOT disjoint pixel partitions -- a pixel touching two boundaries is counted on "
            "both. interior_* IS a disjoint pixel partition. Compare like with like."
        ),
    }


@dataclass(frozen=True)
class ExhaustionIndicator:
    """Residual vs a FORMULATION-scoped reference statistic.

    Operator 2026-08-02: *"there are no floors."* This type deliberately has no field
    named 'floor' and no 'headroom'. ``ratio >= 1`` licenses exactly one reading:
    **the current representation of this class/edge is exhausted** -- a new representation
    is required, NOT that the residual is irreducible. ``reference_is_a_bound`` is
    hard-wired ``False`` so no consumer can quietly re-import the floor reading.
    """

    name: str
    residual_S: float
    reference_S: float
    reference_scope: str = "FORMULATION (GT-flicker of the smooth-label reference; fl1 records it pierced)"
    reference_is_a_bound: bool = field(default=False, init=False)

    @property
    def ratio(self) -> float:
        return self.residual_S / self.reference_S if self.reference_S else float("inf")

    @property
    def reading(self) -> str:
        if self.reference_S == 0:
            return "REFERENCE_ZERO_UNDEFINED"
        return (
            "REPRESENTATION_EXHAUSTED_NEEDS_NEW_CARRIER"
            if self.ratio >= 1.0
            else "CURRENT_REPRESENTATION_STILL_PAYING"
        )


def exhaustion_table(
    residual: Sequence[float], reference: Sequence[float], names: Sequence[str] = CLASS_ORDER
) -> list[ExhaustionIndicator]:
    if not (len(residual) == len(reference) == len(names)):
        raise ValueError("residual, reference and names must have equal length")
    return [
        ExhaustionIndicator(name=n, residual_S=float(r), reference_S=float(f))
        # strict=True: a silent truncation here would mis-attribute a class's residual
        for n, r, f in zip(names, residual, reference, strict=True)
    ]
