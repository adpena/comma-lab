"""Selective-precision engineering for the CoreML/ANE scorer path (ddm_ane2).

``tac.ane_screening`` holds the *screening contract* (who may rank, who must be
confirmed).  This module holds the *precision-placement* algebra that the
``ddm_ane2`` instrument needs and that must be unit-testable without
``coremltools``, a GPU, or an ANE:

* the op-sequence identity that makes a split point reproducible
  (:func:`compute_op_names`, :func:`assert_op_sequence_stable`);
* the split / group / selective-set constructions
  (:func:`split_fp16_names`, :func:`group_ranges`, :func:`selective_fp32_names`);
* the per-axis verdicts against the measured bars
  (:func:`seg_flip_verdict`, :func:`pose_drift_verdict`);
* the realized-hybrid geometry -- band, halo dilation, tile occupancy and the
  crop boxes an fp32 recompute actually has to run
  (:func:`margin_band_mask`, :func:`dilate_bool`, :func:`occupied_tiles`,
  :func:`crop_boxes`, :func:`hybrid_speedup`).

Nothing here reads a score.  Every consumer row stays
``[macOS-CPU/ANE advisory]`` with ``score_claim=false``; the authority for both
scorers remains 1-thread CPU-torch fp32.

``coremltools`` is imported lazily by :func:`selector_from_names` only for its
type hints at call time -- importing this module from the main ``.venv`` (which
carries no ``coremltools``) must never fail.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

__all__ = [
    "MIL_CONST_OP_TYPES",
    "POSE_D_POSE_ADVISORY_BASE",
    "POSE_D_POSE_T4_EXACT",
    "POSE_PER_DIM_TOLERANCE",
    "SPLIT_BACKENDS",
    "AnePrecisionError",
    "assert_op_sequence_stable",
    "compute_op_names",
    "crop_boxes",
    "crop_boxes_with_cores",
    "crop_pixel_fraction",
    "dilate_bool",
    "fixed_crop_boxes",
    "group_ranges",
    "hybrid_speedup",
    "margin_band_mask",
    "occupied_tiles",
    "pose_drift_verdict",
    "seg_flip_verdict",
    "selective_fp32_names",
    "selector_from_names",
    "split_backend_name",
    "split_fp16_names",
]


class AnePrecisionError(RuntimeError):
    """Raised when a precision-placement construction is not well formed."""


#: Exact contest d_pose the shipped body carries (``ddm_up2_shipping_pose_solve.py:92``).
POSE_D_POSE_T4_EXACT = 7.77e-06

#: pr1's advisory n600 base for d_pose -- the second denominator, always stated.
POSE_D_POSE_ADVISORY_BASE = 6.366e-06

#: Per-dimension absolute tolerance a pose backend must beat to be worth the
#: axis: ``sqrt(d_pose)``.  d_pose is a mean square over the 6 dims, so a single
#: dim carrying this much error *doubles* the quantity being measured.
POSE_PER_DIM_TOLERANCE = math.sqrt(POSE_D_POSE_T4_EXACT)

#: MIL op types that carry no compute; a split point never indexes one of these.
MIL_CONST_OP_TYPES: frozenset[str] = frozenset({"const", "constexpr_cast"})

#: Backend names this lane may add to the screening roster once MEASURED.
SPLIT_BACKENDS: tuple[str, ...] = (
    "ane_split_k1",
    "ane_split_k2",
    "ane_split_k4",
    "ane_split_k8",
    "ane_split_k16",
    "ane_split_k32",
    "ane_split_k64",
    "ane_hybrid_exact",
)


# --------------------------------------------------------------- op sequence


def compute_op_names(records: Iterable[Sequence[Any]]) -> tuple[str, ...]:
    """Ordered names of the COMPUTE ops in a MIL program.

    ``records`` is the ``(op_type, name)`` sequence an op-selector observed, in
    the order the selector was called -- which is the block's topological order.
    ``const`` ops are dropped: they carry weights, not compute, and a split
    point that indexed one would move a weight's dtype without moving any
    arithmetic.

    Duplicate names are a hard error, because the whole split construction
    addresses ops BY NAME.
    """
    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        op_type, name = record[0], record[1]
        if op_type in MIL_CONST_OP_TYPES:
            continue
        if name in seen:
            raise AnePrecisionError(
                f"duplicate MIL op name {name!r}: split points address ops by name, "
                "so a duplicate makes the construction ambiguous"
            )
        seen.add(name)
        names.append(name)
    if not names:
        raise AnePrecisionError("no compute ops found (every op was a const?)")
    return tuple(names)


def assert_op_sequence_stable(
    expected: Sequence[str], observed: Sequence[str], *, context: str = ""
) -> None:
    """Refuse a conversion whose op sequence drifted from the enumerated one.

    Every split point is an ordinal into the enumerated sequence.  If a later
    conversion emits a different sequence, the ordinal names a different op and
    every rung below it is silently mislabelled.  This is the guard that keeps
    the ladder honest; it is not decoration.
    """
    if len(expected) != len(observed):
        raise AnePrecisionError(
            f"op sequence length drifted{' in ' + context if context else ''}: "
            f"enumerated {len(expected)} compute ops, this conversion saw {len(observed)}"
        )
    for index, (want, got) in enumerate(zip(expected, observed, strict=True)):
        if want != got:
            raise AnePrecisionError(
                f"op sequence drifted{' in ' + context if context else ''} at index "
                f"{index}: enumerated {want!r}, this conversion saw {got!r}"
            )


def split_fp16_names(compute_names: Sequence[str], k: int) -> frozenset[str]:
    """fp16 PREFIX names for a split that leaves the LAST ``k`` compute ops fp32.

    ``k = 0`` is the all-fp16 model (ane1's ``ane_fp16``); ``k = len`` is the
    all-fp32 model (ane1's ``coreml_cpu_fp32``).  Both endpoints are already
    MEASURED, so the ladder's rungs interpolate between two known rows.
    """
    total = len(compute_names)
    if k < 0 or k > total:
        raise AnePrecisionError(f"split k={k} outside [0, {total}]")
    return frozenset(compute_names[: total - k])


def group_ranges(total: int, groups: int) -> tuple[tuple[int, int], ...]:
    """Contiguous ``[lo, hi)`` ordinal ranges covering ``total`` ops.

    Used by the per-op sensitivity profile: flipping ONE range to fp16 at a time
    against an all-fp32 reference measures the drift that range is responsible
    for.  Ranges are as equal as integer division allows and the earlier ranges
    absorb the remainder, so the partition is deterministic.
    """
    if total <= 0:
        raise AnePrecisionError(f"total must be positive, got {total}")
    if groups <= 0 or groups > total:
        raise AnePrecisionError(f"groups must be in [1, {total}], got {groups}")
    base, extra = divmod(total, groups)
    out: list[tuple[int, int]] = []
    lo = 0
    for index in range(groups):
        size = base + (1 if index < extra else 0)
        out.append((lo, lo + size))
        lo += size
    if lo != total:
        raise AnePrecisionError("group partition did not cover the op sequence")
    return tuple(out)


def selective_fp32_names(
    compute_names: Sequence[str], fp32_ordinals: Iterable[int]
) -> frozenset[str]:
    """fp16 names for a model where the named ORDINALS alone are held at fp32.

    This is the step-3 construction: the minimal per-op fp32 set the sensitivity
    profile names, with everything else left on the ANE.
    """
    total = len(compute_names)
    held: set[int] = set()
    for ordinal in fp32_ordinals:
        value = int(ordinal)
        if value < 0 or value >= total:
            raise AnePrecisionError(f"fp32 ordinal {value} outside [0, {total})")
        held.add(value)
    return frozenset(
        name for index, name in enumerate(compute_names) if index not in held
    )


def selector_from_names(names: Iterable[str]) -> Callable[[Any], bool]:
    """A ``coremltools`` ``op_selector`` that sends exactly ``names`` to fp16.

    The returned callable also records every ``(op_type, name)`` it is asked
    about on its ``observed`` attribute, so the caller can prove the op sequence
    of THIS conversion against the enumerated one rather than assuming stability
    across traces.
    """
    wanted = frozenset(names)

    def selector(op: Any) -> bool:
        selector.observed.append((op.op_type, op.name))  # type: ignore[attr-defined]
        return op.name in wanted

    selector.observed = []  # type: ignore[attr-defined]
    selector.wanted = wanted  # type: ignore[attr-defined]
    return selector


def split_backend_name(k: int) -> str:
    """Canonical backend name for the ``last k ops fp32`` rung."""
    if k < 0:
        raise AnePrecisionError(f"split k must be >= 0, got {k}")
    return f"ane_split_k{int(k)}"


# ------------------------------------------------------------------ verdicts


def seg_flip_verdict(
    flips: int, total_px: int, bar: float, per_pair: Sequence[float] | None = None
) -> dict[str, Any]:
    """SegNet argmax-flip row against the authority bar.

    ``bar`` is passed in rather than imported so the caller states which bar it
    is being judged against in the same breath as the number.
    """
    if total_px <= 0:
        raise AnePrecisionError(f"total_px must be positive, got {total_px}")
    rate = flips / total_px
    row: dict[str, Any] = {
        "flips": int(flips),
        "total_px": int(total_px),
        "flip_rate": float(rate),
        "bar": float(bar),
        "multiple_of_bar": float(rate / bar) if bar else float("inf"),
        "passes_authority_bar": bool(rate <= bar),
        "bit_exact_argmax": bool(flips == 0),
    }
    if per_pair is not None and len(per_pair):
        array = np.asarray(per_pair, dtype=np.float64)
        row["per_pair_median"] = float(np.median(array))
        row["per_pair_p95"] = float(np.percentile(array, 95))
        row["per_pair_max"] = float(array.max())
        row["pairs_with_any_flip"] = int((array > 0).sum())
        row["pairs"] = int(array.size)
    return row


def pose_drift_verdict(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    d_pose: float = POSE_D_POSE_T4_EXACT,
) -> dict[str, Any]:
    """6-dim pose drift row: per-dim error, self-MSE, and both bar readings.

    Two bars, both reported because they answer different questions:

    ``self_mse_multiple_of_d_pose``
        how large the backend's own perturbation is against the whole quantity
        the axis measures.  Must be far below 1 for the backend to be readable.

    ``max_dim_multiple_of_per_dim_tolerance``
        the same statement per dimension against ``sqrt(d_pose)`` -- the form
        the charter's prior-law prediction is written in.
    """
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape != got.shape:
        raise AnePrecisionError(f"shape mismatch: reference {ref.shape} vs {got.shape}")
    if ref.ndim != 2:
        raise AnePrecisionError(f"expected (pairs, dims), got {ref.shape}")
    delta = got - ref
    per_dim_abs = np.abs(delta)
    self_mse = (delta**2).mean(axis=1)
    tolerance = math.sqrt(d_pose)
    dim_median = per_dim_abs.mean(axis=0)
    return {
        "pairs": int(ref.shape[0]),
        "dims": int(ref.shape[1]),
        "d_pose_denominator": float(d_pose),
        "per_dim_tolerance": float(tolerance),
        "abs_delta_median": float(np.median(per_dim_abs)),
        "abs_delta_p95": float(np.percentile(per_dim_abs, 95)),
        "abs_delta_max": float(per_dim_abs.max()),
        "self_mse_median": float(np.median(self_mse)),
        "self_mse_p95": float(np.percentile(self_mse, 95)),
        "self_mse_max": float(self_mse.max()),
        "self_mse_multiple_of_d_pose": float(np.median(self_mse) / d_pose),
        "max_dim_multiple_of_per_dim_tolerance": float(per_dim_abs.max() / tolerance),
        "per_dim_mean_abs_delta": [float(v) for v in dim_median],
        "per_dim_share_of_mse": [
            float(v) for v in (delta**2).sum(axis=0) / max((delta**2).sum(), 1e-300)
        ],
        "passes_per_dim_tolerance": bool(per_dim_abs.max() <= tolerance),
        "readable_against_d_pose": bool(float(np.median(self_mse)) <= 0.01 * d_pose),
    }


# ------------------------------------------------------- realized-hybrid geometry


def margin_band_mask(margin: np.ndarray, width: float) -> np.ndarray:
    """Pixels whose top-2 logit margin is below ``width`` -- the recompute band."""
    if width < 0:
        raise AnePrecisionError(f"band width must be >= 0, got {width}")
    return np.asarray(margin) < width


def dilate_bool(mask: np.ndarray, halo: int) -> np.ndarray:
    """Square dilation of a 2-D boolean mask by ``halo`` pixels.

    A recompute band is a set of PIXELS, but a convolutional recompute needs the
    receptive field around each of them.  Dilating by the halo before measuring
    area is what turns an area price into a compute price; skipping it is the
    step that made the 07-13 lane's proportional model wrong.

    Implemented with a summed-area table so the cost does not depend on ``halo``.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise AnePrecisionError(f"expected a 2-D mask, got shape {array.shape}")
    if halo < 0:
        raise AnePrecisionError(f"halo must be >= 0, got {halo}")
    if halo == 0 or not array.any():
        return array.copy()
    height, width = array.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(array.astype(np.int64), axis=0), axis=1)
    rows = np.arange(height)
    cols = np.arange(width)
    r0 = np.clip(rows - halo, 0, height)[:, None]
    r1 = np.clip(rows + halo + 1, 0, height)[:, None]
    c0 = np.clip(cols - halo, 0, width)[None, :]
    c1 = np.clip(cols + halo + 1, 0, width)[None, :]
    total = (
        integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    )
    return total > 0


def occupied_tiles(mask: np.ndarray, tile: int) -> tuple[int, int]:
    """``(occupied, total)`` tile counts for a ``tile x tile`` grid over ``mask``.

    Tile occupancy -- not pixel area -- is what a crop-batched recompute pays
    for, and the 07-13 lane's 4.27x negative was an occupancy result (a median
    22.5 of 48 tiles lit up by a 5% band).  Measuring it is the point.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise AnePrecisionError(f"expected a 2-D mask, got shape {array.shape}")
    if tile <= 0:
        raise AnePrecisionError(f"tile must be positive, got {tile}")
    height, width = array.shape
    rows = math.ceil(height / tile)
    cols = math.ceil(width / tile)
    occupied = 0
    for r in range(rows):
        for c in range(cols):
            if array[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile].any():
                occupied += 1
    return occupied, rows * cols


def crop_boxes(
    mask: np.ndarray, tile: int, halo: int
) -> tuple[tuple[int, int, int, int], ...]:
    """Tile-aligned crop boxes (``y0, y1, x0, x1``) an fp32 recompute must run.

    Each occupied ``tile x tile`` cell is expanded by ``halo`` on every side and
    clipped to the image.  The returned boxes are what a crop-BATCHED CoreML
    pass actually evaluates, so their summed area is the realized compute, not
    the band's pixel area.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise AnePrecisionError(f"expected a 2-D mask, got shape {array.shape}")
    if tile <= 0:
        raise AnePrecisionError(f"tile must be positive, got {tile}")
    if halo < 0:
        raise AnePrecisionError(f"halo must be >= 0, got {halo}")
    height, width = array.shape
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(math.ceil(height / tile)):
        for c in range(math.ceil(width / tile)):
            y0, y1 = r * tile, min((r + 1) * tile, height)
            x0, x1 = c * tile, min((c + 1) * tile, width)
            if not array[y0:y1, x0:x1].any():
                continue
            boxes.append(
                (
                    max(0, y0 - halo),
                    min(height, y1 + halo),
                    max(0, x0 - halo),
                    min(width, x1 + halo),
                )
            )
    return tuple(boxes)


def crop_boxes_with_cores(
    mask: np.ndarray, tile: int, halo: int
) -> tuple[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], ...]:
    """``((core, expanded), ...)`` for every occupied tile.

    The CORE is the tile itself; the EXPANDED box is the core plus ``halo`` on
    every side, clipped.  A recompute must EVALUATE the expanded box (that is
    the receptive field it needs) but may only WRITE BACK the core: a band pixel
    lying in a neighbour's halo has less context there than in its own tile, so
    letting a halo write win would silently degrade the pixel the hybrid is
    supposed to be fixing.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise AnePrecisionError(f"expected a 2-D mask, got shape {array.shape}")
    if tile <= 0:
        raise AnePrecisionError(f"tile must be positive, got {tile}")
    if halo < 0:
        raise AnePrecisionError(f"halo must be >= 0, got {halo}")
    height, width = array.shape
    out: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for r in range(math.ceil(height / tile)):
        for c in range(math.ceil(width / tile)):
            y0, y1 = r * tile, min((r + 1) * tile, height)
            x0, x1 = c * tile, min((c + 1) * tile, width)
            if not array[y0:y1, x0:x1].any():
                continue
            out.append(
                (
                    (y0, y1, x0, x1),
                    (
                        max(0, y0 - halo),
                        min(height, y1 + halo),
                        max(0, x0 - halo),
                        min(width, x1 + halo),
                    ),
                )
            )
    return tuple(out)


def fixed_crop_boxes(
    mask: np.ndarray, tile: int, halo: int
) -> tuple[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], ...]:
    """``((core, box), ...)`` where EVERY box is exactly ``tile + 2*halo`` square.

    A CoreML model is converted for ONE input shape, so a recompute pass cannot
    feed it clipped boxes of five different sizes.  Boxes at the frame edge are
    therefore SHIFTED inward instead of clipped: the core keeps its tile
    alignment and the halo simply lands asymmetrically.  Every box then feeds the
    same model, which is what makes a crop-batched recompute realizable at all.

    Raises when the box would not fit the frame -- silently shrinking it would
    change the receptive field the halo was chosen to provide.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise AnePrecisionError(f"expected a 2-D mask, got shape {array.shape}")
    if tile <= 0:
        raise AnePrecisionError(f"tile must be positive, got {tile}")
    if halo < 0:
        raise AnePrecisionError(f"halo must be >= 0, got {halo}")
    height, width = array.shape
    size = tile + 2 * halo
    if size > height or size > width:
        raise AnePrecisionError(
            f"crop box {size}x{size} does not fit a {height}x{width} frame; "
            "reduce tile or halo rather than shrinking the receptive field"
        )
    out: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for r in range(math.ceil(height / tile)):
        for c in range(math.ceil(width / tile)):
            y0, y1 = r * tile, min((r + 1) * tile, height)
            x0, x1 = c * tile, min((c + 1) * tile, width)
            if not array[y0:y1, x0:x1].any():
                continue
            by = min(max(0, y0 - halo), height - size)
            bx = min(max(0, x0 - halo), width - size)
            out.append(((y0, y1, x0, x1), (by, by + size, bx, bx + size)))
    return tuple(out)


def crop_pixel_fraction(
    boxes: Sequence[tuple[int, int, int, int]], height: int, width: int
) -> float:
    """Summed crop area as a fraction of one full frame (overlaps counted twice).

    Overlaps are counted because the recompute pays for them: two crops that
    share a halo strip each evaluate it.
    """
    if height <= 0 or width <= 0:
        raise AnePrecisionError(f"bad frame shape ({height}, {width})")
    area = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in boxes)
    return area / float(height * width)


def hybrid_speedup(
    *, ane_s: float, recompute_s: float, reference_s: float, bar: float = 3.0
) -> dict[str, Any]:
    """Realized hybrid timing row: measured legs in, speedup and verdict out."""
    for label, value in (("ane_s", ane_s), ("recompute_s", recompute_s), ("reference_s", reference_s)):
        if value < 0:
            raise AnePrecisionError(f"{label} must be >= 0, got {value}")
    total = ane_s + recompute_s
    if total <= 0:
        raise AnePrecisionError("hybrid total time must be positive")
    speedup = reference_s / total
    return {
        "ane_s": float(ane_s),
        "recompute_s": float(recompute_s),
        "hybrid_total_s": float(total),
        "reference_s": float(reference_s),
        "speedup": float(speedup),
        "bar": float(bar),
        "passes_speed_bar": bool(speedup >= bar),
        "recompute_share_of_hybrid": float(recompute_s / total),
    }
