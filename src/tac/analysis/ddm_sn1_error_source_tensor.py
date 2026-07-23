# SPDX-License-Identifier: MIT
"""Typed reductions for the DDM SN1 residual error-source tensor.

The tensor is deliberately an attribution of *measured current residual
errors*, not a causal attribution of hidden network state.  Current semantic
reach and one tested vocabulary extension form an exhaustive, ordered
partition:

``DESCRIBED_BUT_REALIZATION_LOST``
    the current receiver's semantic program already requests the target cell;
``NEVER_DESCRIBED``
    the tested enriched vocabulary requests the target, while the current
    receiver does not;
``STRUCTURALLY_HARD_IRREDUCIBLE``
    neither tested semantic program requests the target.

The final label is always scoped to the two tested vocabularies.  It is not a
global impossibility claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
from scipy import ndimage

CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
HEIGHT: Final = 384
WIDTH: Final = 512
N_CLASSES: Final = 5
N600_SITES: Final = 600 * HEIGHT * WIDTH


class ErrorSource(IntEnum):
    """Ordered exhaustive residual-source partition."""

    DESCRIBED_BUT_REALIZATION_LOST = 0
    NEVER_DESCRIBED = 1
    STRUCTURALLY_HARD_IRREDUCIBLE = 2


SOURCE_NAMES: Final = tuple(row.name for row in ErrorSource)
MARGIN_BANDS: Final = (
    "BELOW_SIDED_Q10",
    "Q10_TO_Q90",
    "ABOVE_SIDED_Q90",
    "NO_SIDED_REFERENCE_SUPPORT",
)
CURVATURE_BANDS: Final = ("INTERIOR", "CODIM1_FLAT", "CODIM2_OR_CORNER")
TEMPORAL_PATTERNS: Final = (
    "NO_G4_SAME_TRANSITION_HISTORY",
    "ADVECTED_OR_SINGLETON_HISTORICAL",
    "STATIC_IN_IMAGE_HISTORICAL",
    "EVENT_ADJACENT_G3_PROXY",
)
BOUNDARY_DISTANCE_BANDS: Final = (
    "BOUNDARY_BAND_LE1",
    "ANNULUS_2_TO_5",
    "COSTED_6_TO_8",
    "INTERIOR_GT8",
)
PAINT_FLOOR_MECHANISMS: Final = (
    "COARSE_DESCRIPTION",
    "PAINT_FUNCTION",
    "TEXTURE_PRIOR_REGION_ERF",
)


class ErrorSourceTensorError(RuntimeError):
    """Raised when a tensor invariant does not close."""


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    component_count: int
    component_size_min: int
    component_size_median: float
    component_size_p90: float
    component_size_max: int
    pixel_mass_by_scale: Mapping[str, int]


def classify_error_sources(
    *,
    target: np.ndarray,
    predicted: np.ndarray,
    current_semantic: np.ndarray,
    enriched_semantic: np.ndarray,
    target_class_ids: tuple[int, ...] = (0, 2, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Return one source code for every residual error in the declared strata."""

    arrays = (target, predicted, current_semantic, enriched_semantic)
    if any(np.asarray(row).shape != (HEIGHT, WIDTH) for row in arrays):
        raise ErrorSourceTensorError("error-source planes must be 384x512")
    if any(np.any((np.asarray(row) < 0) | (np.asarray(row) >= N_CLASSES)) for row in arrays):
        raise ErrorSourceTensorError("error-source planes contain an invalid class ID")
    target_array = np.asarray(target, dtype=np.uint8)
    predicted_array = np.asarray(predicted, dtype=np.uint8)
    residual = (predicted_array != target_array) & np.isin(
        target_array,
        np.asarray(target_class_ids, dtype=np.uint8),
    )
    source = np.full((HEIGHT, WIDTH), -1, dtype=np.int8)
    described = residual & (np.asarray(current_semantic) == target_array)
    source[described] = int(ErrorSource.DESCRIBED_BUT_REALIZATION_LOST)
    never = residual & ~described & (np.asarray(enriched_semantic) == target_array)
    source[never] = int(ErrorSource.NEVER_DESCRIBED)
    source[residual & ~described & ~never] = int(ErrorSource.STRUCTURALLY_HARD_IRREDUCIBLE)
    assigned = source >= 0
    if not np.array_equal(assigned, residual):
        raise ErrorSourceTensorError("residual source assignment is not exhaustive")
    return source, residual


def d2_margin_bands(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    logits: np.ndarray,
    head_norms: Mapping[str, float],
    sided_thresholds: Mapping[str, tuple[float | None, float | None]],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute current decision-head D2 and bucket it by frozen sided quantiles."""

    if predicted.shape != (HEIGHT, WIDTH) or target.shape != (HEIGHT, WIDTH):
        raise ErrorSourceTensorError("D2 class planes must be 384x512")
    if logits.shape != (N_CLASSES, HEIGHT, WIDTH):
        raise ErrorSourceTensorError("D2 logits must have shape (5,384,512)")
    rows, columns = np.indices((HEIGHT, WIDTH))
    winner_logits = logits[predicted, rows, columns]
    target_logits = logits[target, rows, columns]
    d2 = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    bands = np.empty((HEIGHT, WIDTH), dtype=np.int8)
    for winner in range(N_CLASSES):
        for rival in range(N_CLASSES):
            if winner == rival:
                continue
            orientation = f"{CLASS_NAMES[winner]}->{CLASS_NAMES[rival]}"
            norm = float(head_norms[orientation])
            q10, q90 = sided_thresholds[orientation]
            if not norm > 0.0:
                raise ErrorSourceTensorError(f"invalid D2 normal for {orientation}: {norm}")
            selected = (predicted == winner) & (target == rival)
            values = (winner_logits[selected] - target_logits[selected]) / norm
            d2[selected] = values
            if q10 is None or q90 is None:
                if q10 is not None or q90 is not None:
                    raise ErrorSourceTensorError(f"partial sided reference for {orientation}: q10={q10}, q90={q90}")
                bands[selected] = 3
            else:
                if not 0.0 <= q10 <= q90:
                    raise ErrorSourceTensorError(f"invalid D2 custody for {orientation}: q10={q10}, q90={q90}")
                bands[selected] = np.where(values < q10, 0, np.where(values <= q90, 1, 2))
    same = predicted == target
    d2[same] = 0.0
    bands[same] = 0
    if not np.all(np.isfinite(d2)):
        raise ErrorSourceTensorError("D2 field contains a non-finite value")
    return d2, bands


def curvature_bands(target: np.ndarray) -> np.ndarray:
    """Return a deterministic four-neighbour discrete curvature proxy.

    For the target class occupying each site, ``4 - n_same`` is zero in the
    four-neighbour interior, one on locally flat codimension-1 support, and at
    least two at a corner, tip, or one-cell feature.
    """

    values = np.asarray(target, dtype=np.uint8)
    if values.shape != (HEIGHT, WIDTH):
        raise ErrorSourceTensorError("curvature target plane must be 384x512")
    output = np.empty((HEIGHT, WIDTH), dtype=np.int8)
    kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    for class_id in range(N_CLASSES):
        mask = values == class_id
        neighbours = ndimage.convolve(
            mask.astype(np.int8),
            kernel,
            mode="constant",
            cval=0,
        )
        deficit = 4 - neighbours
        output[mask] = np.where(deficit[mask] == 0, 0, np.where(deficit[mask] == 1, 1, 2))
    return output


def boundary_distance_bands(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Euclidean distance to a two-sided 4-neighbour target boundary."""

    values = np.asarray(target, dtype=np.uint8)
    if values.shape != (HEIGHT, WIDTH):
        raise ErrorSourceTensorError("boundary-distance target plane must be 384x512")
    boundary = np.zeros((HEIGHT, WIDTH), dtype=bool)
    horizontal = values[:, 1:] != values[:, :-1]
    vertical = values[1:, :] != values[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    distance = ndimage.distance_transform_edt(~boundary).astype(np.float32)
    bands = np.where(
        distance <= 1.0,
        0,
        np.where(distance <= 5.0, 1, np.where(distance <= 8.0, 2, 3)),
    ).astype(np.int8)
    return distance, bands


def paint_floor_mechanism_codes(
    *,
    target: np.ndarray,
    predicted: np.ndarray,
    margin_band: np.ndarray,
    boundary_distance_band: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjudicate paint-floor mechanism using only preregistered observable axes.

    This is an operational association, not hidden-state causal identification:
    a Lane-involving ordered pair establishes continuous-curve availability;
    deep interior support or an above-q90 Lane margin points to texture/ERF;
    other curve-available support points to paint-function resolution; and
    support without a continuous Lane curve points to coarse description.
    """

    arrays = (target, predicted, margin_band, boundary_distance_band)
    if any(np.asarray(row).shape != (HEIGHT, WIDTH) for row in arrays):
        raise ErrorSourceTensorError("paint-floor planes must be 384x512")
    curve_available = (np.asarray(target) == 1) | (np.asarray(predicted) == 1)
    interior = np.asarray(boundary_distance_band) == 3
    high_margin_curve = curve_available & (np.asarray(margin_band) == 2)
    mechanism = np.where(
        interior | high_margin_curve,
        2,
        np.where(curve_available, 1, 0),
    ).astype(np.int8)
    return mechanism, curve_available


def survival_wall_149(
    *,
    target: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    """Measure argmax errors in the historical #149 one-dilation boundary band."""

    values = np.asarray(target, dtype=np.uint8)
    observed = np.asarray(predicted, dtype=np.uint8)
    if values.shape != (HEIGHT, WIDTH) or observed.shape != (HEIGHT, WIDTH):
        raise ErrorSourceTensorError("#149 survival-wall planes must be 384x512")
    boundary = np.zeros((HEIGHT, WIDTH), dtype=bool)
    horizontal = values[:, 1:] != values[:, :-1]
    vertical = values[1:, :] != values[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    band = ndimage.binary_dilation(boundary, iterations=1)
    errors = observed != values

    def summary(mask: np.ndarray) -> dict[str, int | float | None]:
        sites = int(np.count_nonzero(mask))
        error_count = int(np.count_nonzero(errors & mask))
        return {
            "sites": sites,
            "errors": error_count,
            "error_fraction": error_count / sites if sites else None,
        }

    return {
        "all_classes": summary(band),
        "by_target_class": {
            class_name: summary(band & (values == class_id))
            for class_id, class_name in enumerate(CLASS_NAMES)
        },
    }


def temporal_pattern_codes(
    *,
    recurrence: np.ndarray,
    event_adjacent: bool,
) -> np.ndarray:
    """Bucket historical G4 recurrence with an explicit G3 event-proxy override."""

    values = np.asarray(recurrence)
    if values.shape != (HEIGHT, WIDTH) or np.any(values < 0):
        raise ErrorSourceTensorError("historical recurrence plane is invalid")
    if event_adjacent:
        return np.full((HEIGHT, WIDTH), 3, dtype=np.int8)
    return np.where(values == 0, 0, np.where(values == 1, 1, 2)).astype(np.int8)


def summarize_components(mask: np.ndarray) -> ComponentSummary:
    """Summarize 4-connected cluster geometry without persisting pixel lists."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (HEIGHT, WIDTH) or not np.any(selected):
        raise ErrorSourceTensorError("component summary requires a nonempty 384x512 mask")
    labels, count = ndimage.label(
        selected,
        structure=ndimage.generate_binary_structure(2, 1),
    )
    sizes = np.bincount(labels.ravel())[1:].astype(np.int64)
    if len(sizes) != count or int(sizes.sum()) != int(np.count_nonzero(selected)):
        raise ErrorSourceTensorError("component accounting does not close")
    scale_mass = {
        "POINT_LE4": int(sizes[sizes <= 4].sum()),
        "BOUNDARY_SEGMENT_5_TO_64": int(sizes[(sizes >= 5) & (sizes <= 64)].sum()),
        "REGION_GT64": int(sizes[sizes > 64].sum()),
    }
    return ComponentSummary(
        component_count=int(count),
        component_size_min=int(sizes.min()),
        component_size_median=float(np.median(sizes)),
        component_size_p90=float(np.quantile(sizes, 0.9)),
        component_size_max=int(sizes.max()),
        pixel_mass_by_scale=scale_mass,
    )


def encode_group_key(
    *,
    source: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    margin_band: np.ndarray,
    curvature_band: np.ndarray,
    temporal_pattern: np.ndarray,
    boundary_distance_band: np.ndarray,
    paint_floor_mechanism: np.ndarray,
) -> np.ndarray:
    """Encode the joint categorical index into one stable integer field."""

    base = (
        (
            ((source.astype(np.int64) * N_CLASSES + target.astype(np.int64)) * N_CLASSES + predicted.astype(np.int64))
            * len(MARGIN_BANDS)
            + margin_band.astype(np.int64)
        )
        * len(CURVATURE_BANDS)
        + curvature_band.astype(np.int64)
    ) * len(TEMPORAL_PATTERNS) + temporal_pattern.astype(np.int64)
    return (
        base * len(BOUNDARY_DISTANCE_BANDS)
        + boundary_distance_band.astype(np.int64)
    ) * len(PAINT_FLOOR_MECHANISMS) + paint_floor_mechanism.astype(np.int64)


def decode_group_key(value: int) -> dict[str, int]:
    """Inverse of :func:`encode_group_key` for one nonnegative key."""

    if value < 0:
        raise ErrorSourceTensorError("group key must be nonnegative")
    remainder, mechanism = divmod(int(value), len(PAINT_FLOOR_MECHANISMS))
    remainder, boundary_distance = divmod(remainder, len(BOUNDARY_DISTANCE_BANDS))
    remainder, temporal = divmod(remainder, len(TEMPORAL_PATTERNS))
    remainder, curvature = divmod(remainder, len(CURVATURE_BANDS))
    remainder, margin = divmod(remainder, len(MARGIN_BANDS))
    remainder, predicted = divmod(remainder, N_CLASSES)
    source, target = divmod(remainder, N_CLASSES)
    if source >= len(ErrorSource):
        raise ErrorSourceTensorError("group key source code is invalid")
    return {
        "source": source,
        "target": target,
        "predicted": predicted,
        "margin_band": margin,
        "curvature_band": curvature,
        "temporal_pattern": temporal,
        "boundary_distance_band": boundary_distance,
        "paint_floor_mechanism": mechanism,
    }


def source_budget(
    *,
    counts: Mapping[str, Mapping[str, int]],
    target_sites: Mapping[str, int],
) -> dict[str, object]:
    """Build an exact three-way error and global d_seg budget."""

    rows: list[dict[str, object]] = []
    total = 0
    for source_name in SOURCE_NAMES:
        for class_name in ("Road", "Undrivable", "MyCar"):
            errors = int(counts.get(source_name, {}).get(class_name, 0))
            sites = int(target_sites[class_name])
            total += errors
            rows.append(
                {
                    "source": source_name,
                    "stratum": class_name,
                    "errors": errors,
                    "global_d_seg": errors / N600_SITES,
                    "conditional_error_rate": errors / sites if sites else None,
                }
            )
    return {
        "schema": "ddm_sn1_error_source_budget.v1",
        "rows": rows,
        "total_errors": total,
        "global_d_seg": total / N600_SITES,
    }


__all__ = [
    "BOUNDARY_DISTANCE_BANDS",
    "CLASS_NAMES",
    "CURVATURE_BANDS",
    "HEIGHT",
    "MARGIN_BANDS",
    "N600_SITES",
    "PAINT_FLOOR_MECHANISMS",
    "SOURCE_NAMES",
    "TEMPORAL_PATTERNS",
    "WIDTH",
    "ComponentSummary",
    "ErrorSource",
    "ErrorSourceTensorError",
    "boundary_distance_bands",
    "classify_error_sources",
    "curvature_bands",
    "d2_margin_bands",
    "decode_group_key",
    "encode_group_key",
    "paint_floor_mechanism_codes",
    "source_budget",
    "summarize_components",
    "survival_wall_149",
    "temporal_pattern_codes",
]
