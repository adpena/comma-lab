# SPDX-License-Identifier: MIT
"""Spatial lens: containment, overlap, distance, and Laguerre cells."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import LensOperationError, SpatialGeometry, T, TypedResult, immutable_array


@dataclass(frozen=True, slots=True)
class PointContainment:
    point: tuple[float, float]
    axes: tuple[str, str]
    inside: bool
    on_boundary: bool
    boundary_included: bool


@dataclass(frozen=True, slots=True)
class Overlap:
    intersection: int
    union: int
    iou: float
    representation: str


@dataclass(frozen=True, slots=True)
class Distance:
    value: float
    metric: str
    left: tuple[float, ...]
    right: tuple[float, ...]
    axes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LaguerreDiagram:
    sites: tuple[tuple[float, float], ...]
    weights: tuple[float, ...]
    classes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LaguerreCells:
    diagram: LaguerreDiagram
    cell_indices: np.ndarray
    class_labels: np.ndarray
    shape: tuple[int, int]


def _pair(value: Any, *, owner: str) -> tuple[float, float]:
    try:
        out = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise LensOperationError(f"{owner} must be a length-2 coordinate") from exc
    if len(out) != 2 or not all(math.isfinite(item) for item in out):
        raise LensOperationError(f"{owner} must be a finite length-2 coordinate")
    return out[0], out[1]


def _spatial_point(complex_: T, element_id: str) -> tuple[tuple[float, ...], tuple[str, ...]]:
    spatial = complex_.element(element_id).spatial
    if spatial is None or not spatial.coordinates:
        raise LensOperationError(f"element {element_id!r} has no spatial coordinates")
    dims = len(spatial.coordinates[0])
    if any(len(coord) != dims for coord in spatial.coordinates):
        raise LensOperationError(f"element {element_id!r} has inconsistent coordinate arity")
    point = tuple(
        float(sum(coord[axis] for coord in spatial.coordinates) / len(spatial.coordinates))
        for axis in range(dims)
    )
    return point, spatial.axes


def _axes(value: Any | None, dimensions: int, *, owner: str) -> tuple[str, ...]:
    if value is None:
        default = ("x", "y") if dimensions == 2 else tuple(f"axis:{i}" for i in range(dimensions))
        return default
    try:
        axes = tuple(str(axis) for axis in value)
    except TypeError as exc:
        raise LensOperationError(f"{owner} axes must be an iterable of names") from exc
    if len(axes) != dimensions or any(not axis for axis in axes):
        raise LensOperationError(f"{owner} axes must match coordinate dimensions")
    return axes


def _polygon(
    complex_: T,
    value: Any | None,
    element_id: str | None,
    axes_value: Any | None,
) -> tuple[tuple[tuple[float, float], ...], tuple[str, str]]:
    if element_id is not None:
        spatial = complex_.element(element_id).spatial
        if spatial is None:
            raise LensOperationError(f"element {element_id!r} has no spatial geometry")
        value = spatial.coordinates
        axes_value = spatial.axes
    if value is None:
        raise LensOperationError("point_in_polygon requires polygon= or polygon_id=")
    points = tuple(_pair(point, owner="polygon vertex") for point in value)
    if len(points) < 3:
        raise LensOperationError("polygon requires at least three vertices")
    axes = _axes(axes_value, 2, owner="polygon")
    return points, (axes[0], axes[1])


def _on_segment(
    point: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    px, py = point
    x1, y1 = left
    x2, y2 = right
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance
    )


def _point_in_polygon(
    point: tuple[float, float],
    polygon: tuple[tuple[float, float], ...],
    *,
    axes: tuple[str, str],
    include_boundary: bool,
) -> PointContainment:
    on_boundary = any(
        _on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )
    if on_boundary:
        return PointContainment(point, axes, include_boundary, True, include_boundary)
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return PointContainment(point, axes, inside, False, include_boundary)


def _overlap(left: Any, right: Any) -> Overlap:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.ndim > 0 or right_array.ndim > 0:
        if left_array.shape != right_array.shape:
            raise LensOperationError(
                f"IoU masks must have equal shape; got {left_array.shape} and {right_array.shape}"
            )
        left_mask = left_array.astype(bool, copy=False)
        right_mask = right_array.astype(bool, copy=False)
        intersection = int(np.count_nonzero(left_mask & right_mask))
        union = int(np.count_nonzero(left_mask | right_mask))
        return Overlap(intersection, union, intersection / union if union else 1.0, "mask")
    try:
        left_set = frozenset(left)
        right_set = frozenset(right)
    except TypeError as exc:
        raise LensOperationError("IoU inputs must be equal-shape masks or iterables") from exc
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return Overlap(intersection, union, intersection / union if union else 1.0, "set")


def _coordinate(value: Any, *, owner: str) -> tuple[float, ...]:
    try:
        out = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise LensOperationError(f"{owner} must be a numeric coordinate") from exc
    if not out or not all(math.isfinite(item) for item in out):
        raise LensOperationError(f"{owner} must be a finite non-empty coordinate")
    return out


def _distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
    axes: tuple[str, ...],
    metric: str,
) -> Distance:
    if len(left) != len(right):
        raise LensOperationError("distance coordinates must have equal dimensions")
    delta = tuple(abs(a - b) for a, b in zip(left, right, strict=True))
    if metric == "euclidean":
        value = math.sqrt(sum(item * item for item in delta))
    elif metric == "manhattan":
        value = sum(delta)
    elif metric == "chebyshev":
        value = max(delta)
    else:
        raise LensOperationError(
            "distance metric must be 'euclidean', 'manhattan', or 'chebyshev'"
        )
    return Distance(float(value), metric, left, right, axes)


def _laguerre(args: dict[str, Any]) -> LaguerreCells:
    try:
        from tac.boundary_math.partition_collapse import (
            PartitionCollapseError,
            PowerDiagram,
            power_assign,
        )
    except ImportError as exc:
        raise LensOperationError(
            "laguerre_cells requires the optional analysis dependencies; install tac[analysis]"
        ) from exc
    diagram = args.pop("diagram", None)
    if diagram is None:
        if "sites" not in args or "shape" not in args:
            raise LensOperationError("laguerre_cells requires diagram= or sites= and shape=")
        try:
            sites = np.asarray(args.pop("sites"), dtype=np.float32)
            if sites.ndim != 2 or sites.shape[1] != 2:
                raise LensOperationError("Laguerre sites must have shape (K,2)")
            weights = np.asarray(
                args.pop("weights", np.zeros(sites.shape[0])),
                dtype=np.float32,
            )
            classes = np.asarray(
                args.pop("classes", np.arange(sites.shape[0])),
                dtype=np.int32,
            )
            diagram = PowerDiagram(sites=sites, weights=weights, classes=classes)
        except LensOperationError:
            raise
        except (PartitionCollapseError, TypeError, ValueError) as exc:
            raise LensOperationError(f"invalid Laguerre diagram: {exc}") from exc
    elif not isinstance(diagram, PowerDiagram):
        raise LensOperationError("diagram must be a PowerDiagram")
    shape_value = args.pop("shape", None)
    if shape_value is None:
        raise LensOperationError("laguerre_cells requires shape=(height, width)")
    try:
        shape = tuple(int(value) for value in shape_value)
    except (TypeError, ValueError) as exc:
        raise LensOperationError("shape must be a length-2 integer tuple") from exc
    if len(shape) != 2 or min(shape) <= 0:
        raise LensOperationError("shape must contain two positive integers")
    row_chunk = int(args.pop("row_chunk", 32))
    if row_chunk <= 0:
        raise LensOperationError("row_chunk must be positive")
    if args:
        raise LensOperationError(f"unexpected laguerre_cells arguments: {sorted(args)}")
    try:
        cell_indices, class_labels = power_assign(diagram, shape, row_chunk=row_chunk)
    except (PartitionCollapseError, ValueError) as exc:
        raise LensOperationError(f"Laguerre assignment failed: {exc}") from exc
    snapshot = LaguerreDiagram(
        sites=tuple((float(site[0]), float(site[1])) for site in diagram.sites),
        weights=tuple(float(weight) for weight in diagram.weights),
        classes=tuple(int(class_id) for class_id in diagram.classes),
    )
    return LaguerreCells(
        snapshot,
        immutable_array(cell_indices),
        immutable_array(class_labels),
        shape,
    )


class SpatialLens:
    """Typed spatial operations over explicit coordinates, masks, and diagrams."""

    name = "spatial"
    operations = frozenset({"point_in_polygon", "iou_overlap", "distance", "laguerre_cells"})

    def apply(self, complex_: T, op: str, **args: Any) -> TypedResult[Any]:
        if op not in self.operations:
            raise LensOperationError(
                f"spatial operation {op!r} is unsupported; choose {sorted(self.operations)}"
            )
        element_ids: tuple[str, ...] = ()
        value: Any
        provenance: tuple[str, ...]
        if op == "point_in_polygon":
            point_id = args.pop("point_id", None)
            polygon_id = args.pop("polygon_id", None)
            point_value = args.pop("point", None)
            if point_id is not None:
                point_value, point_axes = _spatial_point(complex_, str(point_id))
                element_ids += (str(point_id),)
            else:
                point_axes = _axes(args.pop("point_axes", None), 2, owner="point")
            if point_value is None:
                raise LensOperationError("point_in_polygon requires point= or point_id=")
            point = _pair(point_value, owner="point")
            polygon, polygon_axes = _polygon(
                complex_,
                args.pop("polygon", None),
                polygon_id,
                args.pop("polygon_axes", None),
            )
            if polygon_id is not None:
                element_ids += (str(polygon_id),)
            if tuple(point_axes) != tuple(polygon_axes):
                raise LensOperationError(
                    f"point axes {tuple(point_axes)} != polygon axes {tuple(polygon_axes)}; "
                    "provide an explicit coordinate transform"
                )
            include_boundary = bool(args.pop("include_boundary", True))
            self._reject_extra(args)
            value = _point_in_polygon(
                point,
                polygon,
                axes=(point_axes[0], point_axes[1]),
                include_boundary=include_boundary,
            )
            provenance = ("even-odd ray casting with explicit boundary convention",)
        elif op == "iou_overlap":
            if "left" not in args or "right" not in args:
                raise LensOperationError("iou_overlap requires left= and right=")
            left = args.pop("left")
            right = args.pop("right")
            self._reject_extra(args)
            value = _overlap(left, right)
            provenance = ("exact set or boolean-mask intersection-over-union",)
        elif op == "distance":
            left_id = args.pop("left_id", None)
            right_id = args.pop("right_id", None)
            if left_id is not None:
                left, left_axes = _spatial_point(complex_, str(left_id))
                element_ids += (str(left_id),)
            else:
                if "left" not in args:
                    raise LensOperationError("distance requires left= or left_id=")
                left = _coordinate(args.pop("left"), owner="left")
                left_axes = _axes(args.pop("left_axes", None), len(left), owner="left")
            if right_id is not None:
                right, right_axes = _spatial_point(complex_, str(right_id))
                element_ids += (str(right_id),)
            else:
                if "right" not in args:
                    raise LensOperationError("distance requires right= or right_id=")
                right = _coordinate(args.pop("right"), owner="right")
                right_axes = _axes(args.pop("right_axes", None), len(right), owner="right")
            if len(left) != len(right):
                raise LensOperationError("distance coordinates must have equal dimensions")
            if tuple(left_axes) != tuple(right_axes):
                raise LensOperationError(
                    f"left axes {tuple(left_axes)} != right axes {tuple(right_axes)}; "
                    "provide an explicit coordinate transform"
                )
            metric = str(args.pop("metric", "euclidean"))
            self._reject_extra(args)
            value = _distance(left, right, tuple(left_axes), metric)
            provenance = ("direct distance over explicit T.X coordinates",)
        else:
            value = _laguerre(args)
            provenance = (
                "tac.boundary_math.partition_collapse.PowerDiagram",
                "tac.boundary_math.partition_collapse.power_assign",
            )
        return TypedResult(
            lens=self.name,
            op=op,
            value=value,
            element_ids=tuple(dict.fromkeys(element_ids)),
            provenance=provenance,
        )

    @staticmethod
    def _reject_extra(args: dict[str, Any]) -> None:
        if args:
            raise LensOperationError(f"unexpected spatial arguments: {sorted(args)}")


def geometry_point(x: float, y: float) -> SpatialGeometry:
    """Small convenience constructor used by adapters and downstream callers."""

    return SpatialGeometry(kind="point", coordinates=((float(x), float(y)),))
