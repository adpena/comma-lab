# SPDX-License-Identifier: MIT
"""Deterministic nested prefixes for byte-closed signed-int8 coefficients.

The machine-oriented stopping rule follows Wuyuan Xie, Zhenming Li, Ye Liu,
Jian Jin, Yun Song, and Miaohui Wang (2026), *The Last Byte: Learning Just
Enough for Machine-Oriented Image Compression*, DOI
10.1609/aaai.v40i19.38635: stop removing precision at the receiver/evaluator
boundary, not at a coefficient-domain proxy.

The dead-zone family is a clean-room scalar construction informed by Shaohui
Li, Han Li, Wenrui Dai, Chenglin Li, Junni Zou, and Hongkai Xiong (2023),
*Learned Progressive Image Compression With Dead-Zone Quantizers*, DOI
10.1109/TCSVT.2022.3229701.  That paper motivates analytic dead zones for a
Laplacian source.  Pact's concrete dyadic tail schedule below is our own
derivation; no paper is claimed to establish it for these witness weights.

For a zero-centred Laplace fit with scale ``b``, ``P(|X| >= t)=exp(-t/b)``.
At prefix plane ``k`` we set the surviving-tail target to ``2**-k`` and hence
derive ``t_k = b*k*log(2)``.  Rounding ``t_k`` upward to the current dyadic
step makes every threshold a boundary in the coarser lattice.  Together with
sign-magnitude truncation, that gives a deterministic nested dead-zone chain.
The exact receiver oracle, not this source model, decides admissibility.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

PrefixFamily = Literal["uniform", "laplace_dead_zone"]
PREFIX_FAMILIES: tuple[PrefixFamily, ...] = ("uniform", "laplace_dead_zone")
MAX_INT8_PREFIX_PLANES = 8


@dataclass(frozen=True)
class LaplaceHistogramFit:
    """Content-addressed zero-centred Laplace fit for one int8 section."""

    count: int
    zero_count: int
    mean_abs: float
    scale_b: float
    histogram_sha256: str


@dataclass(frozen=True)
class CoefficientSection:
    """One counted int8 coefficient section inside the two LVLS1 streams."""

    name: str
    stream: Literal["base", "code"]
    offset: int
    count: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PrefixMeasurement:
    """Exact receiver/evaluator row used by the last-safe-plane selectors."""

    section: str
    family: PrefixFamily
    bits_removed: int
    archive_bytes: int
    d_seg: float
    d_pose: float


def _as_int8_coefficients(values: np.ndarray | Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.int8:
        raise TypeError(f"coefficients must have dtype int8; got {array.dtype}")
    if array.size == 0:
        raise ValueError("coefficient section must be non-empty")
    return np.ascontiguousarray(array)


def fit_laplace_histogram(values: np.ndarray | Sequence[int]) -> LaplaceHistogramFit:
    """Fit ``b = mean(abs(q))`` and custody-hash the exact 256-bin histogram.

    ``b`` is derived directly by minimizing the zero-centred Laplace negative
    log likelihood ``n*log(2b) + sum(abs(q))/b``.  No learned model, seed, or
    user-set scale enters the fit.
    """

    q = _as_int8_coefficients(values).reshape(-1)
    q16 = q.astype(np.int16)
    histogram = np.bincount(q16 + 128, minlength=256).astype("<u8", copy=False)
    mean_abs = float(np.mean(np.abs(q16), dtype=np.float64))
    return LaplaceHistogramFit(
        count=int(q.size),
        zero_count=int(histogram[128]),
        mean_abs=mean_abs,
        scale_b=mean_abs,
        histogram_sha256=hashlib.sha256(histogram.tobytes()).hexdigest(),
    )


def dead_zone_threshold(*, bits_removed: int, fit: LaplaceHistogramFit) -> int:
    """Return the dyadic histogram-fit threshold for one prefix plane."""

    _validate_bits_removed(bits_removed)
    if fit.count <= 0 or not math.isfinite(fit.scale_b) or fit.scale_b < 0.0:
        raise ValueError("Laplace fit must have finite non-negative scale and positive count")
    if bits_removed == 0:
        return 0
    if bits_removed == MAX_INT8_PREFIX_PLANES:
        return 1 << MAX_INT8_PREFIX_PLANES
    step = 1 << bits_removed
    raw_threshold = fit.scale_b * bits_removed * math.log(2.0)
    lattice_threshold = math.ceil(raw_threshold / step) * step
    # Sign-magnitude truncation already has a one-step central cell.  This
    # lower bound makes the fitted law explicit when the empirical scale is 0.
    return max(step, lattice_threshold)


def quantize_prefix(
    values: np.ndarray | Sequence[int],
    *,
    bits_removed: int,
    family: PrefixFamily,
    fit: LaplaceHistogramFit | None = None,
) -> np.ndarray:
    """Remove ``bits_removed`` low planes without changing int8 storage.

    ``uniform`` clears low bits in the two's-complement byte stream.  It is the
    literal embedded binary prefix of the existing payload.  ``laplace_dead_zone``
    uses sign-magnitude truncation plus the histogram-derived central threshold.
    Plane 0 is byte-identical and plane 8 is the all-zero completion state.
    """

    q = _as_int8_coefficients(values)
    _validate_bits_removed(bits_removed)
    if family not in PREFIX_FAMILIES:
        raise ValueError(f"unknown prefix family: {family!r}")
    if family == "uniform" and fit is not None:
        raise ValueError("uniform prefix does not accept a Laplace fit")
    if family == "laplace_dead_zone":
        fitted = fit_laplace_histogram(q)
        if fit is not None and fit != fitted:
            raise ValueError(
                "caller-supplied Laplace fit does not match the exact coefficient histogram"
            )
        return _quantize_dead_zone_with_fit(q, bits_removed=bits_removed, fit=fitted)
    if bits_removed == 0:
        return q.copy()
    if bits_removed == MAX_INT8_PREFIX_PLANES:
        return np.zeros_like(q)

    step = 1 << bits_removed
    if family == "uniform":
        raw = q.view(np.uint8)
        mask = np.uint8(0xFF ^ (step - 1))
        return np.bitwise_and(raw, mask).view(np.int8).copy()

    raise AssertionError("registered prefix family was not handled")


def _quantize_dead_zone_with_fit(
    q: np.ndarray, *, bits_removed: int, fit: LaplaceHistogramFit
) -> np.ndarray:
    """Internal fixed-fit map used to prove nesting against the original section."""

    if bits_removed == 0:
        return q.copy()
    if bits_removed == MAX_INT8_PREFIX_PLANES:
        return np.zeros_like(q)
    step = 1 << bits_removed
    threshold = dead_zone_threshold(bits_removed=bits_removed, fit=fit)
    q16 = q.astype(np.int16)
    magnitude = np.abs(q16)
    truncated = (magnitude // step) * step
    truncated[magnitude < threshold] = 0
    signed = np.sign(q16) * truncated
    return signed.astype(np.int8).reshape(q.shape)


def generate_prefix_chain(
    values: np.ndarray | Sequence[int], *, family: PrefixFamily
) -> tuple[np.ndarray, ...]:
    """Generate planes 0..8 and fail closed if the chain is not nested."""

    q = _as_int8_coefficients(values)
    fit = fit_laplace_histogram(q) if family == "laplace_dead_zone" else None
    chain = tuple(
        _quantize_dead_zone_with_fit(q, bits_removed=k, fit=fit)
        if fit is not None
        else quantize_prefix(q, bits_removed=k, family=family)
        for k in range(MAX_INT8_PREFIX_PLANES + 1)
    )
    for k in range(MAX_INT8_PREFIX_PLANES):
        coarsened = (
            _quantize_dead_zone_with_fit(chain[k], bits_removed=k + 1, fit=fit)
            if fit is not None
            else quantize_prefix(chain[k], bits_removed=k + 1, family=family)
        )
        if not np.array_equal(coarsened, chain[k + 1]):
            raise RuntimeError(f"{family} prefix chain is not nested at planes {k}->{k + 1}")
    return chain


def coefficient_sections(
    manifest: Mapping[str, Any],
    *,
    base_raw_len: int,
    code_raw_len: int,
    eval_pairs: int | None = None,
) -> tuple[CoefficientSection, ...]:
    """Re-derive exact LVLS1 sections, optionally pair-matching the code stream.

    Shared base sections are counted once and affect every decoded pair.  The
    code stream is pair-local.  An n-pair evaluator probe must therefore expose
    only the first ``2*eval_pairs`` code rows; changing the remaining rows would
    create archive-byte savings from frames the probe never scored.
    """

    order = manifest.get("base_param_order")
    shapes = manifest.get("base_shapes")
    code_shape = manifest.get("code_shape")
    if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
        raise ValueError("manifest base_param_order must be a list of strings")
    if len(order) != len(set(order)):
        raise ValueError("manifest base_param_order contains duplicate names")
    if not isinstance(shapes, Mapping) or code_shape is None:
        raise ValueError("manifest lacks base_shapes or code_shape")

    sections: list[CoefficientSection] = []
    offset = 0
    for name in order:
        if name not in shapes:
            raise ValueError(f"manifest lacks shape for base section {name!r}")
        shape = _validated_shape(shapes[name], context=name)
        count = math.prod(shape)
        sections.append(CoefficientSection(name, "base", offset, count, shape))
        offset += count
    if offset != int(base_raw_len):
        raise ValueError(
            f"base manifest accounts for {offset} int8 values but stream has {base_raw_len}"
        )

    shape = _validated_shape(code_shape, context="code")
    count = math.prod(shape)
    if count != int(code_raw_len):
        raise ValueError(
            f"code manifest accounts for {count} int8 values but stream has {code_raw_len}"
        )
    if eval_pairs is None:
        sections.append(CoefficientSection("code", "code", 0, count, shape))
    else:
        n_pairs = manifest.get("n_pairs")
        if isinstance(eval_pairs, bool) or not isinstance(eval_pairs, int) or eval_pairs <= 0:
            raise ValueError("eval_pairs must be a positive integer")
        if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs <= 0:
            raise ValueError("manifest n_pairs must be a positive integer for pair-scoped code")
        if shape[0] != 2 * n_pairs:
            raise ValueError(
                f"code_shape {shape} is not the required (2*n_pairs, ...) layout"
            )
        if eval_pairs > n_pairs:
            raise ValueError(f"eval_pairs {eval_pairs} exceeds manifest n_pairs {n_pairs}")
        scoped_shape = (2 * eval_pairs, *shape[1:])
        sections.append(
            CoefficientSection(
                "code_scored_pair_prefix",
                "code",
                0,
                math.prod(scoped_shape),
                scoped_shape,
            )
        )
    return tuple(sections)


def read_section(
    base_raw: bytes, code_raw: bytes, section: CoefficientSection
) -> np.ndarray:
    """Return a copied int8 view of one exactly bounded section."""

    stream = base_raw if section.stream == "base" else code_raw
    stop = section.offset + section.count
    if section.offset < 0 or section.count <= 0 or stop > len(stream):
        raise ValueError(f"section {section.name!r} is outside its {section.stream} stream")
    return np.frombuffer(stream[section.offset:stop], dtype=np.int8).copy().reshape(section.shape)


def replace_section(
    base_raw: bytes,
    code_raw: bytes,
    section: CoefficientSection,
    replacement: np.ndarray | Sequence[int],
) -> tuple[bytes, bytes]:
    """Replace exactly one section while preserving all other raw bytes."""

    q = _as_int8_coefficients(replacement)
    if q.shape != section.shape:
        raise ValueError(
            f"replacement shape {q.shape} does not match {section.name!r} {section.shape}"
        )
    target = bytearray(base_raw if section.stream == "base" else code_raw)
    stop = section.offset + section.count
    if section.offset < 0 or stop > len(target):
        raise ValueError(f"section {section.name!r} is outside its {section.stream} stream")
    target[section.offset:stop] = q.tobytes(order="C")
    if section.stream == "base":
        return bytes(target), bytes(code_raw)
    return bytes(base_raw), bytes(target)


def component_safe(
    measurement: PrefixMeasurement,
    baseline: PrefixMeasurement,
    *,
    seg_tolerance: float,
    pose_tolerance: float,
) -> bool:
    """Exact componentwise Pareto guard; no weighted-score compensation is allowed."""

    _validate_measurement(baseline, context="baseline")
    _validate_measurement(measurement, context="measurement")
    for name, value in (
        ("measurement.d_seg", measurement.d_seg),
        ("measurement.d_pose", measurement.d_pose),
        ("baseline.d_seg", baseline.d_seg),
        ("baseline.d_pose", baseline.d_pose),
        ("seg_tolerance", seg_tolerance),
        ("pose_tolerance", pose_tolerance),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if seg_tolerance < 0.0 or pose_tolerance < 0.0:
        raise ValueError("component tolerances must be non-negative")
    return (
        measurement.d_seg <= baseline.d_seg + seg_tolerance
        and measurement.d_pose <= baseline.d_pose + pose_tolerance
    )


def select_last_safe_plane(
    measurements: Sequence[PrefixMeasurement],
    baseline: PrefixMeasurement,
    *,
    seg_tolerance: float,
    pose_tolerance: float,
) -> PrefixMeasurement | None:
    """Select the coarsest exact-safe plane, independent of byte monotonicity."""

    _validate_selector_rows(measurements, baseline)
    safe = [
        row
        for row in measurements
        if row.bits_removed > 0
        and component_safe(
            row,
            baseline,
            seg_tolerance=seg_tolerance,
            pose_tolerance=pose_tolerance,
        )
    ]
    if not safe:
        return None
    return max(safe, key=lambda row: (row.bits_removed, baseline.archive_bytes - row.archive_bytes))


def select_best_byte_safe(
    measurements: Sequence[PrefixMeasurement],
    baseline: PrefixMeasurement,
    *,
    seg_tolerance: float,
    pose_tolerance: float,
) -> PrefixMeasurement | None:
    """Select the smallest exact archive among safe planes that save at least one byte."""

    _validate_selector_rows(measurements, baseline)
    admissible = [
        row
        for row in measurements
        if row.bits_removed > 0
        and row.archive_bytes < baseline.archive_bytes
        and component_safe(
            row,
            baseline,
            seg_tolerance=seg_tolerance,
            pose_tolerance=pose_tolerance,
        )
    ]
    if not admissible:
        return None
    return min(admissible, key=lambda row: (row.archive_bytes, -row.bits_removed, row.family))


def _validate_measurement(row: PrefixMeasurement, *, context: str) -> None:
    if row.family not in PREFIX_FAMILIES:
        raise ValueError(f"{context}.family is not a registered prefix family")
    if not 0 <= row.bits_removed <= MAX_INT8_PREFIX_PLANES:
        raise ValueError(f"{context}.bits_removed is outside the int8 prefix range")
    if row.archive_bytes < 0:
        raise ValueError(f"{context}.archive_bytes must be non-negative")
    for name, value in (("d_seg", row.d_seg), ("d_pose", row.d_pose)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{context}.{name} must be finite and non-negative")


def _validate_selector_rows(
    measurements: Sequence[PrefixMeasurement], baseline: PrefixMeasurement
) -> None:
    _validate_measurement(baseline, context="baseline")
    if baseline.bits_removed != 0:
        raise ValueError("selector baseline must be prefix plane 0")
    for index, row in enumerate(measurements):
        _validate_measurement(row, context=f"measurements[{index}]")
        if row.bits_removed == 0:
            raise ValueError("selector candidate rows must have bits_removed > 0")
        if row.section != baseline.section or row.family != baseline.family:
            raise ValueError("selector rows must match the baseline section and family")


def _validate_bits_removed(bits_removed: int) -> None:
    if isinstance(bits_removed, bool) or not isinstance(bits_removed, int):
        raise TypeError("bits_removed must be an integer")
    if not 0 <= bits_removed <= MAX_INT8_PREFIX_PLANES:
        raise ValueError(
            f"bits_removed must be in [0,{MAX_INT8_PREFIX_PLANES}]; got {bits_removed}"
        )


def _validated_shape(raw: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"shape for {context!r} must be a non-empty list")
    shape: list[int] = []
    for dim in raw:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"shape for {context!r} has invalid dimension {dim!r}")
        shape.append(dim)
    return tuple(shape)


__all__ = [
    "MAX_INT8_PREFIX_PLANES",
    "PREFIX_FAMILIES",
    "CoefficientSection",
    "LaplaceHistogramFit",
    "PrefixFamily",
    "PrefixMeasurement",
    "coefficient_sections",
    "component_safe",
    "dead_zone_threshold",
    "fit_laplace_histogram",
    "generate_prefix_chain",
    "quantize_prefix",
    "read_section",
    "replace_section",
    "select_best_byte_safe",
    "select_last_safe_plane",
]
