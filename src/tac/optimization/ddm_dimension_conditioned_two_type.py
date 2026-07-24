# SPDX-License-Identifier: MIT
"""Dimension-conditioned five-type primitives for the DDM PF2 charter.

The scorer-native split is explicit:

* a discrete event skeleton records ``predicted -> target`` argmax events;
* a connection transports continuous state before native fiber coding;
* fibers remain opaque byte strings owned by their native coders;
* gauge directions are receiver-canonical zero-byte slack; and
* residuals carry exceptions against decoder-derived context.

Nothing in this module calls a scorer.  The codecs are deterministic NumPy and
stdlib reference paths with strict parse-back.  The Euclidean pose-stat map is
retained only as a labeled identity-metric control.  It cannot become
verdict-bearing until the harness measures and applies the scorer-native
margin-Fisher, PoseNet-quadratic, and dual-metric geometry.
"""

from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass
from typing import Final

import brotli
import numpy as np

VISIBILITY_CLASSES: Final = (
    "ker(A)-invisible",
    "seg-visible",
    "pose-visible",
    "both",
)
TEMPORAL_CLASSES: Final = (
    "STATIC_IN_IMAGE",
    "STATIC_IN_XI_PROXY",
    "TRANSIENT",
)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_STRATA: Final = ("cell", "boundary")
REPRESENTATION_TYPES: Final = (
    "SKELETON",
    "CONNECTION",
    "FIBER",
    "GAUGE",
    "RESIDUAL",
)
METRIC_ACTIVE_SCORER_GEOMETRY: Final = "METRIC_ACTIVE_SCORER_GEOMETRY"
IDENTICAL_CONTENT_CODER_CONTROL: Final = "IDENTICAL_CONTENT_CODER_CONTROL"
IDENTITY_EUCLIDEAN_CONTROL: Final = "IDENTITY_EUCLIDEAN_CONTROL"
EVENT_SENTINEL: Final = 255

_FLAT_HEADER = struct.Struct("<4sBHHHB")
_PROGRAM_HEADER = struct.Struct("<4sBHHHB")
_FLAT_MAGIC = b"PFF3"
_PROGRAM_MAGIC = b"PFE3"
_VERSION = 1
_LZMA_FILTERS = [
    {"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}
]
_CODEC_IDS = {"raw": 0, "zlib9": 1, "brotli11": 2, "lzma_raw": 3}
_CODEC_NAMES = {value: key for key, value in _CODEC_IDS.items()}


class DimensionConditionedTwoTypeError(ValueError):
    """Raised when a PF2 partition, projection, or parse-back invariant fails."""


@dataclass(frozen=True)
class RealCodedPayload:
    """One deterministic real-coder selection including its one-byte tag."""

    codec: str
    payload: bytes
    raw_bytes: int
    candidate_bytes: dict[str, int]


@dataclass(frozen=True)
class EventRateRace:
    """Equal-content event-skeleton versus flat-native rate result."""

    event_count: int
    program_raw: bytes
    program_coded: RealCodedPayload
    flat_raw: bytes
    flat_coded: RealCodedPayload

    @property
    def delta_program_minus_flat_bytes(self) -> int:
        return len(self.program_coded.payload) - len(self.flat_coded.payload)


@dataclass(frozen=True)
class FormulationMetricDisposition:
    """Fail-closed verdict eligibility under the scorer-metric law."""

    metric_status: str
    verdict_eligible: bool
    waterfill_eligible: bool
    reason: str


def resolve_formulation_metric_disposition(
    metric_status: str,
    *,
    identical_content_proven: bool,
) -> FormulationMetricDisposition:
    """Classify whether a formulation may carry a verdict.

    A measured scorer-native geometry is eligible.  A rate-only coder control
    is eligible only when strict parse-back proves identical semantic content,
    because its scorer debt cancels exactly.  Identity/Euclidean geometry is
    always an instance-scoped control and is excluded from routing and
    water-filling even if its final hard-score readback happens to improve.
    """

    if not isinstance(metric_status, str):
        raise DimensionConditionedTwoTypeError("metric status must be a string")
    if not isinstance(identical_content_proven, bool):
        raise DimensionConditionedTwoTypeError(
            "identical-content custody must be bool"
        )
    if metric_status == METRIC_ACTIVE_SCORER_GEOMETRY:
        return FormulationMetricDisposition(
            metric_status=metric_status,
            verdict_eligible=True,
            waterfill_eligible=True,
            reason=(
                "proposal and readback use measured scorer-native geometry"
            ),
        )
    if metric_status == IDENTICAL_CONTENT_CODER_CONTROL:
        if not identical_content_proven:
            raise DimensionConditionedTwoTypeError(
                "coder-control verdict requires identical-content parse-back"
            )
        return FormulationMetricDisposition(
            metric_status=metric_status,
            verdict_eligible=True,
            waterfill_eligible=False,
            reason=(
                "strictly identical semantic content cancels distortion; only "
                "exact counted-byte rate differs"
            ),
        )
    if metric_status == IDENTITY_EUCLIDEAN_CONTROL:
        return FormulationMetricDisposition(
            metric_status=metric_status,
            verdict_eligible=False,
            waterfill_eligible=False,
            reason=(
                "identity/Euclidean geometry is an instance-scoped naive "
                "control pending a metric-active rerun"
            ),
        )
    raise DimensionConditionedTwoTypeError(
        f"unknown formulation metric status: {metric_status!r}"
    )


def _validate_camera_projection_inputs(
    base_camera: np.ndarray,
    winner_camera: np.ndarray,
    camera_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(base_camera)
    winner = np.asarray(winner_camera)
    support = np.asarray(camera_support, dtype=bool)
    if (
        base.dtype != np.uint8
        or winner.dtype != np.uint8
        or base.shape != winner.shape
        or base.ndim != 5
        or base.shape[1] != 2
        or base.shape[-1] != 3
        or support.shape != base[:, 1].shape[:3]
    ):
        raise DimensionConditionedTwoTypeError("camera/support geometry differs")
    if not np.array_equal(base[:, 0], winner[:, 0]):
        raise DimensionConditionedTwoTypeError("parent winner changed frame 0")
    return base, winner, support


def support_rgb_moments(camera: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Return per-pair frame-1 support ``[mean_R/G/B, std_R/G/B]``."""

    value = np.asarray(camera)
    mask = np.asarray(support, dtype=bool)
    if (
        value.dtype != np.uint8
        or value.ndim != 5
        or value.shape[1] != 2
        or value.shape[-1] != 3
        or mask.shape != value[:, 1].shape[:3]
    ):
        raise DimensionConditionedTwoTypeError("support moment geometry differs")
    rows = []
    for pair_index in range(value.shape[0]):
        if not np.any(mask[pair_index]):
            raise DimensionConditionedTwoTypeError("empty per-pair support")
        pixels = value[pair_index, 1][mask[pair_index]].astype(np.float64)
        rows.append(np.concatenate((pixels.mean(axis=0), pixels.std(axis=0))))
    return np.asarray(rows, dtype=np.float64)


def moment_constrained_hood_projection(
    *,
    base_camera: np.ndarray,
    winner_camera: np.ndarray,
    camera_support: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """Apply the identity-metric hood projection control.

    For each pair and RGB channel, the deterministic map first moves a fixed
    fraction ``alpha`` from the MENU1 winner toward the base field, then
    normalizes the moved support back to the winner's support mean and standard
    deviation before uint8 realization.  It is a *moment-constrained
    formulation*, not a PoseNet-safety assertion: the harness must measure the
    exact official resize/YUV6 path and joint action at n600.  This Euclidean
    moment normalization is not a PoseNet-quadratic/Fisher projection and must
    never be routed as one.
    """

    base, winner, support = _validate_camera_projection_inputs(
        base_camera, winner_camera, camera_support
    )
    if not np.isfinite(alpha) or not 0.0 < float(alpha) <= 1.0:
        raise DimensionConditionedTwoTypeError("alpha must be finite in (0,1]")
    result = winner.copy()
    for pair_index in range(result.shape[0]):
        mask = support[pair_index]
        if not np.any(mask):
            raise DimensionConditionedTwoTypeError("empty per-pair hood support")
        for channel in range(3):
            parent = winner[pair_index, 1, :, :, channel][mask].astype(np.float64)
            target = base[pair_index, 1, :, :, channel][mask].astype(np.float64)
            moved = parent + float(alpha) * (target - parent)
            moved_std = float(moved.std())
            parent_std = float(parent.std())
            if moved_std > np.finfo(np.float64).eps:
                moved = (
                    (moved - float(moved.mean())) * (parent_std / moved_std)
                    + float(parent.mean())
                )
            else:
                moved = parent
            realized = np.clip(np.rint(moved), 0.0, 255.0).astype(np.uint8)
            result[pair_index, 1, :, :, channel][mask] = realized
    if not np.array_equal(result[:, 0], base[:, 0]):
        raise DimensionConditionedTwoTypeError("projection changed frame 0")
    if not np.array_equal(result[:, 1][~support], winner[:, 1][~support]):
        raise DimensionConditionedTwoTypeError("projection changed outside support")
    return result


def _uleb128(value: int) -> bytes:
    if int(value) < 0:
        raise DimensionConditionedTwoTypeError("ULEB128 value must be non-negative")
    out = bytearray()
    remaining = int(value)
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        out.append(byte | (0x80 if remaining else 0))
        if not remaining:
            return bytes(out)


def _read_uleb128(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise DimensionConditionedTwoTypeError("truncated or oversized ULEB128")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _validate_event_content(
    event_codes: np.ndarray, *, n_classes: int
) -> np.ndarray:
    value = np.asarray(event_codes)
    if (
        value.dtype != np.uint8
        or value.ndim != 3
        or any(int(size) <= 0 or int(size) > 65_535 for size in value.shape)
        or not 2 <= int(n_classes) <= 15
    ):
        raise DimensionConditionedTwoTypeError("event content geometry differs")
    active = value != EVENT_SENTINEL
    if np.any(value[active] >= int(n_classes) ** 2):
        raise DimensionConditionedTwoTypeError("event transition code is out of range")
    if np.any(
        value[active] // int(n_classes) == value[active] % int(n_classes)
    ):
        raise DimensionConditionedTwoTypeError("event content includes a non-flip")
    return np.ascontiguousarray(value)


def encode_flat_event_content(
    event_codes: np.ndarray, *, n_classes: int = 5
) -> bytes:
    """Serialize exact events in pair-major raster order."""

    value = _validate_event_content(event_codes, n_classes=n_classes)
    pairs, height, width = (int(v) for v in value.shape)
    active = value != EVENT_SENTINEL
    packed = np.packbits(active.reshape(-1), bitorder="little").tobytes()
    codes = value.reshape(-1)[active.reshape(-1)].tobytes()
    return (
        _FLAT_HEADER.pack(
            _FLAT_MAGIC, _VERSION, pairs, height, width, int(n_classes)
        )
        + packed
        + codes
    )


def decode_flat_event_content(payload: bytes) -> np.ndarray:
    """Strict inverse of :func:`encode_flat_event_content`."""

    if len(payload) < _FLAT_HEADER.size:
        raise DimensionConditionedTwoTypeError("flat event payload is truncated")
    magic, version, pairs, height, width, n_classes = _FLAT_HEADER.unpack_from(
        payload
    )
    if (
        magic != _FLAT_MAGIC
        or version != _VERSION
        or min(pairs, height, width) <= 0
        or not 2 <= n_classes <= 15
    ):
        raise DimensionConditionedTwoTypeError("flat event header differs")
    total = int(pairs) * int(height) * int(width)
    mask_bytes = (total + 7) // 8
    body_offset = _FLAT_HEADER.size
    if len(payload) < body_offset + mask_bytes:
        raise DimensionConditionedTwoTypeError("flat event mask is truncated")
    active = np.unpackbits(
        np.frombuffer(
            payload[body_offset : body_offset + mask_bytes], dtype=np.uint8
        ),
        bitorder="little",
        count=total,
    ).astype(bool, copy=False)
    code_payload = payload[body_offset + mask_bytes :]
    if len(code_payload) != int(np.count_nonzero(active)):
        raise DimensionConditionedTwoTypeError("flat event code count differs")
    codes = np.frombuffer(code_payload, dtype=np.uint8)
    if np.any(codes >= int(n_classes) ** 2) or np.any(
        codes // int(n_classes) == codes % int(n_classes)
    ):
        raise DimensionConditionedTwoTypeError("flat event code is invalid")
    result = np.full(total, EVENT_SENTINEL, dtype=np.uint8)
    result[active] = codes
    return result.reshape((int(pairs), int(height), int(width)))


def encode_temporal_event_skeleton(
    event_codes: np.ndarray, *, n_classes: int = 5
) -> bytes:
    """Encode events as transition-tokenized pixel tracks with pair deltas."""

    value = _validate_event_content(event_codes, n_classes=n_classes)
    pairs, height, width = (int(v) for v in value.shape)
    sites = height * width
    body = bytearray(
        _PROGRAM_HEADER.pack(
            _PROGRAM_MAGIC, _VERSION, pairs, height, width, int(n_classes)
        )
    )
    active_codes = [
        code
        for code in range(int(n_classes) ** 2)
        if code // int(n_classes) != code % int(n_classes)
        and np.any(value == code)
    ]
    body.extend(_uleb128(len(active_codes)))
    flat = value.reshape(-1)
    for code in active_codes:
        indices = np.flatnonzero(flat == code)
        pair_ids = indices // sites
        pixels = indices % sites
        order = np.lexsort((pair_ids, pixels))
        pair_ids = pair_ids[order]
        pixels = pixels[order]
        starts = np.r_[0, np.flatnonzero(np.diff(pixels)) + 1]
        stops = np.r_[starts[1:], len(pixels)]
        body.append(code)
        body.extend(_uleb128(len(starts)))
        previous_pixel = -1
        for start, stop in zip(starts, stops, strict=True):
            pixel = int(pixels[start])
            body.extend(_uleb128(pixel - previous_pixel - 1))
            body.extend(_uleb128(int(stop - start)))
            previous_pair = -1
            for pair in pair_ids[start:stop]:
                pair_value = int(pair)
                body.extend(_uleb128(pair_value - previous_pair - 1))
                previous_pair = pair_value
            previous_pixel = pixel
    return bytes(body)


def decode_temporal_event_skeleton(payload: bytes) -> np.ndarray:
    """Strict inverse of :func:`encode_temporal_event_skeleton`."""

    if len(payload) < _PROGRAM_HEADER.size:
        raise DimensionConditionedTwoTypeError("event skeleton is truncated")
    magic, version, pairs, height, width, n_classes = _PROGRAM_HEADER.unpack_from(
        payload
    )
    if (
        magic != _PROGRAM_MAGIC
        or version != _VERSION
        or min(pairs, height, width) <= 0
        or not 2 <= n_classes <= 15
    ):
        raise DimensionConditionedTwoTypeError("event skeleton header differs")
    result = np.full(
        (int(pairs), int(height), int(width)),
        EVENT_SENTINEL,
        dtype=np.uint8,
    )
    offset = _PROGRAM_HEADER.size
    code_count, offset = _read_uleb128(payload, offset)
    seen_codes: set[int] = set()
    sites = int(height) * int(width)
    for _ in range(code_count):
        if offset >= len(payload):
            raise DimensionConditionedTwoTypeError("event skeleton code is truncated")
        code = payload[offset]
        offset += 1
        if (
            code in seen_codes
            or code >= int(n_classes) ** 2
            or code // int(n_classes) == code % int(n_classes)
        ):
            raise DimensionConditionedTwoTypeError("event skeleton code differs")
        seen_codes.add(code)
        group_count, offset = _read_uleb128(payload, offset)
        previous_pixel = -1
        for _ in range(group_count):
            pixel_gap, offset = _read_uleb128(payload, offset)
            pixel = previous_pixel + 1 + pixel_gap
            if pixel >= sites:
                raise DimensionConditionedTwoTypeError(
                    "event skeleton pixel escaped geometry"
                )
            event_count, offset = _read_uleb128(payload, offset)
            if event_count <= 0:
                raise DimensionConditionedTwoTypeError(
                    "event skeleton group is empty"
                )
            previous_pair = -1
            row, col = divmod(pixel, int(width))
            for _ in range(event_count):
                pair_gap, offset = _read_uleb128(payload, offset)
                pair = previous_pair + 1 + pair_gap
                if pair >= int(pairs) or result[pair, row, col] != EVENT_SENTINEL:
                    raise DimensionConditionedTwoTypeError(
                        "event skeleton pair is invalid or duplicated"
                    )
                result[pair, row, col] = code
                previous_pair = pair
            previous_pixel = pixel
    if offset != len(payload):
        raise DimensionConditionedTwoTypeError("event skeleton has trailing bytes")
    return result


def real_code(payload: bytes) -> RealCodedPayload:
    """Select the shortest deterministic real coder with an explicit tag."""

    raw = bytes(payload)
    variants = {
        "raw": raw,
        "zlib9": zlib.compress(raw, 9),
        "brotli11": brotli.compress(raw, quality=11),
        "lzma_raw": lzma.compress(
            raw, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS
        ),
    }
    selected = min(variants, key=lambda name: (len(variants[name]), name))
    coded = bytes([_CODEC_IDS[selected]]) + variants[selected]
    return RealCodedPayload(
        codec=selected,
        payload=coded,
        raw_bytes=len(raw),
        candidate_bytes={name: len(value) + 1 for name, value in variants.items()},
    )


def real_decode(payload: bytes) -> bytes:
    """Decode a payload emitted by :func:`real_code`."""

    if not payload or payload[0] not in _CODEC_NAMES:
        raise DimensionConditionedTwoTypeError("real-coder tag differs")
    name = _CODEC_NAMES[payload[0]]
    body = payload[1:]
    if name == "raw":
        return body
    if name == "zlib9":
        return zlib.decompress(body)
    if name == "brotli11":
        return brotli.decompress(body)
    if name == "lzma_raw":
        return lzma.decompress(
            body, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS
        )
    raise DimensionConditionedTwoTypeError("unreachable real-coder tag")


def race_event_coders(
    event_codes: np.ndarray, *, n_classes: int = 5
) -> EventRateRace:
    """Run the equal-content event-skeleton versus flat-native rate race."""

    value = _validate_event_content(event_codes, n_classes=n_classes)
    program_raw = encode_temporal_event_skeleton(value, n_classes=n_classes)
    flat_raw = encode_flat_event_content(value, n_classes=n_classes)
    program_coded = real_code(program_raw)
    flat_coded = real_code(flat_raw)
    program_roundtrip = decode_temporal_event_skeleton(
        real_decode(program_coded.payload)
    )
    flat_roundtrip = decode_flat_event_content(real_decode(flat_coded.payload))
    if not np.array_equal(program_roundtrip, value) or not np.array_equal(
        flat_roundtrip, value
    ):
        raise DimensionConditionedTwoTypeError(
            "equal-content event coder parse-back differs"
        )
    return EventRateRace(
        event_count=int(np.count_nonzero(value != EVENT_SENTINEL)),
        program_raw=program_raw,
        program_coded=program_coded,
        flat_raw=flat_raw,
        flat_coded=flat_coded,
    )


__all__ = [
    "CLASS_NAMES",
    "CLASS_STRATA",
    "EVENT_SENTINEL",
    "IDENTICAL_CONTENT_CODER_CONTROL",
    "IDENTITY_EUCLIDEAN_CONTROL",
    "METRIC_ACTIVE_SCORER_GEOMETRY",
    "REPRESENTATION_TYPES",
    "TEMPORAL_CLASSES",
    "VISIBILITY_CLASSES",
    "DimensionConditionedTwoTypeError",
    "EventRateRace",
    "FormulationMetricDisposition",
    "RealCodedPayload",
    "decode_flat_event_content",
    "decode_temporal_event_skeleton",
    "encode_flat_event_content",
    "encode_temporal_event_skeleton",
    "moment_constrained_hood_projection",
    "race_event_coders",
    "real_code",
    "real_decode",
    "resolve_formulation_metric_disposition",
    "support_rgb_moments",
]
