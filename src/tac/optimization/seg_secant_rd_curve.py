# SPDX-License-Identifier: MIT
"""Deterministic coarsening primitives for measured Seg rate-distortion curves.

The functions in this module do not score candidates and do not claim receiver
closure.  They transform a custodied uint8 source/predictor pair, measure the
range-coordinate payload with codec parse-back, and derive adjacent secants.
Frozen scorer callbacks remain the authority in the calling measurement tool.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

BREAK_EVEN_BYTES_PER_DSEG = 100.0 * 37_545_489.0 / 25.0
CONTEST_PAIR_COUNT = 600


class SegSecantError(ValueError):
    """Refuse malformed coarsening, codec, or curve inputs."""


@dataclass(frozen=True)
class OperatingPoint:
    """One deterministic residual-coarsening operating point."""

    point_id: str
    family: str
    parameter_name: str
    parameter_value: float | int

    def __post_init__(self) -> None:
        if not self.point_id or not self.family or not self.parameter_name:
            raise SegSecantError("operating-point identifiers must be nonempty")
        value = float(self.parameter_value)
        if not math.isfinite(value) or value < 0:
            raise SegSecantError("operating-point parameter must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "family": self.family,
            "parameter_name": self.parameter_name,
            "parameter_value": self.parameter_value,
        }


def default_operating_points() -> tuple[OperatingPoint, ...]:
    """Return the preregistered four-margin plus three-precision sweep."""

    return (
        OperatingPoint("margin_m0p01", "margin_abandonment", "margin_threshold", 0.01),
        OperatingPoint("margin_m0p03", "margin_abandonment", "margin_threshold", 0.03),
        OperatingPoint("margin_m0p1", "margin_abandonment", "margin_threshold", 0.1),
        OperatingPoint("margin_m0p3", "margin_abandonment", "margin_threshold", 0.3),
        OperatingPoint("precision_drop1", "precision_truncation", "drop_low_bits", 1),
        OperatingPoint("precision_drop2", "precision_truncation", "drop_low_bits", 2),
        OperatingPoint("precision_drop3", "precision_truncation", "drop_low_bits", 3),
        OperatingPoint("spatial_stride8", "spatial_subsample", "sample_stride", 8),
        OperatingPoint("spatial_stride16", "spatial_subsample", "sample_stride", 16),
    )


def _uint8_pair(source: np.ndarray, predictor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(source)
    pred = np.asarray(predictor)
    if src.shape != pred.shape or src.ndim != 3 or src.shape[2] < 1:
        raise SegSecantError("source/predictor must be same-shape nonempty HWC arrays")
    if src.dtype != np.uint8 or pred.dtype != np.uint8:
        raise SegSecantError("source/predictor must be uint8")
    return src, pred


def margin_ordered_abandonment(
    source: np.ndarray,
    predictor: np.ndarray,
    margins: np.ndarray,
    *,
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace low-margin scorer blocks by their predictor blocks.

    ``row_indices`` and ``col_indices`` bind the disjoint resize ownership map.
    Copying a complete owned block preserves exact range reachability without a
    new integer solve.  Low-margin means ``margin < threshold``; equality is
    retained at the source point for deterministic boundary semantics.
    """

    src, pred = _uint8_pair(source, predictor)
    margin = np.asarray(margins, dtype=np.float64)
    rows = np.asarray(row_indices)
    cols = np.asarray(col_indices)
    if rows.ndim != 2 or cols.ndim != 2 or rows.shape[1] == 0 or cols.shape[1] == 0:
        raise SegSecantError("resize ownership indices must be nonempty 2-D arrays")
    if margin.shape != (rows.shape[0], cols.shape[0]):
        raise SegSecantError("margin geometry must match scorer ownership geometry")
    if not np.isfinite(margin).all() or np.any(margin < 0):
        raise SegSecantError("margins must be finite and nonnegative")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise SegSecantError("margin threshold must be finite and nonnegative")
    if np.any(rows < 0) or np.any(rows >= src.shape[0]) or np.any(cols < 0) or np.any(cols >= src.shape[1]):
        raise SegSecantError("resize ownership indices leave camera geometry")

    abandon = margin < float(threshold)
    out = src.copy()
    for row_offset in range(rows.shape[1]):
        for col_offset in range(cols.shape[1]):
            rr = rows[:, row_offset, None]
            cc = cols[None, :, col_offset]
            source_blocks = src[rr, cc, :]
            predictor_blocks = pred[rr, cc, :]
            out[rr, cc, :] = np.where(
                abandon[:, :, None], predictor_blocks, source_blocks
            )
    return out, {
        "abandoned_scorer_pixels": int(np.count_nonzero(abandon)),
        "retained_scorer_pixels": int(abandon.size - np.count_nonzero(abandon)),
        "abandoned_fraction": float(np.mean(abandon)),
        "ordering": "custodied native Seg winner/rival margin ascending",
    }


def truncate_preimage_residual_precision(
    source: np.ndarray,
    predictor: np.ndarray,
    *,
    drop_low_bits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop low bit-planes from signed uint8 preimage residual coefficients.

    Magnitudes are truncated toward zero.  The result remains uint8 and exactly
    reachable because it is itself a concrete camera-lattice preimage.  The
    calling tool counts only its recomputed range-coordinate numerator payload.
    """

    src, pred = _uint8_pair(source, predictor)
    if isinstance(drop_low_bits, bool) or int(drop_low_bits) != drop_low_bits:
        raise SegSecantError("drop_low_bits must be an integer")
    bits = int(drop_low_bits)
    if bits < 0 or bits > 7:
        raise SegSecantError("drop_low_bits must be in [0,7]")
    delta = src.astype(np.int16) - pred.astype(np.int16)
    magnitude = np.abs(delta).astype(np.int16)
    truncated_magnitude = (magnitude >> bits) << bits
    truncated = np.where(delta < 0, -truncated_magnitude, truncated_magnitude)
    reconstructed = pred.astype(np.int16) + truncated
    if np.any(reconstructed < 0) or np.any(reconstructed > 255):
        raise SegSecantError("precision truncation left the uint8 lattice")
    out = reconstructed.astype(np.uint8)
    changed = out != src
    return out, {
        "drop_low_bits": bits,
        "changed_camera_values": int(np.count_nonzero(changed)),
        "changed_camera_fraction": float(np.mean(changed)),
        "residual_coefficients": "signed camera-preimage residual versus generated predictor",
        "rounding": "magnitude truncation toward zero",
    }


def spatial_subsample_preimage_residual(
    source: np.ndarray,
    predictor: np.ndarray,
    *,
    sample_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample a camera residual grid and reconstruct it by separable bilinear interpolation.

    Endpoint rows/columns are always included, so the sample geometry and
    interpolation domain are deterministic for any positive stride.  The
    returned signed-int16 sample grid is the counted frame-1 description; the
    caller verifies its codec parse-back separately.
    """

    src, pred = _uint8_pair(source, predictor)
    if (
        isinstance(sample_stride, bool)
        or int(sample_stride) != sample_stride
        or sample_stride <= 0
    ):
        raise SegSecantError("sample_stride must be a positive integer")
    stride = int(sample_stride)
    ys = np.unique(np.append(np.arange(0, src.shape[0], stride), src.shape[0] - 1))
    xs = np.unique(np.append(np.arange(0, src.shape[1], stride), src.shape[1] - 1))
    delta = src.astype(np.int16) - pred.astype(np.int16)
    samples = np.ascontiguousarray(delta[np.ix_(ys, xs, np.arange(src.shape[2]))])

    full_x = np.arange(src.shape[1])
    full_y = np.arange(src.shape[0])
    horizontal = np.empty((ys.size, src.shape[1], src.shape[2]), dtype=np.float64)
    for sample_row in range(ys.size):
        for channel in range(src.shape[2]):
            horizontal[sample_row, :, channel] = np.interp(
                full_x, xs, samples[sample_row, :, channel]
            )
    upsampled = np.empty(src.shape, dtype=np.float64)
    for column in range(src.shape[1]):
        for channel in range(src.shape[2]):
            upsampled[:, column, channel] = np.interp(
                full_y, ys, horizontal[:, column, channel]
            )
    reconstructed_i16 = pred.astype(np.int16) + np.rint(upsampled).astype(np.int16)
    clipped_values = int(np.count_nonzero((reconstructed_i16 < 0) | (reconstructed_i16 > 255)))
    reconstructed = np.clip(reconstructed_i16, 0, 255).astype(np.uint8)
    return reconstructed, samples, {
        "sample_stride": stride,
        "sample_shape": list(samples.shape),
        "sample_count": int(samples.size),
        "sample_sha256": hashlib.sha256(samples.view(np.uint8)).hexdigest(),
        "sample_dtype": str(samples.dtype),
        "interpolation": "separable float64 numpy.interp then ties-to-even np.rint",
        "endpoint_rows_columns_included": True,
        "clipped_camera_values": clipped_values,
        "counted_frame1_description": "signed int32 encoding of sampled camera-preimage residual",
    }


def _zstd_roundtrip(raw: bytes) -> tuple[bytes, bytes, str]:
    try:
        import zstandard as zstd  # type: ignore[import-not-found]
    except ImportError:
        executable = shutil.which("zstd")
        if executable is None:
            raise SegSecantError("zstandard module or zstd CLI is required") from None
        encoded = subprocess.run(
            [executable, "-19", "--stdout", "--quiet"],
            input=raw,
            capture_output=True,
            check=False,
        )
        if encoded.returncode:
            raise SegSecantError(
                f"zstd-19 encode failed with rc={encoded.returncode}"
            ) from None
        decoded = subprocess.run(
            [executable, "-d", "--stdout", "--quiet"],
            input=encoded.stdout,
            capture_output=True,
            check=False,
        )
        if decoded.returncode:
            raise SegSecantError(
                f"zstd parse-back failed with rc={decoded.returncode}"
            ) from None
        return encoded.stdout, decoded.stdout, "zstd-cli-19"
    compressor = zstd.ZstdCompressor(level=19)
    encoded = compressor.compress(raw)
    decoded = zstd.ZstdDecompressor().decompress(encoded, max_output_size=len(raw))
    return encoded, decoded, f"python-zstandard-{getattr(zstd, '__version__', 'unknown')}-19"


def measure_parseback_payload(
    chosen_numerators: np.ndarray,
    predictor_numerators: np.ndarray,
) -> dict[str, Any]:
    """Compress and parse back one signed-int32 range residual with both codecs."""

    chosen = np.asarray(chosen_numerators)
    predictor = np.asarray(predictor_numerators)
    if chosen.shape != predictor.shape or chosen.ndim != 3:
        raise SegSecantError("chosen/predictor numerators must be same-shape HWC")
    if not np.issubdtype(chosen.dtype, np.integer) or not np.issubdtype(predictor.dtype, np.integer):
        raise SegSecantError("chosen/predictor numerators must be integer arrays")
    residual = chosen.astype(np.int64) - predictor.astype(np.int64)
    info = np.iinfo(np.int32)
    if np.any(residual < info.min) or np.any(residual > info.max):
        raise SegSecantError("numerator residual exceeds signed int32 payload")
    words = np.ascontiguousarray(residual.astype("<i4", copy=False))
    raw = words.tobytes(order="C")

    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment gate
        raise SegSecantError("brotli is required") from exc
    brotli_bytes = bytes(brotli.compress(raw, quality=11))
    try:
        brotli_decoded = bytes(brotli.decompress(brotli_bytes))
    except brotli.error as exc:
        raise SegSecantError("Brotli-Q11 parse-back failed") from exc
    zstd_bytes, zstd_decoded, zstd_codec = _zstd_roundtrip(raw)
    if brotli_decoded != raw or zstd_decoded != raw:
        raise SegSecantError("compressed payload parse-back differs from counted bytes")
    parsed_brotli = np.frombuffer(brotli_decoded, dtype="<i4").reshape(words.shape)
    parsed_zstd = np.frombuffer(zstd_decoded, dtype="<i4").reshape(words.shape)
    if not np.array_equal(parsed_brotli, words) or not np.array_equal(parsed_zstd, words):
        raise SegSecantError("parsed residual geometry/value mismatch")
    return {
        "encoding": "signed little-endian int32 scorer-numerator HWC",
        "shape": list(words.shape),
        "raw_bytes": len(raw),
        "nonzero_values": int(np.count_nonzero(words)),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "brotli_q11": {
            "bytes": len(brotli_bytes),
            "sha256": hashlib.sha256(brotli_bytes).hexdigest(),
            "parseback_sha256": hashlib.sha256(brotli_decoded).hexdigest(),
            "parseback_exact": True,
        },
        "zstd_19": {
            "bytes": len(zstd_bytes),
            "sha256": hashlib.sha256(zstd_bytes).hexdigest(),
            "parseback_sha256": hashlib.sha256(zstd_decoded).hexdigest(),
            "parseback_exact": True,
            "codec": zstd_codec,
        },
        "range_A_only": True,
        "ker_A_payload_bytes": 0,
    }


def summarize_per_class(
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    class_count: int = 5,
) -> dict[str, dict[str, float | int | None]]:
    """Return mismatch counts and conditional distortion for each label class."""

    truth = np.asarray(labels)
    guess = np.asarray(predicted)
    if truth.shape != guess.shape or truth.ndim != 2:
        raise SegSecantError("labels/predicted must be same-shape 2-D arrays")
    if isinstance(class_count, bool) or int(class_count) != class_count or class_count <= 0:
        raise SegSecantError("class_count must be a positive integer")
    result: dict[str, dict[str, float | int | None]] = {}
    for class_id in range(int(class_count)):
        selected = truth == class_id
        pixels = int(np.count_nonzero(selected))
        mismatches = int(np.count_nonzero(selected & (guess != truth)))
        result[str(class_id)] = {
            "pixels": pixels,
            "mismatches": mismatches,
            "d_seg_conditional": None if pixels == 0 else mismatches / pixels,
        }
    return result


def adjacent_seg_secants(
    points: Sequence[Mapping[str, Any]],
    *,
    codec_key: str,
    population_pairs: int = CONTEST_PAIR_COUNT,
) -> list[dict[str, Any]]:
    """Derive adjacent within-family byte-saving/Seg-distortion secants.

    A distortion-increasing move is score-favorable exactly when its saved
    bytes per unit ``d_seg`` exceed ``BREAK_EVEN_BYTES_PER_DSEG``.  This sign is
    derived directly from the contest objective, not inferred from prose.
    """

    if codec_key not in {"brotli_q11_bytes_per_pair", "zstd_19_bytes_per_pair"}:
        raise SegSecantError("unknown aggregate codec key")
    if (
        isinstance(population_pairs, bool)
        or int(population_pairs) != population_pairs
        or population_pairs <= 0
    ):
        raise SegSecantError("population_pairs must be a positive integer")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    reference = [point for point in points if point.get("family") == "reference"]
    if len(reference) > 1:
        raise SegSecantError("curve may contain at most one reference point")
    for point in points:
        family = str(point.get("family", ""))
        if not family:
            raise SegSecantError("curve point lacks family")
        if family != "reference":
            grouped.setdefault(family, []).append(point)
    out: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        family_rows = ([reference[0]] if reference else []) + rows
        ordered = sorted(
            family_rows,
            key=lambda point: (float(point["d_seg"]), -float(point[codec_key]), str(point["point_id"])),
        )
        for low, high in itertools.pairwise(ordered):
            delta_dseg = float(high["d_seg"]) - float(low["d_seg"])
            bytes_saved = float(low[codec_key]) - float(high[codec_key])
            if delta_dseg <= 0 or bytes_saved <= 0:
                continue
            per_pair_ratio = bytes_saved / delta_dseg
            global_ratio = per_pair_ratio * int(population_pairs)
            out.append(
                {
                    "family": family,
                    "codec": codec_key,
                    "lower_distortion_point": str(low["point_id"]),
                    "higher_distortion_point": str(high["point_id"]),
                    "delta_d_seg": delta_dseg,
                    "bytes_saved_per_pair": bytes_saved,
                    "bytes_saved_per_pair_per_unit_d_seg": per_pair_ratio,
                    "bytes_saved_per_pair_per_1e_minus_6_d_seg": per_pair_ratio
                    * 1e-6,
                    "population_pairs": int(population_pairs),
                    "n600_equivalent_bytes_saved": bytes_saved * int(population_pairs),
                    "n600_equivalent_bytes_saved_per_unit_d_seg": global_ratio,
                    "n600_equivalent_bytes_saved_per_1e_minus_6_d_seg": global_ratio
                    * 1e-6,
                    "break_even_bytes_per_unit_d_seg": BREAK_EVEN_BYTES_PER_DSEG,
                    "break_even_bytes_per_pair_per_1e_minus_6_d_seg": (
                        BREAK_EVEN_BYTES_PER_DSEG / int(population_pairs) * 1e-6
                    ),
                    "accept_higher_d_seg_improves_two_term_score": global_ratio
                    > BREAK_EVEN_BYTES_PER_DSEG,
                    "derived_two_term_delta_score": 100.0 * delta_dseg
                    - 25.0
                    * bytes_saved
                    * int(population_pairs)
                    / 37_545_489.0,
                    "normalization": (
                        "measured mean bytes/pair multiplied by the declared contest "
                        "population before applying the global archive-byte score term"
                    ),
                    "verdict_scope": "measured adjacent within-family secant only",
                }
            )
    return out


__all__ = [
    "BREAK_EVEN_BYTES_PER_DSEG",
    "CONTEST_PAIR_COUNT",
    "OperatingPoint",
    "SegSecantError",
    "adjacent_seg_secants",
    "default_operating_points",
    "margin_ordered_abandonment",
    "measure_parseback_payload",
    "spatial_subsample_preimage_residual",
    "summarize_per_class",
    "truncate_preimage_residual_precision",
]
