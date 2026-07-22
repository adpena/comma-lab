"""Spatial stationarity analysis for the DDM G4 n600 flip field.

This module operates on already-measured argmax cells.  It never invokes a
scorer and never treats cell-space correction gains as receiver-realized score
gains.  The three stationarity classes are deliberately disjoint:

``STATIC_IN_IMAGE``
    the same ``predicted -> target`` transition occurs at the same image pixel
    at least twice;
``STATIC_IN_XI_PROXY``
    a remaining transition belongs to a length-two-or-more track under the
    SHA-bound target-cache metric-Pose6 G1 homography proxy;
``TRANSIENT``
    every remaining flip event.

The xi category is not an independently observed physical BEV coordinate.  The
distinction is part of the typed output so downstream consumers cannot launder
the proxy into a BEV claim.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from scipy import ndimage, special

from tac.lie._se3_numpy import exp_se3, rotation_of, translation_of

N_PAIRS = 600
HEIGHT = 384
WIDTH = 512
N_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
AXIS = "[macOS-CPU frozen-scorer advisory]"
RATE_PER_BYTE = 25.0 / 37_545_489.0

# Settled G1 calibration.  Its s_r=0 translation-only nature is why every
# output calls this xi-proxy rather than independently observed BEV.
NATIVE_HEIGHT = 874
NATIVE_WIDTH = 1164
NATIVE_INTRINSICS = {"fx": 910.0, "fy": 910.0, "cx": 582.0, "cy": 437.0}
G1_CALIBRATION = {"s_t": -0.00143, "s_r": 0.0, "pitch_rad": -0.05}
CAMERA_HEIGHT_M = 1.22

STATIONARITY_CLASSES = ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT")
STRATA = ("all", "lane_corridor", "movable_band", "hood_rim", "boundaries")


class StationarityError(RuntimeError):
    """Raised when source custody or a stationarity invariant fails."""


class DdmG4SpatialStationarityConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["DdmG4SpatialStationarityConfigV1"] = Field(alias="schema")
    run_id: StrictStr
    g3_receipt_path: StrictStr
    g3_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    v12_receipt_path: StrictStr
    v12_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    output_directory: StrictStr
    compact_receipt_directory: StrictStr
    n_pairs: StrictInt = Field(ge=N_PAIRS, le=N_PAIRS)
    chunk_pairs: StrictInt = Field(ge=1, le=64)
    seed: StrictInt
    research_only: Literal[True]
    execution_allowed: Literal[False]
    score_claim: Literal[False]

    def typed_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class DdmG4LedgerRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: Literal["ddm_g4_spatial_stationarity_ledger.v1"] = Field(alias="schema")
    record_type: Literal["global", "stratum", "opportunity", "free_context"]
    record_id: StrictStr
    payload: dict[str, Any]
    evidence_axis: Literal["[macOS-CPU frozen-scorer advisory]"]
    research_only: Literal[True]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]


@dataclass(frozen=True)
class XiTrack:
    start_pair: int
    start_row: int
    start_col: int
    transition_code: int
    length: int
    event_ids: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    x = np.asarray(labels)
    out = np.zeros(x.shape, dtype=bool)
    horizontal = x[:, 1:] != x[:, :-1]
    out[:, 1:] |= horizontal
    out[:, :-1] |= horizontal
    vertical = x[1:, :] != x[:-1, :]
    out[1:, :] |= vertical
    out[:-1, :] |= vertical
    return out


def transition_codes(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    if predicted.shape != target.shape:
        raise StationarityError("predicted/target cell shapes differ")
    if np.any(predicted >= N_CLASSES) or np.any(target >= N_CLASSES):
        raise StationarityError("cell id outside the five-class vocabulary")
    return predicted * N_CLASSES + target


def concentration_fractions(frequency: np.ndarray) -> dict[str, Any]:
    values = np.asarray(frequency, dtype=np.int64).reshape(-1)
    total = int(values.sum())
    ordered = np.sort(values)[::-1]
    result: dict[str, Any] = {"total_flip_mass": total, "pixel_count": int(values.size)}
    for fraction, name in ((0.01, "top_1pct"), (0.05, "top_5pct"), (0.10, "top_10pct")):
        count = math.ceil(values.size * fraction)
        mass = int(ordered[:count].sum())
        result[name] = {
            "pixels": count,
            "flip_mass": mass,
            "fraction_of_flip_mass": (mass / total if total else 0.0),
        }
    return result


def recurrence_histogram(transition_counts: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(transition_counts, dtype=np.int64)
    if counts.shape != (N_CLASSES * N_CLASSES, HEIGHT, WIDTH):
        raise StationarityError(f"unexpected transition-count shape {counts.shape}")
    flip_codes = [code for code in range(N_CLASSES * N_CLASSES) if code // N_CLASSES != code % N_CLASSES]
    loci = counts[flip_codes].reshape(-1)
    loci = loci[loci > 0]
    exact = np.bincount(loci, minlength=N_PAIRS + 1)
    bins = ((1, 1), (2, 4), (5, 9), (10, 29), (30, 59), (60, 119), (120, N_PAIRS))
    return {
        "exact_k": [
            {"k": int(k), "locus_count": int(n), "flip_event_mass": int(k * n)}
            for k, n in enumerate(exact)
            if k > 0 and n > 0
        ],
        "bands": [
            {
                "k_min": low,
                "k_max": high,
                "locus_count": int(exact[low : high + 1].sum()),
                "flip_event_mass": int(
                    sum(k * int(exact[k]) for k in range(low, min(high, N_PAIRS) + 1))
                ),
            }
            for low, high in bins
        ],
        "definition": "k counts exact same image pixel and exact predicted-to-target transition across n600",
    }


def _intrinsics() -> np.ndarray:
    sx, sy = WIDTH / NATIVE_WIDTH, HEIGHT / NATIVE_HEIGHT
    return np.array(
        [
            [NATIVE_INTRINSICS["fx"] * sx, 0.0, NATIVE_INTRINSICS["cx"] * sx],
            [0.0, NATIVE_INTRINSICS["fy"] * sy, NATIVE_INTRINSICS["cy"] * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_homography(pose6: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose6, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise StationarityError("registered pose6 is not one finite six-vector")
    xi = np.empty(6, dtype=np.float64)
    xi[:3] = G1_CALIBRATION["s_t"] * np.array([pose[2], pose[1], pose[0]])
    xi[3:] = G1_CALIBRATION["s_r"] * pose[3:6]
    transform = exp_se3(xi)
    rotation = rotation_of(transform)
    translation = translation_of(transform)
    pitch = G1_CALIBRATION["pitch_rad"]
    normal = np.array([0.0, -math.cos(pitch), -math.sin(pitch)], dtype=np.float64)
    plane = rotation - np.outer(translation, normal) / CAMERA_HEIGHT_M
    k = _intrinsics()
    result = k @ plane @ np.linalg.inv(k)
    if not np.isfinite(result).all():
        raise StationarityError("G1 pose proxy produced a non-finite homography")
    return result


def _forward_pixels(rows: np.ndarray, cols: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.stack([cols, rows, np.ones_like(cols)], axis=0).astype(np.float64)
    if not np.isfinite(points).all() or not np.isfinite(homography).all():
        raise StationarityError("xi transport received non-finite pixels or homography")
    # Spell out the 3xN product.  NumPy 2.4 on Accelerate can report stale
    # floating-point flags from the small BLAS ``matmul`` even when both inputs
    # and all results are finite; explicit ufuncs keep this reference path
    # deterministic and fail on the actual operation that overflowed.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        moved = np.stack(
            [
                homography[index, 0] * points[0]
                + homography[index, 1] * points[1]
                + homography[index, 2] * points[2]
                for index in range(3)
            ],
            axis=0,
        )
    if not np.isfinite(moved).all():
        raise StationarityError("xi transport produced non-finite pixels")
    valid_z = np.abs(moved[2]) > 1e-12
    out_col = np.full(cols.shape, -1, dtype=np.int32)
    out_row = np.full(rows.shape, -1, dtype=np.int32)
    out_col[valid_z] = np.rint(moved[0, valid_z] / moved[2, valid_z]).astype(np.int32)
    out_row[valid_z] = np.rint(moved[1, valid_z] / moved[2, valid_z]).astype(np.int32)
    return out_row, out_col


def build_xi_tracks(
    codes: np.ndarray,
    transition_counts: np.ndarray,
    registered_pose6: np.ndarray,
) -> tuple[list[XiTrack], np.ndarray, dict[str, Any]]:
    """Build disjoint recurrent tracks on events not recurrent in image coordinates."""

    if codes.shape != (N_PAIRS, HEIGHT, WIDTH):
        raise StationarityError("xi-track code shape mismatch")
    if registered_pose6.shape != (N_PAIRS, 6) or not np.isfinite(registered_pose6).all():
        raise StationarityError("xi-track registered-pose shape/finite invariant failed")
    sites = HEIGHT * WIDTH
    successors: dict[int, int] = {}
    predecessors: set[int] = set()
    duplicate_destinations = 0
    link_count = 0
    for pair_index in range(1, N_PAIRS):
        previous = codes[pair_index - 1]
        current = codes[pair_index]
        previous_flip = previous // N_CLASSES != previous % N_CLASSES
        current_flip = current // N_CLASSES != current % N_CLASSES
        previous_k = transition_counts[previous, np.arange(HEIGHT)[:, None], np.arange(WIDTH)[None, :]]
        current_k = transition_counts[current, np.arange(HEIGHT)[:, None], np.arange(WIDTH)[None, :]]
        candidate = previous_flip & (previous_k == 1)
        rows, cols = np.nonzero(candidate)
        if rows.size == 0:
            continue
        moved_rows, moved_cols = _forward_pixels(
            rows, cols, pose_homography(registered_pose6[pair_index])
        )
        valid = (
            (moved_rows >= 0)
            & (moved_rows < HEIGHT)
            & (moved_cols >= 0)
            & (moved_cols < WIDTH)
        )
        rows, cols, moved_rows, moved_cols = rows[valid], cols[valid], moved_rows[valid], moved_cols[valid]
        if rows.size == 0:
            continue
        source_codes = previous[rows, cols]
        valid = (
            current_flip[moved_rows, moved_cols]
            & (current_k[moved_rows, moved_cols] == 1)
            & (current[moved_rows, moved_cols] == source_codes)
        )
        rows, cols = rows[valid], cols[valid]
        moved_rows, moved_cols = moved_rows[valid], moved_cols[valid]
        source_codes = source_codes[valid]
        if rows.size == 0:
            continue
        destinations = moved_rows.astype(np.int64) * WIDTH + moved_cols
        order = np.lexsort((cols, rows, destinations))
        destinations = destinations[order]
        keep = np.ones(destinations.shape, dtype=bool)
        keep[1:] = destinations[1:] != destinations[:-1]
        duplicate_destinations += int(np.count_nonzero(~keep))
        for row, col, moved_row, moved_col, _code in zip(
            rows[order][keep],
            cols[order][keep],
            moved_rows[order][keep],
            moved_cols[order][keep],
            source_codes[order][keep],
            strict=True,
        ):
            source_id = (pair_index - 1) * sites + int(row) * WIDTH + int(col)
            target_id = pair_index * sites + int(moved_row) * WIDTH + int(moved_col)
            if source_id in successors or target_id in predecessors:
                continue
            successors[source_id] = target_id
            predecessors.add(target_id)
            link_count += 1

    starts = sorted(source for source in successors if source not in predecessors)
    tracks: list[XiTrack] = []
    membership = np.zeros(codes.shape, dtype=bool)
    for start in starts:
        chain = [start]
        while chain[-1] in successors:
            chain.append(successors[chain[-1]])
        if len(chain) < 2:
            continue
        pair = start // sites
        flat = start % sites
        row, col = divmod(flat, WIDTH)
        code = int(codes[pair, row, col])
        for event_id in chain:
            event_pair = event_id // sites
            event_flat = event_id % sites
            event_row, event_col = divmod(event_flat, WIDTH)
            membership[event_pair, event_row, event_col] = True
        tracks.append(XiTrack(pair, row, col, code, len(chain), tuple(chain)))
    return tracks, membership, {
        "directed_links": link_count,
        "duplicate_destinations_dropped": duplicate_destinations,
        "track_count": len(tracks),
        "tracked_event_count": int(membership.sum()),
        "track_length_max": max((track.length for track in tracks), default=0),
        "track_length_mean": (
            float(np.mean([track.length for track in tracks])) if tracks else 0.0
        ),
        "registration_scope": (
            "SHA-bound target-cache metric Pose6 nearest-target G1 translation-only homography proxy; "
            "pose[p] is f0[p]-to-f1[p], not the f1[p-1]-to-f1[p] transport; source metric Pose6 is "
            "not decoder-free; not independently observed physical BEV"
        ),
        "independent_bev_fraction": None,
    }


def stratum_masks(target: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.uint8)
    predicted = np.asarray(predicted, dtype=np.uint8)
    rows = np.arange(HEIGHT)[:, None]
    target_boundary = boundary_mask(target)
    mycar = (target == 4) | (predicted == 4)
    mycar_edge = target_boundary & ndimage.binary_dilation(mycar, iterations=2)
    return {
        "all": np.ones((HEIGHT, WIDTH), dtype=bool),
        "lane_corridor": ((target == 1) | (predicted == 1)) & (rows >= 174),
        "movable_band": ((target == 3) | (predicted == 3)) & (rows >= 174) & (rows <= 215),
        "hood_rim": mycar_edge & (rows >= 290) & (rows <= 379),
        "boundaries": target_boundary,
    }


def stationarity_decomposition(
    codes: np.ndarray,
    transition_counts: np.ndarray,
    xi_membership: np.ndarray,
    target_cells: np.ndarray,
    predicted_cells: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    totals = {name: dict.fromkeys(STATIONARITY_CLASSES, 0) for name in STRATA}
    totals_by_stratum = dict.fromkeys(STRATA, 0)
    category_maps = {category: np.zeros((HEIGHT, WIDTH), dtype=np.uint32) for category in STATIONARITY_CLASSES}
    rows_index = np.arange(HEIGHT)[:, None]
    cols_index = np.arange(WIDTH)[None, :]
    for pair_index in range(N_PAIRS):
        code = codes[pair_index]
        flip = code // N_CLASSES != code % N_CLASSES
        recurrence = transition_counts[code, rows_index, cols_index]
        image_static = flip & (recurrence >= 2)
        xi_static = flip & ~image_static & xi_membership[pair_index]
        transient = flip & ~image_static & ~xi_static
        categories = {
            "STATIC_IN_IMAGE": image_static,
            "STATIC_IN_XI_PROXY": xi_static,
            "TRANSIENT": transient,
        }
        masks = stratum_masks(target_cells[pair_index], predicted_cells[pair_index])
        for category, category_mask in categories.items():
            category_maps[category] += category_mask
        for stratum, stratum_mask in masks.items():
            total = int(np.count_nonzero(flip & stratum_mask))
            totals_by_stratum[stratum] += total
            for category, category_mask in categories.items():
                totals[stratum][category] += int(np.count_nonzero(category_mask & stratum_mask))
    result = {}
    for stratum in STRATA:
        total = totals_by_stratum[stratum]
        if sum(totals[stratum].values()) != total:
            raise StationarityError(f"stationarity partition does not close for {stratum}")
        result[stratum] = {
            "flip_mass": total,
            "classes": {
                category: {
                    "flip_mass": totals[stratum][category],
                    "fraction": (totals[stratum][category] / total if total else 0.0),
                }
                for category in STATIONARITY_CLASSES
            },
        }
    return result, category_maps


def _beta_kt_bits(successes: np.ndarray | int, trials: np.ndarray | int) -> np.ndarray:
    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    return -(
        special.betaln(k + 0.5, n - k + 0.5) - special.betaln(0.5, 0.5)
    ) / math.log(2.0)


def entropy_bits_binary(successes: np.ndarray | int, trials: np.ndarray | int) -> np.ndarray:
    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
        left = np.where(k > 0, -k * np.log2(p), 0.0)
        right_count = n - k
        right = np.where(right_count > 0, -right_count * np.log2(1.0 - p), 0.0)
    return left + right


def _coder_sizes(raw: bytes) -> dict[str, int | str]:
    variants = {
        "raw": raw,
        "zlib9": zlib.compress(raw, 9),
        "brotli11": brotli.compress(raw, quality=11),
        "lzma_raw": lzma.compress(
            raw,
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}],
        ),
    }
    selected = min(variants, key=lambda name: (len(variants[name]), name != "raw", name))
    return {
        **{f"{name}_bytes": len(payload) + 1 for name, payload in variants.items()},
        "selected_coder": selected,
        "selected_bytes": len(variants[selected]) + 1,
        "codec_tag_bytes": 1,
    }


def free_context_measurement(
    codes: np.ndarray,
    predicted_cells: np.ndarray,
    frequency: np.ndarray,
) -> dict[str, Any]:
    flips = codes // N_CLASSES != codes % N_CLASSES
    total_sites = int(flips.size)
    total_flips = int(flips.sum())
    context_free_kt = float(_beta_kt_bits(total_flips, total_sites))
    pixel_kt = float(_beta_kt_bits(frequency, N_PAIRS).sum())
    context_free_empirical = float(entropy_bits_binary(total_flips, total_sites))
    pixel_empirical = float(entropy_bits_binary(frequency, N_PAIRS).sum())

    row_distance_counts = np.zeros((12, 7, 2), dtype=np.int64)
    context_stream = bytearray()
    distance_edges = np.array([0, 1, 2, 3, 4, 8, 16, np.iinfo(np.int32).max])
    for pair_index in range(N_PAIRS):
        predicted = np.asarray(predicted_cells[pair_index], dtype=np.uint8)
        boundary = boundary_mask(predicted)
        distance = ndimage.distance_transform_edt(~boundary).astype(np.int32)
        distance_bin = np.searchsorted(distance_edges[1:], distance, side="right")
        row_bin = np.minimum(11, np.arange(HEIGHT) * 12 // HEIGHT)[:, None]
        flip = flips[pair_index]
        key = ((row_bin * 7 + distance_bin) * 2 + flip.astype(np.int8)).reshape(-1)
        row_distance_counts += np.bincount(key, minlength=row_distance_counts.size).reshape(
            row_distance_counts.shape
        )
        for rbin in range(12):
            for dbin in range(7):
                mask = (row_bin == rbin) & (distance_bin == dbin)
                context_stream.extend(np.packbits(flip[mask], bitorder="little").tobytes())

    row_distance_kt = float(
        _beta_kt_bits(row_distance_counts[..., 1], row_distance_counts.sum(axis=-1)).sum()
    )
    row_distance_empirical = float(
        entropy_bits_binary(row_distance_counts[..., 1], row_distance_counts.sum(axis=-1)).sum()
    )
    raster_raw = np.packbits(flips.reshape(-1), bitorder="little").tobytes()
    pixel_time_raw = np.packbits(np.moveaxis(flips, 0, -1), axis=-1, bitorder="little").tobytes()
    coders = {
        "context_free_raster": _coder_sizes(raster_raw),
        "aggregate_pixel_time_order": _coder_sizes(pixel_time_raw),
        "predictor_boundary_distance_context": _coder_sizes(bytes(context_stream)),
    }
    selected_context_free = int(coders["context_free_raster"]["selected_bytes"])
    for name in ("aggregate_pixel_time_order", "predictor_boundary_distance_context"):
        selected = int(coders[name]["selected_bytes"])
        coders[name]["gain_bytes_vs_context_free"] = selected_context_free - selected
        coders[name]["gain_fraction_vs_context_free"] = (
            (selected_context_free - selected) / selected_context_free if selected_context_free else 0.0
        )
    return {
        "total_sites": total_sites,
        "total_flips": total_flips,
        "context_free": {
            "empirical_entropy_bits": context_free_empirical,
            "kt_prequential_bits": context_free_kt,
        },
        "aggregate_spatial_pixel_prior": {
            "empirical_conditional_entropy_bits": pixel_empirical,
            "kt_prequential_bits": pixel_kt,
            "kt_gain_bits": context_free_kt - pixel_kt,
            "kt_gain_fraction": (context_free_kt - pixel_kt) / context_free_kt,
            "context_payload_bytes": 0,
            "context_derivation": "pixel coordinate plus causally decoded earlier pair flips; Jeffreys-KT reset is generic",
        },
        "predictor_boundary_distance_margin_proxy": {
            "empirical_conditional_entropy_bits": row_distance_empirical,
            "kt_prequential_bits": row_distance_kt,
            "kt_gain_bits": context_free_kt - row_distance_kt,
            "kt_gain_fraction": (context_free_kt - row_distance_kt) / context_free_kt,
            "context_payload_bytes": 0,
            "context_derivation": (
                "12 row bins x 7 distance-to-predicted-argmax-boundary bins; decoder derives from predictor cells"
            ),
            "scope": "topological boundary-distance proxy; predictor logit margin was not retained",
        },
        "real_coder_measurement": coders,
        "real_coder_scope": (
            "exact innovation bits only; generic traversal/context reset is free; archive section/container overhead excluded"
        ),
    }


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError("uleb128 value must be non-negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def encode_sparse_rules(indices: np.ndarray, codes: np.ndarray) -> bytes:
    flat = np.asarray(indices, dtype=np.int64)
    transitions = np.asarray(codes, dtype=np.uint8)
    if flat.ndim != 1 or transitions.shape != flat.shape or np.any(flat[1:] <= flat[:-1]):
        raise StationarityError("sparse rules must have sorted unique indices")
    body = bytearray(struct.pack(">4sBHHI", b"G4SR", 1, HEIGHT, WIDTH, len(flat)))
    previous = -1
    for index, code in zip(flat, transitions, strict=True):
        body.extend(_uleb128(int(index) - previous - 1))
        body.append(int(code))
        previous = int(index)
    return bytes(body)


def encode_xi_tracks(tracks: Sequence[XiTrack]) -> bytes:
    body = bytearray(struct.pack(">4sBI", b"G4XI", 1, len(tracks)))
    for track in tracks:
        body.extend(
            struct.pack(
                ">HHHHB",
                track.start_pair,
                track.start_row,
                track.start_col,
                track.length,
                track.transition_code,
            )
        )
    return bytes(body)


def _best_transition_gain(transition_counts: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    best = (0, 0, 0)
    for source in range(N_CLASSES):
        collateral = transition_counts[source * N_CLASSES + source][mask].sum(dtype=np.int64)
        for target in range(N_CLASSES):
            if source == target:
                continue
            corrected = transition_counts[source * N_CLASSES + target][mask].sum(dtype=np.int64)
            gain = int(corrected - collateral)
            if gain > best[0]:
                best = (gain, source, target)
    return best


def _opportunity_row(
    opportunity_id: str,
    kind: str,
    raw_payload: bytes,
    net_fixed_flips: int,
    touched_pairs: int,
    parameterization: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    coding = _coder_sizes(raw_payload)
    selected_bytes = int(coding["selected_bytes"])
    delta_d_seg = net_fixed_flips / (N_PAIRS * HEIGHT * WIDTH)
    delta_seg_score = 100.0 * delta_d_seg
    return {
        "opportunity_id": opportunity_id,
        "kind": kind,
        "parameterization": dict(parameterization),
        "net_cell_flips_fixed": int(net_fixed_flips),
        "touched_pairs": int(touched_pairs),
        "cell_space_delta_d_seg": delta_d_seg,
        "cell_space_delta_seg_score": delta_seg_score,
        "receiver_realized_delta_d_seg": None,
        "byte_measurement": coding,
        "seg_score_gain_per_selected_byte": (delta_seg_score / selected_bytes if selected_bytes else 0.0),
        "rate_break_even_score_per_byte": RATE_PER_BYTE,
        "cell_space_above_rate_break_even": (
            delta_seg_score / selected_bytes > RATE_PER_BYTE if selected_bytes else False
        ),
        "scope": scope,
        "evidence_axis": AXIS,
        "score_claim": False,
    }


def sparse_rule_opportunity(
    transition_counts: np.ndarray,
    recurrence_counts: np.ndarray,
    support: np.ndarray,
    opportunity_id: str,
) -> dict[str, Any]:
    gains = np.zeros((N_CLASSES * N_CLASSES, HEIGHT, WIDTH), dtype=np.int16)
    for source in range(N_CLASSES):
        collateral = transition_counts[source * N_CLASSES + source].astype(np.int32)
        for target in range(N_CLASSES):
            code = source * N_CLASSES + target
            if source != target:
                gains[code] = np.clip(
                    transition_counts[code].astype(np.int32) - collateral,
                    np.iinfo(np.int16).min,
                    np.iinfo(np.int16).max,
                )
    best_code = gains.argmax(axis=0).astype(np.uint8)
    best_gain = np.take_along_axis(gains, best_code[None, ...], axis=0)[0]
    recurrent = np.take_along_axis(recurrence_counts, best_code[None, ...], axis=0)[0] >= 2
    selected = support & recurrent & (best_gain > 0)
    flat = np.flatnonzero(selected)
    codes = best_code.reshape(-1)[flat]
    payload = encode_sparse_rules(flat, codes)
    return _opportunity_row(
        opportunity_id,
        "one_time_static_image_sparse_rule_field",
        payload,
        int(best_gain[selected].sum(dtype=np.int64)),
        N_PAIRS,
        {"rule_count": int(flat.size), "support_pixels": int(np.count_nonzero(support))},
        scope=(
            "measured argmax-cell intervention including collateral at every pair; sparse section is real-coded, "
            "but RGB receiver realization is unmeasured"
        ),
    )


def parametric_opportunities(transition_counts: np.ndarray, flip_frequency: np.ndarray) -> list[dict[str, Any]]:
    rows = np.arange(HEIGHT)[:, None]
    cols = np.arange(WIDTH)[None, :]
    opportunities: list[dict[str, Any]] = []

    # Horizon: exact horizontal row-band rule, searched over the settled horizon neighborhood.
    best_horizon: tuple[int, int, int, int, int] = (0, 174, 0, 0, 1)
    for center in range(150, 231):
        for halfwidth in range(0, 5):
            mask = np.broadcast_to((np.abs(rows - center) <= halfwidth), (HEIGHT, WIDTH))
            gain, source, target = _best_transition_gain(transition_counts, mask)
            if gain > best_horizon[0]:
                best_horizon = (gain, center, halfwidth, source, target)
    gain, center, halfwidth, source, target = best_horizon
    horizon_payload = struct.pack(">4sBHBBBB", b"G4HR", 1, center, halfwidth, source, target, 0)
    opportunities.append(
        _opportunity_row(
            "horizon_row_parametric",
            "one_time_horizontal_band_rule",
            horizon_payload,
            gain,
            N_PAIRS,
            {"center_row": center, "halfwidth": halfwidth, "source": source, "target": target},
            scope="exhaustive row/width/transition search on cell counts; RGB realization unmeasured",
        )
    )

    # Settled Movable mid-band, priced as a fixed horizontal band.
    movable_mask = np.broadcast_to((rows >= 174) & (rows <= 215), (HEIGHT, WIDTH))
    gain, source, target = _best_transition_gain(transition_counts, movable_mask)
    movable_payload = struct.pack(">4sBHHBB", b"G4MB", 1, 174, 215, source, target)
    opportunities.append(
        _opportunity_row(
            "movable_midband_parametric",
            "one_time_fixed_row_interval_rule",
            movable_payload,
            gain,
            N_PAIRS,
            {"row_start": 174, "row_stop_inclusive": 215, "source": source, "target": target},
            scope="settled row interval with measured best transition; class/receiver realization unmeasured",
        )
    )

    # Lane corridor: fit two straight edge charts to recurrence-weighted Lane flips.
    lane_codes = [code for code in range(25) if (code // 5 == 1) ^ (code % 5 == 1)]
    lane_weight = transition_counts[lane_codes].sum(axis=0, dtype=np.int64)
    fit_rows: list[float] = []
    left_cols: list[float] = []
    right_cols: list[float] = []
    weights: list[float] = []
    for row in range(174, HEIGHT):
        x = np.flatnonzero(lane_weight[row] > 0)
        if x.size < 2:
            continue
        w = lane_weight[row, x].astype(np.float64)
        cumulative = np.cumsum(w)
        total = cumulative[-1]
        left_cols.append(float(x[np.searchsorted(cumulative, 0.15 * total)]))
        right_cols.append(float(x[np.searchsorted(cumulative, 0.85 * total)]))
        fit_rows.append(float(row))
        weights.append(float(total))
    if len(fit_rows) >= 4:
        left_coef = np.polyfit(fit_rows, left_cols, deg=1, w=np.sqrt(weights))
        right_coef = np.polyfit(fit_rows, right_cols, deg=1, w=np.sqrt(weights))
        best: tuple[int, int, int, int] = (0, 1, 0, 1)
        for width in range(0, 5):
            left = left_coef[0] * rows + left_coef[1]
            right = right_coef[0] * rows + right_coef[1]
            mask = (rows >= 174) & (
                (np.abs(cols - left) <= width) | (np.abs(cols - right) <= width)
            )
            gain, source, target = _best_transition_gain(transition_counts, mask)
            if gain > best[0]:
                best = (gain, width, source, target)
        gain, width, source, target = best
        lane_payload = struct.pack(
            ">4sBffffHBBB",
            b"G4LW",
            1,
            float(left_coef[0]),
            float(left_coef[1]),
            float(right_coef[0]),
            float(right_coef[1]),
            174,
            width,
            source,
            target,
        )
        opportunities.append(
            _opportunity_row(
                "lane_corridor_wedge_parametric",
                "one_time_two_line_wedge_edge_rule",
                lane_payload,
                gain,
                N_PAIRS,
                {
                    "left_x_of_y": [float(left_coef[0]), float(left_coef[1])],
                    "right_x_of_y": [float(right_coef[0]), float(right_coef[1])],
                    "row_start": 174,
                    "halfwidth": width,
                    "source": source,
                    "target": target,
                },
                scope="weighted line fit plus exhaustive width/transition search; RGB realization unmeasured",
            )
        )

    # Hood rim: fit a quadratic y(x) to MyCar-transition flip mass.
    hood_codes = [code for code in range(25) if (code // 5 == 4) ^ (code % 5 == 4)]
    hood_weight = transition_counts[hood_codes].sum(axis=0, dtype=np.int64)
    hood_weight[:290] = 0
    hood_weight[380:] = 0
    fit_x: list[float] = []
    fit_y: list[float] = []
    fit_w: list[float] = []
    for col in range(WIDTH):
        y = np.flatnonzero(hood_weight[:, col] > 0)
        if y.size == 0:
            continue
        weight = hood_weight[y, col].astype(np.float64)
        fit_x.append(float(col))
        fit_y.append(float(np.average(y, weights=weight)))
        fit_w.append(float(weight.sum()))
    if len(fit_x) >= 8:
        hood_coef = np.polyfit(fit_x, fit_y, deg=2, w=np.sqrt(fit_w))
        curve = hood_coef[0] * cols**2 + hood_coef[1] * cols + hood_coef[2]
        best = (0, 1, 0, 4)
        for width in range(0, 5):
            mask = (np.abs(rows - curve) <= width) & (rows >= 290) & (rows <= 379)
            gain, source, target = _best_transition_gain(transition_counts, mask)
            if gain > best[0]:
                best = (gain, width, source, target)
        gain, width, source, target = best
        hood_payload = struct.pack(
            ">4sBfffBBB",
            b"G4HC",
            1,
            float(hood_coef[0]),
            float(hood_coef[1]),
            float(hood_coef[2]),
            width,
            source,
            target,
        )
        opportunities.append(
            _opportunity_row(
                "hood_rim_quadratic_parametric",
                "one_time_quadratic_curve_rule",
                hood_payload,
                gain,
                N_PAIRS,
                {
                    "row_of_col_quadratic": [float(value) for value in hood_coef],
                    "halfwidth": width,
                    "source": source,
                    "target": target,
                },
                scope="weighted quadratic fit plus exhaustive width/transition search; RGB realization unmeasured",
            )
        )
    return opportunities


def build_opportunities(
    transition_counts: np.ndarray,
    flip_frequency: np.ndarray,
    xi_tracks: Sequence[XiTrack],
    stratum_frequency: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    all_support = np.ones((HEIGHT, WIDTH), dtype=bool)
    opportunities = [
        sparse_rule_opportunity(
            transition_counts, transition_counts, all_support, "static_image_sparse_all"
        )
    ]
    for stratum in ("lane_corridor", "movable_band", "hood_rim", "boundaries"):
        support = np.asarray(stratum_frequency[stratum]) > 0
        opportunities.append(
            sparse_rule_opportunity(
                transition_counts,
                transition_counts,
                support,
                f"static_image_sparse_{stratum}",
            )
        )
    opportunities.extend(parametric_opportunities(transition_counts, flip_frequency))
    xi_payload = encode_xi_tracks(xi_tracks)
    opportunities.append(
        _opportunity_row(
            "xi_proxy_track_seeds",
            "one_time_xi_track_seed_plus_lifetime_field",
            xi_payload,
            sum(track.length for track in xi_tracks),
            len({event_id // (HEIGHT * WIDTH) for track in xi_tracks for event_id in track.event_ids}),
            {
                "track_count": len(xi_tracks),
                "tracked_events": sum(track.length for track in xi_tracks),
                "generic_transport": "G1 target-cache metric-Pose6 homography proxy",
                "transport_side_information_bytes": None,
            },
            scope=(
                "real-coded track seeds/lifetimes only; target-cache metric Pose6 side-information bytes, physical "
                "BEV, and RGB receiver realization are unmeasured"
            ),
        )
    )
    opportunities.sort(
        key=lambda row: (
            -float(row["seg_score_gain_per_selected_byte"]),
            int(row["byte_measurement"]["selected_bytes"]),
            row["opportunity_id"],
        )
    )
    for rank, row in enumerate(opportunities, start=1):
        row["rank"] = rank
    return opportunities


def typed_ledger_rows(
    *,
    concentration: Mapping[str, Any],
    recurrence: Mapping[str, Any],
    decomposition: Mapping[str, Any],
    free_context: Mapping[str, Any],
    opportunities: Sequence[Mapping[str, Any]],
    xi_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[DdmG4LedgerRowV1] = [
        DdmG4LedgerRowV1(
            schema="ddm_g4_spatial_stationarity_ledger.v1",
            record_type="global",
            record_id="global",
            payload={
                "concentration": dict(concentration),
                "recurrence_k_distribution": dict(recurrence),
                "xi_registration": dict(xi_summary),
            },
            evidence_axis=AXIS,
            research_only=True,
            score_claim=False,
            promotion_eligible=False,
        )
    ]
    for stratum in STRATA:
        rows.append(
            DdmG4LedgerRowV1(
                schema="ddm_g4_spatial_stationarity_ledger.v1",
                record_type="stratum",
                record_id=stratum,
                payload=dict(decomposition[stratum]),
                evidence_axis=AXIS,
                research_only=True,
                score_claim=False,
                promotion_eligible=False,
            )
        )
    rows.append(
        DdmG4LedgerRowV1(
            schema="ddm_g4_spatial_stationarity_ledger.v1",
            record_type="free_context",
            record_id="free_context",
            payload=dict(free_context),
            evidence_axis=AXIS,
            research_only=True,
            score_claim=False,
            promotion_eligible=False,
        )
    )
    for opportunity in opportunities:
        rows.append(
            DdmG4LedgerRowV1(
                schema="ddm_g4_spatial_stationarity_ledger.v1",
                record_type="opportunity",
                record_id=str(opportunity["opportunity_id"]),
                payload=dict(opportunity),
                evidence_axis=AXIS,
                research_only=True,
                score_claim=False,
                promotion_eligible=False,
            )
        )
    return [row.model_dump(mode="json", by_alias=True) for row in rows]


__all__ = [
    "AXIS",
    "HEIGHT",
    "N_PAIRS",
    "STATIONARITY_CLASSES",
    "STRATA",
    "WIDTH",
    "DdmG4LedgerRowV1",
    "DdmG4SpatialStationarityConfigV1",
    "StationarityError",
    "XiTrack",
    "boundary_mask",
    "build_opportunities",
    "build_xi_tracks",
    "concentration_fractions",
    "free_context_measurement",
    "recurrence_histogram",
    "sha256_file",
    "stationarity_decomposition",
    "stratum_masks",
    "transition_codes",
    "typed_ledger_rows",
]
