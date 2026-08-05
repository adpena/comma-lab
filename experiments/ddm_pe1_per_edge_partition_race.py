#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""pe1 scorer-free per-edge partition representation race.

This arm works only in the MASK/partition domain. It reads the real cached n600
GT argmax labels, extracts class-pair boundary components, races an explicit
curve chart against a local Laguerre/power-diagram generator-pair chart, measures
real coder bytes, and appends counted PE1 sections to the qo1 IX2 archive for
parse-back. It does not run SegNet or PoseNet and makes no score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO / "src", _REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.lie import _se3_numpy as se3_np
from tac.optimization.ddm_ix2_archive_container import (
    build_payload,
    build_single_member_zip,
    parse_payload,
)

SEG_H: Final = 384
SEG_W: Final = 512
N_PAIRS: Final = 600
RATE_DENOM: Final = 37_545_489
BASELINE_S: Final = 0.7539807296911207
BASELINE_BYTES: Final = 357_836
BASELINE_AXIS: Final = "[macOS-CPU advisory]"
BASELINE_ARCHIVE_SHA256: Final = (
    "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
)

CLASS_NAMES: Final = ("Road", "Lane", "Undriv", "Movable", "MyCar")
ROAD: Final = 0
LANE: Final = 1
EDGE_PAIRS: Final = tuple((a, b) for a in range(5) for b in range(a + 1, 5))

PE1_MAGIC: Final = b"PE1EDGE1"
PE1_VERSION: Final = 1
PE1_HEADER: Final = struct.Struct("<8sBHHHBBII32s")
PE1_CURVE: Final = 1
PE1_GENERATOR: Final = 2
PE1_GENERATOR_TRANSPORT: Final = 3
PE1_KIND_NAMES: Final = {
    PE1_CURVE: "explicit_curve_spline",
    PE1_GENERATOR: "generator_pair_bisector",
    PE1_GENERATOR_TRANSPORT: "generator_pair_xi_transport",
}
CODEC_IDS: Final = {
    "brotli-q11": bd1.BD1_BROTLI_Q11,
    "lzma1-raw": bd1.BD1_LZMA1_RAW,
    "smevr-r7-nibble": bd1.BD1_SMEVR_R7_NIBBLE,
}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}

DEFAULT_BASE_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_GT_CACHE: Final = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CURRENT_ARGMAX: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy"
)
DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_pe1_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pe1_20260805")


class PE1Error(ValueError):
    """The pe1 representation, archive, or parse-back failed closed."""


@dataclass(frozen=True, slots=True)
class CoderResult:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None


@dataclass(frozen=True, slots=True)
class Component:
    uid: int
    pair: int
    edge: tuple[int, int]
    flat: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid_yx: tuple[float, float]
    flip_mass: int


@dataclass(frozen=True, slots=True)
class CurveParams:
    edge: tuple[int, int]
    axis: int
    half_width: int
    primary_start: int
    primary_count: int
    knot_stride: int
    knots: tuple[int, ...]
    dim: int


@dataclass(frozen=True, slots=True)
class GeneratorParams:
    edge: tuple[int, int]
    bbox: tuple[int, int, int, int]
    gen_a_q4: tuple[int, int]
    gen_b_q4: tuple[int, int]
    dim: int = 4


@dataclass(frozen=True, slots=True)
class RepresentationBuild:
    surface_id: str
    kind: int
    raw: bytes
    frame_records: tuple[bytes, ...]
    component_masks: dict[int, np.ndarray]
    component_dims: dict[int, int]
    component_bytes: dict[int, int]
    scope_note: str
    falls_out: bool
    selected_component_ids: frozenset[int]
    metadata: dict[str, Any]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderResult):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def varint(value: int) -> bytes:
    if value < 0:
        raise PE1Error("varint cannot encode negative values")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(payload):
            raise PE1Error("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise PE1Error("varint too long")


def zigzag(value: int) -> int:
    return (int(value) << 1) ^ (int(value) >> 63)


def unzigzag(value: int) -> int:
    return (int(value) >> 1) ^ -(int(value) & 1)


def write_zigzag(value: int) -> bytes:
    return varint(zigzag(value))


def read_zigzag(payload: bytes, offset: int) -> tuple[int, int]:
    value, offset = read_varint(payload, offset)
    return unzigzag(value), offset


def lzma1_raw(payload: bytes) -> bytes:
    return bd1.lzma1_raw(payload)


def unlzma1_raw(payload: bytes, expected_len: int) -> bytes:
    return bd1.unlzma1_raw(payload, expected_len)


def race_coders(
    *,
    surface_id: str,
    raw: bytes,
    frame_records: tuple[bytes, ...],
    artifact_dir: Path,
    store_best: bool,
) -> tuple[tuple[CoderResult, ...], str, bytes]:
    encoded = {
        "brotli-q11": bytes(brotli.compress(raw, quality=11)),
        "lzma1-raw": lzma1_raw(raw),
        "smevr-r7-nibble": bd1.smevr_records(list(frame_records)),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise PE1Error(f"{surface_id}: Brotli roundtrip failed")
    if unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise PE1Error(f"{surface_id}: LZMA1 roundtrip failed")
    if tuple(bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != frame_records:
        raise PE1Error(f"{surface_id}: SMEVR frame-record roundtrip failed")
    best_codec = min(encoded, key=lambda name: len(encoded[name]))
    rows: list[CoderResult] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if store_best and codec == best_codec:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            safe = surface_id.replace("/", "_").replace(":", "_")
            path = artifact_dir / f"{safe}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        rows.append(CoderResult(codec, len(payload), sha256_bytes(payload), artifact_path))
    return tuple(rows), best_codec, encoded[best_codec]


def edge_name(edge: tuple[int, int]) -> str:
    return f"{CLASS_NAMES[edge[0]]}<->{CLASS_NAMES[edge[1]]}"


def edge_band(labels: np.ndarray, a: int, b: int) -> np.ndarray:
    """Return 4-neighbor class-pair boundary endpoints for one argmax frame."""
    lab = np.asarray(labels)
    out = np.zeros(lab.shape, dtype=bool)
    up = ((lab[:-1, :] == a) & (lab[1:, :] == b)) | ((lab[:-1, :] == b) & (lab[1:, :] == a))
    out[:-1, :] |= up
    out[1:, :] |= up
    left = ((lab[:, :-1] == a) & (lab[:, 1:] == b)) | ((lab[:, :-1] == b) & (lab[:, 1:] == a))
    out[:, :-1] |= left
    out[:, 1:] |= left
    return out


def residual_target(labels: np.ndarray, current: np.ndarray, a: int, b: int) -> np.ndarray:
    return ((labels == a) & (current == b)) | ((labels == b) & (current == a))


def extract_components(lstars: np.ndarray, current: np.ndarray | None) -> tuple[list[Component], dict[str, Any]]:
    if tuple(lstars.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise PE1Error(f"unexpected lstar shape {lstars.shape}")
    if current is not None and tuple(current.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise PE1Error(f"unexpected current argmax shape {current.shape}")

    structure = ndimage.generate_binary_structure(2, 2)
    components: list[Component] = []
    uid = 0
    edge_stats = {
        edge: {"source_band_pixels": 0, "components": 0, "flip_mass": 0}
        for edge in EDGE_PAIRS
    }
    per_pair_components: list[int] = []
    per_pair_band_pixels: list[int] = []
    for pair in range(N_PAIRS):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        cur = np.asarray(current[pair], dtype=np.uint8) if current is not None else labels
        pair_components = 0
        pair_pixels = 0
        for edge in EDGE_PAIRS:
            a, b = edge
            band = edge_band(labels, a, b)
            target = residual_target(labels, cur, a, b)
            labeled, n_comp = ndimage.label(band, structure=structure)
            objects = ndimage.find_objects(labeled)
            for comp_id, slc in enumerate(objects, start=1):
                if slc is None:
                    continue
                local = labeled[slc] == comp_id
                yy, xx = np.nonzero(local)
                y0 = int(slc[0].start)
                x0 = int(slc[1].start)
                ys = yy + y0
                xs = xx + x0
                flat = (ys * SEG_W + xs).astype(np.int32)
                if flat.size == 0:
                    continue
                bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
                flip_mass = int(target.reshape(-1)[flat.astype(np.int64)].sum())
                components.append(
                    Component(
                        uid=uid,
                        pair=pair,
                        edge=edge,
                        flat=np.ascontiguousarray(flat),
                        bbox=bbox,
                        centroid_yx=(float(ys.mean()), float(xs.mean())),
                        flip_mass=flip_mass,
                    )
                )
                uid += 1
                pair_components += 1
                pair_pixels += int(flat.size)
                edge_stats[edge]["source_band_pixels"] += int(flat.size)
                edge_stats[edge]["components"] += 1
                edge_stats[edge]["flip_mass"] += flip_mass
            if n_comp == 0:
                edge_stats[edge]["source_band_pixels"] += 0
        per_pair_components.append(pair_components)
        per_pair_band_pixels.append(pair_pixels)

    ranked_edges = sorted(
        [
            {
                "edge": edge_name(edge),
                "class_ids": list(edge),
                "source_band_pixels": values["source_band_pixels"],
                "components": values["components"],
                "flip_mass": values["flip_mass"],
                "flip_mass_fraction": (
                    values["flip_mass"] / max(1, sum(v["flip_mass"] for v in edge_stats.values()))
                ),
            }
            for edge, values in edge_stats.items()
            if values["source_band_pixels"] or values["flip_mass"]
        ],
        key=lambda row: (-int(row["flip_mass"]), row["edge"]),
    )
    meta = {
        "schema": "pe1_component_extraction.v1",
        "components": len(components),
        "source_band_pixels_sum_over_edges": int(sum(c.flat.size for c in components)),
        "per_pair_components_min": int(min(per_pair_components)),
        "per_pair_components_mean": float(np.mean(per_pair_components)),
        "per_pair_components_max": int(max(per_pair_components)),
        "per_pair_band_pixels_min": int(min(per_pair_band_pixels)),
        "per_pair_band_pixels_mean": float(np.mean(per_pair_band_pixels)),
        "per_pair_band_pixels_max": int(max(per_pair_band_pixels)),
        "total_flip_mass_in_edge_bands": int(sum(v["flip_mass"] for v in edge_stats.values())),
        "edge_ranking": ranked_edges,
        "road_lane_first": next((row for row in ranked_edges if row["edge"] == "Road<->Lane"), None),
    }
    return components, meta


def component_mask(flat: np.ndarray) -> np.ndarray:
    mask = np.zeros((SEG_H, SEG_W), dtype=bool)
    mask.reshape(-1)[np.asarray(flat, dtype=np.int64)] = True
    return mask


def _raster_line(axis: int, primary_start: int, centers: np.ndarray, half_width: int) -> np.ndarray:
    mask = np.zeros((SEG_H, SEG_W), dtype=bool)
    for i, center in enumerate(centers.tolist()):
        primary = primary_start + i
        c = round(center)
        if axis == 0:
            if 0 <= primary < SEG_H:
                x0 = max(0, c - half_width)
                x1 = min(SEG_W - 1, c + half_width)
                if x0 <= x1:
                    mask[primary, x0 : x1 + 1] = True
        else:
            if 0 <= primary < SEG_W:
                y0 = max(0, c - half_width)
                y1 = min(SEG_H - 1, c + half_width)
                if y0 <= y1:
                    mask[y0 : y1 + 1, primary] = True
    return mask


def fit_curve_params(component: Component, knot_stride: int) -> CurveParams:
    if knot_stride < 1:
        raise PE1Error("knot stride must be positive")
    ys = component.flat.astype(np.int64) // SEG_W
    xs = component.flat.astype(np.int64) % SEG_W
    y_span = int(ys.max() - ys.min())
    x_span = int(xs.max() - xs.min())
    axis = 0 if y_span >= x_span else 1
    primary = ys if axis == 0 else xs
    secondary = xs if axis == 0 else ys
    primary_start = int(primary.min())
    primary_stop = int(primary.max()) + 1
    primary_count = primary_stop - primary_start
    centers = np.zeros(primary_count, dtype=np.float64)
    residual_max = 0
    last_center = round(float(np.mean(secondary)))
    for offset, p in enumerate(range(primary_start, primary_stop)):
        sec = secondary[primary == p]
        if sec.size:
            center = round(float(np.mean(sec)))
            last_center = center
            residual_max = max(residual_max, int(np.max(np.abs(sec - center))))
        centers[offset] = last_center
    knot_indices = list(range(0, primary_count, knot_stride))
    if not knot_indices or knot_indices[-1] != primary_count - 1:
        knot_indices.append(primary_count - 1)
    knots = tuple(round(float(centers[i])) for i in knot_indices)
    return CurveParams(
        edge=component.edge,
        axis=axis,
        half_width=max(0, int(residual_max)),
        primary_start=primary_start,
        primary_count=primary_count,
        knot_stride=knot_stride,
        knots=knots,
        dim=4 + len(knots),
    )


def encode_curve_params(params: CurveParams) -> bytes:
    out = bytearray([params.edge[0], params.edge[1], params.axis])
    out += varint(params.half_width)
    out += varint(params.primary_start)
    out += varint(params.primary_count)
    out += varint(params.knot_stride)
    out += varint(len(params.knots))
    prev = 0
    for i, knot in enumerate(params.knots):
        out += write_zigzag(knot if i == 0 else knot - prev)
        prev = knot
    return bytes(out)


def decode_curve_params(record: bytes) -> CurveParams:
    if len(record) < 3:
        raise PE1Error("curve record truncated before class/axis")
    a = int(record[0])
    b = int(record[1])
    axis = int(record[2])
    offset = 3
    half_width, offset = read_varint(record, offset)
    primary_start, offset = read_varint(record, offset)
    primary_count, offset = read_varint(record, offset)
    knot_stride, offset = read_varint(record, offset)
    knot_count, offset = read_varint(record, offset)
    if axis not in (0, 1) or a >= b or primary_count < 1 or knot_stride < 1 or knot_count < 1:
        raise PE1Error("curve record header invalid")
    knots: list[int] = []
    prev = 0
    for i in range(knot_count):
        value, offset = read_zigzag(record, offset)
        knot = value if i == 0 else prev + value
        knots.append(knot)
        prev = knot
    if offset != len(record):
        raise PE1Error("curve record has trailing bytes")
    return CurveParams(
        edge=(a, b),
        axis=axis,
        half_width=int(half_width),
        primary_start=int(primary_start),
        primary_count=int(primary_count),
        knot_stride=int(knot_stride),
        knots=tuple(knots),
        dim=4 + len(knots),
    )


def rasterize_curve(params: CurveParams) -> np.ndarray:
    knot_positions = list(range(0, params.primary_count, params.knot_stride))
    if not knot_positions or knot_positions[-1] != params.primary_count - 1:
        knot_positions.append(params.primary_count - 1)
    if len(knot_positions) != len(params.knots):
        raise PE1Error("curve knot positions/count differ")
    centers = np.interp(
        np.arange(params.primary_count, dtype=np.float64),
        np.asarray(knot_positions, dtype=np.float64),
        np.asarray(params.knots, dtype=np.float64),
    )
    return _raster_line(params.axis, params.primary_start, centers, params.half_width)


def quantize_q4(value: float) -> int:
    return round(float(value) * 4.0)


def fit_generator_params(component: Component, labels: np.ndarray) -> GeneratorParams:
    a, b = component.edge
    y0, x0, y1, x1 = component.bbox
    py0 = max(0, y0 - 2)
    px0 = max(0, x0 - 2)
    py1 = min(SEG_H, y1 + 2)
    px1 = min(SEG_W, x1 + 2)
    roi = labels[py0:py1, px0:px1]
    yy, xx = np.indices(roi.shape)
    coords_y = yy + py0
    coords_x = xx + px0
    mask_a = roi == a
    mask_b = roi == b
    if not np.any(mask_a) or not np.any(mask_b):
        cy, cx = component.centroid_yx
        gen_a = (cy - 0.5, cx)
        gen_b = (cy + 0.5, cx)
    else:
        gen_a = (float(coords_y[mask_a].mean()), float(coords_x[mask_a].mean()))
        gen_b = (float(coords_y[mask_b].mean()), float(coords_x[mask_b].mean()))
    return GeneratorParams(
        edge=component.edge,
        bbox=(py0, px0, py1, px1),
        gen_a_q4=(quantize_q4(gen_a[0]), quantize_q4(gen_a[1])),
        gen_b_q4=(quantize_q4(gen_b[0]), quantize_q4(gen_b[1])),
    )


def encode_generator_params(params: GeneratorParams) -> bytes:
    y0, x0, y1, x1 = params.bbox
    return struct.pack(
        "<BBHHHHhhhh",
        params.edge[0],
        params.edge[1],
        y0,
        x0,
        y1 - y0,
        x1 - x0,
        params.gen_a_q4[0],
        params.gen_a_q4[1],
        params.gen_b_q4[0],
        params.gen_b_q4[1],
    )


def decode_generator_params(record: bytes) -> GeneratorParams:
    if len(record) != struct.calcsize("<BBHHHHhhhh"):
        raise PE1Error("generator record length differs")
    a, b, y0, x0, h, w, gay, gax, gby, gbx = struct.unpack("<BBHHHHhhhh", record)
    if a >= b or h < 1 or w < 1 or y0 + h > SEG_H or x0 + w > SEG_W:
        raise PE1Error("generator record geometry invalid")
    return GeneratorParams(
        edge=(int(a), int(b)),
        bbox=(int(y0), int(x0), int(y0 + h), int(x0 + w)),
        gen_a_q4=(int(gay), int(gax)),
        gen_b_q4=(int(gby), int(gbx)),
    )


def rasterize_generator(params: GeneratorParams) -> np.ndarray:
    y0, x0, y1, x1 = params.bbox
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    yy += y0
    xx += x0
    ay, ax = (params.gen_a_q4[0] / 4.0, params.gen_a_q4[1] / 4.0)
    by, bx = (params.gen_b_q4[0] / 4.0, params.gen_b_q4[1] / 4.0)
    da = (yy - ay) ** 2 + (xx - ax) ** 2
    db = (yy - by) ** 2 + (xx - bx) ** 2
    side = da <= db
    local = np.zeros(side.shape, dtype=bool)
    vert = side[:-1, :] != side[1:, :]
    local[:-1, :] |= vert
    local[1:, :] |= vert
    horiz = side[:, :-1] != side[:, 1:]
    local[:, :-1] |= horiz
    local[:, 1:] |= horiz
    out = np.zeros((SEG_H, SEG_W), dtype=bool)
    out[y0:y1, x0:x1] = local
    return out


def frame_records_from_component_records(
    components: list[Component],
    selected_ids: frozenset[int],
    component_record_bytes: dict[int, bytes],
) -> tuple[bytes, ...]:
    by_pair: list[list[bytes]] = [[] for _ in range(N_PAIRS)]
    for comp in components:
        if comp.uid not in selected_ids:
            continue
        rec = component_record_bytes[comp.uid]
        by_pair[comp.pair].append(varint(len(rec)) + rec)
    records: list[bytes] = []
    for recs in by_pair:
        records.append(varint(len(recs)) + b"".join(recs))
    return tuple(records)


def iter_frame_component_records(raw: bytes, n_pairs: int) -> Iterable[tuple[int, bytes]]:
    offset = 0
    for pair in range(n_pairs):
        count, offset = read_varint(raw, offset)
        for _ in range(count):
            length, offset = read_varint(raw, offset)
            record = raw[offset : offset + length]
            if len(record) != length:
                raise PE1Error("component record truncated")
            offset += length
            yield pair, record
    if offset != len(raw):
        raise PE1Error("component record stream has trailing bytes")


def build_curve_representation(
    *,
    components: list[Component],
    knot_stride: int,
    selected_ids: frozenset[int],
) -> RepresentationBuild:
    record_bytes: dict[int, bytes] = {}
    decoded_masks: dict[int, np.ndarray] = {}
    dims: dict[int, int] = {}
    for comp in components:
        if comp.uid not in selected_ids:
            continue
        params = fit_curve_params(comp, knot_stride)
        rec = encode_curve_params(params)
        parsed = decode_curve_params(rec)
        record_bytes[comp.uid] = rec
        decoded_masks[comp.uid] = rasterize_curve(parsed)
        dims[comp.uid] = parsed.dim
    frame_records = frame_records_from_component_records(components, selected_ids, record_bytes)
    raw = b"".join(frame_records)
    return RepresentationBuild(
        surface_id=f"explicit_curve_k{knot_stride}",
        kind=PE1_CURVE,
        raw=raw,
        frame_records=frame_records,
        component_masks=decoded_masks,
        component_dims=dims,
        component_bytes={uid: len(rec) for uid, rec in record_bytes.items()},
        scope_note="arc-length curve chart with stride-selected secondary-coordinate knots; MDL row is selected from real-coded strides",
        falls_out=False,
        selected_component_ids=selected_ids,
        metadata={"knot_stride": knot_stride},
    )


def build_generator_representation(
    *,
    components: list[Component],
    lstars: np.ndarray,
    selected_ids: frozenset[int],
) -> tuple[RepresentationBuild, dict[int, GeneratorParams]]:
    params_by_uid: dict[int, GeneratorParams] = {}
    for comp in components:
        if comp.uid not in selected_ids:
            continue
        labels = np.asarray(lstars[comp.pair], dtype=np.uint8)
        params = fit_generator_params(comp, labels)
        params_by_uid[comp.uid] = decode_generator_params(encode_generator_params(params))
    return (
        build_generator_representation_from_params(
            components=components,
            params_by_uid=params_by_uid,
            selected_ids=selected_ids,
            surface_id="generator_pair_bisector",
        ),
        params_by_uid,
    )


def build_generator_representation_from_params(
    *,
    components: list[Component],
    params_by_uid: dict[int, GeneratorParams],
    selected_ids: frozenset[int],
    surface_id: str,
) -> RepresentationBuild:
    record_bytes: dict[int, bytes] = {}
    decoded_masks: dict[int, np.ndarray] = {}
    dims: dict[int, int] = {}
    for comp in components:
        if comp.uid not in selected_ids:
            continue
        params = params_by_uid[comp.uid]
        rec = encode_generator_params(params)
        parsed = decode_generator_params(rec)
        record_bytes[comp.uid] = rec
        decoded_masks[comp.uid] = rasterize_generator(parsed)
        dims[comp.uid] = parsed.dim
    frame_records = frame_records_from_component_records(components, selected_ids, record_bytes)
    raw = b"".join(frame_records)
    return RepresentationBuild(
        surface_id=surface_id,
        kind=PE1_GENERATOR,
        raw=raw,
        frame_records=frame_records,
        component_masks=decoded_masks,
        component_dims=dims,
        component_bytes={uid: len(rec) for uid, rec in record_bytes.items()},
        scope_note="local Laguerre/power-diagram generator pair; boundary values fall out from the bisector at decode",
        falls_out=True,
        selected_component_ids=selected_ids,
        metadata={"generator_quantization": "quarter-pixel q4 int16"},
    )


def build_section(rep: RepresentationBuild, selected_codec: str, selected_payload: bytes) -> bytes:
    header = PE1_HEADER.pack(
        PE1_MAGIC,
        PE1_VERSION,
        SEG_H,
        SEG_W,
        N_PAIRS,
        rep.kind,
        CODEC_IDS[selected_codec],
        len(rep.raw),
        len(rep.frame_records),
        hashlib.sha256(rep.raw).digest(),
    )
    section = header + selected_payload
    parsed = parse_pe1_section(section)
    if parsed["raw_sha256"] != sha256_bytes(rep.raw):
        raise PE1Error("section parse-back raw SHA differs")
    return section


def decode_body(codec: str, payload: bytes, expected_len: int) -> bytes:
    if codec == "brotli-q11":
        try:
            raw = brotli.decompress(payload)
        except brotli.error as exc:
            raise PE1Error("Brotli decode failed") from exc
        if len(raw) != expected_len:
            raise PE1Error("Brotli raw length mismatch")
        return raw
    if codec == "lzma1-raw":
        return unlzma1_raw(payload, expected_len)
    if codec == "smevr-r7-nibble":
        raw = b"".join(bd1.unsmevr_records(payload))
        if len(raw) != expected_len:
            raise PE1Error("SMEVR raw length mismatch")
        return raw
    raise PE1Error(f"unknown codec {codec!r}")


def parse_pe1_section(section: bytes) -> dict[str, Any]:
    if len(section) < PE1_HEADER.size:
        raise PE1Error("PE1 section header truncated")
    magic, version, seg_h, seg_w, n_pairs, kind, codec_id, raw_len, record_count, raw_sha = PE1_HEADER.unpack_from(
        section, 0
    )
    if magic != PE1_MAGIC or version != PE1_VERSION:
        raise PE1Error("PE1 section magic/version differs")
    if (seg_h, seg_w, n_pairs) != (SEG_H, SEG_W, N_PAIRS):
        raise PE1Error("PE1 section geometry differs from the n600 contract")
    codec = ID_CODECS.get(int(codec_id))
    if codec is None:
        raise PE1Error(f"unknown PE1 codec id {codec_id}")
    raw = decode_body(codec, section[PE1_HEADER.size :], int(raw_len))
    if hashlib.sha256(raw).digest() != raw_sha:
        raise PE1Error("PE1 raw SHA-256 mismatch")
    frames_seen = 0
    component_records = 0
    offset = 0
    for _pair in range(n_pairs):
        frames_seen += 1
        count, offset = read_varint(raw, offset)
        component_records += int(count)
        for _ in range(count):
            length, offset = read_varint(raw, offset)
            offset += int(length)
            if offset > len(raw):
                raise PE1Error("PE1 raw component stream truncated")
    if offset != len(raw):
        raise PE1Error("PE1 raw body has trailing bytes")
    if frames_seen != record_count:
        raise PE1Error("PE1 record_count differs from decoded frames")
    return {
        "codec": codec,
        "kind": int(kind),
        "kind_name": PE1_KIND_NAMES.get(int(kind), "unknown"),
        "raw_bytes": len(raw),
        "raw_sha256": sha256_bytes(raw),
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "frame_records": frames_seen,
        "component_records": component_records,
    }


def component_metrics(components: list[Component], rep: RepresentationBuild) -> dict[str, Any]:
    source_total = 0
    decoded_total = 0
    intersection_total = 0
    union_total = 0
    hausdorff_values: list[float] = []
    selected = rep.selected_component_ids
    for comp in components:
        if comp.uid in selected:
            source = component_mask(comp.flat)
            decoded = rep.component_masks[comp.uid]
        else:
            count = int(comp.flat.size)
            source_total += count
            union_total += count
            continue
        inter = source & decoded
        union = source | decoded
        source_total += int(source.sum())
        decoded_total += int(decoded.sum())
        intersection_total += int(inter.sum())
        union_total += int(union.sum())
        if comp.uid in selected and np.any(decoded):
            hausdorff_values.append(component_hausdorff(source, decoded, comp.bbox))
    return {
        "source_band_pixels": source_total,
        "decoded_band_pixels": decoded_total,
        "intersection_pixels": intersection_total,
        "union_pixels": union_total,
        "recall": intersection_total / source_total if source_total else 1.0,
        "precision": intersection_total / decoded_total if decoded_total else 1.0,
        "iou": intersection_total / union_total if union_total else 1.0,
        "hausdorff_px_mean": float(np.mean(hausdorff_values)) if hausdorff_values else None,
        "hausdorff_px_p95": float(np.percentile(hausdorff_values, 95)) if hausdorff_values else None,
        "hausdorff_px_max": float(np.max(hausdorff_values)) if hausdorff_values else None,
        "selected_components": len(selected),
        "all_components": len(components),
        "component_recall": len(selected) / len(components) if components else 1.0,
    }


def component_hausdorff(source: np.ndarray, decoded: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    y0, x0, y1, x1 = bbox
    py0 = max(0, y0 - 4)
    px0 = max(0, x0 - 4)
    py1 = min(SEG_H, y1 + 4)
    px1 = min(SEG_W, x1 + 4)
    s = source[py0:py1, px0:px1]
    d = decoded[py0:py1, px0:px1]
    if not np.any(s) and not np.any(d):
        return 0.0
    if not np.any(s) or not np.any(d):
        return float("inf")
    dist_to_d = ndimage.distance_transform_edt(~d)
    dist_to_s = ndimage.distance_transform_edt(~s)
    return float(max(float(dist_to_d[s].max()), float(dist_to_s[d].max())))


def archive_projection(base_payload: bytes, section: bytes) -> dict[str, Any]:
    bulk, sections = parse_payload(base_payload)
    new_payload = build_payload(bulk, [*sections, section])
    archive_zip = build_single_member_zip(new_payload, name="0.bin")
    return {
        "projected_archive_bytes": len(archive_zip),
        "projected_archive_sha256": sha256_bytes(archive_zip),
        "payload_bytes": len(new_payload),
        "payload_sha256": sha256_bytes(new_payload),
        "joint_section_count": len(sections) + 1,
    }


def build_row(
    rep: RepresentationBuild,
    *,
    components: list[Component],
    base_payload: bytes,
    artifact_dir: Path,
    store_best: bool,
) -> tuple[dict[str, Any], bytes, str, bytes]:
    coders, selected_codec, selected_payload = race_coders(
        surface_id=rep.surface_id,
        raw=rep.raw,
        frame_records=rep.frame_records,
        artifact_dir=artifact_dir,
        store_best=store_best,
    )
    section = build_section(rep, selected_codec, selected_payload)
    parsed = parse_pe1_section(section)
    metrics = component_metrics(components, rep)
    projection = archive_projection(base_payload, section)
    best = min(coders, key=lambda row: row.bytes)
    selected_flip_mass = sum(comp.flip_mass for comp in components if comp.uid in rep.selected_component_ids)
    total_flip_mass = sum(comp.flip_mass for comp in components)
    dim_values = list(rep.component_dims.values())
    row = {
        "surface_id": rep.surface_id,
        "representation": PE1_KIND_NAMES[rep.kind],
        "scope_note": rep.scope_note,
        "falls_out": rep.falls_out,
        "raw_bytes": len(rep.raw),
        "raw_sha256": sha256_bytes(rep.raw),
        "frame_records": len(rep.frame_records),
        "component_records": parsed["component_records"],
        "dim_per_edge_mean": float(np.mean(dim_values)) if dim_values else 0.0,
        "dim_per_edge_p95": float(np.percentile(dim_values, 95)) if dim_values else 0.0,
        "best_codec": best.codec,
        "best_body_bytes": best.bytes,
        "best_body_sha256": best.sha256,
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "section_bits_per_band_px": 8.0 * len(section) / max(1, metrics["source_band_pixels"]),
        "body_bits_per_band_px": 8.0 * best.bytes / max(1, metrics["source_band_pixels"]),
        "coder_race": list(coders),
        "mask_domain_fidelity": metrics,
        "selected_flip_mass": selected_flip_mass,
        "total_flip_mass": total_flip_mass,
        "flip_recall": selected_flip_mass / total_flip_mass if total_flip_mass else 1.0,
        "archive_projection": projection,
        "parse_back": parsed,
        "claim_label": "MEASURED n600 mask-domain coder bytes and parse-back; scorer-free",
        "metadata": rep.metadata,
    }
    return row, section, selected_codec, selected_payload


def _absolute_generator_fields(params: GeneratorParams) -> tuple[int, ...]:
    y0, x0, y1, x1 = params.bbox
    return (
        y0,
        x0,
        y1 - y0,
        x1 - x0,
        params.gen_a_q4[0],
        params.gen_a_q4[1],
        params.gen_b_q4[0],
        params.gen_b_q4[1],
    )


def track_generator_components(
    components: list[Component],
    params_by_uid: dict[int, GeneratorParams],
    *,
    max_distance_px: float,
) -> dict[int, int]:
    tracks: dict[int, int] = {}
    next_track = 0
    active: dict[tuple[int, int], list[Component]] = {}
    by_pair: list[list[Component]] = [[] for _ in range(N_PAIRS)]
    for comp in components:
        if comp.uid in params_by_uid:
            by_pair[comp.pair].append(comp)
    for pair in range(N_PAIRS):
        grouped: dict[tuple[int, int], list[Component]] = {}
        for comp in by_pair[pair]:
            grouped.setdefault(comp.edge, []).append(comp)
        for edge, current in grouped.items():
            previous = active.get(edge, [])
            matched_current: set[int] = set()
            matched_previous: set[int] = set()
            if previous and current:
                prev_xy = np.asarray([p.centroid_yx for p in previous], dtype=np.float64)
                cur_xy = np.asarray([c.centroid_yx for c in current], dtype=np.float64)
                dist = np.linalg.norm(prev_xy[:, None, :] - cur_xy[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(dist)
                for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                    if float(dist[r, c]) <= max_distance_px:
                        tracks[current[c].uid] = tracks[previous[r].uid]
                        matched_current.add(c)
                        matched_previous.add(r)
            for i, comp in enumerate(current):
                if i not in matched_current:
                    tracks[comp.uid] = next_track
                    next_track += 1
            active[edge] = current
    return tracks


def build_generator_transport_representation(
    *,
    components: list[Component],
    params_by_uid: dict[int, GeneratorParams],
    selected_ids: frozenset[int],
    max_distance_px: float,
) -> RepresentationBuild:
    selected_components = [comp for comp in components if comp.uid in selected_ids]
    tracks = track_generator_components(selected_components, params_by_uid, max_distance_px=max_distance_px)
    previous_by_track: dict[int, tuple[int, ...]] = {}
    frame_records: list[bytes] = []
    record_bytes_by_uid: dict[int, bytes] = {}
    dims: dict[int, int] = {}
    xi_errors: list[float] = []
    components_by_pair: list[list[Component]] = [[] for _ in range(N_PAIRS)]
    for comp in selected_components:
        components_by_pair[comp.pair].append(comp)
    for _pair, comps in enumerate(components_by_pair):
        out = bytearray(varint(len(comps)))
        for comp in comps:
            params = params_by_uid[comp.uid]
            fields = _absolute_generator_fields(params)
            track_id = tracks[comp.uid]
            rec = bytearray([params.edge[0], params.edge[1]])
            rec += varint(track_id)
            previous = previous_by_track.get(track_id)
            if previous is None:
                rec.append(0)
                for value in fields:
                    rec += write_zigzag(value)
            else:
                rec.append(1)
                for value, prev in zip(fields, previous, strict=True):
                    rec += write_zigzag(value - prev)
                dy = (fields[4] - previous[4]) / 4.0
                dx = (fields[5] - previous[5]) / 4.0
                xi = np.asarray([dy, dx, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                rt = se3_np.log_se3(se3_np.exp_se3(xi))
                xi_errors.append(float(np.max(np.abs(rt - xi))))
            previous_by_track[track_id] = fields
            rec_b = bytes(rec)
            record_bytes_by_uid[comp.uid] = rec_b
            dims[comp.uid] = 4
            out += varint(len(rec_b)) + rec_b
        frame_records.append(bytes(out))

    raw = b"".join(frame_records)
    masks = {uid: rasterize_generator(params_by_uid[uid]) for uid in selected_ids}
    return RepresentationBuild(
        surface_id="generator_pair_xi_transport",
        kind=PE1_GENERATOR_TRANSPORT,
        raw=raw,
        frame_records=tuple(frame_records),
        component_masks=masks,
        component_dims=dims,
        component_bytes={uid: len(rec) for uid, rec in record_bytes_by_uid.items()},
        scope_note="tracks generator parameters across n600 and codes xi translation innovations over parameters, not pixels",
        falls_out=True,
        selected_component_ids=selected_ids,
        metadata={
            "track_count": len(set(tracks.values())),
            "max_distance_px": max_distance_px,
            "xi_transport_roundtrip_max_abs": max(xi_errors) if xi_errors else 0.0,
            "lie_convention": se3_np.CONVENTION,
        },
    )


def selected_by_waterfill(components: list[Component], count: int) -> frozenset[int]:
    ranked = sorted(components, key=lambda c: (-c.flip_mass, c.edge, c.pair, c.uid))
    return frozenset(comp.uid for comp in ranked[:count])


def build_waterfill_rows(
    *,
    components: list[Component],
    base_payload: bytes,
    artifact_dir: Path,
    lstars: np.ndarray,
    explicit_stride: int,
    curve_component_bytes: dict[int, int],
    generator_component_bytes: dict[int, int],
    generator_params: dict[int, GeneratorParams],
    store_best: bool,
    target_section_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    sections: dict[str, bytes] = {}
    ranked = sorted(components, key=lambda c: (-c.flip_mass, c.edge, c.pair, c.uid))

    def estimated_counts_from_component_bytes(component_bytes: dict[int, int]) -> list[int]:
        cumulative = 0
        estimate = 1
        for index, comp in enumerate(ranked, start=1):
            record_len = component_bytes.get(comp.uid, 0)
            cumulative += len(varint(record_len)) + record_len
            if cumulative <= target_section_bytes:
                estimate = index
        guesses = {
            1,
            estimate,
            max(1, estimate // 2),
            max(1, (estimate * 3) // 4),
            min(len(components), max(1, (estimate * 5) // 4)),
            min(len(components), max(1, (estimate * 3) // 2)),
            min(len(components), max(1, estimate * 2)),
        }
        return sorted(guesses)

    # Targeted 75KB search per representation family. Prefix counts are estimated
    # from actual component-record byte sizes, then each candidate is priced with
    # the real coder race. This avoids spending q11/LZMA time on redundant curve
    # points while preserving measured coder evidence for the selected rows.
    for family in ("curve", "generator"):
        best: tuple[int, dict[str, Any], bytes] | None = None
        estimates = (
            estimated_counts_from_component_bytes(curve_component_bytes)
            if family == "curve"
            else estimated_counts_from_component_bytes(generator_component_bytes)
        )
        priced_rows: list[dict[str, Any]] = []
        for count in estimates:
            selected = frozenset(comp.uid for comp in ranked[:count])
            if family == "curve":
                rep = build_curve_representation(components=components, knot_stride=explicit_stride, selected_ids=selected)
            else:
                rep = build_generator_representation_from_params(
                    components=components,
                    params_by_uid=generator_params,
                    selected_ids=selected,
                    surface_id="generator_pair_bisector",
                )
            row, section, _codec, _payload = build_row(
                rep,
                components=components,
                base_payload=base_payload,
                artifact_dir=artifact_dir,
                store_best=False,
            )
            priced_rows.append(row)
            if row["section_bytes"] <= target_section_bytes and (
                best is None or row["selected_flip_mass"] > best[1]["selected_flip_mass"]
            ):
                best = (count, row, section)
        if best is None and priced_rows:
            smallest = min(priced_rows, key=lambda row: row["section_bytes"])
            # Rebuild the smallest section so the returned section matches the row.
            count = int(smallest["mask_domain_fidelity"]["selected_components"])
            selected = frozenset(comp.uid for comp in ranked[:count])
            if family == "curve":
                rep = build_curve_representation(components=components, knot_stride=explicit_stride, selected_ids=selected)
            else:
                rep = build_generator_representation_from_params(
                    components=components,
                    params_by_uid=generator_params,
                    selected_ids=selected,
                    surface_id="generator_pair_bisector",
                )
            row, section, _codec, _payload = build_row(
                rep,
                components=components,
                base_payload=base_payload,
                artifact_dir=artifact_dir,
                store_best=False,
            )
            best = (count, row, section)
        if best is not None:
            count, row, section = best
            row["surface_id"] = f"{row['surface_id']}_waterfill_75kb"
            row["waterfill_rank_prefix_components"] = count
            row["waterfill_target_section_bytes"] = target_section_bytes
            row["waterfill_target"] = family
            row["waterfill_priced_prefix_counts"] = estimates
            rows.append(row)
            sections[row["surface_id"]] = section
    rows.sort(key=lambda row: (row.get("waterfill_target_section_bytes", 10**12), -row["selected_flip_mass"]))
    return rows, sections


def write_candidate(
    *,
    base_sub: Path,
    candidate_dir: Path,
    section: bytes,
    joint_name: str,
) -> dict[str, Any]:
    if candidate_dir.exists():
        raise PE1Error(f"candidate dir already exists: {candidate_dir}")
    bd1.copy_runtime_tree(base_sub, candidate_dir)
    base_payload = bd1.read_archive_payload(base_sub / "archive.zip")
    bulk, sections = parse_payload(base_payload)
    payload = build_payload(bulk, [*sections, section])
    archive_zip = build_single_member_zip(payload, name="0.bin")
    (candidate_dir / "archive" / "0.bin").write_bytes(payload)
    (candidate_dir / "archive.zip").write_bytes(archive_zip)
    parse_back = bd1.build_local_ledger(
        candidate_dir / "archive.zip",
        ("config", "renderer", "selector", "pose_warp", "frame0_pose_repair", joint_name),
    )
    pe1_parse = parse_pe1_section(parse_payload(payload)[1][-1])
    return {
        "submission_dir": str(candidate_dir),
        "archive_bytes": (candidate_dir / "archive.zip").stat().st_size,
        "archive_sha256": sha256_file(candidate_dir / "archive.zip"),
        "payload_sha256": sha256_bytes(payload),
        "delta_bytes_vs_qo1": (candidate_dir / "archive.zip").stat().st_size - BASELINE_BYTES,
        "rate_delta_S_vs_qo1": 25.0 * ((candidate_dir / "archive.zip").stat().st_size - BASELINE_BYTES) / RATE_DENOM,
        "parse_back": parse_back,
        "pe1_section_parse_back": pe1_parse,
        "label": "BYTE-CLOSED / SECTION-PARSE-BACK / RUNTIME-SURVIVAL-UNMEASURED / score_claim=false",
    }


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    rows = receipt["representation_rows"]
    lines = [
        "# ddm_pe1 per-edge partition representation race - 2026-08-05",
        "",
        "Status: **BYTE-CLOSED / SECTION-PARSE-BACK / RUNTIME-SURVIVAL-UNMEASURED / score_claim=false**.",
        "",
        f"Own-vehicle baseline: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`. No scorer job was run.",
        "",
        "## RECALL EVIDENCE",
        "",
    ]
    for item in receipt["recall_evidence"]:
        lines.append(f"- `{item['source']}`: {item['finding']} Plan impact: {item['plan_impact']}")
    lines.extend(
        [
            "",
            "## Typed Table",
            "",
            "| representation | dim/edge | falls-out? | amortization factor | bits/band-px | recall | IoU | coded bytes full | coded bytes @75KB | vs bf1 205,196 B | vs 0.60 currency |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    waterfill_75 = {
        row["waterfill_target"]: row
        for row in receipt["waterfill_rows"]
        if row.get("waterfill_target_section_bytes") == 75_000
    }
    for row in rows:
        wf = waterfill_75.get("curve" if row["representation"] == "explicit_curve_spline" else "generator")
        amort = receipt["transport"].get("amortization_factor") if row["representation"] == "generator_pair_bisector" else None
        metrics = row["mask_domain_fidelity"]
        amort_text = f"{amort:.6f}" if amort is not None else "n/a"
        lines.append(
            f"| `{row['surface_id']}` | `{row['dim_per_edge_mean']:.2f}` | "
            f"`{row['falls_out']}` | `{amort_text}` | "
            f"`{row['section_bits_per_band_px']:.6f}` | `{metrics['recall']:.6f}` | "
            f"`{metrics['iou']:.6f}` | `{row['section_bytes']}` | "
            f"`{wf['section_bytes'] if wf else 'n/a'}` | "
            f"`{row['section_bytes'] / 205196:.6f}x` | "
            f"`{row['section_bits_per_band_px'] / 0.60:.6f}x` |"
        )
    lines.extend(
        [
            "",
            "## Edge Ranking",
            "",
        ]
    )
    for edge in receipt["edge_extraction"]["edge_ranking"][:10]:
        lines.append(
            f"- `{edge['edge']}`: source band `{edge['source_band_pixels']}` px, "
            f"components `{edge['components']}`, flip mass `{edge['flip_mass']}` "
            f"({edge['flip_mass_fraction']:.6f} of in-band flip mass)."
        )
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            f"- Full-component candidate: `{receipt['full_candidate']['archive_bytes']}` B, sha256 `{receipt['full_candidate']['archive_sha256']}`, path `{receipt['full_candidate']['submission_dir']}`.",
            f"- 75KB surgical candidate: `{receipt['surgical_candidate']['archive_bytes']}` B, sha256 `{receipt['surgical_candidate']['archive_sha256']}`, path `{receipt['surgical_candidate']['submission_dir']}`.",
            "",
            "## Boundaries",
            "",
        ]
    )
    for boundary in receipt["boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend(["", "## Follow-On Disposition", ""])
    for item in receipt["follow_on_disposition"]:
        lines.append(f"- **{item['status']}** `{item['id']}`: {item['action']}")
    lines.extend(
        [
            "",
            "## NEXT-IF-RESUMED",
            "",
            receipt["next_if_resumed"],
            "",
            f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; pe1 did not run a scorer and did not move the contest pointer.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def load_current_argmax(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise PE1Error(f"unexpected current argmax shape {arr.shape}")
    return arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sub", type=Path, default=DEFAULT_BASE_SUB)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--curve-strides", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--transport-max-distance-px", type=float, default=24.0)
    parser.add_argument("--store-best", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    args = parser.parse_args(argv)

    run_id = args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    run_ssd = args.ssd_dir / run_id
    run_ssd.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_ssd / "payloads"
    free = shutil.disk_usage(args.ssd_dir).free
    if free < 512 * 1024 * 1024:
        raise PE1Error(f"SSD storage preflight failed: only {free} bytes free at {args.ssd_dir}")

    base_archive = args.base_sub / "archive.zip"
    base_sha = sha256_file(base_archive)
    if base_sha != BASELINE_ARCHIVE_SHA256:
        raise PE1Error(f"base archive SHA drift: {base_sha}")
    base_payload = bd1.read_archive_payload(base_archive)
    bulk, sections = parse_payload(base_payload)
    if len(sections) != 5:
        raise PE1Error(f"qo1 base expected 5 IX2 sections, got {len(sections)}")

    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    current = load_current_argmax(args.current_argmax)
    components, extraction = extract_components(lstars, current)
    all_ids = frozenset(comp.uid for comp in components)

    rows: list[dict[str, Any]] = []
    sections_by_surface: dict[str, bytes] = {}
    explicit_rows: list[dict[str, Any]] = []
    explicit_sections: dict[str, bytes] = {}
    explicit_reps: dict[str, RepresentationBuild] = {}
    for stride in args.curve_strides:
        rep = build_curve_representation(components=components, knot_stride=stride, selected_ids=all_ids)
        row, section, _codec, _payload = build_row(
            rep,
            components=components,
            base_payload=base_payload,
            artifact_dir=artifact_dir,
            store_best=args.store_best,
        )
        explicit_rows.append(row)
        explicit_sections[row["surface_id"]] = section
        explicit_reps[row["surface_id"]] = rep
    # MDL: smallest full-edge section among rows with at least 98% component-band recall.
    viable_explicit = [r for r in explicit_rows if r["mask_domain_fidelity"]["recall"] >= 0.98]
    selected_explicit = min(viable_explicit or explicit_rows, key=lambda row: row["section_bytes"])
    rows.extend(explicit_rows)
    sections_by_surface.update(explicit_sections)

    gen_rep, gen_params = build_generator_representation(components=components, lstars=lstars, selected_ids=all_ids)
    gen_row, gen_section, _codec, _payload = build_row(
        gen_rep,
        components=components,
        base_payload=base_payload,
        artifact_dir=artifact_dir,
        store_best=args.store_best,
    )
    rows.append(gen_row)
    sections_by_surface[gen_row["surface_id"]] = gen_section

    transport_rep = build_generator_transport_representation(
        components=components,
        params_by_uid=gen_params,
        selected_ids=all_ids,
        max_distance_px=args.transport_max_distance_px,
    )
    transport_row, transport_section, _codec, _payload = build_row(
        transport_rep,
        components=components,
        base_payload=base_payload,
        artifact_dir=artifact_dir,
        store_best=args.store_best,
    )
    rows.append(transport_row)
    sections_by_surface[transport_row["surface_id"]] = transport_section
    transport = {
        "independent_body_bytes": gen_row["best_body_bytes"],
        "transport_body_bytes": transport_row["best_body_bytes"],
        "amortization_factor": gen_row["best_body_bytes"] / max(1, transport_row["best_body_bytes"]),
        "track_count": transport_row["metadata"]["track_count"],
        "xi_transport_roundtrip_max_abs": transport_row["metadata"]["xi_transport_roundtrip_max_abs"],
        "verdict": "MEASURED on generator parameters; not a pixel transport claim",
    }

    waterfill_rows, waterfill_sections = build_waterfill_rows(
        components=components,
        base_payload=base_payload,
        artifact_dir=artifact_dir,
        lstars=lstars,
        explicit_stride=int(selected_explicit["metadata"]["knot_stride"]),
        curve_component_bytes=explicit_reps[selected_explicit["surface_id"]].component_bytes,
        generator_component_bytes=gen_rep.component_bytes,
        generator_params=gen_params,
        store_best=args.store_best,
        target_section_bytes=75_000,
    )

    full_candidates = [
        row for row in rows if row["mask_domain_fidelity"]["recall"] >= 0.98
    ]
    full_winner = min(full_candidates or rows, key=lambda row: row["section_bytes"])
    surgical_candidates = [
        row for row in waterfill_rows if row.get("waterfill_target_section_bytes") == 75_000
    ]
    if not surgical_candidates:
        raise PE1Error("waterfill search found no 75KB candidate")
    surgical_winner = max(
        surgical_candidates,
        key=lambda row: (row["selected_flip_mass"], row["mask_domain_fidelity"]["iou"], -row["section_bytes"]),
    )

    if args.skip_candidates:
        full_candidate = {"skipped": True, "reason": "--skip-candidates"}
        surgical_candidate = {"skipped": True, "reason": "--skip-candidates"}
    else:
        full_candidate = write_candidate(
            base_sub=args.base_sub,
            candidate_dir=run_ssd / "sub_auto_pairbit_pe1_full",
            section=sections_by_surface[full_winner["surface_id"]],
            joint_name="pe1_per_edge_full",
        )
        surgical_section = waterfill_sections[surgical_winner["surface_id"]]
        surgical_candidate = write_candidate(
            base_sub=args.base_sub,
            candidate_dir=run_ssd / "sub_auto_pairbit_pe1_surgical_75kb",
            section=surgical_section,
            joint_name="pe1_per_edge_surgical",
        )

    receipt = {
        "schema": "ddm_pe1_per_edge_partition_race.v1",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free mask-domain byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "storage_preflight": {
            "path": str(args.ssd_dir),
            "observed_free_bytes": free,
            "required_free_bytes": 512 * 1024 * 1024,
            "status": "PASS",
        },
        "base": {
            "submission_dir": str(args.base_sub),
            "archive_bytes": base_archive.stat().st_size,
            "archive_sha256": base_sha,
            "payload_sha256": sha256_bytes(base_payload),
            "joint_section_count": len(sections),
            "own_vehicle_S": BASELINE_S,
            "axis": BASELINE_AXIS,
        },
        "inputs": {
            "gt_cache": str(args.gt_cache),
            "gt_cache_sha256": sha256_file(args.gt_cache) if args.hash_inputs else None,
            "current_argmax": str(args.current_argmax),
            "current_argmax_sha256": sha256_file(args.current_argmax) if args.hash_inputs else None,
            "selection_mode": "n600 all pairs; no prefix",
            "shape": [N_PAIRS, SEG_H, SEG_W],
            "class_order": dict(enumerate(CLASS_NAMES)),
        },
        "edge_extraction": extraction,
        "mdl_selection": {
            "explicit_curve_selected_surface": selected_explicit["surface_id"],
            "selection_rule": "min section bytes among full n600 rows with component-band recall >= 0.98; else min section bytes",
            "curve_strides": args.curve_strides,
        },
        "representation_rows": sorted(rows, key=lambda row: row["section_bytes"]),
        "transport": transport,
        "waterfill_rows": waterfill_rows,
        "full_winner": full_winner,
        "surgical_winner": surgical_winner,
        "full_candidate": full_candidate,
        "surgical_candidate": surgical_candidate,
        "recall_evidence": [
            {
                "source": ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md",
                "finding": "five criteria rank PE1 choices: dimensionality, falls-out-free, n600 reuse, surgical targeting, full-stack survival.",
                "plan_impact": "table reports every criterion explicitly and does not choose by bytes alone.",
            },
            {
                "source": "experiments/ddm_pc2_edge_decomp.py and .omx/research/ddm_pc2_perclass_road_edges_20260802.md",
                "finding": "Road<->Lane is the hub separatrix and interiors are negligible; edge view beats per-class splitting.",
                "plan_impact": "extracts class-pair edge components, ranks Road<->Lane first, and waterfills by component flip mass.",
            },
            {
                "source": ".omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md",
                "finding": "BD1CLF1 v2 receiver-section pattern and BF1 205,196 B n600 lane-crop row are the live comparison surface.",
                "plan_impact": "PE1 appends a counted IX2 section to the same qo1 base and reports vs 205,196 B and 0.60 bits/band-px.",
            },
            {
                "source": ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/ddm_g4_spatial_stationarity_receipt.json and tac.lie",
                "finding": "n600 reuse must be parameter/worldsheet transport, not pixel transport.",
                "plan_impact": "generator transport codes track-id generator-parameter innovations and records tac.lie SE(3) xi round-trip closure.",
            },
            {
                "source": ".venv/bin/python tools/list_canonical_equations.py --json",
                "finding": "partition temporal transport amortization and trajectory-derived stopping laws exist; no PE1-specific equation already settles this representation race.",
                "plan_impact": "PE1 measures the missing generator-transport cell instead of promoting prior equations as evidence.",
            },
        ],
        "boundaries": [
            "No SegNet/PoseNet scorer forward was run; sq2 owns the scorer slot.",
            "All byte counts are measured on real serialized coder outputs over n600 cached masks.",
            "The full candidate selects all extracted components, but the selected curve representation is not lossless unless its mask-domain row says so.",
            "PE1 candidate archives append counted video-derived sections; no data is hidden in free code.",
            "Candidate directories carry qo1 runtime plus PE1 bytes, but runtime RGB consumption is not claimed here.",
            "Generator fitting negative, if any, is FORMULATION-scoped to local two-generator bisectors on curved components.",
            "No upstream/ files were edited and no /tmp evidence is cited.",
        ],
        "follow_on_disposition": [
            {
                "id": "#941-per-edge-head",
                "status": "BUILT+PRICED",
                "action": "use this PE1 receipt's section rows and waterfill table as the priced per-edge head; runtime survival remains queued.",
            },
            {
                "id": "generator-transport-cell",
                "status": "MEASURED",
                "action": "consume the generator independent-vs-xi-transport amortization factor before any transport-family claim.",
            },
            {
                "id": "pe1-runtime-consumption",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "patch inflate_runner to consume PE1 sections into frame_1 RGB, prove absent identity, then queue one n600 scorer job after sq2 frees the slot.",
            },
            {
                "id": "cq2-75kb-student-reopen",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "route the surgical_75kb candidate only after runtime consumption proves nonzero scorer-cell survival.",
            },
        ],
        "next_if_resumed": (
            "Start from ddm_pe1_repr_race_receipt.json and the two candidate archives under the SSD run directory. "
            "Do not run a scorer while sq2 owns the slot. The next executable unit is PE1 runtime consumption with an absent-section identity proof, "
            "then a queued n600 scorer job on the runtime-consuming surgical candidate."
        ),
    }

    json_path = args.research_dir / "ddm_pe1_repr_race_receipt.json"
    md_path = args.research_dir / "PE1_RECEIPT_20260805.md"
    json_path.write_text(json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(json_path),
                "markdown": str(md_path),
                "full_winner": full_winner["surface_id"],
                "surgical_winner": surgical_winner["surface_id"],
                "own_vehicle_frontier": f"S = {BASELINE_S} @ {BASELINE_BYTES} B {BASELINE_AXIS}",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
