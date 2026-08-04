#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""bf1 scorer-free band-field representation race into BD1CLF1.

This extends the counted BD1CLF1 payload grammar to version 2 record layouts
that decode back to the same receiver class-field object.  It measures real
coder bytes on the n600 cached Road/Lane band from ``lstars`` and builds one
receiver-closed qo1 candidate for the byte-winning representation.  It does not
run SegNet or PoseNet and makes no score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

_REPO: Final = Path(__file__).resolve().parents[1]
for _path in (_REPO / "src", _REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_ix2_archive_container import (  # noqa: E402
    build_payload,
    build_single_member_zip,
    parse_payload,
)

import ddm_bd1_class_field_receiver as bd1  # noqa: E402


SEG_H: Final = 384
SEG_W: Final = 512
N_PAIRS: Final = 600
ROAD: Final = 0
LANE: Final = 1
RATE_DENOM: Final = 37_545_489
TOTAL_SITES: Final = N_PAIRS * SEG_H * SEG_W
BASELINE_S: Final = 0.7539807296911207
BASELINE_BYTES: Final = 357_836
BASELINE_AXIS: Final = "[macOS-CPU advisory]"
BASELINE_ARCHIVE_SHA256: Final = (
    "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a"
)
PER_EDGE_REFERENCE_BITS_PER_BAND_PIXEL: Final = 0.60

DEFAULT_BASE_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
DEFAULT_GT_CACHE: Final = _REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CURRENT_ARGMAX: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy")
DEFAULT_RESEARCH_DIR: Final = _REPO / ".omx/research/ddm_bf1_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_bf1_20260805")

BD1_HEADER_V2: Final = struct.Struct("<8sBHHHBBBBBBII32s")
BD1_VERSION_V2: Final = 2
REPR_DENSE_BITMAP: Final = 0
REPR_ROW_RUNS: Final = 1
REPR_LANE_CROP: Final = 2
REPR_NAMES: Final = {
    REPR_DENSE_BITMAP: "dense_bitmap",
    REPR_ROW_RUNS: "row_runs",
    REPR_LANE_CROP: "lane_crop",
}
CODEC_IDS: Final = {
    "lzma1-raw": bd1.BD1_LZMA1_RAW,
    "brotli-q11": bd1.BD1_BROTLI_Q11,
    "smevr-r7-nibble": bd1.BD1_SMEVR_R7_NIBBLE,
}

CQ2_THRESHOLDS: Final = {
    "25KB_student_side_implied": 0.516810,
    "25KB_student_explicit_direction": 0.611747,
    "75KB_student_side_implied": 0.759752,
    "75KB_student_explicit_direction": 0.854688,
}


class BF1Error(ValueError):
    """The bf1 representation build failed closed."""


@dataclass(frozen=True)
class CoderResult:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str | None


@dataclass(frozen=True)
class FieldPair:
    indices: np.ndarray
    lane_bits: np.ndarray


@dataclass(frozen=True)
class Representation:
    surface_id: str
    candidate_kind: str
    representation_kind: int
    raw: bytes
    records: tuple[bytes, ...]
    aux0: int
    field_pairs: tuple[FieldPair, ...]
    scope_note: str


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
        return value.__dict__
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_absent_identity_proof(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    proof = json.loads(path.read_text())
    required = {
        "byte_identical_to_qo1_shipped_decode",
        "raw_bytes",
        "raw_sha256",
        "expected_raw_bytes",
        "expected_raw_sha256",
        "output_raw",
        "command",
    }
    missing = sorted(required.difference(proof))
    if missing:
        raise BF1Error(f"absent identity proof missing keys: {missing}")
    if "/tmp" in str(proof):
        raise BF1Error("absent identity proof cites /tmp")
    if not proof["byte_identical_to_qo1_shipped_decode"]:
        raise BF1Error(f"absent identity proof failed: {proof}")
    return proof


def varint(value: int) -> bytes:
    if value < 0:
        raise BF1Error("varint cannot encode negative values")
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
            raise BF1Error("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7
        if shift > 63:
            raise BF1Error("varint too long")


def pack_bits(bits: np.ndarray) -> bytes:
    return bd1.pack_bits(np.asarray(bits, dtype=bool))


def unpack_bits(payload: bytes, count: int) -> np.ndarray:
    return bd1.unpack_bits(payload, count)


def band_for(labels: np.ndarray) -> np.ndarray:
    return bd1.road_lane_band(np.asarray(labels, dtype=np.uint8), radius=1)


def row_bounded_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return []
    rows = indices // SEG_W
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    prev_row = int(rows[0])
    length = 1
    for index, row in zip(indices[1:].tolist(), rows[1:].tolist(), strict=True):
        index = int(index)
        row = int(row)
        if row == prev_row and index == prev + 1:
            length += 1
        else:
            runs.append((start, length))
            start = index
            length = 1
            prev_row = row
        prev = index
    runs.append((start, length))
    return runs


def encode_row_runs(indices: np.ndarray, lane_bits: np.ndarray) -> bytes:
    runs = row_bounded_runs(indices)
    out = bytearray(varint(len(runs)))
    prev_start = 0
    total = 0
    for n, (start, length) in enumerate(runs):
        out += varint(start if n == 0 else start - prev_start)
        out += varint(length)
        prev_start = start
        total += length
    if total != int(np.asarray(indices).size):
        raise BF1Error("run coverage count drifted")
    out += pack_bits(lane_bits)
    return bytes(out)


def decode_row_runs_record(record: bytes) -> FieldPair:
    run_count, offset = read_varint(record, 0)
    indices: list[int] = []
    prev_start = 0
    prev_end = 0
    total = 0
    for n in range(run_count):
        start_delta, offset = read_varint(record, offset)
        length, offset = read_varint(record, offset)
        if length == 0:
            raise BF1Error("row-run length is zero")
        start = start_delta if n == 0 else prev_start + start_delta
        end = start + length
        if start < prev_end or end > SEG_H * SEG_W:
            raise BF1Error("row-run order or bounds differ")
        indices.extend(range(start, end))
        prev_start = start
        prev_end = end
        total += length
    side_bytes = (total + 7) // 8
    lane_bits = unpack_bits(record[offset:offset + side_bytes], total)
    if offset + side_bytes != len(record):
        raise BF1Error("row-run record has trailing bytes")
    return FieldPair(np.asarray(indices, dtype=np.int32), lane_bits)


def decode_row_runs_raw(raw: bytes) -> tuple[FieldPair, ...]:
    pairs: list[FieldPair] = []
    offset = 0
    for _ in range(N_PAIRS):
        start = offset
        run_count, offset = read_varint(raw, offset)
        total = 0
        for _run in range(run_count):
            _start_delta, offset = read_varint(raw, offset)
            length, offset = read_varint(raw, offset)
            total += length
        side_bytes = (total + 7) // 8
        offset += side_bytes
        pairs.append(decode_row_runs_record(raw[start:offset]))
    if offset != len(raw):
        raise BF1Error("row-run raw body has trailing bytes")
    return tuple(pairs)


def dense_records(lstars: np.ndarray) -> Representation:
    records: list[bytes] = []
    fields: list[FieldPair] = []
    band_bytes = (SEG_H * SEG_W + 7) // 8
    for pair in range(N_PAIRS):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        band = band_for(labels)
        indices = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        lane_bits = np.asarray(labels.reshape(-1)[indices] == LANE, dtype=bool)
        records.append(pack_bits(band) + pack_bits(lane_bits))
        fields.append(FieldPair(indices, lane_bits))
    raw = b"".join(records)
    return Representation(
        surface_id="dense_band_bitmap_v2",
        candidate_kind="lossless dense scorer-grid band bitmap plus Road/Lane side bits",
        representation_kind=REPR_DENSE_BITMAP,
        raw=raw,
        records=tuple(records),
        aux0=band_bytes,
        field_pairs=tuple(fields),
        scope_note="lossless full source band; BD1 v1-equivalent content under v2 header",
    )


def row_run_records(lstars: np.ndarray) -> Representation:
    records: list[bytes] = []
    fields: list[FieldPair] = []
    for pair in range(N_PAIRS):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        band = band_for(labels)
        indices = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        lane_bits = np.asarray(labels.reshape(-1)[indices] == LANE, dtype=bool)
        record = encode_row_runs(indices, lane_bits)
        decoded = decode_row_runs_record(record)
        if not np.array_equal(decoded.indices, indices) or not np.array_equal(decoded.lane_bits, lane_bits):
            raise BF1Error("row-run local roundtrip failed")
        records.append(record)
        fields.append(decoded)
    return Representation(
        surface_id="lossless_row_runs",
        candidate_kind="lossless row-bounded runs over the Road/Lane band plus side bits",
        representation_kind=REPR_ROW_RUNS,
        raw=b"".join(records),
        records=tuple(records),
        aux0=0,
        field_pairs=tuple(fields),
        scope_note="lossless full source band; receiver-closed without scorer-derived chart",
    )


def lane_crop_records(lstars: np.ndarray) -> Representation:
    records: list[bytes] = []
    fields: list[FieldPair] = []
    st3 = ndimage.generate_binary_structure(2, 2)
    for pair in range(N_PAIRS):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        lane = labels == LANE
        ys, xs = np.nonzero(lane)
        if ys.size == 0:
            record = struct.pack("<HHHH", 0, 0, 0, 0)
            decoded_lane = np.zeros((SEG_H, SEG_W), dtype=bool)
        else:
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            crop = np.ascontiguousarray(lane[y0:y1, x0:x1])
            record = struct.pack("<HHHH", y0, x0, y1 - y0, x1 - x0) + pack_bits(crop)
            decoded_lane = np.zeros((SEG_H, SEG_W), dtype=bool)
            decoded_lane[y0:y1, x0:x1] = crop
        band = ndimage.binary_dilation(decoded_lane, st3, iterations=1)
        indices = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        lane_bits = np.asarray(decoded_lane.reshape(-1)[indices], dtype=bool)
        records.append(record)
        fields.append(FieldPair(indices, lane_bits))
    return Representation(
        surface_id="rl1_lane_crop_bf1",
        candidate_kind="lossy per-pair Lane crop; receiver derives band as 3x3 Lane dilation",
        representation_kind=REPR_LANE_CROP,
        raw=b"".join(records),
        records=tuple(records),
        aux0=0,
        field_pairs=tuple(fields),
        scope_note="settles #939 description-half at n600; loss is false-positive band expansion, not missed source band",
    )


def se3_receiver_closed_sparse_records(lstars: np.ndarray, current_argmax: np.ndarray) -> Representation:
    if tuple(current_argmax.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise BF1Error(f"unexpected current argmax shape {current_argmax.shape}")
    records: list[bytes] = []
    fields: list[FieldPair] = []
    for pair in range(N_PAIRS):
        gt_labels = np.asarray(lstars[pair], dtype=np.uint8)
        cur = np.asarray(current_argmax[pair], dtype=np.uint8)
        target = (gt_labels != cur) & (
            ((gt_labels == ROAD) & (cur == LANE)) | ((gt_labels == LANE) & (cur == ROAD))
        )
        indices = np.flatnonzero(target.reshape(-1)).astype(np.int32)
        lane_bits = np.asarray(gt_labels.reshape(-1)[indices] == LANE, dtype=bool)
        record = encode_row_runs(indices, lane_bits)
        decoded = decode_row_runs_record(record)
        if not np.array_equal(decoded.indices, indices) or not np.array_equal(decoded.lane_bits, lane_bits):
            raise BF1Error("SE3 sparse local roundtrip failed")
        records.append(record)
        fields.append(decoded)
    return Representation(
        surface_id="se3_receiver_closed_sparse_corrections",
        candidate_kind="receiver-closed sparse Road/Lane correction positions plus target side bits",
        representation_kind=REPR_ROW_RUNS,
        raw=b"".join(records),
        records=tuple(records),
        aux0=0,
        field_pairs=tuple(fields),
        scope_note=(
            "pays absolute correction support to close the RF1 blocker; not the prior 81KB/101KB "
            "assumption-scoped SE3 chart stream"
        ),
    )


def decode_dense_raw(raw: bytes, band_bytes: int) -> tuple[FieldPair, ...]:
    fields: list[FieldPair] = []
    offset = 0
    slots = SEG_H * SEG_W
    for _ in range(N_PAIRS):
        band = unpack_bits(raw[offset:offset + band_bytes], slots)
        offset += band_bytes
        indices = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        side_bytes = (indices.size + 7) // 8
        lane_bits = unpack_bits(raw[offset:offset + side_bytes], int(indices.size))
        offset += side_bytes
        fields.append(FieldPair(indices, lane_bits))
    if offset != len(raw):
        raise BF1Error("dense raw body has trailing bytes")
    return tuple(fields)


def decode_lane_crop_raw(raw: bytes) -> tuple[FieldPair, ...]:
    fields: list[FieldPair] = []
    offset = 0
    st3 = ndimage.generate_binary_structure(2, 2)
    for _ in range(N_PAIRS):
        if offset + 8 > len(raw):
            raise BF1Error("Lane-crop record header is truncated")
        y0, x0, height, width = struct.unpack_from("<HHHH", raw, offset)
        offset += 8
        if y0 + height > SEG_H or x0 + width > SEG_W:
            raise BF1Error("Lane crop exceeds grid")
        bit_count = int(height) * int(width)
        payload_bytes = (bit_count + 7) // 8
        lane = np.zeros((SEG_H, SEG_W), dtype=bool)
        if bit_count:
            crop = unpack_bits(raw[offset:offset + payload_bytes], bit_count).reshape(int(height), int(width))
            lane[int(y0):int(y0) + int(height), int(x0):int(x0) + int(width)] = crop
        offset += payload_bytes
        band = ndimage.binary_dilation(lane, st3, iterations=1)
        indices = np.flatnonzero(band.reshape(-1)).astype(np.int32)
        lane_bits = lane.reshape(-1)[indices]
        fields.append(FieldPair(indices, lane_bits))
    if offset != len(raw):
        raise BF1Error("Lane-crop raw body has trailing bytes")
    return tuple(fields)


def parse_section_v2(section: bytes) -> tuple[str, tuple[FieldPair, ...]]:
    if len(section) < BD1_HEADER_V2.size:
        raise BF1Error("BD1 v2 section truncated")
    (magic, version, seg_h, seg_w, n_pairs, radius, road_cls, lane_cls, paint_mode,
     repr_kind, codec, raw_len, aux0, raw_sha) = BD1_HEADER_V2.unpack_from(section, 0)
    if magic != bd1.BD1_MAGIC or version != BD1_VERSION_V2:
        raise BF1Error("BD1 v2 section magic/version differs")
    if (seg_h, seg_w, n_pairs, radius, road_cls, lane_cls, paint_mode) != (
        SEG_H,
        SEG_W,
        N_PAIRS,
        1,
        ROAD,
        LANE,
        1,
    ):
        raise BF1Error("BD1 v2 geometry/class contract differs")
    codec_name = {v: k for k, v in CODEC_IDS.items()}.get(int(codec))
    if codec_name is None:
        raise BF1Error(f"unknown BD1 v2 codec id {codec}")
    raw = bd1.decode_body(codec_name, section[BD1_HEADER_V2.size:], int(raw_len))
    if len(raw) != raw_len or hashlib.sha256(raw).digest() != raw_sha:
        raise BF1Error("BD1 v2 raw length or sha differs")
    if repr_kind == REPR_DENSE_BITMAP:
        fields = decode_dense_raw(raw, int(aux0))
    elif repr_kind == REPR_ROW_RUNS:
        fields = decode_row_runs_raw(raw)
    elif repr_kind == REPR_LANE_CROP:
        fields = decode_lane_crop_raw(raw)
    else:
        raise BF1Error(f"unknown representation kind {repr_kind}")
    return codec_name, fields


def race_coders(
    *,
    surface_id: str,
    raw: bytes,
    records: tuple[bytes, ...],
    artifact_dir: Path,
    store_best: bool,
) -> tuple[tuple[CoderResult, ...], str, bytes]:
    encoded = {
        "brotli-q11": bytes(brotli.compress(raw, quality=11)),
        "lzma1-raw": bd1.lzma1_raw(raw),
        "smevr-r7-nibble": bd1.smevr_records(list(records)),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise BF1Error(f"{surface_id}: Brotli roundtrip failed")
    if bd1.unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise BF1Error(f"{surface_id}: LZMA roundtrip failed")
    if tuple(bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != records:
        raise BF1Error(f"{surface_id}: SMEVR record roundtrip failed")
    best_codec = min(encoded, key=lambda key: len(encoded[key]))
    rows: list[CoderResult] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        artifact_path = None
        if store_best and codec == best_codec:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / f"{surface_id}.{codec}.bin"
            path.write_bytes(payload)
            artifact_path = str(path)
        rows.append(CoderResult(codec, len(payload), sha256_bytes(payload), artifact_path))
    return tuple(rows), best_codec, encoded[best_codec]


def build_section(rep: Representation, selected_codec: str, selected_payload: bytes) -> bytes:
    header = BD1_HEADER_V2.pack(
        bd1.BD1_MAGIC,
        BD1_VERSION_V2,
        SEG_H,
        SEG_W,
        N_PAIRS,
        1,
        ROAD,
        LANE,
        1,
        rep.representation_kind,
        CODEC_IDS[selected_codec],
        len(rep.raw),
        rep.aux0,
        hashlib.sha256(rep.raw).digest(),
    )
    section = header + selected_payload
    _codec, decoded = parse_section_v2(section)
    if len(decoded) != N_PAIRS:
        raise BF1Error("section parse-back pair count drifted")
    return section


def source_band_metrics(lstars: np.ndarray, pairs: tuple[FieldPair, ...]) -> dict[str, Any]:
    if len(pairs) != N_PAIRS:
        raise BF1Error("field pair count differs")
    source_support_total = 0
    decoded_support_total = 0
    support_intersection = 0
    class_source = {ROAD: 0, LANE: 0}
    class_decoded = {ROAD: 0, LANE: 0}
    class_intersection = {ROAD: 0, LANE: 0}
    fp_by_gt: dict[int, int] = {}
    fn_by_gt: dict[int, int] = {}
    per_pair_support_recall: list[float] = []
    for pair, field in enumerate(pairs):
        labels = np.asarray(lstars[pair], dtype=np.uint8)
        source_band = band_for(labels)
        decoded_band = np.zeros(SEG_H * SEG_W, dtype=bool)
        decoded_lane = np.zeros(SEG_H * SEG_W, dtype=bool)
        decoded_band[field.indices.astype(np.int64)] = True
        decoded_lane[field.indices.astype(np.int64)] = field.lane_bits
        decoded_band = decoded_band.reshape(SEG_H, SEG_W)
        decoded_lane = decoded_lane.reshape(SEG_H, SEG_W)
        inter = source_band & decoded_band
        source_count = int(source_band.sum())
        source_support_total += source_count
        decoded_support_total += int(decoded_band.sum())
        support_intersection += int(inter.sum())
        per_pair_support_recall.append(int(inter.sum()) / source_count if source_count else 1.0)
        source_lane = source_band & (labels == LANE)
        source_road_side = source_band & ~source_lane
        for cls in (ROAD, LANE):
            src = source_lane if cls == LANE else source_road_side
            dec = decoded_band & (decoded_lane if cls == LANE else ~decoded_lane)
            class_source[cls] += int(src.sum())
            class_decoded[cls] += int(dec.sum())
            class_intersection[cls] += int((src & dec).sum())
        fp = decoded_band & ~source_band
        fn = source_band & ~decoded_band
        for cls, count in zip(*np.unique(labels[fp], return_counts=True), strict=False):
            fp_by_gt[int(cls)] = fp_by_gt.get(int(cls), 0) + int(count)
        for cls, count in zip(*np.unique(labels[fn], return_counts=True), strict=False):
            fn_by_gt[int(cls)] = fn_by_gt.get(int(cls), 0) + int(count)
    union = source_support_total + decoded_support_total - support_intersection
    class_iou = {}
    for cls in (ROAD, LANE):
        denom = class_source[cls] + class_decoded[cls] - class_intersection[cls]
        class_iou["Road" if cls == ROAD else "Lane"] = class_intersection[cls] / denom if denom else 1.0
    return {
        "source_band_pixels": source_support_total,
        "decoded_band_pixels": decoded_support_total,
        "support_intersection_pixels": support_intersection,
        "support_false_positive_pixels": decoded_support_total - support_intersection,
        "support_false_negative_pixels": source_support_total - support_intersection,
        "band_pixel_recall": support_intersection / source_support_total,
        "band_pixel_precision": support_intersection / decoded_support_total if decoded_support_total else 1.0,
        "support_iou": support_intersection / union if union else 1.0,
        "per_class_iou": class_iou,
        "per_class_source_pixels": {"Road_side": class_source[ROAD], "Lane": class_source[LANE]},
        "per_class_decoded_pixels": {"Road_side": class_decoded[ROAD], "Lane": class_decoded[LANE]},
        "false_positive_pixels_by_gt_class": fp_by_gt,
        "false_negative_pixels_by_gt_class": fn_by_gt,
        "per_pair_support_recall_min": float(np.min(per_pair_support_recall)),
        "per_pair_support_recall_mean": float(np.mean(per_pair_support_recall)),
        "lossless_mask_domain": (
            source_support_total == decoded_support_total
            and source_support_total == support_intersection
            and all(class_intersection[c] == class_source[c] == class_decoded[c] for c in (ROAD, LANE))
        ),
    }


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
    rep: Representation,
    *,
    lstars: np.ndarray,
    base_payload: bytes,
    artifact_dir: Path,
    store_best: bool,
) -> tuple[dict[str, Any], bytes]:
    coders, selected_codec, selected_payload = race_coders(
        surface_id=rep.surface_id,
        raw=rep.raw,
        records=rep.records,
        artifact_dir=artifact_dir,
        store_best=store_best,
    )
    section = build_section(rep, selected_codec, selected_payload)
    _codec, decoded = parse_section_v2(section)
    metrics = source_band_metrics(lstars, decoded)
    projection = archive_projection(base_payload, section)
    best = min(coders, key=lambda row: row.bytes)
    section_bits_per_band_pixel = 8.0 * len(section) / metrics["source_band_pixels"]
    body_bits_per_band_pixel = 8.0 * best.bytes / metrics["source_band_pixels"]
    row = {
        "surface_id": rep.surface_id,
        "candidate_kind": rep.candidate_kind,
        "representation_kind": REPR_NAMES[rep.representation_kind],
        "scope_note": rep.scope_note,
        "raw_bytes": len(rep.raw),
        "raw_sha256": sha256_bytes(rep.raw),
        "records": len(rep.records),
        "best_codec": best.codec,
        "best_body_bytes": best.bytes,
        "best_body_sha256": best.sha256,
        "section_bytes": len(section),
        "section_sha256": sha256_bytes(section),
        "body_bits_per_source_band_pixel": body_bits_per_band_pixel,
        "section_bits_per_source_band_pixel": section_bits_per_band_pixel,
        "vs_0p60_bits_per_band_pixel": section_bits_per_band_pixel / PER_EDGE_REFERENCE_BITS_PER_BAND_PIXEL,
        "coder_race": list(coders),
        "mask_domain_fidelity": metrics,
        "archive_projection": projection,
        "cq2_composed_archive_bytes": {
            "with_25KB_student": projection["projected_archive_bytes"] + 25_000,
            "with_75KB_student": projection["projected_archive_bytes"] + 75_000,
            "leaves_75KB_student_lane_open_vs_bd1_dense_candidate": (
                projection["projected_archive_bytes"] + 75_000 < 726_027
            ),
        },
        "claim_label": "MEASURED n600 coder bytes and MASK-domain reconstruction; scorer-free",
    }
    return row, section


def copy_runtime_tree(base_sub: Path, out_dir: Path) -> None:
    bd1.copy_runtime_tree(base_sub, out_dir)


def build_candidate(
    *,
    base_sub: Path,
    candidate_dir: Path,
    winning_section: bytes,
) -> dict[str, Any]:
    if candidate_dir.exists():
        raise BF1Error(f"candidate dir already exists: {candidate_dir}")
    copy_runtime_tree(base_sub, candidate_dir)
    base_payload = bd1.read_archive_payload(base_sub / "archive.zip")
    bulk, sections = parse_payload(base_payload)
    new_payload = build_payload(bulk, [*sections, winning_section])
    archive_zip = build_single_member_zip(new_payload, name="0.bin")
    (candidate_dir / "archive" / "0.bin").write_bytes(new_payload)
    (candidate_dir / "archive.zip").write_bytes(archive_zip)
    parse_back = bd1.build_local_ledger(
        candidate_dir / "archive.zip",
        ("config", "renderer", "selector", "pose_warp", "frame0_pose_repair", "bf1_band_field"),
    )
    receiver = bd1.receiver_smoke(candidate_dir, pair_index=0)
    if not receiver["receiver_class_field_present"] or not receiver["mutated"]:
        raise BF1Error(f"receiver did not consume/mutate from BF1 field: {receiver}")
    return {
        "submission_dir": str(candidate_dir),
        "archive_bytes": (candidate_dir / "archive.zip").stat().st_size,
        "archive_sha256": sha256_file(candidate_dir / "archive.zip"),
        "payload_sha256": sha256_bytes(new_payload),
        "delta_bytes_vs_qo1": (candidate_dir / "archive.zip").stat().st_size - BASELINE_BYTES,
        "rate_delta_S_vs_qo1": 25.0 * ((candidate_dir / "archive.zip").stat().st_size - BASELINE_BYTES) / RATE_DENOM,
        "label": "RECEIVER-CLOSED / SURVIVAL-UNMEASURED / score_claim=false",
        "parse_back": parse_back,
        "receiver_smoke": receiver,
    }


def write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    lines = [
        "# bf1 band-field representation race - 2026-08-05",
        "",
        "Status: **RECEIVER-CLOSED / SURVIVAL-UNMEASURED / score_claim=false**.",
        "",
        f"Own-vehicle baseline: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`. No scorer job was run.",
    ]
    identity = receipt.get("absent_section_identity_proof")
    if identity is not None:
        lines.extend([
            "",
            "## Receiver-Absent Identity Proof",
            "",
            f"- Updated receiver without a BF1 section reproduced qo1 exactly: `{identity['byte_identical_to_qo1_shipped_decode']}`.",
            f"- Raw output: `{identity['output_raw']}`, bytes `{identity['raw_bytes']}`, sha256 `{identity['raw_sha256']}`.",
            f"- Command: `{identity['command']}`.",
        ])
    lines.extend([
        "",
        "## RECALL EVIDENCE",
        "",
    ])
    for item in receipt["recall_evidence"]:
        lines.append(f"- `{item['source']}`: {item['finding']} Plan impact: {item['plan_impact']}")
    lines.extend([
        "",
        "## Representation Table",
        "",
        "| representation | section B | archive B | lossless | recall | support IoU | Road-side IoU | Lane IoU | bits/band-px | 75KB lane open |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in receipt["representation_rows"]:
        m = row["mask_domain_fidelity"]
        lines.append(
            f"| `{row['surface_id']}` | `{row['section_bytes']}` | "
            f"`{row['archive_projection']['projected_archive_bytes']}` | "
            f"`{m['lossless_mask_domain']}` | `{m['band_pixel_recall']:.6f}` | "
            f"`{m['support_iou']:.6f}` | `{m['per_class_iou']['Road']:.6f}` | "
            f"`{m['per_class_iou']['Lane']:.6f}` | "
            f"`{row['section_bits_per_source_band_pixel']:.6f}` | "
            f"`{row['cq2_composed_archive_bytes']['leaves_75KB_student_lane_open_vs_bd1_dense_candidate']}` |"
        )
    lines.extend([
        "",
        f"Reference currency: `{PER_EDGE_REFERENCE_BITS_PER_BAND_PIXEL}` bits/band-pixel (#916).",
        "",
        "## Winner",
        "",
        f"Full-band byte winner: `{receipt['winner']['surface_id']}`. It is `{receipt['winner']['loss_status']}` in mask domain.",
        f"Candidate archive: `{receipt['candidate']['archive_bytes']}` B, sha256 `{receipt['candidate']['archive_sha256']}`.",
        f"Candidate path: `{receipt['candidate']['submission_dir']}`.",
        f"Receiver smoke pair 0 changed `{receipt['candidate']['receiver_smoke']['camera_pixels_changed']}` camera pixels.",
        "",
        "## Boundaries",
        "",
    ])
    for boundary in receipt["boundaries"]:
        lines.append(f"- {boundary}")
    lines.extend([
        "",
        "## Follow-On Disposition",
        "",
    ])
    for item in receipt["follow_on_disposition"]:
        lines.append(f"- **{item['status']}** `{item['id']}`: {item['action']}")
    lines.extend([
        "",
        "## NEXT-IF-RESUMED",
        "",
        receipt["next_if_resumed"],
        "",
        f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; bf1 did not run a scorer and did not move the contest pointer.",
        "",
    ])
    path.write_text("\n".join(lines))


def build(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sub", type=Path, default=DEFAULT_BASE_SUB)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--absent-identity-json", type=Path, default=None)
    parser.add_argument("--store-best", action="store_true")
    parser.add_argument("--hash-inputs", action="store_true")
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.ssd_dir / "payloads"
    candidate_dir = args.candidate_dir or args.ssd_dir / "sub_auto_pairbit_bf1_rl1_lane_crop_r2"

    base_archive = args.base_sub / "archive.zip"
    base_sha = sha256_file(base_archive)
    if base_sha != BASELINE_ARCHIVE_SHA256:
        raise BF1Error(f"base archive SHA drift: {base_sha}")
    base_payload = bd1.read_archive_payload(base_archive)
    bulk, sections = parse_payload(base_payload)
    if len(sections) != 5:
        raise BF1Error(f"qo1 base expected 5 joint sections, got {len(sections)}")

    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    if tuple(lstars.shape) != (N_PAIRS, SEG_H, SEG_W):
        raise BF1Error(f"unexpected lstar shape {lstars.shape}")
    current_argmax = np.load(args.current_argmax, mmap_mode="r")

    reps = [
        dense_records(lstars),
        row_run_records(lstars),
        lane_crop_records(lstars),
        se3_receiver_closed_sparse_records(lstars, current_argmax),
    ]

    rows: list[dict[str, Any]] = []
    sections_by_surface: dict[str, bytes] = {}
    for rep in reps:
        row, section = build_row(
            rep,
            lstars=lstars,
            base_payload=base_payload,
            artifact_dir=artifact_dir,
            store_best=args.store_best,
        )
        rows.append(row)
        sections_by_surface[rep.surface_id] = section

    rows = sorted(rows, key=lambda row: row["section_bytes"])
    full_band_rows = [
        row for row in rows
        if row["mask_domain_fidelity"]["band_pixel_recall"] >= 0.99
    ]
    if not full_band_rows:
        raise BF1Error("no full-band representation row reached recall >= 0.99")
    winner_row = min(full_band_rows, key=lambda row: row["section_bytes"])
    candidate = build_candidate(
        base_sub=args.base_sub,
        candidate_dir=candidate_dir,
        winning_section=sections_by_surface[winner_row["surface_id"]],
    )

    receipt = {
        "schema": "ddm_bf1_band_field_repr_race.v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free receiver-byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
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
            "class_order": {"Road": ROAD, "Lane": LANE},
        },
        "bd1_versioning": {
            "section_magic": bd1.BD1_MAGIC.decode("ascii"),
            "version": BD1_VERSION_V2,
            "header": "<8sBHHHBBBBBBII32s",
            "representations": REPR_NAMES,
            "old_absent_path": "qo1 base keeps 5 sections; receiver only parses BF1 when tagged section is present",
        },
        "absent_section_identity_proof": load_absent_identity_proof(args.absent_identity_json),
        "representation_rows": rows,
        "external_rows_cited_not_rerun": [
            {
                "id": "bd1_dense_v1_baseline",
                "section_bytes": 367_929,
                "archive_bytes": 726_027,
                "source": ".omx/research/ddm_bd1_20260805/BD1_RECEIPT_20260805.md",
                "reason": "baseline cited, not recomputed",
            },
            {
                "id": "se3_prior_assumption_scoped",
                "body_bytes_side_implied": 81_365,
                "body_bytes_explicit_direction": 100_904,
                "source": ".omx/research/ddm_se3_20260804/se3_receipt.md",
                "reason": "requires receiver-derived chart; RF1 failed on qo1, so BF1 also reports receiver-closed sparse-support price",
            },
            {
                "id": "sp1_explicit_contour_support",
                "support_bytes": 444_394,
                "lzma_support_bytes": 421_366,
                "source": ".omx/research/ddm_sp1_contour_support_coder_20260728.md",
                "reason": "explicit support contour formulation already measured worse than the BF1 top rows",
            },
        ],
        "winner": {
            "surface_id": winner_row["surface_id"],
            "section_bytes": winner_row["section_bytes"],
            "projected_archive_bytes": winner_row["archive_projection"]["projected_archive_bytes"],
            "loss_status": "lossless" if winner_row["mask_domain_fidelity"]["lossless_mask_domain"] else "lossy",
            "selection_rule": (
                "smallest measured BD1CLF1-v2 section bytes among rows with source-band recall >= 0.99; "
                "scorer-free, no survival claim"
            ),
        },
        "candidate": candidate,
        "cq2_thresholds": CQ2_THRESHOLDS,
        "recall_evidence": [
            {
                "source": "rg over .omx/research for sy1/g4/ph1/#941/pf3/#725/worldsheet/m91 plus canonical equation list",
                "finding": "beyond the charter seeds, the live corpus has regional phase, g4 spatial context, v13 worldsheet, PF3/recursive coordinate, BN-stratum, and row-band address families.",
                "plan_impact": "only receiver-decodeable band fields were raced now; non-field carriers are queued with fire orders instead of priced by implication.",
            },
            {
                "source": ".omx/research/ddm_rf1_20260804/RF1_RECEIPT_20260804.md",
                "finding": "qo1 has no legal receiver-derived Road/Lane chart, and local public comma10k model custody was absent.",
                "plan_impact": "prior SE3 81KB/101KB rows remain cited targets; BF1 built a receiver-closed sparse-support control that pays the chart explicitly.",
            },
            {
                "source": ".omx/research/ddm_ph1_phase_mass_reach_ceiling_20260803.md",
                "finding": "regional phase block16/block8 is a large argmax-field lever but not yet a receiver-realized BD1 field.",
                "plan_impact": "queued for scorer/receiver realization, not admitted into this section-byte race.",
            },
            {
                "source": ".omx/research/codex_findings_ddm_v13_worldsheet_event_predictor_20260722_codex.md and ddm_g4_spatial_stationarity receipt",
                "finding": "worldsheet/g4 are shared-context generators and stationarity priors, not immediate class-field record formats.",
                "plan_impact": "queued as successor address/generator sources after BD1 field survival is measured.",
            },
        ],
        "follow_on_disposition": [
            {
                "id": "#939-description-half",
                "status": "SETTLED",
                "action": "use `rl1_lane_crop_bf1` n600 section/body/candidate bytes as the real re-price; realization/scorer survival remains owed.",
            },
            {
                "id": "sp1-contour",
                "status": "FOLDED",
                "action": "explicit contour support stays formulation-negative for this BF1 field race; do not rerun unless a new contour decoder beats row-run/Lane-crop bytes.",
            },
            {
                "id": "se3-rf-chart",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "when a qo1 successor carries a legal receiver-side class chart, rerun the 81KB/101KB SE3 rows against that chart and then append a scorer-slot job behind sq2.",
            },
            {
                "id": "phase-worldsheet-g4-pf3-successors",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "after BF1 survival is known, test one receiver-realized generator/address source at a time; require BD1 section parse-back plus mask-domain table before scorer dispatch.",
            },
            {
                "id": "n600-scorer",
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "do not fire while sq2 owns the scorer slot; candidate is ready only as receiver-closed/survival-unmeasured input.",
            },
        ],
        "boundaries": [
            "No SegNet/PoseNet forward was run.",
            "No upstream/ files were edited.",
            "No /tmp evidence is cited.",
            "All BF1 sections are video-derived counted payload.",
            "The byte winner is lossy in mask domain and is not a score improvement claim.",
            "Candidate survival through scorer cells is unmeasured.",
        ],
        "next_if_resumed": (
            "Start from bf1_repr_race_receipt.json and the candidate archive under the SSD directory. "
            "If scorer slot is free, run exactly one n600 scorer job on the receiver-closed lane-crop "
            "candidate or first replace it with a legal SE3 receiver-chart row if RF closure has landed."
        ),
        "own_vehicle_frontier_line": f"S = {BASELINE_S} @ {BASELINE_BYTES} B {BASELINE_AXIS}",
    }

    json_path = args.research_dir / "bf1_repr_race_receipt.json"
    md_path = args.research_dir / "BF1_RECEIPT_20260805.md"
    json_path.write_text(json.dumps(jsonable(receipt), indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, jsonable(receipt))
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "candidate_archive": str(candidate_dir / "archive.zip")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
