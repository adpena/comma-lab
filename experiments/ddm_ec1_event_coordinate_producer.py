#!/usr/bin/env python3
"""Build and price the EC1 typed event-coordinate proposal alphabet.

The exact coded objects are coordinate-domain edit streams.  ``base_to_hy1``
reconstructs the retained HY1 token plane from the SHA-bound CP135 plane.
``*_temporal`` reconstructs an entire n600 plane from an absolute first frame
and exact consecutive-frame edits.  Every raw and coded payload is retained.

Receiver admission is deliberately narrower than a score claim: one proposal
is decoded onto one CP135 semantic-token frame, rendered by the shipped CP135
semantic receiver, bilinearly lifted to camera resolution, rounded to uint8,
and bilinearly mapped back to the scorer lattice.  No scorer is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np
from scipy import ndimage

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "experiments", REPO / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1
import ddm_js1_stage0_per_edge as js1
import measure_contour_string_flip_coding as sp1

from tac.optimization.direct_description_g1_worldsheet import (
    decode_g1_movable_worldsheet,
    encode_g1_movable_worldsheet,
)

RUN_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_ec1_20260812")
PROPOSAL_ROOT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/follow_on/realized_acceptance_200"
)
BASE_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/retained/coders/hp3_step2/decoded_spatial_tokens.fresh_rc64.bin"
)
HY1_TOKENS: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_hy1_capstone_hybrid_20260811/retained/c1_solved_tokens_n600.u8"
)
BASE_RAW: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/candidates/cp135_base/retained/0.raw"
)
COMPOSED_RAW: Final = Path(
    "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/"
    "candidates/t1r1_c1_composed/retained/0.raw"
)
JS5_CUSTODY: Final = Path("/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/inputs/CUSTODY.json")
CP135_ARCHIVE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime/archive.zip")

N: Final = 600
H: Final = 384
W: Final = 512
CAM_H: Final = 874
CAM_W: Final = 1164
CLASSES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
PIXELS: Final = N * H * W
RAW_VIDEO_BYTES: Final = N * 2 * CAM_H * CAM_W * 3
BASE_TOKEN_SHA: Final = "c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece"
HY1_TOKEN_SHA: Final = "2b0bdfc38a131ab1ebc3a2c2153a79b1ba23be0037adda66d01ab56f29f4fed5"
CP135_ARCHIVE_SHA: Final = "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6"
BASE_RAW_SHA: Final = "a641d1ef149f8da8f06af3da9234d6d2f6be9702c3f606b7acf838b4b298ed47"
INTRA_BAR: Final = 356_636
XOR_BAR: Final = 453_449
AXIS: Final = "[macOS-CPU scorer-free event/coder and receiver-uint8 measurement; n600+n32]"
MAGIC: Final = b"EC1CLS1\0"
HEADER: Final = struct.Struct("<8sBBBBI")
CONTAINER_MAGIC: Final = b"EC1FULL1"
CONTAINER_HEADER: Final = struct.Struct("<8sBBI")
ENTRY_HEADER: Final = struct.Struct("<BBII")
SP1_MAGIC: Final = b"EC1SP1\0\0"
SP1_HEADER: Final = struct.Struct("<8sBBI")
SP1_ENTRY: Final = struct.Struct("<BI")
SP1_STREAM_NAMES: Final = ("counts", "anchor", "chain", "cls")
CODERS: Final = ("brotli-q11", "lzma1-raw", "smevr-r7-nibble")
CODER_ID: Final = {name: index for index, name in enumerate(CODERS)}
MODE_ID: Final = {"base_to_hy1": 0, "cp135_temporal": 1, "hy1_temporal": 2}
EVENT_TYPE: Final = {
    "absolute_seed": 0,
    "boundary_offset": 1,
    "island_birth": 2,
    "island_death": 3,
    "lane_program_delta": 4,
}
EVENT_NAME: Final = {value: key for key, value in EVENT_TYPE.items()}
CONNECT8: Final = np.ones((3, 3), dtype=np.uint8)
MIN_FREE_BYTES: Final = 3 * 1024**3


class EC1Error(RuntimeError):
    """Fail-closed EC1 contract violation."""


@dataclass(frozen=True, slots=True)
class EventSummary:
    mode: str
    frame: int
    event_type: str
    source_class: int
    target_class: int
    seed_y: int
    seed_x: int
    sites: int
    bbox_y0: int
    bbox_x0: int
    bbox_y1: int
    bbox_x1: int
    delta_q4_min: int
    delta_q4_median: float
    delta_q4_max: int
    lane_poly_q8: tuple[int, int, int, int] | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def atomic_json(path: Path, payload: Any) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, payload: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with partial.open("wb") as stream:
        np.save(stream, np.asarray(payload), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return file_record(path)


def save_stage_state(root: Path, stage: str, completed: list[str]) -> None:
    atomic_json(
        root / "state.json",
        {
            "schema": "ddm_ec1_state.v1",
            "status": "RUNNING",
            "resumable": True,
            "active_stage": stage,
            "completed_stages": completed,
            "score_claim": False,
        },
    )


def require(path: Path, *, size: int | None = None, digest: str | None = None) -> None:
    if not path.is_file():
        raise EC1Error(f"required artifact missing: {path}")
    if size is not None and path.stat().st_size != size:
        raise EC1Error(f"byte count differs for {path}")
    if digest is not None and sha256_file(path) != digest:
        raise EC1Error(f"SHA-256 differs for {path}")


def put_uvarint(output: bytearray, value: int) -> None:
    if value < 0:
        raise EC1Error("uvarint cannot encode a negative value")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def get_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise EC1Error("truncated or oversized uvarint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def classify_sites(source: np.ndarray, target: np.ndarray, target_class: int) -> tuple[np.ndarray, np.ndarray]:
    changed = (source != target) & (target == target_class)
    types = np.full((H, W), EVENT_TYPE["boundary_offset"], dtype=np.uint8)
    if not changed.any():
        return types, np.zeros((H, W), dtype=np.uint8)
    near_old_target = ndimage.binary_dilation(source == target_class, structure=CONNECT8)
    births = changed & ~near_old_target
    types[births] = EVENT_TYPE["island_birth"]
    lane = changed & ((source == 1) | (target == 1))
    types[lane] = EVENT_TYPE["lane_program_delta"]
    source_classes = np.unique(source[changed])
    for old_class in source_classes.tolist():
        sites = changed & (source == old_class)
        survives = ndimage.binary_dilation(target == old_class, structure=CONNECT8)
        deaths = sites & ~survives
        types[deaths] = EVENT_TYPE["island_death"]
    distance = ndimage.distance_transform_edt(source != target_class)
    delta_q4 = np.clip(np.rint(4.0 * distance), 0, 255).astype(np.uint8)
    delta_q4[~changed] = 0
    return types, delta_q4


def encode_class_stream(
    mode: str,
    class_id: int,
    sources: np.ndarray,
    targets: np.ndarray,
) -> tuple[bytes, list[bytes], list[EventSummary]]:
    records: list[bytes] = []
    summaries: list[EventSummary] = []
    for frame in range(N):
        source = np.asarray(sources[frame])
        target = np.asarray(targets[frame])
        if mode.endswith("temporal") and frame == 0:
            changed = target == class_id
            types = np.full((H, W), EVENT_TYPE["absolute_seed"], dtype=np.uint8)
            delta_q4 = np.zeros((H, W), dtype=np.uint8)
        else:
            changed = (source != target) & (target == class_id)
            types, delta_q4 = classify_sites(source, target, class_id)
        indices = np.flatnonzero(changed).astype(np.int64)
        record = bytearray()
        put_uvarint(record, frame)
        put_uvarint(record, len(indices))
        previous = 0
        for position, index in enumerate(indices.tolist()):
            put_uvarint(record, index if position == 0 else index - previous)
            y, x = divmod(index, W)
            record.append(int(types[y, x]))
            record.append(int(delta_q4[y, x]))
            previous = index
        records.append(bytes(record))

        if indices.size:
            labels, count = ndimage.label(changed, structure=CONNECT8)
            for component in range(1, count + 1):
                ys, xs = np.nonzero(labels == component)
                if not ys.size:
                    continue
                component_types = types[ys, xs]
                type_id = int(np.bincount(component_types, minlength=len(EVENT_TYPE)).argmax())
                source_values = source[ys, xs]
                source_class = int(np.bincount(source_values, minlength=len(CLASSES)).argmax())
                offsets = delta_q4[ys, xs]
                lane_poly: tuple[int, int, int, int] | None = None
                if type_id == EVENT_TYPE["lane_program_delta"] and np.unique(ys).size >= 4:
                    coeff = np.polyfit(ys.astype(np.float64) / (H - 1), xs.astype(np.float64) / (W - 1), 3)
                    lane_poly = tuple(int(value) for value in np.clip(np.rint(coeff * 256), -32768, 32767))
                order = np.lexsort((xs, ys))
                summaries.append(
                    EventSummary(
                        mode=mode,
                        frame=frame,
                        event_type=EVENT_NAME[type_id],
                        source_class=source_class,
                        target_class=class_id,
                        seed_y=int(ys[order[0]]),
                        seed_x=int(xs[order[0]]),
                        sites=int(ys.size),
                        bbox_y0=int(ys.min()),
                        bbox_x0=int(xs.min()),
                        bbox_y1=int(ys.max()) + 1,
                        bbox_x1=int(xs.max()) + 1,
                        delta_q4_min=int(offsets.min()),
                        delta_q4_median=float(np.median(offsets)),
                        delta_q4_max=int(offsets.max()),
                        lane_poly_q8=lane_poly,
                    )
                )
    body = b"".join(struct.pack("<I", len(record)) + record for record in records)
    payload = HEADER.pack(MAGIC, 1, MODE_ID[mode], class_id, 0, N) + body
    return payload, records, summaries


def decode_class_stream(payload: bytes) -> tuple[str, int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    if len(payload) < HEADER.size:
        raise EC1Error("class stream is truncated")
    magic, version, mode_id, class_id, reserved, frames = HEADER.unpack_from(payload)
    modes = {value: key for key, value in MODE_ID.items()}
    if magic != MAGIC or version != 1 or reserved or mode_id not in modes or class_id >= len(CLASSES) or frames != N:
        raise EC1Error("class stream header differs")
    offset = HEADER.size
    decoded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for expected_frame in range(N):
        if offset + 4 > len(payload):
            raise EC1Error("class record header is truncated")
        (size,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        stop = offset + size
        if stop > len(payload):
            raise EC1Error("class record body is truncated")
        record = payload[offset:stop]
        offset = stop
        cursor = 0
        frame, cursor = get_uvarint(record, cursor)
        count, cursor = get_uvarint(record, cursor)
        if frame != expected_frame or count > H * W:
            raise EC1Error("class record frame/count differs")
        indices = np.empty(count, dtype=np.int64)
        types = np.empty(count, dtype=np.uint8)
        offsets = np.empty(count, dtype=np.uint8)
        previous = 0
        for index in range(count):
            gap, cursor = get_uvarint(record, cursor)
            value = gap if index == 0 else previous + gap
            if value >= H * W or (index and value <= previous) or cursor + 2 > len(record):
                raise EC1Error("class coordinate ordering differs")
            indices[index] = value
            types[index] = record[cursor]
            offsets[index] = record[cursor + 1]
            if int(types[index]) not in EVENT_NAME:
                raise EC1Error("class event type differs")
            cursor += 2
            previous = value
        if cursor != len(record):
            raise EC1Error("class record has trailing bytes")
        decoded.append((indices, types, offsets))
    if offset != len(payload):
        raise EC1Error("class stream has trailing bytes")
    return modes[mode_id], class_id, decoded


def reconstruct(mode: str, streams: dict[int, bytes], base: np.ndarray | None) -> np.ndarray:
    parsed = {class_id: decode_class_stream(payload) for class_id, payload in streams.items()}
    if set(parsed) != set(range(len(CLASSES))) or any(row[0] != mode for row in parsed.values()):
        raise EC1Error("full stream class/mode set differs")
    output = np.empty((N, H, W), dtype=np.uint8)
    for frame in range(N):
        if mode == "base_to_hy1":
            if base is None:
                raise EC1Error("base_to_hy1 requires its SHA-bound base")
            current = np.asarray(base[frame]).copy()
        elif frame == 0:
            current = np.full((H, W), 255, dtype=np.uint8)
        else:
            current = output[frame - 1].copy()
        seen = np.zeros(H * W, dtype=bool)
        for class_id in range(len(CLASSES)):
            indices = parsed[class_id][2][frame][0]
            if np.any(seen[indices]):
                raise EC1Error("two target classes address the same coordinate")
            seen[indices] = True
            current.reshape(-1)[indices] = class_id
        if frame == 0 and mode.endswith("temporal") and not seen.all():
            raise EC1Error("temporal frame zero is not an absolute description")
        output[frame] = current
    return output


def encode_coder(coder: str, raw: bytes, records: list[bytes]) -> bytes:
    if coder == "brotli-q11":
        return brotli.compress(raw, quality=11)
    if coder == "lzma1-raw":
        return bd1.lzma1_raw(raw)
    if coder == "smevr-r7-nibble":
        return bd1.smevr_records(records)
    raise EC1Error(f"unknown coder: {coder}")


def decode_coder(coder: str, payload: bytes, raw_bytes: int) -> bytes:
    if coder == "brotli-q11":
        raw = brotli.decompress(payload)
    elif coder == "lzma1-raw":
        raw = bd1.unlzma1_raw(payload, raw_bytes)
    elif coder == "smevr-r7-nibble":
        raw = b"".join(bd1.unsmevr_records(payload))
    else:
        raise EC1Error(f"unknown coder: {coder}")
    if len(raw) != raw_bytes:
        raise EC1Error(f"{coder} decoded byte count differs")
    return raw


def preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    if usage.free < MIN_FREE_BYTES:
        raise EC1Error(f"storage preflight failed: {usage.free} < {MIN_FREE_BYTES}")
    require(BASE_TOKENS, size=PIXELS, digest=BASE_TOKEN_SHA)
    require(HY1_TOKENS, size=PIXELS, digest=HY1_TOKEN_SHA)
    require(BASE_RAW, size=RAW_VIDEO_BYTES, digest=BASE_RAW_SHA)
    require(COMPOSED_RAW, size=RAW_VIDEO_BYTES)
    require(CP135_ARCHIVE, size=186_252, digest=CP135_ARCHIVE_SHA)
    require(JS5_CUSTODY)
    result = {
        "schema": "ddm_ec1_preflight.v1",
        "axis": AXIS,
        "score_claim": False,
        "storage": {"free_bytes": usage.free, "required_bytes": MIN_FREE_BYTES},
        "inputs": {
            "cp135_tokens": file_record(BASE_TOKENS),
            "hy1_tokens": file_record(HY1_TOKENS),
            "cp135_raw": file_record(BASE_RAW),
            "t1r1_c1_raw": file_record(COMPOSED_RAW),
            "cp135_archive": file_record(CP135_ARCHIVE),
            "js5_custody": file_record(JS5_CUSTODY),
        },
    }
    atomic_json(root / "00_PREFLIGHT.json", result)
    return result


def source_target(mode: str, base: np.ndarray, hy1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == "base_to_hy1":
        return base, hy1
    target = base if mode == "cp135_temporal" else hy1
    source = np.empty_like(target)
    source[0].fill(255)
    source[1:] = target[:-1]
    return source, target


def extract(root: Path) -> dict[str, Any]:
    receipt_path = root / "20_EXTRACT_RESULT.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        for mode_row in prior["modes"].values():
            for row in mode_row["class_streams"]:
                require(Path(row["raw"]["path"]), size=row["raw"]["bytes"], digest=row["raw"]["sha256"])
        return prior
    base = np.memmap(BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    hy1 = np.memmap(HY1_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    modes: dict[str, Any] = {}
    for mode in MODE_ID:
        sources, targets = source_target(mode, base, hy1)
        class_rows = []
        summaries: list[EventSummary] = []
        changed_total = 0
        for class_id in range(len(CLASSES)):
            payload, records, class_summaries = encode_class_stream(mode, class_id, sources, targets)
            raw_path = root / "retained/events" / mode / f"{class_id}_{CLASSES[class_id]}.ec1raw"
            raw_row = atomic_bytes(raw_path, payload)
            records_path = root / "retained/events" / mode / f"{class_id}_{CLASSES[class_id]}.records.npz"
            offsets = np.cumsum([0, *[len(value) for value in records]], dtype=np.int64)
            joined = np.frombuffer(b"".join(records), dtype=np.uint8).copy()
            records_row = atomic_npy(records_path.with_suffix(".offsets.npy"), offsets)
            joined_row = atomic_npy(records_path.with_suffix(".bytes.npy"), joined)
            changed = sum(len(decode_class_stream(payload)[2][frame][0]) for frame in range(N))
            changed_total += changed
            class_rows.append(
                {
                    "class_id": class_id,
                    "class": CLASSES[class_id],
                    "changed_sites": changed,
                    "raw": raw_row,
                    "record_offsets": records_row,
                    "record_bytes": joined_row,
                }
            )
            summaries.extend(class_summaries)
        streams = {row["class_id"]: Path(row["raw"]["path"]).read_bytes() for row in class_rows}
        restored = reconstruct(mode, streams, base if mode == "base_to_hy1" else None)
        if not np.array_equal(restored, targets):
            raise EC1Error(f"{mode} semantic reconstruction differs")
        ledger_path = root / "retained/events" / mode / "typed_events.jsonl"
        atomic_bytes(
            ledger_path, b"".join((json.dumps(asdict(row), sort_keys=True) + "\n").encode() for row in summaries)
        )
        type_counts: dict[str, int] = dict.fromkeys(EVENT_TYPE, 0)
        type_sites: dict[str, int] = dict.fromkeys(EVENT_TYPE, 0)
        for row in summaries:
            type_counts[row.event_type] += 1
            type_sites[row.event_type] += row.sites
        modes[mode] = {
            "changed_sites": changed_total,
            "event_count": len(summaries),
            "event_counts_by_type": type_counts,
            "event_sites_by_type": type_sites,
            "class_streams": class_rows,
            "typed_event_ledger": file_record(ledger_path),
            "exact_semantic_reconstruction": True,
        }
    # Reuse the productionized V13 EVENT/CENTROID/SHAPE machinery as a typed
    # lossy island-worldsheet adjunct; it is not substituted for the exact stream.
    v13_payload, v13_meta = encode_g1_movable_worldsheet(np.asarray(hy1))
    decoded_v13, decoded_meta = decode_g1_movable_worldsheet(v13_payload, expected_pairs=N)
    if decoded_meta.payload_sha256 != v13_meta.payload_sha256 or decoded_v13.shape != (N, H, W):
        raise EC1Error("V13 worldsheet parse-back differs")
    v13_row = atomic_bytes(root / "retained/v13/hy1_movable.g1s", v13_payload)

    # SP1 contour support coder on the sparse C1-vs-CP135 layer, including
    # target-class labels.  Decode is checked before the byte count is admitted.
    flip_maps = [np.asarray(base[index] != hy1[index]) for index in range(N)]
    class_maps = [np.asarray(hy1[index], dtype=np.int64) for index in range(N)]
    contour = sp1.contour_encode_frames(flip_maps, class_maps)
    decoded_flips, decoded_classes = sp1.contour_decode_frames(contour["streams"], N, H, W)
    for index in range(N):
        if not np.array_equal(decoded_flips[index], flip_maps[index]):
            raise EC1Error("SP1 contour support decode differs")
        if not np.array_equal(decoded_classes[index][flip_maps[index]], class_maps[index][flip_maps[index]]):
            raise EC1Error("SP1 contour class decode differs")
    contour_rows = {
        name: atomic_bytes(root / "retained/sp1" / f"base_to_hy1.{name}.range", payload)
        for name, payload in contour["streams"].items()
    }
    result = {
        "schema": "ddm_ec1_extract_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "modes": modes,
        "v13_reuse": {"payload": v13_row, "metadata": asdict(v13_meta), "exact_partition_stream": False},
        "sp1_reuse": {
            "payloads": contour_rows,
            "support_bytes": sum(contour["stream_bytes"][name] for name in ("counts", "anchor", "chain")),
            "label_bytes": contour["stream_bytes"]["cls"],
            "total_bytes": contour["total_bytes"],
            "n_flips": contour["n_flips"],
            "n_components": contour["n_components"],
            "lossless_roundtrip": True,
        },
    }
    atomic_json(receipt_path, result)
    return result


def _records_for_row(row: dict[str, Any]) -> list[bytes]:
    offsets = np.load(row["record_offsets"]["path"], allow_pickle=False)
    values = np.load(row["record_bytes"]["path"], allow_pickle=False).tobytes()
    return [values[int(offsets[index]) : int(offsets[index + 1])] for index in range(N)]


def build_container(mode: str, entries: dict[int, tuple[str, bytes, int]]) -> bytes:
    output = bytearray(CONTAINER_HEADER.pack(CONTAINER_MAGIC, 1, MODE_ID[mode], N))
    for class_id in range(len(CLASSES)):
        coder, payload, raw_bytes = entries[class_id]
        output.extend(ENTRY_HEADER.pack(class_id, CODER_ID[coder], len(payload), raw_bytes))
        output.extend(payload)
    return bytes(output)


def parse_container(payload: bytes) -> tuple[str, dict[int, tuple[str, bytes, int]]]:
    if len(payload) < CONTAINER_HEADER.size:
        raise EC1Error("complete container is truncated")
    magic, version, mode_id, frames = CONTAINER_HEADER.unpack_from(payload)
    modes = {value: key for key, value in MODE_ID.items()}
    coders = {value: key for key, value in CODER_ID.items()}
    if magic != CONTAINER_MAGIC or version != 1 or mode_id not in modes or frames != N:
        raise EC1Error("complete container header differs")
    cursor = CONTAINER_HEADER.size
    entries: dict[int, tuple[str, bytes, int]] = {}
    for expected_class in range(len(CLASSES)):
        if cursor + ENTRY_HEADER.size > len(payload):
            raise EC1Error("complete container entry header is truncated")
        class_id, coder_id, coded_bytes, raw_bytes = ENTRY_HEADER.unpack_from(payload, cursor)
        cursor += ENTRY_HEADER.size
        stop = cursor + coded_bytes
        if class_id != expected_class or coder_id not in coders or stop > len(payload):
            raise EC1Error("complete container entry differs")
        entries[class_id] = (coders[coder_id], payload[cursor:stop], raw_bytes)
        cursor = stop
    if cursor != len(payload):
        raise EC1Error("complete container has trailing bytes")
    return modes[mode_id], entries


def build_sp1_container(mode: str, streams: dict[str, bytes]) -> bytes:
    output = bytearray(SP1_HEADER.pack(SP1_MAGIC, 1, MODE_ID[mode], N))
    for stream_id, name in enumerate(SP1_STREAM_NAMES):
        payload = streams[name]
        output.extend(SP1_ENTRY.pack(stream_id, len(payload)))
        output.extend(payload)
    return bytes(output)


def parse_sp1_container(payload: bytes) -> tuple[str, dict[str, bytes]]:
    if len(payload) < SP1_HEADER.size:
        raise EC1Error("SP1 container is truncated")
    magic, version, mode_id, frames = SP1_HEADER.unpack_from(payload)
    modes = {value: key for key, value in MODE_ID.items()}
    if magic != SP1_MAGIC or version != 1 or mode_id not in modes or frames != N:
        raise EC1Error("SP1 container header differs")
    cursor = SP1_HEADER.size
    streams: dict[str, bytes] = {}
    for expected_id, name in enumerate(SP1_STREAM_NAMES):
        if cursor + SP1_ENTRY.size > len(payload):
            raise EC1Error("SP1 entry header is truncated")
        stream_id, size = SP1_ENTRY.unpack_from(payload, cursor)
        cursor += SP1_ENTRY.size
        stop = cursor + size
        if stream_id != expected_id or stop > len(payload):
            raise EC1Error("SP1 entry differs")
        streams[name] = payload[cursor:stop]
        cursor = stop
    if cursor != len(payload):
        raise EC1Error("SP1 container has trailing bytes")
    return modes[mode_id], streams


def price_mode(root: Path, mode: str, mode_source: dict[str, Any]) -> dict[str, Any]:
    receipt_path = root / f"40_PRICE_{mode}.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        require(
            Path(prior["winner"]["payload"]["path"]),
            size=prior["winner"]["payload"]["bytes"],
            digest=prior["winner"]["payload"]["sha256"],
        )
        return prior
    class_rows = []
    encoded_by_class: dict[int, dict[str, bytes]] = {}
    for row in mode_source["class_streams"]:
        class_id = int(row["class_id"])
        raw = Path(row["raw"]["path"]).read_bytes()
        records = _records_for_row(row)
        framed_records = [raw[: HEADER.size]] + [struct.pack("<I", len(record)) + record for record in records]
        if b"".join(framed_records) != raw:
            raise EC1Error(f"{mode}/{class_id} retained record framing differs")
        encoded_by_class[class_id] = {}
        coder_rows = []
        for coder in CODERS:
            suffix = {"brotli-q11": "br", "lzma1-raw": "lzma1", "smevr-r7-nibble": "smevr"}[coder]
            payload_path = root / "retained/coders" / mode / f"{class_id}_{CLASSES[class_id]}.{suffix}"
            payload = payload_path.read_bytes() if payload_path.is_file() else encode_coder(coder, raw, framed_records)
            if decode_coder(coder, payload, len(raw)) != raw:
                raise EC1Error(f"{mode}/{class_id}/{coder} round-trip differs")
            payload_row = file_record(payload_path) if payload_path.is_file() else atomic_bytes(payload_path, payload)
            encoded_by_class[class_id][coder] = payload
            coder_rows.append({"coder": coder, "payload": payload_row, "roundtrip": True})
        selected = min(coder_rows, key=lambda value: (value["payload"]["bytes"], CODER_ID[value["coder"]]))
        class_rows.append({**row, "coders": coder_rows, "winner": selected})
    candidates = []
    for coder_name in (*CODERS, "per-class-best"):
        entries: dict[int, tuple[str, bytes, int]] = {}
        for class_id in range(len(CLASSES)):
            selected_coder = coder_name
            if coder_name == "per-class-best":
                selected_coder = min(CODERS, key=lambda name: (len(encoded_by_class[class_id][name]), CODER_ID[name]))
            raw_bytes = int(mode_source["class_streams"][class_id]["raw"]["bytes"])
            entries[class_id] = (selected_coder, encoded_by_class[class_id][selected_coder], raw_bytes)
        container = build_container(mode, entries)
        parsed_mode, parsed_entries = parse_container(container)
        if parsed_mode != mode or parsed_entries != entries:
            raise EC1Error(f"{mode}/{coder_name} complete container parse-back differs")
        payload_row = atomic_bytes(root / "retained/full_vehicle" / mode / f"{coder_name}.ec1", container)
        candidates.append(
            {
                "coder": coder_name,
                "payload": payload_row,
                "entry_coders": {str(k): v[0] for k, v in entries.items()},
            }
        )
    winner = min(candidates, key=lambda value: (value["payload"]["bytes"], value["coder"]))
    winner_mode, winner_entries = parse_container(Path(winner["payload"]["path"]).read_bytes())
    winner_streams = {
        class_id: decode_coder(coder, payload, raw_bytes)
        for class_id, (coder, payload, raw_bytes) in winner_entries.items()
    }
    base = np.memmap(BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    hy1 = np.memmap(HY1_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    expected = hy1 if mode in ("base_to_hy1", "hy1_temporal") else base
    restored = reconstruct(winner_mode, winner_streams, base if mode == "base_to_hy1" else None)
    if not np.array_equal(restored, expected):
        raise EC1Error(f"{mode} winning complete container semantic decode differs")
    result = {
        "class_rows": class_rows,
        "full_candidates": candidates,
        "winner": winner,
        "ratio_vs_intra_356636": winner["payload"]["bytes"] / INTRA_BAR,
        "ratio_vs_xor_453449": winner["payload"]["bytes"] / XOR_BAR,
        "f1_event_ge_intra": winner["payload"]["bytes"] >= INTRA_BAR,
        "winning_container_parse_back_exact": True,
        "winning_container_semantic_roundtrip_exact": True,
        "selection": "minimum retained complete container bytes across three real coders and per-class mixture",
    }
    atomic_json(receipt_path, result)
    return result


def price(root: Path, extracted: dict[str, Any]) -> dict[str, Any]:
    receipt_path = root / "40_PRICE_RESULT.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        for mode in prior["modes"].values():
            require(
                Path(mode["winner"]["payload"]["path"]),
                size=mode["winner"]["payload"]["bytes"],
                digest=mode["winner"]["payload"]["sha256"],
            )
        return prior
    mode_results = {mode: price_mode(root, mode, mode_source) for mode, mode_source in extracted["modes"].items()}
    result = {"schema": "ddm_ec1_price_result.v1", "axis": AXIS, "score_claim": False, "modes": mode_results}
    atomic_json(receipt_path, result)
    return result


def contour_price_mode(
    root: Path, mode: str, extracted: dict[str, Any], base: np.ndarray, hy1: np.ndarray
) -> dict[str, Any]:
    receipt_path = root / f"50_CURVE_PRICE_{mode}.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        require(
            Path(prior["winner"]["payload"]["path"]),
            size=prior["winner"]["payload"]["bytes"],
            digest=prior["winner"]["payload"]["sha256"],
        )
        return prior
    sources, targets = source_target(mode, base, hy1)
    if mode == "base_to_hy1":
        streams = {
            name: Path(extracted["sp1_reuse"]["payloads"][name]["path"]).read_bytes() for name in SP1_STREAM_NAMES
        }
        metadata = {
            "n_flips": extracted["sp1_reuse"]["n_flips"],
            "n_components": extracted["sp1_reuse"]["n_components"],
            "adaptive_range_stream_bytes": extracted["sp1_reuse"]["total_bytes"],
            "reused_from_extract": True,
        }
    else:
        flip_maps = [np.asarray(sources[index] != targets[index]) for index in range(N)]
        class_maps = [np.asarray(targets[index], dtype=np.uint8) for index in range(N)]
        stream_paths = {name: root / "retained/sp1" / f"{mode}.{name}.range" for name in SP1_STREAM_NAMES}
        adopted_partial = all(path.is_file() for path in stream_paths.values())
        if adopted_partial:
            streams = {name: path.read_bytes() for name, path in stream_paths.items()}
            contour = {
                "n_flips": sum(int(value.sum()) for value in flip_maps),
                "n_components": sum(int(ndimage.label(value, structure=CONNECT8)[1]) for value in flip_maps),
                "total_bytes": sum(len(value) for value in streams.values()),
            }
        else:
            contour = sp1.contour_encode_frames(flip_maps, class_maps)
            streams = contour["streams"]
        decoded_flips, decoded_classes = sp1.contour_decode_frames(streams, N, H, W)
        for index in range(N):
            current = np.asarray(sources[index]).copy()
            current[decoded_flips[index]] = decoded_classes[index][decoded_flips[index]]
            if not np.array_equal(current, targets[index]):
                raise EC1Error(f"{mode} SP1 curve/event semantic reconstruction differs")
        stream_rows = {
            name: file_record(stream_paths[name])
            if adopted_partial
            else atomic_bytes(stream_paths[name], streams[name])
            for name in SP1_STREAM_NAMES
        }
        metadata = dict(contour)
        metadata.pop("streams", None)
        metadata["adaptive_range_stream_bytes"] = contour["total_bytes"]
        metadata["streams"] = stream_rows
        metadata["reused_from_extract"] = False
        metadata["adopted_partial_streams"] = adopted_partial
    native = build_sp1_container(mode, streams)
    parsed_mode, parsed_streams = parse_sp1_container(native)
    if parsed_mode != mode or parsed_streams != streams:
        raise EC1Error(f"{mode} SP1 complete container parse-back differs")
    native_row = atomic_bytes(root / "retained/curve_vehicle" / mode / "sp1-range-native.ec1", native)
    framed_records = [native[start : start + 60_000] for start in range(0, len(native), 60_000)]
    candidates = [{"coder": "sp1-adaptive-range-native", "payload": native_row, "roundtrip": True}]
    for coder in CODERS:
        payload = encode_coder(coder, native, framed_records)
        if decode_coder(coder, payload, len(native)) != native:
            raise EC1Error(f"{mode}/SP1/{coder} round-trip differs")
        suffix = {"brotli-q11": "br", "lzma1-raw": "lzma1", "smevr-r7-nibble": "smevr"}[coder]
        payload_row = atomic_bytes(root / "retained/curve_vehicle" / mode / f"sp1.{suffix}", payload)
        candidates.append({"coder": coder, "payload": payload_row, "roundtrip": True})
    winner = min(candidates, key=lambda row: (row["payload"]["bytes"], row["coder"]))
    result = {
        "schema": "ddm_ec1_curve_event_price_mode.v1",
        "mode": mode,
        "representation": (
            "SP1 exact 8-connected component anchors plus DFS chain symbols plus target-class symbols; "
            "typed birth/death/boundary/lane fields are deterministically re-derived from source and decoded target"
        ),
        "metadata": metadata,
        "candidates": candidates,
        "winner": winner,
        "ratio_vs_intra_356636": winner["payload"]["bytes"] / INTRA_BAR,
        "ratio_vs_xor_453449": winner["payload"]["bytes"] / XOR_BAR,
        "f1_event_ge_intra": winner["payload"]["bytes"] >= INTRA_BAR,
        "exact_semantic_reconstruction": True,
        "score_claim": False,
    }
    atomic_json(receipt_path, result)
    return result


def contour_price(root: Path, extracted: dict[str, Any]) -> dict[str, Any]:
    receipt_path = root / "50_CURVE_PRICE_RESULT.json"
    if receipt_path.is_file():
        return json.loads(receipt_path.read_text())
    base = np.memmap(BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    hy1 = np.memmap(HY1_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    modes = {mode: contour_price_mode(root, mode, extracted, base, hy1) for mode in MODE_ID}
    result = {
        "schema": "ddm_ec1_curve_event_price_result.v1",
        "axis": AXIS,
        "modes": modes,
        "score_claim": False,
    }
    atomic_json(receipt_path, result)
    return result


def proposal_payload(frame: int, source_class: int, target_class: int, indices: np.ndarray, event_type: int) -> bytes:
    output = bytearray(b"EC1PROP1")
    output.extend(struct.pack("<HBBB", frame, source_class, target_class, event_type))
    put_uvarint(output, len(indices))
    previous = 0
    for position, index in enumerate(indices.tolist()):
        put_uvarint(output, int(index) if position == 0 else int(index) - previous)
        previous = int(index)
    return bytes(output)


def decode_proposal(payload: bytes) -> tuple[int, int, int, int, np.ndarray]:
    if len(payload) < 13 or payload[:8] != b"EC1PROP1":
        raise EC1Error("proposal header differs")
    frame, source_class, target_class, event_type = struct.unpack_from("<HBBB", payload, 8)
    if frame >= N or source_class >= len(CLASSES) or target_class >= len(CLASSES) or event_type not in EVENT_NAME:
        raise EC1Error("proposal address differs")
    offset = 13
    count, offset = get_uvarint(payload, offset)
    indices = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        gap, offset = get_uvarint(payload, offset)
        value = gap if position == 0 else previous + gap
        if value >= H * W or (position and value <= previous):
            raise EC1Error("proposal coordinates differ")
        indices[position] = value
        previous = value
    if offset != len(payload):
        raise EC1Error("proposal has trailing bytes")
    return frame, source_class, target_class, event_type, indices


def candidate_groups(
    base: np.ndarray, hy1: np.ndarray, sample: list[int]
) -> list[tuple[int, int, int, int, np.ndarray]]:
    groups: list[tuple[int, int, int, int, np.ndarray]] = []
    for frame in sample:
        source = np.asarray(base[frame])
        target = np.asarray(hy1[frame])
        diff = source != target
        labels, count = ndimage.label(diff, structure=CONNECT8)
        for component in range(1, count + 1):
            component_mask = labels == component
            for source_class in range(len(CLASSES)):
                for target_class in range(len(CLASSES)):
                    if source_class == target_class:
                        continue
                    sites = np.flatnonzero(component_mask & (source == source_class) & (target == target_class))
                    if not sites.size:
                        continue
                    types, _offsets = classify_sites(source, target, target_class)
                    type_id = int(np.bincount(types.reshape(-1)[sites], minlength=len(EVENT_TYPE)).argmax())
                    for start in range(0, len(sites), 32):
                        groups.append((frame, source_class, target_class, type_id, sites[start : start + 32].copy()))
    groups.sort(key=lambda row: (row[0], row[3], row[2], row[1], int(row[4][0])))
    return groups


def render_master(semantic: Any, tokens: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.nn import functional

    with torch.inference_mode():
        value = semantic(
            torch.from_numpy(np.asarray(tokens).copy())[None].long(),
            torch.tensor([frame], dtype=torch.long),
        )
        camera = (
            functional.interpolate(value, size=(CAM_H, CAM_W), mode="bilinear", align_corners=False)
            .clamp(0.0, 255.0)
            .round()
            .to(torch.uint8)
        )
        scorer = functional.interpolate(camera.float(), size=(H, W), mode="bilinear", align_corners=False)
    return camera[0].permute(1, 2, 0).numpy(), scorer[0].half().numpy()


def receiver(root: Path) -> dict[str, Any]:
    receipt_path = root / "60_RECEIVER_RESULT.json"
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text())
        require(
            Path(prior["proposal_index"]["path"]),
            size=prior["proposal_index"]["bytes"],
            digest=prior["proposal_index"]["sha256"],
        )
        return prior
    custody = json.loads(JS5_CUSTODY.read_text())
    sample = [int(value) for value in custody["sample"]]
    if len(sample) != 32 or len(set(sample)) != 32:
        raise EC1Error("JS5 stratified sample differs from n32")
    base = np.memmap(BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    hy1 = np.memmap(HY1_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
    base_raw = np.memmap(BASE_RAW, mode="r", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    composed_raw = np.memmap(COMPOSED_RAW, mode="r", dtype=np.uint8, shape=(N * 2, CAM_H, CAM_W, 3))
    proof_root = root / "retained/receiver_proof"
    try:
        *_, semantic, _basis, _coefficients = js1.parse_receiver_state(js1.CANDIDATES["cp135_base"], proof_root)
        semantic = semantic.eval().cpu()
        inactive_rows = []
        base_scorer_by_frame: dict[int, np.ndarray] = {}
        aggregate_rows = []
        for frame in sample:
            replay_camera, replay_scorer = render_master(semantic, np.asarray(base[frame]), frame)
            replay_path = proof_root / "inactive" / f"pair_{frame:03d}.camera.uint8.npy"
            scorer_path = proof_root / "inactive" / f"pair_{frame:03d}.scorer_input.float16.npy"
            replay_row = atomic_npy(replay_path, replay_camera)
            scorer_row = atomic_npy(scorer_path, replay_scorer)
            base_scorer_by_frame[frame] = replay_scorer
            expected = np.asarray(base_raw[2 * frame + 1])
            mismatch = int(np.count_nonzero(replay_camera != expected))
            inactive_rows.append(
                {"pair": frame, "mismatch_values": mismatch, "camera": replay_row, "scorer_input": scorer_row}
            )
            token_edits = int(np.count_nonzero(base[frame] != hy1[frame]))
            camera_values = int(np.count_nonzero(base_raw[2 * frame + 1] != composed_raw[2 * frame + 1]))
            aggregate_rows.append({"pair": frame, "token_edits": token_edits, "camera_changed_values": camera_values})
        if any(row["mismatch_values"] for row in inactive_rows):
            raise EC1Error("inactive CP135 receiver replay is not byte-identical")

        groups = candidate_groups(base, hy1, sample)
        if len(groups) < 200:
            raise EC1Error(f"only {len(groups)} content-distinct event groups exist in stratified n32")
        proposal_root = PROPOSAL_ROOT
        proposal_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(proposal_root)
        if usage.free < MIN_FREE_BYTES:
            raise EC1Error("proposal-store storage preflight failed")
        progress_path = root / "retained/proposal_attempts/PROGRESS.json"
        progress = (
            json.loads(progress_path.read_text())
            if progress_path.is_file()
            else {"next_group": 0, "attempts": 0, "accepted": 0, "proposal_ids": []}
        )
        attempts = int(progress["attempts"])
        accepted = int(progress["accepted"])
        proposal_ids = [str(value) for value in progress["proposal_ids"]]
        for group_index in range(int(progress["next_group"]), len(groups)):
            if accepted >= 200:
                break
            frame, source_class, target_class, type_id, indices = groups[group_index]
            attempts += 1
            payload = proposal_payload(frame, source_class, target_class, indices, type_id)
            decoded = decode_proposal(payload)
            candidate = np.asarray(base[frame]).copy()
            if np.any(candidate.reshape(-1)[decoded[4]] != source_class):
                raise EC1Error("proposal source-class precondition differs")
            candidate.reshape(-1)[decoded[4]] = target_class
            candidate_root = root / "retained/proposal_attempts" / f"attempt_{attempts:04d}"
            payload_row = atomic_bytes(candidate_root / "event.ec1p", payload)
            tokens_row = atomic_npy(candidate_root / "candidate_tokens.uint8.npy", candidate)
            camera, scorer = render_master(semantic, candidate, frame)
            camera_row = atomic_npy(candidate_root / "camera.uint8.npy", camera)
            scorer_row = atomic_npy(candidate_root / "scorer_input.float16.npy", scorer)
            base_camera = np.asarray(base_raw[2 * frame + 1])
            camera_changed = int(np.count_nonzero(camera != base_camera))
            scorer_changed = int(np.count_nonzero(scorer != base_scorer_by_frame[frame]))
            event_brotli = brotli.compress(payload, quality=11)
            event_lzma = lzma.compress(payload, preset=9 | lzma.PRESET_EXTREME)
            brotli_row = atomic_bytes(candidate_root / "event.ec1p.br", event_brotli)
            lzma_row = atomic_bytes(candidate_root / "event.ec1p.xz", event_lzma)
            effective = camera_changed > 0 and scorer_changed > 0
            attempt_receipt = {
                "schema": "ddm_ec1_receiver_proposal_attempt.v1",
                "axis": AXIS,
                "proposal_id": None,
                "pair": frame,
                "event_type": EVENT_NAME[type_id],
                "source_class": CLASSES[source_class],
                "target_class": CLASSES[target_class],
                "site_count": len(indices),
                "seed_yx": list(divmod(int(indices[0]), W)),
                "source_archive_sha256": CP135_ARCHIVE_SHA,
                "parse_back_exact": True,
                "inactive_receiver_byte_identical_n32": True,
                "camera_changed_values": camera_changed,
                "scorer_lattice_changed_values": scorer_changed,
                "receiver_effective": effective,
                "payload": payload_row,
                "candidate_tokens": tokens_row,
                "camera_uint8": camera_row,
                "scorer_input_float16": scorer_row,
                "real_coder_payloads": {"brotli_q11": brotli_row, "lzma_xz9e": lzma_row},
                "score_claim": False,
                "acceptance_tested": False,
            }
            if effective:
                proposal_id = f"ec1_{accepted:04d}_{hashlib.sha256(payload).hexdigest()[:12]}"
                attempt_receipt["proposal_id"] = proposal_id
                target_root = proposal_root / "proposals" / proposal_id
                target_root.mkdir(parents=True, exist_ok=True)
                # Retain proposal-owned copies: the JS5 store stays consumable even if
                # the producer run is later cold-stored.
                for name, source_path in {
                    "event.ec1p": Path(payload_row["path"]),
                    "candidate_tokens.uint8.npy": Path(tokens_row["path"]),
                    "camera.uint8.npy": Path(camera_row["path"]),
                    "scorer_input.float16.npy": Path(scorer_row["path"]),
                    "event.ec1p.br": Path(brotli_row["path"]),
                    "event.ec1p.xz": Path(lzma_row["path"]),
                }.items():
                    destination = target_root / name
                    atomic_bytes(destination, source_path.read_bytes())
                attempt_receipt["consumer_payloads"] = {
                    name: file_record(target_root / name)
                    for name in (
                        "event.ec1p",
                        "candidate_tokens.uint8.npy",
                        "camera.uint8.npy",
                        "scorer_input.float16.npy",
                        "event.ec1p.br",
                        "event.ec1p.xz",
                    )
                }
                atomic_json(target_root / "proposal.json", attempt_receipt)
                proposal_ids.append(proposal_id)
                accepted += 1
            atomic_json(candidate_root / "ATTEMPT_RESULT.json", attempt_receipt)
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_ec1_receiver_progress.v1",
                    "next_group": group_index + 1,
                    "attempts": attempts,
                    "accepted": accepted,
                    "proposal_ids": proposal_ids,
                    "score_claim": False,
                },
            )
        if accepted < 200:
            raise EC1Error(f"F3 fired: only {accepted} receiver-effective proposals after {attempts} attempts")
        index_path = proposal_root / "proposal_index.jsonl"
        index_lines = []
        for proposal_id in proposal_ids:
            proposal_receipt = proposal_root / "proposals" / proposal_id / "proposal.json"
            receipt = json.loads(proposal_receipt.read_text())
            index_lines.append(
                (
                    json.dumps(
                        {**receipt, "proposal_receipt": file_record(proposal_receipt)},
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
            )
        index_row = atomic_bytes(index_path, b"".join(index_lines))
        store_state = {
            "schema": "ddm_js5_realized_acceptance_200_store.v1",
            "producer": "ddm_ec1",
            "status": "PRODUCED_NOT_ACCEPTANCE_TESTED",
            "proposal_count": accepted,
            "receiver_effective_count": accepted,
            "attempt_count": attempts,
            "sample": sample,
            "source_archive_sha256": CP135_ARCHIVE_SHA,
            "proposal_index": index_row,
            "acceptance_tested": False,
            "score_claim": False,
            "fire_boundary": "MAIN owns any SegNet/PoseNet acceptance burn and the sole n600 scorer slot",
        }
        atomic_json(proposal_root / "state.json", store_state)
        result = {
            "schema": "ddm_ec1_receiver_result.v1",
            "axis": AXIS,
            "score_claim": False,
            "stratified_sample": sample,
            "inactive": {
                "byte_identical_pairs": sum(row["mismatch_values"] == 0 for row in inactive_rows),
                "denominator_pairs": 32,
                "rows": inactive_rows,
            },
            "aggregate_c1_receiver": {
                "active_pairs": sum(row["token_edits"] > 0 for row in aggregate_rows),
                "active_pairs_with_camera_change": sum(
                    row["token_edits"] > 0 and row["camera_changed_values"] > 0 for row in aggregate_rows
                ),
                "rows": aggregate_rows,
                "boundary": "matched semantic receiver/master frame only; T1R1 carrier differences are excluded",
            },
            "proposal_attempts": attempts,
            "receiver_effective_proposals": accepted,
            "f3_lt20_fired": accepted < 20,
            "proposal_store": str(proposal_root.resolve()),
            "proposal_index": index_row,
            "store_state": file_record(proposal_root / "state.json"),
            "acceptance_tested": False,
        }
        atomic_json(receipt_path, result)
        return result
    finally:
        js1.release_runtime()


def finalize(
    root: Path,
    extracted: dict[str, Any],
    priced: dict[str, Any],
    curve_priced: dict[str, Any],
    receiver_result: dict[str, Any],
) -> dict[str, Any]:
    producer_source = file_record(Path(__file__))
    producer_invocations = [
        ".venv/bin/python experiments/ddm_ec1_event_coordinate_producer.py extract",
        ".venv/bin/python experiments/ddm_ec1_event_coordinate_producer.py price --price-mode <mode>",
        ".venv/bin/python experiments/ddm_ec1_event_coordinate_producer.py contour --contour-mode <mode>",
        ".venv/bin/python experiments/ddm_ec1_event_coordinate_producer.py receiver",
        ".venv/bin/python experiments/ddm_ec1_event_coordinate_producer.py finalize",
    ]
    queue_rows = [
        {
            "action": "JS5 realized acceptance over EC1 event-coordinate proposals",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN training-leg router",
            "consumer_store": str(PROPOSAL_ROOT),
            "fire_trigger": "MAIN owns the training leg, observes the sole n600 scorer slot free, verifies the EC1 store state and source archive SHA, then runs the existing JS5 pose-gated robust-improvement acceptance loop without regenerating proposal payloads",
        },
        {
            "action": "HY1 terminal whole-container replacement",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "HY1/js1 whole-container builder",
            "consumer_store": "/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/",
            "fire_trigger": "ps135 emits its terminal same-parent pose carrier; replace the stale T1R1 carrier, reuse the EC1 exact base-to-HY1 coordinate receipt for proposal ordering, rebuild the complete archive, and prove independent decode before any scorer request",
        },
        {
            "action": "local event-coordinate prior in HPAC",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "ddm_cl1_capacity MAIN executor/harvester",
            "consumer_store": "/Volumes/VertigoDataTier/pact/ddm_cl1_capacity_20260809/",
            "fire_trigger": "the existing CL1 lambda-1 twin equality and decode gates pass; condition only on the retained EC1 local event coordinates, race the complete model-plus-token package against the no-context control, and retain both payloads",
        },
    ]
    annex = (
        "# ddm_ec1 queue annex\n\n"
        + "\n".join(
            f"- **Action:** {row['action']}. **Disposition:** {row['disposition']}. **Owner:** {row['owner']}. **Consumer store:** `{row['consumer_store']}`. **Fire trigger:** {row['fire_trigger']}."
            for row in queue_rows
        )
        + "\n"
    )
    annex_row = atomic_bytes(root / "QUEUE_ANNEX.md", annex.encode())
    proposal_state_path = PROPOSAL_ROOT / "state.json"
    proposal_state = json.loads(proposal_state_path.read_text())
    proposal_state["producer_source"] = producer_source
    proposal_state["producer_invocations"] = producer_invocations
    atomic_json(proposal_state_path, proposal_state)
    result = {
        "schema": "ddm_ec1_final_result.v1",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "producer_source": producer_source,
        "producer_invocations": producer_invocations,
        "event_extraction": extracted,
        "pricing": priced,
        "curve_event_pricing": curve_priced,
        "receiver_effectiveness": receiver_result,
        "proposal_store": str(PROPOSAL_ROOT),
        "queued_actions": queue_rows,
        "queue_annex": annex_row,
        "falsifiers": {
            "F1": {
                mode: {
                    "fired": bool(curve_priced["modes"][mode]["f1_event_ge_intra"]),
                    "scope": "INSTANCE SP1 curve/event grammar with exact semantic reconstruction and real-coder race",
                }
                for mode in priced["modes"]
            },
            "F2": {
                "fired": receiver_result["receiver_effective_proposals"] == 0,
                "scope": "INSTANCE CP135 semantic receiver/R/uint8",
            },
            "F3": {
                "fired": receiver_result["receiver_effective_proposals"] < 20,
                "scope": "INSTANCE stratified JS5 n32 event alphabet",
            },
        },
        "boundaries": {
            "full_n600_scorer_run": False,
            "pose_scorer_run": False,
            "candidate_archive_built": False,
            "exact_score_measured": False,
            "receiver_axis": "shipped CP135 semantic renderer on CPU with exact camera uint8 and scorer-lattice R; no SegNet/PoseNet",
            "temporal_price": "exact semantic reconstruction; coder bytes are achievable description lengths, not archive scores or representation-free lower bounds",
            "proposal_acceptance": "not tested; MAIN owns the existing JS5 acceptance loop and scorer slot",
        },
    }
    final_row = atomic_json(root / "FINAL_RESULT.json", result)
    state = {
        "schema": "ddm_ec1_state.v1",
        "status": "COMPLETE",
        "resumable": True,
        "completed_stages": ["preflight", "extract", "contour", "price", "receiver", "finalize"],
        "final_result": final_row,
        "proposal_store_state": file_record(PROPOSAL_ROOT / "state.json"),
        "score_claim": False,
    }
    atomic_json(root / "state.json", state)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("preflight", "extract", "contour", "price", "receiver", "finalize", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--price-mode", choices=tuple(MODE_ID))
    parser.add_argument("--contour-mode", choices=tuple(MODE_ID))
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    if not str(root).startswith(("/Volumes/VertigoDataTier/pact/", "/Volumes/APDataStore/pact/")):
        raise EC1Error("run root must use an approved SSD tier")
    preflight(root)
    save_stage_state(root, "preflight", ["preflight"])
    if args.stage == "preflight":
        return 0
    extracted = extract(root)
    save_stage_state(root, "extract", ["preflight", "extract"])
    if args.stage == "extract":
        return 0
    if args.stage == "contour" and args.contour_mode:
        base = np.memmap(BASE_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
        hy1 = np.memmap(HY1_TOKENS, mode="r", dtype=np.uint8, shape=(N, H, W))
        contour_price_mode(root, args.contour_mode, extracted, base, hy1)
        save_stage_state(root, f"contour-{args.contour_mode}", ["preflight", "extract"])
        return 0
    curve_priced = contour_price(root, extracted)
    save_stage_state(root, "contour", ["preflight", "extract", "contour"])
    if args.stage == "contour":
        return 0
    if args.stage == "price" and args.price_mode:
        price_mode(root, args.price_mode, extracted["modes"][args.price_mode])
        save_stage_state(
            root,
            f"price-{args.price_mode}",
            ["preflight", "extract", "contour"],
        )
        return 0
    priced = price(root, extracted)
    save_stage_state(root, "price", ["preflight", "extract", "contour", "price"])
    if args.stage == "price":
        return 0
    receiver_result = receiver(root)
    save_stage_state(
        root,
        "receiver",
        ["preflight", "extract", "contour", "price", "receiver"],
    )
    if args.stage == "receiver":
        return 0
    result = finalize(root, extracted, priced, curve_priced, receiver_result)
    print(
        json.dumps(
            {"final": file_record(root / "FINAL_RESULT.json"), "proposal_store": result["proposal_store"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
