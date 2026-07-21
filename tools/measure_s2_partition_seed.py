#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the finite G3-to-S2 partition-event seed on real n600 custody.

The output is a component packet, not an archive and not a score.  It counts
sites, class identities, packet headers, the finite zlib stream, and CRC.  It
stores no RGB/YUV plane values.  Generic receiver logic remains in
``tac.optimization.s2_partition_seed`` and is reported separately as free
interpreter source LOC under the rule-118 boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.optimization.s2_partition_seed import (  # noqa: E402
    SEMANTIC_NAMES,
    PartitionEventSeed,
    decode_partition_seed,
    detect_partition_semantics,
    encode_partition_seed,
    events_from_rows,
    packet_accounting,
)

SCHEMA: Final = "s2_partition_seed_measurement.v1"
AXIS: Final = "[macOS-CPU advisory]"
POINTER: Final = "0.19108"
N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
EXPECTED_EVENTS: Final = 17_926
EXPECTED_STAGES: Final = 38
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
IDEAL_SPATIAL_TEMPORAL_BYTES: Final = 2724.873306413088
RAW_COORDINATE_BYTES: Final = 62741.0
RATE_PRICE_PER_BYTE: Final = 25.0 / 37_545_489.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _npz_member_memmap(path: Path, key: str) -> np.memmap:
    member_name = f"{key}.npy"
    with zipfile.ZipFile(path, "r") as archive:
        info = archive.getinfo(member_name)
        if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
            raise ValueError(f"{member_name} must be ZIP_STORED for bounded memmap access")
        with path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            if len(local) != 30:
                raise ValueError(f"truncated ZIP local header for {member_name}")
            fields = struct.unpack("<IHHHHHIIIHH", local)
            if fields[0] != 0x04034B50:
                raise ValueError(f"invalid ZIP local header for {member_name}")
            name_length, extra_length = fields[-2:]
            handle.seek(info.header_offset + 30 + name_length + extra_length)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
            offset = handle.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def _tree_hash(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _load_inventory(stage_dir: Path) -> tuple[list[list[float | int]], dict[str, Any]]:
    files = sorted(stage_dir.glob("batch-*.json"))
    if len(files) != EXPECTED_STAGES:
        raise ValueError(f"expected {EXPECTED_STAGES} inventory stages, found {len(files)}")
    rows: list[list[float | int]] = []
    cursor = 0
    cache_label_mismatches = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        start, stop = int(payload["pair_start"]), int(payload["pair_stop"])
        if start != cursor or not start < stop <= N_PAIRS:
            raise ValueError(f"non-contiguous inventory stage {path}: {start}:{stop}")
        flips = payload["flips"]
        if int(payload["flip_count"]) != len(flips):
            raise ValueError(f"flip_count drift in {path}")
        rows.extend(flips)
        cache_label_mismatches += int(payload["cache_label_mismatches"])
        cursor = stop
    if cursor != N_PAIRS or len(rows) != EXPECTED_EVENTS:
        raise ValueError(f"inventory custody drift: pairs={cursor}, events={len(rows)}")
    keys = [(int(row[0]), int(row[1]), int(row[2])) for row in rows]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("inventory sites must be unique and sorted")
    return rows, {
        "stage_dir": str(stage_dir),
        "stage_count": len(files),
        "stage_tree_sha256": _tree_hash(files, stage_dir),
        "cache_label_mismatches": cache_label_mismatches,
    }


def _edge_stratum(labels: np.ndarray, pair: int, row: int, col: int, road: int, lane: int) -> str:
    center = int(labels[pair, row, col])
    neighbors: list[int] = []
    if row:
        neighbors.append(int(labels[pair, row - 1, col]))
    if row + 1 < HEIGHT:
        neighbors.append(int(labels[pair, row + 1, col]))
    if col:
        neighbors.append(int(labels[pair, row, col - 1]))
    if col + 1 < WIDTH:
        neighbors.append(int(labels[pair, row, col + 1]))
    is_edge = any(value != center for value in neighbors)
    is_road_lane = center in (road, lane) and any(
        {center, value} == {road, lane} for value in neighbors
    )
    return "road_lane_edge" if is_road_lane else ("other_edge" if is_edge else "nonedge")


def _standalone_packet_row(
    rows: list[list[float | int]],
    semantic_ids: tuple[int, ...],
) -> dict[str, Any]:
    seed = PartitionEventSeed(
        n_pairs=N_PAIRS,
        height=HEIGHT,
        width=WIDTH,
        semantic_class_ids=semantic_ids,
        events=events_from_rows(rows),
    )
    payload = encode_partition_seed(seed)
    decoded = decode_partition_seed(payload)
    if decoded != seed:
        raise ValueError("standalone stratum packet failed exact parse-back")
    return packet_accounting(payload)


def measure(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_sha = _sha256_file(args.gt_cache)
    if cache_sha != args.expected_gt_cache_sha256:
        raise ValueError(f"GT cache SHA drift: {cache_sha} != {args.expected_gt_cache_sha256}")
    labels = _npz_member_memmap(args.gt_cache, "lstars")
    if labels.shape != (N_PAIRS, HEIGHT, WIDTH):
        raise ValueError(f"GT label geometry drift: {labels.shape}")
    detection = detect_partition_semantics(labels)
    detection_payload = {
        "schema": "s2_partition_semantic_detection_stage.v1",
        "gt_cache_sha256": cache_sha,
        **detection.to_dict(),
    }
    _atomic_json(args.output_dir / "stage_01_semantic_detection.json", detection_payload)

    rows, inventory = _load_inventory(args.flip_stage_dir)
    seed = PartitionEventSeed(
        n_pairs=N_PAIRS,
        height=HEIGHT,
        width=WIDTH,
        semantic_class_ids=detection.semantic_class_ids,
        events=events_from_rows(rows),
    )
    payload = encode_partition_seed(seed)
    payload_repeat = encode_partition_seed(seed)
    if payload_repeat != payload or decode_partition_seed(payload) != seed:
        raise ValueError("full partition seed is not deterministic and exact on parse-back")
    packet_path = args.output_dir / "s2_partition_event_seed.bin"
    _atomic_bytes(packet_path, payload)
    accounting = packet_accounting(payload)
    accounting.update(
        {
            "path": str(packet_path),
            "double_encode_byte_identical": True,
            "parse_back_event_identity": True,
        }
    )
    _atomic_json(args.output_dir / "stage_02_packet_accounting.json", accounting)

    inverse_semantics = {
        class_id: SEMANTIC_NAMES[index]
        for index, class_id in enumerate(detection.semantic_class_ids)
    }
    grouped: dict[str, dict[str, list[list[float | int]]]] = {
        "edge_stratum": defaultdict(list),
        "margin_stratum": defaultdict(list),
        "class_transition": defaultdict(list),
    }
    target_mismatches = 0
    road, lane = detection.semantic_class_ids[:2]
    for row in rows:
        pair, y, x = int(row[0]), int(row[1]), int(row[2])
        target, baseline = int(row[3]), int(row[4])
        margin = abs(float(row[5]))
        target_mismatches += int(int(labels[pair, y, x]) != target)
        grouped["edge_stratum"][_edge_stratum(labels, pair, y, x, road, lane)].append(row)
        grouped["margin_stratum"][
            "moderate_[1e-3,1)" if margin >= 1e-3 else "tight_<1e-3"
        ].append(row)
        grouped["class_transition"][
            f"{inverse_semantics[target]}->{inverse_semantics[baseline]}"
        ].append(row)
    strata: dict[str, dict[str, Any]] = {}
    for family, family_rows in grouped.items():
        strata[family] = {}
        for name, selected in sorted(family_rows.items()):
            strata[family][name] = _standalone_packet_row(
                selected, detection.semantic_class_ids
            )

    interpreter_path = REPO / "src/tac/optimization/s2_partition_seed.py"
    interpreter_source = interpreter_path.read_bytes()
    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_unmoved": POINTER,
        "n_pairs": N_PAIRS,
        "seed": 1234,
        "content_lineage": {
            "from_scratch_our_solve": True,
            "inherited_bytes_in_candidate": 0,
            "input_flip_inventory_role": "harvest-only measurement fixture",
            "stored_rgb_or_yuv_plane_value_bytes": 0,
        },
        "gt_cache": {
            "path": str(args.gt_cache),
            "bytes": args.gt_cache.stat().st_size,
            "sha256": cache_sha,
            "access": "ZIP_STORED read-only memmap, one label plane at a time",
        },
        "semantic_detection": detection_payload,
        "inventory_custody": {**inventory, "cache_target_mismatches": target_mismatches},
        "finite_packet": accounting,
        "standalone_per_stratum_packets": strata,
        "economics": {
            "prior_ideal_spatial_temporal_bytes_excluding_sites_headers": IDEAL_SPATIAL_TEMPORAL_BYTES,
            "prior_raw_coordinate_bytes_excluding_class_stream_headers": RAW_COORDINATE_BYTES,
            "finite_vs_ideal_ratio": len(payload) / IDEAL_SPATIAL_TEMPORAL_BYTES,
            "finite_vs_raw_coordinate_ratio": len(payload) / RAW_COORDINATE_BYTES,
            "rate_price_s_per_byte": RATE_PRICE_PER_BYTE,
            "finite_packet_rate_term": len(payload) * RATE_PRICE_PER_BYTE,
        },
        "rule118_split": {
            "counted_video_derived_seed_bytes": len(payload),
            "free_interpreter_path": str(interpreter_path.relative_to(REPO)),
            "free_interpreter_source_lines": len(interpreter_source.splitlines()),
            "free_interpreter_source_sha256": hashlib.sha256(interpreter_source).hexdigest(),
            "free_generic_algorithms": [
                "spatial/static semantic detector",
                "site-delta ULEB128 parser",
                "finite zlib context decoder",
                "deterministic cell-constraint application",
            ],
            "video_derived_table_embedded_in_interpreter": False,
        },
        "measured_seconds": time.monotonic() - started,
        "verdict": "FINITE_G3_CELL_EVENT_SEED_PARSEBACK_COMPLETE_BASE_PARTITION_RECEIVER_OPEN",
        "verdict_scope": (
            "real n600 measured G3 event sites and target/baseline cell identities; all finite "
            "site/header/coder/CRC bytes counted and exact on parse-back. This is not a full "
            "partition, xi, plane-realization, archive, d_seg, or d_pose receiver row because "
            "the baseline partition predictor remains external."
        ),
    }
    _atomic_json(args.output_dir / "receipt.json", receipt)
    _atomic_json(args.receipt, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--flip-stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-gt-cache-sha256", default=GT_CACHE_SHA256)
    return parser.parse_args()


def main() -> int:
    receipt = measure(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
